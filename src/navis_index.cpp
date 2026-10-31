#include "navis/index/navis_index.h"
#include "navis/io/uring_backend.h"
#include "navis/util/file_utils.h"

#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace navis {

// ===========================================================================
// Lifecycle
// ===========================================================================

template <typename T>
NavisIndex<T>::~NavisIndex() {
    // Stop background write threads
    bg_write_queue_.stop();
    for (auto& t : bg_write_threads_) {
        if (t.joinable()) t.join();
    }

    // Free buffer pool
    for (auto* buf : all_buffers_) {
        navis_dma_free(buf->edge_scratch);
        navis_dma_free(buf->vec_scratch);
        free_aligned(buf->coord_scratch);
        free_aligned(buf->pq_dist_scratch);
        free_aligned(buf->pq_coord_scratch);
        free_aligned(buf->dist_scratch);
        free_aligned(buf->aligned_query);
        delete buf;
    }
}

template <typename T>
void NavisIndex<T>::load(const std::string& prefix,
                         IOBackend* io_edge,
                         IOBackend* io_vec,
                         const SearchConfig& config,
                         const CacheConfig& cache_cfg,
                         uint32_t n_threads,
                         Metric metric) {
    io_edge_ = io_edge;
    io_vec_  = io_vec;
    config_  = config;
    metric_  = metric;

    // 1. Read metadata from the _disk.edge file (sector 0).
    std::string edge_file = prefix + "_disk.edge";
    std::string vec_file  = prefix + "_disk.vec";
    if (!file_exists(edge_file) || !file_exists(vec_file)) {
        throw std::runtime_error(
            "NAVIS separated-layout files missing: " + edge_file + ", " + vec_file);
    }

    std::ifstream meta_in(edge_file, std::ios::binary);
    char meta_buf[META_SECTOR_LEN];
    meta_in.read(meta_buf, META_SECTOR_LEN);

    int32_t meta_count, meta_ndim;
    std::memcpy(&meta_count, meta_buf, sizeof(int32_t));
    std::memcpy(&meta_ndim,  meta_buf + sizeof(int32_t), sizeof(int32_t));
    if (meta_count != 9 || meta_ndim != 1) {
        throw std::runtime_error(
            "NAVIS separated-layout header invalid: count=" +
            std::to_string(meta_count) + " ndim=" + std::to_string(meta_ndim) +
            ". Rebuild with navis_build.");
    }

    const uint64_t* meta = reinterpret_cast<const uint64_t*>(
        meta_buf + 2 * sizeof(int32_t));
    num_points_          = meta[0];
    uint64_t dim         = meta[1];
    uint64_t medoid      = meta[2];
    uint64_t max_vec_len = meta[3];
    uint64_t max_edge_len = meta[4];
    uint64_t epp         = meta[5];
    uint64_t file_kind   = meta[6];
    if (file_kind != 0) {
        throw std::runtime_error("Expected edge file, got file_kind=" +
                                 std::to_string(file_kind));
    }
    if (num_points_ == 0 || dim == 0 || epp == 0) {
        throw std::runtime_error("Invalid metadata in " + edge_file);
    }

    layout_.init_from_metadata(num_points_, dim, sizeof(T),
                               max_vec_len, max_edge_len, epp);
    aligned_dim_ = round_up(dim, 8);

    // Medoid fallback entry point.
    medoids_.clear();
    medoids_.push_back(static_cast<uint32_t>(medoid));

    // 2. PQ table / PQ data.
    std::string pq_pivots = prefix + "_pq_pivots.bin";
    std::string pq_data   = prefix + "_pq_compressed.bin";
    uint32_t pq_npts, pq_dim;
    get_bin_metadata(pq_data, pq_npts, pq_dim);
    n_chunks_ = pq_dim;
    pq_table_.load(pq_pivots, n_chunks_);

    uint8_t* pq_raw = nullptr;
    uint32_t pq_n, pq_d;
    load_bin<uint8_t>(pq_data, pq_raw, pq_n, pq_d);
    // Pre-allocate pq_data_ to 2× initial points — prevents concurrent
    // resize() during insertion (std::vector resize is NOT thread-safe for
    // concurrent readers, causes segfault in compute_pq_dists_internal).
    uint64_t pq_reserve = 2 * static_cast<uint64_t>(pq_n) * pq_d;
    pq_data_.reserve(pq_reserve);
    pq_data_.assign(pq_raw, pq_raw + static_cast<uint64_t>(pq_n) * pq_d);
    pq_data_.resize(pq_reserve);  // fill up to capacity (zeros for unwritten)
    free_aligned(pq_raw);
    dist_cmp_ = make_distance<T>(metric_);

    // 3. Indirection table — load permuted mapping if present, otherwise
    //    fall back to identity (id2edge_loc(id) == id, no backing array).
    //
    // Pre-size the forward maps + reverse loc map to 2× num_points_. Each
    // insert allocates O(R+1) fresh edge slots (97 at R=96), so a full 1M
    // insert workload appends ~100M entries to loc2id. Pre-sizing avoids
    // a runtime grow entirely, which closes a race where a concurrent
    // id2edge_loc / edge_loc2id reader can observe an old base pointer
    // while a writer is publishing a new one (the grow publishes the new
    // base + capacity with release stores, but readers that captured the
    // old base before the write still index into the smaller array —
    // OOB when id/loc ≥ old cap). 2× covers any realistic insert workload
    // on a 50M base while keeping static memory under 1 GB per map.
    indirection_.set_geometry(static_cast<uint32_t>(layout_.edges_per_page));
    uint64_t indir_cap = num_points_ * 2 + 1024;
    if (!indirection_.load(prefix, indir_cap)) {
        indirection_.init_identity(num_points_);
    }

    // 4. Entrance graph (NAVIS-Update keeps it growing during insertion).
    std::string mem_index = prefix + "_mem.index";
    if (file_exists(mem_index)) {
        entrance_graph_ = std::make_unique<EntranceGraph<T>>();
        entrance_graph_->load(mem_index, dist_cmp_.get(), dim);
    }

    // 5. Buffer pool / locks.
    // Pre-size vertex lock table to 2x num_points_ so we never grow at
    // runtime. The grow path has a race between the copy pass and
    // concurrent unlocks: a live lock (true) snapshotted into the new
    // array can be cleared on OLD by a concurrent unlock before the
    // grow's publish, leaving NEW[id]=true forever — a ghost lock that
    // stalls all future acquirers indefinitely. 2x headroom is 2 MB per 1M
    // points and covers the expected insert volume.
    vertex_locks_ = VertexLockTable(num_points_ * 2 + 1024);
    cur_id_.store(num_points_);
    deletion_table_.init(num_points_ * 2 + 1024);
    init_buffer_pool(n_threads);

    // 6. NAVIS-Cache (disabled when size_gb == 0). The indirection table
    //    sends it an eviction hint when an edge page is fully invalidated.
    navis_cache_.init(cache_cfg);
    indirection_.set_navis_cache(&navis_cache_);

    // 7. Background writer.
    bg_write_threads_.emplace_back(&NavisIndex::bg_write_worker, this);
}

// ===========================================================================
// Buffer pool
// ===========================================================================

template <typename T>
void NavisIndex<T>::init_buffer_pool(uint32_t n_threads) {
    // edge_scratch slots are always 4KB (one edge page per beam entry).
    // vec_scratch stride accommodates a max_vec_len vector straddling up to
    // two sectors: round_up(max_vec_len, SECTOR_LEN) + SECTOR_LEN.
    uint64_t vec_stride = layout_.vec_read_stride();

    uint32_t n_buffers = n_threads * 2;
    for (uint32_t i = 0; i < n_buffers; i++) {
        auto* buf = new QueryBuffer<T>();

        buf->edge_scratch = static_cast<char*>(
            navis_dma_alloc(MAX_N_SECTOR_READS * SECTOR_LEN));
        buf->vec_scratch = static_cast<char*>(
            navis_dma_alloc(MAX_N_SECTOR_READS * vec_stride));

        buf->coord_scratch = static_cast<T*>(
            alloc_aligned(MAX_N_CMPS * aligned_dim_ * sizeof(T)));
        buf->pq_dist_scratch = static_cast<float*>(
            alloc_aligned(NUM_PQ_CENTROIDS * MAX_PQ_CHUNKS * sizeof(float)));
        buf->pq_coord_scratch = static_cast<uint8_t*>(
            alloc_aligned(MAX_PQ_CHUNKS * MAX_N_EDGES));
        buf->dist_scratch = static_cast<float*>(
            alloc_aligned(MAX_N_EDGES * sizeof(float)));
        buf->aligned_query = static_cast<T*>(
            alloc_aligned(aligned_dim_ * sizeof(T)));

        all_buffers_.push_back(buf);
        buffer_pool_.push(buf);
    }
}

template <typename T>
QueryBuffer<T>* NavisIndex<T>::acquire_buffer() {
    return buffer_pool_.pop();
}

template <typename T>
void NavisIndex<T>::release_buffer(QueryBuffer<T>* buf) {
    buf->reset();
    buffer_pool_.push(buf);
}

// ===========================================================================
// Background write worker
// ===========================================================================

// bg_write_worker — drain queue non-blockingly, merge whatever contiguity
// already exists in the snapshot.
//
// Default I/O unit is 4 KiB per sector. On each outer iteration we block
// for the first task, then drain every other task already pending via
// try_pop (no time wait). All collected edge writes (resp. vec writes)
// are sorted by offset and merged wherever sectors happen to be adjacent,
// with no size cap — if natural contiguity from the fresh-tail allocator
// gives us a 100-sector run, it submits as one 400 KiB IORequest. Ranges
// that don't touch stay as independent 4 KiB writes.
//
// No dedup across tasks: vec writes are already gated by pair-completion
// in the foreground (each sector appears at most once), and edge writes
// come from fresh-allocated dest locs that are unique to each insert.
template <typename T>
void NavisIndex<T>::bg_write_worker() {
    io_edge_->register_thread();
    io_vec_->register_thread();

    auto merge_adjacent = [&](std::vector<IORequest>& writes,
                              std::vector<IORequest>& out,
                              std::vector<void*>& scratch) {
        if (writes.empty()) return;
        std::sort(writes.begin(), writes.end(),
                  [](const IORequest& a, const IORequest& b) {
                      return a.offset < b.offset;
                  });
        size_t i = 0;
        while (i < writes.size()) {
            size_t j = i;
            while (j + 1 < writes.size() &&
                   writes[j + 1].offset == writes[j].offset + SECTOR_LEN) {
                ++j;
            }
            size_t run_len = j - i + 1;
            if (run_len == 1) {
                out.push_back(writes[i]);
            } else {
                size_t bytes = run_len * SECTOR_LEN;
                void* merged = navis_dma_alloc(bytes);
                scratch.push_back(merged);
                for (size_t k = 0; k < run_len; ++k) {
                    std::memcpy(static_cast<char*>(merged) + k * SECTOR_LEN,
                                writes[i + k].buf, SECTOR_LEN);
                }
                out.emplace_back(writes[i].offset, bytes, merged);
            }
            i = j + 1;
        }
    };

    while (true) {
        auto* first = bg_write_queue_.pop();
        if (!first) break;  // queue stopped

        // Drain current queue state without blocking. Batch size depends
        // purely on how many tasks are already pending.
        std::vector<BgWriteTask*> batch;
        batch.push_back(first);
        while (true) {
            auto nxt = bg_write_queue_.try_pop();
            if (!nxt || *nxt == nullptr) break;
            batch.push_back(*nxt);
        }

        std::vector<IORequest> all_edge, all_vec;
        for (auto* t : batch) {
            for (auto& w : t->edge_writes) all_edge.push_back(w);
            for (auto& w : t->vec_writes)  all_vec .push_back(w);
        }

        std::vector<IORequest> edge_io, vec_io;
        std::vector<void*>     scratch;
        merge_adjacent(all_edge, edge_io, scratch);
        merge_adjacent(all_vec,  vec_io,  scratch);

        if (!edge_io.empty()) io_edge_->write_batch(edge_io);
        if (!vec_io.empty())  io_vec_->write_batch(vec_io);

        // Write-through to NAVIS-Cache: for every edge sector just written
        // to disk, memcpy fresh bytes into the cache entry if it exists.
        // No admission on miss, no frequency bump, no promotion.
        //
        // This must come BEFORE the mark_clean_and_deref loop below —
        // otherwise there's a brief window where RMW has erased the entry
        // but NAVIS-Cache still serves pre-write bytes (stale read hole).
        //
        // Iterates all_edge (pre-merge per-sector list) so each call gets
        // a SECTOR_LEN buffer, even when the merged IORequests span
        // multi-sector runs.
        for (auto& w : all_edge) {
            uint64_t sec = w.offset / SECTOR_LEN;
            navis_cache_.update_if_present(
                sec, reinterpret_cast<const uint8_t*>(w.buf));
        }

        // Release every batched task's state.
        for (auto* t : batch) {
            unlock_pages(page_locks_, t->pages_to_unlock);
            for (uint64_t s : t->pages_to_deref) {
                rmw_cache_.mark_clean_and_deref(s);
            }
            for (void* buf : t->owned_bufs) navis_dma_free(buf);
            delete t;
        }
        for (void* b : scratch) navis_dma_free(b);
    }

    io_edge_->deregister_thread();
    io_vec_ ->deregister_thread();
}

template <typename T>
bool NavisIndex<T>::erase(uint32_t tag) {
    uint64_t active = cur_id_.load(std::memory_order_acquire);
    return deletion_table_.erase(tag, active);
}

template <typename T>
uint64_t NavisIndex<T>::deletion_adjusted_search_L(uint64_t requested_L) const {
    uint64_t limit = MAX_N_CMPS - 1;
    uint64_t base = requested_L > limit ? limit : requested_L;
    if (base == 0) return 0;

    uint64_t active = cur_id_.load(std::memory_order_acquire);
    double frac = deletion_table_.deleted_fraction(active);
    if (frac <= 0.0) return base;

    double live = 1.0 - frac;
    if (live < 0.10) live = 0.10;
    uint64_t expanded = static_cast<uint64_t>(
        static_cast<double>(base) / live + 0.999999);
    expanded += kCasrTopK;

    uint64_t cap = base * 4 + 64;
    if (cap < base) cap = base;
    if (cap > limit) cap = limit;
    if (expanded > cap) expanded = cap;
    return expanded < base ? base : expanded;
}

// ===========================================================================
// Search entry point
// ===========================================================================

template <typename T>
void NavisIndex<T>::search(const T* query, uint64_t K, uint32_t mem_L,
                           uint64_t search_L, uint32_t beam_width,
                           uint32_t* out_ids, float* out_dists,
                           QueryStats* stats) {
    auto* buf = acquire_buffer();

    std::vector<Neighbor> results;
    do_search(query, mem_L, search_L, beam_width,
              results, nullptr, stats, false, buf);

    uint64_t out = 0;
    for (const auto& n : results) {
        if (out >= K) break;
        if (deletion_table_.is_deleted(n.id)) continue;
        out_ids[out] = n.id;
        out_dists[out] = n.distance;
        out++;
    }
    for (uint64_t i = out; i < K; i++) {
        out_ids[i]   = UINT32_MAX;
        out_dists[i] = std::numeric_limits<float>::max();
    }

    release_buffer(buf);
}

// ===========================================================================
// Insert entry point
// ===========================================================================

template <typename T>
int NavisIndex<T>::insert(const T* point, uint32_t tag) {
    auto* buf = acquire_buffer();

    if (deletion_table_.is_deleted(tag)) {
        release_buffer(buf);
        return -1;
    }

    // 1. Assign id: the caller's tag is the vertex id (e.g. its row in the
    //    data file, which keeps ground-truth ids valid). Tags must be unique
    //    and >= the initial number of points. cur_id_ is a high-water mark.
    //
    //    id2edge_loc(target_id) / id2vec_loc(target_id) are bound only in
    //    do_insert, after the target's data is written: binding earlier would
    //    expose an uninitialized slot to concurrent position seeking.
    uint32_t target_id = tag;
    uint64_t expected = cur_id_.load(std::memory_order_relaxed);
    while (expected <= target_id) {
        if (cur_id_.compare_exchange_weak(expected, target_id + 1,
                                          std::memory_order_relaxed)) {
            break;
        }
    }

    // 2. Deflate vector to PQ codes.
    std::vector<uint8_t> pq_codes(n_chunks_);
    if constexpr (std::is_same_v<T, float>) {
        pq_table_.deflate_vec(point, pq_codes.data());
    } else {
        std::vector<float> fp_vec(layout_.data_dim);
        for (uint64_t i = 0; i < layout_.data_dim; i++) {
            fp_vec[i] = static_cast<float>(point[i]);
        }
        pq_table_.deflate_vec(fp_vec.data(), pq_codes.data());
    }

    uint64_t pq_offset = static_cast<uint64_t>(target_id) * n_chunks_;
    // pq_data_ is pre-allocated at load time (2x the initial points) so that
    // concurrent readers never observe a reallocation.
    if (pq_offset + n_chunks_ > pq_data_.size()) {
        std::fprintf(stderr,
            "[insert] pq_data_ capacity exhausted (target_id=%u, cap=%zu ids). "
            "Increase pq_reserve factor in load().\n",
            target_id, pq_data_.size() / n_chunks_);
        std::abort();
    }
    std::memcpy(pq_data_.data() + pq_offset, pq_codes.data(), n_chunks_);

    // 3. Position seeking: entry-point selection with an ent_R-sized list,
    //    then on-disk traversal with frontier insert_L and beam width 4.
    constexpr uint32_t kPosSeekBeamWidth = 4;
    std::vector<Neighbor> candidates;
    tsl::robin_map<uint32_t, T*> coord_map;
    std::vector<Neighbor> explored, explored_ent;
    do_search(point, config_.ent_R, config_.insert_L, kPosSeekBeamWidth,
              candidates, &coord_map, nullptr, true, buf,
              &explored, &explored_ent);

    // 4. Prune neighbors.
    std::vector<uint32_t> neighbors;
    prune_neighbors(candidates, neighbors, &coord_map);

    if (neighbors.empty()) {
        release_buffer(buf);
        return -1;
    }

    // 5. Structural update on disk.
    int rc = do_insert(target_id, point, neighbors, buf);

    // 6. NAVIS-Update: maybe promote the new vertex into the entrance graph,
    //    reusing the explored sets of this insertion.
    if (entrance_graph_) {
        entrance_graph_->maybe_promote(target_id, point,
                                        cur_id_.load(),
                                        explored_ent, explored, config_,
                                        [this](uint32_t id) {
                                            return deletion_table_.is_deleted(id);
                                        });
    }

    release_buffer(buf);
    return rc >= 0 ? static_cast<int>(target_id) : -1;
}

// ===========================================================================
// Explicit instantiations
// ===========================================================================

template class NavisIndex<float>;
template class NavisIndex<int8_t>;
template class NavisIndex<uint8_t>;

} // namespace navis
