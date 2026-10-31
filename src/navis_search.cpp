#include "navis/index/navis_index.h"
#include "navis/cache/navis_cache.h"
#include "navis/io/uring_backend.h"
#include "navis/pq_table.h"

#include <tsl/robin_map.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__SSE__) || defined(__AVX__)
#include <xmmintrin.h>
#endif

namespace navis {

// Promotion threshold for edge pages first seen by user search. Position
// seeking uses the configured NAVIS-Cache threshold (default 2); search uses
// a much larger one so transient query-side accesses cycle through the
// admission window instead of churning the frozen region that holds the
// working set near the entry points.
static constexpr uint32_t kSearchPromoteThreshold = 16;

// ---------------------------------------------------------------------------
// insert_into_pool — sorted-array insert with binary search + memmove.
//
// Returns the position the new entry was inserted at, or K+1 if rejected
// (duplicate id or worse than the K-th entry when full).
// ---------------------------------------------------------------------------
static inline uint32_t insert_into_pool(Neighbor* addr, uint32_t K,
                                        const Neighbor& nn) {
    if (K == 0) {
        addr[0] = nn;
        return 0;
    }
    // Best so far → just shift everything and prepend.
    if (addr[0].distance > nn.distance) {
        std::memmove(&addr[1], &addr[0], K * sizeof(Neighbor));
        addr[0] = nn;
        return 0;
    }
    // Worse than the worst → append (caller decides whether to keep).
    if (addr[K - 1].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    // Binary search for the first slot whose distance ≥ nn.distance.
    uint32_t left = 0, right = K - 1;
    while (right > 1 && left < right - 1) {
        uint32_t mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance) right = mid; else left = mid;
    }
    // Tie-on-distance: walk back to dedupe by id.
    while (left > 0) {
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
    std::memmove(&addr[right + 1], &addr[right],
                 (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

// ===========================================================================
// do_search — shared by user search and insertion position seeking.
//
// Stage 1: entry-point selection on the in-memory entrance graph (+ medoid).
// Stage 2: on-disk traversal. Beam search with PQ distances that reads only
//          4 KiB edge pages; candidates sharing a page share one read. Pages
//          are looked up in the RMW cache, then NAVIS-Cache, then storage.
// Stage 3: CASR reranking. Full vectors of the PQ-ranked explored set are
//          compared one at a time until the exact top-K has been unchanged
//          for a calibrated window of w loads.
//
// For position seeking (coord_map != nullptr) the reranked vertices keep their
// exact vectors in coord_map, and the explored vertices that were not reranked
// are appended with their PQ distance and a deflated-PQ vector, so
// prune_neighbors can fill the remaining neighbor slots.
// ===========================================================================

template <typename T>
void NavisIndex<T>::do_search(const T* query, uint32_t mem_L, uint64_t search_L,
                              uint32_t beam_width,
                              std::vector<Neighbor>& results,
                              tsl::robin_map<uint32_t, T*>* coord_map,
                              QueryStats* stats,
                              bool is_position_seeking,
                              QueryBuffer<T>* buf,
                              std::vector<Neighbor>* out_explored,
                              std::vector<Neighbor>* out_explored_ent) {

    const uint64_t active_points = cur_id_.load(std::memory_order_acquire);
    search_L = deletion_adjusted_search_L(search_L);

    // Prepare aligned query
    std::memset(buf->aligned_query, 0, aligned_dim_ * sizeof(T));
    std::memcpy(buf->aligned_query, query, layout_.data_dim * sizeof(T));

    pq_table_.populate_chunk_distances(buf->aligned_query, buf->pq_dist_scratch);

    // Prefetch the scratch buffers touched on every beam iteration.
#if defined(__SSE__) || defined(__AVX__)
    _mm_prefetch(reinterpret_cast<const char*>(buf->dist_scratch),
                 _MM_HINT_T1);
    _mm_prefetch(reinterpret_cast<const char*>(buf->pq_coord_scratch),
                 _MM_HINT_T1);
#endif

    // -----------------------------------------------------------------------
    // Stage 1: Entry-point selection
    // -----------------------------------------------------------------------
    std::vector<Neighbor> retset(search_L + 1);
    for (uint64_t i = 0; i <= search_L; i++) {
        retset[i].distance = std::numeric_limits<float>::max();
        retset[i].id = UINT32_MAX;
        retset[i].flag = false;
    }
    uint64_t cur_list_size = 0;

    std::vector<Neighbor> explored_ent;

    if (entrance_graph_ && entrance_graph_->is_loaded() && mem_L > 0) {
        std::vector<Neighbor> ent_results;
        entrance_graph_->search(buf->aligned_query, mem_L, ent_results, &explored_ent);
        if (out_explored_ent) *out_explored_ent = explored_ent;
        for (auto& n : ent_results) {
            if (deletion_table_.is_deleted(n.id)) continue;
            if (cur_list_size < search_L) {
                float pq_dist = pq_dist_single(
                    pq_data_.data() + static_cast<uint64_t>(n.id) * n_chunks_,
                    n_chunks_, buf->pq_dist_scratch);
                Neighbor nn(n.id, pq_dist, true);
                insert_into_pool(retset.data(),
                                 static_cast<uint32_t>(cur_list_size), nn);
                cur_list_size++;
                buf->visited.insert(n.id);
            }
        }
    }

    bool added_medoid = false;
    for (uint32_t mid : medoids_) {
        if (deletion_table_.is_deleted(mid)) continue;
        if (!buf->visited.insert(mid).second) continue;
        float dist = pq_dist_single(pq_data_.data() + static_cast<uint64_t>(mid) * n_chunks_,
                                    n_chunks_, buf->pq_dist_scratch);
        if (cur_list_size < search_L) {
            Neighbor nn(mid, dist, true);
            insert_into_pool(retset.data(),
                             static_cast<uint32_t>(cur_list_size), nn);
            cur_list_size++;
            added_medoid = true;
        }
    }
    if (search_L > 0 && !added_medoid && cur_list_size == 0 && !medoids_.empty()) {
        uint32_t mid = medoids_[0];
        if (buf->visited.insert(mid).second) {
            float dist = pq_dist_single(pq_data_.data() + static_cast<uint64_t>(mid) * n_chunks_,
                                        n_chunks_, buf->pq_dist_scratch);
            Neighbor nn(mid, dist, true);
            insert_into_pool(retset.data(),
                             static_cast<uint32_t>(cur_list_size), nn);
            cur_list_size++;
        }
    }

    // -----------------------------------------------------------------------
    // Stage 2: On-disk traversal
    // -----------------------------------------------------------------------
    uint64_t hop_count = 0;

    // Per beam entry: which edge page it reads. page_sp (when set) is an
    // immutable snapshot captured from NAVIS-Cache, read in place without a
    // 4 KiB copy; RMW-cache hits and storage reads use an edge_scratch slot.
    struct BeamEntry {
        uint32_t id;
        uint32_t edge_loc;
        uint64_t edge_page_sec;
        uint32_t edge_slot;    // index into buf->edge_scratch (when page_sp is null)
        NavisCache::PageBufSp page_sp;
    };

    std::vector<BeamEntry> beam;

    // The explored set E: every expanded vertex with its PQ distance. CASR
    // reranks over E, and position seeking fills neighbor slots from it.
    std::vector<Neighbor> approx_explored;
    approx_explored.reserve(static_cast<size_t>(search_L) * 10);

    const uint32_t promote_thresh = is_position_seeking
        ? 0 /* configured NAVIS-Cache default */
        : kSearchPromoteThreshold;

    uint64_t k = 0;
    while (k < cur_list_size) {
        buf->edge_slot_idx = 0;
        beam.clear();

        // Pick beam_width unexpanded candidates.
        for (uint64_t i = k; i < cur_list_size && beam.size() < beam_width; i++) {
            if (retset[i].flag) {
                retset[i].flag = false;
                uint32_t id = retset[i].id;
                uint32_t edge_loc = indirection_.id2edge_loc(id);
                approx_explored.emplace_back(id, retset[i].distance, false);
                beam.push_back(BeamEntry{
                    id, edge_loc, layout_.edge_page_no(edge_loc), 0, nullptr
                });
            }
        }
        if (beam.empty()) break;

        hop_count++;

        // Resolve each beam entry's edge page: RMW cache → NAVIS-Cache →
        // storage. Entries sharing a page share one lookup/read. Storage
        // misses are submitted as ONE batched read, and admitted to the
        // NAVIS-Cache window only after the read fills the buffer.
        tsl::robin_map<uint64_t, uint32_t> page_to_slot;
        tsl::robin_map<uint64_t, NavisCache::PageBufSp> page_to_sp;
        std::vector<IORequest> edge_reads;
        std::vector<uint64_t>  miss_secs;
        for (auto& be : beam) {
            auto sp_it = page_to_sp.find(be.edge_page_sec);
            if (sp_it != page_to_sp.end()) {
                be.page_sp = sp_it->second;
                continue;
            }
            auto it = page_to_slot.find(be.edge_page_sec);
            if (it != page_to_slot.end()) {
                be.edge_slot = it->second;
                continue;
            }

            uint32_t slot = static_cast<uint32_t>(buf->edge_slot_idx++);
            be.edge_slot = slot;
            uint8_t* io_buf = reinterpret_cast<uint8_t*>(
                buf->edge_scratch + slot * SECTOR_LEN);

            // RMW cache first: it holds the freshest bytes of pages whose
            // write is still in flight.
            if (rmw_cache_.get(be.edge_page_sec, io_buf)) {
                page_to_slot.emplace(be.edge_page_sec, slot);
                continue;
            }
            // NAVIS-Cache: pointer return, no copy.
            NavisCache::PageBufSp sp;
            if (navis_cache_.get_sp(be.edge_page_sec, &sp, promote_thresh)) {
                page_to_sp.emplace(be.edge_page_sec, sp);
                be.page_sp = std::move(sp);
                buf->edge_slot_idx--;  // slot wasn't used; release it
                continue;
            }
            // Miss: queue for the batched storage read.
            page_to_slot.emplace(be.edge_page_sec, slot);
            edge_reads.emplace_back(
                layout_.edge_file_offset_from_sector(be.edge_page_sec),
                SECTOR_LEN, io_buf);
            miss_secs.push_back(be.edge_page_sec);
        }

        if (!edge_reads.empty()) {
            io_edge_->read_batch(edge_reads);
            if (stats) stats->n_ios += edge_reads.size();
        }
        for (uint64_t sec : miss_secs) {
            uint32_t slot = page_to_slot[sec];
            navis_cache_.populate(sec, reinterpret_cast<uint8_t*>(
                buf->edge_scratch + slot * SECTOR_LEN));
        }

        uint64_t nk = cur_list_size;
        for (auto& be : beam) {
            const char* page_buf = be.page_sp
                ? reinterpret_cast<const char*>(be.page_sp->data())
                : (buf->edge_scratch + be.edge_slot * SECTOR_LEN);
            const char* edge_ptr = layout_.edge_in_page(page_buf, be.edge_loc);

            // Skip entries whose degree exceeds R: such bytes are not a valid
            // edge list (e.g. a zero-initialized slot past the built file).
            uint32_t degree = SeparatedLayout::edge_degree(edge_ptr);
            if (degree > layout_.max_degree) continue;
            const uint32_t* nbrs = SeparatedLayout::edge_neighbors(edge_ptr);

            uint32_t new_ids[MAX_N_EDGES];
            uint32_t n_new = 0;
            for (uint32_t j = 0; j < degree; j++) {
                uint32_t nbr_id = nbrs[j];
                if (nbr_id < active_points && buf->visited.insert(nbr_id).second) {
                    new_ids[n_new++] = nbr_id;
                }
            }
            if (n_new == 0) continue;

            aggregate_pq_coords(new_ids, n_new, pq_data_.data(),
                                n_chunks_, buf->pq_coord_scratch);
            pq_dist_lookup(buf->pq_coord_scratch, n_new, n_chunks_,
                           buf->pq_dist_scratch, buf->dist_scratch);
            if (stats) stats->n_cmps += n_new;

            for (uint32_t j = 0; j < n_new; j++) {
                float pq_dist = buf->dist_scratch[j];
                if (cur_list_size >= search_L &&
                    pq_dist >= retset[cur_list_size - 1].distance)
                    continue;

                Neighbor nn(new_ids[j], pq_dist, true);
                uint32_t pos = insert_into_pool(
                    retset.data(), static_cast<uint32_t>(cur_list_size), nn);
                if (pos > cur_list_size) continue;  // duplicate id
                if (cur_list_size < search_L) cur_list_size++;
                if (pos < nk) nk = pos;
            }
        }

        if (nk <= k) k = nk;
        else k++;
    }
    if (stats) stats->n_hops = hop_count;

    // Explored set of the on-disk traversal, consumed by NAVIS-Update.
    if (out_explored) {
        out_explored->clear();
        out_explored->reserve(cur_list_size);
        for (uint64_t i = 0; i < cur_list_size; i++) {
            if (retset[i].id != UINT32_MAX) out_explored->push_back(retset[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Stage 3: CASR reranking
    // -----------------------------------------------------------------------
    // Candidates: E sorted by PQ distance, capped at the per-query vector
    // scratch capacity, minus deleted vertices.
    std::sort(approx_explored.begin(), approx_explored.end());
    const size_t n_fill = std::min<size_t>(approx_explored.size(), MAX_N_SECTOR_READS);
    std::vector<uint32_t> rerank_ids;
    rerank_ids.reserve(n_fill);
    for (size_t i = 0; i < n_fill; ++i) {
        if (deletion_table_.is_deleted(approx_explored[i].id)) continue;
        rerank_ids.push_back(approx_explored[i].id);
    }

    // One aligned vector read per candidate, each into its own scratch slot.
    const uint64_t vec_stride = layout_.vec_read_stride();
    std::vector<IORequest> vec_reads;
    vec_reads.reserve(rerank_ids.size());
    for (size_t i = 0; i < rerank_ids.size(); ++i) {
        uint64_t aligned_off, aligned_len;
        layout_.vec_aligned_read(indirection_.id2vec_loc(rerank_ids[i]),
                                 aligned_off, aligned_len);
        vec_reads.emplace_back(aligned_off, aligned_len,
                               buf->vec_scratch + i * vec_stride);
    }

    // Windowed CASR (navis/casr.h). Until this path is calibrated (w == 0)
    // the rerank profiles: it loads every candidate in one batch and records
    // the smallest window that would not have stopped it early.
    CasrCalibrator& cal = is_position_seeking ? casr_cal_insert_ : casr_cal_search_;
    const uint32_t window = cal.window();
    const uint32_t n_cand = static_cast<uint32_t>(rerank_ids.size());
    CasrTracker tracker(kCasrTopK, window, n_cand);

    results.clear();
    results.reserve(rerank_ids.size());
    tsl::robin_set<uint32_t> rerank_set;
    rerank_set.reserve(rerank_ids.size());

    // Vectors are compared one at a time in PQ order. When the comparison
    // reaches the last issued read, everything currently needed (up to the
    // last top-K change + w) is read speculatively as one batch.
    uint32_t issued = 0;
    while (!tracker.done()) {
        const uint32_t i = tracker.loaded();
        if (i == issued) {
            const uint32_t end = tracker.needed();
            std::vector<IORequest> reqs(vec_reads.begin() + issued,
                                        vec_reads.begin() + end);
            io_vec_->read_batch(reqs);
            issued = end;
        }
        const uint32_t rid = rerank_ids[i];
        const uint64_t vf_off = layout_.vec_file_offset(indirection_.id2vec_loc(rid));
        const T* vec = reinterpret_cast<const T*>(
            buf->vec_scratch + i * vec_stride + (vf_off - vec_reads[i].offset));
        float exact = dist_cmp_->compare(buf->aligned_query, vec,
                                         static_cast<uint32_t>(layout_.data_dim));
        results.emplace_back(rid, exact);
        rerank_set.insert(rid);
        tracker.add(Neighbor(rid, exact));
        // Position seeking: keep the exact vector so neighbor pruning uses
        // exact distances between reranked candidates.
        if (coord_map) {
            T* stored = buf->coord_scratch + buf->coord_idx * aligned_dim_;
            std::memcpy(stored, vec, layout_.data_dim * sizeof(T));
            (*coord_map)[rid] = stored;
            buf->coord_idx++;
        }
    }
    const uint32_t vec_loads = issued;

    if (window == 0 && n_cand > 0 && config_.warmup_queries > 0) {
        cal.record(tracker.min_safe_window(), config_.warmup_queries,
                   is_position_seeking ? config_.casr_percentile_insert
                                       : config_.casr_percentile_search);
    }
    if (stats) {
        stats->n_ios += vec_loads;
        stats->n_vec_loads += vec_loads;
    }
    auto& loads_total = is_position_seeking ? insert_vec_loads_total_ : search_vec_loads_total_;
    auto& loads_calls = is_position_seeking ? insert_vec_loads_calls_ : search_vec_loads_calls_;
    loads_total.fetch_add(vec_loads, std::memory_order_relaxed);
    loads_calls.fetch_add(1, std::memory_order_relaxed);

    // Position seeking: append the explored vertices that were not reranked,
    // with their PQ distance and a deflated-PQ (reconstructed) vector in
    // coord_map. prune_neighbors treats exact and reconstructed vectors
    // uniformly, which yields a full R-sized neighbor list. User search must
    // NOT do this: mixing PQ and exact distances in one sort breaks top-K.
    if (is_position_seeking && coord_map) {
        for (uint64_t i = 0; i < cur_list_size; ++i) {
            uint32_t rid = retset[i].id;
            if (rid == UINT32_MAX) continue;
            if (deletion_table_.is_deleted(rid)) continue;
            if (rerank_set.count(rid)) continue;
            if (coord_map->count(rid)) {
                results.emplace_back(rid, retset[i].distance);
                continue;
            }
            T* stored = buf->coord_scratch + buf->coord_idx * aligned_dim_;
            buf->coord_idx++;
            (*coord_map)[rid] = stored;
            pq_table_.inflate_vector(
                pq_data_.data() + static_cast<uint64_t>(rid) * n_chunks_, stored);
            results.emplace_back(rid, retset[i].distance);
        }
    }

    if (deletion_table_.deleted_count() > 0) {
        results.erase(std::remove_if(results.begin(), results.end(),
            [this](const Neighbor& n) {
                return n.id == UINT32_MAX || deletion_table_.is_deleted(n.id);
            }), results.end());
    }

    std::sort(results.begin(), results.end());
}

// Explicit instantiations
template void NavisIndex<float>::do_search(const float*, uint32_t, uint64_t, uint32_t,
    std::vector<Neighbor>&, tsl::robin_map<uint32_t, float*>*, QueryStats*, bool, QueryBuffer<float>*,
    std::vector<Neighbor>*, std::vector<Neighbor>*);
template void NavisIndex<int8_t>::do_search(const int8_t*, uint32_t, uint64_t, uint32_t,
    std::vector<Neighbor>&, tsl::robin_map<uint32_t, int8_t*>*, QueryStats*, bool, QueryBuffer<int8_t>*,
    std::vector<Neighbor>*, std::vector<Neighbor>*);
template void NavisIndex<uint8_t>::do_search(const uint8_t*, uint32_t, uint64_t, uint32_t,
    std::vector<Neighbor>&, tsl::robin_map<uint32_t, uint8_t*>*, QueryStats*, bool, QueryBuffer<uint8_t>*,
    std::vector<Neighbor>*, std::vector<Neighbor>*);

} // namespace navis
