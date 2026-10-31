#include "navis/index/navis_index.h"
#include "navis/io/uring_backend.h"
#include "navis/pq_table.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace navis {

// ===========================================================================
// compute_pq_dists_internal — PQ distance from one source to multiple targets
// ===========================================================================
template <typename T>
void NavisIndex<T>::compute_pq_dists_internal(uint32_t src, const uint32_t* ids,
                                               float* dists, uint32_t count,
                                               uint8_t* scratch) const {
    // Sanity: count should never exceed R+1. If huge, we're being fed garbage
    // from a corrupted state upstream. Abort with diagnostic.
    if (count > layout_.max_degree + 2) {
        std::fprintf(stderr,
            "[compute_pq_dists_internal] bogus count=%u (max expected=%lu), src=%u, ids[0]=%u\n",
            count, layout_.max_degree + 2, src, count > 0 ? ids[0] : 0);
        std::abort();
    }
    thread_local std::vector<T> src_vec_T;
    thread_local std::vector<float> chunk_dists;
    uint64_t dim = pq_table_.get_ndims();

    if (src_vec_T.size() < dim) src_vec_T.resize(dim);
    if (chunk_dists.size() < NUM_PQ_CENTROIDS * n_chunks_)
        chunk_dists.resize(NUM_PQ_CENTROIDS * n_chunks_);

    const uint8_t* src_pq = pq_data_.data() + static_cast<uint64_t>(src) * n_chunks_;
    pq_table_.inflate_vector(src_pq, src_vec_T.data());
    pq_table_.populate_chunk_distances(src_vec_T.data(), chunk_dists.data());
    aggregate_pq_coords(ids, count, pq_data_.data(), n_chunks_, scratch);
    pq_dist_lookup(scratch, count, n_chunks_, chunk_dists.data(), dists);
}

// ===========================================================================
// precompute_src_chunk_dists — PQ distance table of a source vertex, so a
// loop-invariant source can reuse it across many lookups.
// ===========================================================================
template <typename T>
void NavisIndex<T>::precompute_src_chunk_dists(uint32_t src,
                                               float* chunk_dists_out) const {
    thread_local std::vector<T> src_vec_T;
    uint64_t dim = pq_table_.get_ndims();
    if (src_vec_T.size() < dim) src_vec_T.resize(dim);

    const uint8_t* src_pq = pq_data_.data() + static_cast<uint64_t>(src) * n_chunks_;
    pq_table_.inflate_vector(src_pq, src_vec_T.data());
    pq_table_.populate_chunk_distances(src_vec_T.data(), chunk_dists_out);
}

// ===========================================================================
// delta_prune — fast O(R) pruning for insertion overflow
// ===========================================================================
template <typename T>
void NavisIndex<T>::delta_prune(std::vector<TriangleNeighbor>& pool,
                                std::vector<uint32_t>& pruned_list,
                                uint8_t* scratch, int tgt_idx) {
    if (pool.size() != layout_.max_degree + 1) return;

    pruned_list.clear();
    constexpr float alpha = 1.2f;
    float cur_alpha = alpha;
    int to_evict = -1;
    float tgt_nbr_dis = pool[tgt_idx].distance;

    while (cur_alpha >= (1.0f - 1e-5f) && to_evict == -1) {
        for (int i = static_cast<int>(pool.size()) - 1; i >= 0; --i) {
            if (i == tgt_idx) continue;

            if (pool[i].distance > tgt_nbr_dis) {
                if (pool[i].tgt_dis > 0 && pool[i].distance / pool[i].tgt_dis > cur_alpha) {
                    to_evict = i;
                    break;
                }
            } else {
                if (pool[i].tgt_dis > 0 && tgt_nbr_dis / pool[i].tgt_dis > cur_alpha) {
                    to_evict = tgt_idx;
                    break;
                }
            }
        }
        cur_alpha /= 1.2f;
    }

    auto finish = [&]() {
        for (int i = 0; i < static_cast<int>(pool.size()); i++) {
            if (i != to_evict) {
                pruned_list.push_back(pool[i].id);
            }
        }
    };

    if (to_evict != -1) {
        finish();
        return;
    }

    std::vector<uint32_t> ids(pool.size());
    for (uint32_t i = 0; i < pool.size(); i++) ids[i] = pool[i].id;

    std::vector<float> dists(pool.size(), 0.0f);
    for (int start = 0; start < static_cast<int>(pool.size()); ++start) {
        if (start == tgt_idx) continue;

        compute_pq_dists_internal(pool[start].id,
                                   ids.data() + start + 1,
                                   dists.data() + start + 1,
                                   static_cast<uint32_t>(pool.size() - start - 1),
                                   scratch);

        for (uint32_t t = start + 1; t < pool.size(); t++) {
            if (dists[t] > 0 && pool[t].distance / dists[t] > alpha) {
                to_evict = t;
                finish();
                return;
            }
        }
    }

    to_evict = static_cast<int>(pool.size()) - 1;
    finish();
}

// ===========================================================================
// prune_neighbors — RNG-based neighbor pruning (occlusion, alpha=1.2)
//
// Distances between candidates use exact vectors when both are in coord_map
// and deflated-PQ distances otherwise; the PQ side is batched into one
// compute_pq_dists_internal call per candidate to amortize its preamble.
//
// After occlusion, any remaining slots up to R are filled from the explored
// set E (do_search appends it to `candidates` for position seeking) without
// further occlusion: these longer edges preserve graph navigability.
// ===========================================================================
template <typename T>
void NavisIndex<T>::prune_neighbors(const std::vector<Neighbor>& candidates,
                                     std::vector<uint32_t>& pruned,
                                     tsl::robin_map<uint32_t, T*>* coord_map) {
    pruned.clear();
    if (candidates.empty()) return;

    auto sorted = candidates;
    std::sort(sorted.begin(), sorted.end());

    constexpr float alpha = 1.2f;
    uint32_t max_deg = static_cast<uint32_t>(layout_.max_degree);

    thread_local std::vector<uint8_t>  scratch;
    thread_local std::vector<uint32_t> pq_ids;
    thread_local std::vector<float>    pq_dists;
    if (scratch.size() < n_chunks_ * (max_deg + 8)) {
        scratch.resize(n_chunks_ * (max_deg + 8));
    }

    for (auto& cand : sorted) {
        if (deletion_table_.is_deleted(cand.id)) continue;
        if (pruned.size() >= max_deg) break;

        bool occluded = false;
        const bool cand_loaded = coord_map && coord_map->count(cand.id);
        pq_ids.clear();

        if (cand_loaded) {
            const T* cand_vec = (*coord_map)[cand.id];
            for (uint32_t sel_id : pruned) {
                if (coord_map->count(sel_id)) {
                    float d = dist_cmp_->compare(cand_vec,
                                                  (*coord_map)[sel_id],
                                                  static_cast<uint32_t>(layout_.data_dim));
                    if (d * alpha < cand.distance) { occluded = true; break; }
                } else {
                    pq_ids.push_back(sel_id);
                }
            }
        } else {
            pq_ids.insert(pq_ids.end(), pruned.begin(), pruned.end());
        }

        if (!occluded && !pq_ids.empty()) {
            pq_dists.assign(pq_ids.size(), 0.0f);
            compute_pq_dists_internal(cand.id, pq_ids.data(),
                                      pq_dists.data(),
                                      static_cast<uint32_t>(pq_ids.size()),
                                      scratch.data());
            for (float d : pq_dists) {
                if (d * alpha < cand.distance) { occluded = true; break; }
            }
        }

        if (!occluded) pruned.push_back(cand.id);
    }

    // R-P fill from full E (already appended to candidates by do_search).
    if (pruned.size() < max_deg) {
        tsl::robin_set<uint32_t> in_pruned(pruned.begin(), pruned.end());
        for (auto& cand : sorted) {
            if (pruned.size() >= max_deg) break;
            if (deletion_table_.is_deleted(cand.id)) continue;
            if (in_pruned.insert(cand.id).second) {
                pruned.push_back(cand.id);
            }
        }
    }
}

// ===========================================================================
// do_insert — structural update with out-of-place edge writes.
//
// For the new vertex and each chosen neighbor n:
//   1. Allocate a FRESH edge slot (empty pages → pages with holes → tail).
//   2. Read n's current edge page from the RMW cache, NAVIS-Cache, or storage.
//   3. Copy n's edge list into the fresh slot, then append the new vertex
//      (or delta-prune on overflow).
//   4. Bind n → fresh slot in the indirection table and invalidate the old one.
// Only destination pages are written, and they are disjoint across concurrent
// insertions; the new vertex's vector is written once into its own slot.
// ===========================================================================
template <typename T>
int NavisIndex<T>::do_insert(uint32_t target_id, const T* point,
                              const std::vector<uint32_t>& neighbors,
                              QueryBuffer<T>* buf) {
    if (neighbors.empty()) return -1;

    uint32_t target_vec_loc = indirection_.id2vec_loc(target_id);

    // --- Edge-slot allocation ---
    // hint_pages = pages where neighbors' edges currently live (they may have
    // holes from earlier out-of-place writes that can be reused).
    std::set<uint64_t> hint_pages;
    for (uint32_t nbr : neighbors) {
        uint32_t loc = indirection_.id2edge_loc(nbr);
        hint_pages.insert(layout_.edge_page_no(loc));
    }

    std::set<uint64_t> need_to_read;
    std::vector<uint32_t> fresh_locs;

    fresh_locs = indirection_.alloc_edge_locs(
        static_cast<int>(neighbors.size()) + 1, hint_pages, need_to_read);
    // fresh_locs[i] (i < neighbors.size()) = new slot for neighbors[i]
    // fresh_locs.back() = new slot for target's own edge entry

    // --- Destination pages (only these are written, so only these are locked) ---
    std::set<uint64_t> dest_pages;
    for (auto l : fresh_locs) dest_pages.insert(layout_.edge_page_no(l));

    // --- Lock acquisition: ALL page locks first, then vertex locks ---
    // Compute vec lock key upfront so all page-level locks (edge dest + vec sector)
    // are acquired BEFORE any vertex lock, preventing cross-dependency deadlocks.
    constexpr uint64_t VEC_KEY_BIT = 1ULL << 63;
    uint64_t vec_aoff, vec_alen;
    layout_.vec_aligned_read(target_vec_loc, vec_aoff, vec_alen);
    uint64_t vec_sec = vec_aoff / SECTOR_LEN;
    uint64_t vec_lock_key = vec_sec | VEC_KEY_BIT;

    // Page locks: dest edge pages + vec sector. Dest edge pages are fresh
    // alloc'd (nearly disjoint per insert); vec sector is distinguished by
    // high-bit prefix so it can't collide with edge page numbers.
    std::vector<uint64_t> locked_dest_pages;
    locked_dest_pages.reserve(dest_pages.size());
    for (auto sec : dest_pages) {
        page_locks_.wrlock(sec);
        locked_dest_pages.push_back(sec);
    }
    page_locks_.wrlock(vec_lock_key);

    // Vertex locks: target + all neighbors, sorted ascending.
    std::vector<uint32_t> verts_to_lock;
    verts_to_lock.reserve(neighbors.size() + 1);
    verts_to_lock.push_back(target_id);
    for (auto n : neighbors) verts_to_lock.push_back(n);
    std::sort(verts_to_lock.begin(), verts_to_lock.end());
    verts_to_lock.erase(std::unique(verts_to_lock.begin(), verts_to_lock.end()),
                        verts_to_lock.end());
    for (auto v : verts_to_lock) vertex_locks_.lock(v);

    // --- Vector write: read-modify-write of the target's aligned window ---
    // The window spans one or two sectors (a high-dimensional vector may
    // straddle a sector boundary), so it gets its own DMA buffer sized to
    // vec_alen and goes straight to the bg writer, bypassing the RMW cache.
    char* vec_snapshot_buf = static_cast<char*>(navis_dma_alloc(vec_alen));
    {
        std::vector<IORequest> rs{IORequest(vec_aoff, vec_alen, vec_snapshot_buf)};
        io_vec_->read_batch(rs);

        uint64_t in_page = layout_.vec_file_offset(target_vec_loc) - vec_aoff;
        // in_page + data_dim*sizeof(T) <= vec_alen by construction of
        // vec_aligned_read, so this memcpy is always in-bounds.
        std::memcpy(vec_snapshot_buf + in_page, point,
                    layout_.data_dim * sizeof(T));
    }
    page_locks_.unlock(vec_lock_key);

    // --- Collect every unique edge page touched (sources + destinations) ---
    //
    // Each page is read into PRIVATE DMA scratch, mutated there, and only
    // then published to the RMW cache via put_dirty (a pointer swap), so a
    // concurrent reader never sees a partially composed page.
    uint32_t target_loc = fresh_locs.back();
    uint64_t target_sec = layout_.edge_page_no(target_loc);
    std::set<uint64_t> all_secs;
    all_secs.insert(target_sec);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        uint32_t old_loc_i = indirection_.id2edge_loc(neighbors[i]);
        if (old_loc_i != IndirectionTable::kInvalidID &&
            old_loc_i != IndirectionTable::kAllocatedID) {
            all_secs.insert(layout_.edge_page_no(old_loc_i));
        }
        all_secs.insert(layout_.edge_page_no(fresh_locs[i]));
    }

    // --- Allocate + populate one DMA scratch per unique page ---
    // Try RMW cache → NAVIS-Cache → storage. Storage reads are batched.
    // Scratches come from the thread-local 4 KiB pool; written ones are
    // handed to the bg writer, read-only ones go back to the pool.
    std::map<uint64_t, uint8_t*> page_scratch;
    {
        std::vector<IORequest> disk_reads;
        for (uint64_t sec : all_secs) {
            uint8_t* scratch = static_cast<uint8_t*>(navis_dma_alloc_4k());
            if (rmw_cache_.get(sec, scratch)) {
                // RMW hit: the newest bytes of a page still being written.
            } else if (navis_cache_.get(sec, scratch)) {
                // NAVIS-Cache hit.
            } else {
                // Miss on both caches: queue for the batched storage read.
                // Fresh pages past the end of the built file read back as
                // zeros (or short), which is fine: our slots are written
                // below and the whole page is published at the end.
                std::memset(scratch, 0, SECTOR_LEN);
                disk_reads.emplace_back(
                    layout_.edge_file_offset_from_sector(sec),
                    SECTOR_LEN,
                    reinterpret_cast<char*>(scratch));
            }
            page_scratch[sec] = scratch;
        }
        if (!disk_reads.empty()) {
            io_edge_->read_batch(disk_reads);
        }
    }

    // --- Compose the new vertex's own edge list ---
    //
    // Must happen BEFORE the neighbor loop: once bind_edge_loc(n, new_loc)
    // publishes a neighbor's new slot, that list contains target_id, and a
    // concurrent searcher following it must find target_id already bound to
    // valid bytes.
    {
        uint8_t* target_page = page_scratch[target_sec];
        char* target_entry = layout_.edge_in_page(
            reinterpret_cast<char*>(target_page), target_loc);
        uint32_t degree = static_cast<uint32_t>(neighbors.size());
        SeparatedLayout::set_edge_degree(target_entry, degree);
        std::memcpy(SeparatedLayout::edge_neighbors_mut(target_entry),
                    neighbors.data(), degree * sizeof(uint32_t));
        indirection_.bind_edge_loc(target_id, target_loc);
    }

    // Track which sectors actually received writes (target + each
    // neighbor's new_loc). Only these get put()'d and handed to bg.
    std::set<uint64_t> dst_secs_written;
    dst_secs_written.insert(target_sec);

    // --- Compose each neighbor's new slot ---
    //
    // target_id is loop-invariant, so its PQ distance table is computed once
    // here and reused by every overflow's delta prune.
    std::vector<float> target_chunk_dists(NUM_PQ_CENTROIDS * n_chunks_);
    precompute_src_chunk_dists(target_id, target_chunk_dists.data());

    // Per-thread scratch vectors for the overflow branch — avoids R allocations
    // per insert. Sized once and reused across iterations and across inserts.
    thread_local std::vector<uint32_t> tl_nhood;
    thread_local std::vector<float>    tl_tgt_dists;
    thread_local std::vector<float>    tl_nbr_dists;
    thread_local std::vector<TriangleNeighbor> tl_tri_pool;
    thread_local std::vector<uint32_t> tl_pruned_nhood;
    thread_local std::vector<float>    tl_n_chunk_dists;
    thread_local std::vector<uint8_t>  tl_nhood_pq_scratch;
    if (tl_n_chunk_dists.size() < NUM_PQ_CENTROIDS * n_chunks_)
        tl_n_chunk_dists.resize(NUM_PQ_CENTROIDS * n_chunks_);
    if (tl_nhood_pq_scratch.size() < n_chunks_ * (layout_.max_degree + 2))
        tl_nhood_pq_scratch.resize(n_chunks_ * (layout_.max_degree + 2));

    for (size_t i = 0; i < neighbors.size(); ++i) {
        uint32_t n       = neighbors[i];
        uint32_t old_loc = indirection_.id2edge_loc(n);
        uint32_t new_loc = fresh_locs[i];

        // Defensive: skip neighbors with sentinel mapping (another insert
        // is mid-bind for n).
        if (old_loc == IndirectionTable::kInvalidID ||
            old_loc == IndirectionTable::kAllocatedID) {
            continue;
        }

        uint64_t src_sec = layout_.edge_page_no(old_loc);
        uint64_t dst_sec = layout_.edge_page_no(new_loc);

        uint8_t* src_page = page_scratch[src_sec];
        uint8_t* dst_page = page_scratch[dst_sec];
        if (!src_page || !dst_page) {
            std::fprintf(stderr,
                "[insert %u] missing scratch src_sec=%lu (%p) dst_sec=%lu (%p)\n",
                target_id, (unsigned long)src_sec, (void*)src_page,
                (unsigned long)dst_sec, (void*)dst_page);
            std::abort();
        }

        char* src_entry = layout_.edge_in_page(reinterpret_cast<char*>(src_page), old_loc);
        char* dst_entry = layout_.edge_in_page(reinterpret_cast<char*>(dst_page), new_loc);
        std::memcpy(dst_entry, src_entry, layout_.max_edge_len);

        uint32_t deg = SeparatedLayout::edge_degree(dst_entry);
        if (deg > layout_.max_degree) {
            // Garbage degree from a fresh zero'd sector where the slot
            // happens to overlap what used to be data. Reset and treat
            // as empty.
            deg = 0;
            SeparatedLayout::set_edge_degree(dst_entry, 0);
        }
        if (deg < layout_.max_degree) {
            uint32_t* dnbrs = SeparatedLayout::edge_neighbors_mut(dst_entry);
            dnbrs[deg] = target_id;
            SeparatedLayout::set_edge_degree(dst_entry, deg + 1);
        } else {
            // Overflow → delta prune.
            uint32_t* dnbrs = SeparatedLayout::edge_neighbors_mut(dst_entry);
            uint32_t deg_recheck = SeparatedLayout::edge_degree(dst_entry);
            if (deg_recheck != deg || deg_recheck > layout_.max_degree) {
                std::fprintf(stderr,
                    "[insert %u] degree corruption: deg=%u recheck=%u max=%lu "
                    "n=%u new_loc=%u dst_sec=%lu\n",
                    target_id, deg, deg_recheck, layout_.max_degree,
                    n, new_loc, (unsigned long)dst_sec);
                std::abort();
            }
            tl_nhood.assign(dnbrs, dnbrs + deg);
            tl_nhood.push_back(target_id);
            const uint32_t nhood_n = static_cast<uint32_t>(tl_nhood.size());
            tl_tgt_dists.assign(nhood_n, 0.0f);
            tl_nbr_dists.assign(nhood_n, 0.0f);

            // Gather the neighborhood's PQ codes once, then look up distances
            // from the new vertex (hoisted table) and from n.
            aggregate_pq_coords(tl_nhood.data(), nhood_n,
                                pq_data_.data(), n_chunks_,
                                tl_nhood_pq_scratch.data());
            pq_dist_lookup(tl_nhood_pq_scratch.data(), nhood_n, n_chunks_,
                           target_chunk_dists.data(), tl_tgt_dists.data());
            precompute_src_chunk_dists(n, tl_n_chunk_dists.data());
            pq_dist_lookup(tl_nhood_pq_scratch.data(), nhood_n, n_chunks_,
                           tl_n_chunk_dists.data(), tl_nbr_dists.data());

            tl_tri_pool.resize(nhood_n);
            for (uint32_t kk = 0; kk < nhood_n; kk++) {
                tl_tri_pool[kk].id       = tl_nhood[kk];
                tl_tri_pool[kk].tgt_dis  = tl_tgt_dists[kk];
                tl_tri_pool[kk].distance = tl_nbr_dists[kk];
            }
            std::sort(tl_tri_pool.begin(), tl_tri_pool.end());

            int tgt_idx = -1;
            for (int kk = 0; kk < static_cast<int>(tl_tri_pool.size()); kk++) {
                if (tl_tri_pool[kk].id == target_id) { tgt_idx = kk; break; }
            }
            if (tgt_idx == -1) tgt_idx = static_cast<int>(tl_tri_pool.size()) - 1;

            tl_pruned_nhood.clear();
            delta_prune(tl_tri_pool, tl_pruned_nhood, buf->pq_coord_scratch, tgt_idx);
            uint32_t new_deg = static_cast<uint32_t>(tl_pruned_nhood.size());
            SeparatedLayout::set_edge_degree(dst_entry, new_deg);
            std::memcpy(SeparatedLayout::edge_neighbors_mut(dst_entry),
                        tl_pruned_nhood.data(), new_deg * sizeof(uint32_t));
        }

        indirection_.bind_edge_loc(n, new_loc);
        indirection_.invalidate_edge_loc(old_loc);
        dst_secs_written.insert(dst_sec);
    }

    // --- Publish written pages to the RMW cache; free read-only scratches ---
    //
    // put_dirty pins each page until the bg writer's mark_clean_and_deref.
    // NAVIS-Cache is NOT updated here: the bg writer refreshes resident
    // entries after the write reaches storage, and readers probe the RMW
    // cache first, so they still see the new bytes in the meantime.
    std::vector<std::pair<uint64_t, uint8_t*>> dirty_entries;
    for (uint64_t sec : dst_secs_written) {
        auto it = page_scratch.find(sec);
        if (it == page_scratch.end()) continue;
        rmw_cache_.put_dirty(sec, it->second);
        dirty_entries.emplace_back(sec, it->second);
        page_scratch.erase(it);
    }
    for (auto& [sec, scratch] : page_scratch) {
        (void)sec;
        navis_dma_free_4k(scratch);
    }
    page_scratch.clear();

    for (auto v : verts_to_lock) vertex_locks_.unlock(v);

    // --- Hand the writes to the bg writer ---
    // The composed scratches are already DMA-aligned: they become the write
    // buffers directly, owned (and freed) by the bg writer. The destination
    // page locks ride along and are released only after the write lands.
    auto* task = new BgWriteTask();
    for (auto& [sec, page] : dirty_entries) {
        uint64_t offset = layout_.edge_file_offset_from_sector(sec);
        task->edge_writes.emplace_back(offset, SECTOR_LEN,
                                       reinterpret_cast<char*>(page));
        task->owned_bufs.push_back(page);
        task->pages_to_deref.push_back(sec);
    }
    task->vec_writes.emplace_back(vec_aoff, vec_alen, vec_snapshot_buf);
    task->owned_bufs.push_back(vec_snapshot_buf);
    task->pages_to_unlock = std::move(locked_dest_pages);
    bg_write_queue_.push(task);
    return 0;
}

// Explicit instantiations
template void NavisIndex<float>::prune_neighbors(
    const std::vector<Neighbor>&, std::vector<uint32_t>&, tsl::robin_map<uint32_t, float*>*);
template void NavisIndex<int8_t>::prune_neighbors(
    const std::vector<Neighbor>&, std::vector<uint32_t>&, tsl::robin_map<uint32_t, int8_t*>*);
template void NavisIndex<uint8_t>::prune_neighbors(
    const std::vector<Neighbor>&, std::vector<uint32_t>&, tsl::robin_map<uint32_t, uint8_t*>*);

template int NavisIndex<float>::do_insert(uint32_t, const float*,
    const std::vector<uint32_t>&, QueryBuffer<float>*);
template int NavisIndex<int8_t>::do_insert(uint32_t, const int8_t*,
    const std::vector<uint32_t>&, QueryBuffer<int8_t>*);
template int NavisIndex<uint8_t>::do_insert(uint32_t, const uint8_t*,
    const std::vector<uint32_t>&, QueryBuffer<uint8_t>*);

template void NavisIndex<float>::compute_pq_dists_internal(uint32_t, const uint32_t*,
    float*, uint32_t, uint8_t*) const;
template void NavisIndex<int8_t>::compute_pq_dists_internal(uint32_t, const uint32_t*,
    float*, uint32_t, uint8_t*) const;
template void NavisIndex<uint8_t>::compute_pq_dists_internal(uint32_t, const uint32_t*,
    float*, uint32_t, uint8_t*) const;

template void NavisIndex<float>::precompute_src_chunk_dists(uint32_t, float*) const;
template void NavisIndex<int8_t>::precompute_src_chunk_dists(uint32_t, float*) const;
template void NavisIndex<uint8_t>::precompute_src_chunk_dists(uint32_t, float*) const;

template void NavisIndex<float>::delta_prune(std::vector<NavisIndex<float>::TriangleNeighbor>&,
    std::vector<uint32_t>&, uint8_t*, int);
template void NavisIndex<int8_t>::delta_prune(std::vector<NavisIndex<int8_t>::TriangleNeighbor>&,
    std::vector<uint32_t>&, uint8_t*, int);
template void NavisIndex<uint8_t>::delta_prune(std::vector<NavisIndex<uint8_t>::TriangleNeighbor>&,
    std::vector<uint32_t>&, uint8_t*, int);

} // namespace navis
