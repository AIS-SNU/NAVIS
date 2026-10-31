#pragma once

#include "navis/types.h"
#include <libcuckoo/cuckoohash_map.hh>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <vector>

namespace navis { class NavisCache; }

namespace navis {

// ---------------------------------------------------------------------------
// IndirectionTable — host-memory id ↔ loc translation for both edges and
// vectors.
//
// Sentinel constants
// ------------------
//   kInvalidID   = UINT32_MAX     — slot is free (no owner)
//   kAllocatedID = UINT32_MAX - 1 — slot is claimed (owner not yet bound)
//
// Edge side (id → edge_loc)
// -------------------------
// Two modes:
//   - Identity  (is_identity_ = true): id2edge_loc(id) returns id. No backing
//     array is allocated. Used when no `{prefix}_indirection.bin` exists.
//   - Permuted (is_identity_ = false): id2edge_loc(id) reads from a dense
//     uint32_t array. Populated by the build from a Starling partition file;
//     inserts append via the 3-tier allocator.
//
// Vector side (id → vec_loc)
// --------------------------
//   - Identity  (vec_is_identity_ = true): id2vec_loc(id) returns id. Default.
//   - Permuted  (vec_is_identity_ = false): reads from a dense uint32_t array.
//
// Reverse edge map (edge_loc → id)
// ----------------------------------
//   loc2id_base_[edge_loc] holds the id that currently owns that slot, or
//   kInvalidID if the slot is free.  Used by the 3-tier allocator to reclaim
//   freed slots.  Mirrors the same atomic-pointer + retired-buffers growth
//   pattern as the forward maps.
//
// 3-tier edge-slot allocator (alloc_edge_locs)
// ---------------------------------------------
//   Tier 1: reuse a page where all other slots are already loaded (cache hot).
//   Tier 2: reclaim a slot from the empty_pages_queue_ FIFO (returned pages).
//   Tier 3: append fresh slots at the tail (cur_edge_loc_).
//   Returns `n` edge_loc values.  hint_pages is the set of page numbers
//   already in the RMW page cache (populated by the caller); on return
//   pages_need_to_read is filled with pages not in cache that must be
//   fetched before the slots can be written.
//
// Empty-pages queue
// -----------------
//   When invalidate_edge_loc(loc) is called (vertex deleted), the enclosing
//   page is checked: if all its slots become free it is pushed onto
//   empty_pages_queue_.  empty_pages_index_ (libcuckoo) prevents duplicates.
//
// Growth
// ------
//   Atomic base pointer + mutex-protected doubling. Retired buffers are kept
//   alive until destruction so readers that captured a stale pointer remain
//   valid.  alloc_lock_ serializes alloc + invalidate; grow_mu_ serializes
//   buffer growth (held separately, never simultaneously with alloc_lock_).
//
// Persistence file format ({prefix}_indirection.bin):
//   [u64 npts][u32 * npts id2edge_loc][u32 * npts id2vec_loc]
//   (If file was written by an older build with only edge locs, vec_loc
//   section may be absent; callers should fall back to identity for vec.)
// ---------------------------------------------------------------------------
class IndirectionTable {
public:
    // Sentinel values
    static constexpr uint32_t kInvalidID   = UINT32_MAX;       // slot is free
    static constexpr uint32_t kAllocatedID = UINT32_MAX - 1;   // claimed, not yet bound

    IndirectionTable() = default;
    ~IndirectionTable();

    IndirectionTable(const IndirectionTable&)            = delete;
    IndirectionTable& operator=(const IndirectionTable&) = delete;

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    // Put the edge side in identity mode (no backing array).
    // Also sets edge-side geometry if edges_per_page > 0.
    void init_identity(uint64_t npts);

    // Try to load {prefix}_indirection.bin.
    // Returns false if the file does not exist — caller should then call
    // init_identity().
    bool load(const std::string& prefix, uint64_t capacity_hint);

    // Persist current mapping (first `npts` entries only).
    // Only valid when the edge side is in permuted mode.
    void save(const std::string& prefix, uint64_t npts) const;

    // Set edge-page geometry needed by the allocator (edges_per_page).
    // Must be called before alloc_edge_locs() when in permuted mode.
    void set_geometry(uint32_t edges_per_page);

    // Build pipeline: populate both forward maps from permutation arrays.
    // `edge_perm[i]` = edge_loc for id i; `vec_perm[i]` = vec_loc for id i.
    // If vec_perm is nullptr the vector side stays in identity mode.
    void init_from_perm(const uint32_t* edge_perm, const uint32_t* vec_perm,
                        uint64_t npts);

    // -----------------------------------------------------------------------
    // Forward edge map — id → edge_loc
    // -----------------------------------------------------------------------

    // Lookup. Safe under concurrent set_edge_loc / grow.
    inline uint32_t id2edge_loc(uint32_t id) const {
        if (is_identity_.load(std::memory_order_acquire)) return id;
        return edge_base_.load(std::memory_order_acquire)[id];
    }

    // Write a forward-edge mapping entry. No-op under identity.
    void set_edge_loc(uint32_t id, uint32_t edge_loc);

    // Grow the edge forward map to at least new_cap entries.
    void ensure_edge_capacity(uint64_t new_cap);

    // Legacy alias — delegates to ensure_edge_capacity.
    inline void ensure_capacity(uint64_t new_cap) { ensure_edge_capacity(new_cap); }

    uint64_t edge_capacity() const { return capacity_.load(std::memory_order_relaxed); }
    uint64_t capacity()      const { return edge_capacity(); }

    bool is_identity() const { return is_identity_.load(std::memory_order_acquire); }

    uint64_t cur_edge_loc() const { return cur_edge_loc_.load(std::memory_order_relaxed); }

    // After loading from a partition file, set the high-water mark so
    // append-only allocation continues from the right offset.
    void set_cur_edge_loc(uint64_t cur) {
        cur_edge_loc_.store(cur, std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Forward vector map — id → vec_loc
    // -----------------------------------------------------------------------

    // Lookup. Safe under concurrent set_vec_loc / grow.
    inline uint32_t id2vec_loc(uint32_t id) const {
        if (vec_is_identity_.load(std::memory_order_acquire)) return id;
        return vec_base_.load(std::memory_order_acquire)[id];
    }

    // Atomically reserve the next vec_loc; used during identity-free inserts.
    inline uint32_t alloc_vec_loc(uint32_t target_id) {
        if (vec_is_identity_.load(std::memory_order_acquire)) return target_id;
        return static_cast<uint32_t>(
            cur_vec_loc_.fetch_add(1, std::memory_order_relaxed));
    }

    // Write a forward-vector mapping entry. No-op under identity.
    void set_vec_loc(uint32_t id, uint32_t vec_loc);

    bool vec_is_identity() const { return vec_is_identity_.load(std::memory_order_acquire); }

    // Grow the vector forward map to at least new_cap entries.
    void ensure_vec_capacity(uint64_t new_cap);

    // -----------------------------------------------------------------------
    // Reverse edge map — edge_loc → id
    // -----------------------------------------------------------------------

    // Returns the id that owns edge_loc, or kInvalidID if free.
    // Safe for concurrent reads.
    inline uint32_t edge_loc2id(uint32_t edge_loc) const {
        uint32_t* base = loc2id_base_.load(std::memory_order_acquire);
        if (!base) return kInvalidID;
        if (edge_loc >= loc_capacity_.load(std::memory_order_relaxed)) return kInvalidID;
        return base[edge_loc];
    }

    // Grow the reverse map to at least new_cap entries (fills new slots with
    // kInvalidID).
    void ensure_loc_capacity(uint64_t new_cap);

    // -----------------------------------------------------------------------
    // 3-tier edge-slot allocator
    // -----------------------------------------------------------------------

    // Allocate `n` edge_locs for a new vertex.
    //   hint_pages          — page numbers already held in the RMW cache
    //                         (tier-1 prefers pages in this set)
    //   pages_need_to_read  — on return: page numbers that must be fetched
    //                         before the caller can perform RMW writes
    // Returns a vector of `n` freshly reserved edge_locs.
    // Requires set_geometry() to have been called once with edges_per_page.
    // Throws std::invalid_argument for negative n and std::overflow_error if
    // Tier 3 cannot append and page-pad within [0, kAllocatedID).
    // Thread-safe (protected by alloc_lock_).
    std::vector<uint32_t> alloc_edge_locs(int n,
                                          const std::set<uint64_t>& hint_pages,
                                          std::set<uint64_t>& pages_need_to_read);

    // Simple single-slot allocator (legacy / identity path).
    // Under identity, returns target_id. Under permuted, atomically appends at
    // the tail without going through the 3-tier logic.
    inline uint32_t alloc_edge_loc(uint32_t target_id) {
        if (is_identity_.load(std::memory_order_acquire)) return target_id;
        return static_cast<uint32_t>(
            cur_edge_loc_.fetch_add(1, std::memory_order_relaxed));
    }

    // -----------------------------------------------------------------------
    // Bind / Invalidate
    // -----------------------------------------------------------------------

    // Complete an alloc_edge_locs() reservation: write the forward mapping
    // (id → edge_loc) and the reverse mapping (edge_loc → id), replacing the
    // kAllocatedID sentinel with the real id.
    void bind_edge_loc(uint32_t id, uint32_t edge_loc);

    // Release an edge_loc: mark it free in the reverse map and, if the page
    // becomes fully empty, push it onto empty_pages_queue_.
    // Protected by alloc_lock_.
    void invalidate_edge_loc(uint32_t edge_loc);

    // Non-owning pointer to the NAVIS-Cache, set by NavisIndex::load(). Used
    // by invalidate_edge_loc to emit an eviction hint when every slot of a
    // page has been invalidated. Forward-declared to avoid an include cycle.
    void set_navis_cache(NavisCache* nc) { navis_cache_ = nc; }

private:
    // --- Edge forward map ---
    std::atomic<bool>                        is_identity_{true};
    std::atomic<uint32_t*>                   edge_base_{nullptr};
    std::atomic<uint64_t>                    capacity_{0};
    std::atomic<uint64_t>                    cur_edge_loc_{0};
    std::vector<std::unique_ptr<uint32_t[]>> retired_edge_;

    // --- Vector forward map ---
    std::atomic<bool>                        vec_is_identity_{true};
    std::atomic<uint32_t*>                   vec_base_{nullptr};
    std::atomic<uint64_t>                    vec_capacity_{0};
    std::atomic<uint64_t>                    cur_vec_loc_{0};
    std::vector<std::unique_ptr<uint32_t[]>> retired_vec_;

    // --- Reverse edge map (edge_loc → id) ---
    std::atomic<uint32_t*>                   loc2id_base_{nullptr};
    std::atomic<uint64_t>                    loc_capacity_{0};
    std::vector<std::unique_ptr<uint32_t[]>> retired_loc_;

    // --- Allocator geometry ---
    uint32_t                                 edges_per_page_{0};

    // --- Alloc / invalidate serialization ---
    std::mutex                               alloc_lock_;

    // --- Empty pages queue (recycled from invalidate_edge_loc) ---
    // Protected by alloc_lock_. empty_pages_index_ deduplicates so that a
    // page that is invalidated multiple times is only queued once.
    std::queue<uint64_t>                     empty_pages_queue_;
    libcuckoo::cuckoohash_map<uint64_t, bool> empty_pages_index_;

    // --- Growth serialization (separate from alloc_lock_) ---
    std::mutex                               grow_mu_;

    // --- Private growth helpers ---
    void grow_edge_to_locked(uint64_t new_cap);
    void grow_vec_to_locked(uint64_t new_cap);
    void grow_loc_to_locked(uint64_t new_cap);

    // Legacy internal alias kept for existing .cpp callers
    void grow_to_locked(uint64_t new_cap) { grow_edge_to_locked(new_cap); }

    // Non-owning pointer, set by NavisIndex::load(). Used by
    // invalidate_edge_loc to evict fully-invalidated pages.
    NavisCache* navis_cache_ = nullptr;
};

} // namespace navis
