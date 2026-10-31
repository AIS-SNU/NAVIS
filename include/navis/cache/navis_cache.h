#pragma once

#include "navis/types.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace navis {

// ---------------------------------------------------------------------------
// NavisCache — NAVIS-Cache, the prioritized in-memory edgelist-page cache.
//
//   - Mostly-frozen region (90% by default): protected from eviction, so
//     one-off traversal accesses cannot pollute it. When full, the victim is
//     the earliest-promoted page (FIFO over promotion order, O(1)).
//   - Admission window (10%): LRU. A page must be accessed at least
//     `promote_threshold` times in the window to be promoted to frozen.
//   - Write-through: dirty pages live in the RMW PageCache; the background
//     writer refreshes resident entries via update_if_present() after the
//     write reaches storage, so NAVIS-Cache never holds a page newer than disk.
//
// Storage: an open-addressed hash table per region with seqlock-protected
// reads and a single writer mutex. Writes (promote, evict) are rare; reads
// (every traversal edge probe) are lock-free: a few atomic loads plus one
// atomic shared_ptr copy. CacheSlot::buf_sp is never mutated in place —
// update_if_present swaps the whole shared_ptr, so readers keep a consistent
// immutable snapshot validated by the per-slot seqlock version.
// ---------------------------------------------------------------------------
class NavisCache {
public:
    NavisCache() = default;

    void init(const CacheConfig& config);

    // Look up an edge sector. Returns true if hit.
    bool get(uint64_t sector_no, uint8_t* value);

    // Pointer-return variant: on hit, captures the cached page's
    // shared_ptr<const PageBuf> into *out_sp. Avoids the 4 KB memcpy
    // that get() pays — caller can read directly from out_sp->data().
    // The shared_ptr keeps the page alive even if a concurrent
    // update_if_present swaps the cache's pointer; readers see the
    // immutable snapshot they captured. Search path uses this; insert
    // READ_TIER2 cannot (RMW needs a writable 4 KB buffer to compose
    // and write back to disk).
    using PageBufSp = std::shared_ptr<const std::array<uint8_t, EDGE_SECTOR_LEN>>;
    // Per-call promote threshold lets the caller widen the window→frozen
    // gate. Insert position-seeking uses the configured default (2) — its
    // pages are reused so it benefits from sticky frozen residency.
    // Search uses a much larger threshold so transient query-side accesses
    // don't churn the frozen FIFO and starve insert (or earlier search)
    // working sets out of the cache.
    bool get_sp(uint64_t sector_no, PageBufSp* out_sp,
                uint32_t promote_threshold_override = 0);

    // Admit a new entry on miss; insert into admission window with freq=1.
    // On hit (window or frozen): no-op. Admission is read-driven only;
    // write-path updates go through update_if_present instead.
    void populate(uint64_t sector_no, const uint8_t* value);

    // Write-through update: if the entry exists in window or frozen, swap
    // its shared_ptr to a fresh page holding the new bytes. Does NOT bump
    // frequency, does NOT promote, does NOT admit on miss. Miss → no-op.
    // Called by bg writer per flushed edge sector.
    void update_if_present(uint64_t sector_no, const uint8_t* value);

    // Remove entry from window or frozen if present. No-op if absent.
    void evict(uint64_t sector_no);

    // Reset hit/miss counters to zero (for per-batch reporting).
    void reset_stats() {
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
    }

    // Statistics
    uint64_t hits()      const { return hits_.load(std::memory_order_relaxed); }
    uint64_t misses()    const { return misses_.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return total_evictions_.load(std::memory_order_relaxed); }
    size_t   size()      const { return frozen_count_.load() + window_count_.load(); }
    size_t   size_window() const { return window_count_.load(std::memory_order_relaxed); }
    size_t   size_frozen() const { return frozen_count_.load(std::memory_order_relaxed); }

    bool is_enabled() const { return enabled_; }

private:
    using PageBuf = std::array<uint8_t, EDGE_SECTOR_LEN>;

    // Per-slot seqlock-protected entry.
    //
    // Reader pattern:
    //   v1 = version.load(acq); if (v1 & 1) retry;
    //   occ = occupied; k = key; sp = atomic_load(&buf_sp); freq = frequency;
    //   v2 = version.load(acq); if (v1 != v2) retry;
    //
    // Writer pattern (under writer_mu_):
    //   version.fetch_add(1);                 // → odd
    //   mutate fields (atomic_store for buf_sp);
    //   version.fetch_add(1);                 // → even
    struct Slot {
        std::atomic<uint64_t>          version{0};
        uint64_t                       key{0};          // sector no
        bool                           occupied{false};
        std::shared_ptr<const PageBuf> buf_sp;          // atomic-loaded by readers
        uint8_t                        frequency{0};    // only used in window
    };

    struct Region {
        std::vector<Slot>    slots;
        std::size_t          mask = 0;           // slots.size()-1 (pow2)
        std::size_t          max_entries = 0;
        std::atomic<std::size_t> count{0};
        std::mutex           writer_mu;
    };

    Region frozen_region_;
    Region window_region_;

    // Frozen FIFO for O(1) eviction (promoted-oldest at front).
    std::deque<uint64_t> frozen_fifo_;
    std::mutex           frozen_fifo_mu_;

    // Window LRU + lookup.
    std::list<uint64_t>                              window_lru_;
    std::unordered_map<uint64_t,
        std::list<uint64_t>::iterator>               window_lru_map_;
    std::mutex                                       window_lru_mu_;

    // Counters
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> total_evictions_{0};
    std::atomic<std::size_t>& frozen_count_ = frozen_region_.count;
    std::atomic<std::size_t>& window_count_ = window_region_.count;

    // --- Config ---
    uint32_t promote_threshold_ = 2;
    bool     enabled_           = false;
    std::size_t max_window_     = 0;
    std::size_t max_frozen_     = 0;

    // --- Random eviction ---
    thread_local static std::mt19937 tl_rng_;

    // --- Probe / seqlock internals ---
    static constexpr std::size_t kMaxProbe = 64;
    static constexpr int         kMaxRetry = 8;

    static uint64_t hash_key(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    void region_init(Region& r, std::size_t max_entries);

    // Seqlock read. Returns true if key found; copies shared_ptr into *out_sp
    // and frequency into *out_freq (if non-null).
    bool region_find(const Region& r, uint64_t key,
                     std::shared_ptr<const PageBuf>* out_sp,
                     uint8_t* out_freq) const;

    // Under-writer-mu find: returns slot pointer or nullptr.
    Slot* region_find_locked(Region& r, uint64_t key) const;

    // Insert-or-fail under writer_mu. Returns true if inserted.
    bool region_insert_locked(Region& r, uint64_t key,
                              std::shared_ptr<const PageBuf> buf_sp,
                              uint8_t frequency);

    // Mutate in place under writer_mu. Returns true if entry found.
    // buf_sp is swapped via atomic_store; frequency set via seqlock.
    bool region_swap_buf_locked(Region& r, uint64_t key,
                                std::shared_ptr<const PageBuf> new_buf);

    // Increment frequency + optionally read current. Returns true if found.
    // Sets *out_freq to the new (post-increment) value.
    bool region_bump_freq_and_get(Region& r, uint64_t key,
                                   std::shared_ptr<const PageBuf>* out_sp,
                                   uint8_t* out_freq);

    // Erase by key. Caller holds writer_mu.
    bool region_erase_locked(Region& r, uint64_t key);

    // After marking slot i empty under writer_mu, shift probe-displaced
    // successors back to preserve linear-probe invariants.
    void region_fixup_after_erase_locked(Region& r, std::size_t i);

    // W-TinyLFU policy helpers
    void promote_to_frozen(uint64_t sector_no,
                           std::shared_ptr<const PageBuf> buf_sp,
                           uint8_t freq);
    void evict_one_frozen();
    void evict_window_lru();
};

} // namespace navis
