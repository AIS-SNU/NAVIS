#pragma once

// SECTOR_LEN is also defined by ssd_index.h and v2/lock_table.h in this
// codebase; we guard so downstream includes don't conflict.
#ifndef SECTOR_LEN
#define SECTOR_LEN 4096
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace v2 {

// ---------------------------------------------------------------------------
// OdinPackedCache — ported from NAVIS's NavisCache (W-TinyLFU).
//
// Edgelist-only cache with frozen/window design (W-TinyLFU):
//   - Mostly-frozen region (98%): protected from eviction, robust to
//     cache pollution from one-off graph traversal accesses.
//   - Admission window (2%): LRU-based. An edgelist must be accessed at
//     least `promote_threshold` times to be promoted to frozen.
//   - Frozen eviction: FIFO over promotion order (O(1) evict_one_frozen).
//   - Write-through: all write-back handled by PageCache (RMW cache).
//
// Storage design:
//
// Inline, custom open-addressed hash table with seqlock-protected reads
// and a global writer mutex. Writes are rare (promote from window,
// evict), reads are hot (every search/insert edge probe). Reader fast
// path: 3 atomic loads + 1 atomic shared_ptr copy (~30 ns). Writer
// path: brief exclusive work under a single mutex, version bump
// odd → mutate → version bump even.
//
// Key immutability invariant: CacheSlot::buf_sp is never mutated in
// place. `update_if_resident` swaps the entire shared_ptr via
// std::atomic_store. Readers obtain a consistent snapshot via
// std::atomic_load, validated by the per-slot seqlock version.
// ---------------------------------------------------------------------------

struct OdinPackedCacheConfig {
  uint64_t size_gb;           // total budget, GB. 0 → disabled.
  double   window_ratio;      // 0.02 matches NAVIS.
  uint32_t promote_threshold; // 2 matches NAVIS.
  uint32_t eviction_probes;   // 8 matches NAVIS.
};

class OdinPackedCache {
public:
    OdinPackedCache() = default;

    // Grouped statistics counters (atomic for lock-free reads from
    // hot paths). bytes_resident is maintained after each admission /
    // eviction as (frozen_count + window_count) * SECTOR_LEN.
    struct Stats {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> admissions{0};
        std::atomic<uint64_t> window_promotions{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<uint64_t> write_through_hits{0};
        std::atomic<uint64_t> bytes_resident{0};
    };
    const Stats& stats() const { return stats_; }

    void init(const OdinPackedCacheConfig& config);

    // Look up an edge sector. Returns true if hit.
    bool get(uint64_t sector_no, uint8_t* value);

    // Pointer-return variant: on hit, captures the cached page's
    // shared_ptr<const PageBuf> into *out_sp. Avoids the 4 KB memcpy
    // that get() pays — caller can read directly from out_sp->data().
    // The shared_ptr keeps the page alive even if a concurrent
    // update_if_resident swaps the cache's pointer; readers see the
    // immutable snapshot they captured. Search path uses this; insert
    // READ_TIER2 cannot (RMW needs a writable 4 KB buffer to compose
    // and write back to disk).
    using PageBufSp = std::shared_ptr<const std::array<uint8_t, SECTOR_LEN>>;
    bool get_sp(uint64_t sector_no, PageBufSp* out_sp);

    // Readonly variants: hit-path reads that DO NOT drive W-TinyLFU
    // admission state. No window frequency bump, no window→frozen
    // promotion. Designed for front-end query traffic, which has high
    // access rate but whose pattern should not crowd out insert-path
    // position-seeking from the cache. On miss, caller can still
    // admit() explicitly (front-end does, after SSD read).
    //
    // Rationale: benchmark query repetition would otherwise monopolize
    // W-TinyLFU promotion; this path makes admission insert-driven
    // while search freely reads whatever is resident.
    bool get_readonly(uint64_t sector_no, uint8_t* value);
    bool get_sp_readonly(uint64_t sector_no, PageBufSp* out_sp);

    // Admit a new entry on miss; insert into admission window with freq=1.
    // On hit (window or frozen): no-op. Admission is read-driven only;
    // write-path updates go through update_if_resident instead.
    void admit(uint64_t sector_no, const uint8_t* value);

    // Write-through update: if the entry exists in window or frozen, swap
    // its shared_ptr to a fresh page holding the new bytes. Does NOT bump
    // frequency, does NOT promote, does NOT admit on miss. Miss → no-op.
    // Called by bg writer per flushed edge sector.
    void update_if_resident(uint64_t sector_no, const uint8_t* value);

    // Remove entry from window or frozen if present. No-op if absent.
    void evict(uint64_t sector_no);

    // Invalidate every resident page while preserving configuration and
    // cumulative counters. Required when the backing index file is replaced.
    std::size_t clear();

    // Diagnostic / allocator hint: returns true if `sector_no` is currently
    // resident in either the frozen region or the admission window. Does
    // NOT count as an access. Cheap — two seqlock probes.
    bool is_cached(uint64_t sector_no) const;

    // Reset hit/miss counters to zero (for per-batch reporting).
    void reset_stats() {
        stats_.hits.store(0, std::memory_order_relaxed);
        stats_.misses.store(0, std::memory_order_relaxed);
    }

    // Legacy scalar accessors (kept so callers porting from NavisCache
    // continue to compile). These read from the grouped Stats struct.
    uint64_t hits()      const { return stats_.hits.load(std::memory_order_relaxed); }
    uint64_t misses()    const { return stats_.misses.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return stats_.evictions.load(std::memory_order_relaxed); }
    size_t   size()      const { return frozen_count_.load() + window_count_.load(); }
    size_t   size_window() const { return window_count_.load(std::memory_order_relaxed); }
    size_t   size_frozen() const { return frozen_count_.load(std::memory_order_relaxed); }

    bool is_enabled() const { return enabled_; }

private:
    using PageBuf = std::array<uint8_t, SECTOR_LEN>;

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
        std::atomic<uint64_t>          key{0};          // sector no
        std::atomic<bool>              occupied{false};
        std::shared_ptr<const PageBuf> buf_sp;          // atomic-loaded by readers
        std::atomic<uint8_t>           frequency{0};    // only used in window
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
    // Window hash and LRU metadata are both guarded by window_region_.writer_mu.

    // Counters (grouped).
    Stats stats_;
    std::atomic<std::size_t>& frozen_count_ = frozen_region_.count;
    std::atomic<std::size_t>& window_count_ = window_region_.count;

    // --- Config ---
    uint32_t promote_threshold_ = 2;
    uint32_t eviction_probes_   = 8;
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
    bool evict_window_lru_locked();

    // Keep bytes_resident counter in sync with region counts.
    void update_bytes_resident() {
        stats_.bytes_resident.store(
            (frozen_count_.load(std::memory_order_relaxed) +
             window_count_.load(std::memory_order_relaxed)) * SECTOR_LEN,
            std::memory_order_relaxed);
    }
};

}  // namespace v2
