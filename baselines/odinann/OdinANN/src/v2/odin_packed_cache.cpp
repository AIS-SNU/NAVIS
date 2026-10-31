#include "v2/odin_packed_cache.h"
#include <algorithm>

namespace v2 {

thread_local std::mt19937 OdinPackedCache::tl_rng_{std::random_device{}()};

// ---------------------------------------------------------------------------
// Region init — size slot array to next power-of-two ≥ 2 × max_entries
// (load factor capped at 0.5 to keep linear-probe chains short).
// ---------------------------------------------------------------------------
void OdinPackedCache::region_init(Region& r, std::size_t max_entries) {
    std::size_t target = 16;
    while (target < max_entries * 2) target <<= 1;
    r.slots = std::vector<Slot>(target);
    r.mask = target - 1;
    r.max_entries = max_entries;
    r.count.store(0, std::memory_order_relaxed);
}

void OdinPackedCache::init(const OdinPackedCacheConfig& config) {
    if (config.size_gb == 0) {
        enabled_ = false;
        return;
    }
    uint64_t total_sectors = (config.size_gb * 1024UL * 1024UL * 1024UL) / SECTOR_LEN;
    max_window_ = static_cast<std::size_t>(total_sectors * config.window_ratio);
    max_frozen_ = total_sectors - max_window_;
    promote_threshold_ = config.promote_threshold;
    eviction_probes_   = config.eviction_probes;
    enabled_ = true;

    region_init(frozen_region_, max_frozen_);
    region_init(window_region_, max_window_);
    update_bytes_resident();
}

// ---------------------------------------------------------------------------
// region_find — seqlock-protected read.
// ---------------------------------------------------------------------------
bool OdinPackedCache::region_find(const Region& r, uint64_t key,
                                   std::shared_ptr<const PageBuf>* out_sp,
                                   uint8_t* out_freq) const {
    if (r.slots.empty()) return false;
    std::size_t idx = hash_key(key) & r.mask;
    for (std::size_t p = 0; p < kMaxProbe; ++p) {
        const Slot& s = r.slots[(idx + p) & r.mask];
        for (int retry = 0; retry < kMaxRetry; ++retry) {
            uint64_t v1 = s.version.load(std::memory_order_acquire);
            if (v1 & 1) { std::this_thread::yield(); continue; }

            bool occ = s.occupied;
            uint64_t k = s.key;
            if (!occ) {
                uint64_t v2 = s.version.load(std::memory_order_acquire);
                if (v1 != v2) continue;
                return false;  // empty slot terminates probe
            }
            if (k != key) {
                uint64_t v2 = s.version.load(std::memory_order_acquire);
                if (v1 != v2) continue;
                break;  // wrong key, probe next slot
            }

            // Key matches → atomic shared_ptr copy.
            auto sp = std::atomic_load_explicit(&s.buf_sp,
                                                std::memory_order_acquire);
            uint8_t freq = s.frequency;
            uint64_t v2 = s.version.load(std::memory_order_acquire);
            if (v1 != v2) continue;
            if (!sp) return false;

            if (out_sp)   *out_sp   = std::move(sp);
            if (out_freq) *out_freq = freq;
            return true;
        }
    }
    return false;
}

OdinPackedCache::Slot* OdinPackedCache::region_find_locked(Region& r, uint64_t key) const {
    if (r.slots.empty()) return nullptr;
    std::size_t idx = hash_key(key) & r.mask;
    for (std::size_t p = 0; p < kMaxProbe; ++p) {
        Slot& s = r.slots[(idx + p) & r.mask];
        if (!s.occupied) return nullptr;
        if (s.key == key) return &s;
    }
    return nullptr;
}

bool OdinPackedCache::region_insert_locked(Region& r, uint64_t key,
                                            std::shared_ptr<const PageBuf> buf_sp,
                                            uint8_t frequency) {
    if (r.slots.empty()) return false;
    std::size_t idx = hash_key(key) & r.mask;
    for (std::size_t p = 0; p < kMaxProbe; ++p) {
        Slot& s = r.slots[(idx + p) & r.mask];
        if (s.occupied && s.key == key) return false;  // already present
        if (!s.occupied) {
            s.version.fetch_add(1, std::memory_order_release);   // odd
            s.key = key;
            std::atomic_store_explicit(&s.buf_sp, std::move(buf_sp),
                                       std::memory_order_release);
            s.frequency = frequency;
            s.occupied = true;
            s.version.fetch_add(1, std::memory_order_release);   // even
            r.count.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;  // probe chain full
}

bool OdinPackedCache::region_swap_buf_locked(Region& r, uint64_t key,
                                              std::shared_ptr<const PageBuf> new_buf) {
    Slot* s = region_find_locked(r, key);
    if (!s) return false;
    s->version.fetch_add(1, std::memory_order_release);          // odd
    std::atomic_store_explicit(&s->buf_sp, std::move(new_buf),
                               std::memory_order_release);
    s->version.fetch_add(1, std::memory_order_release);          // even
    return true;
}

bool OdinPackedCache::region_bump_freq_and_get(Region& r, uint64_t key,
                                                std::shared_ptr<const PageBuf>* out_sp,
                                                uint8_t* out_freq) {
    std::lock_guard<std::mutex> lk(r.writer_mu);
    Slot* s = region_find_locked(r, key);
    if (!s) return false;
    s->version.fetch_add(1, std::memory_order_release);          // odd
    if (s->frequency < 255) s->frequency++;
    uint8_t new_freq = s->frequency;
    auto sp = std::atomic_load_explicit(&s->buf_sp,
                                        std::memory_order_acquire);
    s->version.fetch_add(1, std::memory_order_release);          // even
    if (!sp) return false;
    if (out_sp)   *out_sp   = std::move(sp);
    if (out_freq) *out_freq = new_freq;
    return true;
}

bool OdinPackedCache::region_erase_locked(Region& r, uint64_t key) {
    if (r.slots.empty()) return false;
    std::size_t idx = hash_key(key) & r.mask;
    for (std::size_t p = 0; p < kMaxProbe; ++p) {
        std::size_t i = (idx + p) & r.mask;
        Slot& s = r.slots[i];
        if (!s.occupied) return false;  // gap → not present
        if (s.key == key) {
            s.version.fetch_add(1, std::memory_order_release);   // odd
            std::atomic_store_explicit(&s.buf_sp,
                std::shared_ptr<const PageBuf>{},
                std::memory_order_release);
            s.occupied = false;
            s.key = 0;
            s.frequency = 0;
            s.version.fetch_add(1, std::memory_order_release);   // even
            r.count.fetch_sub(1, std::memory_order_relaxed);
            region_fixup_after_erase_locked(r, i);
            return true;
        }
    }
    return false;
}

// Close the probe-chain gap at `i` by shifting back any slot whose
// natural position is ≤ i (modularly). Preserves the invariant that
// find() stops at the first empty slot.
void OdinPackedCache::region_fixup_after_erase_locked(Region& r, std::size_t i) {
    std::size_t empty_pos = i;
    for (std::size_t step = 1; step < kMaxProbe; ++step) {
        std::size_t j = (i + step) & r.mask;
        Slot& sj = r.slots[j];
        if (!sj.occupied) return;

        std::size_t desired = hash_key(sj.key) & r.mask;
        bool in_segment =
            (empty_pos <= j)
                ? (desired <= empty_pos || desired > j)
                : (desired <= empty_pos && desired > j);
        if (in_segment) {
            Slot& dst = r.slots[empty_pos];
            dst.version.fetch_add(1, std::memory_order_release);  // odd
            dst.key.store(sj.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
            auto sp = std::atomic_load_explicit(&sj.buf_sp,
                                                std::memory_order_acquire);
            std::atomic_store_explicit(&dst.buf_sp, std::move(sp),
                                       std::memory_order_release);
            dst.frequency.store(sj.frequency.load(std::memory_order_relaxed), std::memory_order_relaxed);
            dst.occupied = true;
            dst.version.fetch_add(1, std::memory_order_release);  // even

            sj.version.fetch_add(1, std::memory_order_release);   // odd
            std::atomic_store_explicit(&sj.buf_sp,
                std::shared_ptr<const PageBuf>{},
                std::memory_order_release);
            sj.occupied = false;
            sj.key = 0;
            sj.frequency = 0;
            sj.version.fetch_add(1, std::memory_order_release);   // even

            empty_pos = j;
        }
    }
}

// ---------------------------------------------------------------------------
// get — frozen-first probe (hot path, lockless), window-second (with
// frequency bump under writer_mu for promotion tracking).
// ---------------------------------------------------------------------------
bool OdinPackedCache::get(uint64_t sector_no, uint8_t* value) {
    if (!enabled_) return false;

    // Frozen: pure seqlock read, no lock acquisition.
    std::shared_ptr<const PageBuf> sp;
    if (region_find(frozen_region_, sector_no, &sp, nullptr)) {
        std::memcpy(value, sp->data(), SECTOR_LEN);
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Window: bump frequency under writer_mu; possibly promote.
    uint8_t freq = 0;
    if (region_bump_freq_and_get(window_region_, sector_no, &sp, &freq)) {
        std::memcpy(value, sp->data(), SECTOR_LEN);
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        if (freq >= promote_threshold_) {
            promote_to_frozen(sector_no, sp, freq);
        }
        return true;
    }

    stats_.misses.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// ---------------------------------------------------------------------------
// get_sp — pointer-return variant of get.
//
// Captures the cached page's shared_ptr<const PageBuf> instead of
// memcpy-ing the 4 KB. Readers consume the page bytes directly via
// (*out_sp)->data(). Saves ~25 GB/s of cache-page memcpy on the
// search-side workload (where each probe consumes ~388 bytes from a
// 4 KB page → 10× write amplification eliminated).
//
// Frozen-region path is the most common (~85% of cache hits): a single
// seqlock-protected atomic_load on the buf_sp. Window-region path goes
// through region_bump_freq_and_get to maintain the freq-based promote
// policy.
// ---------------------------------------------------------------------------
bool OdinPackedCache::get_sp(uint64_t sector_no, PageBufSp* out_sp) {
    if (!enabled_) return false;

    if (region_find(frozen_region_, sector_no, out_sp, nullptr)) {
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    uint8_t freq = 0;
    PageBufSp sp;
    if (region_bump_freq_and_get(window_region_, sector_no, &sp, &freq)) {
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        if (freq >= promote_threshold_) {
            promote_to_frozen(sector_no, sp, freq);
        }
        if (out_sp) *out_sp = std::move(sp);
        return true;
    }

    stats_.misses.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// ---------------------------------------------------------------------------
// get_readonly — hit-path read that DOES NOT drive W-TinyLFU admission.
//
// Same as get() except the window path skips frequency bump + promotion
// check. Designed for front-end query traffic to avoid monopolizing
// cache admission with (possibly-repeated) query access patterns.
//
// Frozen hits are identical to get() (no freq tracking on frozen anyway).
// Window hits do a plain seqlock read via region_find.
// ---------------------------------------------------------------------------
bool OdinPackedCache::get_readonly(uint64_t sector_no, uint8_t* value) {
    if (!enabled_) return false;

    std::shared_ptr<const PageBuf> sp;
    if (region_find(frozen_region_, sector_no, &sp, nullptr)) {
        std::memcpy(value, sp->data(), SECTOR_LEN);
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (region_find(window_region_, sector_no, &sp, nullptr)) {
        std::memcpy(value, sp->data(), SECTOR_LEN);
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    stats_.misses.fetch_add(1, std::memory_order_relaxed);
    return false;
}

bool OdinPackedCache::get_sp_readonly(uint64_t sector_no, PageBufSp* out_sp) {
    if (!enabled_) return false;

    if (region_find(frozen_region_, sector_no, out_sp, nullptr)) {
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (region_find(window_region_, sector_no, out_sp, nullptr)) {
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    stats_.misses.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// ---------------------------------------------------------------------------
// admit — admit-on-miss into window; evict LRU if full.
// ---------------------------------------------------------------------------
void OdinPackedCache::admit(uint64_t sector_no, const uint8_t* value) {
    if (!enabled_) return;

    auto page = std::make_shared<PageBuf>();
    std::memcpy(page->data(), value, SECTOR_LEN);

    bool inserted = false;
    bool evicted = false;
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);

        // Recheck frozen after taking the window policy lock. A concurrent
        // promotion may have completed after the initial miss.
        if (region_find(frozen_region_, sector_no, nullptr, nullptr) ||
            region_find_locked(window_region_, sector_no) != nullptr) {
            return;
        }
        if (window_count_.load(std::memory_order_relaxed) >= max_window_) {
            evicted = evict_window_lru_locked();
        }
        inserted = region_insert_locked(window_region_, sector_no, page, 1);
        if (inserted) {
            window_lru_.push_back(sector_no);
            window_lru_map_[sector_no] = std::prev(window_lru_.end());
        }
    }

    if (evicted) stats_.evictions.fetch_add(1, std::memory_order_relaxed);
    if (!inserted) return;
    stats_.admissions.fetch_add(1, std::memory_order_relaxed);
    update_bytes_resident();
}

// ---------------------------------------------------------------------------
// promote_to_frozen — move entry from window to frozen. Called outside
// the window's writer_mu (get() above releases before calling).
// ---------------------------------------------------------------------------
void OdinPackedCache::promote_to_frozen(uint64_t sector_no,
                                        std::shared_ptr<const PageBuf> /*buf_sp*/,
                                        uint8_t /*freq*/) {
    if (frozen_count_.load(std::memory_order_relaxed) >= max_frozen_) {
        evict_one_frozen();
    }

    bool inserted = false;
    {
        // Frozen-before-window is the only nested policy-lock order. Admit
        // takes only the window lock and rechecks frozen while holding it.
        std::lock_guard<std::mutex> frozen_lk(frozen_region_.writer_mu);
        std::lock_guard<std::mutex> window_lk(window_region_.writer_mu);
        Slot* window_slot = region_find_locked(window_region_, sector_no);
        if (window_slot == nullptr) return;

        auto current_sp = std::atomic_load_explicit(
            &window_slot->buf_sp, std::memory_order_acquire);
        if (!current_sp) return;

        Slot* frozen_slot = region_find_locked(frozen_region_, sector_no);
        if (frozen_slot == nullptr) {
            inserted = region_insert_locked(
                frozen_region_, sector_no, std::move(current_sp), 0);
            if (!inserted) return;
            std::lock_guard<std::mutex> fifo_lk(frozen_fifo_mu_);
            frozen_fifo_.push_back(sector_no);
        }

        region_erase_locked(window_region_, sector_no);
        auto lru_it = window_lru_map_.find(sector_no);
        if (lru_it != window_lru_map_.end()) {
            window_lru_.erase(lru_it->second);
            window_lru_map_.erase(lru_it);
        }
    }

    if (inserted) {
        stats_.window_promotions.fetch_add(1, std::memory_order_relaxed);
    }
    update_bytes_resident();
}

void OdinPackedCache::evict_one_frozen() {
    uint64_t victim;
    bool have_victim = false;
    while (!have_victim) {
        {
            std::lock_guard<std::mutex> lk(frozen_fifo_mu_);
            if (frozen_fifo_.empty()) return;
            victim = frozen_fifo_.front();
            frozen_fifo_.pop_front();
        }
        bool erased;
        {
            std::lock_guard<std::mutex> lk(frozen_region_.writer_mu);
            erased = region_erase_locked(frozen_region_, victim);
        }
        if (erased) {
            stats_.evictions.fetch_add(1, std::memory_order_relaxed);
            have_victim = true;
        }
    }
    update_bytes_resident();
}

bool OdinPackedCache::evict_window_lru_locked() {
    while (!window_lru_.empty()) {
        auto victim_it = window_lru_.begin();
        const uint64_t victim = *victim_it;
        auto map_it = window_lru_map_.find(victim);
        if (map_it != window_lru_map_.end() && map_it->second == victim_it) {
            window_lru_map_.erase(map_it);
        }
        window_lru_.erase(victim_it);
        if (region_erase_locked(window_region_, victim)) return true;
    }
    return false;
}

void OdinPackedCache::evict_window_lru() {
    bool erased;
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        erased = evict_window_lru_locked();
    }
    if (erased) {
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        update_bytes_resident();
    }
}

// ---------------------------------------------------------------------------
// update_if_resident — swap shared_ptr under the region's writer mutex.
// Readers still see a consistent snapshot via atomic_load + seqlock.
// ---------------------------------------------------------------------------
void OdinPackedCache::update_if_resident(uint64_t sector_no, const uint8_t* value) {
    if (!enabled_) return;
    auto new_sp = std::make_shared<PageBuf>();
    std::memcpy(new_sp->data(), value, SECTOR_LEN);

    {
        std::lock_guard<std::mutex> lk(frozen_region_.writer_mu);
        if (region_swap_buf_locked(frozen_region_, sector_no, new_sp)) {
            stats_.write_through_hits.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        if (region_swap_buf_locked(window_region_, sector_no, new_sp)) {
            stats_.write_through_hits.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

bool OdinPackedCache::is_cached(uint64_t sector_no) const {
    if (!enabled_) return false;
    if (region_find(frozen_region_, sector_no, nullptr, nullptr)) return true;
    return region_find(window_region_, sector_no, nullptr, nullptr);
}

std::size_t OdinPackedCache::clear() {
    if (!enabled_) return 0;

    std::lock_guard<std::mutex> frozen_lk(frozen_region_.writer_mu);
    std::lock_guard<std::mutex> window_lk(window_region_.writer_mu);
    const std::size_t cleared =
        frozen_count_.load(std::memory_order_relaxed) +
        window_count_.load(std::memory_order_relaxed);

    auto clear_region = [](Region& region) {
        for (Slot& slot : region.slots) {
            if (!slot.occupied.load(std::memory_order_relaxed)) continue;
            slot.version.fetch_add(1, std::memory_order_release);
            std::atomic_store_explicit(
                &slot.buf_sp, std::shared_ptr<const PageBuf>{},
                std::memory_order_release);
            slot.occupied.store(false, std::memory_order_relaxed);
            slot.key.store(0, std::memory_order_relaxed);
            slot.frequency.store(0, std::memory_order_relaxed);
            slot.version.fetch_add(1, std::memory_order_release);
        }
        region.count.store(0, std::memory_order_relaxed);
    };

    clear_region(frozen_region_);
    clear_region(window_region_);
    {
        std::lock_guard<std::mutex> fifo_lk(frozen_fifo_mu_);
        frozen_fifo_.clear();
    }
    window_lru_.clear();
    window_lru_map_.clear();
    update_bytes_resident();
    return cleared;
}

void OdinPackedCache::evict(uint64_t sector_no) {
    if (!enabled_) return;
    bool erased_frozen;
    {
        std::lock_guard<std::mutex> lk(frozen_region_.writer_mu);
        erased_frozen = region_erase_locked(frozen_region_, sector_no);
    }
    if (erased_frozen) {
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        update_bytes_resident();
        return;
    }
    bool erased_window;
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        erased_window = region_erase_locked(window_region_, sector_no);
        if (erased_window) {
            auto it = window_lru_map_.find(sector_no);
            if (it != window_lru_map_.end()) {
                window_lru_.erase(it->second);
                window_lru_map_.erase(it);
            }
        }
    }
    if (erased_window) {
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        update_bytes_resident();
    }
}

}  // namespace v2
