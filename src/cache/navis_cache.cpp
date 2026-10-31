#include "navis/cache/navis_cache.h"
#include <algorithm>
#include <cstdio>

namespace navis {

thread_local std::mt19937 NavisCache::tl_rng_{std::random_device{}()};

// ---------------------------------------------------------------------------
// Region init — size slot array to next power-of-two ≥ 2 × max_entries
// (load factor capped at 0.5 to keep linear-probe chains short).
// ---------------------------------------------------------------------------
void NavisCache::region_init(Region& r, std::size_t max_entries) {
    std::size_t target = 16;
    while (target < max_entries * 2) target <<= 1;
    r.slots = std::vector<Slot>(target);
    r.mask = target - 1;
    r.max_entries = max_entries;
    r.count.store(0, std::memory_order_relaxed);
}

void NavisCache::init(const CacheConfig& config) {
    if (config.size_gb == 0) {
        enabled_ = false;
        return;
    }
    uint64_t total_sectors = (config.size_gb * 1024UL * 1024UL * 1024UL) / EDGE_SECTOR_LEN;
    max_window_ = static_cast<std::size_t>(total_sectors * config.window_ratio);
    max_frozen_ = total_sectors - max_window_;
    promote_threshold_ = config.promote_threshold;
    enabled_ = true;

    region_init(frozen_region_, max_frozen_);
    region_init(window_region_, max_window_);
}

// ---------------------------------------------------------------------------
// region_find — seqlock-protected read.
// ---------------------------------------------------------------------------
bool NavisCache::region_find(const Region& r, uint64_t key,
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

            if (out_sp)   *out_sp   = std::move(sp);
            if (out_freq) *out_freq = freq;
            return true;
        }
    }
    return false;
}

NavisCache::Slot* NavisCache::region_find_locked(Region& r, uint64_t key) const {
    if (r.slots.empty()) return nullptr;
    std::size_t idx = hash_key(key) & r.mask;
    for (std::size_t p = 0; p < kMaxProbe; ++p) {
        Slot& s = r.slots[(idx + p) & r.mask];
        if (!s.occupied) return nullptr;
        if (s.key == key) return &s;
    }
    return nullptr;
}

bool NavisCache::region_insert_locked(Region& r, uint64_t key,
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

bool NavisCache::region_swap_buf_locked(Region& r, uint64_t key,
                                        std::shared_ptr<const PageBuf> new_buf) {
    Slot* s = region_find_locked(r, key);
    if (!s) return false;
    s->version.fetch_add(1, std::memory_order_release);          // odd
    std::atomic_store_explicit(&s->buf_sp, std::move(new_buf),
                               std::memory_order_release);
    s->version.fetch_add(1, std::memory_order_release);          // even
    return true;
}

bool NavisCache::region_bump_freq_and_get(Region& r, uint64_t key,
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
    if (out_sp)   *out_sp   = std::move(sp);
    if (out_freq) *out_freq = new_freq;
    return true;
}

bool NavisCache::region_erase_locked(Region& r, uint64_t key) {
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
void NavisCache::region_fixup_after_erase_locked(Region& r, std::size_t i) {
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
            dst.key = sj.key;
            auto sp = std::atomic_load_explicit(&sj.buf_sp,
                                                std::memory_order_acquire);
            std::atomic_store_explicit(&dst.buf_sp, std::move(sp),
                                       std::memory_order_release);
            dst.frequency = sj.frequency;
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
bool NavisCache::get(uint64_t sector_no, uint8_t* value) {
    if (!enabled_) return false;

    // Frozen: pure seqlock read, no lock acquisition.
    std::shared_ptr<const PageBuf> sp;
    if (region_find(frozen_region_, sector_no, &sp, nullptr)) {
        std::memcpy(value, sp->data(), EDGE_SECTOR_LEN);
        hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Window: bump frequency under writer_mu; possibly promote.
    uint8_t freq = 0;
    if (region_bump_freq_and_get(window_region_, sector_no, &sp, &freq)) {
        std::memcpy(value, sp->data(), EDGE_SECTOR_LEN);
        hits_.fetch_add(1, std::memory_order_relaxed);
        if (freq >= promote_threshold_) {
            promote_to_frozen(sector_no, sp, freq);
        }
        return true;
    }

    misses_.fetch_add(1, std::memory_order_relaxed);
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
bool NavisCache::get_sp(uint64_t sector_no, PageBufSp* out_sp,
                        uint32_t promote_threshold_override) {
    if (!enabled_) return false;

    if (region_find(frozen_region_, sector_no, out_sp, nullptr)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    uint8_t freq = 0;
    PageBufSp sp;
    if (region_bump_freq_and_get(window_region_, sector_no, &sp, &freq)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        const uint32_t thresh = promote_threshold_override
                                    ? promote_threshold_override
                                    : promote_threshold_;
        if (freq >= thresh) {
            promote_to_frozen(sector_no, sp, freq);
        }
        if (out_sp) *out_sp = std::move(sp);
        return true;
    }

    misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// ---------------------------------------------------------------------------
// populate — admit-on-miss into window; evict LRU if full.
// ---------------------------------------------------------------------------
void NavisCache::populate(uint64_t sector_no, const uint8_t* value) {
    if (!enabled_) return;

    // Skip if already frozen.
    if (region_find(frozen_region_, sector_no, nullptr, nullptr)) return;

    if (window_count_.load(std::memory_order_relaxed) >= max_window_) {
        evict_window_lru();
    }

    auto page = std::make_shared<PageBuf>();
    std::memcpy(page->data(), value, EDGE_SECTOR_LEN);

    bool inserted;
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        inserted = region_insert_locked(window_region_, sector_no, page, 1);
    }
    if (!inserted) return;  // race (another thread inserted concurrently)

    {
        std::lock_guard<std::mutex> lk(window_lru_mu_);
        window_lru_.push_back(sector_no);
        window_lru_map_[sector_no] = std::prev(window_lru_.end());
    }
}

// ---------------------------------------------------------------------------
// promote_to_frozen — move entry from window to frozen. Called outside
// the window's writer_mu (get() above releases before calling).
// ---------------------------------------------------------------------------
void NavisCache::promote_to_frozen(uint64_t sector_no,
                                    std::shared_ptr<const PageBuf> buf_sp,
                                    uint8_t /*freq*/) {
    if (frozen_count_.load(std::memory_order_relaxed) >= max_frozen_) {
        evict_one_frozen();
    }

    bool inserted;
    {
        std::lock_guard<std::mutex> lk(frozen_region_.writer_mu);
        inserted = region_insert_locked(frozen_region_, sector_no, buf_sp, 0);
    }
    if (inserted) {
        std::lock_guard<std::mutex> lk(frozen_fifo_mu_);
        frozen_fifo_.push_back(sector_no);
    }

    // Remove from window (hash + LRU list).
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        region_erase_locked(window_region_, sector_no);
    }
    {
        std::lock_guard<std::mutex> lk(window_lru_mu_);
        auto it = window_lru_map_.find(sector_no);
        if (it != window_lru_map_.end()) {
            window_lru_.erase(it->second);
            window_lru_map_.erase(it);
        }
    }
}

void NavisCache::evict_one_frozen() {
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
            total_evictions_.fetch_add(1, std::memory_order_relaxed);
            have_victim = true;
        }
    }
}

void NavisCache::evict_window_lru() {
    uint64_t victim;
    {
        std::lock_guard<std::mutex> lk(window_lru_mu_);
        if (window_lru_.empty()) return;
        victim = window_lru_.front();
        window_lru_.pop_front();
        window_lru_map_.erase(victim);
    }
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        region_erase_locked(window_region_, victim);
    }
    total_evictions_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// update_if_present — swap shared_ptr under the region's writer mutex.
// Readers still see a consistent snapshot via atomic_load + seqlock.
// ---------------------------------------------------------------------------
void NavisCache::update_if_present(uint64_t sector_no, const uint8_t* value) {
    if (!enabled_) return;
    auto new_sp = std::make_shared<PageBuf>();
    std::memcpy(new_sp->data(), value, EDGE_SECTOR_LEN);

    {
        std::lock_guard<std::mutex> lk(frozen_region_.writer_mu);
        if (region_swap_buf_locked(frozen_region_, sector_no, new_sp)) return;
    }
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        region_swap_buf_locked(window_region_, sector_no, new_sp);
    }
}

void NavisCache::evict(uint64_t sector_no) {
    if (!enabled_) return;
    bool erased_frozen;
    {
        std::lock_guard<std::mutex> lk(frozen_region_.writer_mu);
        erased_frozen = region_erase_locked(frozen_region_, sector_no);
    }
    if (erased_frozen) {
        total_evictions_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    bool erased_window;
    {
        std::lock_guard<std::mutex> lk(window_region_.writer_mu);
        erased_window = region_erase_locked(window_region_, sector_no);
    }
    if (erased_window) {
        std::lock_guard<std::mutex> lk(window_lru_mu_);
        auto it = window_lru_map_.find(sector_no);
        if (it != window_lru_map_.end()) {
            window_lru_.erase(it->second);
            window_lru_map_.erase(it);
        }
        total_evictions_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace navis
