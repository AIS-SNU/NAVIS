#pragma once

#include "navis/types.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace navis {

// ---------------------------------------------------------------------------
// Windowed CASR (convergence-aware speculative reranking)
//
// Candidates are reranked in PQ order, one vector at a time. Reranking
// converges once the exact top-K has gone `w` consecutive loads without
// changing; the first K loads always change it, so the earliest stop is
// load K + w. After a change at load i the next w vectors are needed no
// matter what, so they are read speculatively as one batch: the number of
// loads is exactly min(candidates, last change + w), with no wasted reads.
//
// The window is calibrated per path. The first warm-up reranks load every
// candidate and record the smallest window that would not have stopped them
// early (longest unchanged run before the final top-K change, plus one);
// w is a percentile of those values.
// ---------------------------------------------------------------------------

// Follows the exact top-K of one rerank and tracks how many loads it needs.
class CasrTracker {
public:
    // window == 0 profiles: every candidate is needed.
    CasrTracker(uint32_t k, uint32_t window, uint32_t n_candidates)
        : k_(k), window_(window), n_(n_candidates) {
        topk_.reserve(k);
    }

    // Feed the exact distance of the next candidate (in load order).
    // Returns true if it entered the top-K.
    bool add(const Neighbor& n) {
        ++loaded_;
        bool changed = false;
        if (topk_.size() < k_) {
            topk_.insert(std::upper_bound(topk_.begin(), topk_.end(), n), n);
            changed = true;
        } else if (k_ > 0 && n < topk_.back()) {
            topk_.pop_back();
            topk_.insert(std::upper_bound(topk_.begin(), topk_.end(), n), n);
            changed = true;
        }
        if (changed) {
            max_gap_ = std::max(max_gap_, loaded_ - last_change_ - 1);
            last_change_ = loaded_;
        }
        return changed;
    }

    // Loads required so far.
    uint32_t needed() const {
        if (window_ == 0) return n_;
        uint64_t need = static_cast<uint64_t>(std::max(last_change_, k_)) + window_;
        return static_cast<uint32_t>(std::min<uint64_t>(n_, need));
    }
    uint32_t loaded() const { return loaded_; }
    bool done() const { return loaded_ >= needed(); }

    // Smallest window that would not have stopped this rerank before its
    // final top-K change.
    uint32_t min_safe_window() const { return max_gap_ + 1; }

private:
    uint32_t k_;
    uint32_t window_;
    uint32_t n_;
    uint32_t loaded_      = 0;
    uint32_t last_change_ = 0;
    uint32_t max_gap_     = 0;
    std::vector<Neighbor> topk_;   // sorted ascending
};

// Per-path window calibration from warm-up samples. Thread-safe.
class CasrCalibrator {
public:
    // Calibrated window, or 0 while still warming up.
    uint32_t window() const { return window_.load(std::memory_order_acquire); }

    // Record one warm-up rerank's min_safe_window(). Once `target` samples
    // exist, the `percentile`-th percentile becomes the window.
    void record(uint32_t min_safe_window, uint32_t target, uint32_t percentile) {
        std::lock_guard<std::mutex> lk(mu_);
        if (window_.load(std::memory_order_relaxed) > 0) return;
        samples_.push_back(min_safe_window);
        if (samples_.size() < std::max<uint32_t>(target, 1)) return;
        std::vector<uint32_t> copy = samples_;
        const uint32_t p = std::min<uint32_t>(percentile, 100);
        const size_t idx = std::min(copy.size() - 1, (copy.size() * p) / 100);
        std::nth_element(copy.begin(), copy.begin() + idx, copy.end());
        window_.store(std::max<uint32_t>(1, copy[idx]), std::memory_order_release);
    }

private:
    std::mutex            mu_;
    std::vector<uint32_t> samples_;
    std::atomic<uint32_t> window_{0};
};

} // namespace navis
