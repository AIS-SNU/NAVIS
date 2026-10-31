#include "navis/index/deletion_table.h"

namespace navis {

void DeletionTable::init(uint64_t capacity) {
    capacity_ = capacity;
    deleted_count_.store(0, std::memory_order_relaxed);
    if (capacity_ == 0) {
        bits_.reset();
        return;
    }
    bits_ = std::make_unique<std::atomic<uint8_t>[]>(capacity_);
    for (uint64_t i = 0; i < capacity_; ++i) {
        bits_[i].store(0, std::memory_order_relaxed);
    }
}

bool DeletionTable::erase(uint32_t id, uint64_t active_limit) {
    if (!bits_ || id >= capacity_ || id >= active_limit) return false;
    uint8_t expected = 0;
    if (bits_[id].compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        deleted_count_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

bool DeletionTable::is_deleted(uint32_t id) const {
    if (!bits_ || id >= capacity_) return false;
    return bits_[id].load(std::memory_order_acquire) != 0;
}

double DeletionTable::deleted_fraction(uint64_t active_points) const {
    if (active_points == 0) return 0.0;
    return static_cast<double>(deleted_count()) / static_cast<double>(active_points);
}

} // namespace navis
