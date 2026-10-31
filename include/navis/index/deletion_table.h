#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace navis {

class DeletionTable {
public:
    DeletionTable() = default;
    ~DeletionTable() = default;

    DeletionTable(const DeletionTable&) = delete;
    DeletionTable& operator=(const DeletionTable&) = delete;

    void init(uint64_t capacity);
    bool erase(uint32_t id, uint64_t active_limit);
    bool is_deleted(uint32_t id) const;

    uint64_t deleted_count() const {
        return deleted_count_.load(std::memory_order_relaxed);
    }

    uint64_t capacity() const { return capacity_; }
    double deleted_fraction(uint64_t active_points) const;

private:
    std::unique_ptr<std::atomic<uint8_t>[]> bits_;
    uint64_t capacity_ = 0;
    std::atomic<uint64_t> deleted_count_{0};
};

} // namespace navis
