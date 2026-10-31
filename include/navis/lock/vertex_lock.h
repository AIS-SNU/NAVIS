#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>

namespace navis {

// ---------------------------------------------------------------------------
// VertexLockTable — growable atomic spinlock array (one lock per vertex)
//
// Each lock is a single atomic<bool>. Locking uses exchange with pause-based
// backoff. The table grows automatically when a vertex ID exceeds capacity.
//
// Grow-safety:
//   1. Old arrays are RETIRED, not freed (concurrent lockers may hold raw
//      pointers into them).
//   2. locks_ is atomic, so spinners re-read it every iteration. After a
//      grow, a spinner eventually picks up the new pointer, and the same
//      unlocker writes to the same pointer they read — no divergence.
// ---------------------------------------------------------------------------
class VertexLockTable {
public:
    explicit VertexLockTable(size_t initial_capacity = 1024)
        : capacity_(initial_capacity) {
        auto buf = std::make_unique<std::atomic<bool>[]>(initial_capacity);
        for (size_t i = 0; i < initial_capacity; i++) {
            buf[i].store(false, std::memory_order_relaxed);
        }
        locks_.store(buf.get(), std::memory_order_release);
        retired_.push_back(std::move(buf));
    }

    ~VertexLockTable() = default;  // retired_ cleans up all arrays

    // Non-copyable, but movable
    VertexLockTable(const VertexLockTable&) = delete;
    VertexLockTable& operator=(const VertexLockTable&) = delete;

    VertexLockTable(VertexLockTable&& other) noexcept
        : locks_(other.locks_.load(std::memory_order_acquire)),
          capacity_(other.capacity_.load()),
          grow_mutex_(),
          retired_(std::move(other.retired_)) {
        other.locks_.store(nullptr, std::memory_order_release);
        other.capacity_.store(0);
    }

    VertexLockTable& operator=(VertexLockTable&& other) noexcept {
        if (this != &other) {
            retired_ = std::move(other.retired_);
            locks_.store(other.locks_.load(std::memory_order_acquire),
                         std::memory_order_release);
            capacity_.store(other.capacity_.load());
            other.locks_.store(nullptr, std::memory_order_release);
            other.capacity_.store(0);
        }
        return *this;
    }

    void lock(uint32_t id) {
        if (id >= capacity_.load(std::memory_order_acquire)) {
            grow_to_fit(id + 1);
        }
        // Fast path: try to acquire on the currently-published array.
        auto* arr = locks_.load(std::memory_order_acquire);
        while (arr[id].exchange(true, std::memory_order_acquire)) {
            // Inner spin: re-read locks_ every iteration so that if a grow
            // happens mid-spin, we eventually switch to the new array. The
            // unlocker writes to whatever locks_ they see on their call,
            // so divergence between arrays resolves once both sides
            // observe the new pointer.
            while (true) {
                auto* cur = locks_.load(std::memory_order_acquire);
                if (!cur[id].load(std::memory_order_relaxed)) break;
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
            // Retry exchange on the *current* array.
            arr = locks_.load(std::memory_order_acquire);
        }
    }

    void unlock(uint32_t id) {
        auto* arr = locks_.load(std::memory_order_acquire);
        arr[id].store(false, std::memory_order_release);
    }

    size_t capacity() const { return capacity_.load(std::memory_order_relaxed); }

private:
    void grow_to_fit(size_t required) {
        std::lock_guard<std::mutex> guard(grow_mutex_);
        size_t cur = capacity_.load(std::memory_order_relaxed);
        if (required <= cur) return;

        size_t new_cap = std::max(required, cur + cur / 2); // grow by 1.5x
        auto new_locks = std::make_unique<std::atomic<bool>[]>(new_cap);
        auto* old_arr = locks_.load(std::memory_order_acquire);
        for (size_t i = 0; i < cur; i++) {
            new_locks[i].store(old_arr[i].load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        }
        for (size_t i = cur; i < new_cap; i++) {
            new_locks[i].store(false, std::memory_order_relaxed);
        }

        // Publish pointer first (release), then capacity (release). Readers
        // check capacity (acquire) first; if they see new capacity, they
        // see the new pointer too via the locks_ release-store.
        locks_.store(new_locks.get(), std::memory_order_release);
        capacity_.store(new_cap, std::memory_order_release);
        // Retire the new buffer (keeps it alive). The OLD array stays in
        // retired_ from its initial allocation or a prior grow — never
        // freed while the table lives.
        retired_.push_back(std::move(new_locks));
    }

    std::atomic<std::atomic<bool>*> locks_;
    std::atomic<size_t> capacity_;
    std::mutex          grow_mutex_;
    // Retired arrays kept alive for the lifetime of the table. Appended to
    // only under grow_mutex_; readers never touch this field.
    std::vector<std::unique_ptr<std::atomic<bool>[]>> retired_;
};

// ---------------------------------------------------------------------------
// Deadlock-safe multi-vertex locking: sort IDs ascending, lock in order
// ---------------------------------------------------------------------------
inline void lock_vertices(VertexLockTable& table, uint32_t target,
                          const std::vector<uint32_t>& neighbors) {
    std::vector<uint32_t> ids;
    ids.reserve(neighbors.size() + 1);
    ids.push_back(target);
    ids.insert(ids.end(), neighbors.begin(), neighbors.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (uint32_t id : ids) {
        table.lock(id);
    }
}

inline void unlock_vertices(VertexLockTable& table, uint32_t target,
                            const std::vector<uint32_t>& neighbors) {
    std::vector<uint32_t> ids;
    ids.reserve(neighbors.size() + 1);
    ids.push_back(target);
    ids.insert(ids.end(), neighbors.begin(), neighbors.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (uint32_t id : ids) {
        table.unlock(id);
    }
}

} // namespace navis
