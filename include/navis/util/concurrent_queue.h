#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace navis {

// ---------------------------------------------------------------------------
// ConcurrentQueue — thread-safe MPMC queue for background task dispatch
// ---------------------------------------------------------------------------
template <typename T>
class ConcurrentQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    // Blocking pop — waits until an item is available
    T pop() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) {
            return T{};
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Non-blocking try_pop
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.size();
    }

    // Signal all waiting threads to stop
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    bool is_stopped() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stopped_;
    }

private:
    std::queue<T>           queue_;
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    bool                    stopped_ = false;
};

} // namespace navis
