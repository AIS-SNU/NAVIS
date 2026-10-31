#ifndef LOCK_TABLE_H_
#define LOCK_TABLE_H_
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <thread>
#include <vector>
#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "libcuckoo/cuckoohash_map.hh"
#include "log.h"

#define SECTOR_LEN 4096

inline void thread_pause() {
  // Use pause instruction to reduce contention in tight loops.
#ifdef __x86_64__
  asm volatile("pause" ::: "memory");
#endif
}

namespace v2 {
  // Odin hands page locks from insert workers to the background I/O completion
  // thread. pthread_rwlock_t cannot support that ownership transfer, so the
  // striped table uses an ownerless reader-writer state machine instead.
  class OwnerlessRwLock {
   public:
    OwnerlessRwLock() = default;
    OwnerlessRwLock(const OwnerlessRwLock &) = delete;
    OwnerlessRwLock &operator=(const OwnerlessRwLock &) = delete;

    int tryrdlock() {
      if (waiting_writers_.load(std::memory_order_acquire) != 0) {
        return EBUSY;
      }

      uint32_t state = state_.load(std::memory_order_relaxed);
      while ((state & kWriterBit) == 0) {
        if ((state & kReaderMask) == kReaderMask) {
          return EAGAIN;
        }
        if (waiting_writers_.load(std::memory_order_acquire) != 0) {
          return EBUSY;
        }
        if (state_.compare_exchange_weak(state, state + 1,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
          return 0;
        }
      }
      return EBUSY;
    }

    int trywrlock() {
      uint32_t expected = 0;
      if (!state_.compare_exchange_strong(expected, kWriterBit,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
        return EBUSY;
      }
      writer_tid_.store(current_tid(), std::memory_order_relaxed);
      return 0;
    }

    int rdlock() {
      for (;;) {
        int ret = tryrdlock();
        if (ret == 0 || ret == EAGAIN) {
          return ret;
        }
        wait_for_read();
      }
    }

    int wrlock() {
      waiting_writers_.fetch_add(1, std::memory_order_acq_rel);
      for (;;) {
        uint32_t expected = 0;
        if (state_.compare_exchange_weak(expected, kWriterBit,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
          writer_tid_.store(current_tid(), std::memory_order_relaxed);
          waiting_writers_.fetch_sub(1, std::memory_order_release);
          return 0;
        }
        wait_for_write();
      }
    }

    int unlock() {
      uint32_t state = state_.load(std::memory_order_relaxed);
      for (;;) {
        if (state == kWriterBit) {
          writer_tid_.store(0, std::memory_order_relaxed);
          uint32_t expected = kWriterBit;
          if (!state_.compare_exchange_strong(expected, 0,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
            return EPERM;
          }
          wake_all();
          return 0;
        }
        if ((state & kWriterBit) != 0 || (state & kReaderMask) == 0) {
          return EPERM;
        }
        uint32_t next = state - 1;
        if (state_.compare_exchange_weak(state, next,
                                         std::memory_order_release,
                                         std::memory_order_relaxed)) {
          if (next == 0 &&
              waiting_writers_.load(std::memory_order_acquire) != 0) {
            wake_all();
          }
          return 0;
        }
      }
    }

    void wait_read() { wait_for_read(); }
    void wait_write() { wait_for_write(); }
    void add_waiting_writer() {
      waiting_writers_.fetch_add(1, std::memory_order_acq_rel);
    }
    void remove_waiting_writer() {
      waiting_writers_.fetch_sub(1, std::memory_order_release);
    }
    uint32_t debug_state() const { return state_.load(std::memory_order_acquire); }
    uint32_t debug_waiting_writers() const {
      return waiting_writers_.load(std::memory_order_acquire);
    }
    uint32_t debug_writer_tid() const {
      return writer_tid_.load(std::memory_order_acquire);
    }
    static uint32_t current_thread_tid() { return current_tid(); }

   private:
    static constexpr uint32_t kWriterBit = 1U << 31;
    static constexpr uint32_t kReaderMask = kWriterBit - 1;

    void wait_for_read() {
      uint32_t epoch = wake_epoch_.load(std::memory_order_acquire);
      uint32_t state = state_.load(std::memory_order_acquire);
      if ((state & kWriterBit) != 0 ||
          waiting_writers_.load(std::memory_order_acquire) != 0) {
        wait(epoch);
      }
    }

    void wait_for_write() {
      uint32_t epoch = wake_epoch_.load(std::memory_order_acquire);
      if (state_.load(std::memory_order_acquire) != 0) {
        wait(epoch);
      }
    }

    void wait(uint32_t expected_epoch) {
#ifdef __linux__
      const timespec timeout{1, 0};
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(&wake_epoch_),
              FUTEX_WAIT_PRIVATE, expected_epoch, &timeout, nullptr, 0);
#else
      while (wake_epoch_.load(std::memory_order_acquire) == expected_epoch) {
        std::this_thread::yield();
      }
#endif
    }

    void wake_all() {
      wake_epoch_.fetch_add(1, std::memory_order_release);
#ifdef __linux__
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(&wake_epoch_),
              FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#endif
    }

    static uint32_t current_tid() {
#ifdef __linux__
      static thread_local const uint32_t tid =
          static_cast<uint32_t>(syscall(SYS_gettid));
      return tid;
#else
      return 0;
#endif
    }

    alignas(uint32_t) std::atomic<uint32_t> state_{0};
    alignas(uint32_t) std::atomic<uint32_t> wake_epoch_{0};
    std::atomic<uint32_t> waiting_writers_{0};
    std::atomic<uint32_t> writer_tid_{0};
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseLockTable {
   public:
    static constexpr size_t kDefaultStripes = 1U << 20;

    explicit SparseLockTable(size_t stripe_count = kDefaultStripes)
        : SparseLockTable("unnamed", stripe_count) {}

    explicit SparseLockTable(const char *name,
                             size_t stripe_count = kDefaultStripes)
        : name_(name),
          stripe_count_(std::max<size_t>(1, stripe_count)),
          locks_(new OwnerlessRwLock[stripe_count_]) {}

    ~SparseLockTable() { delete[] locks_; }

    SparseLockTable(const SparseLockTable &) = delete;
    SparseLockTable &operator=(const SparseLockTable &) = delete;

    size_t stripe_for(const K &key) const {
      return hasher_(key) % stripe_count_;
    }

    template<typename Value>
    void sort_and_deduplicate(std::vector<Value> &keys) const {
      std::sort(keys.begin(), keys.end(), [this](const Value &left, const Value &right) {
        size_t left_stripe = stripe_for(static_cast<K>(left));
        size_t right_stripe = stripe_for(static_cast<K>(right));
        if (left_stripe != right_stripe) {
          return left_stripe < right_stripe;
        }
        return left < right;
      });
      auto last = std::unique(keys.begin(), keys.end(), [this](const Value &left, const Value &right) {
        return stripe_for(static_cast<K>(left)) == stripe_for(static_cast<K>(right));
      });
      keys.erase(last, keys.end());
    }

    int tryrdlock(const K &key) {
      int ret = locks_[stripe_for(key)].tryrdlock();
      if (ret == 0) {
        active_locks_.fetch_add(1, std::memory_order_relaxed);
      }
      return ret;
    }

    int trywrlock(const K &key) {
      int ret = locks_[stripe_for(key)].trywrlock();
      if (ret == 0) {
        active_locks_.fetch_add(1, std::memory_order_relaxed);
      }
      return ret;
    }

    void rdlock(const K &key) {
      const size_t stripe = stripe_for(key);
      std::chrono::steady_clock::time_point start;
      bool contended = false;
      for (;;) {
        int ret = locks_[stripe].tryrdlock();
        if (ret == 0) {
          break;
        }
        if (ret == EAGAIN) {
          LOG(ERROR) << "SparseLockTable[" << name_
                     << "]: reader count overflow for key " << key;
          __builtin_trap();
        }
        if (!contended) {
          start = std::chrono::steady_clock::now();
          contended = true;
        }
        locks_[stripe].wait_read();
        report_stall("read", key, stripe, start);
      }
      active_locks_.fetch_add(1, std::memory_order_relaxed);
    }

    void wrlock(const K &key) {
      const size_t stripe = stripe_for(key);
      std::chrono::steady_clock::time_point start;
      bool contended = false;
      locks_[stripe].add_waiting_writer();
      for (;;) {
        int ret = locks_[stripe].trywrlock();
        if (ret == 0) {
          break;
        }
        if (!contended) {
          start = std::chrono::steady_clock::now();
          contended = true;
        }
        locks_[stripe].wait_write();
        report_stall("write", key, stripe, start);
      }
      locks_[stripe].remove_waiting_writer();
      active_locks_.fetch_add(1, std::memory_order_relaxed);
    }

    void unlock(const K &key) {
      const size_t stripe = stripe_for(key);
      int ret = locks_[stripe].unlock();
      if (ret != 0) {
        const uint32_t state = locks_[stripe].debug_state();
        LOG(ERROR) << "[odin-lock-invalid-unlock] table=" << name_
                   << " key=" << key
                   << " stripe=" << stripe
                   << " error=" << ret
                   << " writer=" << ((state >> 31) != 0)
                   << " readers=" << (state & 0x7fffffffU)
                   << " waiting_writers=" << locks_[stripe].debug_waiting_writers()
                   << " writer_tid=" << locks_[stripe].debug_writer_tid()
                   << " caller_tid=" << OwnerlessRwLock::current_thread_tid()
                   << " active=" << active_locks_.load(std::memory_order_relaxed);
        __builtin_trap();
      }
      size_t previous = active_locks_.fetch_sub(1, std::memory_order_relaxed);
      if (previous == 0) {
        LOG(ERROR) << "SparseLockTable: active lock count underflow for key " << key;
        __builtin_trap();
      }
    }

    size_t size() const {
      return active_locks_.load(std::memory_order_relaxed);
    }

   private:
    void report_stall(const char *mode, const K &key, size_t stripe,
                      std::chrono::steady_clock::time_point start) {
      const auto now = std::chrono::steady_clock::now();
      const int64_t elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (elapsed_ms < 5000) {
        return;
      }

      const int64_t now_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
      int64_t next = next_stall_report_ns_.load(std::memory_order_relaxed);
      if (now_ns < next ||
          !next_stall_report_ns_.compare_exchange_strong(
              next, now_ns + 5000000000LL, std::memory_order_relaxed)) {
        return;
      }

      const uint32_t state = locks_[stripe].debug_state();
      LOG(WARNING) << "[odin-lock-stall] table=" << name_
                   << " mode=" << mode
                   << " key=" << key
                   << " stripe=" << stripe
                   << " wait_ms=" << elapsed_ms
                   << " writer=" << ((state >> 31) != 0)
                   << " readers=" << (state & 0x7fffffffU)
                   << " waiting_writers=" << locks_[stripe].debug_waiting_writers()
                   << " writer_tid=" << locks_[stripe].debug_writer_tid()
                   << " active=" << active_locks_.load(std::memory_order_relaxed);
    }

    const char *name_;
    size_t stripe_count_;
    OwnerlessRwLock *locks_;
    HashFunction hasher_;
    std::atomic<size_t> active_locks_{0};
    std::atomic<int64_t> next_stall_report_ns_{0};
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseReadLockGuard {
   public:
    SparseReadLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->rdlock(key_);
    }

    ~SparseReadLockGuard() {
      table_->unlock(key_);
    }

   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseWriteLockGuard {
   public:
    SparseWriteLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->wrlock(key_);
    }

    ~SparseWriteLockGuard() {
      table_->unlock(key_);
    }

   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  class LockTable {
   public:
    LockTable(size_t size) : size_(size) {
      locks_ = new pthread_rwlock_t[size];
      for (size_t i = 0; i < size; i++) {
        pthread_rwlock_init(&locks_[i], nullptr);
      }
    }
    ~LockTable() {
    }

    inline pthread_rwlock_t *rdlock(uint32_t key) {
      auto lock = &locks_[Hash(key) % size_];
      pthread_rwlock_rdlock(lock);
      return lock;
    }

    inline uint64_t pos(uint64_t key) {
      return Hash(key) % size_;
    }

    inline pthread_rwlock_t *wrlock(uint32_t key) {
      auto lock = &locks_[Hash(key) % size_];
      pthread_rwlock_wrlock(lock);
      return lock;
    }

    inline bool tryrdlock(uint32_t key) {
      return (pthread_rwlock_tryrdlock(&locks_[Hash(key) % size_]) == 0);
    }

    inline bool trywrlock(uint32_t key) {
      return (pthread_rwlock_trywrlock(&locks_[Hash(key) % size_]) == 0);
    }

    inline void unlock(pthread_rwlock_t *lock) {
      pthread_rwlock_unlock(lock);
    }

    inline void unlock(uint32_t key) {
      pthread_rwlock_unlock(&locks_[Hash(key) % size_]);
    }

   private:
    size_t size_;
    pthread_rwlock_t *locks_;

    static const uint32_t c1 = 0xcc9e2d51;
    static const uint32_t c2 = 0x1b873593;

    static uint32_t fmix(uint32_t h) {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;
      return h;
    }

    static uint32_t Rotate32(uint32_t val, int shift) {
      // Avoid shifting by 32: doing so yields an undefined result.
      return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
    }

    static uint32_t Mur(uint32_t a, uint32_t h) {
      // Helper from Murmur3 for combining two 32-bit values.
      a *= c1;
      a = Rotate32(a, 17);
      a *= c2;
      h ^= a;
      h = Rotate32(h, 19);
      return h * 5 + 0xe6546b64;
    }

    static uint32_t Hash32Len0to4(const char *s, size_t len) {
      uint32_t b = 0;
      uint32_t c = 9;
      for (size_t i = 0; i < len; i++) {
        signed char v = static_cast<signed char>(s[i]);
        b = b * c1 + static_cast<uint32_t>(v);
        c ^= b;
      }
      return fmix(Mur(b, Mur(static_cast<uint32_t>(len), c)));
    }

    static uint32_t Hash(uint32_t x) {
      return Hash32Len0to4((const char *) &x, sizeof(uint32_t));
    }
  };

  // RAII to avoid forgetting to unlock.
  class LockGuard {
   public:
    LockGuard(pthread_rwlock_t *lock) : lock_(lock) {
    }
    LockGuard &operator=(const LockGuard &) = delete;
    LockGuard &operator=(LockGuard &&rhs) {
      lock_ = rhs.lock_;
      rhs.lock_ = nullptr;
      return *this;
    }
    ~LockGuard() {
      if (lock_) {
        pthread_rwlock_unlock(lock_);
        lock_ = nullptr;
      }
    }

   private:
    pthread_rwlock_t *lock_;
  };

}  // namespace v2

#endif  // LOCK_TABLE_H_