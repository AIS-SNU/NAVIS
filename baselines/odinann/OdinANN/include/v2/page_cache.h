#ifndef PAGE_CACHE_H_
#define PAGE_CACHE_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "log.h"

#ifndef SECTOR_LEN
#define SECTOR_LEN 4096
#endif

namespace v2 {
  // User-space page cache for update acceleration. Entries contain the
  // freshest in-flight sector bytes and live until all users drop their refs.
  struct PageCacheItem {
    std::unique_ptr<uint8_t[]> buf{new uint8_t[SECTOR_LEN]};
    uint64_t ref_cnt = 0;
  };

  class PageCache {
   public:
    bool get(uint64_t block_no, uint8_t *value, bool ref = false) {
      Shard &shard = shard_for(block_no);
      std::lock_guard<std::mutex> lock(shard.mu);
      auto it = shard.items.find(block_no);
      if (it == shard.items.end()) return false;
      std::memcpy(value, it->second.buf.get(), SECTOR_LEN);
      if (ref) ++it->second.ref_cnt;
      return true;
    }

    bool put(uint64_t block_no, uint8_t *value, bool ref = false) {
      Shard &shard = shard_for(block_no);
      std::lock_guard<std::mutex> lock(shard.mu);
      auto [it, inserted] = shard.items.try_emplace(block_no);
      if (ref) ++it->second.ref_cnt;
      std::memcpy(it->second.buf.get(), value, SECTOR_LEN);
      if (inserted) size_.fetch_add(1, std::memory_order_relaxed);
      return inserted;
    }

    bool deref(uint64_t block_no) {
      Shard &shard = shard_for(block_no);
      std::lock_guard<std::mutex> lock(shard.mu);
      auto it = shard.items.find(block_no);
      if (it == shard.items.end()) {
        LOG(ERROR) << "PageCache: deref a non-existent block_no: " << block_no;
        return false;
      }
      if (it->second.ref_cnt == 0) {
        LOG(ERROR) << "PageCache: deref block_no with zero refs: " << block_no;
        return false;
      }
      if (--it->second.ref_cnt == 0) {
        shard.items.erase(it);
        size_.fetch_sub(1, std::memory_order_relaxed);
      }
      return true;
    }

    void clear() {
      for (auto &shard : shards_) {
        std::lock_guard<std::mutex> lock(shard.mu);
        shard.items.clear();
      }
      size_.store(0, std::memory_order_relaxed);
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }

   private:
    static constexpr size_t kShardCount = 4096;

    struct Shard {
      std::mutex mu;
      std::unordered_map<uint64_t, PageCacheItem> items;
    };

    static uint64_t hash_key(uint64_t x) {
      x ^= x >> 33;
      x *= 0xff51afd7ed558ccdULL;
      x ^= x >> 33;
      x *= 0xc4ceb9fe1a85ec53ULL;
      x ^= x >> 33;
      return x;
    }

    Shard &shard_for(uint64_t block_no) {
      return shards_[hash_key(block_no) & (kShardCount - 1)];
    }

    std::array<Shard, kShardCount> shards_;
    std::atomic<size_t> size_{0};
  };

  inline PageCache cache;
}  // namespace v2

#endif  // PAGE_CACHE_H_
