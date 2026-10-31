#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace v2 {

template<typename Value>
class SegmentedAtomicMap {
 public:
  explicit SegmentedAtomicMap(Value empty_value = std::numeric_limits<Value>::max())
      : empty_value_(empty_value) {
    static_assert(std::is_trivially_copyable<Value>::value,
                  "SegmentedAtomicMap values must be atomic-compatible");
    for (auto &segment : segments_) segment.store(nullptr, std::memory_order_relaxed);
  }

  ~SegmentedAtomicMap() { clear(); }
  SegmentedAtomicMap(const SegmentedAtomicMap &) = delete;
  SegmentedAtomicMap &operator=(const SegmentedAtomicMap &) = delete;

  bool find(uint32_t key, Value &value) const {
    Segment *segment = segments_[key >> kSegmentBits].load(std::memory_order_acquire);
    if (segment == nullptr) return false;
    value = segment->values[key & kSegmentMask].load(std::memory_order_acquire);
    return value != empty_value_;
  }

  bool insert(uint32_t key, Value value) {
    if (value == empty_value_) return false;
    Segment *segment = ensure_segment(key >> kSegmentBits);
    Value expected = empty_value_;
    if (!segment->values[key & kSegmentMask].compare_exchange_strong(
            expected, value, std::memory_order_release, std::memory_order_relaxed)) {
      return false;
    }
    size_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  void insert_or_assign(uint32_t key, Value value) {
    if (value == empty_value_) {
      erase(key);
      return;
    }
    Segment *segment = ensure_segment(key >> kSegmentBits);
    Value old = segment->values[key & kSegmentMask].exchange(value, std::memory_order_acq_rel);
    if (old == empty_value_) size_.fetch_add(1, std::memory_order_relaxed);
  }

  bool erase(uint32_t key) {
    Segment *segment = segments_[key >> kSegmentBits].load(std::memory_order_acquire);
    if (segment == nullptr) return false;
    Value old = segment->values[key & kSegmentMask].exchange(empty_value_, std::memory_order_acq_rel);
    if (old == empty_value_) return false;
    size_.fetch_sub(1, std::memory_order_relaxed);
    return true;
  }

  void reserve(size_t entries) {
    size_t segment_count = (entries + kSegmentSize - 1) / kSegmentSize;
    if (segment_count > kSegmentCount) segment_count = kSegmentCount;
    for (size_t i = 0; i < segment_count; ++i) ensure_segment(i);
  }

  void clear() {
    for (auto &slot : segments_) {
      delete slot.exchange(nullptr, std::memory_order_acq_rel);
    }
    size_.store(0, std::memory_order_relaxed);
  }

  size_t size() const { return size_.load(std::memory_order_relaxed); }

 private:
  static constexpr size_t kSegmentBits = 16;
  static constexpr size_t kSegmentSize = size_t{1} << kSegmentBits;
  static constexpr size_t kSegmentMask = kSegmentSize - 1;
  static constexpr size_t kSegmentCount = size_t{1} << (32 - kSegmentBits);

  struct Segment {
    explicit Segment(Value empty_value) {
      for (auto &value : values) value.store(empty_value, std::memory_order_relaxed);
    }
    std::array<std::atomic<Value>, kSegmentSize> values;
  };

  Segment *ensure_segment(size_t segment_index) {
    Segment *segment = segments_[segment_index].load(std::memory_order_acquire);
    if (segment != nullptr) return segment;
    Segment *candidate = new Segment(empty_value_);
    if (!segments_[segment_index].compare_exchange_strong(
            segment, candidate, std::memory_order_release, std::memory_order_acquire)) {
      delete candidate;
      return segment;
    }
    return candidate;
  }

  Value empty_value_;
  std::array<std::atomic<Segment *>, kSegmentCount> segments_;
  std::atomic<size_t> size_{0};
};

template<typename Key, typename Value, size_t ShardCount = 4096,
         typename Hash = std::hash<Key>>
class ShardedMap {
 public:
  static_assert((ShardCount & (ShardCount - 1)) == 0,
                "ShardedMap shard count must be a power of two");

  bool insert(const Key &key, const Value &value) {
    Shard &shard = shard_for(key);
    std::lock_guard<std::mutex> lock(shard.mu);
    auto [unused, inserted] = shard.items.emplace(key, value);
    if (inserted) size_.fetch_add(1, std::memory_order_relaxed);
    return inserted;
  }

  void insert_or_assign(const Key &key, const Value &value) {
    Shard &shard = shard_for(key);
    std::lock_guard<std::mutex> lock(shard.mu);
    auto [unused, inserted] = shard.items.insert_or_assign(key, value);
    if (inserted) size_.fetch_add(1, std::memory_order_relaxed);
  }

  bool find(const Key &key, Value &value) const {
    const Shard &shard = shard_for(key);
    std::lock_guard<std::mutex> lock(shard.mu);
    auto it = shard.items.find(key);
    if (it == shard.items.end()) return false;
    value = it->second;
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
  struct Shard {
    mutable std::mutex mu;
    std::unordered_map<Key, Value, Hash> items;
  };

  Shard &shard_for(const Key &key) { return shards_[hasher_(key) & (ShardCount - 1)]; }
  const Shard &shard_for(const Key &key) const { return shards_[hasher_(key) & (ShardCount - 1)]; }

  Hash hasher_;
  std::array<Shard, ShardCount> shards_;
  std::atomic<size_t> size_{0};
};

}  // namespace v2
