#pragma once
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <string>
#include <set>

// Include lock table implementations BEFORE other headers

#include "v2/page_cache.h"
#include "v2/odin_packed_cache.h"
#include "v2/concurrent_maps.h"
#include <iostream>
#include <iomanip>
#include <omp.h>

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "neighbor.h"
#include "index.h"
#include <queue>
#include <thread>
#include <condition_variable>

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 128
#define SECTOR_LEN 4096
#define EDGE_SECTOR_LEN 4096
#define META_SECTOR_LEN 4096

constexpr int kIndexSizeFactor = 2;

enum SearchMode { BEAM_SEARCH = 0, PAGE_SEARCH = 1 };

namespace {
  inline void aggregate_coords(const unsigned *ids, const _u64 n_ids, const _u8 *all_coords, const _u64 ndims,
                               _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  inline void prefetch_chunk_dists(const float *ptr) {
    _mm_prefetch((char *) ptr, _MM_HINT_NTA);
    _mm_prefetch((char *) (ptr + 64), _MM_HINT_NTA);
    _mm_prefetch((char *) (ptr + 128), _MM_HINT_NTA);
    _mm_prefetch((char *) (ptr + 192), _MM_HINT_NTA);
  }

  inline void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts, const _u64 pq_nchunks, const float *pq_dists,
                             float *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);

    prefetch_chunk_dists(pq_dists);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        prefetch_chunk_dists(chunk_dists + 256);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }
}  // namespace

namespace pipeann {

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader, bool single_file_index,
             bool tags = false, Parameters *parameters = nullptr);

    ~SSDIndex();

    // returns region of `node_buf` containing [COORD(T)]
    inline T *offset_to_node_coords(const char *node_buf) {
      return (T *) node_buf;
    }

    // returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
    inline unsigned *offset_to_node_nhood(const char *node_buf) {
      return (unsigned *) (node_buf + data_dim * sizeof(T));
    }

    // obtains region of sector containing node
    inline char *offset_to_node(const char *sector_buf, uint32_t node_id) {
      return offset_to_loc(sector_buf, id2loc(node_id));
    }

    // sector # on disk where node_id is present
    inline uint64_t node_sector_no(uint32_t node_id) {
      return loc_sector_no(id2loc(node_id));
    }

    inline uint64_t u_node_offset(uint32_t node_id) {
      return u_loc_offset(id2loc(node_id));
    }

    // unaligned offset to location
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * max_node_len;  // compacted store.
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) {
      return loc * max_node_len + data_dim * sizeof(T);
    }

    inline char *offset_to_loc(const char *sector_buf, uint64_t loc) {
      return (char *) sector_buf + (nnodes_per_sector == 0 ? 0 : (loc % nnodes_per_sector) * max_node_len);
    }

    // avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (nnodes_per_sector > 0 ? loc / nnodes_per_sector : loc * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return (sector_no - 1) * nnodes_per_sector + sector_off;
    }

    void init_query_buf(QueryBuffer<T> &buf, _u64 n_chunks = 0) {
      _u64 coord_alloc_size = ROUND_UP(MAX_N_CMPS * this->aligned_dim * sizeof(T), 256);
      pipeann::alloc_aligned((void **) &buf.coord_scratch, coord_alloc_size, 256);
      pipeann::alloc_aligned((void **) &buf.sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.aligned_pq_coord_scratch, 32768 * 32 * sizeof(_u8), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_pqtable_dist_scratch, 256 * MAX_PQ_CHUNKS * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, 512 * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      pipeann::alloc_aligned((void **) &buf.update_buf, (2 * MAX_N_EDGES + 1) * SECTOR_LEN,
                             SECTOR_LEN);  // 2x for read + write

      buf.visited = new tsl::robin_set<_u64>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * SECTOR_LEN);
      memset(buf.coord_scratch, 0, coord_alloc_size);
      memset(buf.aligned_query_T, 0, this->aligned_dim * sizeof(T));
      memset(buf.update_buf, 0, (2 * MAX_N_EDGES + 1) * SECTOR_LEN);
    }

    QueryBuffer<T> *pop_query_buf(const T *query) {
      QueryBuffer<T> *data = this->thread_data_queue.pop();
      while (data == nullptr) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (data_is_normalized) {
          // Data has been normalized. Normalize search vector too.
          float norm = pipeann::compute_l2_norm(query, this->data_dim);
          for (uint32_t i = 0; i < this->data_dim; i++) {
            data->aligned_query_T[i] = query[i] / norm;
          }
        } else {
          memcpy(data->aligned_query_T, query, this->data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer<T> *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // load compressed data, and obtains the handle to the disk-resident index
    int load(const char *index_prefix, uint32_t num_threads, bool new_index_format = true,
             bool use_page_search = false);

    void load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path);

    void load_page_layout(const std::string &index_prefix, const _u64 nnodes_per_sector = 0, const _u64 num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);

    // Enable the packed-layout read cache. Must be called before insertions start.
    // byte_budget = 0 disables (no-op). Re-calling replaces the cache.
    void enable_odin_packed_cache(size_t byte_budget,
                                  double frozen_ratio = 0.98,
                                  uint32_t promote_threshold = 2,
                                  uint32_t eviction_probes = 8) {
      if (byte_budget == 0) { odin_packed_cache_.reset(); return; }
      odin_packed_cache_.reset(new v2::OdinPackedCache());
      v2::OdinPackedCacheConfig cfg{};
      cfg.size_gb           = byte_budget / (1024ULL * 1024ULL * 1024ULL);
      cfg.window_ratio      = 1.0 - frozen_ratio;
      cfg.promote_threshold = promote_threshold;
      cfg.eviction_probes   = eviction_probes;
      odin_packed_cache_->init(cfg);
    }

    v2::OdinPackedCache* odin_packed_cache() { return odin_packed_cache_.get(); }

    _u64 return_nd();

    // search supporting update.
    size_t beam_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, TagT *res_tags,
                       float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr,
                       tsl::robin_set<uint32_t> *deleted_nodes = nullptr, bool dyn_search_l = true);


    // read-only search algorithms.
    size_t page_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, TagT *res_tags,
                       float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr);

    std::vector<uint32_t> get_init_ids() {
      return std::vector<uint32_t>(this->medoids, this->medoids + this->num_medoids);
    }

    // computes PQ dists between src->[ids] into fp_dists (merge, insert)
    void compute_pq_dists(const _u32 src, const _u32 *ids, float *fp_dists, const _u32 count,
                          uint8_t *aligned_scratch = nullptr);

    // deflates `vec` into PQ ids
    std::vector<_u8> deflate_vector(const T *vec);
    std::pair<_u8 *, _u32> get_pq_config() {
      return std::make_pair(this->data.data(), (uint32_t) this->n_chunks);
    }

    _u64 get_num_frozen_points() {
      return this->num_frozen_points;
    }

    _u64 get_frozen_loc() {
      return this->frozen_location;
    }

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;
  

   protected:
    void use_medoids_data_as_centroids();
    void init_buffers(_u64 nthreads);
    void destroy_thread_data();

   public:
    // data info
    _u64 num_points = 0;
    _u64 init_num_pts = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 aligned_dim = 0;
    _u64 size_per_io = 0;
  

    std::string _disk_index_file;

    std::shared_ptr<AlignedFileReader> &reader;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    std::vector<_u8> data;
    _u64 chunk_size;
    _u64 n_chunks;
    FixedChunkPQTable<T> pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>> dist_cmp;

   public:
    // in-place update.
    int insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set = nullptr);

    void disk_iterate_to_fixed_point_dyn(const T *vec, const uint32_t Lsize, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         QueryBuffer<T> *passthrough_data, tsl::robin_set<uint32_t> *exclude_nodes,
                                         std::vector<uint64_t> *page_ref);
    void do_beam_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                        QueryStats *stats = nullptr, tsl::robin_set<uint32_t> *exclude_nodes = nullptr,
                        bool dyn_search_l = true, std::vector<uint64_t> *passthrough_page_ref = nullptr,
                        bool is_position_seeking = false);

    void occlude_list(std::vector<Neighbor> &pool, const tsl::robin_map<uint32_t, T *> &coord_map,
                      std::vector<Neighbor> &result, std::vector<float> &occlude_factor);
    void prune_neighbors(const tsl::robin_map<uint32_t, T *> &coord_map, std::vector<Neighbor> &pool,
                         std::vector<uint32_t> &pruned_list);
    void occlude_list_pq(std::vector<Neighbor> &pool, std::vector<Neighbor> &result, std::vector<float> &occlude_factor,
                         uint8_t *scratch);
    void prune_neighbors_pq(std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list, uint8_t *scratch);

    // delta pruning.
    struct TriangleNeighbor {
      unsigned id;
      float tgt_dis;
      float distance;

      inline bool operator<(const TriangleNeighbor &other) const {
        return (distance < other.distance) || (distance == other.distance && id < other.id);
      }
      inline bool operator==(const TriangleNeighbor &other) const {
        return (id == other.id);
      }
    };
    void delta_prune_neighbors_pq(std::vector<TriangleNeighbor> &pool, std::vector<uint32_t> &pruned_list,
                                  uint8_t *scratch, int tgt_idx);
    void reload(const char *index_prefix, uint32_t num_threads);
    // background I/O commit.
    struct BgTask {
      QueryBuffer<T> *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      std::chrono::high_resolution_clock::time_point enqueue_time; // Added
    };
    // its concurrency should not be the bottleneck.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    void bg_io_thread();
    static constexpr int kBgIOThreads = 1;
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // Global stats for reporting
    std::atomic<uint64_t> global_search_us{0};
    std::atomic<uint64_t> global_structural_update_us{0};
    std::atomic<uint64_t> global_insert_op_count{0};

    // test the estimation efficacy.
    uint32_t beam_width, l_index, range, maxc;
    float alpha;
    // assumed max thread, only the first nthreads are initialized.
    AlignedFileReader *pq_reader = nullptr;
    
    // Fixed striped locks avoid dynamic registry churn on the search/update hot path.
    v2::SparseLockTable<uint64_t> page_lock_table{"page"};

    v2::SparseLockTable<uint64_t> vec_lock_table{"vec"};
    v2::SparseLockTable<uint64_t> page_idx_lock_table{"page_idx"};
    v2::SparseLockTable<uint64_t> idx_lock_table{"idx"};
    std::shared_mutex merge_lock;  // serve search during merge.

    // if ID == tag, then it is not stored.
    v2::SegmentedAtomicMap<TagT> tags;
    TagT id2tag(uint32_t id) {
#ifdef NO_MAPPING
      return id;  // use ID to replace tags.
#else
      TagT ret;
      if (tags.find(id, ret)) {
        return ret;
      } else {
        return id;
      }
#endif
    }

    int get_vector_by_id(const uint32_t &id, T *vector);
    static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;

    static std::vector<uint32_t> lock_vec(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                          const std::vector<uint32_t> &neighbors, bool rd = false) {
      std::vector<uint32_t> to_lock = get_to_lock_idx(lock_table, target, neighbors);
      for (auto &id : to_lock) {
        rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
      }
      return to_lock;
    }

    static void unlock_vec(v2::SparseLockTable<uint64_t> &lock_table,
                           const std::vector<uint32_t> &to_lock) {
      for (auto &id : to_lock) {
        lock_table.unlock(id);
      }
    }

    static void unlock_vec(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                           const std::vector<uint32_t> &neighbors) {
      unlock_vec(lock_table, get_to_lock_idx(lock_table, target, neighbors));
    }

   private:
    static std::vector<uint32_t> get_to_lock_idx(uint32_t target,
                                                  const std::vector<uint32_t> &neighbors) {
      std::vector<uint32_t> to_lock;
      to_lock.assign(neighbors.begin(), neighbors.end());
      if (target != kInvalidID) {
        to_lock.push_back(target);
      }
      // sort and deduplicate
      std::sort(to_lock.begin(), to_lock.end());
      auto last = std::unique(to_lock.begin(), to_lock.end());
      to_lock.erase(last, to_lock.end());
      return to_lock;
    }

    static std::vector<uint32_t> get_to_lock_idx(v2::SparseLockTable<uint64_t> &lock_table,
                                                  uint32_t target,
                                                  const std::vector<uint32_t> &neighbors) {
      std::vector<uint32_t> to_lock = get_to_lock_idx(target, neighbors);
      lock_table.sort_and_deduplicate(to_lock);
      return to_lock;
    }

   public:
    // lock the mapping for target/page if use_page_search == false/true.
    std::vector<uint32_t> lock_idx(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false) {
#ifndef READ_ONLY_TESTS
      std::vector<uint32_t> to_lock = get_to_lock_idx(lock_table, target, neighbors);
      for (auto &id : to_lock) {
        rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
      }
      return to_lock;
#else
      return std::vector<uint32_t>();
#endif
    }

    void unlock_idx(v2::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
      for (auto &id : to_lock) {
        lock_table.unlock(id);
      }
#endif
    }

    void unlock_idx(v2::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock) {
#ifndef READ_ONLY_TESTS
      lock_table.unlock(to_lock);
#endif
    }

    // two-level, as id2page may change before and after grabbing the lock.
    std::vector<uint32_t> lock_page_idx(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                        const std::vector<uint32_t> &neighbors, bool rd = false) {
#ifndef READ_ONLY_TESTS
      if (!use_page_search_) {
        return std::vector<uint32_t>();
      }
      std::vector<uint32_t> to_lock(neighbors.begin(), neighbors.end());
      if (target != kInvalidID) {
        to_lock.push_back(target);
      }

      for (size_t i = 0; i < to_lock.size(); ++i) {
        to_lock[i] = id2page(to_lock[i]);
      }

      lock_table.sort_and_deduplicate(to_lock);

      for (auto &id : to_lock) {
        rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
      }
      return to_lock;
#else
      return std::vector<uint32_t>();
#endif
    }

    void unlock_page_idx(v2::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
      if (!use_page_search_) {
        return;
      }
      for (auto &id : to_lock) {
        lock_table.unlock(id);
      }
#endif
    }

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;

    // Global counters for insertion hops
    std::atomic<uint64_t> total_hops_front{0};
    std::atomic<uint64_t> total_hops_pos{0};

    uint64_t get_total_hops_front() const { return total_hops_front.load(); }
    uint64_t get_total_hops_pos() const { return total_hops_pos.load(); }

    // page search
    bool use_page_search_ = true;

    v2::SegmentedAtomicMap<uint32_t> id2loc_;  // id -> loc (start from 0)
    uint32_t id2loc(uint32_t id) {
#ifdef NO_MAPPING
      return id;
#else
      uint32_t loc = 0;
      if (id2loc_.find(id, loc)) {
        return loc;
      } else {
        // RACE CONDITION HANDLING: During concurrent insertion, id2loc mapping may not exist yet
        // for IDs that have been allocated but not fully inserted. Return kInvalidID to signal
        // search code to skip this ID gracefully (it's being inserted by another thread).
        // Only crash if ID is beyond cur_id (truly invalid).
        uint64_t limit = cur_id.load(std::memory_order_acquire);
        if (id >= limit) {
          LOG(ERROR) << "id " << id << " not found in id2loc (limit=" << limit << ")";
          crash();
        }
        // ID is < cur_id but not in id2loc yet - being inserted concurrently
        return kInvalidID;
      }
#endif
    }

    uint64_t id2page(uint32_t id) {
      uint32_t loc = id2loc(id);
      if (loc == kInvalidID) {
        return kInvalidID;
      }
      return loc_sector_no(loc);
    }

    static constexpr uint32_t kMaxElemInAPage = 16;
    using PageArr = std::array<uint32_t, kMaxElemInAPage>;
    v2::SegmentedAtomicMap<uint32_t> loc2id_;  // loc -> id

    std::mutex alloc_lock;
    uint32_t loc2id(uint32_t loc) {
      uint32_t id = kInvalidID;
      loc2id_.find(loc, id);
      return id;
    }

    void set_loc2id(uint32_t loc, uint32_t id) {
      loc2id_.insert_or_assign(loc, id);
    }

    void erase_loc2id(uint32_t loc) {
      const uint32_t page = loc_sector_no(loc);
      loc2id_.erase(loc);
      bool empty = true;
      for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
        uint32_t id = kInvalidID;
        if (loc2id_.find(sector_to_loc(page, i), id)) {
          empty = false;
          break;
        }
      }
      if (empty) empty_pages.push(page);
    }

    bool get_page_layout(uint32_t page, PageArr &layout) {
      layout.fill(kInvalidID);
      bool found = false;
      for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
        uint32_t id = kInvalidID;
        if (loc2id_.find(sector_to_loc(page, i), id)) {
          layout[i] = id;
          found = true;
        }
      }
      return found;
    }

    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids) {
      std::lock_guard<std::mutex> lock(alloc_lock);
      for (uint32_t i = 0; i < new_locs.size(); ++i) {
        set_loc2id(new_locs[i], new_ids[i]);
      }
      for (auto &l : old_locs) {
        erase_loc2id(l);
      }
    }

    // returns <loc, need_read>
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read) {
      std::lock_guard<std::mutex> lock(alloc_lock);
      std::vector<uint64_t> ret;
      int cur = 0;
      // reuse.
      uint32_t threshold = (nnodes_per_sector + kIndexSizeFactor - 1) / kIndexSizeFactor;

      uint32_t empty_page = kInvalidID;
      while ((empty_page = empty_pages.pop()) != kInvalidID) {
#ifdef NO_POLLUTE_ORIGINAL
        if (empty_page < loc_sector_no(init_num_pts)) {
          continue;
        }
#endif
        // allocate all the locations in the recycled page.
        for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
          const uint32_t loc = sector_to_loc(empty_page, i);
          const uint32_t id = loc2id(loc);
          if (id != kInvalidID) {
            LOG(ERROR) << "Page " << empty_page << " is not empty " << i << " " << id;
            crash();
          }
          set_loc2id(loc, kAllocatedID);
          ret.push_back(loc);
          cur++;
          if (cur == n) break;
        }
        if (cur == n) {
          return ret;
        }
      }

      for (auto &p : hint_pages) {
#ifdef NO_POLLUTE_ORIGINAL
        if (p < loc_sector_no(init_num_pts)) {
          continue;
        }
#endif
        // see the hole number
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
          if (loc2id(sector_to_loc(p, i)) == kInvalidID) cnt++;
        }
        if (cnt < threshold) continue;

        if (cnt < nnodes_per_sector) page_need_to_read.insert(p);
        for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
          const uint32_t loc = sector_to_loc(p, i);
          if (loc2id(loc) == kInvalidID) {
            set_loc2id(loc, kAllocatedID);
            ret.push_back(loc);
            cur++;
            if (cur == n) break;
          }
        }

        if (cur == n) {
          return ret;
        }
      }

      // allocate new space.
      int remaining = n - cur;
      for (int i = 0; i < remaining; i++) {
        set_loc2id(cur_loc + i, kAllocatedID);
        ret.push_back(cur_loc + i);
      }

      // ensure that cur_loc is aligned.
      // the hole will eventually be recycled using either empty page queue or hint pages.
      cur_loc += remaining;
      if (cur_loc % nnodes_per_sector != 0) {
        cur_loc += (nnodes_per_sector - (cur_loc % nnodes_per_sector));
      }
      return ret;
    }

    void verify_id2loc() {
      // verify id -> loc -> id map.
      LOG(INFO) << "ID2loc size: " << id2loc_.size() << ", cur_loc: " << cur_loc.load() << ", cur_id: " << cur_id
                << ", nnodes_per_sector: " << nnodes_per_sector;
      for (uint32_t i = 0; i < cur_id; ++i) {
        auto loc = id2loc(i);
        if (unlikely(loc >= cur_loc.load())) {
          LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << loc << ", cur_loc: " << cur_loc.load();
          crash();
        }
        if (unlikely(loc2id(loc) != i)) {
          LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << id2loc(i)
                     << ", loc2id: " << loc2id(id2loc(i));
          crash();
        }
      }

      LOG(INFO) << "ID2loc consistency check passed.";

      // verify loc2id do not contain duplicate ids.
      for (uint32_t i = 0; i < cur_loc; ++i) {
        auto id = loc2id(i);
        if (id != kInvalidID && id != kAllocatedID) {
          uint32_t loc = id2loc(id);
          if (unlikely(loc != i)) {
            LOG(ERROR) << "loc2id inconsistency at loc: " << i << ", id: " << id << ", loc: " << loc;
          }
        }
      }
      LOG(INFO) << "loc2ID consistency check passed.";
    }
    std::atomic<_u64> cur_id, cur_loc;

    // merge deletes (NOTE: index read-only during merge.)
    void merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                       const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                       uint32_t nthreads, const uint32_t &n_sampled_nbrs);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               const uint64_t &new_npoints, const uint64_t &new_medoid,
                               std::vector<TagT> *new_tags = nullptr);

   private:
    // Are we dealing with normalized data? This will be true
    // if distance == COSINE and datatype == float. Required
    // because going forward, we will normalize vectors when
    // asked to search with COSINE similarity. Of course, this
    // will be done only for floating point vectors.
    bool data_is_normalized = false;

    // medoid/start info
    uint32_t *medoids = nullptr;  // by default it is just one entry point of graph, we
                                  // can optionally have multiple starting points
    size_t num_medoids = 1;       // by default it is set to 1

    // thread-specific scratch
    ConcurrentQueue<QueryBuffer<T> *> thread_data_queue;
    std::vector<QueryBuffer<T> *> thread_data_bufs;  // pre-allocated thread data
    std::vector<uint8_t *> thread_pq_bufs;           // for merge deletes
    _u64 max_nthreads;

    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing

    // Packed-layout read cache (W-TinyLFU, ported from NAVIS).
    // Default nullptr → baseline behavior (no cache).
    std::unique_ptr<v2::OdinPackedCache> odin_packed_cache_;

    // Batched 3-tier cached read for beam-search traversal.
    //   1. RMW v2::cache → hit serves + refs.
    //   2. OdinPackedCache (if enabled; admission mode driven by is_position_seeking).
    //   3. Disk via reader->read_alloc or reader->read; on miss, admit into OdinPackedCache.
    // Signature matches existing `reader->read_alloc(reads, ctx, &page_ref)` call site.
    void cached_sector_read_batch(std::vector<IORequest> &reads,
                                  void *ctx,
                                  std::vector<uint64_t> *page_ref,
                                  bool is_position_seeking);
  };
}  // namespace pipeann
