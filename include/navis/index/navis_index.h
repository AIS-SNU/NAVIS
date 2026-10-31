#pragma once

#include "navis/types.h"
#include "navis/casr.h"
#include "navis/distance.h"
#include "navis/pq_table.h"
#include "navis/separated_layout.h"
#include "navis/query_buffer.h"
#include "navis/io/io_backend.h"
#include "navis/cache/page_cache.h"
#include "navis/cache/navis_cache.h"
#include "navis/lock/vertex_lock.h"
#include "navis/lock/page_lock.h"
#include "navis/index/entrance_graph.h"
#include "navis/index/indirection_table.h"
#include "navis/index/deletion_table.h"
#include "navis/util/concurrent_queue.h"

#include <tsl/robin_map.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace navis {

// ---------------------------------------------------------------------------
// BgWriteTask — storage writes of one insertion, handed to the bg writer
// ---------------------------------------------------------------------------
struct BgWriteTask {
    std::vector<IORequest>  edge_writes;       // writes to _disk.edge
    std::vector<IORequest>  vec_writes;        // writes to _disk.vec
    std::vector<uint64_t>   pages_to_unlock;   // page locks to release after write
    std::vector<uint64_t>   pages_to_deref;    // RMW cache pins to drop after write
    std::vector<void*>      owned_bufs;        // DMA buffers freed after write
};

// ---------------------------------------------------------------------------
// NavisIndex — on-disk graph-based vector index with concurrent search,
// insertion, and deletion.
//
// Template parameter T is the vector element type (float, int8_t, uint8_t).
// ---------------------------------------------------------------------------
template <typename T>
class NavisIndex {
public:
    NavisIndex() = default;
    ~NavisIndex();

    NavisIndex(const NavisIndex&) = delete;
    NavisIndex& operator=(const NavisIndex&) = delete;

    // Load index from disk: metadata, PQ table and codes, indirection table,
    // and the entrance graph ({prefix}_mem.index, if present).
    //   io_edge / io_vec: backends opened on {prefix}_disk.edge / _disk.vec
    //   n_threads:        number of worker threads (sizes the buffer pool)
    void load(const std::string& prefix,
              IOBackend* io_edge,
              IOBackend* io_vec,
              const SearchConfig& config,
              const CacheConfig& cache_cfg,
              uint32_t n_threads,
              Metric metric = Metric::L2);

    // Top-K search.
    //   mem_L:      entrance-graph search list size (0 = start from the medoid)
    //   search_L:   on-disk search frontier size |E_search|
    //   beam_width: edge pages read per traversal hop
    // Unused output slots are filled with UINT32_MAX / FLT_MAX.
    void search(const T* query, uint64_t K, uint32_t mem_L,
                uint64_t search_L, uint32_t beam_width,
                uint32_t* out_ids, float* out_dists,
                QueryStats* stats = nullptr);

    // Insert `point` with external id `tag` (the tag becomes the vertex id,
    // so tags must be unique and >= the initial number of points).
    // Returns the vertex id, or -1 on failure.
    int insert(const T* point, uint32_t tag);

    // Lazily delete vertex `tag`: it is excluded from results and new
    // neighbor lists immediately, while remaining traversable. Returns false
    // if it was already deleted or out of range.
    bool erase(uint32_t tag);
    bool is_deleted(uint32_t id) const { return deletion_table_.is_deleted(id); }
    uint64_t deleted_count() const { return deletion_table_.deleted_count(); }
    double deleted_fraction() const {
        return deletion_table_.deleted_fraction(cur_id_.load(std::memory_order_acquire));
    }

    // --- Accessors ---
    const SeparatedLayout& layout() const { return layout_; }
    uint64_t num_points() const { return num_points_; }
    uint64_t num_active_points() const { return cur_id_.load(std::memory_order_relaxed); }
    uint64_t data_dim() const { return layout_.data_dim; }

    // --- Statistics ---
    uint64_t navis_cache_hits()   const { return navis_cache_.hits(); }
    uint64_t navis_cache_misses() const { return navis_cache_.misses(); }
    void     navis_cache_reset_stats() { navis_cache_.reset_stats(); }

    // Mean full-vector loads per CASR rerank, per path.
    double search_vec_loads_mean() const { return mean_loads(search_vec_loads_total_, search_vec_loads_calls_); }
    double insert_vec_loads_mean() const { return mean_loads(insert_vec_loads_total_, insert_vec_loads_calls_); }
    void reset_vec_load_stats() {
        search_vec_loads_total_.store(0, std::memory_order_relaxed);
        search_vec_loads_calls_.store(0, std::memory_order_relaxed);
        insert_vec_loads_total_.store(0, std::memory_order_relaxed);
        insert_vec_loads_calls_.store(0, std::memory_order_relaxed);
    }
    // Calibrated CASR window per path: w_search, w_insert (0 while warming up).
    uint32_t casr_window_search() const { return casr_cal_search_.window(); }
    uint32_t casr_window_insert() const { return casr_cal_insert_.window(); }

private:
    // --- Core state ---
    IOBackend*                     io_edge_ = nullptr;
    IOBackend*                     io_vec_  = nullptr;
    SeparatedLayout                layout_;
    IndirectionTable               indirection_;
    DeletionTable                  deletion_table_;
    FixedChunkPQTable<T>           pq_table_;
    std::vector<uint8_t>           pq_data_;    // [capacity * n_chunks]
    std::unique_ptr<Distance<T>>   dist_cmp_;
    std::unique_ptr<EntranceGraph<T>> entrance_graph_;

    // --- Caching: RMW page cache (in-flight writes) + NAVIS-Cache (reads) ---
    PageCache                      rmw_cache_;
    NavisCache                     navis_cache_;

    // --- Locking ---
    VertexLockTable                vertex_locks_;
    PageLockTable                  page_locks_;

    // --- Index metadata ---
    uint64_t                       num_points_  = 0;
    uint64_t                       aligned_dim_ = 0;
    uint64_t                       n_chunks_    = 0;
    std::vector<uint32_t>          medoids_;
    Metric                         metric_      = Metric::L2;

    // --- Id allocation for insertion (high-water mark of assigned ids) ---
    std::atomic<uint64_t>          cur_id_{0};

    // --- Background writer ---
    ConcurrentQueue<BgWriteTask*>  bg_write_queue_;
    std::vector<std::thread>       bg_write_threads_;
    void bg_write_worker();

    // --- Per-thread buffer pool ---
    ConcurrentQueue<QueryBuffer<T>*> buffer_pool_;
    std::vector<QueryBuffer<T>*>     all_buffers_;
    void init_buffer_pool(uint32_t n_threads);
    QueryBuffer<T>* acquire_buffer();
    void release_buffer(QueryBuffer<T>* buf);

    SearchConfig config_;

    // --- CASR window calibration, one per path (search, position seeking):
    //     they explore very different set sizes and converge differently.
    CasrCalibrator casr_cal_search_;
    CasrCalibrator casr_cal_insert_;

    // --- Vector-load counters (CASR), per path ---
    std::atomic<uint64_t> search_vec_loads_total_{0};
    std::atomic<uint64_t> search_vec_loads_calls_{0};
    std::atomic<uint64_t> insert_vec_loads_total_{0};
    std::atomic<uint64_t> insert_vec_loads_calls_{0};
    static double mean_loads(const std::atomic<uint64_t>& total,
                             const std::atomic<uint64_t>& calls) {
        uint64_t n = calls.load(std::memory_order_relaxed);
        return n == 0 ? 0.0 : double(total.load(std::memory_order_relaxed)) / double(n);
    }

    // --- Search / insert internals (navis_search.cpp / navis_insert.cpp) ---
    void do_search(const T* query, uint32_t mem_L, uint64_t search_L,
                   uint32_t beam_width,
                   std::vector<Neighbor>& results,
                   tsl::robin_map<uint32_t, T*>* coord_map,
                   QueryStats* stats,
                   bool is_position_seeking,
                   QueryBuffer<T>* buf,
                   std::vector<Neighbor>* out_explored     = nullptr,
                   std::vector<Neighbor>* out_explored_ent = nullptr);

    int do_insert(uint32_t target_id, const T* point,
                  const std::vector<uint32_t>& neighbors,
                  QueryBuffer<T>* buf);

    // Enlarge the frontier to compensate for deleted vertices still on the
    // graph (capped at MAX_N_CMPS - 1 regardless).
    uint64_t deletion_adjusted_search_L(uint64_t requested_L) const;

    void prune_neighbors(const std::vector<Neighbor>& candidates,
                         std::vector<uint32_t>& pruned,
                         tsl::robin_map<uint32_t, T*>* coord_map);

    // PQ distance between one source vertex and multiple targets.
    void compute_pq_dists_internal(uint32_t src, const uint32_t* ids,
                                   float* dists, uint32_t count,
                                   uint8_t* scratch) const;

    // PQ distance table of a source vertex (inflate + chunk distances), so a
    // loop-invariant source can reuse it across many lookups.
    void precompute_src_chunk_dists(uint32_t src, float* chunk_dists_out) const;

    // Delta pruning for a neighbor list that overflows when the new vertex is
    // appended.
    struct TriangleNeighbor {
        uint32_t id;
        float tgt_dis;   // distance from the new vertex to this neighbor
        float distance;  // distance from the overflowing vertex to this neighbor

        bool operator<(const TriangleNeighbor& o) const {
            return (distance < o.distance) || (distance == o.distance && id < o.id);
        }
        bool operator==(const TriangleNeighbor& o) const { return id == o.id; }
    };
    void delta_prune(std::vector<TriangleNeighbor>& pool,
                     std::vector<uint32_t>& pruned_list,
                     uint8_t* scratch, int tgt_idx);
};

extern template class NavisIndex<float>;
extern template class NavisIndex<int8_t>;
extern template class NavisIndex<uint8_t>;

} // namespace navis
