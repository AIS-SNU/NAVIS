#pragma once

#include "navis/types.h"
#include "navis/distance.h"

#include <tsl/robin_map.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace navis {

// ---------------------------------------------------------------------------
// EntranceGraph — in-memory proximity graph for entry point selection
//
// A small (1% sample) in-memory graph used to find good starting points
// for on-disk traversal. Supports NAVIS-Update: incremental insertion of
// new vertices to keep the graph fresh during continuous insertions.
// ---------------------------------------------------------------------------
template <typename T>
class EntranceGraph {
public:
    EntranceGraph() = default;
    ~EntranceGraph();

    // Load from {prefix}_mem.index files (DiskANN in-memory index format)
    //   prefix: e.g., "/data/sift1m_mem.index"
    //   dist:   distance function
    //   dim:    vector dimension
    void load(const std::string& prefix, Distance<T>* dist, uint64_t dim);

    // Search the entrance graph for entry points
    //   query:     query vector [dim]
    //   L:         search list size
    //   out_ids:   output node IDs (mapped to on-disk IDs via tags)
    //   out_dists: output distances
    //   explored:  output explored set (for NAVIS-Update reuse)
    // Returns number of results found
    uint32_t search(const T* query, uint32_t L,
                    std::vector<Neighbor>& results,
                    std::vector<Neighbor>* explored = nullptr);

    // NAVIS-Update: maybe promote a newly inserted vertex into the entrance graph
    //   id:               on-disk vertex ID
    //   vec:              vertex coordinates
    //   total_disk_pts:   total points in on-disk index
    //   explored_ent:     explored set from entrance graph search
    //   explored_disk:    explored set from on-disk search
    //   config:           search config (ent_ratio, ent_R)
    void maybe_promote(uint32_t id, const T* vec,
                       uint64_t total_disk_pts,
                       const std::vector<Neighbor>& explored_ent,
                       const std::vector<Neighbor>& explored_disk,
                       const SearchConfig& config,
                       const std::function<bool(uint32_t)>& is_deleted = {});

    uint64_t size() const { return num_points_; }
    bool is_loaded() const { return loaded_; }

private:
    // Graph storage (adjacency list)
    std::vector<std::vector<uint32_t>> graph_;   // internal IDs
    T*                                 data_     = nullptr; // [num_points * dim]
    std::vector<uint32_t>              tags_;     // internal ID -> on-disk ID
    uint64_t                           num_points_ = 0;
    uint64_t                           dim_        = 0;
    uint32_t                           max_degree_ = 32;
    uint32_t                           medoid_     = 0;
    Distance<T>*                       dist_       = nullptr;
    bool                               loaded_     = false;

    mutable std::shared_mutex          graph_mu_;   // shared for search, exclusive for NAVIS-Update
    std::atomic<uint64_t>              nav_size_{0};
    uint64_t                           data_capacity_{0}; // reserved slots in data_

    // Cached on-disk-id → internal-id index. Populated once at load(), updated
    // incrementally on promote. Avoids the per-call O(num_points_) rebuild that
    // previously dominated ENT_PROMOTE on long sweeps. Read under graph_mu_
    // shared lock, written under graph_mu_ exclusive lock.
    tsl::robin_map<uint32_t, uint32_t> disk_to_internal_;
};

extern template class EntranceGraph<float>;
extern template class EntranceGraph<int8_t>;
extern template class EntranceGraph<uint8_t>;

} // namespace navis
