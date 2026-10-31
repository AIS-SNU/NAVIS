#include "navis/index/entrance_graph.h"
#include "navis/util/file_utils.h"

#include <tsl/robin_set.h>
#include <tsl/robin_map.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

namespace navis {

template <typename T>
EntranceGraph<T>::~EntranceGraph() {
    free_aligned(data_);
}

template <typename T>
void EntranceGraph<T>::load(const std::string& prefix,
                            Distance<T>* dist, uint64_t dim) {
    dist_ = dist;
    dim_  = dim;

    // Load graph structure from Vamana save format:
    // [8B total_size][4B max_degree][4B entry_point][8B frozen_num][per-node: 4B deg + neighbors]
    std::string graph_file = prefix;
    std::ifstream gin(graph_file, std::ios::binary);
    if (!gin.is_open()) {
        throw std::runtime_error("Cannot open entrance graph: " + graph_file);
    }

    // Vamana graph format: [8B total_size][4B max_degree][4B entry_point][8B frozen_num]
    // followed by per-node: [4B degree][degree * 4B neighbor_ids]
    uint64_t index_file_size;
    uint32_t max_deg, entry_point;
    uint64_t frozen_num;

    gin.read(reinterpret_cast<char*>(&index_file_size), sizeof(uint64_t));
    gin.read(reinterpret_cast<char*>(&max_deg), sizeof(uint32_t));
    gin.read(reinterpret_cast<char*>(&entry_point), sizeof(uint32_t));
    gin.read(reinterpret_cast<char*>(&frozen_num), sizeof(uint64_t));

    max_degree_ = max_deg;

    // Count nodes by reading the graph data (total_size includes header)
    // Header is 24 bytes (8+4+4+8). Remaining bytes = graph data.
    // We read nodes until EOF or exhausted size.
    std::streampos data_start = gin.tellg();
    gin.seekg(0, std::ios::end);
    std::streampos file_end = gin.tellg();
    gin.seekg(data_start);

    // Read all nodes
    graph_.clear();
    while (gin.tellg() < file_end && gin.good()) {
        uint32_t deg;
        gin.read(reinterpret_cast<char*>(&deg), sizeof(uint32_t));
        if (gin.fail()) break;

        std::vector<uint32_t> neighbors(deg);
        gin.read(reinterpret_cast<char*>(neighbors.data()), deg * sizeof(uint32_t));
        if (gin.fail()) break;

        graph_.push_back(std::move(neighbors));
    }
    num_points_ = static_cast<uint64_t>(graph_.size());

    // Load vectors: prefix + ".data"
    std::string data_file = prefix + ".data";
    if (file_exists(data_file)) {
        uint32_t d_npts, d_dim;
        load_bin<T>(data_file, data_, d_npts, d_dim);
        if (d_npts != num_points_ || d_dim != dim_) {
            throw std::runtime_error("Entrance graph data dimension mismatch");
        }
    }

    // Load tags (internal ID -> on-disk ID mapping): prefix + ".tags"
    std::string tag_file = prefix + ".tags";
    if (file_exists(tag_file)) {
        uint32_t t_npts, t_dim;
        uint32_t* tag_data = nullptr;
        load_bin<uint32_t>(tag_file, tag_data, t_npts, t_dim);
        tags_.assign(tag_data, tag_data + t_npts);
        free_aligned(tag_data);
    } else {
        // Identity mapping
        tags_.resize(num_points_);
        for (uint64_t i = 0; i < num_points_; i++) {
            tags_[i] = static_cast<uint32_t>(i);
        }
    }

    // Find medoid: node closest to the centroid of all vectors
    if (data_ && num_points_ > 0) {
        std::vector<double> center(dim_, 0.0);
        for (uint64_t i = 0; i < num_points_; i++) {
            for (uint64_t j = 0; j < dim_; j++) {
                center[j] += static_cast<double>(data_[i * dim_ + j]);
            }
        }
        for (uint64_t j = 0; j < dim_; j++) center[j] /= num_points_;

        float best_dist = std::numeric_limits<float>::max();
        medoid_ = 0;
        for (uint32_t i = 0; i < num_points_; i++) {
            float d = 0;
            for (uint64_t j = 0; j < dim_; j++) {
                float diff = static_cast<float>(center[j]) -
                             static_cast<float>(data_[i * dim_ + j]);
                d += diff * diff;
            }
            if (d < best_dist) {
                best_dist = d;
                medoid_ = i;
            }
        }
    } else {
        medoid_ = 0;
    }

    // Pre-reserve to 2× so promotions don't need to reallocate.
    uint64_t reserve = 2 * num_points_;
    data_capacity_ = reserve;
    T* new_data = static_cast<T*>(alloc_aligned(reserve * dim_ * sizeof(T), 256));
    std::memcpy(new_data, data_, num_points_ * dim_ * sizeof(T));
    free_aligned(data_);
    data_ = new_data;
    tags_.reserve(reserve);
    graph_.reserve(reserve);

    // Build the disk_id → internal_id index once; keep it in sync on promote.
    disk_to_internal_.reserve(reserve);
    for (uint64_t i = 0; i < num_points_; i++) {
        disk_to_internal_[tags_[i]] = static_cast<uint32_t>(i);
    }

    nav_size_.store(num_points_);
    loaded_ = true;
}

template <typename T>
uint32_t EntranceGraph<T>::search(const T* query, uint32_t L,
                                  std::vector<Neighbor>& results,
                                  std::vector<Neighbor>* explored) {
    if (!loaded_ || num_points_ == 0) return 0;

    std::shared_lock<std::shared_mutex> lock(graph_mu_);

    tsl::robin_set<uint32_t> visited;
    std::vector<Neighbor> candidates;
    candidates.reserve(L * 2);

    // Start from medoid
    float init_dist = dist_->compare(query, data_ + medoid_ * dim_,
                                     static_cast<uint32_t>(dim_));
    candidates.emplace_back(medoid_, init_dist, true);
    visited.insert(medoid_);

    while (true) {
        // Find closest unexpanded node
        int best_idx = -1;
        for (int i = 0; i < static_cast<int>(candidates.size()); i++) {
            if (candidates[i].flag) {
                if (best_idx < 0 || candidates[i].distance < candidates[best_idx].distance) {
                    best_idx = i;
                }
            }
        }
        if (best_idx < 0) break;

        candidates[best_idx].flag = false;
        uint32_t cur = candidates[best_idx].id;

        if (cur >= graph_.size()) continue;

        // Expand neighbors
        for (uint32_t nbr : graph_[cur]) {
            if (nbr >= num_points_) continue;
            if (!visited.insert(nbr).second) continue;

            float d = dist_->compare(query, data_ + nbr * dim_,
                                     static_cast<uint32_t>(dim_));
            candidates.emplace_back(nbr, d, true);
        }

        // Keep only top-L candidates
        std::sort(candidates.begin(), candidates.end());
        if (candidates.size() > L) {
            candidates.resize(L);
        }
    }

    // Map internal IDs to on-disk IDs via tags
    results.clear();
    results.reserve(candidates.size());
    for (auto& c : candidates) {
        uint32_t disk_id = (c.id < tags_.size()) ? tags_[c.id] : c.id;
        results.emplace_back(disk_id, c.distance);
    }

    if (explored) {
        *explored = candidates; // internal IDs for NAVIS-Update
    }

    return static_cast<uint32_t>(results.size());
}

template <typename T>
void EntranceGraph<T>::maybe_promote(uint32_t id, const T* vec,
                                      uint64_t total_disk_pts,
                                      const std::vector<Neighbor>& explored_ent,
                                      const std::vector<Neighbor>& explored_disk,
                                      const SearchConfig& config,
                                      const std::function<bool(uint32_t)>& is_deleted) {

    // Cheap atomic pre-check: skip if entrance graph already at target.
    uint64_t cur_size = nav_size_.load(std::memory_order_relaxed);
    double target_size = config.ent_ratio * total_disk_pts;
    if (cur_size >= static_cast<uint64_t>(target_size)) return;

    // Read phase: pick up to `max_degree_` candidate neighbors from
    //   E_disk ∩ G_ent  ∪  E_ent
    // Uses the cached disk_to_internal_ index (populated at load(), updated
    // in-place on promote) and reads tags_ — both protected by shared_lock.
    std::vector<uint32_t> internal_neighbors;
    internal_neighbors.reserve(max_degree_);
    {
        std::shared_lock<std::shared_mutex> rlock(graph_mu_);
        tsl::robin_set<uint32_t> seen;
        seen.reserve(max_degree_);

        for (auto& n : explored_disk) {
            if (internal_neighbors.size() >= max_degree_) break;
            if (is_deleted && is_deleted(n.id)) continue;
            auto it = disk_to_internal_.find(n.id);
            if (it != disk_to_internal_.end() && seen.insert(n.id).second) {
                internal_neighbors.push_back(it->second);
            }
        }
        for (auto& n : explored_ent) {
            if (internal_neighbors.size() >= max_degree_) break;
            // explored_ent carries internal IDs (see search()).
            uint32_t disk_id = (n.id < tags_.size()) ? tags_[n.id] : n.id;
            if (is_deleted && is_deleted(disk_id)) continue;
            if (!seen.insert(disk_id).second) continue;
            auto it = disk_to_internal_.find(disk_id);
            if (it != disk_to_internal_.end()) {
                internal_neighbors.push_back(it->second);
            }
        }
    }

    if (internal_neighbors.empty()) return;

    std::unique_lock<std::shared_mutex> lock(graph_mu_);

    cur_size = nav_size_.load(std::memory_order_relaxed);
    if (cur_size >= static_cast<uint64_t>(target_size)) return;

    if (num_points_ >= data_capacity_) {
        std::fprintf(stderr,
            "[navis-update] entrance graph capacity exhausted (%lu); skipping\n",
            num_points_);
        return;
    }

    uint32_t new_internal_id = static_cast<uint32_t>(num_points_);
    std::memcpy(data_ + new_internal_id * dim_, vec, dim_ * sizeof(T));
    graph_.push_back(internal_neighbors);
    tags_.push_back(id);
    disk_to_internal_[id] = new_internal_id;

    for (uint32_t nbr_internal : internal_neighbors) {
        if (graph_[nbr_internal].size() < max_degree_) {
            graph_[nbr_internal].push_back(new_internal_id);
        } else {
            // Evict farthest neighbor from nbr_internal.
            float farthest_d = -1.f;
            int   farthest_k = -1;
            for (size_t k = 0; k < graph_[nbr_internal].size(); ++k) {
                uint32_t nid = graph_[nbr_internal][k];
                float d = dist_->compare(data_ + nbr_internal * dim_,
                                         data_ + nid * dim_,
                                         static_cast<uint32_t>(dim_));
                if (d > farthest_d) { farthest_d = d; farthest_k = static_cast<int>(k); }
            }
            float new_d = dist_->compare(data_ + nbr_internal * dim_,
                                         data_ + new_internal_id * dim_,
                                         static_cast<uint32_t>(dim_));
            if (new_d < farthest_d && farthest_k >= 0) {
                graph_[nbr_internal][farthest_k] = new_internal_id;
            }
        }
    }

    num_points_++;
    nav_size_.store(num_points_, std::memory_order_release);
}

// Explicit instantiations
template class EntranceGraph<float>;
template class EntranceGraph<int8_t>;
template class EntranceGraph<uint8_t>;

} // namespace navis
