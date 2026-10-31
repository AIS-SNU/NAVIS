#include "navis/build/build.h"
#include "navis/io/uring_io.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

namespace navis {
namespace build {

// NOTE: Uses per-vertex fine-grained locks (VertexLockTable) instead of a
// global critical section, as in Microsoft's DiskANN reference
// implementation (~1.8x faster builds).
// The graph topology may differ slightly between runs due to thread interleaving
// in inter_insert(), but graph quality (search recall) is equivalent.
static constexpr float SLACK_FACTOR = 1.3f;

// ---------------------------------------------------------------------------
// Binary-search insert into a sorted Neighbor pool (matching original)
// Returns insertion position, or pool_size+1 if duplicate ID found.
// ---------------------------------------------------------------------------
template <typename T>
uint32_t VamanaGraph<T>::insert_into_pool(Neighbor* pool, uint32_t pool_size, Neighbor nn) {
    // Find insertion point via binary search
    uint32_t lo = 0, hi = pool_size;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (pool[mid] < nn) lo = mid + 1;
        else hi = mid;
    }

    // Duplicate check: scan backward from insertion point for same ID
    // (matching original InsertIntoPool behavior)
    if (lo > 0) {
        for (uint32_t check = lo - 1; ; check--) {
            if (pool[check].id == nn.id) return pool_size + 1; // duplicate
            if (pool[check].distance < nn.distance) break;
            if (check == 0) break;
        }
    }
    // Also check forward
    for (uint32_t check = lo; check < pool_size; check++) {
        if (pool[check].id == nn.id) return pool_size + 1; // duplicate
        if (pool[check].distance > nn.distance) break;
    }

    // Shift right and insert
    std::memmove(&pool[lo + 1], &pool[lo], (pool_size - lo) * sizeof(Neighbor));
    pool[lo] = nn;
    return lo;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

template <typename T>
void VamanaGraph<T>::build(const std::string& data_path, uint32_t R,
                           uint32_t L, float alpha, uint32_t num_threads) {
    uint32_t dim;
    load_bin<T>(data_path, data_, npts_, dim);
    if (dim != dim_) {
        throw std::runtime_error("Data dimension mismatch: expected " +
            std::to_string(dim_) + ", got " + std::to_string(dim));
    }
    std::printf("[build] Loaded %u points x %u dim\n", npts_, dim);

    graph_.resize(npts_);
    for (auto& g : graph_) g.reserve(R);
    locks_ = VertexLockTable(npts_ + 1);

    ep_ = compute_medoid();
    std::printf("[build] Entry point (medoid): %u\n", ep_);

    omp_set_num_threads(num_threads);
    const uint32_t C = 750;
    auto t_start = std::chrono::high_resolution_clock::now();

    // Main Vamana link: for each node, greedy search + prune + reverse links
    std::atomic<uint32_t> link_done{0};
    uint32_t link_report = std::max(npts_ / 10, 1u);  // report every 10%

    #pragma omp parallel for schedule(dynamic, 128)
    for (int64_t node = 0; node < npts_; node++) {
        const T* query = data_ + node * aligned_dim_;

        // Greedy search from entry point
        std::vector<Neighbor> expanded;
        tsl::robin_set<uint32_t> visited;
        greedy_search(query, L, {ep_}, expanded, visited);

        // Remove self
        expanded.erase(
            std::remove_if(expanded.begin(), expanded.end(),
                [node](const Neighbor& n) { return n.id == static_cast<uint32_t>(node); }),
            expanded.end());

        // RNG pruning
        std::vector<uint32_t> pruned;
        prune_neighbors(static_cast<uint32_t>(node), expanded, R, alpha, C, pruned);

        // Set forward edges (per-vertex lock)
        {
            locks_.lock(static_cast<uint32_t>(node));
            graph_[node] = pruned;
            locks_.unlock(static_cast<uint32_t>(node));
        }

        // Reverse links with per-vertex fine-grained locks
        inter_insert(static_cast<uint32_t>(node), pruned, R, alpha, C);

        uint32_t done = link_done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % link_report == 0 || done == npts_) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            double eta = elapsed / done * (npts_ - done);
            int pct = static_cast<int>(100.0 * done / npts_);
            int bar = pct / 2;
            std::printf("\r[build] Linking: [");
            for (int i = 0; i < 50; i++) std::putchar(i < bar ? '#' : '-');
            int eta_m = static_cast<int>(eta) / 60, eta_s = static_cast<int>(eta) % 60;
            std::printf("] %3d%% %dm%02ds", pct, eta_m, eta_s);
            std::fflush(stdout);
        }
    }
    std::printf("\n");

    // Final pass: trim oversized lists
    std::printf("[build] Trimming oversized neighbor lists...\n");
    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t node = 0; node < npts_; node++) {
        std::vector<uint32_t> cur;
        {
            locks_.lock(static_cast<uint32_t>(node));
            cur = graph_[node];
            locks_.unlock(static_cast<uint32_t>(node));
        }
        if (cur.size() > R) {
            std::vector<Neighbor> candidates;
            candidates.reserve(cur.size());
            const T* query = data_ + node * aligned_dim_;
            for (uint32_t id : cur) {
                float d = dist_->compare(query, data_ + id * aligned_dim_,
                                         static_cast<uint32_t>(dim_));
                candidates.emplace_back(id, d);
            }
            std::vector<uint32_t> trimmed;
            prune_neighbors(static_cast<uint32_t>(node), candidates, R, alpha, C, trimmed,
                            false /* final trim, not saturation */);
            locks_.lock(static_cast<uint32_t>(node));
            graph_[node] = trimmed;
            locks_.unlock(static_cast<uint32_t>(node));
        }
    }

    max_degree_ = 0;
    for (auto& g : graph_) {
        max_degree_ = std::max(max_degree_, static_cast<uint32_t>(g.size()));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::printf("[build] Graph built in %.2fs, max_degree=%u\n",
                std::chrono::duration<double>(t_end - t_start).count(), max_degree_);
}

// ---------------------------------------------------------------------------
// Reverse link insertion with per-vertex locks (replaces omp critical)
// ---------------------------------------------------------------------------

template <typename T>
void VamanaGraph<T>::inter_insert(uint32_t node,
                                   const std::vector<uint32_t>& pruned,
                                   uint32_t R, float alpha, uint32_t C) {
    for (uint32_t des : pruned) {
        bool need_prune = false;
        std::vector<uint32_t> copy_of_neighbors;

        locks_.lock(des);
        {
            auto& des_pool = graph_[des];
            // Check if already a neighbor
            if (std::find(des_pool.begin(), des_pool.end(), node) == des_pool.end()) {
                if (des_pool.size() < static_cast<size_t>(SLACK_FACTOR * R)) {
                    des_pool.push_back(node);
                } else {
                    copy_of_neighbors = des_pool;
                    copy_of_neighbors.push_back(node);
                    need_prune = true;
                }
            }
        }
        locks_.unlock(des);

        if (need_prune) {
            // Re-prune the oversized list (distance computation is lock-free)
            std::vector<Neighbor> candidates;
            candidates.reserve(copy_of_neighbors.size());
            const T* des_vec = data_ + des * aligned_dim_;
            for (uint32_t id : copy_of_neighbors) {
                if (id == des) continue;
                float d = dist_->compare(des_vec, data_ + id * aligned_dim_,
                                         static_cast<uint32_t>(dim_));
                candidates.emplace_back(id, d);
            }
            std::vector<uint32_t> new_pruned;
            prune_neighbors(des, candidates, R, alpha, C, new_pruned,
                            false /* no saturation for reverse-link re-pruning */);

            locks_.lock(des);
            graph_[des] = new_pruned;
            locks_.unlock(des);
        }
    }
}

// ---------------------------------------------------------------------------
// Greedy search with prefetching (used during construction)
// ---------------------------------------------------------------------------

template <typename T>
void VamanaGraph<T>::greedy_search(const T* query, uint32_t L,
                                   const std::vector<uint32_t>& init_ids,
                                   std::vector<Neighbor>& expanded,
                                   tsl::robin_set<uint32_t>& visited) {
    // Fixed-size pool with binary-search insertion (top-L candidates)
    std::vector<Neighbor> pool(L + 1);
    for (uint32_t i = 0; i < L + 1; i++) {
        pool[i].distance = std::numeric_limits<float>::max();
        pool[i].id = UINT32_MAX;
        pool[i].flag = false;
    }

    // Track ALL visited nodes with distances (for RNG pruning, matching original)
    // The original passes expanded_nodes_info (all visited) to prune_neighbors,
    // not just the top-L pool. This gives pruning access to distant-but-useful
    // shortcut edges that fell out of the pool.
    expanded.clear();
    expanded.reserve(10 * L);

    uint32_t pool_size = 0;

    for (uint32_t id : init_ids) {
        if (id >= npts_ || !visited.insert(id).second) continue;
        float d = dist_->compare(query, data_ + id * aligned_dim_,
                                 static_cast<uint32_t>(dim_));
        Neighbor nn(id, d, true);
        expanded.push_back(nn);
        if (pool_size < L) {
            insert_into_pool(pool.data(), pool_size, nn);
            pool_size++;
        }
    }

    uint32_t k = 0;
    while (k < pool_size) {
        if (!pool[k].flag) { k++; continue; }
        pool[k].flag = false;
        uint32_t cur = pool[k].id;

        // Snapshot neighbors under lock to prevent torn reads from
        // concurrent inter_insert() writes on other threads
        std::vector<uint32_t> nbrs;
        {
            locks_.lock(cur);
            nbrs = graph_[cur];
            locks_.unlock(cur);
        }
        uint32_t nbr_count = static_cast<uint32_t>(nbrs.size());

        uint32_t nk = pool_size;
        for (uint32_t m = 0; m < nbr_count; m++) {
            uint32_t id = nbrs[m];
            if (id >= npts_ || !visited.insert(id).second) continue;

            // Prefetch next neighbor's vector
            if (m + 1 < nbr_count && nbrs[m + 1] < npts_) {
                _mm_prefetch(reinterpret_cast<const char*>(
                    data_ + nbrs[m + 1] * aligned_dim_), _MM_HINT_T0);
            }

            float d = dist_->compare(query, data_ + id * aligned_dim_,
                                     static_cast<uint32_t>(dim_));

            Neighbor nn(id, d, true);
            expanded.push_back(nn);  // track ALL visited nodes

            if (d >= pool[pool_size - 1].distance && pool_size == L)
                continue;

            uint32_t pos = insert_into_pool(pool.data(), pool_size, nn);
            if (pos > pool_size) continue; // duplicate in pool, skip
            if (pool_size < L) pool_size++;
            if (pos < nk) nk = pos;
        }

        if (nk <= k) k = nk;
        else k++;
    }
}

// ---------------------------------------------------------------------------
// RNG pruning (occlude_list — identical algorithm to original)
// ---------------------------------------------------------------------------

template <typename T>
void VamanaGraph<T>::occlude_list(std::vector<Neighbor>& pool, float alpha,
                                  uint32_t R, uint32_t C,
                                  std::vector<Neighbor>& result) {
    if (pool.empty()) return;

    uint32_t pool_limit = std::min(static_cast<uint32_t>(pool.size()), C);
    std::vector<float> occlude_factor(pool_limit, 0.0f);
    float cur_alpha = 1.0f;

    while (cur_alpha <= alpha && result.size() < R) {
        for (uint32_t i = 0; i < pool_limit && result.size() < R; i++) {
            if (occlude_factor[i] > cur_alpha) continue;

            occlude_factor[i] = std::numeric_limits<float>::max();
            result.push_back(pool[i]);

            for (uint32_t t = i + 1; t < pool_limit; t++) {
                if (occlude_factor[t] > alpha) continue;
                float djk = dist_->compare(data_ + pool[t].id * aligned_dim_,
                                           data_ + pool[i].id * aligned_dim_,
                                           static_cast<uint32_t>(dim_));
                if (djk > 0.0f) {
                    occlude_factor[t] = std::max(occlude_factor[t],
                                                  pool[t].distance / djk);
                }
            }
        }
        cur_alpha *= 1.2f;
    }
}

template <typename T>
void VamanaGraph<T>::prune_neighbors(uint32_t node,
                                      std::vector<Neighbor>& candidates,
                                      uint32_t R, float alpha, uint32_t C,
                                      std::vector<uint32_t>& pruned,
                                      bool saturate) {
    std::sort(candidates.begin(), candidates.end());

    std::vector<Neighbor> result;
    result.reserve(R);
    occlude_list(candidates, alpha, R, C, result);

    pruned.clear();
    pruned.reserve(R);
    for (auto& n : result) {
        if (n.id != node) pruned.push_back(n.id);
    }

    // Graph saturation: if RNG pruning left us under R neighbors and alpha > 1,
    // fill remaining slots with closest unpicked candidates.
    // This ensures every node has close to R edges, improving navigability
    // for nodes in sparse regions of the graph.
    if (saturate && alpha > 1.0f && pruned.size() < R) {
        tsl::robin_set<uint32_t> pruned_set(pruned.begin(), pruned.end());
        pruned_set.insert(node);

        for (auto& c : candidates) {
            if (pruned.size() >= R) break;
            if (pruned_set.find(c.id) == pruned_set.end()) {
                pruned.push_back(c.id);
                pruned_set.insert(c.id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Medoid (closest point to centroid)
// ---------------------------------------------------------------------------

template <typename T>
uint32_t VamanaGraph<T>::compute_medoid() {
    std::vector<double> center(aligned_dim_, 0.0);
    for (uint32_t i = 0; i < npts_; i++) {
        for (uint64_t j = 0; j < dim_; j++) {
            center[j] += static_cast<double>(data_[i * aligned_dim_ + j]);
        }
    }
    for (uint64_t j = 0; j < dim_; j++) center[j] /= npts_;

    uint32_t best = 0;
    float best_dist = std::numeric_limits<float>::max();

    #pragma omp parallel
    {
        uint32_t local_best = 0;
        float local_best_dist = std::numeric_limits<float>::max();

        #pragma omp for schedule(static, 65536)
        for (int64_t i = 0; i < npts_; i++) {
            float d = 0;
            for (uint64_t j = 0; j < dim_; j++) {
                float diff = static_cast<float>(center[j]) -
                             static_cast<float>(data_[i * aligned_dim_ + j]);
                d += diff * diff;
            }
            if (d < local_best_dist) {
                local_best_dist = d;
                local_best = static_cast<uint32_t>(i);
            }
        }

        #pragma omp critical
        {
            if (local_best_dist < best_dist) {
                best_dist = local_best_dist;
                best = local_best;
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Serialization (DiskANN in-memory index format)
// ---------------------------------------------------------------------------

template <typename T>
void VamanaGraph<T>::save(const std::string& path) {
    uint64_t total_size = 24;
    for (uint32_t i = 0; i < npts_; i++) {
        total_size += sizeof(uint32_t) + graph_[i].size() * sizeof(uint32_t);
    }

    UringWriter out(path);
    out.write(&total_size, sizeof(uint64_t));
    out.write(&max_degree_, sizeof(uint32_t));
    out.write(&ep_, sizeof(uint32_t));
    uint64_t frozen = 0;
    out.write(&frozen, sizeof(uint64_t));

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = static_cast<uint32_t>(graph_[i].size());
        out.write(&deg, sizeof(uint32_t));
        out.write(graph_[i].data(), deg * sizeof(uint32_t));
    }

    std::printf("[save] Graph saved to %s (%.2f MB)\n",
                path.c_str(), total_size / (1024.0 * 1024.0));
}

template <typename T>
void VamanaGraph<T>::save_data(const std::string& path) {
    save_bin<T>(path, data_, npts_, static_cast<uint32_t>(dim_));
    std::printf("[save] Data saved to %s\n", path.c_str());
}

template class VamanaGraph<float>;
template class VamanaGraph<int8_t>;
template class VamanaGraph<uint8_t>;

} // namespace build
} // namespace navis
