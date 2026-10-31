#pragma once

#include "navis/types.h"
#include "navis/distance.h"
#include "navis/lock/vertex_lock.h"
#include "navis/util/file_utils.h"

#include <memory>
#include <string>
#include <vector>

#include <tsl/robin_set.h>

namespace navis {
namespace build {

// ---------------------------------------------------------------------------
// Vamana graph construction
// ---------------------------------------------------------------------------

template <typename T>
class VamanaGraph {
public:
    VamanaGraph(uint64_t dim, Metric metric)
        : dim_(dim), aligned_dim_(round_up(dim, 8)),
          dist_(make_distance<T>(metric)) {}

    ~VamanaGraph() { free_aligned(data_); }

    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, uint32_t num_threads);
    void save(const std::string& path);
    void save_data(const std::string& path);

    uint32_t num_points() const { return npts_; }
    uint32_t max_degree() const { return max_degree_; }
    uint32_t entry_point() const { return ep_; }

private:
    uint64_t dim_;
    uint64_t aligned_dim_;
    std::unique_ptr<Distance<T>> dist_;

    T* data_ = nullptr;
    std::vector<std::vector<uint32_t>> graph_;
    VertexLockTable locks_{1024};
    uint32_t npts_ = 0;
    uint32_t max_degree_ = 0;
    uint32_t ep_ = 0;

    uint32_t compute_medoid();
    static uint32_t insert_into_pool(Neighbor* pool, uint32_t pool_size, Neighbor nn);
    void greedy_search(const T* query, uint32_t L,
                       const std::vector<uint32_t>& init_ids,
                       std::vector<Neighbor>& expanded,
                       tsl::robin_set<uint32_t>& visited);
    void occlude_list(std::vector<Neighbor>& pool, float alpha,
                      uint32_t R, uint32_t C, std::vector<Neighbor>& result);
    void prune_neighbors(uint32_t node, std::vector<Neighbor>& candidates,
                         uint32_t R, float alpha, uint32_t C,
                         std::vector<uint32_t>& pruned, bool saturate = true);
    void inter_insert(uint32_t node, const std::vector<uint32_t>& pruned,
                      uint32_t R, float alpha, uint32_t C);
};

// ---------------------------------------------------------------------------
// PQ generation
// ---------------------------------------------------------------------------

// Sample a random subset of vectors for PQ training (in-memory)
template <typename T>
void gen_random_slice(const std::string& data_path, double sample_rate,
                      std::vector<float>& out_data, uint32_t& out_npts,
                      uint32_t& out_dim);

// Sample to file: outputs {prefix}_data.bin and {prefix}_ids.bin
template <typename T>
void gen_random_slice_to_file(const std::string& data_path,
                               const std::string& output_prefix,
                               double sample_rate);

// Generate PQ pivot centroids via k-means. Dimensions are assigned to
// chunks contiguously in their original order.
void generate_pq_pivots(const float* train_data, uint32_t num_train,
                        uint32_t dim, uint32_t num_centers,
                        uint32_t num_chunks, uint32_t max_kmeans_iters,
                        const std::string& output_path);

// Compress all vectors to PQ codes
template <typename T>
void generate_pq_data(const std::string& data_path,
                      const std::string& pq_pivots_path,
                      uint32_t num_centers, uint32_t num_chunks,
                      const std::string& output_path);

// ---------------------------------------------------------------------------
// Sharded build (for datasets exceeding host memory)
// ---------------------------------------------------------------------------

// RAM estimation for Vamana graph (matching DiskANN formula)
double estimate_ram_usage(size_t npts, size_t dim, size_t datasize, size_t R);

// Partition data into shards via k-means, each fitting in ram_budget_gb.
// Returns number of shards. Writes {prefix}_subshard-{p}.bin and _ids_uint32.bin.
template <typename T>
uint32_t partition_with_ram_budget(const std::string& data_path,
                                    double ram_budget_gb, uint32_t R,
                                    uint32_t dim,
                                    const std::string& prefix);

// Build independent Vamana graph on each shard (R' = 2R/3).
template <typename T>
void build_shards(const std::string& prefix, uint32_t num_shards,
                  uint32_t dim, Metric metric, uint32_t R, uint32_t L,
                  float alpha, uint32_t threads);

// Merge shard graphs into a single unified graph with ID remapping.
void merge_shards(const std::string& prefix, uint32_t num_shards,
                  uint32_t npts, uint32_t R,
                  const std::string& output_path);

// Remove temporary shard files.
void cleanup_shards(const std::string& prefix, uint32_t num_shards);

// ---------------------------------------------------------------------------
// Disk layout creation (NAVIS separated layout: independent edge + vec files
// with a host-memory indirection table; Starling partition reorder optional)
// ---------------------------------------------------------------------------

template <typename T>
void create_separated_layout(const std::string& graph_path,
                             const std::string& data_path,
                             const std::string& output_prefix);

// ---------------------------------------------------------------------------
// Starling graph partitioning (locality-aware edge-page layout).
// Produces {prefix}_partition.bin.aligned consumed by create_separated_layout.
// ---------------------------------------------------------------------------

template <typename T>
void create_starling_partition(const std::string& graph_path,
                               uint32_t npts,
                               uint64_t edges_per_page,
                               const std::string& output_prefix);

// ---------------------------------------------------------------------------
// Full build pipeline orchestrator
// ---------------------------------------------------------------------------

template <typename T>
void build_full_index(const std::string& data_path,
                      const std::string& output_prefix,
                      Metric metric,
                      uint32_t R, uint32_t L,
                      float pq_bytes_per_vec,
                      float mem_budget_gb,
                      uint32_t num_threads,
                      float sample_rate,
                      uint32_t ent_R, uint32_t ent_L,
                      float alpha);

} // namespace build
} // namespace navis
