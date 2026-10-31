#include "navis/build/build.h"
#include "navis/util/file_utils.h"
#include "navis/io/uring_io.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

namespace navis {
namespace build {

// ===========================================================================
// Sharded Vamana build for datasets that exceed host memory.
//
// Follows DiskANN's merged-shard build (build_merged_vamana_index,
// partition_with_ram_budget, shard_data_into_clusters).
//
// Three stages:
//   1. Partition: k-means clustering, assign each point to 2 nearest clusters
//   2. Per-shard build: independent Vamana with R' = 2R/3
//   3. Merge: combine shard graphs with ID remapping and deduplication
// ===========================================================================

static constexpr double RAM_OVERHEAD_FACTOR = 1.1;
static constexpr double RAM_SLACK_FACTOR    = 1.3;
static constexpr uint32_t K_BASE           = 2;    // replication factor
static constexpr uint32_t INITIAL_NUM_PARTS = 3;
static constexpr uint32_t MAX_KMEANS_ITERS  = 20;
static constexpr size_t   STREAM_BLOCK      = 16384;

double estimate_ram_usage(size_t npts, size_t dim, size_t datasize, size_t R) {
    double graph = static_cast<double>(npts) * R * sizeof(uint32_t) * RAM_SLACK_FACTOR;
    double data  = static_cast<double>(npts) * round_up(dim, 8) * datasize;
    return RAM_OVERHEAD_FACTOR * (graph + data);
}

// ---------------------------------------------------------------------------
// K-means for partitioning (simplified: no per-chunk, just global clustering)
// ---------------------------------------------------------------------------

static void kmeans_pp_init(const float* data, uint32_t n, uint32_t dim,
                           uint32_t k, float* centroids) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> first_pick(0, n - 1);

    uint32_t first = first_pick(rng);
    std::memcpy(centroids, data + first * dim, dim * sizeof(float));

    std::vector<float> min_dists(n, std::numeric_limits<float>::max());
    for (uint32_t c = 1; c < k; c++) {
        const float* prev = centroids + (c - 1) * dim;
        double total = 0.0;
        #pragma omp parallel for schedule(static, 4096) reduction(+:total)
        for (int64_t i = 0; i < n; i++) {
            float d = 0;
            for (uint32_t j = 0; j < dim; j++) {
                float diff = data[i * dim + j] - prev[j];
                d += diff * diff;
            }
            if (d < min_dists[i]) min_dists[i] = d;
            total += min_dists[i];
        }

        std::uniform_real_distribution<double> uni(0.0, total);
        double target = uni(rng);
        uint32_t selected = n - 1;
        double cumsum = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            cumsum += min_dists[i];
            if (cumsum >= target) { selected = i; break; }
        }
        std::memcpy(centroids + c * dim, data + selected * dim, dim * sizeof(float));
    }
}

static void kmeans_lloyd(const float* data, uint32_t n, uint32_t dim,
                         uint32_t k, uint32_t max_iters,
                         float* centroids, uint32_t* assignments) {
    kmeans_pp_init(data, n, dim, k, centroids);

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        double residual = 0.0, old_residual = std::numeric_limits<double>::max();

        #pragma omp parallel for schedule(static, 512) reduction(+:residual)
        for (int64_t i = 0; i < n; i++) {
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; c++) {
                float d = 0;
                for (uint32_t j = 0; j < dim; j++) {
                    float diff = data[i * dim + j] - centroids[c * dim + j];
                    d += diff * diff;
                }
                if (d < best_dist) { best_dist = d; best_c = c; }
            }
            assignments[i] = best_c;
            residual += best_dist;
        }

        if (iter > 0) {
            double rel = (old_residual - residual) / (residual + 1e-30);
            if (rel < 1e-5) break;
        }
        old_residual = residual;

        // Recompute centroids
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<double>> tl_sums(num_threads, std::vector<double>(k * dim, 0.0));
        std::vector<std::vector<uint32_t>> tl_counts(num_threads, std::vector<uint32_t>(k, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static, 512)
            for (int64_t i = 0; i < n; i++) {
                uint32_t c = assignments[i];
                tl_counts[tid][c]++;
                for (uint32_t j = 0; j < dim; j++)
                    tl_sums[tid][c * dim + j] += data[i * dim + j];
            }
        }

        std::memset(centroids, 0, k * dim * sizeof(float));
        std::vector<uint32_t> counts(k, 0);
        for (int t = 0; t < num_threads; t++) {
            for (uint32_t c = 0; c < k; c++) {
                counts[c] += tl_counts[t][c];
                for (uint32_t j = 0; j < dim; j++)
                    centroids[c * dim + j] += static_cast<float>(tl_sums[t][c * dim + j]);
            }
        }
        for (uint32_t c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (uint32_t j = 0; j < dim; j++)
                    centroids[c * dim + j] /= counts[c];
            }
        }
    }
}

// Find k_base closest centroids for each point in a block.
// closest[i * k_base + p] = p-th closest centroid for point i.
static void find_k_closest(const float* block, uint32_t count, uint32_t dim,
                           const float* centroids, uint32_t num_parts,
                           uint32_t* closest) {
    #pragma omp parallel for schedule(static, 256)
    for (int64_t i = 0; i < count; i++) {
        // Compute distances to all centroids
        std::vector<std::pair<float, uint32_t>> dists(num_parts);
        for (uint32_t c = 0; c < num_parts; c++) {
            float d = 0;
            for (uint32_t j = 0; j < dim; j++) {
                float diff = block[i * dim + j] - centroids[c * dim + j];
                d += diff * diff;
            }
            dists[c] = {d, c};
        }
        std::partial_sort(dists.begin(), dists.begin() + K_BASE, dists.end());
        for (uint32_t p = 0; p < K_BASE; p++)
            closest[i * K_BASE + p] = dists[p].second;
    }
}

// ===========================================================================
// Stage 1: Partition with RAM budget
// ===========================================================================

template <typename T>
uint32_t partition_with_ram_budget(const std::string& data_path,
                                    double ram_budget_gb, uint32_t R,
                                    uint32_t dim,
                                    const std::string& prefix) {
    uint32_t npts;
    { uint32_t d; get_bin_metadata(data_path, npts, d); }

    uint32_t shard_R = 2 * R / 3;

    // Sample 1% for k-means training
    std::printf("[shard] Sampling 1%% of %u points...\n", npts);
    std::fflush(stdout);
    std::vector<float> sample_data;
    uint32_t sample_npts, sample_dim;
    gen_random_slice<T>(data_path, 0.01, sample_data, sample_npts, sample_dim);
    std::printf("[shard] Sampled %u points (%.1f MB) for k-means\n",
                sample_npts, sample_npts * dim * sizeof(float) / 1e6);
    std::fflush(stdout);

    // Find num_parts such that each shard fits in RAM
    uint32_t num_parts = INITIAL_NUM_PARTS;
    while (true) {
        std::printf("[shard] Trying %u parts: running k-means...", num_parts);
        std::fflush(stdout);

        std::vector<float> centroids(num_parts * dim);
        std::vector<uint32_t> assignments(sample_npts);
        kmeans_lloyd(sample_data.data(), sample_npts, dim,
                     num_parts, MAX_KMEANS_ITERS, centroids.data(), assignments.data());

        // Estimate max shard size (with k_base=2 replication, each point in 2 shards)
        std::vector<uint32_t> counts(num_parts, 0);
        for (uint32_t i = 0; i < sample_npts; i++) counts[assignments[i]]++;

        uint32_t max_count = *std::max_element(counts.begin(), counts.end());
        // Scale up from sample: est_shard_size ≈ max_count / sample_rate * k_base
        uint64_t est_shard = static_cast<uint64_t>(max_count) * npts / sample_npts * K_BASE;
        double shard_ram = estimate_ram_usage(est_shard, dim, sizeof(T), shard_R);

        std::printf(" max_shard ~%lu pts, ~%.1f GB (budget %.1f GB)\n",
                    est_shard, shard_ram / 1e9, ram_budget_gb);
        std::fflush(stdout);

        if (shard_ram <= ram_budget_gb * 1e9) break;
        num_parts++;
        if (num_parts > 256) {
            std::fprintf(stderr, "Cannot partition into <=256 shards to fit RAM budget\n");
            exit(1);
        }
    }

    // Final k-means with chosen num_parts
    std::vector<float> centroids(num_parts * dim);
    std::vector<uint32_t> assignments(sample_npts);
    kmeans_lloyd(sample_data.data(), sample_npts, dim,
                 num_parts, MAX_KMEANS_ITERS, centroids.data(), assignments.data());

    std::printf("[shard] Partitioning into %u shards (k_base=%u)\n", num_parts, K_BASE);

    // Stream data and assign each point to K_BASE closest centroids
    UringReader data_in(data_path);
    data_in.seek(8);

    // Open per-shard writers with buffered I/O
    std::vector<std::unique_ptr<UringWriter>> data_writers(num_parts);
    std::vector<std::unique_ptr<UringWriter>> id_writers(num_parts);
    std::vector<uint32_t> shard_counts(num_parts, 0);

    for (uint32_t p = 0; p < num_parts; p++) {
        std::string dp = prefix + "_subshard-" + std::to_string(p) + ".bin";
        std::string ip = prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
        data_writers[p] = std::make_unique<UringWriter>(dp);
        id_writers[p] = std::make_unique<UringWriter>(ip);
        // Write placeholder headers (npts=0, will be patched)
        uint32_t zero = 0;
        data_writers[p]->write(&zero, 4);
        data_writers[p]->write(&dim, 4);
        uint32_t one = 1;
        id_writers[p]->write(&zero, 4);
        id_writers[p]->write(&one, 4);
    }

    // Stream blocks, assign to shards
    std::vector<T> block_raw(STREAM_BLOCK * dim);
    std::vector<float> block_float(STREAM_BLOCK * dim);
    std::vector<uint32_t> closest(STREAM_BLOCK * K_BASE);

    for (uint32_t base = 0; base < npts; base += STREAM_BLOCK) {
        uint32_t count = std::min(static_cast<uint32_t>(STREAM_BLOCK), npts - base);
        data_in.read(block_raw.data(), count * dim * sizeof(T));

        // Convert to float for distance computation
        for (uint64_t i = 0; i < static_cast<uint64_t>(count) * dim; i++)
            block_float[i] = static_cast<float>(block_raw[i]);

        find_k_closest(block_float.data(), count, dim,
                       centroids.data(), num_parts, closest.data());

        for (uint32_t i = 0; i < count; i++) {
            uint32_t global_id = base + i;
            for (uint32_t p = 0; p < K_BASE; p++) {
                uint32_t shard = closest[i * K_BASE + p];
                data_writers[shard]->write(
                    block_raw.data() + static_cast<size_t>(i) * dim,
                    dim * sizeof(T));
                id_writers[shard]->write(&global_id, sizeof(uint32_t));
                shard_counts[shard]++;
            }
        }

        if (base % (STREAM_BLOCK * 100) == 0 && base > 0) {
            int pct = static_cast<int>(100.0 * base / npts);
            int bar = pct / 2;
            std::printf("\r[shard] Assigning: [");
            for (int b = 0; b < 50; b++) std::putchar(b < bar ? '#' : '-');
            std::printf("] %3d%% (%u/%u)", pct, base, npts);
            std::fflush(stdout);
        }
    }

    std::printf("\r[shard] Assigning: [");
    for (int b = 0; b < 50; b++) std::putchar('#');
    std::printf("] 100%% (%u/%u)\n", npts, npts);
    std::fflush(stdout);

    // Patch headers with actual counts
    for (uint32_t p = 0; p < num_parts; p++) {
        data_writers[p]->seek(0);
        data_writers[p]->write(&shard_counts[p], 4);
        data_writers[p].reset();  // flush and close

        id_writers[p]->seek(0);
        id_writers[p]->write(&shard_counts[p], 4);
        id_writers[p].reset();

        std::printf("[shard] Shard %u: %u points\n", p, shard_counts[p]);
    }

    // Save centroids
    save_bin<float>(prefix + "_centroids.bin", centroids.data(), num_parts, dim);

    return num_parts;
}

// ===========================================================================
// Stage 2: Build per-shard Vamana graphs
// ===========================================================================

template <typename T>
void build_shards(const std::string& prefix, uint32_t num_shards,
                  uint32_t dim, Metric metric, uint32_t R, uint32_t L,
                  float alpha, uint32_t threads) {
    uint32_t shard_R = 2 * R / 3;
    std::printf("[shard] Building %u shard graphs (R'=%u, L=%u)\n",
                num_shards, shard_R, L);

    for (uint32_t p = 0; p < num_shards; p++) {
        std::string shard_data = prefix + "_subshard-" + std::to_string(p) + ".bin";
        std::string shard_graph = prefix + "_subshard-" + std::to_string(p) + "_mem.index";

        if (file_exists(shard_graph)) {
            std::printf("[shard] Shard %u graph already exists, skipping\n", p);
            continue;
        }

        std::printf("[shard] Building shard %u/%u...\n", p + 1, num_shards);
        VamanaGraph<T> g(dim, metric);
        g.build(shard_data, shard_R, L, alpha, threads);
        g.save(shard_graph);
        std::printf("[shard] Shard %u: %u pts, max_degree=%u\n",
                    p, g.num_points(), g.max_degree());
    }
}

// ===========================================================================
// Stage 3: Merge shard graphs into unified graph
// ===========================================================================

void merge_shards(const std::string& prefix, uint32_t num_shards,
                  uint32_t npts, uint32_t R,
                  const std::string& output_path) {
    std::printf("[merge] Loading %u shard ID maps...\n", num_shards);

    // Load all ID maps
    std::vector<std::vector<uint32_t>> idmaps(num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = prefix + "_subshard-" + std::to_string(s) + "_ids_uint32.bin";
        uint32_t n, d;
        load_bin<uint32_t>(path, idmaps[s], n, d);
    }

    // Build sorted (global_id, shard_id) pairs
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    for (uint32_t s = 0; s < num_shards; s++) {
        for (uint32_t i = 0; i < idmaps[s].size(); i++) {
            node_shard.push_back({idmaps[s][i], s});
        }
    }
    std::sort(node_shard.begin(), node_shard.end());
    std::printf("[merge] %zu total (global_id, shard) entries\n", node_shard.size());

    // Open shard graph readers, skip headers, collect medoids
    std::vector<std::unique_ptr<UringReader>> readers(num_shards);
    std::vector<uint32_t> shard_medoids(num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = prefix + "_subshard-" + std::to_string(s) + "_mem.index";
        readers[s] = std::make_unique<UringReader>(path);
        uint64_t index_size;
        uint32_t max_deg, medoid;
        uint64_t frozen;
        readers[s]->read(&index_size, 8);
        readers[s]->read(&max_deg, 4);
        readers[s]->read(&medoid, 4);
        readers[s]->read(&frozen, 8);
        // Map local medoid to global
        shard_medoids[s] = idmaps[s][medoid];
    }

    // Track read position per shard (which local vertex we're at)
    std::vector<uint32_t> shard_pos(num_shards, 0);

    // Merge: iterate through node_shard in global_id order
    // Collect neighbors from all shards, remap local→global, dedup, truncate
    UringWriter out(output_path);

    // Write header placeholder (will patch later)
    uint64_t placeholder_size = 0;
    uint32_t merged_max_deg = 0;
    uint32_t global_medoid = shard_medoids[num_shards - 1];  // last shard's medoid
    uint64_t frozen = 0;
    out.write(&placeholder_size, 8);
    out.write(&merged_max_deg, 4);
    out.write(&global_medoid, 4);
    out.write(&frozen, 8);

    std::vector<bool> nhood_set(npts, false);
    std::vector<uint32_t> final_nhood;
    std::mt19937 urng(42);

    uint32_t cur_global = 0;
    size_t idx = 0;

    auto flush_node = [&]() {
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        uint32_t deg = static_cast<uint32_t>(std::min(final_nhood.size(), static_cast<size_t>(R)));
        out.write(&deg, 4);
        out.write(final_nhood.data(), deg * sizeof(uint32_t));
        if (deg > merged_max_deg) merged_max_deg = deg;

        for (auto id : final_nhood) nhood_set[id] = false;
        final_nhood.clear();
    };

    while (idx < node_shard.size()) {
        uint32_t gid = node_shard[idx].first;

        // Flush any skipped nodes (nodes not in any shard — shouldn't happen with k_base=2)
        while (cur_global < gid) {
            uint32_t zero_deg = 0;
            out.write(&zero_deg, 4);
            cur_global++;
        }

        // Collect neighbors from all shards for this global_id
        while (idx < node_shard.size() && node_shard[idx].first == gid) {
            uint32_t shard = node_shard[idx].second;

            // Read this shard's local neighbors for the current point
            uint32_t local_deg;
            readers[shard]->read(&local_deg, 4);
            std::vector<uint32_t> local_nbrs(local_deg);
            readers[shard]->read(local_nbrs.data(),
                                local_deg * sizeof(uint32_t));
            shard_pos[shard]++;

            // Remap local→global and deduplicate
            for (uint32_t j = 0; j < local_deg; j++) {
                uint32_t global_nbr = idmaps[shard][local_nbrs[j]];
                if (global_nbr != gid && !nhood_set[global_nbr]) {
                    nhood_set[global_nbr] = true;
                    final_nhood.push_back(global_nbr);
                }
            }
            idx++;
        }

        flush_node();
        cur_global = gid + 1;
    }

    // Flush remaining nodes
    while (cur_global < npts) {
        uint32_t zero_deg = 0;
        out.write(&zero_deg, 4);
        cur_global++;
    }

    // Patch header
    uint64_t total_size = out.tell();
    out.seek(0);
    out.write(&total_size, 8);
    out.write(&merged_max_deg, 4);

    std::printf("[merge] Merged graph: %u pts, max_degree=%u, medoid=%u\n",
                npts, merged_max_deg, global_medoid);
    std::printf("[merge] Written to %s\n", output_path.c_str());
}

// ===========================================================================
// Cleanup: remove temporary shard files
// ===========================================================================

void cleanup_shards(const std::string& prefix, uint32_t num_shards) {
    for (uint32_t p = 0; p < num_shards; p++) {
        std::string base = prefix + "_subshard-" + std::to_string(p);
        std::remove((base + ".bin").c_str());
        std::remove((base + "_ids_uint32.bin").c_str());
        std::remove((base + "_mem.index").c_str());
    }
    std::remove((prefix + "_centroids.bin").c_str());
    std::printf("[shard] Cleaned up temporary files\n");
}

// Explicit instantiations
template uint32_t partition_with_ram_budget<float>(const std::string&, double, uint32_t, uint32_t, const std::string&);
template uint32_t partition_with_ram_budget<int8_t>(const std::string&, double, uint32_t, uint32_t, const std::string&);
template uint32_t partition_with_ram_budget<uint8_t>(const std::string&, double, uint32_t, uint32_t, const std::string&);

template void build_shards<float>(const std::string&, uint32_t, uint32_t, Metric, uint32_t, uint32_t, float, uint32_t);
template void build_shards<int8_t>(const std::string&, uint32_t, uint32_t, Metric, uint32_t, uint32_t, float, uint32_t);
template void build_shards<uint8_t>(const std::string&, uint32_t, uint32_t, Metric, uint32_t, uint32_t, float, uint32_t);

} // namespace build
} // namespace navis
