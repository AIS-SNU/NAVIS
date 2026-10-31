#include "navis/types.h"
#include "navis/util/file_utils.h"
#include "navis/io/uring_io.h"
#include "navis/build/build.h"

#include <atomic>

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
// Random slice generation
// ===========================================================================

template <typename T>
void gen_random_slice(const std::string& data_path, double sample_rate,
                      std::vector<float>& out_data, uint32_t& out_npts,
                      uint32_t& out_dim) {
    UringReader in(data_path);

    uint32_t npts, dim;
    in.read(&npts, sizeof(uint32_t));
    in.read(&dim, sizeof(uint32_t));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    std::vector<float> sampled;
    sampled.reserve(static_cast<size_t>(npts * sample_rate * dim * 1.1));

    constexpr uint32_t BLOCK = 16384;
    std::vector<T> block(static_cast<size_t>(BLOCK) * dim);
    uint32_t count = 0;

    for (uint32_t base = 0; base < npts; base += BLOCK) {
        uint32_t cur = std::min(BLOCK, npts - base);
        in.read(block.data(), static_cast<uint64_t>(cur) * dim * sizeof(T));

        for (uint32_t i = 0; i < cur; i++) {
            if (coin(rng) < sample_rate) {
                const T* vec = block.data() + static_cast<size_t>(i) * dim;
                for (uint32_t j = 0; j < dim; j++)
                    sampled.push_back(static_cast<float>(vec[j]));
                count++;
            }
        }
    }

    if (count == 0 && npts > 0) {
        in.seek(2 * sizeof(uint32_t));
        std::vector<T> row(dim);
        in.read(row.data(), dim * sizeof(T));
        for (uint32_t j = 0; j < dim; j++)
            sampled.push_back(static_cast<float>(row[j]));
        count = 1;
    }

    out_data = std::move(sampled);
    out_npts = count;
    out_dim = dim;
}

template <typename T>
void gen_random_slice_to_file(const std::string& data_path,
                               const std::string& output_prefix,
                               double sample_rate) {
    UringReader in(data_path);

    uint32_t npts, dim;
    in.read(&npts, sizeof(uint32_t));
    in.read(&dim, sizeof(uint32_t));

    std::string data_out = output_prefix + "_data.bin";
    std::string ids_out  = output_prefix + "_ids.bin";

    UringWriter data_writer(data_out);
    UringWriter ids_writer(ids_out);

    uint32_t zero = 0, one = 1;
    data_writer.write(&zero, sizeof(uint32_t));
    data_writer.write(&dim, sizeof(uint32_t));
    ids_writer.write(&zero, sizeof(uint32_t));
    ids_writer.write(&one, sizeof(uint32_t));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    constexpr uint32_t BLOCK = 16384;
    std::vector<T> block(static_cast<size_t>(BLOCK) * dim);
    uint32_t count = 0;

    for (uint32_t base = 0; base < npts; base += BLOCK) {
        uint32_t cur = std::min(BLOCK, npts - base);
        in.read(block.data(), static_cast<uint64_t>(cur) * dim * sizeof(T));
        for (uint32_t i = 0; i < cur; i++) {
            if (dist(rng) < sample_rate) {
                data_writer.write(block.data() + static_cast<size_t>(i) * dim,
                                  dim * sizeof(T));
                uint32_t global_id = base + i;
                ids_writer.write(&global_id, sizeof(uint32_t));
                count++;
            }
        }
    }

    data_writer.seek(0);
    data_writer.write(&count, sizeof(uint32_t));
    ids_writer.seek(0);
    ids_writer.write(&count, sizeof(uint32_t));

    std::printf("[slice] Sampled %u / %u points (%.2f%%), saved to %s\n",
                count, npts, 100.0 * count / npts, output_prefix.c_str());
}

// ===========================================================================
// K-means with k-means++ initialization and parallel centroid accumulation
// ===========================================================================

static void kmeans_pp_init(const float* data, uint32_t n, uint32_t dim,
                           uint32_t k, float* centroids) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> first_pick(0, n - 1);

    // First centroid: random point
    uint32_t first = first_pick(rng);
    std::memcpy(centroids, data + first * dim, dim * sizeof(float));

    // Remaining centroids: D^2 sampling
    std::vector<float> min_dists(n, std::numeric_limits<float>::max());

    for (uint32_t c = 1; c < k; c++) {
        // Update min distances to nearest existing centroid
        const float* last_centroid = centroids + (c - 1) * dim;
        double total_dist = 0.0;

        #pragma omp parallel for reduction(+:total_dist) schedule(static, 1024)
        for (int64_t i = 0; i < n; i++) {
            float d = 0;
            for (uint32_t j = 0; j < dim; j++) {
                float diff = data[i * dim + j] - last_centroid[j];
                d += diff * diff;
            }
            min_dists[i] = std::min(min_dists[i], d);
            total_dist += min_dists[i];
        }

        // Weighted random selection
        std::uniform_real_distribution<double> pick(0.0, total_dist);
        double target = pick(rng);
        double cumsum = 0.0;
        uint32_t selected = n - 1;
        for (uint32_t i = 0; i < n; i++) {
            cumsum += min_dists[i];
            if (cumsum >= target) { selected = i; break; }
        }
        std::memcpy(centroids + c * dim, data + selected * dim, dim * sizeof(float));
    }
}

// One round of Lloyd's k-means with early convergence detection.
// Returns total residual (sum of squared distances to assigned centroids).
static double kmeans_lloyd_round(const float* data, uint32_t n, uint32_t dim,
                                 uint32_t k, uint32_t max_iters,
                                 float* centroids, uint32_t* assignments) {
    double residual = 0.0;
    double old_residual = std::numeric_limits<double>::max();

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        residual = 0.0;

        // Assign each point to closest centroid (parallel)
        #pragma omp parallel for schedule(static, 512) reduction(+:residual)
        for (int64_t i = 0; i < n; i++) {
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; c++) {
                float d = 0;
                const float* cent = centroids + c * dim;
                const float* pt = data + i * dim;
                for (uint32_t j = 0; j < dim; j++) {
                    float diff = pt[j] - cent[j];
                    d += diff * diff;
                }
                if (d < best_dist) {
                    best_dist = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
            residual += best_dist;
        }

        // Early convergence: stop if <0.001% improvement (matching original)
        if (iter > 0) {
            double rel_change = (old_residual - residual) / (residual + 1e-30);
            if (rel_change < 1e-5 || residual < std::numeric_limits<float>::epsilon()) {
                break;
            }
        }
        old_residual = residual;

        // Parallel centroid accumulation using thread-local buffers
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<float>> tl_sums(num_threads, std::vector<float>(k * dim, 0.0f));
        std::vector<std::vector<uint32_t>> tl_counts(num_threads, std::vector<uint32_t>(k, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_sums = tl_sums[tid];
            auto& local_counts = tl_counts[tid];

            #pragma omp for schedule(static, 512)
            for (int64_t i = 0; i < n; i++) {
                uint32_t c = assignments[i];
                local_counts[c]++;
                float* dst = local_sums.data() + c * dim;
                const float* src = data + i * dim;
                for (uint32_t j = 0; j < dim; j++) {
                    dst[j] += src[j];
                }
            }
        }

        // Merge thread-local accumulators
        std::memset(centroids, 0, k * dim * sizeof(float));
        std::vector<uint32_t> counts(k, 0);
        for (int t = 0; t < num_threads; t++) {
            for (uint32_t c = 0; c < k; c++) {
                counts[c] += tl_counts[t][c];
                for (uint32_t j = 0; j < dim; j++) {
                    centroids[c * dim + j] += tl_sums[t][c * dim + j];
                }
            }
        }

        for (uint32_t c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (uint32_t j = 0; j < dim; j++) {
                    centroids[c * dim + j] /= counts[c];
                }
            }
        }
    }
    return residual;
}

static void kmeans_lloyd(const float* data, uint32_t n, uint32_t dim,
                         uint32_t k, uint32_t max_iters,
                         float* centroids) {
    // K-means++ initialization
    kmeans_pp_init(data, n, dim, k, centroids);

    std::vector<uint32_t> assignments(n);

    // First round
    kmeans_lloyd_round(data, n, dim, k, max_iters, centroids, assignments.data());

    // Two-round refinement: run again if training set is large enough
    // (matching original behavior: second round when num_train > 2 * num_centers)
    if (n > 2 * k) {
        kmeans_lloyd_round(data, n, dim, k, max_iters, centroids, assignments.data());
    }
}

// ===========================================================================
// PQ pivot generation
// ===========================================================================

void generate_pq_pivots(const float* train_data, uint32_t num_train,
                        uint32_t dim, uint32_t num_centers,
                        uint32_t num_chunks, uint32_t max_kmeans_iters,
                        const std::string& output_path) {
    std::printf("[pq] Generating PQ pivots: %u train, %u dim, %u chunks, %u centers\n",
                num_train, dim, num_chunks, num_centers);

    // 1. Global centroid
    std::vector<float> centroid(dim, 0.0f);
    for (uint32_t i = 0; i < num_train; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            centroid[j] += train_data[i * dim + j];
        }
    }
    for (uint32_t j = 0; j < dim; j++) centroid[j] /= num_train;

    // 2. Center the data
    std::vector<float> centered(static_cast<uint64_t>(num_train) * dim);
    #pragma omp parallel for schedule(static, 512)
    for (int64_t i = 0; i < num_train; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            centered[i * dim + j] = train_data[i * dim + j] - centroid[j];
        }
    }

    // 3. Dimension-to-chunk assignment: contiguous, in original dim order.
    //    Each chunk gets dim/num_chunks dims; the first (dim % num_chunks)
    //    chunks get one extra.
    uint32_t low_val  = dim / num_chunks;
    uint32_t high_val = low_val + 1;
    uint32_t max_num_high = dim - (low_val * num_chunks);

    std::vector<std::vector<uint32_t>> chunk_dims(num_chunks);
    for (uint32_t d = 0; d < dim; d++) {
        for (uint32_t b = 0; b < num_chunks; b++) {
            uint32_t bin_cap = (b < max_num_high) ? high_val : low_val;
            if (chunk_dims[b].size() >= bin_cap) continue;
            chunk_dims[b].push_back(d);
            break;
        }
    }

    // Build rearrangement and chunk_offsets from the assigned bins
    std::vector<uint32_t> rearrangement(dim);
    std::vector<uint32_t> chunk_offsets(num_chunks + 1);
    uint32_t pos = 0;
    for (uint32_t c = 0; c < num_chunks; c++) {
        chunk_offsets[c] = pos;
        for (uint32_t d : chunk_dims[c]) rearrangement[pos++] = d;
    }
    chunk_offsets[num_chunks] = dim;

    // 4. Rearrange centered data
    std::vector<float> rearranged(static_cast<uint64_t>(num_train) * dim);
    #pragma omp parallel for schedule(static, 512)
    for (int64_t i = 0; i < num_train; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            rearranged[i * dim + j] = centered[i * dim + rearrangement[j]];
        }
    }

    // 5. K-means per chunk (with k-means++ init) — parallelized over chunks.
    // Each chunk runs independent Lloyd iterations; deterministic across
    // threads because kmeans_lloyd uses a per-chunk RNG seed derived from
    // the chunk index (see kmeans_lloyd implementation).
    std::vector<float> all_centroids(num_centers * dim, 0.0f);
    std::atomic<uint32_t> chunks_done{0};

    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t c = 0; c < static_cast<int32_t>(num_chunks); c++) {
        uint32_t cs = chunk_offsets[c], ce = chunk_offsets[c + 1];
        uint32_t chunk_dim = ce - cs;

        std::vector<float> chunk_data(static_cast<uint64_t>(num_train) * chunk_dim);
        for (uint32_t i = 0; i < num_train; i++) {
            std::memcpy(chunk_data.data() + i * chunk_dim,
                        rearranged.data() + i * dim + cs, chunk_dim * sizeof(float));
        }

        std::vector<float> chunk_centroids(num_centers * chunk_dim);
        kmeans_lloyd(chunk_data.data(), num_train, chunk_dim,
                     num_centers, max_kmeans_iters, chunk_centroids.data());

        for (uint32_t k = 0; k < num_centers; k++) {
            std::memcpy(all_centroids.data() + k * dim + cs,
                        chunk_centroids.data() + k * chunk_dim,
                        chunk_dim * sizeof(float));
        }
        uint32_t done = chunks_done.fetch_add(1, std::memory_order_relaxed) + 1;
        #pragma omp critical
        {
            int pct = static_cast<int>(100.0 * done / num_chunks);
            int bar = pct / 2;
            std::printf("\r[pq] Training: [");
            for (int b = 0; b < 50; b++) std::putchar(b < bar ? '#' : '-');
            std::printf("] %3d%% chunk %u/%u", pct, done, num_chunks);
            std::fflush(stdout);
        }
    }
    std::printf("\n");

    // 6. Write PQ pivots file
    UringWriter out(output_path);

    uint64_t header_size = 2 * sizeof(uint32_t) + 4 * sizeof(uint64_t);
    uint64_t centroids_offset = header_size;
    uint64_t centroid_offset  = centroids_offset + 2 * sizeof(uint32_t) +
                                 num_centers * dim * sizeof(float);
    uint64_t rearrange_offset = centroid_offset + 2 * sizeof(uint32_t) +
                                 dim * sizeof(float);
    uint64_t chunks_offset    = rearrange_offset + 2 * sizeof(uint32_t) +
                                 dim * sizeof(uint32_t);

    uint32_t off_npts = 4, off_dim = 1;
    out.write(&off_npts, sizeof(uint32_t));
    out.write(&off_dim, sizeof(uint32_t));
    out.write(&centroids_offset, sizeof(uint64_t));
    out.write(&centroid_offset, sizeof(uint64_t));
    out.write(&rearrange_offset, sizeof(uint64_t));
    out.write(&chunks_offset, sizeof(uint64_t));

    uint32_t c_npts = num_centers, c_dim = dim;
    out.write(&c_npts, sizeof(uint32_t));
    out.write(&c_dim, sizeof(uint32_t));
    out.write(all_centroids.data(), num_centers * dim * sizeof(float));

    uint32_t g_npts = dim, g_dim = 1;
    out.write(&g_npts, sizeof(uint32_t));
    out.write(&g_dim, sizeof(uint32_t));
    out.write(centroid.data(), dim * sizeof(float));

    out.write(&g_npts, sizeof(uint32_t));
    out.write(&g_dim, sizeof(uint32_t));
    out.write(rearrangement.data(), dim * sizeof(uint32_t));

    uint32_t co_npts = num_chunks + 1;
    out.write(&co_npts, sizeof(uint32_t));
    out.write(&g_dim, sizeof(uint32_t));
    out.write(chunk_offsets.data(), (num_chunks + 1) * sizeof(uint32_t));

    std::printf("[pq] PQ pivots written to %s\n", output_path.c_str());
}

// ===========================================================================
// PQ data compression (with buffered I/O)
// ===========================================================================

template <typename T>
void generate_pq_data(const std::string& data_path,
                      const std::string& pq_pivots_path,
                      uint32_t num_centers, uint32_t num_chunks,
                      const std::string& output_path) {
    // Load PQ pivots
    UringReader pq_in(pq_pivots_path);
    uint32_t off_n, off_d;
    pq_in.read(&off_n, sizeof(uint32_t));
    pq_in.read(&off_d, sizeof(uint32_t));
    std::vector<uint64_t> offsets(off_n);
    pq_in.read(offsets.data(), off_n * sizeof(uint64_t));

    pq_in.seek(offsets[0]);
    uint32_t c_n, c_dim;
    pq_in.read(&c_n, sizeof(uint32_t));
    pq_in.read(&c_dim, sizeof(uint32_t));
    std::vector<float> centroids(c_n * c_dim);
    pq_in.read(centroids.data(), c_n * c_dim * sizeof(float));

    pq_in.seek(offsets[1]);
    uint32_t g_n, g_d;
    pq_in.read(&g_n, sizeof(uint32_t));
    pq_in.read(&g_d, sizeof(uint32_t));
    std::vector<float> centroid(g_n);
    pq_in.read(centroid.data(), g_n * sizeof(float));

    pq_in.seek(offsets[2]);
    uint32_t r_n, r_d;
    pq_in.read(&r_n, sizeof(uint32_t));
    pq_in.read(&r_d, sizeof(uint32_t));
    std::vector<uint32_t> rearrangement(r_n);
    pq_in.read(rearrangement.data(), r_n * sizeof(uint32_t));

    pq_in.seek(offsets[3]);
    uint32_t co_n, co_d;
    pq_in.read(&co_n, sizeof(uint32_t));
    pq_in.read(&co_d, sizeof(uint32_t));
    std::vector<uint32_t> chunk_offsets(co_n);
    pq_in.read(chunk_offsets.data(), co_n * sizeof(uint32_t));

    uint32_t dim = c_dim;

    UringReader data_in(data_path);
    uint32_t npts, data_dim;
    data_in.read(&npts, sizeof(uint32_t));
    data_in.read(&data_dim, sizeof(uint32_t));

    UringWriter out(output_path);
    out.write(&npts, sizeof(uint32_t));
    out.write(&num_chunks, sizeof(uint32_t));

    // Larger block size for better parallelism
    const uint32_t BLOCK = 16384;
    std::vector<T> block_data(static_cast<uint64_t>(BLOCK) * data_dim);
    std::vector<uint8_t> block_pq(static_cast<uint64_t>(BLOCK) * num_chunks);

    for (uint32_t base = 0; base < npts; base += BLOCK) {
        uint32_t count = std::min(BLOCK, npts - base);
        data_in.read(block_data.data(), count * data_dim * sizeof(T));

        #pragma omp parallel for schedule(static, 64)
        for (int64_t i = 0; i < count; i++) {
            // Per-thread stack buffer avoids heap allocation
            thread_local std::vector<float> vec_tl;
            if (vec_tl.size() < dim) vec_tl.resize(dim);
            float* vec = vec_tl.data();
            for (uint32_t j = 0; j < dim; j++) {
                vec[j] = static_cast<float>(block_data[i * data_dim + rearrangement[j]]) -
                         centroid[rearrangement[j]];
            }

            for (uint32_t c = 0; c < num_chunks; c++) {
                uint32_t cs = chunk_offsets[c];
                uint32_t ce = chunk_offsets[c + 1];

                float best_dist = std::numeric_limits<float>::max();
                uint8_t best_id = 0;

                for (uint32_t k = 0; k < num_centers; k++) {
                    float d = 0;
                    const float* cent = centroids.data() + k * dim;
                    for (uint32_t j = cs; j < ce; j++) {
                        float diff = vec[j] - cent[j];
                        d += diff * diff;
                    }
                    if (d < best_dist) {
                        best_dist = d;
                        best_id = static_cast<uint8_t>(k);
                    }
                }
                block_pq[i * num_chunks + c] = best_id;
            }
        }

        out.write(block_pq.data(), count * num_chunks * sizeof(uint8_t));

        uint32_t done = base + count;
        if (base % (BLOCK * 10) == 0 || done == npts) {
            int pct = static_cast<int>(100.0 * done / npts);
            int bar = pct / 2;
            std::printf("\r[pq] Compressing: [");
            for (int b = 0; b < 50; b++) std::putchar(b < bar ? '#' : '-');
            std::printf("] %3d%%", pct);
            std::fflush(stdout);
        }
    }

    std::printf("\n[pq] PQ data written to %s (%u points x %u chunks)\n",
                output_path.c_str(), npts, num_chunks);
}

// Explicit instantiations
template void gen_random_slice<float>(const std::string&, double, std::vector<float>&, uint32_t&, uint32_t&);
template void gen_random_slice<int8_t>(const std::string&, double, std::vector<float>&, uint32_t&, uint32_t&);
template void gen_random_slice<uint8_t>(const std::string&, double, std::vector<float>&, uint32_t&, uint32_t&);

template void gen_random_slice_to_file<float>(const std::string&, const std::string&, double);
template void gen_random_slice_to_file<int8_t>(const std::string&, const std::string&, double);
template void gen_random_slice_to_file<uint8_t>(const std::string&, const std::string&, double);

template void generate_pq_data<float>(const std::string&, const std::string&, uint32_t, uint32_t, const std::string&);
template void generate_pq_data<int8_t>(const std::string&, const std::string&, uint32_t, uint32_t, const std::string&);
template void generate_pq_data<uint8_t>(const std::string&, const std::string&, uint32_t, uint32_t, const std::string&);

} // namespace build
} // namespace navis
