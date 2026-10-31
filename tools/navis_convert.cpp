#include "navis/util/file_utils.h"
#include "navis/types.h"
#include "navis/distance.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>
#ifdef NAVIS_BLAS_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

using namespace navis;

// ===========================================================================
// Subcommand: subset — extract first N vectors
// ===========================================================================

static void subset(const std::string& input, const std::string& output, uint32_t count) {
    std::ifstream in(input, std::ios::binary);
    if (!in.is_open()) {
        std::fprintf(stderr, "Cannot open input: %s\n", input.c_str());
        exit(1);
    }

    uint32_t npts, dim;
    in.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    uint32_t actual = std::min(npts, count);
    uint64_t data_bytes = static_cast<uint64_t>(actual) * dim * sizeof(float);

    std::printf("Extracting %u / %u vectors (%u dim) from %s\n",
                actual, npts, dim, input.c_str());

    std::ofstream out(output, std::ios::binary);
    if (!out.is_open()) {
        std::fprintf(stderr, "Cannot open output: %s\n", output.c_str());
        exit(1);
    }
    out.write(reinterpret_cast<char*>(&actual), sizeof(uint32_t));
    out.write(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    // Stream copy
    constexpr size_t BUF_SIZE = 1024 * 1024;
    std::vector<char> buf(BUF_SIZE);
    uint64_t remaining = data_bytes;

    while (remaining > 0) {
        size_t to_read = std::min(remaining, static_cast<uint64_t>(BUF_SIZE));
        in.read(buf.data(), to_read);
        if (!in) {
            std::fprintf(stderr, "Read error: expected %lu more bytes\n", remaining);
            exit(1);
        }
        out.write(buf.data(), to_read);
        remaining -= to_read;
    }
    out.flush();

    // Verify output
    std::ifstream verify(output, std::ios::binary);
    uint32_t v_npts, v_dim;
    verify.read(reinterpret_cast<char*>(&v_npts), sizeof(uint32_t));
    verify.read(reinterpret_cast<char*>(&v_dim), sizeof(uint32_t));
    verify.seekg(0, std::ios::end);
    uint64_t file_size = verify.tellg();
    uint64_t expected_size = 8 + data_bytes;

    if (v_npts != actual || v_dim != dim || file_size != expected_size) {
        std::fprintf(stderr, "Verification FAILED:\n"
                     "  header: %u x %u (expected %u x %u)\n"
                     "  file size: %lu (expected %lu)\n",
                     v_npts, v_dim, actual, dim, file_size, expected_size);
        exit(1);
    }
    std::printf("Verified: %u vectors x %u dim, %lu bytes\n", v_npts, v_dim, file_size);
    std::printf("Written to %s\n", output.c_str());
}

// ===========================================================================
// Subcommand: batch-gt — exact ground truth for an insert stream
//
// Ids [0, total) are split into segments of g = gcd(index, batch) vectors.
// One brute-force pass records, for every query, its exact top-K within each
// segment. Every live set is a union of whole segments:
//   insert workload : [0, index + offset)
//   delete workload : [offset, index + offset)   (the oldest vectors deleted)
// so the exact top-K of a live set is the top-K of its segments' lists.
// The data is streamed in partitions sized to fit --mem-budget; distances
// are computed with BLAS sgemm in batches of GT_QBATCH queries.
// ===========================================================================

static constexpr size_t GT_QBATCH  = 1024;        // queries per sgemm batch
static constexpr size_t GT_ALIGN   = 512;         // buffer alignment for AVX-512

// aligned_alloc requires size to be a multiple of alignment
static void* aligned_new(size_t bytes) {
    size_t aligned_sz = (bytes + GT_ALIGN - 1) / GT_ALIGN * GT_ALIGN;
    return std::aligned_alloc(GT_ALIGN, aligned_sz);
}

// Read npts vectors from an open file, converting to float if needed.
template <typename T>
static void load_partition_as_float(std::ifstream& in, float* out,
                                     size_t npts, size_t dim) {
    if constexpr (std::is_same_v<T, float>) {
        in.read(reinterpret_cast<char*>(out), npts * dim * sizeof(float));
    } else {
        std::vector<T> raw(npts * dim);
        in.read(reinterpret_cast<char*>(raw.data()), npts * dim * sizeof(T));
        #pragma omp parallel for schedule(static, 65536)
        for (int64_t i = 0; i < (int64_t)(npts * dim); i++)
            out[i] = static_cast<float>(raw[i]);
    }
}

// ||v||^2 for each row of a row-major [npts x dim] matrix.
static void compute_l2sq(float* l2sq, const float* data, size_t npts, size_t dim) {
    #pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t)npts; i++)
        l2sq[i] = cblas_sdot(dim, data + i * dim, 1, data + i * dim, 1);
}

// Compute L2 distance matrix: dist[p][q] = ||points[p] - queries[q]||^2
// via  ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b  using three sgemm calls.
// Output is col-major [npoints x nqueries]: dist[p + q*npoints].
static void compute_dist_matrix(size_t dim, float* dist_matrix,
                                 size_t npoints, const float* points,
                                 const float* points_l2sq,
                                 size_t nqueries, const float* queries,
                                 const float* queries_l2sq) {
    size_t n = std::max(npoints, nqueries);
    std::vector<float> ones(n, 1.0f);

    // dist = -2 * points^T * queries
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                npoints, nqueries, dim, -2.0f,
                points, dim, queries, dim,
                0.0f, dist_matrix, npoints);
    // dist += ||points||^2  (broadcast rows)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                npoints, nqueries, 1, 1.0f,
                points_l2sq, npoints, ones.data(), nqueries,
                1.0f, dist_matrix, npoints);
    // dist += ||queries||^2  (broadcast cols)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                npoints, nqueries, 1, 1.0f,
                ones.data(), npoints, queries_l2sq, nqueries,
                1.0f, dist_matrix, npoints);
}

// (distance, id): orders by distance, ties broken by the smaller id.
using DistId = std::pair<float, uint32_t>;

// Exact top-K of every query within every segment of `seg` vectors over the
// first `total` vectors of data_path. Layout [query][segment][K], ascending,
// padded with (FLT_MAX, UINT32_MAX) where a segment has fewer than K vectors.
template <typename T>
static std::vector<DistId> segment_topk(const std::string& data_path,
                                        const float* queries, size_t nquery, size_t dim,
                                        uint64_t total, uint64_t seg, uint32_t K,
                                        double mem_budget_gb) {
    const size_t nseg = total / seg;
    std::vector<DistId> top(nquery * nseg * K, DistId{FLT_MAX, UINT32_MAX});

    // Partition: a whole number of segments, sized so that the vectors, their
    // distance-matrix columns for one query batch, and the per-segment
    // results fit within the budget (3% is left for BLAS/OpenMP workspace).
    const double fixed = top.size() * sizeof(DistId) + nquery * (dim + 1.0) * sizeof(float);
    const double per_point = (dim + GT_QBATCH + 2) * sizeof(float);
    const double avail = 0.97 * mem_budget_gb * 1e9 - fixed;
    if (avail < per_point * seg) {
        std::fprintf(stderr, "--mem-budget %.1f GB is too small: per-segment results need "
                     "%.1f GB and one segment of %lu vectors needs %.2f GB more\n",
                     mem_budget_gb, fixed / 1e9, (unsigned long)seg, per_point * seg / 1e9);
        exit(1);
    }
    const uint64_t part = std::min<uint64_t>(total,
        static_cast<uint64_t>(avail / per_point / seg) * seg);
    const size_t num_parts = (total + part - 1) / part;
    std::printf("Ground truth: %lu vectors x %zu queries, %zu segments of %lu, "
                "%zu partitions of <= %lu\n", (unsigned long)total, nquery, nseg,
                (unsigned long)seg, num_parts, (unsigned long)part);

    float* part_buf     = static_cast<float*>(aligned_new(part * dim * sizeof(float)));
    float* dist         = static_cast<float*>(aligned_new(GT_QBATCH * part * sizeof(float)));
    float* points_l2sq  = new float[part];
    float* queries_l2sq = new float[nquery];
    compute_l2sq(queries_l2sq, queries, nquery, dim);

    std::ifstream in(data_path, std::ios::binary);
    in.seekg(8);
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t start = 0, p = 0; start < total; start += part, ++p) {
        const uint64_t n = std::min(part, total - start);
        load_partition_as_float<T>(in, part_buf, n, dim);
        compute_l2sq(points_l2sq, part_buf, n, dim);

        for (size_t qb = 0; qb < nquery; qb += GT_QBATCH) {
            const size_t qn = std::min(GT_QBATCH, nquery - qb);
            compute_dist_matrix(dim, dist, n, part_buf, points_l2sq,
                                qn, queries + qb * dim, queries_l2sq + qb);
            #pragma omp parallel for schedule(dynamic, 16)
            for (int64_t q = (int64_t)qb; q < (int64_t)(qb + qn); q++) {
                const float* col = dist + (q - qb) * n;
                std::vector<DistId> heap;   // max-heap of the K closest so far
                heap.reserve(K);
                for (uint64_t s0 = 0; s0 < n; s0 += seg) {
                    heap.clear();
                    for (uint64_t i = s0; i < s0 + seg; i++) {
                        DistId c{col[i], static_cast<uint32_t>(start + i)};
                        if (heap.size() < K) {
                            heap.push_back(c);
                            std::push_heap(heap.begin(), heap.end());
                        } else if (c < heap.front()) {
                            std::pop_heap(heap.begin(), heap.end());
                            heap.back() = c;
                            std::push_heap(heap.begin(), heap.end());
                        }
                    }
                    std::sort_heap(heap.begin(), heap.end());
                    std::copy(heap.begin(), heap.end(),
                              top.begin() + (q * nseg + (start + s0) / seg) * K);
                }
            }
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  [partition %lu/%zu] done (%.0fs elapsed, ETA %.0fs)\n",
                    (unsigned long)(p + 1), num_parts, el, el / (p + 1) * (num_parts - p - 1));
        std::fflush(stdout);
    }
    std::free(part_buf);
    std::free(dist);
    delete[] points_l2sq;
    delete[] queries_l2sq;
    return top;
}

// Write gt_<offset>.bin for offset = 0, batch, ..., total - index: the exact
// top-K over the live ids [begin, index + offset), where begin = 0 for the
// insert workload and begin = offset for the delete (sliding-window) workload.
static void write_step_gt(const std::vector<DistId>& top, size_t nquery, size_t nseg,
                          uint32_t K, uint64_t seg, uint64_t index, uint64_t total,
                          uint64_t batch, bool sliding, const std::string& dir) {
    std::filesystem::create_directories(dir);
    const uint64_t n_steps = (total - index) / batch + 1;
    std::vector<uint32_t> gt(nquery * K);
    for (uint64_t s = 0; s < n_steps; s++) {
        const uint64_t offset = s * batch;
        const uint64_t sb = (sliding ? offset : 0) / seg;
        const uint64_t se = (index + offset) / seg;
        #pragma omp parallel for schedule(static)
        for (int64_t q = 0; q < (int64_t)nquery; q++) {
            std::vector<DistId> best(K);
            auto first = top.begin() + (q * nseg + sb) * K;
            auto last  = top.begin() + (q * nseg + se) * K;
            std::partial_sort_copy(first, last, best.begin(), best.end());
            for (uint32_t j = 0; j < K; j++) gt[q * K + j] = best[j].second;
        }
        const std::string path = dir + "/gt_" + std::to_string(offset) + ".bin";
        save_bin<uint32_t>(path, gt.data(), nquery, K);
    }
    std::printf("Wrote %lu files gt_0 ... gt_%lu.bin to %s (%s)\n",
                (unsigned long)n_steps, (unsigned long)(total - index), dir.c_str(),
                sliding ? "live ids [offset, index+offset)" : "live ids [0, index+offset)");
}

template <typename T>
static void batch_groundtruth(const std::string& data_path, const std::string& query_path,
                              uint64_t index, uint64_t total, uint64_t batch, uint32_t K,
                              const std::string& insert_dir, const std::string& delete_dir,
                              uint32_t threads, double mem_budget_gb) {
    if (total <= index) {
        std::fprintf(stderr, "total-npts (%lu) must be > index-npts (%lu)\n",
                     (unsigned long)total, (unsigned long)index);
        exit(1);
    }
    if ((total - index) % batch != 0) {
        std::fprintf(stderr, "(total-npts - index-npts) (%lu) must be divisible by batch (%lu)\n",
                     (unsigned long)(total - index), (unsigned long)batch);
        exit(1);
    }
    if (index < K) {
        std::fprintf(stderr, "index-npts (%lu) must be >= K (%u)\n", (unsigned long)index, K);
        exit(1);
    }
    uint32_t file_npts, dim, nquery, qdim;
    get_bin_metadata(data_path, file_npts, dim);
    get_bin_metadata(query_path, nquery, qdim);
    if (file_npts < total) {
        std::fprintf(stderr, "%s has %u vectors, fewer than --total-npts %lu\n",
                     data_path.c_str(), file_npts, (unsigned long)total);
        exit(1);
    }
    if (dim != qdim) {
        std::fprintf(stderr, "Dimension mismatch: data=%u, query=%u\n", dim, qdim);
        exit(1);
    }

    omp_set_num_threads(threads);
#ifdef NAVIS_BLAS_MKL
    mkl_set_num_threads(threads);
#endif
    float* queries = static_cast<float*>(aligned_new(size_t(nquery) * dim * sizeof(float)));
    {
        std::ifstream qin(query_path, std::ios::binary);
        qin.seekg(8);
        load_partition_as_float<T>(qin, queries, nquery, dim);
    }

    const uint64_t seg = std::gcd(index, batch);
    const auto t0 = std::chrono::steady_clock::now();
    auto top = segment_topk<T>(data_path, queries, nquery, dim, total, seg, K, mem_budget_gb);
    std::free(queries);

    const size_t nseg = total / seg;
    if (!insert_dir.empty())
        write_step_gt(top, nquery, nseg, K, seg, index, total, batch, false, insert_dir);
    if (!delete_dir.empty())
        write_step_gt(top, nquery, nseg, K, seg, index, total, batch, true, delete_dir);
    std::printf("batch-gt complete in %.0fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// ===========================================================================
// Main
// ===========================================================================

static void print_usage() {
    std::printf(
        "Usage: navis_convert <subcommand> [args...]\n"
        "\n"
        "Subcommands:\n"
        "  subset    <input.bin> <output.bin> --count <N>\n"
        "            Copy the first N vectors.\n"
        "  batch-gt  <data.bin> <query.bin> --index-npts <N> --total-npts <N>\n"
        "            --batch <N> [-K <K>] [-o <insert_dir>] [--delete-o <delete_dir>]\n"
        "            [--threads <T>] [--mem-budget <GB>] [--type float|int8|uint8]\n"
        "            Exact top-K ground truth for every batch boundary of an insert\n"
        "            stream over the first total-npts vectors of data.bin, written as\n"
        "            gt_<offset>.bin for offset = 0, batch, ..., total-index:\n"
        "              -o          insert workload, live ids [0, index+offset)\n"
        "              --delete-o  delete workload, live ids [offset, index+offset)\n"
        "            Both come from one pass. Peak memory stays within --mem-budget\n"
        "            (default 32 GB).\n"
    );
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "subset" && argc >= 4) {
        uint32_t count = 0;
        for (int i = 4; i < argc; i++) {
            if (std::string(argv[i]) == "--count" && i + 1 < argc) {
                count = std::stoul(argv[i + 1]);
            }
        }
        if (count == 0) {
            std::fprintf(stderr, "Missing --count <N>\n");
            return 1;
        }
        subset(argv[2], argv[3], count);
    } else if (cmd == "batch-gt" && argc >= 4) {
        std::string data = argv[2];
        std::string query = argv[3];
        std::string insert_dir, delete_dir;
        std::string type = "float";
        uint64_t index_npts = 0, total_npts = 0, batch_npts = 0;
        uint32_t K = 10;
        uint32_t threads = 1;
        double mem_budget_gb = 32.0;

        for (int i = 4; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--index-npts" && i + 1 < argc) index_npts = std::stoull(argv[++i]);
            else if (arg == "--total-npts" && i + 1 < argc) total_npts = std::stoull(argv[++i]);
            else if (arg == "--batch" && i + 1 < argc) batch_npts = std::stoull(argv[++i]);
            else if (arg == "-K" && i + 1 < argc) K = std::stoul(argv[++i]);
            else if (arg == "-o" && i + 1 < argc) insert_dir = argv[++i];
            else if (arg == "--delete-o" && i + 1 < argc) delete_dir = argv[++i];
            else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (arg == "--mem-budget" && i + 1 < argc) mem_budget_gb = std::stod(argv[++i]);
            else if (arg == "--type" && i + 1 < argc) type = argv[++i];
            else {
                std::fprintf(stderr, "Unknown batch-gt argument: %s\n", arg.c_str());
                return 1;
            }
        }
        if (index_npts == 0 || total_npts == 0 || batch_npts == 0 ||
            (insert_dir.empty() && delete_dir.empty())) {
            std::fprintf(stderr,
                "batch-gt requires --index-npts --total-npts --batch and -o and/or --delete-o\n");
            return 1;
        }

        if (type == "float")       batch_groundtruth<float>(data, query, index_npts, total_npts, batch_npts, K, insert_dir, delete_dir, threads, mem_budget_gb);
        else if (type == "int8")   batch_groundtruth<int8_t>(data, query, index_npts, total_npts, batch_npts, K, insert_dir, delete_dir, threads, mem_budget_gb);
        else if (type == "uint8")  batch_groundtruth<uint8_t>(data, query, index_npts, total_npts, batch_npts, K, insert_dir, delete_dir, threads, mem_budget_gb);
        else {
            std::fprintf(stderr, "Unknown type: %s\n", type.c_str());
            return 1;
        }
    } else {
        print_usage();
        return 1;
    }

    return 0;
}
