// baselines/freshdiskann/navis_to_freshdiskann.cpp
// NAVIS index -> FreshDiskANN index (packed layout) + empty in-memory indexes.
//
// Produces:
//   {out_prefix}_disk.index             — Fresh disk index (same packed layout as Odin)
//   {out_prefix}_disk.index.tags        — identity uint32 tags (bin format)
//   {out_prefix}_pq_pivots.bin          — retrained PQ pivots (contiguous chunks)
//   {out_prefix}_pq_compressed.bin      — PQ-encoded vectors
//   {out_prefix}_mem_0                  — empty Vamana mem-index graph (R/ep/frozen)
//   {out_prefix}_mem_0.data             — empty mem-index data (1 frozen-point slot)
//   {out_prefix}_mem_0.tags             — empty mem-index tags (1 slot)
//   {out_prefix}_mem_1                  — same as mem_0
//   {out_prefix}_mem_1.data             — same as mem_0.data
//   {out_prefix}_mem_1.tags             — same as mem_0.tags
// (The _shadow1_* and _merge_* working copies are made by run_insert.sh before
//  every run.)
//
// Fresh disk.index header (verified against FreshDiskANN/src/aux_utils.cpp
// create_disk_layout(), line 742-884):
//   Sector 0 layout: save_bin<uint64>[9][1]  at offset 0, zero-padded to 4096B
//   [int32 n=9][int32 d=1] preamble
//   uint64 npts
//   uint64 ndims
//   uint64 medoid
//   uint64 max_node_len        = ((R+1)*4 + dim*4)
//   uint64 nnodes_per_sector   = 4096 / max_node_len
//   uint64 frozen_num          = 0 (static disk index)
//   uint64 frozen_loc          = 0
//   uint64 file_size           = (1 + n_sectors) * 4096
//   uint64 tags_end_offset     = file_size + tag_bytes_written
//
// Node layout (same as Odin):
//   [vec: float*dim][nnbr: uint32][nbrs: uint32*R]  (zero-padded to max_node_len)
//
// PQ pivots / compressed: identical format to navis_to_odinann.cpp.
//
// Mem-index stub format (non-single-file mode, verified against
// FreshDiskANN/src/index.cpp save_graph/save_data/save_tags):
//   Graph file (no ext): [u64 index_size=24][u32 max_deg=0][u32 ep=0][u64 frozen_num=1]
//   .data:  save_bin<float>[1][dim]  (one zero frozen-point vector)
//   .tags:  save_bin<uint32>[1][1]   (one frozen-point tag = 0)

#include "navis_layout_reader.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace bench;

static constexpr size_t kSector     = 4096;
static constexpr size_t kMetaSector = 4096;   // PQ pivots METADATA_SIZE
static constexpr int    kNumCenters = 256;
static constexpr int    kKMeansIter = 25;

// ---------------------------------------------------------------------------
// Low-level I/O helpers
// ---------------------------------------------------------------------------

static void xpwrite_all(int fd, const void* buf, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = ::pwrite(fd, static_cast<const char*>(buf) + done,
                             n - done, off + (off_t)done);
        if (r < 0) { std::perror("pwrite"); std::exit(1); }
        done += (size_t)r;
    }
}

static void xwrite_all(int fd, const void* buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = ::write(fd, static_cast<const char*>(buf) + done, n - done);
        if (r < 0) { std::perror("write"); std::exit(1); }
        done += (size_t)r;
    }
}

static void xpread_all(int fd, void* buf, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = ::pread(fd, static_cast<char*>(buf) + done,
                            n - done, off + (off_t)done);
        if (r < 0) { std::perror("pread"); std::exit(1); }
        if (r == 0) { std::fprintf(stderr, "pread: unexpected EOF\n"); std::exit(1); }
        done += (size_t)r;
    }
}

// ---------------------------------------------------------------------------
// save_bin helpers (matches Fresh's diskann::save_bin template)
// Writes [int32 npts][int32 ndims] preamble + data at byte offset `off`.
// Returns bytes written (preamble + data).
// ---------------------------------------------------------------------------

static size_t save_bin_u64(int fd, const uint64_t* data, size_t npts, size_t ndims, off_t off) {
    int32_t n32 = (int32_t)npts, d32 = (int32_t)ndims;
    size_t total = 8 + npts * ndims * sizeof(uint64_t);
    std::vector<char> buf(total);
    std::memcpy(buf.data() + 0, &n32, 4);
    std::memcpy(buf.data() + 4, &d32, 4);
    std::memcpy(buf.data() + 8, data, npts * ndims * sizeof(uint64_t));
    xpwrite_all(fd, buf.data(), total, off);
    return total;
}

static size_t save_bin_f32(int fd, const float* data, size_t npts, size_t ndims, off_t off) {
    int32_t n32 = (int32_t)npts, d32 = (int32_t)ndims;
    size_t total = 8 + npts * ndims * sizeof(float);
    std::vector<char> buf(total);
    std::memcpy(buf.data() + 0, &n32, 4);
    std::memcpy(buf.data() + 4, &d32, 4);
    std::memcpy(buf.data() + 8, data, npts * ndims * sizeof(float));
    xpwrite_all(fd, buf.data(), total, off);
    return total;
}

static size_t save_bin_u32(int fd, const uint32_t* data, size_t npts, size_t ndims, off_t off) {
    int32_t n32 = (int32_t)npts, d32 = (int32_t)ndims;
    size_t total = 8 + npts * ndims * sizeof(uint32_t);
    std::vector<char> buf(total);
    std::memcpy(buf.data() + 0, &n32, 4);
    std::memcpy(buf.data() + 4, &d32, 4);
    std::memcpy(buf.data() + 8, data, npts * ndims * sizeof(uint32_t));
    xpwrite_all(fd, buf.data(), total, off);
    return total;
}

static size_t save_bin_u8(int fd, const uint8_t* data, size_t npts, size_t ndims, off_t off) {
    int32_t n32 = (int32_t)npts, d32 = (int32_t)ndims;
    // Write preamble separately (data may be large)
    xpwrite_all(fd, &n32, 4, off);
    xpwrite_all(fd, &d32, 4, off + 4);
    xpwrite_all(fd, data, npts * ndims, off + 8);
    return 8 + npts * ndims;
}

// ---------------------------------------------------------------------------
// Minimal k-means PQ trainer (contiguous chunks, no rearrangement)
// Copied from navis_to_odinann.cpp — the math util is identical; only the
// writer functions are Fresh-specific.
// ---------------------------------------------------------------------------

static void kmeans_lloyd(const float* train_data, size_t num_train,
                         size_t chunk_dim,
                         float* centroids, int n_iters) {
    std::vector<int>    assign(num_train);
    std::vector<float>  new_centroids(kNumCenters * chunk_dim);
    std::vector<size_t> counts(kNumCenters);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Assignment step
        for (size_t i = 0; i < num_train; ++i) {
            const float* v = train_data + i * chunk_dim;
            float best = std::numeric_limits<float>::max();
            int   best_c = 0;
            for (int c = 0; c < kNumCenters; ++c) {
                const float* cv = centroids + c * chunk_dim;
                float d = 0.0f;
                for (size_t k = 0; k < chunk_dim; ++k) {
                    float diff = v[k] - cv[k];
                    d += diff * diff;
                }
                if (d < best) { best = d; best_c = c; }
            }
            assign[i] = best_c;
        }
        // Update step
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < num_train; ++i) {
            int c = assign[i];
            const float* v = train_data + i * chunk_dim;
            float* nc = new_centroids.data() + c * chunk_dim;
            for (size_t k = 0; k < chunk_dim; ++k) nc[k] += v[k];
            counts[c]++;
        }
        for (int c = 0; c < kNumCenters; ++c) {
            if (counts[c] > 0) {
                float* nc = new_centroids.data() + c * chunk_dim;
                for (size_t k = 0; k < chunk_dim; ++k) nc[k] /= (float)counts[c];
                std::memcpy(centroids + c * chunk_dim, nc, chunk_dim * sizeof(float));
            }
            // else: keep old centroid (empty cluster — don't reset)
        }
    }
}

// ---------------------------------------------------------------------------
// retrain_and_write_pq_contiguous — Fresh-specific writer
//
// Same PQ math as navis_to_odinann.cpp but writes Fresh's {prefix}_pq_pivots.bin
// and {prefix}_pq_compressed.bin.  Format is identical to Odin's (both derive
// from the same DiskANN generate_pq_pivots), but this is a separate function
// to keep the FreshDiskANN writer self-contained.
// ---------------------------------------------------------------------------
static bool retrain_and_write_fresh_pq(
        const std::string& out_prefix,
        uint64_t npts, uint64_t dim,
        int pq_bytes, int num_train,
        NavisLayoutReader& reader,
        std::vector<float>& out_centroid_ret,
        std::vector<float>& out_pivot_data_ret) {

    if (dim % (uint64_t)pq_bytes != 0) {
        std::fprintf(stderr,
            "ERROR: dim=%lu not divisible by pq_bytes=%d\n",
            (unsigned long)dim, pq_bytes);
        return false;
    }
    const size_t n_chunks  = (size_t)pq_bytes;
    const size_t chunk_dim = (size_t)(dim / (uint64_t)pq_bytes);

    std::fprintf(stderr,
        "  PQ retrain: dim=%lu n_chunks=%zu chunk_dim=%zu num_train=%d\n",
        (unsigned long)dim, n_chunks, chunk_dim, num_train);

    // --- 1. Sample training vectors ---
    size_t actual_train = (size_t)std::min((uint64_t)num_train, npts);
    std::vector<uint64_t> train_ids(actual_train);
    {
        std::mt19937_64 rng(0xDEADBEEF1234ULL);
        for (size_t i = 0; i < actual_train; ++i) train_ids[i] = i;
        for (uint64_t i = actual_train; i < npts; ++i) {
            uint64_t j = rng() % (i + 1);
            if (j < (uint64_t)actual_train) train_ids[j] = i;
        }
    }

    std::vector<float> train_data(actual_train * dim);
    {
        std::vector<float> vbuf(dim);
        for (size_t i = 0; i < actual_train; ++i) {
            reader.read_vector(train_ids[i], vbuf.data(), dim * sizeof(float));
            std::memcpy(train_data.data() + i * dim, vbuf.data(), dim * sizeof(float));
        }
    }
    std::fprintf(stderr, "  Loaded %zu training vectors\n", actual_train);

    // --- 2. Compute global centroid and subtract ---
    std::vector<float> centroid(dim, 0.0f);
    for (size_t i = 0; i < actual_train; ++i) {
        const float* v = train_data.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) centroid[d] += v[d];
    }
    for (size_t d = 0; d < dim; ++d) centroid[d] /= (float)actual_train;

    for (size_t i = 0; i < actual_train; ++i) {
        float* v = train_data.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) v[d] -= centroid[d];
    }

    // --- 3. Train each chunk with k-means ---
    // full_pivot_data: [kNumCenters][dim] row-major (Fresh/Odin shared layout)
    std::vector<float> full_pivot_data((size_t)kNumCenters * dim, 0.0f);

    std::atomic<size_t> done_counter{0};
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
        size_t off = chunk * chunk_dim;

        // Per-chunk deterministic RNG.
        std::mt19937_64 rng2(0xCAFEBABE ^ (0x9E3779B97F4A7C15ULL * chunk));

        std::vector<float> chunk_train(actual_train * chunk_dim);
        for (size_t i = 0; i < actual_train; ++i)
            std::memcpy(chunk_train.data() + i * chunk_dim,
                        train_data.data() + i * dim + off,
                        chunk_dim * sizeof(float));

        std::vector<float> centroids((size_t)kNumCenters * chunk_dim);
        for (int c = 0; c < kNumCenters; ++c) {
            uint64_t idx = rng2() % actual_train;
            std::memcpy(centroids.data() + c * chunk_dim,
                        chunk_train.data() + idx * chunk_dim,
                        chunk_dim * sizeof(float));
        }

        kmeans_lloyd(chunk_train.data(), actual_train, chunk_dim,
                     centroids.data(), kKMeansIter);

        size_t d = done_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        if (d % 8 == 0)
            std::fprintf(stderr, "  k-means chunks done %zu/%zu\n", d, n_chunks);

        for (int c = 0; c < kNumCenters; ++c)
            std::memcpy(full_pivot_data.data() + c * dim + off,
                        centroids.data() + c * chunk_dim,
                        chunk_dim * sizeof(float));
    }
    std::fprintf(stderr, "  k-means done\n");

    // --- 4. Rearrangement (identity) and chunk_offsets ---
    std::vector<uint32_t> rearrangement(dim);
    for (size_t d = 0; d < dim; ++d) rearrangement[d] = (uint32_t)d;

    std::vector<uint32_t> chunk_offsets(n_chunks + 1);
    for (size_t c = 0; c <= n_chunks; ++c)
        chunk_offsets[c] = (uint32_t)(c * chunk_dim);

    // --- 5. Write _pq_pivots.bin ---
    std::string pivots_path = out_prefix + "_pq_pivots.bin";
    int pfd = ::open(pivots_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (pfd < 0) { std::perror(pivots_path.c_str()); return false; }

    {
        std::vector<char> zeros(kMetaSector, 0);
        xpwrite_all(pfd, zeros.data(), kMetaSector, 0);
    }

    std::vector<uint64_t> cumul(5, 0);
    cumul[0] = kMetaSector;
    cumul[1] = cumul[0] + save_bin_f32(pfd, full_pivot_data.data(),
                                        (size_t)kNumCenters, dim,
                                        (off_t)cumul[0]);
    cumul[2] = cumul[1] + save_bin_f32(pfd, centroid.data(),
                                        dim, 1, (off_t)cumul[1]);
    cumul[3] = cumul[2] + save_bin_u32(pfd, rearrangement.data(),
                                        dim, 1, (off_t)cumul[2]);
    cumul[4] = cumul[3] + save_bin_u32(pfd, chunk_offsets.data(),
                                        n_chunks + 1, 1, (off_t)cumul[3]);

    save_bin_u64(pfd, cumul.data(), 5, 1, 0);
    ::close(pfd);
    std::fprintf(stderr, "  pq_pivots: %s (%zu bytes)\n",
                 pivots_path.c_str(), (size_t)cumul[4]);

    // --- 6. Encode all vectors and write _pq_compressed.bin ---
    std::string comp_path = out_prefix + "_pq_compressed.bin";
    int cfd = ::open(comp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (cfd < 0) { std::perror(comp_path.c_str()); return false; }

    {
        int32_t n32 = (int32_t)npts, d32 = (int32_t)n_chunks;
        xpwrite_all(cfd, &n32, 4, 0);
        xpwrite_all(cfd, &d32, 4, 4);
    }

    // Per-vector PQ encode is embarrassingly parallel; OpenMP each batch so the
    // 50M × 768D × 128-byte pass runs in ~10 min instead of ~5 h single-thread.
    const size_t kBlock = 4096;
    std::vector<uint8_t> codes_block(kBlock * n_chunks);
    off_t comp_off = 8;

    for (uint64_t base = 0; base < npts; base += kBlock) {
        size_t batch = (size_t)std::min((uint64_t)kBlock, npts - base);
        #pragma omp parallel
        {
            std::vector<float> vbuf(dim);
            #pragma omp for schedule(dynamic, 64)
            for (size_t i = 0; i < batch; ++i) {
                reader.read_vector(base + i, vbuf.data(), dim * sizeof(float));
                for (size_t d = 0; d < dim; ++d) vbuf[d] -= centroid[d];
                for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
                    size_t coff = chunk * chunk_dim;
                    const float* v_chunk = vbuf.data() + coff;
                    float best = std::numeric_limits<float>::max();
                    uint8_t best_c = 0;
                    for (int c = 0; c < kNumCenters; ++c) {
                        const float* cv = full_pivot_data.data() + c * dim + coff;
                        float d = 0.0f;
                        for (size_t k = 0; k < chunk_dim; ++k) {
                            float diff = v_chunk[k] - cv[k];
                            d += diff * diff;
                        }
                        if (d < best) { best = d; best_c = (uint8_t)c; }
                    }
                    codes_block[i * n_chunks + chunk] = best_c;
                }
            }
        }
        xpwrite_all(cfd, codes_block.data(), batch * n_chunks, comp_off);
        comp_off += (off_t)(batch * n_chunks);
        if ((base & 0xFFFFF) == 0)
            std::fprintf(stderr, "  encoded %lu / %lu\n",
                         (unsigned long)base, (unsigned long)npts);
    }
    ::close(cfd);
    std::fprintf(stderr, "  pq_compressed: %s\n", comp_path.c_str());

    // Return centroid and pivot data for caller (not needed currently, but kept)
    out_centroid_ret   = centroid;
    out_pivot_data_ret = full_pivot_data;
    return true;
}

// ---------------------------------------------------------------------------
// write_mem_index_stub — writes three files for an empty Vamana mem-index.
//
// Format verified against FreshDiskANN/src/index.cpp (non-single-file mode):
//   save_graph(): [u64 index_size=24][u32 max_deg=0][u32 ep=0][u64 frozen_num=1]
//   save_data(): save_bin<float>[1][dim] — one frozen-point zero-vector
//   save_tags(): save_bin<uint32>[1][1]  — one frozen-point tag = 0
//
// Fresh constructs Index with frozen_num=1 for the dynamic (insertable) case.
// npts=0 means no actual user points; just the frozen navigating point.
// ---------------------------------------------------------------------------
static bool write_mem_index_stub(const std::string& prefix, uint64_t dim) {
    // --- Graph file (prefix, no extension) ---
    {
        int fd = ::open(prefix.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { std::perror(prefix.c_str()); return false; }

        uint64_t index_size  = 24;      // 8B size + 4B max_deg + 4B ep + 8B frozen_num
        uint32_t max_degree  = 0;
        uint32_t ep          = 0;
        uint64_t frozen_num  = 1;       // dynamic index has 1 frozen navigating point

        xwrite_all(fd, &index_size, sizeof(uint64_t));
        xwrite_all(fd, &max_degree, sizeof(uint32_t));
        xwrite_all(fd, &ep,         sizeof(uint32_t));
        xwrite_all(fd, &frozen_num, sizeof(uint64_t));
        // Note: save_graph() goes back and overwrites the first u64 with the
        // actual size. Since there are no actual nodes, size = 24. The
        // max_degree field is also overwritten: still 0 for empty graph.
        ::close(fd);
    }

    // --- .data file: save_bin<float>[1][dim] — frozen-point zero-vector ---
    {
        std::string data_path = prefix + ".data";
        int fd = ::open(data_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { std::perror(data_path.c_str()); return false; }

        int32_t n32 = 1, d32 = (int32_t)dim;
        xwrite_all(fd, &n32, 4);
        xwrite_all(fd, &d32, 4);
        std::vector<float> zeros(dim, 0.0f);
        xwrite_all(fd, zeros.data(), dim * sizeof(float));
        ::close(fd);
    }

    // --- .tags file: save_bin<uint32>[1][1] — frozen-point tag = 0 ---
    {
        std::string tags_path = prefix + ".tags";
        int fd = ::open(tags_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { std::perror(tags_path.c_str()); return false; }

        int32_t  n32 = 1, d32 = 1;
        uint32_t tag = 0;
        xwrite_all(fd, &n32, 4);
        xwrite_all(fd, &d32, 4);
        xwrite_all(fd, &tag, sizeof(uint32_t));
        ::close(fd);
    }

    std::fprintf(stderr, "  mem-index stub: %s (.data .tags)\n", prefix.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Args {
    std::string navis_prefix;
    std::string out_prefix;
    std::string base_path;
    int    R           = 96;
    int    pq_bytes    = 0;
    int    num_train   = 256000;
    int    sample_n    = 200;
    bool   skip_validate = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (k == "--navis-prefix")  a.navis_prefix = next();
        else if (k == "--out-prefix")    a.out_prefix   = next();
        else if (k == "--base")          a.base_path    = next();
        else if (k == "--R")             a.R            = std::stoi(next());
        else if (k == "--pq-bytes")      a.pq_bytes     = std::stoi(next());
        else if (k == "--num-train")     a.num_train    = std::stoi(next());
        else if (k == "--sample-n")      a.sample_n     = std::stoi(next());
        else if (k == "--skip-validate") a.skip_validate = true;
        else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.navis_prefix.empty() || a.out_prefix.empty() ||
        a.base_path.empty()   || a.pq_bytes == 0) {
        std::fprintf(stderr,
            "usage: navis_to_freshdiskann --navis-prefix P --out-prefix Q "
            "--base B.bin --pq-bytes N "
            "[--R 96] [--num-train 256000] "
            "[--sample-n 200] [--skip-validate]\n");
        std::exit(2);
    }
    return a;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::fprintf(stderr,
        "navis_to_freshdiskann: %s -> %s (R=%d, PQ=%dB)\n",
        a.navis_prefix.c_str(), a.out_prefix.c_str(), a.R, a.pq_bytes);

    // --- Open NAVIS reader ---
    NavisLayoutReader reader(a.navis_prefix);
    const auto& m = reader.meta();
    std::fprintf(stderr, "  NAVIS: npts=%lu dim=%lu medoid=%lu\n",
                 (unsigned long)m.npts, (unsigned long)m.dim,
                 (unsigned long)m.medoid);

    // --- Compute DiskANN layout parameters ---
    // max_node_len = (R+1)*4 + dim*4  (matches Fresh's ((width+1)*4 + ndims*4))
    uint64_t max_node_len = ((uint64_t)a.R + 1) * sizeof(uint32_t)
                           + m.dim * sizeof(float);
    uint64_t nnodes_per_sector = kSector / max_node_len;
    if (nnodes_per_sector == 0) {
        std::fprintf(stderr,
            "ERROR: node too large for 4KB sector (max_node_len=%lu)\n",
            (unsigned long)max_node_len);
        return 1;
    }
    uint64_t n_data_sectors =
        (m.npts + nnodes_per_sector - 1) / nnodes_per_sector;
    uint64_t disk_file_size = (1 + n_data_sectors) * kSector;

    std::fprintf(stderr,
        "  Fresh disk.index: max_node_len=%lu nnodes_per_sector=%lu "
        "n_data_sectors=%lu disk_file_size=%lu\n",
        (unsigned long)max_node_len, (unsigned long)nnodes_per_sector,
        (unsigned long)n_data_sectors, (unsigned long)disk_file_size);

    // --- Open output disk.index ---
    std::string out_disk = a.out_prefix + "_disk.index";
    int ofd = ::open(out_disk.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { std::perror(out_disk.c_str()); return 1; }

    // Write sector 0 placeholder (filled after tags are written)
    {
        std::vector<char> sec0(kSector, 0);
        xwrite_all(ofd, sec0.data(), sec0.size());
    }

    // --- Write packed node sectors ---
    // Layout per node: [float*dim][uint32 degree][uint32*R neighbors]
    std::vector<char>     sector_buf(kSector, 0);
    std::vector<float>    vec_buf(m.dim);
    std::vector<uint32_t> nbr_buf((size_t)a.R, 0);
    size_t in_sector = 0;

    for (uint64_t id = 0; id < m.npts; ++id) {
        reader.read_vector(id, vec_buf.data(), m.dim * sizeof(float));
        uint32_t deg = 0;
        std::fill(nbr_buf.begin(), nbr_buf.end(), 0);
        reader.read_edges(id, nbr_buf.data(), &deg);
        if (deg > (uint32_t)a.R) deg = (uint32_t)a.R;

        char* node = sector_buf.data() + in_sector * max_node_len;
        std::memcpy(node, vec_buf.data(), m.dim * sizeof(float));
        std::memcpy(node + m.dim * sizeof(float), &deg, sizeof(uint32_t));
        std::memcpy(node + m.dim * sizeof(float) + sizeof(uint32_t),
                    nbr_buf.data(), (size_t)a.R * sizeof(uint32_t));
        ++in_sector;

        if (in_sector == nnodes_per_sector || id + 1 == m.npts) {
            xwrite_all(ofd, sector_buf.data(), sector_buf.size());
            std::memset(sector_buf.data(), 0, sector_buf.size());
            in_sector = 0;
        }

        if ((id & 0xFFFFF) == 0)
            std::fprintf(stderr, "  packed %lu / %lu\n",
                         (unsigned long)id, (unsigned long)m.npts);
    }
    ::fsync(ofd);

    // Verify file size
    {
        struct stat st{};
        ::fstat(ofd, &st);
        if ((uint64_t)st.st_size != disk_file_size) {
            std::fprintf(stderr,
                "ERROR: disk.index size mismatch: %ld vs expected %lu\n",
                (long)st.st_size, (unsigned long)disk_file_size);
            ::close(ofd);
            return 1;
        }
    }
    ::close(ofd);
    std::fprintf(stderr, "  disk.index written: %s\n", out_disk.c_str());

    // --- Write identity tags ---
    std::string tags_path = out_disk + ".tags";
    size_t tag_bytes_written = 0;
    {
        int tfd = ::open(tags_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (tfd < 0) { std::perror(tags_path.c_str()); return 1; }
        int32_t n32 = (int32_t)m.npts, d32 = 1;
        xwrite_all(tfd, &n32, 4);
        xwrite_all(tfd, &d32, 4);
        tag_bytes_written = 8;
        const size_t kTagBlock = 65536;
        std::vector<uint32_t> tag_buf(kTagBlock);
        for (uint64_t base = 0; base < m.npts; base += kTagBlock) {
            size_t batch = (size_t)std::min((uint64_t)kTagBlock, m.npts - base);
            for (size_t i = 0; i < batch; ++i)
                tag_buf[i] = (uint32_t)(base + i);
            xwrite_all(tfd, tag_buf.data(), batch * sizeof(uint32_t));
            tag_bytes_written += batch * sizeof(uint32_t);
        }
        ::fsync(tfd);
        ::close(tfd);
        std::fprintf(stderr, "  tags: %s (%zu bytes)\n",
                     tags_path.c_str(), tag_bytes_written);
    }

    // --- Write sector-0 header (Fresh-specific) ---
    // Verified against create_disk_layout() lines 742-884 in
    // FreshDiskANN/src/aux_utils.cpp.  output_file_meta accumulates 8 fields,
    // then pushes a 9th (tags_end) after tags are written:
    //   [npts, ndims, medoid, max_node_len, nnodes_per_sector,
    //    frozen_num, frozen_loc, file_size, tags_end_offset]
    // Written as save_bin<u64>[9][1] at offset 0.
    {
        uint64_t tags_end = disk_file_size + (uint64_t)tag_bytes_written;
        uint64_t fields[9] = {
            m.npts,            // npts
            m.dim,             // ndims
            m.medoid,          // medoid
            max_node_len,      // max_node_len
            nnodes_per_sector, // nnodes_per_sector
            0ULL,              // frozen_num  (0 = static disk index)
            0ULL,              // frozen_loc  (0)
            disk_file_size,    // file_size
            tags_end           // tags_end_offset
        };
        std::vector<char> sec0(kSector, 0);
        int32_t n32 = 9, d32 = 1;
        std::memcpy(sec0.data() + 0, &n32, 4);
        std::memcpy(sec0.data() + 4, &d32, 4);
        std::memcpy(sec0.data() + 8, fields, sizeof(fields));

        int hfd = ::open(out_disk.c_str(), O_WRONLY, 0644);
        if (hfd < 0) { std::perror(out_disk.c_str()); return 1; }
        xpwrite_all(hfd, sec0.data(), sec0.size(), 0);
        ::fsync(hfd);
        ::close(hfd);
        std::fprintf(stderr, "  sector-0 header written (9 fields)\n");
    }

    // --- PQ retrain ---
    std::vector<float> centroid_ret, pivot_ret;
    if (!retrain_and_write_fresh_pq(a.out_prefix, m.npts, m.dim,
                                    a.pq_bytes, a.num_train,
                                    reader, centroid_ret, pivot_ret)) {
        std::fprintf(stderr, "ERROR: PQ retrain failed\n");
        return 1;
    }

    // --- Validation (sample_n random nodes) ---
    if (!a.skip_validate) {
        std::fprintf(stderr, "Validating %d random nodes...\n", a.sample_n);
        std::mt19937_64 rng(0xBEEF1234ULL);
        size_t ok_edges = 0, ok_vecs = 0;
        std::vector<uint64_t> ids((size_t)a.sample_n);
        for (int i = 0; i < a.sample_n; ++i)
            ids[i] = rng() % m.npts;

        int vfd = ::open(out_disk.c_str(), O_RDONLY);
        if (vfd < 0) { std::perror(out_disk.c_str()); return 1; }

        std::vector<float>    vbuf_check(m.dim);
        std::vector<uint32_t> nbuf_check((size_t)a.R);
        std::vector<float>    ref_vec(m.dim);
        std::vector<uint32_t> ref_nbr((size_t)a.R, 0);

        for (uint64_t id : ids) {
            uint64_t sec_idx = id / nnodes_per_sector;
            uint64_t slot    = id % nnodes_per_sector;
            off_t off = (off_t)(kSector
                                + sec_idx * kSector
                                + slot    * max_node_len);

            xpread_all(vfd, vbuf_check.data(), m.dim * sizeof(float), off);
            uint32_t nnbr = 0;
            xpread_all(vfd, &nnbr, sizeof(uint32_t),
                       off + (off_t)(m.dim * sizeof(float)));
            if (nnbr > (uint32_t)a.R) nnbr = (uint32_t)a.R;
            xpread_all(vfd, nbuf_check.data(), nnbr * sizeof(uint32_t),
                       off + (off_t)(m.dim * sizeof(float) + sizeof(uint32_t)));

            reader.read_vector(id, ref_vec.data(), m.dim * sizeof(float));
            bool vec_match = (std::memcmp(vbuf_check.data(), ref_vec.data(),
                                          m.dim * sizeof(float)) == 0);

            uint32_t ref_deg = 0;
            std::fill(ref_nbr.begin(), ref_nbr.end(), 0);
            reader.read_edges(id, ref_nbr.data(), &ref_deg);
            if (ref_deg > (uint32_t)a.R) ref_deg = (uint32_t)a.R;

            bool deg_match = (nnbr == ref_deg);
            bool nbr_match = deg_match && (std::memcmp(nbuf_check.data(),
                                                        ref_nbr.data(),
                                                        ref_deg * sizeof(uint32_t)) == 0);
            if (vec_match)               ok_vecs++;
            if (deg_match && nbr_match)  ok_edges++;
        }
        ::close(vfd);
        std::fprintf(stderr,
            "  vectors match: %zu/%d   edges match: %zu/%d\n",
            ok_vecs, a.sample_n, ok_edges, a.sample_n);

        if (ok_vecs  != (size_t)a.sample_n ||
            ok_edges != (size_t)a.sample_n) {
            std::fprintf(stderr, "VALIDATION FAILED\n");
            return 1;
        }
        std::fprintf(stderr, "  VALIDATION PASSED\n");
    }

    // --- Write empty Vamana mem-index stubs (_mem_0, _mem_1) ---
    // Fresh's MergeInsert uses {mem_prefix}_0 and {mem_prefix}_1 where
    // mem_prefix is passed as out_prefix + "_mem".
    // verified from merge_insert.cpp save() and Index::save():
    //   _mem_index_{0,1}->save((_mem_index_prefix + "_N").c_str())
    {
        std::string mem_pfx_0 = a.out_prefix + "_mem_0";
        std::string mem_pfx_1 = a.out_prefix + "_mem_1";
        if (!write_mem_index_stub(mem_pfx_0, m.dim)) return 1;
        if (!write_mem_index_stub(mem_pfx_1, m.dim)) return 1;
    }

    // The _shadow1_* and _merge_* working copies are made by run_insert.sh
    // before every run, so each run starts from this pristine state.

    std::fprintf(stderr, "navis_to_freshdiskann OK\n");
    return 0;
}
