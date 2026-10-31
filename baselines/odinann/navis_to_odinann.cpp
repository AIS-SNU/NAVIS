// baselines/odinann/navis_to_odinann.cpp
// NAVIS index -> OdinANN index (DiskANN packed layout).
//
// Produces:
//   {out_prefix}_disk.index        — sector-0 Odin header + packed node sectors
//   {out_prefix}_disk.index.tags   — identity uint32 tags (bin format)
//   {out_prefix}_pq_pivots.bin     — retrained PQ centroids (contiguous chunks)
//   {out_prefix}_pq_compressed.bin — PQ-encoded vectors
//
// Header format: matches OdinANN/src/utils/aux_utils.cpp
// create_disk_layout(). Sector 0 layout:
//   [int32 n_fields=9][int32 d=1]        <- save_bin preamble
//   uint64 npts
//   uint64 ndims
//   uint64 medoid
//   uint64 max_node_len
//   uint64 nnodes_per_sector
//   uint64 frozen_num  (0 for static index)
//   uint64 frozen_loc  (0)
//   uint64 file_size
//   uint64 tags_end_offset
//   zero-padded to 4096 bytes
//
// Node layout (per create_disk_layout):
//   [vec: float*dim][nnbr: uint32][nbrs: uint32*R]  (zero-padded to max_node_len)
//
// PQ pivots format: verified against partition_and_pq.cpp generate_pq_pivots().
//   Offset 0: save_bin<uint64>[n=5][d=1] + 5 cumul_byte offsets
//   cumul_bytes[0]=4096: save_bin<float>[256][dim]   — centroids table
//   cumul_bytes[1]:      save_bin<float>[dim][1]     — global centroid
//   cumul_bytes[2]:      save_bin<uint32][dim][1]    — rearrangement (identity)
//   cumul_bytes[3]:      save_bin<uint32>[n_chunks+1][1] — chunk_offsets
//
// PQ compressed: save_bin<uint8>[npts][n_chunks]

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
// save_bin helpers (matches Odin's pipeann::save_bin template)
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
    size_t total = 8 + npts * ndims * sizeof(uint8_t);
    // Write preamble separately (may be large dataset)
    xpwrite_all(fd, &n32, 4, off);
    xpwrite_all(fd, &d32, 4, off + 4);
    xpwrite_all(fd, data, npts * ndims, off + 8);
    return total;
}

// ---------------------------------------------------------------------------
// Minimal k-means PQ trainer (contiguous chunks, no rearrangement)
// Matches the subset of generate_pq_pivots() needed for Odin compatibility.
// ---------------------------------------------------------------------------

// Lloyd's iteration on a single chunk of dimension chunk_dim.
// train_data: [num_train * chunk_dim] floats
// centroids:  [kNumCenters * chunk_dim] floats (in/out)
static void kmeans_lloyd(const float* train_data, size_t num_train,
                         size_t chunk_dim,
                         float* centroids, int n_iters) {
    std::vector<int> assign(num_train);
    std::vector<float> new_centroids(kNumCenters * chunk_dim);
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
            // else: keep old centroid (empty cluster)
        }
    }
}

// Encode a single vector chunk into a PQ code (nearest centroid index).
static uint8_t pq_encode_chunk(const float* v, const float* centroids, size_t chunk_dim) {
    float best = std::numeric_limits<float>::max();
    uint8_t best_c = 0;
    for (int c = 0; c < kNumCenters; ++c) {
        const float* cv = centroids + c * chunk_dim;
        float d = 0.0f;
        for (size_t k = 0; k < chunk_dim; ++k) {
            float diff = v[k] - cv[k];
            d += diff * diff;
        }
        if (d < best) { best = d; best_c = (uint8_t)c; }
    }
    return best_c;
}

// ---------------------------------------------------------------------------
// retrain_and_write_pq_contiguous
//
// Reads up to num_train vectors from base_path (float32 bin format),
// trains a PQ table with pq_bytes chunks (contiguous, no rearrangement),
// writes {out_prefix}_pq_pivots.bin and {out_prefix}_pq_compressed.bin.
// ---------------------------------------------------------------------------
static bool retrain_and_write_pq_contiguous(
        const std::string& base_path,
        const std::string& out_prefix,
        uint64_t npts, uint64_t dim,
        int pq_bytes, int num_train,
        NavisLayoutReader& reader) {

    if (dim % (uint64_t)pq_bytes != 0) {
        std::fprintf(stderr,
            "ERROR: dim=%lu is not divisible by pq_bytes=%d\n",
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
        // Reservoir sample without replacement
        for (size_t i = 0; i < actual_train; ++i) train_ids[i] = i;
        for (uint64_t i = actual_train; i < npts; ++i) {
            uint64_t j = rng() % (i + 1);
            if (j < (uint64_t)actual_train) train_ids[j] = i;
        }
    }

    // Load training vectors
    std::vector<float> train_data(actual_train * dim);
    {
        std::vector<float> vbuf(dim);
        for (size_t i = 0; i < actual_train; ++i) {
            reader.read_vector(train_ids[i], vbuf.data(), dim * sizeof(float));
            std::memcpy(train_data.data() + i * dim, vbuf.data(), dim * sizeof(float));
        }
    }
    std::fprintf(stderr, "  Loaded %zu training vectors\n", actual_train);

    // --- 2. Compute global centroid ---
    std::vector<float> centroid(dim, 0.0f);
    for (size_t i = 0; i < actual_train; ++i) {
        const float* v = train_data.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) centroid[d] += v[d];
    }
    for (size_t d = 0; d < dim; ++d) centroid[d] /= (float)actual_train;

    // Subtract centroid from training data
    for (size_t i = 0; i < actual_train; ++i) {
        float* v = train_data.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) v[d] -= centroid[d];
    }

    // --- 3. Train each chunk with k-means ---
    // full_pivot_data: [kNumCenters][dim] row-major (Odin's layout)
    std::vector<float> full_pivot_data((size_t)kNumCenters * dim, 0.0f);

    // Chunks are independent — parallelize across them (matches how NAVIS's
    // own PQ trainer is multi-threaded). Deterministic per-chunk seeding so
    // the result is reproducible regardless of thread schedule.
    std::atomic<size_t> done_counter{0};
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
        size_t off = chunk * chunk_dim;

        // Per-chunk RNG seeded from the chunk index for determinism.
        std::mt19937_64 rng2(0xCAFEBABE ^ (0x9E3779B97F4A7C15ULL * chunk));

        // Extract chunk training data
        std::vector<float> chunk_train(actual_train * chunk_dim);
        for (size_t i = 0; i < actual_train; ++i)
            std::memcpy(chunk_train.data() + i * chunk_dim,
                        train_data.data() + i * dim + off,
                        chunk_dim * sizeof(float));

        // Initialize centroids from random training samples
        std::vector<float> centroids((size_t)kNumCenters * chunk_dim);
        for (int c = 0; c < kNumCenters; ++c) {
            uint64_t idx = rng2() % actual_train;
            std::memcpy(centroids.data() + c * chunk_dim,
                        chunk_train.data() + idx * chunk_dim,
                        chunk_dim * sizeof(float));
        }

        // Run k-means
        kmeans_lloyd(chunk_train.data(), actual_train, chunk_dim,
                     centroids.data(), kKMeansIter);

        // Copy into full_pivot_data at the correct subspace offset.
        // Odin layout: full_pivot_data[c * dim + off .. off+chunk_dim]
        for (int c = 0; c < kNumCenters; ++c)
            std::memcpy(full_pivot_data.data() + c * dim + off,
                        centroids.data() + c * chunk_dim,
                        chunk_dim * sizeof(float));

        size_t d = done_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        if (d % 8 == 0)
            std::fprintf(stderr, "  k-means chunks done %zu/%zu\n", d, n_chunks);
    }
    std::fprintf(stderr, "  k-means done\n");

    // --- 4. Build rearrangement (identity) and chunk_offsets ---
    std::vector<uint32_t> rearrangement(dim);
    for (size_t d = 0; d < dim; ++d) rearrangement[d] = (uint32_t)d;

    std::vector<uint32_t> chunk_offsets(n_chunks + 1);
    for (size_t c = 0; c <= n_chunks; ++c)
        chunk_offsets[c] = (uint32_t)(c * chunk_dim);

    // --- 5. Write _pq_pivots.bin ---
    std::string pivots_path = out_prefix + "_pq_pivots.bin";
    int pfd = ::open(pivots_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (pfd < 0) { std::perror(pivots_path.c_str()); return false; }

    // Allocate first kMetaSector bytes (will write cumul_bytes there last)
    // Pre-allocate the file to at least kMetaSector
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

    // Write cumul_bytes at offset 0
    save_bin_u64(pfd, cumul.data(), 5, 1, 0);
    ::close(pfd);
    std::fprintf(stderr, "  pq_pivots: %s (%zu bytes)\n",
                 pivots_path.c_str(), (size_t)cumul[4]);

    // --- 6. Encode all vectors and write _pq_compressed.bin ---
    std::string comp_path = out_prefix + "_pq_compressed.bin";
    int cfd = ::open(comp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (cfd < 0) { std::perror(comp_path.c_str()); return false; }

    // Write preamble (npts x n_chunks)
    {
        int32_t n32 = (int32_t)npts, d32 = (int32_t)n_chunks;
        xpwrite_all(cfd, &n32, 4, 0);
        xpwrite_all(cfd, &d32, 4, 4);
    }

    // Encode in blocks of 4096 vectors to amortize syscall overhead.
    // Within each block, vectors are processed in parallel with OpenMP — each
    // vector is independent (own vbuf, own codes row). On 50M × 768D × 128 PQ
    // bytes this reduces wallclock from ~5h (single-thread) to ~10 min (32T).
    const size_t kBlock = 4096;
    std::vector<uint8_t> codes_block(kBlock * n_chunks);
    off_t comp_off = 8;  // after preamble

    for (uint64_t base = 0; base < npts; base += kBlock) {
        size_t batch = (size_t)std::min((uint64_t)kBlock, npts - base);
        #pragma omp parallel
        {
            std::vector<float> vbuf(dim);  // per-thread scratch
            #pragma omp for schedule(dynamic, 64)
            for (size_t i = 0; i < batch; ++i) {
                reader.read_vector(base + i, vbuf.data(), dim * sizeof(float));
                // Subtract global centroid then encode
                for (size_t d = 0; d < dim; ++d) vbuf[d] -= centroid[d];
                // Encode each chunk: full_pivot_data is [kNumCenters][dim] row-major,
                // so centroid c for chunk `chunk` is at full_pivot_data[c*dim + chunk*chunk_dim].
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
    return true;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    std::string navis_prefix;
    std::string out_prefix;
    std::string base_path;
    int R           = 96;
    int pq_bytes    = 0;
    int num_train   = 256000;
    int sample_n    = 200;
    bool skip_validate = false;
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
            "usage: navis_to_odinann --navis-prefix P --out-prefix Q "
            "--base B.bin --pq-bytes N "
            "[--R 96] [--num-train 256000] [--sample-n 200] [--skip-validate]\n");
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
        "navis_to_odinann: %s -> %s (R=%d, PQ=%dB)\n",
        a.navis_prefix.c_str(), a.out_prefix.c_str(), a.R, a.pq_bytes);

    // --- Open NAVIS reader ---
    NavisLayoutReader reader(a.navis_prefix);
    const auto& m = reader.meta();
    std::fprintf(stderr, "  NAVIS: npts=%lu dim=%lu medoid=%lu\n",
                 (unsigned long)m.npts, (unsigned long)m.dim,
                 (unsigned long)m.medoid);

    // --- Compute DiskANN layout parameters ---
    // max_node_len = (R+1)*4 + dim*4  (matches Odin's ((width+1)*sizeof(unsigned) + ndims*sizeof(T)))
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
        "  DiskANN: max_node_len=%lu nnodes_per_sector=%lu "
        "n_data_sectors=%lu disk_file_size=%lu\n",
        (unsigned long)max_node_len, (unsigned long)nnodes_per_sector,
        (unsigned long)n_data_sectors, (unsigned long)disk_file_size);

    // --- Open output disk.index ---
    std::string out_disk = a.out_prefix + "_disk.index";
    int ofd = ::open(out_disk.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { std::perror(out_disk.c_str()); return 1; }

    // --- Write sector 0 placeholder (will be filled after tags are written) ---
    {
        std::vector<char> sec0(kSector, 0);
        xwrite_all(ofd, sec0.data(), sec0.size());
    }

    // --- Write packed node sectors ---
    std::vector<char>     sector_buf(kSector, 0);
    std::vector<float>    vec_buf(m.dim);
    std::vector<uint32_t> nbr_buf((size_t)a.R, 0);
    size_t in_sector = 0;

    for (uint64_t id = 0; id < m.npts; ++id) {
        reader.read_vector(id, vec_buf.data(), m.dim * sizeof(float));
        uint32_t deg = 0;
        std::fill(nbr_buf.begin(), nbr_buf.end(), 0);
        reader.read_edges(id, nbr_buf.data(), &deg);
        // Cap degree at R (shouldn't exceed, but be safe)
        if (deg > (uint32_t)a.R) deg = (uint32_t)a.R;

        char* node = sector_buf.data() + in_sector * max_node_len;
        std::memcpy(node,
                    vec_buf.data(),
                    m.dim * sizeof(float));
        std::memcpy(node + m.dim * sizeof(float),
                    &deg, sizeof(uint32_t));
        std::memcpy(node + m.dim * sizeof(float) + sizeof(uint32_t),
                    nbr_buf.data(),
                    (size_t)a.R * sizeof(uint32_t));
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
        // save_bin<uint32>[npts][1]
        int32_t n32 = (int32_t)m.npts, d32 = 1;
        xwrite_all(tfd, &n32, 4);
        xwrite_all(tfd, &d32, 4);
        tag_bytes_written = 8;
        // Write tags in blocks
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

    // --- Write sector-0 header (now that we know tag_bytes_written) ---
    // Odin format: save_bin<uint64>[9][1] at offset 0 of disk.index.
    // Fields: npts, ndims, medoid, max_node_len, nnodes_per_sector,
    //         frozen_num, frozen_loc, file_size, tags_end_offset
    {
        // tags are written to a separate file; Odin's loader reads tags from
        // disk.index.tags directly. The 9th field (tags_end_offset) in the
        // original code is used only for embedded single-file mode.
        // For multi-file mode (our case), we set it to disk_file_size +
        // tag_bytes_written (matching Odin's push_back pattern).
        uint64_t tags_end = disk_file_size + (uint64_t)tag_bytes_written;
        uint64_t fields[9] = {
            m.npts,           // npts
            m.dim,            // ndims
            m.medoid,         // medoid
            max_node_len,     // max_node_len
            nnodes_per_sector,// nnodes_per_sector
            0ULL,             // frozen_num  (0 = static index, no frozen pt)
            0ULL,             // frozen_loc  (0)
            disk_file_size,   // file_size
            tags_end          // tags_end_offset (Odin's 9th field)
        };
        std::vector<char> sec0(kSector, 0);
        int32_t n32 = 9, d32 = 1;
        std::memcpy(sec0.data() + 0, &n32, 4);
        std::memcpy(sec0.data() + 4, &d32, 4);
        std::memcpy(sec0.data() + 8, fields, sizeof(fields));

        // Reopen to write sector 0
        int hfd = ::open(out_disk.c_str(), O_WRONLY, 0644);
        if (hfd < 0) { std::perror(out_disk.c_str()); return 1; }
        xpwrite_all(hfd, sec0.data(), sec0.size(), 0);
        ::fsync(hfd);
        ::close(hfd);
        std::fprintf(stderr, "  sector-0 header written\n");
    }

    // --- PQ retrain ---
    if (!retrain_and_write_pq_contiguous(a.base_path, a.out_prefix,
                                         m.npts, m.dim,
                                         a.pq_bytes, a.num_train,
                                         reader)) {
        std::fprintf(stderr, "ERROR: PQ retrain failed\n");
        return 1;
    }

    // --- Validation (200 random nodes) ---
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

            // Read vector
            xpread_all(vfd, vbuf_check.data(), m.dim * sizeof(float), off);
            // Read degree
            uint32_t nnbr = 0;
            xpread_all(vfd, &nnbr, sizeof(uint32_t),
                       off + (off_t)(m.dim * sizeof(float)));
            if (nnbr > (uint32_t)a.R) nnbr = (uint32_t)a.R;
            // Read neighbors
            xpread_all(vfd, nbuf_check.data(), nnbr * sizeof(uint32_t),
                       off + (off_t)(m.dim * sizeof(float) + sizeof(uint32_t)));

            // Compare with NAVIS source
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

    std::fprintf(stderr, "navis_to_odinann OK: %s\n", out_disk.c_str());
    return 0;
}
