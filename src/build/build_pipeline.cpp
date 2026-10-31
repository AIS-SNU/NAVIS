#include "navis/build/build.h"
#include "navis/util/file_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sys/stat.h>

namespace navis {
namespace build {

static constexpr double PQ_TRAINING_FRACTION = 0.1;
static constexpr uint32_t MAX_PQ_TRAINING_SIZE = 256000;
static constexpr uint32_t NUM_KMEANS_ITERS = 15;

// ===========================================================================
// Post-build integrity checks for all 5 stages
// ===========================================================================

static uint64_t get_file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_size;
}

static bool validate_pq_pivots(const std::string& path, uint32_t dim, uint32_t num_chunks) {
    std::printf("  [check] PQ pivots: %s\n", path.c_str());
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { std::fprintf(stderr, "    FAIL: cannot open\n"); return false; }

    uint32_t off_n, off_d;
    in.read(reinterpret_cast<char*>(&off_n), 4);
    in.read(reinterpret_cast<char*>(&off_d), 4);
    if (off_n != 4 || off_d != 1) {
        std::fprintf(stderr, "    FAIL: header %u x %u (expected 4 x 1)\n", off_n, off_d);
        return false;
    }

    std::vector<uint64_t> offsets(4);
    in.read(reinterpret_cast<char*>(offsets.data()), 4 * sizeof(uint64_t));

    // Check centroids section
    in.seekg(offsets[0]);
    uint32_t c_n, c_dim;
    in.read(reinterpret_cast<char*>(&c_n), 4);
    in.read(reinterpret_cast<char*>(&c_dim), 4);
    if (c_n != NUM_PQ_CENTROIDS || c_dim != dim) {
        std::fprintf(stderr, "    FAIL: centroids %u x %u (expected %u x %u)\n",
                     c_n, c_dim, (uint32_t)NUM_PQ_CENTROIDS, dim);
        return false;
    }

    // Check rearrangement is valid permutation
    in.seekg(offsets[2]);
    uint32_t r_n, r_d;
    in.read(reinterpret_cast<char*>(&r_n), 4);
    in.read(reinterpret_cast<char*>(&r_d), 4);
    if (r_n != dim) {
        std::fprintf(stderr, "    FAIL: rearrangement size %u (expected %u)\n", r_n, dim);
        return false;
    }
    std::vector<uint32_t> rearr(dim);
    in.read(reinterpret_cast<char*>(rearr.data()), dim * 4);
    std::set<uint32_t> seen(rearr.begin(), rearr.end());
    if (seen.size() != dim || *seen.begin() != 0 || *seen.rbegin() != dim - 1) {
        std::fprintf(stderr, "    FAIL: rearrangement is not a valid permutation\n");
        return false;
    }

    // Check chunk offsets are monotonic, cover [0, dim]
    in.seekg(offsets[3]);
    uint32_t co_n, co_d;
    in.read(reinterpret_cast<char*>(&co_n), 4);
    in.read(reinterpret_cast<char*>(&co_d), 4);
    if (co_n != num_chunks + 1) {
        std::fprintf(stderr, "    FAIL: chunk offsets count %u (expected %u)\n", co_n, num_chunks + 1);
        return false;
    }
    std::vector<uint32_t> co(co_n);
    in.read(reinterpret_cast<char*>(co.data()), co_n * 4);
    if (co[0] != 0 || co[num_chunks] != dim) {
        std::fprintf(stderr, "    FAIL: chunk offsets range [%u, %u] (expected [0, %u])\n",
                     co[0], co[num_chunks], dim);
        return false;
    }
    for (uint32_t i = 1; i <= num_chunks; i++) {
        if (co[i] <= co[i - 1]) {
            std::fprintf(stderr, "    FAIL: chunk offsets not monotonic at %u\n", i);
            return false;
        }
    }

    std::printf("    OK (256 centroids x %u dim, %u chunks)\n", dim, num_chunks);
    return true;
}

static bool validate_pq_data(const std::string& path, uint32_t npts, uint32_t num_chunks) {
    std::printf("  [check] PQ compressed: %s\n", path.c_str());
    uint64_t expected = 8 + static_cast<uint64_t>(npts) * num_chunks;
    uint64_t actual = get_file_size(path);
    if (actual != expected) {
        std::fprintf(stderr, "    FAIL: size %lu (expected %lu)\n", actual, expected);
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    uint32_t h_npts, h_chunks;
    in.read(reinterpret_cast<char*>(&h_npts), 4);
    in.read(reinterpret_cast<char*>(&h_chunks), 4);
    if (h_npts != npts || h_chunks != num_chunks) {
        std::fprintf(stderr, "    FAIL: header %u x %u (expected %u x %u)\n",
                     h_npts, h_chunks, npts, num_chunks);
        return false;
    }
    std::printf("    OK (%u pts x %u chunks, %lu bytes)\n", npts, num_chunks, actual);
    return true;
}

static bool validate_graph(const std::string& path, uint32_t npts, uint32_t R,
                            const char* label, bool exact_R = false) {
    std::printf("  [check] %s: %s\n", label, path.c_str());
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { std::fprintf(stderr, "    FAIL: cannot open\n"); return false; }

    uint64_t index_size;
    uint32_t max_deg, medoid;
    uint64_t frozen;
    in.read(reinterpret_cast<char*>(&index_size), 8);
    in.read(reinterpret_cast<char*>(&max_deg), 4);
    in.read(reinterpret_cast<char*>(&medoid), 4);
    in.read(reinterpret_cast<char*>(&frozen), 8);

    if (max_deg > R) {
        std::fprintf(stderr, "    FAIL: max_degree %u > R=%u\n", max_deg, R);
        return false;
    }
    if (exact_R && max_deg != R && npts > R) {
        std::fprintf(stderr, "    FAIL: max_degree %u != R=%u (parameter mismatch)\n", max_deg, R);
        return false;
    }
    if (medoid >= npts) {
        std::fprintf(stderr, "    FAIL: medoid %u >= npts %u\n", medoid, npts);
        return false;
    }

    uint32_t errors = 0;
    for (uint32_t v = 0; v < npts; v++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        if (deg > max_deg) {
            if (errors < 3) std::fprintf(stderr, "    FAIL: vertex %u degree %u > max %u\n", v, deg, max_deg);
            errors++;
        }
        std::vector<uint32_t> nbrs(deg);
        in.read(reinterpret_cast<char*>(nbrs.data()), deg * 4);
        for (uint32_t j = 0; j < deg; j++) {
            if (nbrs[j] >= npts) {
                if (errors < 3) std::fprintf(stderr, "    FAIL: vertex %u neighbor %u out of range\n", v, nbrs[j]);
                errors++;
            }
            if (nbrs[j] == v) {
                if (errors < 3) std::fprintf(stderr, "    FAIL: vertex %u self-loop\n", v);
                errors++;
            }
        }
    }
    if (errors > 0) {
        std::fprintf(stderr, "    FAIL: %u errors found\n", errors);
        return false;
    }
    std::printf("    OK (%u vertices, max_degree=%u, medoid=%u)\n", npts, max_deg, medoid);
    return true;
}

static bool validate_separated_layout(const std::string& prefix, uint32_t npts,
                                      uint32_t ndims, uint32_t max_degree) {
    std::string edge_path  = prefix + "_disk.edge";
    std::string vec_path   = prefix + "_disk.vec";
    std::string indir_path = prefix + "_indirection.bin";

    std::printf("  [check] Separated layout: %s_disk.{edge,vec}\n", prefix.c_str());

    auto check_meta = [&](const std::string& path, uint64_t expect_kind) -> bool {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) { std::fprintf(stderr, "    FAIL: cannot open %s\n", path.c_str()); return false; }
        int32_t meta_count, meta_dim;
        in.read(reinterpret_cast<char*>(&meta_count), 4);
        in.read(reinterpret_cast<char*>(&meta_dim), 4);
        if (meta_count != 9 || meta_dim != 1) {
            std::fprintf(stderr, "    FAIL: %s metadata header %d x %d\n",
                         path.c_str(), meta_count, meta_dim);
            return false;
        }
        uint64_t meta[9];
        in.read(reinterpret_cast<char*>(meta), 9 * sizeof(uint64_t));
        if (meta[0] != npts || meta[1] != ndims) {
            std::fprintf(stderr, "    FAIL: %s npts=%lu dim=%lu (expected %u x %u)\n",
                         path.c_str(), meta[0], meta[1], npts, ndims);
            return false;
        }
        if (meta[6] != expect_kind) {
            std::fprintf(stderr, "    FAIL: %s file_kind=%lu (expected %lu)\n",
                         path.c_str(), meta[6], expect_kind);
            return false;
        }
        uint64_t size = get_file_size(path);
        if (size != meta[7]) {
            std::fprintf(stderr, "    FAIL: %s size %lu != metadata %lu\n",
                         path.c_str(), size, meta[7]);
            return false;
        }
        return true;
    };

    if (!check_meta(edge_path, /*edge*/ 0)) return false;
    if (!check_meta(vec_path,  /*vec*/  1)) return false;

    // Indirection file is optional — present only when Starling reorder was
    // applied at build time. If missing, the loader falls back to identity.
    if (file_exists(indir_path)) {
        std::ifstream indir(indir_path, std::ios::binary);
        uint64_t n_indir;
        indir.read(reinterpret_cast<char*>(&n_indir), sizeof(uint64_t));
        if (n_indir != npts) {
            std::fprintf(stderr, "    FAIL: indirection npts=%lu (expected %u)\n",
                         n_indir, npts);
            return false;
        }
        std::printf("    OK (%u pts, R=%u, edge %.2f MB, vec %.2f MB, Starling on)\n",
                    npts, max_degree,
                    get_file_size(edge_path) / (1024.0 * 1024.0),
                    get_file_size(vec_path)  / (1024.0 * 1024.0));
    } else {
        std::printf("    OK (%u pts, R=%u, edge %.2f MB, vec %.2f MB, identity)\n",
                    npts, max_degree,
                    get_file_size(edge_path) / (1024.0 * 1024.0),
                    get_file_size(vec_path)  / (1024.0 * 1024.0));
    }
    return true;
}

static bool validate_tags(const std::string& path, uint32_t npts) {
    std::printf("  [check] Tags: %s\n", path.c_str());
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { std::fprintf(stderr, "    FAIL: cannot open\n"); return false; }

    uint32_t n, d;
    in.read(reinterpret_cast<char*>(&n), 4);
    in.read(reinterpret_cast<char*>(&d), 4);
    if (d != 1) {
        std::fprintf(stderr, "    FAIL: tag dim %u (expected 1)\n", d);
        return false;
    }
    uint64_t expected_size = 8 + static_cast<uint64_t>(n) * 4;
    uint64_t actual_size = get_file_size(path);
    if (actual_size != expected_size) {
        std::fprintf(stderr, "    FAIL: size %lu (expected %lu)\n", actual_size, expected_size);
        return false;
    }
    // Check all tag IDs < npts
    std::vector<uint32_t> tags(n);
    in.read(reinterpret_cast<char*>(tags.data()), n * 4);
    for (uint32_t i = 0; i < n; i++) {
        if (tags[i] >= npts) {
            std::fprintf(stderr, "    FAIL: tag[%u]=%u >= npts %u\n", i, tags[i], npts);
            return false;
        }
    }
    std::printf("    OK (%u sample IDs, all < %u)\n", n, npts);
    return true;
}

// Calculate number of PQ chunks from target bytes per vector
static uint32_t calculate_pq_chunks(float pq_bytes_per_vec, uint32_t dim) {
    uint32_t chunks = static_cast<uint32_t>(pq_bytes_per_vec);
    chunks = std::max(chunks, 1u);
    chunks = std::min(chunks, dim);
    return chunks;
}

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
                      float alpha) {
    auto total_start = std::chrono::high_resolution_clock::now();

    // File paths
    std::string pq_pivots_path    = output_prefix + "_pq_pivots.bin";
    std::string pq_data_path      = output_prefix + "_pq_compressed.bin";
    std::string mem_index_path    = output_prefix + "_mem_graph";
    std::string ent_sample_prefix = output_prefix + "_ent_sample";
    std::string ent_index_path    = output_prefix + "_mem.index";

    // Get data metadata
    uint32_t npts, dim;
    get_bin_metadata(data_path, npts, dim);
    std::printf("\n=== NAVIS Index Build ===\n");
    std::printf("Data: %s (%u points x %u dim, type=%lu bytes)\n",
                data_path.c_str(), npts, dim, sizeof(T));
    std::printf("Output: %s\n", output_prefix.c_str());
    std::printf("R=%u, L=%u, alpha=%.2f, threads=%u\n\n", R, L, alpha, num_threads);

    // -----------------------------------------------------------------------
    // Stage 1: PQ pivot generation
    // -----------------------------------------------------------------------
    std::printf("--- Stage 1: PQ Pivot Generation ---\n");
    auto stage_start = std::chrono::high_resolution_clock::now();

    uint32_t num_pq_chunks = calculate_pq_chunks(pq_bytes_per_vec, dim);

    if (file_exists(pq_pivots_path)) {
        // Validate cached pivots match current parameters
        if (!validate_pq_pivots(pq_pivots_path, dim, num_pq_chunks)) {
            std::printf("Cached PQ pivots incompatible with current params, rebuilding\n");
            std::remove(pq_pivots_path.c_str());
            std::remove(pq_data_path.c_str());  // compressed data depends on pivots
        } else {
            std::printf("PQ pivots already exist (validated): %s\n", pq_pivots_path.c_str());
        }
    }
    if (!file_exists(pq_pivots_path)) {
        uint32_t training_size = static_cast<uint32_t>(
            std::min(static_cast<double>(MAX_PQ_TRAINING_SIZE),
                     PQ_TRAINING_FRACTION * npts));
        training_size = std::max(training_size, 1u);
        double p_val = static_cast<double>(training_size) / npts;

        std::vector<float> train_data;
        uint32_t train_npts, train_dim;
        gen_random_slice<T>(data_path, p_val, train_data, train_npts, train_dim);

        std::printf("Training set: %u points (sampled %.2f%%)\n",
                    train_npts, p_val * 100);

        generate_pq_pivots(train_data.data(), train_npts, dim,
                           NUM_PQ_CENTROIDS, num_pq_chunks,
                           NUM_KMEANS_ITERS, pq_pivots_path);
    }

    auto stage_end = std::chrono::high_resolution_clock::now();
    std::printf("Stage 1 done in %.2fs\n\n",
                std::chrono::duration<double>(stage_end - stage_start).count());

    // -----------------------------------------------------------------------
    // Stage 2: PQ data compression
    // -----------------------------------------------------------------------
    std::printf("--- Stage 2: PQ Data Compression ---\n");
    stage_start = std::chrono::high_resolution_clock::now();

    if (file_exists(pq_data_path)) {
        if (!validate_pq_data(pq_data_path, npts, num_pq_chunks)) {
            std::printf("Cached PQ data incompatible, rebuilding\n");
            std::remove(pq_data_path.c_str());
        } else {
            std::printf("PQ data already exists (validated): %s\n", pq_data_path.c_str());
        }
    }
    if (!file_exists(pq_data_path)) {
        generate_pq_data<T>(data_path, pq_pivots_path,
                            NUM_PQ_CENTROIDS, num_pq_chunks, pq_data_path);
    }

    stage_end = std::chrono::high_resolution_clock::now();
    std::printf("Stage 2 done in %.2fs\n\n",
                std::chrono::duration<double>(stage_end - stage_start).count());

    // -----------------------------------------------------------------------
    // Stage 3: Vamana graph construction
    //   Single-shot if data fits in mem_budget, sharded otherwise.
    // -----------------------------------------------------------------------
    std::printf("--- Stage 3: Vamana Graph Construction ---\n");
    stage_start = std::chrono::high_resolution_clock::now();

    if (file_exists(mem_index_path)) {
        if (!validate_graph(mem_index_path, npts, R, "Cached Vamana graph", true)) {
            std::printf("Cached graph incompatible with current params, rebuilding\n");
            std::remove(mem_index_path.c_str());
            // Disk layout depends on graph, remove it too
            std::remove((output_prefix + "_disk.edge").c_str());
            std::remove((output_prefix + "_disk.vec").c_str());
            std::remove((output_prefix + "_indirection.bin").c_str());
        } else {
            std::printf("Memory graph already exists (validated): %s\n", mem_index_path.c_str());
        }
    }
    if (!file_exists(mem_index_path)) {
        double ram_needed = estimate_ram_usage(npts, dim, sizeof(T), R);
        double ram_budget_bytes = mem_budget_gb * 1e9;

        if (ram_needed < ram_budget_bytes) {
            std::printf("Single-shot build (est %.1f GB, budget %.1f GB)\n",
                        ram_needed / 1e9, mem_budget_gb);
            VamanaGraph<T> vamana(dim, metric);
            vamana.build(data_path, R, L, alpha, num_threads);
            vamana.save(mem_index_path);
        } else {
            std::printf("Sharded build (est %.1f GB > budget %.1f GB)\n",
                        ram_needed / 1e9, mem_budget_gb);
            std::string shard_prefix = output_prefix + "_tempFiles";
            uint32_t num_shards = partition_with_ram_budget<T>(
                data_path, mem_budget_gb, R, dim, shard_prefix);
            build_shards<T>(shard_prefix, num_shards, dim, metric,
                            R, L, alpha, num_threads);
            merge_shards(shard_prefix, num_shards, npts, R, mem_index_path);
            cleanup_shards(shard_prefix, num_shards);
        }
    }

    stage_end = std::chrono::high_resolution_clock::now();
    std::printf("Stage 3 done in %.2fs\n\n",
                std::chrono::duration<double>(stage_end - stage_start).count());

    // -----------------------------------------------------------------------
    // Stage 3.5: Starling graph partitioning — greedy page-locality placement
    // of edge lists. Consumed by Stage 4 to permute id2edge_loc.
    // -----------------------------------------------------------------------
    std::string partition_path = output_prefix + "_partition.bin.aligned";
    {
        std::printf("--- Stage 3.5: Starling Graph Partitioning ---\n");
        stage_start = std::chrono::high_resolution_clock::now();

        // Derive edges_per_page from R + type exactly like SeparatedLayout::init.
        uint64_t max_edge_len   = sizeof(uint32_t) * (1 + R);
        uint64_t edges_per_page = SECTOR_LEN / max_edge_len;
        if (edges_per_page == 0) edges_per_page = 1;

        if (file_exists(partition_path)) {
            std::printf("Partition file already exists, reusing: %s\n",
                        partition_path.c_str());
        } else {
            create_starling_partition<T>(mem_index_path, npts, edges_per_page,
                                         output_prefix);
            // Force re-build of Stage 4 because the partition is the input
            // that determines edge-page contents.
            std::remove((output_prefix + "_disk.edge").c_str());
            std::remove((output_prefix + "_disk.vec" ).c_str());
            std::remove((output_prefix + "_indirection.bin").c_str());
        }

        stage_end = std::chrono::high_resolution_clock::now();
        std::printf("Stage 3.5 done in %.2fs\n\n",
                    std::chrono::duration<double>(stage_end - stage_start).count());
    }

    // -----------------------------------------------------------------------
    // Stage 4: Disk layout creation (NAVIS separated layout: edge/vec + indir)
    // -----------------------------------------------------------------------
    std::printf("--- Stage 4: Disk Layout Creation (NAVIS separated) ---\n");
    stage_start = std::chrono::high_resolution_clock::now();

    std::string edge_path  = output_prefix + "_disk.edge";
    std::string vec_path   = output_prefix + "_disk.vec";
    std::string indir_path = output_prefix + "_indirection.bin";
    bool have_all = file_exists(edge_path) && file_exists(vec_path) && file_exists(indir_path);

    if (have_all) {
        if (!validate_separated_layout(output_prefix, npts, dim, R)) {
            std::printf("Cached disk layout incompatible, rebuilding\n");
            std::remove(edge_path.c_str());
            std::remove(vec_path.c_str());
            std::remove(indir_path.c_str());
            have_all = false;
        } else {
            std::printf("Disk layout already exists (validated)\n");
        }
    }
    if (!have_all) {
        create_separated_layout<T>(mem_index_path, data_path, output_prefix);
    }

    stage_end = std::chrono::high_resolution_clock::now();
    std::printf("Stage 4 done in %.2fs\n\n",
                std::chrono::duration<double>(stage_end - stage_start).count());

    // -----------------------------------------------------------------------
    // Stage 5: Entrance graph (in-memory sample graph)
    // -----------------------------------------------------------------------
    std::printf("--- Stage 5: Entrance Graph ---\n");
    stage_start = std::chrono::high_resolution_clock::now();

    if (file_exists(ent_index_path) && file_exists(ent_index_path + ".tags")) {
        // Validate cached entrance graph: tags must reference valid IDs
        std::string tags_path = ent_index_path + ".tags";
        if (!validate_tags(tags_path, npts)) {
            std::printf("Cached entrance graph incompatible, rebuilding\n");
            std::remove(ent_index_path.c_str());
            std::remove((ent_index_path + ".data").c_str());
            std::remove((ent_index_path + ".tags").c_str());
        } else {
            // Also validate graph structure
            uint32_t sample_npts;
            { std::ifstream t(tags_path, std::ios::binary); t.read(reinterpret_cast<char*>(&sample_npts), 4); }
            if (!validate_graph(ent_index_path, sample_npts, ent_R, "Cached entrance graph", true)) {
                std::printf("Cached entrance graph invalid, rebuilding\n");
                std::remove(ent_index_path.c_str());
                std::remove((ent_index_path + ".data").c_str());
                std::remove((ent_index_path + ".tags").c_str());
            } else {
                std::printf("Entrance graph already exists (validated): %s\n", ent_index_path.c_str());
            }
        }
    }
    if (!file_exists(ent_index_path)) {
        // Generate random slice for entrance graph
        gen_random_slice_to_file<T>(data_path, ent_sample_prefix, sample_rate);

        // Build entrance graph on the sample
        std::string sample_data = ent_sample_prefix + "_data.bin";
        VamanaGraph<T> ent_graph(dim, metric);
        ent_graph.build(sample_data, ent_R, ent_L, alpha, num_threads);
        ent_graph.save(ent_index_path);
        ent_graph.save_data(ent_index_path + ".data");

        // Save tags (sample IDs → disk IDs mapping)
        std::string ids_file = ent_sample_prefix + "_ids.bin";
        if (file_exists(ids_file)) {
            std::ifstream src(ids_file, std::ios::binary);
            std::ofstream dst(ent_index_path + ".tags", std::ios::binary);
            dst << src.rdbuf();
        }
    }

    stage_end = std::chrono::high_resolution_clock::now();
    std::printf("Stage 5 done in %.2fs\n\n",
                std::chrono::duration<double>(stage_end - stage_start).count());

    // -----------------------------------------------------------------------
    // Integrity checks
    // -----------------------------------------------------------------------
    std::printf("--- Integrity Checks ---\n");
    stage_start = std::chrono::high_resolution_clock::now();

    bool all_ok = true;
    all_ok &= validate_pq_pivots(pq_pivots_path, dim, num_pq_chunks);
    all_ok &= validate_pq_data(pq_data_path, npts, num_pq_chunks);
    all_ok &= validate_graph(mem_index_path, npts, R, "Vamana graph");
    all_ok &= validate_separated_layout(output_prefix, npts, dim, R);

    // Entrance graph: read sample count from tags file
    std::string tags_path = ent_index_path + ".tags";
    if (file_exists(tags_path)) {
        uint32_t sample_npts;
        { std::ifstream t(tags_path, std::ios::binary); t.read(reinterpret_cast<char*>(&sample_npts), 4); }
        all_ok &= validate_graph(ent_index_path, sample_npts, ent_R, "Entrance graph");
        all_ok &= validate_tags(tags_path, npts);
    }

    stage_end = std::chrono::high_resolution_clock::now();
    if (!all_ok) {
        std::fprintf(stderr, "\n=== INTEGRITY CHECK FAILED ===\n");
        exit(1);
    }
    std::printf("All checks passed (%.2fs)\n\n",
                std::chrono::duration<double>(stage_end - stage_start).count());

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_secs = std::chrono::duration<double>(total_end - total_start).count();

    std::printf("=== Build Complete (%.2fs total) ===\n", total_secs);
    std::printf("Files:\n");
    std::printf("  Disk edges:     %s_disk.edge\n", output_prefix.c_str());
    std::printf("  Disk vectors:   %s_disk.vec\n", output_prefix.c_str());
    std::printf("  Indirection:    %s_indirection.bin\n", output_prefix.c_str());
    std::printf("  PQ pivots:      %s\n", pq_pivots_path.c_str());
    std::printf("  PQ compressed:  %s\n", pq_data_path.c_str());
    std::printf("  Entrance graph: %s\n", ent_index_path.c_str());
    std::printf("\nRun: navis_bench -i %s -q <queries.bin> --insert-data <data.bin>\n",
                output_prefix.c_str());
}

// Explicit instantiations
template void build_full_index<float>(const std::string&, const std::string&, Metric,
    uint32_t, uint32_t, float, float, uint32_t, float, uint32_t, uint32_t, float);
template void build_full_index<int8_t>(const std::string&, const std::string&, Metric,
    uint32_t, uint32_t, float, float, uint32_t, float, uint32_t, uint32_t, float);
template void build_full_index<uint8_t>(const std::string&, const std::string&, Metric,
    uint32_t, uint32_t, float, float, uint32_t, float, uint32_t, uint32_t, float);

} // namespace build
} // namespace navis
