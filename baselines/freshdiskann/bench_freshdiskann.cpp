// baselines/freshdiskann/bench_freshdiskann.cpp
// Drives FreshDiskANN (MergeInsert) through the shared benchmark harness:
// batches of concurrent insertion and search, with a search pass after each
// batch (Recall@K against gt_<offset>.bin).
//
// FreshDiskANN buffers insertions in an in-memory index and merges it into
// the disk index once it holds --merge-size points. The merge runs inside the
// insertion phase, so the 5 s search passes continue during a merge.
// With --entrance-graph, searches start from the entrance graph's --entry-L
// nearest vertices instead of the medoid.
// Following OdinANN's evaluation, the process exits after each merge and
// run_insert.sh restarts it from the merged index (--start-batch, --ckpt),
// avoiding FreshDiskANN's memory corruption after a merge.

#include "bench_harness.h"
#include "disk_index_probe.h"
#include "csv_schema.h"
#include "rss_sampler.h"

// FreshDiskANN headers
#include "v2/merge_insert.h"
#include "aux_utils.h"
#include "percentile_stats.h"
#include "utils.h"
#include "distance.h"
#include "parameters.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <ctime>
#include <fstream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace bench;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
struct Args {
    std::string index_prefix;
    std::string query_path;
    std::string insert_data;
    std::string groundtruth_dir;
    std::string csv_path;
    std::string dataset;
    std::string metric = "l2";
    std::string entrance_graph;  // static entrance graph; empty: start from the medoid

    int base_n        = 0;
    int batch_size    = 0;
    int num_batches   = 0;

    int search_threads  = 22;
    int insert_threads  = 10;
    int K               = 10;
    int search_L        = 40;
    int insert_L        = 100;
    int mem_L           = 128;     // L for insertions into the in-memory index
    int entry_L         = 10;      // entry points taken from the entrance graph
    int merge_size      = 100000;  // merge threshold (MERGE_TH default = 3001000 in Fresh)
    int start_batch     = 0;       // Restart after a merge: skip batches < this.
    double elapsed_offset_s = 0.0; // CSV t_elapsed_s continues from prior chunk.
    int ckpt            = 0;       // Restart-after-merge checkpoint index.
                                   // Fresh's overall_performance.cpp pattern:
                                   //   need_copy = (ckpt == 0)
                                   //   need_swap = (ckpt % 2 != 0)
                                   // ensures the merged graph from chunk N-1 is
                                   // loaded as the live index for chunk N.
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  --index        <prefix>  FreshDiskANN index file prefix\n"
        "  --query        <path>    Query vectors (.bin)\n"
        "  --insert-data  <path>    Insert vectors (.bin, superset of index)\n"
        "  --csv          <path>    Output CSV path\n"
        "  --dataset      <name>    Dataset name recorded in the CSV\n"
        "  --base-n       <int>     Number of vectors in the base index\n"
        "  --batch-size   <int>     Vectors per batch\n"
        "  --num-batches  <int>     Number of batches\n"
        "\n"
        "Optional:\n"
        "  --groundtruth  <dir>     GT directory (gt_0.bin, gt_N.bin, ...)\n"
        "  --metric       <l2|ip>   (default: l2)\n"
        "  --search-threads <int>   (default: 22)\n"
        "  --insert-threads <int>   (default: 10)\n"
        "  -K             <int>     Top-K (default: 10)\n"
        "  --search-L     <int>     Search frontier (default: 40)\n"
        "  --insert-L     <int>     Position-seeking frontier (default: 100)\n"
        "  --mem-L        <int>     L for insertions into the in-memory index (default: 128)\n"
        "  --entrance-graph <path>  In-memory entrance graph whose tags are disk ids (e.g.\n"
        "                           NAVIS's _mem.index); searches start from its --entry-L\n"
        "                           nearest vertices (default: from the medoid)\n"
        "  --entry-L      <int>     Entry points from the entrance graph (default: 10)\n"
        "  --merge-size   <int>     Init mem index size / merge threshold (default: 100000)\n"
        "  --start-batch  <int>     Resume at batch N (restart after a merge; default: 0)\n"
        "  --elapsed-offset <float> Prior chunk's wall seconds (for continuous fair throughput)\n"
        "  --ckpt         <int>     Checkpoint index for restart-after-merge (default: 0)\n"
        "                           ckpt==0 copies pristine→shadow; ckpt>0 alternates swap\n",
        prog);
}

static Args parse(int argc, char** argv) {
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
        if      (k == "--index")          a.index_prefix   = next();
        else if (k == "--query")          a.query_path     = next();
        else if (k == "--insert-data")    a.insert_data    = next();
        else if (k == "--groundtruth")    a.groundtruth_dir = next();
        else if (k == "--csv")            a.csv_path       = next();
        else if (k == "--dataset")        a.dataset        = next();
        else if (k == "--metric")         a.metric         = next();
        else if (k == "--base-n")         a.base_n         = std::stoi(next());
        else if (k == "--batch-size")     a.batch_size     = std::stoi(next());
        else if (k == "--num-batches")    a.num_batches    = std::stoi(next());
        else if (k == "--search-threads") a.search_threads = std::stoi(next());
        else if (k == "--insert-threads") a.insert_threads = std::stoi(next());
        else if (k == "-K")               a.K              = std::stoi(next());
        else if (k == "--search-L")       a.search_L       = std::stoi(next());
        else if (k == "--insert-L")       a.insert_L       = std::stoi(next());
        else if (k == "--mem-L")          a.mem_L          = std::stoi(next());
        else if (k == "--entrance-graph") a.entrance_graph = next();
        else if (k == "--entry-L")        a.entry_L        = std::stoi(next());
        else if (k == "--merge-size")     a.merge_size     = std::stoi(next());
        else if (k == "--start-batch")    a.start_batch    = std::stoi(next());
        else if (k == "--elapsed-offset") a.elapsed_offset_s = std::stod(next());
        else if (k == "--ckpt")           a.ckpt           = std::stoi(next());
        else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            print_usage(argv[0]);
            std::exit(2);
        }
    }

    bool ok = true;
    if (a.index_prefix.empty()) { std::fprintf(stderr, "missing --index\n");        ok = false; }
    if (a.query_path.empty())   { std::fprintf(stderr, "missing --query\n");         ok = false; }
    if (a.insert_data.empty())  { std::fprintf(stderr, "missing --insert-data\n");  ok = false; }
    if (a.csv_path.empty())     { std::fprintf(stderr, "missing --csv\n");           ok = false; }
    if (a.dataset.empty())      { std::fprintf(stderr, "missing --dataset\n");       ok = false; }
    if (a.base_n <= 0)          { std::fprintf(stderr, "missing/invalid --base-n\n"); ok = false; }
    if (a.batch_size <= 0)      { std::fprintf(stderr, "missing/invalid --batch-size\n"); ok = false; }
    if (a.num_batches <= 0)     { std::fprintf(stderr, "missing/invalid --num-batches\n"); ok = false; }
    if (!a.entrance_graph.empty() && a.entry_L <= 0) { std::fprintf(stderr, "invalid --entry-L\n"); ok = false; }
    if (!ok) { print_usage(argv[0]); std::exit(2); }

    return a;
}

// ---------------------------------------------------------------------------
// load_insert_batch — read batch_size vectors of dim floats from .bin starting
// at vector index start_id (0-based global id within the full .bin file).
// Header: [int32 npts][int32 dim][float data...].
// ---------------------------------------------------------------------------
static void load_insert_batch(const std::string& path,
                               float* out,
                               size_t start_id,
                               size_t batch_size,
                               size_t dim) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "load_insert_batch: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    long offset = 8L + static_cast<long>(start_id) * static_cast<long>(dim) * sizeof(float);
    if (std::fseek(f, offset, SEEK_SET) != 0) {
        std::fprintf(stderr, "load_insert_batch: fseek failed\n");
        std::fclose(f);
        std::exit(1);
    }
    size_t n_read = std::fread(out, sizeof(float), batch_size * dim, f);
    if (n_read != batch_size * dim) {
        std::fprintf(stderr, "load_insert_batch: read %zu floats, expected %zu\n",
                     n_read, batch_size * dim);
        // Non-fatal if partial (last batch might be smaller).
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// run_search_pass — lifted and adapted from overall_performance.cpp's
// sync_search_kernel. Stripped of stdout printing; returns TickMetrics.
// ---------------------------------------------------------------------------
static TickMetrics run_search_pass(diskann::MergeInsert<float, uint32_t>& idx,
                                   float* queries,
                                   size_t query_count,
                                   size_t query_aligned_dim,
                                   int K,
                                   uint64_t search_L,
                                   const std::string& truthset_file,
                                   int n_threads,
                                   bool report_recall = true) {
    std::vector<float>    result_dists(static_cast<size_t>(K) * query_count,
                                       std::numeric_limits<float>::max());
    std::vector<uint32_t> result_tags(static_cast<size_t>(K) * query_count,
                                      std::numeric_limits<uint32_t>::max());

    std::vector<double> latency_ms(query_count, 0.0);
    std::vector<diskann::QueryStats> stats(query_count);

    auto t_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(query_count); ++i) {
        auto q0 = std::chrono::high_resolution_clock::now();
        idx.search_sync(queries + i * query_aligned_dim,
                        static_cast<uint64_t>(K),
                        search_L,
                        result_tags.data()  + i * K,
                        result_dists.data() + i * K,
                        stats.data() + i);
        auto q1 = std::chrono::high_resolution_clock::now();
        latency_ms[i] = std::chrono::duration<double, std::milli>(q1 - q0).count();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();
    double qps = static_cast<double>(query_count) / wall_s;

    std::vector<double> sorted_lats = latency_ms;
    std::sort(sorted_lats.begin(), sorted_lats.end());
    size_t n = sorted_lats.size();
    auto pct = [&](double p) -> double {
        if (n == 0) return 0.0;
        return sorted_lats[static_cast<size_t>(p * static_cast<double>(n - 1))];
    };
    double mean_lat = n > 0
        ? std::accumulate(sorted_lats.begin(), sorted_lats.end(), 0.0) / static_cast<double>(n)
        : 0.0;

    double mean_ios = diskann::get_mean_stats(stats.data(), query_count,
        [](const diskann::QueryStats& s) { return s.n_ios; });

    double recall = 0.0;
    if (!truthset_file.empty()) {
        struct stat st{};
        if (::stat(truthset_file.c_str(), &st) == 0) {
            uint32_t* gt_ids   = nullptr;
            float*    gt_dists = nullptr;
            size_t    gt_num   = 0;
            size_t    gt_dim   = 0;
            diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
            if (gt_num > 0 && gt_dim > 0) {
                recall = diskann::calculate_recall(
                    static_cast<unsigned>(query_count),
                    reinterpret_cast<unsigned*>(gt_ids),
                    gt_dists,
                    static_cast<unsigned>(gt_dim),
                    reinterpret_cast<unsigned*>(result_tags.data()),
                    static_cast<unsigned>(K),
                    static_cast<unsigned>(K)) / 100.0;
            }
            delete[] gt_ids;
            delete[] gt_dists;
        }
    }

    TickMetrics m{};
    m.qps           = qps;
    m.mean_lat_ms   = mean_lat;
    m.p50_ms        = pct(0.500);
    m.p90_ms        = pct(0.900);
    m.p95_ms        = pct(0.950);
    m.p99_ms        = pct(0.990);
    m.p999_ms       = pct(0.999);
    m.recall_at_10  = report_recall ? recall : 0.0;
    m.mean_disk_ios = mean_ios;
    m.cache_hit_rate = 0.0;  // Fresh does not expose a unified cache hit rate
    return m;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::filesystem::path csv_dir = std::filesystem::path(a.csv_path).parent_path();
    if (!csv_dir.empty()) std::filesystem::create_directories(csv_dir);

    // -----------------------------------------------------------------------
    // Peek at insert data header to get dimension before constructing index.
    // -----------------------------------------------------------------------
    size_t insert_n       = 0;
    size_t insert_dim_val = 0;
    {
        std::ifstream hdr(a.insert_data, std::ios::binary);
        if (!hdr) {
            std::fprintf(stderr, "Cannot open insert-data: %s\n", a.insert_data.c_str());
            return 1;
        }
        int32_t nn, dd;
        hdr.read(reinterpret_cast<char*>(&nn), 4);
        hdr.read(reinterpret_cast<char*>(&dd), 4);
        insert_n       = static_cast<size_t>(nn);
        insert_dim_val = static_cast<size_t>(dd);
    }
    std::fprintf(stderr, "Insert data: %zu vectors x %zu dim\n", insert_n, insert_dim_val);
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Build FreshDiskANN parameters.
    // -----------------------------------------------------------------------
    diskann::Parameters paras;
    paras.Set<unsigned>("L_mem",  static_cast<unsigned>(a.mem_L));
    paras.Set<unsigned>("R_mem",  64u);
    paras.Set<float>   ("alpha_mem", 1.2f);
    paras.Set<unsigned>("L_disk", static_cast<unsigned>(a.insert_L));
    paras.Set<unsigned>("R_disk", 96u);
    paras.Set<float>   ("alpha_disk", 1.2f);
    paras.Set<unsigned>("C", 384u);
    paras.Set<unsigned>("beamwidth", 4u);
    paras.Set<unsigned>("nodes_to_cache", 0u);
    // Use search_threads as the PQ/disk search parallelism knob.
    paras.Set<unsigned>("num_search_threads",
                        static_cast<unsigned>(a.search_threads));

    // Metric.
    diskann::Metric metric = diskann::Metric::L2;
    if (a.metric == "ip" || a.metric == "inner_product") {
        metric = diskann::Metric::INNER_PRODUCT;
    }

    std::fprintf(stderr, "Opening FreshDiskANN index: %s\n", a.index_prefix.c_str());
    std::fflush(stderr);

    // Construct MergeInsert.
    // disk_prefix_in  = a.index_prefix
    // disk_prefix_out = a.index_prefix + "_merge"  (output for merged index)
    // working_folder  = a.index_prefix  (TMP_FOLDER for merge temp files)
    // need_copy       = true  (first run — copy disk index to shadow area)
    // need_swap       = false (no swap on first run)
    // merge_npts      = 0     — we call init_mem_index() explicitly with our size
    diskann::DistanceL2 dist_cmp;
    diskann::MergeInsert<float, uint32_t> idx(
        paras,
        insert_dim_val,             // real dim read from insert-data header
        a.index_prefix + "_mem",
        a.index_prefix,
        a.index_prefix + "_merge",
        &dist_cmp,
        metric,
        /*single_file_index=*/false,
        /*working_folder=*/a.index_prefix,
        // Restart-after-merge (Fresh overall_performance.cpp:380-382 pattern):
        //   ckpt==0 → need_copy=true  (first run copies pristine→shadow)
        //   ckpt>0  → need_copy=false (reuse merged graph from prior chunk)
        //   need_swap alternates each chunk so the live index toggles between
        //   _shadow1 and _merge, each chunk writing to whichever is NOT live.
        /*need_copy=*/(a.ckpt == 0),
        /*need_swap=*/(a.ckpt % 2 != 0),
        /*merge_npts=*/0);

    // Initialize mem index with our batch-proportional merge threshold.
    idx.init_mem_index(static_cast<uint64_t>(a.merge_size));
    if (!a.entrance_graph.empty()) {
        idx.load_entrance_graph(a.entrance_graph, static_cast<unsigned>(a.entry_L));
    }

    std::fprintf(stderr, "Index opened. dim=%zu, threads: %d search + %d insert, merge_size=%d\n",
                insert_dim_val, a.search_threads, a.insert_threads, a.merge_size);
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Load queries (aligned bin format).
    // -----------------------------------------------------------------------
    float* queries         = nullptr;
    size_t query_count     = 0;
    size_t query_dim       = 0;
    size_t query_aligned_dim = 0;
    diskann::load_aligned_bin<float>(a.query_path, queries, query_count,
                                     query_dim, query_aligned_dim);
    std::fprintf(stderr, "Queries loaded: %zu x %zu (aligned %zu)\n",
                query_count, query_dim, query_aligned_dim);
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Set up bench harness.
    // -----------------------------------------------------------------------
    HarnessConfig cfg{};
    cfg.system           = "freshdiskann";
    cfg.dataset          = a.dataset;
    cfg.shadow_prefix    = a.index_prefix;
    cfg.csv_path         = a.csv_path;
    cfg.base_n           = a.base_n;
    cfg.batch_size       = a.batch_size;
    cfg.num_batches      = a.num_batches;
    cfg.search_threads   = a.search_threads;
    cfg.insert_threads   = a.insert_threads;
    cfg.cache_gb         = 0;
    cfg.search_L         = a.search_L;
    cfg.insert_L         = a.insert_L;
    cfg.K                = a.K;
    cfg.mem_L            = a.entrance_graph.empty() ? 0 : a.entry_L;

    // Shadow bytes function: total of all Fresh file components.
    auto shadow_fn = [&]() -> uint64_t {
        return sum_glob_bytes(a.index_prefix + "_shadow*")
             + sum_glob_bytes(a.index_prefix + "_mem.index*")
             + sum_glob_bytes(a.index_prefix + "_mem_*")
             + sum_glob_bytes(a.index_prefix + "_merge*");
    };

    BenchHarness H(cfg, shadow_fn);
    H.begin_run();

    // -----------------------------------------------------------------------
    // Batch buffer.
    // -----------------------------------------------------------------------
    std::vector<float> batch_data(static_cast<size_t>(a.batch_size) * insert_dim_val);

    // -----------------------------------------------------------------------
    // Per-batch loop.
    // -----------------------------------------------------------------------
    // Accumulated trigger_merge() wall time across batches — folded into
    // total_merge_drain_s in emit_final_summary so fair_throughput =
    // total_inserts / (insert_time + merge_time).
    double total_inline_merge_s = 0.0;
    // Restart-after-merge resume: skip to start_batch
    // and seed harness counters so CSV t_elapsed_s / inserted_so_far continue
    // from the prior chunk. Called after begin_run so run_start_ gets shifted.
    if (a.start_batch > 0 || a.elapsed_offset_s > 0.0) {
        uint64_t inserted_offset =
            static_cast<uint64_t>(a.start_batch) * static_cast<uint64_t>(a.batch_size);
        H.seed_resume(inserted_offset, a.elapsed_offset_s);
        std::fprintf(stderr,
            "[bench_freshdiskann] resume: start_batch=%d inserted_offset=%llu elapsed_offset=%.2fs\n",
            a.start_batch, (unsigned long long)inserted_offset, a.elapsed_offset_s);
        std::fflush(stderr);
    }

    // -----------------------------------------------------------------------
    // Sync merge inside insert_fn + restart-after-merge (OdinANN's evaluation
    // method). Async merge was evaluated but
    // Fresh's _mem_index_1 buffer caps at 2×merge_th (line merge_insert.h:58);
    // unbounded inserts during the 53-min FineWeb merge overflow that cap
    // → `munmap_chunk(): invalid pointer`. Fresh's own overall_performance.cpp
    // rate-limits inserts to match merge rate, so async+rate-limit and sync
    // converge on the same fair throughput (bounded by merger throughput).
    // Sync is simpler and segfault-free under our restart wrapper.
    // -----------------------------------------------------------------------
    bool pending_exit = false;
    for (int b = a.start_batch; b < a.num_batches; ++b) {
        size_t batch_start = static_cast<size_t>(a.base_n) + b * a.batch_size;
        size_t batch_count = static_cast<size_t>(a.batch_size);
        if (batch_start >= insert_n) break;
        if (batch_start + batch_count > insert_n)
            batch_count = insert_n - batch_start;

        load_insert_batch(a.insert_data, batch_data.data(),
                          batch_start, batch_count, insert_dim_val);

        std::string gt_path;
        if (!a.groundtruth_dir.empty()) {
            gt_path = a.groundtruth_dir + "/gt_"
                    + std::to_string(b * a.batch_size) + ".bin";
        }

        // GT for post-batch pure-search: state after (b+1)*batch_size inserts.
        std::string post_gt_path;
        if (!a.groundtruth_dir.empty()) {
            post_gt_path = a.groundtruth_dir + "/gt_"
                         + std::to_string((b + 1) * a.batch_size) + ".bin";
        }

        // insert_fn — OMP parallel-for + inline trigger_merge() so harness's
        // 5s cadence keeps ticking during merge → search_fn fires during
        // merge → tick_idx=0,1,... rows capture search-during-merge latency
        // and QPS. trigger_merge() is synchronous (takes ~50s for ImageNet),
        // giving plenty of mid-merge ticks.
        auto insert_fn = [&]() {
#pragma omp parallel for num_threads(a.insert_threads) schedule(dynamic)
            for (int i = 0; i < static_cast<int>(batch_count); ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                idx.insert(batch_data.data() + i * insert_dim_val,
                           static_cast<uint32_t>(batch_start + i));
                auto t1 = std::chrono::high_resolution_clock::now();
                H.record_insert_us(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            // Sync trigger_merge: returns 0 if mem_points < merge_th (no-op),
            // else blocks until the 53-min merge completes. The search_fn
            // still fires at 5 s cadence during the merge window because the
            // BenchHarness runs insert_fn in a separate thread (async future).
            auto tm0 = std::chrono::steady_clock::now();
            int merge_fired = idx.trigger_merge();
            auto tm1 = std::chrono::steady_clock::now();
            double tm_s = std::chrono::duration<double>(tm1 - tm0).count();
            if (merge_fired) {
                total_inline_merge_s += tm_s;
                std::fprintf(stderr,
                    "[bench_freshdiskann] trigger_merge inline after batch %d (%.2fs)\n",
                    b, tm_s);
                std::fflush(stderr);
            }
        };

        // search_fn — called by the harness at each tick.
        auto search_fn = [&](int /*tick_idx*/) -> TickMetrics {
            // Run search pass.
            // No recall while inserting; only the post-batch pass reports recall.
            TickMetrics m = run_search_pass(idx, queries, query_count,
                                            query_aligned_dim,
                                            a.K,
                                            static_cast<uint64_t>(a.search_L),
                                            gt_path,
                                            a.search_threads,
                                            /*report_recall=*/false);

            return m;
        };

        // post_search_fn — clean-state search after inserts drain, uses post_gt.
        auto post_search_fn = [&](int /*bi*/) -> TickMetrics {
            return run_search_pass(idx, queries, query_count,
                                   query_aligned_dim,
                                   a.K,
                                   static_cast<uint64_t>(a.search_L),
                                   post_gt_path,
                                   a.search_threads);
        };

        // Capture elapsed before run_batch so we can detect merge-completion
        // post-facto via total_inline_merge_s bumping inside insert_fn.
        double merge_before = total_inline_merge_s;
        H.run_batch(b, insert_fn, search_fn, post_search_fn);

        // Restart-after-merge: if insert_fn's sync trigger_merge ran a merge
        // this batch (total_inline_merge_s increased), write a checkpoint and
        // exit so the wrapper respawns — dodging Fresh's post-merge memory
        // corruption. Each restarted process does at
        // most ONE merge, which keeps Fresh in a state it never hits upstream
        // (where their tests also exit after final_merge).
        if (total_inline_merge_s > merge_before) {
            double elapsed_now = H.elapsed_s();
            std::string ckpt_path = a.csv_path + ".next_start";
            if (std::FILE* cf = std::fopen(ckpt_path.c_str(), "w")) {
                std::fprintf(cf, "next_start_batch=%d\nelapsed_s=%.6f\n",
                             b + 1, elapsed_now);
                std::fclose(cf);
            }
            std::fprintf(stderr,
                "[bench_freshdiskann] merge completed in batch %d — "
                "exiting for restart\n", b);
            std::fflush(stderr);
            pending_exit = true;
        }
        if (pending_exit) break;
    }

    // Emit run-wide fair throughput summary (tick_idx=-4).
    // drain_fn returns ONLY final_merge() time; total_inline_merge_s is
    // already included inside total_insert_time_s (since trigger_merge runs
    // synchronously inside insert_fn). Returning inline+final would
    // double-count the inline merges. fair_throughput formula in the harness:
    //   inserts / (total_insert_time + drain) is correct when drain = final_merge only.
    H.emit_final_summary([&]() -> double {
        std::fprintf(stderr, "[bench_freshdiskann] draining pending merges via final_merge()...\n");
        std::fflush(stderr);
        auto t0 = std::chrono::steady_clock::now();
        idx.final_merge();
        auto t1 = std::chrono::steady_clock::now();
        double final_drain_s = std::chrono::duration<double>(t1 - t0).count();
        double drain_s = final_drain_s;
        std::fprintf(stderr,
            "[bench_freshdiskann] merge time: inline=%.2fs (inside insert_fn) + final=%.2fs = %.2fs total\n",
            total_inline_merge_s, final_drain_s, total_inline_merge_s + final_drain_s);
        std::fflush(stderr);
        return drain_s;
    });

    H.end_run();

    delete[] queries;
    std::fprintf(stderr, "CSV written: %s\n", a.csv_path.c_str());
    return 0;
}
