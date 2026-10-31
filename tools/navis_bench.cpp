// navis_bench — run NAVIS: a search-only pass, then batches of concurrent
// search + insertion (optionally with deletion), reporting throughput,
// latency, and recall.

#include "navis/index/navis_index.h"
#include "navis/io/uring_backend.h"
#include "navis/util/file_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <getopt.h>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace navis;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string index_prefix;
    std::string query_path;
    std::string insert_data_path;
    std::string gt_dir;
    std::string type   = "float";
    std::string metric = "l2";

    uint32_t K              = 10;
    uint32_t search_L       = 40;
    uint32_t insert_L       = 100;
    uint32_t mem_L          = 10;
    uint32_t beam_width     = 4;
    uint32_t search_threads = 22;
    uint32_t insert_threads = 10;
    uint32_t batch_size     = 100000;
    uint32_t num_batches    = 10;

    bool     workload_change = false;
    uint32_t delete_threads  = 1;
    uint32_t warmup_passes   = 0;

    uint32_t warmup_queries  = 1000;
    uint32_t casr_pct_search = 100;
    uint32_t casr_pct_insert = 90;
    float    ent_ratio       = 0.01f;
    uint32_t ent_R           = 32;

    uint64_t cache_gb      = 16;
    float    cache_window  = 0.10f;
    uint32_t cache_promote = 2;

    std::vector<uint32_t> L_values;   // search-L sweep (trailing arguments)
};

static void print_usage(const char* prog) {
    std::printf(
        "Usage: %s -i <prefix> -q <query.bin> [options] [search-L ...]\n"
        "\n"
        "Runs a search-only pass, then --num-batches batches of concurrent search\n"
        "and insertion. Inserts modify a shadow copy of the index ({prefix}_shadow_*);\n"
        "the original files are never written.\n"
        "\n"
        "Input:\n"
        "  -i, --index <prefix>       Index prefix (from navis_build)\n"
        "  -q, --query <path>         Query vectors (.bin)\n"
        "  -g, --groundtruth <dir>    Directory with gt_<offset>.bin (navis_convert batch-gt)\n"
        "      --insert-data <path>   Data file; row i is inserted with id i, starting\n"
        "                             at the index size (required if --num-batches > 0)\n"
        "  -t, --type <str>           float|int8|uint8 (default: float)\n"
        "  -m, --metric <str>         l2|cosine|ip (default: l2)\n"
        "\n"
        "Workload:\n"
        "  -K <int>                   Top-K (default: 10)\n"
        "      --search-L <int>       Search frontier |E_search| (default: 40)\n"
        "      --insert-L <int>       Position-seeking frontier |E_pos| (default: 100)\n"
        "      --mem-L <int>          Entrance-graph search list (default: 10)\n"
        "      --beam-width <int>     Edge pages read per hop (default: 4)\n"
        "      --search-threads <int> (default: 22)\n"
        "      --insert-threads <int> (default: 10)\n"
        "      --batch-size <int>     Insertions per batch (default: 100000)\n"
        "      --num-batches <int>    Batches; 0 = search only (default: 10)\n"
        "      --warmup-passes <int>  Unreported search passes over all queries before\n"
        "                             the first measured pass, to warm the cache (default: 0)\n"
        "      --workload-change      Each batch also deletes the batch_size oldest\n"
        "                             vectors (ids [b*batch, (b+1)*batch)).\n"
        "      --delete-threads <int> Deletion threads (default: 1)\n"
        "\n"
        "NAVIS:\n"
        "      --cache-gb <int>       NAVIS-Cache size in GiB, 0 = off (default: 16)\n"
        "      --cache-window <f>     Admission-window share (default: 0.10)\n"
        "      --cache-promote <int>  Window hits needed for promotion (default: 2)\n"
        "      --warmup-queries <int> CASR warm-up reranks per path (default: 1000)\n"
        "      --casr-percentile-search <int>  Percentile of warm-up windows used as\n"
        "                             w_search (default: 100)\n"
        "      --casr-percentile-insert <int>  ... as w_insert (default: 90)\n"
        "      --casr-percentile <int> Set both\n"
        "      --ent-ratio <f>        Entrance-graph target size / |G| (default: 0.01)\n"
        "      --ent-R <int>          Entrance-graph degree (default: 32)\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    enum : int {
        kInsertData = 1000, kSearchL, kInsertL, kMemL, kBeamWidth, kSearchThreads,
        kInsertThreads, kBatchSize, kNumBatches, kWorkloadChange, kDeleteThreads,
        kCacheGb, kCacheWindow, kCachePromote, kWarmupQueries, kCasrPercentile,
        kCasrPctSearch, kCasrPctInsert,
        kEntRatio, kEntR, kWarmupPasses,
    };
    static struct option long_opts[] = {
        {"index",           required_argument, 0, 'i'},
        {"query",           required_argument, 0, 'q'},
        {"groundtruth",     required_argument, 0, 'g'},
        {"type",            required_argument, 0, 't'},
        {"metric",          required_argument, 0, 'm'},
        {"insert-data",     required_argument, 0, kInsertData},
        {"search-L",        required_argument, 0, kSearchL},
        {"insert-L",        required_argument, 0, kInsertL},
        {"mem-L",           required_argument, 0, kMemL},
        {"beam-width",      required_argument, 0, kBeamWidth},
        {"search-threads",  required_argument, 0, kSearchThreads},
        {"insert-threads",  required_argument, 0, kInsertThreads},
        {"batch-size",      required_argument, 0, kBatchSize},
        {"num-batches",     required_argument, 0, kNumBatches},
        {"workload-change", no_argument,       0, kWorkloadChange},
        {"delete-threads",  required_argument, 0, kDeleteThreads},
        {"warmup-passes",   required_argument, 0, kWarmupPasses},
        {"cache-gb",        required_argument, 0, kCacheGb},
        {"cache-window",    required_argument, 0, kCacheWindow},
        {"cache-promote",   required_argument, 0, kCachePromote},
        {"warmup-queries",  required_argument, 0, kWarmupQueries},
        {"casr-percentile", required_argument, 0, kCasrPercentile},
        {"casr-percentile-search", required_argument, 0, kCasrPctSearch},
        {"casr-percentile-insert", required_argument, 0, kCasrPctInsert},
        {"ent-ratio",       required_argument, 0, kEntRatio},
        {"ent-R",           required_argument, 0, kEntR},
        {"help",            no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "i:q:g:K:t:m:h", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'i': a.index_prefix = optarg; break;
            case 'q': a.query_path   = optarg; break;
            case 'g': a.gt_dir       = optarg; break;
            case 'K': a.K            = std::stoi(optarg); break;
            case 't': a.type         = optarg; break;
            case 'm': a.metric       = optarg; break;
            case kInsertData:     a.insert_data_path = optarg; break;
            case kSearchL:        a.search_L         = std::stoi(optarg); break;
            case kInsertL:        a.insert_L         = std::stoi(optarg); break;
            case kMemL:           a.mem_L            = std::stoi(optarg); break;
            case kBeamWidth:      a.beam_width       = std::stoi(optarg); break;
            case kSearchThreads:  a.search_threads   = std::stoi(optarg); break;
            case kInsertThreads:  a.insert_threads   = std::stoi(optarg); break;
            case kBatchSize:      a.batch_size       = std::stoi(optarg); break;
            case kNumBatches:     a.num_batches      = std::stoi(optarg); break;
            case kWorkloadChange: a.workload_change  = true; break;
            case kDeleteThreads:  a.delete_threads   = std::stoi(optarg); break;
            case kWarmupPasses:   a.warmup_passes    = std::stoi(optarg); break;
            case kCacheGb:        a.cache_gb         = std::stoull(optarg); break;
            case kCacheWindow:    a.cache_window     = std::stof(optarg); break;
            case kCachePromote:   a.cache_promote    = std::stoi(optarg); break;
            case kWarmupQueries:  a.warmup_queries   = std::stoi(optarg); break;
            case kCasrPercentile: a.casr_pct_search  = a.casr_pct_insert = std::stoi(optarg); break;
            case kCasrPctSearch:  a.casr_pct_search  = std::stoi(optarg); break;
            case kCasrPctInsert:  a.casr_pct_insert  = std::stoi(optarg); break;
            case kEntRatio:       a.ent_ratio        = std::stof(optarg); break;
            case kEntR:           a.ent_R            = std::stoi(optarg); break;
            case 'h': print_usage(argv[0]); std::exit(0);
            default:  print_usage(argv[0]); std::exit(1);
        }
    }
    for (int i = optind; i < argc; i++) a.L_values.push_back(std::stoi(argv[i]));
    if (a.L_values.empty()) a.L_values.push_back(a.search_L);

    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "error: %s\n\n", msg);
        print_usage(argv[0]);
        std::exit(1);
    };
    if (a.index_prefix.empty() || a.query_path.empty())
        fail("-i and -q are required");
    if (a.num_batches > 0 && a.insert_data_path.empty())
        fail("--insert-data is required when --num-batches > 0");
    if (a.num_batches > 0 && (a.insert_threads == 0 || a.batch_size == 0))
        fail("--insert-threads and --batch-size must be > 0 when inserting");
    if (a.workload_change && a.delete_threads == 0)
        fail("--delete-threads must be > 0 with --workload-change");
    if (a.cache_window <= 0.0f || a.cache_window >= 1.0f)
        fail("--cache-window must be in (0, 1)");
    if (a.cache_promote == 0)
        fail("--cache-promote must be >= 1");
    return a;
}

// Fraction of the top-K results that appear in the ground-truth top-K.
static double compute_recall(const uint32_t* results, const uint32_t* gt,
                             uint32_t K, uint32_t gt_K, uint32_t n_queries) {
    uint64_t correct = 0;
    const uint32_t k_gt = std::min(K, gt_K);
    for (uint32_t q = 0; q < n_queries; q++) {
        const uint32_t* r = results + static_cast<uint64_t>(q) * K;
        const uint32_t* g = gt + static_cast<uint64_t>(q) * gt_K;
        for (uint32_t i = 0; i < K; i++) {
            if (std::find(g, g + k_gt, r[i]) != g + k_gt) correct++;
        }
    }
    return 100.0 * static_cast<double>(correct) / (static_cast<double>(n_queries) * K);
}

// Ground truth for one batch boundary: {gt_dir}/gt_<offset>.bin.
struct GroundTruth {
    uint32_t* ids = nullptr;
    uint32_t  K   = 0;
    GroundTruth() = default;
    GroundTruth(const std::string& dir, uint64_t offset) {
        if (dir.empty()) return;
        std::string path = dir + "/gt_" + std::to_string(offset) + ".bin";
        if (!file_exists(path)) return;
        uint32_t n;
        load_bin<uint32_t>(path, ids, n, K);
    }
    ~GroundTruth() { if (ids) free_aligned(ids); }
    GroundTruth(const GroundTruth&) = delete;
    GroundTruth& operator=(const GroundTruth&) = delete;
};

static double host_rss_gib() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0.0;
    char line[256];
    size_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) std::sscanf(line + 6, "%zu", &kb);
    }
    std::fclose(f);
    return kb / (1024.0 * 1024.0);
}

// Copy the mutable index files to {prefix}_shadow_* and symlink the
// read-only ones, so insertions never modify the original index.
static std::string make_shadow_copy(const std::string& prefix) {
    const std::string shadow = prefix + "_shadow";
    for (const char* suffix : {"_disk.edge", "_disk.vec", "_indirection.bin"}) {
        fs::remove(shadow + suffix);
        // An index without _indirection.bin uses the identity mapping.
        if (fs::exists(prefix + suffix)) {
            fs::copy_file(prefix + suffix, shadow + suffix);
        }
    }
    for (const char* suffix : {"_pq_pivots.bin", "_pq_compressed.bin",
                               "_mem.index", "_mem.index.data", "_mem.index.tags"}) {
        fs::remove(shadow + suffix);
        if (fs::exists(prefix + suffix)) {
            fs::create_symlink(fs::absolute(prefix + suffix), shadow + suffix);
        }
    }
    return shadow;
}

template <typename T>
static void run_bench(const Args& a) {
    const bool updates = a.num_batches > 0;

    std::string prefix = a.index_prefix;
    if (updates) {
        std::printf("Copying index to %s_shadow_* ...\n", prefix.c_str());
        std::fflush(stdout);
        prefix = make_shadow_copy(prefix);
    }

    UringBackend edge_io, vec_io;
    edge_io.open(prefix + "_disk.edge", updates);
    vec_io.open(prefix + "_disk.vec", updates);

    SearchConfig config;
    config.warmup_queries  = a.warmup_queries;
    config.casr_percentile_search = a.casr_pct_search;
    config.casr_percentile_insert = a.casr_pct_insert;
    config.ent_ratio       = a.ent_ratio;
    config.ent_R           = a.ent_R;
    config.insert_L        = a.insert_L;

    CacheConfig cache_cfg;
    cache_cfg.size_gb           = a.cache_gb;
    cache_cfg.window_ratio      = a.cache_window;
    cache_cfg.promote_threshold = a.cache_promote;

    const uint32_t worker_threads = a.search_threads + (updates ? a.insert_threads : 0);
    NavisIndex<T> index;
    index.load(prefix, &edge_io, &vec_io, config, cache_cfg,
               std::max<uint32_t>(worker_threads, 1), metric_from_string(a.metric));

    std::printf("Index: %lu points, dim %lu\n", index.num_points(), index.data_dim());
    if (updates) {
        std::printf("Config: K=%u search-L=%u insert-L=%u mem-L=%u beam=%u threads=%u search/%u insert",
                    a.K, a.search_L, a.insert_L, a.mem_L, a.beam_width,
                    a.search_threads, a.insert_threads);
        if (a.workload_change) std::printf("/%u delete", a.delete_threads);
    } else {
        std::printf("Config: K=%u search-L=%u mem-L=%u beam=%u threads=%u search (search only)",
                    a.K, a.search_L, a.mem_L, a.beam_width, a.search_threads);
    }
    std::printf("\n        NAVIS-Cache=%lu GiB (window %.0f%%, promote@%u)  CASR windows P%u search / P%u insert over %u warm-up reranks\n",
                (unsigned long)a.cache_gb, a.cache_window * 100.0, a.cache_promote,
                a.casr_pct_search, a.casr_pct_insert, a.warmup_queries);
    std::fflush(stdout);

    T* queries = nullptr;
    uint32_t n_queries, q_dim;
    load_bin<T>(a.query_path, queries, n_queries, q_dim);

    // One full pass over all queries: {QPS, mean us, p50 us, p99 us, recall %}.
    auto run_search_pass = [&](uint32_t L, const GroundTruth& gt) {
        std::vector<std::vector<double>> tlats(a.search_threads);
        std::vector<uint32_t> ids(static_cast<size_t>(n_queries) * a.K);
        auto t0 = Clock::now();
        std::vector<std::thread> workers;
        for (uint32_t tid = 0; tid < a.search_threads; tid++) {
            uint32_t qs = static_cast<uint32_t>((uint64_t(n_queries) * tid) / a.search_threads);
            uint32_t qe = static_cast<uint32_t>((uint64_t(n_queries) * (tid + 1)) / a.search_threads);
            workers.emplace_back([&, tid, L, qs, qe]() {
                edge_io.register_thread();
                vec_io.register_thread();
                std::vector<float> dists(a.K);
                for (uint32_t qi = qs; qi < qe; qi++) {
                    auto s = Clock::now();
                    index.search(queries + static_cast<uint64_t>(qi) * q_dim, a.K,
                                 a.mem_L, L, a.beam_width,
                                 ids.data() + static_cast<uint64_t>(qi) * a.K, dists.data());
                    tlats[tid].push_back(
                        std::chrono::duration<double, std::micro>(Clock::now() - s).count());
                }
                edge_io.deregister_thread();
                vec_io.deregister_thread();
            });
        }
        for (auto& w : workers) w.join();
        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

        std::vector<double> lats;
        for (auto& v : tlats) lats.insert(lats.end(), v.begin(), v.end());
        std::sort(lats.begin(), lats.end());
        size_t n = lats.size();
        double recall = gt.ids ? compute_recall(ids.data(), gt.ids, a.K, gt.K, n_queries) : -1.0;
        return std::array<double, 5>{
            n / elapsed,
            std::accumulate(lats.begin(), lats.end(), 0.0) / n,
            lats[n / 2],
            lats[std::min(n - 1, n * 99 / 100)],
            recall};
    };
    auto print_search = [](uint32_t L, const std::array<double, 5>& r) {
        std::printf("L=%-4u QPS %9.1f  mean %8.1f us  p50 %8.1f  p99 %8.1f",
                    L, r[0], r[1], r[2], r[3]);
        if (r[4] >= 0) std::printf("  Recall@K %6.2f%%", r[4]);
        std::printf("\n");
    };

    // ------------------------------------------------------------------
    // Search-only pass on the initial index.
    // ------------------------------------------------------------------
    if (a.search_threads > 0) {
        GroundTruth gt0(a.gt_dir, 0);
        if (a.warmup_passes > 0) {
            std::printf("\nWarming up: %u search passes at L=%u ...\n",
                        a.warmup_passes, a.L_values[0]);
            std::fflush(stdout);
            for (uint32_t w = 0; w < a.warmup_passes; w++) run_search_pass(a.L_values[0], GroundTruth{});
        }
        std::printf("\n--- Search ---\n");
        for (uint32_t L : a.L_values) {
            index.reset_vec_load_stats();
            auto r = run_search_pass(L, gt0);
            std::printf("  ");
            print_search(L, r);
            std::printf("         vec loads/query %.1f\n", index.search_vec_loads_mean());
            std::fflush(stdout);
        }
    }
    if (!updates) {
        std::printf("\nCASR w_search=%u  NAVIS-Cache hit rate %.1f%%  RSS %.1f GiB\n",
                    index.casr_window_search(),
                    100.0 * index.navis_cache_hits() /
                        std::max<uint64_t>(1, index.navis_cache_hits() + index.navis_cache_misses()),
                    host_rss_gib());
        free_aligned(queries);
        return;
    }

    // ------------------------------------------------------------------
    // Batches of concurrent search + insertion (+ deletion).
    // ------------------------------------------------------------------
    uint32_t n_data, d_dim;
    get_bin_metadata(a.insert_data_path, n_data, d_dim);
    const uint64_t base_n = index.num_points();
    T* batch_data = static_cast<T*>(
        alloc_aligned(static_cast<uint64_t>(a.batch_size) * d_dim * sizeof(T)));

    for (uint32_t b = 0; b < a.num_batches; b++) {
        const uint64_t offset = static_cast<uint64_t>(b) * a.batch_size;
        const uint64_t batch_start = base_n + offset;
        if (batch_start >= n_data) {
            std::printf("\nInsert data exhausted after %u batches.\n", b);
            break;
        }
        const uint64_t batch_count = std::min<uint64_t>(a.batch_size, n_data - batch_start);
        load_bin_range<T>(a.insert_data_path, batch_data, batch_start, batch_count, d_dim);

        std::printf("\n--- Batch %u: insert ids [%lu, %lu)", b,
                    (unsigned long)batch_start, (unsigned long)(batch_start + batch_count));
        if (a.workload_change) {
            std::printf(", delete ids [%lu, %lu)",
                        (unsigned long)offset, (unsigned long)(offset + batch_count));
        }
        std::printf(" ---\n");
        std::fflush(stdout);
        index.reset_vec_load_stats();
        index.navis_cache_reset_stats();

        std::atomic<uint64_t> inserted{0}, insert_us{0}, deleted{0};
        std::atomic<uint32_t> active_inserters{a.insert_threads};
        std::mutex done_mu;
        std::condition_variable done_cv;
        bool inserts_done = false;
        Clock::time_point t_insert_end;   // set by the last inserter to finish
        const auto t0 = Clock::now();

        std::vector<std::thread> workers;
        for (uint32_t tid = 0; tid < a.insert_threads; tid++) {
            uint64_t is = (batch_count * tid) / a.insert_threads;
            uint64_t ie = (batch_count * (tid + 1)) / a.insert_threads;
            workers.emplace_back([&, is, ie]() {
                edge_io.register_thread();
                vec_io.register_thread();
                for (uint64_t i = is; i < ie; i++) {
                    auto s = Clock::now();
                    index.insert(batch_data + i * d_dim, static_cast<uint32_t>(batch_start + i));
                    insert_us.fetch_add(static_cast<uint64_t>(
                        std::chrono::duration<double, std::micro>(Clock::now() - s).count()),
                        std::memory_order_relaxed);
                    inserted.fetch_add(1, std::memory_order_relaxed);
                }
                if (active_inserters.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(done_mu);
                    t_insert_end = Clock::now();
                    inserts_done = true;
                    done_cv.notify_all();
                }
                edge_io.deregister_thread();
                vec_io.deregister_thread();
            });
        }
        if (a.workload_change) {
            for (uint32_t tid = 0; tid < a.delete_threads; tid++) {
                uint64_t ds = (batch_count * tid) / a.delete_threads;
                uint64_t de = (batch_count * (tid + 1)) / a.delete_threads;
                workers.emplace_back([&, ds, de]() {
                    for (uint64_t i = ds; i < de; i++) {
                        if (index.erase(static_cast<uint32_t>(offset + i)))
                            deleted.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
        }

        // While insertions run: wait up to 5 s; if they are still running,
        // report progress, run one full concurrent search pass, and pause 1 s.
        // Its recall is not reported: the index is changing during the pass,
        // so no ground-truth file matches it.
        constexpr auto kTickWait  = std::chrono::seconds(5);
        constexpr auto kTickPause = std::chrono::seconds(1);
        while (true) {
            {
                std::unique_lock<std::mutex> lk(done_mu);
                if (done_cv.wait_for(lk, kTickWait, [&] { return inserts_done; })) break;
            }
            double el = std::chrono::duration<double>(Clock::now() - t0).count();
            uint64_t n = inserted.load(std::memory_order_relaxed);
            std::printf("  [%4.0fs] inserted %lu/%lu (%.0f/s)", el,
                        (unsigned long)n, (unsigned long)batch_count, n / el);
            if (a.search_threads > 0) {
                auto r = run_search_pass(a.L_values[0], GroundTruth{});
                std::printf("  | ");
                print_search(a.L_values[0], r);
            } else {
                std::printf("\n");
            }
            std::fflush(stdout);
            std::this_thread::sleep_for(kTickPause);
        }
        for (auto& w : workers) w.join();

        double total_s = std::chrono::duration<double>(t_insert_end - t0).count();
        uint64_t n = inserted.load();
        std::printf("  Batch %u done in %.1fs: %lu inserts (%.1f/s, mean latency %.0f us)",
                    b, total_s, (unsigned long)n, n / total_s,
                    n ? double(insert_us.load()) / n : 0.0);
        if (a.workload_change) {
            std::printf(", %lu deletes (%.2f%% of ids deleted)",
                        (unsigned long)deleted.load(), 100.0 * index.deleted_fraction());
        }
        uint64_t h = index.navis_cache_hits(), m = index.navis_cache_misses();
        std::printf("\n  vec loads/rerank: search %.1f, insert %.1f (CASR w_search=%u w_insert=%u)"
                    "  NAVIS-Cache hit %.1f%%  RSS %.1f GiB\n",
                    index.search_vec_loads_mean(), index.insert_vec_loads_mean(),
                    index.casr_window_search(), index.casr_window_insert(),
                    100.0 * h / std::max<uint64_t>(1, h + m), host_rss_gib());

        // Search pass on the post-batch state.
        if (a.search_threads > 0) {
            GroundTruth post_gt(a.gt_dir, offset + batch_count);
            for (uint32_t L : a.L_values) {
                std::printf("  after batch %u: ", b);
                print_search(L, run_search_pass(L, post_gt));
            }
        }
        std::fflush(stdout);
    }

    free_aligned(batch_data);
    free_aligned(queries);
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.type == "float")      run_bench<float>(a);
    else if (a.type == "int8")  run_bench<int8_t>(a);
    else if (a.type == "uint8") run_bench<uint8_t>(a);
    else {
        std::fprintf(stderr, "Unknown type: %s\n", a.type.c_str());
        return 1;
    }
    return 0;
}
