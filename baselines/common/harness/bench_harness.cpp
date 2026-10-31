// baselines/common/harness/bench_harness.cpp

#include "bench_harness.h"
#include "rss_sampler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace bench {

// ===== CsvWriter =====
CsvWriter::CsvWriter(const std::string& path) {
    fp_ = std::fopen(path.c_str(), "w");
    if (!fp_) throw std::runtime_error("CsvWriter: cannot open " + path);
    std::fputs(kCsvHeader, fp_);
    std::fflush(fp_);
}

CsvWriter::~CsvWriter() {
    if (fp_) std::fclose(fp_);
}

void CsvWriter::write_row(const CsvRow& r) {
    std::lock_guard<std::mutex> g(mu_);
    std::fprintf(fp_,
        "%s,%s,%d,%d,%d,%d,%.3f,%lu,"
        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
        "%.4f,%.3f,%lu,%.4f,%lu,"
        "%d,%d,%d,%d,%d,%d,%s\n",
        r.system, r.dataset, r.base_n, r.batch_idx, r.batch_size, r.tick_idx,
        r.t_elapsed_s, (unsigned long)r.inserted_so_far,
        r.qps, r.mean_lat_ms, r.p50_ms, r.p90_ms, r.p95_ms, r.p99_ms, r.p999_ms,
        r.recall_at_10, r.mean_disk_ios, (unsigned long)r.rss_mib,
        r.cache_hit_rate, (unsigned long)r.disk_index_bytes,
        r.search_threads, r.insert_threads, r.cache_gb,
        r.search_L, r.insert_L, r.K, r.notes.c_str());
    std::fflush(fp_);
}

// ===== BenchHarness =====
BenchHarness::BenchHarness(HarnessConfig cfg, ShadowBytesFn shadow_bytes_fn)
    : cfg_(std::move(cfg)),
      shadow_bytes_fn_(std::move(shadow_bytes_fn)),
      csv_(cfg_.csv_path) {}

BenchHarness::~BenchHarness() = default;

void BenchHarness::begin_run() {
    run_start_ = std::chrono::steady_clock::now();
}

void BenchHarness::seed_resume(uint64_t inserted_offset, double elapsed_offset_s) {
    inserted_total_ = inserted_offset;
    // Shift the virtual run start into the past so subsequent t_elapsed_s
    // readings continue from the prior chunk. Safe to call before begin_run()
    // (which will overwrite run_start_) or after (to apply the offset).
    run_start_ = std::chrono::steady_clock::now()
               - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                     std::chrono::duration<double>(elapsed_offset_s));
}

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

void BenchHarness::record_insert_us(double us) {
    std::lock_guard<std::mutex> g(insert_us_mu_);
    insert_us_.push_back(us);
}

void BenchHarness::emit_row(const CsvRow& r) {
    csv_.write_row(r);
}

double BenchHarness::elapsed_s() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - run_start_).count();
}

void BenchHarness::run_batch(int batch_idx,
                             std::function<void()> insert_fn,
                             std::function<TickMetrics(int)> search_fn,
                             std::function<TickMetrics(int)> post_search_fn,
                             BatchNotesFn batch_notes_fn) {
    { std::lock_guard<std::mutex> g(insert_us_mu_); insert_us_.clear(); }

    auto batch_start = std::chrono::steady_clock::now();
    auto fut = std::async(std::launch::async, insert_fn);
    if (!search_fn) {
        fut.wait();
    } else {
        int tick_idx = 0;
        std::future_status status;
        do {
            status = fut.wait_for(cfg_.tick_interval);
            auto now = std::chrono::steady_clock::now();
            double t_elapsed = std::chrono::duration<double>(now - run_start_).count();

            if (status == std::future_status::timeout) {
                TickMetrics m = search_fn(tick_idx);
                CsvRow r{};
                r.system = cfg_.system.c_str(); r.dataset = cfg_.dataset.c_str();
                r.base_n = cfg_.base_n; r.batch_idx = batch_idx;
                r.batch_size = cfg_.batch_size; r.tick_idx = tick_idx;
                r.t_elapsed_s = t_elapsed;
                r.inserted_so_far = inserted_total_;
                r.qps = m.qps; r.mean_lat_ms = m.mean_lat_ms;
                r.p50_ms = m.p50_ms; r.p90_ms = m.p90_ms; r.p95_ms = m.p95_ms;
                r.p99_ms = m.p99_ms; r.p999_ms = m.p999_ms;
                r.recall_at_10 = m.recall_at_10; r.mean_disk_ios = m.mean_disk_ios;
                r.rss_mib = read_rss_mib();
                r.cache_hit_rate = m.cache_hit_rate;
                r.disk_index_bytes = shadow_bytes_fn_ ? shadow_bytes_fn_() : 0;
                r.search_threads = cfg_.search_threads;
                r.insert_threads = cfg_.insert_threads;
                r.cache_gb = cfg_.cache_gb;
                r.search_L = cfg_.search_L; r.insert_L = cfg_.insert_L; r.K = cfg_.K;
                if (!m.notes.empty()) r.notes = m.notes;
                csv_.write_row(r);
                std::this_thread::sleep_for(cfg_.tick_sleep);
                ++tick_idx;
            }
        } while (status != std::future_status::ready);
    }

    fut.get();  // propagate exceptions.
    auto batch_end = std::chrono::steady_clock::now();
    double batch_wall = std::chrono::duration<double>(batch_end - batch_start).count();
    batch_wall_s_.push_back(batch_wall);
    inserted_total_ += cfg_.batch_size;

    // Per-batch summary row. Compute the same percentile set the Odin driver
    // emits (P10/P50/P90/P99/P99.9 µs) so downstream tooling (converter,
    // paper plots) can treat every system's CSV uniformly.
    std::vector<double> ius;
    { std::lock_guard<std::mutex> g(insert_us_mu_); ius = insert_us_; }
    double p10  = percentile(ius, 0.10);
    double p50  = percentile(ius, 0.50);
    double p90  = percentile(ius, 0.90);
    double p99  = percentile(ius, 0.99);
    double p999 = percentile(ius, 0.999);
    double vps = ius.empty() ? 0 : (static_cast<double>(ius.size()) / batch_wall);

    char base_notes[384];
    std::snprintf(base_notes, sizeof(base_notes),
                  "insert_throughput_vps=%.2f;"
                  "insert_10p_us=%.0f;insert_50p_us=%.0f;insert_90p_us=%.0f;"
                  "insert_99p_us=%.0f;insert_99_9p_us=%.0f;"
                  "batch_wall_s=%.3f",
                  vps, p10, p50, p90, p99, p999, batch_wall);
    std::string notes = base_notes;
    if (batch_notes_fn) {
        std::string extra = batch_notes_fn();
        if (!extra.empty()) notes += ";" + extra;
    }

    CsvRow r{};
    r.system = cfg_.system.c_str(); r.dataset = cfg_.dataset.c_str();
    r.base_n = cfg_.base_n; r.batch_idx = batch_idx;
    r.batch_size = cfg_.batch_size; r.tick_idx = -1;
    r.t_elapsed_s = std::chrono::duration<double>(batch_end - run_start_).count();
    r.inserted_so_far = inserted_total_;
    r.rss_mib = read_rss_mib();
    r.disk_index_bytes = shadow_bytes_fn_ ? shadow_bytes_fn_() : 0;
    r.search_threads = cfg_.search_threads;
    r.insert_threads = cfg_.insert_threads;
    r.cache_gb = cfg_.cache_gb;
    r.search_L = cfg_.search_L; r.insert_L = cfg_.insert_L; r.K = cfg_.K;
    r.notes = notes;
    csv_.write_row(r);

    // Post-batch pure-search pass (tick_idx=-2): no concurrent insert.
    // Measures clean recall/QPS baseline after inserts drain.
    if (post_search_fn) {
        TickMetrics pm = post_search_fn(batch_idx);
        auto post_now = std::chrono::steady_clock::now();
        CsvRow pr{};
        pr.system           = cfg_.system.c_str();
        pr.dataset          = cfg_.dataset.c_str();
        pr.base_n           = cfg_.base_n;
        pr.batch_idx        = batch_idx;
        pr.batch_size       = cfg_.batch_size;
        pr.tick_idx         = -2;
        pr.t_elapsed_s      = std::chrono::duration<double>(post_now - run_start_).count();
        pr.inserted_so_far  = inserted_total_;
        pr.qps              = pm.qps;
        pr.mean_lat_ms      = pm.mean_lat_ms;
        pr.p50_ms           = pm.p50_ms;
        pr.p90_ms           = pm.p90_ms;
        pr.p95_ms           = pm.p95_ms;
        pr.p99_ms           = pm.p99_ms;
        pr.p999_ms          = pm.p999_ms;
        pr.recall_at_10     = pm.recall_at_10;
        pr.mean_disk_ios    = pm.mean_disk_ios;
        pr.rss_mib          = read_rss_mib();
        pr.cache_hit_rate   = pm.cache_hit_rate;
        pr.disk_index_bytes = shadow_bytes_fn_ ? shadow_bytes_fn_() : 0;
        pr.search_threads   = cfg_.search_threads;
        pr.insert_threads   = cfg_.insert_threads;
        pr.cache_gb         = cfg_.cache_gb;
        pr.search_L         = cfg_.search_L;
        pr.insert_L         = cfg_.insert_L;
        pr.K                = cfg_.K;
        pr.notes            = "post_batch_pure_search=1";
        if (!pm.notes.empty()) pr.notes += ";" + pm.notes;
        csv_.write_row(pr);
    }
}

void BenchHarness::emit_final_summary(std::function<double()> drain_fn) {
    // Total insert wall time = sum of per-batch wall times.
    double total_insert_s = 0.0;
    for (double w : batch_wall_s_) total_insert_s += w;

    // Drain async work (e.g. Fresh merge) and measure its time.
    double drain_s = 0.0;
    if (drain_fn) {
        auto drain_start = std::chrono::steady_clock::now();
        drain_s = drain_fn();  // drain_fn blocks and returns its own measured time
        // If drain_fn returns 0.0 (e.g. fallback sleep), measure wall time ourselves.
        if (drain_s <= 0.0) {
            auto drain_end = std::chrono::steady_clock::now();
            drain_s = std::chrono::duration<double>(drain_end - drain_start).count();
        }
    }

    double total_s = total_insert_s + drain_s;
    double fair_vps = (total_s > 0.0 && inserted_total_ > 0)
        ? static_cast<double>(inserted_total_) / total_s
        : 0.0;

    char notes[512];
    std::snprintf(notes, sizeof(notes),
        "fair_throughput_vps=%.2f;total_insert_time_s=%.3f;"
        "total_merge_drain_s=%.3f;total_time_s=%.3f",
        fair_vps, total_insert_s, drain_s, total_s);

    CsvRow r{};
    r.system           = cfg_.system.c_str();
    r.dataset          = cfg_.dataset.c_str();
    r.base_n           = cfg_.base_n;
    r.batch_idx        = -1;   // run-wide, no specific batch
    r.batch_size       = cfg_.batch_size;
    r.tick_idx         = -4;   // run-wide fair throughput summary
    r.t_elapsed_s      = elapsed_s();
    r.inserted_so_far  = inserted_total_;
    r.rss_mib          = read_rss_mib();
    r.disk_index_bytes = shadow_bytes_fn_ ? shadow_bytes_fn_() : 0;
    r.search_threads   = cfg_.search_threads;
    r.insert_threads   = cfg_.insert_threads;
    r.cache_gb         = cfg_.cache_gb;
    r.search_L         = cfg_.search_L;
    r.insert_L         = cfg_.insert_L;
    r.K                = cfg_.K;
    r.notes            = notes;
    csv_.write_row(r);
}

void BenchHarness::end_run() {
    // CsvWriter flushes every row; nothing to do beyond explicit closure.
}

}  // namespace bench
