// baselines/common/harness/bench_harness.h
// Shared cadence timer + CSV emitter + RSS sampler.
// Used by bench_freshdiskann.

#pragma once

#include "csv_schema.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace bench {

struct HarnessConfig {
    // Identity.
    std::string system;         // "navis" | "odin" | "fresh"
    std::string dataset;        // "imagenet" | "fineweb"
    std::string shadow_prefix;  // for disk_index_probe
    std::string csv_path;

    // Workload.
    int base_n{0};
    int batch_size{0};
    int num_batches{0};

    // Threads + params.
    int search_threads{22};
    int insert_threads{10};
    int cache_gb{4};
    int search_L{40};
    int insert_L{100};
    int K{10};
    int mem_L{10};              // Forced uniform across all systems.

    // Cadence.
    std::chrono::seconds tick_interval{5};
    std::chrono::seconds tick_sleep{1};
};

struct TickMetrics {
    double qps{0};
    double mean_lat_ms{0};
    double p50_ms{0};
    double p90_ms{0};
    double p95_ms{0};
    double p99_ms{0};
    double p999_ms{0};
    double recall_at_10{0};
    double mean_disk_ios{0};
    double cache_hit_rate{0};   // 0 if system has no cache
    std::string notes;          // Optional per-tick k=v annotation (appended to CSV notes col).
};

// Callback that returns the current shadow-bytes for this system.
using ShadowBytesFn = std::function<uint64_t()>;

// Callback that returns extra k=v notes for the completed insertion batch.
using BatchNotesFn = std::function<std::string()>;

class BenchHarness {
 public:
    explicit BenchHarness(HarnessConfig cfg, ShadowBytesFn shadow_bytes_fn);
    ~BenchHarness();

    // Writes the CSV header. Call once.
    void begin_run();

    // Drives one batch.
    //   insert_fn       : blocks until the batch drains.
    //                     Must record per-insert latency samples through `record_insert_us`
    //                     for per-batch percentile reporting.
    //   search_fn       : runs one search pass over the query set, fills out TickMetrics.
    //                     If empty, the harness waits directly for insert completion and
    //                     emits no concurrent-search rows (tick_idx>=0).
    //   post_search_fn  : optional. If set, called AFTER insert completes and the per-batch
    //                     summary row (tick_idx=-1) is emitted. Runs a clean-state search
    //                     pass with NO concurrent insert. Emits one additional CSV row with
    //                     tick_idx=-2 and notes="post_batch_pure_search=1".
    //   batch_notes_fn  : optional. Called after insert_fn's future drains, but before the
    //                     per-batch summary row and post_search_fn. A nonempty result is
    //                     appended to the standard tick_idx=-1 insertion notes with ";";
    //                     it must not contain commas or newlines. Exceptions propagate and
    //                     suppress this batch's tick_idx=-1 and post-search rows.
    //
    // Behavior: spawn insert_fn as std::async. If search_fn is empty, wait directly for
    //   insert completion. Otherwise loop: wait_for(tick_interval); if still running,
    //   call search_fn, emit a CSV tick row, and sleep(tick_sleep). After insertion is
    //   ready, emit the per-batch row and return.
    // Watchdog hard-exits the process if batch exceeds T_max.
    void run_batch(int batch_idx,
                   std::function<void()> insert_fn,
                   std::function<TickMetrics(int tick_idx)> search_fn,
                   std::function<TickMetrics(int batch_idx)> post_search_fn = {},
                   BatchNotesFn batch_notes_fn = {});

    // Called from within insert_fn (by each insert thread, per vector).
    // Used to compute per-batch p50/p99 insert latency.
    void record_insert_us(double us);

    // Emit an arbitrary pre-populated CSV row directly (thread-safe).
    // Used by Fresh driver to emit tick_idx=-3 during-merge rows.
    void emit_row(const CsvRow& r);

    // Elapsed seconds since begin_run().
    double elapsed_s() const;

    // Emit a run-wide fair throughput summary row (tick_idx=-4, batch_idx=-1).
    // Call BEFORE end_run() so the CSV file is still open.
    //
    // drain_fn: optional callback that blocks until all pending async work
    //   (e.g. Fresh merges) completes and returns the drain wall time in seconds.
    //   If null, drain_time=0 (synchronous systems: NAVIS, Odin).
    //
    // Fair throughput formula:
    //   fair_throughput_vps = total_inserts / (total_insert_time_s + drain_time_s)
    // where total_insert_time_s is the sum of per-batch wall times from run_batch().
    void emit_final_summary(std::function<double()> drain_fn = {});

    // Final flush + close. Call once.
    void end_run();

    // Seed accumulators for a resumed run (restart-after-merge for Fresh).
    // Call BEFORE begin_run() / run_batch(). Sets the inserted-so-far counter
    // and the run-start timestamp so CSV rows use continuous values.
    // `elapsed_offset_s` shifts the run_start_ into the past so t_elapsed_s
    // continues from the prior chunk's last tick.
    void seed_resume(uint64_t inserted_offset, double elapsed_offset_s);

 private:
    HarnessConfig     cfg_;
    ShadowBytesFn     shadow_bytes_fn_;
    CsvWriter         csv_;
    std::chrono::steady_clock::time_point run_start_;
    uint64_t          inserted_total_{0};
    std::vector<double> batch_wall_s_;
    // Per-batch insert latencies (reset each batch_start).
    std::vector<double> insert_us_;
    std::mutex          insert_us_mu_;
};

}  // namespace bench
