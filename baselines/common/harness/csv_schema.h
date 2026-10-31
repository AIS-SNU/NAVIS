// baselines/common/harness/csv_schema.h
// Single CSV schema for the baseline benchmark drivers (FreshDiskANN, IP-DiskANN).

#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace bench {

// Header line written once at the top of every CSV.
inline constexpr const char* kCsvHeader =
    "system,dataset,base_n,batch_idx,batch_size,tick_idx,t_elapsed_s,"
    "inserted_so_far,qps,mean_lat_ms,p50_ms,p90_ms,p95_ms,p99_ms,p999_ms,"
    "recall_at_10,mean_disk_ios,rss_mib,cache_hit_rate,disk_index_bytes,"
    "search_threads,insert_threads,cache_gb,search_L,insert_L,K,notes\n";

// tick_idx conventions:
//   >= 0   : mid-insertion search tick (fires every tick_interval while insert_fn runs)
//   -1     : per-batch summary row (search cols unused; insert stats in notes as k=v;...)
//   -2     : post-batch pure-search row (insert complete, no concurrent insert; clean
//            recall/QPS baseline; notes includes post_batch_pure_search=1)
//   -4     : run-wide fair throughput summary (batch_idx=-1, emitted once after all batches
//            complete and any async merge drain finishes).
//            notes includes fair_throughput_vps=X;total_insert_time_s=Y;
//              total_merge_drain_s=Z;total_time_s=Y+Z

// notes k=v keys (non-exhaustive):
//   insert_throughput_vps, batch_wall_s, fair_throughput_vps, total_insert_time_s,
//   total_merge_drain_s, total_time_s, post_batch_pure_search.

// One row. tick_idx conventions documented above.
struct CsvRow {
    const char* system{""};
    const char* dataset{""};
    int         base_n{0};
    int         batch_idx{0};
    int         batch_size{0};
    int         tick_idx{0};   // see conventions above
    double      t_elapsed_s{0};
    uint64_t    inserted_so_far{0};
    double      qps{0};
    double      mean_lat_ms{0};
    double      p50_ms{0};
    double      p90_ms{0};
    double      p95_ms{0};
    double      p99_ms{0};
    double      p999_ms{0};
    double      recall_at_10{0};
    double      mean_disk_ios{0};
    uint64_t    rss_mib{0};
    double      cache_hit_rate{0};
    uint64_t    disk_index_bytes{0};
    int         search_threads{0};
    int         insert_threads{0};
    int         cache_gb{0};
    int         search_L{0};
    int         insert_L{0};
    int         K{0};
    std::string notes;         // Free-form k=v;k=v; must not contain newlines or commas outside k=v pairs
};

// Thread-safe row writer. Flushes after every row so killed runs still yield usable CSV.
class CsvWriter {
 public:
    // Opens the file and writes the header. Throws on I/O failure.
    explicit CsvWriter(const std::string& path);
    ~CsvWriter();

    void write_row(const CsvRow& r);

 private:
    std::FILE* fp_{nullptr};
    std::mutex mu_;
};

}  // namespace bench
