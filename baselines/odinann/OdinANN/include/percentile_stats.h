#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace pipeann {
  struct QueryStats {
    double total_us = 0;        // total time to process query in micros
    double n_4k = 0;            // # of 4kB reads
    double n_8k = 0;            // # of 8kB reads
    double n_12k = 0;           // # of 12kB reads
    double n_ios = 0;           // total # of IOs issued
    double read_size = 0;       // total # of bytes read
    double io_us = 0;           // total time spent in IO
    double io_us1 = 0;          // total time spent in IO
    double head_us = 0;         // total time spent in in-memory index
    double cpu_us = 0;          // total time spent in CPU
    double cpu_us1 = 0;         // total time spent in CPU
    double cpu_us2 = 0;         // total time spent in CPU
    double n_cmps_saved = 0;    // # cmps saved
    double n_cmps = 0;          // # cmps
    double n_cache_hits = 0;    // # cache_hits
    double n_hops = 0;          // # search hops
    double n_hops_front = 0;    // # search hops (front)
    double n_hops_pos = 0;      // # search hops (position seeking)
    double n_current_used = 0;  // # force return for latency limit
  };

  inline double get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                                     const std::function<double(const QueryStats &)> &member_fn) {
    std::vector<double> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const double &left, const double &right) { return left < right; });

    auto retval = vals[(uint64_t) (percentile * ((float) len))];
    vals.clear();
    return retval;
  }

  inline double get_mean_stats(QueryStats *stats, uint64_t len,
                               const std::function<double(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) len);
  }
}  // namespace pipeann
