// baselines/common/harness/rss_sampler.h
// Read VmRSS (current resident set size, KiB → MiB) from /proc/self/status.
// Called once per cadence tick by BenchHarness. Cheap: ~1 read + ~10 strncmp.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace bench {

// Returns VmRSS in MiB. Returns 0 on failure.
inline uint64_t read_rss_mib() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            std::sscanf(line + 6, "%lu", &kb);
            break;
        }
    }
    std::fclose(f);
    return kb / 1024;  // MiB
}

}  // namespace bench
