// baselines/common/harness/disk_index_probe.h
// Size of the files a run is modifying (its working copy of the index).

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <glob.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace bench {

// Sum of st_size over every file matching `pattern`. Missing paths → 0 addend.
inline uint64_t sum_glob_bytes(const std::string& pattern) {
    glob_t g;
    if (::glob(pattern.c_str(), 0, nullptr, &g) != 0) { ::globfree(&g); return 0; }
    uint64_t total = 0;
    struct stat st{};
    for (size_t i = 0; i < g.gl_pathc; ++i) {
        if (::stat(g.gl_pathv[i], &st) == 0) total += static_cast<uint64_t>(st.st_size);
    }
    ::globfree(&g);
    return total;
}

}  // namespace bench
