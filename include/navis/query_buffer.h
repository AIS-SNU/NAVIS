#pragma once

#include "navis/types.h"
#include "navis/separated_layout.h"
#include <tsl/robin_set.h>
#include <cstring>

namespace navis {

// ---------------------------------------------------------------------------
// QueryBuffer — per-thread scratch space for search/insert operations
//
//   edge_scratch: MAX_N_SECTOR_READS * 4KB slots (edge pages of one hop)
//   vec_scratch : MAX_N_SECTOR_READS * vec_read_stride bytes — one aligned
//                 vector read per CASR candidate (a vector may straddle two
//                 sectors, so stride = round_up(max_vec_len) + SECTOR_LEN).
// ---------------------------------------------------------------------------
template <typename T>
struct QueryBuffer {
    // --- Aligned memory for O_DIRECT I/O ---
    char* edge_scratch           = nullptr;  // [MAX_N_SECTOR_READS * SECTOR_LEN]
    char* vec_scratch            = nullptr;  // [MAX_N_SECTOR_READS * vec_read_stride]

    // --- Regular memory (CPU only) ---
    T*       coord_scratch     = nullptr;  // [MAX_N_CMPS * aligned_dim]
    float*   pq_dist_scratch   = nullptr;  // [NUM_PQ_CENTROIDS * MAX_PQ_CHUNKS]
    uint8_t* pq_coord_scratch  = nullptr;  // [MAX_PQ_CHUNKS * MAX_N_EDGES]
    float*   dist_scratch      = nullptr;  // [MAX_N_EDGES]
    T*       aligned_query     = nullptr;  // [aligned_dim]

    // --- Visited tracking ---
    tsl::robin_set<uint32_t> visited;

    // --- Per-query counters ---
    uint64_t coord_idx     = 0;
    uint64_t edge_slot_idx = 0;

    void reset() {
        coord_idx     = 0;
        edge_slot_idx = 0;
        visited.clear();
    }
};

} // namespace navis
