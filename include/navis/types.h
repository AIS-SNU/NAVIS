#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>
#include <string>

namespace navis {

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using _u8  = uint8_t;
using _s8  = int8_t;
using _u16 = uint16_t;
using _u32 = uint32_t;
using _s32 = int32_t;
using _u64 = uint64_t;
using _s64 = int64_t;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr uint64_t SECTOR_LEN          = 4096;
inline constexpr uint64_t EDGE_SECTOR_LEN     = 4096;
inline constexpr uint64_t META_SECTOR_LEN     = 4096;
inline constexpr uint32_t NUM_PQ_CENTROIDS    = 256;
inline constexpr uint32_t MAX_PQ_CHUNKS       = 128;
inline constexpr uint32_t MAX_N_SECTOR_READS  = 512;
inline constexpr uint32_t MAX_N_CMPS          = 16384;
inline constexpr uint32_t MAX_N_EDGES         = 512;

// ---------------------------------------------------------------------------
// Alignment utilities
// ---------------------------------------------------------------------------
inline constexpr uint64_t round_up(uint64_t x, uint64_t y) {
    return ((x / y) + (x % y != 0)) * y;
}

inline constexpr uint64_t div_round_up(uint64_t x, uint64_t y) {
    return (x / y) + (x % y != 0);
}

inline constexpr bool is_aligned(uint64_t x, uint64_t y) {
    return (x % y) == 0;
}

inline constexpr bool is_sector_aligned(uint64_t x) {
    return is_aligned(x, SECTOR_LEN);
}

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------
enum class Metric { L2 = 0, INNER_PRODUCT = 1, COSINE = 2 };

Metric metric_from_string(const std::string& s);

// ---------------------------------------------------------------------------
// IORequest — one sector-aligned read or write
// ---------------------------------------------------------------------------
struct IORequest {
    uint64_t offset = 0;       // disk byte offset (sector-aligned)
    uint64_t len    = 0;       // transfer size (sector-aligned)
    void*    buf    = nullptr; // buffer (must be 4KB-aligned for O_DIRECT)

    IORequest() = default;
    IORequest(uint64_t off, uint64_t l, void* b)
        : offset(off), len(l), buf(b) {}
};

// ---------------------------------------------------------------------------
// Neighbor — graph node with distance
// ---------------------------------------------------------------------------
struct Neighbor {
    uint32_t id       = 0;
    float    distance = 0.0f;
    bool     flag     = false;   // unexpanded marker for beam search

    Neighbor() = default;
    Neighbor(uint32_t i, float d, bool f = false)
        : id(i), distance(d), flag(f) {}

    bool operator<(const Neighbor& o) const {
        return (distance < o.distance) ||
               (distance == o.distance && id < o.id);
    }
    bool operator==(const Neighbor& o) const { return id == o.id; }
    bool operator>(const Neighbor& o) const { return o < *this; }
};

// ---------------------------------------------------------------------------
// QueryStats — optional per-query counters filled by NavisIndex::search
// ---------------------------------------------------------------------------
struct QueryStats {
    uint64_t n_hops      = 0;   // traversal hops
    uint64_t n_ios       = 0;   // storage reads issued (edge pages + vectors)
    uint64_t n_cmps      = 0;   // PQ distance computations
    uint64_t n_vec_loads = 0;   // full vectors loaded by CASR reranking
};

// ---------------------------------------------------------------------------
// SearchConfig — runtime parameters for search and insertion
// ---------------------------------------------------------------------------
struct SearchConfig {
    // Windowed CASR (see navis/casr.h). Reranking stops once the exact
    // top-kCasrTopK is unchanged for w consecutive vector loads. The first
    // `warmup_queries` reranks of each path (search, position seeking) load
    // every candidate; w is a percentile of the smallest windows that would
    // not have stopped them early, chosen per path. With warmup_queries == 0
    // the window is never calibrated (full rerank).
    uint32_t warmup_queries         = 1000;
    uint32_t casr_percentile_search = 100;
    uint32_t casr_percentile_insert = 90;

    // NAVIS-Update: the entrance graph grows while |G_ent| < ent_ratio * |G|.
    float    ent_ratio       = 0.01f;
    uint32_t ent_R           = 32;     // entrance-graph max degree

    // Position-seeking frontier size |E_pos| (insertion).
    uint32_t insert_L        = 100;
};

inline constexpr uint32_t kCasrTopK = 10;   // top-K whose convergence CASR tracks

// ---------------------------------------------------------------------------
// CacheConfig — NAVIS-Cache parameters
// ---------------------------------------------------------------------------
struct CacheConfig {
    uint64_t size_gb           = 16;    // total cache size (GiB)
    float    window_ratio      = 0.10f; // admission window share (rest: frozen)
    uint32_t promote_threshold = 2;     // window accesses needed for promotion
};

} // namespace navis
