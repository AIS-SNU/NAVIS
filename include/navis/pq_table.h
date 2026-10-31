#pragma once

#include "navis/types.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <immintrin.h>

namespace navis {

// ---------------------------------------------------------------------------
// FixedChunkPQTable — Product Quantization table
//
// Splits vectors into n_chunks subspaces, each quantized to 256 centroids.
// Supports:
//   - PQ distance computation (query-to-all-centroids per chunk)
//   - Deflation (float vector -> PQ codes)
//   - Inflation (PQ codes -> approximate float vector)
//   - PQ-to-PQ distance lookup
// ---------------------------------------------------------------------------
template <typename T>
class FixedChunkPQTable {
public:
    FixedChunkPQTable() = default;
    ~FixedChunkPQTable();

    // Non-copyable
    FixedChunkPQTable(const FixedChunkPQTable&) = delete;
    FixedChunkPQTable& operator=(const FixedChunkPQTable&) = delete;

    // Load PQ pivots from binary file
    // Format: [offsets][tables (256 x ndims)][centroid (ndims)][rearrangement (ndims)][chunk_offsets (n_chunks+1)]
    void load(const std::string& pq_pivots_path, uint64_t num_chunks);

    // Compute distances from query to all 256 centroids for each chunk
    // Output: dist_out[256 * n_chunks] — chunk-major layout
    void populate_chunk_distances(const T* query, float* dist_out) const;

    // Compress a float vector into PQ codes (one uint8 per chunk)
    void deflate_vec(const float* fp_vec, uint8_t* pq_out) const;

    // Reconstruct approximate vector from PQ codes
    void inflate_vector(const uint8_t* pq_ids, T* out_vec) const;

    // Batch inflate for better cache locality
    void inflate_vector_batch(const uint8_t* pq_ids, T* out_vecs,
                              uint64_t n_vecs) const;

    // Accessors
    uint64_t get_ndims() const { return ndims_; }
    uint64_t get_n_chunks() const { return n_chunks_; }

private:
    float*    tables_     = nullptr;  // [256 * ndims] row-major: centroid i, dim j
    float*    tables_T_   = nullptr;  // [ndims * 256] col-major: dim j, centroid i
    float*    centroid_   = nullptr;  // [ndims] global centroid
    uint32_t* rearrange_  = nullptr;  // [ndims] dimension permutation
    uint32_t* chunk_off_  = nullptr;  // [n_chunks + 1] dimension ranges per chunk
    uint64_t  ndims_      = 0;
    uint64_t  n_chunks_   = 0;

    void build_tables_T();
};

// ---------------------------------------------------------------------------
// PQ distance lookup — used during graph traversal
// ---------------------------------------------------------------------------

// Gather PQ codes for a batch of node IDs from the global PQ data array.
// Prefetches 2 entries ahead to overlap DRAM latency (~100ns) for random
// accesses into the multi-GB PQ array.
inline void aggregate_pq_coords(const uint32_t* ids, uint64_t n_ids,
                                const uint8_t* all_pq_data, uint64_t n_chunks,
                                uint8_t* out) {
    for (uint64_t i = 0; i < n_ids; i++) {
        if (i + 2 < n_ids)
            __builtin_prefetch(all_pq_data + static_cast<uint64_t>(ids[i + 2]) * n_chunks);
        std::memcpy(out + i * n_chunks,
                    all_pq_data + static_cast<uint64_t>(ids[i]) * n_chunks,
                    n_chunks);
    }
}

// Compute PQ distances for a batch of points using pre-computed chunk distances.
//
// This is a gather-indirect loop: `chunk_dists[pq_ids[...]]`. The gather
// pattern doesn't vectorize cleanly on Zen 4 (vpgatherdd is microcoded
// and slower than scalar + prefetch), so the loop stays scalar and
// prefetches the next chunk's distance table (1 KB = 16 cache lines) with
// NTA hints so it does not evict useful data.
//
// pq_ids:    [n_pts * n_chunks] — PQ codes for each point
// pq_dists:  [256 * n_chunks]   — query-to-centroid distances
// dists_out: [n_pts]            — output approximate distances
inline void pq_dist_lookup(const uint8_t* pq_ids, uint64_t n_pts,
                           uint64_t n_chunks, const float* pq_dists,
                           float* dists_out) {
#if defined(__AVX512F__) || defined(__SSE__)
    // Prefetch the outputs + first chunks of code/table data up front.
    __builtin_prefetch(dists_out);
    __builtin_prefetch(pq_ids);
    __builtin_prefetch(pq_ids + 64);
    __builtin_prefetch(pq_dists);
    __builtin_prefetch(pq_dists + 64);
    __builtin_prefetch(pq_dists + 128);
    __builtin_prefetch(pq_dists + 192);
#endif
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
        const float* chunk_dists = pq_dists + 256 * chunk;
        if (chunk + 1 < n_chunks) {
            const float* next_chunk_dists = pq_dists + 256 * (chunk + 1);
            __builtin_prefetch(next_chunk_dists);
            __builtin_prefetch(next_chunk_dists + 64);
            __builtin_prefetch(next_chunk_dists + 128);
            __builtin_prefetch(next_chunk_dists + 192);
        }
        if (chunk == 0) {
            for (uint64_t idx = 0; idx < n_pts; idx++)
                dists_out[idx] = chunk_dists[pq_ids[idx * n_chunks]];
        } else {
            for (uint64_t idx = 0; idx < n_pts; idx++)
                dists_out[idx] += chunk_dists[pq_ids[idx * n_chunks + chunk]];
        }
    }
}

// Single-point PQ distance (returns sum of chunk distances)
inline float pq_dist_single(const uint8_t* pq_id, uint64_t n_chunks,
                            const float* pq_dists) {
    float dist = 0.0f;
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
        dist += pq_dists[256 * chunk + pq_id[chunk]];
    }
    return dist;
}

} // namespace navis
