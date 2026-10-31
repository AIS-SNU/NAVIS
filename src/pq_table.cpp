#include "navis/pq_table.h"
#include "navis/util/file_utils.h"

#include <fstream>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace navis {

// The pivots file starts with 4 section offsets. Files written by DiskANN-
// family tools carry a 5th (end-of-file marker), which is accepted and ignored.
static constexpr uint64_t MIN_PQ_OFFSETS = 4;
static constexpr uint64_t MAX_PQ_OFFSETS = 5;

// ---------------------------------------------------------------------------
// Helper: load a typed array from a binary stream at a given byte offset
// Format at offset: [4B nrows] [4B ncols] [nrows * ncols * sizeof(T) data]
// ---------------------------------------------------------------------------
template <typename U>
static void load_bin_at(std::istream& in, U*& data, uint64_t& nr, uint64_t& nc, uint64_t offset) {
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    uint32_t nrows, ncols;
    in.read(reinterpret_cast<char*>(&nrows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&ncols), sizeof(uint32_t));
    nr = nrows;
    nc = ncols;
    uint64_t total = nr * nc;
    data = new U[total];
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(total * sizeof(U)));
    if (in.fail()) {
        delete[] data;
        data = nullptr;
        throw std::runtime_error("Failed to read binary data at offset " + std::to_string(offset));
    }
}

// ===========================================================================
// FixedChunkPQTable — implementation
// ===========================================================================

template <typename T>
FixedChunkPQTable<T>::~FixedChunkPQTable() {
    delete[] tables_;
    delete[] tables_T_;
    delete[] centroid_;
    delete[] rearrange_;
    delete[] chunk_off_;
}

template <typename T>
void FixedChunkPQTable<T>::load(const std::string& pq_pivots_path, uint64_t num_chunks) {
    std::ifstream in(pq_pivots_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open PQ pivots file: " + pq_pivots_path);
    }

    // 1. Read the section offset table.
    uint64_t* offsets = nullptr;
    uint64_t nr, nc;
    load_bin_at<uint64_t>(in, offsets, nr, nc, 0);
    if (nr < MIN_PQ_OFFSETS || nr > MAX_PQ_OFFSETS) {
        delete[] offsets;
        throw std::runtime_error("PQ pivots: expected " + std::to_string(MIN_PQ_OFFSETS) +
                                 "-" + std::to_string(MAX_PQ_OFFSETS) +
                                 " offsets, got " + std::to_string(nr));
    }

    // 2. Load centroids table [256 x ndims]
    load_bin_at<float>(in, tables_, nr, nc, offsets[0]);
    if (nr != NUM_PQ_CENTROIDS) {
        delete[] offsets;
        throw std::runtime_error("PQ pivots: expected 256 centroids, got " + std::to_string(nr));
    }
    ndims_ = nc;

    // 3. Load global centroid [ndims x 1]
    load_bin_at<float>(in, centroid_, nr, nc, offsets[1]);
    if (nr != ndims_ || nc != 1) {
        delete[] offsets;
        throw std::runtime_error("PQ pivots: centroid dimension mismatch");
    }

    // 4. Load rearrangement permutation [ndims x 1]
    load_bin_at<uint32_t>(in, rearrange_, nr, nc, offsets[2]);
    if (nr != ndims_ || nc != 1) {
        delete[] offsets;
        throw std::runtime_error("PQ pivots: rearrangement dimension mismatch");
    }

    // 5. Load chunk offsets [n_chunks+1 x 1]
    load_bin_at<uint32_t>(in, chunk_off_, nr, nc, offsets[3]);
    if (nr != num_chunks + 1 || nc != 1) {
        delete[] offsets;
        throw std::runtime_error("PQ pivots: chunk_offsets size mismatch, got " +
                                 std::to_string(nr) + ", expected " + std::to_string(num_chunks + 1));
    }

    n_chunks_ = num_chunks;
    delete[] offsets;

    // 6. Build transposed table for cache-friendly access
    build_tables_T();
}


template <typename T>
void FixedChunkPQTable<T>::build_tables_T() {
    // tables_: [256][ndims] row-major (centroid i at row i)
    // tables_T_: [ndims][256] — dim j, centroid i for vectorized access
    tables_T_ = new float[ndims_ * 256];
    for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < ndims_; j++) {
            tables_T_[j * 256 + i] = tables_[i * ndims_ + j];
        }
    }
}

// Hot path: called once per search/insert query (and per iteration in
// NBR_LOOP overflow), then consumed by ~R pq_dist_lookup calls. Compiler
// auto-vectorization over a loop-variant rearrange_[] + indirect query
// read is flaky, especially for gather fallback on Zen 4. Hand-written
// AVX-512 is ~3-4× faster than the scalar path at 768 dim / 128 chunks.
//
// Dimensions are visited through the `rearrange_[]` permutation written by
// the PQ build, and each dimension's contribution uses one fused
// multiply-add per centroid.
//
// For the INNER_PRODUCT metric the per-chunk formula is different
// (`acc - q * center`). We fall back to scalar for IP for now since our
// primary configs are L2; plumbing metric() through would add complexity
// without benefit for the FineWeb/ImageNet/MSMARCO workloads.
template <typename T>
void FixedChunkPQTable<T>::populate_chunk_distances(const T* query, float* dist_out) const {
    // Precompute per-dim query values: q_vals[j] = query[rearrange[j]] − centroid[rearrange[j]]
    std::vector<float> q_vals(ndims_);
    for (uint64_t j = 0; j < ndims_; j++) {
        uint64_t perm = rearrange_[j];
        q_vals[j] = static_cast<float>(query[perm]) - centroid_[perm];
    }
#if defined(__AVX512F__)
    for (uint64_t chunk = 0; chunk < n_chunks_; chunk++) {
        float* chunk_dists = dist_out + 256 * chunk;
        uint64_t first = chunk_off_[chunk];
        uint64_t last  = chunk_off_[chunk + 1];

        // First dim: init chunk_dists[*] = (center - q)^2.
        {
            const float* centers = tables_T_ + 256 * first;
            float q_val = q_vals[first];
            __m512 q_vec = _mm512_set1_ps(q_val);
            for (uint64_t idx = 0; idx < 256; idx += 16) {
                __m512 c  = _mm512_loadu_ps(centers + idx);
                __m512 d  = _mm512_sub_ps(c, q_vec);
                __m512 dd = _mm512_mul_ps(d, d);
                _mm512_storeu_ps(chunk_dists + idx, dd);
            }
        }
        // Subsequent dims: chunk_dists[*] += (center - q)^2 via fmadd.
        for (uint64_t j = first + 1; j < last; j++) {
            const float* centers = tables_T_ + 256 * j;
            float q_val = q_vals[j];
            __m512 q_vec = _mm512_set1_ps(q_val);
            for (uint64_t idx = 0; idx < 256; idx += 16) {
                __m512 c   = _mm512_loadu_ps(centers + idx);
                __m512 d   = _mm512_sub_ps(c, q_vec);
                __m512 acc = _mm512_loadu_ps(chunk_dists + idx);
                acc = _mm512_fmadd_ps(d, d, acc);
                _mm512_storeu_ps(chunk_dists + idx, acc);
            }
        }
    }
#else
    // Scalar fallback — identical semantics to the AVX-512 path.
    for (uint64_t chunk = 0; chunk < n_chunks_; chunk++) {
        float* chunk_dists = dist_out + 256 * chunk;
        uint64_t first = chunk_off_[chunk];
        for (uint64_t j = first; j < chunk_off_[chunk + 1]; j++) {
            const float* centers = tables_T_ + 256 * j;
            float q_val = q_vals[j];
            if (j == first) {
                for (uint64_t idx = 0; idx < 256; idx++) {
                    float diff = centers[idx] - q_val;
                    chunk_dists[idx] = diff * diff;
                }
            } else {
                for (uint64_t idx = 0; idx < 256; idx++) {
                    float diff = centers[idx] - q_val;
                    chunk_dists[idx] += diff * diff;
                }
            }
        }
    }
#endif
}

template <typename T>
void FixedChunkPQTable<T>::deflate_vec(const float* fp_vec, uint8_t* pq_out) const {
    for (uint64_t c = 0; c < n_chunks_; c++) {
        float best_dist = std::numeric_limits<float>::max();
        uint8_t best_id = 0;
        for (uint32_t i = 0; i < 256; i++) {
            float cur_dist = 0.0f;
            for (uint64_t d = chunk_off_[c]; d < chunk_off_[c + 1]; d++) {
                float diff = tables_[i * ndims_ + d] -
                             (fp_vec[rearrange_[d]] - centroid_[rearrange_[d]]);
                cur_dist += diff * diff;
            }
            if (cur_dist < best_dist) {
                best_dist = cur_dist;
                best_id = static_cast<uint8_t>(i);
            }
        }
        pq_out[c] = best_id;
    }
}

template <typename T>
void FixedChunkPQTable<T>::inflate_vector(const uint8_t* pq_ids, T* out_vec) const {
    // Start with global centroid
    for (uint64_t i = 0; i < ndims_; i++) {
        out_vec[i] = static_cast<T>(centroid_[i]);
    }
    // Add per-chunk residuals
    for (uint64_t chunk = 0; chunk < n_chunks_; chunk++) {
        uint8_t code = pq_ids[chunk];
        for (uint64_t j = chunk_off_[chunk]; j < chunk_off_[chunk + 1]; j++) {
            uint64_t perm = rearrange_[j];
            out_vec[perm] += static_cast<T>(tables_T_[j * 256 + code]);
        }
    }
}

template <typename T>
void FixedChunkPQTable<T>::inflate_vector_batch(const uint8_t* pq_ids, T* out_vecs,
                                                 uint64_t n_vecs) const {
    for (uint64_t v = 0; v < n_vecs; v++) {
        inflate_vector(pq_ids + v * n_chunks_, out_vecs + v * ndims_);
    }
}

// Explicit instantiations
template class FixedChunkPQTable<float>;
template class FixedChunkPQTable<int8_t>;
template class FixedChunkPQTable<uint8_t>;

} // namespace navis
