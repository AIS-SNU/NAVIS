#pragma once

#include "utils.h"
#include <immintrin.h>
#include <sstream>
#include <string_view>
#include <vector>

#define NUM_PQ_CENTROIDS 256
#define NUM_PQ_OFFSETS 5

namespace pipeann {
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    //    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    float *tables = nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    float *centroid = nullptr;
    _u64 ndims;  // ndims = chunk_size * n_chunks
    _u64 n_chunks;
    _u32 *chunk_offsets = nullptr;
    _u32 *rearrangement = nullptr;
    float *tables_T = nullptr;  // same as pq_tables, but col-major
    float *all_to_all_dists = nullptr;

   public:
    uint64_t all_to_all_dist_size() {
      return sizeof(float) * n_chunks * NUM_PQ_CENTROIDS * NUM_PQ_CENTROIDS;
    }

    FixedChunkPQTable() {
    }

    virtual ~FixedChunkPQTable() {
      if (tables != nullptr)
        delete[] tables;
      if (tables_T != nullptr)
        delete[] tables_T;
      if (rearrangement != nullptr)
        delete[] rearrangement;
      if (chunk_offsets != nullptr)
        delete[] chunk_offsets;
      if (centroid != nullptr)
        delete[] centroid;
      if (all_to_all_dists != nullptr)
        delete[] all_to_all_dists;
    }

    _u64 get_dim() {
      return ndims;
    }

    void load_pq_pivots_new(std::basic_istream<char> &reader, size_t num_chunks, size_t offset) {
      _u64 nr, nc;
      std::unique_ptr<_u64[]> file_offset_data;
      _u64 *file_offset_data_raw;
      pipeann::load_bin_impl<_u64>(reader, file_offset_data_raw, nr, nc, offset);
      file_offset_data.reset(file_offset_data_raw);

      if (nr != NUM_PQ_OFFSETS) {
        LOG(ERROR) << "Pivot offset incorrect, # offsets = " << nr << ", but expecting " << NUM_PQ_OFFSETS;
        crash();
      }

      pipeann::load_bin_impl<float>(reader, tables, nr, nc, file_offset_data[0] + offset);

      if ((nr != NUM_PQ_CENTROIDS)) {
        LOG(ERROR) << "Num centers incorrect, centers = " << nr << " but expecting " << NUM_PQ_CENTROIDS;
        crash();
      }

      this->ndims = nc;
      pipeann::load_bin_impl<float>(reader, centroid, nr, nc, file_offset_data[1] + offset);

      if ((nr != this->ndims) || (nc != 1)) {
        LOG(ERROR) << "Centroid file dim incorrect: row " << nr << ", col " << nc << " expecting " << this->ndims;
        crash();
      }

      pipeann::load_bin_impl<uint32_t>(reader, rearrangement, nr, nc, file_offset_data[2] + offset);
      if ((nr != this->ndims) || (nc != 1)) {
        LOG(ERROR) << "Rearrangement incorrect: row " << nr << ", col " << nc << " expecting " << this->ndims;
        crash();
      }

      pipeann::load_bin_impl<uint32_t>(reader, chunk_offsets, nr, nc, file_offset_data[3] + offset);

      if (nr != (uint64_t) num_chunks + 1 || nc != 1) {
        LOG(ERROR) << "Chunk offsets: nr=" << nr << ", nc=" << nc << ", expecting nr=" << num_chunks + 1 << ", nc=1.";
        crash();
      }

      this->n_chunks = num_chunks;
      LOG(INFO) << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS << ", #dims: " << this->ndims
                << ", #chunks: " << this->n_chunks;
    }

    void post_load_pq_table() {
      // alloc and compute transpose
      pipeann::alloc_aligned((void **) &tables_T, 256 * ndims * sizeof(float), 64);
      // tables_T = new float[256 * ndims];
      for (_u64 i = 0; i < 256; i++) {
        for (_u64 j = 0; j < ndims; j++) {
          tables_T[j * 256 + i] = tables[i * ndims + j];
        }
      }

      // added this for easy PQ-PQ squared-distance calculations
      // TODO: Create only for StreamingMerger.
      all_to_all_dists = new float[256 * 256 * n_chunks];
      std::memset(all_to_all_dists, 0, 256 * 256 * n_chunks * sizeof(float));
      // should perhaps optimize later
      for (_u32 i = 0; i < 256; i++) {
        for (_u32 j = 0; j < 256; j++) {
          for (_u32 c = 0; c < n_chunks; c++) {
            for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
              float diff = (tables[i * ndims + d] - tables[j * ndims + d]);
              all_to_all_dists[i * 256 * n_chunks + j * n_chunks + c] += diff * diff;
            }
          }
        }
      }
    }

    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks, size_t offset = 0) {
      std::string pq_pivots_path(pq_table_file);
      _u64 nr, nc;

      get_bin_metadata(pq_table_file, nr, nc, offset);
      std::ifstream reader(pq_table_file, std::ios::binary | std::ios::ate);
      reader.seekg(0);
      load_pq_pivots_new(reader, num_chunks, offset);
      post_load_pq_table();
      LOG(INFO) << "Finished optimizing for PQ-PQ distance compuation";
    }

    void populate_chunk_distances(const T *query_vec, float *dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64 permuted_dim_in_query = rearrangement[j];
          const float *centers_dim_vec = tables_T + (256 * j);
          for (_u64 idx = 0; idx < 256; idx++) {
            double diff = centers_dim_vec[idx] - (query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query]);
            chunk_dists[idx] += (float) (diff * diff);
          }
        }
      }
    }

    void populate_chunk_distances_nt(const T *query_vec, float *dist_vec) {
#ifdef USE_AVX512
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64 permuted_dim_in_query = rearrangement[j];
          float *centers_dim_vec = tables_T + (256 * j);
          for (_u64 idx = 0; idx < 256; idx += 16) {
            __m512i center_i = _mm512_stream_load_si512(centers_dim_vec + idx);  // avoid cache thrashing
            __m512 center_f = _mm512_castsi512_ps(center_i);
            __m512 query_f = _mm512_set1_ps(query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query]);
            __m512 diff = _mm512_sub_ps(center_f, query_f);
            __m512 diff_sq = _mm512_mul_ps(diff, diff);
            __m512 chunk_dists_v = _mm512_load_ps(chunk_dists + idx);
            chunk_dists_v = _mm512_add_ps(chunk_dists_v, diff_sq);
            _mm512_store_ps(chunk_dists + idx, chunk_dists_v);  // dist_vec should be in cache.
          }
        }
      }
#else
      // TODO: for non-AVX512 machines, do non-temporal population.
      return populate_chunk_distances(query_vec, dist_vec);
#endif
    }

    // computes PQ distance between comp_src and comp_dsts in efficient manner
    // comp_src: [nchunks]
    // comp_dsts: count * [nchunks]
    // dists: [count]
    // TODO (perf) :: re-order computation to get better locality
    void compute_distances_alltoall(const _u8 *comp_src, const _u8 *comp_dsts, float *dists, const _u32 count) {
      std::memset(dists, 0, count * sizeof(float));
      for (_u64 i = 0; i < count; i++) {
        for (_u64 c = 0; c < n_chunks; c++) {
          dists[i] +=
              all_to_all_dists[(_u64) comp_src[c] * 256 * n_chunks + (_u64) comp_dsts[i * n_chunks + c] * n_chunks + c];
        }
      }
    }

    // fp_vec: [ndims]
    // out_pq_vec : [nchunks]
    void deflate_vec(const float *fp_vec, _u8 *out_pq_vec) {
      // permute the vector according to PQ rearrangement, compute all distances
      // to 256 centroids and choose the closest (for each chunk)
      for (_u32 c = 0; c < n_chunks; c++) {
        float closest_dist = std::numeric_limits<float>::max();
        for (_u32 i = 0; i < 256; i++) {
          float cur_dist = 0;
          for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
            float diff = (tables[i * ndims + d] - ((float) fp_vec[rearrangement[d]] - centroid[rearrangement[d]]));
            cur_dist += diff * diff;
          }
          if (cur_dist < closest_dist) {
            closest_dist = cur_dist;
            out_pq_vec[c] = (_u8) i;
          }
        }
      }
    }

    // Reconstruct approximate vector from PQ codes
    // pq_ids: [n_chunks]
    // out_vec: [ndims] (pre-allocated)
    void inflate_vector(const _u8 *pq_ids, T *out_vec) {
      // Start with global centroid
      for (_u64 i = 0; i < ndims; i++) {
        out_vec[i] = (T)centroid[i];
      }

      // Add residuals from PQ centroids
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        _u8 code = pq_ids[chunk];
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64 perm = rearrangement[j];
          // tables_T is [ndims][256] flattened to [ndims * 256]
          // Value for dim j, centroid code is at index (j * 256 + code)
          float val = tables_T[j * 256 + code];
          out_vec[perm] += (T)val;
        }
      }
    }

    // Reconstruct approximate vectors from PQ codes in a batch
    // pq_ids_batch: vector of [n_chunks]
    // out_vecs_batch: vector of [ndims] (pre-allocated)
    void inflate_vector_batch(const std::vector<const _u8*>& pq_ids_batch, 
                              const std::vector<T*>& out_vecs_batch) {
      size_t batch_size = pq_ids_batch.size();
      if (batch_size == 0) return;

      // 1. Initialize all output vectors with centroid
      for (size_t k = 0; k < batch_size; k++) {
          for (_u64 i = 0; i < ndims; i++) {
             out_vecs_batch[k][i] = (T)centroid[i];
          }
      }

      // 2. Iterate through chunks and dimensions once to maximize cache hits on tables_T
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
         for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
            _u64 perm = rearrangement[j];
            // tables_T is [ndims][256]
            // Pointer to the start of the 256 values for dimension j
            const float* lookup_row = tables_T + (j * 256);

            for (size_t k = 0; k < batch_size; k++) {
                _u8 code = pq_ids_batch[k][chunk];
                // Add the residual value
                out_vecs_batch[k][perm] += (T)lookup_row[code];
            }
         }
      }
    }
  };  // namespace pipeann
}  // namespace pipeann