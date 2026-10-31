#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>

#include "cached_io.h"
#include "index.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <tsl/robin_map.h>

#include <cassert>
#include "partition_and_pq.h"

#define MAX_BLOCK_SIZE 16384  // 64MB for 1024-dim float vectors, 2MB for 128-dim uint8 vectors.

template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset) {
  std::ifstream base_reader(base_file.c_str());
  base_reader.seekg(offset, std::ios::beg);

  std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
  std::ofstream sample_id_writer(std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  auto x = rd();
  std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  size_t npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));
  LOG(INFO) << "Loading base " << base_file << ". #points: " << npts_u32 << ". #dim: " << nd_u32 << ".";
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

  npts = npts_u32;
  nd = nd_u32;
  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

  for (size_t i = 0; i < npts; i++) {
    float sample = distribution(generator);
    if (sample < (float) sampling_rate) {
      base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
      sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
      uint32_t cur_i_u32 = (_u32) i;
      sample_id_writer.write((char *) &cur_i_u32, sizeof(uint32_t));
      num_sampled_pts_u32++;
    } else {
      base_reader.seekg(sizeof(T) * nd, base_reader.cur);  // skip this vector
    }
  }

  if (num_sampled_pts_u32 == 0) {
    // We have read something from file, so write it.
    sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
    num_sampled_pts_u32 = 1;
  }
  sample_writer.seekp(0, std::ios::beg);
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.seekp(0, std::ios::beg);
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.close();
  sample_id_writer.close();
  LOG(INFO) << "Wrote " << num_sampled_pts_u32 << " points to sample file: " << output_prefix + "_data.bin";
}

// Overload with max_points to limit sampling to first N vectors
template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset, size_t max_points) {
  std::ifstream base_reader(base_file.c_str());
  base_reader.seekg(offset, std::ios::beg);

  std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
  std::ofstream sample_id_writer(std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

  std::random_device rd;
  auto x = rd();
  std::mt19937 generator(x);
  std::uniform_real_distribution<float> distribution(0, 1);

  size_t npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));

  // Limit to max_points if specified and smaller than file size
  size_t actual_npts = (max_points > 0 && max_points < npts_u32) ? max_points : npts_u32;

  LOG(INFO) << "Loading base " << base_file << ". #points in file: " << npts_u32
            << ". Sampling from first " << actual_npts << " points. #dim: " << nd_u32 << ".";

  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

  npts = actual_npts;  // Use limited number
  nd = nd_u32;
  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

  for (size_t i = 0; i < npts; i++) {
    float sample = distribution(generator);
    if (sample < (float) sampling_rate) {
      base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
      sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
      uint32_t cur_i_u32 = (_u32) i;
      sample_id_writer.write((char *) &cur_i_u32, sizeof(uint32_t));
      num_sampled_pts_u32++;
    } else {
      base_reader.seekg(sizeof(T) * nd, base_reader.cur);
    }
  }

  if (num_sampled_pts_u32 == 0) {
    // Ensure at least one sample
    base_reader.seekg(8, std::ios::beg);  // Skip header
    base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
    sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
    num_sampled_pts_u32 = 1;
  }

  sample_writer.seekp(0, std::ios::beg);
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.seekp(0, std::ios::beg);
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.close();
  sample_id_writer.close();
  LOG(INFO) << "Wrote " << num_sampled_pts_u32 << " points (from first " << actual_npts
            << ") to sample file: " << output_prefix + "_data.bin";
}

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                      size_t &slice_size, size_t &ndims) {
  float *sampled_ptr = sampled_data.get();
  gen_random_slice<T>(data_file, p_val, sampled_ptr, slice_size, ndims);
  sampled_data.reset(sampled_ptr);
}
template<typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims) {
  size_t npts;
  uint32_t npts32, ndims32;
  std::vector<std::vector<float>> sampled_vectors;

  // amount to read in one shot
  _u64 read_blk_size = 64 * 1024 * 1024;
  std::ifstream base_reader(data_file.c_str());

  // metadata: npts, ndims
  base_reader.read((char *) &npts32, sizeof(unsigned));
  base_reader.read((char *) &ndims32, sizeof(unsigned));
  npts = npts32;
  ndims = ndims32;

  std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
  p_val = p_val < 1 ? p_val : 1;

  std::random_device rd;  // Will be used to obtain a seed for the random number
  size_t x = rd();
  std::mt19937 generator((unsigned) x);
  std::uniform_real_distribution<float> distribution(0, 1);

  for (size_t i = 0; i < npts; i++) {
    float rnd_val = distribution(generator);
    if (rnd_val < (float) p_val) {
      base_reader.read((char *) cur_vector_T.get(), ndims * sizeof(T));
      std::vector<float> cur_vector_float;
      for (size_t d = 0; d < ndims; d++)
        cur_vector_float.push_back(cur_vector_T[d]);
      sampled_vectors.push_back(cur_vector_float);
    } else {
      base_reader.seekg(ndims * sizeof(T), base_reader.cur);  // skip this vector
    }
  }
  slice_size = sampled_vectors.size();
  if (slice_size == 0) {
    slice_size = 1;
    std::vector<float> cur_vector_float(cur_vector_T.get(), cur_vector_T.get() + ndims);
    sampled_vectors.push_back(cur_vector_float);
  }
  sampled_data = new float[slice_size * ndims];

  for (size_t i = 0; i < slice_size; i++) {
    for (size_t j = 0; j < ndims; j++) {
      sampled_data[i * ndims + j] = sampled_vectors[i][j];
    }
  }
}

// given training data in train_data of dimensions num_train * dim, generate PQ
// pivots using k-means algorithm to partition the co-ordinates into
// num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
// k-means in each chunk to compute the PQ pivots and stores in bin format in
// file pq_pivots_path as a s num_centers*dim floating point binary file
template<typename T>
int generate_pq_pivots(const std::unique_ptr<T[]> &passed_train_data, size_t num_train, unsigned dim,
                       unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
                       std::string pq_pivots_path) {
  std::unique_ptr<float[]> train_float = std::make_unique<float[]>(num_train * (size_t) (dim));
  float *flt_ptr = train_float.get();
  T *T_ptr = passed_train_data.get();

  for (_u64 i = 0; i < num_train; i++) {
    for (_u64 j = 0; j < (_u64) dim; j++) {
      flt_ptr[i * (_u64) dim + j] = (float) T_ptr[i * (_u64) dim + j];
    }
  }
  if (generate_pq_pivots(flt_ptr, num_train, dim, num_centers, num_pq_chunks, max_k_means_reps, pq_pivots_path) != 0)
    return -1;
  return 0;
}

int generate_pq_pivots(const float *passed_train_data, size_t num_train, unsigned dim, unsigned num_centers,
                       unsigned num_pq_chunks, unsigned max_k_means_reps, std::string pq_pivots_path) {
  if (num_pq_chunks > dim) {
    LOG(ERROR) << " Error: number of chunks more than dimension";
    return -1;
  }

  std::unique_ptr<float[]> train_data = std::make_unique<float[]>(num_train * dim);
  std::memcpy(train_data.get(), passed_train_data, num_train * dim * sizeof(float));

  for (uint64_t i = 0; i < num_train; i++) {
    for (uint64_t j = 0; j < dim; j++) {
      if (passed_train_data[i * dim + j] != train_data[i * dim + j])
        LOG(ERROR) << "error in copy";
    }
  }

  std::unique_ptr<float[]> full_pivot_data;

  // Calculate centroid and center the training data
  std::unique_ptr<float[]> centroid = std::make_unique<float[]>(dim);
  for (uint64_t d = 0; d < dim; d++) {
    centroid[d] = 0;
    for (uint64_t p = 0; p < num_train; p++) {
      centroid[d] += train_data[p * dim + d];
    }
    centroid[d] /= (float) num_train;
  }

  //  std::memset(centroid, 0 , dim*sizeof(float));

  for (uint64_t d = 0; d < dim; d++) {
    for (uint64_t p = 0; p < num_train; p++) {
      train_data[p * dim + d] -= centroid[d];
    }
  }

  std::vector<uint32_t> rearrangement;
  std::vector<uint32_t> chunk_offsets;

  size_t low_val = (size_t) std::floor((double) dim / (double) num_pq_chunks);
  size_t high_val = (size_t) std::ceil((double) dim / (double) num_pq_chunks);
  size_t max_num_high = dim - (low_val * num_pq_chunks);
  size_t cur_num_high = 0;
  size_t cur_bin_threshold = high_val;

  std::vector<std::vector<uint32_t>> bin_to_dims(num_pq_chunks);
  tsl::robin_map<uint32_t, uint32_t> dim_to_bin;
  std::vector<float> bin_loads(num_pq_chunks, 0);

  // Process dimensions not inserted by previous loop
  for (uint32_t d = 0; d < dim; d++) {
    if (dim_to_bin.find(d) != dim_to_bin.end())
      continue;
    auto cur_best = num_pq_chunks + 1;
    float cur_best_load = std::numeric_limits<float>::max();
    for (uint32_t b = 0; b < num_pq_chunks; b++) {
      if (bin_loads[b] < cur_best_load && bin_to_dims[b].size() < cur_bin_threshold) {
        cur_best = b;
        cur_best_load = bin_loads[b];
      }
    }
    bin_to_dims[cur_best].push_back(d);
    if (bin_to_dims[cur_best].size() == high_val) {
      cur_num_high++;
      if (cur_num_high == max_num_high)
        cur_bin_threshold = low_val;
    }
  }

  rearrangement.clear();
  chunk_offsets.clear();
  chunk_offsets.push_back(0);

  for (uint32_t b = 0; b < num_pq_chunks; b++) {
    for (auto p : bin_to_dims[b]) {
      rearrangement.push_back(p);
    }
    if (b > 0)
      chunk_offsets.push_back(chunk_offsets[b - 1] + (unsigned) bin_to_dims[b - 1].size());
  }
  chunk_offsets.push_back(dim);

  full_pivot_data.reset(new float[num_centers * dim]);

  // DEBUG ONLY
  double kmeans_time = 0.0, lloyds_time = 0.0, copy_time = 0.0;

  for (size_t i = 0; i < num_pq_chunks; i++) {
    size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];

    if (cur_chunk_size == 0)
      continue;
    std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
    std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_train * cur_chunk_size);
    std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_train);

    memset((void *) cur_pivot_data.get(), 0, num_centers * cur_chunk_size * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, 65536)
    for (int64_t j = 0; j < (_s64) num_train; j++) {
      std::memcpy(cur_data.get() + j * cur_chunk_size, train_data.get() + j * dim + chunk_offsets[i],
                  cur_chunk_size * sizeof(float));
    }
    auto end = std::chrono::high_resolution_clock::now();
    copy_time += std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    // kmeans::kmeanspp_selecting_pivots(cur_data.get(), num_train,
    // cur_chunk_size,
    //                                  cur_pivot_data.get(), num_centers);
    kmeans::selecting_pivots(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers);

    unsigned k_means_reps = max_k_means_reps;

    kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers, k_means_reps,
                       nullptr, closest_center.get());
    end = std::chrono::high_resolution_clock::now();
    kmeans_time += std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    if (num_train > 2 * num_centers) {
      kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers, max_k_means_reps,
                         NULL, closest_center.get());
    }
    end = std::chrono::high_resolution_clock::now();
    lloyds_time += std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (uint64_t j = 0; j < num_centers; j++) {
      std::memcpy(full_pivot_data.get() + j * dim + chunk_offsets[i], cur_pivot_data.get() + j * cur_chunk_size,
                  cur_chunk_size * sizeof(float));
    }
    end = std::chrono::high_resolution_clock::now();
    copy_time += std::chrono::duration<double>(end - start).count();
  }
  LOG(INFO) << "Kmeans time: " << kmeans_time << " Lloyds time: " << lloyds_time << " Copy time: " << copy_time;

  std::vector<size_t> cumul_bytes(5, 0);
  cumul_bytes[0] = METADATA_SIZE;
  cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data.get(),
                                                             (size_t) num_centers, dim, cumul_bytes[0]);
  cumul_bytes[2] = cumul_bytes[1] +
                   pipeann::save_bin<float>(pq_pivots_path.c_str(), centroid.get(), (size_t) dim, 1, cumul_bytes[1]);
  cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement.data(),
                                                                rearrangement.size(), 1, cumul_bytes[2]);
  cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets.data(),
                                                                chunk_offsets.size(), 1, cumul_bytes[3]);
  pipeann::save_bin<_u64>(pq_pivots_path.c_str(), cumul_bytes.data(), cumul_bytes.size(), 1, 0);

  LOG(INFO) << "Saved pq pivot data to " << pq_pivots_path << " of size " << cumul_bytes[cumul_bytes.size() - 1]
            << "B.";

  return 0;
}

// streams the base file (data_file), and computes the closest centers in each
// chunk to generate the compressed data_file and stores it in
// pq_compressed_vectors_path.
// If the numbber of centers is < 256, it stores as byte vector, else as 4-byte
// vector in binary format.
template<typename T>
int generate_pq_data_from_pivots(const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
                                 std::string pq_pivots_path, std::string pq_compressed_vectors_path, size_t offset) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
  _u32 npts32;
  _u32 basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

#ifdef SAVE_INFLATED_PQ
  std::string inflated_pq_file = pq_compressed_vectors_path + "_full.bin";
#endif

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);

  std::unique_ptr<float[]> full_pivot_data;
  std::unique_ptr<float[]> centroid;
  std::unique_ptr<uint32_t[]> rearrangement;
  std::unique_ptr<uint32_t[]> chunk_offsets;

  if (!file_exists(pq_pivots_path)) {
    LOG(INFO) << "ERROR: PQ k-means pivot file not found";
    crash();
  } else {
    _u64 nr, nc;
    std::unique_ptr<_u64[]> file_offset_data;

    pipeann::load_bin<_u64>(pq_pivots_path.c_str(), file_offset_data, nr, nc, 0);

    if (nr != 5) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path
                << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting 5.";
      crash();
    }

    pipeann::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data, nr, nc, file_offset_data[0]);

    if ((nr != num_centers) || (nc != dim)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_num_centers  = " << nr
                << ", file_dim = " << nc << " but expecting " << num_centers << " centers in " << dim << " dimensions.";
      crash();
    }

    pipeann::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc, file_offset_data[1]);

    if ((nr != dim) || (nc != 1)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr << ", file_cols = " << nc
                << " but expecting " << dim << " entries in 1 dimension.";
      crash();
    }

    pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement, nr, nc, file_offset_data[2]);

    if ((nr != dim) || (nc != 1)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr << ", file_cols = " << nc
                << " but expecting " << dim << " entries in 1 dimension.";
      crash();
    }

    pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc, file_offset_data[3]);

    if (nr != (uint64_t) num_pq_chunks + 1 || nc != 1) {
      LOG(INFO) << "Error reading pq_pivots file at chunk offsets; file has nr=" << nr << ",nc=" << nc
                << ", expecting nr=" << num_pq_chunks + 1 << ", nc=1.";
      crash();
    }

    LOG(INFO) << "Loaded PQ pivot information";
  }

  std::ofstream compressed_file_writer(pq_compressed_vectors_path, std::ios::binary);
  _u32 num_pq_chunks_u32 = num_pq_chunks;

  compressed_file_writer.write((char *) &num_points, sizeof(uint32_t));
  compressed_file_writer.write((char *) &num_pq_chunks_u32, sizeof(uint32_t));

#ifdef SAVE_INFLATED_PQ
  std::ofstream inflated_file_writer(inflated_pq_file, std::ios::binary);
  inflated_file_writer.write((char *) &npts32, sizeof(uint32_t));
  inflated_file_writer.write((char *) &basedim32, sizeof(uint32_t));

  std::unique_ptr<float[]> block_inflated_base = std::make_unique<float[]>(BLOCK_SIZE * (_u64) dim);
  std::memset(block_inflated_base.get(), 0, BLOCK_SIZE * (_u64) dim * sizeof(float));
#endif

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<_u32[]> block_compressed_base = std::make_unique<_u32[]>(block_size * (_u64) num_pq_chunks);
  std::memset(block_compressed_base.get(), 0, block_size * (_u64) num_pq_chunks * sizeof(uint32_t));

  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) (block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
    pipeann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

    for (uint64_t p = 0; p < cur_blk_size; p++) {
      for (uint64_t d = 0; d < dim; d++) {
        block_data_tmp[p * dim + d] -= centroid[d];
      }
    }

    for (uint64_t p = 0; p < cur_blk_size; p++) {
      for (uint64_t d = 0; d < dim; d++) {
        block_data_float[p * dim + d] = block_data_tmp[p * dim + rearrangement[d]];
      }
    }

    for (size_t i = 0; i < num_pq_chunks; i++) {
      size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
      if (cur_chunk_size == 0)
        continue;

      std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
      std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(cur_blk_size * cur_chunk_size);
      std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(cur_blk_size);

#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          cur_data[j * cur_chunk_size + k] = block_data_float[j * dim + chunk_offsets[i] + k];
      }

#pragma omp parallel for schedule(static, 1)
      for (int64_t j = 0; j < (_s64) num_centers; j++) {
        std::memcpy(cur_pivot_data.get() + j * cur_chunk_size, full_pivot_data.get() + j * dim + chunk_offsets[i],
                    cur_chunk_size * sizeof(float));
      }

      math_utils::compute_closest_centers(cur_data.get(), cur_blk_size, cur_chunk_size, cur_pivot_data.get(),
                                          num_centers, 1, closest_center.get());
#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        block_compressed_base[j * num_pq_chunks + i] = closest_center[j];
#ifdef SAVE_INFLATED_PQ
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          block_inflated_base[j * dim + chunk_offsets[i] + k] =
              cur_pivot_data[closest_center[j] * cur_chunk_size + k] + centroid[chunk_offsets[i] + k];
#endif
      }
    }

#ifdef SAVE_INFLATED_PQ
    inflated_file_writer.write((char *) block_inflated_base.get(), cur_blk_size * dim * sizeof(float));
#endif

    if (num_centers > 256) {
      compressed_file_writer.write((char *) (block_compressed_base.get()),
                                   cur_blk_size * num_pq_chunks * sizeof(uint32_t));
    } else {
      std::unique_ptr<uint8_t[]> pVec = std::make_unique<uint8_t[]>(cur_blk_size * num_pq_chunks);
      pipeann::convert_types<uint32_t, uint8_t>(block_compressed_base.get(), pVec.get(), cur_blk_size, num_pq_chunks);
      compressed_file_writer.write((char *) (pVec.get()), cur_blk_size * num_pq_chunks * sizeof(uint8_t));
    }
    // LOG(INFO) << ".done.";
  }
  // Splittng diskann_dll into separate DLLs for search and build.
  // This code should only be available in the "build" DLL.
  compressed_file_writer.close();
#ifdef SAVE_INFLATED_PQ
  inflated_file_writer.close();
#endif
  return 0;
}

template<typename T>
int estimate_cluster_sizes(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                           const size_t k_base, std::vector<size_t> &cluster_sizes) {
  cluster_sizes.clear();

  size_t num_test, test_dim;
  float *test_data_float;
  double sampling_rate = 0.01;

  gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test, test_dim);

  if (test_dim != dim) {
    LOG(INFO) << "Error. dimensions dont match for pivot set and base set";
    return -1;
  }

  size_t *shard_counts = new size_t[num_centers];

  for (size_t i = 0; i < num_centers; i++) {
    shard_counts[i] = 0;
  }

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_test);
  size_t num_points = 0, num_dim = 0;
  pipeann::get_bin_metadata(data_file, num_points, num_dim);
  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  _u32 *block_closest_centers = new _u32[block_size * k_base];
  float *block_data_float;

  size_t num_blocks = DIV_ROUND_UP(num_test, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_test);
    size_t cur_blk_size = end_id - start_id;

    block_data_float = test_data_float + start_id * test_dim;

    math_utils::compute_closest_centers(block_data_float, cur_blk_size, dim, pivots, num_centers, k_base,
                                        block_closest_centers);

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        shard_counts[shard_id]++;
      }
    }
  }

  LOG(INFO) << "Estimated cluster sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    cluster_sizes.push_back(size_t(((double) cur_shard_count) * (1.0 / sampling_rate)));
    std::cerr << cur_shard_count * (1.0 / sampling_rate) << " ";
  }
  std::cerr << "\n";
  delete[] shard_counts;
  delete[] block_closest_centers;
  return 0;
}

template<typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  //  _u64 write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32 npts32;
  _u32 basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    LOG(INFO) << "Error. dimensions dont match for train set and base set";
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
  std::vector<std::ofstream> shard_data_writer(num_centers);
  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  _u32 dummy_size = 0;
  _u32 const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string data_filename = prefix_path + "_subshard-" + std::to_string(i) + ".bin";
    std::string idmap_filename = prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_data_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_data_writer[i].write((char *) &basedim32, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);
  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<_u32[]> block_closest_centers = std::make_unique<_u32[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    pipeann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t) (start_id + p);
        shard_data_writer[shard_id].write((char *) (block_data_T.get() + p * dim), sizeof(T) * dim);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  LOG(INFO) << "Actual shard sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    total_count += cur_shard_count;
    LOG(INFO) << cur_shard_count << " ";
    shard_data_writer[i].seekp(0);
    shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_data_writer[i].close();
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  LOG(INFO) << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get " << total_count
            << " points across " << num_centers << " shards ";
  return 0;
}

template<typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;
  size_t max_k_means_reps = 20;

  int num_parts = 3;
  bool fit_in_ram = false;

  gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

  float *pivot_data = nullptr;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  while (!fit_in_ram) {
    fit_in_ram = true;

    double max_ram_usage = 0;
    if (pivot_data != nullptr)
      delete[] pivot_data;

    pivot_data = new float[num_parts * train_dim];
    // Process Global k-means for kmeans_partitioning Step
    LOG(INFO) << "Processing global k-means (kmeans_partitioning Step)";
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);

    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

    // now pivots are ready. need to stream base points and assign them to
    // closest clusters.

    std::vector<size_t> cluster_sizes;
    estimate_cluster_sizes<T>(data_file, pivot_data, num_parts, train_dim, k_base, cluster_sizes);

    for (auto &p : cluster_sizes) {
      double cur_shard_ram_estimate = pipeann::estimate_ram_usage(p, train_dim, sizeof(T), graph_degree);

      if (cur_shard_ram_estimate > max_ram_usage)
        max_ram_usage = cur_shard_ram_estimate;
    }
    LOG(INFO) << "With " << num_parts << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
              << "GB, budget given is " << ram_budget;
    if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget) {
      fit_in_ram = false;
      num_parts++;
    }
  }

  LOG(INFO) << "Saving global k-center pivots";
  pipeann::save_bin<float>(output_file.c_str(), pivot_data, (size_t) num_parts, train_dim);

  shard_data_into_clusters<T>(data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
  delete[] pivot_data;
  delete[] train_data_float;
  return num_parts;
}

// Instantations of supported templates
template void gen_random_slice<int8_t>(const std::string data_file, double p_val,
                                       std::unique_ptr<float[]> &sampled_data, size_t &slice_size, size_t &ndims);
template void gen_random_slice<uint8_t>(const std::string data_file, double p_val,
                                        std::unique_ptr<float[]> &sampled_data, size_t &slice_size, size_t &ndims);
template void gen_random_slice<float>(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                                      size_t &slice_size, size_t &ndims);

template void gen_random_slice<int8_t>(const std::string base_file, const std::string output_prefix,
                                       double sampling_rate, size_t offset);
template void gen_random_slice<uint8_t>(const std::string base_file, const std::string output_prefix,
                                        double sampling_rate, size_t offset);
template void gen_random_slice<float>(const std::string base_file, const std::string output_prefix,
                                      double sampling_rate, size_t offset);

template void gen_random_slice<int8_t>(const std::string base_file, const std::string output_prefix,
                                       double sampling_rate, size_t offset, size_t max_points);
template void gen_random_slice<uint8_t>(const std::string base_file, const std::string output_prefix,
                                        double sampling_rate, size_t offset, size_t max_points);
template void gen_random_slice<float>(const std::string base_file, const std::string output_prefix,
                                      double sampling_rate, size_t offset, size_t max_points);

template void gen_random_slice<float>(const std::string data_file, double p_val, float *&sampled_data,
                                      size_t &slice_size, size_t &ndims);
template void gen_random_slice<uint8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                        size_t &slice_size, size_t &ndims);
template void gen_random_slice<int8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                       size_t &slice_size, size_t &ndims);

template int partition_with_ram_budget<int8_t>(const std::string data_file, const double sampling_rate,
                                               double ram_budget, size_t graph_degree, const std::string prefix_path,
                                               size_t k_base);
template int partition_with_ram_budget<uint8_t>(const std::string data_file, const double sampling_rate,
                                                double ram_budget, size_t graph_degree, const std::string prefix_path,
                                                size_t k_base);
template int partition_with_ram_budget<float>(const std::string data_file, const double sampling_rate,
                                              double ram_budget, size_t graph_degree, const std::string prefix_path,
                                              size_t k_base);

template int generate_pq_pivots<float>(const std::unique_ptr<float[]> &passed_train_data, size_t num_train,
                                       unsigned dim, unsigned num_centers, unsigned num_pq_chunks,
                                       unsigned max_k_means_reps, std::string pq_pivots_path);
template int generate_pq_pivots<int8_t>(const std::unique_ptr<int8_t[]> &passed_train_data, size_t num_train,
                                        unsigned dim, unsigned num_centers, unsigned num_pq_chunks,
                                        unsigned max_k_means_reps, std::string pq_pivots_path);
template int generate_pq_pivots<uint8_t>(const std::unique_ptr<uint8_t[]> &passed_train_data, size_t num_train,
                                         unsigned dim, unsigned num_centers, unsigned num_pq_chunks,
                                         unsigned max_k_means_reps, std::string pq_pivots_path);

template int generate_pq_data_from_pivots<int8_t>(const std::string data_file, unsigned num_centers,
                                                  unsigned num_pq_chunks, std::string pq_pivots_path,
                                                  std::string pq_compressed_vectors_path, size_t offset);
template int generate_pq_data_from_pivots<uint8_t>(const std::string data_file, unsigned num_centers,
                                                   unsigned num_pq_chunks, std::string pq_pivots_path,
                                                   std::string pq_compressed_vectors_path, size_t offset);
template int generate_pq_data_from_pivots<float>(const std::string data_file, unsigned num_centers,
                                                 unsigned num_pq_chunks, std::string pq_pivots_path,
                                                 std::string pq_compressed_vectors_path, size_t offset);
