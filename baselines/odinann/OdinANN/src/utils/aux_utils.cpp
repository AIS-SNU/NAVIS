#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <cblas.h>

#include "aux_utils.h"
#include "cached_io.h"
#include "index.h"
#include "omp.h"
#include "partition_and_pq.h"
#include "percentile_stats.h"
#include "ssd_index.h"
#include "utils.h"

#include "ssd_index.h"
#include "tsl/robin_set.h"
#include "utils.h"

#define NUM_KMEANS 15

namespace pipeann {

  void add_new_file_to_single_index(std::string index_file, std::string new_file) {
    std::unique_ptr<_u64[]> metadata;
    _u64 nr, nc;
    pipeann::load_bin<_u64>(index_file, metadata, nr, nc, 0);
    if (nc != 1) {
      LOG(ERROR) << "Error, index file specified does not have correct metadata. ";
      crash();
    }
    size_t index_ending_offset = metadata[nr - 1];
    _u64 read_blk_size = 64 * 1024 * 1024;
    cached_ofstream writer(index_file, read_blk_size, index_ending_offset);
    _u64 check_file_size = get_file_size(index_file);
    if (check_file_size != index_ending_offset) {
      LOG(ERROR) << "Error, index file specified does not have correct metadata "
                    "(last entry must match the filesize). ";
      crash();
    }

    cached_ifstream reader(new_file, read_blk_size);
    size_t fsize = reader.get_file_size();
    if (fsize == 0) {
      LOG(ERROR) << "Error, new file specified is empty. Not appending.";
      crash();
    }

    size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
    char *dump = new char[read_blk_size];
    for (_u64 i = 0; i < num_blocks; i++) {
      size_t cur_block_size = read_blk_size > fsize - (i * read_blk_size) ? fsize - (i * read_blk_size) : read_blk_size;
      reader.read(dump, cur_block_size);
      writer.write(dump, cur_block_size);
    }
    reader.close();
    writer.close();

    delete[] dump;
    std::vector<_u64> new_meta;
    for (_u64 i = 0; i < nr; i++)
      new_meta.push_back(metadata[i]);
    new_meta.push_back(metadata[nr - 1] + fsize);

    pipeann::save_bin<_u64>(index_file, new_meta.data(), new_meta.size(), 1, 0);
  }

  double get_memory_budget(double search_ram_budget) {
    double final_index_ram_limit = search_ram_budget;
    if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by
                                                                                           // cached nodes
      final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
  }

  double get_memory_budget(const std::string &mem_budget_str) {
    double search_ram_budget = atof(mem_budget_str.c_str());
    return get_memory_budget(search_ram_budget);
  }

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim) {
    size_t num_pq_chunks = (size_t) (std::floor)(_u64(final_index_ram_limit / (double) points_num));

    LOG(INFO) << "Calculated num_pq_chunks :" << num_pq_chunks;
    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    LOG(INFO) << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector.";
    return num_pq_chunks;
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at) {
    double total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);

      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags) {
    double total_recall = 0;
    std::set<unsigned> gt, res;
    bool printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      unsigned active_points_count = 0;
      unsigned cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
          active_points_count++;
        }
        cur_counter++;
      }
      if (active_tags.empty())
        cur_counter = recall_at;

      if ((active_points_count < recall_at && !active_tags.empty()) && !printed) {
        LOG(INFO) << "Warning: Couldn't find enough closest neighbors " << active_points_count << "/" << recall_at
                  << " from truthset for query # " << i << ". Will result in under-reported value of recall.";
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((double) (total_recall / (num_queries))) * ((double) (100.0 / recall_at));
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t npts32, dim;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
      LOG(ERROR) << "Error reading idmap file. Check if the file is bin file with 1 dimensional data. Actual: "
                 << actual_file_size << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t);

      crash();
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << max_degree;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      LOG(INFO) << "Creating inverse map -- shard #" << shard;
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((_u32) node_id, (_u32) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
      return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    LOG(INFO) << "Finished computing node -> shards map";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
    }

    size_t merged_index_size = 24;
    size_t merged_index_frozen = 0;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    LOG(INFO) << "Max input width: " << max_input_width << ", output width: " << output_width;

    diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    _u32 nshards_u32 = (_u32) nshards;
    _u32 one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    _u64 vamana_index_frozen = 0;
    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(_u64));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    diskann_writer.write((char *) &merged_index_frozen, sizeof(_u64));
    medoid_writer.close();

    LOG(INFO) << "Starting merge";

    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          LOG(INFO) << cur_id << "...";
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(), shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    // random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    LOG(INFO) << "Expected size: " << merged_index_size;

    diskann_writer.reset();
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    LOG(INFO) << "Finished merge";
    return 0;
  }

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_file_index,
                                unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                const char *tag_file) {
    if (unlikely(single_file_index)) {
      LOG(INFO) << "Single file index is not supported for merged Vamana index, setting to false.";
      single_file_index = false;
    }

    size_t base_num, base_dim;
    pipeann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram = estimate_ram_usage(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 1);  // was 0 earlier.
      paras.Set<std::string>("save_path", mem_index_path);

      bool tags_enabled;
      if (tag_file == nullptr)
        tags_enabled = false;
      else
        tags_enabled = true;

      std::unique_ptr<pipeann::Index<T>> _pvamanaIndex = std::unique_ptr<pipeann::Index<T>>(
          new pipeann::Index<T>(_compareMetric, base_dim, base_num, false, single_file_index, tags_enabled));
      if (tags_enabled)
        _pvamanaIndex->build(base_file.c_str(), base_num, paras, tag_file);
      else
        _pvamanaIndex->build(base_file.c_str(), base_num, paras);

      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
    }

    if (single_file_index || tag_file != nullptr) {
      LOG(INFO) << "Cannot build merged index if single_file_index is "
                   "required or if tags are specified.";
      return 1;
    }

    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      pipeann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);

      _u64 shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      std::unique_ptr<pipeann::Index<T>> _pvamanaIndex = std::unique_ptr<pipeann::Index<T>>(
          new pipeann::Index<T>(_compareMetric, shard_base_dim, shard_base_pts, false,
                                single_file_index));  // TODO: Single?
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    pipeann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                          "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    return 0;
  }

  // if single_index format is true, we assume that the entire mem index is in
  // mem_index_file, and the entire disk index will be in output_file.
  template<typename T, typename TagT>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &pq_pivots_file, const std::string &pq_vectors_file, bool single_file_index,
                          const std::string &output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64 read_blk_size = 64 * 1024 * 1024;
    _u64 write_blk_size = read_blk_size;
    cached_ifstream base_reader;
    std::ifstream vamana_reader;
    _u64 base_offset = 0, vamana_offset = 0, tags_offset = 0;
    bool tags_enabled = false;

    base_reader.open(base_file, read_blk_size);
    vamana_reader.open(mem_index_file, std::ios::binary);
    tags_enabled = tag_file != "";

    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    //    size_t          actual_file_size = get_file_size(mem_index_file);
    std::remove(output_file.c_str());
    cached_ofstream diskann_writer;
    diskann_writer.open(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));

    _u64 vamana_frozen_num = false, vamana_frozen_loc = 0;
    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));
    vamana_reader.read((char *) &vamana_frozen_num, sizeof(_u64));
    // compute
    _u64 medoid, max_node_len, nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
    if (vamana_frozen_num == 1)
      vamana_frozen_loc = medoid;
    max_node_len = (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;  // 0 if max_node_len > SECTOR_LEN

    LOG(INFO) << "medoid: " << medoid << "B";
    LOG(INFO) << "max_node_len: " << max_node_len << "B";
    LOG(INFO) << "nnodes_per_sector: " << nnodes_per_sector << "B";

    // SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
    std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(ROUND_UP(max_node_len, SECTOR_LEN));
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    unsigned &nnbrs = *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T));
    unsigned *nhood_buf = (unsigned *) (node_buf.get() + (ndims_64 * sizeof(T)) + sizeof(unsigned));

    // number of sectors (1 for meta data)
    _u64 n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                           : npts_64 * DIV_ROUND_UP(max_node_len, SECTOR_LEN);
    _u64 disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;

    std::vector<_u64> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back(disk_index_file_size);

    diskann_writer.write(sector_buf.get(), SECTOR_LEN);  // write out the empty
                                                         // first sector, will
                                                         // be populated at the
                                                         // end.

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    LOG(INFO) << "# sectors: " << n_sectors;
    _u64 cur_node_id = 0;

    if (nnodes_per_sector > 0) {
      for (_u64 sector = 0; sector < n_sectors; sector++) {
        if (sector % 100000 == 0) {
          LOG(INFO) << "Sector #" << sector << "written";
        }
        memset(sector_buf.get(), 0, SECTOR_LEN);
        for (_u64 sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64; sector_node_id++) {
          memset(node_buf.get(), 0, max_node_len);
          // read cur node's nnbrs
          vamana_reader.read((char *) &nnbrs, sizeof(unsigned));

          // sanity checks on nnbrs
          if (nnbrs == 0) {
            LOG(INFO) << "ERROR. Found point with no out-neighbors; Point#: " << cur_node_id;
            exit(-1);
          }

          // read node's nhood
          vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(unsigned));
          if (nnbrs > width_u32) {
            vamana_reader.seekg((nnbrs - width_u32) * sizeof(unsigned), vamana_reader.cur);
          }

          // write coords of node first
          //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
          base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
          memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

          // write nnbrs
          *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

          // write nhood next
          memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned), nhood_buf,
                 (std::min)(nnbrs, width_u32) * sizeof(unsigned));

          // get offset into sector_buf
          char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

          // copy node buf into sector_node_buf
          memcpy(sector_node_buf, node_buf.get(), max_node_len);
          cur_node_id++;
        }
        // flush sector to disk
        diskann_writer.write(sector_buf.get(), SECTOR_LEN);
      }
    } else {
      uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, SECTOR_LEN);
      for (uint64_t i = 0; i < npts_64; i++) {
        if ((i * nsectors_per_node) % 100000 == 0) {
          LOG(INFO) << "Sector #" << i * nsectors_per_node << "written";
        }
        memset(multisector_buf.get(), 0, nsectors_per_node * SECTOR_LEN);

        memset(node_buf.get(), 0, max_node_len);
        // read cur node's nnbrs
        vamana_reader.read((char *) &nnbrs, sizeof(uint32_t));

        // read node's nhood
        vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
        if (nnbrs > width_u32) {
          vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
        }

        // write coords of node first
        //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
        memcpy(multisector_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

        // write nnbrs
        *(uint32_t *) (multisector_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

        // write nhood next
        memcpy(multisector_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
               (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

        // flush sector to disk
        diskann_writer.write(multisector_buf.get(), nsectors_per_node * SECTOR_LEN);
      }
    }

    diskann_writer.close();
    size_t tag_bytes_written = 0;

    // frozen point implies dynamic index which must have tags
    if (vamana_frozen_num > 0) {
      std::unique_ptr<TagT[]> mem_index_tags;
      size_t nr, nc;
      pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

      if (nr != npts_64 && nc != 1) {
        LOG(ERROR) << "Error loading tags file. File dims are " << nr << ", " << nc << ", but expecting " << npts_64
                   << " tags in 1 dimension (bin format).";

        crash();
      }

      pipeann::save_bin<TagT>(output_file + std::string(".tags"), mem_index_tags.get(), nr, nc);
    } else {
      if (tags_enabled) {
        std::unique_ptr<TagT[]> mem_index_tags;
        size_t nr, nc;

        if (!file_exists(tag_file)) {
          LOG(INFO) << "Static vamana index, tag file " << tag_file << "does not exist. Exiting....";
          exit(-1);
        }

        pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

        if (nr != npts_64 && nc != 1) {
          LOG(ERROR) << "Error loading tags file. File dims are " << nr << ", " << nc << ", but expecting " << npts_64
                     << " tags in 1 dimension (bin format).";
          crash();
        }

        pipeann::save_bin<TagT>(output_file + std::string(".tags"), mem_index_tags.get(), nr, nc);
      }
    }

    output_file_meta.push_back(output_file_meta[output_file_meta.size() - 1] + tag_bytes_written);
    pipeann::save_bin<_u64>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    LOG(INFO) << "Output file written.";
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, const char *indexBuildParameters,
                        pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 5 && param_list.size() != 6) {
      LOG(INFO) << "Correct usage of parameters is: R (max degree)"
                   " L (indexing list size, should be >= R) "
                   " B (RAM limit of final index in GB) "
                   " M (memory limit while indexing in GB)"
                   " T (number of threads for indexing) "
                   " [C (compression ratio for PQ. Overrides parameter value B)] ";
      return false;
    }

    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                    "format). Should be > 0.";
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      LOG(ERROR) << "Not building index. Please provide more RAM budget";
      return false;
    }
    _u32 num_threads = (_u32) atoi(param_list[4].c_str());

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
              << " Indexing RAM budget: " << indexing_ram_budget << " T: " << num_threads << " Final index will be in "
              << (single_file_index ? "single file" : "multiple files");

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(dataFilePath, normalized_file_path);
        _compareMetric = pipeann::Metric::L2;
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    pipeann::get_bin_metadata(normalized_file_path, points_num, dim);
    auto training_set_size = PQ_TRAINING_SET_FRACTION * points_num > MAX_PQ_TRAINING_SET_SIZE
                                 ? MAX_PQ_TRAINING_SET_SIZE
                                 : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num);
    training_set_size = (training_set_size == 0) ? 1 : training_set_size;
    LOG(INFO) << "(Normalized, if required) file : " << normalized_file_path << " has: " << points_num
              << " points. Changing training set size to " << training_set_size << " points";

    size_t num_pq_chunks = calculate_num_pq_chunks(final_index_ram_limit, points_num, dim);

    size_t train_size, train_dim;
    float *train_data;  // maximum: 256000 * dim * data_size, 1GB for 1024-dim float vector.

    auto start = std::chrono::high_resolution_clock::now();
    double p_val = ((double) training_set_size / (double) points_num);
    
    // Check if PQ pivots already exist, otherwise generate them
    if (!file_exists(pq_pivots_path)) {
      // generates random sample and sets it to train_data and updates train_size
      gen_random_slice<T>(normalized_file_path, p_val, train_data, train_size, train_dim);

      LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size
                << " num PQ chunks: " << num_pq_chunks;
      generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                         pq_pivots_path);
      delete[] train_data;
      train_data = nullptr;
    } else {
      LOG(INFO) << "PQ pivots file already exists, loading from: " << pq_pivots_path;
    }
    auto end = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "PQ pivots processed in " << std::chrono::duration<double>(end - start).count() << "s.";
    start = std::chrono::high_resolution_clock::now();
    
    // Check if PQ compressed data already exists, otherwise generate it
    if (!file_exists(pq_compressed_vectors_path)) {
      LOG(INFO) << "Generating PQ compressed data...";
      generate_pq_data_from_pivots<T>(normalized_file_path, 256, (uint32_t) num_pq_chunks, pq_pivots_path,
                                      pq_compressed_vectors_path);  // 64MB.
    } else {
      LOG(INFO) << "PQ compressed data file already exists, loading from: " << pq_compressed_vectors_path;
    }
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "PQ compressed data processed in: " << std::chrono::duration<double>(end - start).count()
              << "s.";
    start = std::chrono::high_resolution_clock::now();
    
    // Check if memory index already exists, otherwise build it
    if (!file_exists(mem_index_path) || !file_exists(mem_index_path + ".data")) {
      LOG(INFO) << "Building Vamana memory index...";
      pipeann::build_merged_vamana_index<T>(normalized_file_path, _compareMetric, single_file_index, L, R, p_val,
                                            indexing_ram_budget, mem_index_path, medoids_path, centroids_path, tag_file);
    } else {
      LOG(INFO) << "Memory index file already exists, loading from: " << mem_index_path << " and " << mem_index_path
                << ".data";
    }
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index processed in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (tag_file == nullptr) {
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    } else {
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    }

    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template<typename T, typename TagT>
  bool build_disk_index_py(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                           uint32_t num_threads, uint32_t PQ_bytes, pipeann::Metric _compareMetric,
                           bool single_file_index, const char *tag_file) {
    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";
    std::string sample_base_prefix = index_prefix_path + "_sample";

    double final_index_ram_limit = get_memory_budget(M);
    if (final_index_ram_limit <= 0) {
      LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                    "format). Should be > 0.";
      return false;
    }

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
              << " T: " << num_threads << " Final index will be in "
              << (single_file_index ? "single file" : "multiple files");

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(dataFilePath, normalized_file_path);
        _compareMetric = pipeann::Metric::L2;
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    pipeann::get_bin_metadata(normalized_file_path, points_num, dim);
    auto training_set_size = PQ_TRAINING_SET_FRACTION * points_num > MAX_PQ_TRAINING_SET_SIZE
                                 ? MAX_PQ_TRAINING_SET_SIZE
                                 : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num);
    training_set_size = (training_set_size == 0) ? 1 : training_set_size;
    LOG(INFO) << "(Normalized, if required) file : " << normalized_file_path << " has: " << points_num
              << " points. Changing training set size to " << training_set_size << " points";

    size_t num_pq_chunks = PQ_bytes;

    size_t train_size, train_dim;
    float *train_data;

    auto start = std::chrono::high_resolution_clock::now();
    double p_val = ((double) training_set_size / (double) points_num);
    
    // Check if PQ pivots already exist, otherwise generate them
    if (!file_exists(pq_pivots_path)) {
      // generates random sample and sets it to train_data and updates train_size
      gen_random_slice<T>(normalized_file_path, p_val, train_data, train_size, train_dim);

      LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size
                << " num PQ chunks: " << num_pq_chunks;
      generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                         pq_pivots_path);
      delete[] train_data;
      train_data = nullptr;
    } else {
      LOG(INFO) << "PQ pivots file already exists, loading from: " << pq_pivots_path;
    }
    auto end = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "PQ pivots processed in " << std::chrono::duration<double>(end - start).count() << "s.";
    start = std::chrono::high_resolution_clock::now();
    
    // Check if PQ compressed data already exists, otherwise generate it
    if (!file_exists(pq_compressed_vectors_path)) {
      LOG(INFO) << "Generating PQ compressed data...";
      generate_pq_data_from_pivots<T>(normalized_file_path, 256, (uint32_t) num_pq_chunks, pq_pivots_path,
                                      pq_compressed_vectors_path);
    } else {
      LOG(INFO) << "PQ compressed data file already exists, loading from: " << pq_compressed_vectors_path;
    }
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "PQ compressed data processed in: " << std::chrono::duration<double>(end - start).count()
              << "s.";
    start = std::chrono::high_resolution_clock::now();
    
    // Check if memory index already exists, otherwise build it
    if (!file_exists(mem_index_path)) {
      LOG(INFO) << "Building Vamana memory index...";
      pipeann::build_merged_vamana_index<T>(normalized_file_path, _compareMetric, single_file_index, L, R, p_val, M,
                                            mem_index_path, medoids_path, centroids_path, tag_file);
    } else {
      LOG(INFO) << "Memory index file already exists, loading from: " << mem_index_path;
    }
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index processed in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (tag_file == nullptr) {
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    } else {
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    }

    double ten_percent_points = std::ceil(points_num * 0.1);
    double num_sample_points =
        ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : ten_percent_points;
    double sample_sampling_rate = num_sample_points / points_num;
    LOG(INFO) << "Generating warmup file with " << num_sample_points
              << " points using a sampling rate of: " << sample_sampling_rate;
    gen_random_slice<T>(normalized_file_path, sample_base_prefix, sample_sampling_rate);

    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template void create_disk_layout<int8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                     const std::string &tag_file, const std::string &pq_pivots_file,
                                                     const std::string &pq_compressed_vectors_file,
                                                     bool single_file_index, const std::string &output_file);
  template void create_disk_layout<uint8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                      const std::string &tag_file, const std::string &pq_pivots_file,
                                                      const std::string &pq_compressed_vectors_file,
                                                      bool single_file_index, const std::string &output_file);
  template void create_disk_layout<float, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                    const std::string &tag_file, const std::string &pq_pivots_file,
                                                    const std::string &pq_compressed_vectors_file,
                                                    bool single_file_index, const std::string &output_file);
  // template void create_disk_layout<int8_t, uint64_t>(
  //     const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
  //     const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file, bool single_file_index,
  //     const std::string &output_file);
  // template void create_disk_layout<uint8_t, uint64_t>(
  //     const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
  //     const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file, bool single_file_index,
  //     const std::string &output_file);
  // template void create_disk_layout<float, uint64_t>(
  //     const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
  //     const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file, bool single_file_index,
  //     const std::string &output_file);

  template bool build_disk_index<int8_t, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                   const char *indexBuildParameters, pipeann::Metric _compareMetric,
                                                   bool singleFileIndex, const char *tag_file);
  template bool build_disk_index<uint8_t, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                    const char *indexBuildParameters, pipeann::Metric _compareMetric,
                                                    bool singleFileIndex, const char *tag_file);
  template bool build_disk_index<float, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                  const char *indexBuildParameters, pipeann::Metric _compareMetric,
                                                  bool singleFileIndex, const char *tag_file);
  // template bool build_disk_index<int8_t, uint64_t>(const char *dataFilePath,
  //                                                                    const char *indexFilePath,
  //                                                                    const char *indexBuildParameters,
  //                                                                    pipeann::Metric _compareMetric,
  //                                                                    bool singleFileIndex, const char *tag_file);
  // template bool build_disk_index<uint8_t, uint64_t>(const char *dataFilePath,
  //                                                                     const char *indexFilePath,
  //                                                                     const char *indexBuildParameters,
  //                                                                     pipeann::Metric _compareMetric,
  //                                                                     bool singleFileIndex, const char *tag_file);
  // template bool build_disk_index<float, uint64_t>(const char *dataFilePath, const char
  // *indexFilePath,
  //                                                                   const char *indexBuildParameters,
  //                                                                   pipeann::Metric _compareMetric,
  //                                                                   bool singleFileIndex, const char *tag_file);

  template int build_merged_vamana_index<int8_t>(std::string base_file, pipeann::Metric _compareMetric,
                                                 bool single_file_index, unsigned L, unsigned R, double sampling_rate,
                                                 double ram_budget, std::string mem_index_path,
                                                 std::string medoids_path, std::string centroids_file,
                                                 const char *tag_file);
  template int build_merged_vamana_index<float>(std::string base_file, pipeann::Metric _compareMetric,
                                                bool single_file_index, unsigned L, unsigned R, double sampling_rate,
                                                double ram_budget, std::string mem_index_path, std::string medoids_path,
                                                std::string centroids_file, const char *tag_file);
  template int build_merged_vamana_index<uint8_t>(std::string base_file, pipeann::Metric _compareMetric,
                                                  bool single_file_index, unsigned L, unsigned R, double sampling_rate,
                                                  double ram_budget, std::string mem_index_path,
                                                  std::string medoids_path, std::string centroids_file,
                                                  const char *tag_file);
};  // namespace pipeann
