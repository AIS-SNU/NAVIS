#pragma once
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
  const size_t MAX_PQ_TRAINING_SET_SIZE = 256000;
  const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 1000000;
  const double PQ_TRAINING_SET_FRACTION = 0.1;
  const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t WARMUP_L = 20;

  template<typename T, typename TagT>
  class SSDIndex;

  double get_memory_budget(const std::string &mem_budget_str);
  double get_memory_budget(double search_ram_budget_in_gb);
  void add_new_file_to_single_index(std::string index_file, std::string new_file);

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags);

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs);

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file);

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_index_file,
                                unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                const char *tag_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  bool build_disk_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                        pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file = nullptr);
  template<typename T, typename TagT = uint32_t>
  bool build_disk_index_py(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                           uint32_t num_threads, uint32_t PQ_bytes, pipeann::Metric _compareMetric,
                           bool single_file_index, const char *tag_file);

  template<typename T, typename TagT = uint32_t>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file,
                          bool single_file_index, const std::string &output_file);
}  // namespace pipeann
