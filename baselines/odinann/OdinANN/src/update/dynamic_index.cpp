#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/dynamic_index.h"
#include <csignal>
#include <cstdint>
#include <mutex>
#include <vector>

#include <algorithm>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <shared_mutex>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "aux_utils.h"
#include "ssd_index.h"
#include "parameters.h"

#include "linux_aligned_file_reader.h"

namespace pipeann {
  void copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out;
    std::filesystem::copy(prefix_in + "_disk.index", prefix_out + "_disk.index",
                          std::filesystem::copy_options::overwrite_existing);
    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    } else {
      // remove the original tags.
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }
    std::filesystem::copy(prefix_in + "_pq_pivots.bin", prefix_out + "_pq_pivots.bin",
                          std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy(prefix_in + "_pq_compressed.bin", prefix_out + "_pq_compressed.bin",
                          std::filesystem::copy_options::overwrite_existing);
    // partition data
    if (std::filesystem::exists(prefix_in + "_partition.bin.aligned")) {
      std::filesystem::copy(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(Parameters &parameters, const std::string disk_prefix_in,
                                            const std::string disk_prefix_out, Distance<T> *dist,
                                            pipeann::Metric dist_metric, int search_mode, bool use_mem_index,
                                            uint32_t num_insert_threads) {
    // check if file exists.
    if (!std::filesystem::exists(disk_prefix_in + "_disk.index")) {
      LOG(ERROR) << "Disk index file does not exist: " << disk_prefix_in << "_disk.index";
      exit(-1);
    }
    if (use_mem_index && !std::filesystem::exists(disk_prefix_in + "_mem.index")) {
      LOG(ERROR) << "In-memory index file does not exist: " << disk_prefix_in << "_mem.index";
      exit(-1);
    }

    this->active_del[0] = true;
    this->active_del[1] = false;
    this->_dist_metric = dist_metric;
    this->journal = new v2::Journal<TagT>(disk_prefix_out + "_journal");

    _paras_disk.Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    _paras_disk.Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    _paras_disk.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_disk.Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    _paras_disk.Set<unsigned>("beamwidth", parameters.Get<unsigned>("beamwidth"));
    _paras_disk.Set<bool>("saturate_graph", 0);

    _num_threads = parameters.Get<_u32>("num_threads");
    _beamwidth = parameters.Get<uint32_t>("beamwidth");

    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;

    reader.reset(new LinuxAlignedFileReader());
    _disk_index = new pipeann::SSDIndex<T, TagT>(this->_dist_metric, reader, false, true, &_paras_disk);

#ifndef NO_POLLUTE_ORIGINAL
    std::string disk_index_prefix_shadow = _disk_index_prefix_in + "_shadow";
    copy_index(_disk_index_prefix_in, disk_index_prefix_shadow);
    LOG(INFO) << "Copy disk index file to " << disk_index_prefix_shadow << "_disk.index";
    _disk_index_prefix_in = disk_index_prefix_shadow;
#endif

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH) {
      this->search_mode = search_mode;
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode << ". Must be BEAM_SEARCH or PAGE_SEARCH.";
      exit(-1);
    }
    bool use_page_search = (search_mode == PAGE_SEARCH);
    int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, true, use_page_search);
    if (res != 0) {
      LOG(INFO) << "Failed to load disk index in DynamicSSDIndex constructor";
      exit(-1);
    }

    this->_use_mem_index = use_mem_index;
    if (use_mem_index) {
      std::string mem_index_path = disk_prefix_in + "_mem.index";  // use the original one.
      LOG(INFO) << "Use static in-memory index for acceleration, path: " << mem_index_path;
      _disk_index->load_mem_index(this->_dist_metric, _disk_index->data_dim, mem_index_path);
    }
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::~DynamicSSDIndex() {
    // put in destructor code
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::checkpoint() {
    // TODO(gh): checkpoint the index.
    journal->checkpoint();
  }

  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::insert(const T *point, const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);  // prevent merge during insert
    journal->append(v2::TxType::kInsert, tag);
    auto *deletion_set = &deletion_sets[active_delete_set];
    return _disk_index->insert_in_place(point, tag, deletion_set);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                                        const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                                        bool dyn_search_l) {
    std::vector<TagT> result_tags(4096);
    std::vector<float> result_distances(4096);
    auto *deletion_set = &deletion_sets[active_delete_set];
    size_t n = 0;
    if (search_mode == BEAM_SEARCH) {
      n = _disk_index->beam_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats, deletion_set, dyn_search_l);
    } else if (search_mode == PAGE_SEARCH) {
      n = _disk_index->page_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode << ". Must be BEAM_SEARCH or PAGE_SEARCH.";
      exit(-1);
    }
    std::vector<NeighborTag<TagT>> best_vec;
    for (size_t i = 0; i < n; i++) {
      best_vec.emplace_back(result_tags[i], result_distances[i]);
    }
    std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
    size_t pos = 0;

    for (auto iter : best_vec) {
      if (deletion_set->find(iter.tag) == deletion_set->end()) {
        tags[pos] = iter.tag;
        distances[pos] = iter.dist;
        pos++;
      }
      if (pos == K) {
        return;
      }
    }
    // LOG(INFO) << "Failed to find enough tags after " << i + 1 << " attempts";
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::lazy_delete(const TagT &tag) {
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    journal->append(v2::TxType::kDelete, tag);

    if (active_del[active_delete_set].load() == false) {
      LOG(ERROR) << "Active deletion set indicated as _deletion_set_" << active_delete_set
                 << " but it cannot accept deletions";
    }
    if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
      deletion_sets[active_delete_set].insert(tag);
      deleted_tags[active_delete_set].push_back(tag);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::save_del_set() {
    int nxt_idx = 1 - active_delete_set, cur_idx = active_delete_set;
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    deletion_sets[nxt_idx].clear();
    deleted_tags[nxt_idx].clear();
    bool expected_active = false;
    if (active_del[nxt_idx].compare_exchange_strong(expected_active, true)) {
      LOG(INFO) << "Cleared deletion set " << nxt_idx << " - ready to accept new points";
    } else {
      LOG(INFO) << "Failed to clear deletion set " << nxt_idx;
    }
    active_delete_set = nxt_idx;
    active_del[cur_idx].store(false);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::final_merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);  // only one merge at a time
    // _disk_index_in -> _disk_index_out
    save_del_set();
    pipeann::Timer timer;
    merge(nthreads, n_sampled_nbrs);

    // TODO(gh): do we really need to reload disk index?
    std::swap(_disk_index_prefix_in, _disk_index_prefix_out);
    _disk_index->reload(_disk_index_prefix_in.c_str(), _num_threads);
    LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    _disk_index->merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out, deleted_tags[1 - active_delete_set],
                               deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
  }

  template class DynamicSSDIndex<float>;
  template class DynamicSSDIndex<uint8_t>;
  template class DynamicSSDIndex<int8_t>;
}  // namespace pipeann
