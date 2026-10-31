#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <omp.h>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "tsl/robin_set.h"
#include <unordered_map>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "index.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"
#include "query_buf.h"
#include "v2/lock_table.h"

// only L2 implemented. Need to implement inner product search

namespace pipeann {
  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>

  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
                        const bool save_index_in_one_file, const bool enable_tags)
      : _dist_metric(m), _dim(dim), _max_points(max_points), _save_as_one_file(save_index_in_one_file),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags) {
    if (dynamic_index && !enable_tags) {
      LOG(ERROR) << "WARNING: Dynamic Indices must have tags enabled. Auto-enabling.";
      _enable_tags = true;
    }
    // data is stored to _nd * aligned_dim matrix with necessary
    // zero-padding
    _aligned_dim = ROUND_UP(_dim, 8);

    if (dynamic_index)
      _num_frozen_pts = 1;

    if (_max_points == 0) {
      _max_points = 1;
    }

    alloc_aligned(((void **) &_data), (_max_points + 1) * _aligned_dim * sizeof(T), 8 * sizeof(T));
    // std::memset(_data, 0, (_max_points + 1) * _aligned_dim * sizeof(T));

    _ep = (unsigned) _max_points;

    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);

    for (size_t i = 0; i < _max_points + _num_frozen_pts; i++)
      _final_graph[i].clear();

    constexpr uint64_t kLockTableEntries = 131072;  // ~1MB lock table.
    this->_locks = new v2::LockTable(kLockTableEntries);
    LOG(INFO) << "Getting distance function for metric: " << (m == pipeann::Metric::COSINE ? "cosine" : "l2");
    this->_distance = get_distance_function<T>(m);
    _width = 0;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_distance;
    delete this->_locks;
    aligned_free(_data);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::clear_index() {
    memset(_data, 0, _aligned_dim * (_max_points + _num_frozen_pts) * sizeof(T));
    _nd = 0;
    for (size_t i = 0; i < _final_graph.size(); i++)
      _final_graph[i].clear();

    _tag_to_location.clear();
    _location_to_tag.clear();

    _delete_set.clear();
    _empty_slots.clear();
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_tags(std::string tags_file, size_t offset) {
    if (!_enable_tags) {
      LOG(INFO) << "Not saving tags as they are not enabled.";
      return 0;
    }
    size_t tag_bytes_written;
    TagT *tag_data = new TagT[_nd + _num_frozen_pts];
    for (_u32 i = 0; i < _nd; i++) {
      if (_location_to_tag.find(i) != _location_to_tag.end()) {
        tag_data[i] = _location_to_tag[i];
      } else {
        // catering to future when tagT can be any type.
        std::memset((char *) &tag_data[i], 0, sizeof(TagT));
      }
    }
    if (_num_frozen_pts > 0) {
      std::memset((char *) &tag_data[_ep], 0, sizeof(TagT));
    }
    tag_bytes_written = save_bin<TagT>(tags_file, tag_data, _nd + _num_frozen_pts, 1, offset);
    delete[] tag_data;
    return tag_bytes_written;
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_data(std::string data_file, size_t offset) {
    return save_data_in_base_dimensions(data_file, _data, _nd + _num_frozen_pts, _dim, _aligned_dim, offset);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_graph(std::string graph_file, size_t offset) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    out.seekp(offset, out.beg);
    _u64 index_size = 24;
    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    unsigned ep_u32 = _ep;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &_num_frozen_pts, sizeof(_u64));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = _final_graph[i].size() > max_degree ? (_u32) _final_graph[i].size() : max_degree;
      index_size += (_u64) (sizeof(unsigned) * (GK + 1));
    }
    out.seekp(offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_delete_list(const std::string &filename, _u64 file_offset) {
    if (_delete_set.size() == 0) {
      return 0;
    }
    std::unique_ptr<_u32[]> delete_list = std::make_unique<_u32[]>(_delete_set.size());
    _u32 i = 0;
    for (auto &del : _delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<_u32>(filename, delete_list.get(), _delete_set.size(), 1, file_offset);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    // compact_data();
    compact_frozen_point();
    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      save_graph(graph_file);
      delete_file(data_file);
      save_data(data_file);
      delete_file(tags_file);
      save_tags(tags_file);
      delete_file(delete_list_file);
      save_delete_list(delete_list_file);
    } else {
      delete_file(filename);
      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = METADATA_SIZE;
      cumul_bytes[1] = cumul_bytes[0] + save_graph(std::string(filename), cumul_bytes[0]);
      cumul_bytes[2] = cumul_bytes[1] + save_data(std::string(filename), cumul_bytes[1]);
      cumul_bytes[3] = cumul_bytes[2] + save_tags(std::string(filename), cumul_bytes[2]);
      cumul_bytes[4] = cumul_bytes[3] + save_delete_list(filename, cumul_bytes[3]);
      pipeann::save_bin<_u64>(filename, cumul_bytes.data(), cumul_bytes.size(), 1, 0);

      LOG(INFO) << "Saved index as one file to " << filename << " of size " << cumul_bytes[cumul_bytes.size() - 1]
                << "B.";
    }

    reposition_frozen_point_to_end();

    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    LOG(INFO) << "Time taken for save: " << timespan.count() << "s.";
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(const std::string tag_filename, size_t offset) {
    if (_enable_tags && !file_exists(tag_filename)) {
      LOG(ERROR) << "Tag file provided does not exist!";
      crash();
    }

    if (!_enable_tags) {
      LOG(INFO) << "Tags not loaded as tags not enabled.";
      return 0;
    }

    size_t file_dim, file_num_points;
    TagT *tag_data;
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points, file_dim, offset);

    if (file_dim != 1) {
      LOG(ERROR) << "ERROR: Loading " << file_dim << " dimensions for tags,"
                 << "but tag file must have 1 dimension.";
      crash();
    }

    size_t num_data_points = _num_frozen_pts > 0 ? file_num_points - 1 : file_num_points;
    for (_u32 i = 0; i < (_u32) num_data_points; i++) {
      TagT tag = *(tag_data + i);
      if (_delete_set.find(i) == _delete_set.end()) {
        _location_to_tag[i] = tag;
        _tag_to_location[tag] = (_u32) i;
      }
    }
    LOG(INFO) << "Tags loaded.";
    delete[] tag_data;
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_data(std::string filename, size_t offset) {
    LOG(INFO) << "Loading data from " << filename << " offset " << offset;
    if (!file_exists(filename)) {
      LOG(ERROR) << "ERROR: data file " << filename << " does not exist.";
      aligned_free(_data);
      crash();
    }
    size_t file_dim, file_num_points;
    pipeann::get_bin_metadata(filename, file_num_points, file_dim, offset);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim) {
      LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                 << "but file has " << file_dim << " dimension.";
      aligned_free(_data);
      crash();
    }

    if (file_num_points > _max_points + _num_frozen_pts) {
      //_change_lock is already locked in load()
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      resize(file_num_points);
    }

    copy_aligned_data_from_file<T>(std::string(filename), _data, file_num_points, file_dim, _aligned_dim, offset);
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(const std::string &filename, size_t offset) {
    std::unique_ptr<_u32[]> delete_list;
    _u64 npts, ndim;
    load_bin<_u32>(filename, delete_list, npts, ndim, offset);
    assert(ndim == 1);
    for (size_t i = 0; i < npts; i++) {
      _delete_set.insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename) {
    _change_lock.lock();

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;

    if (!_save_as_one_file) {
      std::string data_file = std::string(filename) + ".data";
      std::string tags_file = std::string(filename) + ".tags";
      std::string delete_set_file = std::string(filename) + ".del";
      std::string graph_file = std::string(filename);
      data_file_num_pts = load_data(data_file);
      if (file_exists(delete_set_file)) {
        load_delete_set(delete_set_file);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(tags_file);
      }
      graph_num_pts = load_graph(graph_file, data_file_num_pts);

    } else {
      _u64 nr, nc;
      std::unique_ptr<_u64[]> file_offset_data;

      std::string index_file(filename);

      pipeann::load_bin<_u64>(index_file, file_offset_data, nr, nc, 0);
      // Loading data first so that we know how many points to expect.
      data_file_num_pts = load_data(index_file, file_offset_data[1]);
      graph_num_pts = load_graph(index_file, data_file_num_pts, file_offset_data[0]);
      if (file_offset_data[3] != file_offset_data[4]) {
        load_delete_set(index_file, file_offset_data[3]);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(index_file, file_offset_data[2]);
      }
    }

    if (data_file_num_pts != graph_num_pts || (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
      LOG(ERROR) << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
                 << graph_num_pts << " from graph, and " << tags_file_num_pts
                 << " tags, with num_frozen_pts being set to " << _num_frozen_pts << " in constructor.";
      aligned_free(_data);
      crash();
    }

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    _lazy_done = _delete_set.size() != 0;

    reposition_frozen_point_to_end();
    LOG(INFO) << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd << " _ep: " << _ep
              << " size(_location_to_tag): " << _location_to_tag.size()
              << " size(_tag_to_location):" << _tag_to_location.size() << " Max points: " << _max_points;

    _change_lock.unlock();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::load_from_disk_index(const std::string &filename) {
    // only load V and E.
    std::ifstream in(filename + "_disk.index", std::ios::binary);
    _u32 nr, nc;
    _u64 disk_nnodes, disk_ndims, medoid_id_on_file, max_node_len, nnodes_per_sector;

    in.read((char *) &nr, sizeof(_u32));
    in.read((char *) &nc, sizeof(_u32));

    in.read((char *) &disk_nnodes, sizeof(_u64));
    in.read((char *) &disk_ndims, sizeof(_u64));

    in.read((char *) &medoid_id_on_file, sizeof(_u64));
    in.read((char *) &max_node_len, sizeof(_u64));
    in.read((char *) &nnodes_per_sector, sizeof(_u64));

    LOG(INFO) << "Loading disk index from " << filename << "_disk.index";
    LOG(INFO) << "Disk index has " << disk_nnodes << " nodes and " << disk_ndims << " dimensions.";
    LOG(INFO) << "Medoid id on file: " << medoid_id_on_file << " Max node len: " << max_node_len
              << " Nodes per sector: " << nnodes_per_sector;

    _ep = medoid_id_on_file;
    _u64 data_dim = disk_ndims;
    range = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

    constexpr int kSectorsPerRead = 65536;
    constexpr int kSectorLen = 4096;
    char *buf;
    pipeann::alloc_aligned((void **) &buf, kSectorsPerRead * kSectorLen, kSectorLen);
    uint64_t n_sectors = ROUND_UP(disk_nnodes, nnodes_per_sector) / nnodes_per_sector;
    in.seekg(4096, in.beg);
    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += kSectorsPerRead) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + kSectorsPerRead, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(disk_nnodes, ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      in.read(buf, n_sectors_to_read * kSectorLen);

#pragma omp parallel for
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc;
#pragma omp critical
        {
          _location_to_tag[id] = id;
          _tag_to_location[id] = id;
        }

        auto page_rbuf = buf + (loc / nnodes_per_sector - st_sector) * kSectorLen;
        auto node_rbuf = page_rbuf + (nnodes_per_sector == 0 ? 0 : ((_u64) loc % nnodes_per_sector) * max_node_len);
        DiskNode<T> node(id, (T *) node_rbuf, (unsigned *) (node_rbuf + data_dim * sizeof(T)));

        // load data and nhood.
        memcpy(_data + id * data_dim, node.coords, data_dim * sizeof(T));
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          nhood.push_back(node.nbrs[i]);
        }
        _final_graph[id] = nhood;
      }
    }

    disk_npts = disk_nnodes;
    _nd = disk_nnodes - _num_frozen_pts;
    _empty_slots.clear();
    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }
    reposition_frozen_point_to_end();
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(std::string filename, size_t expected_num_points, size_t offset) {
    std::ifstream in(filename, std::ios::binary);
    in.seekg(offset, in.beg);
    size_t expected_file_size;
    _u64 file_frozen_pts;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    in.read((char *) &file_frozen_pts, sizeof(_u64));

    if (file_frozen_pts != _num_frozen_pts) {
      if (file_frozen_pts == 1) {
        LOG(ERROR) << "ERROR: When loading index, detected dynamic index, but "
                      "constructor asks for static index. Exitting.";
      } else {
        LOG(ERROR) << "ERROR: When loading index, detected static index, but "
                      "constructor asks for dynamic index. Exitting.";
      }
      aligned_free(_data);
      crash();
    }
    LOG(INFO) << "Loading vamana index " << filename << "...";

    // Sanity check. In case the user gave us fewer points as max_points than
    // the number
    // of points in the dataset, resize the _final_graph to the larger size.
    if (_max_points < (expected_num_points - _num_frozen_pts)) {
      LOG(INFO) << "Number of points in data: " << expected_num_points
                << " is more than max_points argument: " << _final_graph.size()
                << " Setting max points to: " << expected_num_points;
      _final_graph.resize(expected_num_points);
      _max_points = expected_num_points - _num_frozen_pts;
      // changed expected_num to expected_num - frozen_num
    }

    size_t bytes_read = 24;
    size_t cc = 0;
    unsigned nodes = 0;
    while (bytes_read != expected_file_size) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (k == 0) {
        LOG(ERROR) << "ERROR: Point found with no out-neighbors, point#" << nodes;
      }
      //      if (in.eof())
      //        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      tmp.reserve(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes - 1].swap(tmp);
      bytes_read += sizeof(uint32_t) * ((_u64) k + 1);
      if (nodes % 10000000 == 0)
        LOG(INFO) << ".";
    }

    LOG(INFO) << "done. Index has " << nodes << " nodes and " << cc << " out-edges, _ep is set to " << _ep;
    return nodes;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      LOG(INFO) << "Tag " << tag << " does not exist";
      return -1;
    }
    unsigned location = _tag_to_location[tag];
    // memory should be allocated for vec before calling this function
    memcpy((void *) vec, (void *) (_data + (size_t) (location * _aligned_dim)), (size_t) _aligned_dim * sizeof(T));
    return 0;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    // allocate and init centroid
    std::vector<float> center(_aligned_dim, 0.0f);

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _aligned_dim; j++)
        center[j] += (float) _data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= (float) _nd;

    // compute all to one distance, updating the atomic variables should not be the bottleneck.
    constexpr uint64_t kDistNum = 256;
    struct alignas(64) AtomicDistance {
      unsigned idx = 0;
      float dist = std::numeric_limits<float>::max();
      std::mutex lk;

      void update(unsigned i, float d) {
        std::lock_guard<std::mutex> guard(lk);
        if (d < dist) {
          dist = d;
          idx = i;
        }
      }
    };
    AtomicDistance atomic_dists[kDistNum];

#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) _nd; i++) {
      // extract point and distance reference
      float dist = 0;
      const T *cur_vec = _data + (i * (size_t) _aligned_dim);
      for (size_t j = 0; j < _aligned_dim; j++) {
        dist += (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
      }
      atomic_dists[(i / 65536) % kDistNum].update(i, dist);
    }

    unsigned min_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (unsigned i = 0; i < kDistNum; i++) {
      if (atomic_dists[i].dist < min_dist) {
        min_idx = atomic_dists[i].idx;
        min_dist = atomic_dists[i].dist;
      }
    }
    return min_idx;
  }

  /* iterate_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_width: beam_width when performing indexing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(const T *node_coords, const unsigned Lsize,
                                                                       const std::vector<unsigned> &init_ids,
                                                                       std::vector<Neighbor> &expanded_nodes_info,
                                                                       tsl::robin_set<unsigned> &expanded_nodes_ids,
                                                                       std::vector<Neighbor> &best_L_nodes,
                                                                       bool ret_frozen) {
    best_L_nodes.resize(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned l = 0;
    Neighbor nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      assert(id < _max_points + _num_frozen_pts);
      nn = Neighbor(id, _distance->compare(_data + _aligned_dim * (size_t) id, node_coords, (unsigned) _aligned_dim),
                    true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        if (!(best_L_nodes[k].id == _ep && _num_frozen_pts > 0 && !ret_frozen)) {
          expanded_nodes_info.emplace_back(best_L_nodes[k]);
          expanded_nodes_ids.insert(n);
        }
        std::vector<unsigned> des;

        {
          // v2::SparseReadLockGuard<uint64_t> guard(&_locks, n);
          v2::LockGuard guard(_locks->rdlock(n));
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              LOG(ERROR) << "Wrong id found: " << _final_graph[n][m];
              crash();
            }
            des.emplace_back(_final_graph[n][m]);
          }
        }

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < des.size()) {
              auto nextn = des[m + 1];
              pipeann::prefetch_vector((const char *) _data + _aligned_dim * (size_t) nextn, sizeof(T) * _aligned_dim);
            }

            cmps++;
            float dist = _distance->compare(node_coords, _data + _aligned_dim * (size_t) id, (unsigned) _aligned_dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::iterate_to_fixed_point(const T *node_coords, const unsigned Lindex,
                                              std::vector<Neighbor> &expanded_nodes_info,
                                              tsl::robin_map<uint32_t, T *> &coord_map, bool return_frozen_pt) {
    std::vector<uint32_t> init_ids;
    init_ids.push_back(this->_ep);
    std::vector<Neighbor> best_L_nodes;
    tsl::robin_set<uint32_t> expanded_nodes_ids;
    this->iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes,
                                 return_frozen_pt);
    for (Neighbor &einf : expanded_nodes_info) {
      T *coords = this->_data + (uint64_t) einf.id * (uint64_t) this->_aligned_dim;
      coord_map.insert(std::make_pair(einf.id, coords));
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(const size_t node_id, const unsigned Lindex, std::vector<unsigned> init_ids,
                                          std::vector<Neighbor> &expanded_nodes_info,
                                          tsl::robin_set<unsigned> &expanded_nodes_ids) {
    const T *node_coords = _data + _aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree,
                                    const unsigned maxc, std::vector<Neighbor> &result) {
    auto pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree,
                                    const unsigned maxc, std::vector<Neighbor> &result,
                                    std::vector<float> &occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;

      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          float djk = _distance->compare(_data + _aligned_dim * (size_t) pool[t].id,
                                         _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                                       const Parameters &parameter, std::vector<unsigned> &pruned_list) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float alpha = parameter.Get<float>("alpha");

    if (pool.size() == 0) {
      crash();
    }

    _width = (std::max)(_width, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, alpha, range, maxc, result, occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }

    if (_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end()) &&
            pool[i].id != location)
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned n, std::vector<unsigned> &pruned_list, const Parameters &parameter) {
    const auto range = parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _nd + _num_frozen_pts);

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      /* des_pool contains the neighbors of the neighbors of n */
      auto &des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool prune_needed = false;
      {
        // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
        v2::LockGuard guard(_locks->wrlock(des));
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (_u64) (SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor> dummy_pool(0);

        size_t reserveSize = (size_t) (std::ceil(1.05 * SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) {
            float dist = _distance->compare(_data + _aligned_dim * (size_t) des,
                                            _data + _aligned_dim * (size_t) cur_nbr, (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, parameter, new_out_neighbors);
        {
          // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
          v2::LockGuard guard(_locks->wrlock(des));
          _final_graph[des].assign(new_out_neighbors.begin(), new_out_neighbors.end());
        }
      }
    }
  }

  // one-pass graph building.
  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    unsigned num_threads = parameters.Get<unsigned>("num_threads");
    _saturate_graph = parameters.Get<bool>("saturate_graph");
    unsigned L = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range = parameters.Get<unsigned>("R");

    LOG(INFO) << "Parameters: " << "L: " << L << ", R: " << range
              << ", saturate_graph: " << (_saturate_graph ? "true" : "false") << ", num_threads: " << num_threads
              << ", alpha: " << parameters.Get<float>("alpha");
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    int64_t n_vecs_to_visit = _nd + _num_frozen_pts;
    _ep = _num_frozen_pts > 0 ? _max_points : calculate_entry_point();

    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_ep);

    pipeann::Timer link_timer;
#pragma omp parallel for schedule(dynamic)
    for (int64_t node = 0; node < n_vecs_to_visit; node++) {
      // search.
      std::vector<Neighbor> pool;
      tsl::robin_set<unsigned> visited;
      pool.reserve(2 * L);
      visited.reserve(2 * L);
      get_expanded_nodes(node, L, init_ids, pool, visited);
      // remove the node itself from pool.
      for (auto it = pool.begin(); it != pool.end();) {
        if (it->id == node) {
          it = pool.erase(it);
        } else {
          ++it;
        }
      }
      // prune neighbors.
      std::vector<unsigned> pruned_list;
      prune_neighbors(node, pool, parameters, pruned_list);

      {
        // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, node);
        v2::LockGuard guard(_locks->wrlock(node));
        _final_graph[node].assign(pruned_list.begin(), pruned_list.end());
      }

      inter_insert(node, pruned_list, parameters);

      if (node % 100000 == 0) {
        std::cerr << "\r" << (100.0 * node) / (n_vecs_to_visit) << "% of index build completed.";
      }
    }

    if (_nd > 0) {
      LOG(INFO) << "Starting final cleanup..";
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < n_vecs_to_visit; node_ctr++) {
      auto node = node_ctr;
      if (_final_graph[node].size() > range) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor> dummy_pool(0);
        std::vector<unsigned> new_out_neighbors;

        for (auto cur_nbr : _final_graph[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node) {
            float dist = _distance->compare(_data + _aligned_dim * (size_t) node,
                                            _data + _aligned_dim * (size_t) cur_nbr, (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

        _final_graph[node].clear();
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
    if (_nd > 0) {
      LOG(INFO) << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, Parameters &parameters,
                             const std::vector<TagT> &tags) {
    if (!file_exists(filename)) {
      LOG(ERROR) << "Data file " << filename << " does not exist!!! Exiting....";
      crash();
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG(INFO) << "Starting with an empty index.";
      _nd = 0;
    } else {
      pipeann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points || num_points_to_load > file_num_points) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points << " points, but "
                   << "index can support only " << _max_points << " points as specified in constructor.";
        crash();
      }
      if (file_dim != _dim) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                   << "but file has " << file_dim << " dimension.";
        crash();
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data, file_num_points, file_dim, _aligned_dim);

      LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;

      if (_enable_tags && tags.size() != num_points_to_load) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points from file,"
                   << "but tags vector is of size " << tags.size() << ".";
        crash();
      }
      if (_enable_tags) {
        for (size_t i = 0; i < tags.size(); ++i) {
          _tag_to_location[tags[i]] = (unsigned) i;
          _location_to_tag[(unsigned) i] = tags[i];
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG(INFO) << "Index built with degree: max:" << max << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, Parameters &parameters,
                             const char *tag_filename) {
    if (!file_exists(filename)) {
      LOG(ERROR) << "Data file provided " << filename << " does not exist.";
      crash();
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG(INFO) << "Starting with an empty index.";
      _nd = 0;
    } else {
      pipeann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points || num_points_to_load > file_num_points) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points << " points, but "
                   << "index can support only " << _max_points << " points as specified in constructor.";
        crash();
      }
      if (file_dim != _dim) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                   << "but file has " << file_dim << " dimension.";
        crash();
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data, file_num_points, file_dim, _aligned_dim);

      LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;
      if (_enable_tags) {
        if (tag_filename == nullptr) {
          for (unsigned i = 0; i < num_points_to_load; i++) {
            _tag_to_location[i] = i;
            _location_to_tag[i] = i;
          }
        } else {
          if (file_exists(tag_filename)) {
            LOG(INFO) << "Loading tags from " << tag_filename << " for vamana index build";
            TagT *tag_data = nullptr;
            size_t npts, ndim;
            pipeann::load_bin(tag_filename, tag_data, npts, ndim);
            if (npts != num_points_to_load) {
              std::stringstream sstream;
              sstream << "Loaded " << npts << " tags instead of expected number: " << num_points_to_load;
              LOG(ERROR) << sstream.str();
              crash();
            }
            for (size_t i = 0; i < npts; i++) {
              _tag_to_location[tag_data[i]] = (unsigned) i;
              _location_to_tag[(unsigned) i] = tag_data[i];
            }
            delete[] tag_data;
          } else {
            LOG(ERROR) << "Tag file " << tag_filename << " does not exist. Exiting...";
            crash();
          }
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG(INFO) << "Index built with degree: max:" << max << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       std::vector<NeighborTag<TagT>> &best_K_tags) {
    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    assert(best_K_tags.size() == 0);
    std::vector<unsigned> init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }

    T *aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids, best, false);

    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    for (auto iter : best) {
      if (_location_to_tag.find(iter.id) != _location_to_tag.end())
        best_K_tags.emplace_back(NeighborTag<TagT>(_location_to_tag[iter.id], iter.distance));
      if (best_K_tags.size() == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       unsigned *indices, float *distances) {
    std::vector<unsigned> init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < _max_points) {
        indices[pos] = it.id;
        if (distances != nullptr)
          distances[pos] = it.distance;
        pos++;
      }
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const uint64_t K, const unsigned L,
                                                       std::vector<unsigned> init_ids, uint64_t *indices,
                                                       float *distances) {
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval = iterate_to_fixed_point(aligned_query, (unsigned) L, init_ids, expanded_nodes_info, expanded_nodes_ids,
                                         best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      distances[pos] = it.distance;
      pos++;
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K, const unsigned L, TagT *tags,
                                          float *distances, std::vector<T *> &res_vectors) {
    _u32 *indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t pos = 0;
    for (int i = 0; i < (int) L; ++i)
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        res_vectors[i] = _data + indices[i] * _aligned_dim;

        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags,
                                          float *distances) {
    _u32 *indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t pos = 0;
    for (int i = 0; i < (int) L; ++i) {
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  uint32_t Index<T, TagT>::search_with_tags_fast(const T *node_coords, const unsigned Lsize, TagT *tags, float *dists) {
    std::vector<Neighbor> best_L_nodes(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }

    unsigned l = 0;
    Neighbor nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    auto id = _ep;
    nn = Neighbor(id, _distance->compare(_data + _aligned_dim * (size_t) id, node_coords, _aligned_dim), true);
    inserted_into_pool.insert(id);
    best_L_nodes[l++] = nn;

    unsigned k = 0, cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;

        auto &cur_v = _final_graph[n];
        for (unsigned m = 0; m < cur_v.size(); ++m) {
          unsigned id = cur_v[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < cur_v.size()) {
              auto nextn = cur_v[m + 1];
              pipeann::prefetch_vector((const char *) _data + _aligned_dim * (size_t) nextn, sizeof(T) * _aligned_dim);
            }

            float dist = _distance->compare(node_coords, _data + _aligned_dim * (size_t) id, (unsigned) _aligned_dim);
            cmps++;

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else {
        k++;
      }
    }
    for (uint32_t i = 0; i < Lsize; ++i) {
      tags[i] = _location_to_tag[best_L_nodes[i].id];
      dists[i] = best_L_nodes[i].distance;
    }
    return cmps;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  template<typename T, typename TagT>
  T *Index<T, TagT>::get_data() {
    if (_num_frozen_pts > 0) {
      T *ret_data = nullptr;
      size_t allocSize = _nd * _aligned_dim * sizeof(T);
      alloc_aligned(((void **) &ret_data), allocSize, 8 * sizeof(T));
      memset(ret_data, 0, _nd * _aligned_dim * sizeof(T));
      memcpy(ret_data, _data, _nd * _aligned_dim * sizeof(T));
      return ret_data;
    }
    return _data;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
  template<typename T, typename TagT>
  int Index<T, TagT>::generate_frozen_point() {
    if (_num_frozen_pts == 0)
      return 0;

    if (_nd == 0) {
      memset(_data + (_max_points) *_aligned_dim, 0, _aligned_dim * sizeof(T));
      return 1;
    }
    size_t res = calculate_entry_point();
    memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim, _aligned_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    assert(_enable_tags);

    if (!_enable_tags) {
      LOG(ERROR) << "Tags must be instantiated for deletions";
      return -2;
    }

    if (_data_compacted) {
      for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot) {
        _empty_slots.insert(slot);
      }
    }

    _lazy_done = false;
    _eager_done = false;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::release_location() {
    LockGuard guard(_change_lock);
    _nd--;
  }

  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  // proxy inserts all nghrs of deleted points
  // original approach
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    LOG(INFO) << "Inside Index::consolidate_deletes()";
    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float alpha = parameters.Get<float>("alpha");

    _u64 total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    _s64 total_blocks = DIV_ROUND_UP(total_pts, block_size);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
    for (_s64 block = 0; block < total_blocks; ++block) {
      tsl::robin_set<unsigned> candidate_set;
      std::vector<Neighbor> expanded_nghrs;
      std::vector<Neighbor> result;

      for (_s64 i = block * block_size;
           i < (_s64) ((block + 1) * block_size) && i < (_s64) (_max_points + _num_frozen_pts); i++) {
        if ((_delete_set.find((_u32) i) == _delete_set.end()) && (_empty_slots.find((_u32) i) == _empty_slots.end())) {
          candidate_set.clear();
          expanded_nghrs.clear();
          result.clear();

          bool modify = false;
          for (auto ngh : _final_graph[(_u32) i]) {
            if (_delete_set.find(ngh) != _delete_set.end()) {
              modify = true;

              // Add outgoing links from
              for (auto j : _final_graph[ngh])
                if (_delete_set.find(j) == _delete_set.end())
                  candidate_set.insert(j);
            } else {
              candidate_set.insert(ngh);
            }
          }
          if (modify) {
            for (auto j : candidate_set) {
              expanded_nghrs.push_back(
                  Neighbor(j,
                           _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * (size_t) j,
                                              (unsigned) _aligned_dim),
                           true));
            }

            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
            occlude_list(expanded_nghrs, alpha, range, maxc, result);

            _final_graph[(_u32) i].clear();
            for (auto j : result) {
              if (j.id != (_u32) i && (_delete_set.find(j.id) == _delete_set.end()))
                _final_graph[(_u32) i].push_back(j.id);
            }
          }
        }
      }
    }

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::consolidate(Parameters &parameters) {
    consolidate_deletes(parameters);
    compact_data();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_frozen_point() {
    if (_nd < _max_points) {
      if (_num_frozen_pts > 0) {
        // set new _ep to be frozen point
        _ep = (_u32) _nd;
        if (!_final_graph[_max_points].empty()) {
          for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
              if (_final_graph[i][j] == _max_points)
                _final_graph[i][j] = (_u32) _nd;

          _final_graph[_nd].clear();
          for (unsigned k = 0; k < _final_graph[_max_points].size(); k++)
            _final_graph[_nd].emplace_back(_final_graph[_max_points][k]);

          _final_graph[_max_points].clear();

          memcpy((void *) (_data + (size_t) _aligned_dim * _nd), _data + (size_t) _aligned_dim * _max_points,
                 sizeof(T) * _dim);
          memset((_data + (size_t) _aligned_dim * _max_points), 0, sizeof(T) * _aligned_dim);
        }
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data() {
    if (!_dynamic_index)
      return;

    if (!_lazy_done && !_eager_done)
      return;

    if (_data_compacted) {
      LOG(ERROR) << "Warning! Calling compact_data() when _data_compacted is true!";
      return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto fnstart = start;

    std::vector<unsigned> new_location = std::vector<unsigned>(_max_points + _num_frozen_pts, (_u32) _max_points);

    _u32 new_counter = 0;

    for (_u32 old_counter = 0; old_counter < _max_points + _num_frozen_pts; old_counter++) {
      if (_location_to_tag.find(old_counter) != _location_to_tag.end()) {
        new_location[old_counter] = new_counter;
        new_counter++;
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for initial setup: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";
    // If start node is removed, replace it.
    if (_delete_set.find(_ep) != _delete_set.end()) {
      LOG(ERROR) << "Replacing start node which has been deleted... ";
      auto old_ep = _ep;
      // First active neighbor of old start node is new start node
      for (auto iter : _final_graph[_ep])
        if (_delete_set.find(iter) != _delete_set.end()) {
          _ep = iter;
          break;
        }
      if (_ep == old_ep) {
        LOG(ERROR) << "ERROR: Did not find a replacement for start node.";
        crash();
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
      }
    }

    start = std::chrono::high_resolution_clock::now();
    double copy_time = 0;
    for (unsigned old = 0; old <= _max_points; ++old) {
      if ((new_location[old] < _max_points) || (old == _max_points)) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          if (new_location[_final_graph[old][i]] > _final_graph[old][i]) {
            std::stringstream sstream;
            sstream << "Error in compact_data(). Found point: " << old << " whose " << i
                    << "th neighbor has new location " << new_location[_final_graph[old][i]]
                    << " that is greater than its old location: " << _final_graph[old][i];
            if (_delete_set.find(_final_graph[old][i]) != _delete_set.end()) {
              sstream << " Point: " << old << " index: " << i << " neighbor: " << _final_graph[old][i]
                      << " found in delete set of size: " << _delete_set.size();
            } else {
              sstream << " Point: " << old << " neighbor: " << _final_graph[old][i]
                      << " NOT found in delete set of size: " << _delete_set.size();
            }

            LOG(ERROR) << sstream.str();
            crash();
          }
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        // Move the data and adj list to the correct position
        auto c_start = std::chrono::high_resolution_clock::now();
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old), _aligned_dim * sizeof(T));
        }
        auto c_stop = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration_cast<std::chrono::duration<double>>(c_stop - c_start).count();

      } else {
        _final_graph[old].clear();
      }
    }
    stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for moving data around: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count()
              << "s. Of which copy_time: " << copy_time << "s.";

    start = std::chrono::high_resolution_clock::now();
    _tag_to_location.clear();
    for (auto iter : _location_to_tag) {
      _tag_to_location[iter.second] = new_location[iter.first];
    }
    _location_to_tag.clear();
    for (auto iter : _tag_to_location) {
      _location_to_tag[iter.second] = iter.first;
    }

    for (_u64 old = _nd; old < _max_points; ++old) {
      _final_graph[old].clear();
    }
    _delete_set.clear();
    _empty_slots.clear();
    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    _lazy_done = false;
    _eager_done = false;
    _data_compacted = true;
    stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for tag<->index consolidation: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";
    LOG(INFO) << "Time taken for compact_data(): "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - fnstart).count() << "s.";
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    LockGuard guard(_change_lock);
    if (_nd >= _max_points) {
      return -1;
    }
    unsigned location;
    if (_data_compacted) {
      location = (unsigned) _nd;
      _empty_slots.erase(location);
    } else {
      // no need of delete_lock here, _change_lock will ensure no other thread
      // executes this block of code
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
      _delete_set.erase(location);
    }

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_point(unsigned old_location, unsigned new_location) {
    for (unsigned i = 0; i < _nd; i++)
      for (unsigned j = 0; j < _final_graph[i].size(); j++)
        if (_final_graph[i][j] == old_location)
          _final_graph[i][j] = (unsigned) new_location;

    _final_graph[new_location].clear();
    for (unsigned k = 0; k < _final_graph[_nd].size(); k++)
      _final_graph[new_location].emplace_back(_final_graph[old_location][k]);

    _final_graph[old_location].clear();

    memcpy((void *) (_data + (size_t) _aligned_dim * new_location), _data + (size_t) _aligned_dim * old_location,
           sizeof(T) * _aligned_dim);
    memset((_data + (size_t) _aligned_dim * old_location), 0, sizeof(T) * _aligned_dim);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_frozen_point_to_end() {
    if (_num_frozen_pts == 0)
      return;

    if (_nd == _max_points) {
      LOG(INFO) << "Not repositioning frozen point as it is already at the end.";
      return;
    }
    reposition_point(_nd, _max_points);
    _ep = (_u32) _max_points;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(uint32_t new_max_points) {
    // TODO: Check if the _change_lock and _update_lock are both locked.

    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() == 0);  // should not resize if there are empty slots.

    T *new_data;
    alloc_aligned((void **) &new_data, (new_max_points + 1) * _aligned_dim * sizeof(T), 8 * sizeof(T));
    LOG(INFO) << "Resize to " << new_max_points << " " << _max_points << " with ptr " << (void *) _data << " "
              << (void *) new_data;
    memcpy(new_data, _data, (_max_points + 1) * _aligned_dim * sizeof(T));
    aligned_free(_data);
    _data = new_data;

    _final_graph.resize(new_max_points + 1);

    if (_num_frozen_pts > 0) {
      reposition_point(_max_points, new_max_points);
      _ep = new_max_points;
    }
    _max_points = new_max_points;

    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Resizing took: " << std::chrono::duration<double>(stop - start).count() << "s";
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const Parameters &parameters, const TagT tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    unsigned range = parameters.Get<unsigned>("R");
    //    assert(_has_built);
    std::vector<Neighbor> pool;
    std::vector<Neighbor> tmp;
    tsl::robin_set<unsigned> visited;

    {
      std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
      std::shared_lock<std::shared_timed_mutex> tsl(_tag_lock);
      if (_enable_tags && (_tag_to_location.find(tag) != _tag_to_location.end())) {
        // LOG(INFO) << "Into Locking" ;
        // TODO! This is a repeat of lazy_delete, but we can't call
        // that function because we are taking many locks here. Hence
        // the repeated code.
        tsl.unlock();
        std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);
        std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
        _lazy_done = true;
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
        // LOG(INFO) << "Out Locking" ;
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << " location  == -1. Waiting for unique_lock. ";
      lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      LOG(INFO) << "Thread: " << std::this_thread::get_id() << " Obtained unique_lock. ";
      if (_nd >= _max_points) {
        auto new_max_points = (size_t) (_max_points * INDEX_GROWTH_FACTOR);
        LOG(ERROR) << "Thread: " << std::this_thread::get_id() << ": Increasing _max_points from " << _max_points
                   << " to " << new_max_points << " _nd is: " << _nd;
        resize(new_max_points);
      }
      growth_lock.unlock();
      lock.lock();
      location = reserve_location();
      // TODO: Consider making this a while/do_while loop so that we retry
      // instead of terminating.
      if (location == -1) {
        crash();
      }
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);

      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    pool.clear();
    tmp.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned Lindex = parameters.Get<unsigned>("L");

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited);

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        break;
      }

    prune_neighbors(location, pool, parameters, pruned_list);
    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((_u64) (range * SLACK_FACTOR * 1.05));

    if (pruned_list.empty()) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << "Tag id: " << tag
                << " pruned_list.size(): " << pruned_list.size();
    }

    assert(!pruned_list.empty());
    {
      // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, location);
      v2::LockGuard guard(_locks->wrlock(location));
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
      }
    }

    assert(_final_graph[location].size() <= range);
    inter_insert(location, pruned_list, parameters);
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const Parameters &parameters, const TagT tag,
                                    const std::vector<Neighbor> &initial_pool) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    unsigned range = parameters.Get<unsigned>("R");
    std::vector<Neighbor> pool;
    std::vector<Neighbor> tmp;
    tsl::robin_set<unsigned> visited;

    {
      std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
      std::shared_lock<std::shared_timed_mutex> tsl(_tag_lock);
      if (_enable_tags && (_tag_to_location.find(tag) != _tag_to_location.end())) {
        tsl.unlock();
        std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);
        std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
        _lazy_done = true;
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << " location  == -1. Waiting for unique_lock. ";
      lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      LOG(INFO) << "Thread: " << std::this_thread::get_id() << " Obtained unique_lock. ";
      if (_nd >= _max_points) {
        auto new_max_points = (size_t) (_max_points * INDEX_GROWTH_FACTOR);
        LOG(ERROR) << "Thread: " << std::this_thread::get_id() << ": Increasing _max_points from " << _max_points
                   << " to " << new_max_points << " _nd is: " << _nd;
        resize(new_max_points);
      }
      growth_lock.unlock();
      lock.lock();
      location = reserve_location();
      if (location == -1) {
        crash();
      }
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    pool.clear();
    tmp.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned Lindex = parameters.Get<unsigned>("L");

    // Reuse initial_pool from position seeking instead of starting from scratch
    if (!initial_pool.empty()) {
      // Convert tag-based neighbors from initial_pool to location-based neighbors
      // initial_pool contains tags (from mem_index search_with_tags), need to map to locations
      std::shared_lock<std::shared_timed_mutex> tag_lock(_tag_lock);

      for (const auto &neighbor : initial_pool) {
        auto tag_iter = _tag_to_location.find(neighbor.id);
        if (tag_iter != _tag_to_location.end()) {
          unsigned neighbor_location = tag_iter->second;
          pool.push_back(Neighbor(neighbor_location, neighbor.distance, neighbor.flag));
          visited.insert(neighbor_location);
        }
      }
      tag_lock.unlock();

      // Optionally expand further if initial pool is smaller than Lindex
      // (position seeking may use a different L than insertion)
      if (pool.size() < Lindex) {
        std::vector<unsigned> init_ids;
        for (const auto &neighbor : pool) {
          init_ids.push_back(neighbor.id);
        }
        get_expanded_nodes(location, Lindex, init_ids, pool, visited);
      }
    } else {
      // Fallback to standard search if no initial pool provided
      std::vector<unsigned> init_ids;
      get_expanded_nodes(location, Lindex, init_ids, pool, visited);
    }

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        break;
      }

    prune_neighbors(location, pool, parameters, pruned_list);
    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((_u64) (range * SLACK_FACTOR * 1.05));

    if (pruned_list.empty()) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << "Tag id: " << tag
                << " pruned_list.size(): " << pruned_list.size();
    }

    assert(!pruned_list.empty());
    {
      v2::LockGuard guard(_locks->wrlock(location));
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
      }
    }

    assert(_final_graph[location].size() <= range);
    inter_insert(location, pruned_list, parameters);
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    if ((_eager_done) && (!_data_compacted)) {
      LOG(ERROR) << "Eager delete requests were issued but data was not "
                    "compacted, cannot proceed with lazy_deletes";
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    {
      std::shared_lock<std::shared_timed_mutex> l(_tag_lock);

      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        //        LOG(ERROR) << "Delete tag not found";
        return -1;
      }
      assert(_tag_to_location[tag] < _max_points);
    }

    {
      std::unique_lock<std::shared_timed_mutex> l(_delete_lock);
      std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
      _delete_set.insert(_tag_to_location[tag]);
    }

    {
      std::unique_lock<std::shared_timed_mutex> l(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    return 0;
  }

  // TODO: Check if this function needs a shared_lock on _tag_lock.
  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const tsl::robin_set<TagT> &tags, std::vector<TagT> &failed_tags) {
    if (failed_tags.size() > 0) {
      LOG(ERROR) << "failed_tags should be passed as an empty list";
      return -3;
    }
    if ((_eager_done) && (!_data_compacted)) {
      LOG(INFO) << "Eager delete requests were issued but data was not "
                   "compacted, cannot proceed with lazy_deletes";
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    for (auto tag : tags) {
      //      assert(_tag_to_location[tag] < _max_points);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  /*  Internals of the library */
  // EXPORTS
  template class Index<float, uint32_t>;
  template class Index<int8_t, uint32_t>;
  template class Index<uint8_t, uint32_t>;
}  // namespace pipeann
