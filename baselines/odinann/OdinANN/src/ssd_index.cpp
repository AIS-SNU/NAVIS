#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>

#include <omp.h>
#include <cmath>
#include "parameters.h"
#include "query_buf.h"
#include "timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "tsl/robin_set.h"

namespace pipeann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader, bool single_file_index,
                              bool tags, Parameters *params)
      : reader(fileReader), data_is_normalized(false), enable_tags(tags) {
    if (m == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = pipeann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

#ifdef DEBUG_BREAK_DOWN_INSERT
    LOG(INFO) << "DEBUG_BREAK_DOWN_INSERT is ENABLED. Detailed insert timing breakdown will be printed per thread.";
#endif

    // this->pq_reader = new LinuxAlignedFileReader();
    if (params != nullptr) {
      this->beam_width = params->Get<uint32_t>("beamwidth");
      this->l_index = params->Get<uint32_t>("L");
      this->range = params->Get<uint32_t>("R");
      this->maxc = params->Get<uint32_t>("C");
      this->alpha = params->Get<float>("alpha");
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.size();

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
    }

    if (medoids != nullptr) {
      delete[] medoids;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(_u64 n_threads) {
    _u64 n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data, this->n_chunks);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
    }

    for (uint64_t i = 0; i < n_buffers; ++i) {
      uint8_t *thread_pq_buf;
      pipeann::alloc_aligned((void **) &thread_pq_buf, 16ul << 20, 256);
      thread_pq_bufs.push_back(thread_pq_buf);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }
#endif

    load_flag = true;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_thread_data() {
    // TODO(gh): destruct thread_queue and other readers.
    for (auto &buf : this->thread_data_bufs) {
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->aligned_pq_coord_scratch);
      pipeann::aligned_free((void *) buf->aligned_pqtable_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      pipeann::aligned_free((void *) buf->update_buf);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(1);
    }

    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());

  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, _u32 num_threads, bool new_index_format, bool use_page_search) {
    std::string pq_table_bin, pq_compressed_vectors, disk_index_file, centroids_file;

    std::string iprefix = std::string(index_prefix);
    pq_table_bin = iprefix + "_pq_pivots.bin";
    pq_compressed_vectors = iprefix + "_pq_compressed.bin";
    disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    centroids_file = disk_index_file + "_centroids.bin";

    std::ifstream index_metadata(disk_index_file, std::ios::binary);

    size_t tags_offset = 0;
    size_t pq_pivots_offset = 0;
    size_t pq_vectors_offset = 0;
    _u64 disk_nnodes;
    _u64 disk_ndims;
    size_t medoid_id_on_file;
    _u64 file_frozen_id;

    if (new_index_format) {
      _u32 nr, nc;

      READ_U32(index_metadata, nr);
      READ_U32(index_metadata, nc);
      LOG(INFO) << "Loading disk index... " << nr << "metadata entries, " << nc << " dimensions.";

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, disk_ndims);
      LOG(INFO) <<"Disk index has " << disk_nnodes << " nodes, " << disk_ndims << " dimensions.";

      READ_U64(index_metadata, medoid_id_on_file);
      LOG(INFO) << "Medoid id of the disk index: " << medoid_id_on_file;

      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      LOG(INFO) << "Max node length (bytes): " << max_node_len;
      LOG(INFO) << "Nodes per sector: " << nnodes_per_sector;

      data_dim = disk_ndims;
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
      if (max_degree != this->range) {
        LOG(ERROR) << "Range mismatch: " << max_degree << " vs " << this->range << ", setting range to " << max_degree;
        this->range = max_degree;
      }

      LOG(INFO) << "Meta-data: # nodes per sector: " << nnodes_per_sector << ", max node len (bytes): " << max_node_len
                << ", max node degree: " << max_degree << ", npts: " << nr << ", dim: " << nc
                << " disk_nnodes: " << disk_nnodes << " disk_ndims: " << disk_ndims;

      if (nnodes_per_sector > this->kMaxElemInAPage) {
        LOG(ERROR) << "nnodes_per_sector: " << nnodes_per_sector << " is greater than " << this->kMaxElemInAPage
                   << ". Please recompile with a higher value of kMaxElemInAPage.";
        return -1;
      }

      READ_U64(index_metadata, this->num_frozen_points);
      READ_U64(index_metadata, file_frozen_id);
      if (this->num_frozen_points == 1) {
        this->frozen_location = file_frozen_id;
        // if (this->num_frozen_points == 1) {
        LOG(INFO) << " Detected frozen point in index at location " << this->frozen_location
                  << ". Will not output it at search time.";
      }
      READ_U64(index_metadata, tags_offset);
      READ_U64(index_metadata, pq_pivots_offset);
      READ_U64(index_metadata, pq_vectors_offset);

      LOG(INFO) << "Tags offset: " << tags_offset << " PQ Pivots offset: " << pq_pivots_offset
                << " PQ Vectors offset: " << pq_vectors_offset;
    } else {  // old index file format
      size_t actual_index_size = get_file_size(disk_index_file);
      size_t expected_file_size;
      READ_U64(index_metadata, expected_file_size);
      if (actual_index_size != expected_file_size) {
        LOG(INFO) << "File size mismatch for " << disk_index_file << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size;
        return -1;
      }

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

      LOG(INFO) << "Disk-Index File Meta-data: # nodes per sector: " << nnodes_per_sector;
      LOG(INFO) << ", max node len (bytes): " << max_node_len;
      LOG(INFO) << ", max node degree: " << max_degree;
    }

    size_per_io = SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    LOG(INFO) << "Size per IO: " << size_per_io;

    index_metadata.close();

    pq_pivots_offset = 0;
    pq_vectors_offset = 0;

    LOG(INFO) << "After single file index check, Tags offset: " << tags_offset
              << " PQ Pivots offset: " << pq_pivots_offset << " PQ Vectors offset: " << pq_vectors_offset;

    size_t npts_u64, nchunks_u64;
    pipeann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
    this->num_points = this->init_num_pts = npts_u64;
    this->n_chunks = nchunks_u64;

    this->cur_id = this->num_points;

    LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors << " offset: " << pq_vectors_offset
              << " num points: " << npts_u64 << " n_chunks: " << nchunks_u64;

    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64, pq_pivots_offset);

    if (disk_nnodes != num_points) {
      LOG(INFO) << "Mismatch in #points for compressed data file and disk "
                   "index file: "
                << disk_nnodes << " vs " << num_points;
      return -1;
    }

    this->data_dim = pq_table.get_dim();
    this->aligned_dim = ROUND_UP(this->data_dim, 8);

    LOG(INFO) << "Loaded PQ centroids and in-memory compressed vectors. #points: " << num_points
              << " #dim: " << data_dim << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;

    // read index metadata
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout and set cur_loc
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    num_medoids = 1;
    medoids = new uint32_t[1];
    medoids[0] = (_u32) (medoid_id_on_file);

    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  _u64 SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::compute_pq_dists(const _u32 src, const _u32 *ids, float *fp_dists, const _u32 count,
                                           uint8_t *aligned_scratch) {
    const _u8 *src_ptr = this->data.data() + (this->n_chunks * src);
    if (unlikely(aligned_scratch == nullptr || count >= 32768)) {
      LOG(ERROR) << "Aligned scratch buffer is null or count is too large: " << count
                 << ". This will lead to memory issues.";
      crash();
    }
    // aggregate PQ coords into scratch
    ::aggregate_coords(ids, count, this->data.data(), this->n_chunks, aligned_scratch);
    // compute distances
    this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, fp_dists, count);
  }

  template<typename T, typename TagT>
  std::vector<_u8> SSDIndex<T, TagT>::deflate_vector(const T *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    std::vector<float> fp_vec(this->data_dim);
    for (uint32_t i = 0; i < this->data_dim; i++) {
      fp_vec[i] = (float) vec[i];
    }
    this->pq_table.deflate_vec(fp_vec.data(), pq_coords.data());
    return pq_coords;
  }

  template<>
  std::vector<_u8> SSDIndex<float>::deflate_vector(const float *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    this->pq_table.deflate_vec(vec, pq_coords.data());
    return pq_coords;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());
      id2loc_.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
    }
    LOG(INFO) << "Loaded " << tags.size() << " tags";
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    size_t num_sectors = node_sector_no(pos);
    std::ifstream disk_reader(_disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    char *node_coords = (offset_to_node(sector_buf.get(), pos));
    memcpy((void *) vector_coords, (void *) node_coords, data_dim * sizeof(T));
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
