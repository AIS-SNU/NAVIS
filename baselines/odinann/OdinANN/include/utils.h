#pragma once
#include <fcntl.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <sstream>
#include <malloc.h>
#include <unistd.h>
#include <sys/stat.h>

typedef int FileHandle;

#include "distance.h"
#include "log.h"

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) ((((uint64_t) (X) / (Y)) + ((uint64_t) (X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t) (X) / (Y)) + ((uint64_t) (X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t) (X) / (Y)) * (Y))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t) (X) % (uint64_t) (Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)
#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
        // 4KB for unified files
typedef uint64_t _u64;
typedef int64_t _s64;
typedef uint32_t _u32;
typedef int32_t _s32;
typedef uint16_t _u16;
typedef int16_t _s16;
typedef uint8_t _u8;
typedef int8_t _s8;

/* Used to replace exception. */
inline void crash() {
  // #ifndef NDEBUG
  __builtin_trap();
  // #endif
}

static inline bool file_exists(const std::string &name, bool dirCheck = false) {
  int val;
  struct stat buffer;
  val = stat(name.c_str(), &buffer);

  LOG(INFO) << " Stat(" << name.c_str() << ") returned: " << val;
  if (val != 0) {
    switch (errno) {
      case EINVAL:
        LOG(INFO) << "Invalid argument passed to stat()";
        break;
      case ENOENT:
        LOG(INFO) << "File " << name.c_str() << " does not exist";
        break;
      default:
        LOG(INFO) << "Unexpected error in stat():" << errno;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}

inline void open_file_to_write(std::ofstream &writer, const std::string &filename) {
  writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  if (!file_exists(filename))
    writer.open(filename, std::ios::binary | std::ios::out);
  else
    writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);
  if (writer.fail()) {
    LOG(ERROR) << std::string("Failed to open file") + filename + " for write because " << std::strerror(errno);
    crash();
  }
}

inline _u64 get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    _u64 end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    LOG(ERROR) << "Could not open file: " << fname;
    return 0;
  }
}

inline int delete_file(const std::string &fileName) {
  if (file_exists(fileName)) {
    auto rc = ::remove(fileName.c_str());
    if (rc != 0) {
      LOG(ERROR) << "Could not delete file: " << fileName
                 << " even though it exists. This might indicate a permissions issue. "
                    "If you see this message, please contact the pipeann team.";
    }
    return rc;
  } else {
    return 0;
  }
}

namespace pipeann {
  static const size_t MAX_SIZE_OF_STREAMBUF = 2LL * 1024 * 1024 * 1024;

  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3, COSINE = 4 };

  float calc_recall_set_tags(unsigned num_queries, unsigned *gold_std, unsigned dim_gs, unsigned *our_results_tags,
                             unsigned dim_or, unsigned recall_at, unsigned subset_size, std::string gt_tag_filename,
                             std::string current_tag_filename);

  inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
    *ptr = ::aligned_alloc(align, size);
    assert(*ptr != nullptr);
  }

  inline void check_stop(std::string arnd) {
    int brnd;
    LOG(INFO) << arnd;
    std::cin >> brnd;
  }

  inline void aligned_free(void *ptr) {
    if (ptr == nullptr) {
      return;
    }
    free(ptr);
  }

  inline void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
      addr[i] = rng() % (N - size);
    }

    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
      if (addr[i] <= addr[i - 1]) {
        addr[i] = addr[i - 1] + 1;
      }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
      addr[i] = (addr[i] + off) % N;
    }
  }

  // get_bin_metadata functions START
  inline void get_bin_metadata_impl(std::basic_istream<char> &reader, size_t &nrows, size_t &ncols, size_t offset = 0) {
    int nrows_32, ncols_32;
    reader.seekg(offset, reader.beg);
    reader.read((char *) &nrows_32, sizeof(int));
    reader.read((char *) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

  inline void get_bin_metadata(const std::string &bin_file, size_t &nrows, size_t &ncols, size_t offset = 0) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols, offset);
  }
  // get_bin_metadata functions END
  // load_bin functions START
  template<typename T>
  inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim,
                            size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    data = new T[npts * dim];
    reader.read((char *) data, npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin_impl(std::basic_istream<char> &reader, std::vector<T> &data, size_t &npts, size_t &dim,
                            size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    data.resize(npts * dim);
    reader.read((char *) (data.data()), npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0) {
    // OLS
    //_u64            read_blk_size = 64 * 1024 * 1024;
    // cached_ifstream reader(bin_file, read_blk_size);
    // size_t actual_file_size = reader.get_file_size();
    // END OLS
    LOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, std::vector<T> &data, size_t &npts, size_t &dim,
                       size_t offset = 0) {
    // OLS
    //_u64            read_blk_size = 64 * 1024 * 1024;
    // cached_ifstream reader(bin_file, read_blk_size);
    // size_t actual_file_size = reader.get_file_size();
    // END OLS
    LOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }
  // load_bin functions END

  // template<typename T>
  // inline void load_bin(const std::string& bin_file, std::vector<T>& data, size_t& npts, size_t& dim,
  //                      size_t offset = 0) {
  //   // OLS
  //   //_u64            read_blk_size = 64 * 1024 * 1024;
  //   // cached_ifstream reader(bin_file, read_blk_size);
  //   // size_t actual_file_size = reader.get_file_size();
  //   // END OLS
  //   LOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
  //   std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
  //   uint64_t fsize = reader.tellg();
  //   reader.seekg(0);

  //   load_bin_impl<T>(reader, data, npts, dim, offset);
  //   if (fsize != reader.tellg()) {
  //     LOG(ERROR) << "File size mismatch. Expected: " << fsize << ", actual: " << reader.tellg();
  //     exit(-1);
  //   }
  // }

  template<typename T>
  inline uint64_t save_bin_sector_aligned(const std::string &filename, T *data, size_t npts, size_t ndims,
                                          size_t offset = 0) {
    std::ofstream writer;
    open_file_to_write(writer, filename);

    LOG(INFO) << "Writing bin sector aligned: " << filename.c_str();
    writer.seekp(offset, writer.beg);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    LOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

    writer.seekp(offset + METADATA_SIZE, writer.beg);
    writer.write((char *) data, npts * ndims * sizeof(T));
    writer.close();
    LOG(INFO) << "Finished writing bin.";
    return bytes_written;
  }

  inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim,
                            uint32_t **tags = nullptr) {
    std::ifstream reader(bin_file, std::ios::binary);
    LOG(INFO) << "Reading truthset file " << bin_file.c_str() << "...";
    size_t actual_file_size = get_file_size(bin_file);

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    size_t with_tags_actual_file_size = 3 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

    if (actual_file_size == with_tags_actual_file_size)
      truthset_type = 3;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
             << expected_file_size_just_ids;
      LOG(INFO) << stream.str();
      crash();
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *) ids, npts * dim * sizeof(uint32_t));

    if ((truthset_type == 1) || (truthset_type == 3)) {
      dists = new float[npts * dim];
      reader.read((char *) dists, npts * dim * sizeof(float));
    }
    if (truthset_type == 3) {
      *tags = new uint32_t[npts * dim];
      reader.read((char *) *tags, npts * dim * sizeof(uint32_t));
    }
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                       size_t offset = 0) {
    T *ptr;
    load_bin<T>(bin_file, ptr, npts, dim, offset);
    data.reset(ptr);
  }

  template<typename T>
  inline uint64_t save_bin(const std::string &filename, T *data, size_t npts, size_t ndims, size_t offset = 0) {
    std::ofstream writer;
    open_file_to_write(writer, filename);

    LOG(INFO) << "Writing bin: " << filename.c_str();
    writer.seekp(offset, writer.beg);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    LOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

    writer.write((char *) data, npts * ndims * sizeof(T));
    writer.close();
    LOG(INFO) << "Finished writing bin.";
    return bytes_written;
  }

  template<typename T>
  inline uint64_t append_bin(const std::string &filename, T *data, size_t npts, size_t ndims, size_t offset = 0) {
    std::ofstream writer;
    open_file_to_write(writer, filename);

    LOG(INFO) << "Appending bin: " << filename.c_str() << " at offset " << offset;
    writer.seekp(offset, writer.beg);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    LOG(INFO) << "bin append: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

    writer.write((char *) data, npts * ndims * sizeof(T));
    writer.close();
    LOG(INFO) << "Finished appending bin.";
    return bytes_written;
  }

  // load_aligned_bin functions START

  template<typename T>
  inline void load_aligned_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim,
                                    size_t &rounded_dim, size_t offset = 0) {
    int npts_i32, dim_i32;
    reader.seekg(offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));

    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;
    rounded_dim = ROUND_UP(dim, 8);
    LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << ", aligned_dim = " << rounded_dim << "...";
    size_t allocSize = npts * rounded_dim * sizeof(T);
    alloc_aligned(((void **) &data), allocSize, 8 * sizeof(T));

    for (size_t i = 0; i < npts; i++) {
      reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    LOG(INFO) << " Allocated " << allocSize << "bytes and copied data ";
  }

  template<typename T>
  inline void load_aligned_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t &rounded_dim,
                               size_t offset = 0) {
    LOG(INFO) << "Reading bin file " << bin_file << " at offset " << offset << "...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_aligned_bin_impl(reader, data, npts, dim, rounded_dim, offset);
  }

  template<typename InType, typename OutType>
  void convert_types(const InType *srcmat, OutType *destmat, size_t npts, size_t dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (_s64) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
      }
    }
  }

  template<typename T>
  inline void load_aligned_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                               size_t &rounded_dim, size_t offset = 0) {
    T *ptr;
    load_aligned_bin<T>(bin_file, ptr, npts, dim, rounded_dim, offset);
    data.reset(ptr);
  }

  template<typename T>
  inline uint64_t save_data_in_base_dimensions(const std::string &filename, T *data, size_t npts, size_t ndims,
                                               size_t aligned_dim, size_t offset = 0) {
    std::ofstream writer;  //(filename, std::ios::binary | std::ios::out);
    open_file_to_write(writer, filename);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    _u64 bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
    writer.seekp(offset, writer.beg);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    for (size_t i = 0; i < npts; i++) {
      writer.write((char *) (data + i * aligned_dim), ndims * sizeof(T));
    }
    writer.close();
    return bytes_written;
  }

  template<typename T>
  inline void copy_aligned_data_from_file(const std::string bin_file, T *&data, size_t &npts, size_t &dim,
                                          const size_t &rounded_dim, size_t offset = 0) {
    if (data == nullptr) {
      LOG(INFO) << "Memory was not allocated for " << data << " before calling the load function. Exiting...";
      exit(-1);
    }
    std::ifstream reader(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    for (size_t i = 0; i < npts; i++) {
      reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector_l2(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char *) vec + d, _MM_HINT_T1);
  }

  // NOTE: Implementation in utils.cpp.
  void block_convert(std::ofstream &writr, std::ifstream &readr, float *read_buf, _u64 npts, _u64 ndims);

  void normalize_data_file(const std::string &inFileName, const std::string &outFileName);

  // Appends the contents of source_file to the end of destination_file
  inline uint64_t append_to_file(const std::string &destination_file, const std::string &source_file) {
    std::ifstream source(source_file, std::ios::binary);
    if (!source) {
      LOG(ERROR) << "Failed to open source file: " << source_file;
      crash();
    }
    
    std::ofstream destination;
    open_file_to_write(destination, destination_file);
    destination.seekp(0, std::ios::end); // Move to the end of destination file
    
    // Get source file size
    source.seekg(0, std::ios::end);
    uint64_t source_size = source.tellg();
    source.seekg(0, std::ios::beg);
    
    LOG(INFO) << "Appending file " << source_file << " (" << source_size << " bytes) to " << destination_file;
    
    // Use buffer for efficient copy
    const size_t buffer_size = 8 * 1024 * 1024; // 8MB buffer
    std::vector<char> buffer(buffer_size);
    
    uint64_t total_bytes_copied = 0;
    while (source) {
      source.read(buffer.data(), buffer_size);
      std::streamsize bytes_read = source.gcount();
      if (bytes_read > 0) {
        destination.write(buffer.data(), bytes_read);
        total_bytes_copied += bytes_read;
      } else {
        break;
      }
    }
    
    source.close();
    destination.close();
    
    LOG(INFO) << "Successfully appended " << total_bytes_copied << " bytes from " << source_file;
    return total_bytes_copied;
  }

  template<typename T>
  Distance<T> *get_distance_function(Metric m);
}  // namespace pipeann

struct PivotContainer {
  PivotContainer() = default;

  PivotContainer(size_t pivo_id, float pivo_dist) : piv_id{pivo_id}, piv_dist{pivo_dist} {
  }

  bool operator<(const PivotContainer &p) const {
    return p.piv_dist < piv_dist;
  }

  bool operator>(const PivotContainer &p) const {
    return p.piv_dist > piv_dist;
  }

  size_t piv_id;
  float piv_dist;
};

template<typename T>
pipeann::Distance<T> *get_distance_function(pipeann::Metric m);
