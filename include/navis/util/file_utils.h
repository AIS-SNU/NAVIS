#pragma once

#include "navis/types.h"
#include <string>
#include <vector>
#include <fstream>
#include <memory>

namespace navis {

// ---------------------------------------------------------------------------
// .bin format: [4B: num_points] [4B: dimension] [num_points * dim * sizeof(T)]
// ---------------------------------------------------------------------------

// Read metadata only (num_points, dimension)
void get_bin_metadata(const std::string& path, uint32_t& npts, uint32_t& dim);

// Load entire .bin file into memory (caller owns the returned pointer)
template <typename T>
void load_bin(const std::string& path, T*& data, uint32_t& npts, uint32_t& dim);

// Load .bin file into a vector
template <typename T>
void load_bin(const std::string& path, std::vector<T>& data, uint32_t& npts, uint32_t& dim);

// Load a range of vectors from a .bin file (seek to start, read count vectors)
// Useful for streaming large files that don't fit in memory.
template <typename T>
void load_bin_range(const std::string& path, T* data,
                    uint64_t start, uint64_t count, uint32_t dim);

// Save data in .bin format
template <typename T>
void save_bin(const std::string& path, const T* data, uint32_t npts, uint32_t dim);

// ---------------------------------------------------------------------------
// Aligned memory allocation (for CPU buffers, not DMA)
// ---------------------------------------------------------------------------
void* alloc_aligned(size_t size, size_t alignment = 256);
void  free_aligned(void* ptr);

// ---------------------------------------------------------------------------
// File existence check
// ---------------------------------------------------------------------------
bool file_exists(const std::string& path);

// ---------------------------------------------------------------------------
// File size
// ---------------------------------------------------------------------------
uint64_t file_size(const std::string& path);

} // namespace navis
