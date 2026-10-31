#include "navis/util/file_utils.h"
#include "navis/io/uring_io.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>

namespace navis {

void get_bin_metadata(const std::string& path, uint32_t& npts, uint32_t& dim) {
    UringReader in(path);
    in.read(&npts, sizeof(uint32_t));
    in.read(&dim, sizeof(uint32_t));
}

template <typename T>
void load_bin(const std::string& path, T*& data, uint32_t& npts, uint32_t& dim) {
    UringReader in(path);
    in.read(&npts, sizeof(uint32_t));
    in.read(&dim, sizeof(uint32_t));

    uint64_t total = static_cast<uint64_t>(npts) * dim;
    data = static_cast<T*>(alloc_aligned(total * sizeof(T)));
    try {
        in.read(data, total * sizeof(T));
    } catch (...) {
        free_aligned(data);
        data = nullptr;
        throw;
    }
}

template <typename T>
void load_bin(const std::string& path, std::vector<T>& data, uint32_t& npts, uint32_t& dim) {
    UringReader in(path);
    in.read(&npts, sizeof(uint32_t));
    in.read(&dim, sizeof(uint32_t));

    uint64_t total = static_cast<uint64_t>(npts) * dim;
    data.resize(total);
    in.read(data.data(), total * sizeof(T));
}

template <typename T>
void load_bin_range(const std::string& path, T* data,
                    uint64_t start, uint64_t count, uint32_t dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);

    // Skip header (8 bytes) + start vectors
    uint64_t offset = 2 * sizeof(uint32_t) + start * dim * sizeof(T);
    in.seekg(static_cast<std::streamoff>(offset));
    if (!in) throw std::runtime_error("Seek failed in: " + path);

    uint64_t bytes = count * dim * sizeof(T);
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("Read failed in: " + path);
}

template <typename T>
void save_bin(const std::string& path, const T* data, uint32_t npts, uint32_t dim) {
    UringWriter out(path);
    out.write(&npts, sizeof(uint32_t));
    out.write(&dim, sizeof(uint32_t));
    uint64_t total = static_cast<uint64_t>(npts) * dim;
    out.write(data, total * sizeof(T));
}

void* alloc_aligned(size_t size, size_t alignment) {
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        throw std::runtime_error("posix_memalign failed with error " + std::to_string(ret));
    }
    std::memset(ptr, 0, size);
    return ptr;
}

void free_aligned(void* ptr) {
    free(ptr);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

uint64_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("Cannot stat file: " + path);
    }
    return static_cast<uint64_t>(st.st_size);
}

// Explicit template instantiations
template void load_bin<float>(const std::string&, float*&, uint32_t&, uint32_t&);
template void load_bin<int8_t>(const std::string&, int8_t*&, uint32_t&, uint32_t&);
template void load_bin<uint8_t>(const std::string&, uint8_t*&, uint32_t&, uint32_t&);
template void load_bin<uint32_t>(const std::string&, uint32_t*&, uint32_t&, uint32_t&);

template void load_bin_range<float>(const std::string&, float*, uint64_t, uint64_t, uint32_t);
template void load_bin_range<int8_t>(const std::string&, int8_t*, uint64_t, uint64_t, uint32_t);
template void load_bin_range<uint8_t>(const std::string&, uint8_t*, uint64_t, uint64_t, uint32_t);

template void load_bin<float>(const std::string&, std::vector<float>&, uint32_t&, uint32_t&);
template void load_bin<int8_t>(const std::string&, std::vector<int8_t>&, uint32_t&, uint32_t&);
template void load_bin<uint8_t>(const std::string&, std::vector<uint8_t>&, uint32_t&, uint32_t&);
template void load_bin<uint32_t>(const std::string&, std::vector<uint32_t>&, uint32_t&, uint32_t&);

template void save_bin<float>(const std::string&, const float*, uint32_t, uint32_t);
template void save_bin<int8_t>(const std::string&, const int8_t*, uint32_t, uint32_t);
template void save_bin<uint8_t>(const std::string&, const uint8_t*, uint32_t, uint32_t);
template void save_bin<uint32_t>(const std::string&, const uint32_t*, uint32_t, uint32_t);

} // namespace navis
