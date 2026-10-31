// baselines/common/navis_layout_reader.h
// Read a NAVIS separated-layout index and yield per-id (vector, edges).
// Shared between navis_to_odinann.cpp and navis_to_freshdiskann.cpp.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bench {

struct NavisLayoutMeta {
    uint64_t npts{0};
    uint64_t dim{0};
    uint64_t medoid{0};
    uint64_t max_vec_len{0};      // bytes per vec slot in _disk.vec
    uint64_t max_edge_len{0};     // bytes per edge slot in _disk.edge (degree+nbrs packed)
    uint64_t edges_per_page{0};
    uint64_t version{0};
};

class NavisLayoutReader {
 public:
    // Opens {prefix}_disk.edge, {prefix}_disk.vec, {prefix}_indirection.bin.
    // Throws std::runtime_error on missing or malformed files.
    explicit NavisLayoutReader(const std::string& prefix);
    ~NavisLayoutReader();

    const NavisLayoutMeta& meta() const { return meta_; }

    // Reads the vector for node `id` into `out` (must be at least
    // meta_.dim * sizeof(float) bytes — assumed float32).
    // Thread-safe via pread on read-only fds.
    void read_vector(uint64_t id, void* out, size_t out_bytes);

    // Decodes the edge entry for node `id`. Writes the actual degree into
    // `*out_degree` and up to (max_edge_len/4 - 1) neighbor ids into `out_nbrs`.
    // `out_nbrs` must have capacity for at least (max_edge_len/4 - 1) u32s.
    void read_edges(uint64_t id, uint32_t* out_nbrs, uint32_t* out_degree);

 private:
    int edge_fd_{-1};
    int vec_fd_{-1};
    NavisLayoutMeta meta_{};
    std::vector<uint32_t> id2edge_loc_;
    std::vector<uint32_t> id2vec_loc_;
};

}  // namespace bench
