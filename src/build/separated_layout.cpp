#include "navis/build/build.h"
#include "navis/separated_layout.h"
#include "navis/index/indirection_table.h"
#include "navis/util/file_utils.h"
#include "navis/io/uring_io.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace navis {
namespace build {

// ===========================================================================
// create_separated_layout — write NAVIS separated on-disk layout.
//
// Produces three files:
//   {prefix}_disk.edge         — [ 4KB meta | 4KB edge pages ]
//   {prefix}_disk.vec          — [ 4KB meta | packed vectors ]
//   {prefix}_indirection.bin   — [ u64 npts | edge_loc[npts] | vec_loc[npts] ]
//
// Metadata sector 0 (identical in each file's header for standalone parse):
//   int32 count=9, int32 ndim=1
//   meta[0]=npts, [1]=dim, [2]=medoid, [3]=max_vec_len, [4]=max_edge_len,
//   meta[5]=edges_per_page, [6]=file_kind (0=edge, 1=vec),
//   meta[7]=file_size, [8]=version (=1)
//
// If {prefix}_partition.bin.aligned exists, uses it to reorder edge pages
// (Starling-style locality). Vector region keeps identity mapping.
// ===========================================================================

static constexpr uint64_t kLayoutVersion = 1;
static constexpr int32_t  kMetaCount     = 9;

static void write_meta_sector(UringWriter& out, uint64_t npts, uint64_t dim,
                              uint64_t medoid, uint64_t max_vec_len,
                              uint64_t max_edge_len, uint64_t edges_per_page,
                              uint64_t file_kind, uint64_t file_size) {
    auto meta_buf = std::make_unique<char[]>(SECTOR_LEN);
    std::memset(meta_buf.get(), 0, SECTOR_LEN);
    int32_t count = kMetaCount;
    int32_t ndim  = 1;
    std::memcpy(meta_buf.get(),                   &count, sizeof(int32_t));
    std::memcpy(meta_buf.get() + sizeof(int32_t), &ndim,  sizeof(int32_t));
    uint64_t* m = reinterpret_cast<uint64_t*>(meta_buf.get() + 2 * sizeof(int32_t));
    m[0] = npts;
    m[1] = dim;
    m[2] = medoid;
    m[3] = max_vec_len;
    m[4] = max_edge_len;
    m[5] = edges_per_page;
    m[6] = file_kind;
    m[7] = file_size;
    m[8] = kLayoutVersion;
    out.write(meta_buf.get(), SECTOR_LEN);
}

// Parse the Vamana graph file into an in-memory adjacency list. The file
// format starts with a small header (index_size, max_degree, medoid,
// frozen_num) followed by per-vertex [degree][neighbors].
// Returns the edges indexed by id; also fills out_max_degree / out_medoid.
template <typename T>
static void load_vamana_adjacency(const std::string& graph_path,
                                  uint32_t npts,
                                  std::vector<std::vector<uint32_t>>& out_adj,
                                  uint32_t& out_max_degree,
                                  uint32_t& out_medoid) {
    UringReader graph_in(graph_path);
    uint64_t index_size;
    uint32_t max_degree, medoid;
    uint64_t frozen_num;
    graph_in.read(&index_size, sizeof(uint64_t));
    graph_in.read(&max_degree, sizeof(uint32_t));
    graph_in.read(&medoid,     sizeof(uint32_t));
    graph_in.read(&frozen_num, sizeof(uint64_t));

    out_max_degree = max_degree;
    out_medoid     = medoid;
    out_adj.assign(npts, {});

    for (uint32_t i = 0; i < npts; i++) {
        uint32_t deg;
        graph_in.read(&deg, sizeof(uint32_t));
        uint32_t kept = std::min(deg, max_degree);
        out_adj[i].resize(kept);
        graph_in.read(out_adj[i].data(), kept * sizeof(uint32_t));
        if (deg > max_degree) {
            graph_in.skip((deg - max_degree) * sizeof(uint32_t));
        }
    }
}

// Read a Starling partition file and return id2edge_loc mapping.
// File format:
//   [u64 C (ids per page)]
//   [u64 partition_nums]
//   [u64 nd (total nodes)]
//   For each partition: [u32 count][u32 * C node_ids (padded with garbage)]
static bool try_load_partition(const std::string& prefix,
                               uint32_t npts,
                               uint64_t edges_per_page,
                               std::vector<uint32_t>& id2edge_loc) {
    std::string partition_file = prefix + "_partition.bin.aligned";
    if (!std::filesystem::exists(partition_file)) return false;

    std::ifstream pf(partition_file, std::ios::binary);
    if (!pf.is_open()) return false;

    uint64_t C, partition_nums, nd;
    pf.read(reinterpret_cast<char*>(&C), sizeof(uint64_t));
    pf.read(reinterpret_cast<char*>(&partition_nums), sizeof(uint64_t));
    pf.read(reinterpret_cast<char*>(&nd), sizeof(uint64_t));

    if (C != edges_per_page) {
        std::fprintf(stderr,
            "[layout] partition C=%lu != edges_per_page=%lu — ignoring\n",
            C, edges_per_page);
        return false;
    }

    id2edge_loc.assign(npts, UINT32_MAX);

    std::vector<uint32_t> page_buf(1 + C);
    for (uint64_t p = 0; p < partition_nums; p++) {
        pf.read(reinterpret_cast<char*>(page_buf.data()),
                sizeof(uint32_t) * (1 + C));
        uint32_t cnt = page_buf[0];
        for (uint32_t j = 0; j < cnt && j < C; j++) {
            uint32_t id = page_buf[1 + j];
            if (id < npts) {
                id2edge_loc[id] = static_cast<uint32_t>(p * C + j);
            }
        }
    }

    // Any ids not covered (shouldn't happen with a complete partition) get
    // appended to the tail in order.
    uint32_t tail_loc = static_cast<uint32_t>(partition_nums * C);
    for (uint32_t id = 0; id < npts; id++) {
        if (id2edge_loc[id] == UINT32_MAX) {
            id2edge_loc[id] = tail_loc++;
        }
    }
    std::printf("[layout] loaded Starling partition: %lu pages, C=%lu\n",
                partition_nums, C);
    return true;
}

template <typename T>
void create_separated_layout(const std::string& graph_path,
                             const std::string& data_path,
                             const std::string& output_prefix) {
    // 1. Read metadata from base data file.
    uint32_t npts, ndims;
    get_bin_metadata(data_path, npts, ndims);

    // 2. Load Vamana adjacency (needed for reordering + per-edge writes).
    std::vector<std::vector<uint32_t>> adj;
    uint32_t max_degree, medoid;
    load_vamana_adjacency<T>(graph_path, npts, adj, max_degree, medoid);

    // 3. Compute layout parameters.
    SeparatedLayout layout;
    layout.init(ndims, sizeof(T), max_degree);
    layout.num_points = npts;

    uint64_t max_vec_len    = layout.max_vec_len;
    uint64_t max_edge_len   = layout.max_edge_len;
    uint64_t edges_per_page = layout.edges_per_page;

    // 4. id2edge_loc: identity unless a partition file is present.
    std::vector<uint32_t> id2edge_loc(npts);
    bool used_partition = try_load_partition(output_prefix, npts,
                                             edges_per_page, id2edge_loc);
    if (!used_partition) {
        for (uint32_t i = 0; i < npts; i++) id2edge_loc[i] = i;
    }

    // 5. Determine total edge pages (may be > npts/edges_per_page if
    //    partition emitted a sparse mapping).
    uint32_t max_edge_loc = 0;
    for (uint32_t id = 0; id < npts; id++) {
        max_edge_loc = std::max(max_edge_loc, id2edge_loc[id]);
    }
    uint64_t n_edge_pages = (max_edge_loc / edges_per_page) + 1;
    uint64_t edge_file_size = SECTOR_LEN + n_edge_pages * SECTOR_LEN;
    uint64_t vec_file_size  = layout.total_vec_file_size(npts);

    std::printf("[layout] npts=%u, dim=%u, medoid=%u, R=%u\n",
                npts, ndims, medoid, max_degree);
    std::printf("[layout] max_vec_len=%lu, max_edge_len=%lu, edges_per_page=%lu\n",
                max_vec_len, max_edge_len, edges_per_page);
    std::printf("[layout] edge pages: %lu (%.2f MB), vec region: %.2f MB%s\n",
                n_edge_pages, edge_file_size / (1024.0 * 1024.0),
                vec_file_size / (1024.0 * 1024.0),
                used_partition ? " [Starling reorder]" : "");

    // 6. Write edge file.
    std::string edge_path = output_prefix + "_disk.edge";
    std::string vec_path  = output_prefix + "_disk.vec";

    {
        UringWriter edge_out(edge_path);
        write_meta_sector(edge_out, npts, ndims, medoid,
                          max_vec_len, max_edge_len, edges_per_page,
                          /*file_kind=*/0, edge_file_size);

        // Build page-indexed buffers, one page at a time. For each page we
        // scan id2edge_loc to find which ids map into it.
        // To avoid the O(n_pages * npts) cost, invert the mapping once.
        // Each edge_loc holds at most one id in a valid partition.
        std::vector<std::vector<uint32_t>> page_ids(n_edge_pages);
        std::vector<std::vector<uint32_t>> page_slots(n_edge_pages);
        for (uint32_t id = 0; id < npts; id++) {
            uint32_t loc = id2edge_loc[id];
            uint64_t page = loc / edges_per_page;
            uint64_t slot = loc % edges_per_page;
            page_ids[page].push_back(id);
            page_slots[page].push_back(static_cast<uint32_t>(slot));
        }

        auto page_buf = std::make_unique<char[]>(SECTOR_LEN);
        for (uint64_t p = 0; p < n_edge_pages; p++) {
            std::memset(page_buf.get(), 0, SECTOR_LEN);
            const auto& ids   = page_ids[p];
            const auto& slots = page_slots[p];
            for (size_t k = 0; k < ids.size(); k++) {
                uint32_t id   = ids[k];
                uint32_t slot = slots[k];
                char* entry = page_buf.get() + slot * max_edge_len;
                uint32_t deg = static_cast<uint32_t>(adj[id].size());
                std::memcpy(entry, &deg, sizeof(uint32_t));
                std::memcpy(entry + sizeof(uint32_t),
                            adj[id].data(), deg * sizeof(uint32_t));
            }
            edge_out.write(page_buf.get(), SECTOR_LEN);

            if (p % 4096 == 0 || p + 1 == n_edge_pages) {
                int pct = static_cast<int>(100.0 * (p + 1) / n_edge_pages);
                std::printf("\r[layout] edge: %3d%%", pct);
                std::fflush(stdout);
            }
        }
        std::printf("\n");
    }

    // 7. Write vector file. Sequential in id order under identity mapping.
    {
        UringReader  base_in(data_path);
        base_in.seek(2 * sizeof(uint32_t));
        UringWriter  vec_out(vec_path);

        write_meta_sector(vec_out, npts, ndims, medoid,
                          max_vec_len, max_edge_len, edges_per_page,
                          /*file_kind=*/1, vec_file_size);

        // Pre-buffer in chunks of ~1MB.
        constexpr uint64_t batch = 4096;
        auto vec_batch_buf = std::make_unique<T[]>(batch * ndims);
        auto out_chunk     = std::make_unique<char[]>(batch * max_vec_len);

        uint64_t written = META_SECTOR_LEN;
        uint64_t remaining = npts;
        uint64_t i = 0;
        while (remaining > 0) {
            uint64_t k = std::min<uint64_t>(batch, remaining);
            base_in.read(vec_batch_buf.get(), k * ndims * sizeof(T));
            std::memset(out_chunk.get(), 0, k * max_vec_len);
            for (uint64_t j = 0; j < k; j++) {
                std::memcpy(out_chunk.get() + j * max_vec_len,
                            vec_batch_buf.get() + j * ndims,
                            ndims * sizeof(T));
            }
            vec_out.write(out_chunk.get(), k * max_vec_len);
            written += k * max_vec_len;
            i += k;
            remaining -= k;
            int pct = static_cast<int>(100.0 * i / npts);
            std::printf("\r[layout] vec: %3d%%", pct);
            std::fflush(stdout);
        }
        std::printf("\n");

        // Pad to sector alignment so file size is sector-aligned.
        uint64_t pad = vec_file_size - written;
        if (pad > 0) {
            std::vector<char> zeros(pad, 0);
            vec_out.write(zeros.data(), pad);
        }
    }

    // 8. Save indirection table (v2 format: forward edge + forward vec +
    //    reverse edge + empty_pages). Always written so the loader gets
    //    the reverse map and allocator metadata.
    {
        IndirectionTable indir;
        indir.set_geometry(static_cast<uint32_t>(edges_per_page));
        if (used_partition) {
            indir.init_from_perm(id2edge_loc.data(), /*vec_perm=*/nullptr, npts);
        } else {
            indir.init_identity(npts);
        }
        indir.save(output_prefix, npts);
    }

    std::printf("[layout] Separated layout written:\n");
    std::printf("  edges:       %s (%.2f MB)\n", edge_path.c_str(),
                edge_file_size / (1024.0 * 1024.0));
    std::printf("  vectors:     %s (%.2f MB)\n", vec_path.c_str(),
                vec_file_size / (1024.0 * 1024.0));
    std::printf("  indirection: %s_indirection.bin\n", output_prefix.c_str());
}

// Explicit instantiations
template void create_separated_layout<float>(const std::string&,
                                             const std::string&,
                                             const std::string&);
template void create_separated_layout<int8_t>(const std::string&,
                                              const std::string&,
                                              const std::string&);
template void create_separated_layout<uint8_t>(const std::string&,
                                               const std::string&,
                                               const std::string&);

} // namespace build
} // namespace navis
