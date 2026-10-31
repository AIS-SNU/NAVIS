// baselines/common/navis_layout_reader.cpp
//
// Parses the NAVIS separated-layout on-disk index format.
//
// Metadata sector layout (actual, from src/build/separated_layout.cpp):
//   Byte 0-3  : int32_t  count = 9
//   Byte 4-7  : int32_t  ndim  = 1
//   Byte 8+   : uint64_t[9]:
//     [0] npts          [1] dim          [2] medoid
//     [3] max_vec_len   [4] max_edge_len [5] edges_per_page
//     [6] file_kind     [7] file_size    [8] version
//
// Indirection file (v2 format from src/index/indirection_table.cpp):
//   [u64 npts][u64 cur_edge][u64 cur_vec]
//   [u32 id2edge_loc[npts]][u32 id2vec_loc[npts]]
//   [u64 loc2id_n][u32 loc2id[loc2id_n]]
//   [u64 num_empty][u64 empty_pages[num_empty]]
//
// v1 format (legacy, identity-only): [u64 npts][u32 id2edge_loc[npts]]
// Detection: v2 if file_size > 8 + npts*4 + 16.

#include "navis_layout_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace bench {

// Sector size matches NAVIS kSector / META_SECTOR_LEN.
static constexpr size_t kSector = 4096;

// Byte offset of the first uint64_t metadata field inside the meta sector.
// The meta sector starts with [int32_t count=9][int32_t ndim=1] before the
// uint64_t array (see write_meta_sector in separated_layout.cpp).
static constexpr size_t kMetaHdrBytes = 2 * sizeof(int32_t);  // = 8

static int xopen(const std::string& p) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open " + p + ": " + std::strerror(errno));
    return fd;
}

static void xpread(int fd, void* buf, size_t n, off_t off) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::pread(fd, static_cast<char*>(buf) + got, n - got, off + static_cast<off_t>(got));
        if (r < 0) throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
        if (r == 0) throw std::runtime_error("pread: unexpected EOF");
        got += static_cast<size_t>(r);
    }
}

// Return the file size in bytes; throws on error.
static uint64_t file_size_bytes(int fd) {
    struct stat st{};
    if (::fstat(fd, &st) < 0)
        throw std::runtime_error(std::string("fstat: ") + std::strerror(errno));
    return static_cast<uint64_t>(st.st_size);
}

NavisLayoutReader::NavisLayoutReader(const std::string& prefix) {
    // ------------------------------------------------------------------
    // 1. Open edge and vec files and parse the metadata sector.
    // ------------------------------------------------------------------
    edge_fd_ = xopen(prefix + "_disk.edge");
    vec_fd_  = xopen(prefix + "_disk.vec");

    // The meta sector has [int32 count][int32 ndim] then uint64_t[9].
    // We only need the 9 uint64_t fields; read them from offset kMetaHdrBytes.
    uint64_t hdr[9]{};
    xpread(edge_fd_, hdr, sizeof(hdr), static_cast<off_t>(kMetaHdrBytes));

    meta_.npts           = hdr[0];
    meta_.dim            = hdr[1];
    meta_.medoid         = hdr[2];
    meta_.max_vec_len    = hdr[3];
    meta_.max_edge_len   = hdr[4];
    meta_.edges_per_page = hdr[5];
    // hdr[6] = file_kind, hdr[7] = file_size (not stored in NavisLayoutMeta)
    meta_.version        = hdr[8];

    if (meta_.npts == 0 || meta_.dim == 0
        || meta_.max_edge_len == 0 || meta_.edges_per_page == 0) {
        throw std::runtime_error("NAVIS metadata corrupt or empty for prefix: " + prefix);
    }

    // ------------------------------------------------------------------
    // 2. Load indirection table.
    //    v1: [u64 npts][u32 edge_loc[npts]]                    (identity)
    //    v2: [u64 npts][u64 cur_edge][u64 cur_vec]
    //        [u32 edge_loc[npts]][u32 vec_loc[npts]] ...
    // ------------------------------------------------------------------
    std::string indir_path = prefix + "_indirection.bin";
    int ifd = xopen(indir_path);
    uint64_t indir_fsz = file_size_bytes(ifd);

    uint64_t npts_on_disk = 0;
    xpread(ifd, &npts_on_disk, sizeof(npts_on_disk), 0);
    if (npts_on_disk != meta_.npts) {
        ::close(ifd);
        throw std::runtime_error(
            "indirection npts mismatch: expected " + std::to_string(meta_.npts)
            + " got " + std::to_string(npts_on_disk));
    }

    // v1 expected size: 8 + npts*4
    uint64_t v1_expected = 8 + meta_.npts * 4;
    bool is_v2 = (indir_fsz > v1_expected + 16);

    id2edge_loc_.resize(meta_.npts);
    id2vec_loc_.resize(meta_.npts);

    off_t off = static_cast<off_t>(sizeof(uint64_t));  // past npts
    if (is_v2) {
        // Skip cur_edge (u64) and cur_vec (u64).
        off += static_cast<off_t>(2 * sizeof(uint64_t));
    }

    // Read forward edge map.
    xpread(ifd, id2edge_loc_.data(), meta_.npts * sizeof(uint32_t), off);
    off += static_cast<off_t>(meta_.npts * sizeof(uint32_t));

    if (is_v2) {
        // Read forward vec map.
        xpread(ifd, id2vec_loc_.data(), meta_.npts * sizeof(uint32_t), off);
    } else {
        // v1: vec loc is identity.
        for (uint32_t i = 0; i < meta_.npts; ++i) id2vec_loc_[i] = i;
    }

    ::close(ifd);
}

NavisLayoutReader::~NavisLayoutReader() {
    if (edge_fd_ >= 0) ::close(edge_fd_);
    if (vec_fd_  >= 0) ::close(vec_fd_);
}

// ---------------------------------------------------------------------------
// read_vector
//
// Vector file layout:
//   [4KB meta sector][vec_loc_0 ... vec_loc_N-1]
// Each slot is exactly max_vec_len bytes; not necessarily sector-aligned.
// We read exactly min(out_bytes, dim*sizeof(float)) bytes from the slot.
// ---------------------------------------------------------------------------
void NavisLayoutReader::read_vector(uint64_t id, void* out, size_t out_bytes) {
    if (id >= meta_.npts) throw std::out_of_range("read_vector: id out of range");
    uint64_t loc_idx = id2vec_loc_[static_cast<size_t>(id)];
    // vec_file_offset from SeparatedLayout: META_SECTOR_LEN + loc * max_vec_len
    off_t file_off = static_cast<off_t>(kSector + loc_idx * meta_.max_vec_len);
    size_t n = std::min(out_bytes, static_cast<size_t>(meta_.dim * sizeof(float)));
    xpread(vec_fd_, out, n, file_off);
}

// ---------------------------------------------------------------------------
// read_edges
//
// Edge file layout:
//   [4KB meta sector][edge page 1][edge page 2]...
// Page numbering: page_no = 1 + edge_loc / edges_per_page  (1-based)
// Absolute file offset of the page = page_no * SECTOR_LEN
// Slot within page = (edge_loc % edges_per_page) * max_edge_len
//
// Edge entry format: [degree: u32][neighbors: degree * u32]
// ---------------------------------------------------------------------------
void NavisLayoutReader::read_edges(uint64_t id, uint32_t* out_nbrs, uint32_t* out_degree) {
    if (id >= meta_.npts) throw std::out_of_range("read_edges: id out of range");
    uint64_t loc_idx  = id2edge_loc_[static_cast<size_t>(id)];
    uint64_t page_no  = 1 + loc_idx / meta_.edges_per_page;
    uint64_t slot_idx = loc_idx % meta_.edges_per_page;
    off_t file_off = static_cast<off_t>(page_no * kSector
                                        + slot_idx * meta_.max_edge_len);

    // Read the entire edge entry (degree + all neighbor slots).
    std::vector<char> buf(meta_.max_edge_len);
    xpread(edge_fd_, buf.data(), buf.size(), file_off);

    uint32_t degree = 0;
    std::memcpy(&degree, buf.data(), sizeof(uint32_t));

    // Clamp to the maximum possible neighbors in this layout.
    uint32_t max_nbrs = static_cast<uint32_t>(meta_.max_edge_len / sizeof(uint32_t) - 1);
    if (degree > max_nbrs) degree = max_nbrs;
    *out_degree = degree;

    std::memcpy(out_nbrs, buf.data() + sizeof(uint32_t),
                degree * sizeof(uint32_t));
}

}  // namespace bench
