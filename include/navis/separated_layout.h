#pragma once

#include "navis/types.h"
#include <cstring>

namespace navis {

// ---------------------------------------------------------------------------
// SeparatedLayout — fully separated on-disk layout for vectors and edges.
//
// Unlike the older weaved layout, there is no chunk packing. Instead:
//   _disk.edge  [ 4KB meta | 4KB edge page × N ]   ids resolved via id2edge_loc
//   _disk.vec   [ 4KB meta | packed max_vec_len ]  ids resolved via id2vec_loc
//
// Edge entry (inside a 4KB edge page):
//   [degree: 4B] [neighbors: max_degree * 4B]
// Vector-position metadata is *not* stored on disk — a host-memory
// IndirectionTable (id → edge_loc, id → vec_loc) resolves both.
//
// A *loc* is a dense slot index:
//   edge_loc ∈ [0, cur_edge_loc)   — which slot in which edge page
//   vec_loc  ∈ [0, cur_vec_loc)    — which packed slot in the vector region
//
// Decoupling edge-page density from vector size is the key win: a 4KB edge
// page now always holds `edges_per_page = 4096 / max_edge_len` entries
// regardless of dim, restoring ~60% of the packing the weaved layout lost
// at high dim. Starling partition files can rewrite id2edge_loc to place
// co-traversed ids on the same page.
// ---------------------------------------------------------------------------
struct SeparatedLayout {
    // Geometry
    uint64_t data_dim         = 0;   // vector dimension
    uint64_t type_size        = 0;   // sizeof(T)
    uint64_t max_vec_len      = 0;   // bytes per vector slot (dim * type_size)
    uint64_t max_edge_len     = 0;   // bytes per edge entry (4 + 4 * max_degree)
    uint64_t edges_per_page   = 0;   // floor(SECTOR_LEN / max_edge_len)
    uint64_t max_degree       = 0;   // max neighbors per node
    uint64_t num_points       = 0;   // total logical points at load

    // File size tracking (updated as we insert)
    uint64_t edge_file_size   = 0;   // bytes, including meta sector
    uint64_t vec_file_size    = 0;   // bytes, including meta sector

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------
    void init(uint64_t dim, uint64_t tsize, uint64_t degree) {
        data_dim        = dim;
        type_size       = tsize;
        max_vec_len     = dim * tsize;
        max_edge_len    = sizeof(uint32_t) * (1 + degree);
        max_degree      = degree;
        edges_per_page  = SECTOR_LEN / max_edge_len;
        if (edges_per_page == 0) {
            // Edge entry larger than a sector — should never happen at sane R/type
            edges_per_page = 1;
        }
    }

    // Called from NavisIndex::load() when loading from disk metadata.
    void init_from_metadata(uint64_t npts, uint64_t dim, uint64_t tsize,
                            uint64_t mvl, uint64_t mel, uint64_t epp) {
        data_dim       = dim;
        type_size      = tsize;
        max_vec_len    = mvl;
        max_edge_len   = mel;
        edges_per_page = epp;
        max_degree     = (mel / sizeof(uint32_t)) - 1;
        num_points     = npts;
    }

    // -----------------------------------------------------------------------
    // Edge-region addressing (by edge_loc)
    // -----------------------------------------------------------------------

    // 1-based edge page number (page 0 holds the metadata sector).
    inline uint64_t edge_page_no(uint64_t edge_loc) const {
        return 1 + edge_loc / edges_per_page;
    }

    // Byte offset of the edge entry within its 4KB page.
    inline uint64_t edge_offset_in_page(uint64_t edge_loc) const {
        return (edge_loc % edges_per_page) * max_edge_len;
    }

    // Absolute byte offset of the 4KB edge page that contains edge_loc.
    inline uint64_t edge_file_offset(uint64_t edge_loc) const {
        return edge_page_no(edge_loc) * SECTOR_LEN;
    }

    // Convert an edge page sector number (1-based) to an absolute byte offset.
    inline uint64_t edge_file_offset_from_sector(uint64_t sector_no) const {
        return sector_no * SECTOR_LEN;
    }

    // -----------------------------------------------------------------------
    // Vector-region addressing (by vec_loc)
    // -----------------------------------------------------------------------

    // Byte offset of the exact vector slot (NOT necessarily sector-aligned).
    inline uint64_t vec_file_offset(uint64_t vec_loc) const {
        return META_SECTOR_LEN + vec_loc * max_vec_len;
    }

    // Sector-aligned read window covering the vector slot.
    // A vector of max_vec_len bytes can straddle up to two sectors if it is
    // smaller than SECTOR_LEN, or up to ceil(max_vec_len/SECTOR_LEN)+1 otherwise.
    inline void vec_aligned_read(uint64_t vec_loc,
                                 uint64_t& aligned_off,
                                 uint64_t& aligned_len) const {
        uint64_t off = vec_file_offset(vec_loc);
        aligned_off  = (off / SECTOR_LEN) * SECTOR_LEN;
        aligned_len  = round_up(off + max_vec_len, SECTOR_LEN) - aligned_off;
    }

    // Max window size produced by vec_aligned_read(): useful for buffer sizing.
    inline uint64_t vec_read_stride() const {
        return round_up(max_vec_len, SECTOR_LEN) + SECTOR_LEN;
    }

    // -----------------------------------------------------------------------
    // Edge entry field accessors (pointer into a loaded edge page)
    // -----------------------------------------------------------------------

    static inline uint32_t edge_degree(const char* edge_ptr) {
        uint32_t d;
        std::memcpy(&d, edge_ptr, sizeof(uint32_t));
        return d;
    }

    static inline void set_edge_degree(char* edge_ptr, uint32_t deg) {
        std::memcpy(edge_ptr, &deg, sizeof(uint32_t));
    }

    static inline const uint32_t* edge_neighbors(const char* edge_ptr) {
        return reinterpret_cast<const uint32_t*>(edge_ptr + sizeof(uint32_t));
    }

    static inline uint32_t* edge_neighbors_mut(char* edge_ptr) {
        return reinterpret_cast<uint32_t*>(edge_ptr + sizeof(uint32_t));
    }

    // Get pointer to the edge entry inside a loaded 4KB page buffer.
    inline char* edge_in_page(char* page_buf, uint64_t edge_loc) const {
        return page_buf + edge_offset_in_page(edge_loc);
    }
    inline const char* edge_in_page(const char* page_buf, uint64_t edge_loc) const {
        return page_buf + edge_offset_in_page(edge_loc);
    }

    // -----------------------------------------------------------------------
    // Size helpers
    // -----------------------------------------------------------------------

    inline uint64_t total_edge_pages(uint64_t n_locs) const {
        return div_round_up(n_locs, edges_per_page);
    }

    inline uint64_t total_edge_file_size(uint64_t n_locs) const {
        return SECTOR_LEN + total_edge_pages(n_locs) * SECTOR_LEN;
    }

    inline uint64_t total_vec_file_size(uint64_t n_vecs) const {
        // Pad to sector alignment so the file size stays sector-aligned.
        return round_up(META_SECTOR_LEN + n_vecs * max_vec_len, SECTOR_LEN);
    }
};

} // namespace navis
