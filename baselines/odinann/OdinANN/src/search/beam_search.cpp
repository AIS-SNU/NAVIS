#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_beam_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref,
                                         bool is_position_seeking) {
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    auto query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(mem_L + 10 * l_search);
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);
    _u32 best_medoid = medoids[0];

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // Do not use optimized start point.
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;
    bool coord_scratch_exhausted = false;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    while (k < cur_list_size && !coord_scratch_exhausted) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (is_position_seeking) {
            total_hops_pos.fetch_add(1, std::memory_order_relaxed);
        } else {
            total_hops_front.fetch_add(1, std::memory_order_relaxed);
        }
        
        if (stats != nullptr) {
          stats->n_hops++;
          if (is_position_seeking) {
            stats->n_hops_pos++;
          } else {
            stats->n_hops_front++;
          }
        }
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (_u64 i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          if (loc == kInvalidID) {
            continue;
          }
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
        cached_sector_read_batch(frontier_read_reqs, ctx, &page_ref, is_position_seeking);

        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
        this->unlock_idx(idx_lock_table, locked);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        char *node_disk_buf = offset_to_loc(sector_buf, loc);
        unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
        _u64 nnbrs = (_u64) (*node_buf);
        T *node_fp_coords = offset_to_node_coords(node_disk_buf);
        // Search only needs one temporary vector. Position-seeking insertions
        // retain distinct pointers in coord_map, so bound their scratch use.
        if (coord_map != nullptr && unlikely(data_buf_idx >= MAX_N_CMPS)) {
          LOG(ERROR) << "Beam-search coordinate scratch exhausted after " << data_buf_idx << " vectors";
          coord_scratch_exhausted = true;
          break;
        }
        T *node_fp_coords_copy = coord_map == nullptr
                                   ? data_buf
                                   : data_buf + (data_buf_idx++ * aligned_dim);
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

        if (coord_map != nullptr) {
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
        }
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

        unsigned *node_nbrs = (node_buf + 1);
        if (unlikely(nnbrs > MAX_N_EDGES)) {
          LOG(ERROR) << "Invalid neighbor count " << nnbrs << " for node " << id;
          continue;
        }

        std::vector<unsigned> valid_nbrs;
        valid_nbrs.reserve(nnbrs);
        uint64_t id_limit = this->cur_id.load(std::memory_order_acquire);
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned raw_id = node_nbrs[m];
          if (unlikely(raw_id == kInvalidID || raw_id >= id_limit)) {
            continue;
          }
          valid_nbrs.push_back(raw_id);
        }

        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        if (!valid_nbrs.empty()) {
          compute_dists(valid_nbrs.data(), valid_nbrs.size(), dist_scratch);
        }
        if (stats != nullptr) {
          stats->n_cmps += (double) valid_nbrs.size();
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < valid_nbrs.size(); ++m) {
          unsigned id = valid_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            // variable search_L for deleted nodes.
            // Return position in sorted list where nn inserted.

            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

            if (cur_list_size < l_search) {
              ++cur_list_size;
              if (unlikely(cur_list_size >= retset.size())) {
                retset.resize(2 * cur_list_size);
              }
            }

            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated due to neighbors of n.
          }
        }

        if (dyn_search_l) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          _u32 tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::beam_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search(query, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, nullptr, stats,
                         deleted_nodes, dyn_search_l);
    _u64 res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::cached_sector_read_batch(std::vector<IORequest> &reads,
                                                   void *ctx,
                                                   std::vector<uint64_t> *page_ref,
                                                   bool is_position_seeking) {
    // Fast path: OdinPackedCache disabled → byte-for-byte baseline behavior.
    if (!odin_packed_cache_) {
#ifdef DIRECT_READ_CC
      reader->read(reads, ctx);
#else
      reader->read_alloc(reads, ctx, page_ref);
#endif
      return;
    }

    // Cache enabled. 3-tier lookup:
    //   (1) RMW v2::cache: freshest in-flight bytes from the update path.
    //       Mirrors read_alloc: on hit, ref-increment and push sector into
    //       page_ref so the later reader->deref(&page_ref, ctx) drops the
    //       refcount symmetrically.
    //   (2) OdinPackedCache: hot read-cached edge/vector pages. Admission
    //       is driven by position-seeking (W-TinyLFU); search-path hits
    //       go through get_readonly() which does NOT promote.
    //   (3) Miss: collect into miss_reqs, issue a single batched disk
    //       read, admit into OdinPackedCache on position-seeking.
    std::vector<IORequest> miss_reqs;
    miss_reqs.reserve(reads.size());

    for (size_t i = 0; i < reads.size(); ++i) {
      const auto &r = reads[i];
      uint64_t sector_no = r.offset / SECTOR_LEN;

      // Tier 1: RMW page cache (freshest bytes — mirrors read_alloc path).
      // ref=true is required when page_ref is non-null so the later
      // reader->deref drops the refcount; get() is a no-op on miss so
      // there's no spurious refcount bump.
      bool v2_ref = (page_ref != nullptr);
      if (v2::cache.get(sector_no, (uint8_t *) r.buf, v2_ref)) {
        if (page_ref != nullptr) {
          page_ref->push_back(sector_no);
        }
        continue;
      }

      // Tier 2: OdinPackedCache. Position-seeking drives W-TinyLFU
      // admission state; search is readonly (no promotion).
      bool odin_hit = is_position_seeking
                        ? odin_packed_cache_->get(sector_no, (uint8_t *) r.buf)
                        : odin_packed_cache_->get_readonly(sector_no, (uint8_t *) r.buf);
      if (odin_hit) {
        // No v2::cache refcount taken → do NOT push into page_ref.
        continue;
      }

      // Tier 3: miss. Defer to batched disk read below.
      miss_reqs.push_back(r);
    }

    if (!miss_reqs.empty()) {
#ifdef DIRECT_READ_CC
      reader->read(miss_reqs, ctx);
#else
      // read_alloc will: (a) probe v2::cache once more per req (harmless
      // double-check; tier-1 already drained the resident subset), (b) on
      // real miss, send_io + v2::cache.put(..., ref=true), (c) push every
      // sector in miss_reqs into page_ref. That keeps the deref bookkeeping
      // consistent with the tier-1 hits we pushed explicitly above.
      reader->read_alloc(miss_reqs, ctx, page_ref);
#endif

      // Admission into OdinPackedCache is insert-path only (driven by
      // position-seeking); search-path misses are NOT promoted, by design
      // — otherwise benchmark query repetition would monopolize the cache.
      if (is_position_seeking) {
        for (auto &m : miss_reqs) {
          uint64_t sector_no = m.offset / SECTOR_LEN;
          odin_packed_cache_->admit(sector_no, (uint8_t *) m.buf);
        }
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
