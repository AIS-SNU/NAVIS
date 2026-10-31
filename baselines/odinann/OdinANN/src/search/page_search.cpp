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
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const _u64 nnodes_per_sector,
                                           const _u64 num_points) {
    const uint64_t mapping_points = num_points ? num_points : this->num_points;
    id2loc_.reserve(mapping_points);
    loc2id_.reserve(mapping_points + nnodes_per_sector);
    std::string partition_file = index_prefix + "_partition.bin.aligned";
    if (std::filesystem::exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      _u64 C, partition_nums, nd;
      part.read((char *) &C, sizeof(_u64));
      part.read((char *) &partition_nums, sizeof(_u64));
      part.read((char *) &nd, sizeof(_u64));
      if (nnodes_per_sector && num_points && (C != nnodes_per_sector)) {
        LOG(ERROR) << "partition information not correct.";
        exit(-1);
      }
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        for (uint64_t i = p; i < nxt_p; ++i) {
          uint32_t s = part_buf[(i - p) * (1 + nnodes_per_sector)];
          PageArr tmp_arr;
          memcpy(tmp_arr.data(), part_buf.data() + (i - p) * (1 + nnodes_per_sector) + 1,
                 sizeof(unsigned) * nnodes_per_sector);
          for (uint32_t j = 0; j < nnodes_per_sector; ++j) {
            uint64_t loc = i * nnodes_per_sector + j;
            if (j < s) {
              id2loc_.insert_or_assign(tmp_arr[j], loc);
              loc2id_.insert_or_assign(loc, tmp_arr[j]);
            }
          }
        }
      }
      this->cur_loc = partition_nums * nnodes_per_sector;  // aligned.

      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";
    } else {
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and loc2id.
#ifndef NO_MAPPING
#pragma omp parallel for
      for (size_t i = 0; i < this->num_points; ++i) {
        id2loc_.insert_or_assign(i, i);
        loc2id_.insert_or_assign(i, i);
      }
      this->cur_loc = num_points;
      // aligned.
      if (num_points % nnodes_per_sector != 0) {
        cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
      }
#endif
    }
    LOG(INFO) << "Cur location: " << this->cur_loc;
    LOG(INFO) << "Page layout loaded.";
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::page_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
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

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(4096);
    tsl::robin_set<_u64> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    _u32 best_medoid = 0;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids,
                                                               float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      if (nbors_cand_size) {
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < l_search)
            ++cur_list_size;
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    stats->io_us = 0;
    stats->cpu_us = 0;
    // search in in-memory index.
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>;  // <node_id, page_id, page_layout, page_buf>
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;  // <node_id, page_id, page_layout>
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(2 * beam_width);

    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);

    // search on disk.
    while (k < cur_list_size) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;
      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;

      // distribute cache and disk-read nodes
      // 100 us
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        const unsigned pid = id2page(retset[marker].id);
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          // disable nhood cache.
          frontier.push_back(retset[marker].id);
          page_visited.insert(pid);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked, page_locked;
      int n_ios = 0;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;

        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);

        for (_u64 i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          uint64_t page_id = id2page(id);
          auto buf = sector_scratch + sector_scratch_idx * size_per_io;
          PageArr layout;
          if (unlikely(!get_page_layout(page_id, layout))) {
            LOG(ERROR) << "Page layout not found for page " << page_id;
            crash();
          }
          page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          // read the page to the temporary buffer
          frontier_read_reqs.emplace_back(
              IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }

        n_ios = reader->send_read_no_alloc(frontier_read_reqs, ctx);
      }

      // compute remaining nodes in the pages that are fetched in the previous
      // round
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
        char *sector_buf = last_pages.data() + i * SECTOR_LEN;

        // minus one for the vector that is computed previously
        std::vector<std::pair<float, const char *>> vis_cand;
        vis_cand.reserve(nnodes_per_sector);

        // compute exact distances of the vectors within the page
        for (unsigned j = 0; j < nnodes_per_sector; ++j) {
          const unsigned id = page_layout[j];
          if (id == last_io_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          const char *node_buf = sector_buf + j * max_node_len;
          float dist = compute_exact_dists_and_push(node_buf, id);
          vis_cand.emplace_back(dist, node_buf);
        }
        if (vis_cand.size() > 0) {
          std::sort(vis_cand.begin(), vis_cand.end());
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (unsigned j = 0; j < vis_cand.size(); ++j) {
          compute_and_push_nbrs(vis_cand[j].second, nk);
        }
      }
      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      auto io_time_st = std::chrono::high_resolution_clock::now();
      // get last submitted io results, blocking
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; ++i) {
          reader->poll_wait(ctx);
        }
        this->unlock_page_idx(page_idx_lock_table, page_locked);
        this->unlock_idx(idx_lock_table, locked);
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      auto cpu_st = std::chrono::high_resolution_clock::now();
      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        // fill in the last_io_ids() and last_pages() with neighbor buffers.
        memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        for (unsigned j = 0; j < nnodes_per_sector; ++j) {
          unsigned cur_id = layout[j];
          if (cur_id == id) {
            char *node_buf = sector_buf + j * max_node_len;
            compute_exact_dists_and_push(node_buf, id);
            compute_and_push_nbrs(node_buf, nk);
          }
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
