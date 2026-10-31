
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

#ifdef DEBUG_BREAK_DOWN_INSERT
#include <atomic>
#include <iomanip>

namespace {
thread_local int tl_thread_id = -1;
std::atomic<int> g_thread_counter{0};

int get_thread_id() {
  if (tl_thread_id < 0) {
    tl_thread_id = g_thread_counter.fetch_add(1);
  }
  return tl_thread_id;
}

struct InsertBreakdown {
  uint64_t prep_us = 0;   // New
  uint64_t search_us = 0;
  uint64_t prune_us = 0;
  uint64_t lock_us = 0;
  uint64_t read_us = 0;
  uint64_t update_us = 0; // compute/memcpy
  uint64_t write_us = 0;
  uint64_t backpressure_us = 0; // New
  uint64_t total_us = 0;
  uint64_t hops_front = 0; // New
  uint64_t hops_pos = 0;   // New
  uint64_t count = 0;
};
thread_local InsertBreakdown tl_breakdown;
thread_local auto tl_last_report = std::chrono::steady_clock::now();

void check_and_print_stats(int tid, std::atomic<uint64_t>& g_search, std::atomic<uint64_t>& g_update, std::atomic<uint64_t>& g_count) {
    tl_breakdown.count++;
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - tl_last_report).count() >= 5) {
       if (tl_breakdown.count > 0) {
          double avg_total = (double)tl_breakdown.total_us / tl_breakdown.count;
          double avg_prep = (double)tl_breakdown.prep_us / tl_breakdown.count;
          double avg_search = (double)tl_breakdown.search_us / tl_breakdown.count;
          double avg_prune = (double)tl_breakdown.prune_us / tl_breakdown.count;
          double avg_lock = (double)tl_breakdown.lock_us / tl_breakdown.count;
          double avg_read = (double)tl_breakdown.read_us / tl_breakdown.count;
          double avg_update = (double)tl_breakdown.update_us / tl_breakdown.count;
          double avg_write = (double)tl_breakdown.write_us / tl_breakdown.count;
          double avg_backpressure = (double)tl_breakdown.backpressure_us / tl_breakdown.count;
          double avg_hops_front = (double)tl_breakdown.hops_front / tl_breakdown.count;
          double avg_hops_pos = (double)tl_breakdown.hops_pos / tl_breakdown.count;
          
          LOG(INFO) << "[Thread " << tid << "] Breakdown (us): "
                    << "Total=" << avg_total
                    << " | Prep=" << avg_prep
                    << " | Search=" << avg_search
                    << " | Prune=" << avg_prune
                    << " | Lock=" << avg_lock
                    << " | Read=" << avg_read
                    << " | Update=" << avg_update
                    << " | Write=" << avg_write
                    << " | Backpressure=" << avg_backpressure
                    << " | Hops(F)=" << avg_hops_front
                    << " | Hops(P)=" << avg_hops_pos;

          // Update global stats
          g_search.fetch_add(tl_breakdown.search_us);
          // Structural update = Total - Search - Backpressure
          uint64_t structural = 0;
          if (tl_breakdown.total_us > tl_breakdown.search_us + tl_breakdown.backpressure_us) {
              structural = tl_breakdown.total_us - tl_breakdown.search_us - tl_breakdown.backpressure_us;
          }
          g_update.fetch_add(structural);
          g_count.fetch_add(tl_breakdown.count);
       }
       tl_breakdown = InsertBreakdown(); // reset
       tl_last_report = now;
    }
}
}
#endif

namespace pipeann {
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);
    void *ctx = reader->get_ctx();

#ifdef DEBUG_BREAK_DOWN_INSERT
    Timer total_timer;
    Timer timer;
#endif

    uint32_t target_id = cur_id++;

    // Phase 1: Setup and Neighbor Discovery

    // 1 - 1) Update PQ data
    std::vector<uint8_t> pq_coords = deflate_vector(point);
    uint64_t pq_offset = target_id * n_chunks;
    {
      static std::mutex pq_mu;
      std::lock_guard<std::mutex> lock(pq_mu);
      if (this->data.size() < pq_offset + n_chunks) {
        while (this->data.size() < pq_offset + n_chunks) {
          this->data.resize(1.5 * this->data.size());
        }
      }
      memcpy(this->data.data() + pq_offset, pq_coords.data(), n_chunks);
    }

    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    coord_map.reserve(2 * this->l_index);

    // 1 - 2) Find neighbors using beam search
    //        Find candidate neighbors and build coord_map
    std::vector<uint64_t> page_ref{};
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.prep_us += timer.elapsed();
    timer.reset();
    QueryStats stats;
#endif
    this->do_beam_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, 
#ifdef DEBUG_BREAK_DOWN_INSERT
                         &stats, 
#else
                         nullptr, 
#endif
                         deletion_set, false,
                         &page_ref, true /* is_position_seeking */);

#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.search_us += timer.elapsed();
    tl_breakdown.hops_front += stats.n_hops_front;
    tl_breakdown.hops_pos += stats.n_hops_pos;
    timer.reset();
#endif

    std::vector<uint32_t> new_nhood;

    // 1 - 3) Select final neighbors using prune_neighbors
    prune_neighbors(coord_map, exp_node_info, new_nhood);

    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
    // lock the pages to write
    std::set<uint64_t> pages_need_to_read;

#ifdef IN_PLACE_RECORD_UPDATE
    std::vector<uint64_t> locs;
    for (auto &nbr : new_nhood) {
      locs.emplace_back(id2loc(nbr));
      pages_need_to_read.insert(node_sector_no(nbr));
    }
    locs.push_back(target_id);
    pages_need_to_read.insert(loc_sector_no(target_id));
    id2loc_.insert_or_assign(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);
#else
    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);
#endif

    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.prune_us += timer.elapsed();
    timer.reset();
#endif
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    auto vec_locked = lock_vec(vec_lock_table, target_id, new_nhood);
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.lock_us += timer.elapsed();
    timer.reset();
#endif

    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    auto &update_buf = read_data->update_buf;
    std::vector<IORequest> reads, writes_4k, writes;
    assert(new_nhood.size() < MAX_N_EDGES);
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      reads.push_back(
          IORequest(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0));
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;
    }

    // write destination pages (for RMW)
    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      writes_4k.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        reads.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      }
      page_buf_map[off / SECTOR_LEN] = update_buf + i * size_per_io;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    for (uint32_t i = 1; i < writes_4k.size(); ++i) {
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        start_idx = i;
      }
      cur_off = writes_4k[i].offset;
    }
    writes_4k.pop_back(); // removes the dummy one.

#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.update_us += timer.elapsed(); // Prep reads
    timer.reset();
#endif

#ifdef DIRECT_READ_CC
    reader->read(reads, ctx);
#else
    reader->read_alloc(reads, ctx, &page_ref);
#endif

#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.read_us += timer.elapsed();
    timer.reset();
#endif

    // update the target node.
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    auto node_buf = offset_to_loc(page_buf_map[sector], locs[new_nhood.size()]);
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);

#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.update_us += timer.elapsed();
    timer.reset();
#endif

    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto r_sector = node_sector_no(new_nhood[i]);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " " << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      if (nhood.size() > this->range) {  // prune neighbors
#ifdef DELTA_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        compute_pq_dists(target_id, nhood.data(), tgt_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        compute_pq_dists(r_nbr_node.id, nhood.data(), nbr_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        std::vector<TriangleNeighbor> tri_pool(nhood.size());

        for (uint32_t k = 0; k < nhood.size(); k++) {
          tri_pool[k].id = nhood[k];
          tri_pool[k].tgt_dis = tgt_dists[k];
          tri_pool[k].distance = nbr_dists[k];
        }
        std::sort(tri_pool.begin(), tri_pool.end());

        int tgt_idx = -1;
        for (int k = 0; k < (int) nhood.size(); ++k) {
          if (tri_pool[k].id == target_id) {
            tgt_idx = k;
            break;
          }
        }
        if (unlikely(tgt_idx == -1)) {
          LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
          exit(-1);
        }
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
#else
        std::vector<float> dists(nhood.size(), 0.0f);
        std::vector<Neighbor> pool(nhood.size());
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        compute_pq_dists(r_nbr_node.id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);
        for (uint32_t k = 0; k < nhood.size(); k++) {
          pool[k].id = nhood[k];
          pool[k].distance = dists[k];
        }
        nhood.clear();
        std::sort(pool.begin(), pool.end());
        this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
#endif
      }

      auto w_sector = loc_sector_no(locs[i]);
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      w_nbr_node.nnbrs = (_u32) nhood.size();
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T));
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
    }

#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.prune_us += timer.elapsed();
    timer.reset();
#endif

    std::vector<uint64_t> write_page_ref;
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.update_us += timer.elapsed();
    timer.reset();
#endif

    reader->wbc_write(writes, ctx, &write_page_ref);

    // Write-through to OdinPackedCache if enabled. update_if_resident is a
    // no-insert, in-place overwrite — we never admit on writes. Admission
    // stays read-driven (via cached_sector_read_batch on misses).
    if (odin_packed_cache_) {
      for (auto &w : writes) {
        // w.len may span multiple sectors (wbc_write handles coalesced writes).
        for (uint64_t off = 0; off < w.len; off += SECTOR_LEN) {
          uint64_t sector_no = (w.offset + off) / SECTOR_LEN;
          odin_packed_cache_->update_if_resident(
              sector_no, (const uint8_t*)((char*)w.buf + off));
        }
      }
    }

#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.write_us += timer.elapsed();
    timer.reset();
#endif

#ifndef IN_PLACE_RECORD_UPDATE
    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      id2loc_.insert_or_assign(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
#endif
    unlock_vec(vec_lock_table, vec_locked);

    // LOG(INFO) << "ID " << target_id << " Target loc " << id2loc(target_id);

    // commit writes (in the background thread.)
#ifdef BG_IO_THREAD
    if (!page_ref.empty()) {
      auto bg_task = new BgTask{
          .thread_data = read_data,
          .writes = std::move(writes),
          .pages_to_unlock = std::move(pages_locked),
          .pages_to_deref = std::move(write_page_ref),
          .enqueue_time = std::chrono::high_resolution_clock::now(),
      };
#ifdef DEBUG_BREAK_DOWN_INSERT
      timer.reset();
#endif
      bg_tasks.push(bg_task);
      bg_tasks.push_notify_all();
#ifdef DEBUG_BREAK_DOWN_INSERT
      tl_breakdown.backpressure_us += timer.elapsed();
#endif
    } else {
      v2::unlockReqs(this->page_lock_table, pages_locked);
    }
    reader->deref(&page_ref, ctx);
#else
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.update_us += timer.elapsed();
    timer.reset();
#endif
    reader->write(writes, ctx);
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.write_us += timer.elapsed();
    timer.reset();
#endif
    v2::unlockReqs(this->page_lock_table, pages_locked);
    reader->deref(&write_page_ref, ctx);

    reader->deref(&page_ref, ctx);
    this->push_query_buf(read_data);
#endif
#ifdef DEBUG_BREAK_DOWN_INSERT
    tl_breakdown.total_us += total_timer.elapsed();
    check_and_print_stats(get_thread_id(), this->global_search_us, this->global_structural_update_us, this->global_insert_op_count);
#endif
    return target_id;
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();
    uint8_t *buf = nullptr;
    pipeann::alloc_aligned((void **) &buf, (MAX_N_EDGES + 1) * SECTOR_LEN, SECTOR_LEN);
    auto timer = pipeann::Timer();
    uint64_t n_tasks = 0;

    // Profiling
    uint64_t total_io_time_us = 0;
    uint64_t bytes_written = 0;
    uint64_t total_write_latency_us = 0;
    uint64_t write_latency_count = 0;

    while (true) {
      auto task = bg_tasks.pop();
      while (task == nullptr) {
        this->bg_tasks.wait_for_push_notify();
        task = bg_tasks.pop();
      }

      pipeann::Timer io_timer;
      reader->write(task->writes, ctx);
      total_io_time_us += io_timer.elapsed();
      
      for (const auto& req : task->writes) {
          bytes_written += req.len;
      }

      // Measure latency just before unlock (or right after write)
      auto now = std::chrono::high_resolution_clock::now();
      auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(now - task->enqueue_time).count();
      total_write_latency_us += latency_us;
      write_latency_count++;

      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        double elapsed_s = (double)timer.elapsed() / 1e6;
        double throughput = (double)n_tasks / elapsed_s;
        double io_util = (double)total_io_time_us / timer.elapsed() * 100.0;
        double mb_per_sec = (double)bytes_written / (1024.0 * 1024.0) / elapsed_s;
        double avg_write_latency = write_latency_count > 0 ? (double)total_write_latency_us / write_latency_count : 0.0;
        
        // Get global stats
        uint64_t g_count = this->global_insert_op_count.load();
        double avg_search = g_count > 0 ? (double)this->global_search_us.load() / g_count : 0.0;
        double avg_struct = g_count > 0 ? (double)this->global_structural_update_us.load() / g_count : 0.0;
        
        double total_latency_sum = avg_search + avg_struct + avg_write_latency;
        double search_pct = total_latency_sum > 0 ? (avg_search / total_latency_sum) * 100.0 : 0.0;
        double struct_pct = total_latency_sum > 0 ? (avg_struct / total_latency_sum) * 100.0 : 0.0;
        double write_pct = total_latency_sum > 0 ? (avg_write_latency / total_latency_sum) * 100.0 : 0.0;

#ifdef IN_PLACE_RECORD_UPDATE
        LOG(INFO) << "[IN_PLACE] Processed " << n_tasks << " tasks, throughput: " << throughput
                  << " tasks/sec."
                  << " [IO Util: " << std::fixed << std::setprecision(1) << io_util << "%"
                  << ", Bandwidth: " << mb_per_sec << " MB/s"
                  << ", Avg Write Latency: " << avg_write_latency << " us]";
#else
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << throughput
                  << " tasks/sec."
                  << " [IO Util: " << std::fixed << std::setprecision(1) << io_util << "%"
                  << ", Bandwidth: " << mb_per_sec << " MB/s"
                  << ", Avg Write Latency: " << avg_write_latency << " us]";
#endif
        
        LOG(INFO) << "Latency Breakdown: Total=" << total_latency_sum << " us "
                  << "| Search: " << avg_search << " us (" << search_pct << "%) "
                  << "| Structural: " << avg_struct << " us (" << struct_pct << "%) "
                  << "| Write: " << avg_write_latency << " us (" << write_pct << "%)";

        if (this->odin_packed_cache_) {
          const auto &cs = this->odin_packed_cache_->stats();
          uint64_t h = cs.hits.load(std::memory_order_relaxed);
          uint64_t m = cs.misses.load(std::memory_order_relaxed);
          uint64_t tot = h + m;
          double hr = tot > 0 ? (double)h / (double)tot * 100.0 : 0.0;
          double br_gb = (double)cs.bytes_resident.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0);
          LOG(INFO) << "[odin-cache] hit=" << std::fixed << std::setprecision(2) << hr << "% "
                    << "hits=" << h << " misses=" << m
                    << " adm=" << cs.admissions.load(std::memory_order_relaxed)
                    << " promo=" << cs.window_promotions.load(std::memory_order_relaxed)
                    << " evict=" << cs.evictions.load(std::memory_order_relaxed)
                    << " wt=" << cs.write_through_hits.load(std::memory_order_relaxed)
                    << " resident=" << std::fixed << std::setprecision(2) << br_gb << "GB";
        }

        timer.reset();
        total_io_time_us = 0;
        bytes_written = 0;
        n_tasks = 0;
        total_write_latency_us = 0;
        write_latency_count = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
