#include "ssd_index.h"
#include "v2/dynamic_index.h"

#include <index.h>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <string>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int NUM_INSERT_THREADS = 10;
int NUM_SEARCH_THREADS = 32;

int search_mode = BEAM_SEARCH;

int begin_time = 0;
pipeann::Timer globalTimer;

// acutually also shows disk size
void ShowMemoryStatus(const std::string &filename) {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  std::string index_file_name = filename + "_disk.index";
  stat(index_file_name.c_str(), &st);

  LOG(INFO) << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB";
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(const std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T>
inline uint64_t save_bin_test(const std::string &filename, T *id, float *dist, size_t npts, size_t ndims,
                              size_t offset = 0) {
  std::ofstream writer;
  open_file_to_write(writer, filename);

  LOG(INFO) << "Writing bin: " << filename.c_str();
  writer.seekp(offset, writer.beg);
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  LOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < ndims; j++) {
      writer.write((char *) (id + i * ndims + j), sizeof(T));
      writer.write((char *) (dist + i * ndims + j), sizeof(float));
    }
  }
  writer.close();
  LOG(INFO) << "Finished writing bin.";
  return bytes_written;
}

template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, _u32 mem_L, _u64 L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  if (NUM_SEARCH_THREADS == 0) {
    return;
  }
  unsigned *gt_ids = NULL;
  float *gt_dists = NULL;
  size_t gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    calRecall = false;
  }

  if (calRecall) {
    LOG(INFO) << "current truthfile: " << truthset_file;
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float *query_result_dists = new float[recall_at * query_num];
  TagT *query_result_tags = new TagT[recall_at * query_num];

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cerr << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs" 
            << std::setw(12) << "Hop (F)" << std::setw(12) << "Hop (P)" << std::endl;
  std::cerr << "==============================================================="
               "======================================="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);
    latency_stats[i] = stats[i].total_us / 1000.0;  // convert to ms
    // SLEEP DISABLED for matched-harness comparison with bench_odin/bench_navis.
    // if (search_mode == BEAM_SEARCH) {
    //   std::this_thread::sleep_for(std::chrono::milliseconds(2));
    // }
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());
  float recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;
  }

  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });
  float mean_hops_front =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_hops_front; });
  float mean_hops_pos =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_hops_pos; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cerr << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num
            << std::setw(12) << (float) latency_stats[(_u64) (0.50 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.90 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.95 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.99 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.999 * ((double) query_num))] << std::setw(12) << recall << std::setw(12)
            << mean_ios << std::setw(12) << mean_hops_front << std::setw(12) << mean_hops_pos << std::endl;

  LOG(INFO) << "search current time: " << current_time;
  disk_io = mean_ios;

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  sync_index.final_merge(NUM_INSERT_THREADS);
}

template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &insert_vec,
                      size_t dim) {
  pipeann::Timer timer;
  size_t npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  LOG(INFO) << "Begin Insert";
  std::atomic_size_t success(0);
  
  // Capture start hop counts
  uint64_t start_hops_front = sync_index._disk_index->get_total_hops_front();
  uint64_t start_hops_pos = sync_index._disk_index->get_total_hops_pos();

#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
    pipeann::Timer insert_timer;
    sync_index.insert(data_load + dim * i, insert_vec[i]);
    success++;
    insert_latencies[i] = ((double) insert_timer.elapsed());
  }

  float time_secs = timer.elapsed() / 1.0e6f;
  
  // Capture end hop counts
  uint64_t end_hops_front = sync_index._disk_index->get_total_hops_front();
  uint64_t end_hops_pos = sync_index._disk_index->get_total_hops_pos();
  
  double avg_hops_front = (double)(end_hops_front - start_hops_front) / npts;
  double avg_hops_pos = (double)(end_hops_pos - start_hops_pos) / npts;

  std::sort(insert_latencies.begin(), insert_latencies.end());
  LOG(INFO) << "Inserted " << insert_vec.size() << " points in " << time_secs << "s";
  LOG(INFO) << "Avg Insertion Hops (F): " << avg_hops_front;
  LOG(INFO) << "Avg Insertion Hops (P): " << avg_hops_pos;
  LOG(INFO) << "10p insertion time : " << insert_latencies[(size_t) (0.10 * ((double) npts))] << " us";
  LOG(INFO) << "50p insertion time : " << insert_latencies[(size_t) (0.5 * ((double) npts))] << " us";
  LOG(INFO) << "90p insertion time : " << insert_latencies[(size_t) (0.90 * ((double) npts))] << " us";
  LOG(INFO) << "99p insertion time : " << insert_latencies[(size_t) (0.99 * ((double) npts))] << " us";
  LOG(INFO) << "99.9p insertion time : " << insert_latencies[(size_t) (0.999 * ((double) npts))] << " us";
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start, uint64_t n, std::vector<TagT> &delete_tags,
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
  LOG(INFO) << "l_start: " << l_start << " r_start: " << r_start << " n: " << n;

  for (uint64_t i = l_start; i < l_start + n; ++i) {
    delete_tags.push_back(i);
  }

  for (uint64_t i = r_start; i < r_start + n; ++i) {
    insert_tags.push_back(i);
  }

  // load data, load n vecs from r_start.
  int npts_i32, dim_i32;
  std::ifstream reader(data_bin, std::ios::binary | std::ios::ate);
  reader.seekg(0, reader.beg);
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));

  size_t data_dim = dim_i32;
  data_load.resize(n * data_dim);
  reader.seekg(2 * sizeof(int) + r_start * data_dim * sizeof(T), reader.beg);
  reader.read((char *) data_load.data(), sizeof(T) * n * data_dim);
}

template<typename T, typename TagT>
void update(const std::string &data_bin, const unsigned L_disk, int vecs_per_step, int num_steps,
            const std::string &index_prefix, const std::string &query_file, const std::string &truthset_file,
            size_t truthset_l_offset, const int recall_at, const std::vector<_u64> &Lsearch, const unsigned beam_width,
            const uint32_t search_beam_width, const uint32_t search_mem_L, pipeann::Distance<T> *dist_cmp) {
  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_threads", NUM_SEARCH_THREADS + NUM_INSERT_THREADS);
  std::vector<T> data_load;
  size_t dim{};

  pipeann::Timer timer;

  LOG(INFO) << "Loading queries";
  T *query = NULL;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  dim = query_dim;
  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               search_mode, (search_mem_L > 0), NUM_INSERT_THREADS);

  // --- OdinPackedCache plumbing: read ODIN_PACKED_CACHE_GB env var.
  {
    const char* env = std::getenv("ODIN_PACKED_CACHE_GB");
    uint64_t cache_gb = env ? (uint64_t) std::atoll(env) : 0;
    if (cache_gb > 0) {
      LOG(INFO) << "[odin-cache] Enabling OdinPackedCache with "
                << cache_gb << " GB budget";
      sync_index._disk_index->enable_odin_packed_cache(cache_gb << 30);
    } else {
      LOG(INFO) << "[odin-cache] OdinPackedCache DISABLED (baseline odin)";
    }
  }

  // --- OdinPackedCache stats dump helper. Called at both normal and exit(0) paths.
  auto dump_packed_cache_stats = [&]() {
    if (auto* c = sync_index._disk_index->odin_packed_cache()) {
      const auto &s = c->stats();
      const std::string stats_path = std::string(index_prefix) + "_packed_cache_stats.json";
      std::ofstream fout(stats_path);
      if (fout) {
        fout << "{\n"
             << "  \"hits\": "              << s.hits.load()              << ",\n"
             << "  \"misses\": "            << s.misses.load()            << ",\n"
             << "  \"admissions\": "        << s.admissions.load()        << ",\n"
             << "  \"window_promotions\": " << s.window_promotions.load() << ",\n"
             << "  \"evictions\": "         << s.evictions.load()         << ",\n"
             << "  \"write_through_hits\": "<< s.write_through_hits.load()<< ",\n"
             << "  \"bytes_resident\": "    << s.bytes_resident.load()    << "\n"
             << "}\n";
        LOG(INFO) << "[odin-cache] Stats written to " << stats_path;
      }
    }
  };

  // ODIN_SEARCH_ONLY=1: skip insert path entirely, run warmup + measure
  // search loops, reset cache stats between phases. Used by NAVIS misc
  // experiments (Exp 3 / 4) -- the JSON file then reflects the measure
  // phase only.
  {
    bool search_only_env = (std::getenv("ODIN_SEARCH_ONLY") &&
                            std::string(std::getenv("ODIN_SEARCH_ONLY")) == "1");
    if (search_only_env) {
      int warmup_passes_env  = std::getenv("ODIN_WARMUP_PASSES")
                              ? std::atoi(std::getenv("ODIN_WARMUP_PASSES")) : 0;
      int measure_passes_env = std::getenv("ODIN_MEASURE_PASSES")
                              ? std::atoi(std::getenv("ODIN_MEASURE_PASSES")) : 1;
      LOG(INFO) << "[odin-search-only] warmup=" << warmup_passes_env
                << " measure=" << measure_passes_env;
      std::string baseFn = GetTruthFileName(truthset_file, truthset_l_offset);
      double dummy = 0.0;
      for (int w = 0; w < warmup_passes_env; ++w) {
        sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L,
                           Lsearch[0], search_beam_width, sync_index, baseFn,
                           false, true, dummy);
      }
      if (auto* c = sync_index._disk_index->odin_packed_cache()) c->reset_stats();
      for (int m = 0; m < measure_passes_env; ++m) {
        sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L,
                           Lsearch[0], search_beam_width, sync_index, baseFn,
                           false, true, dummy);
      }
      dump_packed_cache_stats();
      exit(0);
    }
  }

  uint64_t res = 0;

  std::string currentFileName = GetTruthFileName(truthset_file, res + truthset_l_offset);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);

  std::vector<double> ref_diskio;

  for (size_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width, sync_index,
                       currentFileName, false, true, diskio);
    ref_diskio.push_back(diskio);
  }

  // LOG(ERROR) << "If you see this message, the search before update is done.";
  // exit(-1);

  int inMemorySize = 0;
  std::future<void> merge_future;
  uint64_t index_npts = sync_index._disk_index->num_points;
  const bool force_final_store = []() {
    const char *value = std::getenv("ODIN_FORCE_FINAL_STORE");
    return value != nullptr && std::string(value) == "1";
  }();
  for (int i = 0; i < num_steps; i++) {
    LOG(INFO) << "Batch: " << i << " Total Batch : " << num_steps;
    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    uint64_t ed = st + index_npts;
    LOG(INFO) << "st: " << st << " ed: " << ed;
    get_trace<T, TagT>(data_bin, st, ed, vecs_per_step, delete_vec, insert_vec, data_load);

#ifdef SEARCH_ONLY_TEST_WO_INSERTION
    LOG(INFO) << "Search only test without insertion.";
    // set dummy empty function for insertion kernel.
    auto dummy_insertion_kernel = [](T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index,
                                     std::vector<TagT> &insert_vec, size_t dim) {
      return;
    };
    std::future<void> insert_future = std::async(std::launch::async, dummy_insertion_kernel, data_load.data(),
                                                 std::ref(sync_index), std::ref(insert_vec), dim);
#else
    std::future<void> insert_future = std::async(std::launch::async, insertion_kernel<T, TagT>, data_load.data(),
                                                 std::ref(sync_index), std::ref(insert_vec), dim);
#endif

    int total_queries = 0;
    std::future_status insert_status;
    do {
      insert_status = insert_future.wait_for(std::chrono::seconds(5));
      if (insert_status == std::future_status::deferred) {
        LOG(INFO) << "deferred\n";
      } else if (insert_status == std::future_status::timeout) {
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        LOG(INFO) << "Number of vectors: " << sync_index._disk_index->cur_id;
        double dummy;
        // for (uint32_t j = 0; j < Lsearch.size(); ++j) {
        sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[0], search_beam_width,
                           sync_index, currentFileName, false, false, dummy);
        sleep(1);
        // }
        total_queries += query_num;
        LOG(INFO) << "Queries processed: " << total_queries;
      }
      if (insert_status == std::future_status::ready) {
        LOG(INFO) << "Insertions complete!\n";
      }
    } while (insert_status != std::future_status::ready);

    inMemorySize += insert_vec.size();

    LOG(INFO) << "Search after update, current vector number: " << res;

    res += vecs_per_step;
    currentFileName = GetTruthFileName(truthset_file, res + truthset_l_offset);

    std::vector<double> disk_ios;
    for (size_t j = 0; j < Lsearch.size(); ++j) {
      double diskio = 0;
      sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width,
                         sync_index, currentFileName, false, true, diskio);
      disk_ios.push_back(diskio);
    }

    if (i == num_steps - 1 && (num_steps >= 10 || force_final_store)) {
      // Normal runs preserve the historical >=10-step condition. The env
      // override exists for storage-tracking smoke runs only.
      LOG(INFO) << "Store the last index to disk.";
      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>, std::ref(sync_index));
      std::future_status merge_status;
      do {
        merge_status = merge_future.wait_for(std::chrono::seconds(10));
      } while (merge_status != std::future_status::ready);
      LOG(INFO) << "Store finished.";
      dump_packed_cache_stats();
      exit(0);
    }
  }

  // Normal exit path (num_steps < 10): dump cache stats before index destruction.
  dump_packed_cache_stats();
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--storage-capabilities") {
#ifdef NO_POLLUTE_ORIGINAL
    std::cout << "shadow-copy=0 force-final-store=1" << std::endl;
#else
    std::cout << "shadow-copy=1 force-final-store=1" << std::endl;
#endif
    return 0;
  }

  if (argc < 18) {
    LOG(INFO) << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk>"
              << " <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode>"
              << " <index_prefix> <query_file> <truthset_prefix> <truthset_l_offset> <recall@>"
              << " <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    exit(-1);
  }

  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);

  // 1M vectors per step.
  int vecs_per_step = (int) std::atoi(argv[arg_no++]);

  // 100 steps for 100M + 100M test, 200 steps for 800M + 200M test.
  int num_steps = (int) std::atoi(argv[arg_no++]);

  NUM_INSERT_THREADS = (int) std::atoi(argv[arg_no++]);
  NUM_SEARCH_THREADS = (int) std::atoi(argv[arg_no++]);
  LOG(INFO) << "num insert threads: " << NUM_INSERT_THREADS;
  LOG(INFO) << "num search threads: " << NUM_SEARCH_THREADS;

  search_mode = std::atoi(argv[arg_no++]);
  LOG(INFO) << "search mode: " << search_mode;

  if (search_mode != BEAM_SEARCH && search_mode != PAGE_SEARCH) {
    LOG(INFO) << "search mode should be 0 (beam) or 1 (page)";
    exit(-1);
  }

  std::string index_prefix(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);

  // truthsets lie in the same folder, with gt_X.bin as the truthset of [0, x + 100M) vectors.
  std::string truthset(argv[arg_no++]);

  // We generate truth files every 1M vectors, gt_X.bin is the truthset of [0, x + 100M) vectors.
  // This should be 0 for 100M + 100M test, and 700M for 800M + 200M test.
  size_t truthset_l_offset = (size_t) std::atoll(argv[arg_no++]);
  int recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned search_beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned search_mem_L = (unsigned) std::atoi(argv[arg_no++]);  // 0 if no mem_index is used.
  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;

  if (std::string(argv[1]) == std::string("int8")) {
    pipeann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                             truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                             &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    pipeann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                              truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                              &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    pipeann::DistanceL2 dist_cmp;
    update<float, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                            truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                            &dist_cmp);
  } else
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
}
