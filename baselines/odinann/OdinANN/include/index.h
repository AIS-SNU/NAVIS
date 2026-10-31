#pragma once

#include <cassert>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"

#include "v2/lock_table.h"

#include "distance.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"

#include "neighbor.h"

#define OVERHEAD_FACTOR 1.1
#define SLACK_FACTOR 1.3

namespace pipeann {
  inline double estimate_ram_usage(size_t size, size_t dim, size_t datasize, size_t degree) {
    double graph_size = (double) size * (double) degree * (double) sizeof(unsigned) * SLACK_FACTOR;
    size_t data_size = size * ROUND_UP(dim, 8) * datasize;
    return OVERHEAD_FACTOR * (graph_size + data_size);
  }

  template<typename T, typename TagT = uint32_t>
  class Index {
   public:
    Index(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
          const bool save_index_in_one_file, const bool enable_tags = false);

    ~Index();

    // Public Functions for Static Support

    // checks if data is consolidated, saves graph, metadata and associated
    // tags.
    void save(const char *filename);

    _u64 save_graph(std::string filename, size_t offset = 0);
    _u64 save_data(std::string filename, size_t offset = 0);
    _u64 save_tags(std::string filename, size_t offset = 0);
    _u64 save_delete_list(const std::string &filename, size_t offset = 0);

    void load(const char *index_file);

    void load_from_disk_index(const std::string &filename);
    size_t disk_npts, range;

    size_t load_graph(const std::string filename, size_t expected_num_points, size_t offset = 0);

    size_t load_data(std::string filename, size_t offset = 0);

    size_t load_tags(const std::string tag_file_name, size_t offset = 0);
    size_t load_delete_set(const std::string &filename, size_t offset = 0);

    size_t get_num_points();

    void build(const char *filename, const size_t num_points_to_load, Parameters &parameters,
               const std::vector<TagT> &tags = std::vector<TagT>());

    void build(const char *filename, const size_t num_points_to_load, Parameters &parameters, const char *tag_filename);
    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    std::pair<uint32_t, uint32_t> search(const T *query, const size_t K, const unsigned L, unsigned *indices,
                                         float *distances = nullptr);

    std::pair<uint32_t, uint32_t> search(const T *query, const uint64_t K, const unsigned L,
                                         std::vector<unsigned> init_ids, uint64_t *indices, float *distances);

    size_t search_with_tags(const T *query, const uint64_t K, const unsigned L, TagT *tags, float *distances,
                            std::vector<T *> &res_vectors);

    size_t search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags, float *distances);

    std::pair<uint32_t, uint32_t> search(const T *query, const size_t K, const unsigned L,
                                         std::vector<NeighborTag<TagT>> &best_L_tags);

    void clear_index();

    // Public Functions for Incremental Support

    /* insertions possible only when id corresponding to tag does not already
     * exist in the graph */
    int insert_point(const T *point, const Parameters &parameter,
                     const TagT tag);  // only keep point, tag, parameters

    // Overload that accepts an initial explored pool to reuse neighbors from position seeking
    int insert_point(const T *point, const Parameters &parameter,
                     const TagT tag, const std::vector<Neighbor> &initial_pool);

    // call before triggering deleteions - sets important flags required for
    // deletion related operations
    int enable_delete();

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK.
    int lazy_delete(const TagT &tag);

    // Record deleted points now and restructure graph later. Add to failed_tags
    // if tag not found.
    int lazy_delete(const tsl::robin_set<TagT> &tags, std::vector<TagT> &failed_tags);

    // return _data and tag_to_location offset
    // diskv2 API
    void iterate_to_fixed_point(const T *node_coords, const unsigned Lindex, std::vector<Neighbor> &expanded_nodes_info,
                                tsl::robin_map<uint32_t, T *> &coord_map, bool return_frozen_pt = true);

    // return immediately after "approx" converge.
    uint32_t search_with_tags_fast(const T *node_coords, const unsigned L, TagT *tags, float *dists);

    // convenient access to graph + data (aligned)
    const std::vector<std::vector<unsigned>> *get_graph() const {
      return &this->_final_graph;
    }
    T *get_data();
    const std::unordered_map<unsigned, TagT> *get_tags() const {
      return &this->_location_to_tag;
    };
    // Replaces every tag t with f(t), e.g. after the disk index that the tags
    // refer to renumbers its vertices.
    template<typename F>
    void remap_tags(F f) {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _tag_to_location.clear();
      for (auto &[loc, tag] : _location_to_tag) {
        tag = f(tag);
        _tag_to_location[tag] = loc;
      }
    }
    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    void reposition_frozen_point_to_end();
    void reposition_point(unsigned old_location, unsigned new_location);

    void compact_frozen_point();

    void consolidate(Parameters &parameters);

    // void save_index_as_one_file(bool flag);

    void get_active_tags(tsl::robin_set<TagT> &active_tags);

    int get_vector_by_tag(TagT &tag, T *vec);

    // This variable MUST be updated if the number of entries in the metadata
    // change.
    static const int METADATA_ROWS = 5;

    /*  Internals of the library */
   public:
    std::vector<std::vector<unsigned>> _final_graph;

    // generates one frozen point that will never get deleted from the
    // graph
    int generate_frozen_point();

    // determines navigating node of the graph by calculating medoid of data
    unsigned calculate_entry_point();

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(const T *node_coords, const unsigned Lindex,
                                                         const std::vector<unsigned> &init_ids,
                                                         std::vector<Neighbor> &expanded_nodes_info,
                                                         tsl::robin_set<unsigned> &expanded_nodes_ids,
                                                         std::vector<Neighbor> &best_L_nodes, bool ret_frozen = true);

    void get_expanded_nodes(const size_t node, const unsigned Lindex, std::vector<unsigned> init_ids,
                            std::vector<Neighbor> &expanded_nodes_info, tsl::robin_set<unsigned> &expanded_nodes_ids);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list, const Parameters &parameter);

    void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool, const Parameters &parameter,
                         std::vector<unsigned> &pruned_list);

    void occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result);

    void occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result, std::vector<float> &occlude_factor);

    void link(Parameters &parameters);

    // Support for Incremental Indexing
    int reserve_location();
    void release_location();

    // Support for resizing the index
    // This function must be called ONLY after taking the _change_lock and
    // _update_lock.
    // Anything else in a MT environment will lead to an inconsistent index.
    void resize(uint32_t new_max_points);

    // renumber nodes, update tag and location maps and compact the graph, mode
    // = _compacted_lazy_deletions in case of lazy deletion and
    // _compacted_eager_deletions in
    // case of eager deletion
    void compact_data();

    // WARNING: Do not call consolidate_deletes without acquiring change_lock_
    // Returns number of live points left after consolidation
    size_t consolidate_deletes(const Parameters &parameters);

   public:
    std::shared_timed_mutex _tag_lock;  // reader-writer lock on
                                        // _tag_to_location and
    std::mutex _change_lock;            // Lock taken to synchronously modify _nd

    T *_data = nullptr;  // coordinates of all base points
    // T *_pq_data =
    //    nullptr;  // coordinates of pq centroid corresponding to every point
    Distance<T> *_distance = nullptr;
    pipeann::Metric _dist_metric;

    size_t _dim;
    size_t _aligned_dim;
    size_t _nd = 0;          // number of active points i.e. existing in the graph
    size_t _max_points = 0;  // total number of points in given data set
    size_t _num_frozen_pts = 0;
    unsigned _width = 0;
    unsigned _ep = 0;
    bool _has_built = false;
    bool _saturate_graph = false;
    bool _save_as_one_file = false;
    bool _dynamic_index = false;
    bool _enable_tags = false;

    char *_opt_graph;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    // flags for dynamic indexing
    std::unordered_map<TagT, unsigned> _tag_to_location;
    std::unordered_map<unsigned, TagT> _location_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    // bool _can_delete = false;  // only true if deletes can be done (if
    // enabled)
    bool _eager_done = false;     // true if eager deletions have been made
    bool _lazy_done = false;      // true if lazy deletions have been made
    bool _data_compacted = true;  // true if data has been consolidated
    bool _is_saved = false;       // Gopal. Checking if the index is already saved.

    v2::LockTable *_locks = nullptr;
    // v2::SparseLockTable<uint64_t> _locks;  // Sparse lock table for _final_graph.

    std::shared_timed_mutex _delete_lock;  // Lock on _delete_set and
                                           // _empty_slots when reading and
                                           // writing to them
    // _location_to_tag, has a shared lock
    // and exclusive lock associated with
    // it.
    std::shared_timed_mutex _update_lock;  // coordinate save() and any change
                                           // being done to the graph.

    const float INDEX_GROWTH_FACTOR = 1.5f;
  };
}  // namespace pipeann
