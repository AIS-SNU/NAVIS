// Probe-order test for the read path: RMW PageCache first (freshest bytes of
// in-flight writes), then NAVIS-Cache, then storage. Mirrors the lookup in
// NavisIndex::do_search against bare cache instances (no disk I/O; a miss on
// both caches is reported as MISS). Also checks that fully invalidating an
// edge page in the IndirectionTable evicts it from NAVIS-Cache.

#include "navis/cache/page_cache.h"
#include "navis/cache/navis_cache.h"
#include "navis/index/indirection_table.h"
#include "navis/types.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

static int probe(PageCache& rmw, NavisCache& nav, uint64_t sec, uint8_t* dst) {
    if (rmw.get(sec, dst)) return 0;  // HIT_RMW
    if (nav.get(sec, dst)) return 1;  // HIT_NAVIS
    return 2;  // MISS (real impl would read disk + populate)
}

static void test_rmw_beats_navis() {
    PageCache rmw;
    NavisCache nav;
    CacheConfig cfg; cfg.size_gb = 1; nav.init(cfg);

    uint8_t stale[SECTOR_LEN]; std::memset(stale, 0xAA, SECTOR_LEN);
    uint8_t fresh[SECTOR_LEN]; std::memset(fresh, 0xBB, SECTOR_LEN);

    nav.populate(42, stale);

    // RMW populated as if an insert is mid-RMW (ref + dirty).
    rmw.put_dirty(42, fresh);

    uint8_t out[SECTOR_LEN];
    CHECK(probe(rmw, nav, 42, out) == 0);    // HIT_RMW
    CHECK(out[0] == 0xBB);                   // fresh wins

    rmw.mark_clean_and_deref(42);
}

static void test_navis_hit_when_rmw_empty() {
    PageCache rmw;
    NavisCache nav;
    CacheConfig cfg; cfg.size_gb = 1; nav.init(cfg);

    uint8_t bytes[SECTOR_LEN]; std::memset(bytes, 0xCC, SECTOR_LEN);
    nav.populate(77, bytes);

    uint8_t out[SECTOR_LEN];
    CHECK(probe(rmw, nav, 77, out) == 1);    // HIT_NAVIS
    CHECK(out[0] == 0xCC);
}

static void test_miss_both() {
    PageCache rmw;
    NavisCache nav;
    CacheConfig cfg; cfg.size_gb = 1; nav.init(cfg);

    uint8_t out[SECTOR_LEN];
    CHECK(probe(rmw, nav, 999, out) == 2);   // MISS
}

static void test_eager_evict_on_full_invalidation() {
    NavisCache nav;
    CacheConfig cfg; cfg.size_gb = 1;
    nav.init(cfg);

    IndirectionTable indir;
    indir.set_geometry(10);            // 10 entries per page
    indir.set_navis_cache(&nav);

    // init_identity() sets up forward id2edge_loc but leaves loc2id
    // backing array null — invalidate_edge_loc would early-return.
    // Bind each id explicitly to populate both forward and reverse maps.
    for (uint32_t id = 0; id < 10; ++id) {
        indir.bind_edge_loc(id, id);
    }

    // Pre-populate NAVIS-Cache for page 1 (== sector 1 in the edge file).
    uint8_t bytes[SECTOR_LEN]; std::memset(bytes, 0xEE, SECTOR_LEN);
    nav.populate(1, bytes);

    uint8_t out[SECTOR_LEN];
    CHECK(nav.get(1, out));                      // hit before invalidation

    // Invalidate every slot on page 1 (locs 0..9). The last call makes
    // all_free=true and pushes page_no=1 onto empty_pages_queue_, which
    // in turn calls navis_cache_.evict(1).
    for (uint32_t loc = 0; loc < 10; ++loc) {
        indir.invalidate_edge_loc(loc);
    }

    // Entry should be gone from NAVIS-Cache.
    CHECK(!nav.get(1, out));
}

int main() {
    test_rmw_beats_navis();
    test_navis_hit_when_rmw_empty();
    test_miss_both();
    test_eager_evict_on_full_invalidation();
    std::printf("test_cache_integration OK\n");
    return 0;
}
