#include "navis/cache/navis_cache.h"
#include "navis/types.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

static void test_populate_admit_on_miss_noop_on_hit() {
    NavisCache c;
    CacheConfig cfg;
    cfg.size_gb = 1;   // small, just for test
    c.init(cfg);

    uint8_t v1[EDGE_SECTOR_LEN];
    std::memset(v1, 0xAA, EDGE_SECTOR_LEN);
    c.populate(42, v1);

    // populate again with different bytes — should be a no-op.
    uint8_t v2[EDGE_SECTOR_LEN];
    std::memset(v2, 0xBB, EDGE_SECTOR_LEN);
    c.populate(42, v2);

    uint8_t read[EDGE_SECTOR_LEN];
    CHECK(c.get(42, read));
    CHECK(read[0] == 0xAA);   // still the original bytes, not 0xBB
}

static void test_update_if_present_updates_on_hit_noop_on_miss() {
    NavisCache c;
    CacheConfig cfg;
    cfg.size_gb = 1;
    c.init(cfg);

    uint8_t v1[EDGE_SECTOR_LEN];
    std::memset(v1, 0xAA, EDGE_SECTOR_LEN);
    c.populate(42, v1);

    uint8_t v2[EDGE_SECTOR_LEN];
    std::memset(v2, 0xCC, EDGE_SECTOR_LEN);
    c.update_if_present(42, v2);

    uint8_t read[EDGE_SECTOR_LEN];
    CHECK(c.get(42, read));
    CHECK(read[0] == 0xCC);   // now shows updated bytes

    // update_if_present on a miss is a no-op (no admission).
    c.update_if_present(999, v2);
    CHECK(!c.get(999, read));
}

static void test_evict_removes_entry() {
    NavisCache c;
    CacheConfig cfg;
    cfg.size_gb = 1;
    c.init(cfg);

    uint8_t v[EDGE_SECTOR_LEN];
    std::memset(v, 0xAA, EDGE_SECTOR_LEN);
    c.populate(42, v);

    uint8_t read[EDGE_SECTOR_LEN];
    CHECK(c.get(42, read));

    c.evict(42);
    CHECK(!c.get(42, read));
    // evict on a miss is a no-op
    c.evict(999);
}

static void test_disabled_cache_all_noop() {
    NavisCache c;
    CacheConfig cfg;
    cfg.size_gb = 0;   // disabled
    c.init(cfg);

    uint8_t v[EDGE_SECTOR_LEN];
    std::memset(v, 0xAA, EDGE_SECTOR_LEN);
    c.populate(42, v);
    c.update_if_present(42, v);
    c.evict(42);
    CHECK(!c.get(42, v));
    CHECK(!c.is_enabled());
}

static void test_hit_miss_counters() {
    NavisCache c;
    CacheConfig cfg; cfg.size_gb = 1;
    c.init(cfg);

    uint8_t v[EDGE_SECTOR_LEN]; std::memset(v, 0xAA, EDGE_SECTOR_LEN);
    c.populate(42, v);
    uint8_t read[EDGE_SECTOR_LEN];
    (void)c.get(42, read);   // 1 hit
    (void)c.get(43, read);   // 1 miss
    CHECK(c.hits() == 1);
    CHECK(c.misses() == 1);

    c.reset_stats();
    CHECK(c.hits() == 0 && c.misses() == 0);
}

int main() {
    test_populate_admit_on_miss_noop_on_hit();
    test_update_if_present_updates_on_hit_noop_on_miss();
    test_evict_removes_entry();
    test_disabled_cache_all_noop();
    test_hit_miss_counters();
    std::printf("test_navis_cache OK\n");
    return 0;
}
