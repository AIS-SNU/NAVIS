#include "navis/cache/page_cache.h"
#include "navis/types.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

static void test_put_dirty_lifecycle() {
    PageCache c;

    uint8_t buf[SECTOR_LEN];
    std::memset(buf, 0xA5, SECTOR_LEN);

    // put_dirty pins the page (ref_cnt=1) and marks it dirty.
    c.put_dirty(1001, buf);
    CHECK(c.ref_cnt_for_test(1001) == 1);

    // Read back via get: contents match, ref_cnt unchanged.
    uint8_t out[SECTOR_LEN];
    std::memset(out, 0, SECTOR_LEN);
    CHECK(c.get(1001, out));
    CHECK(std::memcmp(out, buf, SECTOR_LEN) == 0);
    CHECK(c.ref_cnt_for_test(1001) == 1);

    // A second in-flight write of the same page takes another ref.
    c.put_dirty(1001, buf);
    CHECK(c.ref_cnt_for_test(1001) == 2);

    // First write lands: page stays (still pinned by the second write).
    c.mark_clean_and_deref(1001);
    CHECK(c.size() == 1);

    // Second write lands: unpinned and clean, so the entry is dropped.
    c.mark_clean_and_deref(1001);
    CHECK(c.size() == 0);
    CHECK(!c.get(1001, out));
}

static void test_put_overwrites_contents() {
    PageCache c;

    uint8_t buf_a[SECTOR_LEN];
    uint8_t buf_b[SECTOR_LEN];
    std::memset(buf_a, 0x11, SECTOR_LEN);
    std::memset(buf_b, 0x22, SECTOR_LEN);

    c.put_dirty(42, buf_a);
    c.put_dirty(42, buf_b);

    uint8_t out[SECTOR_LEN];
    CHECK(c.get(42, out));
    CHECK(out[0] == 0x22);
    CHECK(out[SECTOR_LEN - 1] == 0x22);

    c.mark_clean_and_deref(42);
    c.mark_clean_and_deref(42);
    CHECK(c.size() == 0);
}

static void test_concurrent_get_during_put() {
    // Stress test: many concurrent get()s alongside put() swaps should
    // never torn-read thanks to the atomic shared_ptr swap + bucket lock.
    PageCache c;
    uint8_t init[SECTOR_LEN];
    std::memset(init, 0x00, SECTOR_LEN);
    c.put_dirty(999, init);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> read_ops{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 8; t++) {
        readers.emplace_back([&]() {
            uint8_t out[SECTOR_LEN];
            while (!stop.load()) {
                if (c.get(999, out)) {
                    // All bytes must be uniform: either all-X or all-Y,
                    // never a mix (would indicate torn read).
                    uint8_t v = out[0];
                    for (size_t i = 1; i < SECTOR_LEN; i++) CHECK(out[i] == v);
                    read_ops.fetch_add(1);
                }
            }
        });
    }

    // Writer: alternate two fill patterns.
    for (int i = 0; i < 1000; i++) {
        uint8_t buf[SECTOR_LEN];
        std::memset(buf, (i & 1) ? 0xAA : 0x55, SECTOR_LEN);
        c.put_dirty(999, buf);
    }
    stop.store(true);
    for (auto& t : readers) t.join();

    CHECK(read_ops.load() > 0);
}

int main() {
    test_put_dirty_lifecycle();
    test_put_overwrites_contents();
    test_concurrent_get_during_put();
    std::printf("test_page_cache OK\n");
    return 0;
}
