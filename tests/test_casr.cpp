// Unit tests for windowed CASR: convergence tracking and window calibration.

#include "navis/casr.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

// Feed distances in order until the tracker is done; returns loads used.
static uint32_t run(CasrTracker& t, const std::vector<float>& dists) {
    uint32_t id = 0;
    while (!t.done() && id < dists.size()) {
        t.add(Neighbor(id, dists[id]));
        ++id;
    }
    return t.loaded();
}

static void test_stops_after_window_of_unchanged_loads() {
    // k=2, w=3: loads 1-3 change the top-2 (5, 4, then 3 replaces 5);
    // loads 4-6 do not, so the rerank stops at 3 + 3 = 6.
    CasrTracker t(2, 3, 10);
    CHECK(t.needed() == 5);   // before any load: k + w
    CHECK(run(t, {5, 4, 3, 9, 9, 9, 1, 1, 1, 1}) == 6);
    CHECK(t.done());
}

static void test_never_needs_more_than_candidates() {
    CasrTracker t(10, 5, 4);
    CHECK(t.needed() == 4);
    CHECK(run(t, {4, 3, 2, 1}) == 4);
    CasrTracker empty(10, 5, 0);
    CHECK(empty.done());
}

static void test_profiling_loads_all_and_finds_min_safe_window() {
    // k=1: top-1 changes at loads 1 (5), 4 (4), 6 (3). Unchanged runs before
    // a change: 2 and 1, so the smallest safe window is 3.
    const std::vector<float> d = {5, 6, 7, 4, 8, 3, 9};
    CasrTracker prof(1, 0, d.size());
    CHECK(run(prof, d) == d.size());
    CHECK(prof.min_safe_window() == 3);

    // w = 3 reaches the final change (load 6) ...
    CasrTracker ok(1, 3, d.size());
    run(ok, d);
    CHECK(ok.loaded() >= 6);
    // ... w = 2 stops after load 3 and misses it.
    CasrTracker early(1, 2, d.size());
    CHECK(run(early, d) == 3);
}

static void test_ties_do_not_change_topk() {
    // Equal distance with a larger id does not enter the top-1.
    CasrTracker t(1, 2, 5);
    CHECK(t.add(Neighbor(1, 1.0f)));
    CHECK(!t.add(Neighbor(2, 1.0f)));
    CHECK(t.add(Neighbor(0, 1.0f)));
}

static void test_calibrator_percentile() {
    CasrCalibrator c;
    for (uint32_t v = 1; v <= 99; ++v) c.record(v, 100, 90);
    CHECK(c.window() == 0);                // still warming up
    c.record(100, 100, 90);
    CHECK(c.window() == 91);               // 90th percentile of 1..100
    c.record(1000, 100, 90);               // ignored once calibrated
    CHECK(c.window() == 91);

    CasrCalibrator max_c;
    for (uint32_t v = 1; v <= 10; ++v) max_c.record(v, 10, 100);
    CHECK(max_c.window() == 10);
}

static void test_calibrator_concurrent() {
    CasrCalibrator c;
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&c]() {
            for (uint32_t v = 1; v <= 50; ++v) c.record(v, 100, 50);
        });
    }
    for (auto& t : ts) t.join();
    CHECK(c.window() >= 1 && c.window() <= 50);
}

int main() {
    test_stops_after_window_of_unchanged_loads();
    test_never_needs_more_than_candidates();
    test_profiling_loads_all_and_finds_min_safe_window();
    test_ties_do_not_change_topk();
    test_calibrator_percentile();
    test_calibrator_concurrent();
    std::printf("test_casr OK\n");
    return 0;
}
