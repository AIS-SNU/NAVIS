#include "navis/index/indirection_table.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

int main() {
    constexpr uint32_t EPP        = 10;
    constexpr uint32_t NPTS       = 100;
    constexpr int      THREADS    = 8;
    constexpr int      ITERS      = 500;
    constexpr int      ALLOC_N    = 5;

    std::vector<uint32_t> perm(NPTS);
    for (uint32_t i = 0; i < NPTS; ++i) perm[i] = NPTS - 1 - i;

    IndirectionTable t;
    t.set_geometry(EPP);
    t.init_from_perm(perm.data(), nullptr, NPTS);

    std::mutex collect_mu;
    std::unordered_set<uint32_t> all_seen;
    std::atomic<int> errors{0};

    auto worker = [&]() {
        for (int it = 0; it < ITERS; ++it) {
            std::set<uint64_t> need_to_read;
            std::set<uint64_t> hint;
            auto locs = t.alloc_edge_locs(ALLOC_N, hint, need_to_read);
            if (static_cast<int>(locs.size()) != ALLOC_N) {
                errors.fetch_add(1);
                continue;
            }
            for (auto l : locs) {
                t.bind_edge_loc(l + 100000, l);
            }
            for (auto l : locs) {
                if (t.id2edge_loc(l + 100000) != l) errors.fetch_add(1);
            }
            for (auto l : locs) {
                t.invalidate_edge_loc(l);
            }
            std::lock_guard g(collect_mu);
            for (auto l : locs) all_seen.insert(l);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < THREADS; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    CHECK(errors.load() == 0);
    CHECK(!all_seen.empty());
    std::printf("test_indirection_concurrent OK (slots touched: %zu)\n", all_seen.size());
    return 0;
}
