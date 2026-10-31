#include "navis/index/indirection_table.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

static uint64_t page_no(uint32_t loc, uint32_t epp) {
    return 1 + loc / epp;
}

static std::unique_ptr<IndirectionTable> make_permuted(uint32_t npts, uint32_t epp) {
    std::vector<uint32_t> perm(npts);
    for (uint32_t i = 0; i < npts; ++i) perm[i] = npts - 1 - i;
    auto t = std::make_unique<IndirectionTable>();
    t->set_geometry(epp);
    t->init_from_perm(perm.data(), nullptr, npts);
    return t;
}

static void test_tier3() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);
    std::set<uint64_t> need, hint;
    auto locs = t->alloc_edge_locs(15, hint, need);
    CHECK(locs.size() == 15);
    for (auto l : locs) CHECK(l >= NPTS);
    CHECK(need.empty());
}

static void test_tier1() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);
    for (uint32_t i = 0; i < EPP; ++i) t->invalidate_edge_loc(i);
    std::set<uint64_t> need, hint;
    auto locs = t->alloc_edge_locs(EPP, hint, need);
    std::sort(locs.begin(), locs.end());
    CHECK(locs.size() == EPP);
    for (uint32_t i = 0; i < EPP; ++i) CHECK(locs[i] == i);
    CHECK(need.empty());
}

static void test_tier2() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);
    for (uint32_t i = 15; i < 20; ++i) t->invalidate_edge_loc(i);
    std::set<uint64_t> hint = {page_no(10, EPP)};
    std::set<uint64_t> need;
    auto locs = t->alloc_edge_locs(5, hint, need);
    CHECK(locs.size() == 5);
    std::sort(locs.begin(), locs.end());
    for (uint32_t i = 0; i < 5; ++i) CHECK(locs[i] == 15 + i);
    CHECK(need.count(page_no(10, EPP)) == 1);
}

static void test_tier3_overflow_is_rejected_before_mutation() {
    constexpr uint32_t EPP = 1, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);

    const uint64_t tail = static_cast<uint64_t>(IndirectionTable::kAllocatedID);
    const uint32_t existing_owner = t->edge_loc2id(0);
    t->set_cur_edge_loc(tail);

    std::set<uint64_t> hint;
    std::set<uint64_t> need = {1234};
    bool rejected = false;
    try {
        (void)t->alloc_edge_locs(1, hint, need);
    } catch (const std::overflow_error&) {
        rejected = true;
    }

    CHECK(rejected);
    CHECK(t->cur_edge_loc() == tail);
    CHECK(t->edge_loc2id(0) == existing_owner);
    CHECK(t->edge_loc2id(static_cast<uint32_t>(tail)) ==
          IndirectionTable::kInvalidID);
    CHECK(need == std::set<uint64_t>{1234});
}

static void test_recycled_page_succeeds_with_exhausted_tail() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);
    for (uint32_t loc = 0; loc < EPP; ++loc) {
        t->invalidate_edge_loc(loc);
    }

    const uint64_t tail = static_cast<uint64_t>(IndirectionTable::kAllocatedID);
    t->set_cur_edge_loc(tail);

    std::set<uint64_t> hint, need;
    auto locs = t->alloc_edge_locs(EPP, hint, need);
    std::sort(locs.begin(), locs.end());

    CHECK(locs.size() == EPP);
    for (uint32_t loc = 0; loc < EPP; ++loc) CHECK(locs[loc] == loc);
    CHECK(t->cur_edge_loc() == tail);
    CHECK(need.empty());
}

static void test_tier3_overflow_rolls_back_recycled_slots() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);

    for (uint32_t loc = 0; loc < EPP; ++loc) {
        t->invalidate_edge_loc(loc);
    }
    for (uint32_t loc = 15; loc < 20; ++loc) {
        t->invalidate_edge_loc(loc);
    }

    const uint64_t tail = static_cast<uint64_t>(IndirectionTable::kAllocatedID);
    t->set_cur_edge_loc(tail);

    const std::set<uint64_t> hint = {page_no(15, EPP)};
    std::set<uint64_t> need = {1234};
    bool rejected = false;
    try {
        (void)t->alloc_edge_locs(16, hint, need);
    } catch (const std::overflow_error&) {
        rejected = true;
    }

    CHECK(rejected);
    CHECK(t->cur_edge_loc() == tail);
    CHECK(need == std::set<uint64_t>{1234});
    for (uint32_t loc = 0; loc < EPP; ++loc) {
        CHECK(t->edge_loc2id(loc) == IndirectionTable::kInvalidID);
    }
    for (uint32_t loc = 15; loc < 20; ++loc) {
        CHECK(t->edge_loc2id(loc) == IndirectionTable::kInvalidID);
    }

    std::set<uint64_t> no_hint, empty_need;
    auto empty_page_locs = t->alloc_edge_locs(EPP, no_hint, empty_need);
    std::sort(empty_page_locs.begin(), empty_page_locs.end());
    CHECK(empty_page_locs.size() == EPP);
    for (uint32_t loc = 0; loc < EPP; ++loc) {
        CHECK(empty_page_locs[loc] == loc);
    }
    CHECK(empty_need.empty());

    std::set<uint64_t> hint_need;
    auto hint_locs = t->alloc_edge_locs(5, hint, hint_need);
    std::sort(hint_locs.begin(), hint_locs.end());
    CHECK(hint_locs.size() == 5);
    for (uint32_t i = 0; i < 5; ++i) CHECK(hint_locs[i] == 15 + i);
    CHECK(hint_need == hint);
    CHECK(t->cur_edge_loc() == tail);
}

static void test_negative_count_is_rejected() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);
    const uint64_t tail = t->cur_edge_loc();

    std::set<uint64_t> hint, need;
    bool rejected = false;
    try {
        (void)t->alloc_edge_locs(-1, hint, need);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    CHECK(rejected);
    CHECK(t->cur_edge_loc() == tail);
    CHECK(need.empty());
}

static void test_reverse_capacity_rejects_sentinel_slot() {
    constexpr uint32_t EPP = 10, NPTS = 100;
    auto t = make_permuted(NPTS, EPP);
    const uint32_t existing_owner = t->edge_loc2id(0);

    bool rejected = false;
    try {
        t->ensure_loc_capacity(
            static_cast<uint64_t>(IndirectionTable::kAllocatedID) + 1);
    } catch (const std::overflow_error&) {
        rejected = true;
    }

    CHECK(rejected);
    CHECK(t->edge_loc2id(0) == existing_owner);
}

int main() {
    test_tier3();
    test_tier1();
    test_tier2();
    test_tier3_overflow_is_rejected_before_mutation();
    test_recycled_page_succeeds_with_exhausted_tail();
    test_tier3_overflow_rolls_back_recycled_slots();
    test_negative_count_is_rejected();
    test_reverse_capacity_rejects_sentinel_slot();
    std::printf("test_indirection_alloc OK\n");
    return 0;
}
