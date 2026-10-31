#include "navis/index/indirection_table.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

static void test_identity() {
    IndirectionTable t;
    t.set_geometry(10);
    t.init_identity(1000);
    CHECK(t.is_identity());
    CHECK(t.vec_is_identity());
    for (uint32_t i = 0; i < 1000; ++i) {
        CHECK(t.id2edge_loc(i) == i);
        CHECK(t.id2vec_loc(i)  == i);
    }
    CHECK(t.cur_edge_loc() == 1000);
}

static void test_persistence_identity() {
    std::string prefix = "/tmp/test_indir_basic_id";
    {
        IndirectionTable t;
        t.set_geometry(10);
        t.init_identity(500);
        t.save(prefix, 500);
    }
    {
        IndirectionTable t2;
        t2.set_geometry(10);
        CHECK(t2.load(prefix, 500));
        for (uint32_t i = 0; i < 500; ++i) {
            CHECK(t2.id2edge_loc(i) == i);
            CHECK(t2.id2vec_loc(i)  == i);
        }
    }
    std::filesystem::remove(prefix + "_indirection.bin");
}

static void test_persistence_permuted() {
    std::string prefix = "/tmp/test_indir_basic_perm";
    constexpr uint32_t N = 100;
    constexpr uint32_t EPP = 10;

    uint32_t perm[N];
    for (uint32_t i = 0; i < N; ++i) perm[i] = N - 1 - i;

    {
        IndirectionTable t;
        t.set_geometry(EPP);
        t.init_from_perm(perm, nullptr, N);
        CHECK(!t.is_identity());
        CHECK(t.vec_is_identity());
        for (uint32_t i = 0; i < N; ++i) CHECK(t.id2edge_loc(i) == N - 1 - i);
        for (uint32_t i = 0; i < N; ++i) CHECK(t.edge_loc2id(N - 1 - i) == i);
        t.save(prefix, N);
    }
    {
        IndirectionTable t2;
        t2.set_geometry(EPP);
        CHECK(t2.load(prefix, N));
        CHECK(!t2.is_identity());
        for (uint32_t i = 0; i < N; ++i) CHECK(t2.id2edge_loc(i) == N - 1 - i);
        for (uint32_t i = 0; i < N; ++i) CHECK(t2.edge_loc2id(N - 1 - i) == i);
    }
    std::filesystem::remove(prefix + "_indirection.bin");
}

static void test_bind_invalidate() {
    constexpr uint32_t N = 100;
    constexpr uint32_t EPP = 10;

    uint32_t perm[N];
    for (uint32_t i = 0; i < N; ++i) perm[i] = i;

    IndirectionTable t;
    t.set_geometry(EPP);
    t.init_from_perm(perm, nullptr, N);

    uint32_t old_loc = t.id2edge_loc(5);
    CHECK(old_loc == 5);
    CHECK(t.edge_loc2id(5) == 5);

    uint32_t new_loc = 200;
    t.ensure_loc_capacity(201);
    t.bind_edge_loc(5, new_loc);
    CHECK(t.id2edge_loc(5) == new_loc);
    CHECK(t.edge_loc2id(new_loc) == 5);

    t.invalidate_edge_loc(old_loc);
    CHECK(t.edge_loc2id(old_loc) == IndirectionTable::kInvalidID);
}

int main() {
    test_identity();
    test_persistence_identity();
    test_persistence_permuted();
    test_bind_invalidate();
    std::printf("test_indirection_basic OK\n");
    return 0;
}
