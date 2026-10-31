#include "navis/index/deletion_table.h"
#include "navis/index/navis_index.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace navis;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while(0)

static void test_empty_table() {
    DeletionTable d;
    CHECK(d.capacity() == 0);
    CHECK(d.deleted_count() == 0);
    CHECK(!d.is_deleted(0));
    CHECK(!d.erase(0, 10));
}

static void test_idempotent_erase() {
    DeletionTable d;
    d.init(16);
    CHECK(d.capacity() == 16);
    CHECK(!d.is_deleted(3));
    CHECK(d.erase(3, 10));
    CHECK(d.is_deleted(3));
    CHECK(d.deleted_count() == 1);
    CHECK(!d.erase(3, 10));
    CHECK(d.deleted_count() == 1);
}

static void test_bounds() {
    DeletionTable d;
    d.init(8);
    CHECK(!d.erase(8, 100));
    CHECK(!d.erase(7, 7));
    CHECK(d.erase(6, 7));
    CHECK(d.is_deleted(6));
    CHECK(!d.is_deleted(7));
    CHECK(!d.is_deleted(8));
}

static void test_fraction() {
    DeletionTable d;
    d.init(100);
    CHECK(d.erase(1, 50));
    CHECK(d.erase(2, 50));
    double f = d.deleted_fraction(50);
    CHECK(std::fabs(f - 0.04) < 0.000001);
    CHECK(d.deleted_fraction(0) == 0.0);
}

static void test_init_resets_table() {
    DeletionTable d;
    d.init(10);
    CHECK(d.erase(2, 10));
    CHECK(d.erase(4, 10));
    CHECK(d.deleted_count() == 2);
    CHECK(d.is_deleted(2));
    CHECK(d.is_deleted(4));

    d.init(0);
    CHECK(d.capacity() == 0);
    CHECK(d.deleted_count() == 0);
    CHECK(!d.is_deleted(2));
    CHECK(!d.erase(2, 10));

    d.init(10);
    CHECK(d.capacity() == 10);
    CHECK(d.deleted_count() == 0);
    CHECK(!d.is_deleted(2));
    CHECK(!d.is_deleted(4));
    CHECK(d.erase(2, 10));
    CHECK(d.deleted_count() == 1);
}

static void test_default_navis_delete_api() {
    NavisIndex<float> idx;
    CHECK(!idx.erase(0));
    CHECK(!idx.is_deleted(0));
    CHECK(idx.deleted_count() == 0);
    CHECK(idx.deleted_fraction() == 0.0);
}

int main() {
    test_empty_table();
    test_idempotent_erase();
    test_bounds();
    test_fraction();
    test_init_resets_table();
    test_default_navis_delete_api();
    std::printf("test_deletion_table OK\n");
    return 0;
}
