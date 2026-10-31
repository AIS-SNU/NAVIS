#pragma once

#include "navis/types.h"
#include <libcuckoo/cuckoohash_map.hh>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

namespace navis {

// ---------------------------------------------------------------------------
// PageCache — the RMW page cache for edge pages written by insertions.
//
// An insertion composes each modified edge page in private scratch and
// publishes it with put_dirty(), which pins the page (ref) and marks it
// dirty. The background writer calls mark_clean_and_deref() after the page
// reaches storage; the entry is dropped once it is unpinned and clean.
// Readers (search and position seeking) probe this cache first, so they
// always see the newest bytes of a page that is still in flight.
//
// The 4 KiB page lives in a std::shared_ptr<const PageBuf>, not inline in
// the cuckoo bucket: readers atomic_load the pointer under the bucket lock
// and memcpy off-lock, writers build a new buffer and atomic_store it. The
// bucket lock therefore only guards ref/dirty metadata and the pointer swap.
// ---------------------------------------------------------------------------
class PageCache {
public:
    using PageBuf   = std::array<uint8_t, SECTOR_LEN>;
    using PageBufSp = std::shared_ptr<const PageBuf>;

    struct Item {
        PageBufSp buf_sp;        // atomic_load/store by readers/writers
        uint64_t  ref_cnt = 0;
        bool      dirty   = false;
    };

    PageCache() = default;
    ~PageCache() { cache_.clear(); }

    // Copy the cached page into `value` (off-lock). Returns true on hit.
    bool get(uint64_t sector, uint8_t* value) {
        PageBufSp sp;
        bool found = false;
        cache_.update_fn(sector, [&](Item& item) {
            sp = std::atomic_load_explicit(&item.buf_sp, std::memory_order_acquire);
            found = true;
        });
        if (found && value && sp) {
            std::memcpy(value, sp->data(), SECTOR_LEN);
        }
        return found;
    }

    // Publish a modified page: insert or overwrite its bytes, take a ref and
    // mark it dirty. Readers holding an older snapshot keep consistent old
    // bytes; new readers see the new bytes.
    void put_dirty(uint64_t sector, const uint8_t* value) {
        auto mutable_sp = std::make_shared<PageBuf>();
        std::memcpy(mutable_sp->data(), value, SECTOR_LEN);
        PageBufSp new_sp = std::move(mutable_sp);

        cache_.upsert(sector,
            [&](Item& item, libcuckoo::UpsertContext ctx) {
                if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
                    item.ref_cnt = 0;
                    item.dirty   = false;
                }
                std::atomic_store_explicit(&item.buf_sp, new_sp,
                                           std::memory_order_release);
                item.ref_cnt++;
                item.dirty = true;
            });
    }

    // Called by the background writer once the page is on storage.
    void mark_clean_and_deref(uint64_t sector) {
        cache_.erase_fn(sector, [](Item& item) -> bool {
            item.dirty = false;
            if (item.ref_cnt > 0) item.ref_cnt--;
            return item.ref_cnt == 0 && !item.dirty;
        });
    }

    size_t size() const { return cache_.size(); }

    // ref_cnt of an entry (-1 if absent). For tests.
    int ref_cnt_for_test(uint64_t sector) {
        int rc = -1;
        cache_.update_fn(sector, [&](Item& item) {
            rc = static_cast<int>(item.ref_cnt);
        });
        return rc;
    }

private:
    libcuckoo::cuckoohash_map<uint64_t, Item> cache_;
};

} // namespace navis
