#pragma once

#include <libcuckoo/cuckoohash_map.hh>
#include <pthread.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace navis {

// ---------------------------------------------------------------------------
// PageLockTable — sparse RW lock table keyed by page number
//
// Uses libcuckoo upsert with UpsertContext to atomically create-and-lock
// in a single operation, avoiding the race window of update_fn → insert.
// ---------------------------------------------------------------------------
class PageLockTable {
public:
    PageLockTable() = default;

    ~PageLockTable() {
        auto lt = locks_.lock_table();
        for (auto& [key, val] : lt) {
            pthread_rwlock_destroy(val.first);
            delete val.first;
        }
    }

    PageLockTable(const PageLockTable&) = delete;
    PageLockTable& operator=(const PageLockTable&) = delete;

    int tryrdlock(uint64_t key) {
        int ret = 0;
        locks_.upsert(key, [&](LockEntry& v, libcuckoo::UpsertContext ctx) {
            if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
                v = std::make_pair(new pthread_rwlock_t, 0);
                pthread_rwlock_init(v.first, nullptr);
            }
            ret = pthread_rwlock_tryrdlock(v.first);
            if (ret == 0) v.second++;
        });
        return ret;
    }

    int trywrlock(uint64_t key) {
        int ret = 0;
        locks_.upsert(key, [&](LockEntry& v, libcuckoo::UpsertContext ctx) {
            if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
                v = std::make_pair(new pthread_rwlock_t, 0);
                pthread_rwlock_init(v.first, nullptr);
            }
            ret = pthread_rwlock_trywrlock(v.first);
            if (ret == 0) v.second++;
        });
        return ret;
    }

    void rdlock(uint64_t key) {
        while (tryrdlock(key) != 0) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        }
    }

    void wrlock(uint64_t key) {
        while (trywrlock(key) != 0) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        }
    }

    void unlock(uint64_t key) {
        locks_.erase_fn(key, [](LockEntry& v) -> bool {
            if (v.second == 0) {
                std::fprintf(stderr, "PageLockTable: unlock non-locked key\n");
                __builtin_trap();
            }
            pthread_rwlock_unlock(v.first);
            if (v.second == 1) {
                pthread_rwlock_destroy(v.first);
                delete v.first;
            }
            v.second--;
            return v.second == 0;
        });
    }

private:
    using LockEntry = std::pair<pthread_rwlock_t*, int>;
    libcuckoo::cuckoohash_map<uint64_t, LockEntry> locks_;
};

// ---------------------------------------------------------------------------
// Deadlock-safe multi-page locking: sort sector numbers ascending
// ---------------------------------------------------------------------------
inline std::vector<uint64_t> lock_pages(PageLockTable& table,
                                         const std::vector<uint64_t>& sectors) {
    auto sorted = sectors;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (uint64_t s : sorted) {
        table.wrlock(s);
    }
    return sorted;
}

inline void unlock_pages(PageLockTable& table,
                          const std::vector<uint64_t>& locked_sectors) {
    for (uint64_t s : locked_sectors) {
        table.unlock(s);
    }
}

} // namespace navis
