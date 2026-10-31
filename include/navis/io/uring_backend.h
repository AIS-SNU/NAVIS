#pragma once

#include "navis/io/io_backend.h"

#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <unordered_map>
#include <vector>

// ===========================================================================
// io_uring random I/O backend for runtime search/insert.
//
// C API implemented in src/io/uring_backend.c (compiled as C because liburing
// headers conflict with C++). Each worker thread owns a private ring (queue
// depth 256) per backend instance, with the file registered as a fixed file.
// ===========================================================================

extern "C" {
    struct uring_io_ctx;
    struct uring_io_ctx* uring_ctx_create(int fd, unsigned queue_depth);
    void uring_ctx_destroy(struct uring_io_ctx* ctx);

    int uring_read_batch(struct uring_io_ctx* ctx,
                         void** bufs, uint64_t* offsets, uint64_t* lens, int count);
    int uring_write_batch(struct uring_io_ctx* ctx,
                          void** bufs, uint64_t* offsets, uint64_t* lens, int count);

    void* uring_alloc_aligned(size_t size, size_t align);
    void  uring_free_aligned(void* ptr);
}

namespace navis {

class UringBackend : public IOBackend {
public:
    static constexpr unsigned kDefaultQueueDepth = 256;

    UringBackend() : instance_id_(next_instance_id_.fetch_add(1)) {}
    ~UringBackend() override { close(); }

    UringBackend(const UringBackend&) = delete;
    UringBackend& operator=(const UringBackend&) = delete;

    // Per-instance, per-thread ctx: multiple UringBackend instances (e.g.
    // one for _disk.edge and one for _disk.vec) must not share a ring
    // context, or submissions collide. Each instance owns a slot in the
    // per-thread map, keyed by its instance_id_.
    struct uring_io_ctx* tl_ctx() const {
        auto it = tl_ctxs_().find(instance_id_);
        return (it != tl_ctxs_().end()) ? it->second : nullptr;
    }

    void open(const std::string& file_path, bool writable) override {
        int flags = O_DIRECT | O_LARGEFILE;
        flags |= writable ? O_RDWR : O_RDONLY;
        fd_ = ::open(file_path.c_str(), flags);
        if (fd_ < 0) {
            throw std::runtime_error("UringBackend: failed to open " + file_path +
                                     ": " + std::strerror(errno));
        }
        writable_ = writable;
    }

    void close() override {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void read_batch(std::vector<IORequest>& reqs) override {
        if (reqs.empty()) return;
        int n = static_cast<int>(reqs.size());

        // Unpack IORequests into parallel arrays for C API
        void**    bufs    = static_cast<void**>(alloca(n * sizeof(void*)));
        uint64_t* offsets = static_cast<uint64_t*>(alloca(n * sizeof(uint64_t)));
        uint64_t* lens    = static_cast<uint64_t*>(alloca(n * sizeof(uint64_t)));
        for (int i = 0; i < n; i++) {
            bufs[i]    = reqs[i].buf;
            offsets[i] = reqs[i].offset;
            lens[i]    = reqs[i].len;
        }

        int rc = uring_read_batch(tl_ctx(), bufs, offsets, lens, n);
        if (rc < 0) {
            throw std::runtime_error("UringBackend::read_batch failed: " +
                                     std::string(std::strerror(-rc)));
        }
    }

    void write_batch(std::vector<IORequest>& reqs) override {
        if (reqs.empty()) return;
        int n = static_cast<int>(reqs.size());

        void**    bufs    = static_cast<void**>(alloca(n * sizeof(void*)));
        uint64_t* offsets = static_cast<uint64_t*>(alloca(n * sizeof(uint64_t)));
        uint64_t* lens    = static_cast<uint64_t*>(alloca(n * sizeof(uint64_t)));
        for (int i = 0; i < n; i++) {
            bufs[i]    = reqs[i].buf;
            offsets[i] = reqs[i].offset;
            lens[i]    = reqs[i].len;
        }

        int rc = uring_write_batch(tl_ctx(), bufs, offsets, lens, n);
        if (rc < 0) {
            throw std::runtime_error("UringBackend::write_batch failed: " +
                                     std::string(std::strerror(-rc)));
        }
    }

    void register_thread() override {
        auto& map = tl_ctxs_();
        auto it = map.find(instance_id_);
        if (it != map.end() && it->second) return; // already registered
        struct uring_io_ctx* ctx = uring_ctx_create(fd_, kDefaultQueueDepth);
        if (!ctx) {
            throw std::runtime_error("UringBackend: failed to create io_uring context");
        }
        map[instance_id_] = ctx;
    }

    void deregister_thread() override {
        auto& map = tl_ctxs_();
        auto it = map.find(instance_id_);
        if (it != map.end() && it->second) {
            uring_ctx_destroy(it->second);
            map.erase(it);
        }
    }

private:
    int  fd_          = -1;
    bool writable_    = false;
    int  instance_id_ = 0;

    static std::atomic<int> next_instance_id_;
    static std::unordered_map<int, struct uring_io_ctx*>& tl_ctxs_() {
        // One map per thread; one slot per UringBackend instance.
        static thread_local std::unordered_map<int, struct uring_io_ctx*> m;
        return m;
    }

};

inline std::atomic<int> UringBackend::next_instance_id_{0};

// ---------------------------------------------------------------------------
// Aligned memory allocation (O_DIRECT needs 4 KiB alignment)
// ---------------------------------------------------------------------------
inline void* navis_dma_alloc(size_t size, size_t align = 4096) {
    void* ptr = uring_alloc_aligned(size, align);
    if (!ptr) {
        throw std::runtime_error("navis_dma_alloc failed for size=" +
                                 std::to_string(size));
    }
    return ptr;
}

inline void navis_dma_free(void* ptr) {
    uring_free_aligned(ptr);
}

// ---------------------------------------------------------------------------
// 4 KB DMA scratch pool — thread-local free-list for O_DIRECT-aligned 4 KB
// buffers. The insert path allocates ~R sector scratches per insert and frees
// most of them before returning; reusing them across inserts removes ~R
// `aligned_alloc + memset(4KB)` calls per insert (~200 µs on FineWeb 50M).
//
// Not used for dst scratches handed to the bg writer — those get freed on
// the writer thread and would not replenish the insert thread's pool.
// Callers that want pool semantics explicitly use `navis_dma_alloc_4k` /
// `navis_dma_free_4k`. Generic `navis_dma_alloc / navis_dma_free` stay
// pool-unaware so they remain size-agnostic.
// ---------------------------------------------------------------------------
inline std::vector<void*>& _navis_dma_pool_4k() {
    thread_local std::vector<void*> pool;
    return pool;
}

inline void* navis_dma_alloc_4k() {
    auto& pool = _navis_dma_pool_4k();
    if (!pool.empty()) {
        void* p = pool.back();
        pool.pop_back();
        std::memset(p, 0, 4096);
        return p;
    }
    return navis_dma_alloc(4096, 4096);
}

inline void navis_dma_free_4k(void* ptr) {
    if (!ptr) return;
    auto& pool = _navis_dma_pool_4k();
    // Cap pool size to bound memory footprint (128 * 4 KB = 512 KB/thread).
    constexpr size_t kMaxPool = 128;
    if (pool.size() < kMaxPool) {
        pool.push_back(ptr);
        return;
    }
    navis_dma_free(ptr);
}

} // namespace navis
