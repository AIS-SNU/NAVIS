#define _GNU_SOURCE

/*
 * uring_io.c — io_uring-based sequential reader and writer for build pipeline.
 *
 * Compiled as C because liburing's header includes C++ <atomic>, making it
 * impossible to use from C++ code. Exposes opaque C APIs (uring_reader_*,
 * uring_writer_*) that the C++ wrapper classes in uring_io.h call.
 *
 * Reader: io_uring + O_DIRECT, 64MB aligned buffer, 4MB SQEs, queue depth 64.
 *         Achieves 5.9 GB/s on PCIe 5.0 NVMe (vs 4.5 GB/s with dd).
 * Writer: io_uring flush of 64MB buffer in 4MB chunks. Small writes are
 *         buffered; only the flush path uses io_uring for throughput.
 */

#include "liburing.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct uring_reader {
    int fd;
    int direct;
    struct io_uring ring;
    char* buf;
    size_t buf_size;
    size_t buf_pos;
    size_t buf_end;
    uint64_t file_off;
    uint64_t file_size;
};

#define ALIGN 4096
#define CHUNK (4 * 1024 * 1024)  /* 4MB per SQE */
#define QD 64

struct uring_reader* uring_reader_open(const char* path, size_t buf_size) {
    struct uring_reader* r = calloc(1, sizeof(*r));
    if (!r) return NULL;

    r->buf_size = buf_size & ~(ALIGN - 1);

    /* Try O_DIRECT */
    r->fd = open(path, O_RDONLY | O_DIRECT);
    if (r->fd >= 0) {
        r->direct = 1;
    } else {
        r->fd = open(path, O_RDONLY);
        r->direct = 0;
    }
    if (r->fd < 0) { free(r); return NULL; }

    struct stat st;
    fstat(r->fd, &st);
    r->file_size = st.st_size;

    if (io_uring_queue_init(QD, &r->ring, 0) < 0) {
        close(r->fd);
        free(r);
        return NULL;
    }

    r->buf = aligned_alloc(ALIGN, r->buf_size);
    if (!r->buf) {
        io_uring_queue_exit(&r->ring);
        close(r->fd);
        free(r);
        return NULL;
    }

    return r;
}

void uring_reader_close(struct uring_reader* r) {
    if (!r) return;
    io_uring_queue_exit(&r->ring);
    free(r->buf);
    close(r->fd);
    free(r);
}

static int drain_cqes(struct uring_reader* r, int count) {
    for (int i = 0; i < count; i++) {
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&r->ring, &cqe);
        if (ret < 0) return ret;
        if (cqe->res < 0) { io_uring_cqe_seen(&r->ring, cqe); return cqe->res; }
        io_uring_cqe_seen(&r->ring, cqe);
    }
    return 0;
}

static int refill(struct uring_reader* r) {
    uint64_t remaining = r->file_size - r->file_off;
    if (remaining == 0) { r->buf_pos = r->buf_end = 0; return 0; }

    uint64_t aligned_off = r->direct ? (r->file_off & ~((uint64_t)ALIGN - 1)) : r->file_off;
    size_t skip_bytes = r->file_off - aligned_off;
    size_t to_read = r->buf_size;
    if (to_read > remaining + skip_bytes) to_read = remaining + skip_bytes;

    if (r->direct) {
        to_read = (to_read + ALIGN - 1) & ~((size_t)ALIGN - 1);
        if (to_read > r->buf_size) to_read = r->buf_size;
        uint64_t max_aligned = ((r->file_size + ALIGN - 1) & ~((uint64_t)ALIGN - 1)) - aligned_off;
        if (to_read > max_aligned) to_read = max_aligned;
    }

    /* Submit io_uring SQEs */
    size_t n_chunks = (to_read + CHUNK - 1) / CHUNK;
    int submitted = 0;

    for (size_t i = 0; i < n_chunks; i++) {
        size_t chunk_off = i * CHUNK;
        size_t chunk_len = to_read - chunk_off;
        if (chunk_len > CHUNK) chunk_len = CHUNK;
        if (r->direct) chunk_len = (chunk_len + ALIGN - 1) & ~((size_t)ALIGN - 1);

        struct io_uring_sqe* sqe = io_uring_get_sqe(&r->ring);
        if (!sqe) {
            io_uring_submit(&r->ring);
            int rc = drain_cqes(r, submitted);
            if (rc < 0) return rc;
            submitted = 0;
            sqe = io_uring_get_sqe(&r->ring);
            if (!sqe) return -ENOMEM;
        }
        io_uring_prep_read(sqe, r->fd, r->buf + chunk_off, chunk_len, aligned_off + chunk_off);
        sqe->user_data = i;
        submitted++;
    }

    if (submitted > 0) {
        io_uring_submit(&r->ring);
        int rc = drain_cqes(r, submitted);
        if (rc < 0) return rc;
    }

    size_t valid = to_read;
    if (valid > r->file_size - aligned_off) valid = r->file_size - aligned_off;

    if (skip_bytes > 0 && valid > skip_bytes) {
        memmove(r->buf, r->buf + skip_bytes, valid - skip_bytes);
        valid -= skip_bytes;
    }

    r->buf_pos = 0;
    r->buf_end = valid < remaining ? valid : remaining;
    r->file_off += r->buf_end;
    return 0;
}

int uring_reader_read(struct uring_reader* r, void* dst, size_t n) {
    char* out = dst;
    while (n > 0) {
        size_t avail = r->buf_end - r->buf_pos;
        if (avail > 0) {
            size_t take = n < avail ? n : avail;
            memcpy(out, r->buf + r->buf_pos, take);
            r->buf_pos += take;
            out += take;
            n -= take;
        } else {
            int rc = refill(r);
            if (rc < 0) return rc;
            if (r->buf_end == 0) return -1; /* EOF */
        }
    }
    return 0;
}

void uring_reader_seek(struct uring_reader* r, uint64_t offset) {
    r->file_off = offset;
    r->buf_pos = r->buf_end = 0;
}

void uring_reader_skip(struct uring_reader* r, size_t n) {
    size_t avail = r->buf_end - r->buf_pos;
    if (n <= avail) {
        r->buf_pos += n;
    } else {
        n -= avail;
        r->file_off += n;
        r->buf_pos = r->buf_end = 0;
    }
}

uint64_t uring_reader_file_size(struct uring_reader* r) {
    return r->file_size;
}

/* ======================================================================
 * uring_writer — Buffered sequential writer with io_uring flush.
 *
 * Small writes are buffered in a 64MB user-space buffer. On flush,
 * the buffer is written to disk via io_uring in 4MB chunks for
 * maximum throughput.
 * ====================================================================== */

struct uring_writer {
    int fd;
    struct io_uring ring;
    int ring_ok;
    char* buf;
    size_t buf_size;
    size_t buf_pos;
    uint64_t file_off;
    uint64_t wb_off;   /* start of region being written back asynchronously */
    size_t   wb_len;   /* length of that region (0 = nothing pending) */
};

struct uring_writer* uring_writer_open(const char* path, size_t buf_size) {
    struct uring_writer* w = calloc(1, sizeof(*w));
    if (!w) return NULL;

    w->buf_size = buf_size;
    w->fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (w->fd < 0) { free(w); return NULL; }

    if (io_uring_queue_init(QD, &w->ring, 0) < 0) {
        close(w->fd);
        free(w);
        return NULL;
    }
    w->ring_ok = 1;

    w->buf = malloc(buf_size);
    if (!w->buf) {
        io_uring_queue_exit(&w->ring);
        close(w->fd);
        free(w);
        return NULL;
    }
    return w;
}

static int drain_cqes_w(struct uring_writer* w, int count) {
    for (int i = 0; i < count; i++) {
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&w->ring, &cqe);
        if (ret < 0) return ret;
        if (cqe->res < 0) { io_uring_cqe_seen(&w->ring, cqe); return cqe->res; }
        io_uring_cqe_seen(&w->ring, cqe);
    }
    return 0;
}

static int uring_writer_flush(struct uring_writer* w) {
    if (w->buf_pos == 0) return 0;

    /* Drain previous async writeback before reusing the page cache region.
       Then evict those clean pages so dirty pages stay bounded at ~128 MB
       (previous batch + current batch) and never trigger kernel writeback
       stalls that throttle the SSD to ~100 MB/s. */
    if (w->wb_len > 0) {
        sync_file_range(w->fd, w->wb_off, w->wb_len,
                        SYNC_FILE_RANGE_WAIT_BEFORE);
        posix_fadvise(w->fd, w->wb_off, w->wb_len, POSIX_FADV_DONTNEED);
        w->wb_len = 0;
    }

    size_t total = w->buf_pos;
    size_t n_chunks = (total + CHUNK - 1) / CHUNK;
    int submitted = 0;

    for (size_t i = 0; i < n_chunks; i++) {
        size_t off = i * CHUNK;
        size_t len = total - off;
        if (len > CHUNK) len = CHUNK;

        struct io_uring_sqe* sqe = io_uring_get_sqe(&w->ring);
        if (!sqe) {
            io_uring_submit(&w->ring);
            int rc = drain_cqes_w(w, submitted);
            if (rc < 0) return rc;
            submitted = 0;
            sqe = io_uring_get_sqe(&w->ring);
            if (!sqe) return -1;
        }
        io_uring_prep_write(sqe, w->fd, w->buf + off, len, w->file_off + off);
        sqe->user_data = i;
        submitted++;
    }

    if (submitted > 0) {
        io_uring_submit(&w->ring);
        int rc = drain_cqes_w(w, submitted);
        if (rc < 0) return rc;
    }

    /* Kick off async writeback for this batch — the kernel flushes to SSD
       in the background while we fill the next buffer. */
    sync_file_range(w->fd, w->file_off, total, SYNC_FILE_RANGE_WRITE);
    w->wb_off = w->file_off;
    w->wb_len = total;

    w->file_off += total;
    w->buf_pos = 0;
    return 0;
}

void uring_writer_close(struct uring_writer* w) {
    if (!w) return;
    uring_writer_flush(w);
    /* Wait for final writeback to reach disk */
    if (w->wb_len > 0) {
        sync_file_range(w->fd, w->wb_off, w->wb_len,
                        SYNC_FILE_RANGE_WAIT_BEFORE |
                        SYNC_FILE_RANGE_WRITE |
                        SYNC_FILE_RANGE_WAIT_AFTER);
    }
    if (w->ring_ok) io_uring_queue_exit(&w->ring);
    if (w->buf) free(w->buf);
    close(w->fd);
    free(w);
}

int uring_writer_write(struct uring_writer* w, const void* src, size_t n) {
    const char* in = src;
    while (n > 0) {
        size_t space = w->buf_size - w->buf_pos;
        if (space > 0) {
            size_t put = n < space ? n : space;
            memcpy(w->buf + w->buf_pos, in, put);
            w->buf_pos += put;
            in += put;
            n -= put;
        }
        if (w->buf_pos == w->buf_size) {
            int rc = uring_writer_flush(w);
            if (rc < 0) return rc;
        }
    }
    return 0;
}

int uring_writer_seek(struct uring_writer* w, uint64_t offset) {
    int rc = uring_writer_flush(w);
    if (rc < 0) return rc;
    w->file_off = offset;
    lseek(w->fd, offset, SEEK_SET);
    return 0;
}

uint64_t uring_writer_tell(struct uring_writer* w) {
    return w->file_off + w->buf_pos;
}
