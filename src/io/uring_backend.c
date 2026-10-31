#define _GNU_SOURCE

/*
 * uring_backend.c — io_uring random I/O backend for runtime search/insert.
 *
 * Compiled as C because liburing's header includes C++ <atomic>, making it
 * unusable from C++ code. Exposes opaque C APIs (uring_ctx_*) that the C++
 * UringBackend class in uring_backend.h calls.
 *
 *   - io_uring_submit_and_wait(): one syscall submits a batch and waits
 *   - io_uring_peek_batch_cqe(): bulk CQE harvesting
 *   - io_uring_register_files() + IOSQE_FIXED_FILE: no per-I/O fd lookup
 */

#include "liburing.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* ======================================================================
 * uring_io_ctx — per-thread io_uring context bound to one file
 * ====================================================================== */

struct uring_io_ctx {
    struct io_uring ring;
    int             fd;
    int             fixed_file_ok; /* 1 if io_uring_register_files succeeded */
    unsigned        queue_depth;
};

struct uring_io_ctx *uring_ctx_create(int fd, unsigned queue_depth) {
    struct uring_io_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->fd = fd;
    ctx->queue_depth = queue_depth;

    int ret = io_uring_queue_init(queue_depth, &ctx->ring, 0);
    if (ret < 0) {
        free(ctx);
        return NULL;
    }

    /* Register the fd as fixed file 0. Non-fatal on failure; we fall back
     * to regular fd lookup per SQE. */
    ret = io_uring_register_files(&ctx->ring, &ctx->fd, 1);
    if (ret == 0) ctx->fixed_file_ok = 1;

    return ctx;
}

void uring_ctx_destroy(struct uring_io_ctx *ctx) {
    if (!ctx) return;
    if (ctx->fixed_file_ok) {
        io_uring_unregister_files(&ctx->ring);
    }
    io_uring_queue_exit(&ctx->ring);
    free(ctx);
}

/* ======================================================================
 * Internal: prep one SQE (read or write), using the fixed fd if available
 * ====================================================================== */

static struct io_uring_sqe *prep_sqe(struct uring_io_ctx *ctx,
                                     void *buf, uint64_t offset, uint64_t len,
                                     uint64_t user_data, int is_write) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ctx->ring);
    if (!sqe) return NULL;

    int sqe_fd = ctx->fixed_file_ok ? 0 : ctx->fd;
    if (is_write) {
        io_uring_prep_write(sqe, sqe_fd, buf, (unsigned)len, offset);
    } else {
        io_uring_prep_read(sqe, sqe_fd, buf, (unsigned)len, offset);
    }
    if (ctx->fixed_file_ok) {
        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
    }
    sqe->user_data = user_data;
    return sqe;
}

/* ======================================================================
 * Synchronous batch I/O
 *
 * io_uring_submit_and_wait(batch) — single syscall for submit + wait.
 * io_uring_peek_batch_cqe() — bulk CQE harvesting.
 * Handles count > queue_depth by submitting in chunks.
 * ====================================================================== */

static int batch_io(struct uring_io_ctx *ctx,
                    void **bufs, uint64_t *offsets, uint64_t *lens,
                    int count, int is_write) {
    if (count <= 0) return 0;

    int base = 0;
    int remaining = count;

    while (remaining > 0) {
        int batch = remaining;
        if (batch > (int)ctx->queue_depth)
            batch = (int)ctx->queue_depth;

        /* Fill SQEs */
        for (int i = 0; i < batch; i++) {
            struct io_uring_sqe *sqe = prep_sqe(ctx,
                bufs[base + i], offsets[base + i], lens[base + i],
                (uint64_t)(base + i), is_write);
            if (!sqe) {
                /* SQ full — flush and retry */
                int ret = io_uring_submit(&ctx->ring);
                if (ret < 0) return ret;
                struct io_uring_cqe *cqe;
                while (ret > 0) {
                    int rc = io_uring_wait_cqe(&ctx->ring, &cqe);
                    if (rc < 0) return rc;
                    io_uring_cqe_seen(&ctx->ring, cqe);
                    ret--;
                }
                sqe = prep_sqe(ctx,
                    bufs[base + i], offsets[base + i], lens[base + i],
                    (uint64_t)(base + i), is_write);
                if (!sqe) return -ENOSPC;
            }
        }

        /* Single syscall: submit all + wait for all completions */
        int ret = io_uring_submit_and_wait(&ctx->ring, batch);
        if (ret < 0) return ret;

        /* Harvest all CQEs */
        int harvested = 0;
        while (harvested < batch) {
            struct io_uring_cqe *cqes[256];
            unsigned to_peek = (unsigned)(batch - harvested);
            if (to_peek > 256) to_peek = 256;

            unsigned n = io_uring_peek_batch_cqe(&ctx->ring, cqes, to_peek);
            if (n == 0) {
                struct io_uring_cqe *cqe;
                ret = io_uring_wait_cqe(&ctx->ring, &cqe);
                if (ret < 0) return ret;
                if (cqe->res < 0) {
                    int err = cqe->res;
                    io_uring_cqe_seen(&ctx->ring, cqe);
                    return err;
                }
                io_uring_cqe_seen(&ctx->ring, cqe);
                harvested++;
                continue;
            }
            for (unsigned j = 0; j < n; j++) {
                if (cqes[j]->res < 0) {
                    int err = cqes[j]->res;
                    for (unsigned k = j; k < n; k++)
                        io_uring_cqe_seen(&ctx->ring, cqes[k]);
                    return err;
                }
                io_uring_cqe_seen(&ctx->ring, cqes[j]);
            }
            harvested += (int)n;
        }

        base += batch;
        remaining -= batch;
    }

    return 0;
}

int uring_read_batch(struct uring_io_ctx *ctx,
                     void **bufs, uint64_t *offsets, uint64_t *lens, int count) {
    return batch_io(ctx, bufs, offsets, lens, count, 0);
}

int uring_write_batch(struct uring_io_ctx *ctx,
                      void **bufs, uint64_t *offsets, uint64_t *lens, int count) {
    return batch_io(ctx, bufs, offsets, lens, count, 1);
}

/* ======================================================================
 * Aligned memory allocation
 *
 * io_uring + O_DIRECT only needs 4KB alignment, no hugepages.
 * ====================================================================== */

void *uring_alloc_aligned(size_t size, size_t align) {
    /* aligned_alloc requires size to be a multiple of align */
    size_t alloc_size = ((size + align - 1) / align) * align;
    void *ptr = aligned_alloc(align, alloc_size);
    if (ptr) memset(ptr, 0, alloc_size);
    return ptr;
}

void uring_free_aligned(void *ptr) {
    free(ptr);
}
