#ifndef ORCHKV_IO_WORKER_H
#define ORCHKV_IO_WORKER_H

#include "orchfs_tier.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Async IO Worker Pool
 *
 *  Wraps orchfs_tier_write/read into an asynchronous task queue
 *  backed by a fixed-size thread pool.  Callers submit tasks and
 *  optionally flush (barrier) to wait for completion.
 * ======================================================================== */

typedef enum {
    IO_OP_WRITE = 0,
    IO_OP_READ  = 1,
} IOOpType;

typedef void (*io_callback_fn)(int status, void *user_data);

typedef struct io_task {
    IOOpType             op;
    orchfs_tier_t       *tier;
    orchfs_file_ctx_t   *fctx;
    uint32_t             layer;
    uint32_t             head;
    uint32_t             block_idx;
    void                *buf;
    size_t               size;
    io_callback_fn       callback;
    void                *user_data;
} io_task_t;

typedef struct io_worker_pool {
    int              num_workers;
    pthread_t       *threads;

    io_task_t       *queue;
    uint32_t         queue_cap;
    uint32_t         queue_head;
    uint32_t         queue_tail;
    uint32_t         queue_count;

    pthread_mutex_t  lock;
    pthread_cond_t   not_empty;
    pthread_cond_t   not_full;
    pthread_cond_t   all_done;

    uint64_t         submitted;
    uint64_t         completed;
    bool             shutdown;
} io_worker_pool_t;

int  io_worker_init(io_worker_pool_t *pool, int num_workers, uint32_t queue_cap);
void io_worker_destroy(io_worker_pool_t *pool);

int  io_worker_submit(io_worker_pool_t *pool, const io_task_t *task);
void io_worker_flush(io_worker_pool_t *pool);

static inline uint64_t io_worker_submitted(const io_worker_pool_t *p) { return p->submitted; }
static inline uint64_t io_worker_completed(const io_worker_pool_t *p) { return p->completed; }

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_IO_WORKER_H */
