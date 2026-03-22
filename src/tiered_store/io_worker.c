#include "io_worker.h"
#include <stdlib.h>
#include <string.h>

static void *worker_thread(void *arg)
{
    io_worker_pool_t *pool = (io_worker_pool_t *)arg;

    for (;;) {
        pthread_mutex_lock(&pool->lock);

        while (pool->queue_count == 0 && !pool->shutdown)
            pthread_cond_wait(&pool->not_empty, &pool->lock);

        if (pool->shutdown && pool->queue_count == 0) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        io_task_t task = pool->queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % pool->queue_cap;
        pool->queue_count--;

        pthread_cond_signal(&pool->not_full);
        pthread_mutex_unlock(&pool->lock);

        int status;
        if (task.op == IO_OP_WRITE) {
            status = orchfs_tier_write(task.fctx, task.layer, task.head,
                                       task.block_idx, task.buf, task.size);
            if (status == ORCHKV_OK)
                orchfs_tier_record_write(task.tier, task.size);
        } else {
            status = orchfs_tier_read(task.fctx, task.layer, task.head,
                                      task.block_idx, task.buf, task.size);
            if (status == ORCHKV_OK)
                orchfs_tier_record_read(task.tier, task.size);
        }

        if (task.callback)
            task.callback(status, task.user_data);

        pthread_mutex_lock(&pool->lock);
        pool->completed++;
        if (pool->completed == pool->submitted)
            pthread_cond_broadcast(&pool->all_done);
        pthread_mutex_unlock(&pool->lock);
    }
    return NULL;
}

int io_worker_init(io_worker_pool_t *pool, int num_workers, uint32_t queue_cap)
{
    memset(pool, 0, sizeof(*pool));

    if (num_workers <= 0) num_workers = 4;
    if (queue_cap == 0)   queue_cap = 1024;

    pool->queue = (io_task_t *)calloc(queue_cap, sizeof(io_task_t));
    if (!pool->queue) return ORCHKV_ERR_OOM;

    pool->threads = (pthread_t *)calloc((size_t)num_workers, sizeof(pthread_t));
    if (!pool->threads) { free(pool->queue); return ORCHKV_ERR_OOM; }

    pool->num_workers = num_workers;
    pool->queue_cap   = queue_cap;

    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->not_empty, NULL);
    pthread_cond_init(&pool->not_full, NULL);
    pthread_cond_init(&pool->all_done, NULL);

    for (int i = 0; i < num_workers; i++)
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);

    LOG_INFO("io_worker_pool: %d workers, queue_cap=%u", num_workers, queue_cap);
    return ORCHKV_OK;
}

void io_worker_destroy(io_worker_pool_t *pool)
{
    if (!pool->threads) return;

    pthread_mutex_lock(&pool->lock);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->not_empty);
    pthread_mutex_unlock(&pool->lock);

    for (int i = 0; i < pool->num_workers; i++)
        pthread_join(pool->threads[i], NULL);

    LOG_INFO("io_worker_pool: submitted=%lu completed=%lu",
             (unsigned long)pool->submitted, (unsigned long)pool->completed);

    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->not_empty);
    pthread_cond_destroy(&pool->not_full);
    pthread_cond_destroy(&pool->all_done);

    free(pool->queue);
    free(pool->threads);
    pool->threads = NULL;
}

int io_worker_submit(io_worker_pool_t *pool, const io_task_t *task)
{
    pthread_mutex_lock(&pool->lock);

    while (pool->queue_count == pool->queue_cap && !pool->shutdown)
        pthread_cond_wait(&pool->not_full, &pool->lock);

    if (pool->shutdown) {
        pthread_mutex_unlock(&pool->lock);
        return ORCHKV_ERR_INIT;
    }

    pool->queue[pool->queue_tail] = *task;
    pool->queue_tail = (pool->queue_tail + 1) % pool->queue_cap;
    pool->queue_count++;
    pool->submitted++;

    pthread_cond_signal(&pool->not_empty);
    pthread_mutex_unlock(&pool->lock);
    return ORCHKV_OK;
}

void io_worker_flush(io_worker_pool_t *pool)
{
    pthread_mutex_lock(&pool->lock);
    while (pool->completed < pool->submitted)
        pthread_cond_wait(&pool->all_done, &pool->lock);
    pthread_mutex_unlock(&pool->lock);
}
