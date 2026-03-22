extern "C" {
#include "dram_tier.h"
}
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

int dram_tier_init(dram_tier_t *t, size_t pool_bytes,
                   size_t slab_size, bool use_pinned)
{
    memset(t, 0, sizeof(*t));

    if (slab_size == 0 || pool_bytes < slab_size)
        return ORCHKV_ERR_INVALID;

    t->slab_size   = slab_size;
    t->total_slabs = (uint32_t)(pool_bytes / slab_size);
    t->pool_size   = (size_t)t->total_slabs * slab_size;
    t->is_pinned   = false;

    if (use_pinned) {
        cudaError_t err = cudaMallocHost(&t->pool_base, t->pool_size);
        if (err == cudaSuccess) {
            t->is_pinned = true;
        } else {
            LOG_WARN("cudaMallocHost(%zu MB) failed (%s), falling back to malloc",
                     t->pool_size >> 20, cudaGetErrorString(err));
        }
    }

    if (!t->is_pinned) {
        t->pool_base = aligned_alloc(4096, t->pool_size);
        if (!t->pool_base)
            return ORCHKV_ERR_OOM;
    }

    t->free_stack = (uint32_t *)malloc(t->total_slabs * sizeof(uint32_t));
    if (!t->free_stack) {
        if (t->is_pinned) cudaFreeHost(t->pool_base);
        else free(t->pool_base);
        return ORCHKV_ERR_OOM;
    }

    for (uint32_t i = 0; i < t->total_slabs; i++)
        t->free_stack[i] = t->total_slabs - 1 - i;
    t->free_top   = t->total_slabs;
    t->used_slabs = 0;

    pthread_mutex_init(&t->lock, NULL);

    LOG_INFO("dram_tier: pool %zu MB, slab %zu KB, %u slabs, pinned=%d",
             t->pool_size >> 20, slab_size >> 10, t->total_slabs, t->is_pinned);
    return ORCHKV_OK;
}

void dram_tier_destroy(dram_tier_t *t)
{
    if (t->pool_base) {
        if (t->is_pinned)
            cudaFreeHost(t->pool_base);
        else
            free(t->pool_base);
        t->pool_base = NULL;
    }
    free(t->free_stack);
    t->free_stack = NULL;
    pthread_mutex_destroy(&t->lock);
}

int dram_tier_alloc(dram_tier_t *t, void **out_ptr)
{
    pthread_mutex_lock(&t->lock);
    if (t->free_top == 0) {
        pthread_mutex_unlock(&t->lock);
        return ORCHKV_ERR_TIER_FULL;
    }
    uint32_t idx = t->free_stack[--t->free_top];
    t->used_slabs++;
    pthread_mutex_unlock(&t->lock);

    *out_ptr = (char *)t->pool_base + (size_t)idx * t->slab_size;
    return ORCHKV_OK;
}

int dram_tier_free(dram_tier_t *t, void *host_ptr)
{
    ptrdiff_t off = (char *)host_ptr - (char *)t->pool_base;
    if (off < 0 || (size_t)off >= t->pool_size || (size_t)off % t->slab_size != 0) {
        LOG_ERR("dram_tier_free: pointer not in pool");
        return ORCHKV_ERR_INVALID;
    }
    uint32_t idx = (uint32_t)((size_t)off / t->slab_size);

    pthread_mutex_lock(&t->lock);
    t->free_stack[t->free_top++] = idx;
    t->used_slabs--;
    pthread_mutex_unlock(&t->lock);
    return ORCHKV_OK;
}
