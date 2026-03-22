#ifndef ORCHKV_GPU_TIER_H
#define ORCHKV_GPU_TIER_H

#include "tier_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_tier {
    int          device_id;
    void        *pool_base;      /* device pointer from cudaMalloc */
    size_t       pool_size;
    size_t       slab_size;      /* bytes per slab (= KV block data size) */
    uint32_t     total_slabs;
    uint32_t     used_slabs;
    uint32_t    *free_stack;     /* host-side LIFO stack of free slab indices */
    uint32_t     free_top;       /* stack pointer (points to next push slot) */
    pthread_mutex_t lock;
} gpu_tier_t;

int  gpu_tier_init(gpu_tier_t *t, int device_id,
                   size_t pool_bytes, size_t slab_size);
void gpu_tier_destroy(gpu_tier_t *t);

/* O(1) slab alloc; returns device pointer via out_ptr */
int  gpu_tier_alloc(gpu_tier_t *t, void **out_ptr);
/* O(1) slab free */
int  gpu_tier_free(gpu_tier_t *t, void *dev_ptr);

static inline float gpu_tier_usage(const gpu_tier_t *t)
{
    return t->total_slabs ? (float)t->used_slabs / t->total_slabs : 0.f;
}

static inline uint32_t gpu_tier_free_count(const gpu_tier_t *t)
{
    return t->total_slabs - t->used_slabs;
}

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_GPU_TIER_H */
