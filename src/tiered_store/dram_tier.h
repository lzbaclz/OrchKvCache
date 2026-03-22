#ifndef ORCHKV_DRAM_TIER_H
#define ORCHKV_DRAM_TIER_H

#include "tier_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dram_tier {
    void        *pool_base;      /* host pointer (pinned or malloc) */
    size_t       pool_size;
    size_t       slab_size;
    uint32_t     total_slabs;
    uint32_t     used_slabs;
    uint32_t    *free_stack;
    uint32_t     free_top;
    bool         is_pinned;      /* true if allocated with cudaMallocHost */
    pthread_mutex_t lock;
} dram_tier_t;

int  dram_tier_init(dram_tier_t *t, size_t pool_bytes,
                    size_t slab_size, bool use_pinned);
void dram_tier_destroy(dram_tier_t *t);

int  dram_tier_alloc(dram_tier_t *t, void **out_ptr);
int  dram_tier_free(dram_tier_t *t, void *host_ptr);

static inline float dram_tier_usage(const dram_tier_t *t)
{
    return t->total_slabs ? (float)t->used_slabs / t->total_slabs : 0.f;
}

static inline uint32_t dram_tier_free_count(const dram_tier_t *t)
{
    return t->total_slabs - t->used_slabs;
}

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_DRAM_TIER_H */
