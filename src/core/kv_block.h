#ifndef ORCHKV_KV_BLOCK_H
#define ORCHKV_KV_BLOCK_H

#include "kv_types.h"

#ifndef __CUDACC__
#include <stdatomic.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  KV Block – the fundamental management unit.
 *
 *  Each block holds K and V data for a single KV-head, single layer,
 *  covering up to `tokens_per_block` contiguous tokens.
 *  With d_head=128, FP16, 64 tokens: data size = 32 KB = ORCH_BLOCK_SIZE.
 * ======================================================================== */

typedef struct kv_block {
    /* --- identity --- */
    uint64_t        block_id;           /* globally unique, assigned by atomic counter */
    uint64_t        request_id;
    uint16_t        layer_id;
    uint16_t        head_id;            /* KV-head index (GQA-aware) */
    uint32_t        token_start;        /* first token position in the sequence */
    uint16_t        token_count;        /* actual tokens stored (≤ tokens_per_block) */
    uint16_t        _pad0;

    /* --- storage location --- */
    StorageTier     tier;
    void           *data_ptr;           /* GPU or DRAM pointer; NULL when on NVM/SSD */
    uint64_t        persistent_offset;  /* OrchFS file offset (Phase B) */

    /* --- hotness (Phase C fills these; Phase A keeps defaults) --- */
    float           hotness;
    uint64_t        last_access_step;
    uint32_t        access_count;

    /* --- lifecycle --- */
    KVBlockState    state;
    uint8_t         flags;              /* KV_FLAG_* bitmask */

    /* --- concurrency --- */
    pthread_rwlock_t lock;

    /* --- intrusive doubly-linked list pointers (free list, LRU, etc.) --- */
    struct kv_block *prev;
    struct kv_block *next;
} kv_block_t;

/* ---------------- Global block-ID counter (module-internal) ------------- */
#ifndef __CUDACC__
extern atomic_uint_fast64_t g_block_id_counter;
#endif

/* ---------------- API --------------------------------------------------- */

/*
 * Initialise a block struct.  Assigns a new unique block_id.
 * Does NOT allocate the data buffer – that is the tier's job.
 */
void kv_block_init(kv_block_t *blk,
                   uint64_t    request_id,
                   uint16_t    layer_id,
                   uint16_t    head_id,
                   uint32_t    token_start,
                   uint16_t    token_count);

/* Destroy: releases the rwlock.  Does NOT free the data buffer. */
void kv_block_destroy(kv_block_t *blk);

/* Compute data payload size in bytes for this block's geometry. */
static inline size_t kv_block_payload_size(const kv_block_t *blk,
                                           uint32_t d_head,
                                           DataType dtype)
{
    return kv_block_data_bytes(blk->token_count, d_head, dtype);
}

/* Update tier + data pointer atomically under write-lock. */
void kv_block_set_location(kv_block_t *blk, StorageTier tier, void *ptr);

/* State transition (validates legality). Returns ORCHKV_OK or ORCHKV_ERR_STATE. */
int kv_block_set_state(kv_block_t *blk, KVBlockState new_state);

/* Lock helpers (thin wrappers for readability) */
static inline void kv_block_rdlock(kv_block_t *blk)  { pthread_rwlock_rdlock(&blk->lock); }
static inline void kv_block_wrlock(kv_block_t *blk)  { pthread_rwlock_wrlock(&blk->lock); }
static inline void kv_block_unlock(kv_block_t *blk)  { pthread_rwlock_unlock(&blk->lock); }

/* Check flag helpers */
static inline bool kv_block_is_pinned(const kv_block_t *blk)
{
    return (blk->flags & KV_FLAG_PIN) != 0;
}

static inline bool kv_block_is_dirty(const kv_block_t *blk)
{
    return (blk->flags & KV_FLAG_DIRTY) != 0;
}

/* Reset the block ID counter (used in tests). */
void kv_block_reset_id_counter(void);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_KV_BLOCK_H */
