#ifndef ORCHKV_KV_REQUEST_H
#define ORCHKV_KV_REQUEST_H

#include "kv_types.h"
#include "kv_block.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  KV Request Context
 *
 *  Owns all KV blocks for a single inference request.
 *  Blocks are indexed as blocks[layer][head][block_idx].
 *  The inner dimension grows dynamically as tokens are appended.
 * ======================================================================== */

/* Per-head block vector (dynamic array) */
typedef struct kv_head_vec {
    kv_block_t **items;
    uint32_t     count;         /* number of blocks currently held */
    uint32_t     capacity;      /* allocated slots */
} kv_head_vec_t;

typedef struct kv_request_ctx {
    uint64_t        request_id;
    uint32_t        n_layers;
    uint32_t        n_kv_heads;
    uint32_t        d_head;
    DataType        dtype;
    uint32_t        tokens_per_block;   /* KV_TOKEN_GROUP_SIZE */

    /* Current sequence state */
    uint32_t        seq_len;

    /*
     * 2-D array of head-vectors:  hvecs[layer * n_kv_heads + head]
     * Flattened for cache-friendliness; each entry is a kv_head_vec_t.
     */
    kv_head_vec_t  *hvecs;

    /* Aggregate counters */
    uint64_t        total_blocks;
    uint64_t        blocks_on_gpu;
    uint64_t        blocks_on_dram;

    bool            active;
    pthread_mutex_t lock;
} kv_request_ctx_t;

/* ---------- Lifecycle --------------------------------------------------- */

/*
 * Allocate and initialise a request context.
 * Returns NULL on OOM.
 */
kv_request_ctx_t *kv_request_create(uint64_t  request_id,
                                    uint32_t  n_layers,
                                    uint32_t  n_kv_heads,
                                    uint32_t  d_head,
                                    DataType  dtype,
                                    uint32_t  tokens_per_block);

/*
 * Destroy the context and all owned kv_block_t objects (metadata only;
 * the caller must free the actual GPU/DRAM data buffers first).
 */
void kv_request_destroy(kv_request_ctx_t *ctx);

/* ---------- Block access ------------------------------------------------ */

/* Get the head-vector for (layer, head).  Returns NULL on invalid indices. */
static inline kv_head_vec_t *kv_request_hvec(kv_request_ctx_t *ctx,
                                             uint32_t layer,
                                             uint32_t head)
{
    if (layer >= ctx->n_layers || head >= ctx->n_kv_heads)
        return NULL;
    return &ctx->hvecs[layer * ctx->n_kv_heads + head];
}

/* Get a specific block.  Returns NULL if out of range. */
kv_block_t *kv_request_get_block(kv_request_ctx_t *ctx,
                                 uint32_t layer,
                                 uint32_t head,
                                 uint32_t block_idx);

/* ---------- Allocation -------------------------------------------------- */

/*
 * Allocate blocks for `n_tokens` new tokens across ALL heads for one layer.
 * Blocks are created with state FREE; caller is responsible for assigning
 * storage (slab alloc) and transitioning to ALLOCATED → HOT.
 *
 * Returns the number of new blocks created per head, or negative error.
 */
int kv_request_alloc_blocks_for_layer(kv_request_ctx_t *ctx,
                                      uint32_t layer,
                                      uint32_t n_tokens);

/*
 * Notify the context that the overall sequence grew by `n_tokens`.
 * This is bookkeeping only — call alloc_blocks_for_layer per layer first.
 */
void kv_request_advance_seq(kv_request_ctx_t *ctx, uint32_t n_tokens);

/* Total number of blocks across all layers & heads */
static inline uint64_t kv_request_total_blocks(const kv_request_ctx_t *ctx)
{
    return ctx->total_blocks;
}

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_KV_REQUEST_H */
