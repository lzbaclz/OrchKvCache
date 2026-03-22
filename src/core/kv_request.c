#include "kv_request.h"
#include <stdlib.h>
#include <string.h>

/* ---- head vector helpers ---- */

static int hvec_init(kv_head_vec_t *hv, uint32_t initial_cap)
{
    if (initial_cap == 0)
        initial_cap = 4;
    hv->items    = (kv_block_t **)calloc(initial_cap, sizeof(kv_block_t *));
    if (!hv->items)
        return ORCHKV_ERR_OOM;
    hv->count    = 0;
    hv->capacity = initial_cap;
    return ORCHKV_OK;
}

static void hvec_destroy(kv_head_vec_t *hv)
{
    /* destroy each block's metadata */
    for (uint32_t i = 0; i < hv->count; i++) {
        if (hv->items[i]) {
            kv_block_destroy(hv->items[i]);
            free(hv->items[i]);
        }
    }
    free(hv->items);
    hv->items    = NULL;
    hv->count    = 0;
    hv->capacity = 0;
}

static int hvec_push(kv_head_vec_t *hv, kv_block_t *blk)
{
    if (hv->count >= hv->capacity) {
        uint32_t new_cap = hv->capacity * 2;
        kv_block_t **tmp = (kv_block_t **)realloc(hv->items,
                                                   new_cap * sizeof(kv_block_t *));
        if (!tmp)
            return ORCHKV_ERR_OOM;
        hv->items    = tmp;
        hv->capacity = new_cap;
    }
    hv->items[hv->count++] = blk;
    return ORCHKV_OK;
}

/* ---- public API ---- */

kv_request_ctx_t *kv_request_create(uint64_t  request_id,
                                    uint32_t  n_layers,
                                    uint32_t  n_kv_heads,
                                    uint32_t  d_head,
                                    DataType  dtype,
                                    uint32_t  tokens_per_block)
{
    kv_request_ctx_t *ctx = (kv_request_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;

    ctx->request_id       = request_id;
    ctx->n_layers         = n_layers;
    ctx->n_kv_heads       = n_kv_heads;
    ctx->d_head           = d_head;
    ctx->dtype            = dtype;
    ctx->tokens_per_block = tokens_per_block;
    ctx->seq_len          = 0;
    ctx->total_blocks     = 0;
    ctx->blocks_on_gpu    = 0;
    ctx->blocks_on_dram   = 0;
    ctx->active           = true;
    pthread_mutex_init(&ctx->lock, NULL);

    size_t n_hvecs = (size_t)n_layers * n_kv_heads;
    ctx->hvecs = (kv_head_vec_t *)calloc(n_hvecs, sizeof(kv_head_vec_t));
    if (!ctx->hvecs) {
        free(ctx);
        return NULL;
    }

    for (size_t i = 0; i < n_hvecs; i++) {
        if (hvec_init(&ctx->hvecs[i], 4) != ORCHKV_OK) {
            /* rollback */
            for (size_t j = 0; j < i; j++)
                hvec_destroy(&ctx->hvecs[j]);
            free(ctx->hvecs);
            free(ctx);
            return NULL;
        }
    }

    LOG_INFO("request %lu created: %u layers, %u kv_heads, d=%u, %s",
             (unsigned long)request_id, n_layers, n_kv_heads, d_head,
             dtype_name(dtype));
    return ctx;
}

void kv_request_destroy(kv_request_ctx_t *ctx)
{
    if (!ctx) return;

    size_t n_hvecs = (size_t)ctx->n_layers * ctx->n_kv_heads;
    for (size_t i = 0; i < n_hvecs; i++)
        hvec_destroy(&ctx->hvecs[i]);

    free(ctx->hvecs);
    pthread_mutex_destroy(&ctx->lock);

    LOG_INFO("request %lu destroyed, was holding %lu blocks",
             (unsigned long)ctx->request_id,
             (unsigned long)ctx->total_blocks);

    free(ctx);
}

kv_block_t *kv_request_get_block(kv_request_ctx_t *ctx,
                                 uint32_t layer,
                                 uint32_t head,
                                 uint32_t block_idx)
{
    kv_head_vec_t *hv = kv_request_hvec(ctx, layer, head);
    if (!hv || block_idx >= hv->count)
        return NULL;
    return hv->items[block_idx];
}

int kv_request_alloc_blocks_for_layer(kv_request_ctx_t *ctx,
                                      uint32_t layer,
                                      uint32_t n_tokens)
{
    if (layer >= ctx->n_layers)
        return ORCHKV_ERR_INVALID;
    if (n_tokens == 0)
        return 0;

    /*
     * Figure out how many new blocks each head needs.
     * Current head already has some blocks that may have spare capacity.
     */
    kv_head_vec_t *hv0 = kv_request_hvec(ctx, layer, 0);
    uint32_t existing_tokens = 0;
    if (hv0->count > 0) {
        /* All heads in the same layer have identical token layout */
        kv_block_t *last = hv0->items[hv0->count - 1];
        existing_tokens = last->token_start + last->token_count;
    }

    uint32_t total_after   = existing_tokens + n_tokens;
    uint32_t blocks_needed = ORCHKV_DIV_CEIL(total_after, ctx->tokens_per_block);
    uint32_t blocks_have   = hv0->count;

    if (blocks_needed <= blocks_have) {
        /* last block has room: just update its token_count */
        uint32_t remaining = n_tokens;
        for (uint32_t h = 0; h < ctx->n_kv_heads; h++) {
            kv_head_vec_t *hv = kv_request_hvec(ctx, layer, h);
            kv_block_t *last = hv->items[hv->count - 1];
            last->token_count = (uint16_t)ORCHKV_MIN(
                ctx->tokens_per_block,
                last->token_count + remaining);
        }
        return 0;
    }

    int new_count = (int)(blocks_needed - blocks_have);

    /* Possibly extend the last existing block to full capacity first */
    if (blocks_have > 0) {
        for (uint32_t h = 0; h < ctx->n_kv_heads; h++) {
            kv_head_vec_t *hv = kv_request_hvec(ctx, layer, h);
            kv_block_t *last = hv->items[hv->count - 1];
            last->token_count = (uint16_t)ctx->tokens_per_block;
        }
    }

    /* Allocate new blocks */
    for (int bi = 0; bi < new_count; bi++) {
        uint32_t tok_start = (blocks_have + (uint32_t)bi) * ctx->tokens_per_block;
        bool is_last = (bi == new_count - 1);
        uint32_t tok_cnt;

        if (is_last) {
            tok_cnt = total_after - tok_start;
            if (tok_cnt > ctx->tokens_per_block)
                tok_cnt = ctx->tokens_per_block;
        } else {
            tok_cnt = ctx->tokens_per_block;
        }

        for (uint32_t h = 0; h < ctx->n_kv_heads; h++) {
            kv_block_t *blk = (kv_block_t *)calloc(1, sizeof(kv_block_t));
            if (!blk)
                return ORCHKV_ERR_OOM;

            kv_block_init(blk, ctx->request_id, (uint16_t)layer,
                          (uint16_t)h, tok_start, (uint16_t)tok_cnt);

            kv_head_vec_t *hv = kv_request_hvec(ctx, layer, h);
            int rc = hvec_push(hv, blk);
            if (rc != ORCHKV_OK) {
                kv_block_destroy(blk);
                free(blk);
                return rc;
            }
            ctx->total_blocks++;
        }
    }

    return new_count;
}

void kv_request_advance_seq(kv_request_ctx_t *ctx, uint32_t n_tokens)
{
    ctx->seq_len += n_tokens;
}
