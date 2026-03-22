#include <assert.h>
#include <stdio.h>
#include "core/kv_request.h"

static void test_create_destroy(void)
{
    kv_block_reset_id_counter();
    kv_request_ctx_t *ctx = kv_request_create(
        /*request_id=*/1, /*n_layers=*/32, /*n_kv_heads=*/8,
        /*d_head=*/128, DTYPE_FP16, /*tokens_per_block=*/64);
    assert(ctx != NULL);
    assert(ctx->request_id == 1);
    assert(ctx->n_layers == 32);
    assert(ctx->n_kv_heads == 8);
    assert(ctx->d_head == 128);
    assert(ctx->dtype == DTYPE_FP16);
    assert(ctx->tokens_per_block == 64);
    assert(ctx->seq_len == 0);
    assert(ctx->total_blocks == 0);
    assert(ctx->active == true);

    kv_request_destroy(ctx);
    printf("  [PASS] create_destroy\n");
}

static void test_alloc_prefill(void)
{
    kv_block_reset_id_counter();

    /* Simulate LLaMA-3-8B: 32 layers, 8 KV heads, d=128, FP16 */
    kv_request_ctx_t *ctx = kv_request_create(1, 32, 8, 128, DTYPE_FP16, 64);
    assert(ctx);

    /* Prefill 256 tokens → ceil(256/64)=4 blocks per head per layer */
    for (uint32_t l = 0; l < 32; l++) {
        int new_blocks = kv_request_alloc_blocks_for_layer(ctx, l, 256);
        assert(new_blocks == 4);
    }
    kv_request_advance_seq(ctx, 256);

    assert(ctx->seq_len == 256);
    /* total = 32 layers × 8 heads × 4 blocks = 1024 */
    assert(ctx->total_blocks == 1024);

    /* Verify block geometry */
    kv_block_t *b0 = kv_request_get_block(ctx, 0, 0, 0);
    assert(b0 != NULL);
    assert(b0->layer_id == 0);
    assert(b0->head_id == 0);
    assert(b0->token_start == 0);
    assert(b0->token_count == 64);
    assert(b0->request_id == 1);

    kv_block_t *b3 = kv_request_get_block(ctx, 0, 0, 3);
    assert(b3 != NULL);
    assert(b3->token_start == 192);
    assert(b3->token_count == 64);  /* 256-192 = 64, full */

    /* Last layer, last head, last block */
    kv_block_t *blast = kv_request_get_block(ctx, 31, 7, 3);
    assert(blast != NULL);
    assert(blast->layer_id == 31);
    assert(blast->head_id == 7);

    /* Out of range */
    assert(kv_request_get_block(ctx, 0, 0, 4) == NULL);
    assert(kv_request_get_block(ctx, 32, 0, 0) == NULL);
    assert(kv_request_get_block(ctx, 0, 8, 0) == NULL);

    kv_request_destroy(ctx);
    printf("  [PASS] alloc_prefill\n");
}

static void test_alloc_partial_block(void)
{
    kv_block_reset_id_counter();

    kv_request_ctx_t *ctx = kv_request_create(2, 2, 2, 128, DTYPE_FP16, 64);
    assert(ctx);

    /* 100 tokens → 2 blocks: [0..63] full, [64..99] partial (36 tokens) */
    for (uint32_t l = 0; l < 2; l++) {
        int new_b = kv_request_alloc_blocks_for_layer(ctx, l, 100);
        assert(new_b == 2);
    }
    kv_request_advance_seq(ctx, 100);

    kv_block_t *b0 = kv_request_get_block(ctx, 0, 0, 0);
    assert(b0->token_count == 64);

    kv_block_t *b1 = kv_request_get_block(ctx, 0, 0, 1);
    assert(b1->token_start == 64);
    assert(b1->token_count == 36);

    kv_request_destroy(ctx);
    printf("  [PASS] alloc_partial_block\n");
}

static void test_decode_append(void)
{
    kv_block_reset_id_counter();

    kv_request_ctx_t *ctx = kv_request_create(3, 2, 2, 128, DTYPE_FP16, 64);
    assert(ctx);

    /* Prefill 60 tokens (1 partial block with 60 tokens) */
    for (uint32_t l = 0; l < 2; l++)
        kv_request_alloc_blocks_for_layer(ctx, l, 60);
    kv_request_advance_seq(ctx, 60);
    assert(ctx->total_blocks == 2 * 2 * 1);  /* 4 blocks */

    kv_block_t *b = kv_request_get_block(ctx, 0, 0, 0);
    assert(b->token_count == 60);

    /* Decode: append 4 tokens → still fits in the existing block */
    for (uint32_t l = 0; l < 2; l++) {
        int new_b = kv_request_alloc_blocks_for_layer(ctx, l, 4);
        assert(new_b == 0);  /* no new block needed */
    }
    kv_request_advance_seq(ctx, 4);

    b = kv_request_get_block(ctx, 0, 0, 0);
    assert(b->token_count == 64);  /* 60+4=64 = full */
    assert(ctx->total_blocks == 4);

    /* Decode: append 1 more → needs a new block */
    for (uint32_t l = 0; l < 2; l++) {
        int new_b = kv_request_alloc_blocks_for_layer(ctx, l, 1);
        assert(new_b == 1);
    }
    kv_request_advance_seq(ctx, 1);

    assert(ctx->total_blocks == 4 + 2 * 2);  /* 8 */
    kv_block_t *bnew = kv_request_get_block(ctx, 0, 0, 1);
    assert(bnew != NULL);
    assert(bnew->token_start == 64);
    assert(bnew->token_count == 1);

    kv_request_destroy(ctx);
    printf("  [PASS] decode_append\n");
}

static void test_payload_sizes(void)
{
    kv_block_reset_id_counter();

    kv_request_ctx_t *ctx = kv_request_create(4, 1, 1, 128, DTYPE_FP16, 64);
    assert(ctx);

    kv_request_alloc_blocks_for_layer(ctx, 0, 64);
    kv_request_advance_seq(ctx, 64);

    kv_block_t *blk = kv_request_get_block(ctx, 0, 0, 0);
    size_t sz = kv_block_payload_size(blk, ctx->d_head, ctx->dtype);
    assert(sz == 32768);  /* 32 KB – matches ORCH_BLOCK_SIZE */

    kv_request_destroy(ctx);
    printf("  [PASS] payload_sizes\n");
}

int main(void)
{
    printf("=== test_kv_request ===\n");
    test_create_destroy();
    test_alloc_prefill();
    test_alloc_partial_block();
    test_decode_append();
    test_payload_sizes();
    printf("=== ALL PASSED ===\n");
    return 0;
}
