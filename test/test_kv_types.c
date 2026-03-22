/*
 * test_kv_types.c – verify kv_types.h compiles correctly and helpers work.
 * This is the A1 smoke test.
 */
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "core/kv_types.h"

static void test_dtype_size(void)
{
    assert(dtype_size(DTYPE_FP16) == 2);
    assert(dtype_size(DTYPE_BF16) == 2);
    assert(dtype_size(DTYPE_FP32) == 4);
    assert(dtype_size(DTYPE_INT8) == 1);
    assert(dtype_size(DTYPE_INT4) == 1);
    printf("  [PASS] dtype_size\n");
}

static void test_dtype_name(void)
{
    assert(strcmp(dtype_name(DTYPE_FP16), "fp16") == 0);
    assert(strcmp(dtype_name(DTYPE_FP32), "fp32") == 0);
    printf("  [PASS] dtype_name\n");
}

static void test_tier_name(void)
{
    assert(strcmp(tier_name(TIER_GPU_HBM), "GPU_HBM") == 0);
    assert(strcmp(tier_name(TIER_HOST_DRAM), "HOST_DRAM") == 0);
    assert(strcmp(tier_name(TIER_NVM), "NVM") == 0);
    assert(strcmp(tier_name(TIER_SSD), "SSD") == 0);
    assert(strcmp(tier_name(TIER_NONE), "NONE") == 0);
    printf("  [PASS] tier_name\n");
}

static void test_block_state_name(void)
{
    assert(strcmp(block_state_name(KV_STATE_FREE), "FREE") == 0);
    assert(strcmp(block_state_name(KV_STATE_HOT), "HOT") == 0);
    assert(strcmp(block_state_name(KV_STATE_MIGRATING), "MIGRATING") == 0);
    printf("  [PASS] block_state_name\n");
}

static void test_kv_block_data_bytes(void)
{
    /* 64 tokens, d_head=128, FP16 → 64×128×2(K+V)×2B = 32768 = 32KB */
    size_t sz = kv_block_data_bytes(64, 128, DTYPE_FP16);
    assert(sz == 32768);
    assert(sz == KV_BLOCK_SIZE);

    /* 64 tokens, d_head=128, FP32 → 64×128×2×4B = 65536 = 64KB */
    sz = kv_block_data_bytes(64, 128, DTYPE_FP32);
    assert(sz == 65536);

    /* 64 tokens, d_head=128, INT8 → 64×128×2×1B = 16384 = 16KB */
    sz = kv_block_data_bytes(64, 128, DTYPE_INT8);
    assert(sz == 16384);

    printf("  [PASS] kv_block_data_bytes\n");
}

static void test_align_macros(void)
{
    assert(ORCHKV_ALIGN_UP(1, 4096) == 4096);
    assert(ORCHKV_ALIGN_UP(4096, 4096) == 4096);
    assert(ORCHKV_ALIGN_UP(4097, 4096) == 8192);
    assert(ORCHKV_ALIGN_DOWN(8191, 4096) == 4096);
    assert(ORCHKV_ALIGN_DOWN(8192, 4096) == 8192);
    printf("  [PASS] align macros\n");
}

static void test_util_macros(void)
{
    assert(ORCHKV_MIN(3, 5) == 3);
    assert(ORCHKV_MAX(3, 5) == 5);
    assert(ORCHKV_DIV_CEIL(10, 3) == 4);
    assert(ORCHKV_DIV_CEIL(9, 3) == 3);
    assert(ORCHKV_DIV_CEIL(1, 64) == 1);
    printf("  [PASS] util macros\n");
}

static void test_config_default(void)
{
    orchkv_config_t cfg;
    orchkv_config_default(&cfg);

    assert(cfg.gpu_device_id == 0);
    assert(cfg.tokens_per_block == KV_TOKEN_GROUP_SIZE);
    assert(cfg.d_head == 128);
    assert(cfg.dtype == DTYPE_FP16);
    assert(cfg.dram_use_pinned == true);
    assert(cfg.gpu_hwm > 0.8f && cfg.gpu_hwm < 1.0f);
    assert(cfg.orchfs_nvm_path == NULL);
    printf("  [PASS] config_default\n");
}

static void test_flags(void)
{
    uint8_t flags = KV_FLAG_NONE;
    flags |= KV_FLAG_PIN;
    assert(flags & KV_FLAG_PIN);
    assert(!(flags & KV_FLAG_DIRTY));
    flags |= KV_FLAG_DIRTY | KV_FLAG_ATTN_SINK;
    assert(flags & KV_FLAG_DIRTY);
    assert(flags & KV_FLAG_ATTN_SINK);
    flags &= ~KV_FLAG_PIN;
    assert(!(flags & KV_FLAG_PIN));
    printf("  [PASS] flags\n");
}

int main(void)
{
    printf("=== test_kv_types ===\n");
    test_dtype_size();
    test_dtype_name();
    test_tier_name();
    test_block_state_name();
    test_kv_block_data_bytes();
    test_align_macros();
    test_util_macros();
    test_config_default();
    test_flags();
    printf("=== ALL PASSED ===\n");
    return 0;
}
