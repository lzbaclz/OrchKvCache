extern "C" {
#include "api/orchkv_api.h"
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

/* ---- Config for small-scale 4-tier test ---- */
#define N_LAYERS       4
#define N_KV_HEADS     4
#define D_HEAD         128
#define TOKENS_PER_BLK 64
#define SEQ_LEN        128   /* 2 blocks per head */
#define GPU_POOL_MB    32
#define DRAM_POOL_MB   32

static void fill_pattern(void *buf, size_t len, uint8_t seed)
{
    uint8_t *p = (uint8_t *)buf;
    for (size_t i = 0; i < len; i++)
        p[i] = (uint8_t)(seed + (i & 0xFF));
}

static bool check_pattern(const void *buf, size_t len, uint8_t seed)
{
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < len; i++)
        if (p[i] != (uint8_t)(seed + (i & 0xFF))) return false;
    return true;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Test 1: full lifecycle GPU → DRAM → Storage → DRAM → GPU ---- */
static void test_4tier_roundtrip(void)
{
    printf("  test_4tier_roundtrip ... ");

    kv_request_ctx_t *ctx = orchkv_request_create(1, N_LAYERS, N_KV_HEADS);
    assert(ctx);
    assert(ctx->orchfs_fctx != NULL);

    size_t head_bytes = (size_t)SEQ_LEN * D_HEAD * 2;
    size_t layer_bytes = head_bytes * N_KV_HEADS;
    uint8_t *k_data = (uint8_t *)malloc(layer_bytes);
    uint8_t *v_data = (uint8_t *)malloc(layer_bytes);
    assert(k_data && v_data);

    for (uint32_t l = 0; l < N_LAYERS; l++) {
        fill_pattern(k_data, layer_bytes, (uint8_t)(0x10 + l));
        fill_pattern(v_data, layer_bytes, (uint8_t)(0x50 + l));
        int rc = orchkv_prefill(ctx, l, k_data, v_data, SEQ_LEN);
        assert(rc == ORCHKV_OK);
    }

    /* All blocks on GPU */
    assert(ctx->blocks_on_gpu == (uint64_t)N_LAYERS * N_KV_HEADS * 2);
    assert(ctx->blocks_on_dram == 0);
    assert(ctx->blocks_on_storage == 0);

    /* Evict layer 0 to DRAM */
    for (uint32_t h = 0; h < N_KV_HEADS; h++) {
        for (uint32_t bi = 0; bi < 2; bi++) {
            int rc = orchkv_evict_to_dram(ctx, 0, h, bi);
            assert(rc == ORCHKV_OK);
        }
    }
    assert(ctx->blocks_on_dram == N_KV_HEADS * 2);

    /* Evict layer 0 from DRAM to Storage */
    for (uint32_t h = 0; h < N_KV_HEADS; h++) {
        for (uint32_t bi = 0; bi < 2; bi++) {
            int rc = orchkv_evict_to_storage(ctx, 0, h, bi);
            assert(rc == ORCHKV_OK);
        }
    }
    assert(ctx->blocks_on_storage == N_KV_HEADS * 2);
    assert(ctx->blocks_on_dram == 0);

    /* Verify blocks are on NVM tier */
    kv_block_t *blk = kv_request_get_block(ctx, 0, 0, 0);
    assert(blk->tier == TIER_NVM);
    assert(blk->data_ptr == NULL);

    /* Promote from storage → DRAM */
    for (uint32_t h = 0; h < N_KV_HEADS; h++) {
        for (uint32_t bi = 0; bi < 2; bi++) {
            int rc = orchkv_promote_from_storage(ctx, 0, h, bi);
            assert(rc == ORCHKV_OK);
        }
    }
    assert(ctx->blocks_on_dram == N_KV_HEADS * 2);
    assert(ctx->blocks_on_storage == 0);

    /* Promote from DRAM → GPU and verify data */
    for (uint32_t h = 0; h < N_KV_HEADS; h++) {
        for (uint32_t bi = 0; bi < 2; bi++) {
            int rc = orchkv_promote_to_gpu(ctx, 0, h, bi);
            assert(rc == ORCHKV_OK);
        }
    }

    /* Verify data integrity after full roundtrip */
    size_t slab_size = (size_t)TOKENS_PER_BLK * D_HEAD * 2 * 2;
    size_t kh = slab_size / 2;
    uint8_t *host_buf = (uint8_t *)malloc(slab_size);
    assert(host_buf);

    fill_pattern(k_data, layer_bytes, (uint8_t)(0x10));

    for (uint32_t h = 0; h < N_KV_HEADS; h++) {
        void *k_ptr, *v_ptr;
        int rc = orchkv_get_kv_block(ctx, 0, h, 0, &k_ptr, &v_ptr);
        assert(rc == ORCHKV_OK);

        cudaMemcpy(host_buf, k_ptr, kh, cudaMemcpyDeviceToHost);

        size_t tok_bytes = (size_t)TOKENS_PER_BLK * D_HEAD * 2;
        assert(memcmp(host_buf, k_data + h * head_bytes, tok_bytes) == 0);
    }

    free(host_buf);
    free(k_data);
    free(v_data);
    orchkv_request_destroy(ctx);
    printf("PASS\n");
}

/* ---- Test 2: evict_cold (two-hop: GPU → DRAM → Storage) ---- */
static void test_evict_cold(void)
{
    printf("  test_evict_cold ... ");

    kv_request_ctx_t *ctx = orchkv_request_create(2, 2, 2);
    assert(ctx);

    size_t head_bytes = (size_t)64 * D_HEAD * 2;
    size_t layer_bytes = head_bytes * 2;
    uint8_t *kd = (uint8_t *)malloc(layer_bytes);
    uint8_t *vd = (uint8_t *)malloc(layer_bytes);
    fill_pattern(kd, layer_bytes, 0xAA);
    fill_pattern(vd, layer_bytes, 0xBB);

    for (uint32_t l = 0; l < 2; l++)
        assert(orchkv_prefill(ctx, l, kd, vd, 64) == ORCHKV_OK);

    /* Two-hop evict */
    int rc = orchkv_evict_cold(ctx, 0, 0, 0);
    assert(rc == ORCHKV_OK);

    kv_block_t *blk = kv_request_get_block(ctx, 0, 0, 0);
    assert(blk->tier == TIER_NVM);
    assert(ctx->blocks_on_storage == 1);

    /* Auto-promote via get_kv_block: Storage → DRAM → GPU */
    void *k_out, *v_out;
    rc = orchkv_get_kv_block(ctx, 0, 0, 0, &k_out, &v_out);
    assert(rc == ORCHKV_OK);
    assert(blk->tier == TIER_GPU_HBM);
    assert(ctx->blocks_on_storage == 0);

    /* Verify data survived the full roundtrip */
    size_t slab_half = (size_t)TOKENS_PER_BLK * D_HEAD * 2;
    uint8_t *hbuf = (uint8_t *)malloc(slab_half);
    cudaMemcpy(hbuf, k_out, slab_half, cudaMemcpyDeviceToHost);
    assert(memcmp(hbuf, kd, slab_half) == 0);

    cudaMemcpy(hbuf, v_out, slab_half, cudaMemcpyDeviceToHost);
    assert(memcmp(hbuf, vd, slab_half) == 0);

    free(hbuf);
    free(kd);
    free(vd);
    orchkv_request_destroy(ctx);
    printf("PASS\n");
}

/* ---- Test 3: stats after storage operations ---- */
static void test_storage_stats(void)
{
    printf("  test_storage_stats ... ");

    kv_request_ctx_t *ctx = orchkv_request_create(3, 1, 1);
    assert(ctx);

    size_t head_bytes = (size_t)64 * D_HEAD * 2;
    uint8_t *kd = (uint8_t *)malloc(head_bytes);
    uint8_t *vd = (uint8_t *)malloc(head_bytes);
    fill_pattern(kd, head_bytes, 0xCC);
    fill_pattern(vd, head_bytes, 0xDD);
    assert(orchkv_prefill(ctx, 0, kd, vd, 64) == ORCHKV_OK);

    assert(orchkv_evict_to_dram(ctx, 0, 0, 0) == ORCHKV_OK);
    assert(orchkv_evict_to_storage(ctx, 0, 0, 0) == ORCHKV_OK);
    assert(orchkv_promote_from_storage(ctx, 0, 0, 0) == ORCHKV_OK);

    orchfs_tier_t *st = orchkv_orchfs_tier();
    assert(orchfs_tier_writes(st) >= 1);
    assert(orchfs_tier_reads(st) >= 1);

    free(kd);
    free(vd);
    orchkv_request_destroy(ctx);
    printf("PASS\n");
}

/* ---- Benchmark: measure 4-tier latencies ---- */
static void benchmark_4tier(void)
{
    printf("\n=== 4-Tier Latency Benchmark (LLaMA-7B scale) ===\n");

    uint32_t layers = 32, heads = 8;
    uint32_t seq = 64;
    size_t head_bytes = (size_t)seq * D_HEAD * 2;
    size_t layer_bytes = head_bytes * heads;

    kv_request_ctx_t *ctx = orchkv_request_create(100, layers, heads);
    assert(ctx);

    uint8_t *kd = (uint8_t *)malloc(layer_bytes);
    uint8_t *vd = (uint8_t *)malloc(layer_bytes);
    fill_pattern(kd, layer_bytes, 0x11);
    fill_pattern(vd, layer_bytes, 0x22);

    for (uint32_t l = 0; l < layers; l++)
        assert(orchkv_prefill(ctx, l, kd, vd, seq) == ORCHKV_OK);

    /* Benchmark: evict to DRAM (layer 0-3) */
    double t0 = now_ms();
    for (uint32_t l = 0; l < 4; l++)
        for (uint32_t h = 0; h < heads; h++)
            assert(orchkv_evict_to_dram(ctx, l, h, 0) == ORCHKV_OK);
    double evict_dram_ms = now_ms() - t0;
    uint32_t evict_dram_count = 4 * heads;

    /* Benchmark: evict DRAM → Storage */
    t0 = now_ms();
    for (uint32_t l = 0; l < 4; l++)
        for (uint32_t h = 0; h < heads; h++)
            assert(orchkv_evict_to_storage(ctx, l, h, 0) == ORCHKV_OK);
    double evict_storage_ms = now_ms() - t0;

    /* Benchmark: promote Storage → DRAM */
    t0 = now_ms();
    for (uint32_t l = 0; l < 4; l++)
        for (uint32_t h = 0; h < heads; h++)
            assert(orchkv_promote_from_storage(ctx, l, h, 0) == ORCHKV_OK);
    double promote_storage_ms = now_ms() - t0;

    /* Benchmark: promote DRAM → GPU */
    t0 = now_ms();
    for (uint32_t l = 0; l < 4; l++)
        for (uint32_t h = 0; h < heads; h++)
            assert(orchkv_promote_to_gpu(ctx, l, h, 0) == ORCHKV_OK);
    double promote_gpu_ms = now_ms() - t0;

    /* Benchmark: evict_cold (two-hop) */
    t0 = now_ms();
    for (uint32_t l = 4; l < 8; l++)
        for (uint32_t h = 0; h < heads; h++)
            assert(orchkv_evict_cold(ctx, l, h, 0) == ORCHKV_OK);
    double evict_cold_ms = now_ms() - t0;

    size_t slab_size = (size_t)TOKENS_PER_BLK * D_HEAD * 2 * 2;

    printf("  GPU→DRAM:      %6.2f ms (%u blocks, %.1f us/block)\n",
           evict_dram_ms, evict_dram_count,
           evict_dram_ms * 1000.0 / evict_dram_count);
    printf("  DRAM→Storage:  %6.2f ms (%u blocks, %.1f us/block)\n",
           evict_storage_ms, evict_dram_count,
           evict_storage_ms * 1000.0 / evict_dram_count);
    printf("  Storage→DRAM:  %6.2f ms (%u blocks, %.1f us/block)\n",
           promote_storage_ms, evict_dram_count,
           promote_storage_ms * 1000.0 / evict_dram_count);
    printf("  DRAM→GPU:      %6.2f ms (%u blocks, %.1f us/block)\n",
           promote_gpu_ms, evict_dram_count,
           promote_gpu_ms * 1000.0 / evict_dram_count);
    printf("  GPU→Storage:   %6.2f ms (%u blocks, %.1f us/block, two-hop)\n",
           evict_cold_ms, evict_dram_count,
           evict_cold_ms * 1000.0 / evict_dram_count);
    printf("  Slab size:     %zu B\n", slab_size);

    free(kd);
    free(vd);
    orchkv_request_destroy(ctx);
}

int main(void)
{
    orchkv_config_t cfg;
    orchkv_config_default(&cfg);
    cfg.gpu_pool_bytes  = (size_t)GPU_POOL_MB << 20;
    cfg.dram_pool_bytes = (size_t)DRAM_POOL_MB << 20;
    cfg.d_head          = D_HEAD;
    cfg.dtype           = DTYPE_FP16;
    cfg.tokens_per_block = TOKENS_PER_BLK;
    cfg.num_cuda_streams = 4;
    cfg.orchfs_io_workers = 4;
    cfg.max_blocks_per_head = 256;

    int rc = orchkv_init(&cfg);
    assert(rc == ORCHKV_OK);

    printf("=== test_e2e_4tier ===\n");
    test_4tier_roundtrip();
    test_evict_cold();
    test_storage_stats();

    benchmark_4tier();

    orchkv_shutdown();
    printf("\n=== ALL PASSED ===\n");
    return 0;
}
