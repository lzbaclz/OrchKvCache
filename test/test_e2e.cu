/*
 * test_e2e.cu — End-to-end correctness & benchmark for OrchKvCache Phase A
 *
 * Correctness:  ./test_e2e
 * + Benchmark:  ./test_e2e --bench
 */

extern "C" {
#include "api/orchkv_api.h"
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

/* ---- Helpers ----------------------------------------------------------- */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...)                                             \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "  FAIL [%s:%d] " fmt "\n",                   \
                    __FILE__, __LINE__, ##__VA_ARGS__);                    \
            g_fail++;                                                     \
        } else { g_pass++; }                                              \
    } while (0)

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Deterministic pattern: value at position i is i & 0xFF */
static void fill_pattern(void *buf, size_t bytes, uint8_t seed)
{
    uint8_t *p = (uint8_t *)buf;
    for (size_t i = 0; i < bytes; i++)
        p[i] = (uint8_t)((i + seed) & 0xFF);
}

static bool verify_pattern(const void *buf, size_t bytes, uint8_t seed)
{
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < bytes; i++) {
        if (p[i] != (uint8_t)((i + seed) & 0xFF))
            return false;
    }
    return true;
}

/* Small-scale config for correctness tests */
static void test_config(orchkv_config_t *cfg,
                        size_t gpu_mb, size_t dram_mb)
{
    orchkv_config_default(cfg);
    cfg->gpu_pool_bytes   = gpu_mb << 20;
    cfg->dram_pool_bytes  = dram_mb << 20;
    cfg->dram_use_pinned  = true;
    cfg->num_cuda_streams = 2;
    cfg->d_head           = 128;
    cfg->dtype            = DTYPE_FP16;
    cfg->tokens_per_block = 64;
}

/* ======================================================================== */
/*  Test 1: Init / Shutdown lifecycle                                       */
/* ======================================================================== */

static void test_init_shutdown(void)
{
    printf("--- test_init_shutdown ---\n");
    orchkv_config_t cfg;
    test_config(&cfg, 2, 2);

    CHECK(orchkv_init(&cfg) == ORCHKV_OK, "init");
    CHECK(orchkv_is_initialized(), "is_initialized");
    CHECK(orchkv_init(&cfg) == ORCHKV_ERR_ALREADY, "double init rejected");
    CHECK(orchkv_shutdown() == ORCHKV_OK, "shutdown");
    CHECK(!orchkv_is_initialized(), "not initialized after shutdown");
    CHECK(orchkv_shutdown() == ORCHKV_ERR_INIT, "double shutdown rejected");
    printf("  PASS\n\n");
}

/* ======================================================================== */
/*  Test 2: Prefill and verify data integrity                               */
/* ======================================================================== */

static void test_prefill_verify(void)
{
    printf("--- test_prefill_verify ---\n");

    orchkv_config_t cfg;
    test_config(&cfg, 4, 4);
    int rc = orchkv_init(&cfg);
    CHECK(rc == ORCHKV_OK, "init");

    const uint32_t N_LAYERS  = 2;
    const uint32_t N_HEADS   = 2;
    const uint32_t SEQ_LEN   = 80;
    const uint32_t D_HEAD    = cfg.d_head;
    const uint32_t TPB       = cfg.tokens_per_block;

    kv_request_ctx_t *req = orchkv_request_create(1, N_LAYERS, N_HEADS);
    CHECK(req != NULL, "request create");

    size_t per_head_bytes = (size_t)SEQ_LEN * D_HEAD * dtype_size(cfg.dtype);
    size_t k_total = (size_t)N_HEADS * per_head_bytes;
    size_t slab_kh = kv_block_data_bytes(TPB, D_HEAD, cfg.dtype) / 2;

    uint8_t *k_host, *v_host;
    cudaMallocHost((void **)&k_host, k_total);
    cudaMallocHost((void **)&v_host, k_total);
    fill_pattern(k_host, k_total, 0xA0);
    fill_pattern(v_host, k_total, 0x50);

    for (uint32_t l = 0; l < N_LAYERS; l++) {
        rc = orchkv_prefill(req, l, k_host, v_host, SEQ_LEN);
        CHECK(rc == ORCHKV_OK, "prefill layer %u", l);
    }

    CHECK(req->seq_len == SEQ_LEN, "seq_len=%u expected %u",
          req->seq_len, SEQ_LEN);

    uint32_t blocks_per_head = (SEQ_LEN + TPB - 1) / TPB;
    CHECK(req->total_blocks == (uint64_t)N_LAYERS * N_HEADS * blocks_per_head,
          "total_blocks=%lu", (unsigned long)req->total_blocks);

    uint8_t *read_buf = (uint8_t *)malloc(slab_kh);

    for (uint32_t l = 0; l < N_LAYERS; l++) {
        for (uint32_t h = 0; h < N_HEADS; h++) {
            for (uint32_t bi = 0; bi < blocks_per_head; bi++) {
                void *k_out, *v_out;
                rc = orchkv_get_kv_block(req, l, h, bi, &k_out, &v_out);
                CHECK(rc == ORCHKV_OK, "get_kv l=%u h=%u bi=%u", l, h, bi);

                kv_block_t *blk = kv_request_get_block(req, l, h, bi);
                size_t tok_bytes = (size_t)blk->token_count * D_HEAD
                                   * dtype_size(cfg.dtype);
                size_t tok_off = (size_t)blk->token_start * D_HEAD
                                 * dtype_size(cfg.dtype);

                cudaMemcpy(read_buf, k_out, tok_bytes, cudaMemcpyDeviceToHost);
                const uint8_t *expected_k = k_host + h * per_head_bytes + tok_off;
                CHECK(memcmp(read_buf, expected_k, tok_bytes) == 0,
                      "K verify l=%u h=%u bi=%u", l, h, bi);

                cudaMemcpy(read_buf, v_out, tok_bytes, cudaMemcpyDeviceToHost);
                const uint8_t *expected_v = v_host + h * per_head_bytes + tok_off;
                CHECK(memcmp(read_buf, expected_v, tok_bytes) == 0,
                      "V verify l=%u h=%u bi=%u", l, h, bi);
            }
        }
    }

    free(read_buf);
    orchkv_request_destroy(req);
    cudaFreeHost(k_host);
    cudaFreeHost(v_host);
    orchkv_shutdown();
    printf("  PASS\n\n");
}

/* ======================================================================== */
/*  Test 3: Evict to DRAM, verify, promote back, verify                     */
/* ======================================================================== */

static void test_evict_promote(void)
{
    printf("--- test_evict_promote ---\n");

    orchkv_config_t cfg;
    test_config(&cfg, 4, 4);
    orchkv_init(&cfg);

    const uint32_t N_LAYERS = 1, N_HEADS = 2, SEQ_LEN = 64;
    const uint32_t D_HEAD = cfg.d_head, TPB = cfg.tokens_per_block;

    kv_request_ctx_t *req = orchkv_request_create(2, N_LAYERS, N_HEADS);

    size_t per_head_bytes = (size_t)SEQ_LEN * D_HEAD * dtype_size(cfg.dtype);
    size_t k_total = N_HEADS * per_head_bytes;
    size_t slab_kh = kv_block_data_bytes(TPB, D_HEAD, cfg.dtype) / 2;

    uint8_t *k_host, *v_host;
    cudaMallocHost((void **)&k_host, k_total);
    cudaMallocHost((void **)&v_host, k_total);
    fill_pattern(k_host, k_total, 0x10);
    fill_pattern(v_host, k_total, 0x20);

    orchkv_prefill(req, 0, k_host, v_host, SEQ_LEN);

    int rc = orchkv_evict_to_dram(req, 0, 0, 0);
    CHECK(rc == ORCHKV_OK, "evict");

    kv_block_t *blk = kv_request_get_block(req, 0, 0, 0);
    CHECK(blk->tier == TIER_HOST_DRAM, "block on DRAM after evict");
    CHECK(blk->state == KV_STATE_WARM, "state WARM after evict");

    /* Verify data on DRAM (host pointer, can read directly) */
    size_t tok_bytes = (size_t)blk->token_count * D_HEAD * dtype_size(cfg.dtype);
    const uint8_t *expected_k = k_host;
    const uint8_t *expected_v = v_host;
    CHECK(memcmp(blk->data_ptr, expected_k, tok_bytes) == 0,
          "K verify on DRAM");
    CHECK(memcmp((char *)blk->data_ptr + slab_kh, expected_v, tok_bytes) == 0,
          "V verify on DRAM");

    /* Promote back */
    rc = orchkv_promote_to_gpu(req, 0, 0, 0);
    CHECK(rc == ORCHKV_OK, "promote");
    CHECK(blk->tier == TIER_GPU_HBM, "block on GPU after promote");
    CHECK(blk->state == KV_STATE_HOT, "state HOT after promote");

    /* Verify data on GPU by reading back */
    uint8_t *read_buf = (uint8_t *)malloc(tok_bytes);
    cudaMemcpy(read_buf, blk->data_ptr, tok_bytes, cudaMemcpyDeviceToHost);
    CHECK(memcmp(read_buf, expected_k, tok_bytes) == 0,
          "K verify after promote");

    cudaMemcpy(read_buf, (char *)blk->data_ptr + slab_kh, tok_bytes,
               cudaMemcpyDeviceToHost);
    CHECK(memcmp(read_buf, expected_v, tok_bytes) == 0,
          "V verify after promote");

    free(read_buf);
    orchkv_request_destroy(req);
    cudaFreeHost(k_host);
    cudaFreeHost(v_host);
    orchkv_shutdown();
    printf("  PASS\n\n");
}

/* ======================================================================== */
/*  Test 4: Append tokens (decode) and verify                               */
/* ======================================================================== */

static void test_append_verify(void)
{
    printf("--- test_append_verify ---\n");

    orchkv_config_t cfg;
    test_config(&cfg, 4, 4);
    orchkv_init(&cfg);

    const uint32_t N_LAYERS = 1, N_HEADS = 2;
    const uint32_t D_HEAD = cfg.d_head, TPB = cfg.tokens_per_block;
    const uint32_t PREFILL_LEN = 60;
    const uint32_t DECODE_STEPS = 10;

    kv_request_ctx_t *req = orchkv_request_create(3, N_LAYERS, N_HEADS);

    size_t per_head_bytes = (size_t)PREFILL_LEN * D_HEAD * dtype_size(cfg.dtype);
    size_t k_total = N_HEADS * per_head_bytes;
    size_t slab_kh = kv_block_data_bytes(TPB, D_HEAD, cfg.dtype) / 2;
    size_t head_stride = D_HEAD * dtype_size(cfg.dtype);

    uint8_t *k_host, *v_host;
    cudaMallocHost((void **)&k_host, k_total);
    cudaMallocHost((void **)&v_host, k_total);
    fill_pattern(k_host, k_total, 0x30);
    fill_pattern(v_host, k_total, 0x40);

    orchkv_prefill(req, 0, k_host, v_host, PREFILL_LEN);
    CHECK(req->seq_len == PREFILL_LEN, "seq_len after prefill");

    /* After 60 tokens: 1 block per head, token_count=60, 4 slots remaining */
    uint32_t blocks_per_head = (PREFILL_LEN + TPB - 1) / TPB;
    CHECK(blocks_per_head == 1, "1 block per head");

    /* Append DECODE_STEPS tokens */
    uint8_t *k_tok, *v_tok;
    cudaMallocHost((void **)&k_tok, N_HEADS * head_stride);
    cudaMallocHost((void **)&v_tok, N_HEADS * head_stride);

    for (uint32_t step = 0; step < DECODE_STEPS; step++) {
        fill_pattern(k_tok, N_HEADS * head_stride, (uint8_t)(0x60 + step));
        fill_pattern(v_tok, N_HEADS * head_stride, (uint8_t)(0x70 + step));

        int rc = orchkv_append_token(req, 0, k_tok, v_tok);
        CHECK(rc == ORCHKV_OK, "append step %u", step);
    }

    CHECK(req->seq_len == PREFILL_LEN + DECODE_STEPS,
          "seq_len=%u expected %u", req->seq_len, PREFILL_LEN + DECODE_STEPS);

    /* 60 + 10 = 70 tokens, still 2 blocks (block 0: 64 tokens, block 1: 6 tokens) */
    kv_head_vec_t *hv0 = kv_request_hvec(req, 0, 0);
    if (PREFILL_LEN + DECODE_STEPS > TPB) {
        CHECK(hv0->count == 2, "2 blocks after append past boundary");
    }

    /* Verify last appended token (step 9) for head 0 */
    kv_block_t *last_blk = hv0->items[hv0->count - 1];
    uint32_t last_slot = last_blk->token_count - 1;

    uint8_t *read_k = (uint8_t *)malloc(head_stride);
    uint8_t *read_v = (uint8_t *)malloc(head_stride);

    cudaMemcpy(read_k,
               (char *)last_blk->data_ptr + last_slot * head_stride,
               head_stride, cudaMemcpyDeviceToHost);

    cudaMemcpy(read_v,
               (char *)last_blk->data_ptr + slab_kh + last_slot * head_stride,
               head_stride, cudaMemcpyDeviceToHost);

    uint8_t *expect_k = (uint8_t *)malloc(head_stride);
    uint8_t *expect_v = (uint8_t *)malloc(head_stride);
    fill_pattern(expect_k, head_stride, 0x60 + DECODE_STEPS - 1);
    fill_pattern(expect_v, head_stride, 0x70 + DECODE_STEPS - 1);

    CHECK(memcmp(read_k, expect_k, head_stride) == 0,
          "K verify last decoded token");
    CHECK(memcmp(read_v, expect_v, head_stride) == 0,
          "V verify last decoded token");

    free(read_k); free(read_v);
    free(expect_k); free(expect_v);
    cudaFreeHost(k_tok); cudaFreeHost(v_tok);
    cudaFreeHost(k_host); cudaFreeHost(v_host);
    orchkv_request_destroy(req);
    orchkv_shutdown();
    printf("  PASS\n\n");
}

/* ======================================================================== */
/*  Test 5: Stats                                                           */
/* ======================================================================== */

static void test_stats(void)
{
    printf("--- test_stats ---\n");

    orchkv_config_t cfg;
    test_config(&cfg, 4, 4);
    orchkv_init(&cfg);

    kv_request_ctx_t *req = orchkv_request_create(4, 1, 2);
    orchkv_prefill(req, 0, NULL, NULL, 64);

    orchkv_stats_t st;
    orchkv_get_stats(&st);

    CHECK(st.gpu_slabs_used == 2, "gpu_slabs_used=%u", st.gpu_slabs_used);
    CHECK(st.dram_slabs_used == 0, "dram_slabs_used=%u", st.dram_slabs_used);
    CHECK(st.total_blocks == 2, "total_blocks=%lu", (unsigned long)st.total_blocks);

    orchkv_evict_to_dram(req, 0, 0, 0);
    orchkv_get_stats(&st);

    CHECK(st.gpu_slabs_used == 1, "after evict gpu=%u", st.gpu_slabs_used);
    CHECK(st.dram_slabs_used == 1, "after evict dram=%u", st.dram_slabs_used);
    CHECK(st.transfers_d2h == 1, "transfers_d2h=%lu", (unsigned long)st.transfers_d2h);

    orchkv_promote_to_gpu(req, 0, 0, 0);
    orchkv_get_stats(&st);

    CHECK(st.gpu_slabs_used == 2, "after promote gpu=%u", st.gpu_slabs_used);
    CHECK(st.dram_slabs_used == 0, "after promote dram=%u", st.dram_slabs_used);

    orchkv_request_destroy(req);
    orchkv_get_stats(&st);
    CHECK(st.total_blocks == 0, "after destroy total=%lu", (unsigned long)st.total_blocks);
    CHECK(st.gpu_slabs_used == 0, "after destroy gpu=%u", st.gpu_slabs_used);

    orchkv_shutdown();
    printf("  PASS\n\n");
}

/* ======================================================================== */
/*  Benchmark: simulate LLaMA-7B–scale decode loop                          */
/* ======================================================================== */

static void bench_decode_loop(void)
{
    printf("--- bench_decode_loop (LLaMA-7B scale) ---\n");

    orchkv_config_t cfg;
    orchkv_config_default(&cfg);
    cfg.gpu_pool_bytes   = (size_t)128 << 20;
    cfg.dram_pool_bytes  = (size_t)128 << 20;
    cfg.dram_use_pinned  = true;
    cfg.num_cuda_streams = 4;
    cfg.d_head           = 128;
    cfg.dtype            = DTYPE_FP16;
    cfg.tokens_per_block = 64;

    const uint32_t N_LAYERS    = 32;
    const uint32_t N_HEADS     = 8;    /* GQA heads */
    const uint32_t PREFILL_LEN = 128;
    const uint32_t DECODE_STEPS = 100;

    int rc = orchkv_init(&cfg);
    if (rc != ORCHKV_OK) {
        printf("  SKIP: init failed (%d)\n\n", rc);
        return;
    }

    kv_request_ctx_t *req = orchkv_request_create(100, N_LAYERS, N_HEADS);

    size_t per_head_bytes = (size_t)PREFILL_LEN * cfg.d_head * dtype_size(cfg.dtype);
    size_t k_total = N_HEADS * per_head_bytes;
    size_t head_stride = cfg.d_head * dtype_size(cfg.dtype);

    uint8_t *k_buf, *v_buf;
    cudaMallocHost((void **)&k_buf, k_total);
    cudaMallocHost((void **)&v_buf, k_total);
    memset(k_buf, 0x55, k_total);
    memset(v_buf, 0xAA, k_total);

    uint8_t *k_tok, *v_tok;
    cudaMallocHost((void **)&k_tok, N_HEADS * head_stride);
    cudaMallocHost((void **)&v_tok, N_HEADS * head_stride);
    memset(k_tok, 0x33, N_HEADS * head_stride);
    memset(v_tok, 0xCC, N_HEADS * head_stride);

    /* ---- Prefill ---- */
    double t0 = now_us();
    for (uint32_t l = 0; l < N_LAYERS; l++) {
        rc = orchkv_prefill(req, l, k_buf, v_buf, PREFILL_LEN);
        assert(rc == ORCHKV_OK);
    }
    double prefill_us = now_us() - t0;

    printf("  prefill %u tok × %u layers: %.1f ms  (%.1f us/layer)\n",
           PREFILL_LEN, N_LAYERS, prefill_us / 1e3,
           prefill_us / N_LAYERS);

    /* ---- Decode ---- */
    double decode_total_us = 0;
    for (uint32_t step = 0; step < DECODE_STEPS; step++) {
        double t1 = now_us();
        for (uint32_t l = 0; l < N_LAYERS; l++) {
            rc = orchkv_append_token(req, l, k_tok, v_tok);
            assert(rc == ORCHKV_OK);
        }
        decode_total_us += now_us() - t1;
    }
    printf("  decode %u steps × %u layers: %.1f ms  (%.1f us/step, %.2f us/layer)\n",
           DECODE_STEPS, N_LAYERS, decode_total_us / 1e3,
           decode_total_us / DECODE_STEPS,
           decode_total_us / DECODE_STEPS / N_LAYERS);

    /* ---- Evict half of layer-0 blocks to DRAM ---- */
    kv_head_vec_t *hv00 = kv_request_hvec(req, 0, 0);
    uint32_t evict_count = hv00->count / 2;
    if (evict_count == 0) evict_count = 1;

    double evict_total_us = 0;
    uint32_t evictions = 0;
    for (uint32_t h = 0; h < N_HEADS; h++) {
        for (uint32_t bi = 0; bi < evict_count; bi++) {
            double te = now_us();
            rc = orchkv_evict_to_dram(req, 0, h, bi);
            evict_total_us += now_us() - te;
            if (rc == ORCHKV_OK) evictions++;
        }
    }
    printf("  evict %u blocks: %.1f ms  (%.1f us/block)\n",
           evictions, evict_total_us / 1e3,
           evictions ? evict_total_us / evictions : 0.0);

    /* ---- Promote them back ---- */
    double promote_total_us = 0;
    uint32_t promotions = 0;
    for (uint32_t h = 0; h < N_HEADS; h++) {
        for (uint32_t bi = 0; bi < evict_count; bi++) {
            double tp = now_us();
            rc = orchkv_promote_to_gpu(req, 0, h, bi);
            promote_total_us += now_us() - tp;
            if (rc == ORCHKV_OK) promotions++;
        }
    }
    printf("  promote %u blocks: %.1f ms  (%.1f us/block)\n",
           promotions, promote_total_us / 1e3,
           promotions ? promote_total_us / promotions : 0.0);

    /* ---- Final stats ---- */
    orchkv_stats_t st;
    orchkv_get_stats(&st);
    printf("  final stats: gpu_used=%u/%u  dram_used=%u/%u  "
           "d2h=%lu h2d=%lu\n",
           st.gpu_slabs_used, st.gpu_slabs_total,
           st.dram_slabs_used, st.dram_slabs_total,
           (unsigned long)st.transfers_d2h,
           (unsigned long)st.transfers_h2d);

    /* ---- JSON output for collection ---- */
    printf("\n  === BENCH_JSON_START ===\n");
    printf("  {\n");
    printf("    \"model\": \"LLaMA-7B-like\",\n");
    printf("    \"n_layers\": %u,\n", N_LAYERS);
    printf("    \"n_kv_heads\": %u,\n", N_HEADS);
    printf("    \"d_head\": %u,\n", cfg.d_head);
    printf("    \"dtype\": \"fp16\",\n");
    printf("    \"tokens_per_block\": %u,\n", cfg.tokens_per_block);
    printf("    \"slab_size_bytes\": %zu,\n",
           kv_block_data_bytes(cfg.tokens_per_block, cfg.d_head, cfg.dtype));
    printf("    \"prefill_tokens\": %u,\n", PREFILL_LEN);
    printf("    \"prefill_total_ms\": %.3f,\n", prefill_us / 1e3);
    printf("    \"prefill_per_layer_us\": %.1f,\n", prefill_us / N_LAYERS);
    printf("    \"decode_steps\": %u,\n", DECODE_STEPS);
    printf("    \"decode_total_ms\": %.3f,\n", decode_total_us / 1e3);
    printf("    \"decode_per_step_us\": %.1f,\n", decode_total_us / DECODE_STEPS);
    printf("    \"decode_per_layer_us\": %.2f,\n",
           decode_total_us / DECODE_STEPS / N_LAYERS);
    printf("    \"evict_blocks\": %u,\n", evictions);
    printf("    \"evict_total_ms\": %.3f,\n", evict_total_us / 1e3);
    printf("    \"evict_per_block_us\": %.1f,\n",
           evictions ? evict_total_us / evictions : 0.0);
    printf("    \"promote_blocks\": %u,\n", promotions);
    printf("    \"promote_total_ms\": %.3f,\n", promote_total_us / 1e3);
    printf("    \"promote_per_block_us\": %.1f,\n",
           promotions ? promote_total_us / promotions : 0.0);
    printf("    \"gpu_pool_mb\": %zu,\n", cfg.gpu_pool_bytes >> 20);
    printf("    \"dram_pool_mb\": %zu,\n", cfg.dram_pool_bytes >> 20);
    printf("    \"total_blocks\": %lu,\n", (unsigned long)st.total_blocks);
    printf("    \"transfers_d2h\": %lu,\n", (unsigned long)st.transfers_d2h);
    printf("    \"transfers_h2d\": %lu,\n", (unsigned long)st.transfers_h2d);
    printf("    \"bytes_d2h\": %lu,\n", (unsigned long)st.bytes_d2h);
    printf("    \"bytes_h2d\": %lu\n", (unsigned long)st.bytes_h2d);
    printf("  }\n");
    printf("  === BENCH_JSON_END ===\n\n");

    orchkv_request_destroy(req);
    cudaFreeHost(k_buf); cudaFreeHost(v_buf);
    cudaFreeHost(k_tok); cudaFreeHost(v_tok);
    orchkv_shutdown();
}

/* ======================================================================== */

int main(int argc, char **argv)
{
    bool run_bench = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0)
            run_bench = true;
    }

    printf("=== OrchKvCache E2E Test Suite ===\n\n");

    kv_block_reset_id_counter();

    test_init_shutdown();
    test_prefill_verify();
    test_evict_promote();
    test_append_verify();
    test_stats();

    if (run_bench)
        bench_decode_loop();

    printf("=== Results: %d passed, %d failed ===\n",
           g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
