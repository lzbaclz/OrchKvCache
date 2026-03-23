/*
 * C11: E2E Auto-Scheduling Test + Benchmark
 *
 * Uses the full tiered_manager (C8) with real CUDA transfers to verify
 * automatic eviction, prefetch dispatch, data integrity, and multi-request
 * isolation.  Outputs benchmark results as JSON.
 */
extern "C" {
#include "scheduler/tiered_manager.h"
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

/* ---- Configuration ---- */
#define BLK_SZ       4096     /* bytes per block */
#define N_GPU_BLOCKS 16
#define N_HOT        6
#define N_COLD       (N_GPU_BLOCKS - N_HOT)
#define BATCH_SZ     4
#define MAX_BLOCKS   64

/* ---- Timing ---- */
static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ---- Pattern helpers ---- */
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

/* ========================================================================
 *  CUDA transfer function + allocator
 * ======================================================================== */

static int cuda_transfer(kv_block_t *blk, void *dst,
                         size_t size, MigrateOp op, void *ctx)
{
    (void)ctx;
    if (!blk->data_ptr || !dst) return ORCHKV_ERR_INVALID;

    switch (op) {
    case MIGRATE_DEMOTE_GPU2DRAM:
        return (cudaMemcpy(dst, blk->data_ptr, size,
                           cudaMemcpyDeviceToHost) == cudaSuccess)
                   ? ORCHKV_OK : ORCHKV_ERR_CUDA;
    case MIGRATE_PROMOTE_DRAM2GPU:
        return (cudaMemcpy(dst, blk->data_ptr, size,
                           cudaMemcpyHostToDevice) == cudaSuccess)
                   ? ORCHKV_OK : ORCHKV_ERR_CUDA;
    case MIGRATE_DEMOTE_DRAM2STOR:
    case MIGRATE_PROMOTE_STOR2DRAM:
        memcpy(dst, blk->data_ptr, size);
        return ORCHKV_OK;
    default:
        return ORCHKV_ERR_INVALID;
    }
}

static void *cuda_alloc(size_t size, StorageTier tier, void *ctx)
{
    (void)ctx;
    if (tier == TIER_GPU_HBM) {
        void *p = NULL;
        if (cudaMalloc(&p, size) != cudaSuccess) return NULL;
        return p;
    }
    return calloc(1, size);
}

static void cuda_free(void *ptr, StorageTier tier, void *ctx)
{
    (void)ctx;
    if (!ptr) return;
    if (tier == TIER_GPU_HBM)
        cudaFree(ptr);
    else
        free(ptr);
}

/* ========================================================================
 *  Block management
 * ======================================================================== */

static kv_block_t g_blks[MAX_BLOCKS];
static uint8_t    g_patterns[MAX_BLOCKS];  /* per-block seed for verification */

static void init_gpu_block(int idx, uint64_t id, uint8_t seed)
{
    memset(&g_blks[idx], 0, sizeof(kv_block_t));
    g_blks[idx].block_id    = id;
    g_blks[idx].request_id  = 1;
    g_blks[idx].layer_id    = 0;
    g_blks[idx].head_id     = 0;
    g_blks[idx].token_start = (uint32_t)(id * 64);
    g_blks[idx].token_count = 64;
    g_blks[idx].tier        = TIER_GPU_HBM;
    g_blks[idx].state       = KV_STATE_HOT;
    g_blks[idx].flags       = KV_FLAG_NONE;
    pthread_rwlock_init(&g_blks[idx].lock, NULL);
    g_patterns[idx] = seed;

    void *gpu_buf = NULL;
    assert(cudaMalloc(&gpu_buf, BLK_SZ) == cudaSuccess);

    uint8_t *host_tmp = (uint8_t *)malloc(BLK_SZ);
    fill_pattern(host_tmp, BLK_SZ, seed);
    assert(cudaMemcpy(gpu_buf, host_tmp, BLK_SZ,
                      cudaMemcpyHostToDevice) == cudaSuccess);
    free(host_tmp);

    g_blks[idx].data_ptr = gpu_buf;
}

static void free_block(int idx)
{
    if (!g_blks[idx].data_ptr) return;
    if (g_blks[idx].tier == TIER_GPU_HBM)
        cudaFree(g_blks[idx].data_ptr);
    else
        free(g_blks[idx].data_ptr);
    g_blks[idx].data_ptr = NULL;
}

/* ========================================================================
 *  Manager config helper
 * ======================================================================== */

static tm_config_t make_cfg(void)
{
    tm_config_t cfg;
    tm_config_default(&cfg);
    cfg.tracker_capacity   = 256;
    cfg.ema_lambda         = 0.9f;
    cfg.hcc_params.alpha   = 1.0f;
    cfg.hcc_params.beta    = 0.0f;
    cfg.hcc_params.gamma   = 0.0f;
    cfg.athresh_params.cooldown_sec = 0.0;
    cfg.athresh_params.gpu_hwm  = 0.85f;
    cfg.athresh_params.gpu_lwm  = 0.60f;
    cfg.athresh_params.dram_hwm = 0.85f;
    cfg.athresh_params.dram_lwm = 0.60f;
    cfg.w_heat              = 0.7f;
    cfg.w_lru               = 0.3f;
    cfg.tokens_per_block    = 64;
    cfg.prefetch_budget     = 8;
    cfg.threshold_to_gpu    = 0.3f;
    cfg.threshold_to_dram   = 0.1f;
    cfg.transfer_fn         = cuda_transfer;
    cfg.alloc_fn            = cuda_alloc;
    cfg.free_fn             = cuda_free;
    cfg.block_data_size     = BLK_SZ;
    cfg.schedule_interval_us = 1000;
    cfg.demote_batch_size   = BATCH_SZ;
    cfg.prefetch_batch_size = BATCH_SZ;
    cfg.max_blocks          = MAX_BLOCKS;
    cfg.auto_schedule       = false;
    return cfg;
}

/* ========================================================================
 *  Test 1: GPU overload → cold blocks automatically demoted to DRAM
 * ======================================================================== */

static void test_auto_evict_gpu(void)
{
    printf("  test_auto_evict_gpu ... "); fflush(stdout);

    tiered_manager_t m;
    tm_config_t cfg = make_cfg();
    assert(tm_init(&m, &cfg) == ORCHKV_OK);

    for (int i = 0; i < N_GPU_BLOCKS; i++) {
        init_gpu_block(i, (uint64_t)(100 + i), (uint8_t)(0x10 + i));
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }

    /* Hot blocks: 100..105, Cold: 106..115 */
    for (int step = 0; step < 50; step++) {
        for (int i = 0; i < N_HOT; i++) {
            tm_notify_attn(&m, (uint64_t)(100 + i), 0.9f);
            tm_notify_access(&m, &g_blks[i]);
        }
        tm_step_done(&m);
    }

    tm_set_usage(&m, 0.95f, 0.5f);
    tm_schedule_once(&m);

    /* Hot blocks should remain on GPU */
    for (int i = 0; i < N_HOT; i++)
        assert(g_blks[i].tier == TIER_GPU_HBM);

    /* At least some cold blocks demoted to DRAM */
    int dram_count = 0;
    for (int i = N_HOT; i < N_GPU_BLOCKS; i++)
        if (g_blks[i].tier == TIER_HOST_DRAM) dram_count++;
    assert(dram_count > 0);

    /* Verify DRAM data integrity */
    for (int i = N_HOT; i < N_GPU_BLOCKS; i++) {
        if (g_blks[i].tier != TIER_HOST_DRAM) continue;
        assert(check_pattern(g_blks[i].data_ptr, BLK_SZ, g_patterns[i]));
    }

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    assert(stats.gpu_demotes > 0);

    for (int i = 0; i < N_GPU_BLOCKS; i++) free_block(i);
    tm_destroy(&m);
    printf("PASS (demoted %d blocks)\n", dram_count);
}

/* ========================================================================
 *  Test 2: DRAM overload → blocks demoted to Storage (simulated)
 * ======================================================================== */

static void test_auto_evict_dram(void)
{
    printf("  test_auto_evict_dram ... "); fflush(stdout);

    tiered_manager_t m;
    tm_config_t cfg = make_cfg();
    assert(tm_init(&m, &cfg) == ORCHKV_OK);

    /* Start blocks directly on DRAM (simulated post-GPU-eviction) */
    for (int i = 0; i < 8; i++) {
        memset(&g_blks[i], 0, sizeof(kv_block_t));
        g_blks[i].block_id    = (uint64_t)(200 + i);
        g_blks[i].request_id  = 2;
        g_blks[i].token_start = (uint32_t)(i * 64);
        g_blks[i].token_count = 64;
        g_blks[i].tier        = TIER_HOST_DRAM;
        g_blks[i].state       = KV_STATE_WARM;
        g_blks[i].flags       = KV_FLAG_NONE;
        g_patterns[i]         = (uint8_t)(0x30 + i);
        g_blks[i].data_ptr    = calloc(1, BLK_SZ);
        fill_pattern(g_blks[i].data_ptr, BLK_SZ, g_patterns[i]);
        pthread_rwlock_init(&g_blks[i].lock, NULL);
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }

    for (int step = 0; step < 30; step++)
        tm_step_done(&m);

    tm_set_usage(&m, 0.5f, 0.95f);
    tm_schedule_once(&m);

    int storage_count = 0;
    for (int i = 0; i < 8; i++)
        if (g_blks[i].tier == TIER_NVM) storage_count++;
    assert(storage_count > 0);

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    assert(stats.dram_demotes > 0);

    for (int i = 0; i < 8; i++) free_block(i);
    tm_destroy(&m);
    printf("PASS (demoted %d to storage)\n", storage_count);
}

/* ========================================================================
 *  Test 3: High-attention DRAM blocks flagged for prefetch
 * ======================================================================== */

static void test_auto_prefetch(void)
{
    printf("  test_auto_prefetch ... "); fflush(stdout);

    tiered_manager_t m;
    tm_config_t cfg = make_cfg();
    assert(tm_init(&m, &cfg) == ORCHKV_OK);

    /* 4 DRAM blocks with high attention */
    for (int i = 0; i < 4; i++) {
        memset(&g_blks[i], 0, sizeof(kv_block_t));
        g_blks[i].block_id    = (uint64_t)(300 + i);
        g_blks[i].request_id  = 3;
        g_blks[i].token_start = (uint32_t)(i * 64);
        g_blks[i].token_count = 64;
        g_blks[i].tier        = TIER_HOST_DRAM;
        g_blks[i].state       = KV_STATE_WARM;
        g_blks[i].flags       = KV_FLAG_NONE;
        g_blks[i].data_ptr    = calloc(1, BLK_SZ);
        pthread_rwlock_init(&g_blks[i].lock, NULL);
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }

    for (int step = 0; step < 60; step++) {
        for (int i = 0; i < 4; i++)
            tm_notify_attn(&m, (uint64_t)(300 + i), 0.8f);
        tm_step_done(&m);
    }

    tm_set_usage(&m, 0.5f, 0.5f);
    tm_schedule_once(&m);

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    assert(stats.prefetches_dispatched > 0);

    for (int i = 0; i < 4; i++) free_block(i);
    tm_destroy(&m);
    printf("PASS (dispatched %llu prefetches)\n",
           (unsigned long long)stats.prefetches_dispatched);
}

/* ========================================================================
 *  Test 4: Data integrity after GPU → DRAM eviction → DRAM → GPU promotion
 * ======================================================================== */

static void test_data_integrity(void)
{
    printf("  test_data_integrity ... "); fflush(stdout);

    tiered_manager_t m;
    tm_config_t cfg = make_cfg();
    cfg.demote_batch_size = 8;
    assert(tm_init(&m, &cfg) == ORCHKV_OK);

    /* 8 GPU blocks with unique patterns */
    for (int i = 0; i < 8; i++) {
        init_gpu_block(i, (uint64_t)(400 + i), (uint8_t)(0x40 + i));
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }

    /* All cold (no attention) */
    for (int step = 0; step < 30; step++)
        tm_step_done(&m);

    /* Demote all to DRAM */
    tm_set_usage(&m, 0.95f, 0.5f);
    tm_schedule_once(&m);

    int dram_count = 0;
    for (int i = 0; i < 8; i++)
        if (g_blks[i].tier == TIER_HOST_DRAM) dram_count++;

    /* Verify DRAM data matches original patterns */
    for (int i = 0; i < 8; i++) {
        if (g_blks[i].tier != TIER_HOST_DRAM) continue;
        assert(check_pattern(g_blks[i].data_ptr, BLK_SZ, g_patterns[i]));
    }

    /* Promote DRAM blocks back to GPU */
    for (int i = 0; i < 8; i++) {
        if (g_blks[i].tier != TIER_HOST_DRAM) continue;

        void *gpu_buf = NULL;
        assert(cudaMalloc(&gpu_buf, BLK_SZ) == cudaSuccess);

        void *old_ptr = g_blks[i].data_ptr;
        int rc = mig_execute_one(&m.migration, &g_blks[i], TIER_GPU_HBM,
                                 gpu_buf, NULL, BLK_SZ);
        assert(rc == ORCHKV_OK);
        assert(g_blks[i].tier == TIER_GPU_HBM);
        free(old_ptr);
    }

    /* Verify GPU data matches original patterns (cudaMemcpy D2H) */
    uint8_t *verify_buf = (uint8_t *)malloc(BLK_SZ);
    for (int i = 0; i < 8; i++) {
        if (g_blks[i].tier != TIER_GPU_HBM) continue;
        assert(cudaMemcpy(verify_buf, g_blks[i].data_ptr, BLK_SZ,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(check_pattern(verify_buf, BLK_SZ, g_patterns[i]));
    }
    free(verify_buf);

    for (int i = 0; i < 8; i++) free_block(i);
    tm_destroy(&m);
    printf("PASS (%d blocks roundtripped)\n", dram_count);
}

/* ========================================================================
 *  Test 5: Multi-request blocks don't interfere
 * ======================================================================== */

static void test_multi_request(void)
{
    printf("  test_multi_request ... "); fflush(stdout);

    tiered_manager_t m;
    tm_config_t cfg = make_cfg();
    cfg.demote_batch_size = 4;
    assert(tm_init(&m, &cfg) == ORCHKV_OK);

    /* Request A: blocks 0-3, pattern 0xAA+i */
    for (int i = 0; i < 4; i++) {
        init_gpu_block(i, (uint64_t)(500 + i), (uint8_t)(0xA0 + i));
        g_blks[i].request_id = 10;
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }
    /* Request B: blocks 4-7, pattern 0xBB+i */
    for (int i = 4; i < 8; i++) {
        init_gpu_block(i, (uint64_t)(500 + i), (uint8_t)(0xB0 + i));
        g_blks[i].request_id = 20;
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }

    /* Request A is hot, Request B is cold */
    for (int step = 0; step < 50; step++) {
        for (int i = 0; i < 4; i++) {
            tm_notify_attn(&m, (uint64_t)(500 + i), 0.9f);
            tm_notify_access(&m, &g_blks[i]);
        }
        tm_step_done(&m);
    }

    tm_set_usage(&m, 0.95f, 0.5f);
    tm_schedule_once(&m);

    /* Request A blocks should stay on GPU */
    for (int i = 0; i < 4; i++)
        assert(g_blks[i].tier == TIER_GPU_HBM);

    /* Request B blocks should be on DRAM with correct data */
    int b_dram = 0;
    for (int i = 4; i < 8; i++) {
        if (g_blks[i].tier != TIER_HOST_DRAM) continue;
        b_dram++;
        assert(check_pattern(g_blks[i].data_ptr, BLK_SZ, g_patterns[i]));
    }
    assert(b_dram > 0);

    /* Verify Request A GPU data is still intact */
    uint8_t *verify = (uint8_t *)malloc(BLK_SZ);
    for (int i = 0; i < 4; i++) {
        assert(cudaMemcpy(verify, g_blks[i].data_ptr, BLK_SZ,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(check_pattern(verify, BLK_SZ, g_patterns[i]));
    }
    free(verify);

    for (int i = 0; i < 8; i++) free_block(i);
    tm_destroy(&m);
    printf("PASS (A on GPU, B demoted %d)\n", b_dram);
}

/* ========================================================================
 *  Benchmark: simulated decode loop with auto-scheduling
 * ======================================================================== */

static void benchmark_auto_schedule(void)
{
    printf("\n=== Auto-Schedule Benchmark ===\n");

    tiered_manager_t m;
    tm_config_t cfg = make_cfg();
    cfg.demote_batch_size   = 8;
    cfg.prefetch_batch_size = 8;
    assert(tm_init(&m, &cfg) == ORCHKV_OK);

    const int n_blocks = 32;
    const int n_hot = 12;
    const int n_steps = 100;

    for (int i = 0; i < n_blocks; i++) {
        init_gpu_block(i, (uint64_t)(1000 + i), (uint8_t)(i));
        assert(tm_register_block(&m, &g_blks[i]) == ORCHKV_OK);
    }

    double total_step_us = 0, total_sched_us = 0;
    double step_times[100];
    double sched_times[100];

    for (int step = 0; step < n_steps; step++) {
        double t0 = now_us();

        /* Simulate attention: hot blocks get high scores */
        for (int i = 0; i < n_hot; i++) {
            tm_notify_attn(&m, (uint64_t)(1000 + i), 0.8f + 0.01f * (step % 10));
            tm_notify_access(&m, &g_blks[i]);
        }
        tm_step_done(&m);

        /* Simulate varying GPU pressure */
        float gpu_ratio = 0.6f + 0.35f * sinf((float)step * 0.1f);
        float dram_ratio = 0.5f + 0.3f * sinf((float)step * 0.07f + 1.0f);
        tm_set_usage(&m, gpu_ratio, dram_ratio);

        double t_sched0 = now_us();
        tm_schedule_once(&m);
        double t_sched1 = now_us();

        double t1 = now_us();
        step_times[step]  = t1 - t0;
        sched_times[step] = t_sched1 - t_sched0;
        total_step_us  += step_times[step];
        total_sched_us += sched_times[step];
    }

    tm_stats_t stats;
    tm_get_stats(&m, &stats);

    double avg_step  = total_step_us / n_steps;
    double avg_sched = total_sched_us / n_steps;

    /* Compute p50/p99 for schedule latency */
    double sorted[100];
    memcpy(sorted, sched_times, sizeof(sorted));
    for (int i = 0; i < n_steps - 1; i++)
        for (int j = i + 1; j < n_steps; j++)
            if (sorted[j] < sorted[i]) {
                double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
            }
    double p50_sched = sorted[n_steps / 2];
    double p99_sched = sorted[(int)(n_steps * 0.99)];

    printf("  Blocks: %d (hot=%d, cold=%d)\n", n_blocks, n_hot, n_blocks - n_hot);
    printf("  Steps:  %d\n", n_steps);
    printf("  Avg step latency:       %8.1f us\n", avg_step);
    printf("  Avg schedule latency:   %8.1f us  (p50=%.1f, p99=%.1f)\n",
           avg_sched, p50_sched, p99_sched);
    printf("  Schedule cycles:        %llu\n", (unsigned long long)stats.schedule_cycles);
    printf("  GPU demotes:            %llu\n", (unsigned long long)stats.gpu_demotes);
    printf("  DRAM demotes:           %llu\n", (unsigned long long)stats.dram_demotes);
    printf("  Prefetches dispatched:  %llu\n", (unsigned long long)stats.prefetches_dispatched);
    printf("  Migration blocks total: %llu\n",
           (unsigned long long)stats.migration_stats.blocks_migrated);

    /* Count blocks per tier at end */
    int on_gpu = 0, on_dram = 0, on_stor = 0;
    for (int i = 0; i < n_blocks; i++) {
        switch (g_blks[i].tier) {
        case TIER_GPU_HBM:   on_gpu++;  break;
        case TIER_HOST_DRAM:  on_dram++; break;
        case TIER_NVM: case TIER_SSD: on_stor++; break;
        default: break;
        }
    }
    printf("  Final distribution: GPU=%d, DRAM=%d, Storage=%d\n",
           on_gpu, on_dram, on_stor);

    /* JSON output */
    FILE *fp = fopen("benchmark_e2e_auto.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"n_blocks\": %d,\n", n_blocks);
        fprintf(fp, "  \"n_hot\": %d,\n", n_hot);
        fprintf(fp, "  \"n_steps\": %d,\n", n_steps);
        fprintf(fp, "  \"avg_step_us\": %.2f,\n", avg_step);
        fprintf(fp, "  \"avg_schedule_us\": %.2f,\n", avg_sched);
        fprintf(fp, "  \"p50_schedule_us\": %.2f,\n", p50_sched);
        fprintf(fp, "  \"p99_schedule_us\": %.2f,\n", p99_sched);
        fprintf(fp, "  \"schedule_cycles\": %llu,\n", (unsigned long long)stats.schedule_cycles);
        fprintf(fp, "  \"gpu_demotes\": %llu,\n", (unsigned long long)stats.gpu_demotes);
        fprintf(fp, "  \"dram_demotes\": %llu,\n", (unsigned long long)stats.dram_demotes);
        fprintf(fp, "  \"prefetches_dispatched\": %llu,\n",
                (unsigned long long)stats.prefetches_dispatched);
        fprintf(fp, "  \"blocks_migrated\": %llu,\n",
                (unsigned long long)stats.migration_stats.blocks_migrated);
        fprintf(fp, "  \"final_gpu\": %d,\n", on_gpu);
        fprintf(fp, "  \"final_dram\": %d,\n", on_dram);
        fprintf(fp, "  \"final_storage\": %d,\n", on_stor);
        fprintf(fp, "  \"block_size_bytes\": %d,\n", BLK_SZ);
        fprintf(fp, "  \"per_step_us\": [");
        for (int i = 0; i < n_steps; i++)
            fprintf(fp, "%s%.1f", i ? "," : "", step_times[i]);
        fprintf(fp, "],\n");
        fprintf(fp, "  \"per_sched_us\": [");
        for (int i = 0; i < n_steps; i++)
            fprintf(fp, "%s%.1f", i ? "," : "", sched_times[i]);
        fprintf(fp, "]\n}\n");
        fclose(fp);
        printf("  Benchmark JSON → benchmark_e2e_auto.json\n");
    }

    for (int i = 0; i < n_blocks; i++) free_block(i);
    tm_destroy(&m);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_e2e_auto (C11) ===\n");

    test_auto_evict_gpu();
    test_auto_evict_dram();
    test_auto_prefetch();
    test_data_integrity();
    test_multi_request();

    benchmark_auto_schedule();

    printf("\n=== ALL PASSED ===\n");
    return 0;
}
