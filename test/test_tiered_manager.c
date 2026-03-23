#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "scheduler/tiered_manager.h"

static int tests_run = 0, tests_passed = 0;

#define RUN_TEST(fn) do {                           \
    printf("  %-50s", #fn); fflush(stdout);         \
    tests_run++; fn(); tests_passed++;              \
    printf(" PASS\n"); } while (0)

#define ASSERT_OK(e)          assert((e) == ORCHKV_OK)
#define ASSERT_EQ(a, b)       assert((a) == (b))
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_NEAR(a,b,e)    assert(fabsf((a)-(b)) < (e))

/* ========================================================================
 *  Mock transfer + allocator
 * ======================================================================== */

static int mock_transfer(kv_block_t *blk, void *dst,
                         size_t size, MigrateOp op, void *ctx)
{
    (void)ctx; (void)op;
    if (blk->data_ptr && dst)
        memcpy(dst, blk->data_ptr, size);
    return ORCHKV_OK;
}

#define MAX_ALLOCS 256
static void *g_allocs[MAX_ALLOCS];
static int g_alloc_count = 0;

static void *test_alloc(size_t size, StorageTier tier, void *ctx)
{
    (void)tier; (void)ctx;
    void *p = calloc(1, size);
    if (p && g_alloc_count < MAX_ALLOCS)
        g_allocs[g_alloc_count++] = p;
    return p;
}

static void test_free(void *ptr, StorageTier tier, void *ctx)
{
    (void)tier; (void)ctx;
    for (int i = 0; i < g_alloc_count; i++) {
        if (g_allocs[i] == ptr) {
            free(ptr);
            g_allocs[i] = g_allocs[--g_alloc_count];
            return;
        }
    }
    /* Not allocated by test_alloc — skip (e.g. static g_data buffers) */
}

static void cleanup_allocs(void)
{
    g_alloc_count = 0;
}

/* ========================================================================
 *  Block helpers
 * ======================================================================== */

#define BLK_SZ 256
#define MAX_BLK 32

static kv_block_t g_blks[MAX_BLK];
static uint8_t    g_data[MAX_BLK][BLK_SZ];

static void make_blk(int idx, uint64_t id, StorageTier tier,
                     KVBlockState state, uint8_t flags)
{
    memset(&g_blks[idx], 0, sizeof(kv_block_t));
    memset(g_data[idx], (uint8_t)(id & 0xFF), BLK_SZ);
    g_blks[idx].block_id    = id;
    g_blks[idx].request_id  = 1;
    g_blks[idx].layer_id    = 0;
    g_blks[idx].head_id     = 0;
    g_blks[idx].token_start = (uint32_t)(id * 64);
    g_blks[idx].token_count = 64;
    g_blks[idx].tier        = tier;
    g_blks[idx].state       = state;
    g_blks[idx].flags       = flags;
    g_blks[idx].data_ptr    = g_data[idx];
    pthread_rwlock_init(&g_blks[idx].lock, NULL);
}

static tm_config_t base_cfg(void)
{
    tm_config_t cfg;
    tm_config_default(&cfg);
    cfg.tracker_capacity   = 256;
    cfg.transfer_fn        = mock_transfer;
    cfg.alloc_fn           = test_alloc;
    cfg.free_fn            = test_free;
    cfg.block_data_size    = BLK_SZ;
    cfg.auto_schedule      = false;
    cfg.max_blocks         = MAX_BLK;
    cfg.demote_batch_size  = 4;
    cfg.prefetch_batch_size = 4;

    cfg.athresh_params.cooldown_sec = 0.0;
    cfg.athresh_params.gpu_hwm  = 0.9f;
    cfg.athresh_params.gpu_lwm  = 0.7f;
    cfg.athresh_params.dram_hwm = 0.9f;
    cfg.athresh_params.dram_lwm = 0.7f;
    return cfg;
}

/* ======================================================================== */

static void test_init_destroy(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    ASSERT_OK(tm_init(&m, &cfg));

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_EQ(stats.schedule_cycles, 0ULL);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_register_unregister(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    ASSERT_OK(tm_init(&m, &cfg));

    make_blk(0, 10, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE);
    make_blk(1, 20, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE);

    ASSERT_OK(tm_register_block(&m, &g_blks[0]));
    ASSERT_OK(tm_register_block(&m, &g_blks[1]));
    ASSERT_EQ(m.n_blocks, 2u);
    ASSERT_EQ(evpol_lru_size(&m.evpol), 2u);

    tm_unregister_block(&m, &g_blks[0]);
    ASSERT_EQ(m.n_blocks, 1u);
    ASSERT_EQ(evpol_lru_size(&m.evpol), 1u);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_notify_attn_step(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    ASSERT_OK(tm_init(&m, &cfg));

    make_blk(0, 10, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE);
    ASSERT_OK(tm_register_block(&m, &g_blks[0]));

    for (int step = 0; step < 20; step++) {
        tm_notify_attn(&m, 10, 0.8f);
        tm_step_done(&m);
    }

    hcc_update_all(&m.classifier);
    float score = hcc_get_score(&m.classifier, 10);
    ASSERT_TRUE(score > 0.3f);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_schedule_gpu_demote(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    cfg.hcc_params.alpha = 1.0f;
    cfg.hcc_params.beta  = 0.0f;
    cfg.hcc_params.gamma = 0.0f;
    ASSERT_OK(tm_init(&m, &cfg));

    /* 6 GPU blocks: 2 hot (high attention), 4 cold (no attention) */
    for (int i = 0; i < 6; i++) {
        make_blk(i, (uint64_t)(10 + i * 10), TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE);
        ASSERT_OK(tm_register_block(&m, &g_blks[i]));
    }

    /* Pump attention: blocks 10,20 hot; 30-60 cold.
     * Also LRU-touch the hot blocks so they stay at the head. */
    for (int step = 0; step < 50; step++) {
        tm_notify_attn(&m, 10, 0.9f);
        tm_notify_attn(&m, 20, 0.8f);
        tm_notify_access(&m, &g_blks[0]);
        tm_notify_access(&m, &g_blks[1]);
        tm_step_done(&m);
    }

    /* Set GPU overloaded */
    tm_set_usage(&m, 0.95f, 0.80f);

    /* Run scheduler — should demote cold blocks */
    tm_schedule_once(&m);

    /* At least one block should have been demoted from GPU */
    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_TRUE(stats.gpu_demotes > 0);

    /* Hot blocks (10, 20) should still be on GPU */
    ASSERT_EQ(g_blks[0].tier, TIER_GPU_HBM);
    ASSERT_EQ(g_blks[1].tier, TIER_GPU_HBM);

    /* Some cold blocks should have been moved to DRAM */
    int dram_count = 0;
    for (int i = 2; i < 6; i++)
        if (g_blks[i].tier == TIER_HOST_DRAM) dram_count++;
    ASSERT_TRUE(dram_count > 0);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_schedule_dram_demote(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    cfg.hcc_params.alpha = 1.0f;
    cfg.hcc_params.beta  = 0.0f;
    cfg.hcc_params.gamma = 0.0f;
    ASSERT_OK(tm_init(&m, &cfg));

    /* 4 DRAM blocks, all cold */
    for (int i = 0; i < 4; i++) {
        make_blk(i, (uint64_t)(50 + i), TIER_HOST_DRAM, KV_STATE_WARM, KV_FLAG_NONE);
        ASSERT_OK(tm_register_block(&m, &g_blks[i]));
    }

    for (int step = 0; step < 30; step++)
        tm_step_done(&m);

    /* DRAM overloaded, GPU OK */
    tm_set_usage(&m, 0.5f, 0.95f);

    tm_schedule_once(&m);

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_TRUE(stats.dram_demotes > 0);

    int nvm_count = 0;
    for (int i = 0; i < 4; i++)
        if (g_blks[i].tier == TIER_NVM) nvm_count++;
    ASSERT_TRUE(nvm_count > 0);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_no_demote_in_midrange(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    ASSERT_OK(tm_init(&m, &cfg));

    make_blk(0, 10, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE);
    ASSERT_OK(tm_register_block(&m, &g_blks[0]));

    /* Both usage in midrange */
    tm_set_usage(&m, 0.8f, 0.8f);
    tm_schedule_once(&m);

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_EQ(stats.gpu_demotes, 0ULL);
    ASSERT_EQ(stats.dram_demotes, 0ULL);

    /* Block unchanged */
    ASSERT_EQ(g_blks[0].tier, TIER_GPU_HBM);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_skip_pinned_in_schedule(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    ASSERT_OK(tm_init(&m, &cfg));

    /* All blocks GPU, but all pinned */
    for (int i = 0; i < 4; i++) {
        make_blk(i, (uint64_t)(60 + i), TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_PIN);
        ASSERT_OK(tm_register_block(&m, &g_blks[i]));
    }

    for (int step = 0; step < 20; step++)
        tm_step_done(&m);

    tm_set_usage(&m, 0.95f, 0.5f);
    tm_schedule_once(&m);

    /* No blocks should be demoted (all pinned) */
    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_EQ(stats.gpu_demotes, 0ULL);

    for (int i = 0; i < 4; i++)
        ASSERT_EQ(g_blks[i].tier, TIER_GPU_HBM);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_prefetch_in_schedule(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    cfg.hcc_params.alpha = 1.0f;
    cfg.hcc_params.beta  = 0.0f;
    cfg.hcc_params.gamma = 0.0f;
    cfg.threshold_to_gpu  = 0.5f;
    cfg.threshold_to_dram = 0.2f;
    ASSERT_OK(tm_init(&m, &cfg));

    /* 2 blocks on DRAM with high attention → prefetch candidates */
    make_blk(0, 10, TIER_HOST_DRAM, KV_STATE_WARM, KV_FLAG_NONE);
    make_blk(1, 20, TIER_HOST_DRAM, KV_STATE_WARM, KV_FLAG_NONE);
    ASSERT_OK(tm_register_block(&m, &g_blks[0]));
    ASSERT_OK(tm_register_block(&m, &g_blks[1]));

    for (int step = 0; step < 50; step++) {
        tm_notify_attn(&m, 10, 0.9f);
        tm_notify_attn(&m, 20, 0.7f);
        tm_step_done(&m);
    }

    /* No pressure — just trigger prefetch scan */
    tm_set_usage(&m, 0.5f, 0.5f);
    tm_schedule_once(&m);

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_TRUE(stats.prefetches_dispatched > 0);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_set_policy(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    ASSERT_OK(tm_init(&m, &cfg));

    tm_set_policy(&m, 0.8f, 0.1f, 0.1f);

    ASSERT_NEAR(m.classifier.alpha, 0.8f, 1e-5f);
    ASSERT_NEAR(m.classifier.beta,  0.1f, 1e-5f);
    ASSERT_NEAR(m.classifier.gamma, 0.1f, 1e-5f);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_background_thread(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    cfg.schedule_interval_us = 500;  /* 0.5ms */
    ASSERT_OK(tm_init(&m, &cfg));

    ASSERT_OK(tm_start(&m));
    ASSERT_TRUE(m.running);

    usleep(5000);  /* let it run for 5ms → ~10 cycles */

    tm_stop(&m);
    ASSERT_TRUE(!m.running);

    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_TRUE(stats.schedule_cycles >= 2);

    tm_destroy(&m);
    cleanup_allocs();
}

static void test_full_lifecycle(void)
{
    tiered_manager_t m;
    tm_config_t cfg = base_cfg();
    cfg.hcc_params.alpha = 1.0f;
    cfg.hcc_params.beta  = 0.0f;
    cfg.hcc_params.gamma = 0.0f;
    ASSERT_OK(tm_init(&m, &cfg));

    /* Phase 1: register 8 GPU blocks */
    for (int i = 0; i < 8; i++) {
        make_blk(i, (uint64_t)(100 + i), TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE);
        ASSERT_OK(tm_register_block(&m, &g_blks[i]));
    }

    /* Phase 2: pump attention — blocks 100-103 hot, 104-107 cold */
    for (int step = 0; step < 50; step++) {
        for (int i = 0; i < 4; i++)
            tm_notify_attn(&m, (uint64_t)(100 + i), 0.9f);
        tm_step_done(&m);
    }

    /* Phase 3: GPU overloaded → demote cold blocks */
    tm_set_usage(&m, 0.95f, 0.5f);
    tm_schedule_once(&m);

    int gpu_count = 0, dram_count = 0;
    for (int i = 0; i < 8; i++) {
        if (g_blks[i].tier == TIER_GPU_HBM) gpu_count++;
        if (g_blks[i].tier == TIER_HOST_DRAM) dram_count++;
    }

    ASSERT_TRUE(dram_count > 0);
    /* Hot blocks should remain on GPU */
    for (int i = 0; i < 4; i++)
        ASSERT_EQ(g_blks[i].tier, TIER_GPU_HBM);

    /* Phase 4: unregister a block */
    tm_unregister_block(&m, &g_blks[7]);
    ASSERT_EQ(m.n_blocks, 7u);

    /* Phase 5: verify stats */
    tm_stats_t stats;
    tm_get_stats(&m, &stats);
    ASSERT_TRUE(stats.schedule_cycles >= 1);
    ASSERT_TRUE(stats.gpu_demotes > 0);
    ASSERT_TRUE(stats.migration_stats.blocks_migrated > 0);

    tm_destroy(&m);
    cleanup_allocs();
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_tiered_manager ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_register_unregister);
    RUN_TEST(test_notify_attn_step);
    RUN_TEST(test_schedule_gpu_demote);
    RUN_TEST(test_schedule_dram_demote);
    RUN_TEST(test_no_demote_in_midrange);
    RUN_TEST(test_skip_pinned_in_schedule);
    RUN_TEST(test_prefetch_in_schedule);
    RUN_TEST(test_set_policy);
    RUN_TEST(test_background_thread);
    RUN_TEST(test_full_lifecycle);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
