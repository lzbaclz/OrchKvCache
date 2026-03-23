/*
 * C10: Prefetch Scheduler Integration Test
 *
 * Wires C1 (attention_tracker) + C2 (classifier) + C4 (eviction) +
 *       C5 (prefetch_scheduler) + C6 (pipeline) + C7 (migration_engine)
 * and verifies cross-component prefetch behaviour end-to-end.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/hotcold_classifier.h"
#include "scheduler/eviction_policy.h"
#include "scheduler/prefetch_scheduler.h"
#include "scheduler/pipeline.h"
#include "scheduler/migration_engine.h"

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
 *  Mock transfer
 * ======================================================================== */

static int mock_transfer(kv_block_t *blk, void *dst,
                         size_t size, MigrateOp op, void *ctx)
{
    (void)ctx; (void)op;
    if (blk->data_ptr && dst)
        memcpy(dst, blk->data_ptr, size);
    return ORCHKV_OK;
}

/* ========================================================================
 *  Fixture: tracker + classifier + evpol + prefetch + pipeline + migration
 * ======================================================================== */

typedef struct fixture {
    attention_tracker_t   tracker;
    hotcold_classifier_t  classifier;
    eviction_policy_t     evpol;
    prefetch_scheduler_t  prefetch;
    pipeline_t            pipeline;
    migration_engine_t    migration;
} fixture_t;

static void fixture_init(fixture_t *f, uint32_t budget,
                          float thr_gpu, float thr_dram)
{
    attn_tracker_init(&f->tracker, 256, 0.9f);

    hcc_params_t hp;
    hcc_params_default(&hp);
    hcc_init(&f->classifier, &f->tracker, &hp);

    evpol_init(&f->evpol, &f->classifier, 8, 64, 0.7f, 0.3f);

    prefetch_init(&f->prefetch, &f->tracker,
                  budget, thr_gpu, thr_dram, 0);

    pipeline_init(&f->pipeline, &f->prefetch);

    mig_init(&f->migration, &f->evpol, &f->prefetch,
             mock_transfer, NULL);
}

static void fixture_destroy(fixture_t *f)
{
    mig_destroy(&f->migration);
    pipeline_destroy(&f->pipeline);
    prefetch_destroy(&f->prefetch);
    evpol_destroy(&f->evpol);
    hcc_destroy(&f->classifier);
    attn_tracker_destroy(&f->tracker);
}

/* ========================================================================
 *  Block helpers
 * ======================================================================== */

#define MAX_BLK 16
#define BLK_SZ  256
static kv_block_t g_blks[MAX_BLK];
static uint8_t    g_data[MAX_BLK][BLK_SZ];

static void make_blk(int idx, uint64_t id, StorageTier tier, uint8_t fill)
{
    memset(&g_blks[idx], 0, sizeof(kv_block_t));
    memset(g_data[idx], fill, BLK_SZ);
    g_blks[idx].block_id    = id;
    g_blks[idx].request_id  = 1;
    g_blks[idx].layer_id    = 0;
    g_blks[idx].head_id     = 0;
    g_blks[idx].token_start = (uint32_t)(id * 64);
    g_blks[idx].token_count = 64;
    g_blks[idx].tier        = tier;
    g_blks[idx].state       = (tier == TIER_GPU_HBM) ? KV_STATE_HOT : KV_STATE_WARM;
    g_blks[idx].flags       = KV_FLAG_NONE;
    g_blks[idx].data_ptr    = g_data[idx];
    pthread_rwlock_init(&g_blks[idx].lock, NULL);
}

/* ======================================================================== */

/* No blocks registered → scan produces nothing, dispatch returns 0 */
static void test_prefetch_scan_empty(void)
{
    fixture_t f;
    fixture_init(&f, 16, 0.5f, 0.2f);

    prefetch_block_info_t empty_info[1];
    prefetch_scan_blocks(&f.prefetch, empty_info, 0);
    ASSERT_EQ(f.prefetch.heap_size, 0u);

    prefetch_entry_t out[4];
    uint32_t n = prefetch_dispatch(&f.prefetch, 4, out);
    ASSERT_EQ(n, 0u);

    fixture_destroy(&f);
}

/* High-EMA block is dispatched before low-EMA block */
static void test_prefetch_priority(void)
{
    fixture_t f;
    fixture_init(&f, 16, 0.3f, 0.1f);

    attn_tracker_register(&f.tracker, 10);
    attn_tracker_register(&f.tracker, 20);
    attn_tracker_register(&f.tracker, 30);

    for (int step = 0; step < 60; step++) {
        attn_tracker_update(&f.tracker, 10, 0.9f);
        attn_tracker_update(&f.tracker, 20, 0.5f);
        attn_tracker_update(&f.tracker, 30, 0.1f);
        attn_tracker_step_done(&f.tracker);
    }

    prefetch_block_info_t info[3] = {
        {10, TIER_HOST_DRAM},
        {20, TIER_HOST_DRAM},
        {30, TIER_HOST_DRAM},
    };
    prefetch_scan_blocks(&f.prefetch, info, 3);

    prefetch_entry_t out[3];
    uint32_t n = prefetch_dispatch(&f.prefetch, 3, out);
    ASSERT_TRUE(n >= 2);

    /* First dispatched should have highest priority */
    ASSERT_TRUE(out[0].priority >= out[1].priority);
    /* Block 10 (highest EMA) should be first */
    ASSERT_EQ(out[0].block_id, 10ULL);

    fixture_destroy(&f);
}

/* Budget limits the number of dispatched candidates */
static void test_prefetch_budget(void)
{
    fixture_t f;
    fixture_init(&f, 3, 0.1f, 0.05f);  /* budget = 3 */

    for (int i = 0; i < 8; i++) {
        uint64_t id = (uint64_t)(100 + i);
        attn_tracker_register(&f.tracker, id);
    }

    for (int step = 0; step < 40; step++) {
        for (int i = 0; i < 8; i++)
            attn_tracker_update(&f.tracker, (uint64_t)(100 + i), 0.6f);
        attn_tracker_step_done(&f.tracker);
    }

    prefetch_block_info_t info[8];
    for (int i = 0; i < 8; i++) {
        info[i].block_id = (uint64_t)(100 + i);
        info[i].tier = TIER_HOST_DRAM;
    }
    prefetch_scan_blocks(&f.prefetch, info, 8);

    prefetch_entry_t out[8];
    uint32_t n = prefetch_dispatch(&f.prefetch, 8, out);
    ASSERT_EQ(n, 3u);  /* capped by budget */

    fixture_destroy(&f);
}

/* Hit rate is computed correctly after notify_hit + step_reset */
static void test_prefetch_hit_rate(void)
{
    fixture_t f;
    fixture_init(&f, 16, 0.3f, 0.1f);

    for (int i = 0; i < 4; i++) {
        uint64_t id = (uint64_t)(200 + i);
        attn_tracker_register(&f.tracker, id);
        prefetch_add_candidate(&f.prefetch, id,
                               TIER_HOST_DRAM, TIER_GPU_HBM, 0.8f - i * 0.1f);
    }

    prefetch_entry_t out[4];
    uint32_t n = prefetch_dispatch(&f.prefetch, 4, out);
    ASSERT_EQ(n, 4u);

    /* 3 of 4 were actually accessed */
    ASSERT_TRUE(prefetch_notify_hit(&f.prefetch, 200));
    ASSERT_TRUE(prefetch_notify_hit(&f.prefetch, 201));
    ASSERT_TRUE(prefetch_notify_hit(&f.prefetch, 202));

    prefetch_step_reset(&f.prefetch);

    prefetch_stats_t stats;
    prefetch_get_stats(&f.prefetch, &stats);
    ASSERT_EQ(stats.prefetch_hits, 3ULL);
    ASSERT_EQ(stats.prefetch_wasted, 1ULL);
    ASSERT_NEAR(stats.hit_rate, 0.75f, 0.01f);

    fixture_destroy(&f);
}

/* Blocks already on GPU are skipped during scan */
static void test_prefetch_skip_gpu(void)
{
    fixture_t f;
    fixture_init(&f, 16, 0.3f, 0.1f);

    attn_tracker_register(&f.tracker, 10);
    attn_tracker_register(&f.tracker, 20);

    for (int step = 0; step < 40; step++) {
        attn_tracker_update(&f.tracker, 10, 0.8f);
        attn_tracker_update(&f.tracker, 20, 0.8f);
        attn_tracker_step_done(&f.tracker);
    }

    prefetch_block_info_t info[2] = {
        {10, TIER_GPU_HBM},    /* already on GPU */
        {20, TIER_HOST_DRAM},  /* candidate */
    };
    prefetch_scan_blocks(&f.prefetch, info, 2);

    prefetch_entry_t out[2];
    uint32_t n = prefetch_dispatch(&f.prefetch, 2, out);
    ASSERT_TRUE(n >= 1);

    /* Only block 20 should be dispatched (block 10 skipped) */
    for (uint32_t i = 0; i < n; i++)
        ASSERT_TRUE(out[i].block_id != 10);

    fixture_destroy(&f);
}

/*
 * Integration: prefetch dispatch → migration engine → block tier changes.
 * Verifies data integrity through the full promote path.
 */
static void test_prefetch_async(void)
{
    fixture_t f;
    fixture_init(&f, 16, 0.3f, 0.1f);

    /* Block on DRAM with high attention */
    make_blk(0, 10, TIER_HOST_DRAM, 0xAA);
    attn_tracker_register(&f.tracker, 10);
    hcc_register(&f.classifier, 10, KV_FLAG_NONE);
    evpol_lru_touch(&f.evpol, &g_blks[0]);

    for (int step = 0; step < 50; step++) {
        attn_tracker_update(&f.tracker, 10, 0.9f);
        attn_tracker_step_done(&f.tracker);
    }

    /* Scan and dispatch */
    prefetch_block_info_t info = {10, TIER_HOST_DRAM};
    prefetch_scan_blocks(&f.prefetch, &info, 1);

    prefetch_entry_t out[1];
    uint32_t n = prefetch_dispatch(&f.prefetch, 1, out);
    ASSERT_TRUE(n >= 1);
    ASSERT_EQ(out[0].block_id, 10ULL);

    /* Execute the promotion via migration engine */
    uint8_t gpu_buf[BLK_SZ];
    memset(gpu_buf, 0, BLK_SZ);

    int rc = mig_execute_one(&f.migration, &g_blks[0], TIER_GPU_HBM,
                             gpu_buf, NULL, BLK_SZ);
    ASSERT_OK(rc);

    /* Block should now be on GPU */
    ASSERT_EQ(g_blks[0].tier, TIER_GPU_HBM);
    ASSERT_EQ(g_blks[0].data_ptr, (void *)gpu_buf);

    /* Data integrity: gpu_buf should have the 0xAA pattern */
    uint8_t expected[BLK_SZ];
    memset(expected, 0xAA, BLK_SZ);
    ASSERT_EQ(memcmp(gpu_buf, expected, BLK_SZ), 0);

    /* Notify hit — hit rate should reflect this */
    prefetch_notify_hit(&f.prefetch, 10);
    prefetch_step_reset(&f.prefetch);

    prefetch_stats_t stats;
    prefetch_get_stats(&f.prefetch, &stats);
    ASSERT_EQ(stats.prefetch_hits, 1ULL);
    ASSERT_EQ(stats.prefetch_wasted, 0ULL);

    fixture_destroy(&f);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_prefetch (C10: integration) ===\n");

    RUN_TEST(test_prefetch_scan_empty);
    RUN_TEST(test_prefetch_priority);
    RUN_TEST(test_prefetch_budget);
    RUN_TEST(test_prefetch_hit_rate);
    RUN_TEST(test_prefetch_skip_gpu);
    RUN_TEST(test_prefetch_async);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
