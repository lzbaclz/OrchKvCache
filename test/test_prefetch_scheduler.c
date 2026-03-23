#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/prefetch_scheduler.h"

static int tests_run = 0, tests_passed = 0;

#define RUN_TEST(fn) do {                           \
    printf("  %-50s", #fn); fflush(stdout);         \
    tests_run++; fn(); tests_passed++;              \
    printf(" PASS\n"); } while (0)

#define ASSERT_OK(e)          assert((e) == ORCHKV_OK)
#define ASSERT_EQ(a, b)       assert((a) == (b))
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_FALSE(c)       assert(!(c))
#define ASSERT_NEAR(a, b, e)  assert(fabsf((a)-(b)) < (e))

/* ======================================================================== */

static void test_init_destroy(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 16, 0.5f, 0.25f, 128));
    ASSERT_EQ(s.prefetch_budget, 16u);
    ASSERT_EQ(s.heap_cap, 128u);
    ASSERT_EQ(s.heap_size, 0u);
    prefetch_destroy(&s);
}

static void test_add_candidate_basic(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 8, 0.5f, 0.25f, 64));

    ASSERT_OK(prefetch_add_candidate(&s, 10, TIER_HOST_DRAM, TIER_GPU_HBM, 0.9f));
    ASSERT_OK(prefetch_add_candidate(&s, 20, TIER_NVM,       TIER_HOST_DRAM, 0.3f));
    ASSERT_EQ(s.heap_size, 2u);
    ASSERT_EQ(s.total_enqueued, 2ULL);

    prefetch_destroy(&s);
}

static void test_heap_ordering(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 16, 0.5f, 0.25f, 64));

    prefetch_add_candidate(&s, 1, TIER_HOST_DRAM, TIER_GPU_HBM, 0.3f);
    prefetch_add_candidate(&s, 2, TIER_HOST_DRAM, TIER_GPU_HBM, 0.9f);
    prefetch_add_candidate(&s, 3, TIER_HOST_DRAM, TIER_GPU_HBM, 0.5f);
    prefetch_add_candidate(&s, 4, TIER_HOST_DRAM, TIER_GPU_HBM, 0.7f);
    prefetch_add_candidate(&s, 5, TIER_HOST_DRAM, TIER_GPU_HBM, 0.1f);

    /* Dispatch all 5 — should come out in priority order */
    prefetch_entry_t out[5];
    uint32_t n = prefetch_dispatch(&s, 5, out);
    ASSERT_EQ(n, 5u);

    ASSERT_EQ(out[0].block_id, 2ULL);  /* 0.9 */
    ASSERT_EQ(out[1].block_id, 4ULL);  /* 0.7 */
    ASSERT_EQ(out[2].block_id, 3ULL);  /* 0.5 */
    ASSERT_EQ(out[3].block_id, 1ULL);  /* 0.3 */
    ASSERT_EQ(out[4].block_id, 5ULL);  /* 0.1 */

    prefetch_destroy(&s);
}

static void test_dispatch_respects_budget(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 3, 0.5f, 0.25f, 64));  /* budget = 3 */

    for (int i = 0; i < 10; i++)
        prefetch_add_candidate(&s, (uint64_t)i, TIER_HOST_DRAM, TIER_GPU_HBM,
                               (float)i * 0.1f);

    prefetch_entry_t out[10];
    uint32_t n = prefetch_dispatch(&s, 10, out);
    ASSERT_EQ(n, 3u);  /* budget caps at 3 */

    /* max_n < budget should also work */
    n = prefetch_dispatch(&s, 2, out);
    ASSERT_EQ(n, 2u);

    prefetch_destroy(&s);
}

static void test_dispatch_empty(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 8, 0.5f, 0.25f, 64));

    prefetch_entry_t out[4];
    uint32_t n = prefetch_dispatch(&s, 4, out);
    ASSERT_EQ(n, 0u);

    prefetch_destroy(&s);
}

static void test_notify_hit(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 8, 0.5f, 0.25f, 64));

    prefetch_add_candidate(&s, 10, TIER_HOST_DRAM, TIER_GPU_HBM, 0.8f);
    prefetch_add_candidate(&s, 20, TIER_HOST_DRAM, TIER_GPU_HBM, 0.6f);

    prefetch_entry_t out[2];
    prefetch_dispatch(&s, 2, out);

    /* Block 10 is accessed → hit */
    ASSERT_TRUE(prefetch_notify_hit(&s, 10));
    ASSERT_EQ(s.prefetch_hits, 1ULL);

    /* Block 99 was never dispatched → not a hit */
    ASSERT_FALSE(prefetch_notify_hit(&s, 99));
    ASSERT_EQ(s.prefetch_hits, 1ULL);

    /* Block 10 already removed from tracking → false */
    ASSERT_FALSE(prefetch_notify_hit(&s, 10));

    prefetch_destroy(&s);
}

static void test_step_reset_wasted(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 8, 0.5f, 0.25f, 64));

    prefetch_add_candidate(&s, 10, TIER_HOST_DRAM, TIER_GPU_HBM, 0.9f);
    prefetch_add_candidate(&s, 20, TIER_HOST_DRAM, TIER_GPU_HBM, 0.7f);
    prefetch_add_candidate(&s, 30, TIER_HOST_DRAM, TIER_GPU_HBM, 0.5f);

    prefetch_entry_t out[3];
    prefetch_dispatch(&s, 3, out);

    /* Only block 10 was accessed */
    prefetch_notify_hit(&s, 10);

    /* Step reset: remaining 2 (blocks 20, 30) count as wasted */
    prefetch_step_reset(&s);
    ASSERT_EQ(s.prefetch_wasted, 2ULL);
    ASSERT_EQ(s.prefetch_hits, 1ULL);
    ASSERT_EQ(s.tracked_count, 0u);
    ASSERT_EQ(s.heap_size, 0u);

    prefetch_destroy(&s);
}

static void test_hit_rate(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 16, 0.5f, 0.25f, 64));

    for (int i = 0; i < 10; i++)
        prefetch_add_candidate(&s, (uint64_t)i, TIER_HOST_DRAM, TIER_GPU_HBM,
                               0.5f + (float)i * 0.01f);

    prefetch_entry_t out[10];
    prefetch_dispatch(&s, 10, out);

    /* 7 out of 10 are accessed */
    for (int i = 0; i < 7; i++)
        prefetch_notify_hit(&s, out[i].block_id);

    ASSERT_NEAR(prefetch_hit_rate(&s), 0.7f, 0.01f);

    prefetch_stats_t stats;
    prefetch_get_stats(&s, &stats);
    ASSERT_EQ(stats.total_dispatched, 10ULL);
    ASSERT_EQ(stats.prefetch_hits, 7ULL);

    prefetch_destroy(&s);
}

static void test_scan_with_tracker(void)
{
    attention_tracker_t tracker;
    ASSERT_OK(attn_tracker_init(&tracker, 256, 0.9f));

    /* Register 5 blocks, pump different attention levels */
    for (uint64_t id = 1; id <= 5; id++)
        attn_tracker_register(&tracker, id);

    for (int step = 0; step < 50; step++) {
        attn_tracker_update(&tracker, 1, 0.9f);  /* high → should prefetch to GPU */
        attn_tracker_update(&tracker, 2, 0.6f);  /* medium → should prefetch to GPU */
        attn_tracker_update(&tracker, 3, 0.3f);  /* low → should prefetch to DRAM */
        attn_tracker_update(&tracker, 4, 0.05f); /* very low → below both thresholds */
        /* block 5: never updated → ema ≈ 0, decays */
        attn_tracker_step_done(&tracker);
    }

    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, &tracker, 16, 0.5f, 0.2f, 64));

    /* All 5 blocks are on DRAM */
    prefetch_block_info_t blocks[] = {
        {1, TIER_HOST_DRAM},
        {2, TIER_HOST_DRAM},
        {3, TIER_HOST_DRAM},
        {4, TIER_HOST_DRAM},
        {5, TIER_HOST_DRAM},
    };
    prefetch_scan_blocks(&s, blocks, 5);

    /* Blocks 1,2 should be candidates (ema > 0.5 → GPU target)
     * Block 3 should be candidate (ema > 0.2 → DRAM target, but already on DRAM, so skip)
     * Actually block 3 is on DRAM and target would be DRAM → skipped (tier != TIER_HOST_DRAM check)
     * Wait — the scan logic: if ema >= threshold_to_gpu → target GPU; elif ema >= threshold_to_dram && tier != HOST_DRAM → target DRAM
     * Block 3 (ema ~0.3): 0.3 >= 0.2 but tier == DRAM → skip DRAM target
     * Block 3 doesn't qualify for GPU (0.3 < 0.5)
     * So only blocks 1,2 should be enqueued */

    ASSERT_TRUE(s.total_enqueued >= 2);

    prefetch_entry_t out[4];
    uint32_t n = prefetch_dispatch(&s, 4, out);
    ASSERT_TRUE(n >= 2);

    /* Highest priority should be block 1 (ema ≈ 0.9) */
    ASSERT_EQ(out[0].block_id, 1ULL);
    ASSERT_EQ(out[0].target_tier, TIER_GPU_HBM);

    prefetch_destroy(&s);
    attn_tracker_destroy(&tracker);
}

static void test_scan_skips_gpu_blocks(void)
{
    attention_tracker_t tracker;
    ASSERT_OK(attn_tracker_init(&tracker, 256, 0.9f));

    attn_tracker_register(&tracker, 10);
    for (int i = 0; i < 20; i++) {
        attn_tracker_update(&tracker, 10, 0.9f);
        attn_tracker_step_done(&tracker);
    }

    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, &tracker, 16, 0.5f, 0.25f, 64));

    /* Block is already on GPU → should be skipped */
    prefetch_block_info_t info = {10, TIER_GPU_HBM};
    prefetch_scan_blocks(&s, &info, 1);
    ASSERT_EQ(s.total_enqueued, 0ULL);

    prefetch_destroy(&s);
    attn_tracker_destroy(&tracker);
}

static void test_scan_storage_to_dram(void)
{
    attention_tracker_t tracker;
    ASSERT_OK(attn_tracker_init(&tracker, 256, 0.9f));

    attn_tracker_register(&tracker, 10);
    for (int i = 0; i < 50; i++) {
        attn_tracker_update(&tracker, 10, 0.35f);
        attn_tracker_step_done(&tracker);
    }

    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, &tracker, 16, 0.5f, 0.2f, 64));

    /* Block on NVM with ema ≈ 0.35: above DRAM threshold (0.2), below GPU (0.5) */
    prefetch_block_info_t info = {10, TIER_NVM};
    prefetch_scan_blocks(&s, &info, 1);
    ASSERT_EQ(s.total_enqueued, 1ULL);

    prefetch_entry_t out[1];
    uint32_t n = prefetch_dispatch(&s, 1, out);
    ASSERT_EQ(n, 1u);
    ASSERT_EQ(out[0].target_tier, TIER_HOST_DRAM);
    ASSERT_TRUE(out[0].priority < 0.35f);  /* priority = ema * 0.5 for DRAM target */

    prefetch_destroy(&s);
    attn_tracker_destroy(&tracker);
}

static void test_heap_full(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 8, 0.5f, 0.25f, 4));  /* heap_cap = 4 */

    for (int i = 0; i < 4; i++)
        ASSERT_OK(prefetch_add_candidate(&s, (uint64_t)i, TIER_HOST_DRAM,
                                         TIER_GPU_HBM, (float)i));

    /* 5th entry should fail */
    int rc = prefetch_add_candidate(&s, 99, TIER_HOST_DRAM, TIER_GPU_HBM, 0.5f);
    ASSERT_EQ(rc, ORCHKV_ERR_TIER_FULL);

    prefetch_destroy(&s);
}

static void test_multi_step_stats(void)
{
    prefetch_scheduler_t s;
    ASSERT_OK(prefetch_init(&s, NULL, 8, 0.5f, 0.25f, 64));

    /* Step 1: dispatch 3, hit 2, wasted 1 */
    prefetch_add_candidate(&s, 1, TIER_HOST_DRAM, TIER_GPU_HBM, 0.9f);
    prefetch_add_candidate(&s, 2, TIER_HOST_DRAM, TIER_GPU_HBM, 0.8f);
    prefetch_add_candidate(&s, 3, TIER_HOST_DRAM, TIER_GPU_HBM, 0.7f);

    prefetch_entry_t out[8];
    prefetch_dispatch(&s, 8, out);
    prefetch_notify_hit(&s, 1);
    prefetch_notify_hit(&s, 2);
    prefetch_step_reset(&s);

    /* Step 2: dispatch 2, hit 1, wasted 1 */
    prefetch_add_candidate(&s, 4, TIER_HOST_DRAM, TIER_GPU_HBM, 0.6f);
    prefetch_add_candidate(&s, 5, TIER_HOST_DRAM, TIER_GPU_HBM, 0.5f);
    prefetch_dispatch(&s, 8, out);
    prefetch_notify_hit(&s, 4);
    prefetch_step_reset(&s);

    prefetch_stats_t stats;
    prefetch_get_stats(&s, &stats);

    ASSERT_EQ(stats.total_dispatched, 5ULL);
    ASSERT_EQ(stats.prefetch_hits, 3ULL);
    ASSERT_EQ(stats.prefetch_wasted, 2ULL);
    ASSERT_NEAR(stats.hit_rate, 0.6f, 0.01f);

    prefetch_destroy(&s);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_prefetch_scheduler ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_add_candidate_basic);
    RUN_TEST(test_heap_ordering);
    RUN_TEST(test_dispatch_respects_budget);
    RUN_TEST(test_dispatch_empty);
    RUN_TEST(test_notify_hit);
    RUN_TEST(test_step_reset_wasted);
    RUN_TEST(test_hit_rate);
    RUN_TEST(test_scan_with_tracker);
    RUN_TEST(test_scan_skips_gpu_blocks);
    RUN_TEST(test_scan_storage_to_dram);
    RUN_TEST(test_heap_full);
    RUN_TEST(test_multi_step_stats);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
