/*
 * C9: Hot/Cold Classification Pipeline Integration Test
 *
 * Wires C1 (attention_tracker) → C2 (hotcold_classifier) →
 *       C3 (adaptive_threshold) → C4 (eviction_policy)
 * and verifies cross-component behaviour end-to-end.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/hotcold_classifier.h"
#include "scheduler/adaptive_threshold.h"
#include "scheduler/eviction_policy.h"

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
 *  Shared fixture: tracker + classifier + threshold + eviction
 * ======================================================================== */

typedef struct fixture {
    attention_tracker_t   tracker;
    hotcold_classifier_t  classifier;
    adaptive_threshold_t  threshold;
    eviction_policy_t     evpol;
} fixture_t;

static void fixture_init(fixture_t *f)
{
    attn_tracker_init(&f->tracker, 256, 0.9f);

    hcc_params_t hp;
    hcc_params_default(&hp);
    hcc_init(&f->classifier, &f->tracker, &hp);

    athresh_params_t ap;
    athresh_params_default(&ap);
    ap.cooldown_sec = 0.0;
    athresh_init(&f->threshold, &f->classifier, &ap);

    evpol_init(&f->evpol, &f->classifier, 8, 64, 0.7f, 0.3f);
}

static void fixture_destroy(fixture_t *f)
{
    evpol_destroy(&f->evpol);
    athresh_destroy(&f->threshold);
    hcc_destroy(&f->classifier);
    attn_tracker_destroy(&f->tracker);
}

/* ========================================================================
 *  Block helpers
 * ======================================================================== */

#define MAX_BLK 16
static kv_block_t g_blks[MAX_BLK];

static kv_block_t *make_blk(int idx, uint64_t id, StorageTier tier,
                             uint8_t flags)
{
    memset(&g_blks[idx], 0, sizeof(kv_block_t));
    g_blks[idx].block_id    = id;
    g_blks[idx].request_id  = 1;
    g_blks[idx].layer_id    = 0;
    g_blks[idx].head_id     = 0;
    g_blks[idx].token_start = (uint32_t)(id * 64);
    g_blks[idx].token_count = 64;
    g_blks[idx].tier        = tier;
    g_blks[idx].state       = KV_STATE_HOT;
    g_blks[idx].flags       = flags;
    pthread_rwlock_init(&g_blks[idx].lock, NULL);
    return &g_blks[idx];
}

/* ========================================================================
 *  Tests
 * ======================================================================== */

/* C1 → query: single block update + readback */
static void test_attn_tracker_basic(void)
{
    fixture_t f;
    fixture_init(&f);

    attn_tracker_register(&f.tracker, 100);
    attn_tracker_update(&f.tracker, 100, 0.75f);
    attn_tracker_step_done(&f.tracker);

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&f.tracker, 100, &s));
    ASSERT_TRUE(s.active);
    ASSERT_TRUE(s.ema > 0.0f);
    ASSERT_EQ(s.query_hits, 1u);

    fixture_destroy(&f);
}

/* C1: EMA converges with repeated high scores, decays when idle */
static void test_attn_tracker_ema(void)
{
    fixture_t f;
    fixture_init(&f);

    attn_tracker_register(&f.tracker, 200);

    for (int i = 0; i < 50; i++) {
        attn_tracker_update(&f.tracker, 200, 1.0f);
        attn_tracker_step_done(&f.tracker);
    }

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&f.tracker, 200, &s));
    float ema_high = s.ema;
    ASSERT_TRUE(ema_high > 0.8f);

    /* Now let it decay (no updates for 50 steps) */
    for (int i = 0; i < 50; i++)
        attn_tracker_step_done(&f.tracker);

    ASSERT_OK(attn_tracker_get(&f.tracker, 200, &s));
    ASSERT_TRUE(s.ema < ema_high * 0.1f);

    fixture_destroy(&f);
}

/* C1 → C2: three blocks with different attention → Hot/Warm/Cold */
static void test_classifier_3level(void)
{
    fixture_t f;
    fixture_init(&f);

    attn_tracker_register(&f.tracker, 10);
    attn_tracker_register(&f.tracker, 20);
    attn_tracker_register(&f.tracker, 30);
    hcc_register(&f.classifier, 10, KV_FLAG_NONE);
    hcc_register(&f.classifier, 20, KV_FLAG_NONE);
    hcc_register(&f.classifier, 30, KV_FLAG_NONE);

    for (int step = 0; step < 80; step++) {
        attn_tracker_update(&f.tracker, 10, 0.95f);  /* hot */
        attn_tracker_update(&f.tracker, 20, 0.15f);  /* warm */
        /* block 30: no attention → cold */
        attn_tracker_step_done(&f.tracker);
    }

    hcc_update_all(&f.classifier);

    ASSERT_EQ(hcc_get_heat(&f.classifier, 10), HEAT_HOT);
    ASSERT_EQ(hcc_get_heat(&f.classifier, 30), HEAT_COLD);

    float s10 = hcc_get_score(&f.classifier, 10);
    float s20 = hcc_get_score(&f.classifier, 20);
    float s30 = hcc_get_score(&f.classifier, 30);
    ASSERT_TRUE(s10 > s20);
    ASSERT_TRUE(s20 > s30);

    fixture_destroy(&f);
}

/* C2: ATTN_SINK flag forces HOT regardless of attention */
static void test_classifier_attn_sink(void)
{
    fixture_t f;
    fixture_init(&f);

    attn_tracker_register(&f.tracker, 50);
    hcc_register(&f.classifier, 50, KV_FLAG_ATTN_SINK);

    /* No attention at all for 100 steps */
    for (int i = 0; i < 100; i++)
        attn_tracker_step_done(&f.tracker);

    hcc_update_all(&f.classifier);

    ASSERT_EQ(hcc_get_heat(&f.classifier, 50), HEAT_HOT);

    fixture_destroy(&f);
}

/* C3: GPU > HWM → threshold_hot raises → classifier reclassifies */
static void test_adaptive_threshold_hwm(void)
{
    fixture_t f;
    fixture_init(&f);

    float orig_hot = athresh_get_hot(&f.threshold);

    /* GPU overloaded, DRAM mid-range */
    athresh_update(&f.threshold, 0.95f, 0.80f);

    float new_hot = athresh_get_hot(&f.threshold);
    ASSERT_TRUE(new_hot > orig_hot);

    /* Verify the classifier received the updated threshold */
    ASSERT_NEAR(f.classifier.threshold_hot, new_hot, 1e-5f);

    fixture_destroy(&f);
}

/* C3: GPU < LWM → threshold_hot lowers */
static void test_adaptive_threshold_lwm(void)
{
    fixture_t f;
    fixture_init(&f);

    float orig_hot = athresh_get_hot(&f.threshold);

    /* GPU underutilised, DRAM mid-range */
    athresh_update(&f.threshold, 0.50f, 0.80f);

    float new_hot = athresh_get_hot(&f.threshold);
    ASSERT_TRUE(new_hot < orig_hot);

    fixture_destroy(&f);
}

/* C1→C2→C4: select N victims sorted by eviction_score descending */
static void test_eviction_select_n(void)
{
    fixture_t f;
    fixture_init(&f);

    /* 6 GPU blocks: 2 hot, 4 cold */
    for (int i = 0; i < 6; i++) {
        uint64_t id = (uint64_t)(10 + i);
        kv_block_t *b = make_blk(i, id, TIER_GPU_HBM, KV_FLAG_NONE);
        attn_tracker_register(&f.tracker, id);
        hcc_register(&f.classifier, id, KV_FLAG_NONE);
        evpol_lru_touch(&f.evpol, b);
    }

    for (int step = 0; step < 50; step++) {
        attn_tracker_update(&f.tracker, 10, 0.9f);
        attn_tracker_update(&f.tracker, 11, 0.8f);
        attn_tracker_step_done(&f.tracker);
    }

    hcc_update_all(&f.classifier);

    eviction_candidate_t cands[4];
    uint32_t n = evpol_select_gpu_victims(&f.evpol, 4, cands);
    ASSERT_EQ(n, 4u);

    /* Scores must be descending */
    for (uint32_t i = 1; i < n; i++)
        ASSERT_TRUE(cands[i - 1].score >= cands[i].score);

    /* Hot blocks (10, 11) should NOT be among the 4 victims */
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_TRUE(cands[i].block->block_id != 10);
        ASSERT_TRUE(cands[i].block->block_id != 11);
    }

    fixture_destroy(&f);
}

/* C4: pinned blocks are never selected as victims */
static void test_eviction_skip_pinned(void)
{
    fixture_t f;
    fixture_init(&f);

    for (int i = 0; i < 4; i++) {
        uint64_t id = (uint64_t)(30 + i);
        uint8_t flags = (i < 2) ? KV_FLAG_PIN : KV_FLAG_NONE;
        kv_block_t *b = make_blk(i, id, TIER_GPU_HBM, flags);
        attn_tracker_register(&f.tracker, id);
        hcc_register(&f.classifier, id, flags);
        evpol_lru_touch(&f.evpol, b);
    }

    for (int step = 0; step < 20; step++)
        attn_tracker_step_done(&f.tracker);

    hcc_update_all(&f.classifier);

    eviction_candidate_t cands[4];
    uint32_t n = evpol_select_gpu_victims(&f.evpol, 4, cands);

    /* Only non-pinned blocks (32, 33) can be selected */
    ASSERT_EQ(n, 2u);
    for (uint32_t i = 0; i < n; i++)
        ASSERT_TRUE(!(cands[i].block->flags & KV_FLAG_PIN));

    fixture_destroy(&f);
}

/* C4: LRU touch moves block to head */
static void test_lru_touch(void)
{
    fixture_t f;
    fixture_init(&f);

    kv_block_t *a = make_blk(0, 40, TIER_GPU_HBM, KV_FLAG_NONE);
    kv_block_t *b = make_blk(1, 41, TIER_GPU_HBM, KV_FLAG_NONE);
    kv_block_t *c = make_blk(2, 42, TIER_GPU_HBM, KV_FLAG_NONE);

    evpol_lru_touch(&f.evpol, a);
    evpol_lru_touch(&f.evpol, b);
    evpol_lru_touch(&f.evpol, c);
    /* Order: head=c → b → a=tail */

    ASSERT_EQ(f.evpol.lru_tail, a);
    ASSERT_EQ(f.evpol.lru_head, c);

    /* Touch a → moves to head */
    evpol_lru_touch(&f.evpol, a);
    /* Order: head=a → c → b=tail */

    ASSERT_EQ(f.evpol.lru_head, a);
    ASSERT_EQ(f.evpol.lru_tail, b);

    fixture_destroy(&f);
}

/* ========================================================================
 *  End-to-end integration: attention → classification → threshold
 *  adjustment → victim selection in a single pipeline
 * ======================================================================== */

static void test_pipeline_integration(void)
{
    fixture_t f;
    fixture_init(&f);

    /* 8 GPU blocks */
    for (int i = 0; i < 8; i++) {
        uint64_t id = (uint64_t)(100 + i);
        kv_block_t *b = make_blk(i, id, TIER_GPU_HBM, KV_FLAG_NONE);
        attn_tracker_register(&f.tracker, id);
        hcc_register(&f.classifier, id, KV_FLAG_NONE);
        evpol_lru_touch(&f.evpol, b);
    }

    /* Blocks 100-103 get high attention; 104-107 get none */
    for (int step = 0; step < 60; step++) {
        for (int i = 0; i < 4; i++)
            attn_tracker_update(&f.tracker, (uint64_t)(100 + i), 0.9f);
        attn_tracker_step_done(&f.tracker);
    }

    /* 1. Classify */
    hcc_update_all(&f.classifier);

    hcc_stats_t hs;
    hcc_get_stats(&f.classifier, &hs);
    ASSERT_TRUE(hs.n_hot >= 2);
    ASSERT_TRUE(hs.n_cold >= 2);

    /* 2. Simulate GPU overload → raise threshold */
    float hot_before = athresh_get_hot(&f.threshold);
    athresh_update(&f.threshold, 0.95f, 0.80f);
    float hot_after = athresh_get_hot(&f.threshold);
    ASSERT_TRUE(hot_after > hot_before);

    /* 3. Re-classify with updated thresholds */
    hcc_update_all(&f.classifier);

    /* 4. Select victims — should pick cold blocks */
    eviction_candidate_t cands[4];
    uint32_t n = evpol_select_gpu_victims(&f.evpol, 4, cands);
    ASSERT_TRUE(n >= 2);

    /* All victims should be cold blocks (104-107) */
    for (uint32_t i = 0; i < n; i++)
        ASSERT_TRUE(cands[i].block->block_id >= 104);

    /* Scores should be descending */
    for (uint32_t i = 1; i < n; i++)
        ASSERT_TRUE(cands[i - 1].score >= cands[i].score);

    /* Hot blocks should have higher hotness than cold ones */
    float hot_score = hcc_get_score(&f.classifier, 100);
    float cold_score = hcc_get_score(&f.classifier, 107);
    ASSERT_TRUE(hot_score > cold_score * 3.0f);

    fixture_destroy(&f);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_hotcold (C9: integration) ===\n");

    RUN_TEST(test_attn_tracker_basic);
    RUN_TEST(test_attn_tracker_ema);
    RUN_TEST(test_classifier_3level);
    RUN_TEST(test_classifier_attn_sink);
    RUN_TEST(test_adaptive_threshold_hwm);
    RUN_TEST(test_adaptive_threshold_lwm);
    RUN_TEST(test_eviction_select_n);
    RUN_TEST(test_eviction_skip_pinned);
    RUN_TEST(test_lru_touch);
    RUN_TEST(test_pipeline_integration);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
