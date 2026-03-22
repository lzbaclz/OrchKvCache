#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/hotcold_classifier.h"
#include "scheduler/eviction_policy.h"

static int tests_run = 0, tests_passed = 0;

#define RUN_TEST(fn) do {                           \
    printf("  %-50s", #fn); fflush(stdout);         \
    tests_run++; fn(); tests_passed++;              \
    printf(" PASS\n"); } while (0)

#define ASSERT_OK(e)          assert((e) == ORCHKV_OK)
#define ASSERT_EQ(a, b)       assert((a) == (b))
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_NEAR(a, b, e)  assert(fabsf((a)-(b)) < (e))

/* Shared test infrastructure */
static attention_tracker_t  g_tracker;
static hotcold_classifier_t g_classifier;
static eviction_policy_t    g_pol;

#define MAX_TEST_BLOCKS 16
static kv_block_t g_blocks[MAX_TEST_BLOCKS];

static void global_setup(void)
{
    ASSERT_OK(attn_tracker_init(&g_tracker, 256, 0.9f));
    hcc_params_t hp;
    hcc_params_default(&hp);
    hp.alpha = 1.0f; hp.beta = 0.0f; hp.gamma = 0.0f;
    hp.threshold_hot = 0.6f; hp.threshold_warm = 0.3f;
    ASSERT_OK(hcc_init(&g_classifier, &g_tracker, &hp));
    ASSERT_OK(evpol_init(&g_pol, &g_classifier, 8, 64, 0.7f, 0.3f));
    memset(g_blocks, 0, sizeof(g_blocks));
}

static void global_teardown(void)
{
    evpol_destroy(&g_pol);
    hcc_destroy(&g_classifier);
    attn_tracker_destroy(&g_tracker);
}

static void init_block(kv_block_t *blk, uint64_t id,
                       StorageTier tier, uint8_t flags)
{
    memset(blk, 0, sizeof(*blk));
    blk->block_id    = id;
    blk->request_id  = 1;
    blk->layer_id    = 0;
    blk->head_id     = 0;
    blk->token_start = (uint32_t)(id * 64);
    blk->token_count = 64;
    blk->tier        = tier;
    blk->state       = (tier == TIER_GPU_HBM) ? KV_STATE_HOT : KV_STATE_WARM;
    blk->flags       = flags;
    pthread_rwlock_init(&blk->lock, NULL);
    blk->prev = blk->next = NULL;
}

static void reg_block(kv_block_t *blk)
{
    attn_tracker_register(&g_tracker, blk->block_id);
    hcc_register(&g_classifier, blk->block_id, blk->flags);
}

/* ======================================================================== */

static void test_lru_touch_and_size(void)
{
    global_setup();
    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_GPU_HBM, KV_FLAG_NONE);

    ASSERT_EQ(evpol_lru_size(&g_pol), 0u);

    evpol_lru_touch(&g_pol, &g_blocks[0]);
    ASSERT_EQ(evpol_lru_size(&g_pol), 1u);

    evpol_lru_touch(&g_pol, &g_blocks[1]);
    ASSERT_EQ(evpol_lru_size(&g_pol), 2u);

    /* Re-touch block 0 (move to head) — size unchanged */
    evpol_lru_touch(&g_pol, &g_blocks[0]);
    ASSERT_EQ(evpol_lru_size(&g_pol), 2u);

    global_teardown();
}

static void test_lru_remove(void)
{
    global_setup();
    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_GPU_HBM, KV_FLAG_NONE);

    evpol_lru_touch(&g_pol, &g_blocks[0]);
    evpol_lru_touch(&g_pol, &g_blocks[1]);
    ASSERT_EQ(evpol_lru_size(&g_pol), 2u);

    evpol_lru_remove(&g_pol, &g_blocks[0]);
    ASSERT_EQ(evpol_lru_size(&g_pol), 1u);

    /* Remove again (should be no-op) */
    evpol_lru_remove(&g_pol, &g_blocks[0]);
    ASSERT_EQ(evpol_lru_size(&g_pol), 1u);

    global_teardown();
}

static void test_lru_order(void)
{
    global_setup();
    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[2], 30, TIER_GPU_HBM, KV_FLAG_NONE);

    /* Insert order: 10, 20, 30 → head=30, tail=10 */
    evpol_lru_touch(&g_pol, &g_blocks[0]);
    evpol_lru_touch(&g_pol, &g_blocks[1]);
    evpol_lru_touch(&g_pol, &g_blocks[2]);

    ASSERT_EQ(g_pol.lru_head->block_id, 30ULL);
    ASSERT_EQ(g_pol.lru_tail->block_id, 10ULL);

    /* Touch block 10 → moves to head: head=10, tail=20 */
    evpol_lru_touch(&g_pol, &g_blocks[0]);
    ASSERT_EQ(g_pol.lru_head->block_id, 10ULL);
    ASSERT_EQ(g_pol.lru_tail->block_id, 20ULL);

    global_teardown();
}

static void test_select_gpu_basic(void)
{
    global_setup();

    /* 3 GPU blocks with different hotness */
    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[2], 30, TIER_GPU_HBM, KV_FLAG_NONE);

    for (int i = 0; i < 3; i++) reg_block(&g_blocks[i]);

    /* Pump attention: block 10 = high, 20 = medium, 30 = low */
    for (int s = 0; s < 100; s++) {
        attn_tracker_update(&g_tracker, 10, 0.9f);
        attn_tracker_update(&g_tracker, 20, 0.5f);
        attn_tracker_update(&g_tracker, 30, 0.1f);
        attn_tracker_step_done(&g_tracker);
    }
    hcc_update_all(&g_classifier);

    /* Insert into LRU (order: 10 first, 30 last touched = MRU) */
    evpol_lru_touch(&g_pol, &g_blocks[0]);
    evpol_lru_touch(&g_pol, &g_blocks[1]);
    evpol_lru_touch(&g_pol, &g_blocks[2]);

    eviction_candidate_t cands[3];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 2, cands);

    ASSERT_EQ(n, 2u);
    /* Block 30 (lowest hotness) should have highest eviction score */
    ASSERT_EQ(cands[0].block->block_id, 30ULL);

    global_teardown();
}

static void test_skip_pinned(void)
{
    global_setup();

    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_PIN);   /* pinned */
    init_block(&g_blocks[1], 20, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[2], 30, TIER_GPU_HBM, KV_FLAG_NONE);

    for (int i = 0; i < 3; i++) {
        reg_block(&g_blocks[i]);
        evpol_lru_touch(&g_pol, &g_blocks[i]);
    }

    /* All blocks have same low attention → LRU ordering matters */
    attn_tracker_step_done(&g_tracker);
    hcc_update_all(&g_classifier);

    eviction_candidate_t cands[3];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 3, cands);

    /* Only 2 candidates (block 10 is pinned) */
    ASSERT_EQ(n, 2u);
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_TRUE(cands[i].block->block_id != 10ULL);
    }

    global_teardown();
}

static void test_skip_wrong_tier(void)
{
    global_setup();

    init_block(&g_blocks[0], 10, TIER_GPU_HBM,   KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_HOST_DRAM,  KV_FLAG_NONE);
    init_block(&g_blocks[2], 30, TIER_GPU_HBM,    KV_FLAG_NONE);

    for (int i = 0; i < 3; i++) {
        reg_block(&g_blocks[i]);
        evpol_lru_touch(&g_pol, &g_blocks[i]);
    }
    attn_tracker_step_done(&g_tracker);
    hcc_update_all(&g_classifier);

    eviction_candidate_t cands[3];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 3, cands);

    /* Only GPU blocks (10, 30) should be selected */
    ASSERT_EQ(n, 2u);
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_EQ(cands[i].block->tier, TIER_GPU_HBM);
    }

    global_teardown();
}

static void test_select_dram_victims(void)
{
    global_setup();

    init_block(&g_blocks[0], 10, TIER_HOST_DRAM, KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_HOST_DRAM, KV_FLAG_NONE);
    init_block(&g_blocks[2], 30, TIER_GPU_HBM,   KV_FLAG_NONE);

    for (int i = 0; i < 3; i++) {
        reg_block(&g_blocks[i]);
        evpol_lru_touch(&g_pol, &g_blocks[i]);
    }
    attn_tracker_step_done(&g_tracker);
    hcc_update_all(&g_classifier);

    eviction_candidate_t cands[3];
    uint32_t n = evpol_select_dram_victims(&g_pol, 3, cands);

    ASSERT_EQ(n, 2u);
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_EQ(cands[i].block->tier, TIER_HOST_DRAM);
    }

    global_teardown();
}

static void test_score_ordering(void)
{
    global_setup();

    /* 4 GPU blocks, same LRU order, different hotness */
    for (int i = 0; i < 4; i++) {
        init_block(&g_blocks[i], (uint64_t)(i + 1), TIER_GPU_HBM, KV_FLAG_NONE);
        reg_block(&g_blocks[i]);
    }

    /* Hotness: block 1=0.9, 2=0.6, 3=0.3, 4=0.1 */
    float weights[] = {0.9f, 0.6f, 0.3f, 0.1f};
    for (int s = 0; s < 100; s++) {
        for (int i = 0; i < 4; i++)
            attn_tracker_update(&g_tracker, (uint64_t)(i + 1), weights[i]);
        attn_tracker_step_done(&g_tracker);
    }
    hcc_update_all(&g_classifier);

    /* All added to LRU in same order (1 oldest, 4 newest) */
    for (int i = 0; i < 4; i++)
        evpol_lru_touch(&g_pol, &g_blocks[i]);

    eviction_candidate_t cands[4];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 4, cands);
    ASSERT_EQ(n, 4u);

    /* Candidates should be sorted by eviction score descending.
     * Block 4 has lowest hotness → highest (1-hotness) → should be first. */
    ASSERT_TRUE(cands[0].score >= cands[1].score);
    ASSERT_TRUE(cands[1].score >= cands[2].score);
    ASSERT_TRUE(cands[2].score >= cands[3].score);

    global_teardown();
}

static void test_skip_migrating(void)
{
    global_setup();

    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_NONE);
    init_block(&g_blocks[1], 20, TIER_GPU_HBM, KV_FLAG_NONE);
    g_blocks[1].state = KV_STATE_MIGRATING;

    for (int i = 0; i < 2; i++) {
        reg_block(&g_blocks[i]);
        evpol_lru_touch(&g_pol, &g_blocks[i]);
    }
    attn_tracker_step_done(&g_tracker);
    hcc_update_all(&g_classifier);

    eviction_candidate_t cands[2];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 2, cands);

    ASSERT_EQ(n, 1u);
    ASSERT_EQ(cands[0].block->block_id, 10ULL);

    global_teardown();
}

static void test_block_idx_computed(void)
{
    global_setup();

    init_block(&g_blocks[0], 10, TIER_GPU_HBM, KV_FLAG_NONE);
    g_blocks[0].token_start = 192;  /* block_idx = 192 / 64 = 3 */
    reg_block(&g_blocks[0]);
    evpol_lru_touch(&g_pol, &g_blocks[0]);
    attn_tracker_step_done(&g_tracker);
    hcc_update_all(&g_classifier);

    eviction_candidate_t cands[1];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 1, cands);
    ASSERT_EQ(n, 1u);
    ASSERT_EQ(cands[0].block_idx, 3u);

    global_teardown();
}

static void test_empty_list(void)
{
    global_setup();

    eviction_candidate_t cands[4];
    uint32_t n = evpol_select_gpu_victims(&g_pol, 4, cands);
    ASSERT_EQ(n, 0u);

    n = evpol_select_dram_victims(&g_pol, 4, cands);
    ASSERT_EQ(n, 0u);

    global_teardown();
}

static void test_set_weights(void)
{
    global_setup();

    evpol_set_weights(&g_pol, 1.0f, 0.0f);
    ASSERT_NEAR(g_pol.w_heat, 1.0f, 1e-6f);
    ASSERT_NEAR(g_pol.w_lru,  0.0f, 1e-6f);

    evpol_set_weights(&g_pol, 0.5f, 0.5f);
    ASSERT_NEAR(g_pol.w_heat, 0.5f, 1e-6f);

    global_teardown();
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_eviction_policy ===\n");

    RUN_TEST(test_lru_touch_and_size);
    RUN_TEST(test_lru_remove);
    RUN_TEST(test_lru_order);
    RUN_TEST(test_select_gpu_basic);
    RUN_TEST(test_skip_pinned);
    RUN_TEST(test_skip_wrong_tier);
    RUN_TEST(test_select_dram_victims);
    RUN_TEST(test_score_ordering);
    RUN_TEST(test_skip_migrating);
    RUN_TEST(test_block_idx_computed);
    RUN_TEST(test_empty_list);
    RUN_TEST(test_set_weights);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
