#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/hotcold_classifier.h"

static int tests_run    = 0;
static int tests_passed = 0;

#define RUN_TEST(fn)                                                        \
    do {                                                                    \
        printf("  %-50s", #fn);                                             \
        fflush(stdout);                                                     \
        tests_run++;                                                        \
        fn();                                                               \
        tests_passed++;                                                     \
        printf(" PASS\n");                                                  \
    } while (0)

#define ASSERT_OK(expr)       assert((expr) == ORCHKV_OK)
#define ASSERT_EQ(a, b)       assert((a) == (b))
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_NEAR(a, b, e)  assert(fabsf((a) - (b)) < (e))

/* Helper: create tracker + classifier pair with defaults */
static void setup(attention_tracker_t *t, hotcold_classifier_t *c,
                  const hcc_params_t *params)
{
    ASSERT_OK(attn_tracker_init(t, 256, 0.9f));
    ASSERT_OK(hcc_init(c, t, params));
}

static void setup_default(attention_tracker_t *t, hotcold_classifier_t *c)
{
    hcc_params_t p;
    hcc_params_default(&p);
    setup(t, c, &p);
}

static void teardown(attention_tracker_t *t, hotcold_classifier_t *c)
{
    hcc_destroy(c);
    attn_tracker_destroy(t);
}

/* Register a block in both tracker and classifier */
static void reg_block(attention_tracker_t *t, hotcold_classifier_t *c,
                      uint64_t id, uint8_t flags)
{
    attn_tracker_register(t, id);
    hcc_register(c, id, flags);
}

/* ======================================================================== */

static void test_init_destroy(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    setup_default(&t, &c);

    ASSERT_TRUE(c.capacity >= 256);
    ASSERT_NEAR(c.alpha, 0.5f, 1e-6f);
    ASSERT_NEAR(c.beta,  0.3f, 1e-6f);
    ASSERT_NEAR(c.gamma, 0.2f, 1e-6f);
    ASSERT_NEAR(c.threshold_hot,  0.5f, 1e-6f);
    ASSERT_NEAR(c.threshold_warm, 0.2f, 1e-6f);

    teardown(&t, &c);
}

static void test_decay_table(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.recency_tau = 50.0f;
    setup(&t, &c, &p);

    /* decay_table[0] = exp(0) = 1.0 */
    ASSERT_NEAR(c.decay_table[0], 1.0f, 1e-5f);

    /* decay_table[50] = exp(-1) ≈ 0.3679 */
    ASSERT_NEAR(c.decay_table[50], expf(-1.0f), 1e-4f);

    /* decay_table[100] = exp(-2) ≈ 0.1353 */
    ASSERT_NEAR(c.decay_table[100], expf(-2.0f), 1e-4f);

    teardown(&t, &c);
}

static void test_register_unregister(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    setup_default(&t, &c);

    reg_block(&t, &c, 10, KV_FLAG_NONE);
    ASSERT_EQ(hcc_get_heat(&c, 10), HEAT_COLD);  /* default for new block */
    ASSERT_NEAR(hcc_get_score(&c, 10), 0.0f, 1e-6f);

    hcc_unregister(&c, 10);
    ASSERT_EQ(hcc_get_heat(&c, 10), HEAT_COLD);  /* unregistered → default */

    float score;
    HeatLevel level;
    int rc = hcc_get(&c, 10, &score, &level);
    ASSERT_EQ(rc, ORCHKV_ERR_NOT_FOUND);

    teardown(&t, &c);
}

static void test_attn_sink_always_hot(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    setup_default(&t, &c);

    reg_block(&t, &c, 1, KV_FLAG_ATTN_SINK);

    /* Even without any attention updates, ATTN_SINK → HOT */
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    /* After many idle steps, still HOT */
    for (int i = 0; i < 100; i++)
        attn_tracker_step_done(&t);
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    teardown(&t, &c);
}

static void test_hotness_formula_pure_attention(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 1.0f; p.beta = 0.0f; p.gamma = 0.0f;
    p.threshold_hot = 0.5f; p.threshold_warm = 0.2f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);

    /* Pump a high attention weight through tracker */
    attn_tracker_update(&t, 1, 0.8f);
    attn_tracker_step_done(&t);
    hcc_update_all(&c);

    float score = hcc_get_score(&c, 1);
    /* EMA after 1 step: 0.9 * 0 + 0.1 * 0.8 = 0.08 */
    /* hotness = 1.0 * 0.08 = 0.08 → COLD */
    ASSERT_NEAR(score, 0.08f, 1e-4f);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_COLD);

    /* After many steps with constant weight, EMA → 0.8, hotness → 0.8 → HOT */
    for (int i = 0; i < 200; i++) {
        attn_tracker_update(&t, 1, 0.8f);
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);
    score = hcc_get_score(&c, 1);
    ASSERT_NEAR(score, 0.8f, 0.05f);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    teardown(&t, &c);
}

static void test_hotness_formula_pure_recency(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 0.0f; p.beta = 1.0f; p.gamma = 0.0f;
    p.recency_tau = 10.0f;
    p.threshold_hot = 0.5f; p.threshold_warm = 0.2f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);

    /* Touch block at step 0 */
    attn_tracker_update(&t, 1, 0.1f);
    attn_tracker_step_done(&t);

    /* At step 1: gap = 1 - 0 = 1, recency = exp(-1/10) ≈ 0.905 → HOT */
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    /* Advance 50 steps without touching: gap = 51, recency = exp(-51/10) ≈ 0 → COLD */
    for (int i = 0; i < 50; i++)
        attn_tracker_step_done(&t);
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_COLD);

    teardown(&t, &c);
}

static void test_hotness_formula_pure_frequency(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 0.0f; p.beta = 0.0f; p.gamma = 1.0f;
    p.threshold_hot = 0.5f; p.threshold_warm = 0.2f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);

    /* Touch block every step for 10 steps out of 10 → freq = 1.0 → HOT */
    for (int i = 0; i < 10; i++) {
        attn_tracker_update(&t, 1, 0.1f);
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);
    float score = hcc_get_score(&c, 1);
    /* freq = 10 / 10 = 1.0, hotness = 1.0 * 1.0 = 1.0 → HOT */
    ASSERT_NEAR(score, 1.0f, 0.01f);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    /* Now stop touching for 90 steps → freq = 10 / 100 = 0.1 → COLD */
    for (int i = 0; i < 90; i++)
        attn_tracker_step_done(&t);
    hcc_update_all(&c);
    score = hcc_get_score(&c, 1);
    ASSERT_NEAR(score, 0.1f, 0.01f);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_COLD);

    teardown(&t, &c);
}

static void test_three_level_classification(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 1.0f; p.beta = 0.0f; p.gamma = 0.0f;
    p.threshold_hot  = 0.6f;
    p.threshold_warm = 0.3f;
    setup(&t, &c, &p);

    /* Three blocks with different attention levels */
    reg_block(&t, &c, 1, KV_FLAG_NONE);  /* will be HOT */
    reg_block(&t, &c, 2, KV_FLAG_NONE);  /* will be WARM */
    reg_block(&t, &c, 3, KV_FLAG_NONE);  /* will be COLD */

    for (int i = 0; i < 200; i++) {
        attn_tracker_update(&t, 1, 0.9f);  /* converge EMA → 0.9 */
        attn_tracker_update(&t, 2, 0.4f);  /* converge EMA → 0.4 */
        attn_tracker_update(&t, 3, 0.1f);  /* converge EMA → 0.1 */
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);

    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);
    ASSERT_EQ(hcc_get_heat(&c, 2), HEAT_WARM);
    ASSERT_EQ(hcc_get_heat(&c, 3), HEAT_COLD);

    /* Verify ordering: score(1) > score(2) > score(3) */
    ASSERT_TRUE(hcc_get_score(&c, 1) > hcc_get_score(&c, 2));
    ASSERT_TRUE(hcc_get_score(&c, 2) > hcc_get_score(&c, 3));

    teardown(&t, &c);
}

static void test_update_block_single(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 1.0f; p.beta = 0.0f; p.gamma = 0.0f;
    p.threshold_hot = 0.5f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);
    reg_block(&t, &c, 2, KV_FLAG_NONE);

    for (int i = 0; i < 100; i++) {
        attn_tracker_update(&t, 1, 0.8f);
        attn_tracker_update(&t, 2, 0.8f);
        attn_tracker_step_done(&t);
    }

    /* Only update block 1, not block 2 */
    hcc_update_block(&c, 1);

    ASSERT_TRUE(hcc_get_score(&c, 1) > 0.5f);
    /* Block 2 hasn't been updated via hcc yet, still at initial 0 */
    ASSERT_NEAR(hcc_get_score(&c, 2), 0.0f, 1e-6f);

    teardown(&t, &c);
}

static void test_set_thresholds_runtime(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 1.0f; p.beta = 0.0f; p.gamma = 0.0f;
    p.threshold_hot = 0.8f; p.threshold_warm = 0.4f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);
    for (int i = 0; i < 200; i++) {
        attn_tracker_update(&t, 1, 0.6f);
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);
    /* score ≈ 0.6, threshold_hot = 0.8 → WARM */
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_WARM);

    /* Lower the threshold → block becomes HOT */
    hcc_set_thresholds(&c, 0.5f, 0.2f);
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    teardown(&t, &c);
}

static void test_set_weights_runtime(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 1.0f; p.beta = 0.0f; p.gamma = 0.0f;
    p.threshold_hot = 0.5f; p.threshold_warm = 0.2f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);
    for (int i = 0; i < 200; i++) {
        attn_tracker_update(&t, 1, 0.8f);
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    /* Switch to frequency-only: block accessed 200/200 steps → freq = 1.0 */
    hcc_set_weights(&c, 0.0f, 0.0f, 1.0f);
    hcc_update_all(&c);
    float score = hcc_get_score(&c, 1);
    ASSERT_NEAR(score, 1.0f, 0.01f);

    teardown(&t, &c);
}

static void test_stats(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    p.alpha = 1.0f; p.beta = 0.0f; p.gamma = 0.0f;
    p.threshold_hot = 0.6f; p.threshold_warm = 0.3f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_ATTN_SINK);  /* forced HOT */
    reg_block(&t, &c, 2, KV_FLAG_NONE);        /* will be WARM */
    reg_block(&t, &c, 3, KV_FLAG_NONE);        /* will be COLD */

    for (int i = 0; i < 200; i++) {
        attn_tracker_update(&t, 2, 0.5f);  /* → EMA ≈ 0.5 → WARM */
        attn_tracker_update(&t, 3, 0.1f);  /* → EMA ≈ 0.1 → COLD */
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);

    hcc_stats_t stats;
    hcc_get_stats(&c, &stats);

    ASSERT_EQ(stats.n_hot, 1u);   /* block 1 (ATTN_SINK) */
    ASSERT_EQ(stats.n_warm, 1u);  /* block 2 */
    ASSERT_EQ(stats.n_cold, 1u);  /* block 3 */
    ASSERT_EQ(stats.n_attn_sink, 1u);
    ASSERT_TRUE(stats.max_hotness >= stats.avg_hotness);
    ASSERT_TRUE(stats.avg_hotness >= stats.min_hotness);

    teardown(&t, &c);
}

static void test_cold_after_idle(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    /* Use balanced weights */
    p.recency_tau = 20.0f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);

    /* Make block HOT by active attention */
    for (int i = 0; i < 100; i++) {
        attn_tracker_update(&t, 1, 1.0f);
        attn_tracker_step_done(&t);
    }
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);

    /* Let it go idle: EMA decays, recency decays, freq drops */
    for (int i = 0; i < 500; i++)
        attn_tracker_step_done(&t);
    hcc_update_all(&c);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_COLD);

    teardown(&t, &c);
}

static void test_mixed_formula(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t p;
    hcc_params_default(&p);
    /* α=0.5, β=0.3, γ=0.2, tau=50 */
    p.threshold_hot  = 0.4f;
    p.threshold_warm = 0.15f;
    setup(&t, &c, &p);

    reg_block(&t, &c, 1, KV_FLAG_NONE);

    /* Step 0: update with 0.5 */
    attn_tracker_update(&t, 1, 0.5f);
    attn_tracker_step_done(&t);  /* step = 1 now */
    hcc_update_all(&c);

    float score = hcc_get_score(&c, 1);
    /* attn_ema = 0.9*0 + 0.1*0.5 = 0.05
     * recency = exp(-(1 - 0)/50) = exp(-0.02) ≈ 0.9802
     * freq = 1/1 = 1.0 (capped at 1.0)
     * hotness = 0.5*0.05 + 0.3*0.9802 + 0.2*1.0
     *         = 0.025 + 0.29406 + 0.2 = 0.51906 */
    ASSERT_NEAR(score, 0.519f, 0.01f);
    ASSERT_EQ(hcc_get_heat(&c, 1), HEAT_HOT);  /* 0.519 ≥ 0.4 → HOT */

    teardown(&t, &c);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_hotcold_classifier ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_decay_table);
    RUN_TEST(test_register_unregister);
    RUN_TEST(test_attn_sink_always_hot);
    RUN_TEST(test_hotness_formula_pure_attention);
    RUN_TEST(test_hotness_formula_pure_recency);
    RUN_TEST(test_hotness_formula_pure_frequency);
    RUN_TEST(test_three_level_classification);
    RUN_TEST(test_update_block_single);
    RUN_TEST(test_set_thresholds_runtime);
    RUN_TEST(test_set_weights_runtime);
    RUN_TEST(test_stats);
    RUN_TEST(test_cold_after_idle);
    RUN_TEST(test_mixed_formula);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
