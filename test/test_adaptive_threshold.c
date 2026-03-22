#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/hotcold_classifier.h"
#include "scheduler/adaptive_threshold.h"

static int tests_run = 0, tests_passed = 0;

#define RUN_TEST(fn) do {                           \
    printf("  %-50s", #fn); fflush(stdout);         \
    tests_run++; fn(); tests_passed++;              \
    printf(" PASS\n"); } while (0)

#define ASSERT_OK(e)          assert((e) == ORCHKV_OK)
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_FALSE(c)       assert(!(c))
#define ASSERT_NEAR(a, b, e)  assert(fabsf((a)-(b)) < (e))

/* ======================================================================== */

static void test_init_destroy(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);

    ASSERT_OK(athresh_init(&a, NULL, &p));
    ASSERT_NEAR(athresh_get_hot(&a),  0.5f, 1e-6f);
    ASSERT_NEAR(athresh_get_warm(&a), 0.2f, 1e-6f);
    athresh_destroy(&a);
}

static void test_should_demote_basic(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.gpu_hwm = 0.9f; p.dram_hwm = 0.9f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    ASSERT_TRUE(athresh_should_demote_gpu(&a, 0.95f));
    ASSERT_FALSE(athresh_should_demote_gpu(&a, 0.80f));
    ASSERT_FALSE(athresh_should_demote_gpu(&a, 0.90f));  /* not strictly > */

    ASSERT_TRUE(athresh_should_demote_dram(&a, 0.91f));
    ASSERT_FALSE(athresh_should_demote_dram(&a, 0.50f));

    athresh_destroy(&a);
}

static void test_raise_on_overload(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec = 0.0;  /* disable cooldown for test */
    p.adjust_step  = 0.05f;
    p.threshold_hot = 0.5f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* GPU overloaded, DRAM in mid-range (between LWM and HWM) */
    bool changed = athresh_update(&a, 0.95f, 0.80f);
    ASSERT_TRUE(changed);
    ASSERT_NEAR(athresh_get_hot(&a), 0.55f, 1e-5f);

    /* DRAM in mid-range → threshold_warm unchanged */
    ASSERT_NEAR(athresh_get_warm(&a), 0.2f, 1e-5f);

    athresh_destroy(&a);
}

static void test_lower_on_underutil(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec = 0.0;
    p.adjust_step  = 0.05f;
    p.threshold_hot = 0.5f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* GPU under-utilised, DRAM in mid-range */
    bool changed = athresh_update(&a, 0.50f, 0.80f);
    ASSERT_TRUE(changed);
    ASSERT_NEAR(athresh_get_hot(&a), 0.45f, 1e-5f);

    athresh_destroy(&a);
}

static void test_dram_threshold_adjust(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec = 0.0;
    p.adjust_step  = 0.03f;
    p.threshold_warm = 0.2f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* DRAM overloaded → threshold_warm rises */
    athresh_update(&a, 0.80f, 0.95f);
    ASSERT_NEAR(athresh_get_warm(&a), 0.23f, 1e-5f);

    /* DRAM under-utilised → threshold_warm drops */
    athresh_update(&a, 0.80f, 0.60f);
    ASSERT_NEAR(athresh_get_warm(&a), 0.20f, 1e-5f);

    athresh_destroy(&a);
}

static void test_clamp_prevents_extremes(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec  = 0.0;
    p.adjust_step   = 0.1f;
    p.threshold_hot = 0.85f;
    p.max_hot       = 0.9f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* Try to raise beyond max_hot */
    athresh_update(&a, 0.99f, 0.80f);
    ASSERT_NEAR(athresh_get_hot(&a), 0.9f, 1e-5f);  /* clamped */

    /* Another raise should not change */
    bool changed = athresh_update(&a, 0.99f, 0.80f);
    ASSERT_FALSE(changed);

    athresh_destroy(&a);
}

static void test_clamp_lower_bound(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec  = 0.0;
    p.adjust_step   = 0.1f;
    p.threshold_hot = 0.25f;
    p.min_hot       = 0.2f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* Try to lower below min_hot */
    athresh_update(&a, 0.50f, 0.80f);
    ASSERT_NEAR(athresh_get_hot(&a), 0.2f, 1e-5f);  /* clamped */

    athresh_destroy(&a);
}

static void test_cooldown_blocks_rapid_changes(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec  = 1.0;   /* 1 second cooldown */
    p.adjust_step   = 0.05f;
    p.threshold_hot = 0.5f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* First update should be blocked by cooldown (just initialised) */
    bool changed = athresh_update(&a, 0.95f, 0.80f);
    /* May or may not change depending on timing — the init stamps "now",
     * so with cooldown_sec=1.0 the first call within the same ms is blocked. */
    float hot1 = athresh_get_hot(&a);

    /* Immediate second call definitely blocked */
    changed = athresh_update(&a, 0.95f, 0.80f);
    float hot2 = athresh_get_hot(&a);
    ASSERT_NEAR(hot1, hot2, 1e-6f);  /* no change */

    athresh_destroy(&a);
}

static void test_no_change_in_midrange(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec = 0.0;
    p.gpu_hwm = 0.9f; p.gpu_lwm = 0.7f;
    ASSERT_OK(athresh_init(&a, NULL, &p));

    /* Both at 80% — between LWM(0.7) and HWM(0.9) → no adjustment */
    bool changed = athresh_update(&a, 0.80f, 0.80f);
    ASSERT_FALSE(changed);

    athresh_destroy(&a);
}

static void test_push_to_classifier(void)
{
    attention_tracker_t t;
    hotcold_classifier_t c;
    hcc_params_t hp;
    hcc_params_default(&hp);

    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));
    ASSERT_OK(hcc_init(&c, &t, &hp));

    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    p.cooldown_sec  = 0.0;
    p.adjust_step   = 0.05f;
    p.threshold_hot = 0.5f;
    ASSERT_OK(athresh_init(&a, &c, &p));

    /* Initial push: classifier should have 0.5 / 0.2 */
    ASSERT_NEAR(c.threshold_hot,  0.5f, 1e-5f);
    ASSERT_NEAR(c.threshold_warm, 0.2f, 1e-5f);

    /* Trigger GPU overload → hot threshold rises → pushed to classifier */
    athresh_update(&a, 0.95f, 0.80f);
    ASSERT_NEAR(c.threshold_hot, 0.55f, 1e-5f);

    athresh_destroy(&a);
    hcc_destroy(&c);
    attn_tracker_destroy(&t);
}

static void test_force_thresholds(void)
{
    adaptive_threshold_t a;
    athresh_params_t p;
    athresh_params_default(&p);
    ASSERT_OK(athresh_init(&a, NULL, &p));

    athresh_force_thresholds(&a, 0.7f, 0.35f);
    ASSERT_NEAR(athresh_get_hot(&a),  0.7f, 1e-5f);
    ASSERT_NEAR(athresh_get_warm(&a), 0.35f, 1e-5f);

    /* Force with out-of-range values → clamped */
    athresh_force_thresholds(&a, 99.0f, -1.0f);
    ASSERT_NEAR(athresh_get_hot(&a),  p.max_hot,  1e-5f);
    ASSERT_NEAR(athresh_get_warm(&a), p.min_warm, 1e-5f);

    athresh_destroy(&a);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_adaptive_threshold ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_should_demote_basic);
    RUN_TEST(test_raise_on_overload);
    RUN_TEST(test_lower_on_underutil);
    RUN_TEST(test_dram_threshold_adjust);
    RUN_TEST(test_clamp_prevents_extremes);
    RUN_TEST(test_clamp_lower_bound);
    RUN_TEST(test_cooldown_blocks_rapid_changes);
    RUN_TEST(test_no_change_in_midrange);
    RUN_TEST(test_push_to_classifier);
    RUN_TEST(test_force_thresholds);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
