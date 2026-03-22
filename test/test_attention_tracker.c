#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include "scheduler/attention_tracker.h"

static int tests_run    = 0;
static int tests_passed = 0;

#define RUN_TEST(fn)                                                        \
    do {                                                                    \
        printf("  %-45s", #fn);                                             \
        fflush(stdout);                                                     \
        tests_run++;                                                        \
        fn();                                                               \
        tests_passed++;                                                     \
        printf(" PASS\n");                                                  \
    } while (0)

#define ASSERT_OK(expr)    assert((expr) == ORCHKV_OK)
#define ASSERT_EQ(a, b)    assert((a) == (b))
#define ASSERT_TRUE(c)     assert((c))

static int float_near(float a, float b, float eps)
{
    return fabsf(a - b) < eps;
}

/* ======================================================================== */

static void test_init_destroy(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 128, 0.9f));
    ASSERT_TRUE(t.capacity >= 128);
    ASSERT_TRUE((t.capacity & (t.capacity - 1)) == 0);   /* power of 2 */
    ASSERT_EQ(t.current_step, 0ULL);
    ASSERT_TRUE(float_near(t.ema_lambda, 0.9f, 1e-6f));
    attn_tracker_destroy(&t);
}

static void test_init_rounds_up(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 100, 0.9f));
    ASSERT_EQ(t.capacity, 128u);
    attn_tracker_destroy(&t);

    ASSERT_OK(attn_tracker_init(&t, 1, 0.9f));
    ASSERT_TRUE(t.capacity >= 64);
    attn_tracker_destroy(&t);
}

static void test_init_clamps_lambda(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 64, -0.5f));
    ASSERT_TRUE(float_near(t.ema_lambda, 0.9f, 1e-6f));
    attn_tracker_destroy(&t);

    ASSERT_OK(attn_tracker_init(&t, 64, 1.0f));
    ASSERT_TRUE(float_near(t.ema_lambda, 0.9f, 1e-6f));
    attn_tracker_destroy(&t);
}

static void test_register_and_active_count(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    ASSERT_EQ(attn_tracker_active_count(&t), 0u);

    attn_tracker_register(&t, 10);
    attn_tracker_register(&t, 20);
    attn_tracker_register(&t, 30);
    ASSERT_EQ(attn_tracker_active_count(&t), 3u);

    attn_tracker_reset(&t, 20);
    ASSERT_EQ(attn_tracker_active_count(&t), 2u);

    attn_tracker_destroy(&t);
}

static void test_update_single(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    attn_tracker_register(&t, 42);
    ASSERT_OK(attn_tracker_update(&t, 42, 0.5f));
    ASSERT_OK(attn_tracker_update(&t, 42, 0.3f));

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 42, &s));
    ASSERT_TRUE(float_near(s.sum, 0.8f, 1e-5f));
    ASSERT_TRUE(float_near(s.max, 0.5f, 1e-5f));
    ASSERT_EQ(s.last_hit_step, 0ULL);

    attn_tracker_destroy(&t);
}

static void test_update_inactive_ignored(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    ASSERT_OK(attn_tracker_update(&t, 99, 1.0f));

    attn_stats_t s;
    int rc = attn_tracker_get(&t, 99, &s);
    ASSERT_EQ(rc, ORCHKV_ERR_NOT_FOUND);

    attn_tracker_destroy(&t);
}

static void test_step_done_ema(void)
{
    attention_tracker_t t;
    float lambda = 0.8f;
    ASSERT_OK(attn_tracker_init(&t, 256, lambda));

    attn_tracker_register(&t, 1);

    /* Step 0: update with weight 1.0 */
    attn_tracker_update(&t, 1, 1.0f);
    attn_tracker_step_done(&t);

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 1, &s));
    /* EMA after step 0: 0.8 * 0 + 0.2 * 1.0 = 0.2 */
    ASSERT_TRUE(float_near(s.ema, 0.2f, 1e-5f));
    ASSERT_EQ(s.query_hits, 1u);
    /* sum/max should be reset to 0 for next step */
    ASSERT_TRUE(float_near(s.sum, 0.0f, 1e-5f));
    ASSERT_EQ(t.current_step, 1ULL);

    /* Step 1: update with weight 0.5 */
    attn_tracker_update(&t, 1, 0.5f);
    attn_tracker_step_done(&t);

    ASSERT_OK(attn_tracker_get(&t, 1, &s));
    /* EMA: 0.8 * 0.2 + 0.2 * 0.5 = 0.16 + 0.1 = 0.26 */
    ASSERT_TRUE(float_near(s.ema, 0.26f, 1e-5f));
    ASSERT_EQ(s.query_hits, 2u);
    ASSERT_EQ(t.current_step, 2ULL);

    attn_tracker_destroy(&t);
}

static void test_idle_decay(void)
{
    attention_tracker_t t;
    float lambda = 0.5f;
    ASSERT_OK(attn_tracker_init(&t, 256, lambda));

    attn_tracker_register(&t, 1);

    /* Step 0: set initial EMA */
    attn_tracker_update(&t, 1, 1.0f);
    attn_tracker_step_done(&t);
    /* EMA = 0.5 * 0 + 0.5 * 1.0 = 0.5 */

    /* Step 1: no update → idle decay */
    attn_tracker_step_done(&t);

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 1, &s));
    /* EMA = 0.5 * 0.5 = 0.25 */
    ASSERT_TRUE(float_near(s.ema, 0.25f, 1e-5f));
    ASSERT_EQ(s.query_hits, 1u);  /* only step 0 was a hit */

    /* Step 2: still idle */
    attn_tracker_step_done(&t);
    ASSERT_OK(attn_tracker_get(&t, 1, &s));
    /* EMA = 0.5 * 0.25 = 0.125 */
    ASSERT_TRUE(float_near(s.ema, 0.125f, 1e-5f));

    attn_tracker_destroy(&t);
}

static void test_batch_update(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    uint64_t ids[]    = {10, 20, 30};
    float weights[]   = {0.1f, 0.2f, 0.3f};

    attn_tracker_register(&t, 10);
    attn_tracker_register(&t, 20);
    attn_tracker_register(&t, 30);

    ASSERT_OK(attn_tracker_update_batch(&t, ids, weights, 3));

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 10, &s));
    ASSERT_TRUE(float_near(s.sum, 0.1f, 1e-5f));

    ASSERT_OK(attn_tracker_get(&t, 20, &s));
    ASSERT_TRUE(float_near(s.sum, 0.2f, 1e-5f));

    ASSERT_OK(attn_tracker_get(&t, 30, &s));
    ASSERT_TRUE(float_near(s.sum, 0.3f, 1e-5f));

    attn_tracker_destroy(&t);
}

static void test_reset_clears_stats(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    attn_tracker_register(&t, 7);
    attn_tracker_update(&t, 7, 0.9f);
    attn_tracker_step_done(&t);

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 7, &s));
    ASSERT_TRUE(s.ema > 0.0f);

    attn_tracker_reset(&t, 7);
    int rc = attn_tracker_get(&t, 7, &s);
    ASSERT_EQ(rc, ORCHKV_ERR_NOT_FOUND);

    attn_tracker_destroy(&t);
}

static void test_multi_block_independence(void)
{
    attention_tracker_t t;
    float lambda = 0.5f;
    ASSERT_OK(attn_tracker_init(&t, 256, lambda));

    attn_tracker_register(&t, 100);
    attn_tracker_register(&t, 200);

    /* Step 0: only block 100 updated */
    attn_tracker_update(&t, 100, 2.0f);
    attn_tracker_step_done(&t);

    attn_stats_t s100, s200;
    ASSERT_OK(attn_tracker_get(&t, 100, &s100));
    ASSERT_OK(attn_tracker_get(&t, 200, &s200));

    /* block 100: EMA = 0.5*0 + 0.5*2.0 = 1.0 */
    ASSERT_TRUE(float_near(s100.ema, 1.0f, 1e-5f));
    ASSERT_EQ(s100.query_hits, 1u);

    /* block 200: idle → EMA stays 0 (0.5 * 0 = 0) */
    ASSERT_TRUE(float_near(s200.ema, 0.0f, 1e-5f));
    ASSERT_EQ(s200.query_hits, 0u);

    /* Step 1: only block 200 updated */
    attn_tracker_update(&t, 200, 1.0f);
    attn_tracker_step_done(&t);

    ASSERT_OK(attn_tracker_get(&t, 100, &s100));
    ASSERT_OK(attn_tracker_get(&t, 200, &s200));

    /* block 100 idle: EMA = 0.5 * 1.0 = 0.5 */
    ASSERT_TRUE(float_near(s100.ema, 0.5f, 1e-5f));
    ASSERT_EQ(s100.query_hits, 1u);

    /* block 200: EMA = 0.5 * 0 + 0.5 * 1.0 = 0.5 */
    ASSERT_TRUE(float_near(s200.ema, 0.5f, 1e-5f));
    ASSERT_EQ(s200.query_hits, 1u);

    attn_tracker_destroy(&t);
}

static void test_max_tracking(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    attn_tracker_register(&t, 5);
    attn_tracker_update(&t, 5, 0.1f);
    attn_tracker_update(&t, 5, 0.9f);
    attn_tracker_update(&t, 5, 0.3f);

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 5, &s));
    ASSERT_TRUE(float_near(s.max, 0.9f, 1e-5f));
    ASSERT_TRUE(float_near(s.sum, 1.3f, 1e-5f));

    attn_tracker_destroy(&t);
}

/* ---- Concurrent update test ------------------------------------------ */

typedef struct {
    attention_tracker_t *tracker;
    uint64_t             block_id;
    int                  n_updates;
    float                weight;
} thread_arg_t;

static void *thread_update(void *arg)
{
    thread_arg_t *a = (thread_arg_t *)arg;
    for (int i = 0; i < a->n_updates; i++) {
        attn_tracker_update(a->tracker, a->block_id, a->weight);
    }
    return NULL;
}

static void test_concurrent_updates(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    attn_tracker_register(&t, 1);

    #define N_THREADS 4
    #define N_UPDATES 10000

    pthread_t threads[N_THREADS];
    thread_arg_t args[N_THREADS];

    for (int i = 0; i < N_THREADS; i++) {
        args[i].tracker   = &t;
        args[i].block_id  = 1;
        args[i].n_updates = N_UPDATES;
        args[i].weight    = 0.01f;
        pthread_create(&threads[i], NULL, thread_update, &args[i]);
    }
    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 1, &s));

    float expected_sum = N_THREADS * N_UPDATES * 0.01f;
    ASSERT_TRUE(float_near(s.sum, expected_sum, 0.1f));

    attn_tracker_destroy(&t);

    #undef N_THREADS
    #undef N_UPDATES
}

/* ---- EMA convergence test -------------------------------------------- */

static void test_ema_convergence(void)
{
    attention_tracker_t t;
    float lambda = 0.9f;
    ASSERT_OK(attn_tracker_init(&t, 256, lambda));

    attn_tracker_register(&t, 1);

    /* Update with constant weight=1.0 for many steps.
     * EMA should converge to 1.0.
     * After n steps: EMA_n = 1 - λ^n ≈ 1.0 for large n. */
    for (int step = 0; step < 100; step++) {
        attn_tracker_update(&t, 1, 1.0f);
        attn_tracker_step_done(&t);
    }

    attn_stats_t s;
    ASSERT_OK(attn_tracker_get(&t, 1, &s));
    ASSERT_TRUE(float_near(s.ema, 1.0f, 0.01f));
    ASSERT_EQ(s.query_hits, 100u);

    attn_tracker_destroy(&t);
}

/* ---- Step counter test ----------------------------------------------- */

static void test_step_counter(void)
{
    attention_tracker_t t;
    ASSERT_OK(attn_tracker_init(&t, 256, 0.9f));

    ASSERT_EQ(attn_tracker_current_step(&t), 0ULL);

    attn_tracker_step_done(&t);
    ASSERT_EQ(attn_tracker_current_step(&t), 1ULL);

    for (int i = 0; i < 100; i++)
        attn_tracker_step_done(&t);
    ASSERT_EQ(attn_tracker_current_step(&t), 101ULL);

    attn_tracker_destroy(&t);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_attention_tracker ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_init_rounds_up);
    RUN_TEST(test_init_clamps_lambda);
    RUN_TEST(test_register_and_active_count);
    RUN_TEST(test_update_single);
    RUN_TEST(test_update_inactive_ignored);
    RUN_TEST(test_step_done_ema);
    RUN_TEST(test_idle_decay);
    RUN_TEST(test_batch_update);
    RUN_TEST(test_reset_clears_stats);
    RUN_TEST(test_multi_block_independence);
    RUN_TEST(test_max_tracking);
    RUN_TEST(test_concurrent_updates);
    RUN_TEST(test_ema_convergence);
    RUN_TEST(test_step_counter);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
