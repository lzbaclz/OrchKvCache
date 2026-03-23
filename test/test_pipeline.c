#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/prefetch_scheduler.h"
#include "scheduler/pipeline.h"

static int tests_run = 0, tests_passed = 0;

#define RUN_TEST(fn) do {                           \
    printf("  %-50s", #fn); fflush(stdout);         \
    tests_run++; fn(); tests_passed++;              \
    printf(" PASS\n"); } while (0)

#define ASSERT_OK(e)          assert((e) == ORCHKV_OK)
#define ASSERT_EQ(a, b)       assert((a) == (b))
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_NEAR_D(a,b,e)  assert(fabs((a)-(b)) < (e))

/* ======================================================================== */

static void test_init_destroy(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));
    ASSERT_EQ(pipeline_current_stage(&p), PIPELINE_IDLE);
    ASSERT_EQ(pipeline_step_number(&p), 0ULL);
    pipeline_destroy(&p);
}

static void test_stage_transitions(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));

    pipeline_step_begin(&p);
    ASSERT_EQ(pipeline_current_stage(&p), PIPELINE_COMPUTE);

    pipeline_compute_done(&p);
    ASSERT_EQ(pipeline_current_stage(&p), PIPELINE_PREFETCH);

    pipeline_prefetch_done(&p, 5);
    ASSERT_EQ(pipeline_current_stage(&p), PIPELINE_TRANSFER);

    pipeline_transfer_done(&p, 3);
    ASSERT_EQ(pipeline_current_stage(&p), PIPELINE_IDLE);

    ASSERT_EQ(pipeline_step_number(&p), 1ULL);

    pipeline_destroy(&p);
}

static void test_step_record(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));

    pipeline_step_begin(&p);
    usleep(1000);  /* 1ms compute */
    pipeline_compute_done(&p);
    usleep(500);   /* 0.5ms prefetch */
    pipeline_prefetch_done(&p, 4);
    usleep(500);   /* 0.5ms transfer */
    pipeline_transfer_done(&p, 2);

    pipeline_step_record_t rec;
    pipeline_get_last_record(&p, &rec);

    ASSERT_EQ(rec.step, 0ULL);
    ASSERT_TRUE(rec.compute_us > 500.0);     /* at least 0.5ms */
    ASSERT_TRUE(rec.prefetch_us > 200.0);
    ASSERT_TRUE(rec.transfer_us > 200.0);
    ASSERT_TRUE(rec.step_total_us > 1000.0); /* at least 1ms total */
    ASSERT_EQ(rec.n_prefetched, 4u);
    ASSERT_EQ(rec.n_transferred, 2u);

    pipeline_destroy(&p);
}

static void test_multi_step_accumulation(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));

    for (int i = 0; i < 5; i++) {
        pipeline_step_begin(&p);
        usleep(200);
        pipeline_compute_done(&p);
        usleep(100);
        pipeline_prefetch_done(&p, 2);
        usleep(100);
        pipeline_transfer_done(&p, 1);
    }

    pipeline_stats_t stats;
    pipeline_get_stats(&p, &stats);

    ASSERT_EQ(stats.total_steps, 5ULL);
    ASSERT_TRUE(stats.avg_compute_us > 100.0);
    ASSERT_TRUE(stats.avg_prefetch_us > 50.0);
    ASSERT_TRUE(stats.avg_step_us > 200.0);
    ASSERT_EQ(stats.total_prefetched, 10ULL);
    ASSERT_EQ(stats.total_transferred, 5ULL);

    pipeline_destroy(&p);
}

static void test_overlap_ratio_pure(void)
{
    /* Compute > Prefetch: all IO hidden → ratio = 1.0 */
    ASSERT_NEAR_D(pipeline_compute_overlap(100.0, 50.0), 1.0, 0.01);

    /* Compute = Prefetch: all hidden → ratio = 1.0 */
    ASSERT_NEAR_D(pipeline_compute_overlap(100.0, 100.0), 1.0, 0.01);

    /* Compute < Prefetch: partial overlap */
    ASSERT_NEAR_D(pipeline_compute_overlap(50.0, 100.0), 0.5, 0.01);

    /* Compute much smaller: small overlap */
    ASSERT_NEAR_D(pipeline_compute_overlap(10.0, 100.0), 0.1, 0.01);

    /* Zero prefetch → ratio = 0 */
    ASSERT_NEAR_D(pipeline_compute_overlap(100.0, 0.0), 0.0, 0.01);

    /* Zero compute → ratio = 0 */
    ASSERT_NEAR_D(pipeline_compute_overlap(0.0, 100.0), 0.0, 0.01);
}

static void test_stats_overlap_ratio(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));

    /* Simulate: compute takes 2ms, prefetch takes 1ms
     * → overlap ratio should be ~1.0 (all IO hidden) */
    for (int i = 0; i < 10; i++) {
        pipeline_step_begin(&p);
        usleep(2000);  /* 2ms compute */
        pipeline_compute_done(&p);
        usleep(1000);  /* 1ms prefetch */
        pipeline_prefetch_done(&p, 3);
        usleep(500);   /* 0.5ms transfer */
        pipeline_transfer_done(&p, 2);
    }

    pipeline_stats_t stats;
    pipeline_get_stats(&p, &stats);

    /* avg_compute should be roughly 2× avg_prefetch → overlap ≈ 1.0 */
    ASSERT_TRUE(stats.overlap_ratio > 0.8);
    ASSERT_TRUE(stats.overlap_ratio <= 1.0);

    pipeline_destroy(&p);
}

static void test_zero_steps(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));

    pipeline_stats_t stats;
    pipeline_get_stats(&p, &stats);

    ASSERT_EQ(stats.total_steps, 0ULL);
    ASSERT_NEAR_D(stats.avg_compute_us, 0.0, 0.01);
    ASSERT_NEAR_D(stats.overlap_ratio, 0.0, 0.01);

    pipeline_destroy(&p);
}

static void test_step_number_increments(void)
{
    pipeline_t p;
    ASSERT_OK(pipeline_init(&p, NULL));

    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(pipeline_step_number(&p), (uint64_t)i);
        pipeline_step_begin(&p);
        pipeline_compute_done(&p);
        pipeline_prefetch_done(&p, 0);
        pipeline_transfer_done(&p, 0);
    }
    ASSERT_EQ(pipeline_step_number(&p), 3ULL);

    pipeline_destroy(&p);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_pipeline ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_stage_transitions);
    RUN_TEST(test_step_record);
    RUN_TEST(test_multi_step_accumulation);
    RUN_TEST(test_overlap_ratio_pure);
    RUN_TEST(test_stats_overlap_ratio);
    RUN_TEST(test_zero_steps);
    RUN_TEST(test_step_number_increments);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
