#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "scheduler/attention_tracker.h"
#include "scheduler/hotcold_classifier.h"
#include "scheduler/eviction_policy.h"
#include "scheduler/prefetch_scheduler.h"
#include "scheduler/migration_engine.h"

static int tests_run = 0, tests_passed = 0;

#define RUN_TEST(fn) do {                           \
    printf("  %-50s", #fn); fflush(stdout);         \
    tests_run++; fn(); tests_passed++;              \
    printf(" PASS\n"); } while (0)

#define ASSERT_OK(e)          assert((e) == ORCHKV_OK)
#define ASSERT_EQ(a, b)       assert((a) == (b))
#define ASSERT_TRUE(c)        assert((c))
#define ASSERT_NE(a, b)       assert((a) != (b))

/* ========================================================================
 *  Mock transfer: memcpy between user-space buffers
 * ======================================================================== */

static int g_transfer_calls = 0;
static MigrateOp g_last_ops[8];

static int mock_transfer(kv_block_t *blk, void *dst,
                         size_t size, MigrateOp op, void *ctx)
{
    (void)ctx;
    if (g_transfer_calls < 8)
        g_last_ops[g_transfer_calls] = op;
    g_transfer_calls++;

    if (blk->data_ptr && dst)
        memcpy(dst, blk->data_ptr, size);
    return ORCHKV_OK;
}

static int mock_transfer_fail(kv_block_t *blk, void *dst,
                              size_t size, MigrateOp op, void *ctx)
{
    (void)blk; (void)dst; (void)size; (void)op; (void)ctx;
    return ORCHKV_ERR_IO;
}

static void reset_mock(void)
{
    g_transfer_calls = 0;
    memset(g_last_ops, 0xFF, sizeof(g_last_ops));
}

/* ========================================================================
 *  Block helpers
 * ======================================================================== */

#define BLK_SIZE 256

static void make_block(kv_block_t *blk, uint64_t id,
                       StorageTier tier, KVBlockState state,
                       uint8_t flags, void *data_ptr)
{
    memset(blk, 0, sizeof(*blk));
    blk->block_id    = id;
    blk->request_id  = 1;
    blk->layer_id    = 0;
    blk->head_id     = 0;
    blk->token_start = 0;
    blk->token_count = 64;
    blk->tier        = tier;
    blk->state       = state;
    blk->flags       = flags;
    blk->data_ptr    = data_ptr;
    pthread_rwlock_init(&blk->lock, NULL);
}

/* ======================================================================== */

static void test_init_destroy(void)
{
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));
    mig_destroy(&e);
}

static void test_determine_op(void)
{
    /* Demote paths */
    ASSERT_EQ(mig_determine_op(TIER_GPU_HBM,   TIER_HOST_DRAM), MIGRATE_DEMOTE_GPU2DRAM);
    ASSERT_EQ(mig_determine_op(TIER_HOST_DRAM,  TIER_NVM),      MIGRATE_DEMOTE_DRAM2STOR);
    ASSERT_EQ(mig_determine_op(TIER_HOST_DRAM,  TIER_SSD),      MIGRATE_DEMOTE_DRAM2STOR);
    ASSERT_EQ(mig_determine_op(TIER_GPU_HBM,    TIER_NVM),      MIGRATE_DEMOTE_GPU2STOR);
    ASSERT_EQ(mig_determine_op(TIER_GPU_HBM,    TIER_SSD),      MIGRATE_DEMOTE_GPU2STOR);

    /* Promote paths */
    ASSERT_EQ(mig_determine_op(TIER_NVM,        TIER_HOST_DRAM), MIGRATE_PROMOTE_STOR2DRAM);
    ASSERT_EQ(mig_determine_op(TIER_SSD,        TIER_HOST_DRAM), MIGRATE_PROMOTE_STOR2DRAM);
    ASSERT_EQ(mig_determine_op(TIER_HOST_DRAM,  TIER_GPU_HBM),  MIGRATE_PROMOTE_DRAM2GPU);
    ASSERT_EQ(mig_determine_op(TIER_NVM,        TIER_GPU_HBM),  MIGRATE_PROMOTE_STOR2GPU);
    ASSERT_EQ(mig_determine_op(TIER_SSD,        TIER_GPU_HBM),  MIGRATE_PROMOTE_STOR2GPU);

    /* Invalid: same tier */
    ASSERT_EQ(mig_determine_op(TIER_GPU_HBM, TIER_GPU_HBM), MIGRATE_OP_COUNT);
    ASSERT_EQ(mig_determine_op(TIER_NVM, TIER_NVM), MIGRATE_OP_COUNT);
}

static void test_target_state(void)
{
    ASSERT_EQ(mig_target_state(TIER_GPU_HBM),   KV_STATE_HOT);
    ASSERT_EQ(mig_target_state(TIER_HOST_DRAM),  KV_STATE_WARM);
    ASSERT_EQ(mig_target_state(TIER_NVM),        KV_STATE_COLD);
    ASSERT_EQ(mig_target_state(TIER_SSD),        KV_STATE_COLD);
    ASSERT_EQ(mig_target_state(TIER_NONE),        KV_STATE_EVICTED);
}

static void test_op_names(void)
{
    ASSERT_TRUE(strcmp(mig_op_name(MIGRATE_DEMOTE_GPU2DRAM), "GPU→DRAM") == 0);
    ASSERT_TRUE(strcmp(mig_op_name(MIGRATE_PROMOTE_STOR2GPU), "Storage→GPU") == 0);
}

static void test_demote_gpu2dram(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t src_data[BLK_SIZE], dst_data[BLK_SIZE];
    memset(src_data, 0xAB, BLK_SIZE);
    memset(dst_data, 0, BLK_SIZE);

    kv_block_t blk;
    make_block(&blk, 1, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, src_data);

    int rc = mig_execute_one(&e, &blk, TIER_HOST_DRAM,
                             dst_data, NULL, BLK_SIZE);
    ASSERT_OK(rc);

    /* Block metadata updated */
    ASSERT_EQ(blk.tier, TIER_HOST_DRAM);
    ASSERT_EQ(blk.state, KV_STATE_WARM);
    ASSERT_EQ(blk.data_ptr, dst_data);

    /* Data was copied */
    ASSERT_EQ(memcmp(dst_data, src_data, BLK_SIZE), 0);

    /* Transfer was called once with correct op */
    ASSERT_EQ(g_transfer_calls, 1);
    ASSERT_EQ(g_last_ops[0], MIGRATE_DEMOTE_GPU2DRAM);

    mig_stats_t stats;
    mig_get_stats(&e, &stats);
    ASSERT_EQ(stats.blocks_migrated, 1ULL);

    mig_destroy(&e);
}

static void test_promote_dram2gpu(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t src[BLK_SIZE], dst[BLK_SIZE];
    memset(src, 0xCD, BLK_SIZE);

    kv_block_t blk;
    make_block(&blk, 2, TIER_HOST_DRAM, KV_STATE_WARM, KV_FLAG_NONE, src);

    ASSERT_OK(mig_execute_one(&e, &blk, TIER_GPU_HBM, dst, NULL, BLK_SIZE));

    ASSERT_EQ(blk.tier, TIER_GPU_HBM);
    ASSERT_EQ(blk.state, KV_STATE_HOT);
    ASSERT_EQ(blk.data_ptr, dst);
    ASSERT_EQ(memcmp(dst, src, BLK_SIZE), 0);
    ASSERT_EQ(g_last_ops[0], MIGRATE_PROMOTE_DRAM2GPU);

    mig_destroy(&e);
}

static void test_demote_dram2storage(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t src[BLK_SIZE], dst[BLK_SIZE];
    memset(src, 0xEF, BLK_SIZE);

    kv_block_t blk;
    make_block(&blk, 3, TIER_HOST_DRAM, KV_STATE_WARM, KV_FLAG_NONE, src);

    ASSERT_OK(mig_execute_one(&e, &blk, TIER_NVM, dst, NULL, BLK_SIZE));

    ASSERT_EQ(blk.tier, TIER_NVM);
    ASSERT_EQ(blk.state, KV_STATE_COLD);
    ASSERT_TRUE(blk.data_ptr == NULL);  /* storage tier → no data pointer */
    ASSERT_EQ(g_last_ops[0], MIGRATE_DEMOTE_DRAM2STOR);

    mig_destroy(&e);
}

static void test_two_hop_gpu2storage(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t src[BLK_SIZE], inter[BLK_SIZE], dst[BLK_SIZE];
    memset(src, 0x77, BLK_SIZE);

    kv_block_t blk;
    make_block(&blk, 4, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, src);

    ASSERT_OK(mig_execute_one(&e, &blk, TIER_NVM,
                              dst, inter, BLK_SIZE));

    ASSERT_EQ(blk.tier, TIER_NVM);
    ASSERT_EQ(blk.state, KV_STATE_COLD);
    ASSERT_TRUE(blk.data_ptr == NULL);

    /* Two transfer calls: GPU→DRAM, then DRAM→Storage */
    ASSERT_EQ(g_transfer_calls, 2);
    ASSERT_EQ(g_last_ops[0], MIGRATE_DEMOTE_GPU2DRAM);
    ASSERT_EQ(g_last_ops[1], MIGRATE_DEMOTE_DRAM2STOR);

    /* Data integrity: src → intermediate → dst */
    ASSERT_EQ(memcmp(inter, src, BLK_SIZE), 0);

    mig_destroy(&e);
}

static void test_two_hop_storage2gpu(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t src[BLK_SIZE], inter[BLK_SIZE], dst[BLK_SIZE];
    memset(src, 0x88, BLK_SIZE);

    kv_block_t blk;
    make_block(&blk, 5, TIER_SSD, KV_STATE_COLD, KV_FLAG_NONE, src);

    ASSERT_OK(mig_execute_one(&e, &blk, TIER_GPU_HBM,
                              dst, inter, BLK_SIZE));

    ASSERT_EQ(blk.tier, TIER_GPU_HBM);
    ASSERT_EQ(blk.state, KV_STATE_HOT);
    ASSERT_EQ(blk.data_ptr, dst);

    ASSERT_EQ(g_transfer_calls, 2);
    ASSERT_EQ(g_last_ops[0], MIGRATE_PROMOTE_STOR2DRAM);
    ASSERT_EQ(g_last_ops[1], MIGRATE_PROMOTE_DRAM2GPU);

    mig_destroy(&e);
}

static void test_reject_pinned(void)
{
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t buf[BLK_SIZE];
    kv_block_t blk;
    make_block(&blk, 10, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_PIN, buf);

    int rc = mig_execute_one(&e, &blk, TIER_HOST_DRAM, buf, NULL, BLK_SIZE);
    ASSERT_EQ(rc, ORCHKV_ERR_LOCKED);
    ASSERT_EQ(blk.state, KV_STATE_HOT);  /* unchanged */

    mig_destroy(&e);
}

static void test_reject_already_migrating(void)
{
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t buf[BLK_SIZE];
    kv_block_t blk;
    make_block(&blk, 11, TIER_GPU_HBM, KV_STATE_MIGRATING, KV_FLAG_NONE, buf);

    int rc = mig_execute_one(&e, &blk, TIER_HOST_DRAM, buf, NULL, BLK_SIZE);
    ASSERT_EQ(rc, ORCHKV_ERR_STATE);

    mig_destroy(&e);
}

static void test_reject_same_tier(void)
{
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t buf[BLK_SIZE];
    kv_block_t blk;
    make_block(&blk, 12, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, buf);

    int rc = mig_execute_one(&e, &blk, TIER_GPU_HBM, buf, NULL, BLK_SIZE);
    ASSERT_EQ(rc, ORCHKV_ERR_INVALID);

    mig_destroy(&e);
}

static void test_transfer_failure_restores_state(void)
{
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer_fail, NULL));

    uint8_t buf[BLK_SIZE];
    kv_block_t blk;
    make_block(&blk, 13, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, buf);

    int rc = mig_execute_one(&e, &blk, TIER_HOST_DRAM, buf, NULL, BLK_SIZE);
    ASSERT_EQ(rc, ORCHKV_ERR_IO);

    /* State should be restored */
    ASSERT_EQ(blk.state, KV_STATE_HOT);
    ASSERT_EQ(blk.tier, TIER_GPU_HBM);

    mig_stats_t stats;
    mig_get_stats(&e, &stats);
    ASSERT_EQ(stats.op_errors, 1ULL);
    ASSERT_EQ(stats.blocks_migrated, 0ULL);

    mig_destroy(&e);
}

static void test_two_hop_missing_intermediate(void)
{
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t buf[BLK_SIZE];
    kv_block_t blk;
    make_block(&blk, 14, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, buf);

    /* Two-hop (GPU→Storage) without intermediate buffer → error */
    int rc = mig_execute_one(&e, &blk, TIER_NVM, buf, NULL, BLK_SIZE);
    ASSERT_EQ(rc, ORCHKV_ERR_INVALID);

    mig_destroy(&e);
}

static void test_batch_demote(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    kv_block_t blks[4];
    uint8_t src[4][BLK_SIZE], dst[4][BLK_SIZE];
    void *dst_ptrs[4];

    for (int i = 0; i < 4; i++) {
        memset(src[i], (uint8_t)(i + 1), BLK_SIZE);
        make_block(&blks[i], (uint64_t)(i + 100), TIER_GPU_HBM,
                   KV_STATE_HOT, KV_FLAG_NONE, src[i]);
        dst_ptrs[i] = dst[i];
    }

    /* Pin block 2 — should be skipped */
    blks[2].flags = KV_FLAG_PIN;

    kv_block_t *blk_ptrs[] = {&blks[0], &blks[1], &blks[2], &blks[3]};

    uint32_t ok = mig_demote_batch(&e, blk_ptrs, TIER_HOST_DRAM,
                                   dst_ptrs, NULL, BLK_SIZE, 4);

    ASSERT_EQ(ok, 3u);  /* 3 out of 4 (block 2 was pinned) */

    /* Block 2 unchanged */
    ASSERT_EQ(blks[2].tier, TIER_GPU_HBM);
    ASSERT_EQ(blks[2].state, KV_STATE_HOT);

    /* Others migrated */
    ASSERT_EQ(blks[0].tier, TIER_HOST_DRAM);
    ASSERT_EQ(blks[1].tier, TIER_HOST_DRAM);
    ASSERT_EQ(blks[3].tier, TIER_HOST_DRAM);

    mig_stats_t stats;
    mig_get_stats(&e, &stats);
    ASSERT_EQ(stats.blocks_migrated, 3ULL);

    mig_destroy(&e);
}

static void test_lru_update_on_demote(void)
{
    reset_mock();

    attention_tracker_t trk;
    hotcold_classifier_t cls;
    eviction_policy_t evp;
    hcc_params_t hp;

    ASSERT_OK(attn_tracker_init(&trk, 256, 0.9f));
    hcc_params_default(&hp);
    ASSERT_OK(hcc_init(&cls, &trk, &hp));
    ASSERT_OK(evpol_init(&evp, &cls, 8, 64, 0.7f, 0.3f));

    migration_engine_t e;
    ASSERT_OK(mig_init(&e, &evp, NULL, mock_transfer, NULL));

    uint8_t src[BLK_SIZE], dst[BLK_SIZE];
    kv_block_t blk;
    make_block(&blk, 50, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, src);
    evpol_lru_touch(&evp, &blk);
    ASSERT_EQ(evpol_lru_size(&evp), 1u);

    /* Demote to storage → should remove from LRU */
    ASSERT_OK(mig_execute_one(&e, &blk, TIER_HOST_DRAM, dst, NULL, BLK_SIZE));

    /* Block moved to DRAM → still in LRU (touched) */
    ASSERT_EQ(evpol_lru_size(&evp), 1u);

    /* Now demote to NVM → removed from LRU */
    blk.data_ptr = dst;
    uint8_t stor_dst[BLK_SIZE];
    ASSERT_OK(mig_execute_one(&e, &blk, TIER_NVM, stor_dst, NULL, BLK_SIZE));
    ASSERT_EQ(evpol_lru_size(&evp), 0u);

    mig_destroy(&e);
    evpol_destroy(&evp);
    hcc_destroy(&cls);
    attn_tracker_destroy(&trk);
}

static void test_stats_accumulate(void)
{
    reset_mock();
    migration_engine_t e;
    ASSERT_OK(mig_init(&e, NULL, NULL, mock_transfer, NULL));

    uint8_t buf1[BLK_SIZE], buf2[BLK_SIZE];
    kv_block_t blk;

    /* GPU→DRAM */
    make_block(&blk, 1, TIER_GPU_HBM, KV_STATE_HOT, KV_FLAG_NONE, buf1);
    ASSERT_OK(mig_execute_one(&e, &blk, TIER_HOST_DRAM, buf2, NULL, BLK_SIZE));

    /* DRAM→GPU */
    blk.data_ptr = buf2;
    ASSERT_OK(mig_execute_one(&e, &blk, TIER_GPU_HBM, buf1, NULL, BLK_SIZE));

    mig_stats_t stats;
    mig_get_stats(&e, &stats);
    ASSERT_EQ(stats.blocks_migrated, 2ULL);
    ASSERT_TRUE(stats.op_count[MIGRATE_DEMOTE_GPU2DRAM] >= 1);
    ASSERT_TRUE(stats.op_count[MIGRATE_PROMOTE_DRAM2GPU] >= 1);

    mig_destroy(&e);
}

/* ======================================================================== */

int main(void)
{
    printf("=== test_migration_engine ===\n");

    RUN_TEST(test_init_destroy);
    RUN_TEST(test_determine_op);
    RUN_TEST(test_target_state);
    RUN_TEST(test_op_names);
    RUN_TEST(test_demote_gpu2dram);
    RUN_TEST(test_promote_dram2gpu);
    RUN_TEST(test_demote_dram2storage);
    RUN_TEST(test_two_hop_gpu2storage);
    RUN_TEST(test_two_hop_storage2gpu);
    RUN_TEST(test_reject_pinned);
    RUN_TEST(test_reject_already_migrating);
    RUN_TEST(test_reject_same_tier);
    RUN_TEST(test_transfer_failure_restores_state);
    RUN_TEST(test_two_hop_missing_intermediate);
    RUN_TEST(test_batch_demote);
    RUN_TEST(test_lru_update_on_demote);
    RUN_TEST(test_stats_accumulate);

    printf("\n  Result: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
