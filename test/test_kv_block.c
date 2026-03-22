#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "core/kv_block.h"

static void test_init_and_id(void)
{
    kv_block_reset_id_counter();

    kv_block_t a, b, c;
    kv_block_init(&a, 100, 0, 0, 0, 64);
    kv_block_init(&b, 100, 0, 1, 0, 64);
    kv_block_init(&c, 200, 1, 0, 64, 32);

    assert(a.block_id == 0);
    assert(b.block_id == 1);
    assert(c.block_id == 2);

    assert(a.request_id == 100);
    assert(a.layer_id == 0);
    assert(a.head_id == 0);
    assert(a.token_start == 0);
    assert(a.token_count == 64);
    assert(a.state == KV_STATE_FREE);
    assert(a.tier == TIER_NONE);
    assert(a.data_ptr == NULL);
    assert(a.flags == KV_FLAG_NONE);

    assert(c.request_id == 200);
    assert(c.layer_id == 1);
    assert(c.token_start == 64);
    assert(c.token_count == 32);

    kv_block_destroy(&a);
    kv_block_destroy(&b);
    kv_block_destroy(&c);
    printf("  [PASS] init_and_id\n");
}

static void test_payload_size(void)
{
    kv_block_reset_id_counter();
    kv_block_t blk;

    kv_block_init(&blk, 1, 0, 0, 0, 64);
    assert(kv_block_payload_size(&blk, 128, DTYPE_FP16) == 32768);  /* 32 KB */
    assert(kv_block_payload_size(&blk, 128, DTYPE_FP32) == 65536);  /* 64 KB */

    kv_block_init(&blk, 1, 0, 0, 0, 32);
    assert(kv_block_payload_size(&blk, 128, DTYPE_FP16) == 16384);  /* 16 KB */

    kv_block_destroy(&blk);
    printf("  [PASS] payload_size\n");
}

static void test_state_transitions(void)
{
    kv_block_reset_id_counter();
    kv_block_t blk;
    kv_block_init(&blk, 1, 0, 0, 0, 64);

    /* FREE → ALLOCATED → HOT */
    assert(kv_block_set_state(&blk, KV_STATE_ALLOCATED) == ORCHKV_OK);
    assert(blk.state == KV_STATE_ALLOCATED);
    assert(kv_block_set_state(&blk, KV_STATE_HOT) == ORCHKV_OK);
    assert(blk.state == KV_STATE_HOT);

    /* HOT → WARM → COLD */
    assert(kv_block_set_state(&blk, KV_STATE_WARM) == ORCHKV_OK);
    assert(kv_block_set_state(&blk, KV_STATE_COLD) == ORCHKV_OK);

    /* COLD → WARM → HOT */
    assert(kv_block_set_state(&blk, KV_STATE_WARM) == ORCHKV_OK);
    assert(kv_block_set_state(&blk, KV_STATE_HOT) == ORCHKV_OK);

    /* HOT → MIGRATING → HOT */
    assert(kv_block_set_state(&blk, KV_STATE_MIGRATING) == ORCHKV_OK);
    assert(kv_block_set_state(&blk, KV_STATE_HOT) == ORCHKV_OK);

    /* HOT → EVICTED (always allowed) */
    assert(kv_block_set_state(&blk, KV_STATE_EVICTED) == ORCHKV_OK);

    kv_block_destroy(&blk);
    printf("  [PASS] state_transitions\n");
}

static void test_illegal_transitions(void)
{
    kv_block_reset_id_counter();
    kv_block_t blk;
    kv_block_init(&blk, 1, 0, 0, 0, 64);

    /* FREE → HOT (skip ALLOCATED — illegal) */
    assert(kv_block_set_state(&blk, KV_STATE_HOT) == ORCHKV_ERR_STATE);
    assert(blk.state == KV_STATE_FREE);

    /* FREE → COLD (illegal) */
    assert(kv_block_set_state(&blk, KV_STATE_COLD) == ORCHKV_ERR_STATE);

    /* get to HOT */
    kv_block_set_state(&blk, KV_STATE_ALLOCATED);
    kv_block_set_state(&blk, KV_STATE_HOT);

    /* HOT → COLD (must go through WARM — illegal) */
    assert(kv_block_set_state(&blk, KV_STATE_COLD) == ORCHKV_ERR_STATE);
    assert(blk.state == KV_STATE_HOT);

    /* HOT → ALLOCATED (illegal) */
    assert(kv_block_set_state(&blk, KV_STATE_ALLOCATED) == ORCHKV_ERR_STATE);

    /* EVICTED → anything (illegal) */
    kv_block_set_state(&blk, KV_STATE_EVICTED);
    assert(kv_block_set_state(&blk, KV_STATE_FREE) == ORCHKV_ERR_STATE);
    assert(kv_block_set_state(&blk, KV_STATE_HOT) == ORCHKV_ERR_STATE);

    kv_block_destroy(&blk);
    printf("  [PASS] illegal_transitions\n");
}

static void test_set_location(void)
{
    kv_block_reset_id_counter();
    kv_block_t blk;
    kv_block_init(&blk, 1, 0, 0, 0, 64);

    int dummy_gpu = 42;
    int dummy_dram = 99;

    kv_block_set_location(&blk, TIER_GPU_HBM, &dummy_gpu);
    assert(blk.tier == TIER_GPU_HBM);
    assert(blk.data_ptr == &dummy_gpu);

    kv_block_set_location(&blk, TIER_HOST_DRAM, &dummy_dram);
    assert(blk.tier == TIER_HOST_DRAM);
    assert(blk.data_ptr == &dummy_dram);

    kv_block_set_location(&blk, TIER_NONE, NULL);
    assert(blk.tier == TIER_NONE);
    assert(blk.data_ptr == NULL);

    kv_block_destroy(&blk);
    printf("  [PASS] set_location\n");
}

static void test_flags(void)
{
    kv_block_reset_id_counter();
    kv_block_t blk;
    kv_block_init(&blk, 1, 0, 0, 0, 64);

    assert(!kv_block_is_pinned(&blk));
    assert(!kv_block_is_dirty(&blk));

    blk.flags |= KV_FLAG_PIN;
    assert(kv_block_is_pinned(&blk));

    blk.flags |= KV_FLAG_DIRTY;
    assert(kv_block_is_dirty(&blk));

    blk.flags &= ~KV_FLAG_PIN;
    assert(!kv_block_is_pinned(&blk));
    assert(kv_block_is_dirty(&blk));

    kv_block_destroy(&blk);
    printf("  [PASS] flags\n");
}

static void test_lock(void)
{
    kv_block_reset_id_counter();
    kv_block_t blk;
    kv_block_init(&blk, 1, 0, 0, 0, 64);

    /* basic lock/unlock (single thread, just verify no deadlock) */
    kv_block_rdlock(&blk);
    kv_block_unlock(&blk);

    kv_block_wrlock(&blk);
    blk.hotness = 1.5f;
    kv_block_unlock(&blk);

    /* multiple readers */
    kv_block_rdlock(&blk);
    kv_block_rdlock(&blk);
    kv_block_unlock(&blk);
    kv_block_unlock(&blk);

    kv_block_destroy(&blk);
    printf("  [PASS] lock\n");
}

static void test_many_blocks(void)
{
    kv_block_reset_id_counter();

    #define N_BLOCKS 10000
    kv_block_t blocks[N_BLOCKS];

    for (int i = 0; i < N_BLOCKS; i++) {
        kv_block_init(&blocks[i], 1, (uint16_t)(i % 32), (uint16_t)(i % 8),
                      (uint32_t)(i * 64), 64);
    }

    /* verify monotonic IDs */
    for (int i = 0; i < N_BLOCKS; i++) {
        assert(blocks[i].block_id == (uint64_t)i);
    }

    for (int i = 0; i < N_BLOCKS; i++) {
        kv_block_destroy(&blocks[i]);
    }
    #undef N_BLOCKS

    printf("  [PASS] many_blocks (10000)\n");
}

int main(void)
{
    printf("=== test_kv_block ===\n");
    test_init_and_id();
    test_payload_size();
    test_state_transitions();
    test_illegal_transitions();
    test_set_location();
    test_flags();
    test_lock();
    test_many_blocks();
    printf("=== ALL PASSED ===\n");
    return 0;
}
