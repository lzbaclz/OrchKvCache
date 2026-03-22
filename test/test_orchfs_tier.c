#include "tiered_store/orchfs_tier.h"
#include "tiered_store/io_worker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define SLAB_SIZE  (32 * 1024)
#define TEST_DIR   "/dev/shm/orchkv_test"

static void fill(void *buf, size_t len, uint8_t seed)
{
    uint8_t *p = (uint8_t *)buf;
    for (size_t i = 0; i < len; i++)
        p[i] = (uint8_t)(seed + (i & 0xFF));
}

static bool verify(const void *buf, size_t len, uint8_t seed)
{
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < len; i++)
        if (p[i] != (uint8_t)(seed + (i & 0xFF))) return false;
    return true;
}

/* ---- Test 1: init/destroy ---- */
static void test_init_destroy(void)
{
    printf("  test_init_destroy ... ");
    orchfs_tier_t tier;
    int rc = orchfs_tier_init(&tier, TEST_DIR, SLAB_SIZE);
    assert(rc == ORCHKV_OK);
    assert(tier.initialized);
    assert(tier.slab_size == SLAB_SIZE);

    orchfs_tier_destroy(&tier);
    assert(!tier.initialized);
    printf("PASS\n");
}

/* ---- Test 2: file open/close ---- */
static void test_file_open_close(void)
{
    printf("  test_file_open_close ... ");
    orchfs_tier_t tier;
    orchfs_tier_init(&tier, TEST_DIR, SLAB_SIZE);

    orchfs_file_ctx_t *fctx = orchfs_file_open(&tier, 42, 2, 4, 64);
    assert(fctx != NULL);
    assert(fctx->fd >= 0);
    assert(fctx->request_id == 42);
    assert(fctx->n_layers == 2);
    assert(fctx->n_kv_heads == 4);

    int rc = orchfs_file_close(&tier, fctx);
    assert(rc == ORCHKV_OK);

    orchfs_tier_destroy(&tier);
    printf("PASS\n");
}

/* ---- Test 3: write + read correctness ---- */
static void test_write_read(void)
{
    printf("  test_write_read ... ");
    orchfs_tier_t tier;
    orchfs_tier_init(&tier, TEST_DIR, SLAB_SIZE);

    orchfs_file_ctx_t *fctx = orchfs_file_open(&tier, 100, 2, 4, 16);
    assert(fctx);

    uint8_t *wbuf = (uint8_t *)malloc(SLAB_SIZE);
    uint8_t *rbuf = (uint8_t *)malloc(SLAB_SIZE);
    assert(wbuf && rbuf);

    for (uint32_t layer = 0; layer < 2; layer++) {
        for (uint32_t head = 0; head < 4; head++) {
            uint8_t seed = (uint8_t)(layer * 10 + head);
            fill(wbuf, SLAB_SIZE, seed);

            int rc = orchfs_tier_write(fctx, layer, head, 0, wbuf, SLAB_SIZE);
            assert(rc == ORCHKV_OK);
        }
    }

    for (uint32_t layer = 0; layer < 2; layer++) {
        for (uint32_t head = 0; head < 4; head++) {
            uint8_t seed = (uint8_t)(layer * 10 + head);
            memset(rbuf, 0, SLAB_SIZE);

            int rc = orchfs_tier_read(fctx, layer, head, 0, rbuf, SLAB_SIZE);
            assert(rc == ORCHKV_OK);
            assert(verify(rbuf, SLAB_SIZE, seed));
        }
    }

    free(wbuf);
    free(rbuf);
    orchfs_file_close(&tier, fctx);
    orchfs_tier_destroy(&tier);
    printf("PASS\n");
}

/* ---- Test 4: multiple blocks per head ---- */
static void test_multi_block(void)
{
    printf("  test_multi_block ... ");
    orchfs_tier_t tier;
    orchfs_tier_init(&tier, TEST_DIR, SLAB_SIZE);

    orchfs_file_ctx_t *fctx = orchfs_file_open(&tier, 200, 1, 1, 8);
    assert(fctx);

    uint8_t *buf = (uint8_t *)malloc(SLAB_SIZE);
    assert(buf);

    for (uint32_t bi = 0; bi < 8; bi++) {
        fill(buf, SLAB_SIZE, (uint8_t)(0xA0 + bi));
        assert(orchfs_tier_write(fctx, 0, 0, bi, buf, SLAB_SIZE) == ORCHKV_OK);
    }

    for (uint32_t bi = 0; bi < 8; bi++) {
        memset(buf, 0, SLAB_SIZE);
        assert(orchfs_tier_read(fctx, 0, 0, bi, buf, SLAB_SIZE) == ORCHKV_OK);
        assert(verify(buf, SLAB_SIZE, (uint8_t)(0xA0 + bi)));
    }

    free(buf);
    orchfs_file_close(&tier, fctx);
    orchfs_tier_destroy(&tier);
    printf("PASS\n");
}

/* ---- Test 5: offset computation ---- */
static void test_offset(void)
{
    printf("  test_offset ... ");
    orchfs_file_ctx_t fctx = {
        .fd = -1, .n_layers = 32, .n_kv_heads = 8,
        .max_blocks_per_head = 256, .slab_size = SLAB_SIZE
    };

    int64_t off_0_0_0 = orchfs_compute_offset(&fctx, 0, 0, 0);
    assert(off_0_0_0 == 0);

    int64_t off_0_0_1 = orchfs_compute_offset(&fctx, 0, 0, 1);
    assert(off_0_0_1 == SLAB_SIZE);

    int64_t off_0_1_0 = orchfs_compute_offset(&fctx, 0, 1, 0);
    assert(off_0_1_0 == 256 * SLAB_SIZE);

    int64_t off_1_0_0 = orchfs_compute_offset(&fctx, 1, 0, 0);
    assert(off_1_0_0 == 8 * 256 * SLAB_SIZE);

    printf("PASS\n");
}

/* ---- Test 6: io_worker async write/read ---- */

typedef struct {
    int   status;
    bool  done;
} cb_data_t;

static void test_callback(int status, void *user_data)
{
    cb_data_t *d = (cb_data_t *)user_data;
    d->status = status;
    d->done = true;
}

static void test_io_worker(void)
{
    printf("  test_io_worker ... ");
    orchfs_tier_t tier;
    orchfs_tier_init(&tier, TEST_DIR, SLAB_SIZE);
    orchfs_file_ctx_t *fctx = orchfs_file_open(&tier, 300, 1, 1, 4);
    assert(fctx);

    io_worker_pool_t pool;
    int rc = io_worker_init(&pool, 2, 64);
    assert(rc == ORCHKV_OK);

    uint8_t *wbuf = (uint8_t *)malloc(SLAB_SIZE);
    uint8_t *rbuf = (uint8_t *)malloc(SLAB_SIZE);
    assert(wbuf && rbuf);
    fill(wbuf, SLAB_SIZE, 0x55);

    cb_data_t cb_w = {0, false};
    io_task_t task_w = {
        .op = IO_OP_WRITE, .tier = &tier, .fctx = fctx,
        .layer = 0, .head = 0, .block_idx = 0,
        .buf = wbuf, .size = SLAB_SIZE,
        .callback = test_callback, .user_data = &cb_w
    };
    rc = io_worker_submit(&pool, &task_w);
    assert(rc == ORCHKV_OK);

    io_worker_flush(&pool);
    assert(cb_w.done && cb_w.status == ORCHKV_OK);

    cb_data_t cb_r = {0, false};
    io_task_t task_r = {
        .op = IO_OP_READ, .tier = &tier, .fctx = fctx,
        .layer = 0, .head = 0, .block_idx = 0,
        .buf = rbuf, .size = SLAB_SIZE,
        .callback = test_callback, .user_data = &cb_r
    };
    rc = io_worker_submit(&pool, &task_r);
    assert(rc == ORCHKV_OK);

    io_worker_flush(&pool);
    assert(cb_r.done && cb_r.status == ORCHKV_OK);
    assert(verify(rbuf, SLAB_SIZE, 0x55));

    free(wbuf);
    free(rbuf);
    io_worker_destroy(&pool);
    orchfs_file_close(&tier, fctx);
    orchfs_tier_destroy(&tier);
    printf("PASS\n");
}

/* ---- Test 7: stats recording ---- */
static void test_stats(void)
{
    printf("  test_stats ... ");
    orchfs_tier_t tier;
    orchfs_tier_init(&tier, TEST_DIR, SLAB_SIZE);

    orchfs_tier_record_write(&tier, 1000);
    orchfs_tier_record_write(&tier, 2000);
    orchfs_tier_record_read(&tier, 500);

    assert(orchfs_tier_writes(&tier) == 2);
    assert(orchfs_tier_reads(&tier) == 1);
    assert(tier.bytes_written == 3000);
    assert(tier.bytes_read == 500);

    orchfs_tier_destroy(&tier);
    printf("PASS\n");
}

int main(void)
{
    printf("=== test_orchfs_tier ===\n");
    test_init_destroy();
    test_file_open_close();
    test_write_read();
    test_multi_block();
    test_offset();
    test_io_worker();
    test_stats();
    printf("=== ALL PASSED ===\n");
    return 0;
}
