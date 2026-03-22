/*
 * test_gpu_dram.cu — A5/A6/A7 correctness tests + performance benchmarks.
 *
 * Correctness:
 *   - GPU slab alloc/free (exhaust pool, double-free detection)
 *   - DRAM pinned slab alloc/free
 *   - Async transfer D2H/H2D with data integrity check (round-trip)
 *
 * Benchmarks:
 *   - Slab alloc latency
 *   - Transfer bandwidth for various block sizes
 *   - Multi-stream concurrency
 *
 * JSON results written to stdout when run with --bench flag.
 */
extern "C" {
#include "tiered_store/gpu_tier.h"
#include "tiered_store/dram_tier.h"
#include "tiered_store/transfer.h"
}
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Fill host buffer with a deterministic pattern based on seed */
static void fill_pattern(void *buf, size_t bytes, uint32_t seed)
{
    uint32_t *p = (uint32_t *)buf;
    for (size_t i = 0; i < bytes / sizeof(uint32_t); i++)
        p[i] = seed ^ (uint32_t)i;
}

static int verify_pattern(const void *buf, size_t bytes, uint32_t seed)
{
    const uint32_t *p = (const uint32_t *)buf;
    for (size_t i = 0; i < bytes / sizeof(uint32_t); i++) {
        if (p[i] != (seed ^ (uint32_t)i))
            return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Correctness Tests                                                 */
/* ------------------------------------------------------------------ */

static void test_gpu_tier_basic(void)
{
    gpu_tier_t gt;
    size_t slab = 32 * 1024;  /* 32 KB */
    size_t pool = slab * 100; /* 100 slabs */

    assert(gpu_tier_init(&gt, 0, pool, slab) == ORCHKV_OK);
    assert(gt.total_slabs == 100);
    assert(gt.used_slabs == 0);

    void *ptrs[100];
    for (int i = 0; i < 100; i++) {
        assert(gpu_tier_alloc(&gt, &ptrs[i]) == ORCHKV_OK);
        assert(ptrs[i] != NULL);
    }
    assert(gt.used_slabs == 100);

    /* pool exhausted */
    void *extra;
    assert(gpu_tier_alloc(&gt, &extra) == ORCHKV_ERR_TIER_FULL);

    /* free all */
    for (int i = 0; i < 100; i++)
        assert(gpu_tier_free(&gt, ptrs[i]) == ORCHKV_OK);
    assert(gt.used_slabs == 0);

    /* re-alloc after free */
    assert(gpu_tier_alloc(&gt, &ptrs[0]) == ORCHKV_OK);
    assert(gpu_tier_free(&gt, ptrs[0]) == ORCHKV_OK);

    /* invalid pointer */
    assert(gpu_tier_free(&gt, (void *)0xDEAD) == ORCHKV_ERR_INVALID);

    gpu_tier_destroy(&gt);
    printf("  [PASS] gpu_tier_basic\n");
}

static void test_dram_tier_basic(void)
{
    dram_tier_t dt;
    size_t slab = 32 * 1024;
    size_t pool = slab * 50;

    assert(dram_tier_init(&dt, pool, slab, true) == ORCHKV_OK);
    assert(dt.total_slabs == 50);
    assert(dt.is_pinned == true);

    void *ptrs[50];
    for (int i = 0; i < 50; i++)
        assert(dram_tier_alloc(&dt, &ptrs[i]) == ORCHKV_OK);
    assert(dt.used_slabs == 50);

    void *extra;
    assert(dram_tier_alloc(&dt, &extra) == ORCHKV_ERR_TIER_FULL);

    for (int i = 0; i < 50; i++)
        assert(dram_tier_free(&dt, ptrs[i]) == ORCHKV_OK);

    dram_tier_destroy(&dt);
    printf("  [PASS] dram_tier_basic\n");
}

static void test_dram_tier_fallback(void)
{
    dram_tier_t dt;
    size_t slab = 4096;
    size_t pool = slab * 10;

    /* use_pinned=false → should use malloc */
    assert(dram_tier_init(&dt, pool, slab, false) == ORCHKV_OK);
    assert(dt.is_pinned == false);

    void *p;
    assert(dram_tier_alloc(&dt, &p) == ORCHKV_OK);
    assert(dram_tier_free(&dt, p) == ORCHKV_OK);

    dram_tier_destroy(&dt);
    printf("  [PASS] dram_tier_fallback (non-pinned)\n");
}

static void test_transfer_basic(void)
{
    transfer_engine_t eng;
    assert(transfer_engine_init(&eng, 0, 4) == ORCHKV_OK);
    assert(eng.num_streams == 4);

    transfer_engine_destroy(&eng);
    printf("  [PASS] transfer_basic (init/destroy)\n");
}

static void test_roundtrip(void)
{
    /*
     * Write known pattern to DRAM → H2D to GPU → D2H back to DRAM → verify.
     * This is the core data-integrity test.
     */
    size_t slab = 32 * 1024; /* 32 KB */
    gpu_tier_t gt;
    dram_tier_t dt;
    transfer_engine_t eng;

    assert(gpu_tier_init(&gt, 0, slab * 4, slab) == ORCHKV_OK);
    assert(dram_tier_init(&dt, slab * 4, slab, true) == ORCHKV_OK);
    assert(transfer_engine_init(&eng, 0, 2) == ORCHKV_OK);

    void *gpu_ptr, *dram_src, *dram_dst;
    assert(gpu_tier_alloc(&gt, &gpu_ptr) == ORCHKV_OK);
    assert(dram_tier_alloc(&dt, &dram_src) == ORCHKV_OK);
    assert(dram_tier_alloc(&dt, &dram_dst) == ORCHKV_OK);

    /* fill source with pattern */
    uint32_t seed = 0xCAFEBABE;
    fill_pattern(dram_src, slab, seed);

    /* DRAM → GPU (H2D) */
    int si = transfer_submit(&eng, gpu_ptr, dram_src, slab, XFER_H2D);
    assert(si >= 0);
    assert(transfer_sync_stream(&eng, si) == ORCHKV_OK);

    /* GPU → DRAM (D2H) into a different buffer */
    memset(dram_dst, 0, slab);
    si = transfer_submit(&eng, dram_dst, gpu_ptr, slab, XFER_D2H);
    assert(si >= 0);
    assert(transfer_sync_stream(&eng, si) == ORCHKV_OK);

    /* verify data integrity */
    assert(verify_pattern(dram_dst, slab, seed) == 0);

    /* check counters */
    assert(eng.total_h2d == 1);
    assert(eng.total_d2h == 1);
    assert(eng.bytes_h2d == slab);
    assert(eng.bytes_d2h == slab);

    gpu_tier_free(&gt, gpu_ptr);
    dram_tier_free(&dt, dram_src);
    dram_tier_free(&dt, dram_dst);

    transfer_engine_destroy(&eng);
    dram_tier_destroy(&dt);
    gpu_tier_destroy(&gt);
    printf("  [PASS] roundtrip (32KB data integrity)\n");
}

static void test_roundtrip_large(void)
{
    /* Test with 4 MB blocks (the "sweet spot" from Exp0) */
    size_t slab = 4 * 1024 * 1024;
    gpu_tier_t gt;
    dram_tier_t dt;
    transfer_engine_t eng;

    assert(gpu_tier_init(&gt, 0, slab * 2, slab) == ORCHKV_OK);
    assert(dram_tier_init(&dt, slab * 2, slab, true) == ORCHKV_OK);
    assert(transfer_engine_init(&eng, 0, 2) == ORCHKV_OK);

    void *gpu_ptr, *dram_src, *dram_dst;
    assert(gpu_tier_alloc(&gt, &gpu_ptr) == ORCHKV_OK);
    assert(dram_tier_alloc(&dt, &dram_src) == ORCHKV_OK);
    assert(dram_tier_alloc(&dt, &dram_dst) == ORCHKV_OK);

    fill_pattern(dram_src, slab, 0xDEADBEEF);

    int si = transfer_submit(&eng, gpu_ptr, dram_src, slab, XFER_H2D);
    assert(si >= 0);
    transfer_sync_stream(&eng, si);

    memset(dram_dst, 0, slab);
    si = transfer_submit(&eng, dram_dst, gpu_ptr, slab, XFER_D2H);
    assert(si >= 0);
    transfer_sync_stream(&eng, si);

    assert(verify_pattern(dram_dst, slab, 0xDEADBEEF) == 0);

    gpu_tier_free(&gt, gpu_ptr);
    dram_tier_free(&dt, dram_src);
    dram_tier_free(&dt, dram_dst);
    transfer_engine_destroy(&eng);
    dram_tier_destroy(&dt);
    gpu_tier_destroy(&gt);
    printf("  [PASS] roundtrip_large (4MB data integrity)\n");
}

static void test_multi_stream(void)
{
    size_t slab = 32 * 1024;
    int n_slabs = 16;
    gpu_tier_t gt;
    dram_tier_t dt;
    transfer_engine_t eng;

    assert(gpu_tier_init(&gt, 0, slab * n_slabs, slab) == ORCHKV_OK);
    assert(dram_tier_init(&dt, slab * n_slabs * 2, slab, true) == ORCHKV_OK);
    assert(transfer_engine_init(&eng, 0, 4) == ORCHKV_OK);

    void *gpu_ptrs[16], *dram_srcs[16], *dram_dsts[16];
    for (int i = 0; i < n_slabs; i++) {
        gpu_tier_alloc(&gt, &gpu_ptrs[i]);
        dram_tier_alloc(&dt, &dram_srcs[i]);
        dram_tier_alloc(&dt, &dram_dsts[i]);
        fill_pattern(dram_srcs[i], slab, (uint32_t)(i + 1));
    }

    /* submit all H2D in parallel across 4 streams */
    for (int i = 0; i < n_slabs; i++)
        transfer_submit(&eng, gpu_ptrs[i], dram_srcs[i], slab, XFER_H2D);
    transfer_sync_all(&eng);

    /* submit all D2H in parallel */
    for (int i = 0; i < n_slabs; i++) {
        memset(dram_dsts[i], 0, slab);
        transfer_submit(&eng, dram_dsts[i], gpu_ptrs[i], slab, XFER_D2H);
    }
    transfer_sync_all(&eng);

    /* verify all */
    for (int i = 0; i < n_slabs; i++)
        assert(verify_pattern(dram_dsts[i], slab, (uint32_t)(i + 1)) == 0);

    for (int i = 0; i < n_slabs; i++) {
        gpu_tier_free(&gt, gpu_ptrs[i]);
        dram_tier_free(&dt, dram_srcs[i]);
        dram_tier_free(&dt, dram_dsts[i]);
    }

    transfer_engine_destroy(&eng);
    dram_tier_destroy(&dt);
    gpu_tier_destroy(&gt);
    printf("  [PASS] multi_stream (16 blocks × 4 streams)\n");
}

/* ------------------------------------------------------------------ */
/*  Benchmarks                                                        */
/* ------------------------------------------------------------------ */

static void bench_slab_alloc(void)
{
    gpu_tier_t gt;
    size_t slab = 32 * 1024;
    uint32_t n = 10000;
    size_t pool = slab * n;

    gpu_tier_init(&gt, 0, pool, slab);

    void **ptrs = (void **)malloc(n * sizeof(void *));

    double t0 = now_us();
    for (uint32_t i = 0; i < n; i++)
        gpu_tier_alloc(&gt, &ptrs[i]);
    double alloc_us = now_us() - t0;

    t0 = now_us();
    for (uint32_t i = 0; i < n; i++)
        gpu_tier_free(&gt, ptrs[i]);
    double free_us = now_us() - t0;

    printf("    \"slab_alloc_10k\": {"
           "\"alloc_total_us\": %.1f, \"alloc_avg_ns\": %.1f, "
           "\"free_total_us\": %.1f, \"free_avg_ns\": %.1f}",
           alloc_us, alloc_us * 1000.0 / n,
           free_us, free_us * 1000.0 / n);

    free(ptrs);
    gpu_tier_destroy(&gt);
}

typedef struct {
    double h2d_us;
    double h2d_gbps;
    double d2h_us;
    double d2h_gbps;
    double roundtrip_us;
} bw_result_t;

static bw_result_t bench_transfer_bw(size_t block_size, int num_streams, int iters)
{
    gpu_tier_t gt;
    dram_tier_t dt;
    transfer_engine_t eng;

    gpu_tier_init(&gt, 0, block_size * 2, block_size);
    dram_tier_init(&dt, block_size * 2, block_size, true);
    transfer_engine_init(&eng, 0, num_streams);

    void *gpu_ptr, *host_src, *host_dst;
    gpu_tier_alloc(&gt, &gpu_ptr);
    dram_tier_alloc(&dt, &host_src);
    dram_tier_alloc(&dt, &host_dst);
    fill_pattern(host_src, block_size, 0x12345678);

    /* warmup */
    for (int i = 0; i < 5; i++) {
        int si = transfer_submit(&eng, gpu_ptr, host_src, block_size, XFER_H2D);
        transfer_sync_stream(&eng, si);
    }

    /* H2D */
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        int si = transfer_submit(&eng, gpu_ptr, host_src, block_size, XFER_H2D);
        transfer_sync_stream(&eng, si);
    }
    double h2d_total = now_us() - t0;

    /* D2H */
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        int si = transfer_submit(&eng, host_dst, gpu_ptr, block_size, XFER_D2H);
        transfer_sync_stream(&eng, si);
    }
    double d2h_total = now_us() - t0;

    bw_result_t r;
    r.h2d_us  = h2d_total / iters;
    r.h2d_gbps = (double)block_size / r.h2d_us / 1e3;  /* GB/s */
    r.d2h_us  = d2h_total / iters;
    r.d2h_gbps = (double)block_size / r.d2h_us / 1e3;
    r.roundtrip_us = r.h2d_us + r.d2h_us;

    gpu_tier_free(&gt, gpu_ptr);
    dram_tier_free(&dt, host_src);
    dram_tier_free(&dt, host_dst);
    transfer_engine_destroy(&eng);
    dram_tier_destroy(&dt);
    gpu_tier_destroy(&gt);

    return r;
}

static void run_benchmarks(void)
{
    printf("{\n  \"benchmark\": \"A5_A6_A7_gpu_dram_transfer\",\n");

    /* 1. Slab alloc latency */
    bench_slab_alloc();
    printf(",\n");

    /* 2. Transfer bandwidth for various sizes */
    printf("    \"transfer_bw\": [\n");
    size_t sizes[] = {32*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024};
    const char *labels[] = {"32KB", "64KB", "256KB", "1MB", "4MB", "16MB", "64MB"};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++) {
        int iters = sizes[i] < 1024*1024 ? 500 : (sizes[i] < 16*1024*1024 ? 200 : 50);
        bw_result_t r = bench_transfer_bw(sizes[i], 4, iters);
        printf("      {\"size\": \"%s\", \"size_bytes\": %zu, "
               "\"h2d_us\": %.1f, \"h2d_gbps\": %.2f, "
               "\"d2h_us\": %.1f, \"d2h_gbps\": %.2f, "
               "\"roundtrip_us\": %.1f}%s\n",
               labels[i], sizes[i],
               r.h2d_us, r.h2d_gbps,
               r.d2h_us, r.d2h_gbps,
               r.roundtrip_us,
               i < n_sizes - 1 ? "," : "");
    }
    printf("    ],\n");

    /* 3. Multi-stream scaling */
    printf("    \"stream_scaling\": [\n");
    int stream_counts[] = {1, 2, 4, 8};
    for (int si = 0; si < 4; si++) {
        bw_result_t r = bench_transfer_bw(4*1024*1024, stream_counts[si], 200);
        printf("      {\"streams\": %d, \"block_4MB_h2d_gbps\": %.2f, "
               "\"block_4MB_d2h_gbps\": %.2f}%s\n",
               stream_counts[si], r.h2d_gbps, r.d2h_gbps,
               si < 3 ? "," : "");
    }
    printf("    ]\n");

    printf("}\n");
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    bool do_bench = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0)
            do_bench = true;
    }

    if (!do_bench) {
        printf("=== test_gpu_dram (correctness) ===\n");
        test_gpu_tier_basic();
        test_dram_tier_basic();
        test_dram_tier_fallback();
        test_transfer_basic();
        test_roundtrip();
        test_roundtrip_large();
        test_multi_stream();
        printf("=== ALL PASSED ===\n");
    } else {
        run_benchmarks();
    }
    return 0;
}
