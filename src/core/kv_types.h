#ifndef ORCHKV_KV_TYPES_H
#define ORCHKV_KV_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>

/* ========================================================================
 *  Error Codes
 * ======================================================================== */

#define ORCHKV_OK                0
#define ORCHKV_ERR_OOM          -1   /* out of memory (GPU or DRAM) */
#define ORCHKV_ERR_INVALID      -2   /* invalid argument */
#define ORCHKV_ERR_NOT_FOUND    -3   /* block / request not found */
#define ORCHKV_ERR_TIER_FULL    -4   /* target tier has no free slabs */
#define ORCHKV_ERR_CUDA         -5   /* CUDA runtime error */
#define ORCHKV_ERR_IO           -6   /* storage I/O error */
#define ORCHKV_ERR_LOCKED       -7   /* block is pinned or in-flight */
#define ORCHKV_ERR_STATE        -8   /* illegal state transition */
#define ORCHKV_ERR_INIT         -9   /* system not initialised */
#define ORCHKV_ERR_ALREADY     -10   /* double init / duplicate insert */

/* ========================================================================
 *  Enumerations
 * ======================================================================== */

typedef enum {
    TIER_GPU_HBM   = 0,
    TIER_HOST_DRAM = 1,
    TIER_NVM       = 2,   /* Phase B */
    TIER_SSD       = 3,   /* Phase B */
    TIER_NONE      = 4,   /* not resident anywhere */
} StorageTier;

typedef enum {
    DTYPE_FP16  = 0,   /* 2 bytes */
    DTYPE_BF16  = 1,   /* 2 bytes */
    DTYPE_FP32  = 2,   /* 4 bytes */
    DTYPE_INT8  = 3,   /* 1 byte  */
    DTYPE_INT4  = 4,   /* 0.5 byte (packed) */
} DataType;

typedef enum {
    KV_STATE_FREE       = 0,   /* in free pool, no data */
    KV_STATE_ALLOCATED  = 1,   /* slab assigned, data being written */
    KV_STATE_HOT        = 2,   /* resident on GPU, actively used */
    KV_STATE_WARM       = 3,   /* resident on DRAM (or NVM) */
    KV_STATE_COLD       = 4,   /* resident on NVM / SSD */
    KV_STATE_MIGRATING  = 5,   /* async transfer in progress */
    KV_STATE_EVICTED    = 6,   /* data released */
} KVBlockState;

/* ========================================================================
 *  Size & Alignment Constants
 * ======================================================================== */

/*
 * Per-block token count.  With d_head=128, FP16, single KV-head:
 *   64 tokens × 128 × 2(K+V) × 2B = 32 KB  →  matches ORCH_BLOCK_SIZE.
 */
#define KV_TOKEN_GROUP_SIZE     64

/* OrchFS alignment targets (mirrored from OrchFS config/config.h) */
#define KV_PAGE_SIZE            4096            /* 4 KB  – NVM page */
#define KV_BLOCK_SIZE           (32 * 1024)     /* 32 KB – SSD block */

/* Default pool sizing */
#define KV_DEFAULT_GPU_POOL_GB  40              /* half of A100-80GB */
#define KV_DEFAULT_DRAM_POOL_GB 64

/* Transfer engine */
#define KV_DEFAULT_NUM_STREAMS  4
#define KV_TRANSFER_QUEUE_CAP   1024

/* Address-map hash table */
#define KV_ADDRMAP_INIT_CAP     (1 << 16)       /* 65536 buckets */
#define KV_ADDRMAP_LOAD_FACTOR  0.75

/* ========================================================================
 *  Block Flags  (bitfield stored in uint8_t)
 * ======================================================================== */

#define KV_FLAG_NONE        0x00
#define KV_FLAG_PIN         0x01   /* pinned: do not evict */
#define KV_FLAG_DIRTY       0x02   /* modified since last persist */
#define KV_FLAG_PREFETCHING 0x04   /* async prefetch in progress */
#define KV_FLAG_ATTN_SINK   0x08   /* attention-sink token (always hot) */

/* ========================================================================
 *  Utility Macros
 * ======================================================================== */

#define ORCHKV_ALIGN_UP(x, align)   (((x) + (align) - 1) & ~((align) - 1))
#define ORCHKV_ALIGN_DOWN(x, align) ((x) & ~((align) - 1))
#define ORCHKV_MIN(a, b)            (((a) < (b)) ? (a) : (b))
#define ORCHKV_MAX(a, b)            (((a) > (b)) ? (a) : (b))
#define ORCHKV_DIV_CEIL(a, b)       (((a) + (b) - 1) / (b))

/* ========================================================================
 *  Logging
 * ======================================================================== */

#ifndef ORCHKV_LOG_LEVEL
#define ORCHKV_LOG_LEVEL 2  /* 0=off, 1=error, 2=warn, 3=info, 4=debug */
#endif

#define ORCHKV_LOG(level, tag, fmt, ...)                                      \
    do {                                                                      \
        if ((level) <= ORCHKV_LOG_LEVEL)                                      \
            fprintf(stderr, "[%s] %s:%d: " fmt "\n",                         \
                    (tag), __FILE__, __LINE__, ##__VA_ARGS__);                \
    } while (0)

#define LOG_ERR(fmt, ...)   ORCHKV_LOG(1, "ERR",   fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  ORCHKV_LOG(2, "WARN",  fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  ORCHKV_LOG(3, "INFO",  fmt, ##__VA_ARGS__)
#define LOG_DBG(fmt, ...)   ORCHKV_LOG(4, "DBG",   fmt, ##__VA_ARGS__)

/* ========================================================================
 *  DataType Helpers
 * ======================================================================== */

static inline size_t dtype_size(DataType dt)
{
    switch (dt) {
    case DTYPE_FP16:  return 2;
    case DTYPE_BF16:  return 2;
    case DTYPE_FP32:  return 4;
    case DTYPE_INT8:  return 1;
    case DTYPE_INT4:  return 1; /* 2 values packed per byte */
    default:          return 0;
    }
}

static inline const char *dtype_name(DataType dt)
{
    switch (dt) {
    case DTYPE_FP16:  return "fp16";
    case DTYPE_BF16:  return "bf16";
    case DTYPE_FP32:  return "fp32";
    case DTYPE_INT8:  return "int8";
    case DTYPE_INT4:  return "int4";
    default:          return "unknown";
    }
}

static inline const char *tier_name(StorageTier t)
{
    switch (t) {
    case TIER_GPU_HBM:   return "GPU_HBM";
    case TIER_HOST_DRAM:  return "HOST_DRAM";
    case TIER_NVM:        return "NVM";
    case TIER_SSD:        return "SSD";
    case TIER_NONE:       return "NONE";
    default:              return "UNKNOWN";
    }
}

static inline const char *block_state_name(KVBlockState s)
{
    switch (s) {
    case KV_STATE_FREE:       return "FREE";
    case KV_STATE_ALLOCATED:  return "ALLOCATED";
    case KV_STATE_HOT:        return "HOT";
    case KV_STATE_WARM:       return "WARM";
    case KV_STATE_COLD:       return "COLD";
    case KV_STATE_MIGRATING:  return "MIGRATING";
    case KV_STATE_EVICTED:    return "EVICTED";
    default:                  return "UNKNOWN";
    }
}

/* ========================================================================
 *  KV Block Data Size Calculation
 * ======================================================================== */

/*
 * Compute the byte size of a single KV block's payload.
 *   = token_count × d_head × 2(K+V) × dtype_bytes
 * For a single KV-head.
 */
static inline size_t kv_block_data_bytes(uint32_t token_count,
                                         uint32_t d_head,
                                         DataType dtype)
{
    return (size_t)token_count * d_head * 2 * dtype_size(dtype);
}

/* ========================================================================
 *  System Configuration
 * ======================================================================== */

typedef struct orchkv_config {
    /* GPU */
    int          gpu_device_id;         /* CUDA device index (default 0) */
    size_t       gpu_pool_bytes;        /* GPU KV-cache pool size */

    /* DRAM */
    size_t       dram_pool_bytes;       /* Host DRAM pool size */
    bool         dram_use_pinned;       /* use cudaMallocHost (default true) */

    /* Transfer */
    int          num_cuda_streams;      /* async transfer streams (default 4) */

    /* KV block geometry */
    uint32_t     tokens_per_block;      /* KV_TOKEN_GROUP_SIZE (default 64) */
    uint32_t     d_head;                /* model head dimension */
    DataType     dtype;                 /* data type for KV cache */

    /* Capacity thresholds */
    float        gpu_hwm;              /* GPU high water mark (default 0.9) */
    float        gpu_lwm;              /* GPU low water mark  (default 0.7) */
    float        dram_hwm;             /* DRAM high water mark (default 0.9) */
    float        dram_lwm;             /* DRAM low water mark  (default 0.7) */

    /* OrchFS (Phase B, ignored in Phase A) */
    const char  *orchfs_nvm_path;      /* e.g. "/dev/dax0.0" */
    const char  *orchfs_ssd_path;      /* e.g. "/dev/nvme1n1" */
    int          nvm_io_threads;
    int          ssd_io_threads;
} orchkv_config_t;

/* Fill config with sensible defaults. Always call this before customising. */
static inline void orchkv_config_default(orchkv_config_t *cfg)
{
    cfg->gpu_device_id     = 0;
    cfg->gpu_pool_bytes    = (size_t)KV_DEFAULT_GPU_POOL_GB << 30;
    cfg->dram_pool_bytes   = (size_t)KV_DEFAULT_DRAM_POOL_GB << 30;
    cfg->dram_use_pinned   = true;
    cfg->num_cuda_streams  = KV_DEFAULT_NUM_STREAMS;
    cfg->tokens_per_block  = KV_TOKEN_GROUP_SIZE;
    cfg->d_head            = 128;
    cfg->dtype             = DTYPE_FP16;
    cfg->gpu_hwm           = 0.9f;
    cfg->gpu_lwm           = 0.7f;
    cfg->dram_hwm          = 0.9f;
    cfg->dram_lwm          = 0.7f;
    cfg->orchfs_nvm_path   = NULL;
    cfg->orchfs_ssd_path   = NULL;
    cfg->nvm_io_threads    = 4;
    cfg->ssd_io_threads    = 16;
}

/* ========================================================================
 *  Runtime Statistics
 * ======================================================================== */

typedef struct orchkv_stats {
    /* Pool utilisation */
    size_t   gpu_pool_total;
    size_t   gpu_pool_used;
    uint32_t gpu_slabs_total;
    uint32_t gpu_slabs_used;

    size_t   dram_pool_total;
    size_t   dram_pool_used;
    uint32_t dram_slabs_total;
    uint32_t dram_slabs_used;

    /* Block counts */
    uint64_t total_blocks;
    uint64_t blocks_on_gpu;
    uint64_t blocks_on_dram;
    uint64_t blocks_on_nvm;          /* Phase B */
    uint64_t blocks_on_ssd;          /* Phase B */

    /* Transfer counters */
    uint64_t transfers_d2h;          /* GPU → DRAM completed */
    uint64_t transfers_h2d;          /* DRAM → GPU completed */
    uint64_t bytes_d2h;
    uint64_t bytes_h2d;

    /* Active requests */
    uint32_t active_requests;
} orchkv_stats_t;

/* ========================================================================
 *  CUDA Error Checking (only included when compiled with nvcc)
 * ======================================================================== */

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            LOG_ERR("CUDA error %d (%s) at %s:%d",                            \
                    (int)_err, cudaGetErrorString(_err),                        \
                    __FILE__, __LINE__);                                        \
            return ORCHKV_ERR_CUDA;                                            \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                 \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            LOG_ERR("CUDA error %d (%s) at %s:%d",                            \
                    (int)_err, cudaGetErrorString(_err),                        \
                    __FILE__, __LINE__);                                        \
        }                                                                      \
    } while (0)

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_KV_TYPES_H */
