#ifndef ORCHKV_TIERED_MANAGER_H
#define ORCHKV_TIERED_MANAGER_H

#include "../core/kv_types.h"
#include "../core/kv_block.h"
#include "attention_tracker.h"
#include "hotcold_classifier.h"
#include "adaptive_threshold.h"
#include "eviction_policy.h"
#include "prefetch_scheduler.h"
#include "pipeline.h"
#include "migration_engine.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Tiered Manager — Phase C (C8)
 *
 *  Integrates all C1-C7 scheduler components into a single coordinated
 *  system.  Provides the unified interface for:
 *
 *    - Block registration and lifecycle
 *    - Attention score ingestion
 *    - Periodic scheduling (demote overloaded tiers, dispatch prefetches)
 *    - Runtime policy tuning
 *    - Aggregated statistics
 *
 *  The scheduler can run as a background thread (auto_schedule=true)
 *  or be driven manually via tm_schedule_once() for deterministic testing.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

/* ---------- Allocator callbacks ---------------------------------------- */

typedef void *(*tm_alloc_fn)(size_t size, StorageTier tier, void *ctx);
typedef void  (*tm_free_fn)(void *ptr, StorageTier tier, void *ctx);

/* ---------- Configuration ---------------------------------------------- */

typedef struct tm_config {
    /* Attention tracker */
    uint32_t tracker_capacity;
    float    ema_lambda;

    /* Classifier */
    hcc_params_t hcc_params;

    /* Adaptive threshold */
    athresh_params_t athresh_params;

    /* Eviction policy */
    float    w_heat, w_lru;
    uint32_t tokens_per_block;

    /* Prefetch scheduler */
    uint32_t prefetch_budget;
    float    threshold_to_gpu;
    float    threshold_to_dram;

    /* Migration engine transfer callback */
    mig_transfer_fn transfer_fn;
    void           *transfer_ctx;

    /* Buffer allocator for migration destinations */
    tm_alloc_fn alloc_fn;
    tm_free_fn  free_fn;
    void       *alloc_ctx;
    size_t      block_data_size;

    /* Scheduler loop */
    uint32_t schedule_interval_us;   /* default 1000 (1ms) */
    uint32_t demote_batch_size;      /* default 8 */
    uint32_t prefetch_batch_size;    /* default 8 */
    uint32_t max_blocks;             /* max tracked blocks (default 4096) */
    bool     auto_schedule;          /* start background thread */
} tm_config_t;

void tm_config_default(tm_config_t *cfg);

/* ---------- Aggregated statistics -------------------------------------- */

typedef struct tm_stats {
    uint64_t schedule_cycles;
    uint64_t gpu_demotes;
    uint64_t dram_demotes;
    uint64_t prefetches_dispatched;
    float    gpu_used_ratio;
    float    dram_used_ratio;

    hcc_stats_t      hcc_stats;
    prefetch_stats_t prefetch_stats;
    mig_stats_t      migration_stats;
} tm_stats_t;

/* ---------- Manager state ---------------------------------------------- */

typedef struct tiered_manager {
    /* Sub-systems (owned, embedded) */
    attention_tracker_t   tracker;
    hotcold_classifier_t  classifier;
    adaptive_threshold_t  threshold;
    eviction_policy_t     evpol;
    prefetch_scheduler_t  prefetch;
    migration_engine_t    migration;
    pipeline_t            pipeline;

    /* Scheduler thread */
    bool       running;
    bool       auto_schedule;
    pthread_t  sched_thread;

    /* Scheduler parameters */
    uint32_t   schedule_interval_us;
    uint32_t   demote_batch_size;
    uint32_t   prefetch_batch_size;

    /* Current tier usage (set by caller via tm_set_usage) */
    float      gpu_used_ratio;
    float      dram_used_ratio;

    /* Buffer allocator */
    tm_alloc_fn  alloc_fn;
    tm_free_fn   free_fn;
    void        *alloc_ctx;
    size_t       block_data_size;

    /* Active block registry */
    kv_block_t **blocks;
    uint32_t     n_blocks;
    uint32_t     max_blocks;

    /* Scheduler stats */
    uint64_t   schedule_cycles;
    uint64_t   gpu_demotes;
    uint64_t   dram_demotes;
    uint64_t   prefetches_dispatched;

    pthread_rwlock_t block_lock;
    pthread_mutex_t  lock;
} tiered_manager_t;

/* ---- Lifecycle -------------------------------------------------------- */

int  tm_init(tiered_manager_t *m, const tm_config_t *cfg);
void tm_destroy(tiered_manager_t *m);

/* ---- Block registration ----------------------------------------------- */

int  tm_register_block(tiered_manager_t *m, kv_block_t *blk);
void tm_unregister_block(tiered_manager_t *m, kv_block_t *blk);

/* ---- Notifications from inference engine ------------------------------ */

/*
 * Report an attention score for a block during the current decode step.
 */
void tm_notify_attn(tiered_manager_t *m,
                    uint64_t block_id,
                    float attn_weight);

/*
 * Notify that a block was accessed (updates LRU).
 */
void tm_notify_access(tiered_manager_t *m, kv_block_t *blk);

/*
 * Mark the current decode step as complete.
 * Advances the attention tracker and resets per-step prefetch state.
 */
void tm_step_done(tiered_manager_t *m);

/* ---- Scheduling ------------------------------------------------------- */

/*
 * Run one iteration of the scheduling loop (synchronous).
 * Suitable for deterministic testing without a background thread.
 *
 * Steps:
 *   1. Update classification (hcc_update_all)
 *   2. Update thresholds (athresh_update)
 *   3. GPU demote check → select victims → migrate
 *   4. DRAM demote check → select victims → migrate
 *   5. Prefetch scan → dispatch
 */
void tm_schedule_once(tiered_manager_t *m);

/* Start / stop the background scheduler thread. */
int  tm_start(tiered_manager_t *m);
void tm_stop(tiered_manager_t *m);

/* ---- External state updates ------------------------------------------- */

void tm_set_usage(tiered_manager_t *m,
                  float gpu_used_ratio,
                  float dram_used_ratio);

/* ---- Runtime policy tuning -------------------------------------------- */

void tm_set_policy(tiered_manager_t *m,
                   float alpha, float beta, float gamma);

/* ---- Per-block query -------------------------------------------------- */

/*
 * Get the composite hotness score for a block from the classifier.
 * Returns 0.0f if the block is not registered.
 */
float tm_get_block_score(const tiered_manager_t *m, uint64_t block_id);

/*
 * Get the heat level for a block (HOT / WARM / COLD).
 * Returns HEAT_COLD if the block is not registered.
 */
HeatLevel tm_get_block_heat(const tiered_manager_t *m, uint64_t block_id);

/* ---- Statistics ------------------------------------------------------- */

void tm_get_stats(const tiered_manager_t *m, tm_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_TIERED_MANAGER_H */
