#ifndef ORCHKV_PREFETCH_SCHEDULER_H
#define ORCHKV_PREFETCH_SCHEDULER_H

#include "../core/kv_types.h"
#include "attention_tracker.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Prefetch Scheduler — Phase C (C5)
 *
 *  Predicts which KV blocks will be accessed next based on attention EMA,
 *  ranks candidates via a max-heap priority queue, and tracks hit rate.
 *
 *  Prefetch trigger rules (per block B not on GPU):
 *    attn_ema(B) > threshold_to_gpu  → candidate for GPU  (high priority)
 *    attn_ema(B) > threshold_to_dram → candidate for DRAM (lower priority)
 *
 *  Budget: at most `prefetch_budget` candidates dispatched per step.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

/* ---------- Prefetch candidate entry ----------------------------------- */

typedef struct prefetch_entry {
    uint64_t    block_id;
    StorageTier current_tier;
    StorageTier target_tier;
    float       priority;       /* higher = prefetch first */
} prefetch_entry_t;

/* ---------- Block info for scan input ---------------------------------- */

typedef struct prefetch_block_info {
    uint64_t    block_id;
    StorageTier tier;
} prefetch_block_info_t;

/* ---------- Statistics ------------------------------------------------- */

typedef struct prefetch_stats {
    uint64_t total_scanned;     /* blocks examined during scans */
    uint64_t total_enqueued;    /* candidates added to heap */
    uint64_t total_dispatched;  /* candidates popped via dispatch */
    uint64_t prefetch_hits;     /* dispatched blocks actually accessed */
    uint64_t prefetch_wasted;   /* dispatched blocks never accessed (step reset) */
    float    hit_rate;          /* hits / dispatched (0 if none dispatched) */
} prefetch_stats_t;

/* ---------- Prefetch scheduler ----------------------------------------- */

#define PREFETCH_TRACK_CAP  512   /* max concurrently tracked dispatched entries */

typedef struct prefetch_scheduler {
    /* Max-heap of candidates (sorted by priority descending) */
    prefetch_entry_t *heap;
    uint32_t          heap_size;
    uint32_t          heap_cap;

    /* Configuration */
    uint32_t  prefetch_budget;      /* max candidates per dispatch (default 16) */
    float     threshold_to_gpu;     /* attn_ema ≥ this → prefetch to GPU */
    float     threshold_to_dram;    /* attn_ema ≥ this → prefetch to DRAM */

    /* Reference to attention tracker (read-only) for EMA lookups */
    attention_tracker_t *tracker;

    /* Hit tracking: recently dispatched block IDs */
    uint64_t  tracked_ids[PREFETCH_TRACK_CAP];
    uint32_t  tracked_count;

    /* Accumulated statistics */
    uint64_t  total_scanned;
    uint64_t  total_enqueued;
    uint64_t  total_dispatched;
    uint64_t  prefetch_hits;
    uint64_t  prefetch_wasted;

    pthread_mutex_t lock;
} prefetch_scheduler_t;

/* ---- Lifecycle -------------------------------------------------------- */

int  prefetch_init(prefetch_scheduler_t *s,
                   attention_tracker_t *tracker,
                   uint32_t budget,
                   float threshold_to_gpu,
                   float threshold_to_dram,
                   uint32_t heap_cap);

void prefetch_destroy(prefetch_scheduler_t *s);

/* ---- Candidate management --------------------------------------------- */

/*
 * Scan an array of blocks, look up EMA from the tracker, and add
 * qualifying candidates to the priority queue.
 *
 * Blocks already on GPU are skipped.  Priority is:
 *   - GPU target: priority = ema
 *   - DRAM target: priority = ema × 0.5  (lower to prefer GPU prefetches)
 */
void prefetch_scan_blocks(prefetch_scheduler_t *s,
                          const prefetch_block_info_t *blocks,
                          uint32_t n_blocks);

/*
 * Manually add a candidate (useful for testing or external logic).
 * Returns ORCHKV_OK or ORCHKV_ERR_TIER_FULL (heap full).
 */
int  prefetch_add_candidate(prefetch_scheduler_t *s,
                            uint64_t block_id,
                            StorageTier current_tier,
                            StorageTier target_tier,
                            float priority);

/* ---- Dispatch --------------------------------------------------------- */

/*
 * Pop up to `max_n` highest-priority candidates from the heap.
 * Each popped candidate is added to the tracking set.
 *
 * Returns the actual number of candidates dispatched (≤ min(max_n, budget)).
 * `out` must have space for at least `max_n` entries.
 */
uint32_t prefetch_dispatch(prefetch_scheduler_t *s,
                           uint32_t max_n,
                           prefetch_entry_t *out);

/* ---- Hit tracking ----------------------------------------------------- */

/*
 * Notify that a previously-dispatched block was actually accessed.
 * If the block_id is in the tracking set, increment hits.
 * Returns true if the block was found in the tracking set.
 */
bool prefetch_notify_hit(prefetch_scheduler_t *s, uint64_t block_id);

/*
 * Reset for a new step: remaining tracked entries count as wasted.
 * Clears the heap and tracking set.
 */
void prefetch_step_reset(prefetch_scheduler_t *s);

/* ---- Statistics ------------------------------------------------------- */

float prefetch_hit_rate(const prefetch_scheduler_t *s);

void  prefetch_get_stats(const prefetch_scheduler_t *s, prefetch_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_PREFETCH_SCHEDULER_H */
