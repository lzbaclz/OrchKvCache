#ifndef ORCHKV_ATTENTION_TRACKER_H
#define ORCHKV_ATTENTION_TRACKER_H

#include "../core/kv_types.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Attention Tracker — Phase C (C1)
 *
 *  Collects per-block attention statistics during decode steps.
 *  Provides raw data for the hotcold_classifier (C2) to compute hotness.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 *  Indexing: block_id % capacity  (open-addressing, no collision handling
 *            — caller ensures capacity ≫ concurrent block count).
 * ======================================================================== */

/* Per-block attention statistics, aggregated per decode step. */
typedef struct attn_stats {
    float       sum;            /* cumulative attention weight this step */
    float       max;            /* peak single-token attention this step */
    float       ema;            /* exponential moving average of per-step sum */
    uint32_t    query_hits;     /* total decode steps that touched this block */
    uint64_t    last_hit_step;  /* most recent step that updated this block */
    bool        active;         /* slot occupied by a live block */
} attn_stats_t;

typedef struct attention_tracker {
    attn_stats_t   *stats;          /* array indexed by block_id % capacity */
    uint32_t        capacity;       /* slot count (power-of-2 recommended) */
    uint32_t        mask;           /* capacity - 1, for fast modulo */
    uint64_t        current_step;   /* monotonically increasing step counter */
    float           ema_lambda;     /* EMA decay factor (default 0.9) */

    pthread_mutex_t lock;
} attention_tracker_t;

/* ---- Lifecycle -------------------------------------------------------- */

/*
 * Initialise the tracker.
 *   capacity  — max tracking slots (should be ≥ max concurrent blocks).
 *               Rounded up to next power-of-2 internally.
 *   ema_lambda — EMA decay factor in [0, 1).  0.9 is a good default.
 * Returns ORCHKV_OK or ORCHKV_ERR_OOM.
 */
int attn_tracker_init(attention_tracker_t *t,
                      uint32_t capacity,
                      float ema_lambda);

/* Free internal allocations and destroy the mutex. */
void attn_tracker_destroy(attention_tracker_t *t);

/* ---- Per-step operations ---------------------------------------------- */

/*
 * Record an attention weight for a single block in the current step.
 * Safe to call from multiple threads concurrently.
 *
 *   block_id    — globally unique block identifier
 *   attn_weight — aggregated attention score for this block (≥ 0)
 */
int attn_tracker_update(attention_tracker_t *t,
                        uint64_t block_id,
                        float attn_weight);

/*
 * Batch update: record attention weights for multiple blocks at once.
 * More efficient than calling attn_tracker_update() in a loop
 * (single lock acquisition).
 */
int attn_tracker_update_batch(attention_tracker_t *t,
                              const uint64_t *block_ids,
                              const float *weights,
                              uint32_t count);

/*
 * Mark the current decode step as complete.
 * Flushes per-step accumulators (sum, max) into the EMA,
 * then advances current_step.
 *
 * Must be called exactly once per decode step, after all updates.
 */
void attn_tracker_step_done(attention_tracker_t *t);

/* ---- Query ------------------------------------------------------------ */

/*
 * Read the attention statistics for a block.
 * Returns ORCHKV_OK, or ORCHKV_ERR_NOT_FOUND if the slot is inactive.
 */
int attn_tracker_get(const attention_tracker_t *t,
                     uint64_t block_id,
                     attn_stats_t *out);

/* Return the current decode step number. */
static inline uint64_t attn_tracker_current_step(const attention_tracker_t *t)
{
    return t->current_step;
}

/* ---- Slot management -------------------------------------------------- */

/*
 * Activate a slot for a newly created block.
 * Zeroes all stats.  Must be called before the first update.
 */
void attn_tracker_register(attention_tracker_t *t, uint64_t block_id);

/*
 * Reset (deactivate) a slot when a block is destroyed or evicted.
 * Clears all accumulated stats for this block_id.
 */
void attn_tracker_reset(attention_tracker_t *t, uint64_t block_id);

/* ---- Bulk operations -------------------------------------------------- */

/*
 * Decay EMA for all active slots that were NOT updated in the current step.
 * This ensures idle blocks' EMA naturally decays toward zero.
 * Called internally by attn_tracker_step_done(); exposed for testing.
 */
void attn_tracker_decay_idle(attention_tracker_t *t);

/* Return the number of active (registered) slots. */
uint32_t attn_tracker_active_count(const attention_tracker_t *t);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_ATTENTION_TRACKER_H */
