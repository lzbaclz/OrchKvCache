#include "attention_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 *  Helpers
 * ======================================================================== */

static uint32_t next_power_of_2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static inline uint32_t slot_of(const attention_tracker_t *t, uint64_t block_id)
{
    return (uint32_t)(block_id & t->mask);
}

/* Update a single slot's per-step accumulators (caller holds lock). */
static inline void update_slot(attn_stats_t *s, float w, uint64_t step)
{
    s->sum += w;
    if (w > s->max)
        s->max = w;
    s->last_hit_step = step;
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int attn_tracker_init(attention_tracker_t *t, uint32_t capacity, float ema_lambda)
{
    if (!t || capacity == 0)
        return ORCHKV_ERR_INVALID;

    memset(t, 0, sizeof(*t));

    uint32_t cap = next_power_of_2(capacity < 64 ? 64 : capacity);

    t->stats = (attn_stats_t *)calloc(cap, sizeof(attn_stats_t));
    if (!t->stats)
        return ORCHKV_ERR_OOM;

    t->capacity   = cap;
    t->mask       = cap - 1;
    t->ema_lambda = (ema_lambda > 0.0f && ema_lambda < 1.0f)
                        ? ema_lambda : 0.9f;
    t->current_step = 0;

    pthread_mutex_init(&t->lock, NULL);
    return ORCHKV_OK;
}

void attn_tracker_destroy(attention_tracker_t *t)
{
    if (!t) return;
    free(t->stats);
    t->stats    = NULL;
    t->capacity = 0;
    pthread_mutex_destroy(&t->lock);
}

/* ========================================================================
 *  Per-step operations
 * ======================================================================== */

int attn_tracker_update(attention_tracker_t *t,
                        uint64_t block_id,
                        float attn_weight)
{
    if (!t || !t->stats)
        return ORCHKV_ERR_INIT;

    uint32_t idx = slot_of(t, block_id);

    pthread_mutex_lock(&t->lock);
    attn_stats_t *s = &t->stats[idx];
    if (s->active) {
        update_slot(s, attn_weight, t->current_step);
    }
    pthread_mutex_unlock(&t->lock);

    return ORCHKV_OK;
}

int attn_tracker_update_batch(attention_tracker_t *t,
                              const uint64_t *block_ids,
                              const float *weights,
                              uint32_t count)
{
    if (!t || !t->stats)
        return ORCHKV_ERR_INIT;
    if (count == 0)
        return ORCHKV_OK;
    if (!block_ids || !weights)
        return ORCHKV_ERR_INVALID;

    pthread_mutex_lock(&t->lock);
    uint64_t step = t->current_step;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = slot_of(t, block_ids[i]);
        attn_stats_t *s = &t->stats[idx];
        if (s->active) {
            update_slot(s, weights[i], step);
        }
    }
    pthread_mutex_unlock(&t->lock);

    return ORCHKV_OK;
}

void attn_tracker_step_done(attention_tracker_t *t)
{
    if (!t || !t->stats) return;

    pthread_mutex_lock(&t->lock);

    float lambda     = t->ema_lambda;
    float one_minus  = 1.0f - lambda;
    uint64_t step    = t->current_step;

    for (uint32_t i = 0; i < t->capacity; i++) {
        attn_stats_t *s = &t->stats[i];
        if (!s->active) continue;

        if (s->last_hit_step == step) {
            /* Block was updated this step: blend into EMA */
            s->ema = lambda * s->ema + one_minus * s->sum;
            s->query_hits++;
        } else {
            /* Block was idle this step: decay EMA toward zero */
            s->ema *= lambda;
        }

        /* Reset per-step accumulators for next step */
        s->sum = 0.0f;
        s->max = 0.0f;
    }

    t->current_step++;
    pthread_mutex_unlock(&t->lock);
}

/* ========================================================================
 *  Query
 * ======================================================================== */

int attn_tracker_get(const attention_tracker_t *t,
                     uint64_t block_id,
                     attn_stats_t *out)
{
    if (!t || !t->stats || !out)
        return ORCHKV_ERR_INVALID;

    uint32_t idx = slot_of(t, block_id);

    /* Safe to read without lock for a snapshot — caller accepts
       slight staleness in exchange for no contention on the hot path. */
    const attn_stats_t *s = &t->stats[idx];
    if (!s->active)
        return ORCHKV_ERR_NOT_FOUND;

    *out = *s;
    return ORCHKV_OK;
}

/* ========================================================================
 *  Slot management
 * ======================================================================== */

void attn_tracker_register(attention_tracker_t *t, uint64_t block_id)
{
    if (!t || !t->stats) return;

    uint32_t idx = slot_of(t, block_id);

    pthread_mutex_lock(&t->lock);
    attn_stats_t *s = &t->stats[idx];
    memset(s, 0, sizeof(*s));
    s->active        = true;
    s->last_hit_step = UINT64_MAX;  /* sentinel: "never updated" */
    pthread_mutex_unlock(&t->lock);
}

void attn_tracker_reset(attention_tracker_t *t, uint64_t block_id)
{
    if (!t || !t->stats) return;

    uint32_t idx = slot_of(t, block_id);

    pthread_mutex_lock(&t->lock);
    memset(&t->stats[idx], 0, sizeof(attn_stats_t));
    pthread_mutex_unlock(&t->lock);
}

/* ========================================================================
 *  Bulk operations
 * ======================================================================== */

void attn_tracker_decay_idle(attention_tracker_t *t)
{
    if (!t || !t->stats) return;

    pthread_mutex_lock(&t->lock);
    float lambda  = t->ema_lambda;
    uint64_t step = t->current_step;

    for (uint32_t i = 0; i < t->capacity; i++) {
        attn_stats_t *s = &t->stats[i];
        if (s->active && s->last_hit_step != step) {
            s->ema *= lambda;
        }
    }
    pthread_mutex_unlock(&t->lock);
}

uint32_t attn_tracker_active_count(const attention_tracker_t *t)
{
    if (!t || !t->stats) return 0;

    uint32_t count = 0;
    for (uint32_t i = 0; i < t->capacity; i++) {
        if (t->stats[i].active)
            count++;
    }
    return count;
}
