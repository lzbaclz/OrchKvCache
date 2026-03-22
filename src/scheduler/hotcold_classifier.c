#include "hotcold_classifier.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ========================================================================
 *  Helpers
 * ======================================================================== */

static uint32_t next_pow2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static inline uint32_t slot_of(const hotcold_classifier_t *c, uint64_t block_id)
{
    return (uint32_t)(block_id & c->mask);
}

static void build_decay_table(float *table, float tau)
{
    if (tau < 1.0f) tau = 1.0f;
    float inv_tau = 1.0f / tau;
    for (int k = 0; k < HCC_DECAY_TABLE_MAX; k++) {
        table[k] = expf(-(float)k * inv_tau);
    }
}

static inline float recency_score(const hotcold_classifier_t *c,
                                  uint64_t steps_since_hit)
{
    if (steps_since_hit >= HCC_DECAY_TABLE_MAX)
        return 0.0f;
    return c->decay_table[(uint32_t)steps_since_hit];
}

static inline float frequency_score(uint32_t query_hits, uint64_t total_steps)
{
    if (total_steps == 0) return 0.0f;
    float f = (float)query_hits / (float)total_steps;
    return f > 1.0f ? 1.0f : f;
}

static inline HeatLevel classify(float hotness,
                                 float th_hot,
                                 float th_warm,
                                 uint8_t flags)
{
    if (flags & KV_FLAG_ATTN_SINK)
        return HEAT_HOT;
    if (hotness >= th_hot)
        return HEAT_HOT;
    if (hotness >= th_warm)
        return HEAT_WARM;
    return HEAT_COLD;
}

/*
 * Compute hotness for a single block given its attention stats.
 * Does not acquire any lock — caller is responsible.
 */
static float compute_hotness(const hotcold_classifier_t *c,
                             const attn_stats_t *as,
                             uint64_t current_step)
{
    float attn   = as->ema;
    uint64_t gap = (as->last_hit_step == UINT64_MAX)
                       ? current_step   /* never accessed: full gap */
                       : (current_step > as->last_hit_step
                             ? current_step - as->last_hit_step
                             : 0);
    float rec    = recency_score(c, gap);
    float freq   = frequency_score(as->query_hits, current_step);

    return c->alpha * attn + c->beta * rec + c->gamma * freq;
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int hcc_init(hotcold_classifier_t *c,
             attention_tracker_t *tracker,
             const hcc_params_t *params)
{
    if (!c || !tracker || !params)
        return ORCHKV_ERR_INVALID;

    memset(c, 0, sizeof(*c));

    uint32_t cap = next_pow2(tracker->capacity < 64 ? 64 : tracker->capacity);

    c->slots = (hcc_slot_t *)calloc(cap, sizeof(hcc_slot_t));
    if (!c->slots)
        return ORCHKV_ERR_OOM;

    c->capacity       = cap;
    c->mask           = cap - 1;
    c->alpha          = params->alpha;
    c->beta           = params->beta;
    c->gamma          = params->gamma;
    c->recency_tau    = params->recency_tau;
    c->threshold_hot  = params->threshold_hot;
    c->threshold_warm = params->threshold_warm;
    c->tracker        = tracker;

    build_decay_table(c->decay_table, c->recency_tau);
    pthread_mutex_init(&c->lock, NULL);

    return ORCHKV_OK;
}

void hcc_destroy(hotcold_classifier_t *c)
{
    if (!c) return;
    free(c->slots);
    c->slots    = NULL;
    c->capacity = 0;
    pthread_mutex_destroy(&c->lock);
}

/* ========================================================================
 *  Block registration
 * ======================================================================== */

void hcc_register(hotcold_classifier_t *c, uint64_t block_id, uint8_t flags)
{
    if (!c || !c->slots) return;
    uint32_t idx = slot_of(c, block_id);

    pthread_mutex_lock(&c->lock);
    hcc_slot_t *s   = &c->slots[idx];
    s->hotness      = 0.0f;
    s->heat_level   = (flags & KV_FLAG_ATTN_SINK) ? HEAT_HOT : HEAT_COLD;
    s->block_flags  = flags;
    s->active       = true;
    pthread_mutex_unlock(&c->lock);
}

void hcc_unregister(hotcold_classifier_t *c, uint64_t block_id)
{
    if (!c || !c->slots) return;
    uint32_t idx = slot_of(c, block_id);

    pthread_mutex_lock(&c->lock);
    memset(&c->slots[idx], 0, sizeof(hcc_slot_t));
    pthread_mutex_unlock(&c->lock);
}

/* ========================================================================
 *  Classification
 * ======================================================================== */

void hcc_update_all(hotcold_classifier_t *c)
{
    if (!c || !c->slots || !c->tracker) return;

    pthread_mutex_lock(&c->lock);

    const attention_tracker_t *t = c->tracker;
    uint64_t step = t->current_step;

    for (uint32_t i = 0; i < c->capacity; i++) {
        hcc_slot_t *slot = &c->slots[i];
        if (!slot->active) continue;

        /* Read corresponding tracker stats (same index). */
        const attn_stats_t *as = &t->stats[i];
        if (!as->active) continue;

        float h = compute_hotness(c, as, step);
        slot->hotness    = h;
        slot->heat_level = classify(h, c->threshold_hot,
                                    c->threshold_warm, slot->block_flags);
    }

    pthread_mutex_unlock(&c->lock);
}

void hcc_update_block(hotcold_classifier_t *c, uint64_t block_id)
{
    if (!c || !c->slots || !c->tracker) return;

    uint32_t idx = slot_of(c, block_id);

    pthread_mutex_lock(&c->lock);
    hcc_slot_t *slot = &c->slots[idx];
    if (slot->active) {
        const attn_stats_t *as = &c->tracker->stats[idx];
        if (as->active) {
            float h = compute_hotness(c, as, c->tracker->current_step);
            slot->hotness    = h;
            slot->heat_level = classify(h, c->threshold_hot,
                                        c->threshold_warm, slot->block_flags);
        }
    }
    pthread_mutex_unlock(&c->lock);
}

/* ========================================================================
 *  Query
 * ======================================================================== */

HeatLevel hcc_get_heat(const hotcold_classifier_t *c, uint64_t block_id)
{
    if (!c || !c->slots) return HEAT_COLD;
    uint32_t idx = slot_of(c, block_id);
    const hcc_slot_t *s = &c->slots[idx];
    return s->active ? s->heat_level : HEAT_COLD;
}

float hcc_get_score(const hotcold_classifier_t *c, uint64_t block_id)
{
    if (!c || !c->slots) return 0.0f;
    uint32_t idx = slot_of(c, block_id);
    const hcc_slot_t *s = &c->slots[idx];
    return s->active ? s->hotness : 0.0f;
}

int hcc_get(const hotcold_classifier_t *c,
            uint64_t block_id,
            float *score_out,
            HeatLevel *level_out)
{
    if (!c || !c->slots) return ORCHKV_ERR_INVALID;
    uint32_t idx = slot_of(c, block_id);
    const hcc_slot_t *s = &c->slots[idx];
    if (!s->active)
        return ORCHKV_ERR_NOT_FOUND;
    if (score_out) *score_out = s->hotness;
    if (level_out) *level_out = s->heat_level;
    return ORCHKV_OK;
}

/* ========================================================================
 *  Runtime tuning
 * ======================================================================== */

void hcc_set_thresholds(hotcold_classifier_t *c,
                        float threshold_hot,
                        float threshold_warm)
{
    if (!c) return;
    pthread_mutex_lock(&c->lock);
    c->threshold_hot  = threshold_hot;
    c->threshold_warm = threshold_warm;
    pthread_mutex_unlock(&c->lock);
}

void hcc_set_weights(hotcold_classifier_t *c,
                     float alpha, float beta, float gamma)
{
    if (!c) return;
    pthread_mutex_lock(&c->lock);
    c->alpha = alpha;
    c->beta  = beta;
    c->gamma = gamma;
    pthread_mutex_unlock(&c->lock);
}

/* ========================================================================
 *  Statistics
 * ======================================================================== */

void hcc_get_stats(const hotcold_classifier_t *c, hcc_stats_t *out)
{
    if (!c || !c->slots || !out) return;
    memset(out, 0, sizeof(*out));

    float sum_hotness = 0.0f;
    float max_h       = -FLT_MAX;
    float min_h       =  FLT_MAX;
    uint32_t n_active = 0;

    for (uint32_t i = 0; i < c->capacity; i++) {
        const hcc_slot_t *s = &c->slots[i];
        if (!s->active) continue;

        n_active++;
        sum_hotness += s->hotness;
        if (s->hotness > max_h) max_h = s->hotness;
        if (s->hotness < min_h) min_h = s->hotness;

        switch (s->heat_level) {
        case HEAT_HOT:  out->n_hot++;  break;
        case HEAT_WARM: out->n_warm++; break;
        case HEAT_COLD: out->n_cold++; break;
        }

        if (s->block_flags & KV_FLAG_ATTN_SINK)
            out->n_attn_sink++;
    }

    if (n_active > 0) {
        out->avg_hotness = sum_hotness / (float)n_active;
        out->max_hotness = max_h;
        out->min_hotness = min_h;
    }
}
