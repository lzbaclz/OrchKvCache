#ifndef ORCHKV_HOTCOLD_CLASSIFIER_H
#define ORCHKV_HOTCOLD_CLASSIFIER_H

#include "../core/kv_types.h"
#include "attention_tracker.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Hot/Cold Classifier — Phase C (C2)
 *
 *  Computes a composite hotness score for each KV block and classifies
 *  it into one of three heat levels (Hot / Warm / Cold).
 *
 *  Hotness formula:
 *    hotness(b, t) = α × attn_ema(b)
 *                  + β × recency_score(b, t)
 *                  + γ × frequency_score(b, t)
 *
 *  Where:
 *    attn_ema        = EMA of attention weights       (from C1 tracker)
 *    recency_score   = exp(-(t - last_hit) / τ)       (τ default 50)
 *    frequency_score = min(query_hits / t, 1.0)
 *
 *  Classification:
 *    hotness ≥ threshold_hot   → HEAT_HOT   (stays on GPU)
 *    hotness ≥ threshold_warm  → HEAT_WARM  (DRAM / NVM)
 *    hotness <  threshold_warm → HEAT_COLD  (NVM / SSD)
 *
 *  Blocks with KV_FLAG_ATTN_SINK are unconditionally HEAT_HOT.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

typedef enum {
    HEAT_HOT  = 0,
    HEAT_WARM = 1,
    HEAT_COLD = 2,
} HeatLevel;

static inline const char *heat_level_name(HeatLevel h)
{
    switch (h) {
    case HEAT_HOT:  return "HOT";
    case HEAT_WARM: return "WARM";
    case HEAT_COLD: return "COLD";
    default:        return "UNKNOWN";
    }
}

/* Per-slot classifier state (indexed identically to attention_tracker). */
typedef struct hcc_slot {
    float       hotness;        /* composite score in [0, 1+] */
    HeatLevel   heat_level;
    uint8_t     block_flags;    /* mirrors kv_block_t.flags */
    bool        active;
} hcc_slot_t;

/* Precomputed recency decay table: decay_table[k] = exp(-k / τ). */
#define HCC_DECAY_TABLE_MAX  512

typedef struct hotcold_classifier {
    hcc_slot_t *slots;
    uint32_t    capacity;
    uint32_t    mask;           /* capacity - 1 */

    /* Hotness formula weights (must sum to ~1.0) */
    float       alpha;          /* attention EMA weight   (default 0.5) */
    float       beta;           /* recency weight         (default 0.3) */
    float       gamma;          /* frequency weight       (default 0.2) */

    /* Recency decay */
    float       recency_tau;    /* time constant in steps (default 50) */
    float       decay_table[HCC_DECAY_TABLE_MAX];

    /* Classification thresholds */
    float       threshold_hot;  /* ≥ this → HOT  (default 0.5) */
    float       threshold_warm; /* ≥ this → WARM (default 0.2) */

    /* Reference to the attention tracker (read-only). */
    attention_tracker_t *tracker;

    pthread_mutex_t lock;
} hotcold_classifier_t;

/* ---- Lifecycle -------------------------------------------------------- */

typedef struct hcc_params {
    float alpha, beta, gamma;
    float recency_tau;
    float threshold_hot;
    float threshold_warm;
} hcc_params_t;

/* Fill params with sensible defaults. */
static inline void hcc_params_default(hcc_params_t *p)
{
    p->alpha          = 0.5f;
    p->beta           = 0.3f;
    p->gamma          = 0.2f;
    p->recency_tau    = 50.0f;
    p->threshold_hot  = 0.5f;
    p->threshold_warm = 0.2f;
}

/*
 * Initialise the classifier.
 *   tracker — pointer to an already-initialised attention_tracker (C1).
 *             The classifier reads from it but never modifies it.
 *   params  — hotness formula parameters.
 * Returns ORCHKV_OK or ORCHKV_ERR_OOM / ORCHKV_ERR_INVALID.
 */
int hcc_init(hotcold_classifier_t *c,
             attention_tracker_t *tracker,
             const hcc_params_t *params);

void hcc_destroy(hotcold_classifier_t *c);

/* ---- Block registration ---------------------------------------------- */

/*
 * Register a block for classification.
 *   flags — kv_block_t.flags (KV_FLAG_ATTN_SINK etc.)
 */
void hcc_register(hotcold_classifier_t *c,
                  uint64_t block_id,
                  uint8_t flags);

void hcc_unregister(hotcold_classifier_t *c, uint64_t block_id);

/* ---- Classification -------------------------------------------------- */

/*
 * Recompute hotness and heat level for ALL active blocks.
 * Should be called once per decode step, after attn_tracker_step_done().
 * O(capacity) — run on a non-critical path.
 */
void hcc_update_all(hotcold_classifier_t *c);

/*
 * Recompute hotness and heat level for a single block.
 * Useful for on-demand queries without a full sweep.
 */
void hcc_update_block(hotcold_classifier_t *c, uint64_t block_id);

/* ---- Query ------------------------------------------------------------ */

/*
 * Get the heat level for a block.
 * Returns HEAT_COLD if the block is not registered.
 */
HeatLevel hcc_get_heat(const hotcold_classifier_t *c, uint64_t block_id);

/*
 * Get the raw hotness score for a block.
 * Returns 0.0f if the block is not registered.
 */
float hcc_get_score(const hotcold_classifier_t *c, uint64_t block_id);

/*
 * Get both score and level in one call (avoids double lookup).
 * Returns ORCHKV_OK or ORCHKV_ERR_NOT_FOUND.
 */
int hcc_get(const hotcold_classifier_t *c,
            uint64_t block_id,
            float *score_out,
            HeatLevel *level_out);

/* ---- Runtime tuning -------------------------------------------------- */

void hcc_set_thresholds(hotcold_classifier_t *c,
                        float threshold_hot,
                        float threshold_warm);

void hcc_set_weights(hotcold_classifier_t *c,
                     float alpha, float beta, float gamma);

/* ---- Statistics ------------------------------------------------------ */

typedef struct hcc_stats {
    uint32_t n_hot;
    uint32_t n_warm;
    uint32_t n_cold;
    uint32_t n_attn_sink;       /* blocks forced HOT by ATTN_SINK flag */
    float    avg_hotness;
    float    max_hotness;
    float    min_hotness;       /* among active blocks */
} hcc_stats_t;

void hcc_get_stats(const hotcold_classifier_t *c, hcc_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_HOTCOLD_CLASSIFIER_H */
