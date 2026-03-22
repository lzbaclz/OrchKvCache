#ifndef ORCHKV_ADAPTIVE_THRESHOLD_H
#define ORCHKV_ADAPTIVE_THRESHOLD_H

#include "../core/kv_types.h"
#include "hotcold_classifier.h"
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Adaptive Threshold — Phase C (C3)
 *
 *  Dynamically adjusts the Hot/Warm classification thresholds based on
 *  real-time GPU and DRAM utilisation, using high/low water marks and
 *  a cooldown timer to prevent oscillation.
 *
 *  Logic:
 *    GPU used > HWM  →  raise threshold_hot  (evict more aggressively)
 *    GPU used < LWM  →  lower threshold_hot  (keep more on GPU)
 *    DRAM used > HWM →  raise threshold_warm (evict more to storage)
 *    DRAM used < LWM →  lower threshold_warm (keep more in DRAM)
 *
 *  The thresholds are clamped to [min, max] and adjustments are gated
 *  by a cooldown period to avoid rapid flapping.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

typedef struct adaptive_threshold {
    /* Current thresholds (pushed to classifier on change) */
    float   threshold_hot;
    float   threshold_warm;

    /* Clamp ranges */
    float   min_hot,  max_hot;
    float   min_warm, max_warm;

    /* Water marks (ratios in [0, 1]) */
    float   gpu_hwm,  gpu_lwm;
    float   dram_hwm, dram_lwm;

    /* Adjustment control */
    float   adjust_step;        /* per-adjustment delta (default 0.02) */
    double  cooldown_sec;       /* min seconds between adjustments (default 0.5) */
    struct timespec last_adjust;

    /* Statistics */
    uint64_t adjustments_up;    /* times threshold was raised */
    uint64_t adjustments_down;  /* times threshold was lowered */
    uint64_t demote_checks;     /* total should_demote calls */

    /* Optional: push threshold changes to classifier */
    hotcold_classifier_t *classifier;

    pthread_mutex_t lock;
} adaptive_threshold_t;

/* ---- Configuration ---------------------------------------------------- */

typedef struct athresh_params {
    float threshold_hot;        /* initial (default 0.5) */
    float threshold_warm;       /* initial (default 0.2) */
    float min_hot,  max_hot;    /* clamp (default 0.2, 0.9) */
    float min_warm, max_warm;   /* clamp (default 0.05, 0.5) */
    float gpu_hwm,  gpu_lwm;   /* GPU water marks (default 0.9, 0.7) */
    float dram_hwm, dram_lwm;  /* DRAM water marks (default 0.9, 0.7) */
    float adjust_step;          /* default 0.02 */
    double cooldown_sec;        /* default 0.5 */
} athresh_params_t;

static inline void athresh_params_default(athresh_params_t *p)
{
    p->threshold_hot  = 0.5f;
    p->threshold_warm = 0.2f;
    p->min_hot  = 0.2f;   p->max_hot  = 0.9f;
    p->min_warm = 0.05f;  p->max_warm = 0.5f;
    p->gpu_hwm  = 0.9f;   p->gpu_lwm  = 0.7f;
    p->dram_hwm = 0.9f;   p->dram_lwm = 0.7f;
    p->adjust_step  = 0.02f;
    p->cooldown_sec = 0.5;
}

/* ---- Lifecycle -------------------------------------------------------- */

/*
 * Initialise the adaptive threshold controller.
 *   classifier — optional (may be NULL); if set, threshold changes are
 *                automatically pushed via hcc_set_thresholds().
 */
int athresh_init(adaptive_threshold_t *a,
                 hotcold_classifier_t *classifier,
                 const athresh_params_t *params);

void athresh_destroy(adaptive_threshold_t *a);

/* ---- Core operations -------------------------------------------------- */

/*
 * Feed current GPU and DRAM usage ratios.
 * Adjusts thresholds if water marks are breached and cooldown has elapsed.
 *
 *   gpu_used_ratio  — GPU pool utilisation in [0, 1]
 *   dram_used_ratio — DRAM pool utilisation in [0, 1]
 *
 * Returns true if any threshold was adjusted.
 */
bool athresh_update(adaptive_threshold_t *a,
                    float gpu_used_ratio,
                    float dram_used_ratio);

/* Query whether the tier needs eviction right now. */
bool athresh_should_demote_gpu(adaptive_threshold_t *a, float gpu_used_ratio);
bool athresh_should_demote_dram(adaptive_threshold_t *a, float dram_used_ratio);

/* ---- Getters ---------------------------------------------------------- */

float athresh_get_hot(const adaptive_threshold_t *a);
float athresh_get_warm(const adaptive_threshold_t *a);

/* ---- Runtime override ------------------------------------------------- */

void athresh_force_thresholds(adaptive_threshold_t *a,
                              float threshold_hot,
                              float threshold_warm);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_ADAPTIVE_THRESHOLD_H */
