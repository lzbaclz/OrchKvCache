#include "adaptive_threshold.h"
#include <string.h>

/* ========================================================================
 *  Helpers
 * ======================================================================== */

static inline float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static double timespec_sec(const struct timespec *ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec * 1e-9;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return timespec_sec(&ts);
}

static inline bool cooldown_elapsed(const adaptive_threshold_t *a)
{
    return (now_sec() - timespec_sec(&a->last_adjust)) >= a->cooldown_sec;
}

static void stamp_now(adaptive_threshold_t *a)
{
    clock_gettime(CLOCK_MONOTONIC, &a->last_adjust);
}

static void push_to_classifier(adaptive_threshold_t *a)
{
    if (a->classifier)
        hcc_set_thresholds(a->classifier, a->threshold_hot, a->threshold_warm);
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int athresh_init(adaptive_threshold_t *a,
                 hotcold_classifier_t *classifier,
                 const athresh_params_t *p)
{
    if (!a || !p) return ORCHKV_ERR_INVALID;

    memset(a, 0, sizeof(*a));

    a->threshold_hot  = p->threshold_hot;
    a->threshold_warm = p->threshold_warm;
    a->min_hot  = p->min_hot;   a->max_hot  = p->max_hot;
    a->min_warm = p->min_warm;  a->max_warm = p->max_warm;
    a->gpu_hwm  = p->gpu_hwm;  a->gpu_lwm  = p->gpu_lwm;
    a->dram_hwm = p->dram_hwm; a->dram_lwm = p->dram_lwm;
    a->adjust_step  = p->adjust_step > 0.0f ? p->adjust_step : 0.02f;
    a->cooldown_sec = p->cooldown_sec >= 0.0 ? p->cooldown_sec : 0.5;
    a->classifier   = classifier;

    stamp_now(a);
    pthread_mutex_init(&a->lock, NULL);

    push_to_classifier(a);
    return ORCHKV_OK;
}

void athresh_destroy(adaptive_threshold_t *a)
{
    if (!a) return;
    pthread_mutex_destroy(&a->lock);
}

/* ========================================================================
 *  Core: threshold adaptation
 * ======================================================================== */

bool athresh_update(adaptive_threshold_t *a,
                    float gpu_used_ratio,
                    float dram_used_ratio)
{
    if (!a) return false;

    pthread_mutex_lock(&a->lock);

    if (!cooldown_elapsed(a)) {
        pthread_mutex_unlock(&a->lock);
        return false;
    }

    bool changed = false;
    float step = a->adjust_step;

    /* --- GPU tier: adjust threshold_hot --- */
    if (gpu_used_ratio > a->gpu_hwm) {
        /* Overloaded → raise threshold (evict more) */
        float new_hot = clampf(a->threshold_hot + step,
                               a->min_hot, a->max_hot);
        if (new_hot != a->threshold_hot) {
            a->threshold_hot = new_hot;
            a->adjustments_up++;
            changed = true;
        }
    } else if (gpu_used_ratio < a->gpu_lwm) {
        /* Under-utilised → lower threshold (keep more on GPU) */
        float new_hot = clampf(a->threshold_hot - step,
                               a->min_hot, a->max_hot);
        if (new_hot != a->threshold_hot) {
            a->threshold_hot = new_hot;
            a->adjustments_down++;
            changed = true;
        }
    }

    /* --- DRAM tier: adjust threshold_warm --- */
    if (dram_used_ratio > a->dram_hwm) {
        float new_warm = clampf(a->threshold_warm + step,
                                a->min_warm, a->max_warm);
        if (new_warm != a->threshold_warm) {
            a->threshold_warm = new_warm;
            a->adjustments_up++;
            changed = true;
        }
    } else if (dram_used_ratio < a->dram_lwm) {
        float new_warm = clampf(a->threshold_warm - step,
                                a->min_warm, a->max_warm);
        if (new_warm != a->threshold_warm) {
            a->threshold_warm = new_warm;
            a->adjustments_down++;
            changed = true;
        }
    }

    if (changed) {
        stamp_now(a);
        push_to_classifier(a);
    }

    pthread_mutex_unlock(&a->lock);
    return changed;
}

/* ========================================================================
 *  Demote checks
 * ======================================================================== */

bool athresh_should_demote_gpu(adaptive_threshold_t *a, float gpu_used_ratio)
{
    if (!a) return false;
    pthread_mutex_lock(&a->lock);
    a->demote_checks++;
    bool need = (gpu_used_ratio > a->gpu_hwm);
    pthread_mutex_unlock(&a->lock);
    return need;
}

bool athresh_should_demote_dram(adaptive_threshold_t *a, float dram_used_ratio)
{
    if (!a) return false;
    pthread_mutex_lock(&a->lock);
    a->demote_checks++;
    bool need = (dram_used_ratio > a->dram_hwm);
    pthread_mutex_unlock(&a->lock);
    return need;
}

/* ========================================================================
 *  Getters / overrides
 * ======================================================================== */

float athresh_get_hot(const adaptive_threshold_t *a)
{
    return a ? a->threshold_hot : 0.5f;
}

float athresh_get_warm(const adaptive_threshold_t *a)
{
    return a ? a->threshold_warm : 0.2f;
}

void athresh_force_thresholds(adaptive_threshold_t *a,
                              float threshold_hot,
                              float threshold_warm)
{
    if (!a) return;
    pthread_mutex_lock(&a->lock);
    a->threshold_hot  = clampf(threshold_hot,  a->min_hot,  a->max_hot);
    a->threshold_warm = clampf(threshold_warm, a->min_warm, a->max_warm);
    push_to_classifier(a);
    pthread_mutex_unlock(&a->lock);
}
