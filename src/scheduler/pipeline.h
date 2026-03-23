#ifndef ORCHKV_PIPELINE_H
#define ORCHKV_PIPELINE_H

#include "../core/kv_types.h"
#include "prefetch_scheduler.h"
#include <time.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Three-Stage Pipeline Coordinator — Phase C (C6)
 *
 *  Manages the IO-compute overlap for each decode step:
 *
 *    Step N:
 *      Stage A (Compute):   GPU attention(Q, K, V)  → produces attn scores
 *      Stage B (Prefetch):  Storage → DRAM           ← async, overlaps with A(N+1)
 *      Stage C (Transfer):  DRAM → GPU               ← async, before A(N+1)
 *
 *  The goal is to hide prefetch latency behind GPU compute. The pipeline
 *  measures per-stage durations and computes the overlap ratio — the
 *  fraction of IO time that was successfully hidden.
 *
 *  Usage:
 *    pipeline_step_begin(p);          // start of decode step
 *    ... GPU compute ...
 *    pipeline_compute_done(p);        // compute finished, kick prefetch
 *    ... async IO ...
 *    pipeline_prefetch_done(p, n);    // IO completed
 *    ... H2D transfer ...
 *    pipeline_transfer_done(p, n);    // transfer completed, step done
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

typedef enum {
    PIPELINE_IDLE      = 0,
    PIPELINE_COMPUTE   = 1,
    PIPELINE_PREFETCH  = 2,
    PIPELINE_TRANSFER  = 3,
} PipelineStage;

/* Per-step timing record (microseconds). */
typedef struct pipeline_step_record {
    uint64_t step;
    double   compute_us;
    double   prefetch_us;
    double   transfer_us;
    double   step_total_us;
    uint32_t n_prefetched;
    uint32_t n_transferred;
} pipeline_step_record_t;

/* Accumulated statistics across all steps. */
typedef struct pipeline_stats {
    uint64_t total_steps;
    double   sum_compute_us;
    double   sum_prefetch_us;
    double   sum_transfer_us;
    double   sum_step_us;

    double   avg_compute_us;
    double   avg_prefetch_us;
    double   avg_transfer_us;
    double   avg_step_us;

    /*
     * Overlap ratio: fraction of prefetch IO hidden behind compute.
     *   overlap = min(avg_prefetch, avg_compute) / avg_prefetch
     * Value in [0, 1].  1.0 = all IO hidden; 0.0 = no overlap.
     * Only meaningful when avg_prefetch > 0.
     */
    double   overlap_ratio;

    uint64_t total_prefetched;
    uint64_t total_transferred;
} pipeline_stats_t;

/* ---------- Pipeline state --------------------------------------------- */

typedef struct pipeline {
    PipelineStage stage;
    uint64_t      step_number;

    /* Timestamps for current step */
    struct timespec ts_step_begin;
    struct timespec ts_compute_done;
    struct timespec ts_prefetch_done;
    struct timespec ts_transfer_done;

    /* Current step counters */
    uint32_t cur_prefetched;
    uint32_t cur_transferred;

    /* Accumulated stats (microseconds) */
    double   sum_compute_us;
    double   sum_prefetch_us;
    double   sum_transfer_us;
    double   sum_step_us;
    uint64_t total_steps;
    uint64_t total_prefetched;
    uint64_t total_transferred;

    /* Last completed step record */
    pipeline_step_record_t last_record;

    /* Optional prefetch scheduler reference */
    prefetch_scheduler_t *prefetch;

    pthread_mutex_t lock;
} pipeline_t;

/* ---- Lifecycle -------------------------------------------------------- */

int  pipeline_init(pipeline_t *p, prefetch_scheduler_t *prefetch);
void pipeline_destroy(pipeline_t *p);

/* ---- Step lifecycle --------------------------------------------------- */

void pipeline_step_begin(pipeline_t *p);
void pipeline_compute_done(pipeline_t *p);
void pipeline_prefetch_done(pipeline_t *p, uint32_t n_prefetched);
void pipeline_transfer_done(pipeline_t *p, uint32_t n_transferred);

/* ---- Query ------------------------------------------------------------ */

PipelineStage pipeline_current_stage(const pipeline_t *p);
uint64_t      pipeline_step_number(const pipeline_t *p);

void pipeline_get_last_record(const pipeline_t *p, pipeline_step_record_t *out);
void pipeline_get_stats(const pipeline_t *p, pipeline_stats_t *out);

/*
 * Pure function: compute overlap ratio from average durations.
 * Can be used independently for testing.
 */
double pipeline_compute_overlap(double avg_compute_us,
                                double avg_prefetch_us);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_PIPELINE_H */
