#include "pipeline.h"
#include <string.h>

/* ========================================================================
 *  Internal helpers
 * ======================================================================== */

static void stamp(struct timespec *ts)
{
    clock_gettime(CLOCK_MONOTONIC, ts);
}

static double elapsed_us(const struct timespec *start,
                         const struct timespec *end)
{
    double sec  = (double)(end->tv_sec  - start->tv_sec);
    double nsec = (double)(end->tv_nsec - start->tv_nsec);
    return (sec * 1e6) + (nsec * 1e-3);
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int pipeline_init(pipeline_t *p, prefetch_scheduler_t *prefetch)
{
    if (!p) return ORCHKV_ERR_INVALID;
    memset(p, 0, sizeof(*p));
    p->stage    = PIPELINE_IDLE;
    p->prefetch = prefetch;
    pthread_mutex_init(&p->lock, NULL);
    return ORCHKV_OK;
}

void pipeline_destroy(pipeline_t *p)
{
    if (!p) return;
    pthread_mutex_destroy(&p->lock);
}

/* ========================================================================
 *  Step lifecycle
 * ======================================================================== */

void pipeline_step_begin(pipeline_t *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->lock);

    p->stage = PIPELINE_COMPUTE;
    stamp(&p->ts_step_begin);
    p->cur_prefetched  = 0;
    p->cur_transferred = 0;

    pthread_mutex_unlock(&p->lock);
}

void pipeline_compute_done(pipeline_t *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->lock);

    stamp(&p->ts_compute_done);
    p->stage = PIPELINE_PREFETCH;

    pthread_mutex_unlock(&p->lock);
}

void pipeline_prefetch_done(pipeline_t *p, uint32_t n_prefetched)
{
    if (!p) return;
    pthread_mutex_lock(&p->lock);

    stamp(&p->ts_prefetch_done);
    p->cur_prefetched = n_prefetched;
    p->stage = PIPELINE_TRANSFER;

    pthread_mutex_unlock(&p->lock);
}

void pipeline_transfer_done(pipeline_t *p, uint32_t n_transferred)
{
    if (!p) return;
    pthread_mutex_lock(&p->lock);

    stamp(&p->ts_transfer_done);
    p->cur_transferred = n_transferred;

    /* Compute per-step durations */
    double compute_us  = elapsed_us(&p->ts_step_begin,   &p->ts_compute_done);
    double prefetch_us = elapsed_us(&p->ts_compute_done,  &p->ts_prefetch_done);
    double transfer_us = elapsed_us(&p->ts_prefetch_done, &p->ts_transfer_done);
    double step_us     = elapsed_us(&p->ts_step_begin,    &p->ts_transfer_done);

    /* Record */
    p->last_record.step          = p->step_number;
    p->last_record.compute_us    = compute_us;
    p->last_record.prefetch_us   = prefetch_us;
    p->last_record.transfer_us   = transfer_us;
    p->last_record.step_total_us = step_us;
    p->last_record.n_prefetched  = p->cur_prefetched;
    p->last_record.n_transferred = n_transferred;

    /* Accumulate */
    p->sum_compute_us  += compute_us;
    p->sum_prefetch_us += prefetch_us;
    p->sum_transfer_us += transfer_us;
    p->sum_step_us     += step_us;
    p->total_steps++;
    p->total_prefetched  += p->cur_prefetched;
    p->total_transferred += n_transferred;
    p->step_number++;

    p->stage = PIPELINE_IDLE;

    pthread_mutex_unlock(&p->lock);
}

/* ========================================================================
 *  Query
 * ======================================================================== */

PipelineStage pipeline_current_stage(const pipeline_t *p)
{
    return p ? p->stage : PIPELINE_IDLE;
}

uint64_t pipeline_step_number(const pipeline_t *p)
{
    return p ? p->step_number : 0;
}

void pipeline_get_last_record(const pipeline_t *p, pipeline_step_record_t *out)
{
    if (!p || !out) return;
    pthread_mutex_lock((pthread_mutex_t *)&p->lock);
    *out = p->last_record;
    pthread_mutex_unlock((pthread_mutex_t *)&p->lock);
}

double pipeline_compute_overlap(double avg_compute_us,
                                double avg_prefetch_us)
{
    if (avg_prefetch_us <= 0.0) return 0.0;
    double hidden = avg_compute_us < avg_prefetch_us
                        ? avg_compute_us
                        : avg_prefetch_us;
    if (hidden < 0.0) hidden = 0.0;
    return hidden / avg_prefetch_us;
}

void pipeline_get_stats(const pipeline_t *p, pipeline_stats_t *out)
{
    if (!p || !out) return;
    memset(out, 0, sizeof(*out));

    pthread_mutex_lock((pthread_mutex_t *)&p->lock);

    out->total_steps       = p->total_steps;
    out->sum_compute_us    = p->sum_compute_us;
    out->sum_prefetch_us   = p->sum_prefetch_us;
    out->sum_transfer_us   = p->sum_transfer_us;
    out->sum_step_us       = p->sum_step_us;
    out->total_prefetched  = p->total_prefetched;
    out->total_transferred = p->total_transferred;

    if (p->total_steps > 0) {
        out->avg_compute_us  = p->sum_compute_us  / (double)p->total_steps;
        out->avg_prefetch_us = p->sum_prefetch_us  / (double)p->total_steps;
        out->avg_transfer_us = p->sum_transfer_us  / (double)p->total_steps;
        out->avg_step_us     = p->sum_step_us      / (double)p->total_steps;
        out->overlap_ratio   = pipeline_compute_overlap(
                                   out->avg_compute_us,
                                   out->avg_prefetch_us);
    }

    pthread_mutex_unlock((pthread_mutex_t *)&p->lock);
}
