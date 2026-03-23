#include "tiered_manager.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================
 *  Default configuration
 * ======================================================================== */

void tm_config_default(tm_config_t *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    cfg->tracker_capacity      = 4096;
    cfg->ema_lambda            = 0.9f;

    hcc_params_default(&cfg->hcc_params);
    athresh_params_default(&cfg->athresh_params);

    cfg->w_heat               = 0.7f;
    cfg->w_lru                = 0.3f;
    cfg->tokens_per_block     = 64;

    cfg->prefetch_budget      = 16;
    cfg->threshold_to_gpu     = 0.5f;
    cfg->threshold_to_dram    = 0.2f;

    cfg->transfer_fn          = NULL;
    cfg->transfer_ctx         = NULL;
    cfg->alloc_fn             = NULL;
    cfg->free_fn              = NULL;
    cfg->alloc_ctx            = NULL;
    cfg->block_data_size      = 32 * 1024;  /* 32 KB */

    cfg->schedule_interval_us = 1000;
    cfg->demote_batch_size    = 8;
    cfg->prefetch_batch_size  = 8;
    cfg->max_blocks           = 4096;
    cfg->auto_schedule        = false;
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int tm_init(tiered_manager_t *m, const tm_config_t *cfg)
{
    if (!m || !cfg) return ORCHKV_ERR_INVALID;
    memset(m, 0, sizeof(*m));

    int rc;

    /* C1: attention tracker */
    rc = attn_tracker_init(&m->tracker, cfg->tracker_capacity, cfg->ema_lambda);
    if (rc != ORCHKV_OK) return rc;

    /* C2: classifier */
    rc = hcc_init(&m->classifier, &m->tracker, &cfg->hcc_params);
    if (rc != ORCHKV_OK) goto fail_tracker;

    /* C3: adaptive threshold */
    rc = athresh_init(&m->threshold, &m->classifier, &cfg->athresh_params);
    if (rc != ORCHKV_OK) goto fail_classifier;

    /* C4: eviction policy */
    rc = evpol_init(&m->evpol, &m->classifier,
                    cfg->demote_batch_size, cfg->tokens_per_block,
                    cfg->w_heat, cfg->w_lru);
    if (rc != ORCHKV_OK) goto fail_threshold;

    /* C5: prefetch scheduler */
    rc = prefetch_init(&m->prefetch, &m->tracker,
                       cfg->prefetch_budget,
                       cfg->threshold_to_gpu, cfg->threshold_to_dram, 0);
    if (rc != ORCHKV_OK) goto fail_evpol;

    /* C6: pipeline */
    rc = pipeline_init(&m->pipeline, &m->prefetch);
    if (rc != ORCHKV_OK) goto fail_prefetch;

    /* C7: migration engine */
    rc = mig_init(&m->migration, &m->evpol, &m->prefetch,
                  cfg->transfer_fn, cfg->transfer_ctx);
    if (rc != ORCHKV_OK) goto fail_pipeline;

    /* Active block registry */
    m->max_blocks = cfg->max_blocks > 0 ? cfg->max_blocks : 4096;
    m->blocks = (kv_block_t **)calloc(m->max_blocks, sizeof(kv_block_t *));
    if (!m->blocks) { rc = ORCHKV_ERR_OOM; goto fail_migration; }

    m->schedule_interval_us = cfg->schedule_interval_us;
    m->demote_batch_size    = cfg->demote_batch_size;
    m->prefetch_batch_size  = cfg->prefetch_batch_size;

    m->alloc_fn        = cfg->alloc_fn;
    m->free_fn         = cfg->free_fn;
    m->alloc_ctx       = cfg->alloc_ctx;
    m->block_data_size = cfg->block_data_size;

    m->auto_schedule = cfg->auto_schedule;

    pthread_rwlock_init(&m->block_lock, NULL);
    pthread_mutex_init(&m->lock, NULL);

    if (m->auto_schedule)
        tm_start(m);

    return ORCHKV_OK;

fail_migration:  mig_destroy(&m->migration);
fail_pipeline:   pipeline_destroy(&m->pipeline);
fail_prefetch:   prefetch_destroy(&m->prefetch);
fail_evpol:      evpol_destroy(&m->evpol);
fail_threshold:  athresh_destroy(&m->threshold);
fail_classifier: hcc_destroy(&m->classifier);
fail_tracker:    attn_tracker_destroy(&m->tracker);
    return rc;
}

void tm_destroy(tiered_manager_t *m)
{
    if (!m) return;

    tm_stop(m);

    /* Destroy in reverse order */
    free(m->blocks);
    m->blocks = NULL;

    mig_destroy(&m->migration);
    pipeline_destroy(&m->pipeline);
    prefetch_destroy(&m->prefetch);
    evpol_destroy(&m->evpol);
    athresh_destroy(&m->threshold);
    hcc_destroy(&m->classifier);
    attn_tracker_destroy(&m->tracker);

    pthread_rwlock_destroy(&m->block_lock);
    pthread_mutex_destroy(&m->lock);
}

/* ========================================================================
 *  Block registration
 * ======================================================================== */

int tm_register_block(tiered_manager_t *m, kv_block_t *blk)
{
    if (!m || !blk) return ORCHKV_ERR_INVALID;

    pthread_rwlock_wrlock(&m->block_lock);

    if (m->n_blocks >= m->max_blocks) {
        pthread_rwlock_unlock(&m->block_lock);
        return ORCHKV_ERR_TIER_FULL;
    }

    m->blocks[m->n_blocks++] = blk;

    pthread_rwlock_unlock(&m->block_lock);

    attn_tracker_register(&m->tracker, blk->block_id);
    hcc_register(&m->classifier, blk->block_id, blk->flags);
    evpol_lru_touch(&m->evpol, blk);

    return ORCHKV_OK;
}

void tm_unregister_block(tiered_manager_t *m, kv_block_t *blk)
{
    if (!m || !blk) return;

    pthread_rwlock_wrlock(&m->block_lock);
    for (uint32_t i = 0; i < m->n_blocks; i++) {
        if (m->blocks[i] == blk) {
            m->blocks[i] = m->blocks[--m->n_blocks];
            break;
        }
    }
    pthread_rwlock_unlock(&m->block_lock);

    evpol_lru_remove(&m->evpol, blk);
    hcc_unregister(&m->classifier, blk->block_id);
    attn_tracker_reset(&m->tracker, blk->block_id);
}

/* ========================================================================
 *  Notifications
 * ======================================================================== */

void tm_notify_attn(tiered_manager_t *m,
                    uint64_t block_id,
                    float attn_weight)
{
    if (!m) return;
    attn_tracker_update(&m->tracker, block_id, attn_weight);
}

void tm_notify_access(tiered_manager_t *m, kv_block_t *blk)
{
    if (!m || !blk) return;
    evpol_lru_touch(&m->evpol, blk);
}

void tm_step_done(tiered_manager_t *m)
{
    if (!m) return;
    attn_tracker_step_done(&m->tracker);
    prefetch_step_reset(&m->prefetch);
}

/* ========================================================================
 *  Scheduling: core loop
 * ======================================================================== */

static void do_gpu_demote(tiered_manager_t *m)
{
    if (!athresh_should_demote_gpu(&m->threshold, m->gpu_used_ratio))
        return;

    uint32_t n = m->demote_batch_size;
    eviction_candidate_t *cands = (eviction_candidate_t *)
        alloca(n * sizeof(eviction_candidate_t));

    uint32_t found = evpol_select_gpu_victims(&m->evpol, n, cands);

    for (uint32_t i = 0; i < found; i++) {
        kv_block_t *blk = cands[i].block;

        void *dst = NULL;
        if (m->alloc_fn)
            dst = m->alloc_fn(m->block_data_size, TIER_HOST_DRAM, m->alloc_ctx);

        if (!dst && m->alloc_fn) break;

        void *old_ptr = blk->data_ptr;

        int rc = mig_execute_one(&m->migration, blk, TIER_HOST_DRAM,
                                 dst, NULL, m->block_data_size);
        if (rc == ORCHKV_OK) {
            if (m->free_fn && old_ptr)
                m->free_fn(old_ptr, TIER_GPU_HBM, m->alloc_ctx);
            m->gpu_demotes++;
        } else if (m->free_fn && dst) {
            m->free_fn(dst, TIER_HOST_DRAM, m->alloc_ctx);
        }
    }
}

static void do_dram_demote(tiered_manager_t *m)
{
    if (!athresh_should_demote_dram(&m->threshold, m->dram_used_ratio))
        return;

    uint32_t n = m->demote_batch_size;
    eviction_candidate_t *cands = (eviction_candidate_t *)
        alloca(n * sizeof(eviction_candidate_t));

    uint32_t found = evpol_select_dram_victims(&m->evpol, n, cands);

    for (uint32_t i = 0; i < found; i++) {
        kv_block_t *blk = cands[i].block;

        void *dst = NULL;
        if (m->alloc_fn)
            dst = m->alloc_fn(m->block_data_size, TIER_NVM, m->alloc_ctx);

        if (!dst && m->alloc_fn) break;

        void *old_ptr = blk->data_ptr;

        int rc = mig_execute_one(&m->migration, blk, TIER_NVM,
                                 dst, NULL, m->block_data_size);
        if (rc == ORCHKV_OK) {
            if (m->free_fn && old_ptr)
                m->free_fn(old_ptr, TIER_HOST_DRAM, m->alloc_ctx);
            m->dram_demotes++;
        } else if (m->free_fn && dst) {
            m->free_fn(dst, TIER_NVM, m->alloc_ctx);
        }
    }
}

static void do_prefetch(tiered_manager_t *m)
{
    pthread_rwlock_rdlock(&m->block_lock);

    uint32_t scan_n = 0;
    prefetch_block_info_t *info = (prefetch_block_info_t *)
        alloca(m->n_blocks * sizeof(prefetch_block_info_t));

    for (uint32_t i = 0; i < m->n_blocks; i++) {
        kv_block_t *blk = m->blocks[i];
        if (blk->tier != TIER_GPU_HBM && blk->tier != TIER_NONE) {
            info[scan_n].block_id = blk->block_id;
            info[scan_n].tier     = blk->tier;
            scan_n++;
        }
    }

    pthread_rwlock_unlock(&m->block_lock);

    if (scan_n > 0)
        prefetch_scan_blocks(&m->prefetch, info, scan_n);

    prefetch_entry_t *out = (prefetch_entry_t *)
        alloca(m->prefetch_batch_size * sizeof(prefetch_entry_t));

    uint32_t dispatched = prefetch_dispatch(&m->prefetch,
                                            m->prefetch_batch_size, out);
    m->prefetches_dispatched += dispatched;
}

void tm_schedule_once(tiered_manager_t *m)
{
    if (!m) return;

    /* 1. Update classification */
    hcc_update_all(&m->classifier);

    /* 2. Update thresholds */
    athresh_update(&m->threshold, m->gpu_used_ratio, m->dram_used_ratio);

    /* 3. GPU demote */
    do_gpu_demote(m);

    /* 4. DRAM demote */
    do_dram_demote(m);

    /* 5. Prefetch */
    do_prefetch(m);

    pthread_mutex_lock(&m->lock);
    m->schedule_cycles++;
    pthread_mutex_unlock(&m->lock);
}

/* ========================================================================
 *  Background scheduler thread
 * ======================================================================== */

static void *sched_loop(void *arg)
{
    tiered_manager_t *m = (tiered_manager_t *)arg;
    while (m->running) {
        usleep(m->schedule_interval_us);
        tm_schedule_once(m);
    }
    return NULL;
}

int tm_start(tiered_manager_t *m)
{
    if (!m || m->running) return ORCHKV_ERR_STATE;

    m->running = true;
    int rc = pthread_create(&m->sched_thread, NULL, sched_loop, m);
    if (rc != 0) {
        m->running = false;
        return ORCHKV_ERR_INIT;
    }
    return ORCHKV_OK;
}

void tm_stop(tiered_manager_t *m)
{
    if (!m || !m->running) return;

    m->running = false;
    pthread_join(m->sched_thread, NULL);
}

/* ========================================================================
 *  External state updates
 * ======================================================================== */

void tm_set_usage(tiered_manager_t *m,
                  float gpu_used_ratio,
                  float dram_used_ratio)
{
    if (!m) return;
    pthread_mutex_lock(&m->lock);
    m->gpu_used_ratio  = gpu_used_ratio;
    m->dram_used_ratio = dram_used_ratio;
    pthread_mutex_unlock(&m->lock);
}

/* ========================================================================
 *  Runtime policy tuning
 * ======================================================================== */

void tm_set_policy(tiered_manager_t *m,
                   float alpha, float beta, float gamma)
{
    if (!m) return;
    hcc_set_weights(&m->classifier, alpha, beta, gamma);
}

/* ========================================================================
 *  Statistics
 * ======================================================================== */

void tm_get_stats(const tiered_manager_t *m, tm_stats_t *out)
{
    if (!m || !out) return;
    memset(out, 0, sizeof(*out));

    pthread_mutex_lock((pthread_mutex_t *)&m->lock);

    out->schedule_cycles       = m->schedule_cycles;
    out->gpu_demotes           = m->gpu_demotes;
    out->dram_demotes          = m->dram_demotes;
    out->prefetches_dispatched = m->prefetches_dispatched;
    out->gpu_used_ratio        = m->gpu_used_ratio;
    out->dram_used_ratio       = m->dram_used_ratio;

    pthread_mutex_unlock((pthread_mutex_t *)&m->lock);

    hcc_get_stats(&m->classifier, &out->hcc_stats);
    prefetch_get_stats(&m->prefetch, &out->prefetch_stats);
    mig_get_stats(&m->migration, &out->migration_stats);
}
