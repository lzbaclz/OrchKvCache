#include "prefetch_scheduler.h"
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 *  Max-heap helpers (priority queue by entry.priority descending)
 * ======================================================================== */

static inline void swap_entries(prefetch_entry_t *a, prefetch_entry_t *b)
{
    prefetch_entry_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static void heap_sift_up(prefetch_entry_t *heap, uint32_t idx)
{
    while (idx > 0) {
        uint32_t parent = (idx - 1) / 2;
        if (heap[idx].priority <= heap[parent].priority) break;
        swap_entries(&heap[idx], &heap[parent]);
        idx = parent;
    }
}

static void heap_sift_down(prefetch_entry_t *heap, uint32_t size, uint32_t idx)
{
    while (true) {
        uint32_t largest = idx;
        uint32_t left  = 2 * idx + 1;
        uint32_t right = 2 * idx + 2;

        if (left < size && heap[left].priority > heap[largest].priority)
            largest = left;
        if (right < size && heap[right].priority > heap[largest].priority)
            largest = right;
        if (largest == idx) break;

        swap_entries(&heap[idx], &heap[largest]);
        idx = largest;
    }
}

static bool heap_push(prefetch_entry_t *heap, uint32_t *size,
                      uint32_t cap, const prefetch_entry_t *e)
{
    if (*size >= cap) return false;
    heap[*size] = *e;
    heap_sift_up(heap, *size);
    (*size)++;
    return true;
}

static bool heap_pop(prefetch_entry_t *heap, uint32_t *size,
                     prefetch_entry_t *out)
{
    if (*size == 0) return false;
    *out = heap[0];
    (*size)--;
    if (*size > 0) {
        heap[0] = heap[*size];
        heap_sift_down(heap, *size, 0);
    }
    return true;
}

/* ========================================================================
 *  Tracking set helpers (linear scan, fine for budget-sized arrays)
 * ======================================================================== */

static void track_add(prefetch_scheduler_t *s, uint64_t block_id)
{
    if (s->tracked_count < PREFETCH_TRACK_CAP)
        s->tracked_ids[s->tracked_count++] = block_id;
}

/* Remove block_id from tracking set; returns true if found. */
static bool track_remove(prefetch_scheduler_t *s, uint64_t block_id)
{
    for (uint32_t i = 0; i < s->tracked_count; i++) {
        if (s->tracked_ids[i] == block_id) {
            s->tracked_ids[i] = s->tracked_ids[--s->tracked_count];
            return true;
        }
    }
    return false;
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int prefetch_init(prefetch_scheduler_t *s,
                  attention_tracker_t *tracker,
                  uint32_t budget,
                  float threshold_to_gpu,
                  float threshold_to_dram,
                  uint32_t heap_cap)
{
    if (!s) return ORCHKV_ERR_INVALID;

    memset(s, 0, sizeof(*s));

    if (heap_cap == 0) heap_cap = 1024;
    s->heap = (prefetch_entry_t *)calloc(heap_cap, sizeof(prefetch_entry_t));
    if (!s->heap) return ORCHKV_ERR_OOM;

    s->heap_cap          = heap_cap;
    s->prefetch_budget   = budget > 0 ? budget : 16;
    s->threshold_to_gpu  = threshold_to_gpu;
    s->threshold_to_dram = threshold_to_dram;
    s->tracker           = tracker;

    pthread_mutex_init(&s->lock, NULL);
    return ORCHKV_OK;
}

void prefetch_destroy(prefetch_scheduler_t *s)
{
    if (!s) return;
    free(s->heap);
    s->heap = NULL;
    s->heap_size = 0;
    pthread_mutex_destroy(&s->lock);
}

/* ========================================================================
 *  Scan & candidate management
 * ======================================================================== */

void prefetch_scan_blocks(prefetch_scheduler_t *s,
                          const prefetch_block_info_t *blocks,
                          uint32_t n_blocks)
{
    if (!s || !blocks || n_blocks == 0) return;

    pthread_mutex_lock(&s->lock);

    for (uint32_t i = 0; i < n_blocks; i++) {
        uint64_t    bid  = blocks[i].block_id;
        StorageTier tier = blocks[i].tier;
        s->total_scanned++;

        if (tier == TIER_GPU_HBM) continue;   /* already on GPU */
        if (tier == TIER_NONE)    continue;

        attn_stats_t st;
        if (!s->tracker || attn_tracker_get(s->tracker, bid, &st) != ORCHKV_OK)
            continue;
        if (!st.active) continue;

        float ema = st.ema;
        prefetch_entry_t cand;
        cand.block_id     = bid;
        cand.current_tier = tier;

        if (ema >= s->threshold_to_gpu) {
            cand.target_tier = TIER_GPU_HBM;
            cand.priority    = ema;
        } else if (ema >= s->threshold_to_dram && tier != TIER_HOST_DRAM) {
            cand.target_tier = TIER_HOST_DRAM;
            cand.priority    = ema * 0.5f;
        } else {
            continue;
        }

        if (heap_push(s->heap, &s->heap_size, s->heap_cap, &cand))
            s->total_enqueued++;
    }

    pthread_mutex_unlock(&s->lock);
}

int prefetch_add_candidate(prefetch_scheduler_t *s,
                           uint64_t block_id,
                           StorageTier current_tier,
                           StorageTier target_tier,
                           float priority)
{
    if (!s) return ORCHKV_ERR_INVALID;

    pthread_mutex_lock(&s->lock);

    prefetch_entry_t cand = {
        .block_id     = block_id,
        .current_tier = current_tier,
        .target_tier  = target_tier,
        .priority     = priority,
    };

    bool ok = heap_push(s->heap, &s->heap_size, s->heap_cap, &cand);
    if (ok) s->total_enqueued++;

    pthread_mutex_unlock(&s->lock);
    return ok ? ORCHKV_OK : ORCHKV_ERR_TIER_FULL;
}

/* ========================================================================
 *  Dispatch
 * ======================================================================== */

uint32_t prefetch_dispatch(prefetch_scheduler_t *s,
                           uint32_t max_n,
                           prefetch_entry_t *out)
{
    if (!s || !out || max_n == 0) return 0;

    pthread_mutex_lock(&s->lock);

    uint32_t limit = max_n < s->prefetch_budget ? max_n : s->prefetch_budget;
    uint32_t count = 0;

    while (count < limit) {
        prefetch_entry_t e;
        if (!heap_pop(s->heap, &s->heap_size, &e)) break;

        out[count++] = e;
        track_add(s, e.block_id);
        s->total_dispatched++;
    }

    pthread_mutex_unlock(&s->lock);
    return count;
}

/* ========================================================================
 *  Hit tracking
 * ======================================================================== */

bool prefetch_notify_hit(prefetch_scheduler_t *s, uint64_t block_id)
{
    if (!s) return false;

    pthread_mutex_lock(&s->lock);
    bool found = track_remove(s, block_id);
    if (found)
        s->prefetch_hits++;
    pthread_mutex_unlock(&s->lock);
    return found;
}

void prefetch_step_reset(prefetch_scheduler_t *s)
{
    if (!s) return;

    pthread_mutex_lock(&s->lock);

    /* Remaining tracked entries are wasted prefetches */
    s->prefetch_wasted += s->tracked_count;
    s->tracked_count = 0;

    /* Clear the heap for the next step */
    s->heap_size = 0;

    pthread_mutex_unlock(&s->lock);
}

/* ========================================================================
 *  Statistics
 * ======================================================================== */

float prefetch_hit_rate(const prefetch_scheduler_t *s)
{
    if (!s || s->total_dispatched == 0) return 0.0f;
    return (float)s->prefetch_hits / (float)s->total_dispatched;
}

void prefetch_get_stats(const prefetch_scheduler_t *s, prefetch_stats_t *out)
{
    if (!s || !out) return;
    out->total_scanned    = s->total_scanned;
    out->total_enqueued   = s->total_enqueued;
    out->total_dispatched = s->total_dispatched;
    out->prefetch_hits    = s->prefetch_hits;
    out->prefetch_wasted  = s->prefetch_wasted;
    out->hit_rate         = prefetch_hit_rate(s);
}
