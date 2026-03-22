#include "eviction_policy.h"
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 *  Internal: LRU list manipulation (caller holds lock)
 * ======================================================================== */

/* Detach a block from the doubly-linked list without freeing it. */
static void list_detach(eviction_policy_t *p, kv_block_t *blk)
{
    if (blk->prev)
        blk->prev->next = blk->next;
    else
        p->lru_head = blk->next;

    if (blk->next)
        blk->next->prev = blk->prev;
    else
        p->lru_tail = blk->prev;

    blk->prev = NULL;
    blk->next = NULL;
    p->lru_count--;
}

/* Insert at the head of the list (most recently used). */
static void list_push_front(eviction_policy_t *p, kv_block_t *blk)
{
    blk->prev = NULL;
    blk->next = p->lru_head;
    if (p->lru_head)
        p->lru_head->prev = blk;
    else
        p->lru_tail = blk;
    p->lru_head = blk;
    p->lru_count++;
}

/* Check whether a block is currently in the LRU list.
 * A block is in the list if it's the head, or if it has a prev pointer. */
static inline bool in_list(const eviction_policy_t *p, const kv_block_t *blk)
{
    return (blk == p->lru_head) || (blk->prev != NULL);
}

/* ========================================================================
 *  Internal: candidate selection helpers
 * ======================================================================== */

/*
 * Compute eviction score.
 *   lru_pos: 0 = tail (oldest), lru_count-1 = head (newest)
 */
static float eviction_score(const eviction_policy_t *p,
                            float hotness,
                            uint32_t lru_pos)
{
    float heat_part = (1.0f - hotness) * p->w_heat;
    float lru_part  = (p->lru_count > 1)
                          ? ((float)(p->lru_count - 1 - lru_pos)
                             / (float)(p->lru_count - 1)) * p->w_lru
                          : 0.0f;
    return heat_part + lru_part;
}

/* Insert candidate into a sorted array (descending by score).
 * Returns new count (≤ max_n). */
static uint32_t sorted_insert(eviction_candidate_t *arr,
                               uint32_t count,
                               uint32_t max_n,
                               const eviction_candidate_t *c)
{
    if (count < max_n) {
        /* There's room — insert at correct position */
        uint32_t i = count;
        while (i > 0 && arr[i - 1].score < c->score) {
            arr[i] = arr[i - 1];
            i--;
        }
        arr[i] = *c;
        return count + 1;
    }
    /* Full — replace the last entry if new score is higher */
    if (c->score > arr[count - 1].score) {
        uint32_t i = count - 1;
        while (i > 0 && arr[i - 1].score < c->score) {
            arr[i] = arr[i - 1];
            i--;
        }
        arr[i] = *c;
    }
    return count;
}

/*
 * Generic victim selection: scan from tail, filter by tier,
 * skip pinned blocks, score and collect top N.
 */
static uint32_t select_victims(eviction_policy_t *p,
                                StorageTier target_tier,
                                uint32_t n,
                                eviction_candidate_t *out)
{
    if (!p || !out || n == 0) return 0;

    pthread_mutex_lock(&p->lock);

    uint32_t found    = 0;
    uint32_t scanned  = 0;
    uint32_t max_scan = n * 4;  /* scan up to 4N blocks from tail */
    uint32_t pos      = 0;      /* 0 = tail position */

    kv_block_t *cur = p->lru_tail;
    while (cur && scanned < max_scan) {
        scanned++;

        if (cur->tier != target_tier)  { cur = cur->prev; pos++; continue; }
        if (cur->flags & KV_FLAG_PIN) { cur = cur->prev; pos++; continue; }
        if (cur->state == KV_STATE_MIGRATING) { cur = cur->prev; pos++; continue; }

        float hotness = hcc_get_score(p->classifier, cur->block_id);
        float score   = eviction_score(p, hotness, pos);

        eviction_candidate_t cand = {
            .block      = cur,
            .request_id = cur->request_id,
            .layer      = cur->layer_id,
            .head       = cur->head_id,
            .block_idx  = (p->tokens_per_block > 0)
                              ? cur->token_start / p->tokens_per_block
                              : 0,
            .score      = score,
        };

        found = sorted_insert(out, found, n, &cand);
        cur = cur->prev;
        pos++;
    }

    pthread_mutex_unlock(&p->lock);
    return found;
}

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int evpol_init(eviction_policy_t *p,
               hotcold_classifier_t *classifier,
               uint32_t batch_size,
               uint32_t tokens_per_block,
               float w_heat,
               float w_lru)
{
    if (!p || !classifier) return ORCHKV_ERR_INVALID;

    memset(p, 0, sizeof(*p));
    p->w_heat           = w_heat;
    p->w_lru            = w_lru;
    p->batch_size       = batch_size > 0 ? batch_size : 8;
    p->tokens_per_block = tokens_per_block > 0 ? tokens_per_block : 64;
    p->classifier       = classifier;

    pthread_mutex_init(&p->lock, NULL);
    return ORCHKV_OK;
}

void evpol_destroy(eviction_policy_t *p)
{
    if (!p) return;
    pthread_mutex_destroy(&p->lock);
    p->lru_head = p->lru_tail = NULL;
    p->lru_count = 0;
}

/* ========================================================================
 *  LRU management
 * ======================================================================== */

void evpol_lru_touch(eviction_policy_t *p, kv_block_t *blk)
{
    if (!p || !blk) return;

    pthread_mutex_lock(&p->lock);
    if (in_list(p, blk))
        list_detach(p, blk);
    list_push_front(p, blk);
    pthread_mutex_unlock(&p->lock);
}

void evpol_lru_remove(eviction_policy_t *p, kv_block_t *blk)
{
    if (!p || !blk) return;

    pthread_mutex_lock(&p->lock);
    if (in_list(p, blk))
        list_detach(p, blk);
    pthread_mutex_unlock(&p->lock);
}

uint32_t evpol_lru_size(const eviction_policy_t *p)
{
    return p ? p->lru_count : 0;
}

/* ========================================================================
 *  Candidate selection
 * ======================================================================== */

uint32_t evpol_select_gpu_victims(eviction_policy_t *p,
                                  uint32_t n,
                                  eviction_candidate_t *out)
{
    return select_victims(p, TIER_GPU_HBM, n, out);
}

uint32_t evpol_select_dram_victims(eviction_policy_t *p,
                                   uint32_t n,
                                   eviction_candidate_t *out)
{
    return select_victims(p, TIER_HOST_DRAM, n, out);
}

/* ========================================================================
 *  Runtime tuning
 * ======================================================================== */

void evpol_set_weights(eviction_policy_t *p, float w_heat, float w_lru)
{
    if (!p) return;
    pthread_mutex_lock(&p->lock);
    p->w_heat = w_heat;
    p->w_lru  = w_lru;
    pthread_mutex_unlock(&p->lock);
}
