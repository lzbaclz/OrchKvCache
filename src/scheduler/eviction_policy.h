#ifndef ORCHKV_EVICTION_POLICY_H
#define ORCHKV_EVICTION_POLICY_H

#include "../core/kv_types.h"
#include "../core/kv_block.h"
#include "hotcold_classifier.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Eviction Policy — Phase C (C4)
 *
 *  Maintains an LRU doubly-linked list of all active KV blocks and
 *  selects eviction candidates using a hybrid Heat + LRU score:
 *
 *    eviction_score(b) = (1 - hotness(b)) × w_heat
 *                      + lru_position(b)  × w_lru
 *
 *  Higher score → more likely to be evicted.
 *  Blocks with KV_FLAG_PIN are unconditionally skipped.
 *
 *  The LRU list uses kv_block_t's intrusive prev/next pointers.
 *  Head = most recently used, Tail = least recently used.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

typedef struct eviction_candidate {
    kv_block_t *block;
    uint64_t    request_id;
    uint16_t    layer;
    uint16_t    head;
    uint32_t    block_idx;      /* token_start / tokens_per_block */
    float       score;          /* eviction score (higher = evict first) */
} eviction_candidate_t;

typedef struct eviction_policy {
    /* Scoring weights */
    float       w_heat;         /* weight for (1 - hotness)  (default 0.7) */
    float       w_lru;          /* weight for lru position   (default 0.3) */
    uint32_t    batch_size;     /* default eviction batch (default 8) */
    uint32_t    tokens_per_block; /* for computing block_idx */

    /* LRU doubly-linked list (head = MRU, tail = LRU) */
    kv_block_t *lru_head;
    kv_block_t *lru_tail;
    uint32_t    lru_count;      /* total blocks in list */

    /* Reference to classifier for hotness lookup */
    hotcold_classifier_t *classifier;

    pthread_mutex_t lock;
} eviction_policy_t;

/* ---- Lifecycle -------------------------------------------------------- */

int evpol_init(eviction_policy_t *p,
               hotcold_classifier_t *classifier,
               uint32_t batch_size,
               uint32_t tokens_per_block,
               float w_heat,
               float w_lru);

void evpol_destroy(eviction_policy_t *p);

/* ---- LRU management --------------------------------------------------- */

/*
 * Touch: move block to the head of the LRU list (most recently used).
 * Call when a block is accessed (get_kv, promote, prefill, append).
 * If the block is not in the list, it is inserted at the head.
 */
void evpol_lru_touch(eviction_policy_t *p, kv_block_t *blk);

/*
 * Remove a block from the LRU list.
 * Call when a block is destroyed or evicted to a non-tracked tier.
 * Safe to call on a block that is not in the list (no-op).
 */
void evpol_lru_remove(eviction_policy_t *p, kv_block_t *blk);

/* Return the number of blocks currently in the LRU list. */
uint32_t evpol_lru_size(const eviction_policy_t *p);

/* ---- Candidate selection ---------------------------------------------- */

/*
 * Select up to `n` GPU-resident blocks for eviction.
 * Scans from the LRU tail (least recently used), skipping pinned and
 * non-GPU blocks, scores candidates, and returns the top `n` sorted
 * by eviction_score descending.
 *
 * Returns the actual number of candidates found (may be < n).
 * `out` must have space for at least `n` entries.
 */
uint32_t evpol_select_gpu_victims(eviction_policy_t *p,
                                  uint32_t n,
                                  eviction_candidate_t *out);

/*
 * Select up to `n` DRAM-resident blocks for eviction.
 * Same logic as GPU variant but filters for TIER_HOST_DRAM.
 */
uint32_t evpol_select_dram_victims(eviction_policy_t *p,
                                   uint32_t n,
                                   eviction_candidate_t *out);

/* ---- Runtime tuning --------------------------------------------------- */

void evpol_set_weights(eviction_policy_t *p, float w_heat, float w_lru);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_EVICTION_POLICY_H */
