#include "kv_block.h"
#include <string.h>

/* Global monotonic block-ID counter */
atomic_uint_fast64_t g_block_id_counter = 0;

void kv_block_reset_id_counter(void)
{
    atomic_store(&g_block_id_counter, 0);
}

void kv_block_init(kv_block_t *blk,
                   uint64_t    request_id,
                   uint16_t    layer_id,
                   uint16_t    head_id,
                   uint32_t    token_start,
                   uint16_t    token_count)
{
    memset(blk, 0, sizeof(*blk));

    blk->block_id      = atomic_fetch_add(&g_block_id_counter, 1);
    blk->request_id    = request_id;
    blk->layer_id      = layer_id;
    blk->head_id       = head_id;
    blk->token_start   = token_start;
    blk->token_count   = token_count;

    blk->tier          = TIER_NONE;
    blk->data_ptr      = NULL;
    blk->persistent_offset = 0;

    blk->hotness       = 0.0f;
    blk->last_access_step = 0;
    blk->access_count  = 0;

    blk->state         = KV_STATE_FREE;
    blk->flags         = KV_FLAG_NONE;

    pthread_rwlock_init(&blk->lock, NULL);

    blk->prev = NULL;
    blk->next = NULL;
}

void kv_block_destroy(kv_block_t *blk)
{
    pthread_rwlock_destroy(&blk->lock);
    blk->state    = KV_STATE_EVICTED;
    blk->data_ptr = NULL;
    blk->tier     = TIER_NONE;
}

void kv_block_set_location(kv_block_t *blk, StorageTier tier, void *ptr)
{
    kv_block_wrlock(blk);
    blk->tier     = tier;
    blk->data_ptr = ptr;
    kv_block_unlock(blk);
}

/*
 * Allowed state transitions (row=from, col=to):
 *
 *   FREE → ALLOCATED
 *   ALLOCATED → HOT
 *   HOT → WARM, MIGRATING, EVICTED
 *   WARM → HOT, COLD, MIGRATING, EVICTED
 *   COLD → WARM, MIGRATING, EVICTED
 *   MIGRATING → HOT, WARM, COLD
 *   any → EVICTED  (always allowed)
 */
int kv_block_set_state(kv_block_t *blk, KVBlockState new_state)
{
    if (new_state == KV_STATE_EVICTED) {
        blk->state = new_state;
        return ORCHKV_OK;
    }

    KVBlockState old = blk->state;
    bool ok = false;

    switch (old) {
    case KV_STATE_FREE:
        ok = (new_state == KV_STATE_ALLOCATED);
        break;
    case KV_STATE_ALLOCATED:
        ok = (new_state == KV_STATE_HOT);
        break;
    case KV_STATE_HOT:
        ok = (new_state == KV_STATE_WARM ||
              new_state == KV_STATE_MIGRATING);
        break;
    case KV_STATE_WARM:
        ok = (new_state == KV_STATE_HOT  ||
              new_state == KV_STATE_COLD ||
              new_state == KV_STATE_MIGRATING);
        break;
    case KV_STATE_COLD:
        ok = (new_state == KV_STATE_WARM ||
              new_state == KV_STATE_MIGRATING);
        break;
    case KV_STATE_MIGRATING:
        ok = (new_state == KV_STATE_HOT  ||
              new_state == KV_STATE_WARM ||
              new_state == KV_STATE_COLD);
        break;
    case KV_STATE_EVICTED:
        ok = false;
        break;
    }

    if (!ok) {
        LOG_ERR("illegal state transition: %s → %s (block %lu)",
                block_state_name(old), block_state_name(new_state),
                (unsigned long)blk->block_id);
        return ORCHKV_ERR_STATE;
    }

    blk->state = new_state;
    return ORCHKV_OK;
}
