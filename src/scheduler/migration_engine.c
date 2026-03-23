#include "migration_engine.h"
#include <string.h>

/* ========================================================================
 *  Lifecycle
 * ======================================================================== */

int mig_init(migration_engine_t *e,
             eviction_policy_t *evpol,
             prefetch_scheduler_t *prefetch,
             mig_transfer_fn transfer_fn,
             void *transfer_ctx)
{
    if (!e) return ORCHKV_ERR_INVALID;

    memset(e, 0, sizeof(*e));
    e->evpol        = evpol;
    e->prefetch     = prefetch;
    e->transfer_fn  = transfer_fn;
    e->transfer_ctx = transfer_ctx;

    pthread_mutex_init(&e->lock, NULL);
    return ORCHKV_OK;
}

void mig_destroy(migration_engine_t *e)
{
    if (!e) return;
    pthread_mutex_destroy(&e->lock);
}

/* ========================================================================
 *  Operation routing
 * ======================================================================== */

MigrateOp mig_determine_op(StorageTier from, StorageTier to)
{
    if (from == to) return MIGRATE_OP_COUNT;  /* invalid: same tier */

    /* Demote paths */
    if (from == TIER_GPU_HBM && to == TIER_HOST_DRAM) return MIGRATE_DEMOTE_GPU2DRAM;
    if (from == TIER_HOST_DRAM && (to == TIER_NVM || to == TIER_SSD))
        return MIGRATE_DEMOTE_DRAM2STOR;
    if (from == TIER_GPU_HBM && (to == TIER_NVM || to == TIER_SSD))
        return MIGRATE_DEMOTE_GPU2STOR;

    /* Promote paths */
    if ((from == TIER_NVM || from == TIER_SSD) && to == TIER_HOST_DRAM)
        return MIGRATE_PROMOTE_STOR2DRAM;
    if (from == TIER_HOST_DRAM && to == TIER_GPU_HBM) return MIGRATE_PROMOTE_DRAM2GPU;
    if ((from == TIER_NVM || from == TIER_SSD) && to == TIER_GPU_HBM)
        return MIGRATE_PROMOTE_STOR2GPU;

    return MIGRATE_OP_COUNT;
}

KVBlockState mig_target_state(StorageTier tier)
{
    switch (tier) {
    case TIER_GPU_HBM:   return KV_STATE_HOT;
    case TIER_HOST_DRAM: return KV_STATE_WARM;
    case TIER_NVM:       return KV_STATE_COLD;
    case TIER_SSD:       return KV_STATE_COLD;
    default:             return KV_STATE_EVICTED;
    }
}

const char *mig_op_name(MigrateOp op)
{
    switch (op) {
    case MIGRATE_DEMOTE_GPU2DRAM:   return "GPU→DRAM";
    case MIGRATE_DEMOTE_DRAM2STOR:  return "DRAM→Storage";
    case MIGRATE_DEMOTE_GPU2STOR:   return "GPU→Storage";
    case MIGRATE_PROMOTE_STOR2DRAM: return "Storage→DRAM";
    case MIGRATE_PROMOTE_DRAM2GPU:  return "DRAM→GPU";
    case MIGRATE_PROMOTE_STOR2GPU:  return "Storage→GPU";
    default:                        return "UNKNOWN";
    }
}

/* ========================================================================
 *  Internal: two-hop detection
 * ======================================================================== */

static inline bool is_two_hop(MigrateOp op)
{
    return op == MIGRATE_DEMOTE_GPU2STOR || op == MIGRATE_PROMOTE_STOR2GPU;
}

/*
 * Split a two-hop op into its two single-hop sub-operations.
 *
 * GPU→Storage:  hop1 = GPU→DRAM,      hop2 = DRAM→Storage
 * Storage→GPU:  hop1 = Storage→DRAM,  hop2 = DRAM→GPU
 */
static void two_hop_split(MigrateOp op, MigrateOp *hop1, MigrateOp *hop2)
{
    if (op == MIGRATE_DEMOTE_GPU2STOR) {
        *hop1 = MIGRATE_DEMOTE_GPU2DRAM;
        *hop2 = MIGRATE_DEMOTE_DRAM2STOR;
    } else {
        *hop1 = MIGRATE_PROMOTE_STOR2DRAM;
        *hop2 = MIGRATE_PROMOTE_DRAM2GPU;
    }
}

/* ========================================================================
 *  Core: single-block migration
 * ======================================================================== */

int mig_execute_one(migration_engine_t *e,
                    kv_block_t *blk,
                    StorageTier target_tier,
                    void *dst_buf,
                    void *intermediate_buf,
                    size_t data_size)
{
    if (!e || !blk) return ORCHKV_ERR_INVALID;
    if (!e->transfer_fn) return ORCHKV_ERR_INIT;

    /* --- Validate block --- */
    if (blk->flags & KV_FLAG_PIN) return ORCHKV_ERR_LOCKED;
    if (blk->state == KV_STATE_MIGRATING) return ORCHKV_ERR_STATE;

    MigrateOp op = mig_determine_op(blk->tier, target_tier);
    if (op >= MIGRATE_OP_COUNT) return ORCHKV_ERR_INVALID;

    if (is_two_hop(op) && !intermediate_buf) return ORCHKV_ERR_INVALID;

    /* --- Mark as migrating --- */
    KVBlockState saved_state = blk->state;
    blk->state = KV_STATE_MIGRATING;

    int rc;

    if (is_two_hop(op)) {
        /* ---- Two-hop migration ---- */
        MigrateOp hop1, hop2;
        two_hop_split(op, &hop1, &hop2);

        /* Hop 1 */
        rc = e->transfer_fn(blk, intermediate_buf, data_size,
                            hop1, e->transfer_ctx);
        if (rc != ORCHKV_OK) goto fail;

        /* Update block for the intermediate state */
        void *saved_ptr = blk->data_ptr;
        StorageTier saved_tier = blk->tier;
        blk->data_ptr = intermediate_buf;
        blk->tier     = TIER_HOST_DRAM;

        /* Hop 2 */
        rc = e->transfer_fn(blk, dst_buf, data_size,
                            hop2, e->transfer_ctx);
        if (rc != ORCHKV_OK) {
            /* Rollback intermediate state */
            blk->data_ptr = saved_ptr;
            blk->tier     = saved_tier;
            goto fail;
        }

        /* Record both sub-ops */
        pthread_mutex_lock(&e->lock);
        e->op_count[hop1]++;  e->op_bytes[hop1] += data_size;
        e->op_count[hop2]++;  e->op_bytes[hop2] += data_size;
        pthread_mutex_unlock(&e->lock);

    } else {
        /* ---- Single-hop migration ---- */
        rc = e->transfer_fn(blk, dst_buf, data_size,
                            op, e->transfer_ctx);
        if (rc != ORCHKV_OK) goto fail;

        pthread_mutex_lock(&e->lock);
        e->op_count[op]++;
        e->op_bytes[op] += data_size;
        pthread_mutex_unlock(&e->lock);
    }

    /* --- Update block metadata --- */
    blk->tier = target_tier;
    blk->data_ptr = (target_tier == TIER_NVM || target_tier == TIER_SSD)
                        ? NULL : dst_buf;
    blk->state = mig_target_state(target_tier);

    /* --- Update global stats --- */
    pthread_mutex_lock(&e->lock);
    e->op_count[op]++;
    e->op_bytes[op] += data_size;
    e->blocks_migrated++;
    pthread_mutex_unlock(&e->lock);

    /* --- Update LRU if eviction policy is present --- */
    if (e->evpol) {
        if (target_tier == TIER_NVM || target_tier == TIER_SSD)
            evpol_lru_remove(e->evpol, blk);
        else
            evpol_lru_touch(e->evpol, blk);
    }

    /* --- Notify prefetch scheduler of hit if this was a promote --- */
    if (e->prefetch &&
        (op == MIGRATE_PROMOTE_STOR2DRAM || op == MIGRATE_PROMOTE_DRAM2GPU ||
         op == MIGRATE_PROMOTE_STOR2GPU)) {
        prefetch_notify_hit(e->prefetch, blk->block_id);
    }

    return ORCHKV_OK;

fail:
    blk->state = saved_state;
    pthread_mutex_lock(&e->lock);
    e->op_errors++;
    pthread_mutex_unlock(&e->lock);
    return rc;
}

/* ========================================================================
 *  Batch demote
 * ======================================================================== */

uint32_t mig_demote_batch(migration_engine_t *e,
                          kv_block_t **blocks,
                          StorageTier target_tier,
                          void **dst_bufs,
                          void **inter_bufs,
                          size_t data_size,
                          uint32_t n)
{
    if (!e || !blocks || !dst_bufs || n == 0) return 0;

    uint32_t ok = 0;
    for (uint32_t i = 0; i < n; i++) {
        void *inter = inter_bufs ? inter_bufs[i] : NULL;
        int rc = mig_execute_one(e, blocks[i], target_tier,
                                 dst_bufs[i], inter, data_size);
        if (rc == ORCHKV_OK)
            ok++;
    }
    return ok;
}

/* ========================================================================
 *  Statistics
 * ======================================================================== */

void mig_get_stats(const migration_engine_t *e, mig_stats_t *out)
{
    if (!e || !out) return;
    pthread_mutex_lock((pthread_mutex_t *)&e->lock);
    for (int i = 0; i < MIGRATE_OP_COUNT; i++) {
        out->op_count[i] = e->op_count[i];
        out->op_bytes[i] = e->op_bytes[i];
    }
    out->op_errors       = e->op_errors;
    out->blocks_migrated = e->blocks_migrated;
    pthread_mutex_unlock((pthread_mutex_t *)&e->lock);
}
