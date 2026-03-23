#ifndef ORCHKV_MIGRATION_ENGINE_H
#define ORCHKV_MIGRATION_ENGINE_H

#include "../core/kv_types.h"
#include "../core/kv_block.h"
#include "eviction_policy.h"
#include "prefetch_scheduler.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Migration Engine — Phase C (C7)
 *
 *  Coordinates Demote (eviction) and Promote (prefetch) operations across
 *  storage tiers.  Handles:
 *
 *    GPU ↔ DRAM   (single-hop via CUDA transfer)
 *    DRAM ↔ Storage  (single-hop via OrchFS IO)
 *    GPU ↔ Storage   (two-hop: GPU↔DRAM then DRAM↔Storage)
 *
 *  The engine uses a pluggable transfer callback so it can be tested
 *  without CUDA.  In production, wire up transfer_engine + orchfs_tier.
 *
 *  Block state is set to KV_STATE_MIGRATING during transfer to prevent
 *  double-migration.  Pinned blocks (KV_FLAG_PIN) are unconditionally
 *  rejected.
 *
 *  Thread-safety: all public functions are safe to call concurrently.
 * ======================================================================== */

/* ---------- Migration operations --------------------------------------- */

typedef enum {
    MIGRATE_DEMOTE_GPU2DRAM  = 0,   /* GPU → DRAM */
    MIGRATE_DEMOTE_DRAM2STOR = 1,   /* DRAM → Storage */
    MIGRATE_DEMOTE_GPU2STOR  = 2,   /* GPU → Storage (two-hop) */
    MIGRATE_PROMOTE_STOR2DRAM = 3,  /* Storage → DRAM */
    MIGRATE_PROMOTE_DRAM2GPU  = 4,  /* DRAM → GPU */
    MIGRATE_PROMOTE_STOR2GPU  = 5,  /* Storage → GPU (two-hop) */
    MIGRATE_OP_COUNT          = 6,
} MigrateOp;

/* ---------- Transfer callback ------------------------------------------ */

/*
 * Called by the engine to perform actual data movement for a single hop.
 *
 *   blk      — the block being migrated (read blk->data_ptr for source)
 *   dst_buf  — destination buffer (pre-allocated by caller)
 *   size     — bytes to transfer
 *   op       — which single-hop operation (never a two-hop op)
 *   ctx      — user context passed at init
 *
 * Returns ORCHKV_OK on success, or an error code.
 */
typedef int (*mig_transfer_fn)(kv_block_t *blk,
                               void *dst_buf,
                               size_t size,
                               MigrateOp op,
                               void *ctx);

/* ---------- Statistics ------------------------------------------------- */

typedef struct mig_stats {
    uint64_t op_count[MIGRATE_OP_COUNT];
    uint64_t op_bytes[MIGRATE_OP_COUNT];
    uint64_t op_errors;
    uint64_t blocks_migrated;
} mig_stats_t;

/* ---------- Engine state ----------------------------------------------- */

typedef struct migration_engine {
    eviction_policy_t    *evpol;
    prefetch_scheduler_t *prefetch;

    mig_transfer_fn  transfer_fn;
    void            *transfer_ctx;

    uint64_t op_count[MIGRATE_OP_COUNT];
    uint64_t op_bytes[MIGRATE_OP_COUNT];
    uint64_t op_errors;
    uint64_t blocks_migrated;

    pthread_mutex_t lock;
} migration_engine_t;

/* ---- Lifecycle -------------------------------------------------------- */

int  mig_init(migration_engine_t *e,
              eviction_policy_t *evpol,
              prefetch_scheduler_t *prefetch,
              mig_transfer_fn transfer_fn,
              void *transfer_ctx);

void mig_destroy(migration_engine_t *e);

/* ---- Operation routing ------------------------------------------------ */

/*
 * Determine the migration op from source tier to target tier.
 * Returns MIGRATE_OP_COUNT if the transition is invalid (e.g. same tier).
 */
MigrateOp mig_determine_op(StorageTier from, StorageTier to);

/*
 * Map a storage tier to the expected block state after migration.
 */
KVBlockState mig_target_state(StorageTier tier);

const char *mig_op_name(MigrateOp op);

/* ---- Core migration --------------------------------------------------- */

/*
 * Migrate a single block to target_tier.
 *
 *   blk              — block to migrate (state set to MIGRATING during op)
 *   target_tier      — destination tier
 *   dst_buf          — pre-allocated destination buffer
 *   intermediate_buf — for two-hop ops (GPU↔Storage), a temporary DRAM
 *                      buffer; NULL for single-hop ops
 *   data_size        — bytes to transfer
 *
 * On success: block's tier, data_ptr, and state are updated.
 * On failure: block state is restored; returns an error code.
 *
 * Rejects pinned (KV_FLAG_PIN) or already-migrating blocks.
 */
int mig_execute_one(migration_engine_t *e,
                    kv_block_t *blk,
                    StorageTier target_tier,
                    void *dst_buf,
                    void *intermediate_buf,
                    size_t data_size);

/*
 * Batch demote: migrate an array of blocks to target_tier.
 * Skips blocks that fail validation (pinned, wrong state).
 *
 *   blocks      — array of block pointers
 *   target_tier — where to move them
 *   dst_bufs    — array of pre-allocated destination buffers
 *   inter_bufs  — array of intermediate buffers (for two-hop; NULL entries OK for single-hop)
 *   data_size   — bytes per block
 *   n           — number of blocks
 *
 * Returns the number of blocks successfully migrated.
 */
uint32_t mig_demote_batch(migration_engine_t *e,
                          kv_block_t **blocks,
                          StorageTier target_tier,
                          void **dst_bufs,
                          void **inter_bufs,
                          size_t data_size,
                          uint32_t n);

/* ---- Statistics ------------------------------------------------------- */

void mig_get_stats(const migration_engine_t *e, mig_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_MIGRATION_ENGINE_H */
