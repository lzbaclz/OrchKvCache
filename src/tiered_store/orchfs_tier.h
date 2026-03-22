#ifndef ORCHKV_ORCHFS_TIER_H
#define ORCHKV_ORCHFS_TIER_H

#include "tier_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  OrchFS Backend Tier — persistent storage via OrchFS (NVM + SSD)
 *
 *  Each inference request maps to one OrchFS file.
 *  File layout:
 *    offset = (layer * n_kv_heads + head) * max_blocks_per_head * slab_size
 *             + block_idx * slab_size
 *
 *  This keeps blocks for the same (layer, head) contiguous in the file,
 *  aligning with OrchFS's sequential-write optimisation and SSD block
 *  alignment (slab_size = 32KB = ORCH_BLOCK_SIZE).
 *
 *  All functions are guarded by ORCHKV_HAS_ORCHFS.  When OrchFS is not
 *  available, stubs return ORCHKV_ERR_INIT.
 * ======================================================================== */

/* ---- Per-request file context ---- */

typedef struct orchfs_file_ctx {
    int          fd;                 /* orchfs_open fd, -1 if not open */
    char         path[256];
    uint64_t     request_id;
    uint32_t     n_layers;
    uint32_t     n_kv_heads;
    uint32_t     max_blocks_per_head;
    size_t       slab_size;
} orchfs_file_ctx_t;

/* ---- Global OrchFS tier state ---- */

typedef struct orchfs_tier {
    bool         initialized;
    char         base_dir[256];
    size_t       slab_size;

    uint64_t     total_writes;
    uint64_t     total_reads;
    uint64_t     bytes_written;
    uint64_t     bytes_read;

    pthread_mutex_t lock;
} orchfs_tier_t;

/* ---- Lifecycle ---- */

int  orchfs_tier_init(orchfs_tier_t *t, const char *base_dir, size_t slab_size);
void orchfs_tier_destroy(orchfs_tier_t *t);

/* ---- Per-request file management ---- */

orchfs_file_ctx_t *orchfs_file_open(orchfs_tier_t *t,
                                    uint64_t request_id,
                                    uint32_t n_layers,
                                    uint32_t n_kv_heads,
                                    uint32_t max_blocks_per_head);

int  orchfs_file_close(orchfs_tier_t *t, orchfs_file_ctx_t *fctx);

/* ---- Block IO ---- */

/*
 * Write a KV block slab to OrchFS.
 * `data` is a host pointer of `size` bytes.
 * Returns ORCHKV_OK or ORCHKV_ERR_IO.
 */
int  orchfs_tier_write(orchfs_file_ctx_t *fctx,
                       uint32_t layer, uint32_t head, uint32_t block_idx,
                       const void *data, size_t size);

/*
 * Read a KV block slab from OrchFS.
 * `data` is a host buffer of at least `size` bytes.
 */
int  orchfs_tier_read(orchfs_file_ctx_t *fctx,
                      uint32_t layer, uint32_t head, uint32_t block_idx,
                      void *data, size_t size);

/* ---- Offset computation (public for testing) ---- */

static inline int64_t orchfs_compute_offset(const orchfs_file_ctx_t *fctx,
                                            uint32_t layer,
                                            uint32_t head,
                                            uint32_t block_idx)
{
    uint64_t head_linear = (uint64_t)layer * fctx->n_kv_heads + head;
    return (int64_t)(head_linear * fctx->max_blocks_per_head * fctx->slab_size
                     + (uint64_t)block_idx * fctx->slab_size);
}

/* ---- Stats helpers ---- */

static inline uint64_t orchfs_tier_writes(const orchfs_tier_t *t) { return t->total_writes; }
static inline uint64_t orchfs_tier_reads(const orchfs_tier_t *t)  { return t->total_reads;  }

void orchfs_tier_record_write(orchfs_tier_t *t, size_t bytes);
void orchfs_tier_record_read(orchfs_tier_t *t, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_ORCHFS_TIER_H */
