extern "C" {
#include "orchkv_api.h"
}
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 *  Global system state (singleton)
 *
 *  Slab layout (fixed-offset, no data movement on append):
 *    [0,             slab_size/2)  → K data (up to tokens_per_block tokens)
 *    [slab_size/2,   slab_size)    → V data (up to tokens_per_block tokens)
 *
 *  Within each half, token t occupies byte range:
 *    [t * head_stride, (t+1) * head_stride)
 *  where head_stride = d_head * dtype_size.
 * ======================================================================== */

static struct {
    bool              inited;
    orchkv_config_t   cfg;
    gpu_tier_t        gpu;
    dram_tier_t       dram;
    transfer_engine_t xfer;
    address_map_t     addrmap;
    size_t            slab_size;
} g_sys;

static inline size_t slab_k_half(void)  { return g_sys.slab_size / 2; }

/* ---- init / shutdown --------------------------------------------------- */

int orchkv_init(const orchkv_config_t *config)
{
    if (g_sys.inited) return ORCHKV_ERR_ALREADY;

    g_sys.cfg = *config;
    g_sys.slab_size = kv_block_data_bytes(config->tokens_per_block,
                                          config->d_head,
                                          config->dtype);
    int rc;
    rc = gpu_tier_init(&g_sys.gpu, config->gpu_device_id,
                       config->gpu_pool_bytes, g_sys.slab_size);
    if (rc != ORCHKV_OK) return rc;

    rc = dram_tier_init(&g_sys.dram, config->dram_pool_bytes,
                        g_sys.slab_size, config->dram_use_pinned);
    if (rc != ORCHKV_OK) { gpu_tier_destroy(&g_sys.gpu); return rc; }

    rc = transfer_engine_init(&g_sys.xfer, config->gpu_device_id,
                              config->num_cuda_streams);
    if (rc != ORCHKV_OK) {
        dram_tier_destroy(&g_sys.dram);
        gpu_tier_destroy(&g_sys.gpu);
        return rc;
    }

    rc = address_map_init(&g_sys.addrmap, KV_ADDRMAP_INIT_CAP);
    if (rc != ORCHKV_OK) {
        transfer_engine_destroy(&g_sys.xfer);
        dram_tier_destroy(&g_sys.dram);
        gpu_tier_destroy(&g_sys.gpu);
        return rc;
    }

    g_sys.inited = true;
    LOG_INFO("orchkv_init OK: slab=%zu B, gpu_slabs=%u, dram_slabs=%u, streams=%d",
             g_sys.slab_size, g_sys.gpu.total_slabs, g_sys.dram.total_slabs,
             g_sys.xfer.num_streams);
    return ORCHKV_OK;
}

int orchkv_shutdown(void)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;

    address_map_destroy(&g_sys.addrmap);
    transfer_engine_destroy(&g_sys.xfer);
    dram_tier_destroy(&g_sys.dram);
    gpu_tier_destroy(&g_sys.gpu);

    g_sys.inited = false;
    LOG_INFO("orchkv_shutdown OK");
    return ORCHKV_OK;
}

bool orchkv_is_initialized(void) { return g_sys.inited; }

gpu_tier_t       *orchkv_gpu_tier(void)         { return &g_sys.gpu;     }
dram_tier_t      *orchkv_dram_tier(void)        { return &g_sys.dram;    }
transfer_engine_t *orchkv_transfer_engine(void) { return &g_sys.xfer;    }
address_map_t    *orchkv_address_map(void)      { return &g_sys.addrmap; }

/* ---- Request lifecycle ------------------------------------------------- */

kv_request_ctx_t *orchkv_request_create(uint64_t request_id,
                                        uint32_t n_layers,
                                        uint32_t n_kv_heads)
{
    if (!g_sys.inited) return NULL;
    return kv_request_create(request_id, n_layers, n_kv_heads,
                             g_sys.cfg.d_head, g_sys.cfg.dtype,
                             g_sys.cfg.tokens_per_block);
}

int orchkv_request_destroy(kv_request_ctx_t *ctx)
{
    if (!ctx) return ORCHKV_ERR_INVALID;

    for (uint32_t l = 0; l < ctx->n_layers; l++) {
        for (uint32_t h = 0; h < ctx->n_kv_heads; h++) {
            kv_head_vec_t *hv = kv_request_hvec(ctx, l, h);
            if (!hv) continue;
            for (uint32_t bi = 0; bi < hv->count; bi++) {
                kv_block_t *blk = hv->items[bi];
                if (!blk) continue;

                address_map_remove(&g_sys.addrmap, blk->block_id);

                if (blk->data_ptr) {
                    if (blk->tier == TIER_GPU_HBM)
                        gpu_tier_free(&g_sys.gpu, blk->data_ptr);
                    else if (blk->tier == TIER_HOST_DRAM)
                        dram_tier_free(&g_sys.dram, blk->data_ptr);
                    blk->data_ptr = NULL;
                }
            }
        }
    }
    kv_request_destroy(ctx);
    return ORCHKV_OK;
}

/* ---- Helper: assign GPU slab to a FREE block ---- */

static int assign_gpu_slab(kv_block_t *blk)
{
    void *ptr;
    int rc = gpu_tier_alloc(&g_sys.gpu, &ptr);
    if (rc != ORCHKV_OK) return rc;

    blk->data_ptr = ptr;
    blk->tier     = TIER_GPU_HBM;
    kv_block_set_state(blk, KV_STATE_ALLOCATED);
    kv_block_set_state(blk, KV_STATE_HOT);

    address_map_insert(&g_sys.addrmap, blk);
    return ORCHKV_OK;
}

/* ---- Prefill ----------------------------------------------------------- */

int orchkv_prefill(kv_request_ctx_t *ctx,
                   uint32_t layer_id,
                   const void *k_data,
                   const void *v_data,
                   uint32_t seq_len)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;
    if (!ctx || layer_id >= ctx->n_layers) return ORCHKV_ERR_INVALID;

    int new_blocks = kv_request_alloc_blocks_for_layer(ctx, layer_id, seq_len);
    if (new_blocks < 0) return new_blocks;

    size_t per_head_bytes = (size_t)seq_len * ctx->d_head * dtype_size(ctx->dtype);
    size_t kh = slab_k_half();

    for (uint32_t h = 0; h < ctx->n_kv_heads; h++) {
        kv_head_vec_t *hv = kv_request_hvec(ctx, layer_id, h);

        for (uint32_t bi = 0; bi < hv->count; bi++) {
            kv_block_t *blk = hv->items[bi];
            if (blk->state != KV_STATE_FREE) continue;

            int rc = assign_gpu_slab(blk);
            if (rc != ORCHKV_OK) return rc;

            size_t tok_bytes = (size_t)blk->token_count * ctx->d_head
                               * dtype_size(ctx->dtype);
            size_t tok_off   = (size_t)blk->token_start * ctx->d_head
                               * dtype_size(ctx->dtype);

            if (k_data && tok_off < per_head_bytes) {
                const char *ks = (const char *)k_data
                                 + h * per_head_bytes + tok_off;
                int si = transfer_submit(&g_sys.xfer, blk->data_ptr,
                                         ks, tok_bytes, XFER_H2D);
                if (si >= 0) transfer_sync_stream(&g_sys.xfer, si);
            }

            if (v_data && tok_off < per_head_bytes) {
                const char *vs = (const char *)v_data
                                 + h * per_head_bytes + tok_off;
                char *vd = (char *)blk->data_ptr + kh;
                int si = transfer_submit(&g_sys.xfer, vd,
                                         vs, tok_bytes, XFER_H2D);
                if (si >= 0) transfer_sync_stream(&g_sys.xfer, si);
            }

            ctx->blocks_on_gpu++;
        }
    }

    if (layer_id == 0)
        kv_request_advance_seq(ctx, seq_len);

    return ORCHKV_OK;
}

/* ---- Append token ------------------------------------------------------ */

int orchkv_append_token(kv_request_ctx_t *ctx,
                        uint32_t layer_id,
                        const void *k_token,
                        const void *v_token)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;
    if (!ctx || layer_id >= ctx->n_layers) return ORCHKV_ERR_INVALID;

    int new_blocks = kv_request_alloc_blocks_for_layer(ctx, layer_id, 1);
    if (new_blocks < 0) return new_blocks;

    size_t head_stride = (size_t)ctx->d_head * dtype_size(ctx->dtype);
    size_t kh = slab_k_half();

    for (uint32_t h = 0; h < ctx->n_kv_heads; h++) {
        kv_head_vec_t *hv = kv_request_hvec(ctx, layer_id, h);
        kv_block_t *blk = hv->items[hv->count - 1];

        if (blk->state == KV_STATE_FREE) {
            int rc = assign_gpu_slab(blk);
            if (rc != ORCHKV_OK) return rc;
            ctx->blocks_on_gpu++;
        }

        if (blk->tier != TIER_GPU_HBM) {
            LOG_ERR("append_token: last block not on GPU (layer=%u head=%u)",
                    layer_id, h);
            return ORCHKV_ERR_STATE;
        }

        uint32_t slot = blk->token_count - 1;
        size_t slot_off = (size_t)slot * head_stride;

        if (k_token) {
            const char *ks = (const char *)k_token + h * head_stride;
            char *kd = (char *)blk->data_ptr + slot_off;
            int si = transfer_submit(&g_sys.xfer, kd, ks,
                                     head_stride, XFER_H2D);
            if (si >= 0) transfer_sync_stream(&g_sys.xfer, si);
        }

        if (v_token) {
            const char *vs = (const char *)v_token + h * head_stride;
            char *vd = (char *)blk->data_ptr + kh + slot_off;
            int si = transfer_submit(&g_sys.xfer, vd, vs,
                                     head_stride, XFER_H2D);
            if (si >= 0) transfer_sync_stream(&g_sys.xfer, si);
        }
    }

    if (layer_id == 0)
        kv_request_advance_seq(ctx, 1);

    return ORCHKV_OK;
}

/* ---- Get KV block ------------------------------------------------------ */

int orchkv_get_kv_block(kv_request_ctx_t *ctx,
                        uint32_t layer_id,
                        uint32_t head_id,
                        uint32_t block_idx,
                        void **k_out,
                        void **v_out)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;

    kv_block_t *blk = kv_request_get_block(ctx, layer_id, head_id, block_idx);
    if (!blk) return ORCHKV_ERR_NOT_FOUND;

    if (blk->tier == TIER_HOST_DRAM) {
        int rc = orchkv_promote_to_gpu(ctx, layer_id, head_id, block_idx);
        if (rc != ORCHKV_OK) return rc;
    }

    if (blk->tier != TIER_GPU_HBM)
        return ORCHKV_ERR_STATE;

    *k_out = blk->data_ptr;
    *v_out = (char *)blk->data_ptr + slab_k_half();
    return ORCHKV_OK;
}

/* ---- Evict to DRAM ----------------------------------------------------- */

int orchkv_evict_to_dram(kv_request_ctx_t *ctx,
                         uint32_t layer_id,
                         uint32_t head_id,
                         uint32_t block_idx)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;

    kv_block_t *blk = kv_request_get_block(ctx, layer_id, head_id, block_idx);
    if (!blk) return ORCHKV_ERR_NOT_FOUND;
    if (blk->tier != TIER_GPU_HBM) return ORCHKV_ERR_STATE;
    if (kv_block_is_pinned(blk))   return ORCHKV_ERR_LOCKED;

    void *dram_ptr;
    int rc = dram_tier_alloc(&g_sys.dram, &dram_ptr);
    if (rc != ORCHKV_OK) return rc;

    kv_block_set_state(blk, KV_STATE_MIGRATING);

    int si = transfer_submit(&g_sys.xfer, dram_ptr, blk->data_ptr,
                             g_sys.slab_size, XFER_D2H);
    if (si < 0) {
        dram_tier_free(&g_sys.dram, dram_ptr);
        kv_block_set_state(blk, KV_STATE_HOT);
        return si;
    }
    transfer_sync_stream(&g_sys.xfer, si);

    void *old_gpu = blk->data_ptr;
    kv_block_set_location(blk, TIER_HOST_DRAM, dram_ptr);
    kv_block_set_state(blk, KV_STATE_WARM);

    gpu_tier_free(&g_sys.gpu, old_gpu);

    ctx->blocks_on_gpu--;
    ctx->blocks_on_dram++;
    return ORCHKV_OK;
}

/* ---- Promote to GPU ---------------------------------------------------- */

int orchkv_promote_to_gpu(kv_request_ctx_t *ctx,
                          uint32_t layer_id,
                          uint32_t head_id,
                          uint32_t block_idx)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;

    kv_block_t *blk = kv_request_get_block(ctx, layer_id, head_id, block_idx);
    if (!blk) return ORCHKV_ERR_NOT_FOUND;
    if (blk->tier != TIER_HOST_DRAM) return ORCHKV_ERR_STATE;

    void *gpu_ptr;
    int rc = gpu_tier_alloc(&g_sys.gpu, &gpu_ptr);
    if (rc != ORCHKV_OK) return rc;

    kv_block_set_state(blk, KV_STATE_MIGRATING);

    int si = transfer_submit(&g_sys.xfer, gpu_ptr, blk->data_ptr,
                             g_sys.slab_size, XFER_H2D);
    if (si < 0) {
        gpu_tier_free(&g_sys.gpu, gpu_ptr);
        kv_block_set_state(blk, KV_STATE_WARM);
        return si;
    }
    transfer_sync_stream(&g_sys.xfer, si);

    void *old_dram = blk->data_ptr;
    kv_block_set_location(blk, TIER_GPU_HBM, gpu_ptr);
    kv_block_set_state(blk, KV_STATE_HOT);

    dram_tier_free(&g_sys.dram, old_dram);

    ctx->blocks_on_dram--;
    ctx->blocks_on_gpu++;
    return ORCHKV_OK;
}

/* ---- Stats ------------------------------------------------------------- */

int orchkv_get_stats(orchkv_stats_t *stats)
{
    if (!g_sys.inited) return ORCHKV_ERR_INIT;
    memset(stats, 0, sizeof(*stats));

    stats->gpu_pool_total  = g_sys.gpu.pool_size;
    stats->gpu_pool_used   = (size_t)g_sys.gpu.used_slabs * g_sys.gpu.slab_size;
    stats->gpu_slabs_total = g_sys.gpu.total_slabs;
    stats->gpu_slabs_used  = g_sys.gpu.used_slabs;

    stats->dram_pool_total  = g_sys.dram.pool_size;
    stats->dram_pool_used   = (size_t)g_sys.dram.used_slabs * g_sys.dram.slab_size;
    stats->dram_slabs_total = g_sys.dram.total_slabs;
    stats->dram_slabs_used  = g_sys.dram.used_slabs;

    stats->total_blocks  = address_map_count(&g_sys.addrmap);
    stats->transfers_d2h = g_sys.xfer.total_d2h;
    stats->transfers_h2d = g_sys.xfer.total_h2d;
    stats->bytes_d2h     = g_sys.xfer.bytes_d2h;
    stats->bytes_h2d     = g_sys.xfer.bytes_h2d;

    return ORCHKV_OK;
}
