#ifndef ORCHKV_API_H
#define ORCHKV_API_H

#include "../core/kv_types.h"
#include "../core/kv_block.h"
#include "../core/kv_request.h"
#include "../core/address_map.h"
#include "../tiered_store/gpu_tier.h"
#include "../tiered_store/dram_tier.h"
#include "../tiered_store/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  OrchKvCache Public API  — Phase A subset
 *
 *  Lifecycle:   orchkv_init  →  (use)  →  orchkv_shutdown
 *  Per-request: orchkv_request_create  →  prefill / append / get_kv  →
 *               orchkv_request_destroy
 *  Migration:   orchkv_evict_to_dram / orchkv_promote_to_gpu
 * ======================================================================== */

int  orchkv_init(const orchkv_config_t *config);
int  orchkv_shutdown(void);
bool orchkv_is_initialized(void);

/* ---- Request lifecycle ---- */

kv_request_ctx_t *orchkv_request_create(uint64_t request_id,
                                        uint32_t n_layers,
                                        uint32_t n_kv_heads);

int orchkv_request_destroy(kv_request_ctx_t *ctx);

/* ---- Data operations ---- */

/*
 * Prefill: for a single layer, allocate GPU slabs and copy KV data.
 * k_data / v_data are HOST pointers of shape [seq_len, d_head], dtype.
 * The function copies data to GPU asynchronously and waits.
 */
int orchkv_prefill(kv_request_ctx_t *ctx,
                   uint32_t layer_id,
                   const void *k_data,
                   const void *v_data,
                   uint32_t seq_len);

/*
 * Decode: append a single token's KV for one layer.
 * k_token / v_token are HOST pointers of shape [n_kv_heads, d_head], dtype.
 */
int orchkv_append_token(kv_request_ctx_t *ctx,
                        uint32_t layer_id,
                        const void *k_token,
                        const void *v_token);

/*
 * Get the GPU pointer for all KV data of a layer.
 * If some blocks are not on GPU, they are promoted first.
 * Returns GPU pointers for K and V contiguously per-head.
 *
 * For Phase A: k_out[head] and v_out[head] point into the slab pool.
 * seq_len_out receives the current sequence length.
 */
int orchkv_get_kv_block(kv_request_ctx_t *ctx,
                        uint32_t layer_id,
                        uint32_t head_id,
                        uint32_t block_idx,
                        void **k_out,
                        void **v_out);

/* ---- Migration (Phase A: GPU ↔ DRAM only) ---- */

int orchkv_evict_to_dram(kv_request_ctx_t *ctx,
                         uint32_t layer_id,
                         uint32_t head_id,
                         uint32_t block_idx);

int orchkv_promote_to_gpu(kv_request_ctx_t *ctx,
                          uint32_t layer_id,
                          uint32_t head_id,
                          uint32_t block_idx);

/* ---- Statistics ---- */

int orchkv_get_stats(orchkv_stats_t *stats);

/* ---- Access global subsystems (for tests / advanced use) ---- */

gpu_tier_t       *orchkv_gpu_tier(void);
dram_tier_t      *orchkv_dram_tier(void);
transfer_engine_t *orchkv_transfer_engine(void);
address_map_t    *orchkv_address_map(void);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_API_H */
