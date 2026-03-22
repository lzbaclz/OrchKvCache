#ifndef ORCHKV_TRANSFER_H
#define ORCHKV_TRANSFER_H

#include "tier_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    XFER_D2H = 0,   /* GPU → DRAM */
    XFER_H2D = 1,   /* DRAM → GPU */
} TransferDir;

/*
 * Multi-stream async transfer engine.
 *
 * Wraps cudaMemcpyAsync across a pool of CUDA streams, using round-robin
 * assignment for concurrent transfers.  Provides both single-shot and
 * synchronous convenience APIs.
 */
typedef struct transfer_engine {
    int          device_id;
    int          num_streams;
    void       **streams;          /* cudaStream_t array (void* for C compat) */
    int          next_stream;      /* round-robin counter */

    /* Lifetime counters */
    uint64_t     total_d2h;
    uint64_t     total_h2d;
    uint64_t     bytes_d2h;
    uint64_t     bytes_h2d;

    pthread_mutex_t lock;
} transfer_engine_t;

int  transfer_engine_init(transfer_engine_t *eng, int device_id, int num_streams);
void transfer_engine_destroy(transfer_engine_t *eng);

/*
 * Submit an async transfer on the next stream.
 * Returns the stream index used (for later sync), or negative error.
 */
int  transfer_submit(transfer_engine_t *eng,
                     void *dst, const void *src,
                     size_t bytes, TransferDir dir);

/* Synchronise a specific stream. */
int  transfer_sync_stream(transfer_engine_t *eng, int stream_idx);

/* Synchronise ALL streams. */
int  transfer_sync_all(transfer_engine_t *eng);

/* Get the raw cudaStream_t for advanced usage (e.g. CUDA events). */
void *transfer_get_stream(transfer_engine_t *eng, int stream_idx);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_TRANSFER_H */
