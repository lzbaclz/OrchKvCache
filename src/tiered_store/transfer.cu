extern "C" {
#include "transfer.h"
}
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

int transfer_engine_init(transfer_engine_t *eng, int device_id, int num_streams)
{
    memset(eng, 0, sizeof(*eng));

    if (num_streams <= 0)
        num_streams = KV_DEFAULT_NUM_STREAMS;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        LOG_ERR("cudaSetDevice(%d): %s", device_id, cudaGetErrorString(err));
        return ORCHKV_ERR_CUDA;
    }

    eng->device_id   = device_id;
    eng->num_streams = num_streams;

    eng->streams = (void **)calloc(num_streams, sizeof(void *));
    if (!eng->streams)
        return ORCHKV_ERR_OOM;

    for (int i = 0; i < num_streams; i++) {
        cudaStream_t s;
        err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            LOG_ERR("cudaStreamCreate[%d]: %s", i, cudaGetErrorString(err));
            for (int j = 0; j < i; j++)
                cudaStreamDestroy((cudaStream_t)eng->streams[j]);
            free(eng->streams);
            return ORCHKV_ERR_CUDA;
        }
        eng->streams[i] = (void *)s;
    }

    eng->next_stream = 0;
    pthread_mutex_init(&eng->lock, NULL);

    LOG_INFO("transfer_engine: device %d, %d streams", device_id, num_streams);
    return ORCHKV_OK;
}

void transfer_engine_destroy(transfer_engine_t *eng)
{
    if (eng->streams) {
        cudaSetDevice(eng->device_id);
        for (int i = 0; i < eng->num_streams; i++) {
            if (eng->streams[i])
                cudaStreamDestroy((cudaStream_t)eng->streams[i]);
        }
        free(eng->streams);
        eng->streams = NULL;
    }
    pthread_mutex_destroy(&eng->lock);
}

int transfer_submit(transfer_engine_t *eng,
                    void *dst, const void *src,
                    size_t bytes, TransferDir dir)
{
    cudaSetDevice(eng->device_id);

    cudaMemcpyKind kind;
    if (dir == XFER_D2H)
        kind = cudaMemcpyDeviceToHost;
    else
        kind = cudaMemcpyHostToDevice;

    pthread_mutex_lock(&eng->lock);
    int si = eng->next_stream;
    eng->next_stream = (si + 1) % eng->num_streams;

    if (dir == XFER_D2H) { eng->total_d2h++; eng->bytes_d2h += bytes; }
    else                 { eng->total_h2d++; eng->bytes_h2d += bytes; }
    pthread_mutex_unlock(&eng->lock);

    cudaStream_t stream = (cudaStream_t)eng->streams[si];
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (err != cudaSuccess) {
        LOG_ERR("cudaMemcpyAsync(%zu bytes, %s): %s",
                bytes, dir == XFER_D2H ? "D2H" : "H2D",
                cudaGetErrorString(err));
        return ORCHKV_ERR_CUDA;
    }

    return si;
}

int transfer_sync_stream(transfer_engine_t *eng, int stream_idx)
{
    if (stream_idx < 0 || stream_idx >= eng->num_streams)
        return ORCHKV_ERR_INVALID;

    cudaSetDevice(eng->device_id);
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)eng->streams[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERR("cudaStreamSynchronize[%d]: %s", stream_idx, cudaGetErrorString(err));
        return ORCHKV_ERR_CUDA;
    }
    return ORCHKV_OK;
}

int transfer_sync_all(transfer_engine_t *eng)
{
    cudaSetDevice(eng->device_id);
    for (int i = 0; i < eng->num_streams; i++) {
        cudaError_t err = cudaStreamSynchronize((cudaStream_t)eng->streams[i]);
        if (err != cudaSuccess) {
            LOG_ERR("cudaStreamSynchronize[%d]: %s", i, cudaGetErrorString(err));
            return ORCHKV_ERR_CUDA;
        }
    }
    return ORCHKV_OK;
}

void *transfer_get_stream(transfer_engine_t *eng, int stream_idx)
{
    if (stream_idx < 0 || stream_idx >= eng->num_streams)
        return NULL;
    return eng->streams[stream_idx];
}
