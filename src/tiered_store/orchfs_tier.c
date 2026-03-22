/*
 * orchfs_tier.c — Persistent storage backend for OrchKvCache
 *
 * ORCHKV_HAS_ORCHFS=1 : links against libOrchFS, calls orchfs_pwrite/pread.
 * ORCHKV_HAS_ORCHFS=0 : POSIX fallback — uses standard pwrite/pread on a
 *                        configurable directory (default /dev/shm/orchkv).
 *                        Works on any Linux without special hardware.
 */

#include "orchfs_tier.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

/* ========================================================================
 *  OrchFS backend (linked against libOrchFS)
 * ======================================================================== */

#if ORCHKV_HAS_ORCHFS

#include <orchfs.h>

static bool g_libfs_inited = false;

int orchfs_tier_init(orchfs_tier_t *t, const char *base_dir, size_t slab_size)
{
    memset(t, 0, sizeof(*t));

    if (slab_size == 0)
        return ORCHKV_ERR_INVALID;

    if (!g_libfs_inited) {
        init_libfs();
        g_libfs_inited = true;
    }

    snprintf(t->base_dir, sizeof(t->base_dir), "%s", base_dir);
    t->slab_size = slab_size;
    pthread_mutex_init(&t->lock, NULL);

    orchfs_mkdir(base_dir, 0755);

    t->initialized = true;
    LOG_INFO("orchfs_tier [orchfs]: dir=%s, slab=%zu B", base_dir, slab_size);
    return ORCHKV_OK;
}

void orchfs_tier_destroy(orchfs_tier_t *t)
{
    if (!t->initialized) return;

    LOG_INFO("orchfs_tier: writes=%lu reads=%lu bytes_w=%lu bytes_r=%lu",
             (unsigned long)t->total_writes, (unsigned long)t->total_reads,
             (unsigned long)t->bytes_written, (unsigned long)t->bytes_read);

    pthread_mutex_destroy(&t->lock);
    t->initialized = false;

    if (g_libfs_inited) {
        close_libfs();
        g_libfs_inited = false;
    }
}

orchfs_file_ctx_t *orchfs_file_open(orchfs_tier_t *t,
                                    uint64_t request_id,
                                    uint32_t n_layers,
                                    uint32_t n_kv_heads,
                                    uint32_t max_blocks_per_head)
{
    if (!t || !t->initialized)
        return NULL;

    orchfs_file_ctx_t *fctx = (orchfs_file_ctx_t *)calloc(1, sizeof(*fctx));
    if (!fctx) return NULL;

    snprintf(fctx->path, sizeof(fctx->path),
             "%s/req_%lu", t->base_dir, (unsigned long)request_id);

    fctx->fd = orchfs_open(fctx->path, O_CREAT | O_RDWR, 0644);
    if (fctx->fd < 0) {
        LOG_ERR("orchfs_open(%s) failed", fctx->path);
        free(fctx);
        return NULL;
    }

    fctx->request_id          = request_id;
    fctx->n_layers            = n_layers;
    fctx->n_kv_heads          = n_kv_heads;
    fctx->max_blocks_per_head = max_blocks_per_head;
    fctx->slab_size           = t->slab_size;

    return fctx;
}

int orchfs_file_close(orchfs_tier_t *t, orchfs_file_ctx_t *fctx)
{
    (void)t;
    if (!fctx) return ORCHKV_ERR_INVALID;

    if (fctx->fd >= 0) {
        orchfs_close(fctx->fd);
        fctx->fd = -1;
    }
    orchfs_unlink(fctx->path);
    free(fctx);
    return ORCHKV_OK;
}

int orchfs_tier_write(orchfs_file_ctx_t *fctx,
                      uint32_t layer, uint32_t head, uint32_t block_idx,
                      const void *data, size_t size)
{
    if (!fctx || fctx->fd < 0)
        return ORCHKV_ERR_INVALID;

    int64_t offset  = orchfs_compute_offset(fctx, layer, head, block_idx);
    int64_t written = orchfs_pwrite(fctx->fd, data, (int64_t)size, offset);

    if (written != (int64_t)size) {
        LOG_ERR("orchfs_pwrite: expected %zu got %ld", size, (long)written);
        return ORCHKV_ERR_IO;
    }
    return ORCHKV_OK;
}

int orchfs_tier_read(orchfs_file_ctx_t *fctx,
                     uint32_t layer, uint32_t head, uint32_t block_idx,
                     void *data, size_t size)
{
    if (!fctx || fctx->fd < 0)
        return ORCHKV_ERR_INVALID;

    int64_t offset = orchfs_compute_offset(fctx, layer, head, block_idx);
    int64_t nread  = orchfs_pread(fctx->fd, data, (int64_t)size, offset);

    if (nread != (int64_t)size) {
        LOG_ERR("orchfs_pread: expected %zu got %ld", size, (long)nread);
        return ORCHKV_ERR_IO;
    }
    return ORCHKV_OK;
}

/* ========================================================================
 *  POSIX fallback (no OrchFS — uses standard pwrite/pread on tmpfs)
 * ======================================================================== */

#else /* !ORCHKV_HAS_ORCHFS */

static int posix_mkdir_p(const char *path, mode_t mode)
{
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, mode);
            *p = '/';
        }
    }
    return mkdir(tmp, mode);
}

int orchfs_tier_init(orchfs_tier_t *t, const char *base_dir, size_t slab_size)
{
    memset(t, 0, sizeof(*t));

    if (slab_size == 0)
        return ORCHKV_ERR_INVALID;

    if (!base_dir || base_dir[0] == '\0')
        base_dir = "/dev/shm/orchkv";

    snprintf(t->base_dir, sizeof(t->base_dir), "%s", base_dir);
    t->slab_size = slab_size;
    pthread_mutex_init(&t->lock, NULL);

    posix_mkdir_p(base_dir, 0755);

    t->initialized = true;
    LOG_INFO("orchfs_tier [posix]: dir=%s, slab=%zu B", base_dir, slab_size);
    return ORCHKV_OK;
}

void orchfs_tier_destroy(orchfs_tier_t *t)
{
    if (!t->initialized) return;

    LOG_INFO("orchfs_tier: writes=%lu reads=%lu bytes_w=%lu bytes_r=%lu",
             (unsigned long)t->total_writes, (unsigned long)t->total_reads,
             (unsigned long)t->bytes_written, (unsigned long)t->bytes_read);

    pthread_mutex_destroy(&t->lock);
    t->initialized = false;
}

orchfs_file_ctx_t *orchfs_file_open(orchfs_tier_t *t,
                                    uint64_t request_id,
                                    uint32_t n_layers,
                                    uint32_t n_kv_heads,
                                    uint32_t max_blocks_per_head)
{
    if (!t || !t->initialized)
        return NULL;

    orchfs_file_ctx_t *fctx = (orchfs_file_ctx_t *)calloc(1, sizeof(*fctx));
    if (!fctx) return NULL;

    snprintf(fctx->path, sizeof(fctx->path),
             "%s/req_%lu", t->base_dir, (unsigned long)request_id);

    fctx->fd = open(fctx->path, O_CREAT | O_RDWR, 0644);
    if (fctx->fd < 0) {
        LOG_ERR("open(%s) failed: %s", fctx->path, strerror(errno));
        free(fctx);
        return NULL;
    }

    fctx->request_id          = request_id;
    fctx->n_layers            = n_layers;
    fctx->n_kv_heads          = n_kv_heads;
    fctx->max_blocks_per_head = max_blocks_per_head;
    fctx->slab_size           = t->slab_size;

    return fctx;
}

int orchfs_file_close(orchfs_tier_t *t, orchfs_file_ctx_t *fctx)
{
    (void)t;
    if (!fctx) return ORCHKV_ERR_INVALID;

    if (fctx->fd >= 0) {
        close(fctx->fd);
        fctx->fd = -1;
    }
    unlink(fctx->path);
    free(fctx);
    return ORCHKV_OK;
}

static ssize_t full_pwrite(int fd, const void *buf, size_t count, off_t offset)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t n = pwrite(fd, p, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p         += n;
        offset    += n;
        remaining -= (size_t)n;
    }
    return (ssize_t)count;
}

static ssize_t full_pread(int fd, void *buf, size_t count, off_t offset)
{
    uint8_t *p = (uint8_t *)buf;
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t n = pread(fd, p, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return (ssize_t)(count - remaining);
        p         += n;
        offset    += n;
        remaining -= (size_t)n;
    }
    return (ssize_t)count;
}

int orchfs_tier_write(orchfs_file_ctx_t *fctx,
                      uint32_t layer, uint32_t head, uint32_t block_idx,
                      const void *data, size_t size)
{
    if (!fctx || fctx->fd < 0)
        return ORCHKV_ERR_INVALID;

    int64_t offset = orchfs_compute_offset(fctx, layer, head, block_idx);
    ssize_t written = full_pwrite(fctx->fd, data, size, (off_t)offset);

    if (written != (ssize_t)size) {
        LOG_ERR("pwrite: expected %zu got %zd (offset=%ld, errno=%s)",
                size, written, (long)offset, strerror(errno));
        return ORCHKV_ERR_IO;
    }
    return ORCHKV_OK;
}

int orchfs_tier_read(orchfs_file_ctx_t *fctx,
                     uint32_t layer, uint32_t head, uint32_t block_idx,
                     void *data, size_t size)
{
    if (!fctx || fctx->fd < 0)
        return ORCHKV_ERR_INVALID;

    int64_t offset = orchfs_compute_offset(fctx, layer, head, block_idx);
    ssize_t nread = full_pread(fctx->fd, data, size, (off_t)offset);

    if (nread != (ssize_t)size) {
        LOG_ERR("pread: expected %zu got %zd (offset=%ld, errno=%s)",
                size, nread, (long)offset, strerror(errno));
        return ORCHKV_ERR_IO;
    }
    return ORCHKV_OK;
}

#endif /* ORCHKV_HAS_ORCHFS */

/* ========================================================================
 *  Thread-safe stats update (shared by both paths)
 * ======================================================================== */

void orchfs_tier_record_write(orchfs_tier_t *t, size_t bytes)
{
    if (!t) return;
    pthread_mutex_lock(&t->lock);
    t->total_writes++;
    t->bytes_written += bytes;
    pthread_mutex_unlock(&t->lock);
}

void orchfs_tier_record_read(orchfs_tier_t *t, size_t bytes)
{
    if (!t) return;
    pthread_mutex_lock(&t->lock);
    t->total_reads++;
    t->bytes_read += bytes;
    pthread_mutex_unlock(&t->lock);
}
