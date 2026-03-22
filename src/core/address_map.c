#include "address_map.h"
#include <stdlib.h>
#include <string.h>

/* FNV-1a inspired hash – fast for uint64 keys */
static inline size_t hash_u64(uint64_t key, size_t cap)
{
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (size_t)(key & (cap - 1));   /* cap is always power-of-2 */
}

static size_t next_pow2(size_t n)
{
    n--;
    n |= n >> 1;  n |= n >> 2;
    n |= n >> 4;  n |= n >> 8;
    n |= n >> 16; n |= n >> 32;
    return n + 1;
}

int address_map_init(address_map_t *map, size_t initial_capacity)
{
    if (initial_capacity < 16)
        initial_capacity = 16;
    initial_capacity = next_pow2(initial_capacity);

    map->buckets = (addrmap_entry_t *)malloc(initial_capacity * sizeof(addrmap_entry_t));
    if (!map->buckets)
        return ORCHKV_ERR_OOM;

    for (size_t i = 0; i < initial_capacity; i++) {
        map->buckets[i].key   = ADDRMAP_EMPTY;
        map->buckets[i].value = NULL;
    }

    map->capacity   = initial_capacity;
    map->count      = 0;
    map->tombstones = 0;
    pthread_rwlock_init(&map->lock, NULL);
    return ORCHKV_OK;
}

void address_map_destroy(address_map_t *map)
{
    pthread_rwlock_destroy(&map->lock);
    free(map->buckets);
    map->buckets  = NULL;
    map->capacity = 0;
    map->count    = 0;
}

/* Internal: resize and rehash.  Caller must hold wrlock. */
static int addrmap_resize(address_map_t *map, size_t new_cap)
{
    addrmap_entry_t *old = map->buckets;
    size_t old_cap = map->capacity;

    addrmap_entry_t *new_buckets = (addrmap_entry_t *)malloc(new_cap * sizeof(addrmap_entry_t));
    if (!new_buckets)
        return ORCHKV_ERR_OOM;

    for (size_t i = 0; i < new_cap; i++) {
        new_buckets[i].key   = ADDRMAP_EMPTY;
        new_buckets[i].value = NULL;
    }

    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].key == ADDRMAP_EMPTY || old[i].key == ADDRMAP_DELETED)
            continue;
        size_t idx = hash_u64(old[i].key, new_cap);
        while (new_buckets[idx].key != ADDRMAP_EMPTY)
            idx = (idx + 1) & (new_cap - 1);
        new_buckets[idx] = old[i];
    }

    free(old);
    map->buckets    = new_buckets;
    map->capacity   = new_cap;
    map->tombstones = 0;
    return ORCHKV_OK;
}

int address_map_insert(address_map_t *map, kv_block_t *blk)
{
    pthread_rwlock_wrlock(&map->lock);

    /* grow if load factor > 0.75 (count + tombstones considered) */
    if ((map->count + map->tombstones) * 4 >= map->capacity * 3) {
        int rc = addrmap_resize(map, map->capacity * 2);
        if (rc != ORCHKV_OK) {
            pthread_rwlock_unlock(&map->lock);
            return rc;
        }
    }

    uint64_t key = blk->block_id;
    size_t idx = hash_u64(key, map->capacity);
    size_t first_deleted = SIZE_MAX;

    for (;;) {
        uint64_t k = map->buckets[idx].key;
        if (k == key) {
            pthread_rwlock_unlock(&map->lock);
            return ORCHKV_ERR_ALREADY;
        }
        if (k == ADDRMAP_EMPTY) {
            if (first_deleted != SIZE_MAX)
                idx = first_deleted;
            break;
        }
        if (k == ADDRMAP_DELETED && first_deleted == SIZE_MAX)
            first_deleted = idx;
        idx = (idx + 1) & (map->capacity - 1);
    }

    if (map->buckets[idx].key == ADDRMAP_DELETED)
        map->tombstones--;

    map->buckets[idx].key   = key;
    map->buckets[idx].value = blk;
    map->count++;

    pthread_rwlock_unlock(&map->lock);
    return ORCHKV_OK;
}

kv_block_t *address_map_lookup(address_map_t *map, uint64_t block_id)
{
    pthread_rwlock_rdlock(&map->lock);

    size_t idx = hash_u64(block_id, map->capacity);
    for (;;) {
        uint64_t k = map->buckets[idx].key;
        if (k == block_id) {
            kv_block_t *v = map->buckets[idx].value;
            pthread_rwlock_unlock(&map->lock);
            return v;
        }
        if (k == ADDRMAP_EMPTY) {
            pthread_rwlock_unlock(&map->lock);
            return NULL;
        }
        idx = (idx + 1) & (map->capacity - 1);
    }
}

int address_map_remove(address_map_t *map, uint64_t block_id)
{
    pthread_rwlock_wrlock(&map->lock);

    size_t idx = hash_u64(block_id, map->capacity);
    for (;;) {
        uint64_t k = map->buckets[idx].key;
        if (k == block_id) {
            map->buckets[idx].key   = ADDRMAP_DELETED;
            map->buckets[idx].value = NULL;
            map->count--;
            map->tombstones++;
            pthread_rwlock_unlock(&map->lock);
            return ORCHKV_OK;
        }
        if (k == ADDRMAP_EMPTY) {
            pthread_rwlock_unlock(&map->lock);
            return ORCHKV_ERR_NOT_FOUND;
        }
        idx = (idx + 1) & (map->capacity - 1);
    }
}
