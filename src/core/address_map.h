#ifndef ORCHKV_ADDRESS_MAP_H
#define ORCHKV_ADDRESS_MAP_H

#include "kv_types.h"
#include "kv_block.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 *  Address Map – open-addressing hash table mapping block_id → kv_block_t*.
 *
 *  Supports concurrent readers with a single-writer via rwlock.
 *  Auto-resizes at 75% load factor.
 * ======================================================================== */

#define ADDRMAP_EMPTY    UINT64_MAX   /* sentinel for empty bucket */
#define ADDRMAP_DELETED  (UINT64_MAX - 1)

typedef struct addrmap_entry {
    uint64_t     key;       /* block_id, or ADDRMAP_EMPTY / ADDRMAP_DELETED */
    kv_block_t  *value;
} addrmap_entry_t;

typedef struct address_map {
    addrmap_entry_t *buckets;
    size_t           capacity;
    size_t           count;         /* live entries */
    size_t           tombstones;    /* deleted entries (for load factor) */
    pthread_rwlock_t lock;
} address_map_t;

int  address_map_init(address_map_t *map, size_t initial_capacity);
void address_map_destroy(address_map_t *map);

/* Insert a block.  Returns ORCHKV_ERR_ALREADY if block_id already present. */
int  address_map_insert(address_map_t *map, kv_block_t *blk);

/* Lookup by block_id.  Returns NULL if not found.  Caller holds rdlock internally. */
kv_block_t *address_map_lookup(address_map_t *map, uint64_t block_id);

/* Remove by block_id.  Returns ORCHKV_ERR_NOT_FOUND if absent. */
int  address_map_remove(address_map_t *map, uint64_t block_id);

/* Current live entry count */
static inline size_t address_map_count(const address_map_t *map)
{
    return map->count;
}

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_ADDRESS_MAP_H */
