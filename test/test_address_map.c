#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/address_map.h"

static void test_basic_crud(void)
{
    kv_block_reset_id_counter();
    address_map_t map;
    assert(address_map_init(&map, 16) == ORCHKV_OK);
    assert(address_map_count(&map) == 0);

    kv_block_t a, b, c;
    kv_block_init(&a, 1, 0, 0, 0, 64);
    kv_block_init(&b, 1, 0, 1, 0, 64);
    kv_block_init(&c, 2, 1, 0, 64, 32);

    /* insert */
    assert(address_map_insert(&map, &a) == ORCHKV_OK);
    assert(address_map_insert(&map, &b) == ORCHKV_OK);
    assert(address_map_insert(&map, &c) == ORCHKV_OK);
    assert(address_map_count(&map) == 3);

    /* duplicate insert */
    assert(address_map_insert(&map, &a) == ORCHKV_ERR_ALREADY);
    assert(address_map_count(&map) == 3);

    /* lookup */
    assert(address_map_lookup(&map, a.block_id) == &a);
    assert(address_map_lookup(&map, b.block_id) == &b);
    assert(address_map_lookup(&map, c.block_id) == &c);
    assert(address_map_lookup(&map, 999) == NULL);

    /* remove */
    assert(address_map_remove(&map, b.block_id) == ORCHKV_OK);
    assert(address_map_count(&map) == 2);
    assert(address_map_lookup(&map, b.block_id) == NULL);

    /* remove non-existent */
    assert(address_map_remove(&map, 999) == ORCHKV_ERR_NOT_FOUND);

    /* remaining still accessible */
    assert(address_map_lookup(&map, a.block_id) == &a);
    assert(address_map_lookup(&map, c.block_id) == &c);

    kv_block_destroy(&a);
    kv_block_destroy(&b);
    kv_block_destroy(&c);
    address_map_destroy(&map);
    printf("  [PASS] basic_crud\n");
}

static void test_resize(void)
{
    kv_block_reset_id_counter();
    address_map_t map;
    assert(address_map_init(&map, 16) == ORCHKV_OK);

    #define N 200
    kv_block_t blocks[N];
    for (int i = 0; i < N; i++) {
        kv_block_init(&blocks[i], 1, 0, 0, (uint32_t)i * 64, 64);
        assert(address_map_insert(&map, &blocks[i]) == ORCHKV_OK);
    }
    assert(address_map_count(&map) == N);
    /* capacity must have grown beyond initial 16 */
    assert(map.capacity >= N);

    /* verify all still findable after resizes */
    for (int i = 0; i < N; i++)
        assert(address_map_lookup(&map, blocks[i].block_id) == &blocks[i]);

    for (int i = 0; i < N; i++)
        kv_block_destroy(&blocks[i]);
    address_map_destroy(&map);
    #undef N
    printf("  [PASS] resize\n");
}

static void test_remove_and_reinsert(void)
{
    kv_block_reset_id_counter();
    address_map_t map;
    assert(address_map_init(&map, 16) == ORCHKV_OK);

    #define M 50
    kv_block_t blocks[M];
    for (int i = 0; i < M; i++) {
        kv_block_init(&blocks[i], 1, 0, 0, (uint32_t)i * 64, 64);
        address_map_insert(&map, &blocks[i]);
    }

    /* remove even-indexed blocks */
    for (int i = 0; i < M; i += 2)
        assert(address_map_remove(&map, blocks[i].block_id) == ORCHKV_OK);
    assert(address_map_count(&map) == M / 2);

    /* odd still present */
    for (int i = 1; i < M; i += 2)
        assert(address_map_lookup(&map, blocks[i].block_id) == &blocks[i]);

    /* re-insert even (they keep their old block_id) */
    for (int i = 0; i < M; i += 2)
        assert(address_map_insert(&map, &blocks[i]) == ORCHKV_OK);
    assert(address_map_count(&map) == M);

    for (int i = 0; i < M; i++)
        kv_block_destroy(&blocks[i]);
    address_map_destroy(&map);
    #undef M
    printf("  [PASS] remove_and_reinsert\n");
}

static void test_stress(void)
{
    kv_block_reset_id_counter();
    address_map_t map;
    assert(address_map_init(&map, 64) == ORCHKV_OK);

    #define S 100000
    kv_block_t *blocks = (kv_block_t *)calloc(S, sizeof(kv_block_t));
    assert(blocks);

    for (int i = 0; i < S; i++) {
        kv_block_init(&blocks[i], 1, (uint16_t)(i % 32), (uint16_t)(i % 8),
                      (uint32_t)i * 64, 64);
        assert(address_map_insert(&map, &blocks[i]) == ORCHKV_OK);
    }
    assert(address_map_count(&map) == S);

    /* spot-check every 1000th */
    for (int i = 0; i < S; i += 1000)
        assert(address_map_lookup(&map, blocks[i].block_id) == &blocks[i]);

    /* remove first half */
    for (int i = 0; i < S / 2; i++)
        assert(address_map_remove(&map, blocks[i].block_id) == ORCHKV_OK);
    assert(address_map_count(&map) == S / 2);

    /* second half still intact */
    for (int i = S / 2; i < S; i += 1000)
        assert(address_map_lookup(&map, blocks[i].block_id) == &blocks[i]);

    for (int i = 0; i < S; i++)
        kv_block_destroy(&blocks[i]);
    free(blocks);
    address_map_destroy(&map);
    #undef S
    printf("  [PASS] stress (100000 blocks)\n");
}

int main(void)
{
    printf("=== test_address_map ===\n");
    test_basic_crud();
    test_resize();
    test_remove_and_reinsert();
    test_stress();
    printf("=== ALL PASSED ===\n");
    return 0;
}
