/**
 * @file paged_kv_test.cpp
 * @brief Phase 32 Step 1: PagedKVCache integration tests
 *
 * 3 sub-tests:
 *   1. PagedKVCache_init_destroy
 *   2. PagedKVCache_append_sequence
 *   3. PagedKVCache_multi_sequence
 */

#include "model_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

static void test_init_destroy() {
    nf::PagedKVCache cache;
    cache.init(64, 16, 4, 8, 128);  /* 64 blocks, bs=16, 4 layers, 8 kv_heads, dim=128 */

    CHECK(cache.allocator.num_blocks == 64);
    CHECK(cache.allocator.block_size == 16);
    CHECK(cache.allocator.num_free() == 64);
    CHECK(cache.n_layers == 4);
    CHECK(cache.n_kv_heads == 8);
    CHECK(cache.head_dim == 128);
    CHECK(cache.block_size == 16);
    CHECK(cache.num_sequences == 0);

    /* All sequence slots should be inactive */
    for (uint32_t i = 0; i < nf::NF_PAGED_MAX_SEQUENCES; ++i)
        CHECK(!cache.sequences[i].active);

    std::printf("  [PASS] PagedKVCache_init_destroy\n");
}

static void test_append_sequence() {
    nf::PagedKVCache cache;
    cache.init(32, 8, 2, 4, 64);  /* bs=8 */

    uint32_t seq = cache.alloc_sequence();
    CHECK(seq == 0);

    /* Append 100 tokens — should need ceil(100/8) = 13 blocks */
    for (int i = 0; i < 100; ++i) {
        uint32_t offset = cache.append(seq);
        CHECK(offset != UINT32_MAX);

        /* Verify offset is consistent: phys_block * block_size + offset_in_block */
        uint32_t logical = (uint32_t)i / 8;
        uint32_t in_block = (uint32_t)i % 8;
        uint32_t phys = cache.sequences[seq].block_table[logical];
        CHECK(offset == phys * 8 + in_block);
    }

    CHECK(cache.sequences[seq].num_tokens == 100);
    CHECK(cache.sequences[seq].num_logical_blocks == 13);
    CHECK(cache.allocator.num_used() == 13);
    CHECK(cache.allocator.num_free() == 32 - 13);

    /* Truncate to 50 tokens — should free blocks 7..12 (keep 0..6) */
    cache.truncate(seq, 50);
    CHECK(cache.sequences[seq].num_tokens == 50);
    CHECK(cache.sequences[seq].num_logical_blocks == 7);  /* ceil(50/8) */
    CHECK(cache.allocator.num_used() == 7);
    CHECK(cache.allocator.num_free() == 32 - 7);

    /* Last block should have fill=2 (50 % 8 = 2) */
    uint32_t last_phys = cache.sequences[seq].block_table[6];
    CHECK(cache.allocator.block_meta[last_phys].num_filled == 2);

    /* Free sequence */
    cache.free_sequence(seq);
    CHECK(cache.allocator.num_free() == 32);
    CHECK(!cache.sequences[seq].active);

    std::printf("  [PASS] PagedKVCache_append_sequence\n");
}

static void test_multi_sequence() {
    nf::PagedKVCache cache;
    cache.init(64, 4, 1, 2, 32);  /* bs=4, plenty of blocks */

    /* Allocate 4 sequences */
    uint32_t seqs[4];
    for (int i = 0; i < 4; ++i) {
        seqs[i] = cache.alloc_sequence();
        CHECK(seqs[i] != UINT32_MAX);
    }
    CHECK(cache.num_sequences == 4);

    /* Append different lengths: 8, 12, 4, 20 tokens */
    uint32_t lengths[] = {8, 12, 4, 20};
    for (int s = 0; s < 4; ++s) {
        for (uint32_t t = 0; t < lengths[s]; ++t) {
            uint32_t off = cache.append(seqs[s]);
            CHECK(off != UINT32_MAX);
        }
    }

    /* Verify isolation: each sequence has its own blocks */
    uint32_t expected_blocks[] = {2, 3, 1, 5};  /* ceil(len/4) */
    uint32_t total_blocks = 0;
    std::set<uint32_t> all_phys;

    for (int s = 0; s < 4; ++s) {
        CHECK(cache.sequences[seqs[s]].num_tokens == lengths[s]);
        CHECK(cache.sequences[seqs[s]].num_logical_blocks == expected_blocks[s]);
        total_blocks += expected_blocks[s];

        for (uint32_t b = 0; b < expected_blocks[s]; ++b) {
            uint32_t phys = cache.sequences[seqs[s]].block_table[b];
            CHECK(phys != nf::NF_PAGED_INVALID_BLOCK);
            /* No physical block shared between sequences */
            CHECK(all_phys.find(phys) == all_phys.end());
            all_phys.insert(phys);
        }
    }
    CHECK(cache.allocator.num_used() == total_blocks);

    /* Free seq 1, verify its blocks returned */
    uint32_t freed = expected_blocks[1];
    cache.free_sequence(seqs[1]);
    CHECK(cache.allocator.num_used() == total_blocks - freed);

    /* Remaining sequences unaffected */
    CHECK(cache.sequences[seqs[0]].num_tokens == 8);
    CHECK(cache.sequences[seqs[2]].num_tokens == 4);
    CHECK(cache.sequences[seqs[3]].num_tokens == 20);

    std::printf("  [PASS] PagedKVCache_multi_sequence\n");
}

int main() {
    std::printf("=== paged_kv_test ===\n");
    test_init_destroy();
    test_append_sequence();
    test_multi_sequence();
    std::printf("All 3 paged_kv tests passed.\n");
    return 0;
}
