/**
 * @file model_config_test.cpp
 * @brief Phase 32 Step 1: ModelConfig + BlockAllocator + SequenceKV tests
 *
 * 4 sub-tests:
 *   1. ModelConfig_defaults
 *   2. BlockAllocator_alloc_free
 *   3. BlockAllocator_cow_refcount
 *   4. SequenceKV_block_table
 */

#include "model/model_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

static void test_model_config_defaults() {
    nf::ModelConfig cfg{};
    CHECK(cfg.engine == nullptr);
    CHECK(cfg.prov == nullptr);
    CHECK(cfg.vt == nullptr);
    CHECK(cfg.mem_vt == nullptr);
    CHECK(cfg.model == nullptr);
    CHECK(cfg.max_seq == 512);
    CHECK(cfg.max_prefill_seq == 0);
    CHECK(cfg.use_fp16 == false);
    CHECK(cfg.use_paged_kv == false);
    CHECK(cfg.kv_block_size == 16);
    CHECK(cfg.num_kv_blocks == 0);
    CHECK(cfg.kv_cfg == nullptr);
    CHECK(cfg.arch_override == nullptr);
    std::printf("  [PASS] ModelConfig_defaults\n");
}

static void test_block_allocator_alloc_free() {
    nf::BlockAllocator alloc;
    alloc.init(8, 16);

    CHECK(alloc.num_blocks == 8);
    CHECK(alloc.num_free() == 8);
    CHECK(alloc.num_used() == 0);

    /* Alloc all blocks */
    uint32_t blocks[8];
    for (int i = 0; i < 8; ++i) {
        blocks[i] = alloc.alloc_block();
        CHECK(blocks[i] != nf::NF_PAGED_INVALID_BLOCK);
        CHECK(alloc.block_meta[blocks[i]].ref_count == 1);
    }
    CHECK(alloc.num_free() == 0);
    CHECK(alloc.num_used() == 8);

    /* OOM */
    CHECK(alloc.alloc_block() == nf::NF_PAGED_INVALID_BLOCK);

    /* Free all */
    for (int i = 0; i < 8; ++i)
        alloc.free_block(blocks[i]);
    CHECK(alloc.num_free() == 8);
    CHECK(alloc.num_used() == 0);

    /* Re-alloc should work */
    uint32_t b = alloc.alloc_block();
    CHECK(b != nf::NF_PAGED_INVALID_BLOCK);
    CHECK(alloc.num_free() == 7);

    std::printf("  [PASS] BlockAllocator_alloc_free\n");
}

static void test_block_allocator_cow_refcount() {
    nf::BlockAllocator alloc;
    alloc.init(4, 16);

    uint32_t b0 = alloc.alloc_block();
    CHECK(alloc.block_meta[b0].ref_count == 1);

    /* Add a CoW reference */
    alloc.ref(b0);
    CHECK(alloc.block_meta[b0].ref_count == 2);

    /* First unref: decrement but don't free */
    alloc.unref(b0);
    CHECK(alloc.block_meta[b0].ref_count == 1);
    CHECK(alloc.num_free() == 3);  /* still allocated */

    /* Second unref: ref_count → 0, freed */
    alloc.unref(b0);
    CHECK(alloc.block_meta[b0].ref_count == 0);
    CHECK(alloc.num_free() == 4);  /* freed */

    std::printf("  [PASS] BlockAllocator_cow_refcount\n");
}

static void test_sequence_kv_block_table() {
    nf::PagedKVCache cache;
    cache.init(32, 4, 2, 4, 64);  /* 32 blocks, bs=4, 2 layers, 4 kv_heads, dim=64 */

    uint32_t seq = cache.alloc_sequence();
    CHECK(seq != UINT32_MAX);
    CHECK(cache.sequences[seq].active);
    CHECK(cache.sequences[seq].num_tokens == 0);

    /* Append 10 tokens — should allocate ceil(10/4) = 3 blocks */
    for (int i = 0; i < 10; ++i) {
        uint32_t offset = cache.append(seq);
        CHECK(offset != UINT32_MAX);
    }
    CHECK(cache.sequences[seq].num_tokens == 10);
    CHECK(cache.sequences[seq].num_logical_blocks == 3);

    /* Verify block table has valid physical blocks */
    for (uint32_t i = 0; i < 3; ++i)
        CHECK(cache.sequences[seq].block_table[i] != nf::NF_PAGED_INVALID_BLOCK);

    /* Verify fill counts */
    auto& alloc = cache.allocator;
    CHECK(alloc.block_meta[cache.sequences[seq].block_table[0]].num_filled == 4);
    CHECK(alloc.block_meta[cache.sequences[seq].block_table[1]].num_filled == 4);
    CHECK(alloc.block_meta[cache.sequences[seq].block_table[2]].num_filled == 2);

    /* fill_block_table for GPU upload */
    uint32_t gpu_table[8];
    cache.fill_block_table(seq, gpu_table, 8);
    for (uint32_t i = 0; i < 3; ++i)
        CHECK(gpu_table[i] == cache.sequences[seq].block_table[i]);
    for (uint32_t i = 3; i < 8; ++i)
        CHECK(gpu_table[i] == nf::NF_PAGED_INVALID_BLOCK);

    std::printf("  [PASS] SequenceKV_block_table\n");
}

int main() {
    std::printf("=== model_config_test ===\n");
    test_model_config_defaults();
    test_block_allocator_alloc_free();
    test_block_allocator_cow_refcount();
    test_sequence_kv_block_table();
    std::printf("All 4 model_config tests passed.\n");
    return 0;
}
