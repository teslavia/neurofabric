/**
 * @file spec_vmmu_integration_test.cpp
 * @brief Phase 42A.2 — SpecEngine ↔ vMMU integration tests
 */

#include "neuralOS/kernel/SpecEngine.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== SpecEngine ↔ vMMU Integration Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);

    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);
    neuralOS::L2::vMMU vmmu(&kv, &hub);

    nf::SpeculativeConfig cfg;
    cfg.tree_width = 3;
    cfg.max_depth = 4;
    neuralOS::L2::SpecEngine engine(cfg);
    engine.set_vmmu(&vmmu, &kv);

    /* Build tree with seq_idx on branches */
    auto* root = engine.tree().root.get();
    auto branches = engine.branch(root, {10, 20, 30});
    CHECK(branches.size() == 3, "3 branches created");

    /* Assign seq_idx to branches (simulate KV allocation) */
    for (auto* b : branches) {
        uint32_t seq = kv.alloc_sequence();
        b->seq_idx = seq;
        /* Allocate a block for each */
        kv.sequences[seq].block_table[0] = kv.allocator.alloc_block();
        kv.sequences[seq].num_logical_blocks = 1;
    }

    /* Verify only branch 0 */
    int32_t target[] = {10};
    engine.verify(root, target, 1);

    /* Test 1: Rollback frees unverified sequences */
    uint32_t used_before = kv.allocator.num_used();
    uint32_t pruned = engine.rollback();
    CHECK(pruned == 2, "2 unverified nodes pruned");
    CHECK(engine.freed_sequences() == 2, "2 sequences freed via KV cleanup");

    /* Test 2: SpecEngine without vMMU — no crash */
    neuralOS::L2::SpecEngine engine2(cfg);
    auto* root2 = engine2.tree().root.get();
    auto b2 = engine2.branch(root2, {10, 20});
    for (auto* b : b2) b->seq_idx = 42;
    engine2.rollback();
    CHECK(engine2.freed_sequences() == 0, "no freed sequences without vMMU");

    printf("PASS: all SpecEngine ↔ vMMU integration tests passed\n");
    return 0;
}
