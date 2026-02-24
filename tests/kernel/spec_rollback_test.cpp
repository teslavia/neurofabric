/**
 * @file spec_rollback_test.cpp
 * @brief Phase 36.4 — SpecEngine rollback + checkpoint/restore tests
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
    printf("=== SpecEngine Rollback Test ===\n");

    nf::SpeculativeConfig cfg;
    cfg.tree_width = 3;
    cfg.max_depth = 6;

    neuralOS::L2::SpecEngine engine(cfg);
    auto& tree = engine.tree();

    /* Build a tree:
     *   root -> [10] -> [11] -> [12]
     *        -> [20] -> [21]
     *        -> [30]
     */
    auto b1 = engine.branch(tree.root.get(), {10, 20, 30});
    CHECK(b1.size() == 3, "3 branches");

    auto b1_1 = engine.branch(b1[0], {11});
    auto b1_1_1 = engine.branch(b1_1[0], {12});
    auto b2_1 = engine.branch(b1[1], {21});

    uint32_t total_before = tree.total_nodes;
    CHECK(total_before == 7, "7 nodes total");

    /* Verify path [10, 11] — marks root, 10, 11 as verified */
    int32_t target[] = {10, 11};
    auto vr = engine.verify(tree.root.get(), target, 2);
    CHECK(vr.accepted_count == 2, "accepted 2");

    /* Rollback: prune unverified branches */
    uint32_t pruned = engine.rollback();
    CHECK(pruned > 0, "pruned some nodes");

    /* After rollback: only verified path remains (root -> 10 -> 11)
     * plus 11's child 12 which was unverified → pruned */
    /* Branches 20, 21, 30 should be gone */
    CHECK(tree.root->num_children() == 1, "only verified branch remains");
    CHECK(tree.root->children[0]->token_id == 10, "branch 10 kept");

    /* Test checkpoint */
    auto cp = engine.checkpoint(0, 100, 5);
    CHECK(cp.seq_idx == 0, "checkpoint seq_idx");
    CHECK(cp.num_tokens == 100, "checkpoint num_tokens");
    CHECK(engine.num_checkpoints() == 1, "1 checkpoint");

    /* Test restore */
    neuralOS::L2::SpecCheckpoint restored;
    CHECK(engine.restore(&restored), "restore succeeded");
    CHECK(restored.seq_idx == 0, "restored seq_idx");
    CHECK(restored.num_tokens == 100, "restored num_tokens");
    CHECK(engine.num_checkpoints() == 0, "0 checkpoints after restore");

    /* Tree should be reset after restore */
    CHECK(tree.root->children.empty(), "tree reset after restore");
    CHECK(tree.total_nodes == 1, "only root remains");

    /* Test restore with no checkpoints */
    neuralOS::L2::SpecCheckpoint empty;
    CHECK(!engine.restore(&empty), "restore fails with no checkpoints");

    /* Test multiple checkpoints (LIFO) */
    engine.checkpoint(1, 200, 10);
    engine.checkpoint(2, 300, 15);
    CHECK(engine.num_checkpoints() == 2, "2 checkpoints");

    neuralOS::L2::SpecCheckpoint cp2;
    engine.restore(&cp2);
    CHECK(cp2.seq_idx == 2, "LIFO: last checkpoint first");
    engine.restore(&cp2);
    CHECK(cp2.seq_idx == 1, "LIFO: second checkpoint");

    printf("PASS: all SpecEngine rollback tests passed\n");
    return 0;
}
