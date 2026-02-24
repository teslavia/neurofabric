/**
 * @file spec_tree_test.cpp
 * @brief Phase 36.4 — SpecEngine tree search + branch + verify tests
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
    printf("=== SpecEngine Tree Test ===\n");

    nf::SpeculativeConfig cfg;
    cfg.tree_width = 3;
    cfg.max_depth = 4;
    cfg.num_speculative = 4;

    neuralOS::L2::SpecEngine engine(cfg);
    auto& tree = engine.tree();

    /* Test 1: Tree starts with root */
    CHECK(tree.root != nullptr, "root exists");
    CHECK(tree.total_nodes == 1, "starts with 1 node");

    /* Test 2: Branch from root with 3 candidates */
    auto leaves = engine.branch(tree.root.get(), {10, 20, 30}, {-0.1f, -0.2f, -0.5f});
    CHECK(leaves.size() == 3, "branched into 3");
    CHECK(tree.total_nodes == 4, "4 nodes total");
    CHECK(leaves[0]->token_id == 10, "first branch token=10");
    CHECK(leaves[1]->token_id == 20, "second branch token=20");
    CHECK(leaves[2]->token_id == 30, "third branch token=30");

    /* Test 3: Branch deeper from first leaf */
    auto deeper = engine.branch(leaves[0], {11, 12});
    CHECK(deeper.size() == 2, "branched deeper into 2");
    CHECK(deeper[0]->depth == 2, "depth is 2");

    /* Test 4: Branch even deeper */
    auto deep3 = engine.branch(deeper[0], {111});
    CHECK(deep3.size() == 1, "one more level");
    CHECK(deep3[0]->depth == 3, "depth is 3");

    /* Test 5: Max depth enforcement */
    auto deep4 = engine.branch(deep3[0], {1111});
    CHECK(deep4.size() == 1, "depth 4 allowed");
    auto deep5 = engine.branch(deep4[0], {11111});
    CHECK(deep5.empty(), "depth 5 blocked by max_depth=4");

    /* Test 6: Max width enforcement */
    auto wide = engine.branch(tree.root.get(), {40});
    CHECK(wide.empty(), "root already has 3 children, width=3 blocks more");

    /* Test 7: Path to node */
    auto path = neuralOS::L2::SpecTree::path_to(deep3[0]);
    CHECK(path.size() == 3, "path length 3");
    CHECK(path[0] == 10 && path[1] == 11 && path[2] == 111, "correct path");

    /* Test 8: Collect leaves */
    std::vector<neuralOS::L2::SpecNode*> all_leaves;
    tree.collect_leaves(tree.root.get(), all_leaves);
    CHECK(all_leaves.size() > 0, "has leaves");

    /* Test 9: Verify against target */
    int32_t target[] = {10, 11, 111, 1111};
    auto result = engine.verify(tree.root.get(), target, 4);
    CHECK(result.accepted_count == 4, "all 4 tokens accepted");
    CHECK(result.accepted_tokens.size() == 4, "4 accepted tokens");

    /* Test 10: Partial verify */
    int32_t target2[] = {10, 11, 999};
    auto result2 = engine.verify(tree.root.get(), target2, 3);
    CHECK(result2.accepted_count == 2, "2 tokens accepted before mismatch");

    printf("PASS: all SpecEngine tree tests passed\n");
    return 0;
}
