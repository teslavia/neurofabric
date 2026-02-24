/**
 * @file runtime_spec_test.cpp
 * @brief Phase 43 — NeuralOSRuntime with SpecEngine tests
 */

#include "neuralOS/kernel/NeuralOSRuntime.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== NeuralOSRuntime SpecEngine Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    neuralOS::kernel::RuntimeConfig cfg;
    cfg.enable_spec = true;
    cfg.spec_cfg.tree_width = 2;
    cfg.spec_cfg.max_depth = 4;
    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched, cfg);

    CHECK(runtime.spec_engine() != nullptr, "spec_engine enabled");

    auto* spec = runtime.spec_engine();
    auto* root = spec->tree().root.get();
    CHECK(root != nullptr, "tree root exists");

    /* Branch from root */
    auto children = spec->branch(root, {42, 43}, {0.9f, 0.8f});
    CHECK(children.size() == 2, "2 branches created");
    CHECK(children[0]->token_id == 42, "branch 0 token");
    CHECK(children[1]->token_id == 43, "branch 1 token");

    /* Verify child1, leave child2 unverified */
    children[0]->verified = true;
    children[1]->verified = false;

    /* Rollback unverified */
    spec->rollback();
    CHECK(root->num_children() <= 1, "unverified branch pruned");

    /* Stats reflect freed sequences */
    CHECK(runtime.stats().total_submitted == 0, "no requests submitted");

    printf("PASS: all NeuralOSRuntime SpecEngine tests passed\n");
    return 0;
}
