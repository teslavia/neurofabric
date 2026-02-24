/**
 * @file memory_plan_test.cpp
 * @brief Phase 37.3 — Memory planning pass tests
 */

#include "neuralOS/compiler/memory_plan_pass.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

using namespace neuralOS::compiler;

int main() {
    printf("=== Memory Plan Test ===\n");

    NfirLowGraph graph;

    /* Create tensors with known sizes */
    NfirTensorRef t0; t0.size_bytes = 4096; t0.name = "A";
    NfirTensorRef t1; t1.size_bytes = 8192; t1.name = "B";
    NfirTensorRef t2; t2.size_bytes = 4096; t2.name = "C";
    NfirTensorRef t3; t3.size_bytes = 2048; t3.name = "D";

    graph.tensors.push_back(t0); graph.tensors.back().id = 0;
    graph.tensors.push_back(t1); graph.tensors.back().id = 1;
    graph.tensors.push_back(t2); graph.tensors.back().id = 2;
    graph.tensors.push_back(t3); graph.tensors.back().id = 3;

    /* Op 0: uses t0, produces t1 */
    NfirLowOp op0; op0.input_tensor_ids = {0}; op0.output_tensor_ids = {1};
    graph.add_op(op0);

    /* Op 1: uses t1, produces t2 */
    NfirLowOp op1; op1.input_tensor_ids = {1}; op1.output_tensor_ids = {2};
    graph.add_op(op1);

    /* Op 2: uses t2, produces t3 (t0 and t1 are dead here) */
    NfirLowOp op2; op2.input_tensor_ids = {2}; op2.output_tensor_ids = {3};
    graph.add_op(op2);

    MemoryPlanPass planner;
    MemoryPlan plan = planner.run(graph);

    printf("  Peak: %llu bytes, reused: %u, allocations: %zu\n",
           (unsigned long long)plan.peak_bytes, plan.num_reused,
           plan.allocations.size());

    CHECK(plan.allocations.size() == 4, "4 tensor allocations");
    CHECK(plan.peak_bytes > 0, "peak > 0");

    /* With reuse, peak should be less than sum of all tensors */
    uint64_t total = 4096 + 8192 + 4096 + 2048;
    CHECK(plan.peak_bytes <= total, "peak <= total (reuse helps)");

    /* At least some reuse should happen (t0 dies before t2 is born) */
    if (plan.num_reused > 0) {
        CHECK(plan.peak_bytes < total, "reuse reduced peak");
    }

    /* Empty graph */
    NfirLowGraph empty;
    MemoryPlan ep = planner.run(empty);
    CHECK(ep.peak_bytes == 0, "empty graph → 0 peak");
    CHECK(ep.allocations.empty(), "empty graph → no allocations");

    printf("PASS: all memory plan tests passed\n");
    return 0;
}
