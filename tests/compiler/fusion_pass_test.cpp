/**
 * @file fusion_pass_test.cpp
 * @brief Phase 37.2 — Fusion pass tests
 */

#include "neuralOS/compiler/fusion_pass.hpp"
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
    printf("=== Fusion Pass Test ===\n");

    NfirHighGraph graph;

    /* Build: dequant → matmul → silu → element_mul */
    NfirHighOp dq; dq.kind = HighOpKind::DEQUANT;
    NfirHighOp mm; mm.kind = HighOpKind::MATMUL;
    NfirHighOp si; si.kind = HighOpKind::SILU;
    NfirHighOp em; em.kind = HighOpKind::ELEMENT_MUL;

    uint32_t dq_id = graph.add_op(dq);
    uint32_t mm_id = graph.add_op(mm);
    uint32_t si_id = graph.add_op(si);
    uint32_t em_id = graph.add_op(em);

    graph.add_edge(dq_id, mm_id);
    graph.add_edge(mm_id, si_id);
    graph.add_edge(si_id, em_id);

    FusionPass fuser;
    CHECK(fuser.num_patterns() == 3, "3 built-in patterns");

    uint32_t count = fuser.run(&graph);
    CHECK(count == 2, "2 fusions: dequant+matmul, silu+element_mul");

    /* Custom pattern */
    fuser.add_pattern(HighOpKind::LAYER_NORM, HighOpKind::ROPE,
                      "fused_ln_rope", 1.2f);
    CHECK(fuser.num_patterns() == 4, "4 patterns after custom add");

    /* Null graph */
    CHECK(fuser.run(nullptr) == 0, "null graph returns 0");

    /* No fusible ops */
    NfirHighGraph graph2;
    NfirHighOp softmax; softmax.kind = HighOpKind::SOFTMAX;
    NfirHighOp embedding; embedding.kind = HighOpKind::EMBEDDING;
    graph2.add_op(softmax);
    graph2.add_op(embedding);
    graph2.add_edge(0, 1);
    CHECK(fuser.run(&graph2) == 0, "no fusions for softmax→embedding");

    printf("PASS: all fusion pass tests passed\n");
    return 0;
}
