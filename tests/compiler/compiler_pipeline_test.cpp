/**
 * @file compiler_pipeline_test.cpp
 * @brief Phase 43A.2 — CompilerPipeline tests
 */

#include "neuralOS/compiler/compiler_pipeline.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

using namespace neuralOS::compiler;

/* Helper: build a graph with dead ops, duplicates, fusible chain */
static NfirHighGraph build_test_graph() {
    NfirHighGraph g;

    /* Tensors: t0(input), t1, t2, t3, t4(output), t5(dead) */
    for (int i = 0; i < 6; ++i) {
        NfirTensorRef t;
        t.ndim = 2;
        t.shape[0] = 4; t.shape[1] = 4;
        t.dtype = 0;
        g.add_tensor(t);
    }

    /* op0: DEQUANT t0 → t1 */
    NfirHighOp op0; op0.kind = HighOpKind::DEQUANT;
    op0.input_ids = {0}; op0.output_ids = {1};
    g.add_op(op0);

    /* op1: MATMUL t1 → t2 */
    NfirHighOp op1; op1.kind = HighOpKind::MATMUL;
    op1.input_ids = {1, 1}; op1.output_ids = {2};
    g.add_op(op1);

    /* op2: RMS_NORM t2 → t3 */
    NfirHighOp op2; op2.kind = HighOpKind::RMS_NORM;
    op2.input_ids = {2}; op2.output_ids = {3};
    g.add_op(op2);

    /* op3: SOFTMAX t3 → t4 (output) */
    NfirHighOp op3; op3.kind = HighOpKind::SOFTMAX;
    op3.input_ids = {3}; op3.output_ids = {4};
    g.add_op(op3);

    /* op4: DEAD — SILU t0 → t5 (output t5 not consumed) */
    NfirHighOp op4; op4.kind = HighOpKind::SILU;
    op4.input_ids = {0}; op4.output_ids = {5};
    g.add_op(op4);

    /* Edges */
    g.add_edge(0, 1); /* dequant → matmul */
    g.add_edge(1, 2); /* matmul → rms_norm */
    g.add_edge(2, 3); /* rms_norm → softmax */

    /* Mark output */
    g.output_tensor_ids = {4};

    return g;
}

int main() {
    printf("=== CompilerPipeline Test ===\n");

    /* Test 1: All passes enabled */
    {
        auto g = build_test_graph();
        uint32_t ops_before = g.num_ops();
        CompilerPipeline pipeline;
        auto result = pipeline.run(&g);

        CHECK(result.ops_removed >= 1, "DCE removed dead op");
        CHECK(result.shapes_inferred > 0, "shapes inferred");
        CHECK(result.fusions_found >= 1, "fusion found (dequant+matmul)");
        CHECK(g.num_ops() < ops_before, "total ops reduced");
        printf("  all passes: removed=%u merged=%u shapes=%u fusions=%u\n",
               result.ops_removed, result.ops_merged,
               result.shapes_inferred, result.fusions_found);
    }

    /* Test 2: Disable individual passes */
    {
        auto g = build_test_graph();
        CompilerPipeline::Config cfg;
        cfg.enable_dce = false;
        cfg.enable_fusion = false;
        CompilerPipeline pipeline(cfg);
        auto result = pipeline.run(&g);

        CHECK(result.ops_removed == 0, "DCE disabled");
        CHECK(result.fusions_found == 0, "fusion disabled");
        CHECK(result.shapes_inferred > 0, "shape still runs");
    }

    /* Test 3: Empty graph */
    {
        NfirHighGraph g;
        CompilerPipeline pipeline;
        auto result = pipeline.run(&g);
        CHECK(result.ops_removed == 0, "empty: no crash");
        CHECK(result.fusions_found == 0, "empty: no fusions");
    }

    /* Test 4: Null graph */
    {
        CompilerPipeline pipeline;
        auto result = pipeline.run(nullptr);
        CHECK(result.ops_removed == 0, "null: no crash");
    }

    printf("PASS: all CompilerPipeline tests passed\n");
    return 0;
}
