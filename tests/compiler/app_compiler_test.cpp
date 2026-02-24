/**
 * @file app_compiler_test.cpp
 * @brief Phase 43D.1 — CompilerPipeline in app integration test
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

/* Build a mock transformer-like NfirHighGraph */
static NfirHighGraph build_transformer_graph() {
    NfirHighGraph g;

    /* Tensors */
    for (int i = 0; i < 12; ++i) {
        NfirTensorRef t;
        t.ndim = 2;
        t.shape[0] = 32; t.shape[1] = 128;
        t.dtype = 0;
        g.add_tensor(t);
    }

    /* Layer 0: embedding → rms_norm → linear → silu → element_mul */
    NfirHighOp emb; emb.kind = HighOpKind::EMBEDDING;
    emb.input_ids = {0}; emb.output_ids = {1};
    g.add_op(emb);

    NfirHighOp norm; norm.kind = HighOpKind::RMS_NORM;
    norm.input_ids = {1}; norm.output_ids = {2};
    g.add_op(norm);

    NfirHighOp lin; lin.kind = HighOpKind::LINEAR;
    lin.input_ids = {2, 2}; lin.output_ids = {3};
    g.add_op(lin);

    NfirHighOp silu; silu.kind = HighOpKind::SILU;
    silu.input_ids = {3}; silu.output_ids = {4};
    g.add_op(silu);

    NfirHighOp emul; emul.kind = HighOpKind::ELEMENT_MUL;
    emul.input_ids = {4, 3}; emul.output_ids = {5};
    g.add_op(emul);

    /* Layer 1: attention block */
    NfirHighOp attn; attn.kind = HighOpKind::ATTENTION;
    attn.input_ids = {5}; attn.output_ids = {6};
    g.add_op(attn);

    NfirHighOp norm2; norm2.kind = HighOpKind::RMS_NORM;
    norm2.input_ids = {6}; norm2.output_ids = {7};
    g.add_op(norm2);

    NfirHighOp softmax; softmax.kind = HighOpKind::SOFTMAX;
    softmax.input_ids = {7}; softmax.output_ids = {8};
    g.add_op(softmax);

    /* Dead ops (should be removed by DCE) */
    NfirHighOp dead1; dead1.kind = HighOpKind::GELU;
    dead1.input_ids = {0}; dead1.output_ids = {9};
    g.add_op(dead1);

    NfirHighOp dead2; dead2.kind = HighOpKind::LAYER_NORM;
    dead2.input_ids = {0}; dead2.output_ids = {10};
    g.add_op(dead2);

    /* Edges */
    g.add_edge(0, 1);  /* emb → norm */
    g.add_edge(1, 2);  /* norm → linear */
    g.add_edge(2, 3);  /* linear → silu */
    g.add_edge(3, 4);  /* silu → emul */
    g.add_edge(4, 5);  /* emul → attn */
    g.add_edge(5, 6);  /* attn → norm2 */
    g.add_edge(6, 7);  /* norm2 → softmax */

    g.output_tensor_ids = {8};
    return g;
}

int main() {
    printf("=== App Compiler Integration Test ===\n");

    auto g = build_transformer_graph();
    uint32_t ops_before = g.num_ops();

    CompilerPipeline pipeline;
    auto result = pipeline.run(&g);

    printf("  ops_before=%u ops_after=%u\n", ops_before, g.num_ops());
    printf("  removed=%u merged=%u shapes=%u fusions=%u\n",
           result.ops_removed, result.ops_merged,
           result.shapes_inferred, result.fusions_found);

    CHECK(result.ops_removed >= 2, "DCE removed dead ops");
    CHECK(result.shapes_inferred > 0, "shapes inferred");
    CHECK(g.num_ops() < ops_before, "graph optimized");

    /* Verify all passes produced stats */
    uint32_t total_effect = result.ops_removed + result.ops_merged
                          + result.shapes_inferred + result.fusions_found;
    CHECK(total_effect > 0, "pipeline had effect");

    printf("PASS: all app compiler integration tests passed\n");
    return 0;
}
