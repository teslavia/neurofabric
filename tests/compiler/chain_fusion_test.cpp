/**
 * @file chain_fusion_test.cpp
 * @brief Phase 42B.3 — Chain fusion (multi-op) tests
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

using namespace neuralOS::L1;

int main() {
    printf("=== Chain Fusion Test ===\n");

    /* Test 1: 4-op FFN chain → fused to 1 */
    {
        NfirHighGraph g;
        for (uint32_t i = 0; i < 5; ++i) {
            NfirTensorRef t; t.id = i; g.add_tensor(t);
        }

        NfirHighOp op0; op0.kind = HighOpKind::RMS_NORM;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::LINEAR;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);

        NfirHighOp op2; op2.kind = HighOpKind::SILU;
        op2.input_ids = {2}; op2.output_ids = {3};
        g.add_op(op2);

        NfirHighOp op3; op3.kind = HighOpKind::ELEMENT_MUL;
        op3.input_ids = {3}; op3.output_ids = {4};
        g.add_op(op3);

        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);

        FusionPass fp;
        uint32_t fused = fp.run(&g);
        CHECK(fused >= 1, "FFN chain fused");
        CHECK(g.fusion_candidates.size() >= 1, "at least 1 fusion candidate");
        CHECK(g.fusion_candidates[0].fused_name == "fused_ffn_block",
              "fused name is fused_ffn_block");
    }

    /* Test 2: 3-op attention chain → fused to 1 */
    {
        NfirHighGraph g;
        for (uint32_t i = 0; i < 4; ++i) {
            NfirTensorRef t; t.id = i; g.add_tensor(t);
        }

        NfirHighOp op0; op0.kind = HighOpKind::MATMUL;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::SOFTMAX;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);

        NfirHighOp op2; op2.kind = HighOpKind::MATMUL;
        op2.input_ids = {2}; op2.output_ids = {3};
        g.add_op(op2);

        g.add_edge(0, 1);
        g.add_edge(1, 2);

        FusionPass fp;
        uint32_t fused = fp.run(&g);
        CHECK(fused >= 1, "attention chain fused");
        bool found_attn = false;
        for (auto& fc : g.fusion_candidates)
            if (fc.fused_name == "fused_attention_core") found_attn = true;
        CHECK(found_attn, "fused_attention_core found");
    }

    /* Test 3: Partial chain match → pair fusion fallback */
    {
        NfirHighGraph g;
        for (uint32_t i = 0; i < 3; ++i) {
            NfirTensorRef t; t.id = i; g.add_tensor(t);
        }

        NfirHighOp op0; op0.kind = HighOpKind::SILU;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::ELEMENT_MUL;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);

        g.add_edge(0, 1);

        FusionPass fp;
        uint32_t fused = fp.run(&g);
        CHECK(fused == 1, "pair fusion fallback");
        CHECK(g.fusion_candidates[0].fused_name == "fused_swiglu",
              "fused_swiglu from pair pattern");
    }

    /* Test 4: Custom chain pattern */
    {
        NfirHighGraph g;
        for (uint32_t i = 0; i < 4; ++i) {
            NfirTensorRef t; t.id = i; g.add_tensor(t);
        }

        NfirHighOp op0; op0.kind = HighOpKind::LAYER_NORM;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::GELU;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);

        NfirHighOp op2; op2.kind = HighOpKind::LINEAR;
        op2.input_ids = {2}; op2.output_ids = {3};
        g.add_op(op2);

        g.add_edge(0, 1);
        g.add_edge(1, 2);

        FusionPass fp;
        fp.add_chain_pattern(
            {HighOpKind::LAYER_NORM, HighOpKind::GELU, HighOpKind::LINEAR},
            "fused_custom_block", 1.5f);
        uint32_t fused = fp.run(&g);
        CHECK(fused >= 1, "custom chain fused");
        bool found = false;
        for (auto& fc : g.fusion_candidates)
            if (fc.fused_name == "fused_custom_block") found = true;
        CHECK(found, "custom chain pattern matched");
    }

    printf("PASS: all chain fusion tests passed\n");
    return 0;
}
