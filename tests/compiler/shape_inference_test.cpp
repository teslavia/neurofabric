/**
 * @file shape_inference_test.cpp
 * @brief Phase 42B.4 — Shape inference pass tests
 */

#include "neuralOS/compiler/shape_inference_pass.hpp"
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
    printf("=== Shape Inference Test ===\n");

    /* Test 1: MATMUL shape propagation [4,8] × [8,16] → [4,16] */
    {
        NfirHighGraph g;
        NfirTensorRef t0; t0.ndim = 2; t0.shape[0] = 4; t0.shape[1] = 8;
        g.add_tensor(t0);
        NfirTensorRef t1; t1.ndim = 2; t1.shape[0] = 8; t1.shape[1] = 16;
        g.add_tensor(t1);
        NfirTensorRef t2; t2.ndim = 0; /* unknown */
        g.add_tensor(t2);

        NfirHighOp op; op.kind = HighOpKind::MATMUL;
        op.input_ids = {0, 1}; op.output_ids = {2};
        g.add_op(op);

        ShapeInferencePass si;
        uint32_t inferred = si.run(&g);
        CHECK(inferred == 1, "1 tensor inferred");
        CHECK(g.tensors[2].ndim == 2, "output is 2D");
        CHECK(g.tensors[2].shape[0] == 4, "M = 4");
        CHECK(g.tensors[2].shape[1] == 16, "N = 16");
    }

    /* Test 2: Element-wise broadcast [4,1] + [1,8] → [4,8] */
    {
        NfirHighGraph g;
        NfirTensorRef t0; t0.ndim = 2; t0.shape[0] = 4; t0.shape[1] = 1;
        g.add_tensor(t0);
        NfirTensorRef t1; t1.ndim = 2; t1.shape[0] = 1; t1.shape[1] = 8;
        g.add_tensor(t1);
        NfirTensorRef t2; t2.ndim = 0;
        g.add_tensor(t2);

        NfirHighOp op; op.kind = HighOpKind::ELEMENT_ADD;
        op.input_ids = {0, 1}; op.output_ids = {2};
        g.add_op(op);

        ShapeInferencePass si;
        uint32_t inferred = si.run(&g);
        CHECK(inferred == 1, "1 tensor inferred");
        CHECK(g.tensors[2].ndim == 2, "output is 2D");
        CHECK(g.tensors[2].shape[0] == 4, "broadcast dim 0 = 4");
        CHECK(g.tensors[2].shape[1] == 8, "broadcast dim 1 = 8");
    }

    /* Test 3: Full transformer layer chain */
    {
        NfirHighGraph g;
        /* t0[2,64] → RMS_NORM → t1 → MATMUL(t1,t2[64,128]) → t3[2,128] → SILU → t4 */
        NfirTensorRef t0; t0.ndim = 2; t0.shape[0] = 2; t0.shape[1] = 64;
        g.add_tensor(t0);
        NfirTensorRef t1; t1.ndim = 0; g.add_tensor(t1);
        NfirTensorRef t2; t2.ndim = 2; t2.shape[0] = 64; t2.shape[1] = 128;
        g.add_tensor(t2);
        NfirTensorRef t3; t3.ndim = 0; g.add_tensor(t3);
        NfirTensorRef t4; t4.ndim = 0; g.add_tensor(t4);

        NfirHighOp op0; op0.kind = HighOpKind::RMS_NORM;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::MATMUL;
        op1.input_ids = {1, 2}; op1.output_ids = {3};
        g.add_op(op1);

        NfirHighOp op2; op2.kind = HighOpKind::SILU;
        op2.input_ids = {3}; op2.output_ids = {4};
        g.add_op(op2);

        g.add_edge(0, 1);
        g.add_edge(1, 2);

        ShapeInferencePass si;
        uint32_t inferred = si.run(&g);
        CHECK(inferred == 3, "3 tensors inferred");
        CHECK(g.tensors[1].shape[0] == 2, "RMS_NORM preserves shape");
        CHECK(g.tensors[1].shape[1] == 64, "RMS_NORM preserves shape");
        CHECK(g.tensors[3].shape[0] == 2, "MATMUL M=2");
        CHECK(g.tensors[3].shape[1] == 128, "MATMUL N=128");
        CHECK(g.tensors[4].shape[0] == 2, "SILU preserves shape");
        CHECK(g.tensors[4].shape[1] == 128, "SILU preserves shape");
    }

    /* Test 4: Null / empty */
    {
        ShapeInferencePass si;
        CHECK(si.run(nullptr) == 0, "null graph");
        NfirHighGraph g;
        CHECK(si.run(&g) == 0, "empty graph");
    }

    printf("PASS: all shape inference tests passed\n");
    return 0;
}
