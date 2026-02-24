/**
 * @file cse_pass_test.cpp
 * @brief Phase 42B.2 — Common Subexpression Elimination pass tests
 */

#include "neuralOS/compiler/cse_pass.hpp"
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
    printf("=== CSE Pass Test ===\n");

    /* Test 1: Two identical MATMUL ops → merged to one */
    {
        NfirHighGraph g;
        NfirTensorRef t0; t0.id = 0; g.add_tensor(t0);
        NfirTensorRef t1; t1.id = 1; g.add_tensor(t1);
        NfirTensorRef t2; t2.id = 2; g.add_tensor(t2);
        NfirTensorRef t3; t3.id = 3; g.add_tensor(t3);

        NfirHighOp op0; op0.kind = HighOpKind::MATMUL;
        op0.input_ids = {0, 1}; op0.output_ids = {2};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::MATMUL;
        op1.input_ids = {0, 1}; op1.output_ids = {3};
        g.add_op(op1);

        CSEPass cse;
        uint32_t removed = cse.run(&g);
        CHECK(removed == 1, "1 duplicate MATMUL removed");
        CHECK(g.num_ops() == 1, "1 op remains");
    }

    /* Test 2: Different inputs → not merged */
    {
        NfirHighGraph g;
        NfirTensorRef t0; t0.id = 0; g.add_tensor(t0);
        NfirTensorRef t1; t1.id = 1; g.add_tensor(t1);
        NfirTensorRef t2; t2.id = 2; g.add_tensor(t2);
        NfirTensorRef t3; t3.id = 3; g.add_tensor(t3);

        NfirHighOp op0; op0.kind = HighOpKind::MATMUL;
        op0.input_ids = {0, 1}; op0.output_ids = {2};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::MATMUL;
        op1.input_ids = {0, 3}; op1.output_ids = {3};
        g.add_op(op1);

        CSEPass cse;
        uint32_t removed = cse.run(&g);
        CHECK(removed == 0, "different inputs not merged");
        CHECK(g.num_ops() == 2, "2 ops remain");
    }

    /* Test 3: Chain with duplicates → all merged */
    {
        NfirHighGraph g;
        for (uint32_t i = 0; i < 6; ++i) {
            NfirTensorRef t; t.id = i; g.add_tensor(t);
        }

        /* 3 identical SILU ops on same input */
        for (uint32_t i = 0; i < 3; ++i) {
            NfirHighOp op; op.kind = HighOpKind::SILU;
            op.input_ids = {0}; op.output_ids = {static_cast<uint32_t>(i + 3)};
            g.add_op(op);
        }

        CSEPass cse;
        uint32_t removed = cse.run(&g);
        CHECK(removed == 2, "2 duplicate SILUs removed");
        CHECK(g.num_ops() == 1, "1 op remains");
    }

    /* Test 4: Empty / null */
    {
        NfirHighGraph g;
        CSEPass cse;
        CHECK(cse.run(&g) == 0, "empty graph");
        CHECK(cse.run(nullptr) == 0, "null graph");
    }

    printf("PASS: all CSE pass tests passed\n");
    return 0;
}
