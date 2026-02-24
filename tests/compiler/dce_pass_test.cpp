/**
 * @file dce_pass_test.cpp
 * @brief Phase 42B.1 — Dead Code Elimination pass tests
 */

#include "neuralOS/compiler/dce_pass.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

using namespace neuralOS::compiler;

static NfirTensorRef make_tensor(uint32_t id) {
    NfirTensorRef t;
    t.id = id;
    t.ndim = 2;
    t.shape[0] = 4; t.shape[1] = 4;
    return t;
}

int main() {
    printf("=== DCE Pass Test ===\n");

    /* Test 1: Graph with dead branch → removed.
     * Live path: t0 → MATMUL → t1 → SILU → t2 → SOFTMAX → t5 (graph output)
     * Dead path: t0 → RMS_NORM → t3 → GELU → t4 (t4 consumed by nothing that
     *            leads to t5, so the dead ops are unreachable from output t5) */
    {
        NfirHighGraph g;
        g.add_tensor(make_tensor(0));  /* t0: input */
        g.add_tensor(make_tensor(1));  /* t1 */
        g.add_tensor(make_tensor(2));  /* t2 */
        g.add_tensor(make_tensor(3));  /* t3: dead */
        g.add_tensor(make_tensor(4));  /* t4: dead */
        g.add_tensor(make_tensor(5));  /* t5: graph output */

        NfirHighOp op0; op0.kind = HighOpKind::MATMUL;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);  /* id=0 */

        NfirHighOp op1; op1.kind = HighOpKind::SILU;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);  /* id=1 */

        /* Dead branch */
        NfirHighOp op2; op2.kind = HighOpKind::RMS_NORM;
        op2.input_ids = {0}; op2.output_ids = {3};
        g.add_op(op2);  /* id=2 */

        NfirHighOp op3; op3.kind = HighOpKind::GELU;
        op3.input_ids = {3}; op3.output_ids = {4};
        g.add_op(op3);  /* id=3 */

        /* Final op consuming t2 → makes t5 the sole graph output */
        NfirHighOp op4; op4.kind = HighOpKind::SOFTMAX;
        op4.input_ids = {2}; op4.output_ids = {5};
        g.add_op(op4);  /* id=4 */

        g.add_edge(0, 1);
        g.add_edge(1, 4);
        g.add_edge(2, 3);

        /* Mark t5 as the sole graph output */
        g.output_tensor_ids = {5};

        DCEPass dce;
        uint32_t removed = dce.run(&g);
        CHECK(removed == 2, "2 dead ops removed");
        CHECK(g.num_ops() == 3, "3 live ops remain");
    }

    /* Test 2: All-live graph → unchanged */
    {
        NfirHighGraph g;
        g.add_tensor(make_tensor(0));
        g.add_tensor(make_tensor(1));
        g.add_tensor(make_tensor(2));

        NfirHighOp op0; op0.kind = HighOpKind::MATMUL;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::SILU;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);

        g.add_edge(0, 1);

        DCEPass dce;
        uint32_t removed = dce.run(&g);
        CHECK(removed == 0, "no ops removed from all-live graph");
        CHECK(g.num_ops() == 2, "2 ops remain");
    }

    /* Test 3: Cascading dead ops → all removed.
     * Live: t0 → MATMUL → t1 → SILU → t2 → SOFTMAX → t6 (output)
     * Dead cascade: t0 → GELU → t3 → LAYER_NORM → t4 → RMS_NORM → t5 */
    {
        NfirHighGraph g;
        for (uint32_t i = 0; i < 7; ++i)
            g.add_tensor(make_tensor(i));

        NfirHighOp op0; op0.kind = HighOpKind::MATMUL;
        op0.input_ids = {0}; op0.output_ids = {1};
        g.add_op(op0);

        NfirHighOp op1; op1.kind = HighOpKind::SILU;
        op1.input_ids = {1}; op1.output_ids = {2};
        g.add_op(op1);

        NfirHighOp op_final; op_final.kind = HighOpKind::SOFTMAX;
        op_final.input_ids = {2}; op_final.output_ids = {6};
        g.add_op(op_final);

        /* Dead cascade */
        NfirHighOp op2; op2.kind = HighOpKind::GELU;
        op2.input_ids = {0}; op2.output_ids = {3};
        g.add_op(op2);

        NfirHighOp op3; op3.kind = HighOpKind::LAYER_NORM;
        op3.input_ids = {3}; op3.output_ids = {4};
        g.add_op(op3);

        NfirHighOp op4; op4.kind = HighOpKind::RMS_NORM;
        op4.input_ids = {4}; op4.output_ids = {5};
        g.add_op(op4);

        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(3, 4);
        g.add_edge(4, 5);

        /* Mark t6 as the sole graph output */
        g.output_tensor_ids = {6};

        DCEPass dce;
        uint32_t removed = dce.run(&g);
        CHECK(removed == 3, "3 cascading dead ops removed");
        CHECK(g.num_ops() == 3, "3 live ops remain");
    }

    /* Test 4: Empty graph */
    {
        NfirHighGraph g;
        DCEPass dce;
        CHECK(dce.run(&g) == 0, "empty graph returns 0");
        CHECK(dce.run(nullptr) == 0, "null graph returns 0");
    }

    printf("PASS: all DCE pass tests passed\n");
    return 0;
}
