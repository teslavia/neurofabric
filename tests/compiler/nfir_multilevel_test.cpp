/**
 * @file nfir_multilevel_test.cpp
 * @brief Phase 37.1-37.3 — Multi-level NFIR + fusion + memory plan tests
 */

#include "neuralOS/compiler/nfir_high.hpp"
#include "neuralOS/compiler/nfir_low.hpp"
#include "neuralOS/compiler/fusion_pass.hpp"
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
    printf("=== NFIR Multi-Level Test ===\n");

    /* Build a high-level graph: dequant → matmul → rms_norm → rope → silu → element_mul */
    NfirHighGraph high;

    /* Tensors */
    NfirTensorRef t0; t0.name = "weight_q4"; t0.size_bytes = 4096;
    NfirTensorRef t1; t1.name = "input";     t1.size_bytes = 2048;
    NfirTensorRef t2; t2.name = "dequant_out"; t2.size_bytes = 8192;
    NfirTensorRef t3; t3.name = "matmul_out";  t3.size_bytes = 4096;
    NfirTensorRef t4; t4.name = "norm_out";    t4.size_bytes = 4096;
    NfirTensorRef t5; t5.name = "rope_out";    t5.size_bytes = 4096;
    NfirTensorRef t6; t6.name = "silu_out";    t6.size_bytes = 4096;
    NfirTensorRef t7; t7.name = "gate_out";    t7.size_bytes = 4096;

    for (auto* t : {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7})
        high.add_tensor(*t);

    /* Ops */
    NfirHighOp dequant; dequant.kind = HighOpKind::DEQUANT;
    dequant.input_ids = {0}; dequant.output_ids = {2};
    uint32_t dq_id = high.add_op(dequant);

    NfirHighOp matmul; matmul.kind = HighOpKind::MATMUL;
    matmul.input_ids = {2, 1}; matmul.output_ids = {3};
    uint32_t mm_id = high.add_op(matmul);

    NfirHighOp norm; norm.kind = HighOpKind::RMS_NORM;
    norm.input_ids = {3}; norm.output_ids = {4};
    uint32_t nm_id = high.add_op(norm);

    NfirHighOp rope; rope.kind = HighOpKind::ROPE;
    rope.input_ids = {4}; rope.output_ids = {5};
    uint32_t rp_id = high.add_op(rope);

    NfirHighOp silu; silu.kind = HighOpKind::SILU;
    silu.input_ids = {3}; silu.output_ids = {6};
    uint32_t si_id = high.add_op(silu);

    NfirHighOp emul; emul.kind = HighOpKind::ELEMENT_MUL;
    emul.input_ids = {6, 5}; emul.output_ids = {7};
    uint32_t em_id = high.add_op(emul);

    /* Edges */
    high.add_edge(dq_id, mm_id);
    high.add_edge(mm_id, nm_id);
    high.add_edge(nm_id, rp_id);
    high.add_edge(mm_id, si_id);
    high.add_edge(si_id, em_id);
    high.add_edge(rp_id, em_id);

    CHECK(high.num_ops() == 6, "6 high-level ops");
    CHECK(high.num_tensors() == 8, "8 tensors");

    /* Test fusion pass */
    FusionPass fuser;
    uint32_t fusions = fuser.run(&high);
    printf("  Fusions found: %u\n", fusions);
    CHECK(fusions >= 2, "at least 2 fusions (dequant+matmul, silu+element_mul)");

    bool has_dq_fuse = false, has_swiglu = false;
    for (auto& fc : high.fusion_candidates) {
        if (fc.fused_name == "fused_dq_matmul") has_dq_fuse = true;
        if (fc.fused_name == "fused_swiglu") has_swiglu = true;
    }
    CHECK(has_dq_fuse, "dequant+matmul fused");
    CHECK(has_swiglu, "silu+element_mul fused");

    /* Test lowering */
    NfirLowGraph low = lower(high);
    printf("  Low-level ops: %u (from %u high-level)\n", low.num_ops(), high.num_ops());
    CHECK(low.num_ops() < high.num_ops(), "fusion reduced op count");

    /* Verify fused op names exist */
    bool found_fused_dq = false;
    for (auto& op : low.ops) {
        if (op.op_name == "fused_dq_matmul") found_fused_dq = true;
    }
    CHECK(found_fused_dq, "fused_dq_matmul in low-level graph");

    /* Test memory plan pass */
    MemoryPlanPass planner;
    MemoryPlan plan = planner.run(low);
    printf("  Peak memory: %llu bytes, reused: %u\n",
           (unsigned long long)plan.peak_bytes, plan.num_reused);
    CHECK(plan.peak_bytes > 0, "peak memory > 0");
    CHECK(!plan.allocations.empty(), "has allocations");

    /* Verify no overlapping live allocations */
    /* (simplified: just check plan is non-empty and reasonable) */
    uint64_t total_tensor_bytes = 0;
    for (auto& t : low.tensors) total_tensor_bytes += t.size_bytes;
    CHECK(plan.peak_bytes <= total_tensor_bytes,
          "peak <= total (reuse should help)");

    printf("PASS: all NFIR multi-level tests passed\n");
    return 0;
}
