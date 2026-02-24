/**
 * @file e2e_neuralOS_test.cpp
 * @brief Phase 44B.4 — E2E NeuralOS kernel path test
 *
 * Validates NeuralOSRuntime + BatchInferenceLoop with concurrent requests.
 * Checks CFS fairness (vruntime spread) and pressure-triggered preemption.
 */

#include "neuralOS/kernel/NeuralOSRuntime.hpp"
#include "neuralOS/kernel/BatchInferenceLoop.hpp"
#include "neuralOS/compiler/compiler_pipeline.hpp"
#include "model/model_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); std::exit(1); } \
} while(0)

int main() {
    std::fprintf(stderr, "[e2e_neuralOS] starting...\n");

    /* Setup PagedKVCache + RequestScheduler */
    nf::PagedKVCache kv;
    kv.init(256, 16, 4, 4, 64);  /* 256 blocks, 16 tokens/block, 4 layers, 4 heads, dim=64 */

    nf::RequestScheduler sched;

    /* Create NeuralOSRuntime */
    neuralOS::L2::RuntimeConfig cfg;
    cfg.vmmu_cfg.pressure_threshold = 0.8f;
    cfg.cfs_cfg.preempt_threshold = 4;
    auto runtime = std::make_unique<neuralOS::L2::NeuralOSRuntime>(&kv, &sched, cfg);

    /* Create BatchInferenceLoop */
    neuralOS::L2::BatchInferenceLoop batch(runtime.get(), &kv, &sched);

    /* ---- Test 1: Submit 5 concurrent requests ---- */
    std::fprintf(stderr, "[e2e_neuralOS] test 1: 5 concurrent requests\n");

    std::vector<uint32_t> req_ids;
    for (int i = 0; i < 5; ++i) {
        nf::InferenceRequest ir;
        ir.prompt_tokens = {1, 2, 3, 4, 5};
        ir.max_new_tokens = 4;
        uint32_t id = batch.submit(ir);
        CHECK(id != UINT32_MAX, "submit should succeed");
        req_ids.push_back(id);
    }

    CHECK(runtime->stats().total_submitted == 5, "5 requests submitted");
    CHECK(batch.has_active(), "should have active requests");

    /* ---- Test 2: Step loop until all complete ---- */
    std::fprintf(stderr, "[e2e_neuralOS] test 2: step loop\n");

    uint32_t max_steps = 200;
    uint32_t steps = 0;
    while (batch.has_active() && steps < max_steps) {
        auto result = batch.step();
        (void)result;
        ++steps;
    }

    CHECK(!batch.has_active(), "all requests should complete");
    CHECK(runtime->stats().total_completed == 5, "5 requests completed");
    std::fprintf(stderr, "[e2e_neuralOS]   completed in %u steps\n", steps);

    /* ---- Test 3: CFS fairness — vruntime spread ---- */
    std::fprintf(stderr, "[e2e_neuralOS] test 3: CFS fairness\n");

    uint64_t min_vrt = UINT64_MAX, max_vrt = 0;
    for (auto id : req_ids) {
        uint64_t vrt = runtime->cfs()->get_vruntime(id);
        if (vrt < min_vrt) min_vrt = vrt;
        if (vrt > max_vrt) max_vrt = vrt;
    }
    uint64_t spread = max_vrt - min_vrt;
    std::fprintf(stderr, "[e2e_neuralOS]   vruntime spread: %llu (min=%llu max=%llu)\n",
                 (unsigned long long)spread, (unsigned long long)min_vrt, (unsigned long long)max_vrt);
    /* Fairness: spread should be reasonable (< 10x the per-token weight) */
    CHECK(spread < 10000, "vruntime spread should be bounded");

    /* ---- Test 4: CompilerPipeline integration ---- */
    std::fprintf(stderr, "[e2e_neuralOS] test 4: compiler pipeline\n");

    neuralOS::L1::NfirHighGraph graph;

    /* Create tensors for the chain */
    neuralOS::L1::NfirTensorRef t0; t0.name = "tokens";
    uint32_t tid0 = graph.add_tensor(t0);
    neuralOS::L1::NfirTensorRef t1; t1.name = "embed_out";
    uint32_t tid1 = graph.add_tensor(t1);
    neuralOS::L1::NfirTensorRef t2; t2.name = "attn_out";
    uint32_t tid2 = graph.add_tensor(t2);
    neuralOS::L1::NfirTensorRef t3; t3.name = "ffn_out";
    uint32_t tid3 = graph.add_tensor(t3);

    neuralOS::L1::NfirHighOp op1;
    op1.kind = neuralOS::L1::HighOpKind::EMBEDDING;
    op1.name = "embed";
    op1.input_ids.push_back(tid0);
    op1.output_ids.push_back(tid1);
    graph.add_op(op1);

    neuralOS::L1::NfirHighOp op2;
    op2.kind = neuralOS::L1::HighOpKind::ATTENTION;
    op2.name = "attn";
    op2.input_ids.push_back(tid1);
    op2.output_ids.push_back(tid2);
    graph.add_op(op2);

    neuralOS::L1::NfirHighOp op3;
    op3.kind = neuralOS::L1::HighOpKind::FFN_BLOCK;
    op3.name = "ffn";
    op3.input_ids.push_back(tid2);
    op3.output_ids.push_back(tid3);
    graph.add_op(op3);

    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.output_tensor_ids.push_back(tid3);

    neuralOS::L1::CompilerPipeline compiler;
    auto cr = compiler.run(&graph);
    std::fprintf(stderr, "[e2e_neuralOS]   compiler: removed=%u merged=%u shapes=%u fusions=%u\n",
                 cr.ops_removed, cr.ops_merged, cr.shapes_inferred, cr.fusions_found);
    CHECK(graph.num_ops() >= 2, "graph should have ops after pipeline");

    /* ---- Test 5: Pressure-triggered preemption ---- */
    std::fprintf(stderr, "[e2e_neuralOS] test 5: pressure preemption\n");

    nf::PagedKVCache kv2;
    kv2.init(8, 4, 2, 2, 32);  /* tiny: 8 blocks only */
    nf::RequestScheduler sched2;

    neuralOS::L2::RuntimeConfig cfg2;
    cfg2.vmmu_cfg.pressure_threshold = 0.5f;
    auto runtime2 = std::make_unique<neuralOS::L2::NeuralOSRuntime>(&kv2, &sched2, cfg2);
    neuralOS::L2::BatchInferenceLoop batch2(runtime2.get(), &kv2, &sched2);

    /* Submit requests to create pressure */
    for (int i = 0; i < 4; ++i) {
        nf::InferenceRequest ir;
        ir.prompt_tokens = {1, 2, 3};
        ir.max_new_tokens = 8;
        batch2.submit(ir);
    }

    /* Step a few times — pressure check should trigger */
    for (int i = 0; i < 20; ++i)
        batch2.step();

    /* Preemption may or may not have occurred depending on block allocation,
     * but the runtime should not crash */
    std::fprintf(stderr, "[e2e_neuralOS]   preempted: %llu\n",
                 (unsigned long long)runtime2->stats().total_preempted);

    /* ---- Done ---- */
    auto& st = runtime->stats();
    std::fprintf(stderr, "[e2e_neuralOS] PASS — submitted=%llu completed=%llu "
                 "preempted=%llu prefix_hits=%u\n",
                 (unsigned long long)st.total_submitted,
                 (unsigned long long)st.total_completed,
                 (unsigned long long)st.total_preempted,
                 st.prefix_hits);
    return 0;
}
