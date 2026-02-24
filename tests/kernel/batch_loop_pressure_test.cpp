/**
 * @file batch_loop_pressure_test.cpp
 * @brief Phase 43B.1 — BatchInferenceLoop memory pressure tests
 */

#include "neuralOS/kernel/BatchInferenceLoop.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== BatchInferenceLoop Pressure Test ===\n");

    nf::PagedKVCache kv;
    kv.init(16, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    neuralOS::kernel::RuntimeConfig cfg;
    cfg.vmmu_cfg.pressure_threshold = 0.75f;
    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched, cfg);
    neuralOS::kernel::BatchInferenceLoop loop(&runtime, &kv, &sched);

    /* Submit requests */
    for (int i = 0; i < 3; ++i) {
        nf::InferenceRequest req;
        req.prompt_tokens = {1, 2, 3};
        req.max_new_tokens = 16;
        loop.submit(req);
    }

    /* Allocate sequences and set DECODE */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].seq_id = kv.alloc_sequence();
        sched.requests[i].state = nf::RequestState::DECODE;
    }

    /* Give different vruntimes */
    runtime.cfs()->account_tokens(sched.requests[0].req_id, 5, false);
    runtime.cfs()->account_tokens(sched.requests[1].req_id, 50, false);
    runtime.cfs()->account_tokens(sched.requests[2].req_id, 200, false);

    /* Fill cache to trigger pressure */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        for (int t = 0; t < 16; ++t)
            kv.append(sched.requests[i].seq_id);
    }

    CHECK(kv.allocator.num_used() >= 12, "cache near full");

    /* step() triggers pressure check internally */
    uint64_t preempted_before = runtime.stats().total_preempted;
    loop.step();
    CHECK(runtime.stats().total_preempted > preempted_before,
          "pressure triggered preemption");

    /* Verify preempted request can be resumed */
    bool found_preempted = false;
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        if (sched.requests[i].state == nf::RequestState::PREEMPTED) {
            found_preempted = true;
            runtime.cfs()->resume(sched.requests[i].req_id);
            CHECK(sched.requests[i].state == nf::RequestState::DECODE,
                  "resumed request back to DECODE");
            break;
        }
    }
    CHECK(found_preempted, "at least one request was preempted");

    printf("PASS: all BatchInferenceLoop pressure tests passed\n");
    return 0;
}
