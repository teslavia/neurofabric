/**
 * @file generate_neuralOS_test.cpp
 * @brief Phase 43C.1 — nf_generate --neuralOS mock test
 */

#include "neuralOS/kernel/NeuralOSRuntime.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== nf_generate --neuralOS Mock Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    /* Simulate --neuralOS flag: construct runtime */
    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched);

    /* Submit single request (like nf_generate does) */
    nf::InferenceRequest req;
    req.prompt_tokens = {1, 2, 3, 4, 5};
    req.max_new_tokens = 16;
    uint32_t req_id = runtime.submit(req);
    CHECK(req_id != UINT32_MAX, "submit succeeded");
    CHECK(runtime.stats().total_submitted == 1, "1 submitted");

    /* Allocate sequence and set to DECODE (simulating prefill done) */
    sched.requests[0].seq_id = kv.alloc_sequence();
    sched.requests[0].state = nf::RequestState::DECODE;

    /* schedule_step before each decode step */
    auto sr = runtime.schedule_step();
    CHECK(sr.selected_ids.size() == 1, "single request selected");
    CHECK(sr.selected_ids[0] == req_id, "correct request selected");

    /* Account tokens (CFS) */
    runtime.cfs()->account_tokens(req_id, 1, false);
    uint64_t vrt = runtime.cfs()->get_vruntime(req_id);
    CHECK(vrt > 0, "vruntime increased");

    /* Backward compat: without --neuralOS, runtime not constructed */
    /* (just verify the flag-based construction pattern works) */
    bool use_neuralOS = false;
    neuralOS::kernel::NeuralOSRuntime* opt_runtime = nullptr;
    if (use_neuralOS) {
        opt_runtime = new neuralOS::kernel::NeuralOSRuntime(&kv, &sched);
    }
    CHECK(opt_runtime == nullptr, "runtime null when flag disabled");

    printf("PASS: all nf_generate --neuralOS tests passed\n");
    return 0;
}
