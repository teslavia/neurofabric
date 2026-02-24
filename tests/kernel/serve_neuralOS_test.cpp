/**
 * @file serve_neuralOS_test.cpp
 * @brief Phase 43C.2 — nf_serve --neuralOS mock test
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
    printf("=== nf_serve --neuralOS Mock Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    /* Simulate ServerState with NeuralOSRuntime */
    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched);
    neuralOS::kernel::BatchInferenceLoop loop(&runtime, &kv, &sched);

    /* Submit 2 concurrent requests (like HTTP server) */
    nf::InferenceRequest req1;
    req1.prompt_tokens = {10, 20, 30};
    req1.max_new_tokens = 8;
    uint32_t id1 = loop.submit(req1);

    nf::InferenceRequest req2;
    req2.prompt_tokens = {40, 50, 60};
    req2.max_new_tokens = 8;
    uint32_t id2 = loop.submit(req2);

    CHECK(id1 != UINT32_MAX && id2 != UINT32_MAX, "both submitted");
    CHECK(sched.num_requests == 2, "2 requests");

    /* Set to DECODE */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].seq_id = kv.alloc_sequence();
        sched.requests[i].state = nf::RequestState::DECODE;
    }

    /* CFS scheduling via batch loop */
    auto r = loop.step();
    CHECK(r.num_selected == 2, "both selected in batch");

    /* Stats accessible */
    CHECK(runtime.stats().total_submitted == 2, "stats: 2 submitted");
    uint64_t vrt1 = runtime.cfs()->get_vruntime(id1);
    uint64_t vrt2 = runtime.cfs()->get_vruntime(id2);
    CHECK(vrt1 > 0 && vrt2 > 0, "both have vruntime");

    printf("PASS: all nf_serve --neuralOS tests passed\n");
    return 0;
}
