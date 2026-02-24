/**
 * @file batch_loop_test.cpp
 * @brief Phase 43B.1 — BatchInferenceLoop basic tests
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
    printf("=== BatchInferenceLoop Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched);
    neuralOS::kernel::BatchInferenceLoop loop(&runtime, &kv, &sched);

    /* Submit 4 requests */
    for (int i = 0; i < 4; ++i) {
        nf::InferenceRequest req;
        req.prompt_tokens = {1, 2, 3};
        req.max_new_tokens = 4;
        uint32_t id = loop.submit(req);
        CHECK(id != UINT32_MAX, "submit succeeded");
    }
    CHECK(sched.num_requests == 4, "4 requests in scheduler");
    CHECK(loop.has_active(), "has active requests");

    /* Allocate sequences and set to DECODE */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].seq_id = kv.alloc_sequence();
        sched.requests[i].state = nf::RequestState::DECODE;
    }

    /* Step — CFS selects fair batch */
    auto r1 = loop.step();
    CHECK(r1.num_selected > 0, "step selected requests");
    CHECK(r1.total_tokens > 0, "step has tokens");

    /* Multiple steps — VTC accounting progresses */
    auto r2 = loop.step();
    CHECK(r2.num_selected > 0, "second step selects");

    /* Verify VTC accounting happened */
    uint64_t vrt0 = runtime.cfs()->get_vruntime(sched.requests[0].req_id);
    CHECK(vrt0 > 0, "vruntime increased after steps");

    /* Complete all and drain */
    for (uint32_t i = 0; i < sched.num_requests; ++i)
        runtime.complete(sched.requests[i].req_id);

    auto completed = loop.drain_completed();
    CHECK(completed.size() == 4, "4 completed");
    CHECK(!loop.has_active(), "no active after completion");

    printf("PASS: all BatchInferenceLoop tests passed\n");
    return 0;
}
