/**
 * @file runtime_basic_test.cpp
 * @brief Phase 43A.1 — NeuralOSRuntime basic tests
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
    printf("=== NeuralOSRuntime Basic Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    neuralOS::kernel::RuntimeConfig cfg;
    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched, cfg);

    /* Subsystem accessors */
    CHECK(runtime.vmmu() != nullptr, "vmmu accessible");
    CHECK(runtime.cfs() != nullptr, "cfs accessible");
    CHECK(runtime.context_hub() != nullptr, "context_hub accessible");
    CHECK(runtime.spec_engine() == nullptr, "spec_engine null when disabled");

    /* Submit 3 requests */
    for (int i = 0; i < 3; ++i) {
        nf::InferenceRequest req;
        req.prompt_tokens = {1, 2, 3, 4};
        req.max_new_tokens = 8;
        uint32_t id = runtime.submit(req);
        CHECK(id != UINT32_MAX, "submit succeeded");
    }
    CHECK(runtime.stats().total_submitted == 3, "3 submitted");
    CHECK(sched.num_requests == 3, "scheduler has 3 requests");

    /* Allocate sequences so CFS can schedule them */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].seq_id = kv.alloc_sequence();
        sched.requests[i].state = nf::RequestState::DECODE;
    }

    auto result = runtime.schedule_step();
    CHECK(result.selected_ids.size() > 0, "schedule_step selected requests");
    CHECK(result.total_tokens > 0, "schedule_step has tokens");

    /* Complete a request */
    runtime.complete(sched.requests[0].req_id);
    CHECK(sched.requests[0].state == nf::RequestState::COMPLETE, "request completed");
    CHECK(runtime.stats().total_completed == 1, "1 completed");

    printf("PASS: all NeuralOSRuntime basic tests passed\n");
    return 0;
}
