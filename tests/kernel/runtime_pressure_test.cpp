/**
 * @file runtime_pressure_test.cpp
 * @brief Phase 43A.1 — NeuralOSRuntime memory pressure tests
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
    printf("=== NeuralOSRuntime Pressure Test ===\n");

    nf::PagedKVCache kv;
    kv.init(16, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    neuralOS::kernel::RuntimeConfig cfg;
    cfg.vmmu_cfg.pressure_threshold = 0.80f;
    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched, cfg);

    /* Submit 2 requests */
    for (int i = 0; i < 2; ++i) {
        nf::InferenceRequest req;
        req.prompt_tokens = {1, 2, 3};
        req.max_new_tokens = 16;
        runtime.submit(req);
    }

    /* Put in DECODE state with sequences */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].seq_id = kv.alloc_sequence();
        sched.requests[i].state = nf::RequestState::DECODE;
    }

    /* Different vruntimes so preemption picks highest */
    runtime.cfs()->account_tokens(sched.requests[0].req_id, 10, false);
    runtime.cfs()->account_tokens(sched.requests[1].req_id, 100, false);

    /* Fill KV cache to trigger pressure: 16 blocks, 80% = 13 needed */
    uint32_t seq0 = sched.requests[0].seq_id;
    uint32_t seq1 = sched.requests[1].seq_id;
    /* block_size=4, so 28 tokens per seq = 7 blocks each = 14 blocks total */
    for (int i = 0; i < 56; ++i) {
        kv.append((i < 28) ? seq0 : seq1);
    }

    CHECK(kv.allocator.num_used() >= 13, "cache near full");

    uint64_t preempted_before = runtime.stats().total_preempted;
    runtime.check_pressure();
    CHECK(runtime.stats().total_preempted > preempted_before, "preemption triggered");
    CHECK(sched.requests[1].state == nf::RequestState::PREEMPTED,
          "high-vruntime request preempted");

    printf("PASS: all NeuralOSRuntime pressure tests passed\n");
    return 0;
}
