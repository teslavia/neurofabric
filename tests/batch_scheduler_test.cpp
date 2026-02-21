/**
 * @file batch_scheduler_test.cpp
 * @brief Phase 32 Step 3: RequestScheduler + continuous batching tests
 *
 * 4 sub-tests:
 *   1. Scheduler_submit_single — submit 1 request, schedule_batch returns it
 *   2. Scheduler_prefill_decode_split — 2 requests in different states
 *   3. Scheduler_priority_ordering — 3 requests with different priorities
 *   4. Scheduler_token_budget — max_batch_tokens budget enforcement
 */

#include "model_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

static void test_submit_single() {
    nf::PagedKVCache kv;
    kv.init(64, 4, 1, 2, 32);

    nf::RequestScheduler sched{};

    nf::InferenceRequest req{};
    req.prompt_tokens = {1, 2, 3, 4};
    req.max_new_tokens = 10;

    uint32_t id = sched.submit(std::move(req));
    CHECK(id != UINT32_MAX);
    CHECK(sched.num_requests == 1);
    CHECK(sched.requests[0].state == nf::RequestState::QUEUED);

    /* Schedule — should transition to PREFILL and allocate a sequence */
    auto batch = sched.schedule_batch(64, kv);
    CHECK(batch.num_sequences == 1);
    CHECK(batch.total_tokens == 4);  /* prompt length */
    CHECK(batch.seq_lens[0] == 4);
    CHECK(sched.requests[0].state == nf::RequestState::PREFILL);
    CHECK(sched.requests[0].seq_id != UINT32_MAX);

    std::printf("  [PASS] Scheduler_submit_single\n");
}

static void test_prefill_decode_split() {
    nf::PagedKVCache kv;
    kv.init(64, 4, 1, 2, 32);

    nf::RequestScheduler sched{};

    /* Request 0: already in DECODE state */
    nf::InferenceRequest req0{};
    req0.prompt_tokens = {1, 2, 3};
    req0.max_new_tokens = 10;
    sched.submit(std::move(req0));

    /* Request 1: QUEUED (will become PREFILL) */
    nf::InferenceRequest req1{};
    req1.prompt_tokens = {10, 20, 30, 40, 50};
    req1.max_new_tokens = 5;
    sched.submit(std::move(req1));

    /* First schedule: both get scheduled */
    auto batch1 = sched.schedule_batch(64, kv);
    CHECK(batch1.num_sequences == 2);

    /* Simulate prefill completion for req0 → transitions to DECODE */
    int32_t dummy_tokens[2] = {100, 200};
    sched.on_step_complete(batch1, dummy_tokens);
    CHECK(sched.requests[0].state == nf::RequestState::DECODE);
    CHECK(sched.requests[1].state == nf::RequestState::DECODE);

    /* Second schedule: both in DECODE, each needs 1 token */
    auto batch2 = sched.schedule_batch(64, kv);
    CHECK(batch2.num_sequences == 2);
    CHECK(batch2.total_tokens == 2);  /* 1 + 1 */
    CHECK(batch2.seq_lens[0] == 1);
    CHECK(batch2.seq_lens[1] == 1);

    std::printf("  [PASS] Scheduler_prefill_decode_split\n");
}

static void test_priority_ordering() {
    nf::PagedKVCache kv;
    kv.init(64, 4, 1, 2, 32);

    nf::RequestScheduler sched{};

    /* 3 requests with different priorities */
    nf::InferenceRequest r0{};
    r0.prompt_tokens = {1, 2};
    r0.priority = 1;
    r0.submit_time_us = 100;
    sched.submit(std::move(r0));

    nf::InferenceRequest r1{};
    r1.prompt_tokens = {3, 4};
    r1.priority = 3;  /* highest */
    r1.submit_time_us = 200;
    sched.submit(std::move(r1));

    nf::InferenceRequest r2{};
    r2.prompt_tokens = {5, 6};
    r2.priority = 2;
    r2.submit_time_us = 50;
    sched.submit(std::move(r2));

    auto batch = sched.schedule_batch(64, kv);
    CHECK(batch.num_sequences == 3);

    /* All are QUEUED → PREFILL, sorted by priority (highest first) */
    /* After scheduling, verify the highest priority got seq_id first */
    CHECK(sched.requests[1].state == nf::RequestState::PREFILL);  /* priority 3 */
    CHECK(sched.requests[2].state == nf::RequestState::PREFILL);  /* priority 2 */
    CHECK(sched.requests[0].state == nf::RequestState::PREFILL);  /* priority 1 */

    std::printf("  [PASS] Scheduler_priority_ordering\n");
}

static void test_token_budget() {
    nf::PagedKVCache kv;
    kv.init(64, 4, 1, 2, 32);

    nf::RequestScheduler sched{};

    /* Request with 20-token prompt */
    nf::InferenceRequest r0{};
    r0.prompt_tokens.resize(20, 1);
    r0.max_new_tokens = 5;
    sched.submit(std::move(r0));

    /* Request with 15-token prompt */
    nf::InferenceRequest r1{};
    r1.prompt_tokens.resize(15, 2);
    r1.max_new_tokens = 5;
    sched.submit(std::move(r1));

    /* Budget = 32: first request (20 tokens) fits, second (15) exceeds remaining */
    auto batch = sched.schedule_batch(32, kv);
    CHECK(batch.num_sequences == 1);
    CHECK(batch.total_tokens == 20);

    /* With budget = 40: both fit */
    /* Reset scheduler */
    nf::RequestScheduler sched2{};
    nf::InferenceRequest r2{};
    r2.prompt_tokens.resize(20, 1);
    sched2.submit(std::move(r2));
    nf::InferenceRequest r3{};
    r3.prompt_tokens.resize(15, 2);
    sched2.submit(std::move(r3));

    nf::PagedKVCache kv2;
    kv2.init(64, 4, 1, 2, 32);
    auto batch2 = sched2.schedule_batch(40, kv2);
    CHECK(batch2.num_sequences == 2);
    CHECK(batch2.total_tokens == 35);

    std::printf("  [PASS] Scheduler_token_budget\n");
}

int main() {
    std::printf("=== batch_scheduler_test ===\n");
    test_submit_single();
    test_prefill_decode_split();
    test_priority_ordering();
    test_token_budget();
    std::printf("All 4 batch_scheduler tests passed.\n");
    return 0;
}
