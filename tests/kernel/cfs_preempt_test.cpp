/**
 * @file cfs_preempt_test.cpp
 * @brief Phase 36.3 — CFS preemption + resume tests
 */

#include "neuralOS/kernel/CFS.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== CFS Preempt Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);

    nf::RequestScheduler sched;

    nf::InferenceRequest r1;
    r1.prompt_tokens = {1, 2, 3};
    r1.vtc_weight = 1;
    uint32_t id1 = sched.submit(r1);

    nf::InferenceRequest r2;
    r2.prompt_tokens = {4, 5, 6};
    r2.vtc_weight = 1;
    uint32_t id2 = sched.submit(r2);

    /* Set both to DECODE */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].state = nf::RequestState::DECODE;
        sched.requests[i].seq_id = kv.alloc_sequence();
    }

    neuralOS::L2::CFS cfs(&sched, &kv);

    /* Test 1: Preempt r1 */
    CHECK(cfs.preempt(id1), "preempt r1 succeeded");
    CHECK(sched.requests[0].state == nf::RequestState::PREEMPTED, "r1 is PREEMPTED");
    CHECK(cfs.num_preempted() == 1, "1 preempted");

    /* Test 2: Can't preempt already preempted */
    CHECK(!cfs.preempt(id1), "can't preempt again");

    /* Test 3: Resume r1 */
    CHECK(cfs.resume(id1), "resume r1 succeeded");
    CHECK(sched.requests[0].state == nf::RequestState::DECODE, "r1 back to DECODE");
    CHECK(cfs.num_preempted() == 0, "0 preempted");

    /* Test 4: Can't resume non-preempted */
    CHECK(!cfs.resume(id1), "can't resume non-preempted");

    /* Test 5: Preempt invalid request */
    CHECK(!cfs.preempt(9999), "can't preempt invalid req");

    /* Test 6: Preempted requests list */
    cfs.preempt(id1);
    cfs.preempt(id2);
    auto preempted = cfs.preempted_requests();
    CHECK(preempted.size() == 2, "2 preempted requests");

    /* Test 7: Resume all */
    cfs.resume(id1);
    cfs.resume(id2);
    CHECK(cfs.num_preempted() == 0, "all resumed");

    /* Test 8: Rebalance with VTC-based preemption.
     * Give r1 a huge vruntime, r2 stays low → r1 should get preempted */
    cfs.account_tokens(id1, 1000, false);
    cfs.account_tokens(id2, 1, false);

    auto result = cfs.rebalance();
    /* r1's vruntime is way higher than r2's, should be preempted */
    bool r1_preempted = false;
    for (auto pid : result.preempted_req_ids) {
        if (pid == id1) r1_preempted = true;
    }
    CHECK(r1_preempted, "r1 preempted by rebalance (high vruntime)");

    /* Test 9: migrate_request returns false (Phase 38 stub) */
    CHECK(!cfs.migrate_request(id2, "192.168.1.70:9999"),
          "migrate_request is stub");

    printf("PASS: all CFS preempt tests passed\n");
    return 0;
}
