/**
 * @file cfs_vmmu_integration_test.cpp
 * @brief Phase 42A.1 — CFS ↔ vMMU integration tests
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
    printf("=== CFS ↔ vMMU Integration Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);

    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);
    neuralOS::kernel::vMMU vmmu(&kv, &hub);

    nf::RequestScheduler sched;

    nf::InferenceRequest r1;
    r1.prompt_tokens = {1, 2, 3};
    r1.vtc_weight = 1;
    uint32_t id1 = sched.submit(r1);

    /* Set to DECODE and allocate sequence + blocks */
    sched.requests[0].state = nf::RequestState::DECODE;
    sched.requests[0].seq_id = kv.alloc_sequence();
    auto& seq = kv.sequences[sched.requests[0].seq_id];
    seq.block_table[0] = kv.allocator.alloc_block();
    seq.block_table[1] = kv.allocator.alloc_block();
    seq.num_logical_blocks = 2;

    uint32_t used_before = kv.allocator.num_used();

    /* Test 1: CFS with vMMU — preempt triggers page_out */
    neuralOS::kernel::CFS cfs(&sched, &kv);
    cfs.set_vmmu(&vmmu);

    CHECK(cfs.preempt(id1), "preempt with vMMU succeeded");
    CHECK(sched.requests[0].state == nf::RequestState::PREEMPTED, "r1 PREEMPTED");
    CHECK(cfs.page_out_count() > 0, "page_out was called");
    CHECK(vmmu.num_paged_out() == 2, "2 blocks paged out");

    /* Test 2: Resume triggers page_in */
    CHECK(cfs.resume(id1), "resume with vMMU succeeded");
    CHECK(sched.requests[0].state == nf::RequestState::DECODE, "r1 back to DECODE");
    CHECK(cfs.page_in_count() > 0, "page_in was called");

    /* Test 3: CFS without vMMU — backward compat, no crash */
    neuralOS::kernel::CFS cfs2(&sched, &kv);
    sched.requests[0].state = nf::RequestState::DECODE;
    CHECK(cfs2.preempt(id1), "preempt without vMMU succeeded");
    CHECK(cfs2.page_out_count() == 0, "no page_out without vMMU");
    CHECK(cfs2.resume(id1), "resume without vMMU succeeded");
    CHECK(cfs2.page_in_count() == 0, "no page_in without vMMU");

    printf("PASS: all CFS ↔ vMMU integration tests passed\n");
    return 0;
}
