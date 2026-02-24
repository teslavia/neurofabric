/**
 * @file cfs_vtc_test.cpp
 * @brief Phase 36.3 — CFS Virtual Token Counter + rebalance tests
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
    printf("=== CFS VTC Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);

    nf::RequestScheduler sched;

    /* Submit 3 requests with different weights */
    nf::InferenceRequest r1;
    r1.prompt_tokens = {1, 2, 3, 4};
    r1.max_new_tokens = 16;
    r1.vtc_weight = 1;
    uint32_t id1 = sched.submit(r1);

    nf::InferenceRequest r2;
    r2.prompt_tokens = {5, 6, 7, 8};
    r2.max_new_tokens = 16;
    r2.vtc_weight = 2;  /* higher weight = more share */
    uint32_t id2 = sched.submit(r2);

    nf::InferenceRequest r3;
    r3.prompt_tokens = {9, 10};
    r3.max_new_tokens = 8;
    r3.vtc_weight = 1;
    uint32_t id3 = sched.submit(r3);

    CHECK(id1 != UINT32_MAX && id2 != UINT32_MAX && id3 != UINT32_MAX,
          "all requests submitted");

    /* Transition to DECODE state */
    for (uint32_t i = 0; i < sched.num_requests; ++i) {
        sched.requests[i].state = nf::RequestState::DECODE;
        sched.requests[i].seq_id = kv.alloc_sequence();
    }

    neuralOS::L2::CFS cfs(&sched, &kv);

    /* Account tokens: r1 processes 10, r2 processes 10, r3 processes 10 */
    cfs.account_tokens(id1, 10, false);
    cfs.account_tokens(id2, 10, false);
    cfs.account_tokens(id3, 10, false);

    /* r2 has weight=2, so its vruntime should be lower */
    uint64_t vrt1 = cfs.get_vruntime(id1);
    uint64_t vrt2 = cfs.get_vruntime(id2);
    uint64_t vrt3 = cfs.get_vruntime(id3);
    CHECK(vrt2 < vrt1, "higher weight → lower vruntime");
    CHECK(vrt1 == vrt3, "same weight → same vruntime");

    /* Prefill costs more */
    cfs.account_tokens(id1, 5, true);  /* prefill_weight=2.0 */
    uint64_t vrt1_after = cfs.get_vruntime(id1);
    CHECK(vrt1_after > vrt1, "prefill increases vruntime");

    /* Rebalance should pick lowest vruntime first */
    auto result = cfs.rebalance();
    CHECK(!result.selected_req_ids.empty(), "rebalance selected some requests");
    /* r2 should be first (lowest vruntime) */
    CHECK(result.selected_req_ids[0] == id2, "r2 selected first (lowest vrt)");

    /* min_vruntime */
    uint64_t min_vrt = cfs.min_vruntime();
    CHECK(min_vrt == vrt2, "min_vruntime matches r2");

    printf("PASS: all CFS VTC tests passed\n");
    return 0;
}
