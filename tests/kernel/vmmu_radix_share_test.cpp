/**
 * @file vmmu_radix_share_test.cpp
 * @brief Phase 36.2 — vMMU radix prefix sharing tests
 */

#include "neuralOS/kernel/vMMU.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== vMMU Radix Prefix Share Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);

    /* Create ContextHub with 1MB budget */
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    neuralOS::kernel::vMMU vmmu(&kv, &hub);

    /* Insert some cached entries into ContextHub */
    std::vector<int32_t> key1 = {1, 2, 3, 4, 5};
    std::vector<int32_t> key2 = {1, 2, 3, 6, 7};
    std::vector<int32_t> key3 = {10, 20, 30};

    /* Allocate sequences and register in hub */
    uint32_t seq1 = kv.alloc_sequence();
    uint32_t seq2 = kv.alloc_sequence();
    uint32_t seq3 = kv.alloc_sequence();
    CHECK(seq1 != UINT32_MAX, "alloc seq1");
    CHECK(seq2 != UINT32_MAX, "alloc seq2");
    CHECK(seq3 != UINT32_MAX, "alloc seq3");

    /* Put entries (using dummy tensors — TensorView default is invalid but OK for test) */
    nf::TensorView d1, d2, d3;
    hub.put(key1, "agent1", std::move(d1), 0, seq1);
    hub.put(key2, "agent2", std::move(d2), 0, seq2);
    hub.put(key3, "agent3", std::move(d3), 0, seq3);

    /* Test 1: Query with prefix [1,2,3,4,5] — exact match on key1 */
    auto match = vmmu.radix_prefix_share({1, 2, 3, 4, 5});
    CHECK(match.found, "found match for [1,2,3,4,5]");
    CHECK(match.match_len == 5, "exact match length 5");
    CHECK(match.seq_id == seq1, "matched seq1");

    /* Test 2: Query with prefix [1,2,3,4,5,6,7,8] — best match is key1 (len 5) */
    match = vmmu.radix_prefix_share({1, 2, 3, 4, 5, 6, 7, 8});
    CHECK(match.found, "found match for extended query");
    CHECK(match.match_len == 5, "best prefix match is 5");

    /* Test 3: Query with prefix [1,2,3,6,7,8] — best match is key2 (len 5) */
    match = vmmu.radix_prefix_share({1, 2, 3, 6, 7, 8});
    CHECK(match.found, "found match for key2 prefix");
    CHECK(match.match_len == 5, "key2 match length 5");
    CHECK(match.seq_id == seq2, "matched seq2");

    /* Test 4: Query with prefix [10,20] — partial match on key3 (len 2) */
    match = vmmu.radix_prefix_share({10, 20});
    CHECK(match.found, "found partial match");
    CHECK(match.match_len == 2, "partial match length 2");
    CHECK(match.seq_id == seq3, "matched seq3");

    /* Test 5: Query with no matching prefix */
    match = vmmu.radix_prefix_share({99, 100, 101});
    CHECK(!match.found || match.match_len == 0, "no match for unrelated prefix");

    /* Test 6: Empty query */
    match = vmmu.radix_prefix_share({});
    CHECK(match.match_len == 0, "empty query returns 0 match");

    /* Test 7: vMMU without hub */
    neuralOS::kernel::vMMU vmmu_no_hub(&kv, nullptr);
    match = vmmu_no_hub.radix_prefix_share({1, 2, 3});
    CHECK(!match.found, "no hub returns no match");

    printf("PASS: all vMMU radix prefix share tests passed\n");
    return 0;
}
