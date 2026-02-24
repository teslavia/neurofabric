/**
 * @file cfs_migration_test.cpp
 * @brief Phase 42A.3 — CFS ↔ KVMigration integration tests
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
    printf("=== CFS ↔ KVMigration Integration Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);

    nf::RequestScheduler sched;

    nf::InferenceRequest r1;
    r1.prompt_tokens = {1, 2, 3};
    r1.vtc_weight = 1;
    uint32_t id1 = sched.submit(r1);

    sched.requests[0].state = nf::RequestState::DECODE;
    sched.requests[0].seq_id = kv.alloc_sequence();
    auto& seq = kv.sequences[sched.requests[0].seq_id];
    seq.block_table[0] = kv.allocator.alloc_block();
    seq.num_logical_blocks = 1;
    seq.num_tokens = 3;

    /* Test 1: Migrate without migrator → returns false */
    neuralOS::kernel::CFS cfs(&sched, &kv);
    CHECK(!cfs.migrate_request(id1, "192.168.1.70:9999"),
          "migrate without migrator returns false");

    /* Test 2: Migrate with migrator → succeeds */
    neuralOS::mesh::KVMigrator migrator;
    cfs.set_migrator(&migrator);

    CHECK(cfs.migrate_request(id1, "192.168.1.70:9999"),
          "migrate with migrator succeeded");
    CHECK(cfs.num_migrations() == 1, "1 migration recorded");
    CHECK(cfs.last_migration_target() == "192.168.1.70:9999",
          "target node recorded");
    CHECK(sched.requests[0].state == nf::RequestState::PREEMPTED,
          "request preempted after migration");

    /* Test 3: Migrate invalid request → returns false */
    CHECK(!cfs.migrate_request(9999, "node2"), "invalid req returns false");

    printf("PASS: all CFS ↔ KVMigration integration tests passed\n");
    return 0;
}
