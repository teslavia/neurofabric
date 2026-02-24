/**
 * @file kv_migration_test.cpp
 * @brief Phase 38.3 — KV cache cross-node migration tests
 */

#include "neuralOS/mesh/kv_migration.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== KV Migration Test ===\n");

    const uint32_t BLOCK_SIZE = 4;
    const uint32_t NUM_BLOCKS = 32;
    const uint32_t BYTES_PER_BLOCK = BLOCK_SIZE * 64 * sizeof(float);

    /* Source KV cache */
    nf::PagedKVCache src_kv;
    src_kv.init(NUM_BLOCKS, BLOCK_SIZE, 2, 4, 64);

    /* Destination KV cache */
    nf::PagedKVCache dst_kv;
    dst_kv.init(NUM_BLOCKS, BLOCK_SIZE, 2, 4, 64);

    /* Allocate source sequence with 8 tokens */
    uint32_t src_seq = src_kv.alloc_sequence();
    CHECK(src_seq != UINT32_MAX, "alloc src seq");
    for (int i = 0; i < 8; ++i) {
        uint32_t off = src_kv.append(src_seq);
        CHECK(off != UINT32_MAX, "append token");
    }
    CHECK(src_kv.sequences[src_seq].num_tokens == 8, "8 tokens");
    CHECK(src_kv.sequences[src_seq].num_logical_blocks == 2, "2 blocks");

    /* Simulate block data */
    std::vector<uint8_t> src_mem(NUM_BLOCKS * BYTES_PER_BLOCK, 0);
    std::vector<uint8_t> dst_mem(NUM_BLOCKS * BYTES_PER_BLOCK, 0);

    /* Fill source blocks with pattern */
    uint32_t phys0 = src_kv.sequences[src_seq].block_table[0];
    std::memset(src_mem.data() + phys0 * BYTES_PER_BLOCK, 0xBB, BYTES_PER_BLOCK);

    neuralOS::L5::KVMigrator migrator;

    /* Test 1: Serialize */
    auto wire = migrator.serialize_kv(src_kv, src_seq, src_mem.data(), BYTES_PER_BLOCK);
    CHECK(wire.num_tokens == 8, "wire has 8 tokens");
    CHECK(wire.num_blocks == 2, "wire has 2 blocks");
    CHECK(wire.blocks.size() == 2, "2 serialized blocks");
    CHECK(wire.blocks[0].data.size() == BYTES_PER_BLOCK, "block data size correct");

    /* Test 2: Deserialize */
    uint32_t dst_seq = migrator.deserialize_kv(dst_kv, wire, dst_mem.data(), BYTES_PER_BLOCK);
    CHECK(dst_seq != UINT32_MAX, "deserialize succeeded");
    CHECK(dst_kv.sequences[dst_seq].num_tokens == 8, "dst has 8 tokens");
    CHECK(dst_kv.sequences[dst_seq].num_logical_blocks == 2, "dst has 2 blocks");

    /* Verify data was copied */
    uint32_t dst_phys0 = dst_kv.sequences[dst_seq].block_table[0];
    CHECK(dst_mem[dst_phys0 * BYTES_PER_BLOCK] == 0xBB, "data migrated correctly");

    /* Test 3: Live migration */
    nf::PagedKVCache dst2_kv;
    dst2_kv.init(NUM_BLOCKS, BLOCK_SIZE, 2, 4, 64);
    std::vector<uint8_t> dst2_mem(NUM_BLOCKS * BYTES_PER_BLOCK, 0);

    uint32_t migrated = migrator.live_migrate(
        src_kv, src_seq, dst2_kv,
        src_mem.data(), dst2_mem.data(), BYTES_PER_BLOCK);
    CHECK(migrated != UINT32_MAX, "live_migrate succeeded");
    CHECK(dst2_kv.sequences[migrated].num_tokens == 8, "migrated 8 tokens");

    /* Source sequence should still be intact */
    CHECK(src_kv.sequences[src_seq].num_tokens == 8, "source unchanged");

    /* Test 4: Serialize invalid sequence */
    auto empty = migrator.serialize_kv(src_kv, 999, nullptr, 0);
    CHECK(empty.num_tokens == 0, "invalid seq returns empty");

    printf("PASS: all KV migration tests passed\n");
    return 0;
}
