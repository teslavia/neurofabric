/**
 * @file vmmu_pageout_test.cpp
 * @brief Phase 36.2 — vMMU page-out / page-in tests
 */

#include "neuralOS/kernel/vMMU.hpp"
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
    printf("=== vMMU Page-Out/In Test ===\n");

    const uint32_t BLOCK_SIZE = 4;
    const uint32_t NUM_BLOCKS = 32;
    const uint32_t BYTES_PER_BLOCK = BLOCK_SIZE * 64 * sizeof(float);

    nf::PagedKVCache kv;
    kv.init(NUM_BLOCKS, BLOCK_SIZE, 2, 4, 64);

    neuralOS::L2::vMMU vmmu(&kv, nullptr);

    /* Allocate sequence with 12 tokens (3 blocks) */
    uint32_t seq = kv.alloc_sequence();
    CHECK(seq != UINT32_MAX, "alloc sequence");
    for (int i = 0; i < 12; ++i) {
        uint32_t off = kv.append(seq);
        CHECK(off != UINT32_MAX, "append token");
    }
    CHECK(kv.sequences[seq].num_logical_blocks == 3, "3 blocks allocated");

    /* Simulate block data */
    std::vector<uint8_t> device_mem(NUM_BLOCKS * BYTES_PER_BLOCK, 0);
    /* Fill block 0 with pattern */
    uint32_t phys0 = kv.sequences[seq].block_table[0];
    std::memset(device_mem.data() + phys0 * BYTES_PER_BLOCK, 0xAA, BYTES_PER_BLOCK);

    uint32_t used_before = kv.allocator.num_used();

    /* Page out block 0 */
    uint32_t paged = vmmu.page_out(seq, 0, 1, device_mem.data(), BYTES_PER_BLOCK);
    CHECK(paged == 1, "paged out 1 block");
    CHECK(vmmu.num_paged_out() == 1, "1 block in page-out store");
    CHECK(vmmu.is_paged_out(seq, 0), "block 0 is paged out");
    CHECK(kv.sequences[seq].block_table[0] == nf::NF_PAGED_INVALID_BLOCK,
          "block table entry invalidated");
    CHECK(kv.allocator.num_used() == used_before - 1, "freed 1 device block");

    /* Page out blocks 1-2 */
    paged = vmmu.page_out(seq, 1, 2, device_mem.data(), BYTES_PER_BLOCK);
    CHECK(paged == 2, "paged out 2 more blocks");
    CHECK(vmmu.num_paged_out() == 3, "3 blocks in page-out store");

    /* Page in block 0 */
    uint32_t paged_in = vmmu.page_in(seq, 0, 1, device_mem.data(), BYTES_PER_BLOCK);
    CHECK(paged_in == 1, "paged in 1 block");
    CHECK(!vmmu.is_paged_out(seq, 0), "block 0 no longer paged out");
    uint32_t new_phys = kv.sequences[seq].block_table[0];
    CHECK(new_phys != nf::NF_PAGED_INVALID_BLOCK, "block 0 has new phys");

    /* Verify data was restored */
    uint8_t* restored = device_mem.data() + new_phys * BYTES_PER_BLOCK;
    CHECK(restored[0] == 0xAA, "data restored correctly");

    /* Page in remaining */
    paged_in = vmmu.page_in(seq, 1, 2, device_mem.data(), BYTES_PER_BLOCK);
    CHECK(paged_in == 2, "paged in 2 blocks");
    CHECK(vmmu.num_paged_out() == 0, "no blocks paged out");

    /* Pressure callback */
    bool pressure_fired = false;

    /* Use a very low threshold so current usage triggers it */
    neuralOS::L2::vMMU::Config pcfg;
    pcfg.pressure_threshold = 0.01f;  /* 1% — any usage triggers */
    neuralOS::L2::vMMU vmmu2(&kv, nullptr, pcfg);
    vmmu2.set_pressure_callback([&](uint64_t used, uint64_t total) {
        pressure_fired = true;
    });
    vmmu2.check_pressure();
    CHECK(pressure_fired, "pressure callback fired");

    printf("PASS: all vMMU page-out/in tests passed\n");
    return 0;
}
