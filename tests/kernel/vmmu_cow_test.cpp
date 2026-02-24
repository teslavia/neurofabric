/**
 * @file vmmu_cow_test.cpp
 * @brief Phase 36.2 — vMMU CoW fork/write tests
 */

#include "neuralOS/kernel/vMMU.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== vMMU CoW Test ===\n");

    /* Setup: PagedKVCache with 32 blocks of size 4 */
    nf::PagedKVCache kv;
    kv.init(32, 4, 2, 4, 64);

    /* No ContextHub needed for CoW tests */
    neuralOS::kernel::vMMU vmmu(&kv, nullptr);

    /* Allocate a source sequence and fill some tokens */
    uint32_t src = kv.alloc_sequence();
    CHECK(src != UINT32_MAX, "alloc src sequence");
    for (int i = 0; i < 8; ++i) {
        uint32_t off = kv.append(src);
        CHECK(off != UINT32_MAX, "append token to src");
    }
    CHECK(kv.sequences[src].num_tokens == 8, "src has 8 tokens");
    CHECK(kv.sequences[src].num_logical_blocks == 2, "src has 2 blocks");

    /* Fork the sequence */
    uint32_t dst = vmmu.fork_sequence(src);
    CHECK(dst != UINT32_MAX, "fork_sequence succeeded");
    CHECK(vmmu.has_fork_parent(dst), "dst has fork parent");
    CHECK(kv.sequences[dst].num_tokens == 8, "dst has 8 tokens");
    CHECK(kv.sequences[dst].num_logical_blocks == 2, "dst has 2 blocks");

    /* Verify blocks are shared (ref_count > 1) */
    uint32_t phys0_src = kv.sequences[src].block_table[0];
    uint32_t phys0_dst = kv.sequences[dst].block_table[0];
    CHECK(phys0_src == phys0_dst, "block 0 shared between src and dst");
    CHECK(kv.allocator.block_meta[phys0_src].ref_count == 2, "ref_count == 2");

    /* CoW write on dst block 0 — should trigger copy */
    uint32_t new_phys = vmmu.cow_write(dst, 0);
    CHECK(new_phys != nf::NF_PAGED_INVALID_BLOCK, "cow_write succeeded");
    CHECK(new_phys != phys0_src, "cow_write allocated new block");
    CHECK(kv.allocator.block_meta[phys0_src].ref_count == 1, "src ref back to 1");
    CHECK(kv.allocator.block_meta[new_phys].ref_count == 1, "new block ref == 1");
    CHECK(kv.sequences[dst].block_table[0] == new_phys, "dst block table updated");

    /* CoW write on non-shared block — should return same block */
    uint32_t same = vmmu.cow_write(dst, 0);
    CHECK(same == new_phys, "non-shared cow_write returns same block");

    /* Multiple forks */
    uint32_t fork2 = vmmu.fork_sequence(src);
    uint32_t fork3 = vmmu.fork_sequence(src);
    CHECK(fork2 != UINT32_MAX && fork3 != UINT32_MAX, "multiple forks");
    uint32_t phys1_src = kv.sequences[src].block_table[1];
    /* block 1 shared by: src, dst (from first fork), fork2, fork3 = 4 */
    CHECK(kv.allocator.block_meta[phys1_src].ref_count == 4, "4-way shared block 1");

    printf("PASS: all vMMU CoW tests passed\n");
    return 0;
}
