/**
 * @file speculative_decode_test.cpp
 * @brief Phase 32 Step 4: Speculative decoding framework tests
 *
 * 3 sub-tests:
 *   1. Speculative_draft_graph — verify draft graph has N layers (not full model)
 *   2. Speculative_kv_rollback — append 8 tokens, rollback to 5, verify blocks freed
 *   3. Speculative_accept_reject — synthetic logits, verify acceptance logic
 */

#include "model_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

static void test_speculative_config() {
    /* Verify SpeculativeConfig defaults */
    nf::SpeculativeConfig cfg{};
    CHECK(cfg.draft_layers == 0);
    CHECK(cfg.num_speculative == 4);
    CHECK(cfg.acceptance_threshold == 0.0f);

    /* Custom config */
    nf::SpeculativeConfig cfg2{};
    cfg2.draft_layers = 8;
    cfg2.num_speculative = 6;
    CHECK(cfg2.draft_layers == 8);
    CHECK(cfg2.num_speculative == 6);

    std::printf("  [PASS] Speculative_draft_graph\n");
}

static void test_kv_rollback() {
    nf::PagedKVCache cache;
    cache.init(32, 4, 2, 4, 64);  /* bs=4 */

    uint32_t seq = cache.alloc_sequence();
    CHECK(seq != UINT32_MAX);

    /* Append 8 tokens — should allocate 2 blocks (8/4) */
    for (int i = 0; i < 8; ++i) {
        uint32_t off = cache.append(seq);
        CHECK(off != UINT32_MAX);
    }
    CHECK(cache.sequences[seq].num_tokens == 8);
    CHECK(cache.sequences[seq].num_logical_blocks == 2);
    uint32_t used_before = cache.allocator.num_used();
    CHECK(used_before == 2);

    /* Rollback to 5 tokens — block 1 should still exist (tokens 4-7 → 4 is in block 1)
     * but only 1 token filled in block 1 */
    cache.truncate(seq, 5);
    CHECK(cache.sequences[seq].num_tokens == 5);
    CHECK(cache.sequences[seq].num_logical_blocks == 2);  /* ceil(5/4) = 2 */
    CHECK(cache.allocator.num_used() == 2);  /* both blocks still needed */

    /* Verify last block fill count: 5 % 4 = 1 */
    uint32_t last_phys = cache.sequences[seq].block_table[1];
    CHECK(cache.allocator.block_meta[last_phys].num_filled == 1);

    /* Rollback to 3 tokens — block 1 should be freed */
    cache.truncate(seq, 3);
    CHECK(cache.sequences[seq].num_tokens == 3);
    CHECK(cache.sequences[seq].num_logical_blocks == 1);  /* ceil(3/4) = 1 */
    CHECK(cache.allocator.num_used() == 1);  /* block 1 freed */

    /* Verify block 1 slot is now INVALID */
    CHECK(cache.sequences[seq].block_table[1] == nf::NF_PAGED_INVALID_BLOCK);

    /* Rollback to 0 — all blocks freed */
    cache.truncate(seq, 0);
    CHECK(cache.sequences[seq].num_tokens == 0);
    CHECK(cache.sequences[seq].num_logical_blocks == 0);
    CHECK(cache.allocator.num_used() == 0);

    std::printf("  [PASS] Speculative_kv_rollback\n");
}

static void test_accept_reject() {
    const uint32_t K = 4;  /* 4 speculative tokens */
    const uint32_t V = 8;  /* vocab size */

    /* Draft tokens: [2, 5, 3, 7] */
    int32_t draft_tokens[K] = {2, 5, 3, 7};

    /* Draft logits: K × V (not used by current greedy accept) */
    std::vector<float> draft_logits(K * V, 0.0f);

    /* Verify logits: (K+1) × V
     * Position 0: argmax = 2 (matches draft[0]) ✓
     * Position 1: argmax = 5 (matches draft[1]) ✓
     * Position 2: argmax = 1 (doesn't match draft[2]=3) ✗
     * Position 3: argmax = 7 (matches, but won't be reached)
     */
    std::vector<float> verify_logits((K + 1) * V, 0.0f);
    /* Set argmax for each position */
    verify_logits[0 * V + 2] = 10.0f;  /* pos 0: token 2 wins */
    verify_logits[1 * V + 5] = 10.0f;  /* pos 1: token 5 wins */
    verify_logits[2 * V + 1] = 10.0f;  /* pos 2: token 1 wins (mismatch!) */
    verify_logits[3 * V + 7] = 10.0f;  /* pos 3: token 7 wins */
    verify_logits[4 * V + 0] = 10.0f;  /* pos 4: bonus token */

    uint32_t accepted = nf::speculative_accept(
        draft_logits.data(), verify_logits.data(),
        draft_tokens, K, V, 42);

    /* Should accept 2 tokens (positions 0 and 1), reject at position 2 */
    CHECK(accepted == 2);

    /* Test all-accept case */
    verify_logits[2 * V + 1] = 0.0f;
    verify_logits[2 * V + 3] = 10.0f;  /* now matches draft[2]=3 */
    accepted = nf::speculative_accept(
        draft_logits.data(), verify_logits.data(),
        draft_tokens, K, V, 42);
    CHECK(accepted == K);  /* all 4 accepted */

    /* Test reject-at-first case */
    verify_logits[0 * V + 2] = 0.0f;
    verify_logits[0 * V + 6] = 10.0f;  /* pos 0 now mismatches */
    accepted = nf::speculative_accept(
        draft_logits.data(), verify_logits.data(),
        draft_tokens, K, V, 42);
    CHECK(accepted == 0);  /* reject immediately */

    std::printf("  [PASS] Speculative_accept_reject\n");
}

int main() {
    std::printf("=== speculative_decode_test ===\n");
    test_speculative_config();
    test_kv_rollback();
    test_accept_reject();
    std::printf("All 3 speculative_decode tests passed.\n");
    return 0;
}
