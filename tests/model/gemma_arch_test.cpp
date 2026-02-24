/**
 * @file gemma_arch_test.cpp
 * @brief Gemma architecture registry validation
 *
 * Phase 33: Multi-Architecture Expansion (Qwen2/Gemma/Mixtral).
 *
 * Tests:
 *   1. Architecture registry: gemma strategy lookup + op name resolution
 *   2. Weight naming: GGUF blk.N.xxx format
 *   3. GELU activation selection (key differentiator from LLaMA)
 */

#include "model/arch_registry.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

/* ---- Test 1: Gemma strategy in registry ---- */
static void test_gemma_registry() {
    std::printf("[gemma] Test 1: Registry lookup...\n");

    nf::nf_clear_arch_registry();
    nf::nf_register_llama();
    nf::nf_register_gemma();

    CHECK(nf::nf_arch_count() == 2);

    const auto* strat = nf::nf_find_arch("gemma");
    CHECK(strat != nullptr);
    CHECK(strat->weight_name != nullptr);
    CHECK(strat->attn_op_name != nullptr);
    CHECK(strat->rope_op_name != nullptr);
    CHECK(strat->ffn_op_name != nullptr);
    CHECK(strat->norm_op_name != nullptr);
    CHECK(strat->paged_attn_op_name != nullptr);

    /* Verify op names — GELU is the key difference */
    CHECK(std::strcmp(nf::nf_resolve_attn_op(strat), "flash_attention_cached") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, false), "rope_batch") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, true), "rope_batch_f16") == 0);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(strat), "gelu") == 0);
    CHECK(std::strcmp(nf::nf_resolve_norm_op(strat), "rms_norm") == 0);
    CHECK(std::strcmp(nf::nf_resolve_paged_attn_op(strat), "flash_attention_paged") == 0);

    /* Verify Gemma differs from LLaMA on FFN activation */
    const auto* llama = nf::nf_find_arch("llama");
    CHECK(llama != nullptr);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(llama), "silu") == 0);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(strat), "gelu") == 0);

    std::printf("  PASS\n");
}

/* ---- Test 2: Gemma weight naming ---- */
static void test_gemma_weight_naming() {
    std::printf("[gemma] Test 2: Weight naming...\n");

    const auto* strat = nf::nf_find_arch("gemma");
    CHECK(strat != nullptr);

    char buf[128];

    /* Global tensors */
    CHECK(std::strcmp(strat->weight_name(0, "token_embd", buf, sizeof(buf)),
                      "token_embd.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(0, "output_norm", buf, sizeof(buf)),
                      "output_norm.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(0, "output", buf, sizeof(buf)),
                      "output.weight") == 0);

    /* Per-layer tensors (blk.N.xxx format) */
    CHECK(std::strcmp(strat->weight_name(0, "attn_q", buf, sizeof(buf)),
                      "blk.0.attn_q.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(7, "ffn_up", buf, sizeof(buf)),
                      "blk.7.ffn_up.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(17, "attn_v", buf, sizeof(buf)),
                      "blk.17.attn_v.weight") == 0);

    std::printf("  PASS\n");
}

int main() {
    std::printf("=== Gemma Architecture Tests (Phase 33) ===\n");
    test_gemma_registry();
    test_gemma_weight_naming();
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
