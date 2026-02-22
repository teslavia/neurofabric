/**
 * @file qwen2_arch_test.cpp
 * @brief Qwen2 architecture registry validation
 *
 * Phase 33: Multi-Architecture Expansion (Qwen2/Gemma/Mixtral).
 *
 * Tests:
 *   1. Architecture registry: qwen2 strategy lookup + op name resolution
 *   2. Weight naming: GGUF blk.N.xxx format (same as LLaMA)
 *   3. FP16 RoPE op selection
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

/* ---- Test 1: Qwen2 strategy in registry ---- */
static void test_qwen2_registry() {
    std::printf("[qwen2] Test 1: Registry lookup...\n");

    nf::nf_clear_arch_registry();
    nf::nf_register_llama();
    nf::nf_register_mistral();
    nf::nf_register_phi3();
    nf::nf_register_qwen2();

    CHECK(nf::nf_arch_count() == 4);

    const auto* strat = nf::nf_find_arch("qwen2");
    CHECK(strat != nullptr);
    CHECK(strat->weight_name != nullptr);
    CHECK(strat->attn_op_name != nullptr);
    CHECK(strat->rope_op_name != nullptr);
    CHECK(strat->ffn_op_name != nullptr);
    CHECK(strat->norm_op_name != nullptr);
    CHECK(strat->paged_attn_op_name != nullptr);

    /* Verify op names */
    CHECK(std::strcmp(nf::nf_resolve_attn_op(strat), "flash_attention_cached") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, false), "rope_batch") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, true), "rope_batch_f16") == 0);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(strat), "silu") == 0);
    CHECK(std::strcmp(nf::nf_resolve_norm_op(strat), "rms_norm") == 0);
    CHECK(std::strcmp(nf::nf_resolve_paged_attn_op(strat), "flash_attention_paged") == 0);

    std::printf("  PASS\n");
}

/* ---- Test 2: Qwen2 weight naming ---- */
static void test_qwen2_weight_naming() {
    std::printf("[qwen2] Test 2: Weight naming...\n");

    const auto* strat = nf::nf_find_arch("qwen2");
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
    CHECK(std::strcmp(strat->weight_name(5, "ffn_gate", buf, sizeof(buf)),
                      "blk.5.ffn_gate.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(31, "attn_output", buf, sizeof(buf)),
                      "blk.31.attn_output.weight") == 0);

    std::printf("  PASS\n");
}

int main() {
    std::printf("=== Qwen2 Architecture Tests (Phase 33) ===\n");
    test_qwen2_registry();
    test_qwen2_weight_naming();
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
