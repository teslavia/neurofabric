/**
 * @file mixtral_arch_test.cpp
 * @brief Mixtral (MoE) architecture registry validation
 *
 * Phase 33: Multi-Architecture Expansion (Qwen2/Gemma/Mixtral).
 *
 * Tests:
 *   1. Architecture registry: mixtral strategy lookup + op name resolution
 *   2. Weight naming: GGUF blk.N.xxx format
 *   3. MoE config fields in nf_arch_config
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

/* ---- Test 1: Mixtral strategy in registry ---- */
static void test_mixtral_registry() {
    std::printf("[mixtral] Test 1: Registry lookup...\n");

    nf::nf_clear_arch_registry();
    nf::nf_register_llama();
    nf::nf_register_mistral();
    nf::nf_register_mixtral();

    CHECK(nf::nf_arch_count() == 3);

    const auto* strat = nf::nf_find_arch("mixtral");
    CHECK(strat != nullptr);
    CHECK(strat->weight_name != nullptr);
    CHECK(strat->attn_op_name != nullptr);
    CHECK(strat->rope_op_name != nullptr);
    CHECK(strat->ffn_op_name != nullptr);
    CHECK(strat->norm_op_name != nullptr);
    CHECK(strat->paged_attn_op_name != nullptr);

    /* Verify op names — same as Mistral (SiLU, RMS norm) */
    CHECK(std::strcmp(nf::nf_resolve_attn_op(strat), "flash_attention_cached") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, false), "rope_batch") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, true), "rope_batch_f16") == 0);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(strat), "silu") == 0);
    CHECK(std::strcmp(nf::nf_resolve_norm_op(strat), "rms_norm") == 0);
    CHECK(std::strcmp(nf::nf_resolve_paged_attn_op(strat), "flash_attention_paged") == 0);

    std::printf("  PASS\n");
}

/* ---- Test 2: Mixtral weight naming ---- */
static void test_mixtral_weight_naming() {
    std::printf("[mixtral] Test 2: Weight naming...\n");

    const auto* strat = nf::nf_find_arch("mixtral");
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
    CHECK(std::strcmp(strat->weight_name(3, "ffn_gate", buf, sizeof(buf)),
                      "blk.3.ffn_gate.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(31, "attn_output", buf, sizeof(buf)),
                      "blk.31.attn_output.weight") == 0);

    std::printf("  PASS\n");
}

/* ---- Test 3: MoE config fields ---- */
static void test_mixtral_moe_config() {
    std::printf("[mixtral] Test 3: MoE config fields...\n");

    /* Verify nf_arch_config supports MoE fields */
    nf::nf_arch_config cfg{};
    cfg.arch_name = "mixtral";
    cfg.n_experts = 8;
    cfg.n_experts_used = 2;
    cfg.sliding_window = 4096;

    CHECK(cfg.n_experts == 8);
    CHECK(cfg.n_experts_used == 2);
    CHECK(cfg.sliding_window == 4096);

    std::printf("  PASS\n");
}

int main() {
    std::printf("=== Mixtral Architecture Tests (Phase 33) ===\n");
    test_mixtral_registry();
    test_mixtral_weight_naming();
    test_mixtral_moe_config();
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
