/**
 * @file arch_registry_test.cpp
 * @brief Phase 25 — Architecture Registry Unit Test
 *
 * Verifies register/lookup/fallback for the strategy pattern.
 */

#include "arch_registry.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)

static void test_register_and_find() {
    std::printf("  test_register_and_find...\n");
    nf::nf_clear_arch_registry();

    nf::nf_register_llama();
    CHECK(nf::nf_arch_count() == 1);

    auto* strat = nf::nf_find_arch("llama");
    CHECK(strat != nullptr);
    CHECK(strat->weight_name != nullptr);

    /* Verify weight naming */
    char buf[128];
    strat->weight_name(0, "token_embd", buf, sizeof(buf));
    CHECK(std::strcmp(buf, "token_embd.weight") == 0);

    strat->weight_name(3, "attn_q", buf, sizeof(buf));
    CHECK(std::strcmp(buf, "blk.3.attn_q.weight") == 0);

    strat->weight_name(0, "output_norm", buf, sizeof(buf));
    CHECK(std::strcmp(buf, "output_norm.weight") == 0);

    strat->weight_name(0, "output", buf, sizeof(buf));
    CHECK(std::strcmp(buf, "output.weight") == 0);

    std::printf("    register_and_find verified ✓\n");
}

static void test_lookup_missing() {
    std::printf("  test_lookup_missing...\n");
    nf::nf_clear_arch_registry();

    auto* strat = nf::nf_find_arch("mistral");
    CHECK(strat == nullptr);

    /* Register llama, still can't find mistral */
    nf::nf_register_llama();
    strat = nf::nf_find_arch("mistral");
    CHECK(strat == nullptr);

    /* But llama is found */
    strat = nf::nf_find_arch("llama");
    CHECK(strat != nullptr);

    std::printf("    lookup_missing verified ✓\n");
}

static void test_multiple_archs() {
    std::printf("  test_multiple_archs...\n");
    nf::nf_clear_arch_registry();

    /* Register two architectures */
    nf::nf_register_llama();

    /* Custom "test" arch */
    auto test_weight_name = [](uint32_t layer, const char* component,
                                char* buf, size_t buf_len) -> const char* {
        std::snprintf(buf, buf_len, "test.%u.%s", layer, component);
        return buf;
    };
    nf::nf_arch_strategy test_strat{};
    test_strat.weight_name = test_weight_name;
    CHECK(nf::nf_register_arch("test_arch", test_strat));

    CHECK(nf::nf_arch_count() == 2);

    auto* llama = nf::nf_find_arch("llama");
    auto* test = nf::nf_find_arch("test_arch");
    CHECK(llama != nullptr);
    CHECK(test != nullptr);
    CHECK(llama != test);

    char buf[128];
    test->weight_name(5, "ffn_gate", buf, sizeof(buf));
    CHECK(std::strcmp(buf, "test.5.ffn_gate") == 0);

    std::printf("    multiple_archs verified ✓\n");
}

static void test_max_capacity() {
    std::printf("  test_max_capacity...\n");
    nf::nf_clear_arch_registry();

    /* Fill to max */
    nf::nf_arch_strategy dummy{};
    dummy.weight_name = nf::llama_weight_name;
    for (int i = 0; i < nf::NF_MAX_ARCHS; ++i) {
        CHECK(nf::nf_register_arch("dummy", dummy));
    }
    CHECK(nf::nf_arch_count() == nf::NF_MAX_ARCHS);

    /* One more should fail */
    CHECK(!nf::nf_register_arch("overflow", dummy));

    std::printf("    max_capacity verified ✓\n");
}

int main() {
    std::printf("arch_registry_test: Phase 25 — Architecture Registry\n");

    test_register_and_find();
    test_lookup_missing();
    test_multiple_archs();
    test_max_capacity();

    std::printf("OK: all Phase 25 arch registry tests passed\n");
    return 0;
}
