/**
 * @file mistral_arch_test.cpp
 * @brief Mistral architecture end-to-end validation
 *
 * Phase 31: Multi-Architecture DAG & KV Cache Intelligence.
 *
 * Tests:
 *   1. Architecture registry: mistral strategy lookup + op name resolution
 *   2. KV cache policy: sliding window auto-config from GGUF metadata
 *   3. KV write offset: ring buffer correctness across window boundary
 *   4. Attention range: bounded by window_size
 *   5. (Optional) Real model decode with NF_TEST_GGUF_PATH_MISTRAL
 */

#include "model/arch_registry.hpp"
#include "model/kv_cache_policy.hpp"
#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/engine/PipelineEngine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

/* ---- Test 1: Mistral strategy in registry ---- */
static void test_mistral_registry() {
    std::printf("[mistral] Test 1: Registry lookup...\n");

    nf::nf_clear_arch_registry();
    nf::nf_register_llama();
    nf::nf_register_mistral();
    nf::nf_register_phi3();

    CHECK(nf::nf_arch_count() == 3);

    const auto* strat = nf::nf_find_arch("mistral");
    CHECK(strat != nullptr);
    CHECK(strat->weight_name != nullptr);
    CHECK(strat->attn_op_name != nullptr);
    CHECK(strat->rope_op_name != nullptr);
    CHECK(strat->ffn_op_name != nullptr);
    CHECK(strat->norm_op_name != nullptr);

    /* Verify op names */
    CHECK(std::strcmp(nf::nf_resolve_attn_op(strat), "flash_attention_cached") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, false), "rope_batch") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, true), "rope_batch_f16") == 0);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(strat), "silu") == 0);
    CHECK(std::strcmp(nf::nf_resolve_norm_op(strat), "rms_norm") == 0);

    /* Verify weight naming (same as LLaMA) */
    char buf[128];
    CHECK(std::strcmp(strat->weight_name(0, "token_embd", buf, sizeof(buf)),
                      "token_embd.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(5, "attn_q", buf, sizeof(buf)),
                      "blk.5.attn_q.weight") == 0);

    std::printf("  PASS\n");
}

/* ---- Test 2: Sliding window KV policy auto-config ---- */
static void test_sliding_window_policy() {
    std::printf("[mistral] Test 2: Sliding window KV policy...\n");

    /* Simulate GGUF model with sliding_window = 4096 (Mistral-7B default) */
    const uint32_t WINDOW = 4096;

    nf::nf_kv_cache_config cfg{};
    cfg.eviction = nf::NF_KV_EVICT_SLIDING;
    cfg.window_size = WINDOW;
    cfg.max_seq_len = 32768;
    auto policy = nf::nf_create_kv_policy(cfg);

    /* Write offset: ring buffer */
    CHECK(policy.write_offset(&policy, 0) == 0);
    CHECK(policy.write_offset(&policy, WINDOW - 1) == WINDOW - 1);
    CHECK(policy.write_offset(&policy, WINDOW) == 0);  /* wraps */
    CHECK(policy.write_offset(&policy, WINDOW + 100) == 100);

    /* Attention range: before window fills */
    uint32_t start, len;
    policy.attn_range(&policy, 100, &start, &len);
    CHECK(start == 0 && len == 101);

    /* Attention range: exactly at window boundary */
    policy.attn_range(&policy, WINDOW - 1, &start, &len);
    CHECK(start == 0 && len == WINDOW);

    /* Attention range: after window fills */
    policy.attn_range(&policy, WINDOW, &start, &len);
    CHECK(start == 1 && len == WINDOW);

    policy.attn_range(&policy, WINDOW + 1000, &start, &len);
    CHECK(start == 1001 && len == WINDOW);

    /* Effective length capped at window */
    CHECK(policy.effective_len(&policy, 100) == 101);
    CHECK(policy.effective_len(&policy, WINDOW - 1) == WINDOW);
    CHECK(policy.effective_len(&policy, WINDOW) == WINDOW);
    CHECK(policy.effective_len(&policy, 100000) == WINDOW);

    std::printf("  PASS\n");
}

/* ---- Test 3: KV cache policy integration with LlamaContext ---- */
static void test_context_kv_policy() {
    std::printf("[mistral] Test 3: Context KV policy integration...\n");

    /* Initialize Metal provider */
    nf_provider_vtable vt{};
    nf_provider prov = nullptr;
    nf_status st = nf_plugin_register(&vt, &prov);
    CHECK(st == NF_OK);

    nf_provider_mem_vtable mem_vt{};
    nf_provider mem_prov = nullptr;
    nf_plugin_register_mem(&mem_vt, &mem_prov);

    st = vt.init(prov);
    CHECK(st == NF_OK);

    nf::PipelineEngine engine(2);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);
    engine.register_provider_mem(0, mem_vt);

    /* Register architectures */
    nf::nf_clear_arch_registry();
    nf::nf_register_llama();
    nf::nf_register_mistral();

    /* Create a mock GGUFModel with Mistral-like config */
    nf::GGUFModel model{};
    model.architecture = "mistral";
    model.dim = 64;
    model.n_layers = 1;
    model.n_heads = 4;
    model.n_kv_heads = 2;
    model.ff_dim = 128;
    model.vocab_size = 32;
    model.max_seq = 256;
    model.rope_theta = 10000.0f;
    model.rms_norm_eps = 1e-5f;
    model.sliding_window = 64;  /* small window for testing */

    /* Create synthetic weight tensors */
    const uint32_t D = model.dim;
    const uint32_t HD = D / model.n_heads;
    const uint32_t KV_DIM = model.n_kv_heads * HD;
    const uint32_t FF = model.ff_dim;
    const uint32_t V = model.vocab_size;

    /* Allocate F32 weight data */
    auto make_tensor = [&](const std::string& name, uint64_t rows, uint64_t cols) {
        nf::GGUFTensorInfo ti{};
        ti.gguf_dtype = 0; /* F32 */
        ti.ndim = 2;
        ti.shape[0] = cols; ti.shape[1] = rows;
        ti.n_elements = rows * cols;
        ti.byte_size = ti.n_elements * sizeof(float);
        ti.data = std::calloc(ti.n_elements, sizeof(float));
        /* Fill with small values */
        float* p = (float*)ti.data;
        for (uint64_t i = 0; i < ti.n_elements; ++i)
            p[i] = 0.001f * (float)(i % 100);
        model.tensors[name] = ti;
    };

    make_tensor("token_embd.weight", V, D);
    make_tensor("blk.0.attn_norm.weight", 1, D);
    make_tensor("blk.0.attn_q.weight", D, D);
    make_tensor("blk.0.attn_k.weight", KV_DIM, D);
    make_tensor("blk.0.attn_v.weight", KV_DIM, D);
    make_tensor("blk.0.attn_output.weight", D, D);
    make_tensor("blk.0.ffn_norm.weight", 1, D);
    make_tensor("blk.0.ffn_gate.weight", FF, D);
    make_tensor("blk.0.ffn_up.weight", FF, D);
    make_tensor("blk.0.ffn_down.weight", D, FF);
    make_tensor("output_norm.weight", 1, D);
    make_tensor("output.weight", V, D);

    /* Create context — should auto-detect sliding window from GGUF */
    auto ctx = nf::create_llama_context(engine, model, prov, vt, mem_vt,
                                         256, 16);
    CHECK(ctx != nullptr);

    /* Verify KV policy was auto-configured */
    CHECK(ctx->kv_policy.config.eviction == nf::NF_KV_EVICT_SLIDING);
    CHECK(ctx->kv_policy.config.window_size == 64);
    CHECK(ctx->sliding_window == 64);

    /* Verify arch was resolved */
    CHECK(ctx->arch != nullptr);
    CHECK(std::strcmp(nf::nf_resolve_attn_op(ctx->arch), "flash_attention_cached") == 0);

    /* Verify write_offset uses ring buffer */
    CHECK(ctx->kv_policy.write_offset(&ctx->kv_policy, 63) == 63);
    CHECK(ctx->kv_policy.write_offset(&ctx->kv_policy, 64) == 0);

    /* Build a step graph — should use resolved op names */
    auto sg = nf::build_llama_step_graph(*ctx, 1);
    CHECK(sg.gid != UINT32_MAX);
    CHECK(sg.layer_ids.size() == 1);

    engine.destroy_graph(sg.gid);

    /* Cleanup synthetic tensors */
    for (auto& [name, ti] : model.tensors)
        std::free(const_cast<void*>(ti.data));

    vt.shutdown(prov);
    std::printf("  PASS\n");
}

/* ---- Test 4: Mistral vs LLaMA KV policy difference ---- */
static void test_mistral_vs_llama_policy() {
    std::printf("[mistral] Test 4: Mistral vs LLaMA policy...\n");

    /* LLaMA: no eviction */
    nf::nf_kv_cache_config llama_cfg{};
    llama_cfg.eviction = nf::NF_KV_EVICT_NONE;
    llama_cfg.max_seq_len = 4096;
    auto llama_p = nf::nf_create_kv_policy(llama_cfg);

    /* Mistral: sliding window */
    nf::nf_kv_cache_config mistral_cfg{};
    mistral_cfg.eviction = nf::NF_KV_EVICT_SLIDING;
    mistral_cfg.window_size = 128;
    mistral_cfg.max_seq_len = 4096;
    auto mistral_p = nf::nf_create_kv_policy(mistral_cfg);

    /* At step 200: LLaMA sees all 201 tokens, Mistral sees only 128 */
    uint32_t start, len;
    llama_p.attn_range(&llama_p, 200, &start, &len);
    CHECK(start == 0 && len == 201);

    mistral_p.attn_range(&mistral_p, 200, &start, &len);
    CHECK(start == 73 && len == 128);

    /* LLaMA effective_len grows unbounded, Mistral caps at window */
    CHECK(llama_p.effective_len(&llama_p, 200) == 201);
    CHECK(mistral_p.effective_len(&mistral_p, 200) == 128);

    /* LLaMA write_offset = step, Mistral wraps */
    CHECK(llama_p.write_offset(&llama_p, 200) == 200);
    CHECK(mistral_p.write_offset(&mistral_p, 200) == 200 % 128);

    std::printf("  PASS\n");
}

int main() {
    std::printf("=== Mistral Architecture Tests (Phase 31) ===\n");
    test_mistral_registry();
    test_sliding_window_policy();
    test_context_kv_policy();
    test_mistral_vs_llama_policy();
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
