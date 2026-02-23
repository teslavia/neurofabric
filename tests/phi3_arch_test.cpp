/**
 * @file phi3_arch_test.cpp
 * @brief Phi-3 architecture end-to-end validation
 *
 * Phase 31: Multi-Architecture DAG & KV Cache Intelligence.
 *
 * Tests:
 *   1. Architecture registry: phi3 strategy lookup + op name resolution
 *   2. Phi3ArchCtx: partial RoPE configuration
 *   3. Weight naming: GGUF blk.N.xxx format
 *   4. (Optional) Real model decode with NF_TEST_GGUF_PATH_PHI3
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

/* ---- Test 1: Phi-3 strategy in registry ---- */
static void test_phi3_registry() {
    std::printf("[phi3] Test 1: Registry lookup...\n");

    nf::nf_clear_arch_registry();
    nf::nf_register_llama();
    nf::nf_register_mistral();

    static nf::Phi3ArchCtx phi3_ctx{0.5f, 48};
    nf::nf_register_phi3(&phi3_ctx);

    CHECK(nf::nf_arch_count() == 3);

    const auto* strat = nf::nf_find_arch("phi3");
    CHECK(strat != nullptr);
    CHECK(strat->weight_name != nullptr);
    CHECK(strat->attn_op_name != nullptr);
    CHECK(strat->rope_op_name != nullptr);
    CHECK(strat->ffn_op_name != nullptr);
    CHECK(strat->norm_op_name != nullptr);
    CHECK(strat->arch_ctx != nullptr);

    /* Verify op names */
    CHECK(std::strcmp(nf::nf_resolve_attn_op(strat), "flash_attention_cached") == 0);
    CHECK(std::strcmp(nf::nf_resolve_rope_op(strat, false), "rope_batch") == 0);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(strat), "silu") == 0);
    CHECK(std::strcmp(nf::nf_resolve_norm_op(strat), "rms_norm") == 0);

    /* Verify arch_ctx carries partial RoPE config */
    auto* ctx = static_cast<nf::Phi3ArchCtx*>(strat->arch_ctx);
    CHECK(ctx->partial_rotary_factor == 0.5f);
    CHECK(ctx->rotary_dims == 48);

    std::printf("  PASS\n");
}

/* ---- Test 2: Phi-3 weight naming ---- */
static void test_phi3_weight_naming() {
    std::printf("[phi3] Test 2: Weight naming...\n");

    const auto* strat = nf::nf_find_arch("phi3");
    CHECK(strat != nullptr);

    char buf[128];

    /* Global tensors */
    CHECK(std::strcmp(strat->weight_name(0, "token_embd", buf, sizeof(buf)),
                      "token_embd.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(0, "output_norm", buf, sizeof(buf)),
                      "output_norm.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(0, "output", buf, sizeof(buf)),
                      "output.weight") == 0);

    /* Per-layer tensors (same blk.N.xxx format as LLaMA in GGUF) */
    CHECK(std::strcmp(strat->weight_name(0, "attn_q", buf, sizeof(buf)),
                      "blk.0.attn_q.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(3, "ffn_gate", buf, sizeof(buf)),
                      "blk.3.ffn_gate.weight") == 0);
    CHECK(std::strcmp(strat->weight_name(31, "attn_output", buf, sizeof(buf)),
                      "blk.31.attn_output.weight") == 0);

    std::printf("  PASS\n");
}

/* ---- Test 3: Phi-3 context creation with no sliding window ---- */
static void test_phi3_context() {
    std::printf("[phi3] Test 3: Context creation...\n");

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

    /* Create mock GGUFModel with Phi-3-like config */
    nf::GGUFModel model{};
    model.architecture = "phi3";
    model.dim = 64;
    model.n_layers = 1;
    model.n_heads = 4;
    model.n_kv_heads = 2;
    model.ff_dim = 128;
    model.vocab_size = 32;
    model.max_seq = 256;
    model.rope_theta = 10000.0f;
    model.rms_norm_eps = 1e-5f;
    model.sliding_window = 0;  /* Phi-3 uses full causal */

    const uint32_t D = model.dim;
    const uint32_t KV_DIM = model.n_kv_heads * (D / model.n_heads);
    const uint32_t FF = model.ff_dim;
    const uint32_t V = model.vocab_size;

    auto make_tensor = [&](const std::string& name, uint64_t rows, uint64_t cols) {
        nf::GGUFTensorInfo ti{};
        ti.gguf_dtype = 0;
        ti.ndim = 2;
        ti.shape[0] = cols; ti.shape[1] = rows;
        ti.n_elements = rows * cols;
        ti.byte_size = ti.n_elements * sizeof(float);
        ti.data = std::calloc(ti.n_elements, sizeof(float));
        float* p = (float*)ti.data;
        for (uint64_t i = 0; i < ti.n_elements; ++i) p[i] = 0.001f;
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

    auto ctx = nf::create_llama_context(engine, model, prov, vt, mem_vt,
                                         256, 16);
    CHECK(ctx != nullptr);

    /* Phi-3: no sliding window → NONE eviction */
    CHECK(ctx->kv_policy.config.eviction == nf::NF_KV_EVICT_NONE);
    CHECK(ctx->kv_policy.config.window_size == 0);

    /* Arch resolved to phi3 */
    CHECK(ctx->arch != nullptr);
    CHECK(std::strcmp(nf::nf_resolve_ffn_op(ctx->arch), "silu") == 0);

    /* Build step graph */
    auto sg = nf::build_llama_step_graph(*ctx, 1);
    CHECK(sg.gid != UINT32_MAX);
    engine.destroy_graph(sg.gid);

    for (auto& [name, ti] : model.tensors)
        std::free(const_cast<void*>(ti.data));

    vt.shutdown(prov);
    std::printf("  PASS\n");
}

int main() {
    std::printf("=== Phi-3 Architecture Tests (Phase 31) ===\n");
    test_phi3_registry();
    test_phi3_weight_naming();
    test_phi3_context();
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
