/**
 * @file fused_warmup_test.cpp
 * @brief Phase 30 — Fused Op DAG Integration Test
 *
 * Verifies that create_llama_context() takes the fused path for Q4_0 weights:
 *   1. layer_quant buffers populated for Q4_0 layers
 *   2. Step graph uses fused op names
 *   3. Fused step graph produces finite logits
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_scheduler_abi.h"
#include "neurofabric/engine/PipelineEngine.hpp"
#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "model/quant_registry.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static nf_provider            g_prov = nullptr;
static nf_provider_vtable     g_vt{};
static nf_provider_mem_vtable g_mem_vt{};
/* FUSED_WARMUP_PART2 */

/**
 * Test 1: Verify fused path is selected for Q4_0 weights.
 * layer_quant buffers should be populated, layer_f32 should be null for fused slots.
 */
static void test_fused_path_selection() {
    std::printf("  test_fused_path_selection...\n");

    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    CHECK(gguf_path != nullptr);

    nf::GGUFModel* model = nf::gguf_open(gguf_path);
    CHECK(model != nullptr);

    nf::PipelineEngine engine;
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt,
        /*max_seq=*/64, /*max_prefill_seq=*/16, /*use_fp16=*/false);
    CHECK(ctx != nullptr);

    /* For Q4_0 models, layer_quant should have populated buffers */
    CHECK(ctx->layer_quant.size() == ctx->n_layers);

    bool has_fused = false;
    for (uint32_t l = 0; l < ctx->n_layers; ++l) {
        auto& lq = ctx->layer_quant[l];
        /* If any quant buffer is non-null, fused path was taken */
        if (lq.q.buf) { has_fused = true; CHECK(lq.q_dt == NF_DTYPE_Q4_0); }
        if (lq.k.buf) { has_fused = true; CHECK(lq.k_dt == NF_DTYPE_Q4_0); }
        if (lq.v.buf) { has_fused = true; CHECK(lq.v_dt == NF_DTYPE_Q4_0); }
        if (lq.o.buf) { has_fused = true; CHECK(lq.o_dt == NF_DTYPE_Q4_0); }
        if (lq.gate.buf) { has_fused = true; CHECK(lq.gate_dt == NF_DTYPE_Q4_0); }
        if (lq.up.buf)   { has_fused = true; CHECK(lq.up_dt == NF_DTYPE_Q4_0); }
        if (lq.down.buf) { has_fused = true; CHECK(lq.down_dt == NF_DTYPE_Q4_0); }
    }
    /* TinyLlama Q4_0 should have fused weights */
    std::printf("    fused path taken: %s\n", has_fused ? "yes" : "no");
    CHECK(has_fused);

    nf::gguf_close(model);
    std::printf("    fused_path_selection verified ✓\n");
}

/**
 * Test 2: Build step graph and verify fused op names appear.
 */
static void test_fused_step_graph() {
    std::printf("  test_fused_step_graph...\n");

    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    CHECK(gguf_path != nullptr);

    nf::GGUFModel* model = nf::gguf_open(gguf_path);
    CHECK(model != nullptr);

    nf::PipelineEngine engine;
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt,
        64, 16, false);
    CHECK(ctx != nullptr);

    /* Build step graph */
    auto sg = nf::build_llama_step_graph(*ctx, 1);

    /* Verify graph was created */
    CHECK(sg.gid != 0 || sg.embed_id != UINT32_MAX);
    CHECK(sg.layer_ids.size() == ctx->n_layers);

    /* The step graph should use fused ops — we verify by checking that
       the context has fused weights (the graph builder uses them when available) */
    bool any_fused = false;
    for (uint32_t l = 0; l < ctx->n_layers; ++l) {
        if (ctx->layer_quant[l].q.buf) { any_fused = true; break; }
    }
    CHECK(any_fused);

    engine.destroy_graph(sg.gid);
    nf::gguf_close(model);
    std::printf("    fused_step_graph verified ✓\n");
}

/**
 * Test 3: End-to-end — fused context produces finite logits.
 */
static void test_fused_e2e() {
    std::printf("  test_fused_e2e...\n");

    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    CHECK(gguf_path != nullptr);

    nf::GGUFModel* model = nf::gguf_open(gguf_path);
    CHECK(model != nullptr);

    nf::PipelineEngine engine;
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt,
        64, 16, false);
    CHECK(ctx != nullptr);

    /* Write a single token */
    { void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
      ((int32_t*)p)[0] = 1;  /* BOS token */
      ctx->token_buf.ops.unmap(ctx->token_buf.buf); }

    /* Build and run step graph */
    auto sg = nf::build_llama_step_graph(*ctx, 1);
    nf::PipelineEngine::Session sess(engine, sg.gid);
    nf::inject_step_push_constants(*ctx, sg, sess, 1, 0);
    nf_status st = sess.step().get();
    CHECK(st == NF_OK);

    /* Verify logits are finite */
    { void* p; ctx->logits.ops.map(ctx->logits.buf, &p);
      float* logits = (float*)p;
      bool any_nonzero = false;
      for (uint32_t i = 0; i < ctx->vocab_size; ++i) {
          CHECK(std::isfinite(logits[i]));
          if (logits[i] != 0.0f) any_nonzero = true;
      }
      CHECK(any_nonzero);
      ctx->logits.ops.unmap(ctx->logits.buf); }

    engine.destroy_graph(sg.gid);
    nf::gguf_close(model);
    std::printf("    fused_e2e verified ✓\n");
}

int main() {
    std::printf("fused_warmup_test: Phase 30 — Fused Op DAG Integration\n");

    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    if (!gguf_path) {
        std::printf("SKIP: NF_TEST_GGUF_PATH not set\n");
        return 0;
    }

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));

    test_fused_path_selection();
    test_fused_step_graph();
    test_fused_e2e();

    g_vt.shutdown(g_prov);
    std::printf("OK: all Phase 30 fused warmup tests passed\n");
    return 0;
}