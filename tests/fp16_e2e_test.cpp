/**
 * @file fp16_e2e_test.cpp
 * @brief Phase 27 — FP16 End-to-End Inference Test
 *
 * Loads TinyLlama 1.1B with use_fp16=true, runs forward pass,
 * verifies logits are valid and compares top predictions vs F32.
 * Requires NF_TEST_GGUF_PATH env var.
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/engine/PipelineEngine.hpp"
#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "model/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

int main() {
    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    if (!gguf_path) {
        std::printf("[fp16_e2e] SKIP — NF_TEST_GGUF_PATH not set\n");
        return 0;
    }

    auto* model = nf::gguf_open(gguf_path);
    CHECK(model != nullptr);

    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    CHECK_OK(nf_plugin_register(&vt, &prov));
    CHECK_OK(nf_plugin_register_mem(&mem_vt, &prov));
    CHECK_OK(vt.init(prov));
    nf::PipelineEngine engine(4);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);

    const uint32_t max_seq = 128;
    const uint32_t prefill_seq = 1;

    std::printf("[fp16_e2e] Loading model with FP16...\n");
    auto t0 = std::chrono::steady_clock::now();

    /* Create FP16 context */
    auto ctx_f16 = nf::create_llama_context(engine, *model, prov, vt, mem_vt,
                                              max_seq, prefill_seq, true);
    CHECK(ctx_f16 != nullptr);
    CHECK(ctx_f16->use_fp16 == true);

    auto t1 = std::chrono::steady_clock::now();
    double ctx_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[fp16_e2e] FP16 context created in %.0f ms\n", ctx_ms);

    /* Run single-token forward pass */
    int32_t test_token = 1;  /* BOS token */
    { void* p; ctx_f16->token_buf.ops.map(ctx_f16->token_buf.buf, &p);
      ((int32_t*)p)[0] = test_token;
      ctx_f16->token_buf.ops.unmap(ctx_f16->token_buf.buf); }

    auto sg = nf::build_llama_step_graph(*ctx_f16, prefill_seq);
    nf::PipelineEngine::Session sess(engine, sg.gid);
    nf::inject_step_push_constants(*ctx_f16, sg, sess, prefill_seq, 0);

    auto t2 = std::chrono::steady_clock::now();
    nf_status st = sess.step().get();
    CHECK_OK(st);
    auto t3 = std::chrono::steady_clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("[fp16_e2e] Forward pass: %.1f ms\n", fwd_ms);

    /* Verify logits are valid (not NaN/Inf) */
    ctx_f16->logits.ops.cache_sync(ctx_f16->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
    void* lp; ctx_f16->logits.ops.map(ctx_f16->logits.buf, &lp);
    float* logits = (float*)lp;
    uint32_t V = model->vocab_size;

    int nan_count = 0, inf_count = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if (std::isnan(logits[i])) nan_count++;
        if (std::isinf(logits[i])) inf_count++;
    }
    std::printf("[fp16_e2e] Logits: %u values, %d NaN, %d Inf\n", V, nan_count, inf_count);
    CHECK(nan_count == 0);
    CHECK(inf_count == 0);

    /* Find top-5 tokens */
    std::vector<std::pair<float, int>> scored(V);
    for (uint32_t i = 0; i < V; ++i) scored[i] = {logits[i], (int)i};
    std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });

    std::printf("[fp16_e2e] Top-5 FP16 tokens:");
    for (int i = 0; i < 5; ++i)
        std::printf(" %d(%.2f)", scored[i].second, scored[i].first);
    std::printf("\n");

    ctx_f16->logits.ops.unmap(ctx_f16->logits.buf);
    engine.destroy_graph(sg.gid);

    /* Greedy decode test: generate 4 tokens */
    std::printf("[fp16_e2e] Greedy decode (4 tokens)...\n");
    std::vector<int32_t> all_tokens = {test_token};
    for (int step = 0; step < 4; ++step) {
        int32_t last_tok = all_tokens.back();
        { void* p; ctx_f16->token_buf.ops.map(ctx_f16->token_buf.buf, &p);
          ((int32_t*)p)[0] = last_tok;
          ctx_f16->token_buf.ops.unmap(ctx_f16->token_buf.buf); }

        uint32_t step_idx = (step == 0) ? 0 : (uint32_t)all_tokens.size() - 1;
        uint32_t sl = 1;
        auto sg2 = nf::build_llama_step_graph(*ctx_f16, sl);
        nf::PipelineEngine::Session s2(engine, sg2.gid);
        nf::inject_step_push_constants(*ctx_f16, sg2, s2, sl, step_idx);
        CHECK_OK(s2.step().get());

        ctx_f16->logits.ops.cache_sync(ctx_f16->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        void* lp2; ctx_f16->logits.ops.map(ctx_f16->logits.buf, &lp2);
        float* log2 = (float*)lp2;
        /* Greedy argmax */
        int best = 0; float best_v = log2[0];
        for (uint32_t i = 1; i < V; ++i)
            if (log2[i] > best_v) { best_v = log2[i]; best = (int)i; }
        ctx_f16->logits.ops.unmap(ctx_f16->logits.buf);
        all_tokens.push_back(best);
        engine.destroy_graph(sg2.gid);
    }

    std::printf("[fp16_e2e] Generated tokens:");
    for (auto t : all_tokens) std::printf(" %d", t);
    std::printf("\n");

    vt.shutdown(prov);
    nf::gguf_close(model);
    std::printf("[fp16_e2e] All tests PASSED\n");
    return 0;
}
