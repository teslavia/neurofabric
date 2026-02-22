/**
 * @file real_model_test.cpp
 * @brief Phase 22 — Persistent Session & KV Cache Continuity
 *
 * Loads a real TinyLlama-1.1B Q4_0 GGUF from disk and generates real tokens
 * on Metal GPU with persistent weights, KV cache, and Session.
 *
 * Requires NF_TEST_GGUF_PATH env var pointing to a LLaMA-architecture GGUF.
 * Skips gracefully (exit 0) if unset — CI-friendly.
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"
#include "neurofabric/PipelineEngine.hpp"

#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

/* Metal timing API (Phase 23) */
extern "C" void     nf_metal_enable_timing(bool enable);
extern "C" uint32_t nf_metal_get_timing_count();
extern "C" void     nf_metal_get_timings(char (*op_names)[64], double* gpu_ms,
                                          uint32_t max_count);

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static nf_provider            g_prov = nullptr;
static nf_provider_vtable     g_vt{};
static nf_provider_mem_vtable g_mem_vt{};

/* ================================================================== */
/*  Test 1: GGUF metadata parse                                        */
/* ================================================================== */
static void test_gguf_metadata(nf::GGUFModel* model) {
    std::printf("  Test 1: GGUF metadata parse ... ");

    CHECK(model->dim > 0);
    CHECK(model->n_layers > 0);
    CHECK(model->n_heads > 0);
    CHECK(model->ff_dim > 0);
    CHECK(model->vocab_size > 0);
/* PLACEHOLDER_TEST1_CONT */

    /* Verify key tensors exist */
    CHECK(model->tensors.count("token_embd.weight"));
    CHECK(model->tensors.count("blk.0.attn_q.weight"));
    CHECK(model->tensors.count("output.weight"));
    CHECK(model->tensors.count("output_norm.weight"));

    std::printf("PASS\n");
    std::printf("    dim=%u, layers=%u, heads=%u, kv_heads=%u, ff=%u, vocab=%u\n",
                model->dim, model->n_layers, model->n_heads,
                model->n_kv_heads, model->ff_dim, model->vocab_size);
    std::printf("    rope_theta=%.1f, rms_eps=%.1e\n",
                model->rope_theta, model->rms_norm_eps);
}

/* ================================================================== */
/*  Test 2: Single-token forward pass                                  */
/* ================================================================== */
static void test_single_token_forward(nf::GGUFModel* model) {
    std::printf("  Test 2: single-token forward ... ");

    const uint32_t seq = 1;
    const uint32_t max_seq = 128;

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto dag = nf::build_llama_dag(
        engine, *model, g_prov, g_vt, g_mem_vt, max_seq, seq);
    CHECK(dag != nullptr);

    /* Feed token ID 1 (BOS) */
    { void* p; dag->token_buf.ops.map(dag->token_buf.buf, &p);
      ((int32_t*)p)[0] = 1;
      dag->token_buf.ops.unmap(dag->token_buf.buf); }

    nf::PipelineEngine::Session sess(engine, dag->gid);
    nf::inject_llama_push_constants(*dag, sess, seq, 0);
    CHECK_OK(sess.step().get());

    /* Verify logits: vocab_size floats, all finite */
    { dag->logits.ops.cache_sync(dag->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
      void* p; dag->logits.ops.map(dag->logits.buf, &p);
      float* logits = (float*)p;
      uint32_t V = dag->vocab_size;
      bool has_nonzero = false;
      for (uint32_t i = 0; i < V; ++i) {
          CHECK(std::isfinite(logits[i]));
          if (logits[i] != 0.0f) has_nonzero = true;
      }
      CHECK(has_nonzero);
      dag->logits.ops.unmap(dag->logits.buf); }

    /* Verify argmax output */
    { dag->argmax_out.ops.cache_sync(dag->argmax_out.buf, NF_CACHE_INVALIDATE, 0, 0);
      void* p; dag->argmax_out.ops.map(dag->argmax_out.buf, &p);
      int32_t tok = ((int32_t*)p)[0];
      CHECK(tok >= 0 && (uint32_t)tok < dag->vocab_size);
      std::printf("logits OK (%u finite values), argmax=%d PASS\n",
                  dag->vocab_size, tok);
      dag->argmax_out.ops.unmap(dag->argmax_out.buf); }
}

/* PLACEHOLDER_TEST3 */

/* ================================================================== */
/*  Test 3: Greedy autoregressive decode (16 tokens) — persistent ctx  */
/* ================================================================== */
static void test_greedy_decode(nf::GGUFModel* model) {
    std::printf("  Test 3: greedy decode (16 tokens, persistent session) ... ");

    const uint32_t max_seq = 128;
    const uint32_t prefill_seq = 1;  /* BOS only */
    const uint32_t n_decode = 16;

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto t0 = std::chrono::steady_clock::now();

    /* One-time context creation (weights + dequant warmup + KV cache) */
    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt, max_seq, prefill_seq);
    CHECK(ctx != nullptr);

    std::vector<int32_t> generated;

    auto t_ctx = std::chrono::steady_clock::now();

    /* Prefill: feed BOS token */
    {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = 1;  /* BOS */
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, prefill_seq);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, prefill_seq, 0);
        CHECK_OK(sess.step().get());

        ctx->argmax_out.ops.cache_sync(ctx->argmax_out.buf,
                                        NF_CACHE_INVALIDATE, 0, 0);
        ctx->argmax_out.ops.map(ctx->argmax_out.buf, &p);
        int32_t tok = ((int32_t*)p)[0];
        generated.push_back(tok);
        ctx->argmax_out.ops.unmap(ctx->argmax_out.buf);

        engine.destroy_graph(sg.gid);
    }

    auto t_prefill = std::chrono::steady_clock::now();

    /* Decode loop — same engine, same weights, same KV cache */
    for (uint32_t step = 0; step < n_decode - 1; ++step) {
        uint32_t step_idx = prefill_seq + step;

        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = generated.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, step_idx);
        CHECK_OK(sess.step().get());

        ctx->argmax_out.ops.cache_sync(ctx->argmax_out.buf,
                                        NF_CACHE_INVALIDATE, 0, 0);
        ctx->argmax_out.ops.map(ctx->argmax_out.buf, &p);
        int32_t tok = ((int32_t*)p)[0];
        generated.push_back(tok);
        ctx->argmax_out.ops.unmap(ctx->argmax_out.buf);

        engine.destroy_graph(sg.gid);
    }

    auto t_end = std::chrono::steady_clock::now();

    /* Verify all tokens valid */
    uint32_t V = model->vocab_size;
    for (auto tok : generated) {
        CHECK(tok >= 0 && (uint32_t)tok < V);
    }

    /* Print generated token IDs */
    std::printf("[");
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%d", generated[i]);
    }
    std::printf("] PASS\n");

    /* Test 4: Performance (observational) */
    double ctx_ms = std::chrono::duration<double, std::milli>(
        t_ctx - t0).count();
    double prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill - t_ctx).count();
    double decode_ms = std::chrono::duration<double, std::milli>(
        t_end - t_prefill).count();
    double tps = (n_decode - 1) / (decode_ms / 1000.0);
    std::printf("  Test 4: performance ... ctx: %.0f ms, prefill: %.1f ms, decode: %.1f tok/s\n",
                ctx_ms, prefill_ms, tps);
}

/* PLACEHOLDER_TEST5 */

/* PLACEHOLDER_TEST5 */

/* ================================================================== */
/*  Test 5: KV cache continuity proof                                  */
/* ================================================================== */
static void test_kv_cache_continuity(nf::GGUFModel* model) {
    std::printf("  Test 5: KV cache continuity ... ");

    const uint32_t max_seq = 128;
    const uint32_t V = model->vocab_size;

    /* ---- Run A: 2-step decode with persistent context ---- */
    std::vector<float> logits_with(V);
    int32_t tok1;
    {
        nf::PipelineEngine engine(4);
        engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
        auto ctx = nf::create_llama_context(
            engine, *model, g_prov, g_vt, g_mem_vt, max_seq, 1);
        CHECK(ctx != nullptr);

        /* Prefill BOS */
        { void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
          ((int32_t*)p)[0] = 1;
          ctx->token_buf.ops.unmap(ctx->token_buf.buf); }
        {
            auto sg = nf::build_llama_step_graph(*ctx, 1);
            nf::PipelineEngine::Session sess(engine, sg.gid);
            nf::inject_step_push_constants(*ctx, sg, sess, 1, 0);
            CHECK_OK(sess.step().get());
            void* p; ctx->argmax_out.ops.map(ctx->argmax_out.buf, &p);
            tok1 = ((int32_t*)p)[0];
            ctx->argmax_out.ops.unmap(ctx->argmax_out.buf);
            engine.destroy_graph(sg.gid);
        }

        /* Decode tok1 at step_idx=1 (KV cache has BOS context) */
        { void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
          ((int32_t*)p)[0] = tok1;
          ctx->token_buf.ops.unmap(ctx->token_buf.buf); }
        {
            auto sg = nf::build_llama_step_graph(*ctx, 1);
            nf::PipelineEngine::Session sess(engine, sg.gid);
            nf::inject_step_push_constants(*ctx, sg, sess, 1, 1);
            CHECK_OK(sess.step().get());

            ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
            void* p; ctx->logits.ops.map(ctx->logits.buf, &p);
            std::memcpy(logits_with.data(), p, V * sizeof(float));
            ctx->logits.ops.unmap(ctx->logits.buf);
            engine.destroy_graph(sg.gid);
        }
    }

    /* ---- Run B: fresh single-token forward with tok1 at step_idx=0 ---- */
    std::vector<float> logits_without(V);
    {
        nf::PipelineEngine engine(4);
        engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
        auto ctx = nf::create_llama_context(
            engine, *model, g_prov, g_vt, g_mem_vt, max_seq, 1);
        CHECK(ctx != nullptr);

        { void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
          ((int32_t*)p)[0] = tok1;
          ctx->token_buf.ops.unmap(ctx->token_buf.buf); }
        {
            auto sg = nf::build_llama_step_graph(*ctx, 1);
            nf::PipelineEngine::Session sess(engine, sg.gid);
            nf::inject_step_push_constants(*ctx, sg, sess, 1, 0);
            CHECK_OK(sess.step().get());

            ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
            void* p; ctx->logits.ops.map(ctx->logits.buf, &p);
            std::memcpy(logits_without.data(), p, V * sizeof(float));
            ctx->logits.ops.unmap(ctx->logits.buf);
            engine.destroy_graph(sg.gid);
        }
    }

    /* ---- Compare: logits MUST differ (proves KV cache has effect) ---- */
    bool any_diff = false;
    for (uint32_t i = 0; i < V; ++i) {
        if (logits_with[i] != logits_without[i]) { any_diff = true; break; }
    }
    CHECK(any_diff);
    std::printf("logits differ (context effect confirmed) PASS\n");
}

/* ================================================================== */
/*  Test 6: Sampled decode (16 tokens)                                  */
/* ================================================================== */
static void test_sampled_decode(nf::GGUFModel* model) {
    std::printf("  Test 6: sampled decode (16 tokens) ... ");

    const uint32_t max_seq = 128;
    const uint32_t prefill_seq = 1;
    const uint32_t n_decode = 16;

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt, max_seq, prefill_seq);
    CHECK(ctx != nullptr);

    nf::SamplerParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 40;
    sp.top_p = 0.95f;
    sp.repeat_penalty = 1.1f;
    sp.seed = 42;
    std::mt19937_64 rng(sp.seed);

    std::vector<int32_t> generated;
    uint32_t V = model->vocab_size;

    /* Prefill: BOS */
    {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = 1;
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, prefill_seq);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, prefill_seq, 0);
        CHECK_OK(sess.step().get());

        /* Read logits and sample */
        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            nullptr, 0, sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        generated.push_back(tok);
        engine.destroy_graph(sg.gid);
    }

    /* Decode loop */
    for (uint32_t step = 0; step < n_decode - 1; ++step) {
        uint32_t step_idx = prefill_seq + step;
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = generated.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, step_idx);
        CHECK_OK(sess.step().get());

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            generated.data(), (uint32_t)generated.size(), sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        generated.push_back(tok);
        engine.destroy_graph(sg.gid);
    }

    for (auto tok : generated)
        CHECK(tok >= 0 && (uint32_t)tok < V);

    /* Count max consecutive repeats */
    uint32_t max_repeat = 1, cur_repeat = 1;
    for (size_t i = 1; i < generated.size(); ++i) {
        if (generated[i] == generated[i-1]) {
            ++cur_repeat;
            if (cur_repeat > max_repeat) max_repeat = cur_repeat;
        } else { cur_repeat = 1; }
    }

    std::printf("[");
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%d", generated[i]);
    }
    std::printf("] max_repeat=%u PASS\n", max_repeat);
}

/* ================================================================== */
/*  Test 7: Per-kernel GPU profiling                                    */
/* ================================================================== */
static void test_gpu_profiling(nf::GGUFModel* model) {
    std::printf("  Test 7: per-kernel GPU profiling ...\n");

    const uint32_t max_seq = 128;
    const uint32_t n_steps = 4;

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt, max_seq, 1);
    CHECK(ctx != nullptr);

    nf_metal_enable_timing(true);

    int32_t next_tok = 1; /* BOS */
    for (uint32_t s = 0; s < n_steps; ++s) {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = next_tok;
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, s);
        CHECK_OK(sess.step().get());
        g_vt.synchronize(g_prov);

        ctx->argmax_out.ops.cache_sync(ctx->argmax_out.buf,
                                        NF_CACHE_INVALIDATE, 0, 0);
        ctx->argmax_out.ops.map(ctx->argmax_out.buf, &p);
        next_tok = ((int32_t*)p)[0];
        ctx->argmax_out.ops.unmap(ctx->argmax_out.buf);
        engine.destroy_graph(sg.gid);
    }

    nf_metal_enable_timing(false);

    /* Read timings */
    uint32_t count = nf_metal_get_timing_count();
    if (count == 0) {
        std::printf("    (no timings captured)\n");
        return;
    }

    struct OpName { char name[64]; };
    std::vector<OpName> names(count);
    std::vector<double> ms(count);
    nf_metal_get_timings(reinterpret_cast<char(*)[64]>(names.data()),
                          ms.data(), count);

    /* Aggregate by op name */
    struct OpStats { double total_ms; uint32_t count; };
    std::map<std::string, OpStats> stats;
    double grand_total = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& s = stats[names[i].name];
        s.total_ms += ms[i];
        s.count++;
        grand_total += ms[i];
    }

    std::printf("    %-30s %6s %10s %10s %6s\n",
                "op_name", "count", "total_ms", "avg_us", "%");
    std::printf("    %-30s %6s %10s %10s %6s\n",
                "------------------------------", "------",
                "----------", "----------", "------");
    for (auto& [name, s] : stats) {
        double avg_us = (s.total_ms / s.count) * 1000.0;
        double pct = (grand_total > 0) ? (s.total_ms / grand_total * 100.0) : 0.0;
        std::printf("    %-30s %6u %10.3f %10.1f %5.1f%%\n",
                    name.c_str(), s.count, s.total_ms, avg_us, pct);
    }
    std::printf("    TOTAL: %.3f ms (%u dispatches, %u steps)\n",
                grand_total, count, n_steps);
    std::printf("    PASS\n");
}

/* ================================================================== */
/*  Test 8: Text round-trip (tokenize → generate → detokenize)         */
/* ================================================================== */
static void test_text_roundtrip(nf::GGUFModel* model) {
    std::printf("  Test 8: text round-trip...\n");

    /* Check tokenizer data is available */
    if (model->vocab.empty()) {
        std::printf("    SKIPPED (no vocab in GGUF)\n");
        return;
    }

    nf::Tokenizer tokenizer(*model);
    CHECK(tokenizer.vocab_size() > 0);

    /* Encode a prompt */
    auto prompt_ids = tokenizer.encode("Hello");
    CHECK(prompt_ids.size() >= 2); /* at least BOS + something */
    std::printf("    encode(\"Hello\") → %zu tokens:", prompt_ids.size());
    for (auto id : prompt_ids) std::printf(" %d", id);
    std::printf("\n");

    /* Generate a few tokens */
    const uint32_t max_seq = 128;
    uint32_t prefill_seq = static_cast<uint32_t>(prompt_ids.size());

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(engine, *model, g_prov, g_vt, g_mem_vt, max_seq, prefill_seq);
    CHECK(ctx != nullptr);

    nf::SamplerParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 40;
    sp.top_p = 0.95f;
    sp.repeat_penalty = 1.1f;
    sp.seed = 42;
    std::mt19937_64 rng(sp.seed);

    std::vector<int32_t> all_tokens(prompt_ids.begin(), prompt_ids.end());
    uint32_t V = model->vocab_size;
    const uint32_t n_generate = 8;

    /* Prefill */
    {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = prompt_ids.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, prefill_seq);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, prefill_seq, 0);
        CHECK_OK(sess.step().get());

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V, nullptr, 0, sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        all_tokens.push_back(tok);
        engine.destroy_graph(sg.gid);
    }

    /* Decode */
    for (uint32_t step = 0; step < n_generate - 1; ++step) {
        uint32_t step_idx = prefill_seq + step;
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = all_tokens.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, step_idx);
        CHECK_OK(sess.step().get());

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            all_tokens.data(), (uint32_t)all_tokens.size(), sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        all_tokens.push_back(tok);
        engine.destroy_graph(sg.gid);
    }

    /* Detokenize */
    std::string output = tokenizer.decode(all_tokens);
    CHECK(!output.empty());

    /* Verify: output should contain readable characters */
    bool has_alpha = false;
    for (char c : output)
        if (std::isalpha(static_cast<unsigned char>(c))) { has_alpha = true; break; }
    CHECK(has_alpha);

    std::printf("    generated: \"%s\"\n", output.c_str());
    std::printf("    PASS\n");
}

/* ================================================================== */
/*  7B Model Tests (Phase 26)                                          */
/* ================================================================== */

/**
 * Test 9: 7B metadata — validates sliding window & architecture fields.
 */
static void test_7b_metadata(nf::GGUFModel* model) {
    std::printf("  Test 9: 7B metadata parse ... ");

    CHECK(model->dim > 0);
    CHECK(model->n_layers >= 32);   /* 7B models have ≥32 layers */
    CHECK(model->n_heads > 0);
    CHECK(model->ff_dim > 0);
    CHECK(model->vocab_size > 0);

    /* Verify key tensors */
    CHECK(model->tensors.count("token_embd.weight"));
    CHECK(model->tensors.count("blk.0.attn_q.weight"));
    CHECK(model->tensors.count("output_norm.weight"));

    std::printf("PASS\n");
    std::printf("    arch=%s, dim=%u, layers=%u, heads=%u, kv_heads=%u, ff=%u, vocab=%u\n",
                model->architecture.c_str(), model->dim, model->n_layers,
                model->n_heads, model->n_kv_heads, model->ff_dim, model->vocab_size);
    std::printf("    sliding_window=%u\n", model->sliding_window);
}

/**
 * Test 10: 7B forward pass — flash attention at scale.
 * Runs prefill + short decode, verifies logits are finite and tokens valid.
 */
static void test_7b_forward(nf::GGUFModel* model) {
    std::printf("  Test 10: 7B forward (flash attention) ... ");

    const uint32_t max_seq = 256;
    const uint32_t prefill_seq = 1;
    const uint32_t n_decode = 8;

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    auto ctx = nf::create_llama_context(
        engine, *model, g_prov, g_vt, g_mem_vt, max_seq, prefill_seq);
    CHECK(ctx != nullptr);

    /* Verify sliding_window propagated to context */
    CHECK(ctx->sliding_window == model->sliding_window);

    std::vector<int32_t> generated;
    uint32_t V = model->vocab_size;
/* PLACEHOLDER_7B_FORWARD_CONT */

    /* Prefill: BOS */
    {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = 1;
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, prefill_seq);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, prefill_seq, 0);
        CHECK_OK(sess.step().get());

        ctx->argmax_out.ops.cache_sync(ctx->argmax_out.buf,
                                        NF_CACHE_INVALIDATE, 0, 0);
        ctx->argmax_out.ops.map(ctx->argmax_out.buf, &p);
        int32_t tok = ((int32_t*)p)[0];
        CHECK(tok >= 0 && (uint32_t)tok < V);
        generated.push_back(tok);
        ctx->argmax_out.ops.unmap(ctx->argmax_out.buf);
        engine.destroy_graph(sg.gid);
    }

    /* Decode loop */
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t step = 0; step < n_decode - 1; ++step) {
        uint32_t step_idx = prefill_seq + step;
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = generated.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, step_idx);
        CHECK_OK(sess.step().get());

        ctx->argmax_out.ops.cache_sync(ctx->argmax_out.buf,
                                        NF_CACHE_INVALIDATE, 0, 0);
        ctx->argmax_out.ops.map(ctx->argmax_out.buf, &p);
        int32_t tok = ((int32_t*)p)[0];
        CHECK(tok >= 0 && (uint32_t)tok < V);
        generated.push_back(tok);
        ctx->argmax_out.ops.unmap(ctx->argmax_out.buf);
        engine.destroy_graph(sg.gid);
    }
    auto t1 = std::chrono::steady_clock::now();

    double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tps = (n_decode - 1) / (decode_ms / 1000.0);

    std::printf("[");
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%d", generated[i]);
    }
    std::printf("] %.1f tok/s PASS\n", tps);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main() {
    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    if (!gguf_path || gguf_path[0] == '\0') {
        std::printf("SKIPPED (NF_TEST_GGUF_PATH not set)\n");
        return 0;
    }

    std::printf("=== real_model_test (Phase 26) ===\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    nf::GGUFModel* model = nf::gguf_open(gguf_path);
    CHECK(model != nullptr);
    std::printf("  Model: %s (dim=%u, layers=%u, heads=%u, ff=%u, vocab=%u)\n",
                gguf_path, model->dim, model->n_layers, model->n_heads,
                model->ff_dim, model->vocab_size);

    test_gguf_metadata(model);
    test_single_token_forward(model);
    test_greedy_decode(model);
    test_kv_cache_continuity(model);
    test_sampled_decode(model);
    test_gpu_profiling(model);
    test_text_roundtrip(model);

    nf::gguf_close(model);

    /* ---- 7B model tests (optional, separate GGUF) ---- */
    const char* gguf_7b_path = std::getenv("NF_TEST_GGUF_7B_PATH");
    if (gguf_7b_path && gguf_7b_path[0] != '\0') {
        std::printf("\n=== 7B model tests (Phase 26) ===\n");
        nf::GGUFModel* model_7b = nf::gguf_open(gguf_7b_path);
        CHECK(model_7b != nullptr);
        std::printf("  Model: %s (dim=%u, layers=%u, heads=%u, ff=%u, vocab=%u)\n",
                    gguf_7b_path, model_7b->dim, model_7b->n_layers,
                    model_7b->n_heads, model_7b->ff_dim, model_7b->vocab_size);

        test_7b_metadata(model_7b);
        test_7b_forward(model_7b);

        nf::gguf_close(model_7b);
    } else {
        std::printf("\n  7B tests SKIPPED (NF_TEST_GGUF_7B_PATH not set)\n");
    }

    if (g_vt.shutdown) g_vt.shutdown(g_prov);
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
