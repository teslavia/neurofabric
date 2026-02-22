/**
 * @file nf_generate.cpp
 * @brief End-to-end text generation CLI
 *
 * Phase 24: Tokenizer Integration & MatMul Optimization.
 * Phase 31: Multi-Architecture support (--arch flag, auto-detection).
 *
 * Usage: nf_generate <gguf_path> "prompt text" [--max-tokens N] [--temperature T]
 *                    [--top-k K] [--top-p P] [--seed S] [--fp16] [--arch NAME]
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/PipelineEngine.hpp"
#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "model/arch_registry.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <gguf_path> \"prompt\" [options]\n"
        "Options:\n"
        "  --max-tokens N   (default: 32)\n"
        "  --temperature T  (default: 0.8)\n"
        "  --top-k K        (default: 40)\n"
        "  --top-p P        (default: 0.95)\n"
        "  --seed S         (default: random)\n"
        "  --fp16           use FP16 inference\n"
        "  --paged          use paged KV cache\n"
        "  --arch NAME      override architecture (llama/mistral/phi3)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* gguf_path = argv[1];
    std::string prompt = argv[2];
/* PLACEHOLDER_MAIN_CONTINUE */

    /* Parse options */
    uint32_t max_tokens = 32;
    nf::SamplerParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 40;
    sp.top_p = 0.95f;
    sp.repeat_penalty = 1.1f;
    sp.seed = 0;
    bool has_seed = false;
    bool use_fp16 = false;
    bool use_paged = false;
    const char* arch_override = nullptr;

    /* Check NF_FP16 env var */
    const char* fp16_env = std::getenv("NF_FP16");
    if (fp16_env && std::strcmp(fp16_env, "1") == 0) use_fp16 = true;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            sp.temperature = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            sp.top_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            sp.top_p = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            sp.seed = std::atoi(argv[++i]);
            has_seed = true;
        }
        else if (std::strcmp(argv[i], "--fp16") == 0)
            use_fp16 = true;
        else if (std::strcmp(argv[i], "--paged") == 0)
            use_paged = true;
        else if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc)
            arch_override = argv[++i];
    }
    if (!has_seed) sp.seed = static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(sp.seed);

    /* Load model */
    auto* model = nf::gguf_open(gguf_path);
    if (!model) { std::fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path); return 1; }

    /* Phase 31: override architecture if requested */
    if (arch_override) model->architecture = arch_override;

    /* Register all known architectures */
    nf::nf_register_llama();
    nf::nf_register_mistral();
    nf::nf_register_phi3();

    /* Init tokenizer */
    nf::Tokenizer tokenizer(*model);

    /* Init Metal */
    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    if (nf_plugin_register(&vt, &prov) != NF_OK ||
        nf_plugin_register_mem(&mem_vt, &prov) != NF_OK ||
        vt.init(prov) != NF_OK) {
        std::fprintf(stderr, "Metal init failed\n");
        return 1;
    }

    nf::PipelineEngine engine(4);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);

    const uint32_t max_seq = 512;

    /* Tokenize prompt */
    auto prompt_ids = tokenizer.encode(prompt);
    uint32_t prefill_seq = static_cast<uint32_t>(prompt_ids.size());

    std::fprintf(stderr, "[nf_generate] model: %u layers, %u dim, vocab %u, arch %s\n",
                 model->n_layers, model->dim, model->vocab_size,
                 model->architecture.empty() ? "llama" : model->architecture.c_str());
    std::fprintf(stderr, "[nf_generate] prompt: %u tokens, generating %u tokens\n",
                 prefill_seq, max_tokens);

    /* Create context */
    auto t0 = std::chrono::steady_clock::now();
    nf::LlamaContextPtr ctx(nullptr, nf::release_llama_context);
    if (use_paged) {
        nf::ModelConfig cfg{};
        cfg.engine = &engine;
        cfg.prov = prov;
        cfg.vt = &vt;
        cfg.mem_vt = &mem_vt;
        cfg.model = model;
        cfg.max_seq = max_seq;
        cfg.max_prefill_seq = prefill_seq;
        cfg.use_fp16 = use_fp16;
        cfg.use_paged_kv = true;
        cfg.arch_override = arch_override;
        ctx = nf::create_llama_context(cfg);
    } else {
        ctx = nf::create_llama_context(engine, *model, prov, vt, mem_vt, max_seq, prefill_seq, use_fp16);
    }
    if (!ctx) { std::fprintf(stderr, "Context creation failed\n"); return 1; }
    auto t_ctx = std::chrono::steady_clock::now();

    std::vector<int32_t> all_tokens(prompt_ids.begin(), prompt_ids.end());
    uint32_t V = model->vocab_size;

    /* Prefill */
    {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = prompt_ids.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, prefill_seq);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, prefill_seq, 0);
        sess.step().get();

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            all_tokens.data(), (uint32_t)all_tokens.size(), sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        all_tokens.push_back(tok);
        engine.destroy_graph(sg.gid);

        /* Print first generated token */
        std::printf("%s", tokenizer.id_to_piece(tok).c_str());
        std::fflush(stdout);
    }

    auto t_prefill = std::chrono::steady_clock::now();

    /* Decode loop */
    for (uint32_t step = 0; step < max_tokens - 1; ++step) {
        uint32_t step_idx = prefill_seq + step;
        int32_t last_tok = all_tokens.back();

        if (last_tok == static_cast<int32_t>(tokenizer.eos_id())) break;

        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = last_tok;
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, step_idx);
        sess.step().get();

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            all_tokens.data(), (uint32_t)all_tokens.size(), sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        all_tokens.push_back(tok);
        engine.destroy_graph(sg.gid);

        /* Stream output */
        std::printf("%s", tokenizer.id_to_piece(tok).c_str());
        std::fflush(stdout);
    }

    auto t_end = std::chrono::steady_clock::now();
    std::printf("\n");

    /* Stats to stderr */
    double ctx_ms = std::chrono::duration<double, std::milli>(t_ctx - t0).count();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_ctx).count();
    double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
    uint32_t n_generated = static_cast<uint32_t>(all_tokens.size()) - prefill_seq;
    double tps = (decode_ms > 0) ? (n_generated * 1000.0 / decode_ms) : 0;

    std::fprintf(stderr, "\n[nf_generate] ctx: %.0f ms, prefill: %.0f ms, "
                 "decode: %.0f ms (%u tokens, %.1f tok/s)\n",
                 ctx_ms, prefill_ms, decode_ms, n_generated, tps);

    vt.shutdown(prov);
    nf::gguf_close(model);
    return 0;
}
