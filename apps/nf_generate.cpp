/**
 * @file nf_generate.cpp
 * @brief End-to-end text generation CLI
 *
 * Phase 24: Tokenizer Integration & MatMul Optimization.
 * Phase 31: Multi-Architecture support (--arch flag, auto-detection).
 * Phase 34: Chat mode, streaming callback, error handling, signal handling.
 * Phase 44: NeuralOS auto-enable with --paged, --no-neuralOS opt-out.
 *
 * Usage: nf_generate <gguf_path> "prompt text" [--max-tokens N] [--temperature T]
 *                    [--top-k K] [--top-p P] [--seed S] [--fp16] [--arch NAME]
 *                    [--chat] [--chat-format FORMAT]
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"
#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "model/arch_registry.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "model/chat_template.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

/* Phase 44: NeuralOS runtime (auto with --paged) */
#include "neuralOS/kernel/NeuralOSRuntime.hpp"
#include "neuralOS/compiler/compiler_pipeline.hpp"
#include "neuralOS/compiler/dag_to_nfir.hpp"
#include "neuralOS/compiler/nfir_annotate.hpp"

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

/* Phase 35-F: Graceful shutdown */
static volatile sig_atomic_t g_stop = 0;
static void signal_handler(int) { g_stop = 1; }

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
        "  --arch NAME      override architecture (llama/mistral/phi3/qwen2/gemma/mixtral)\n"
        "  --chat           enable chat template mode\n"
        "  --chat-format F  chatml|llama|phi3 (default: auto)\n"
        "  --no-neuralOS    disable NeuralOS runtime (default: auto with --paged)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* gguf_path = argv[1];
    std::string prompt = argv[2];

    /* Phase 35-F: Install signal handlers */
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

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
    bool chat_mode = false;
    bool no_neuralOS = false;
    const char* arch_override = nullptr;
    const char* chat_format_str = nullptr;

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
        else if (std::strcmp(argv[i], "--chat") == 0)
            chat_mode = true;
        else if (std::strcmp(argv[i], "--chat-format") == 0 && i + 1 < argc)
            chat_format_str = argv[++i];
        else if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc)
            arch_override = argv[++i];
        else if (std::strcmp(argv[i], "--no-neuralOS") == 0)
            no_neuralOS = true;
    }
    if (!has_seed) sp.seed = static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(sp.seed);

    /* Phase 44: NeuralOS auto-enable when paged KV is active */
    bool use_neuralOS = use_paged && !no_neuralOS;

    /* Load model */
    auto* model = nf::gguf_open(gguf_path);
    if (!model) { std::fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path); return 1; }

    /* Phase 31: override architecture if requested */
    if (arch_override) model->architecture = arch_override;

    /* Register all known architectures */
    nf::nf_register_llama();
    nf::nf_register_mistral();
    nf::nf_register_phi3();
    nf::nf_register_qwen2();
    nf::nf_register_gemma();
    nf::nf_register_mixtral();

    /* Init tokenizer */
    nf::Tokenizer tokenizer(*model);

    /* Phase 34-D: Chat template mode */
    if (chat_mode) {
        nf::ChatFormat fmt = nf::ChatFormat::AUTO;
        if (chat_format_str) {
            if (std::strcmp(chat_format_str, "chatml") == 0) fmt = nf::ChatFormat::CHATML;
            else if (std::strcmp(chat_format_str, "llama") == 0) fmt = nf::ChatFormat::LLAMA;
            else if (std::strcmp(chat_format_str, "phi3") == 0) fmt = nf::ChatFormat::PHI3;
        }
        const char* arch_for_chat = arch_override ? arch_override :
            (model->architecture.empty() ? "llama" : model->architecture.c_str());
        std::vector<nf::ChatMessage> msgs;
        msgs.push_back({nf::ChatRole::USER, prompt});
        prompt = nf::apply_chat_template(msgs, fmt, arch_for_chat);
        std::fprintf(stderr, "[nf_generate] chat mode: %s format\n",
                     chat_format_str ? chat_format_str : "auto");
    }

    /* Phase 34-C: Streaming callback setup */
    nf_session_callbacks stream_cb{};
    stream_cb.on_token = nullptr;
    stream_cb.userdata = nullptr;

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

    /* Phase 44: NeuralOS runtime (auto with --paged, opt-out with --no-neuralOS) */
    std::unique_ptr<nf::PagedKVCache> nos_kv;
    std::unique_ptr<nf::RequestScheduler> nos_sched;
    std::unique_ptr<neuralOS::L2::NeuralOSRuntime> nos_runtime;
    if (use_neuralOS) {
        nos_kv = std::make_unique<nf::PagedKVCache>();
        nos_kv->init(256, 16, model->n_layers,
                     model->n_kv_heads ? model->n_kv_heads : model->n_heads,
                     model->dim / model->n_heads);
        nos_sched = std::make_unique<nf::RequestScheduler>();
        nos_runtime = std::make_unique<neuralOS::L2::NeuralOSRuntime>(
            nos_kv.get(), nos_sched.get());

        nf::InferenceRequest ir;
        ir.prompt_tokens.assign(prompt_ids.begin(), prompt_ids.end());
        ir.max_new_tokens = max_tokens;
        nos_runtime->submit(ir);

        std::fprintf(stderr, "[nf_generate] NeuralOS runtime enabled\n");
    }

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
    nf_session_state session_state = NF_SESSION_RUNNING;
    for (uint32_t step = 0; step < max_tokens - 1 && !g_stop; ++step) {
        uint32_t step_idx = prefill_seq + step;
        int32_t last_tok = all_tokens.back();

        if (last_tok == static_cast<int32_t>(tokenizer.eos_id())) break;

        /* Phase 43: NeuralOS schedule_step before each decode */
        if (nos_runtime) nos_runtime->schedule_step();
        if (nos_runtime) nos_runtime->cfs()->account_tokens(0, 1, false);

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

        /* Stream output + callback */
        std::string piece = tokenizer.id_to_piece(tok);
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
        if (stream_cb.on_token)
            stream_cb.on_token(tok, piece.c_str(), stream_cb.userdata);
    }
    session_state = g_stop ? NF_SESSION_IDLE : NF_SESSION_IDLE;
    (void)session_state;

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

    /* Phase 44: NeuralOS runtime stats */
    if (nos_runtime) {
        nos_runtime->complete(0);  /* mark the single request complete */
        auto& st = nos_runtime->stats();
        std::fprintf(stderr, "[nf_generate] NeuralOS: submitted=%llu completed=%llu "
                     "preempted=%llu pages_out=%u prefix_hits=%u\n",
                     (unsigned long long)st.total_submitted,
                     (unsigned long long)st.total_completed,
                     (unsigned long long)st.total_preempted,
                     st.pages_out, st.prefix_hits);
    }

    /* Phase 45B: NFIR bridge — lift real DAG through compiler pipeline */
    {
        auto sg_ir = nf::build_llama_step_graph(*ctx, 1);

        std::vector<neuralOS::L1::DagNodeInfo> dag_nodes;
        dag_nodes.push_back({sg_ir.embed_id, "embedding_lookup", 1, 1});
        for (auto& lid : sg_ir.layer_ids) {
            dag_nodes.push_back({lid.attn_norm,  "rms_norm",         1, 1});
            dag_nodes.push_back({lid.q_lin,      "linear",           2, 1});
            dag_nodes.push_back({lid.k_lin,      "linear",           2, 1});
            dag_nodes.push_back({lid.v_lin,      "linear",           2, 1});
            dag_nodes.push_back({lid.q_rope,     "rope",             1, 1});
            dag_nodes.push_back({lid.k_rope,     "rope",             1, 1});
            dag_nodes.push_back({lid.cached_attn,"causal_attention",  3, 1});
            dag_nodes.push_back({lid.o_lin,      "linear",           2, 1});
            dag_nodes.push_back({lid.resid_add1, "element_add",      2, 1});
            dag_nodes.push_back({lid.ffn_norm,   "rms_norm",         1, 1});
            dag_nodes.push_back({lid.gate_lin,   "linear",           2, 1});
            dag_nodes.push_back({lid.silu_node,  "silu",             1, 1});
            dag_nodes.push_back({lid.up_lin,     "linear",           2, 1});
            dag_nodes.push_back({lid.elem_mul,   "element_mul",      2, 1});
            dag_nodes.push_back({lid.down_lin,   "linear",           2, 1});
            dag_nodes.push_back({lid.resid_add2, "element_add",      2, 1});
        }
        dag_nodes.push_back({sg_ir.final_norm_id, "rms_norm", 1, 1});
        dag_nodes.push_back({sg_ir.lm_head_id,    "linear",   2, 1});

        auto ir_graph = neuralOS::L1::lift_nodes_to_nfir(
            dag_nodes.data(), static_cast<uint32_t>(dag_nodes.size()));

        neuralOS::L1::CompilerPipeline compiler;
        auto cr = compiler.run(&ir_graph);

        auto ann = neuralOS::L1::annotate_from_nfir(ir_graph);

        std::fprintf(stderr, "[nf_generate] Compiler: ops=%u removed=%u merged=%u "
                     "shapes=%u fusions=%u annotated=%zu\n",
                     ir_graph.num_ops(), cr.ops_removed, cr.ops_merged,
                     cr.shapes_inferred, cr.fusions_found, ann.annotations.size());

        engine.destroy_graph(sg_ir.gid);
    }

    vt.shutdown(prov);
    nf::gguf_close(model);
    return 0;
}
