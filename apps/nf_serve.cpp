/**
 * @file nf_serve.cpp
 * @brief Phase 34-E / 44: OpenAI-compatible HTTP inference server
 *
 * Embedded HTTP/1.1 server with SSE streaming. No external dependencies.
 * Endpoints:
 *   POST /v1/chat/completions  — ChatML streaming (SSE)
 *   POST /v1/completions       — Text completion
 *   GET  /v1/models            — Model list
 *   GET  /health               — Health check
 *   GET  /metrics              — Prometheus metrics
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"
#include "neuralOS/ddi/metrics.h"
#include "model/gguf_loader.hpp"
#include "model/llama_dag_builder.hpp"
#include "model/arch_registry.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "model/chat_template.hpp"

/* Phase 43: NeuralOS runtime (optional) */
#include "neuralOS/kernel/NeuralOSRuntime.hpp"
#include "neuralOS/kernel/BatchInferenceLoop.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

static volatile sig_atomic_t g_stop = 0;
static void sig_handler(int) { g_stop = 1; }

/* ---- Minimal JSON helpers ---- */

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out;
}

static std::string json_get_string(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + needle.size());
    if (pos == std::string::npos) return "";
    ++pos;
    auto end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

static int json_get_int(const std::string& json, const char* key, int def) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    return std::atoi(json.c_str() + pos + 1);
}

static double json_get_double(const std::string& json, const char* key, double def) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    return std::atof(json.c_str() + pos + 1);
}

static bool json_get_bool(const std::string& json, const char* key, bool def) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    auto rest = json.substr(pos + 1);
    auto t = rest.find("true");
    auto f = rest.find("false");
    if (t != std::string::npos && (f == std::string::npos || t < f)) return true;
    if (f != std::string::npos) return false;
    return def;
}

/* Extract messages array content from chat completions request */
static std::string extract_prompt_from_messages(const std::string& json, const char* arch) {
    std::vector<nf::ChatMessage> msgs;
    /* Simple extraction: find "messages" array, parse role/content pairs */
    auto mpos = json.find("\"messages\"");
    if (mpos == std::string::npos) return "";
    auto arr_start = json.find('[', mpos);
    if (arr_start == std::string::npos) return "";
    auto arr_end = json.find(']', arr_start);
    if (arr_end == std::string::npos) return "";
    std::string arr = json.substr(arr_start, arr_end - arr_start + 1);

    /* Parse each {role, content} object */
    size_t pos = 0;
    while (pos < arr.size()) {
        auto obj_start = arr.find('{', pos);
        if (obj_start == std::string::npos) break;
        auto obj_end = arr.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        std::string obj = arr.substr(obj_start, obj_end - obj_start + 1);

        std::string role = json_get_string(obj, "role");
        std::string content = json_get_string(obj, "content");

        nf::ChatRole cr = nf::ChatRole::USER;
        if (role == "system") cr = nf::ChatRole::SYSTEM;
        else if (role == "assistant") cr = nf::ChatRole::ASSISTANT;
        msgs.push_back({cr, content});
        pos = obj_end + 1;
    }

    return nf::apply_chat_template(msgs, nf::ChatFormat::AUTO, arch);
}

/* ---- HTTP helpers ---- */

struct HttpRequest {
    std::string method, path, body;
    int content_length = 0;
};

static HttpRequest parse_http_request(int fd) {
    HttpRequest req;
    char buf[8192];
    std::string raw;

    /* Read headers */
    while (raw.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) return req;
        buf[n] = '\0';
        raw += buf;
    }

    /* Parse method + path */
    auto sp1 = raw.find(' ');
    if (sp1 != std::string::npos) {
        req.method = raw.substr(0, sp1);
        auto sp2 = raw.find(' ', sp1 + 1);
        if (sp2 != std::string::npos)
            req.path = raw.substr(sp1 + 1, sp2 - sp1 - 1);
    }

    /* Content-Length */
    auto cl = raw.find("Content-Length:");
    if (cl == std::string::npos) cl = raw.find("content-length:");
    if (cl != std::string::npos) {
        req.content_length = std::atoi(raw.c_str() + cl + 15);
    }

    /* Extract body */
    auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end != std::string::npos) {
        req.body = raw.substr(hdr_end + 4);
    }

    /* Read remaining body */
    while ((int)req.body.size() < req.content_length) {
        ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';
        req.body += buf;
    }

    return req;
}

static void send_response(int fd, int status, const char* content_type,
                           const std::string& body) {
    const char* status_text = (status == 200) ? "OK" : (status == 404) ? "Not Found" : "Bad Request";
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n",
        status, status_text, content_type, body.size());
    send(fd, header, hlen, 0);
    send(fd, body.c_str(), body.size(), 0);
}

static void send_sse_event(int fd, const std::string& data) {
    std::string event = "data: " + data + "\n\n";
    send(fd, event.c_str(), event.size(), 0);
}

static void send_sse_headers(int fd) {
    const char* h =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send(fd, h, strlen(h), 0);
}

/* ---- Server state ---- */

struct ServerState {
    nf::GGUFModel* model = nullptr;
    nf::Tokenizer* tokenizer = nullptr;
    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    nf::PipelineEngine* engine = nullptr;
    std::string model_name;
    const char* arch = nullptr;
    bool use_fp16 = false;
    bool use_paged = false;
    bool use_neuralOS = false;
    std::mutex infer_mu;  /* serialize inference for now */

    /* Phase 43: NeuralOS runtime (nullptr when disabled) */
    std::unique_ptr<nf::PagedKVCache>                  nos_kv;
    std::unique_ptr<nf::RequestScheduler>              nos_sched;
    std::unique_ptr<neuralOS::kernel::NeuralOSRuntime>     nos_runtime;
    std::unique_ptr<neuralOS::kernel::BatchInferenceLoop>  nos_loop;
};

static void handle_completions(int fd, const HttpRequest& req, ServerState& state, bool is_chat) {
    std::string prompt;
    if (is_chat) {
        prompt = extract_prompt_from_messages(req.body, state.arch);
    } else {
        prompt = json_get_string(req.body, "prompt");
    }

    if (prompt.empty()) {
        send_response(fd, 400, "application/json", "{\"error\":\"empty prompt\"}");
        return;
    }

    int max_tokens = json_get_int(req.body, "max_tokens", 128);
    double temperature = json_get_double(req.body, "temperature", 0.8);
    bool stream = json_get_bool(req.body, "stream", false);

    nf::SamplerParams sp{};
    sp.temperature = (float)temperature;
    sp.top_k = json_get_int(req.body, "top_k", 40);
    sp.top_p = (float)json_get_double(req.body, "top_p", 0.95);
    sp.repeat_penalty = 1.1f;
    sp.seed = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(sp.seed);

    std::lock_guard<std::mutex> lk(state.infer_mu);

    /* Phase 44: NeuralOS schedule step before inference */
    if (state.nos_runtime) state.nos_runtime->schedule_step();

    auto prompt_ids = state.tokenizer->encode(prompt);
    uint32_t prefill_seq = (uint32_t)prompt_ids.size();
    uint32_t V = state.model->vocab_size;

    nf::ModelConfig cfg{};
    cfg.engine = state.engine;
    cfg.prov = state.prov;
    cfg.vt = &state.vt;
    cfg.mem_vt = &state.mem_vt;
    cfg.model = state.model;
    cfg.max_seq = 512;
    cfg.max_prefill_seq = prefill_seq;
    cfg.use_fp16 = state.use_fp16;
    cfg.use_paged_kv = state.use_paged;

    auto ctx = nf::create_llama_context(cfg);
    if (!ctx) {
        send_response(fd, 500, "application/json", "{\"error\":\"context creation failed\"}");
        nf_metrics_record_request(0, 1);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<int32_t> all_tokens(prompt_ids.begin(), prompt_ids.end());

    /* Prefill */
    {
        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = prompt_ids.back();
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, prefill_seq);
        nf::PipelineEngine::Session sess(*state.engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, prefill_seq, 0);
        sess.step().get();

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            all_tokens.data(), (uint32_t)all_tokens.size(), sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        all_tokens.push_back(tok);
        state.engine->destroy_graph(sg.gid);
    }

    auto t_prefill = std::chrono::steady_clock::now();
    uint64_t prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(t_prefill - t0).count();
    nf_metrics_record_prefill(prefill_us, prefill_seq);

    if (stream) send_sse_headers(fd);

    std::string generated_text;
    std::string first_piece = state.tokenizer->id_to_piece(all_tokens.back());
    generated_text += first_piece;

    if (stream) {
        std::string chunk = "{\"choices\":[{\"delta\":{\"content\":\"" +
            json_escape(first_piece) + "\"},\"index\":0}]}";
        send_sse_event(fd, chunk);
    }

    /* Decode loop */
    for (int step = 0; step < max_tokens - 1 && !g_stop; ++step) {
        uint32_t step_idx = prefill_seq + step;
        int32_t last_tok = all_tokens.back();
        /* Phase 44: CFS token accounting */
        if (state.nos_runtime) state.nos_runtime->cfs()->account_tokens(0, 1, false);
        if (last_tok == (int32_t)state.tokenizer->eos_id()) break;

        void* p; ctx->token_buf.ops.map(ctx->token_buf.buf, &p);
        ((int32_t*)p)[0] = last_tok;
        ctx->token_buf.ops.unmap(ctx->token_buf.buf);

        auto sg = nf::build_llama_step_graph(*ctx, 1);
        nf::PipelineEngine::Session sess(*state.engine, sg.gid);
        nf::inject_step_push_constants(*ctx, sg, sess, 1, step_idx);
        sess.step().get();

        ctx->logits.ops.cache_sync(ctx->logits.buf, NF_CACHE_INVALIDATE, 0, 0);
        ctx->logits.ops.map(ctx->logits.buf, &p);
        int32_t tok = nf::sample_token((float*)p, V,
            all_tokens.data(), (uint32_t)all_tokens.size(), sp, rng);
        ctx->logits.ops.unmap(ctx->logits.buf);
        all_tokens.push_back(tok);
        state.engine->destroy_graph(sg.gid);

        std::string piece = state.tokenizer->id_to_piece(tok);
        generated_text += piece;

        if (stream) {
            std::string chunk = "{\"choices\":[{\"delta\":{\"content\":\"" +
                json_escape(piece) + "\"},\"index\":0}]}";
            send_sse_event(fd, chunk);
        }
    }

    /* Phase 44: Complete request in NeuralOS */
    if (state.nos_runtime) state.nos_runtime->complete(0);

    auto t_end = std::chrono::steady_clock::now();
    uint64_t decode_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_prefill).count();
    uint32_t n_gen = (uint32_t)all_tokens.size() - prefill_seq;
    nf_metrics_record_decode(decode_us, n_gen);
    nf_metrics_record_request(1, 0);

    if (stream) {
        send_sse_event(fd, "[DONE]");
    } else {
        /* Non-streaming: return full response */
        std::string resp;
        if (is_chat) {
            resp = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" +
                json_escape(generated_text) + "\"},\"index\":0,\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"prompt_tokens\":" + std::to_string(prefill_seq) +
                ",\"completion_tokens\":" + std::to_string(n_gen) + "}}";
        } else {
            resp = "{\"choices\":[{\"text\":\"" + json_escape(generated_text) +
                "\",\"index\":0,\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"prompt_tokens\":" + std::to_string(prefill_seq) +
                ",\"completion_tokens\":" + std::to_string(n_gen) + "}}";
        }
        send_response(fd, 200, "application/json", resp);
    }
}

static void handle_client(int fd, ServerState& state) {
    auto req = parse_http_request(fd);

    if (req.method == "GET" && req.path == "/health") {
        send_response(fd, 200, "application/json", "{\"status\":\"ok\"}");
    } else if (req.method == "GET" && req.path == "/v1/models") {
        std::string resp = "{\"data\":[{\"id\":\"" + json_escape(state.model_name) +
            "\",\"object\":\"model\"}]}";
        send_response(fd, 200, "application/json", resp);
    } else if (req.method == "GET" && req.path == "/metrics") {
        char buf[4096];
        size_t n = nf_metrics_to_prometheus(buf, sizeof(buf));
        std::string metrics(buf, n);
        /* Phase 43: Append NeuralOS runtime stats if available */
        if (state.nos_runtime) {
            auto& st = state.nos_runtime->stats();
            char nos_buf[512];
            int nos_n = snprintf(nos_buf, sizeof(nos_buf),
                "nf_neuralOS_submitted %llu\n"
                "nf_neuralOS_completed %llu\n"
                "nf_neuralOS_preempted %llu\n"
                "nf_neuralOS_pages_out %u\n"
                "nf_neuralOS_prefix_hits %u\n",
                (unsigned long long)st.total_submitted,
                (unsigned long long)st.total_completed,
                (unsigned long long)st.total_preempted,
                st.pages_out, st.prefix_hits);
            metrics.append(nos_buf, nos_n);
        }
        send_response(fd, 200, "text/plain", metrics);
    } else if (req.method == "POST" && req.path == "/v1/chat/completions") {
        handle_completions(fd, req, state, true);
    } else if (req.method == "POST" && req.path == "/v1/completions") {
        handle_completions(fd, req, state, false);
    } else {
        send_response(fd, 404, "application/json", "{\"error\":\"not found\"}");
    }

    close(fd);
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <gguf_path> [options]\n"
        "Options:\n"
        "  --port N     (default: 8080)\n"
        "  --threads N  (default: 4)\n"
        "  --fp16       use FP16 inference\n"
        "  --paged      use paged KV cache\n"
        "  --arch NAME  override architecture\n"
        "  --no-neuralOS disable NeuralOS runtime (default: auto with --paged)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char* gguf_path = argv[1];
    int port = 8080;
    int n_threads = 4;
    bool use_fp16 = false;
    bool use_paged = false;
    bool no_neuralOS = false;
    const char* arch_override = nullptr;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            n_threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fp16") == 0)
            use_fp16 = true;
        else if (std::strcmp(argv[i], "--paged") == 0)
            use_paged = true;
        else if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc)
            arch_override = argv[++i];
        else if (std::strcmp(argv[i], "--no-neuralOS") == 0)
            no_neuralOS = true;
    }

    /* Phase 44: NeuralOS auto-enable when paged KV is active */
    bool use_neuralOS = use_paged && !no_neuralOS;

    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);
    std::signal(SIGPIPE, SIG_IGN);

    /* Load model */
    auto* model = nf::gguf_open(gguf_path);
    if (!model) { std::fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path); return 1; }
    if (arch_override) model->architecture = arch_override;

    nf::nf_register_llama();
    nf::nf_register_mistral();
    nf::nf_register_phi3();
    nf::nf_register_qwen2();
    nf::nf_register_gemma();
    nf::nf_register_mixtral();

    nf::Tokenizer tokenizer(*model);

    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    if (nf_plugin_register(&vt, &prov) != NF_OK ||
        nf_plugin_register_mem(&mem_vt, &prov) != NF_OK ||
        vt.init(prov) != NF_OK) {
        std::fprintf(stderr, "Metal init failed\n");
        return 1;
    }

    nf::PipelineEngine engine(n_threads);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);

    ServerState state;
    state.model = model;
    state.tokenizer = &tokenizer;
    state.prov = prov;
    state.vt = vt;
    state.mem_vt = mem_vt;
    state.engine = &engine;
    state.model_name = gguf_path;
    state.arch = arch_override ? arch_override :
        (model->architecture.empty() ? "llama" : model->architecture.c_str());
    state.use_fp16 = use_fp16;
    state.use_paged = use_paged;
    state.use_neuralOS = use_neuralOS;

    /* Phase 43: Initialize NeuralOS runtime if enabled */
    if (use_neuralOS) {
        state.nos_kv = std::make_unique<nf::PagedKVCache>();
        state.nos_kv->init(512, 16, model->n_layers,
                           model->n_kv_heads ? model->n_kv_heads : model->n_heads,
                           model->dim / model->n_heads);
        state.nos_sched = std::make_unique<nf::RequestScheduler>();
        state.nos_runtime = std::make_unique<neuralOS::kernel::NeuralOSRuntime>(
            state.nos_kv.get(), state.nos_sched.get());
        state.nos_loop = std::make_unique<neuralOS::kernel::BatchInferenceLoop>(
            state.nos_runtime.get(), state.nos_kv.get(), state.nos_sched.get());
        std::fprintf(stderr, "[nf_serve] NeuralOS runtime enabled\n");
    }

    nf_metrics_reset();

    /* Create server socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return 1;
    }
    if (listen(server_fd, 32) < 0) {
        perror("listen"); close(server_fd); return 1;
    }

    std::fprintf(stderr, "[nf_serve] listening on port %d (model: %s, %s)\n",
                 port, gguf_path, use_fp16 ? "FP16" : "FP32");

    /* Accept loop */
    while (!g_stop) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        /* Use a timeout so we can check g_stop */
        struct timeval tv{};
        tv.tv_sec = 1;
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(server_fd, &fds);
        int sel = select(server_fd + 1, &fds, nullptr, nullptr, &tv);
        if (sel <= 0) continue;

        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        /* Handle in detached thread */
        std::thread([client_fd, &state]() {
            handle_client(client_fd, state);
        }).detach();
    }

    std::fprintf(stderr, "\n[nf_serve] shutting down...\n");
    close(server_fd);
    vt.shutdown(prov);
    nf::gguf_close(model);
    return 0;
}
