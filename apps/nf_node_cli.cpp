/**
 * @file nf_node_cli.cpp
 * @brief Phase 12 — Universal Coordinator/Worker CLI
 *
 * Single binary, three modes:
 *   --mode=local   Load .nfir, execute entire DAG locally via PipelineEngine
 *   --mode=coord   Coordinator: load .nfir, dispatch REMOTE-flagged tasks to workers
 *   --mode=worker  Worker: listen on TCP, execute tasks received from coordinator
 *
 * The local mode is the E2E smoke test: Python AOT → .nfir → GraphBuilder → DAG → verify.
 * Coordinator/worker modes reuse the Phase 11 wire protocol (nf_frame_header + nf_tensor_wire).
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/ddi/neuro_network_protocol.h"
#include "neuralOS/ddi/neuro_ir_format.h"
#include "neuralOS/kernel/PipelineEngine.hpp"
#include "neuralOS/kernel/GraphBuilder.hpp"
#include "neuralOS/kernel/ContextHub.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

/* POSIX sockets (coord/worker modes) */
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <signal.h>

/* ================================================================== */
/*  Host Buffer — malloc-backed activation allocator                   */
/* ================================================================== */

struct HostBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data = nullptr;
    nf_tensor_desc        desc{};
    bool                  mapped = false;
};

static uint32_t host_retain(nf_buffer self) {
    return reinterpret_cast<HostBuffer*>(self)->refcount.fetch_add(
        1, std::memory_order_relaxed) + 1;
}
static uint32_t host_release(nf_buffer self) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) { std::free(b->data); delete b; }
    return prev - 1;
}
static nf_status host_map(nf_buffer self, void** p) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    b->mapped = true; *p = b->data; return NF_OK;
}
static nf_status host_unmap(nf_buffer self) {
    reinterpret_cast<HostBuffer*>(self)->mapped = false; return NF_OK;
}
static nf_status host_cache(nf_buffer, nf_cache_op, uint64_t, uint64_t) {
    return NF_OK;
}
static nf_status host_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    info->desc = b->desc; info->domain = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0; info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0; return NF_OK;
}
static nf_status host_export(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_CPU; return NF_OK;
}

static nf_buffer_ops make_host_ops() {
    nf_buffer_ops ops{};
    ops.retain = host_retain; ops.release = host_release;
    ops.map = host_map; ops.unmap = host_unmap;
    ops.cache_sync = host_cache; ops.get_info = host_info;
    ops.export_handle = host_export; ops.import_handle = nullptr;
    return ops;
}

static nf_status host_alloc_fn(const nf_tensor_desc& desc,
                                nf_buffer_ops* ops, nf_buffer* buf) {
    auto* b = new HostBuffer;
    b->desc = desc;
    b->data = std::calloc(1, desc.size_bytes);
    if (!b->data) { delete b; return NF_ERROR_OUT_OF_MEMORY; }
    *ops = make_host_ops();
    *buf = reinterpret_cast<nf_buffer>(b);
    return NF_OK;
}

/* ================================================================== */
/*  Mock Provider — weighted_add + mock_relu (same as ir_loader_test)  */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    /* Recover full task descriptor via cross-dylib bridge */
    auto* desc = reinterpret_cast<const nf_task_desc*>(
        reinterpret_cast<const char*>(inputs) -
        offsetof(nf_task_desc, inputs));

    if (std::strcmp(op_name, "weighted_add") == 0 && n_in >= 2 && n_out >= 1) {
        void *w_ptr, *a_ptr, *o_ptr;
        desc->input_ops[0].map(inputs[0], &w_ptr);
        desc->input_ops[1].map(inputs[1], &a_ptr);
        desc->output_ops[0].map(outputs[0], &o_ptr);

        nf_buffer_info info{};
        desc->output_ops[0].get_info(outputs[0], &info);
        size_t count = info.desc.size_bytes / sizeof(float);

        auto* w = static_cast<const float*>(w_ptr);
        auto* a = static_cast<const float*>(a_ptr);
        auto* o = static_cast<float*>(o_ptr);
        for (size_t i = 0; i < count; ++i) o[i] = w[i] + a[i];

        desc->input_ops[0].unmap(inputs[0]);
        desc->input_ops[1].unmap(inputs[1]);
        desc->output_ops[0].unmap(outputs[0]);
        return NF_OK;
    }

    if (std::strcmp(op_name, "mock_relu") == 0 && n_in >= 1 && n_out >= 1) {
        void *in_ptr, *out_ptr;
        desc->input_ops[0].map(inputs[0], &in_ptr);
        desc->output_ops[0].map(outputs[0], &out_ptr);

        nf_buffer_info info{};
        desc->output_ops[0].get_info(outputs[0], &info);
        size_t count = info.desc.size_bytes / sizeof(float);

        auto* src = static_cast<const float*>(in_ptr);
        auto* dst = static_cast<float*>(out_ptr);
        for (size_t i = 0; i < count; ++i) dst[i] = src[i] > 0.f ? src[i] : 0.f;

        desc->input_ops[0].unmap(inputs[0]);
        desc->output_ops[0].unmap(outputs[0]);
        return NF_OK;
    }

    std::fprintf(stderr, "[mock] unknown op: %s\n", op_name);
    return NF_ERROR_NOT_FOUND;
}

/* ================================================================== */
/*  Socket Helpers (coord/worker modes)                                */
/* ================================================================== */

static bool send_all(int s, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    int flags = 0;
#ifdef __linux__
    flags = MSG_NOSIGNAL;
#endif
    while (sent < len) {
        auto n = ::send(s, p + sent, len - sent, flags);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

static bool recv_all(int s, void* data, size_t len) {
    auto* p = static_cast<uint8_t*>(data);
    size_t got = 0;
    while (got < len) {
        auto n = ::recv(s, p + got, len - got, 0);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

/* ================================================================== */
/*  Frame Helpers                                                      */
/* ================================================================== */

static nf_frame_header make_header(nf_proto_opcode op, const char* op_name,
                                    uint8_t n_in, uint8_t n_out,
                                    uint64_t payload_bytes) {
    nf_frame_header hdr{};
    hdr.magic   = NF_PROTO_MAGIC;
    hdr.version = NF_PROTO_VERSION;
    hdr.opcode  = static_cast<uint8_t>(op);
    hdr.flags   = 0;
    hdr.task_id = 1;
    hdr.seq_num = 0;
    if (op_name) std::strncpy(hdr.op_name, op_name, NF_MAX_OP_NAME - 1);
    hdr.n_input_tensors  = n_in;
    hdr.n_output_tensors = n_out;
    hdr.total_payload_bytes = payload_bytes;
    hdr.header_crc32 = nf_frame_compute_crc(&hdr);
    return hdr;
}

/* ================================================================== */
/*  Local Mode — load .nfir, build DAG, execute via PipelineEngine     */
/* ================================================================== */

static int run_local(const char* nfir_path) {
    std::printf("[local] loading %s\n", nfir_path);
    auto t0 = std::chrono::steady_clock::now();

    /* Create engine + register mock provider */
    nf::PipelineEngine engine(2);

    nf_provider_vtable mock_vt{};
    mock_vt.get_name = [](nf_provider) -> const char* { return "mock_cpu"; };
    mock_vt.dispatch = mock_dispatch;
    mock_vt.init     = [](nf_provider) -> nf_status { return NF_OK; };
    mock_vt.shutdown = [](nf_provider) {};

    nf_provider mock_prov = reinterpret_cast<nf_provider>(0x1);
    engine.register_provider(mock_prov, mock_vt, NF_AFFINITY_ANY);

    /* Build graph from .nfir */
    nf::GraphBuilder builder(engine, host_alloc_fn);
    nf_status st = builder.load(nfir_path);
    if (st != NF_OK) {
        std::fprintf(stderr, "[local] load failed: %d\n", st);
        return 1;
    }

    uint32_t graph_id = 0;
    st = builder.build(&graph_id);
    if (st != NF_OK) {
        std::fprintf(stderr, "[local] build_dag failed: %d\n", st);
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[local] graph built: id=%u (%.2f ms)\n", graph_id, load_ms);

    /* Execute DAG */
    auto future = engine.submit(graph_id);
    st = future.get();
    auto t2 = std::chrono::steady_clock::now();
    double exec_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if (st != NF_OK) {
        std::fprintf(stderr, "[local] execution failed: %d\n", st);
        return 1;
    }
    std::printf("[local] execution complete (%.2f ms)\n", exec_ms);

    /* Dump output tensor summaries */
    /* Scan all tensors; non-weight tensors that are outputs of the last node */
    std::printf("[local] total: %.2f ms (load=%.2f, exec=%.2f)\n",
                load_ms + exec_ms, load_ms, exec_ms);
    return 0;
}

/* ================================================================== */
/*  Coordinator Mode — load .nfir, dispatch REMOTE tasks to workers    */
/* ================================================================== */

static int run_coord(const char* nfir_path, const char* worker_host,
                     uint16_t worker_port) {
    std::printf("[coord] loading %s, worker=%s:%u\n",
                nfir_path, worker_host, worker_port);

    /* Connect to worker */
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(worker_port);
    if (::inet_pton(AF_INET, worker_host, &addr.sin_addr) <= 0) {
        std::fprintf(stderr, "[coord] invalid address: %s\n", worker_host);
        ::close(sock); return 1;
    }
    if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr),
                  sizeof(addr)) < 0) {
        std::perror("connect"); ::close(sock); return 1;
    }
    std::printf("[coord] connected to worker\n");

    /* Phase 14: keepalive + send/recv timeout on coordinator socket */
    {
        int ka = 1;
        ::setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &ka, sizeof(ka));
        int nd = 1;
        ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
        struct timeval tv;
        tv.tv_sec  = 30;
        tv.tv_usec = 0;
        ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    /* Create engine with mock local + network remote provider */
    nf::PipelineEngine engine(2);

    /* Local mock provider for non-remote tasks */
    nf_provider_vtable local_vt{};
    local_vt.get_name = [](nf_provider) -> const char* { return "mock_cpu"; };
    local_vt.dispatch = mock_dispatch;
    local_vt.init     = [](nf_provider) -> nf_status { return NF_OK; };
    local_vt.shutdown = [](nf_provider) {};
    nf_provider local_prov = reinterpret_cast<nf_provider>(0x1);
    engine.register_provider(local_prov, local_vt, NF_AFFINITY_CPU);

    /* Network provider for REMOTE-flagged tasks — sends to worker via TCP */
    struct NetCtx { int sock; };
    auto* net_ctx = new NetCtx{sock};

    nf_provider_vtable net_vt{};
    net_vt.get_name = [](nf_provider) -> const char* { return "net_remote"; };
    net_vt.init     = [](nf_provider) -> nf_status { return NF_OK; };
    net_vt.shutdown = [](nf_provider p) {
        auto* ctx = reinterpret_cast<NetCtx*>(p);
        nf_frame_header shut{};
        shut.magic = NF_PROTO_MAGIC; shut.version = NF_PROTO_VERSION;
        shut.opcode = static_cast<uint8_t>(NF_OP_SHUTDOWN);
        shut.header_crc32 = nf_frame_compute_crc(&shut);
        send_all(ctx->sock, &shut, sizeof(shut));
        ::close(ctx->sock);
        delete ctx;
    };

    net_vt.dispatch = [](nf_provider p, const char* op_name,
                         const nf_buffer* inputs, uint32_t n_in,
                         nf_buffer* outputs, uint32_t n_out) -> nf_status {
        auto* ctx = reinterpret_cast<NetCtx*>(p);
        auto* desc = reinterpret_cast<const nf_task_desc*>(
            reinterpret_cast<const char*>(inputs) -
            offsetof(nf_task_desc, inputs));

        /* Compute total payload size */
        uint64_t total_payload = 0;
        for (uint32_t i = 0; i < n_in; ++i) {
            nf_buffer_info info{};
            desc->input_ops[i].get_info(inputs[i], &info);
            total_payload += info.desc.size_bytes;
        }

        /* Send TASK_SUBMIT header */
        nf_frame_header hdr = make_header(NF_OP_TASK_SUBMIT, op_name,
                                           static_cast<uint8_t>(n_in),
                                           static_cast<uint8_t>(n_out),
                                           total_payload);
        if (!send_all(ctx->sock, &hdr, sizeof(hdr))) return NF_ERROR_INTERNAL;

        /* Send input tensors: wire desc + payload */
        for (uint32_t i = 0; i < n_in; ++i) {
            nf_buffer_info info{};
            desc->input_ops[i].get_info(inputs[i], &info);

            nf_tensor_wire tw{};
            tw.dtype = static_cast<uint8_t>(info.desc.dtype);
            tw.ndim  = static_cast<uint8_t>(info.desc.ndim);
            tw.layout = static_cast<uint16_t>(NF_LAYOUT_NHWC);
            for (uint32_t d = 0; d < info.desc.ndim && d < NF_MAX_DIMS; ++d)
                tw.shape[d] = info.desc.shape[d];
            tw.payload_bytes = info.desc.size_bytes;

            if (!send_all(ctx->sock, &tw, sizeof(tw))) return NF_ERROR_INTERNAL;

            void* ptr = nullptr;
            desc->input_ops[i].map(inputs[i], &ptr);
            bool ok = send_all(ctx->sock, ptr, info.desc.size_bytes);
            desc->input_ops[i].unmap(inputs[i]);
            if (!ok) return NF_ERROR_INTERNAL;
        }

        /* Receive TASK_COMPLETE response */
        nf_frame_header resp{};
        if (!recv_all(ctx->sock, &resp, sizeof(resp))) return NF_ERROR_INTERNAL;
        if (resp.opcode != NF_OP_TASK_COMPLETE) return NF_ERROR_INTERNAL;

        /* Receive output tensors */
        for (uint32_t j = 0; j < n_out && j < resp.n_output_tensors; ++j) {
            nf_tensor_wire otw{};
            if (!recv_all(ctx->sock, &otw, sizeof(otw))) return NF_ERROR_INTERNAL;

            void* ptr = nullptr;
            desc->output_ops[j].map(outputs[j], &ptr);
            bool ok = recv_all(ctx->sock, ptr, otw.payload_bytes);
            desc->output_ops[j].unmap(outputs[j]);
            if (!ok) return NF_ERROR_INTERNAL;
        }
        return NF_OK;
    };

    nf_provider net_prov = reinterpret_cast<nf_provider>(net_ctx);
    engine.register_provider(net_prov, net_vt, NF_AFFINITY_REMOTE);

    /* Build graph from .nfir */
    nf::GraphBuilder builder(engine, host_alloc_fn);
    nf_status st = builder.load(nfir_path);
    if (st != NF_OK) {
        std::fprintf(stderr, "[coord] load failed: %d\n", st);
        return 1;
    }

    uint32_t graph_id = 0;
    st = builder.build(&graph_id);
    if (st != NF_OK) {
        std::fprintf(stderr, "[coord] build_dag failed: %d\n", st);
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto future = engine.submit(graph_id);
    st = future.get();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (st != NF_OK) {
        std::fprintf(stderr, "[coord] execution failed: %d\n", st);
        return 1;
    }
    std::printf("[coord] DAG complete (%.2f ms)\n", ms);

    engine.destroy_graph(graph_id);
    net_vt.shutdown(net_prov);
    return 0;
}

/* ================================================================== */
/*  Worker Mode — listen on TCP, execute tasks from coordinator         */
/* ================================================================== */

static int run_worker(uint16_t port) {
    std::printf("[worker] listening on port %u\n", port);
    signal(SIGPIPE, SIG_IGN);

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { std::perror("socket"); return 1; }
    int opt = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (::bind(srv, reinterpret_cast<struct sockaddr*>(&addr),
               sizeof(addr)) < 0) {
        std::perror("bind"); ::close(srv); return 1;
    }
    ::listen(srv, 4);

    /* Create engine + mock provider for local execution */
    nf::PipelineEngine engine(2);
    nf_provider_vtable mock_vt{};
    mock_vt.get_name = [](nf_provider) -> const char* { return "mock_cpu"; };
    mock_vt.dispatch = mock_dispatch;
    mock_vt.init     = [](nf_provider) -> nf_status { return NF_OK; };
    mock_vt.shutdown = [](nf_provider) {};
    nf_provider mock_prov = reinterpret_cast<nf_provider>(0x1);
    engine.register_provider(mock_prov, mock_vt, NF_AFFINITY_ANY);

    /* Phase 13: ContextHub for stateful KV cache retention */
    nf::ContextHub cache(64 * 1024 * 1024, NF_EVICT_LRU);
    std::printf("[worker] ContextHub: 64 MB budget, LRU eviction\n");

    for (;;) {
        struct sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int conn = ::accept(srv, reinterpret_cast<struct sockaddr*>(&cli),
                            &cli_len);
        if (conn < 0) { std::perror("accept"); continue; }

        char cli_ip[INET_ADDRSTRLEN];
        ::inet_ntop(AF_INET, &cli.sin_addr, cli_ip, sizeof(cli_ip));
        std::printf("[worker] coordinator connected: %s\n", cli_ip);

        /* Phase 14: keepalive + recv/send timeout on accepted socket */
        {
            int ka = 1;
            ::setsockopt(conn, SOL_SOCKET, SO_KEEPALIVE, &ka, sizeof(ka));
            int nd = 1;
            ::setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
            struct timeval tv;
            tv.tv_sec  = 30;
            tv.tv_usec = 0;
            ::setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            ::setsockopt(conn, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        }

        bool done = false;
        while (!done) {
            nf_frame_header req{};
            if (!recv_all(conn, &req, sizeof(req))) break;
            if (req.magic != NF_PROTO_MAGIC) break;
            if (req.opcode == NF_OP_SHUTDOWN) {
                std::printf("[worker] shutdown requested\n");
                done = true; break;
            }
            if (req.opcode != NF_OP_TASK_SUBMIT) break;

            std::printf("[worker] task: %s, %u inputs\n",
                        req.op_name, req.n_input_tensors);
            auto t0 = std::chrono::steady_clock::now();

            /* Receive input tensors — Phase 13: detect stateful/prefix */
            std::vector<nf_buffer> in_bufs(req.n_input_tensors);
            std::vector<nf_buffer_ops> in_ops(req.n_input_tensors);
            std::vector<bool> retained(req.n_input_tensors, false);
            std::vector<int32_t> prefix_tokens;
            bool has_prefix = false;

            for (uint8_t i = 0; i < req.n_input_tensors; ++i) {
                nf_tensor_wire tw{};
                if (!recv_all(conn, &tw, sizeof(tw))) { done = true; break; }

                nf_tensor_desc d{};
                d.dtype = static_cast<nf_dtype>(tw.dtype);
                d.ndim  = tw.ndim;
                for (uint8_t k = 0; k < d.ndim; ++k) d.shape[k] = tw.shape[k];
                d.size_bytes = tw.payload_bytes;

                host_alloc_fn(d, &in_ops[i], &in_bufs[i]);
                void* ptr = nullptr;
                in_ops[i].map(in_bufs[i], &ptr);
                if (!recv_all(conn, ptr, tw.payload_bytes)) { done = true; break; }
                in_ops[i].unmap(in_bufs[i]);

                /* Phase 13: PREFIX tensor — payload is int32_t token IDs */
                if (tw.flags & NF_TENSOR_FLAG_PREFIX) {
                    size_t n_tokens = tw.payload_bytes / sizeof(int32_t);
                    in_ops[i].map(in_bufs[i], &ptr);
                    prefix_tokens.assign(
                        static_cast<const int32_t*>(ptr),
                        static_cast<const int32_t*>(ptr) + n_tokens);
                    in_ops[i].unmap(in_bufs[i]);
                    has_prefix = true;
                    std::printf("[worker] prefix: %zu tokens\n", n_tokens);
                }

                /* Phase 13: STATEFUL tensor — retain in ContextHub */
                if ((tw.flags & NF_TENSOR_FLAG_STATEFUL) && has_prefix) {
                    nf::TensorView tv(in_bufs[i], in_ops[i]);
                    in_ops[i].retain(in_bufs[i]); /* keep our handle too */
                    cache.put(prefix_tokens, "remote",
                              tv.share(), 0, 0);
                    retained[i] = true;
                    std::printf("[worker] cached stateful tensor %u "
                                "(%llu bytes)\n", i,
                                (unsigned long long)tw.payload_bytes);
                }
            }
            if (done) break;

            /* Phase 13: State-routed dispatch — check cache before exec */
            if (has_prefix) {
                auto hit = cache.get(
                    std::span<const int32_t>(prefix_tokens));
                if (hit.found) {
                    std::printf("[worker] cache HIT: %u/%zu tokens matched\n",
                                hit.match_len, prefix_tokens.size());
                    /* Inject cached tensor as input[0] (KV cache slot) */
                    nf_buffer cached_buf = hit.tensor.handle();
                    nf_buffer_ops cached_ops = hit.tensor.ops();
                    if (req.n_input_tensors > 0) {
                        /* Replace first non-prefix input with cached */
                        for (uint8_t i = 0; i < req.n_input_tensors; ++i) {
                            if (!retained[i]) {
                                in_ops[i].release(in_bufs[i]);
                                in_bufs[i] = cached_buf;
                                in_ops[i] = cached_ops;
                                in_ops[i].retain(in_bufs[i]);
                                std::printf("[worker] injected cached tensor "
                                            "at input[%u]\n", i);
                                break;
                            }
                        }
                    }
                } else {
                    std::printf("[worker] cache MISS\n");
                }
            }

            /* Allocate output buffers (same size as first input for mock) */
            nf_buffer_info first_info{};
            in_ops[0].get_info(in_bufs[0], &first_info);
            uint32_t n_out = 1;
            std::vector<nf_buffer> out_bufs(n_out);
            std::vector<nf_buffer_ops> out_ops(n_out);
            for (uint32_t j = 0; j < n_out; ++j) {
                host_alloc_fn(first_info.desc, &out_ops[j], &out_bufs[j]);
            }

            /* Build task descriptor and dispatch */
            nf_task_desc td{};
            std::strncpy(td.op_name, req.op_name, NF_MAX_OP_NAME - 1);
            td.n_inputs = req.n_input_tensors;
            td.n_outputs = n_out;
            for (uint8_t i = 0; i < req.n_input_tensors; ++i) {
                td.inputs[i] = in_bufs[i];
                td.input_ops[i] = in_ops[i];
            }
            for (uint32_t j = 0; j < n_out; ++j) {
                td.outputs[j] = out_bufs[j];
                td.output_ops[j] = out_ops[j];
            }
            td.user_data = &td;

            nf_status st = mock_dispatch(mock_prov, td.op_name,
                                          td.inputs, td.n_inputs,
                                          td.outputs, td.n_outputs);

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (st == NF_OK) {
                std::printf("[worker] done (%.2f ms), sending %u outputs\n",
                            ms, n_out);
                nf_frame_header resp = make_header(NF_OP_TASK_COMPLETE,
                    td.op_name, 0, static_cast<uint8_t>(n_out), 0);
                send_all(conn, &resp, sizeof(resp));

                for (uint32_t j = 0; j < n_out; ++j) {
                    nf_buffer_info oinfo{};
                    out_ops[j].get_info(out_bufs[j], &oinfo);
                    nf_tensor_wire otw{};
                    otw.dtype = static_cast<uint8_t>(oinfo.desc.dtype);
                    otw.ndim  = static_cast<uint8_t>(oinfo.desc.ndim);
                    for (uint32_t d = 0; d < oinfo.desc.ndim; ++d)
                        otw.shape[d] = oinfo.desc.shape[d];
                    otw.payload_bytes = oinfo.desc.size_bytes;
                    send_all(conn, &otw, sizeof(otw));

                    void* ptr = nullptr;
                    out_ops[j].map(out_bufs[j], &ptr);
                    send_all(conn, ptr, oinfo.desc.size_bytes);
                    out_ops[j].unmap(out_bufs[j]);
                }
            } else {
                nf_frame_header resp = make_header(NF_OP_TASK_ERROR,
                    td.op_name, 0, 0, 0);
                send_all(conn, &resp, sizeof(resp));
            }

            /* Cleanup — skip retained (stateful) buffers */
            for (uint32_t j = 0; j < n_out; ++j)
                out_ops[j].release(out_bufs[j]);
            for (uint8_t i = 0; i < req.n_input_tensors; ++i) {
                if (!retained[i])
                    in_ops[i].release(in_bufs[i]);
            }
        }

        ::close(conn);
        std::printf("[worker] coordinator disconnected\n");
    }

    ::close(srv);
    return 0;
}

/* ================================================================== */
/*  Main — argument parsing                                            */
/* ================================================================== */

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --mode=local  --nfir=PATH\n"
        "  %s --mode=coord  --nfir=PATH --worker=HOST:PORT\n"
        "  %s --mode=worker --port=PORT\n", prog, prog, prog);
}

int main(int argc, char** argv) {
    const char* mode   = nullptr;
    const char* nfir   = nullptr;
    const char* worker = nullptr;
    uint16_t    port   = 9876;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0)   mode   = argv[i] + 7;
        if (std::strncmp(argv[i], "--nfir=", 7) == 0)   nfir   = argv[i] + 7;
        if (std::strncmp(argv[i], "--worker=", 9) == 0)  worker = argv[i] + 9;
        if (std::strncmp(argv[i], "--port=", 7) == 0)
            port = static_cast<uint16_t>(std::atoi(argv[i] + 7));
    }

    if (!mode) { usage(argv[0]); return 1; }

    if (std::strcmp(mode, "local") == 0) {
        if (!nfir) {
            std::fprintf(stderr, "local mode requires --nfir=PATH\n");
            return 1;
        }
        return run_local(nfir);
    }

    if (std::strcmp(mode, "coord") == 0) {
        if (!nfir || !worker) {
            std::fprintf(stderr,
                "coord mode requires --nfir=PATH --worker=HOST:PORT\n");
            return 1;
        }
        char host[256] = {};
        uint16_t wport = 9876;
        const char* colon = std::strrchr(worker, ':');
        if (colon) {
            size_t hlen = static_cast<size_t>(colon - worker);
            if (hlen >= sizeof(host)) hlen = sizeof(host) - 1;
            std::memcpy(host, worker, hlen);
            wport = static_cast<uint16_t>(std::atoi(colon + 1));
        } else {
            std::strncpy(host, worker, sizeof(host) - 1);
        }
        return run_coord(nfir, host, wport);
    }

    if (std::strcmp(mode, "worker") == 0) {
        return run_worker(port);
    }

    std::fprintf(stderr, "unknown mode: %s\n", mode);
    usage(argv[0]);
    return 1;
}
