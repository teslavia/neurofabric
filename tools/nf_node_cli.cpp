/**
 * @file nf_node_cli.cpp
 * @brief Phase 11 — Distributed Router CLI (Cloud/Edge)
 *
 * Single binary, two modes:
 *   --mode=cloud  (Mac):     sends inference request over TCP to remote edge
 *   --mode=edge   (Rock 5B+): listens for requests, dispatches via RKNN provider
 *
 * Wire protocol: same nf_frame_header + nf_tensor_wire from neuro_network_protocol.h
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"
#include "neurofabric/neuro_network_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

/* POSIX sockets */
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <signal.h>

#ifdef NF_HAS_RKNN_SDK
#include <rknn_api.h>
#endif

/* ================================================================== */
/*  Socket Helpers                                                     */
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

static nf_tensor_wire make_wire(nf_dtype dtype, uint32_t ndim,
                                 const uint64_t* shape, uint64_t payload) {
    nf_tensor_wire tw{};
    tw.dtype = static_cast<uint8_t>(dtype);
    tw.ndim  = static_cast<uint8_t>(ndim);
    tw.layout = static_cast<uint16_t>(NF_LAYOUT_NHWC);
    for (uint32_t i = 0; i < ndim && i < NF_MAX_DIMS; ++i)
        tw.shape[i] = shape[i];
    tw.payload_bytes = payload;
    return tw;
}

/* ================================================================== */
/*  Cloud Mode — send inference request to remote edge node             */
/* ================================================================== */

static int run_cloud(const char* remote_host, uint16_t remote_port) {
    std::printf("[cloud] connecting to %s:%u...\n", remote_host, remote_port);

    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(remote_port);
    if (::inet_pton(AF_INET, remote_host, &addr.sin_addr) <= 0) {
        std::fprintf(stderr, "[cloud] invalid address: %s\n", remote_host);
        ::close(sock); return 1;
    }
    if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("connect"); ::close(sock); return 1;
    }
    std::printf("[cloud] connected\n");

    /* Generate dummy yolov5s input: 1x3x640x640 uint8 = 1,228,800 bytes */
    uint64_t shape[4] = {1, 3, 640, 640};
    uint64_t payload_sz = 1 * 3 * 640 * 640;
    std::vector<uint8_t> input_data(payload_sz, 128);

    /* Send NF_OP_TASK_SUBMIT frame */
    nf_tensor_wire tw = make_wire(NF_DTYPE_U8, 4, shape, payload_sz);
    nf_frame_header hdr = make_header(NF_OP_TASK_SUBMIT, "rknn_subgraph",
                                       1, 0, payload_sz);

    auto t0 = std::chrono::steady_clock::now();

    if (!send_all(sock, &hdr, sizeof(hdr)) ||
        !send_all(sock, &tw, sizeof(tw)) ||
        !send_all(sock, input_data.data(), payload_sz)) {
        std::fprintf(stderr, "[cloud] send failed\n");
        ::close(sock); return 1;
    }
    std::printf("[cloud] sent %zu bytes input\n", static_cast<size_t>(payload_sz));

    /* Receive NF_OP_TASK_COMPLETE response */
    nf_frame_header resp{};
    if (!recv_all(sock, &resp, sizeof(resp))) {
        std::fprintf(stderr, "[cloud] recv header failed\n");
        ::close(sock); return 1;
    }
    if (resp.magic != NF_PROTO_MAGIC || resp.opcode != NF_OP_TASK_COMPLETE) {
        std::fprintf(stderr, "[cloud] bad response: opcode=0x%02x\n", resp.opcode);
        ::close(sock); return 1;
    }

    /* Receive output tensor descriptors + payloads */
    uint64_t total_out = 0;
    for (uint8_t j = 0; j < resp.n_output_tensors; ++j) {
        nf_tensor_wire out_tw{};
        if (!recv_all(sock, &out_tw, sizeof(out_tw))) {
            std::fprintf(stderr, "[cloud] recv tensor desc failed\n");
            ::close(sock); return 1;
        }
        std::vector<uint8_t> out_data(out_tw.payload_bytes);
        if (!recv_all(sock, out_data.data(), out_tw.payload_bytes)) {
            std::fprintf(stderr, "[cloud] recv payload failed\n");
            ::close(sock); return 1;
        }
        /* Count non-zero bytes */
        size_t nz = 0;
        for (size_t k = 0; k < out_data.size(); ++k)
            if (out_data[k] != 0) ++nz;
        std::printf("[cloud] output[%u]: %zu bytes, non-zero=%zu, shape=[",
                    j, static_cast<size_t>(out_tw.payload_bytes), nz);
        for (uint8_t d = 0; d < out_tw.ndim; ++d)
            std::printf("%s%llu", d ? "," : "", (unsigned long long)out_tw.shape[d]);
        std::printf("]\n");
        total_out += out_tw.payload_bytes;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[cloud] round-trip: %.1f ms, received %zu bytes\n",
                ms, static_cast<size_t>(total_out));

    /* Send shutdown */
    nf_frame_header shut = make_header(NF_OP_SHUTDOWN, nullptr, 0, 0, 0);
    send_all(sock, &shut, sizeof(shut));

    ::close(sock);
    return 0;
}

/* ================================================================== */
/*  Edge Mode — listen for requests, dispatch via RKNN provider         */
/* ================================================================== */

/* RKNN plugin entry points — linked directly */
extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static int run_edge(uint16_t port, const char* model_path) {
    std::printf("[edge] starting on port %u, model=%s\n", port, model_path);

    signal(SIGPIPE, SIG_IGN);

    /* Load model file */
    struct stat sb;
    if (::stat(model_path, &sb) != 0) {
        std::fprintf(stderr, "[edge] model not found: %s\n", model_path);
        return 1;
    }
    int model_fd = ::open(model_path, O_RDONLY);
    if (model_fd < 0) { std::perror("open model"); return 1; }
    void* model_data = ::mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, model_fd, 0);
    ::close(model_fd);
    if (model_data == MAP_FAILED) { std::perror("mmap model"); return 1; }
    std::printf("[edge] model loaded: %.1f KB\n", sb.st_size / 1024.0);

    /* Register RKNN provider */
    nf_provider prov = nullptr;
    nf_provider_vtable vt{};
    nf_provider_mem_vtable mem_vt{};
    nf_plugin_register(&vt, &prov);
    nf_plugin_register_mem(&mem_vt);
    vt.init(prov);
    std::printf("[edge] provider: %s\n", vt.get_name(prov));

    /* Create TCP server socket */
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { std::perror("socket"); return 1; }
    int opt = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (::bind(srv, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind"); ::close(srv); return 1;
    }
    ::listen(srv, 4);
    std::printf("[edge] listening on 0.0.0.0:%u\n", port);

    /* Accept loop (single-threaded, one client at a time) */
    for (;;) {
        struct sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int conn = ::accept(srv, reinterpret_cast<struct sockaddr*>(&cli), &cli_len);
        if (conn < 0) { std::perror("accept"); continue; }

        char cli_ip[INET_ADDRSTRLEN];
        ::inet_ntop(AF_INET, &cli.sin_addr, cli_ip, sizeof(cli_ip));
        std::printf("[edge] client connected: %s\n", cli_ip);

        /* Request loop for this client */
        bool client_done = false;
        while (!client_done) {
            nf_frame_header req{};
            if (!recv_all(conn, &req, sizeof(req))) break;

            if (req.magic != NF_PROTO_MAGIC) {
                std::fprintf(stderr, "[edge] bad magic: 0x%08x\n", req.magic);
                break;
            }
            if (req.opcode == NF_OP_SHUTDOWN) {
                std::printf("[edge] shutdown requested\n");
                client_done = true;
                break;
            }
            if (req.opcode != NF_OP_TASK_SUBMIT) {
                std::fprintf(stderr, "[edge] unexpected opcode: 0x%02x\n", req.opcode);
                break;
            }

            std::printf("[edge] task: %s, %u inputs\n", req.op_name, req.n_input_tensors);
            auto t0 = std::chrono::steady_clock::now();

            /* Receive input tensor descriptors + payloads */
            std::vector<nf_tensor_wire> in_wires(req.n_input_tensors);
            std::vector<std::vector<uint8_t>> in_payloads(req.n_input_tensors);
            for (uint8_t i = 0; i < req.n_input_tensors; ++i) {
                if (!recv_all(conn, &in_wires[i], sizeof(nf_tensor_wire))) {
                    client_done = true; break;
                }
                in_payloads[i].resize(in_wires[i].payload_bytes);
                if (!recv_all(conn, in_payloads[i].data(), in_wires[i].payload_bytes)) {
                    client_done = true; break;
                }
                std::printf("[edge]   input[%u]: %zu bytes\n",
                            i, static_cast<size_t>(in_wires[i].payload_bytes));
            }
            if (client_done) break;

            /* Allocate RKNN buffers for inputs, copy network data in */
            std::vector<nf_buffer> in_bufs(req.n_input_tensors);
            std::vector<nf_buffer_ops> in_ops(req.n_input_tensors);
            for (uint8_t i = 0; i < req.n_input_tensors; ++i) {
                nf_tensor_desc d{};
                d.dtype = static_cast<nf_dtype>(in_wires[i].dtype);
                d.ndim  = in_wires[i].ndim;
                for (uint8_t k = 0; k < d.ndim; ++k) d.shape[k] = in_wires[i].shape[k];
                d.size_bytes = in_wires[i].payload_bytes;

                nf_buffer_alloc_request ar{};
                ar.desc = d;
                ar.preferred = NF_MEM_DOMAIN_DMA_BUF;
                mem_vt.alloc(prov, &ar, &in_ops[i], &in_bufs[i]);

                void* ptr = nullptr;
                in_ops[i].map(in_bufs[i], &ptr);
                std::memcpy(ptr, in_payloads[i].data(), in_wires[i].payload_bytes);
                in_ops[i].unmap(in_bufs[i]);
                if (in_ops[i].cache_sync)
                    in_ops[i].cache_sync(in_bufs[i], NF_CACHE_FLUSH, 0, 0);
            }

            /* Build model blob buffer (mmap'd) */
            nf_tensor_desc model_desc{};
            model_desc.dtype = NF_DTYPE_U8;
            model_desc.ndim = 1;
            model_desc.shape[0] = static_cast<uint64_t>(sb.st_size);
            model_desc.size_bytes = static_cast<uint64_t>(sb.st_size);

            /* Create a simple wrapper buffer for the mmap'd model blob */
            struct ModelBuf {
                void* data; uint64_t size;
            };
            auto* mb = new ModelBuf{model_data, static_cast<uint64_t>(sb.st_size)};
            nf_buffer model_buf = reinterpret_cast<nf_buffer>(mb);
            nf_buffer_ops model_ops{};
            model_ops.map = [](nf_buffer self, void** out) -> nf_status {
                auto* m = reinterpret_cast<ModelBuf*>(self);
                *out = m->data; return NF_OK;
            };
            model_ops.unmap = [](nf_buffer) -> nf_status { return NF_OK; };
            model_ops.get_info = [](nf_buffer self, nf_buffer_info* info) -> nf_status {
                auto* m = reinterpret_cast<ModelBuf*>(self);
                info->desc.dtype = NF_DTYPE_U8;
                info->desc.ndim = 1;
                info->desc.shape[0] = m->size;
                info->desc.size_bytes = m->size;
                info->domain = NF_MEM_DOMAIN_MMAP;
                return NF_OK;
            };
            model_ops.release = [](nf_buffer self) -> uint32_t {
                delete reinterpret_cast<ModelBuf*>(self); return 0;
            };

            /* Probe model for output sizes */
            uint32_t n_out = 0;
            std::vector<nf_buffer> out_bufs;
            std::vector<nf_buffer_ops> out_ops;
            std::vector<uint32_t> out_sizes;

#ifdef NF_HAS_RKNN_SDK
            {
                rknn_context probe_ctx = 0;
                int ret = rknn_init(&probe_ctx, model_data,
                                    static_cast<uint32_t>(sb.st_size), 0, nullptr);
                if (ret == 0) {
                    rknn_input_output_num io_num{};
                    rknn_query(probe_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
                    n_out = io_num.n_output;
                    for (uint32_t j = 0; j < n_out; ++j) {
                        rknn_tensor_attr attr{};
                        attr.index = j;
                        rknn_query(probe_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
                        out_sizes.push_back(attr.size);
                    }
                    rknn_destroy(probe_ctx);
                }
            }
#else
            /* Simulation: mirror each input as an output */
            n_out = req.n_input_tensors;
            for (uint8_t j = 0; j < n_out; ++j)
                out_sizes.push_back(static_cast<uint32_t>(in_wires[j].payload_bytes));
#endif

            /* Allocate output buffers */
            for (uint32_t j = 0; j < n_out; ++j) {
                nf_tensor_desc d{};
                d.dtype = NF_DTYPE_U8;
                d.ndim = 1;
                d.shape[0] = out_sizes[j];
                d.size_bytes = out_sizes[j];

                nf_buffer_alloc_request ar{};
                ar.desc = d;
                ar.preferred = NF_MEM_DOMAIN_DMA_BUF;
                nf_buffer_ops ops{};
                nf_buffer buf = nullptr;
                mem_vt.alloc(prov, &ar, &ops, &buf);
                out_bufs.push_back(buf);
                out_ops.push_back(ops);
            }

            /* Dispatch via provider vtable (rknn_subgraph) */
            /* Build input array: [model_buf, in_bufs[0], in_bufs[1], ...] */
            uint32_t total_in = 1 + req.n_input_tensors;
            std::vector<nf_buffer> dispatch_in(total_in);
            dispatch_in[0] = model_buf;
            for (uint8_t i = 0; i < req.n_input_tensors; ++i)
                dispatch_in[i + 1] = in_bufs[i];

            /* We need input_ops accessible via the cross-dylib bridge.
             * Build a nf_task_desc so the provider can recover ops via offsetof. */
            nf_task_desc td{};
            std::strncpy(td.op_name, "rknn_subgraph", NF_MAX_OP_NAME - 1);
            td.inputs[0]    = model_buf;
            td.input_ops[0] = model_ops;
            for (uint8_t i = 0; i < req.n_input_tensors && i + 1 < NF_MAX_TASK_INPUTS; ++i) {
                td.inputs[i + 1]    = in_bufs[i];
                td.input_ops[i + 1] = in_ops[i];
            }
            td.n_inputs = total_in;
            for (uint32_t j = 0; j < n_out && j < NF_MAX_TASK_OUTPUTS; ++j) {
                td.outputs[j]    = out_bufs[j];
                td.output_ops[j] = out_ops[j];
            }
            td.n_outputs = n_out;
            td.affinity  = NF_AFFINITY_NPU;

            nf_status st = vt.dispatch(prov, td.op_name,
                                        td.inputs, td.n_inputs,
                                        td.outputs, td.n_outputs);

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("[edge] dispatch %s: %.1f ms\n",
                        st == NF_OK ? "OK" : "FAILED", ms);

            /* Send response */
            if (st == NF_OK) {
                uint64_t total_payload = 0;
                for (uint32_t j = 0; j < n_out; ++j)
                    total_payload += out_sizes[j];

                nf_frame_header resp = make_header(NF_OP_TASK_COMPLETE, td.op_name,
                                                    0, static_cast<uint8_t>(n_out),
                                                    total_payload);
                send_all(conn, &resp, sizeof(resp));

                for (uint32_t j = 0; j < n_out; ++j) {
                    uint64_t shape_1d[1] = { out_sizes[j] };
                    nf_tensor_wire otw = make_wire(NF_DTYPE_U8, 1, shape_1d, out_sizes[j]);
                    send_all(conn, &otw, sizeof(otw));

                    void* ptr = nullptr;
                    out_ops[j].map(out_bufs[j], &ptr);
                    send_all(conn, ptr, out_sizes[j]);
                    out_ops[j].unmap(out_bufs[j]);
                }
            } else {
                nf_frame_header resp = make_header(NF_OP_TASK_ERROR, td.op_name, 0, 0, 0);
                send_all(conn, &resp, sizeof(resp));
            }

            /* Cleanup this request */
            for (uint32_t j = 0; j < n_out; ++j)
                out_ops[j].release(out_bufs[j]);
            for (uint8_t i = 0; i < req.n_input_tensors; ++i)
                in_ops[i].release(in_bufs[i]);
            model_ops.release(model_buf);
        }

        ::close(conn);
        std::printf("[edge] client disconnected\n");
    }

    vt.shutdown(prov);
    ::munmap(model_data, sb.st_size);
    ::close(srv);
    return 0;
}

/* ================================================================== */
/*  Main — argument parsing                                            */
/* ================================================================== */

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --mode=cloud --remote=HOST:PORT\n"
        "  %s --mode=edge  --port=PORT --model=PATH\n", prog, prog);
}

int main(int argc, char** argv) {
    const char* mode   = nullptr;
    const char* remote = nullptr;
    const char* model  = nullptr;
    uint16_t port = 9876;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0)   mode   = argv[i] + 7;
        if (std::strncmp(argv[i], "--remote=", 9) == 0)  remote = argv[i] + 9;
        if (std::strncmp(argv[i], "--model=", 8) == 0)   model  = argv[i] + 8;
        if (std::strncmp(argv[i], "--port=", 7) == 0)    port   = static_cast<uint16_t>(std::atoi(argv[i] + 7));
    }

    if (!mode) { usage(argv[0]); return 1; }

    if (std::strcmp(mode, "cloud") == 0) {
        if (!remote) {
            std::fprintf(stderr, "cloud mode requires --remote=HOST:PORT\n");
            return 1;
        }
        /* Parse host:port */
        char host[256] = {};
        uint16_t rport = 9876;
        const char* colon = std::strrchr(remote, ':');
        if (colon) {
            size_t hlen = static_cast<size_t>(colon - remote);
            if (hlen >= sizeof(host)) hlen = sizeof(host) - 1;
            std::memcpy(host, remote, hlen);
            rport = static_cast<uint16_t>(std::atoi(colon + 1));
        } else {
            std::strncpy(host, remote, sizeof(host) - 1);
        }
        return run_cloud(host, rport);
    }

    if (std::strcmp(mode, "edge") == 0) {
        if (!model) {
            std::fprintf(stderr, "edge mode requires --model=PATH\n");
            return 1;
        }
        return run_edge(port, model);
    }

    std::fprintf(stderr, "unknown mode: %s\n", mode);
    usage(argv[0]);
    return 1;
}
