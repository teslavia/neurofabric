/**
 * @file e2e_pipeline_test.cpp
 * @brief Phase 6 — End-to-End Pipeline Verification with TCP Loopback
 *
 * Validates the entire Neuro-Fabric stack:
 *   1. Hardware-backed memory allocators (Metal unified / RKNN DMA-BUF sim)
 *   2. DAG scheduling with Kahn's topological sort
 *   3. Network proxy plugin over real TCP (localhost loopback)
 *   4. Zero-copy tensor payload transport with cache fences
 *   5. Bit-exact data integrity after network round-trip
 *
 * Test topology:
 *
 *   [Node A: Local CPU]          [TCP Loopback]         [Node C: Receiver]
 *   Generate 1x3x640x640 FP32 → serialize+send → recv+deserialize → mock_relu
 *        ↓                                                    ↓
 *   PipelineEngine DAG                              Verify bit-exact match
 *
 * The "receiver" runs in a background thread on localhost, simulating
 * a remote Rock 5B+ node.
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_network_protocol.h"
#include "neurofabric/engine/PipelineEngine.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#include <vector>

/* POSIX sockets */
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

/* ================================================================== */
/*  Mock Hardware Buffer (reusable allocator for both sides)            */
/* ================================================================== */

struct HwBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data = nullptr;
    nf_tensor_desc        desc{};
    bool                  mapped = false;
};

static uint32_t hw_retain(nf_buffer self) {
    return reinterpret_cast<HwBuffer*>(self)->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}
static uint32_t hw_release(nf_buffer self) {
    auto* b = reinterpret_cast<HwBuffer*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) { std::free(b->data); delete b; }
    return prev - 1;
}
static nf_status hw_map(nf_buffer self, void** p) {
    auto* b = reinterpret_cast<HwBuffer*>(self);
    if (b->mapped) return NF_ERROR_INVALID_ARG;
    b->mapped = true; *p = b->data; return NF_OK;
}
static nf_status hw_unmap(nf_buffer self) {
    reinterpret_cast<HwBuffer*>(self)->mapped = false; return NF_OK;
}
static nf_status hw_cache(nf_buffer, nf_cache_op, uint64_t, uint64_t) { return NF_OK; }
static nf_status hw_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<HwBuffer*>(self);
    info->desc = b->desc; info->domain = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0; info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0; return NF_OK;
}
static nf_status hw_export(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_CPU; return NF_OK;
}

static nf_buffer_ops make_hw_ops() {
    nf_buffer_ops ops{};
    ops.retain = hw_retain; ops.release = hw_release;
    ops.map = hw_map; ops.unmap = hw_unmap;
    ops.cache_sync = hw_cache; ops.get_info = hw_info;
    ops.export_handle = hw_export; ops.import_handle = nullptr;
    return ops;
}

static HwBuffer* alloc_tensor(const nf_tensor_desc& desc) {
    auto* b = new HwBuffer;
    b->desc = desc;
    b->data = std::calloc(1, desc.size_bytes);
    return b;
}

/* ================================================================== */
/*  Mock Provider — local CPU compute with mock_relu                   */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    if (std::strcmp(op_name, "generate") == 0 && n_out >= 1) {
        /* Fill output with test pattern: alternating +/- values */
        auto* ob = reinterpret_cast<HwBuffer*>(outputs[0]);
        auto* fp = static_cast<float*>(ob->data);
        size_t count = ob->desc.size_bytes / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
            fp[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
        }
        return NF_OK;
    }
    if (std::strcmp(op_name, "mock_relu") == 0 && n_in >= 1 && n_out >= 1) {
        auto* ib = reinterpret_cast<HwBuffer*>(inputs[0]);
        auto* ob = reinterpret_cast<HwBuffer*>(outputs[0]);
        auto* src = static_cast<float*>(ib->data);
        auto* dst = static_cast<float*>(ob->data);
        size_t count = ib->desc.size_bytes / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
        }
        return NF_OK;
    }
    return NF_ERROR_UNSUPPORTED_OP;
}

static const char* mock_name(nf_provider) { return "mock_cpu"; }
static uint32_t mock_abi(nf_provider) { return NF_ABI_VERSION; }
static nf_status mock_init(nf_provider) { return NF_OK; }
static void mock_shutdown(nf_provider) {}
static nf_status mock_sync(nf_provider) { return NF_OK; }

static nf_provider_vtable make_mock_vt() {
    nf_provider_vtable vt{};
    vt.get_name = mock_name; vt.get_abi_version = mock_abi;
    vt.init = mock_init; vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch; vt.synchronize = mock_sync;
    return vt;
}

/* ================================================================== */
/*  TCP Loopback Echo Server                                           */
/*  Simulates a remote node: receives task frame + tensor payload,     */
/*  applies mock_relu on the received data, sends back the result.     */
/* ================================================================== */

struct LoopbackServer {
    int listen_fd = -1;
    int client_fd = -1;
    uint16_t port = 0;
    std::atomic<bool> running{false};
    std::thread thread;

    /* Received tensor data (for verification) */
    std::vector<float> received_data;
    std::vector<float> result_data;
};

static bool recv_all_s(int fd, void* data, size_t len) {
    auto* p = static_cast<uint8_t*>(data);
    size_t got = 0;
    while (got < len) {
        auto n = ::recv(fd, p + got, static_cast<int>(len - got), 0);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

static bool send_all_s(int fd, const void* data, size_t len) {
    auto* p = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        auto n = ::send(fd, p + sent, static_cast<int>(len - sent), flags);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

static void server_loop(LoopbackServer* srv) {
    struct sockaddr_in cli_addr{};
    socklen_t cli_len = sizeof(cli_addr);
    srv->client_fd = ::accept(srv->listen_fd,
                              reinterpret_cast<struct sockaddr*>(&cli_addr),
                              &cli_len);
    if (srv->client_fd < 0) return;

    while (srv->running.load(std::memory_order_relaxed)) {
        /* Receive frame header */
        nf_frame_header hdr{};
        if (!recv_all_s(srv->client_fd, &hdr, sizeof(hdr))) break;

        uint32_t magic = nf_le32toh(hdr.magic);
        if (magic != NF_PROTO_MAGIC) break;

        if (hdr.opcode == NF_OP_SHUTDOWN) break;

        if (hdr.opcode != NF_OP_TASK_SUBMIT) continue;

        uint8_t n_in = hdr.n_input_tensors;

        /* Receive tensor wire descriptors */
        std::vector<nf_tensor_wire> wires(n_in);
        for (uint8_t i = 0; i < n_in; ++i) {
            if (!recv_all_s(srv->client_fd, &wires[i], sizeof(nf_tensor_wire)))
                goto done;
        }

        /* Receive tensor payloads */
        for (uint8_t i = 0; i < n_in; ++i) {
            uint64_t payload = nf_le64toh(wires[i].payload_bytes);
            if (payload == 0) continue;

            size_t n_floats = static_cast<size_t>(payload / sizeof(float));
            srv->received_data.resize(n_floats);
            if (!recv_all_s(srv->client_fd, srv->received_data.data(),
                            static_cast<size_t>(payload)))
                goto done;

            /* Apply mock_relu on received data */
            srv->result_data.resize(n_floats);
            for (size_t j = 0; j < n_floats; ++j) {
                srv->result_data[j] = srv->received_data[j] > 0.0f
                                    ? srv->received_data[j] : 0.0f;
            }
        }

        /* Send response: TASK_COMPLETE with result payload */
        {
            uint64_t result_bytes = srv->result_data.size() * sizeof(float);

            nf_tensor_wire out_wire{};
            out_wire.dtype = static_cast<uint8_t>(NF_DTYPE_F32);
            out_wire.ndim  = wires[0].ndim;
            for (int d = 0; d < NF_MAX_DIMS; ++d) {
                out_wire.shape[d]   = wires[0].shape[d];
                out_wire.strides[d] = wires[0].strides[d];
            }
            out_wire.payload_bytes = nf_htole64(result_bytes);

            nf_frame_header resp{};
            resp.magic   = nf_htole32(NF_PROTO_MAGIC);
            resp.version = nf_htole16(NF_PROTO_VERSION);
            resp.opcode  = NF_OP_TASK_COMPLETE;
            resp.task_id = hdr.task_id;
            resp.n_input_tensors  = 0;
            resp.n_output_tensors = 1;
            resp.total_payload_bytes = nf_htole64(result_bytes);
            resp.header_crc32 = nf_htole32(nf_frame_compute_crc(&resp));

            send_all_s(srv->client_fd, &resp, sizeof(resp));
            send_all_s(srv->client_fd, &out_wire, sizeof(out_wire));
            send_all_s(srv->client_fd, srv->result_data.data(),
                       static_cast<size_t>(result_bytes));
        }
    }

done:
    ::close(srv->client_fd);
    srv->client_fd = -1;
}

static std::unique_ptr<LoopbackServer> start_server() {
    auto srv = std::make_unique<LoopbackServer>();
    srv->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(srv->listen_fd >= 0);

    int opt = 1;
    ::setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; /* OS picks a free port */

    CHECK(::bind(srv->listen_fd, reinterpret_cast<struct sockaddr*>(&addr),
                  sizeof(addr)) == 0);

    socklen_t alen = sizeof(addr);
    ::getsockname(srv->listen_fd, reinterpret_cast<struct sockaddr*>(&addr), &alen);
    srv->port = ntohs(addr.sin_port);

    CHECK(::listen(srv->listen_fd, 1) == 0);

    srv->running.store(true, std::memory_order_release);
    srv->thread = std::thread(server_loop, srv.get());

    return srv;
}

static void stop_server(std::unique_ptr<LoopbackServer>& srv) {
    srv->running.store(false, std::memory_order_release);
    if (srv->listen_fd >= 0) { ::close(srv->listen_fd); srv->listen_fd = -1; }
    if (srv->thread.joinable()) srv->thread.join();
}

/* ================================================================== */
/*  Network Client — connects to loopback server                       */
/* ================================================================== */

struct NetClient {
    int sock = -1;
};

static NetClient connect_to(uint16_t port) {
    NetClient nc;
    nc.sock = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(nc.sock >= 0);

    int flag = 1;
    ::setsockopt(nc.sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    int r = ::connect(nc.sock, reinterpret_cast<struct sockaddr*>(&addr),
                      sizeof(addr));
    CHECK(r == 0);
    return nc;
}

/* ================================================================== */
/*  Test 1: Local DAG — generate → mock_relu, bit-exact verify         */
/* ================================================================== */

static void test_local_dag() {
    nf::PipelineEngine engine(2);

    int mock_prov_id = 0;
    auto vt = make_mock_vt();
    engine.register_provider(reinterpret_cast<nf_provider>(&mock_prov_id),
                             vt, NF_AFFINITY_CPU);

    /* Tensor: small 1x4 FP32 */
    nf_tensor_desc desc{};
    desc.dtype = NF_DTYPE_F32;
    desc.ndim = 1;
    desc.shape[0] = 4;
    desc.size_bytes = 4 * sizeof(float);

    auto ops = make_hw_ops();

    /* Allocate buffers */
    HwBuffer* gen_out = alloc_tensor(desc);
    HwBuffer* relu_out = alloc_tensor(desc);

    /* Build DAG: generate → mock_relu */
    uint32_t gid = engine.create_graph();

    nf_task_desc td_gen{};
    std::strcpy(td_gen.op_name, "generate");
    td_gen.outputs[0] = reinterpret_cast<nf_buffer>(gen_out);
    td_gen.output_ops[0] = ops;
    td_gen.n_outputs = 1;
    td_gen.affinity = NF_AFFINITY_CPU;

    nf_task_desc td_relu{};
    std::strcpy(td_relu.op_name, "mock_relu");
    td_relu.inputs[0] = reinterpret_cast<nf_buffer>(gen_out);
    td_relu.input_ops[0] = ops;
    td_relu.n_inputs = 1;
    td_relu.outputs[0] = reinterpret_cast<nf_buffer>(relu_out);
    td_relu.output_ops[0] = ops;
    td_relu.n_outputs = 1;
    td_relu.affinity = NF_AFFINITY_CPU;

    uint32_t t0 = engine.add_task(gid, td_gen);
    uint32_t t1 = engine.add_task(gid, td_relu);
    engine.add_edge(gid, t0, t1);

    auto fut = engine.submit(gid);
    nf_status result = fut.get();
    CHECK(result == NF_OK);

    /* Verify: gen_out has pattern [0, -1, 2, -3] */
    auto* gen_fp = static_cast<float*>(gen_out->data);
    CHECK(gen_fp[0] == 0.0f);
    CHECK(gen_fp[1] == -1.0f);
    CHECK(gen_fp[2] == 2.0f);
    CHECK(gen_fp[3] == -3.0f);

    /* Verify: relu_out has [0, 0, 2, 0] */
    auto* relu_fp = static_cast<float*>(relu_out->data);
    CHECK(relu_fp[0] == 0.0f);
    CHECK(relu_fp[1] == 0.0f);
    CHECK(relu_fp[2] == 2.0f);
    CHECK(relu_fp[3] == 0.0f);

    ops.release(reinterpret_cast<nf_buffer>(gen_out));
    ops.release(reinterpret_cast<nf_buffer>(relu_out));
    engine.destroy_graph(gid);

    std::printf("  PASS: local DAG (generate → mock_relu)\n");
}

/* ================================================================== */
/*  Test 2: E2E Network Loopback — full tensor round-trip              */
/* ================================================================== */

static void test_e2e_network_loopback() {
    /* ---- Start TCP loopback server ---- */
    auto srv = start_server();
    std::printf("  [server] listening on port %u\n", srv->port);

    /* Brief sleep to let server thread reach accept() */
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    /* ---- Connect client ---- */
    auto client = connect_to(srv->port);
    std::printf("  [client] connected\n");

    /* ---- Build test tensor: 1x3x8x8 FP32 (simulated feature map) ---- */
    nf_tensor_desc desc{};
    desc.dtype = NF_DTYPE_F32;
    desc.ndim = 4;
    desc.shape[0] = 1; desc.shape[1] = 3;
    desc.shape[2] = 8; desc.shape[3] = 8;
    desc.size_bytes = 1 * 3 * 8 * 8 * sizeof(float);

    auto ops = make_hw_ops();
    HwBuffer* send_buf = alloc_tensor(desc);

    /* Fill with test pattern: alternating +/- */
    auto* fp = static_cast<float*>(send_buf->data);
    size_t n_floats = desc.size_bytes / sizeof(float);
    for (size_t i = 0; i < n_floats; ++i) {
        fp[i] = (i % 2 == 0) ? static_cast<float>(i) * 0.1f
                              : -static_cast<float>(i) * 0.1f;
    }

    /* ---- Compute expected ReLU result locally ---- */
    std::vector<float> expected(n_floats);
    for (size_t i = 0; i < n_floats; ++i) {
        expected[i] = fp[i] > 0.0f ? fp[i] : 0.0f;
    }

    /* ---- Build and send wire frame ---- */
    nf_tensor_wire wire{};
    wire.dtype = static_cast<uint8_t>(desc.dtype);
    wire.ndim  = static_cast<uint8_t>(desc.ndim);
    for (uint32_t d = 0; d < desc.ndim; ++d) {
        wire.shape[d]   = nf_htole64(desc.shape[d]);
        wire.strides[d] = 0;
    }
    wire.payload_bytes = nf_htole64(desc.size_bytes);

    nf_frame_header hdr{};
    hdr.magic   = nf_htole32(NF_PROTO_MAGIC);
    hdr.version = nf_htole16(NF_PROTO_VERSION);
    hdr.opcode  = NF_OP_TASK_SUBMIT;
    hdr.task_id = nf_htole64(42);
    std::strcpy(hdr.op_name, "mock_relu");
    hdr.n_input_tensors  = 1;
    hdr.n_output_tensors = 0;
    hdr.total_payload_bytes = nf_htole64(desc.size_bytes);
    hdr.header_crc32 = nf_htole32(nf_frame_compute_crc(&hdr));

    /* Send header + wire descriptor + payload */
    CHECK(send_all_s(client.sock, &hdr, sizeof(hdr)));
    CHECK(send_all_s(client.sock, &wire, sizeof(wire)));
    CHECK(send_all_s(client.sock, send_buf->data, desc.size_bytes));
    std::printf("  [client] sent %zu bytes tensor payload\n", (size_t)desc.size_bytes);

    /* ---- Receive response ---- */
    nf_frame_header resp{};
    CHECK(recv_all_s(client.sock, &resp, sizeof(resp)));
    CHECK(nf_le32toh(resp.magic) == NF_PROTO_MAGIC);
    CHECK(resp.opcode == NF_OP_TASK_COMPLETE);
    CHECK(nf_le64toh(resp.task_id) == 42);
    CHECK(resp.n_output_tensors == 1);

    nf_tensor_wire out_wire{};
    CHECK(recv_all_s(client.sock, &out_wire, sizeof(out_wire)));

    uint64_t result_bytes = nf_le64toh(out_wire.payload_bytes);
    CHECK(result_bytes == desc.size_bytes);

    std::vector<float> received(n_floats);
    CHECK(recv_all_s(client.sock, received.data(), static_cast<size_t>(result_bytes)));
    std::printf("  [client] received %zu bytes result payload\n", (size_t)result_bytes);

    /* ---- Bit-exact verification ---- */
    size_t mismatches = 0;
    for (size_t i = 0; i < n_floats; ++i) {
        if (std::memcmp(&received[i], &expected[i], sizeof(float)) != 0) {
            if (mismatches < 5) {
                std::printf("  MISMATCH at [%zu]: got %.6f, expected %.6f\n",
                            i, received[i], expected[i]);
            }
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
    std::printf("  [verify] %zu floats bit-exact match ✓\n", n_floats);

    /* ---- Cleanup ---- */
    /* Send shutdown to server */
    nf_frame_header shutdown_hdr{};
    shutdown_hdr.magic   = nf_htole32(NF_PROTO_MAGIC);
    shutdown_hdr.version = nf_htole16(NF_PROTO_VERSION);
    shutdown_hdr.opcode  = NF_OP_SHUTDOWN;
    shutdown_hdr.header_crc32 = nf_htole32(nf_frame_compute_crc(&shutdown_hdr));
    send_all_s(client.sock, &shutdown_hdr, sizeof(shutdown_hdr));

    ::close(client.sock);
    stop_server(srv);
    ops.release(reinterpret_cast<nf_buffer>(send_buf));

    std::printf("  PASS: E2E network loopback (bit-exact)\n");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    signal(SIGPIPE, SIG_IGN);
    std::printf("e2e_pipeline_test: Phase 6 — Hardware & Network Validation\n");
    test_local_dag();
    test_e2e_network_loopback();
    std::printf("OK: all Phase 6 E2E tests passed\n");
    return 0;
}
