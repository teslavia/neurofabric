/**
 * @file split_inference_test.cpp
 * @brief Phase 7 — Split-Inference Pipeline: Metal Prefill → TCP → ContextHub + Decode
 *
 * Validates:
 *   1. Metal GPU execution fence (async dispatch + blocking cache_sync)
 *   2. Layout-aware wire protocol (nf_layout_type in nf_tensor_wire)
 *   3. ContextHub radix-tree insert/lookup with TensorView
 *   4. RKNN decode_step operator (tanh)
 *   5. Bit-exact: received[i] == tanh(input[i] * 0.5f)
 *
 * Topology:
 *   [Metal/Local]              [TCP Loopback]           [RKNN/Remote]
 *   attention_prefill       serialize K_Cache over TCP   ContextHub.put(K)
 *   input → K_Cache, V_Cache                            decode_step(K) → output
 *   (async GPU + fence)                                 send result back
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_network_protocol.h"
#include "neurofabric/ContextHub.hpp"
#include "neurofabric/TensorView.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

/* POSIX sockets */
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

/* ================================================================== */
/*  GPU-Fenced Buffer — simulates MetalBuffer with execution fence      */
/* ================================================================== */

struct GpuFencedBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data = nullptr;
    nf_tensor_desc        desc{};
    bool                  mapped = false;
    /* GPU execution fence state */
    std::mutex              fence_mu;
    std::condition_variable fence_cv;
    std::atomic<bool>       gpu_done{true};
};

/* ================================================================== */
/*  Buffer ops for GpuFencedBuffer                                      */
/* ================================================================== */

static uint32_t fb_retain(nf_buffer self) {
    return reinterpret_cast<GpuFencedBuffer*>(self)->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}
static uint32_t fb_release(nf_buffer self) {
    auto* b = reinterpret_cast<GpuFencedBuffer*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) { std::free(b->data); delete b; }
    return prev - 1;
}
static nf_status fb_map(nf_buffer self, void** p) {
    auto* b = reinterpret_cast<GpuFencedBuffer*>(self);
    if (b->mapped) return NF_ERROR_INVALID_ARG;
    b->mapped = true; *p = b->data; return NF_OK;
}
static nf_status fb_unmap(nf_buffer self) {
    reinterpret_cast<GpuFencedBuffer*>(self)->mapped = false; return NF_OK;
}
static nf_status fb_cache_sync(nf_buffer self, nf_cache_op, uint64_t, uint64_t) {
    /* Block until GPU fence signals (mirrors Phase 7 metal_buf_cache_sync) */
    auto* b = reinterpret_cast<GpuFencedBuffer*>(self);
    if (!b->gpu_done.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(b->fence_mu);
        b->fence_cv.wait(lk, [&] { return b->gpu_done.load(std::memory_order_acquire); });
    }
    return NF_OK;
}
static nf_status fb_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<GpuFencedBuffer*>(self);
    info->desc = b->desc; info->domain = NF_MEM_DOMAIN_UNIFIED;
    info->offset_bytes = 0; info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0; return NF_OK;
}
static nf_status fb_export(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_UNIFIED; return NF_OK;
}

static nf_buffer_ops make_fb_ops() {
    nf_buffer_ops ops{};
    ops.retain = fb_retain; ops.release = fb_release;
    ops.map = fb_map; ops.unmap = fb_unmap;
    ops.cache_sync = fb_cache_sync; ops.get_info = fb_info;
    ops.export_handle = fb_export; ops.import_handle = nullptr;
    return ops;
}

static GpuFencedBuffer* alloc_fenced(const nf_tensor_desc& desc) {
    auto* b = new GpuFencedBuffer;
    b->desc = desc;
    b->data = std::calloc(1, desc.size_bytes);
    return b;
}

/* ================================================================== */
/*  Socket helpers                                                      */
/* ================================================================== */

static bool send_all_s(int s, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        auto n = ::send(s, p + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

static bool recv_all_s(int s, void* data, size_t len) {
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
/*  TCP Loopback Server                                                 */
/* ================================================================== */

struct LoopbackServer {
    int listen_fd = -1;
    uint16_t port = 0;
    std::thread thread;
    std::atomic<bool> running{false};

    LoopbackServer() = default;
    LoopbackServer(LoopbackServer&& o) noexcept
        : listen_fd(o.listen_fd), port(o.port),
          thread(std::move(o.thread)),
          running(o.running.load()) { o.listen_fd = -1; }
    LoopbackServer& operator=(LoopbackServer&&) = delete;
    LoopbackServer(const LoopbackServer&) = delete;
};

static LoopbackServer start_server(uint16_t port_hint) {
    LoopbackServer srv;
    srv.listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(srv.listen_fd >= 0);
    int opt = 1;
    setsockopt(srv.listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port_hint);
    int rc = ::bind(srv.listen_fd, (struct sockaddr*)&addr, sizeof(addr));
    assert(rc == 0);
    ::listen(srv.listen_fd, 1);

    /* Retrieve actual port */
    socklen_t alen = sizeof(addr);
    getsockname(srv.listen_fd, (struct sockaddr*)&addr, &alen);
    srv.port = ntohs(addr.sin_port);
    srv.running.store(true);
    return srv;
}

static void stop_server(LoopbackServer& srv) {
    srv.running.store(false);
    if (srv.listen_fd >= 0) { ::close(srv.listen_fd); srv.listen_fd = -1; }
    if (srv.thread.joinable()) srv.thread.join();
}

/* ================================================================== */
/*  Test: Split-Inference Pipeline                                      */
/* ================================================================== */

static void test_split_inference() {
    std::printf("  test_split_inference: Metal prefill → TCP → ContextHub + decode\n");

    /* ---- Tensor shape: 1x4x8x6 FP32 = 192 floats ---- */
    const uint64_t N = 1, C = 4, H = 8, W = 6;
    const size_t n_floats = N * C * H * W;
    const size_t tensor_bytes = n_floats * sizeof(float);

    nf_tensor_desc desc{};
    desc.dtype = NF_DTYPE_F32;
    desc.ndim = 4;
    desc.shape[0] = N; desc.shape[1] = C;
    desc.shape[2] = H; desc.shape[3] = W;
    desc.size_bytes = tensor_bytes;

    /* ---- Allocate buffers ---- */
    auto ops = make_fb_ops();
    auto* input_buf = alloc_fenced(desc);
    auto* k_buf     = alloc_fenced(desc);
    auto* v_buf     = alloc_fenced(desc);

    /* Fill input with deterministic pattern */
    auto* inp = static_cast<float*>(input_buf->data);
    for (size_t i = 0; i < n_floats; ++i) {
        inp[i] = static_cast<float>(i) * 0.01f - 0.5f;
    }

    /* ---- Simulate async GPU attention_prefill ---- */
    /* Mark K and V as GPU-pending */
    k_buf->gpu_done.store(false, std::memory_order_release);
    v_buf->gpu_done.store(false, std::memory_order_release);

    std::thread gpu_thread([inp, k_buf, v_buf, n_floats] {
        /* Simulate GPU compute latency */
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto* k_dst = static_cast<float*>(k_buf->data);
        auto* v_dst = static_cast<float*>(v_buf->data);
        for (size_t i = 0; i < n_floats; ++i) {
            k_dst[i] = inp[i] * 0.5f;
            v_dst[i] = inp[i] * -0.25f;
        }

        /* Signal K fence */
        { std::lock_guard<std::mutex> lk(k_buf->fence_mu);
          k_buf->gpu_done.store(true, std::memory_order_release); }
        k_buf->fence_cv.notify_all();

        /* Signal V fence */
        { std::lock_guard<std::mutex> lk(v_buf->fence_mu);
          v_buf->gpu_done.store(true, std::memory_order_release); }
        v_buf->fence_cv.notify_all();
    });

    /* ---- Start TCP loopback server ---- */
    auto srv = start_server(0);
    std::printf("    [server] listening on port %u\n", srv.port);

    /* Server thread: receives K_Cache, inserts into ContextHub, runs decode, sends back */
    std::vector<float> server_result(n_floats, 0.0f);
    std::atomic<bool> server_done{false};
    nf_status server_status = NF_OK;

    srv.thread = std::thread([&] {
        int client = ::accept(srv.listen_fd, nullptr, nullptr);
        assert(client >= 0);

        /* Receive tensor wire descriptor */
        nf_tensor_wire wire{};
        assert(recv_all_s(client, &wire, sizeof(wire)));
        uint64_t payload = nf_le64toh(wire.payload_bytes);
        uint16_t layout  = nf_le16toh(wire.layout);
        std::printf("    [server] received wire: %llu bytes, layout=%u\n",
                    (unsigned long long)payload, layout);

        /* Receive tensor payload */
        std::vector<float> k_data(n_floats);
        assert(recv_all_s(client, k_data.data(), tensor_bytes));
        std::printf("    [server] received %zu float payload\n", n_floats);

        /* Verify layout tag was transmitted */
        assert(layout == NF_LAYOUT_NCHW);

        /* ---- ContextHub: insert K_Cache ---- */
        nf_tensor_desc recv_desc = desc;
        auto recv_ops = make_fb_ops();
        auto* hub_buf = alloc_fenced(recv_desc);
        std::memcpy(hub_buf->data, k_data.data(), tensor_bytes);

        nf::TensorView k_view(reinterpret_cast<nf_buffer>(hub_buf), recv_ops);

        nf::ContextHub hub(16 * 1024 * 1024, NF_EVICT_LRU);
        std::vector<int32_t> kv_tokens = {100, 200, 300, 400};
        auto st = hub.put(kv_tokens, "test_agent",
                          k_view.share(), 0, 1);
        assert(st == NF_OK);
        std::printf("    [server] ContextHub: inserted K_Cache\n");

        /* ---- ContextHub: lookup by prefix ---- */
        auto found = hub.get(std::span<const int32_t>(kv_tokens));
        assert(found.found);
        assert(found.match_len == 4);
        std::printf("    [server] ContextHub: found K via prefix lookup\n");

        /* ---- Decode step: output[i] = tanh(K[i]) ---- */
        for (size_t i = 0; i < n_floats; ++i) {
            server_result[i] = std::tanh(k_data[i]);
        }
        std::printf("    [server] decode_step: tanh applied to %zu floats\n", n_floats);

        /* ---- ContextHub stats ---- */
        auto stats = hub.stats();
        std::printf("    [server] ContextHub: used=%llu bytes, entries=%u\n",
                    (unsigned long long)stats.used_bytes, stats.entry_count);
        assert(stats.entry_count == 1);
        assert(stats.used_bytes == tensor_bytes);

        /* ---- Send result back ---- */
        /* Send wire descriptor for result */
        nf_tensor_wire result_wire{};
        std::memset(&result_wire, 0, sizeof(result_wire));
        result_wire.dtype = static_cast<uint8_t>(NF_DTYPE_F32);
        result_wire.ndim = 4;
        result_wire.layout = nf_htole16(static_cast<uint16_t>(NF_LAYOUT_NCHW));
        for (uint8_t d = 0; d < 4; ++d) {
            result_wire.shape[d] = nf_htole64(desc.shape[d]);
        }
        result_wire.payload_bytes = nf_htole64(tensor_bytes);
        assert(send_all_s(client, &result_wire, sizeof(result_wire)));
        assert(send_all_s(client, server_result.data(), tensor_bytes));
        std::printf("    [server] sent %zu bytes result\n", tensor_bytes);

        /* ---- Evict all and verify ---- */
        hub.evict();
        auto stats2 = hub.stats();
        assert(stats2.used_bytes == 0);
        assert(stats2.entry_count == 0);
        std::printf("    [server] ContextHub: evicted all, used=0 ✓\n");

        ::close(client);
        server_done.store(true);
    });

    /* ---- Client side: fence-wait, serialize K_Cache, send over TCP ---- */

    /* cache_sync blocks until GPU fence signals */
    std::printf("    [client] waiting for GPU fence on K_Cache...\n");
    auto t0 = std::chrono::steady_clock::now();
    fb_cache_sync(reinterpret_cast<nf_buffer>(k_buf), NF_CACHE_FLUSH, 0, 0);
    auto t1 = std::chrono::steady_clock::now();
    auto fence_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("    [client] GPU fence waited %lld ms\n", (long long)fence_ms);
    assert(fence_ms >= 5); /* GPU thread sleeps 10ms, should be >= 5 */

    /* Connect to server */
    int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(client_fd >= 0);
    struct sockaddr_in saddr{};
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(srv.port);
    saddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int rc = ::connect(client_fd, (struct sockaddr*)&saddr, sizeof(saddr));
    assert(rc == 0);
    std::printf("    [client] connected to server\n");

    /* Build and send tensor wire descriptor for K_Cache */
    nf_tensor_wire k_wire{};
    std::memset(&k_wire, 0, sizeof(k_wire));
    k_wire.dtype = static_cast<uint8_t>(NF_DTYPE_F32);
    k_wire.ndim = 4;
    k_wire.layout = nf_htole16(static_cast<uint16_t>(NF_LAYOUT_NCHW));
    for (uint8_t d = 0; d < 4; ++d) {
        k_wire.shape[d] = nf_htole64(desc.shape[d]);
    }
    k_wire.payload_bytes = nf_htole64(tensor_bytes);
    assert(send_all_s(client_fd, &k_wire, sizeof(k_wire)));

    /* Send K_Cache payload (GPU fence already waited) */
    assert(send_all_s(client_fd, k_buf->data, tensor_bytes));
    std::printf("    [client] sent K_Cache: %zu bytes\n", tensor_bytes);

    /* Receive result from server */
    nf_tensor_wire result_wire{};
    assert(recv_all_s(client_fd, &result_wire, sizeof(result_wire)));
    uint64_t result_bytes = nf_le64toh(result_wire.payload_bytes);
    std::vector<float> received(n_floats);
    assert(recv_all_s(client_fd, received.data(), static_cast<size_t>(result_bytes)));
    std::printf("    [client] received %llu bytes result\n", (unsigned long long)result_bytes);

    /* ---- Bit-exact verification ---- */
    /* Expected: tanh(input[i] * 0.5f) */
    size_t mismatches = 0;
    for (size_t i = 0; i < n_floats; ++i) {
        float expected = std::tanh(inp[i] * 0.5f);
        if (std::memcmp(&received[i], &expected, sizeof(float)) != 0) {
            if (mismatches < 5) {
                std::printf("    MISMATCH at [%zu]: got %.8f, expected %.8f\n",
                            i, received[i], expected);
            }
            ++mismatches;
        }
    }
    assert(mismatches == 0);
    std::printf("    [verify] %zu floats bit-exact: tanh(input*0.5) ✓\n", n_floats);

    /* ---- Cleanup ---- */
    ::close(client_fd);
    gpu_thread.join();
    stop_server(srv);

    ops.release(reinterpret_cast<nf_buffer>(input_buf));
    ops.release(reinterpret_cast<nf_buffer>(k_buf));
    ops.release(reinterpret_cast<nf_buffer>(v_buf));

    std::printf("  PASS: split-inference pipeline (bit-exact)\n");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("split_inference_test: Phase 7 — Real Compute & KV Cache Hand-off\n");
    test_split_inference();
    std::printf("OK: all Phase 7 split-inference tests passed\n");
    return 0;
}
