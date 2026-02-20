/**
 * @file network_provider.cpp
 * @brief Network Proxy Execution Provider — Edge-Cloud Bridge
 *
 * This plugin implements nf_provider_vtable but instead of dispatching
 * to local hardware, it serializes the task into the NF binary wire
 * protocol and sends it over a TCP socket to a remote Neuro-Fabric node.
 *
 * Architecture:
 *   1. On init(), connects to the configured remote endpoint.
 *   2. On dispatch(), serializes nf_task_desc + input tensor payloads
 *      into nf_frame_header + nf_tensor_wire frames.
 *   3. Sends asynchronously via a dedicated IO thread (kqueue/epoll).
 *   4. On receiving NF_OP_TASK_COMPLETE, deserializes output tensors
 *      and signals the pending future.
 *   5. On synchronize(), drains all in-flight tasks.
 *
 * The Core scheduler sees this as just another provider — zero
 * awareness of networking in PipelineEngine.
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_network_protocol.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

/* Platform socket headers */
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socket_t = SOCKET;
  #define NF_INVALID_SOCKET INVALID_SOCKET
#else
  #include <arpa/inet.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <unistd.h>
  using socket_t = int;
  #define NF_INVALID_SOCKET (-1)
#endif

/* ================================================================== */
/*  Internal: Pending Task Tracker                                     */
/* ================================================================== */

struct PendingTask {
    uint64_t                task_id = 0;
    nf_buffer*              outputs = nullptr;
    uint32_t                n_outputs = 0;
    std::atomic<bool>       completed{false};
    nf_status               result = NF_OK;
    std::condition_variable cv;
    std::mutex              mu;
};

/* ================================================================== */
/*  Network Provider State                                             */
/* ================================================================== */

struct nf_provider_network {
    socket_t    sock = NF_INVALID_SOCKET;
    char        remote_host[256] = "127.0.0.1";
    uint16_t    remote_port = 9876;

    std::atomic<uint64_t>   next_task_id{1};
    std::atomic<bool>       running{false};
    std::thread             io_thread;

    std::mutex              pending_mu;
    std::unordered_map<uint64_t, std::shared_ptr<PendingTask>> pending;
};

static nf_provider_network g_net_provider;

/* ================================================================== */
/*  Socket Helpers                                                     */
/* ================================================================== */

static void close_socket(socket_t s) {
    if (s == NF_INVALID_SOCKET) return;
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
}

/** Send exactly `len` bytes. Returns true on success. */
static bool send_all(socket_t s, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        auto n = ::send(s, reinterpret_cast<const char*>(p + sent),
                        static_cast<int>(len - sent), 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

/** Receive exactly `len` bytes. Returns true on success. */
static bool recv_all(socket_t s, void* data, size_t len) {
    auto* p = static_cast<uint8_t*>(data);
    size_t got = 0;
    while (got < len) {
        auto n = ::recv(s, reinterpret_cast<char*>(p + got),
                        static_cast<int>(len - got), 0);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

/* ================================================================== */
/*  Serialization: task_desc + buffers → wire frames                   */
/* ================================================================== */

static bool serialize_and_send(nf_provider_network* net,
                               uint64_t task_id,
                               const char* op_name,
                               const nf_buffer* inputs,
                               uint32_t n_in) {
    /* -- Build frame header ----------------------------------------- */
    nf_frame_header hdr{};
    hdr.magic   = nf_htole32(NF_PROTO_MAGIC);
    hdr.version = nf_htole16(NF_PROTO_VERSION);
    hdr.opcode  = NF_OP_TASK_SUBMIT;
    hdr.flags   = 0;
    hdr.task_id = nf_htole64(task_id);
    hdr.seq_num = 0;
    std::strncpy(hdr.op_name, op_name, NF_MAX_OP_NAME - 1);
    hdr.n_input_tensors  = static_cast<uint8_t>(n_in);
    hdr.n_output_tensors = 0;

    /* -- Build tensor wire descriptors ------------------------------ */
    /* For Phase 4 skeleton, we send tensor metadata only.             */
    /* Full payload streaming will be added when buffer map is wired.  */
    std::vector<nf_tensor_wire> wires(n_in);
    uint64_t total_payload = 0;

    for (uint32_t i = 0; i < n_in; ++i) {
        /* In a full implementation, we'd call buffer_ops.get_info()   */
        /* and buffer_ops.map() to read the actual tensor data.        */
        /* For now, wire descriptors carry shape metadata.             */
        std::memset(&wires[i], 0, sizeof(nf_tensor_wire));
        wires[i].dtype = 0;
        wires[i].ndim  = 0;
        wires[i].payload_bytes = 0;
        total_payload += wires[i].payload_bytes;
    }

    hdr.total_payload_bytes = nf_htole64(total_payload);
    hdr.header_crc32 = nf_htole32(nf_frame_compute_crc(&hdr));

    /* -- Send: header, then tensor descriptors ---------------------- */
    if (!send_all(net->sock, &hdr, sizeof(hdr))) return false;
    for (uint32_t i = 0; i < n_in; ++i) {
        if (!send_all(net->sock, &wires[i], sizeof(nf_tensor_wire)))
            return false;
    }

    /* -- Send: tensor payloads (placeholder for zero-copy path) ----- */
    /* When buffer map is wired:                                       */
    /*   void* ptr; ops.map(buf, &ptr);                                */
    /*   send_all(net->sock, ptr, wire.payload_bytes);                 */
    /*   ops.unmap(buf);                                               */

    return true;
}

/* ================================================================== */
/*  IO Thread: receives responses from remote node                     */
/* ================================================================== */

static void io_recv_loop(nf_provider_network* net) {
    while (net->running.load(std::memory_order_relaxed)) {
        nf_frame_header resp{};
        if (!recv_all(net->sock, &resp, sizeof(resp))) {
            if (net->running.load(std::memory_order_relaxed)) {
                /* Connection lost — mark all pending as failed */
                std::lock_guard<std::mutex> lk(net->pending_mu);
                for (auto& [id, pt] : net->pending) {
                    pt->result = NF_ERROR_DEVICE_LOST;
                    pt->completed.store(true, std::memory_order_release);
                    pt->cv.notify_all();
                }
                net->pending.clear();
            }
            return;
        }

        /* Validate */
        uint32_t magic = nf_le32toh(resp.magic);
        if (magic != NF_PROTO_MAGIC) continue;

        uint64_t tid = nf_le64toh(resp.task_id);

        /* Skip tensor descriptors + payload for now */
        uint64_t skip = nf_le64toh(resp.total_payload_bytes);
        skip += resp.n_output_tensors * sizeof(nf_tensor_wire);
        if (skip > 0) {
            std::vector<uint8_t> discard(skip);
            recv_all(net->sock, discard.data(), skip);
        }

        /* Resolve pending task */
        std::shared_ptr<PendingTask> pt;
        {
            std::lock_guard<std::mutex> lk(net->pending_mu);
            auto it = net->pending.find(tid);
            if (it != net->pending.end()) {
                pt = it->second;
                net->pending.erase(it);
            }
        }

        if (pt) {
            pt->result = (resp.opcode == NF_OP_TASK_COMPLETE)
                         ? NF_OK : NF_ERROR_INTERNAL;
            pt->completed.store(true, std::memory_order_release);
            pt->cv.notify_all();
        }
    }
}

/* ================================================================== */
/*  Provider VTable Implementation                                     */
/* ================================================================== */

static const char* net_get_name(nf_provider) {
    return "network_proxy";
}

static uint32_t net_get_abi_version(nf_provider) {
    return NF_ABI_VERSION;
}

static nf_status net_init(nf_provider self) {
    auto* net = reinterpret_cast<nf_provider_network*>(self);

    /* Resolve remote endpoint from environment (or use defaults). */
    const char* host_env = std::getenv("NF_REMOTE_HOST");
    const char* port_env = std::getenv("NF_REMOTE_PORT");
    if (host_env) std::strncpy(net->remote_host, host_env, 255);
    if (port_env) net->remote_port = static_cast<uint16_t>(std::atoi(port_env));

    /* Create TCP socket */
    net->sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (net->sock == NF_INVALID_SOCKET) return NF_ERROR_DEVICE_LOST;

    /* Disable Nagle for low-latency framing */
    int flag = 1;
    ::setsockopt(net->sock, IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<const char*>(&flag), sizeof(flag));

    /* Connect */
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(net->remote_port);
    if (::inet_pton(AF_INET, net->remote_host, &addr.sin_addr) != 1) {
        close_socket(net->sock);
        net->sock = NF_INVALID_SOCKET;
        return NF_ERROR_INVALID_ARG;
    }

    if (::connect(net->sock, reinterpret_cast<struct sockaddr*>(&addr),
                  sizeof(addr)) != 0) {
        close_socket(net->sock);
        net->sock = NF_INVALID_SOCKET;
        return NF_ERROR_DEVICE_LOST;
    }

    /* Start IO receive thread */
    net->running.store(true, std::memory_order_release);
    net->io_thread = std::thread(io_recv_loop, net);

    return NF_OK;
}

static void net_shutdown(nf_provider self) {
    auto* net = reinterpret_cast<nf_provider_network*>(self);

    net->running.store(false, std::memory_order_release);

    /* Send graceful shutdown frame */
    if (net->sock != NF_INVALID_SOCKET) {
        nf_frame_header hdr{};
        hdr.magic   = nf_htole32(NF_PROTO_MAGIC);
        hdr.version = nf_htole16(NF_PROTO_VERSION);
        hdr.opcode  = NF_OP_SHUTDOWN;
        hdr.header_crc32 = nf_htole32(nf_frame_compute_crc(&hdr));
        send_all(net->sock, &hdr, sizeof(hdr));

        /* Unblock recv thread */
#ifdef _WIN32
        ::shutdown(net->sock, SD_BOTH);
#else
        ::shutdown(net->sock, SHUT_RDWR);
#endif
        close_socket(net->sock);
        net->sock = NF_INVALID_SOCKET;
    }

    if (net->io_thread.joinable()) {
        net->io_thread.join();
    }

    /* Fail all pending tasks */
    std::lock_guard<std::mutex> lk(net->pending_mu);
    for (auto& [id, pt] : net->pending) {
        pt->result = NF_ERROR_DEVICE_LOST;
        pt->completed.store(true, std::memory_order_release);
        pt->cv.notify_all();
    }
    net->pending.clear();
}

static nf_status net_buffer_alloc(nf_provider, const nf_tensor_desc*,
                                  nf_buffer*) {
    return NF_ERROR_UNSUPPORTED_OP; /* Remote buffers live on the peer */
}

static void      net_buffer_free(nf_provider, nf_buffer) {}
static nf_status net_buffer_map(nf_provider, nf_buffer, void**) {
    return NF_ERROR_UNSUPPORTED_OP;
}
static nf_status net_buffer_unmap(nf_provider, nf_buffer) {
    return NF_ERROR_UNSUPPORTED_OP;
}

/**
 * dispatch() — the critical path.
 *
 * Serializes the task into wire format, sends it, and BLOCKS until
 * the remote node responds. The PipelineEngine calls this from its
 * thread pool, so blocking here does NOT stall the scheduler —
 * other ready tasks continue dispatching on other pool threads.
 *
 * For fully non-blocking IO (Phase 4+), this can be refactored to
 * return immediately and use an eventfd/kqueue callback to signal
 * the graph's remaining counter.
 */
static nf_status net_dispatch(nf_provider self,
                              const char* op_name,
                              const nf_buffer* inputs, uint32_t n_in,
                              nf_buffer* /*outputs*/, uint32_t /*n_out*/) {
    auto* net = reinterpret_cast<nf_provider_network*>(self);
    if (net->sock == NF_INVALID_SOCKET) return NF_ERROR_DEVICE_LOST;

    uint64_t tid = net->next_task_id.fetch_add(1, std::memory_order_relaxed);

    /* Register pending task */
    auto pt = std::make_shared<PendingTask>();
    pt->task_id = tid;
    {
        std::lock_guard<std::mutex> lk(net->pending_mu);
        net->pending[tid] = pt;
    }

    /* Serialize and send */
    if (!serialize_and_send(net, tid, op_name, inputs, n_in)) {
        std::lock_guard<std::mutex> lk(net->pending_mu);
        net->pending.erase(tid);
        return NF_ERROR_DEVICE_LOST;
    }

    /* Wait for response (blocking — pool thread is dedicated) */
    {
        std::unique_lock<std::mutex> lk(pt->mu);
        pt->cv.wait(lk, [&pt] {
            return pt->completed.load(std::memory_order_acquire);
        });
    }

    return pt->result;
}

static nf_status net_synchronize(nf_provider self) {
    auto* net = reinterpret_cast<nf_provider_network*>(self);

    /* Spin until all pending tasks drain */
    for (;;) {
        std::lock_guard<std::mutex> lk(net->pending_mu);
        if (net->pending.empty()) return NF_OK;
        /* Brief yield to avoid busy-spin */
        std::this_thread::yield();
    }
}

/* ================================================================== */
/*  Plugin Entry Point                                                 */
/* ================================================================== */

extern "C" NF_API nf_status nf_plugin_register(nf_provider_vtable* vt,
                                                nf_provider* out) {
    vt->get_name        = net_get_name;
    vt->get_abi_version = net_get_abi_version;
    vt->init            = net_init;
    vt->shutdown        = net_shutdown;
    vt->buffer_alloc    = net_buffer_alloc;
    vt->buffer_free     = net_buffer_free;
    vt->buffer_map      = net_buffer_map;
    vt->buffer_unmap    = net_buffer_unmap;
    vt->dispatch        = net_dispatch;
    vt->synchronize     = net_synchronize;

    *out = reinterpret_cast<nf_provider>(& g_net_provider);
    return NF_OK;
}
