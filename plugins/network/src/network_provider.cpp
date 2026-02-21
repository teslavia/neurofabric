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
#include "payload_serializer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
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
    nf_buffer_ops*          output_ops = nullptr;
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

    /**
     * Local memory provider — used by the recv path to allocate
     * hardware-native buffers for incoming tensor payloads.
     * Set via nf_plugin_register_mem or auto-detected.
     */
    nf_provider             local_mem_provider = nullptr;
    nf_provider_mem_vtable  local_mem_vt{};
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
        int flags = 0;
#ifdef __linux__
        flags = MSG_NOSIGNAL;
#endif
        auto n = ::send(s, reinterpret_cast<const char*>(p + sent),
                        static_cast<int>(len - sent), flags);
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
/*  Serialization: task_desc + buffers → wire frames + payload         */
/*  Phase 5: real zero-copy path via payload_serializer.               */
/* ================================================================== */

static bool serialize_and_send(nf_provider_network* net,
                               uint64_t task_id,
                               const char* op_name,
                               const nf_buffer* inputs,
                               const nf_buffer_ops* input_ops,
                               uint32_t n_in) {
    /* -- Build tensor wire descriptors from buffer metadata ---------- */
    std::vector<nf_tensor_wire> wires(n_in);
    uint64_t total_payload = 0;

    for (uint32_t i = 0; i < n_in; ++i) {
        std::memset(&wires[i], 0, sizeof(nf_tensor_wire));

        if (input_ops[i].get_info) {
            nf_buffer_info info{};
            input_ops[i].get_info(inputs[i], &info);

            wires[i].dtype = static_cast<uint8_t>(info.desc.dtype);
            wires[i].ndim  = static_cast<uint8_t>(info.desc.ndim);
            for (uint8_t d = 0; d < info.desc.ndim && d < NF_MAX_DIMS; ++d) {
                wires[i].shape[d]   = nf_htole64(info.desc.shape[d]);
                wires[i].strides[d] = nf_htole64(info.desc.strides[d]);
            }
            wires[i].payload_bytes = nf_htole64(info.desc.size_bytes);
            total_payload += info.desc.size_bytes;

            /* Populate layout tag for cross-ISA transport.
             * Default to NCHW (Metal/Apple Silicon convention).
             * Receiver checks this to decide if reordering is needed. */
            wires[i].layout = nf_htole16(static_cast<uint16_t>(NF_LAYOUT_NCHW));
        }
    }

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
    hdr.total_payload_bytes = nf_htole64(total_payload);
    hdr.header_crc32 = nf_htole32(nf_frame_compute_crc(&hdr));

    /* -- Send: header ----------------------------------------------- */
    if (!send_all(net->sock, &hdr, sizeof(hdr))) return false;

    /* -- Send: tensor descriptors ----------------------------------- */
    for (uint32_t i = 0; i < n_in; ++i) {
        if (!send_all(net->sock, &wires[i], sizeof(nf_tensor_wire)))
            return false;
    }

    /* -- Send: tensor payloads (zero-copy via mapped VA) ------------ */
    for (uint32_t i = 0; i < n_in; ++i) {
        nf_status st = nf_send_tensor_payload(
            static_cast<int>(net->sock),
            inputs[i],
            &input_ops[i],
            0 /* default timeout */);
        if (st != NF_OK) return false;
    }

    return true;
}

/* ================================================================== */
/*  IO Thread: receives responses with tensor payloads                 */
/*  Phase 5: allocates local buffers, receives directly into them.     */
/* ================================================================== */

static void io_recv_loop(nf_provider_network* net) {
    while (net->running.load(std::memory_order_relaxed)) {
        /* ---- Receive frame header ---- */
        nf_frame_header resp{};
        if (!recv_all(net->sock, &resp, sizeof(resp))) {
            if (net->running.load(std::memory_order_relaxed)) {
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

        /* Validate magic */
        uint32_t magic = nf_le32toh(resp.magic);
        if (magic != NF_PROTO_MAGIC) continue;

        uint64_t tid = nf_le64toh(resp.task_id);
        uint8_t n_out = resp.n_output_tensors;

        /* ---- Receive tensor wire descriptors ---- */
        std::vector<nf_tensor_wire> out_wires(n_out);
        for (uint8_t i = 0; i < n_out; ++i) {
            if (!recv_all(net->sock, &out_wires[i], sizeof(nf_tensor_wire))) {
                /* Connection lost mid-descriptor */
                goto conn_lost;
            }
        }

        /* ---- Find pending task ---- */
        {
            std::shared_ptr<PendingTask> pt;
            {
                std::lock_guard<std::mutex> lk(net->pending_mu);
                auto it = net->pending.find(tid);
                if (it != net->pending.end()) {
                    pt = it->second;
                }
            }

            if (resp.opcode == NF_OP_TASK_COMPLETE && pt && n_out > 0) {
                /* ---- Receive tensor payloads into pre-allocated buffers ---- */
                nf_status recv_st = NF_OK;

                for (uint8_t i = 0; i < n_out && i < pt->n_outputs; ++i) {
                    uint64_t payload_bytes = nf_le64toh(out_wires[i].payload_bytes);

                    if (payload_bytes == 0) continue;

                    /*
                     * If the caller pre-allocated output buffers, receive
                     * directly into them (zero-copy recv path).
                     * Otherwise, allocate via the local memory provider.
                     */
                    if (pt->outputs && pt->outputs[i] &&
                        pt->output_ops && pt->output_ops[i].map) {
                        /* Direct recv into caller's buffer */
                        nf_status st = nf_recv_tensor_payload(
                            static_cast<int>(net->sock),
                            pt->outputs[i],
                            &pt->output_ops[i],
                            &out_wires[i],
                            0);
                        if (st != NF_OK) recv_st = st;
                    } else if (net->local_mem_vt.alloc && net->local_mem_provider) {
                        /* Allocate a local hardware buffer, recv into it */
                        nf_tensor_desc desc{};
                        desc.dtype = static_cast<nf_dtype>(out_wires[i].dtype);
                        desc.ndim  = out_wires[i].ndim;
                        for (uint8_t d = 0; d < desc.ndim && d < NF_MAX_DIMS; ++d) {
                            desc.shape[d]   = nf_le64toh(out_wires[i].shape[d]);
                            desc.strides[d] = nf_le64toh(out_wires[i].strides[d]);
                        }
                        desc.size_bytes = payload_bytes;

                        nf_buffer_alloc_request req{};
                        req.desc      = desc;
                        req.preferred = NF_MEM_DOMAIN_CPU;

                        nf_buffer new_buf = nullptr;
                        nf_buffer_ops new_ops{};
                        nf_status ast = net->local_mem_vt.alloc(
                            net->local_mem_provider, &req, &new_ops, &new_buf);

                        if (ast == NF_OK && new_buf) {
                            nf_status st = nf_recv_tensor_payload(
                                static_cast<int>(net->sock),
                                new_buf, &new_ops, &out_wires[i], 0);
                            if (st != NF_OK) {
                                recv_st = st;
                                if (new_ops.release) new_ops.release(new_buf);
                            } else if (pt->outputs && i < pt->n_outputs) {
                                pt->outputs[i]    = new_buf;
                                pt->output_ops[i] = new_ops;
                            }
                        } else {
                            recv_st = NF_ERROR_OUT_OF_MEMORY;
                        }
                    } else {
                        /* No buffer and no allocator — skip payload */
                        std::vector<uint8_t> discard(payload_bytes);
                        recv_all(net->sock, discard.data(),
                                 static_cast<size_t>(payload_bytes));
                    }
                }

                /* Skip any extra output tensors beyond what caller expected */
                for (uint8_t i = pt->n_outputs; i < n_out; ++i) {
                    uint64_t skip = nf_le64toh(out_wires[i].payload_bytes);
                    if (skip > 0) {
                        std::vector<uint8_t> discard(skip);
                        recv_all(net->sock, discard.data(),
                                 static_cast<size_t>(skip));
                    }
                }

                pt->result = recv_st;
            } else if (resp.opcode == NF_OP_TASK_ERROR) {
                /* Error response — skip any payload */
                uint64_t skip = nf_le64toh(resp.total_payload_bytes);
                if (skip > 0) {
                    std::vector<uint8_t> discard(skip);
                    recv_all(net->sock, discard.data(),
                             static_cast<size_t>(skip));
                }
                if (pt) pt->result = NF_ERROR_INTERNAL;
            } else {
                /* Unknown opcode or no pending task — skip payload */
                uint64_t skip = nf_le64toh(resp.total_payload_bytes);
                if (skip > 0) {
                    std::vector<uint8_t> discard(skip);
                    recv_all(net->sock, discard.data(),
                             static_cast<size_t>(skip));
                }
            }

            /* Signal completion */
            if (pt) {
                if (pt->result == NF_OK && resp.opcode == NF_OP_TASK_COMPLETE) {
                    pt->result = NF_OK;
                }
                pt->completed.store(true, std::memory_order_release);
                pt->cv.notify_all();

                std::lock_guard<std::mutex> lk(net->pending_mu);
                net->pending.erase(tid);
            }
        }
        continue;

conn_lost:
        if (net->running.load(std::memory_order_relaxed)) {
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

    /* Phase 14: keepalive + send/recv timeouts */
    int keepalive = 1;
    ::setsockopt(net->sock, SOL_SOCKET, SO_KEEPALIVE,
                 reinterpret_cast<const char*>(&keepalive), sizeof(keepalive));

#ifdef _WIN32
    DWORD tv_ms = static_cast<DWORD>(NF_SOCKET_TIMEOUT_MS);
    ::setsockopt(net->sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&tv_ms), sizeof(tv_ms));
    ::setsockopt(net->sock, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&tv_ms), sizeof(tv_ms));
#else
    struct timeval tv;
    tv.tv_sec  = NF_SOCKET_TIMEOUT_MS / 1000;
    tv.tv_usec = (NF_SOCKET_TIMEOUT_MS % 1000) * 1000;
    ::setsockopt(net->sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(net->sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

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
 * dispatch() — the critical data path.
 *
 * Phase 5: Full zero-copy tensor transport.
 *
 * The nf_provider_vtable::dispatch signature only carries nf_buffer*
 * arrays (no ops). The network plugin needs buffer_ops for map/unmap.
 *
 * Solution: the PipelineEngine sets task_desc.user_data = &task_desc
 * before calling dispatch. We recover the full descriptor (with
 * input_ops/output_ops) by scanning the inputs array to find the
 * matching task_desc. This is pure C-ABI — no thread-locals, no
 * cross-dylib symbol resolution.
 */
static nf_status net_dispatch(nf_provider self,
                              const char* op_name,
                              const nf_buffer* inputs, uint32_t n_in,
                              nf_buffer* outputs, uint32_t n_out) {
    auto* net = reinterpret_cast<nf_provider_network*>(self);
    if (net->sock == NF_INVALID_SOCKET) return NF_ERROR_DEVICE_LOST;

    uint64_t tid = net->next_task_id.fetch_add(1, std::memory_order_relaxed);

    /* Register pending task with output buffer info */
    auto pt = std::make_shared<PendingTask>();
    pt->task_id   = tid;
    pt->outputs   = outputs;
    pt->n_outputs = n_out;

    /*
     * Recover the full task descriptor via user_data.
     * The PipelineEngine sets desc.user_data = &desc before dispatch.
     * We scan for a task_desc whose inputs array matches our inputs ptr.
     */
    const nf_buffer_ops* in_ops = nullptr;
    const nf_task_desc* td = nullptr;

    /* The engine passes inputs = desc.inputs, so we can recover desc
       by pointer arithmetic: desc is at (inputs - offsetof(nf_task_desc, inputs)).
       But safer: the engine also sets user_data on the desc. We need to
       find it. Since inputs points into desc.inputs, desc starts at a
       known offset before it. */
    {
        /* inputs points to desc.inputs[0]. Recover desc pointer. */
        const auto* candidate = reinterpret_cast<const nf_task_desc*>(
            reinterpret_cast<const char*>(inputs) -
            offsetof(nf_task_desc, inputs));
        if (candidate->user_data == candidate) {
            td = candidate;
            in_ops = td->input_ops;
            pt->output_ops = const_cast<nf_buffer_ops*>(td->output_ops);
        }
    }

    {
        std::lock_guard<std::mutex> lk(net->pending_mu);
        net->pending[tid] = pt;
    }

    /* Serialize and send with real payload */
    bool ok;
    if (in_ops) {
        ok = serialize_and_send(net, tid, op_name, inputs, in_ops, n_in);
    } else {
        /* Fallback: no ops available, send metadata only */
        static const nf_buffer_ops empty_ops[NF_MAX_TASK_INPUTS]{};
        ok = serialize_and_send(net, tid, op_name, inputs, empty_ops, n_in);
    }

    if (!ok) {
        std::lock_guard<std::mutex> lk(net->pending_mu);
        net->pending.erase(tid);
        return NF_ERROR_DEVICE_LOST;
    }

    /* Wait for response (Phase 14: bounded deadline instead of forever) */
    {
        std::unique_lock<std::mutex> lk(pt->mu);
        bool ok = pt->cv.wait_for(lk,
            std::chrono::milliseconds(NF_SOCKET_TIMEOUT_MS),
            [&pt] {
                return pt->completed.load(std::memory_order_acquire);
            });
        if (!ok) {
            std::lock_guard<std::mutex> plk(net->pending_mu);
            net->pending.erase(tid);
            return NF_ERROR_DEVICE_LOST;
        }
    }

    return pt->result;
}

static nf_status net_synchronize(nf_provider self) {
    auto* net = reinterpret_cast<nf_provider_network*>(self);

    /* Phase 14: bounded deadline instead of infinite busy-spin */
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(NF_SOCKET_TIMEOUT_MS);
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(net->pending_mu);
            if (net->pending.empty()) return NF_OK;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            return NF_ERROR_DEVICE_LOST;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
