/**
 * @file rdma_transport.hpp
 * @brief NeuralOS L4 — Unified Transport Abstraction (TCP + RDMA stub)
 *
 * Phase 37.6: TransportOps interface with TCP implementation
 * and RDMA stub behind NF_HAS_RDMA feature gate.
 */

#ifndef NEURALOS_RDMA_TRANSPORT_HPP
#define NEURALOS_RDMA_TRANSPORT_HPP

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

namespace neuralOS { namespace transport {

/* ================================================================== */
/*  TransportOps — unified transport interface                         */
/* ================================================================== */

enum class TransportKind : uint8_t {
    TCP,
    RDMA
};

struct TransportOps {
    TransportKind kind = TransportKind::TCP;

    /* Connection lifecycle */
    std::function<bool(const std::string& addr, uint16_t port)> connect;
    std::function<void()> disconnect;

    /* Data transfer */
    std::function<bool(const void* data, size_t len)> send;
    std::function<bool(void* data, size_t len)> recv;

    /* RDMA-specific (no-op for TCP) */
    std::function<uint64_t(void* addr, size_t len)> register_mr;
    std::function<bool(uint64_t remote_mr, uint64_t local_mr,
                       size_t len, uint64_t remote_offset)> rdma_write;

    bool is_connected = false;
};

/* ================================================================== */
/*  TCP Transport — wraps POSIX sockets                                */
/* ================================================================== */

inline TransportOps make_tcp_transport() {
    TransportOps ops;
    ops.kind = TransportKind::TCP;

    /* Placeholder implementations — real socket code lives in network_provider.cpp.
     * These stubs allow the interface to be tested without actual networking. */
    ops.connect = [&ops](const std::string& addr, uint16_t port) -> bool {
        (void)addr; (void)port;
        ops.is_connected = true;
        return true;
    };

    ops.disconnect = [&ops]() {
        ops.is_connected = false;
    };

    ops.send = [](const void* data, size_t len) -> bool {
        (void)data; (void)len;
        return true;  /* stub */
    };

    ops.recv = [](void* data, size_t len) -> bool {
        (void)data; (void)len;
        return true;  /* stub */
    };

    ops.register_mr = [](void*, size_t) -> uint64_t { return 0; };
    ops.rdma_write = [](uint64_t, uint64_t, size_t, uint64_t) -> bool { return false; };

    return ops;
}

/* ================================================================== */
/*  RDMA Transport — stub behind feature gate                          */
/* ================================================================== */

#ifdef NF_HAS_RDMA
inline TransportOps make_rdma_transport() {
    TransportOps ops;
    ops.kind = TransportKind::RDMA;
    /* Real RDMA implementation would use ibverbs here */
    ops.connect = [&ops](const std::string& addr, uint16_t port) -> bool {
        (void)addr; (void)port;
        ops.is_connected = true;
        return true;
    };
    ops.disconnect = [&ops]() { ops.is_connected = false; };
    ops.send = [](const void*, size_t) -> bool { return true; };
    ops.recv = [](void*, size_t) -> bool { return true; };
    ops.register_mr = [](void* addr, size_t len) -> uint64_t {
        (void)addr; (void)len;
        return reinterpret_cast<uint64_t>(addr);
    };
    ops.rdma_write = [](uint64_t, uint64_t, size_t, uint64_t) -> bool {
        return true;
    };
    return ops;
}
#endif

/* ================================================================== */
/*  select_transport — auto-select best available transport            */
/* ================================================================== */

inline TransportOps select_transport() {
#ifdef NF_HAS_RDMA
    return make_rdma_transport();
#else
    return make_tcp_transport();
#endif
}

}} // namespace neuralOS::transport

#endif // NEURALOS_RDMA_TRANSPORT_HPP
