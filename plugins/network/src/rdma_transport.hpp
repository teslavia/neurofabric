/**
 * @file rdma_transport.hpp
 * @brief NeuralOS L4 — Unified Transport Abstraction (TCP + RDMA stub)
 *
 * Phase 37.6: TransportOps interface with TCP implementation
 * and RDMA stub behind NF_HAS_RDMA feature gate.
 * Phase 45D: Real POSIX socket TCP transport.
 */

#ifndef NEURALOS_MESH_RDMA_TRANSPORT_HPP
#define NEURALOS_MESH_RDMA_TRANSPORT_HPP

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <memory>

namespace neuralOS { namespace mesh { namespace transport {

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

    auto sock = std::make_shared<int>(-1);

    ops.connect = [sock, &ops](const std::string& addr, uint16_t port) -> bool {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return false;
        struct sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(port);
        if (::inet_pton(AF_INET, addr.c_str(), &sa.sin_addr) <= 0) {
            ::close(fd);
            return false;
        }
        if (::connect(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
            ::close(fd);
            return false;
        }
        *sock = fd;
        ops.is_connected = true;
        return true;
    };

    ops.disconnect = [sock, &ops]() {
        if (*sock >= 0) { ::close(*sock); *sock = -1; }
        ops.is_connected = false;
    };

    ops.send = [sock](const void* data, size_t len) -> bool {
        if (*sock < 0) return false;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        size_t sent = 0;
        while (sent < len) {
            ssize_t n = ::send(*sock, p + sent, len - sent, 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    };

    ops.recv = [sock](void* data, size_t len) -> bool {
        if (*sock < 0) return false;
        uint8_t* p = static_cast<uint8_t*>(data);
        size_t got = 0;
        while (got < len) {
            ssize_t n = ::recv(*sock, p + got, len - got, 0);
            if (n <= 0) return false;
            got += static_cast<size_t>(n);
        }
        return true;
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

inline TransportOps select_transport(const char* scheme) {
    if (scheme) {
        std::string s(scheme);
        if (s.find("rdma://") == 0) {
#ifdef NF_HAS_RDMA
            return make_rdma_transport();
#endif
        }
    }
    return make_tcp_transport();
}

}}} // namespace neuralOS::mesh::transport

/* ================================================================== */
/*  Backward-compat alias: neuralOS::transport → neuralOS::mesh::transport */
/* ================================================================== */
namespace neuralOS { namespace transport {
    using neuralOS::mesh::transport::TransportKind;
    using neuralOS::mesh::transport::TransportOps;
    using neuralOS::mesh::transport::make_tcp_transport;
#ifdef NF_HAS_RDMA
    using neuralOS::mesh::transport::make_rdma_transport;
#endif
    using neuralOS::mesh::transport::select_transport;
}} // namespace neuralOS::transport

#endif // NEURALOS_MESH_RDMA_TRANSPORT_HPP
