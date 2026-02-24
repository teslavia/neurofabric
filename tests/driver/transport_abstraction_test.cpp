/**
 * @file transport_abstraction_test.cpp
 * @brief Phase 37.6 — Transport abstraction (TCP/RDMA) tests
 */

#include "rdma_transport.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== Transport Abstraction Test ===\n");

    /* Test 1: TCP transport creation */
    auto tcp = neuralOS::transport::make_tcp_transport();
    CHECK(tcp.kind == neuralOS::transport::TransportKind::TCP, "TCP kind");
    CHECK(!tcp.is_connected, "not connected initially");

    /* Test 2: Connect */
    bool ok = tcp.connect("127.0.0.1", 9876);
    CHECK(ok, "TCP connect succeeded (stub)");
    CHECK(tcp.is_connected, "connected after connect");

    /* Test 3: Send/recv stubs */
    uint8_t buf[64] = {};
    CHECK(tcp.send(buf, 64), "TCP send stub OK");
    CHECK(tcp.recv(buf, 64), "TCP recv stub OK");

    /* Test 4: RDMA ops on TCP (should fail/no-op) */
    uint64_t mr = tcp.register_mr(buf, 64);
    CHECK(mr == 0, "TCP register_mr returns 0");
    CHECK(!tcp.rdma_write(0, 0, 64, 0), "TCP rdma_write returns false");

    /* Test 5: Disconnect */
    tcp.disconnect();
    CHECK(!tcp.is_connected, "disconnected");

    /* Test 6: select_transport (should return TCP without NF_HAS_RDMA) */
    auto selected = neuralOS::transport::select_transport();
#ifdef NF_HAS_RDMA
    CHECK(selected.kind == neuralOS::transport::TransportKind::RDMA, "RDMA selected");
#else
    CHECK(selected.kind == neuralOS::transport::TransportKind::TCP, "TCP selected (no RDMA)");
#endif

    printf("PASS: all transport abstraction tests passed\n");
    return 0;
}
