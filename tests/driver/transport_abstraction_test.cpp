/**
 * @file transport_abstraction_test.cpp
 * @brief Phase 37.6 — Transport abstraction (TCP/RDMA) tests
 */

#include "rdma_transport.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== Transport Abstraction Test ===\n");

    /* Test 1: TCP transport creation */
    auto tcp = neuralOS::mesh::transport::make_tcp_transport();
    CHECK(tcp.kind == neuralOS::mesh::transport::TransportKind::TCP, "TCP kind");
    CHECK(!tcp.is_connected, "not connected initially");

    /* Test 2: Connect to a real loopback server */
    int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(srv_fd >= 0, "server socket");
    int opt = 1;
    setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sa.sin_port = 0;
    CHECK(::bind(srv_fd, (struct sockaddr*)&sa, sizeof(sa)) == 0, "bind");
    CHECK(::listen(srv_fd, 1) == 0, "listen");
    socklen_t slen = sizeof(sa);
    getsockname(srv_fd, (struct sockaddr*)&sa, &slen);
    uint16_t port = ntohs(sa.sin_port);

    std::thread server([srv_fd]() {
        int cli = ::accept(srv_fd, nullptr, nullptr);
        if (cli < 0) return;
        uint8_t buf[64];
        ssize_t n = ::recv(cli, buf, sizeof(buf), 0);
        if (n > 0) ::send(cli, buf, n, 0);
        ::close(cli);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    bool ok = tcp.connect("127.0.0.1", port);
    CHECK(ok, "TCP connect succeeded");
    CHECK(tcp.is_connected, "connected after connect");

    /* Test 3: Send/recv with real data */
    uint8_t send_buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    CHECK(tcp.send(send_buf, 16), "TCP send OK");
    uint8_t recv_buf[16] = {};
    CHECK(tcp.recv(recv_buf, 16), "TCP recv OK");
    CHECK(std::memcmp(send_buf, recv_buf, 16) == 0, "echo data matches");

    /* Test 4: RDMA ops on TCP (should fail/no-op) */
    uint64_t mr = tcp.register_mr(send_buf, 64);
    CHECK(mr == 0, "TCP register_mr returns 0");
    CHECK(!tcp.rdma_write(0, 0, 64, 0), "TCP rdma_write returns false");

    /* Test 5: Disconnect */
    tcp.disconnect();
    CHECK(!tcp.is_connected, "disconnected");

    server.join();
    ::close(srv_fd);

    /* Test 6: select_transport (should return TCP without NF_HAS_RDMA) */
    auto selected = neuralOS::mesh::transport::select_transport();
#ifdef NF_HAS_RDMA
    CHECK(selected.kind == neuralOS::mesh::transport::TransportKind::RDMA, "RDMA selected");
#else
    CHECK(selected.kind == neuralOS::mesh::transport::TransportKind::TCP, "TCP selected (no RDMA)");
#endif

    /* Test 7: select_transport with scheme */
    auto tcp2 = neuralOS::mesh::transport::select_transport("tcp://localhost:1234");
    CHECK(tcp2.kind == neuralOS::mesh::transport::TransportKind::TCP, "tcp scheme");

    printf("PASS: all transport abstraction tests passed\n");
    return 0;
}
