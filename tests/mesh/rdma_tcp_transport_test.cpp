/**
 * @file rdma_tcp_transport_test.cpp
 * @brief Phase 45D — TCP transport loopback test
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

using namespace neuralOS::transport;

static void test_tcp_loopback() {
    /* Start a simple echo server on a random port */
    int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(srv_fd >= 0, "server socket");

    int opt = 1;
    setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sa.sin_port = 0;  /* OS picks port */
    CHECK(::bind(srv_fd, (struct sockaddr*)&sa, sizeof(sa)) == 0, "bind");
    CHECK(::listen(srv_fd, 1) == 0, "listen");

    /* Get assigned port */
    socklen_t slen = sizeof(sa);
    getsockname(srv_fd, (struct sockaddr*)&sa, &slen);
    uint16_t port = ntohs(sa.sin_port);

    /* Server thread: accept, echo back, close */
    std::thread server([srv_fd]() {
        int cli = ::accept(srv_fd, nullptr, nullptr);
        if (cli < 0) return;
        uint8_t buf[256];
        ssize_t n = ::recv(cli, buf, sizeof(buf), 0);
        if (n > 0) ::send(cli, buf, n, 0);
        ::close(cli);
    });

    /* Give server a moment */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    /* Client: connect via TransportOps */
    auto transport = make_tcp_transport();
    CHECK(transport.connect("127.0.0.1", port), "connect");
    CHECK(transport.is_connected, "is_connected");

    /* Send data */
    const char* msg = "Hello NeuralOS Transport!";
    size_t msg_len = std::strlen(msg) + 1;
    CHECK(transport.send(msg, msg_len), "send");

    /* Receive echo */
    char recv_buf[256] = {};
    CHECK(transport.recv(recv_buf, msg_len), "recv");
    CHECK(std::strcmp(recv_buf, msg) == 0, "data integrity");

    transport.disconnect();
    CHECK(!transport.is_connected, "disconnected");

    server.join();
    ::close(srv_fd);
    fprintf(stderr, "  [PASS] test_tcp_loopback\n");
}

static void test_select_transport() {
    auto tcp = select_transport("tcp://192.168.1.1:9999");
    CHECK(tcp.kind == TransportKind::TCP, "tcp scheme");

    auto def = select_transport(nullptr);
    CHECK(def.kind == TransportKind::TCP, "null scheme → TCP");

    auto def2 = select_transport();
    CHECK(def2.kind == TransportKind::TCP, "no-arg → TCP");

    fprintf(stderr, "  [PASS] test_select_transport\n");
}

int main() {
    fprintf(stderr, "[rdma_tcp_transport_test]\n");
    test_tcp_loopback();
    test_select_transport();
    fprintf(stderr, "[rdma_tcp_transport_test] ALL PASSED\n");
    return 0;
}
