/**
 * @file mesh_coordinator_test.cpp
 * @brief Phase 38.1 — MeshCoordinator registration, health, assignment tests
 */

#include "neuralOS/mesh/mesh_coordinator.hpp"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== Mesh Coordinator Test ===\n");

    neuralOS::mesh::MeshCoordinator::Config cfg;
    cfg.heartbeat_timeout_ms = 50;  /* short for testing */
    neuralOS::mesh::MeshCoordinator coord(cfg);

    /* Register nodes */
    CHECK(coord.register_node(0, "m4_pro", "localhost", 32ULL<<30, 10000000000ULL, true),
          "register node 0");
    CHECK(coord.register_node(1, "rk3588", "192.168.1.70:9999", 8ULL<<30, 6000000000ULL),
          "register node 1");
    CHECK(!coord.register_node(0, "dup", "x", 0, 0), "duplicate rejected");
    CHECK(coord.num_nodes() == 2, "2 nodes");
    CHECK(coord.num_alive() == 2, "2 alive");

    /* Add link */
    coord.add_link(0, 1, 1.0, 500.0);

    /* Topology discovery */
    auto topo = coord.discover_topology();
    CHECK(topo.nodes.size() == 2, "topo has 2 nodes");
    CHECK(topo.edges.size() == 1, "topo has 1 edge");

    /* Routing */
    auto r = coord.route(0, 1);
    CHECK(r.found, "route 0→1 found");

    /* Subgraph assignment */
    std::vector<uint32_t> tasks = {0, 1, 2, 3};
    std::vector<uint64_t> flops = {100, 200, 150, 50};
    auto parts = coord.assign_subgraph(tasks, flops);
    CHECK(parts.size() == 2, "2 partitions");

    /* Health monitoring */
    coord.heartbeat(0, 0.5f);
    coord.heartbeat(1, 0.3f);

    auto h0 = coord.get_health(0);
    CHECK(h0 != nullptr, "health for node 0");
    CHECK(h0->alive, "node 0 alive");

    /* Wait for heartbeat timeout */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto dead = coord.monitor_health();
    CHECK(dead.size() == 2, "both nodes timed out");
    CHECK(coord.num_alive() == 0, "0 alive after timeout");

    /* Heartbeat revives */
    coord.heartbeat(0, 0.1f);
    CHECK(coord.get_health(0)->alive, "node 0 revived");

    /* Unregister */
    coord.unregister_node(1);
    CHECK(coord.num_nodes() == 1, "1 node after unregister");

    printf("PASS: all mesh coordinator tests passed\n");
    return 0;
}
