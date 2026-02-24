/**
 * @file mesh_e2e_test.cpp
 * @brief Phase 44D.3 — Mesh E2E test
 *
 * Tests MeshCoordinator dispatch/collect with mock transport.
 * Simulates coordinator → worker subgraph dispatch and result collection.
 */

#include "neuralOS/mesh/mesh_coordinator.hpp"
#include "neuralOS/mesh/async_dataflow.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); std::exit(1); } \
} while(0)

int main() {
    std::fprintf(stderr, "[mesh_e2e] starting...\n");

    /* ---- Setup MeshCoordinator ---- */
    neuralOS::L5::MeshCoordinator::Config cfg;
    cfg.heartbeat_timeout_ms = 2000;
    neuralOS::L5::MeshCoordinator coord(cfg);

    /* Register coordinator (node 0) and worker (node 1) */
    bool ok = coord.register_node(0, "coordinator", "localhost:9000",
                                   16ULL * 1024 * 1024 * 1024, 100000, true);
    CHECK(ok, "register coordinator");

    ok = coord.register_node(1, "worker_0", "localhost:9999",
                              8ULL * 1024 * 1024 * 1024, 50000, false);
    CHECK(ok, "register worker");

    CHECK(coord.num_nodes() == 2, "should have 2 nodes");
    CHECK(coord.num_alive() == 2, "both should be alive");

    /* ---- Test 1: Topology + routing ---- */
    std::fprintf(stderr, "[mesh_e2e] test 1: topology + routing\n");

    coord.add_link(0, 1, 10.0, 100.0);  /* 10 Gbps, 100us latency */
    coord.add_link(1, 0, 10.0, 100.0);

    auto route = coord.route(0, 1);
    CHECK(route.found, "node 1 should be reachable from node 0");
    std::fprintf(stderr, "[mesh_e2e]   route 0→1: hops=%zu cost=%.1f\n",
                 route.path.size(), route.cost);

    /* ---- Test 2: Subgraph assignment ---- */
    std::fprintf(stderr, "[mesh_e2e] test 2: subgraph assignment\n");

    std::vector<uint32_t> task_ids = {0, 1, 2, 3};
    std::vector<uint64_t> task_flops = {1000, 2000, 1500, 500};
    auto partitions = coord.assign_subgraph(task_ids, task_flops);
    std::fprintf(stderr, "[mesh_e2e]   partitions: %zu\n", partitions.size());
    /* Should produce at least 1 partition */
    CHECK(!partitions.empty(), "should have partitions");

    /* ---- Test 3: Dispatch subgraph via DataPlane ---- */
    std::fprintf(stderr, "[mesh_e2e] test 3: dispatch subgraph\n");

    /* Set up mock transport that records transfers */
    uint32_t transport_calls = 0;
    coord.set_transport([&](uint32_t src, uint32_t dst,
                            const void* data, uint64_t size) -> bool {
        std::fprintf(stderr, "[mesh_e2e]   transport: %u → %u (%llu bytes)\n",
                     src, dst, (unsigned long long)size);
        ++transport_calls;
        return true;
    });

    /* Dispatch a payload to worker */
    float payload[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t handle = coord.dispatch_subgraph(1, {0, 1},
                                               payload, sizeof(payload));
    CHECK(handle != 0, "dispatch should return valid handle");
    CHECK(transport_calls == 1, "transport should be called once");
    CHECK(coord.num_dispatches() == 1, "dispatch count should be 1");

    /* ---- Test 4: Collect result (async resolve) ---- */
    std::fprintf(stderr, "[mesh_e2e] test 4: collect result\n");

    void* result_data = nullptr;
    CHECK(!coord.collect_result(handle, &result_data), "result not yet available");

    /* Simulate worker completing and sending result back */
    float result_payload[] = {2.0f, 4.0f, 6.0f, 8.0f};
    coord.resolve_dispatch(handle, result_payload);

    CHECK(coord.collect_result(handle, &result_data), "result should be available now");

    /* ---- Test 5: Heartbeat + health monitoring ---- */
    std::fprintf(stderr, "[mesh_e2e] test 5: heartbeat + health\n");

    coord.heartbeat(1, 0.3f);  /* worker reports 30% load */
    auto* health = coord.get_health(1);
    CHECK(health != nullptr, "should have health for node 1");
    CHECK(health->alive, "worker should be alive");
    CHECK(health->load_factor > 0.2f, "load factor should be ~0.3");

    /* Check no dead nodes */
    auto dead = coord.monitor_health();
    CHECK(dead.empty(), "no nodes should be dead");

    /* ---- Test 6: Multiple dispatches ---- */
    std::fprintf(stderr, "[mesh_e2e] test 6: multiple dispatches\n");

    float payload2[] = {5.0f, 6.0f};
    uint64_t h2 = coord.dispatch_subgraph(1, {2, 3}, payload2, sizeof(payload2));
    CHECK(h2 != 0, "second dispatch should succeed");
    CHECK(coord.num_dispatches() == 2, "dispatch count should be 2");
    CHECK(coord.num_transfers() == 2, "transfer count should be 2");

    /* Resolve second dispatch */
    float result2[] = {10.0f, 12.0f};
    coord.resolve_dispatch(h2, result2);
    CHECK(coord.collect_result(h2, &result_data), "second result should be available");

    /* ---- Done ---- */
    std::fprintf(stderr, "[mesh_e2e] PASS — dispatches=%u transfers=%u\n",
                 coord.num_dispatches(), coord.num_transfers());
    return 0;
}
