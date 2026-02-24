/**
 * @file virtual_bus_test.cpp
 * @brief Phase 36.5 — VirtualBus topology, routing, splitting tests
 */

#include "neuralOS/kernel/VirtualBus.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== VirtualBus Test ===\n");

    neuralOS::L2::VirtualBus bus;

    /* Register 3 nodes: M4 Pro (local), RK3588, Cloud GPU */
    bus.register_provider(0, "m4_pro", "localhost", 32ULL * 1024 * 1024 * 1024,
                          10000000000ULL, true);
    bus.register_provider(1, "rk3588", "192.168.1.70:9999",
                          8ULL * 1024 * 1024 * 1024, 6000000000ULL);
    bus.register_provider(2, "cloud_gpu", "10.0.0.1:9999",
                          80ULL * 1024 * 1024 * 1024, 100000000000ULL);

    CHECK(bus.num_nodes() == 3, "3 nodes registered");

    /* Add links */
    bus.add_link(0, 1, 1.0, 500.0, 1);   /* M4 → RK3588: 1 GB/s, 500us */
    bus.add_link(1, 0, 1.0, 500.0, 1);   /* bidirectional */
    bus.add_link(0, 2, 10.0, 5000.0, 3); /* M4 → Cloud: 10 GB/s, 5ms */
    bus.add_link(2, 0, 10.0, 5000.0, 3);
    bus.add_link(1, 2, 0.5, 10000.0, 4); /* RK → Cloud: 0.5 GB/s, 10ms */
    bus.add_link(2, 1, 0.5, 10000.0, 4);

    CHECK(bus.num_edges() == 6, "6 edges");

    /* Test 1: Route M4 → RK3588 (direct) */
    auto r1 = bus.route(0, 1);
    CHECK(r1.found, "route 0→1 found");
    CHECK(r1.path.size() == 2, "direct path: 2 nodes");
    CHECK(r1.path[0] == 0 && r1.path[1] == 1, "path is [0, 1]");

    /* Test 2: Route M4 → Cloud */
    auto r2 = bus.route(0, 2);
    CHECK(r2.found, "route 0→2 found");
    /* Could be direct (0→2) or via RK (0→1→2), depends on cost */
    CHECK(r2.path.front() == 0 && r2.path.back() == 2, "starts at 0, ends at 2");

    /* Test 3: Route to self */
    auto r3 = bus.route(0, 0);
    CHECK(r3.found, "self-route found");
    CHECK(r3.cost == 0.0, "self-route cost is 0");

    /* Test 4: Route to non-existent node */
    auto r4 = bus.route(0, 99);
    CHECK(!r4.found, "no route to non-existent node");

    /* Test 5: Bandwidth query */
    double bw = bus.bandwidth_query(0, 1);
    CHECK(bw == 1.0, "bandwidth 0→1 is 1.0 GB/s");
    bw = bus.bandwidth_query(0, 2);
    CHECK(bw == 10.0, "bandwidth 0→2 is 10.0 GB/s");
    bw = bus.bandwidth_query(1, 0);
    CHECK(bw == 1.0, "bandwidth 1→0 is 1.0 GB/s");
    bw = bus.bandwidth_query(0, 99);
    CHECK(bw == 0.0, "no bandwidth to non-existent");

    /* Test 6: Split graph — 6 tasks across 3 nodes */
    std::vector<uint32_t> task_ids = {0, 1, 2, 3, 4, 5};
    std::vector<uint64_t> task_flops = {100, 200, 150, 300, 50, 100};

    auto parts = bus.split_graph(task_ids, task_flops);
    CHECK(parts.size() == 3, "3 partitions");

    /* Verify all tasks are assigned */
    uint32_t total_assigned = 0;
    for (auto& p : parts) total_assigned += static_cast<uint32_t>(p.task_ids.size());
    CHECK(total_assigned == 6, "all 6 tasks assigned");

    /* Test 7: Split with single node */
    neuralOS::L2::VirtualBus single;
    single.register_provider(0, "solo", "localhost", 16ULL * 1024 * 1024 * 1024,
                             5000000000ULL, true);
    auto sp = single.split_graph(task_ids, task_flops);
    CHECK(sp.size() == 1, "1 partition for single node");
    CHECK(sp[0].task_ids.size() == 6, "all tasks on single node");

    /* Test 8: Empty split */
    auto ep = bus.split_graph({}, {});
    CHECK(ep.empty(), "empty split returns empty");

    printf("PASS: all VirtualBus tests passed\n");
    return 0;
}
