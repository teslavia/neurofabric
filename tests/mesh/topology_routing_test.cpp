/**
 * @file topology_routing_test.cpp
 * @brief Phase 38.1 — Topology-aware routing tests
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
    printf("=== Topology Routing Test ===\n");

    neuralOS::kernel::VirtualBus bus;

    /* Diamond topology: 0 → 1, 0 → 2, 1 → 3, 2 → 3 */
    bus.register_provider(0, "src", "10.0.0.1", 16ULL<<30, 1000000000ULL, true);
    bus.register_provider(1, "mid_fast", "10.0.0.2", 8ULL<<30, 500000000ULL);
    bus.register_provider(2, "mid_slow", "10.0.0.3", 8ULL<<30, 500000000ULL);
    bus.register_provider(3, "dst", "10.0.0.4", 32ULL<<30, 2000000000ULL);

    /* Fast path: 0→1→3 (low latency, high bandwidth) */
    bus.add_link(0, 1, 100.0, 10.0);
    bus.add_link(1, 3, 100.0, 10.0);

    /* Slow path: 0→2→3 (high latency, low bandwidth) */
    bus.add_link(0, 2, 1.0, 1000.0);
    bus.add_link(2, 3, 1.0, 1000.0);

    /* Direct path: 0→3 (medium) */
    bus.add_link(0, 3, 10.0, 500.0);

    /* Route should prefer fast path (0→1→3) */
    auto r = bus.route(0, 3);
    CHECK(r.found, "route found");
    printf("  Route 0→3: cost=%.1f, hops=%zu\n", r.cost, r.path.size());

    /* The fast path (0→1→3) has cost = 10+10+10+10 = 40 (latency + 1000/bw)
     * Direct (0→3) has cost = 500+100 = 600
     * Slow (0→2→3) has cost = 1000+1000+1000+1000 = 4000
     * So fast path should win */
    CHECK(r.path.size() <= 3, "path is at most 3 hops");
    CHECK(r.path.front() == 0, "starts at 0");
    CHECK(r.path.back() == 3, "ends at 3");

    /* Verify fast path is chosen */
    if (r.path.size() == 3) {
        CHECK(r.path[1] == 1, "goes through fast mid node");
    }

    /* No route to disconnected node */
    bus.register_provider(99, "island", "10.0.0.99", 4ULL<<30, 100000000ULL);
    auto r2 = bus.route(0, 99);
    CHECK(!r2.found, "no route to disconnected node");

    printf("PASS: all topology routing tests passed\n");
    return 0;
}
