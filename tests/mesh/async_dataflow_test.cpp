/**
 * @file async_dataflow_test.cpp
 * @brief Phase 38.2 — Async dataflow (control/data plane) tests
 */

#include "neuralOS/mesh/async_dataflow.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== Async Dataflow Test ===\n");

    neuralOS::mesh::ControlPlane ctrl;
    neuralOS::mesh::DataPlane data;

    /* Test 1: Dispatch async handles */
    uint64_t h1 = ctrl.dispatch_async(0, 1, "hidden_state", 4096);
    uint64_t h2 = ctrl.dispatch_async(0, 2, "kv_cache", 8192);
    uint64_t h3 = ctrl.dispatch_async(1, 2, "logits", 2048);

    CHECK(h1 > 0 && h2 > 0 && h3 > 0, "handles allocated");
    CHECK(ctrl.num_handles() == 3, "3 handles");
    CHECK(ctrl.num_pending() == 3, "3 pending");

    /* Test 2: Not yet resolved */
    CHECK(!ctrl.is_resolved(h1), "h1 not resolved");
    CHECK(!ctrl.gang_schedule({h1, h2, h3}), "gang not ready");

    /* Test 3: Resolve h1 */
    float dummy_data[1024];
    CHECK(ctrl.resolve(h1, dummy_data), "resolve h1");
    CHECK(ctrl.is_resolved(h1), "h1 resolved");
    CHECK(ctrl.num_pending() == 2, "2 pending");

    /* Test 4: Callback on resolve */
    bool callback_fired = false;
    ctrl.on_resolve(h2, [&](neuralOS::mesh::DataHandle* h) {
        callback_fired = true;
        CHECK(h->handle_id == h2, "callback has correct handle");
    });
    ctrl.resolve(h2, dummy_data);
    CHECK(callback_fired, "callback fired on resolve");

    /* Test 5: Gang schedule — still missing h3 */
    CHECK(!ctrl.gang_schedule({h1, h2, h3}), "gang still not ready (h3 pending)");

    /* Resolve h3 */
    ctrl.resolve(h3, dummy_data);
    CHECK(ctrl.gang_schedule({h1, h2, h3}), "gang ready after all resolved");
    CHECK(ctrl.num_pending() == 0, "0 pending");

    /* Test 6: Resolve invalid handle */
    CHECK(!ctrl.resolve(9999, dummy_data), "invalid handle fails");

    /* Test 7: Data plane transfers */
    CHECK(data.transfer(0, 1, dummy_data, 4096), "transfer OK");
    CHECK(data.num_transfers() == 1, "1 transfer");

    data.prefetch(0, 2, "next_layer");
    CHECK(data.num_prefetches() == 1, "1 prefetch");

    printf("PASS: all async dataflow tests passed\n");
    return 0;
}
