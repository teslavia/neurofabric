/**
 * @file dataplane_transport_test.cpp
 * @brief Phase 42D.1 — DataPlane transport callback tests
 */

#include "neuralOS/mesh/async_dataflow.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

static int g_transport_calls = 0;
static uint32_t g_last_src = 0, g_last_dst = 0;
static uint64_t g_last_size = 0;

static bool mock_transport(uint32_t src, uint32_t dst, const void*, uint64_t size) {
    ++g_transport_calls;
    g_last_src = src;
    g_last_dst = dst;
    g_last_size = size;
    return true;
}

static bool mock_transport_fail(uint32_t, uint32_t, const void*, uint64_t) {
    return false;
}

static int g_prefetch_calls = 0;
static void mock_prefetch(uint32_t, uint32_t, const std::string&) {
    ++g_prefetch_calls;
}

int main() {
    printf("=== DataPlane Transport Test ===\n");

    /* Test 1: No callback → counter still works (backward compat) */
    {
        neuralOS::mesh::DataPlane dp;
        uint8_t data[64] = {};
        CHECK(dp.transfer(1, 2, data, 64), "transfer without callback succeeds");
        CHECK(dp.num_transfers() == 1, "transfer counted");
        dp.prefetch(1, 2, "tensor_a");
        CHECK(dp.num_prefetches() == 1, "prefetch counted");
    }

    /* Test 2: Set transport callback → verify called with correct args */
    {
        g_transport_calls = 0;
        neuralOS::mesh::DataPlane dp;
        dp.set_transport(mock_transport);

        uint8_t data[128] = {};
        CHECK(dp.transfer(3, 7, data, 128), "transfer with callback succeeds");
        CHECK(g_transport_calls == 1, "callback called once");
        CHECK(g_last_src == 3, "src node correct");
        CHECK(g_last_dst == 7, "dst node correct");
        CHECK(g_last_size == 128, "size correct");
        CHECK(dp.num_transfers() == 1, "transfer counted on success");
    }

    /* Test 3: Transfer failure → propagated */
    {
        neuralOS::mesh::DataPlane dp;
        dp.set_transport(mock_transport_fail);

        uint8_t data[64] = {};
        CHECK(!dp.transfer(1, 2, data, 64), "failed transfer returns false");
        CHECK(dp.num_transfers() == 0, "failed transfer not counted");
    }

    /* Test 4: Prefetch callback */
    {
        g_prefetch_calls = 0;
        neuralOS::mesh::DataPlane dp;
        dp.set_prefetch(mock_prefetch);

        dp.prefetch(1, 2, "weights");
        CHECK(g_prefetch_calls == 1, "prefetch callback called");
        CHECK(dp.num_prefetches() == 1, "prefetch counted");
    }

    printf("PASS: all DataPlane transport tests passed\n");
    return 0;
}
