/**
 * @file driver_registry_populate_test.cpp
 * @brief Phase 42C.2 — DriverRegistry auto-population tests
 */

#include "neuralOS/kernel/PipelineEngine.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

static nf_status mock_dispatch(nf_provider, const char*, const nf_buffer*, uint32_t,
                                nf_buffer*, uint32_t) { return NF_OK; }
static const char* mock_name_a(nf_provider) { return "metal_gpu"; }
static const char* mock_name_b(nf_provider) { return "rknn_npu"; }

int main() {
    printf("=== DriverRegistry Auto-Population Test ===\n");

    nf::PipelineEngine engine(1);

    /* Test 1: Register provider → registry.count() == 1 */
    nf_provider_vtable vt_a{};
    vt_a.dispatch = mock_dispatch;
    vt_a.get_name = mock_name_a;
    engine.register_provider(reinterpret_cast<nf_provider>(0x1), vt_a, NF_AFFINITY_GPU);

    CHECK(engine.registry().count() == 1, "registry has 1 entry after first register");

    /* Test 2: Register second provider → registry.count() == 2 */
    nf_provider_vtable vt_b{};
    vt_b.dispatch = mock_dispatch;
    vt_b.get_name = mock_name_b;
    engine.register_provider(reinterpret_cast<nf_provider>(0x2), vt_b, NF_AFFINITY_NPU);

    CHECK(engine.registry().count() == 2, "registry has 2 entries");

    /* Test 3: Query by name */
    auto* metal = engine.registry().query("metal_gpu");
    CHECK(metal != nullptr, "metal_gpu found in registry");
    CHECK(metal->affinity == NF_AFFINITY_GPU, "metal_gpu has GPU affinity");

    auto* rknn = engine.registry().query("rknn_npu");
    CHECK(rknn != nullptr, "rknn_npu found in registry");
    CHECK(rknn->affinity == NF_AFFINITY_NPU, "rknn_npu has NPU affinity");

    /* Test 4: Query by affinity */
    auto gpu_list = engine.registry().query_by_affinity(NF_AFFINITY_GPU);
    CHECK(gpu_list.size() == 1, "1 GPU provider");

    auto all_list = engine.registry().query_by_affinity(NF_AFFINITY_ANY);
    CHECK(all_list.size() == 2, "2 providers with ANY query");

    printf("PASS: all DriverRegistry auto-population tests passed\n");
    return 0;
}
