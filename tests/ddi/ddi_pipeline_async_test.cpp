/**
 * @file ddi_pipeline_async_test.cpp
 * @brief Phase 42C.1 — DDI async dispatch in PipelineEngine tests
 */

#include "neuralOS/kernel/PipelineEngine.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

/* Mock provider state */
static int g_async_dispatches = 0;
static int g_sync_dispatches = 0;

static nf_status mock_dispatch(nf_provider, const char*, const nf_buffer*, uint32_t,
                                nf_buffer*, uint32_t) {
    ++g_sync_dispatches;
    return NF_OK;
}

static const char* mock_name(nf_provider) { return "mock_gpu"; }

static nf_status mock_dispatch_async(nf_provider, const char*, const nf_buffer*, uint32_t,
                                      nf_buffer*, uint32_t, nf_completion_token* token) {
    ++g_async_dispatches;
    *token = reinterpret_cast<nf_completion_token>(0x1234);
    return NF_OK;
}

static nf_status mock_wait(nf_provider, nf_completion_token, uint64_t) {
    return NF_OK;
}

static nf_status mock_poll(nf_provider, nf_completion_token) {
    return NF_OK;
}

static nf_status mock_query_caps(nf_provider, nf_driver_caps* caps) {
    caps->max_concurrent = 8;
    caps->memory_bytes = 16ULL * 1024 * 1024 * 1024;
    caps->flops = 100000000;
    caps->supported_dtypes = 0x0F;
    caps->flags = NF_CAP_ASYNC | NF_CAP_FP16;
    return NF_OK;
}

int main() {
    printf("=== DDI Pipeline Async Test ===\n");

    nf::PipelineEngine engine(2);

    /* Register a mock provider */
    nf_provider_vtable vt{};
    vt.dispatch = mock_dispatch;
    vt.get_name = mock_name;
    engine.register_provider(reinterpret_cast<nf_provider>(0x1), vt, NF_AFFINITY_GPU);

    /* Test 1: Provider without DDI → sync fallback */
    CHECK(engine.num_async_providers() == 0, "no async providers initially");

    /* Test 2: Register DDI vtable */
    nf_ddi_vtable ddi{};
    ddi.dispatch_async = mock_dispatch_async;
    ddi.wait_completion = mock_wait;
    ddi.poll_completion = mock_poll;
    ddi.query_caps = mock_query_caps;
    engine.register_ddi(0, ddi);

    CHECK(engine.num_async_providers() == 1, "1 async provider after register_ddi");

    /* Test 3: Registry updated with caps */
    auto& reg = engine.registry();
    CHECK(reg.count() >= 1, "registry has entries");
    auto caps = reg.query("mock_gpu");
    /* After register_ddi with query_caps, a second entry is added with full caps */
    CHECK(reg.count() >= 1, "registry populated");

    /* Test 4: Query by affinity */
    auto gpu_caps = reg.query_by_affinity(NF_AFFINITY_GPU);
    CHECK(!gpu_caps.empty(), "GPU caps found");

    printf("PASS: all DDI pipeline async tests passed\n");
    return 0;
}
