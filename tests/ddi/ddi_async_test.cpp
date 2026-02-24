/**
 * @file ddi_async_test.cpp
 * @brief Phase 37.5 — DDI async completion model tests
 */

#include "neuralOS/ddi/neuro_ddi.h"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

/* Mock DDI implementation for testing */
static nf_status mock_dispatch_async(nf_provider self,
                                      const char* op_name,
                                      const nf_buffer* inputs, uint32_t n_in,
                                      nf_buffer* outputs, uint32_t n_out,
                                      nf_completion_token* token) {
    (void)self; (void)op_name; (void)inputs; (void)n_in;
    (void)outputs; (void)n_out;
    /* Return a dummy token */
    *token = reinterpret_cast<nf_completion_token>((uintptr_t)42);
    return NF_OK;
}

static nf_status mock_wait(nf_provider self, nf_completion_token token,
                            uint64_t timeout_ms) {
    (void)self; (void)timeout_ms;
    if (token == reinterpret_cast<nf_completion_token>((uintptr_t)42))
        return NF_OK;
    return NF_ERROR_INVALID_ARG;
}

static nf_status mock_poll(nf_provider self, nf_completion_token token) {
    (void)self;
    if (token == reinterpret_cast<nf_completion_token>((uintptr_t)42))
        return NF_OK;
    return NF_ERROR_TIMEOUT;
}

static nf_status mock_query_caps(nf_provider self, nf_driver_caps* caps) {
    (void)self;
    caps->max_concurrent = 8;
    caps->memory_bytes = 16ULL * 1024 * 1024 * 1024;
    caps->flops = 10000000000ULL;
    caps->supported_dtypes = (1 << NF_DTYPE_F32) | (1 << NF_DTYPE_F16);
    caps->flags = NF_CAP_ASYNC | NF_CAP_FP16;
    return NF_OK;
}

int main() {
    printf("=== DDI Async Test ===\n");

    /* Test 1: Struct sizes */
    CHECK(sizeof(nf_driver_caps) == 32, "nf_driver_caps is 32 bytes");

    /* Test 2: Fill DDI vtable */
    nf_ddi_vtable ddi;
    ddi.dispatch_async = mock_dispatch_async;
    ddi.wait_completion = mock_wait;
    ddi.poll_completion = mock_poll;
    ddi.query_caps = mock_query_caps;

    /* Test 3: Async dispatch */
    nf_completion_token token = NULL;
    nf_status st = ddi.dispatch_async(NULL, "matmul", NULL, 0, NULL, 0, &token);
    CHECK(st == NF_OK, "dispatch_async OK");
    CHECK(token != NULL, "got completion token");

    /* Test 4: Wait */
    st = ddi.wait_completion(NULL, token, 1000);
    CHECK(st == NF_OK, "wait_completion OK");

    /* Test 5: Poll */
    st = ddi.poll_completion(NULL, token);
    CHECK(st == NF_OK, "poll_completion OK (already complete)");

    /* Test 6: Poll invalid token */
    st = ddi.poll_completion(NULL, NULL);
    CHECK(st == NF_ERROR_TIMEOUT, "poll invalid token → TIMEOUT");

    /* Test 7: Query caps */
    nf_driver_caps caps;
    st = ddi.query_caps(NULL, &caps);
    CHECK(st == NF_OK, "query_caps OK");
    CHECK(caps.max_concurrent == 8, "max_concurrent == 8");
    CHECK(caps.flags & NF_CAP_ASYNC, "has ASYNC cap");
    CHECK(caps.flags & NF_CAP_FP16, "has FP16 cap");
    CHECK(!(caps.flags & NF_CAP_RDMA), "no RDMA cap");

    /* Test 8: Capability flag checks */
    CHECK(NF_CAP_ASYNC == 0x01, "NF_CAP_ASYNC value");
    CHECK(NF_CAP_RDMA == 0x02, "NF_CAP_RDMA value");
    CHECK(NF_CAP_FP16 == 0x04, "NF_CAP_FP16 value");
    CHECK(NF_CAP_INT8 == 0x08, "NF_CAP_INT8 value");
    CHECK(NF_CAP_PAGED == 0x10, "NF_CAP_PAGED value");

    printf("PASS: all DDI async tests passed\n");
    return 0;
}
