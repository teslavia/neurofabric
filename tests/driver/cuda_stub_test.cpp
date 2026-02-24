/**
 * @file cuda_stub_test.cpp
 * @brief Phase 45E — CUDA stub provider test (CPU fallback)
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

/* Helper: alloc a buffer via the provider vtable */
static nf_buffer alloc_buf(nf_provider_vtable& vt, nf_provider prov, uint64_t sz) {
    nf_tensor_desc desc{};
    desc.size_bytes = sz;
    nf_buffer buf;
    nf_status st = vt.buffer_alloc(prov, &desc, &buf);
    if (st != NF_OK) return nullptr;
    return buf;
}

static void test_init_shutdown() {
    nf_provider prov;
    nf_provider_vtable vt{};
    CHECK(nf_plugin_register(&vt, &prov) == NF_OK, "register");
    CHECK(vt.init(prov) == NF_OK, "init");
    CHECK(std::strcmp(vt.get_name(prov), "cuda_cpu_fallback") == 0, "name");
    CHECK(vt.get_abi_version(prov) == NF_ABI_VERSION, "abi version");
    vt.shutdown(prov);
    fprintf(stderr, "  [PASS] test_init_shutdown\n");
}

static void test_buffer_ops() {
    nf_provider prov;
    nf_provider_vtable vt{};
    nf_plugin_register(&vt, &prov);
    vt.init(prov);

    nf_buffer buf = alloc_buf(vt, prov, 256);
    CHECK(buf != nullptr, "alloc");

    void* p;
    CHECK(vt.buffer_map(prov, buf, &p) == NF_OK, "map");
    std::memset(p, 0x42, 256);
    vt.buffer_unmap(prov, buf);
    vt.buffer_free(prov, buf);
    vt.shutdown(prov);
    fprintf(stderr, "  [PASS] test_buffer_ops\n");
}

static void test_vector_add() {
    nf_provider prov;
    nf_provider_vtable vt{};
    nf_plugin_register(&vt, &prov);
    vt.init(prov);

    const uint32_t N = 8;
    const uint64_t sz = N * sizeof(float);

    nf_buffer a_buf = alloc_buf(vt, prov, sz);
    nf_buffer b_buf = alloc_buf(vt, prov, sz);
    nf_buffer c_buf = alloc_buf(vt, prov, sz);

    void* pa; vt.buffer_map(prov, a_buf, &pa);
    void* pb; vt.buffer_map(prov, b_buf, &pb);
    float* fa = static_cast<float*>(pa);
    float* fb = static_cast<float*>(pb);
    for (uint32_t i = 0; i < N; ++i) { fa[i] = (float)i; fb[i] = (float)(i * 2); }
    vt.buffer_unmap(prov, a_buf);
    vt.buffer_unmap(prov, b_buf);

    nf_buffer inputs[] = {a_buf, b_buf};
    nf_buffer outputs[] = {c_buf};
    CHECK(vt.dispatch(prov, "vector_add", inputs, 2, outputs, 1) == NF_OK, "dispatch");

    void* pc; vt.buffer_map(prov, c_buf, &pc);
    float* fc = static_cast<float*>(pc);
    for (uint32_t i = 0; i < N; ++i) {
        float expected = (float)i + (float)(i * 2);
        CHECK(std::fabs(fc[i] - expected) < 1e-5f, "vector_add result");
    }
    vt.buffer_unmap(prov, c_buf);

    vt.buffer_free(prov, a_buf);
    vt.buffer_free(prov, b_buf);
    vt.buffer_free(prov, c_buf);
    vt.shutdown(prov);
    fprintf(stderr, "  [PASS] test_vector_add\n");
}

static void test_element_mul() {
    nf_provider prov;
    nf_provider_vtable vt{};
    nf_plugin_register(&vt, &prov);
    vt.init(prov);

    const uint32_t N = 4;
    const uint64_t sz = N * sizeof(float);

    nf_buffer a_buf = alloc_buf(vt, prov, sz);
    nf_buffer b_buf = alloc_buf(vt, prov, sz);
    nf_buffer c_buf = alloc_buf(vt, prov, sz);

    void* pa; vt.buffer_map(prov, a_buf, &pa);
    void* pb; vt.buffer_map(prov, b_buf, &pb);
    float* fa = static_cast<float*>(pa);
    float* fb = static_cast<float*>(pb);
    for (uint32_t i = 0; i < N; ++i) { fa[i] = (float)(i + 1); fb[i] = 0.5f; }
    vt.buffer_unmap(prov, a_buf);
    vt.buffer_unmap(prov, b_buf);

    nf_buffer inputs[] = {a_buf, b_buf};
    nf_buffer outputs[] = {c_buf};
    CHECK(vt.dispatch(prov, "element_mul", inputs, 2, outputs, 1) == NF_OK, "dispatch");

    void* pc; vt.buffer_map(prov, c_buf, &pc);
    float* fc = static_cast<float*>(pc);
    for (uint32_t i = 0; i < N; ++i) {
        float expected = (float)(i + 1) * 0.5f;
        CHECK(std::fabs(fc[i] - expected) < 1e-5f, "element_mul result");
    }
    vt.buffer_unmap(prov, c_buf);

    vt.buffer_free(prov, a_buf);
    vt.buffer_free(prov, b_buf);
    vt.buffer_free(prov, c_buf);
    vt.shutdown(prov);
    fprintf(stderr, "  [PASS] test_element_mul\n");
}

int main() {
    fprintf(stderr, "[cuda_stub_test]\n");
    test_init_shutdown();
    test_buffer_ops();
    test_vector_add();
    test_element_mul();
    fprintf(stderr, "[cuda_stub_test] ALL PASSED\n");
    return 0;
}
