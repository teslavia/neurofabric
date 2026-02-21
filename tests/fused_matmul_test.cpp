/**
 * @file fused_matmul_test.cpp
 * @brief Phase 29 — Fused Dequant-Q4_0 + Linear Kernel Correctness
 *
 * Verifies fused kernel matches separate dequant→linear pipeline:
 *   1. Small aligned (M=4, N=8, K=32)
 *   2. Non-aligned (M=5, N=7, K=64)
 *   3. Realistic (M=64, N=128, K=256)
 *   4. F16 variant (M=64, N=128, K=256)
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"
#include "neurofabric/PipelineEngine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static nf_provider            g_prov = nullptr;
static nf_provider_vtable     g_vt{};
static nf_provider_mem_vtable g_mem_vt{};

static nf_buffer alloc_buf(const nf_tensor_desc& desc, nf_buffer_ops* ops) {
    nf_buffer buf = nullptr;
    nf_buffer_alloc_request req{};
    req.desc = desc;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    CHECK_OK(g_mem_vt.alloc(g_prov, &req, ops, &buf));
    return buf;
}

static nf_tensor_desc make_desc(nf_dtype dtype, size_t size_bytes) {
    nf_tensor_desc d{};
    d.dtype = dtype;
    d.ndim = 1;
    d.shape[0] = size_bytes;
    d.size_bytes = size_bytes;
    return d;
}
/* NF_FUSED_TEST_PART2 */

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;       /* f16 scale */
    uint8_t  qs[16];  /* 4-bit quants, 2 per byte, 32 elements */
};
#pragma pack(pop)
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
               f = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 31) { f = sign | 0x7F800000u | (mant << 13);
    } else { f = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    float result; std::memcpy(&result, &f, 4); return result;
}

static uint16_t f32_to_f16(float val) {
    uint32_t f; std::memcpy(&f, &val, 4);
    uint16_t sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

static void quantize_q4_0(const float* src, block_q4_0* dst, size_t n_elem) {
    for (size_t b = 0; b < n_elem / 32; ++b) {
        const float* bs = src + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { float av = std::fabs(bs[i]); if (av > amax) amax = av; }
        float d = amax / 7.0f;
        dst[b].d = f32_to_f16(d);
        float df = f16_to_f32(dst[b].d);
        float id = (df != 0.0f) ? 1.0f / df : 0.0f;
        for (int i = 0; i < 16; ++i) {
            int lo = (int)std::roundf(bs[i] * id) + 8;
            int hi = (int)std::roundf(bs[i + 16] * id) + 8;
            lo = std::max(0, std::min(15, lo));
            hi = std::max(0, std::min(15, hi));
            dst[b].qs[i] = (uint8_t)((hi << 4) | lo);
        }
    }
}

/* NF_FUSED_TEST_PART3 */

/* Dispatch helper: sets up nf_task_desc with push constants for M,N,K */
struct FakeTaskDesc {
    nf_task_desc td;
    nf_buffer extra[3];
};

static nf_status dispatch_op(const char* op_name,
                              nf_buffer* in_bufs, uint32_t n_in,
                              nf_buffer* out_bufs, uint32_t n_out,
                              uint32_t M, uint32_t N, uint32_t K) {
    FakeTaskDesc ftd{};
    uint32_t pc[12] = {};
    pc[5] = M; pc[6] = N; pc[7] = K;
    std::memcpy(ftd.td.push_constants, pc, sizeof(pc));
    ftd.td.push_constants_size = sizeof(pc);
    nf_buffer* inp = ftd.td.inputs;
    for (uint32_t i = 0; i < n_in; ++i) inp[i] = in_bufs[i];
    return g_vt.dispatch(g_prov, op_name, inp, n_in, out_bufs, n_out);
}

static void sync() { g_vt.synchronize(g_prov); }

/* ================================================================== */
/*  Test: fused vs separate, parameterized                              */
/* ================================================================== */

static void test_fused_f32(const char* label, uint32_t M, uint32_t N, uint32_t K) {
    std::printf("  [%s] M=%u N=%u K=%u ... ", label, M, N, K);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    /* Activation A [M×K] */
    size_t a_bytes = M * K * sizeof(float);
    nf_buffer_ops ops{};
    auto a_desc = make_desc(NF_DTYPE_F32, a_bytes);
    nf_buffer a_buf = alloc_buf(a_desc, &ops);
    void* a_ptr; CHECK_OK(ops.map(a_buf, &a_ptr));
    float* a_data = (float*)a_ptr;
    for (size_t i = 0; i < M * K; ++i) a_data[i] = dist(rng);
    ops.unmap(a_buf);

    /* Weight B [K×N] as Q4_0 — K*N must be multiple of 32 */
    size_t kn = (size_t)K * N;
    size_t kn_padded = ((kn + 31) / 32) * 32;
    std::vector<float> b_f32(kn_padded, 0.0f);
    for (size_t i = 0; i < kn; ++i) b_f32[i] = dist(rng);
    size_t n_blocks = kn_padded / 32;
    std::vector<block_q4_0> b_q4(n_blocks);
    quantize_q4_0(b_f32.data(), b_q4.data(), kn_padded);

    auto bq_desc = make_desc(NF_DTYPE_Q4_0, n_blocks * sizeof(block_q4_0));
    nf_buffer bq_buf = alloc_buf(bq_desc, &ops);
    void* bq_ptr; CHECK_OK(ops.map(bq_buf, &bq_ptr));
    std::memcpy(bq_ptr, b_q4.data(), n_blocks * sizeof(block_q4_0));
    ops.unmap(bq_buf);

    /* Dequanted weights buffer [kn_padded floats] */
    auto dq_desc = make_desc(NF_DTYPE_F32, kn_padded * sizeof(float));
    nf_buffer dq_buf = alloc_buf(dq_desc, &ops);

    /* Output buffers */
    size_t out_bytes = M * N * sizeof(float);
    auto out_desc = make_desc(NF_DTYPE_F32, out_bytes);
    nf_buffer sep_buf = alloc_buf(out_desc, &ops);
    nf_buffer fused_buf = alloc_buf(out_desc, &ops);

    /* Separate path: dequant → linear_tiled */
    nf_buffer dq_in[] = {bq_buf};
    nf_buffer dq_out[] = {dq_buf};
    CHECK_OK(g_vt.dispatch(g_prov, "dequant_q4_0", dq_in, 1, dq_out, 1));
    sync();
    nf_buffer lin_in[] = {a_buf, dq_buf};
    nf_buffer lin_out[] = {sep_buf};
    CHECK_OK(dispatch_op("linear_tiled", lin_in, 2, lin_out, 1, M, N, K));
    sync();

    /* Fused path */
    nf_buffer fused_in[] = {a_buf, bq_buf};
    nf_buffer fused_out[] = {fused_buf};
    CHECK_OK(dispatch_op("dequant_q4_0_linear", fused_in, 2, fused_out, 1, M, N, K));
    sync();

    /* Compare */
    void *sp, *fp;
    CHECK_OK(ops.map(sep_buf, &sp));
    CHECK_OK(ops.map(fused_buf, &fp));
    float* sep_data = (float*)sp;
    float* fused_data = (float*)fp;
    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float err = std::fabs(sep_data[i] - fused_data[i]);
        if (err > max_err) max_err = err;
    }
    ops.unmap(sep_buf); ops.unmap(fused_buf);
    std::printf("max_err=%.6f %s\n", max_err, max_err < 1e-3f ? "PASS" : "FAIL");
    CHECK(max_err < 1e-3f);

    ops.release(a_buf); ops.release(bq_buf); ops.release(dq_buf);
    ops.release(sep_buf); ops.release(fused_buf);
}

/* NF_FUSED_TEST_PART4 */

static void test_fused_f16(uint32_t M, uint32_t N, uint32_t K) {
    std::printf("  [F16] M=%u N=%u K=%u ... ", M, N, K);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    nf_buffer_ops ops{};

    /* Activation A [M×K] as F16 */
    size_t a_elems = (size_t)M * K;
    auto a_desc = make_desc(NF_DTYPE_F16, a_elems * sizeof(uint16_t));
    nf_buffer a_buf = alloc_buf(a_desc, &ops);
    void* a_ptr; CHECK_OK(ops.map(a_buf, &a_ptr));
    uint16_t* a_h = (uint16_t*)a_ptr;
    for (size_t i = 0; i < a_elems; ++i) a_h[i] = f32_to_f16(dist(rng));
    ops.unmap(a_buf);

    /* Weight B [K×N] as Q4_0 */
    size_t kn = (size_t)K * N;
    size_t kn_padded = ((kn + 31) / 32) * 32;
    std::vector<float> b_f32(kn_padded, 0.0f);
    for (size_t i = 0; i < kn; ++i) b_f32[i] = dist(rng);
    size_t n_blocks = kn_padded / 32;
    std::vector<block_q4_0> b_q4(n_blocks);
    quantize_q4_0(b_f32.data(), b_q4.data(), kn_padded);

    auto bq_desc = make_desc(NF_DTYPE_Q4_0, n_blocks * sizeof(block_q4_0));
    nf_buffer bq_buf = alloc_buf(bq_desc, &ops);
    void* bq_ptr; CHECK_OK(ops.map(bq_buf, &bq_ptr));
    std::memcpy(bq_ptr, b_q4.data(), n_blocks * sizeof(block_q4_0));
    ops.unmap(bq_buf);

    /* Dequanted weights as F16 */
    auto dq_desc = make_desc(NF_DTYPE_F16, kn_padded * sizeof(uint16_t));
    nf_buffer dq_buf = alloc_buf(dq_desc, &ops);

    /* Output buffers (F16) */
    size_t out_bytes = (size_t)M * N * sizeof(uint16_t);
    auto out_desc = make_desc(NF_DTYPE_F16, out_bytes);
    nf_buffer sep_buf = alloc_buf(out_desc, &ops);
    nf_buffer fused_buf = alloc_buf(out_desc, &ops);

    /* Separate: dequant_q4_0_f16 → linear (auto-selects F16 path) */
    nf_buffer dq_in[] = {bq_buf};
    nf_buffer dq_out[] = {dq_buf};
    CHECK_OK(g_vt.dispatch(g_prov, "dequant_q4_0_f16", dq_in, 1, dq_out, 1));
    sync();
    nf_buffer lin_in[] = {a_buf, dq_buf};
    nf_buffer lin_out[] = {sep_buf};
    CHECK_OK(dispatch_op("linear", lin_in, 2, lin_out, 1, M, N, K));
    sync();

    /* Fused */
    nf_buffer fused_in[] = {a_buf, bq_buf};
    nf_buffer fused_out[] = {fused_buf};
    CHECK_OK(dispatch_op("dequant_q4_0_linear_f16", fused_in, 2, fused_out, 1, M, N, K));
    sync();

    /* Compare (F16 → F32 for comparison) */
    void *sp, *fp;
    CHECK_OK(ops.map(sep_buf, &sp));
    CHECK_OK(ops.map(fused_buf, &fp));
    uint16_t* sep_h = (uint16_t*)sp;
    uint16_t* fused_h = (uint16_t*)fp;
    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float err = std::fabs(f16_to_f32(sep_h[i]) - f16_to_f32(fused_h[i]));
        if (err > max_err) max_err = err;
    }
    ops.unmap(sep_buf); ops.unmap(fused_buf);
    std::printf("max_err=%.6f %s\n", max_err, max_err < 0.1f ? "PASS" : "FAIL");
    CHECK(max_err < 0.1f);

    ops.release(a_buf); ops.release(bq_buf); ops.release(dq_buf);
    ops.release(sep_buf); ops.release(fused_buf);
}

int main() {
    std::printf("=== Phase 29: Fused Dequant-Q4_0 + Linear Test ===\n");
    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));

    int pass = 0;
    test_fused_f32("small_aligned", 4, 8, 32);   ++pass;
    test_fused_f32("non_aligned",   5, 7, 64);    ++pass;
    test_fused_f32("realistic",     64, 128, 256); ++pass;
    test_fused_f16(64, 128, 256);                  ++pass;

    g_vt.shutdown(g_prov);
    std::printf("=== %d/4 tests passed ===\n", pass);
    return 0;
}
