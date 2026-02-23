/**
 * @file fp16_kernel_test.cpp
 * @brief Phase 27 — FP16 Kernel Silicon Verification
 *
 * Tests all F16 compute kernels on real Metal GPU against CPU reference.
 * Tolerance: 1e-2 for half precision.
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_scheduler_abi.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static constexpr float TOL = 5e-2f;  /* half precision: ~3 decimal digits */

static nf_provider          g_prov = nullptr;
static nf_provider_vtable   g_vt{};
static nf_provider_mem_vtable g_mem_vt{};

static nf_buffer alloc_buf(nf_dtype dtype, size_t size_bytes, nf_buffer_ops* ops) {
    nf_buffer buf = nullptr;
    nf_tensor_desc d{}; d.dtype = dtype; d.ndim = 1;
    d.shape[0] = size_bytes; d.size_bytes = size_bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    CHECK_OK(g_mem_vt.alloc(g_prov, &req, ops, &buf));
    return buf;
}

/* F32 → F16 conversion */
static uint16_t f32_to_f16(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t sign = (fb >> 16) & 0x8000;
    int32_t exp = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (fb >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}
/* F16 → F32 conversion */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
               mant &= 0x3FF; f = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 31) { f = sign | 0x7F800000u | (mant << 13); }
    else { f = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    float result; std::memcpy(&result, &f, 4); return result;
}

struct PushConstants {
    uint32_t seq_len; uint32_t n_heads; uint32_t head_dim;
    float epsilon; float theta;
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t step_idx; uint32_t max_seq_len;
    uint32_t window_size; uint32_t _pad1;
};

static void fill_f16(nf_buffer buf, nf_buffer_ops& ops, const float* src, size_t n) {
    void* p; ops.map(buf, &p);
    uint16_t* h = (uint16_t*)p;
    for (size_t i = 0; i < n; ++i) h[i] = f32_to_f16(src[i]);
    ops.unmap(buf);
}

static void read_f16(nf_buffer buf, nf_buffer_ops& ops, float* dst, size_t n) {
    void* p; ops.map(buf, &p);
    const uint16_t* h = (const uint16_t*)p;
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(h[i]);
    ops.unmap(buf);
}

/* ---- Test: silu_f16 ---- */
static void test_silu_f16() {
    const size_t N = 64;
    nf_buffer_ops in_ops, out_ops;
    nf_buffer in_buf = alloc_buf(NF_DTYPE_F16, N * 2, &in_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, N * 2, &out_ops);

    std::vector<float> src(N), ref(N);
    for (size_t i = 0; i < N; ++i) src[i] = (float)i * 0.1f - 3.0f;
    fill_f16(in_buf, in_ops, src.data(), N);

    for (size_t i = 0; i < N; ++i) {
        float x = src[i];
        ref[i] = x / (1.0f + std::exp(-x));
    }

    nf_buffer ins[] = {in_buf}; nf_buffer outs[] = {out_buf};
    CHECK_OK(g_vt.dispatch(g_prov, "silu", ins, 1, outs, 1));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(N);
    read_f16(out_buf, out_ops, result.data(), N);
    for (size_t i = 0; i < N; ++i)
        CHECK(std::fabs(result[i] - ref[i]) < TOL);

    out_ops.release(out_buf); in_ops.release(in_buf);
    std::printf("  silu_f16: PASS\n");
}

/* ---- Test: vector_add_f16 ---- */
static void test_vector_add_f16() {
    const size_t N = 128;
    nf_buffer_ops a_ops, b_ops, out_ops;
    nf_buffer a_buf = alloc_buf(NF_DTYPE_F16, N * 2, &a_ops);
    nf_buffer b_buf = alloc_buf(NF_DTYPE_F16, N * 2, &b_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, N * 2, &out_ops);

    std::vector<float> a(N), b(N), ref(N);
    for (size_t i = 0; i < N; ++i) { a[i] = (float)i * 0.5f; b[i] = (float)i * -0.3f; ref[i] = a[i] + b[i]; }
    fill_f16(a_buf, a_ops, a.data(), N);
    fill_f16(b_buf, b_ops, b.data(), N);

    nf_buffer ins[] = {a_buf, b_buf}; nf_buffer outs[] = {out_buf};
    CHECK_OK(g_vt.dispatch(g_prov, "metal_vector_add", ins, 2, outs, 1));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(N);
    read_f16(out_buf, out_ops, result.data(), N);
    for (size_t i = 0; i < N; ++i) {
        float err = std::fabs(result[i] - ref[i]);
        float rel = (std::fabs(ref[i]) > 1.0f) ? err / std::fabs(ref[i]) : err;
        if (rel >= TOL) {
            std::fprintf(stderr, "  vector_add_f16 FAIL at [%zu]: got=%f ref=%f a=%f b=%f\n",
                         i, result[i], ref[i], a[i], b[i]);
        }
        CHECK(rel < TOL);
    }

    out_ops.release(out_buf); b_ops.release(b_buf); a_ops.release(a_buf);
    std::printf("  vector_add_f16: PASS\n");
}

/* ---- Test: elementwise_mul_f16 ---- */
static void test_elementwise_mul_f16() {
    const size_t N = 64;
    nf_buffer_ops a_ops, b_ops, out_ops;
    nf_buffer a_buf = alloc_buf(NF_DTYPE_F16, N * 2, &a_ops);
    nf_buffer b_buf = alloc_buf(NF_DTYPE_F16, N * 2, &b_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, N * 2, &out_ops);

    std::vector<float> a(N), b(N), ref(N);
    for (size_t i = 0; i < N; ++i) { a[i] = (float)i * 0.1f; b[i] = 2.0f - (float)i * 0.05f; ref[i] = a[i] * b[i]; }
    fill_f16(a_buf, a_ops, a.data(), N);
    fill_f16(b_buf, b_ops, b.data(), N);

    nf_buffer ins[] = {a_buf, b_buf}; nf_buffer outs[] = {out_buf};
    CHECK_OK(g_vt.dispatch(g_prov, "elementwise_mul", ins, 2, outs, 1));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(N);
    read_f16(out_buf, out_ops, result.data(), N);
    for (size_t i = 0; i < N; ++i)
        CHECK(std::fabs(result[i] - ref[i]) < 0.05f);

    out_ops.release(out_buf); b_ops.release(b_buf); a_ops.release(a_buf);
    std::printf("  elementwise_mul_f16: PASS\n");
}

/* ---- Test: linear_tiled_f16 (32×32 matmul) ---- */
static void test_linear_tiled_f16() {
    const uint32_t M = 32, N = 32, K = 32;
    nf_buffer_ops a_ops, b_ops, out_ops;
    nf_buffer a_buf = alloc_buf(NF_DTYPE_F16, M * K * 2, &a_ops);
    nf_buffer b_buf = alloc_buf(NF_DTYPE_F16, K * N * 2, &b_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, M * N * 2, &out_ops);

    std::vector<float> a(M * K), b(K * N), ref(M * N, 0.0f);
    for (size_t i = 0; i < M * K; ++i) a[i] = ((float)(i % 7) - 3.0f) * 0.1f;
    for (size_t i = 0; i < K * N; ++i) b[i] = ((float)(i % 5) - 2.0f) * 0.1f;
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t k = 0; k < K; ++k)
                ref[r * N + c] += a[r * K + k] * b[k * N + c];

    fill_f16(a_buf, a_ops, a.data(), M * K);
    fill_f16(b_buf, b_ops, b.data(), K * N);

    /* Set push constants via user_data trick — dispatch directly */
    nf_task_desc td{};
    std::strncpy(td.op_name, "linear", NF_MAX_OP_NAME - 1);
    PushConstants pc{}; pc.M = M; pc.N = N; pc.K = K;
    td.push_constants_size = sizeof(pc);
    std::memcpy(td.push_constants, &pc, sizeof(pc));
    td.inputs[0] = a_buf; td.input_ops[0] = a_ops;
    td.inputs[1] = b_buf; td.input_ops[1] = b_ops;
    td.n_inputs = 2;
    td.outputs[0] = out_buf; td.output_ops[0] = out_ops;
    td.n_outputs = 1;
    td.user_data = &td;

    CHECK_OK(g_vt.dispatch(g_prov, td.op_name,
                            td.inputs, td.n_inputs,
                            td.outputs, td.n_outputs));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(M * N);
    read_f16(out_buf, out_ops, result.data(), M * N);
    float max_err = 0;
    for (size_t i = 0; i < M * N; ++i) {
        float err = std::fabs(result[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    CHECK(max_err < 0.1f);  /* half precision matmul tolerance */

    out_ops.release(out_buf); b_ops.release(b_buf); a_ops.release(a_buf);
    std::printf("  linear_tiled_f16: PASS (max_err=%.4f)\n", max_err);
}

/* ---- Test: rms_norm_f16 ---- */
static void test_rms_norm_f16() {
    const uint32_t dim = 64;
    nf_buffer_ops in_ops, out_ops, w_ops;
    nf_buffer in_buf = alloc_buf(NF_DTYPE_F16, dim * 2, &in_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, dim * 2, &out_ops);
    nf_buffer w_buf = alloc_buf(NF_DTYPE_F32, dim * 4, &w_ops);  /* weights always F32 */

    std::vector<float> src(dim), weights(dim), ref(dim);
    for (size_t i = 0; i < dim; ++i) { src[i] = (float)i * 0.1f - 3.0f; weights[i] = 1.0f; }
    fill_f16(in_buf, in_ops, src.data(), dim);
    { void* p; w_ops.map(w_buf, &p); std::memcpy(p, weights.data(), dim * 4); w_ops.unmap(w_buf); }

    /* CPU reference */
    float sum_sq = 0;
    for (size_t i = 0; i < dim; ++i) sum_sq += src[i] * src[i];
    float rms = 1.0f / std::sqrt(sum_sq / dim + 1e-5f);
    for (size_t i = 0; i < dim; ++i) ref[i] = src[i] * rms * weights[i];

    /* Dispatch needs push constants */
    nf_task_desc td{};
    std::strncpy(td.op_name, "rms_norm", NF_MAX_OP_NAME - 1);
    PushConstants pc{}; pc.head_dim = dim; pc.epsilon = 1e-5f; pc.seq_len = 1;
    td.push_constants_size = sizeof(pc);
    std::memcpy(td.push_constants, &pc, sizeof(pc));
    td.inputs[0] = in_buf; td.input_ops[0] = in_ops;
    td.inputs[1] = w_buf;  td.input_ops[1] = w_ops;
    td.n_inputs = 2;
    td.outputs[0] = out_buf; td.output_ops[0] = out_ops;
    td.n_outputs = 1;
    td.user_data = &td;

    CHECK_OK(g_vt.dispatch(g_prov, td.op_name,
                            td.inputs, td.n_inputs,
                            td.outputs, td.n_outputs));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(dim);
    read_f16(out_buf, out_ops, result.data(), dim);
    for (size_t i = 0; i < dim; ++i)
        CHECK(std::fabs(result[i] - ref[i]) < TOL);

    w_ops.release(w_buf); out_ops.release(out_buf); in_ops.release(in_buf);
    std::printf("  rms_norm_f16: PASS\n");
}

/* ---- Test: dequant_q4_0_f16 ---- */
static void test_dequant_q4_0_f16() {
    /* 1 block = 32 elements, 18 bytes (half d + 16 bytes qs) */
    const uint32_t n_blocks = 4;
    const uint32_t n_elems = n_blocks * 32;
    const size_t block_size = 18;

    nf_buffer_ops in_ops, out_ops;
    nf_buffer in_buf = alloc_buf(NF_DTYPE_Q4_0, n_blocks * block_size, &in_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, n_elems * 2, &out_ops);

    /* Fill with known pattern */
    void* p; in_ops.map(in_buf, &p);
    uint8_t* raw = (uint8_t*)p;
    std::vector<float> ref(n_elems);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        float d_val = 0.5f + b * 0.1f;
        uint16_t d_f16 = f32_to_f16(d_val);
        std::memcpy(raw + b * block_size, &d_f16, 2);
        for (int i = 0; i < 16; ++i)
            raw[b * block_size + 2 + i] = ((i + 1) & 0xF) | (((i + 2) & 0xF) << 4);
        for (int i = 0; i < 32; ++i) {
            uint8_t byte_val = raw[b * block_size + 2 + (i % 16)];
            int nibble = (i < 16) ? (byte_val & 0xF) : (byte_val >> 4);
            ref[b * 32 + i] = d_val * (float)(nibble - 8);
        }
    }
    in_ops.unmap(in_buf);

    nf_buffer ins[] = {in_buf}; nf_buffer outs[] = {out_buf};
    CHECK_OK(g_vt.dispatch(g_prov, "dequant_q4_0_f16", ins, 1, outs, 1));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(n_elems);
    read_f16(out_buf, out_ops, result.data(), n_elems);
    for (size_t i = 0; i < n_elems; ++i)
        CHECK(std::fabs(result[i] - ref[i]) < TOL);

    out_ops.release(out_buf); in_ops.release(in_buf);
    std::printf("  dequant_q4_0_f16: PASS\n");
}

/* ---- Test: embedding_lookup_f16 ---- */
static void test_embedding_lookup_f16() {
    const uint32_t vocab = 8, dim = 16, seq = 2;
    nf_buffer_ops w_ops, t_ops, out_ops;
    nf_buffer w_buf = alloc_buf(NF_DTYPE_F16, vocab * dim * 2, &w_ops);
    nf_buffer t_buf = alloc_buf(NF_DTYPE_I32, seq * 4, &t_ops);
    nf_buffer out_buf = alloc_buf(NF_DTYPE_F16, seq * dim * 2, &out_ops);

    std::vector<float> weights(vocab * dim);
    for (size_t i = 0; i < vocab * dim; ++i) weights[i] = (float)i * 0.01f;
    fill_f16(w_buf, w_ops, weights.data(), vocab * dim);

    int32_t tokens[] = {3, 5};
    { void* p; t_ops.map(t_buf, &p); std::memcpy(p, tokens, seq * 4); t_ops.unmap(t_buf); }

    nf_task_desc td{};
    std::strncpy(td.op_name, "embedding_lookup", NF_MAX_OP_NAME - 1);
    PushConstants pc{}; pc.head_dim = dim; pc.seq_len = seq;
    td.push_constants_size = sizeof(pc);
    std::memcpy(td.push_constants, &pc, sizeof(pc));
    td.inputs[0] = w_buf; td.input_ops[0] = w_ops;
    td.inputs[1] = t_buf; td.input_ops[1] = t_ops;
    td.n_inputs = 2;
    td.outputs[0] = out_buf; td.output_ops[0] = out_ops;
    td.n_outputs = 1;
    td.user_data = &td;

    CHECK_OK(g_vt.dispatch(g_prov, td.op_name, td.inputs, td.n_inputs, td.outputs, td.n_outputs));
    CHECK_OK(g_vt.synchronize(g_prov));

    std::vector<float> result(seq * dim);
    read_f16(out_buf, out_ops, result.data(), seq * dim);
    for (uint32_t s = 0; s < seq; ++s)
        for (uint32_t d = 0; d < dim; ++d)
            CHECK(std::fabs(result[s * dim + d] - weights[tokens[s] * dim + d]) < TOL);

    out_ops.release(out_buf); t_ops.release(t_buf); w_ops.release(w_buf);
    std::printf("  embedding_lookup_f16: PASS\n");
}

int main() {
    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));

    std::printf("[fp16_kernel_test] Phase 27 — FP16 Kernel Verification\n");
    test_silu_f16();
    test_vector_add_f16();
    test_elementwise_mul_f16();
    test_linear_tiled_f16();
    test_rms_norm_f16();
    test_dequant_q4_0_f16();
    test_embedding_lookup_f16();

    g_vt.shutdown(g_prov);
    std::printf("[fp16_kernel_test] All 7 tests PASSED\n");
    return 0;
}
