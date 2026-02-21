/**
 * @file dequant_test.cpp
 * @brief Phase 17 — Dequantization & Tiled MatMul Silicon Verification
 *
 * Verifies on real Metal GPU:
 *   1. dequant_q4_0: bit-exact match against CPU reference
 *   2. dequant_q8_0: bit-exact match against CPU reference
 *   3. linear_tiled: correctness against naive matmul for 64x64
 *   4. Pipeline: dequant_q4_0 → linear_tiled end-to-end
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
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static constexpr float TOLERANCE = 1e-4f;

static nf_provider          g_prov = nullptr;
static nf_provider_vtable   g_vt{};
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

/* ================================================================== */
/*  ggml-compatible block structures (CPU reference)                    */
/* ================================================================== */

/* Q4_0: 18 bytes per block, 32 elements */
#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;       /* f16 scale (IEEE 754 half) */
    uint8_t  qs[16];  /* 4-bit quants, 2 per byte */
};

struct block_q8_0 {
    uint16_t d;       /* f16 scale */
    int8_t   qs[32];  /* 8-bit quants */
};
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

/* f16 ↔ f32 conversion (IEEE 754) */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

static uint16_t f32_to_f16(float val) {
    uint32_t f;
    std::memcpy(&f, &val, 4);
    uint16_t sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

/* CPU reference: dequantize Q4_0 block */
static void ref_dequant_q4_0(const block_q4_0* block, float* out) {
    float d = f16_to_f32(block->d);
    for (int i = 0; i < 32; ++i) {
        int nibble = (i < 16)
            ? (block->qs[i] & 0xF)
            : (block->qs[i - 16] >> 4);
        out[i] = d * (float)(nibble - 8);
    }
}

/* CPU reference: dequantize Q8_0 block */
static void ref_dequant_q8_0(const block_q8_0* block, float* out) {
    float d = f16_to_f32(block->d);
    for (int i = 0; i < 32; ++i) {
        out[i] = d * (float)block->qs[i];
    }
}

/* Push constants layout (matches Metal PushConstants) */
struct PushConstants {
    uint32_t seq_len;
    uint32_t n_heads;
    uint32_t head_dim;
    float    epsilon;
    float    theta;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t step_idx;
    uint32_t _pad0;
};

/* PLACEHOLDER_TESTS */

/* ================================================================== */
/*  Test 1: dequant_q4_0 — bit-exact against CPU reference             */
/* ================================================================== */

static void test_dequant_q4_0() {
    std::printf("  test_dequant_q4_0...\n");

    constexpr int N_BLOCKS = 8;
    constexpr int N_ELEMENTS = N_BLOCKS * 32;

    /* Build Q4_0 blocks with known values */
    block_q4_0 blocks[N_BLOCKS];
    for (int b = 0; b < N_BLOCKS; ++b) {
        float scale = 0.5f + 0.1f * b;
        blocks[b].d = f32_to_f16(scale);
        for (int j = 0; j < 16; ++j) {
            uint8_t lo = (j + b) & 0xF;
            uint8_t hi = (15 - j + b) & 0xF;
            blocks[b].qs[j] = lo | (hi << 4);
        }
    }

    /* CPU reference */
    float ref[N_ELEMENTS];
    for (int b = 0; b < N_BLOCKS; ++b)
        ref_dequant_q4_0(&blocks[b], &ref[b * 32]);

    /* Allocate Metal buffers */
    nf_tensor_desc in_desc = make_desc(NF_DTYPE_Q4_0, sizeof(blocks));
    nf_tensor_desc out_desc = make_desc(NF_DTYPE_F32, N_ELEMENTS * sizeof(float));
    nf_buffer_ops in_ops{}, out_ops{};
    nf_buffer in_buf = alloc_buf(in_desc, &in_ops);
    nf_buffer out_buf = alloc_buf(out_desc, &out_ops);

    /* Upload quantized data */
    void* in_ptr = nullptr;
    CHECK_OK(in_ops.map(in_buf, &in_ptr));
    std::memcpy(in_ptr, blocks, sizeof(blocks));
    in_ops.unmap(in_buf);

    /* Dispatch on GPU */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "dequant_q4_0", NF_MAX_OP_NAME - 1);
    td.inputs[0] = in_buf; td.input_ops[0] = in_ops; td.n_inputs = 1;
    td.outputs[0] = out_buf; td.output_ops[0] = out_ops; td.n_outputs = 1;
    td.affinity = NF_AFFINITY_GPU;
    engine.add_task(gid, td);
    CHECK_OK(engine.submit(gid).get());

    /* Wait for GPU */
    out_ops.cache_sync(out_buf, NF_CACHE_INVALIDATE, 0, 0);

    /* Verify */
    void* out_ptr = nullptr;
    CHECK_OK(out_ops.map(out_buf, &out_ptr));
    auto* gpu_out = static_cast<float*>(out_ptr);
    size_t mismatches = 0;
    for (int i = 0; i < N_ELEMENTS; ++i) {
        float diff = std::fabs(gpu_out[i] - ref[i]);
        if (diff > TOLERANCE) {
            if (mismatches < 5)
                std::printf("    MISMATCH [%d]: gpu=%.6f ref=%.6f\n",
                            i, gpu_out[i], ref[i]);
            ++mismatches;
        }
    }
    out_ops.unmap(out_buf);
    CHECK(mismatches == 0);

    out_ops.release(out_buf);
    in_ops.release(in_buf);
    engine.destroy_graph(gid);
    std::printf("    %d elements verified ✓\n", N_ELEMENTS);
}

/* ================================================================== */
/*  Test 2: dequant_q8_0 — bit-exact against CPU reference             */
/* ================================================================== */

static void test_dequant_q8_0() {
    std::printf("  test_dequant_q8_0...\n");

    constexpr int N_BLOCKS = 8;
    constexpr int N_ELEMENTS = N_BLOCKS * 32;

    block_q8_0 blocks[N_BLOCKS];
    for (int b = 0; b < N_BLOCKS; ++b) {
        float scale = 0.25f + 0.05f * b;
        blocks[b].d = f32_to_f16(scale);
        for (int j = 0; j < 32; ++j)
            blocks[b].qs[j] = static_cast<int8_t>(j - 16 + b);
    }

    float ref[N_ELEMENTS];
    for (int b = 0; b < N_BLOCKS; ++b)
        ref_dequant_q8_0(&blocks[b], &ref[b * 32]);

    nf_tensor_desc in_desc = make_desc(NF_DTYPE_Q8_0, sizeof(blocks));
    nf_tensor_desc out_desc = make_desc(NF_DTYPE_F32, N_ELEMENTS * sizeof(float));
    nf_buffer_ops in_ops{}, out_ops{};
    nf_buffer in_buf = alloc_buf(in_desc, &in_ops);
    nf_buffer out_buf = alloc_buf(out_desc, &out_ops);

    void* in_ptr = nullptr;
    CHECK_OK(in_ops.map(in_buf, &in_ptr));
    std::memcpy(in_ptr, blocks, sizeof(blocks));
    in_ops.unmap(in_buf);

    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "dequant_q8_0", NF_MAX_OP_NAME - 1);
    td.inputs[0] = in_buf; td.input_ops[0] = in_ops; td.n_inputs = 1;
    td.outputs[0] = out_buf; td.output_ops[0] = out_ops; td.n_outputs = 1;
    td.affinity = NF_AFFINITY_GPU;
    engine.add_task(gid, td);
    CHECK_OK(engine.submit(gid).get());

    out_ops.cache_sync(out_buf, NF_CACHE_INVALIDATE, 0, 0);

    void* out_ptr = nullptr;
    CHECK_OK(out_ops.map(out_buf, &out_ptr));
    auto* gpu_out = static_cast<float*>(out_ptr);
    size_t mismatches = 0;
    for (int i = 0; i < N_ELEMENTS; ++i) {
        float diff = std::fabs(gpu_out[i] - ref[i]);
        if (diff > TOLERANCE) {
            if (mismatches < 5)
                std::printf("    MISMATCH [%d]: gpu=%.6f ref=%.6f\n",
                            i, gpu_out[i], ref[i]);
            ++mismatches;
        }
    }
    out_ops.unmap(out_buf);
    CHECK(mismatches == 0);

    out_ops.release(out_buf);
    in_ops.release(in_buf);
    engine.destroy_graph(gid);
    std::printf("    %d elements verified ✓\n", N_ELEMENTS);
}

/* PLACEHOLDER_MATMUL_TEST */

/* ================================================================== */
/*  Test 3: linear_tiled — correctness against naive CPU matmul        */
/* ================================================================== */

static void test_linear_tiled() {
    std::printf("  test_linear_tiled...\n");

    constexpr uint32_t M = 64, N = 48, K = 32;

    /* Generate deterministic A[M×K] and B[K×N] */
    std::vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f);
    for (uint32_t i = 0; i < M * K; ++i)
        A[i] = (float)(i % 17) * 0.1f - 0.8f;
    for (uint32_t i = 0; i < K * N; ++i)
        B[i] = (float)(i % 13) * 0.15f - 0.9f;

    /* CPU reference matmul */
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                acc += A[r * K + k] * B[k * N + c];
            C_ref[r * N + c] = acc;
        }

    /* Allocate Metal buffers */
    nf_tensor_desc a_desc = make_desc(NF_DTYPE_F32, M * K * sizeof(float));
    nf_tensor_desc b_desc = make_desc(NF_DTYPE_F32, K * N * sizeof(float));
    nf_tensor_desc c_desc = make_desc(NF_DTYPE_F32, M * N * sizeof(float));
    nf_buffer_ops a_ops{}, b_ops{}, c_ops{};
    nf_buffer a_buf = alloc_buf(a_desc, &a_ops);
    nf_buffer b_buf = alloc_buf(b_desc, &b_ops);
    nf_buffer c_buf = alloc_buf(c_desc, &c_ops);

    /* Upload */
    void* ptr = nullptr;
    CHECK_OK(a_ops.map(a_buf, &ptr));
    std::memcpy(ptr, A.data(), A.size() * sizeof(float));
    a_ops.unmap(a_buf);

    CHECK_OK(b_ops.map(b_buf, &ptr));
    std::memcpy(ptr, B.data(), B.size() * sizeof(float));
    b_ops.unmap(b_buf);

    /* Build DAG with push constants */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
    uint32_t gid = engine.create_graph();

    nf_task_desc td{};
    std::strncpy(td.op_name, "linear_tiled", NF_MAX_OP_NAME - 1);
    td.inputs[0] = a_buf; td.input_ops[0] = a_ops;
    td.inputs[1] = b_buf; td.input_ops[1] = b_ops;
    td.n_inputs = 2;
    td.outputs[0] = c_buf; td.output_ops[0] = c_ops;
    td.n_outputs = 1;
    td.affinity = NF_AFFINITY_GPU;

    /* Set push constants with M, N, K */
    PushConstants pc{};
    pc.M = M; pc.N = N; pc.K = K;
    std::memcpy(td.push_constants, &pc, sizeof(pc));
    td.push_constants_size = sizeof(pc);

    engine.add_task(gid, td);
    CHECK_OK(engine.submit(gid).get());

    c_ops.cache_sync(c_buf, NF_CACHE_INVALIDATE, 0, 0);

    /* Verify */
    CHECK_OK(c_ops.map(c_buf, &ptr));
    auto* gpu_c = static_cast<float*>(ptr);
    size_t mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
        float diff = std::fabs(gpu_c[i] - C_ref[i]);
        float rel = (std::fabs(C_ref[i]) > 1.0f)
            ? diff / std::fabs(C_ref[i]) : diff;
        if (rel > 1e-3f) {  /* slightly relaxed for tiled accumulation */
            if (mismatches < 5)
                std::printf("    MISMATCH [%u]: gpu=%.6f ref=%.6f\n",
                            i, gpu_c[i], C_ref[i]);
            ++mismatches;
        }
    }
    c_ops.unmap(c_buf);
    CHECK(mismatches == 0);

    c_ops.release(c_buf);
    b_ops.release(b_buf);
    a_ops.release(a_buf);
    engine.destroy_graph(gid);
    std::printf("    %ux%u matmul verified ✓\n", M, N);
}

/* ================================================================== */
/*  Test 4: dequant_q4_0 → linear_tiled pipeline                      */
/* ================================================================== */

static void test_dequant_linear_pipeline() {
    std::printf("  test_dequant_linear_pipeline...\n");

    /* Small pipeline: dequant 32 Q4_0 elements → use as A[1×32],
       multiply by B[32×4] → C[1×4] */
    constexpr uint32_t M = 1, K = 32, N = 4;

    /* Build one Q4_0 block */
    block_q4_0 block;
    block.d = f32_to_f16(1.0f);
    for (int j = 0; j < 16; ++j)
        block.qs[j] = (j & 0xF) | (((j + 1) & 0xF) << 4);

    /* CPU reference dequant */
    float dequant_ref[32];
    ref_dequant_q4_0(&block, dequant_ref);

    /* B matrix [32×4] */
    std::vector<float> B(K * N);
    for (uint32_t i = 0; i < K * N; ++i)
        B[i] = (i % 5) * 0.2f;

    /* CPU reference matmul */
    float c_ref[N];
    for (uint32_t c = 0; c < N; ++c) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; ++k)
            acc += dequant_ref[k] * B[k * N + c];
        c_ref[c] = acc;
    }

    /* Allocate Metal buffers */
    nf_tensor_desc q_desc = make_desc(NF_DTYPE_Q4_0, sizeof(block));
    nf_tensor_desc a_desc = make_desc(NF_DTYPE_F32, K * sizeof(float));
    nf_tensor_desc b_desc = make_desc(NF_DTYPE_F32, K * N * sizeof(float));
    nf_tensor_desc c_desc = make_desc(NF_DTYPE_F32, M * N * sizeof(float));
    nf_buffer_ops q_ops{}, a_ops{}, b_ops{}, c_ops{};
    nf_buffer q_buf = alloc_buf(q_desc, &q_ops);
    nf_buffer a_buf = alloc_buf(a_desc, &a_ops);
    nf_buffer b_buf = alloc_buf(b_desc, &b_ops);
    nf_buffer c_buf = alloc_buf(c_desc, &c_ops);

    /* Upload */
    void* ptr = nullptr;
    CHECK_OK(q_ops.map(q_buf, &ptr));
    std::memcpy(ptr, &block, sizeof(block));
    q_ops.unmap(q_buf);

    CHECK_OK(b_ops.map(b_buf, &ptr));
    std::memcpy(ptr, B.data(), B.size() * sizeof(float));
    b_ops.unmap(b_buf);

    /* Build 2-step pipeline: dequant then linear (sequential GPU sync) */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    /* Step 1: dequant_q4_0(q_buf) → a_buf */
    {
        uint32_t gid = engine.create_graph();
        nf_task_desc td0{};
        std::strncpy(td0.op_name, "dequant_q4_0", NF_MAX_OP_NAME - 1);
        td0.inputs[0] = q_buf; td0.input_ops[0] = q_ops; td0.n_inputs = 1;
        td0.outputs[0] = a_buf; td0.output_ops[0] = a_ops; td0.n_outputs = 1;
        td0.affinity = NF_AFFINITY_GPU;
        engine.add_task(gid, td0);
        CHECK_OK(engine.submit(gid).get());
        g_vt.synchronize(g_prov);  /* Ensure GPU dequant completes */
        engine.destroy_graph(gid);
    }

    /* Step 2: linear_tiled(a_buf, b_buf) → c_buf */
    {
        uint32_t gid = engine.create_graph();
        nf_task_desc td1{};
        std::strncpy(td1.op_name, "linear_tiled", NF_MAX_OP_NAME - 1);
        td1.inputs[0] = a_buf; td1.input_ops[0] = a_ops;
        td1.inputs[1] = b_buf; td1.input_ops[1] = b_ops;
        td1.n_inputs = 2;
        td1.outputs[0] = c_buf; td1.output_ops[0] = c_ops; td1.n_outputs = 1;
        td1.affinity = NF_AFFINITY_GPU;
        PushConstants pc{};
        pc.M = M; pc.N = N; pc.K = K;
        std::memcpy(td1.push_constants, &pc, sizeof(pc));
        td1.push_constants_size = sizeof(pc);
        engine.add_task(gid, td1);
        CHECK_OK(engine.submit(gid).get());
        g_vt.synchronize(g_prov);
        engine.destroy_graph(gid);
    }

    /* Synchronize and verify */
    c_ops.cache_sync(c_buf, NF_CACHE_INVALIDATE, 0, 0);

    CHECK_OK(c_ops.map(c_buf, &ptr));
    auto* gpu_c = static_cast<float*>(ptr);
    size_t mismatches = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float diff = std::fabs(gpu_c[i] - c_ref[i]);
        if (diff > 0.01f) {
            std::printf("    MISMATCH [%u]: gpu=%.6f ref=%.6f\n",
                        i, gpu_c[i], c_ref[i]);
            ++mismatches;
        }
    }
    c_ops.unmap(c_buf);
    CHECK(mismatches == 0);

    c_ops.release(c_buf);
    b_ops.release(b_buf);
    a_ops.release(a_buf);
    q_ops.release(q_buf);
    std::printf("    dequant→linear pipeline verified ✓\n");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("dequant_test: Phase 17 — Dequantization & Tiled MatMul\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    test_dequant_q4_0();
    test_dequant_q8_0();
    test_linear_tiled();
    test_dequant_linear_pipeline();

    g_vt.shutdown(g_prov);
    std::printf("OK: all Phase 17 dequant tests passed\n");
    return 0;
}
