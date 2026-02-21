/**
 * @file kquant_dequant_test.cpp
 * @brief Phase 25 — K-Quant Dequantization Silicon Verification
 *
 * Verifies all 7 new dequant kernels on real Metal GPU against CPU reference:
 *   1. dequant_q4_1
 *   2. dequant_q5_0
 *   3. dequant_q5_1
 *   4. dequant_q2_k
 *   5. dequant_q3_k
 *   6. dequant_q4_k
 *   7. dequant_q5_k
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"

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

static constexpr float TOLERANCE = 1e-3f;

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
    d.dtype = dtype; d.ndim = 1;
    d.shape[0] = size_bytes; d.size_bytes = size_bytes;
    return d;
}
/* PLACEHOLDER_HELPERS */

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

/* ================================================================== */
/*  Block structures (CPU reference, packed)                            */
/* ================================================================== */

#pragma pack(push, 1)
struct block_q4_1 { uint16_t d; uint16_t m; uint8_t qs[16]; };
struct block_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q5_1 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q2_k { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; };
struct block_q3_k { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct block_q4_k { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q5_k { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; uint8_t qh[32]; };
#pragma pack(pop)

static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size");
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 size");
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 size");
static_assert(sizeof(block_q2_k) == 84, "block_q2_k size");
static_assert(sizeof(block_q3_k) == 110, "block_q3_k size");
static_assert(sizeof(block_q4_k) == 144, "block_q4_k size");
static_assert(sizeof(block_q5_k) == 176, "block_q5_k size");

/* ================================================================== */
/*  CPU reference dequant functions                                     */
/* ================================================================== */

static void ref_dequant_q4_1(const block_q4_1* b, float* out) {
    float d = f16_to_f32(b->d);
    float m = f16_to_f32(b->m);
    for (int i = 0; i < 32; ++i) {
        int nibble = (i < 16) ? (b->qs[i] & 0xF) : (b->qs[i - 16] >> 4);
        out[i] = d * (float)nibble + m;
    }
}

static void ref_dequant_q5_0(const block_q5_0* b, float* out) {
    float d = f16_to_f32(b->d);
    for (int i = 0; i < 32; ++i) {
        int nibble = (i < 16) ? (b->qs[i] & 0xF) : (b->qs[i - 16] >> 4);
        int bit = (b->qh[i / 8] >> (i % 8)) & 1;
        out[i] = d * (float)((nibble | (bit << 4)) - 16);
    }
}

static void ref_dequant_q5_1(const block_q5_1* b, float* out) {
    float d = f16_to_f32(b->d);
    float m = f16_to_f32(b->m);
    for (int i = 0; i < 32; ++i) {
        int nibble = (i < 16) ? (b->qs[i] & 0xF) : (b->qs[i - 16] >> 4);
        int bit = (b->qh[i / 8] >> (i % 8)) & 1;
        out[i] = d * (float)(nibble | (bit << 4)) + m;
    }
}

/* PLACEHOLDER_KQUANT_REF */

static void ref_dequant_q2_k(const block_q2_k* b, float* out) {
    float super_d    = f16_to_f32(b->d);
    float super_dmin = f16_to_f32(b->dmin);
    for (int i = 0; i < 256; ++i) {
        int sub = i / 16;
        float sc = (float)(b->scales[sub] & 0xF) * super_d;
        float mn = (float)(b->scales[sub] >> 4) * super_dmin;
        int qs_idx = i / 4;
        int qs_shift = (i % 4) * 2;
        int q2 = (b->qs[qs_idx] >> qs_shift) & 0x3;
        out[i] = sc * (float)q2 - mn;
    }
}

static void ref_dequant_q3_k(const block_q3_k* b, float* out) {
    float super_d = f16_to_f32(b->d);
    for (int i = 0; i < 256; ++i) {
        int sub = i / 16;
        int scale;
        if (sub < 8) {
            scale = (int)(b->scales[sub] & 0xF)
                  | (((int)(b->scales[8 + sub / 2] >> (4 * (sub % 2))) & 0x3) << 4);
        } else {
            int s2 = sub - 8;
            scale = (int)(b->scales[s2] >> 4)
                  | (((int)(b->scales[8 + s2 / 2] >> (4 * (s2 % 2) + 2)) & 0x3) << 4);
        }
        scale -= 32;
        int qs_idx = i / 4;
        int qs_shift = (i % 4) * 2;
        int q_lo = (b->qs[qs_idx] >> qs_shift) & 0x3;
        int q_hi = (b->hmask[i / 8] >> (i % 8)) & 1;
        int q3 = q_lo | (q_hi << 2);
        out[i] = super_d * (float)scale * (float)(q3 - 4);
    }
}

static void ref_dequant_q4_k(const block_q4_k* b, float* out) {
    float super_d    = f16_to_f32(b->d);
    float super_dmin = f16_to_f32(b->dmin);
    for (int i = 0; i < 256; ++i) {
        int sub = i / 32;
        int sc, mn;
        if (sub < 4) {
            sc = (int)(b->scales[sub] & 0x3F);
            mn = (int)(b->scales[sub + 4] & 0x3F);
        } else {
            int s2 = sub - 4;
            sc = (int)((b->scales[s2] >> 6) | ((b->scales[s2 + 8] & 0x0F) << 2));
            mn = (int)((b->scales[s2 + 4] >> 6) | ((b->scales[s2 + 8] >> 4) << 2));
        }
        int qs_idx = i / 2;
        int nibble = (i & 1) ? (b->qs[qs_idx] >> 4) : (b->qs[qs_idx] & 0xF);
        out[i] = super_d * (float)sc * (float)nibble - super_dmin * (float)mn;
    }
}

static void ref_dequant_q5_k(const block_q5_k* b, float* out) {
    float super_d    = f16_to_f32(b->d);
    float super_dmin = f16_to_f32(b->dmin);
    for (int i = 0; i < 256; ++i) {
        int sub = i / 32;
        int sc, mn;
        if (sub < 4) {
            sc = (int)(b->scales[sub] & 0x3F);
            mn = (int)(b->scales[sub + 4] & 0x3F);
        } else {
            int s2 = sub - 4;
            sc = (int)((b->scales[s2] >> 6) | ((b->scales[s2 + 8] & 0x0F) << 2));
            mn = (int)((b->scales[s2 + 4] >> 6) | ((b->scales[s2 + 8] >> 4) << 2));
        }
        int qs_idx = i / 2;
        int nibble = (i & 1) ? (b->qs[qs_idx] >> 4) : (b->qs[qs_idx] & 0xF);
        int bit = (b->qh[i / 8] >> (i % 8)) & 1;
        out[i] = super_d * (float)sc * (float)(nibble | (bit << 4)) - super_dmin * (float)mn;
    }
}

/* ================================================================== */
/*  Generic dequant test runner                                         */
/* ================================================================== */

template <typename Block, typename RefFn>
static void run_dequant_test(const char* name, const char* op,
                              nf_dtype in_dtype, int block_elems,
                              int n_blocks, RefFn ref_fn)
{
    std::printf("  %s...\n", name);
    int n_elements = n_blocks * block_elems;

    /* Build blocks with pseudo-random data */
    std::vector<Block> blocks(n_blocks);
    std::memset(blocks.data(), 0, n_blocks * sizeof(Block));
    auto* raw = reinterpret_cast<uint8_t*>(blocks.data());
    for (size_t i = 0; i < n_blocks * sizeof(Block); ++i)
        raw[i] = (uint8_t)((i * 73 + 17) & 0xFF);

    /* Set scale fields to reasonable f16 values */
    for (int b = 0; b < n_blocks; ++b) {
        float scale = 0.01f * (float)(b + 1);
        auto* bp = reinterpret_cast<uint8_t*>(&blocks[b]);
        uint16_t h = f32_to_f16(scale);
        /* Write scale to first 2 bytes (d field) for all block types */
        std::memcpy(bp, &h, 2);
    }

    /* CPU reference */
    std::vector<float> ref(n_elements);
    for (int b = 0; b < n_blocks; ++b)
        ref_fn(&blocks[b], &ref[b * block_elems]);

    /* GPU: upload quantized data */
    size_t in_bytes = n_blocks * sizeof(Block);
    size_t out_bytes = n_elements * sizeof(float);
    auto in_desc = make_desc(in_dtype, in_bytes);
    auto out_desc = make_desc(NF_DTYPE_F32, out_bytes);

    nf_buffer_ops in_ops, out_ops;
    nf_buffer in_buf = alloc_buf(in_desc, &in_ops);
    nf_buffer out_buf = alloc_buf(out_desc, &out_ops);

    void* ptr;
    CHECK_OK(in_ops.map(in_buf, &ptr));
    std::memcpy(ptr, blocks.data(), in_bytes);
    in_ops.unmap(in_buf);

    /* Dispatch */
    nf_buffer ins[] = {in_buf};
    nf_buffer outs[] = {out_buf};
    CHECK_OK(g_vt.dispatch(g_prov, op, ins, 1, outs, 1));
    CHECK_OK(g_vt.synchronize(g_prov));

    /* Verify */
    CHECK_OK(out_ops.map(out_buf, &ptr));
    auto* gpu = static_cast<float*>(ptr);
    size_t mismatches = 0;
    for (int i = 0; i < n_elements; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        if (diff > TOLERANCE && diff > TOLERANCE * std::fabs(ref[i])) {
            if (mismatches < 5)
                std::printf("    MISMATCH [%d]: gpu=%.6f ref=%.6f diff=%.6f\n",
                            i, gpu[i], ref[i], diff);
            ++mismatches;
        }
    }
    out_ops.unmap(out_buf);
    CHECK(mismatches == 0);

    out_ops.release(out_buf);
    in_ops.release(in_buf);
    std::printf("    %s verified (%d blocks, %d elements) ✓\n", name, n_blocks, n_elements);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("kquant_dequant_test: Phase 25 — K-Quant Dequantization\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    run_dequant_test<block_q4_1>("dequant_q4_1", "dequant_q4_1",
        NF_DTYPE_Q4_1, 32, 8, ref_dequant_q4_1);
    run_dequant_test<block_q5_0>("dequant_q5_0", "dequant_q5_0",
        NF_DTYPE_Q5_0, 32, 8, ref_dequant_q5_0);
    run_dequant_test<block_q5_1>("dequant_q5_1", "dequant_q5_1",
        NF_DTYPE_Q5_1, 32, 8, ref_dequant_q5_1);
    run_dequant_test<block_q2_k>("dequant_q2_k", "dequant_q2_k",
        NF_DTYPE_Q2_K, 256, 4, ref_dequant_q2_k);
    run_dequant_test<block_q3_k>("dequant_q3_k", "dequant_q3_k",
        NF_DTYPE_Q3_K, 256, 4, ref_dequant_q3_k);
    run_dequant_test<block_q4_k>("dequant_q4_k", "dequant_q4_k",
        NF_DTYPE_Q4_K, 256, 4, ref_dequant_q4_k);
    run_dequant_test<block_q5_k>("dequant_q5_k", "dequant_q5_k",
        NF_DTYPE_Q5_K, 256, 4, ref_dequant_q5_k);

    g_vt.shutdown(g_prov);
    std::printf("OK: all Phase 25 K-quant dequant tests passed\n");
    return 0;
}
