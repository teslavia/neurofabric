/**
 * @file kv_cache_policy_test.cpp
 * @brief KV Cache Policy unit tests — pure CPU, no Metal/RKNN dependency
 *
 * Phase 31: Multi-Architecture DAG & KV Cache Intelligence.
 *
 * Tests:
 *   1. NONE policy: write_offset, attn_range, effective_len
 *   2. SLIDING policy: ring buffer correctness, boundary conditions
 *   3. LRU tracker: touch/evict ordering, capacity limits
 *   4. INT8 quantization: round-trip error < 0.5%
 *   5. INT8 F16 variant: round-trip correctness
 */

#include "model/kv_cache_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

/* ---- Test 1: NONE policy ---- */
static void test_none_policy() {
    std::printf("[kv_policy] Test 1: NONE policy...\n");

    nf::nf_kv_cache_config cfg{};
    cfg.eviction = nf::NF_KV_EVICT_NONE;
    cfg.max_seq_len = 2048;
    auto p = nf::nf_create_kv_policy(cfg);

    /* write_offset = step */
    CHECK(p.write_offset(&p, 0) == 0);
    CHECK(p.write_offset(&p, 42) == 42);
    CHECK(p.write_offset(&p, 2047) == 2047);

    /* attn_range = [0, step+1) */
    uint32_t start, len;
    p.attn_range(&p, 0, &start, &len);
    CHECK(start == 0 && len == 1);

    p.attn_range(&p, 99, &start, &len);
    CHECK(start == 0 && len == 100);

    /* effective_len = step+1 */
    CHECK(p.effective_len(&p, 0) == 1);
    CHECK(p.effective_len(&p, 511) == 512);

    std::printf("  PASS\n");
}

/* ---- Test 2: SLIDING policy ---- */
static void test_sliding_policy() {
    std::printf("[kv_policy] Test 2: SLIDING policy...\n");

    nf::nf_kv_cache_config cfg{};
    cfg.eviction = nf::NF_KV_EVICT_SLIDING;
    cfg.window_size = 8;
    cfg.max_seq_len = 2048;
    auto p = nf::nf_create_kv_policy(cfg);

    /* write_offset = step % window_size */
    CHECK(p.write_offset(&p, 0) == 0);
    CHECK(p.write_offset(&p, 7) == 7);
    CHECK(p.write_offset(&p, 8) == 0);  /* wraps */
    CHECK(p.write_offset(&p, 15) == 7);
    CHECK(p.write_offset(&p, 16) == 0);
    CHECK(p.write_offset(&p, 100) == 100 % 8);

    /* attn_range: before window fills */
    uint32_t start, len;
    p.attn_range(&p, 0, &start, &len);
    CHECK(start == 0 && len == 1);

    p.attn_range(&p, 3, &start, &len);
    CHECK(start == 0 && len == 4);

    p.attn_range(&p, 7, &start, &len);
    CHECK(start == 0 && len == 8);  /* exactly full */

    /* attn_range: after window fills */
    p.attn_range(&p, 8, &start, &len);
    CHECK(start == 1 && len == 8);

    p.attn_range(&p, 15, &start, &len);
    CHECK(start == 8 && len == 8);

    p.attn_range(&p, 100, &start, &len);
    CHECK(start == 93 && len == 8);

    /* effective_len */
    CHECK(p.effective_len(&p, 0) == 1);
    CHECK(p.effective_len(&p, 5) == 6);
    CHECK(p.effective_len(&p, 7) == 8);
    CHECK(p.effective_len(&p, 8) == 8);  /* capped at window */
    CHECK(p.effective_len(&p, 1000) == 8);

    /* Edge case: window_size = 1 */
    cfg.window_size = 1;
    p = nf::nf_create_kv_policy(cfg);
    CHECK(p.write_offset(&p, 0) == 0);
    CHECK(p.write_offset(&p, 99) == 0);
    p.attn_range(&p, 99, &start, &len);
    CHECK(start == 99 && len == 1);
    CHECK(p.effective_len(&p, 99) == 1);

    std::printf("  PASS\n");
}

/* ---- Test 3: LRU tracker ---- */
static void test_lru_tracker() {
    std::printf("[kv_policy] Test 3: LRU tracker...\n");

    /* Create tracker: capacity=4, max_positions=16 */
    auto t = nf::nf_lru_create(4, 16);

    /* Empty state */
    CHECK(nf::nf_lru_coldest(t) == nf::NF_LRU_INVALID);
    CHECK(!nf::nf_lru_contains(t, 0));

    /* Touch positions 0,1,2,3 — order: head=3,2,1,0=tail */
    nf::nf_lru_touch(t, 0);
    nf::nf_lru_touch(t, 1);
    nf::nf_lru_touch(t, 2);
    nf::nf_lru_touch(t, 3);
    CHECK(t.count == 4);
    CHECK(nf::nf_lru_coldest(t) == 0);  /* 0 is LRU */
    CHECK(nf::nf_lru_contains(t, 0));
    CHECK(nf::nf_lru_contains(t, 3));

    /* Touch 0 again — moves to front. Now: head=0,3,2,1=tail */
    nf::nf_lru_touch(t, 0);
    CHECK(nf::nf_lru_coldest(t) == 1);  /* 1 is now LRU */

    /* Evict LRU — should return 1 */
    uint32_t evicted = nf::nf_lru_evict(t);
    CHECK(evicted == 1);
    CHECK(t.count == 3);
    CHECK(!nf::nf_lru_contains(t, 1));

    /* Touch new position 5 — fills the freed slot */
    nf::nf_lru_touch(t, 5);
    CHECK(t.count == 4);
    CHECK(nf::nf_lru_contains(t, 5));

    /* Touch position 10 — capacity full, auto-evicts tail (2) */
    nf::nf_lru_touch(t, 10);
    CHECK(t.count == 4);
    CHECK(!nf::nf_lru_contains(t, 2));  /* 2 was evicted */
    CHECK(nf::nf_lru_contains(t, 10));

    /* Verify ordering: head=10,5,0,3=tail */
    CHECK(nf::nf_lru_coldest(t) == 3);

    /* Evict all */
    nf::nf_lru_evict(t); /* 3 */
    nf::nf_lru_evict(t); /* 0 */
    nf::nf_lru_evict(t); /* 5 */
    nf::nf_lru_evict(t); /* 10 */
    CHECK(t.count == 0);
    CHECK(nf::nf_lru_coldest(t) == nf::NF_LRU_INVALID);

    nf::nf_lru_destroy(t);
    std::printf("  PASS\n");
}

/* ---- Test 4: INT8 quantization round-trip (F32) ---- */
static void test_int8_quantize_f32() {
    std::printf("[kv_policy] Test 4: INT8 quantize F32...\n");

    constexpr uint32_t N = 128;  /* 4 blocks of 32 */
    std::vector<float> src(N);
    for (uint32_t i = 0; i < N; ++i)
        src[i] = (float)(i - 64) * 0.1f;  /* range: -6.4 to 6.3 */

    uint32_t n_blocks = N / nf::NF_KV_Q8_BLOCK_SIZE;
    std::vector<nf::nf_kv_q8_block> blocks(n_blocks);
    nf::nf_kv_quantize_i8(src.data(), blocks.data(), N);

    std::vector<float> dst(N);
    nf::nf_kv_dequantize_i8(blocks.data(), dst.data(), N);

    /* Check round-trip error < 0.5% relative to range */
    float max_err = 0.0f;
    float range = 12.7f;  /* -6.4 to 6.3 */
    for (uint32_t i = 0; i < N; ++i) {
        float err = std::fabs(src[i] - dst[i]);
        if (err > max_err) max_err = err;
    }
    float rel_err = max_err / range;
    std::printf("  max_err=%.6f, rel_err=%.4f%%\n", max_err, rel_err * 100.0f);
    CHECK(rel_err < 0.005f);  /* < 0.5% */

    /* Edge case: all zeros */
    std::vector<float> zeros(N, 0.0f);
    nf::nf_kv_quantize_i8(zeros.data(), blocks.data(), N);
    nf::nf_kv_dequantize_i8(blocks.data(), dst.data(), N);
    for (uint32_t i = 0; i < N; ++i)
        CHECK(dst[i] == 0.0f);

    std::printf("  PASS\n");
}

/* ---- Test 5: INT8 quantization round-trip (F16) ---- */
static void test_int8_quantize_f16() {
    std::printf("[kv_policy] Test 5: INT8 quantize F16...\n");

    constexpr uint32_t N = 64;  /* 2 blocks */

    /* Generate F16 values from known F32 */
    auto f32_to_f16 = [](float val) -> uint16_t {
        uint32_t fb; std::memcpy(&fb, &val, 4);
        uint32_t sign = (fb >> 16) & 0x8000;
        int32_t exp = ((fb >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (fb >> 13) & 0x3FF;
        if (exp <= 0) return (uint16_t)sign;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
        return (uint16_t)(sign | (exp << 10) | mant);
    };

    auto f16_to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) f = sign;
        else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
        else f = sign | ((exp + 112) << 23) | (mant << 13);
        float val; std::memcpy(&val, &f, 4);
        return val;
    };

    std::vector<uint16_t> src_f16(N);
    std::vector<float> src_f32(N);
    for (uint32_t i = 0; i < N; ++i) {
        float v = (float)(i - 32) * 0.05f;
        src_f16[i] = f32_to_f16(v);
        src_f32[i] = f16_to_f32(src_f16[i]);  /* actual F16 value */
    }

    uint32_t n_blocks = N / nf::NF_KV_Q8_BLOCK_SIZE;
    std::vector<nf::nf_kv_q8_block> blocks(n_blocks);
    nf::nf_kv_quantize_i8_f16(src_f16.data(), blocks.data(), N);

    std::vector<uint16_t> dst_f16(N);
    nf::nf_kv_dequantize_i8_f16(blocks.data(), dst_f16.data(), N);

    /* Check round-trip error */
    float max_err = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        float orig = src_f32[i];
        float recon = f16_to_f32(dst_f16[i]);
        float err = std::fabs(orig - recon);
        if (err > max_err) max_err = err;
    }
    float range = 3.2f;
    float rel_err = max_err / range;
    std::printf("  max_err=%.6f, rel_err=%.4f%%\n", max_err, rel_err * 100.0f);
    CHECK(rel_err < 0.01f);  /* < 1% (F16 precision + INT8 quantization) */

    std::printf("  PASS\n");
}

int main() {
    std::printf("=== KV Cache Policy Tests (Phase 31) ===\n");
    test_none_policy();
    test_sliding_policy();
    test_lru_tracker();
    test_int8_quantize_f32();
    test_int8_quantize_f16();
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
