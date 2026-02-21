/**
 * @file benchmark_kernels.cpp
 * @brief Phase 28: F16 vs F32 micro-benchmark harness
 *
 * For each key op, dispatches 100 iterations in F32 and F16,
 * prints JSON with avg GPU ms and speedup.
 * Built but NOT added to ctest — opt-in manual run only.
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* Metal timing API */
extern "C" void     nf_metal_enable_timing(bool);
extern "C" uint32_t nf_metal_get_timing_count();
extern "C" void     nf_metal_get_timings(char (*)[64], double*, uint32_t);

/* Plugin entry */
extern "C" nf_status nf_plugin_register(nf_provider_vtable*, nf_provider*);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable*);

static nf_provider_vtable     g_vt;
static nf_provider_mem_vtable g_mem_vt;
static nf_provider            g_prov;

static nf_buffer alloc_buf(uint32_t n_elem, nf_dtype dtype) {
    uint32_t elem_sz = (dtype == NF_DTYPE_F16) ? 2 : 4;
    nf_tensor_desc desc{};
    desc.size_bytes = n_elem * elem_sz;
    desc.dtype = dtype;
    nf_buffer buf = nullptr;
    g_vt.buffer_alloc(g_prov, &desc, &buf);
    return buf;
}

static void fill_rand_f32(nf_buffer buf, uint32_t n) {
    void* ptr = nullptr;
    g_vt.buffer_map(g_prov, buf, &ptr);
    auto* f = static_cast<float*>(ptr);
    for (uint32_t i = 0; i < n; ++i) f[i] = (float)(rand() % 1000) / 1000.0f;
    g_vt.buffer_unmap(g_prov, buf);
}

static void fill_rand_f16(nf_buffer buf, uint32_t n) {
    void* ptr = nullptr;
    g_vt.buffer_map(g_prov, buf, &ptr);
    auto* h = static_cast<uint16_t*>(ptr);
    /* Write F16 as raw uint16_t — 0x3C00 = 1.0h, approximate random */
    for (uint32_t i = 0; i < n; ++i) h[i] = 0x3000 + (uint16_t)(rand() % 0x400);
    g_vt.buffer_unmap(g_prov, buf);
}

struct PushConstants {
    uint32_t seq_len; uint32_t n_heads; uint32_t head_dim;
    float epsilon; float theta;
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t step_idx; uint32_t max_seq_len;
    uint32_t window_size; uint32_t _pad1;
};

static double bench_op(const char* op_name, nf_buffer* ins, uint32_t n_in,
                        nf_buffer* outs, uint32_t n_out, int iters) {
    /* Warmup */
    for (int i = 0; i < 5; ++i) {
        g_vt.dispatch(g_prov, op_name, ins, n_in, outs, n_out);
        g_vt.synchronize(g_prov);
    }

    nf_metal_enable_timing(true);
    for (int i = 0; i < iters; ++i) {
        g_vt.dispatch(g_prov, op_name, ins, n_in, outs, n_out);
        g_vt.synchronize(g_prov);
    }

    uint32_t count = nf_metal_get_timing_count();
    if (count == 0) return 0.0;
    char names[1024][64];
    double ms[1024];
    nf_metal_get_timings(names, ms, count);

    double total = 0.0;
    for (uint32_t i = 0; i < count; ++i) total += ms[i];
    nf_metal_enable_timing(false);
    return total / (double)count;
}

static void bench_unary(const char* label, const char* op_name, uint32_t n_elem) {
    /* F32 */
    nf_buffer in32 = alloc_buf(n_elem, NF_DTYPE_F32);
    nf_buffer out32 = alloc_buf(n_elem, NF_DTYPE_F32);
    fill_rand_f32(in32, n_elem);
    double f32_ms = bench_op(op_name, &in32, 1, &out32, 1, 100);

    /* F16 */
    nf_buffer in16 = alloc_buf(n_elem, NF_DTYPE_F16);
    nf_buffer out16 = alloc_buf(n_elem, NF_DTYPE_F16);
    fill_rand_f16(in16, n_elem);
    double f16_ms = bench_op(op_name, &in16, 1, &out16, 1, 100);

    double speedup = (f16_ms > 0.0) ? f32_ms / f16_ms : 0.0;
    std::printf("{\"op\":\"%s\",\"f32_ms\":%.4f,\"f16_ms\":%.4f,\"speedup\":%.2f}\n",
                label, f32_ms, f16_ms, speedup);

    g_vt.buffer_free(g_prov, in32);
    g_vt.buffer_free(g_prov, out32);
    g_vt.buffer_free(g_prov, in16);
    g_vt.buffer_free(g_prov, out16);
}

static void bench_binary(const char* label, const char* op_name, uint32_t n_elem) {
    /* F32 */
    nf_buffer a32 = alloc_buf(n_elem, NF_DTYPE_F32);
    nf_buffer b32 = alloc_buf(n_elem, NF_DTYPE_F32);
    nf_buffer out32 = alloc_buf(n_elem, NF_DTYPE_F32);
    fill_rand_f32(a32, n_elem); fill_rand_f32(b32, n_elem);
    nf_buffer ins32[] = {a32, b32};
    double f32_ms = bench_op(op_name, ins32, 2, &out32, 1, 100);

    /* F16 */
    nf_buffer a16 = alloc_buf(n_elem, NF_DTYPE_F16);
    nf_buffer b16 = alloc_buf(n_elem, NF_DTYPE_F16);
    nf_buffer out16 = alloc_buf(n_elem, NF_DTYPE_F16);
    fill_rand_f16(a16, n_elem); fill_rand_f16(b16, n_elem);
    nf_buffer ins16[] = {a16, b16};
    double f16_ms = bench_op(op_name, ins16, 2, &out16, 1, 100);

    double speedup = (f16_ms > 0.0) ? f32_ms / f16_ms : 0.0;
    std::printf("{\"op\":\"%s\",\"f32_ms\":%.4f,\"f16_ms\":%.4f,\"speedup\":%.2f}\n",
                label, f32_ms, f16_ms, speedup);

    g_vt.buffer_free(g_prov, a32); g_vt.buffer_free(g_prov, b32);
    g_vt.buffer_free(g_prov, out32);
    g_vt.buffer_free(g_prov, a16); g_vt.buffer_free(g_prov, b16);
    g_vt.buffer_free(g_prov, out16);
}

int main() {
    nf_plugin_register(&g_vt, &g_prov);
    nf_plugin_register_mem(&g_mem_vt);
    if (g_vt.init(g_prov) != NF_OK) {
        std::fprintf(stderr, "Metal init failed\n");
        return 1;
    }

    constexpr uint32_t N = 1024 * 1024;  /* 1M elements */
    std::printf("=== F16 vs F32 Kernel Benchmark (N=%u) ===\n", N);

    bench_unary("silu",           "silu",             N);
    bench_binary("vector_add",    "metal_vector_add", N);
    bench_binary("elementwise_mul","elementwise_mul",  N);

    g_vt.shutdown(g_prov);
    std::printf("=== Benchmark complete ===\n");
    return 0;
}
