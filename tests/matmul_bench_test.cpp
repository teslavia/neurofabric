/**
 * @file matmul_bench_test.cpp
 * @brief MatMul kernel benchmark: linear (auto-simd) vs linear_tiled
 *
 * Phase 24: MatMul Optimization.
 *
 * Benchmarks matmul kernels at 512², 1024², 2048² and reports GFLOPS.
 * Uses PipelineEngine for proper push constant delivery.
 * Requires Metal plugin (macOS only).
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/engine/PipelineEngine.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #expr, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

static nf_provider g_prov;
static nf_provider_vtable g_vt;
static nf_provider_mem_vtable g_mem_vt;

struct BufPair {
    nf_buffer buf = nullptr;
    nf_buffer_ops ops{};
};

static BufPair alloc_buf(size_t bytes) {
    BufPair bp;
    nf_tensor_desc d{}; d.dtype = NF_DTYPE_F32; d.ndim = 1;
    d.shape[0] = bytes; d.size_bytes = bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    g_mem_vt.alloc(g_prov, &req, &bp.ops, &bp.buf);
/* PLACEHOLDER_ALLOC_END */
    return bp;
}

static void fill_random(BufPair& bp, size_t n_floats) {
    void* p; bp.ops.map(bp.buf, &p);
    float* f = (float*)p;
    for (size_t i = 0; i < n_floats; ++i)
        f[i] = ((float)(i % 1000) - 500.0f) * 0.001f;
    bp.ops.unmap(bp.buf);
}

struct PushConstants {
    uint32_t seq_len; uint32_t n_heads; uint32_t head_dim;
    float epsilon; float theta;
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t step_idx; uint32_t max_seq_len;
    uint32_t window_size; uint32_t _pad1;
};

static double bench_op(nf::PipelineEngine& engine, const char* op_name,
                       BufPair& A, BufPair& B, BufPair& C,
                       uint32_t M, uint32_t N, uint32_t K,
                       int warmup, int iters) {
    PushConstants pc{};
    pc.M = M; pc.N = N; pc.K = K;

    auto run_once = [&]() {
        uint32_t gid = engine.create_graph();
        nf_task_desc td{};
        std::strncpy(td.op_name, op_name, NF_MAX_OP_NAME - 1);
        td.inputs[0] = A.buf; td.input_ops[0] = A.ops;
        td.inputs[1] = B.buf; td.input_ops[1] = B.ops;
        td.n_inputs = 2;
        td.outputs[0] = C.buf; td.output_ops[0] = C.ops;
        td.n_outputs = 1;
        td.affinity = NF_AFFINITY_GPU;
        uint32_t nid = engine.add_task(gid, td);

        nf::PipelineEngine::Session sess(engine, gid);
        sess.set_push_constants_by_id(nid, &pc, sizeof(pc));
        auto fut = sess.step();
        fut.get();
        engine.destroy_graph(gid);
    };

    /* Warmup */
    for (int i = 0; i < warmup; ++i) run_once();
    g_vt.synchronize(g_prov);

    /* Timed runs */
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) run_once();
    g_vt.synchronize(g_prov);
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = ms / iters;
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms * 1e-3)) / 1e9;
    return gflops;
}

static void bench_size(nf::PipelineEngine& engine, uint32_t dim) {
    size_t bytes = (size_t)dim * dim * sizeof(float);
    auto A = alloc_buf(bytes);
    auto B = alloc_buf(bytes);
    auto C = alloc_buf(bytes);
    CHECK(A.buf && B.buf && C.buf);

    fill_random(A, (size_t)dim * dim);
    fill_random(B, (size_t)dim * dim);

    int warmup = 2, iters = (dim <= 512) ? 10 : 5;

    std::printf("[matmul_bench] %ux%u x %u:\n", dim, dim, dim);

    double gf_auto  = bench_op(engine, "linear",       A, B, C, dim, dim, dim, warmup, iters);
    double gf_tiled = bench_op(engine, "linear_tiled",  A, B, C, dim, dim, dim, warmup, iters);

    std::printf("  linear (auto):  %7.2f GFLOPS\n", gf_auto);
    std::printf("  linear_tiled:   %7.2f GFLOPS\n", gf_tiled);

    if (gf_auto > 0 && gf_tiled > 0)
        std::printf("  speedup (auto vs tiled): %.1fx\n", gf_auto / gf_tiled);

    /* Verify correctness */
    {
        void* p; C.ops.map(C.buf, &p);
        float* f = (float*)p;
        bool ok = true;
        for (uint32_t i = 0; i < 100 && i < dim * dim; ++i)
            if (!std::isfinite(f[i])) { ok = false; break; }
        CHECK(ok);
        C.ops.unmap(C.buf);
    }

    A.ops.release(A.buf);
    B.ops.release(B.buf);
    C.ops.release(C.buf);
}

int main() {
    CHECK(nf_plugin_register(&g_vt, &g_prov) == NF_OK);
    CHECK(nf_plugin_register_mem(&g_mem_vt, &g_prov) == NF_OK);
    CHECK(g_vt.init(g_prov) == NF_OK);

    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    std::printf("=== MatMul Benchmark ===\n");

    bench_size(engine, 512);
    bench_size(engine, 1024);
    bench_size(engine, 2048);

    g_vt.shutdown(g_prov);
    std::printf("=== MatMul Benchmark PASSED ===\n");
    return 0;
}
