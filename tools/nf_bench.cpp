/**
 * @file nf_bench.cpp
 * @brief Phase 35-D: Benchmark suite — TTFT, TPS, memory profiling
 *
 * Usage: nf_bench [--iterations N] [--warmup N]
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

struct BenchResult {
    double mean_ms;
    double median_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
};

static BenchResult compute_stats(std::vector<double>& samples) {
    BenchResult r{};
    if (samples.empty()) return r;
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    r.mean_ms = sum / n;
    r.median_ms = (n % 2 == 0) ? (samples[n/2-1] + samples[n/2]) / 2.0 : samples[n/2];
    r.p99_ms = samples[(size_t)(n * 0.99)];
    r.min_ms = samples.front();
    r.max_ms = samples.back();
    double sq_sum = 0;
    for (auto s : samples) sq_sum += (s - r.mean_ms) * (s - r.mean_ms);
    r.stddev_ms = std::sqrt(sq_sum / n);
    return r;
}

struct BufPair {
    nf_buffer buf = nullptr;
    nf_buffer_ops ops{};
};

static BufPair alloc_buf(nf_provider prov, nf_provider_mem_vtable& mem_vt,
                          nf_dtype dtype, size_t size_bytes) {
    BufPair bp;
    nf_tensor_desc d{}; d.dtype = dtype; d.ndim = 1;
    d.shape[0] = size_bytes; d.size_bytes = size_bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    mem_vt.alloc(prov, &req, &bp.ops, &bp.buf);
    return bp;
}

struct PushConstants {
    uint32_t seq_len, n_heads, head_dim;
    float epsilon, theta;
    uint32_t M, N, K;
    uint32_t step_idx, max_seq_len;
    uint32_t window_size, _pad;
};

static void bench_buffer_alloc(nf_provider prov, nf_provider_mem_vtable& mem_vt,
                                int iterations) {
    std::printf("\n--- Buffer Allocation Benchmark ---\n");
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto bp = alloc_buf(prov, mem_vt, NF_DTYPE_F32, 4 * 1024 * 1024);
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (bp.buf) bp.ops.release(bp.buf);
    }
    auto r = compute_stats(samples);
    std::printf("  alloc 4MB: mean=%.3f ms, median=%.3f ms, p99=%.3f ms, stddev=%.3f ms\n",
                r.mean_ms, r.median_ms, r.p99_ms, r.stddev_ms);
}

static void bench_dispatch(nf_provider prov, nf_provider_vtable& vt,
                            nf_provider_mem_vtable& mem_vt,
                            nf::PipelineEngine& engine, int iterations) {
    std::printf("\n--- Kernel Dispatch Benchmark ---\n");
    const size_t N = 1024 * 1024;
    auto a = alloc_buf(prov, mem_vt, NF_DTYPE_F32, N * sizeof(float));
    auto b = alloc_buf(prov, mem_vt, NF_DTYPE_F32, N * sizeof(float));
    auto c = alloc_buf(prov, mem_vt, NF_DTYPE_F32, N * sizeof(float));

    { void* p; a.ops.map(a.buf, &p);
      for (size_t i = 0; i < N; ++i) ((float*)p)[i] = 1.0f;
      a.ops.unmap(a.buf); }
    { void* p; b.ops.map(b.buf, &p);
      for (size_t i = 0; i < N; ++i) ((float*)p)[i] = 2.0f;
      b.ops.unmap(b.buf); }

    /* Warmup via PipelineEngine */
    for (int i = 0; i < 3; ++i) {
        uint32_t gid = engine.create_graph();
        nf_task_desc td{};
        std::strncpy(td.op_name, "metal_vector_add", NF_MAX_OP_NAME - 1);
        td.inputs[0] = a.buf; td.input_ops[0] = a.ops;
        td.inputs[1] = b.buf; td.input_ops[1] = b.ops;
        td.n_inputs = 2;
        td.outputs[0] = c.buf; td.output_ops[0] = c.ops;
        td.n_outputs = 1;
        td.affinity = NF_AFFINITY_GPU;
        engine.add_task(gid, td);
        nf::PipelineEngine::Session sess(engine, gid);
        sess.step().get();
        vt.synchronize(prov);
        engine.destroy_graph(gid);
    }

    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        uint32_t gid = engine.create_graph();
        nf_task_desc td{};
        std::strncpy(td.op_name, "metal_vector_add", NF_MAX_OP_NAME - 1);
        td.inputs[0] = a.buf; td.input_ops[0] = a.ops;
        td.inputs[1] = b.buf; td.input_ops[1] = b.ops;
        td.n_inputs = 2;
        td.outputs[0] = c.buf; td.output_ops[0] = c.ops;
        td.n_outputs = 1;
        td.affinity = NF_AFFINITY_GPU;
        engine.add_task(gid, td);
        nf::PipelineEngine::Session sess(engine, gid);
        sess.step().get();
        vt.synchronize(prov);
        engine.destroy_graph(gid);
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    auto r = compute_stats(samples);
    double gbps = (3.0 * N * sizeof(float)) / (r.mean_ms * 1e6);
    std::printf("  vector_add (1M): mean=%.3f ms, median=%.3f ms, p99=%.3f ms (%.1f GB/s)\n",
                r.mean_ms, r.median_ms, r.p99_ms, gbps);

    a.ops.release(a.buf); b.ops.release(b.buf); c.ops.release(c.buf);
}

static void bench_matmul_size(nf_provider prov, nf_provider_vtable& vt,
                               nf_provider_mem_vtable& mem_vt,
                               nf::PipelineEngine& engine,
                               const char* op_name,
                               uint32_t M, uint32_t N, uint32_t K, int iterations) {
    auto a = alloc_buf(prov, mem_vt, NF_DTYPE_F32, M * K * sizeof(float));
    auto b = alloc_buf(prov, mem_vt, NF_DTYPE_F32, K * N * sizeof(float));
    auto c = alloc_buf(prov, mem_vt, NF_DTYPE_F32, M * N * sizeof(float));

    { void* p; a.ops.map(a.buf, &p);
      for (uint32_t i = 0; i < M * K; ++i) ((float*)p)[i] = 0.01f * (i % 100);
      a.ops.unmap(a.buf); }
    { void* p; b.ops.map(b.buf, &p);
      for (uint32_t i = 0; i < K * N; ++i) ((float*)p)[i] = 0.01f * (i % 100);
      b.ops.unmap(b.buf); }

    PushConstants pc{};
    pc.M = M; pc.N = N; pc.K = K;

    /* Warmup */
    for (int i = 0; i < 3; ++i) {
        uint32_t gid = engine.create_graph();
        nf_task_desc td{};
        std::strncpy(td.op_name, op_name, NF_MAX_OP_NAME - 1);
        td.inputs[0] = a.buf; td.input_ops[0] = a.ops;
        td.inputs[1] = b.buf; td.input_ops[1] = b.ops;
        td.n_inputs = 2;
        td.outputs[0] = c.buf; td.output_ops[0] = c.ops;
        td.n_outputs = 1;
        td.affinity = NF_AFFINITY_GPU;
        td.push_constants_size = sizeof(pc);
        std::memcpy(td.push_constants, &pc, sizeof(pc));
        engine.add_task(gid, td);
        nf::PipelineEngine::Session sess(engine, gid);
        sess.step().get();
        vt.synchronize(prov);
        engine.destroy_graph(gid);
    }

    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        uint32_t gid = engine.create_graph();
        nf_task_desc td{};
        std::strncpy(td.op_name, op_name, NF_MAX_OP_NAME - 1);
        td.inputs[0] = a.buf; td.input_ops[0] = a.ops;
        td.inputs[1] = b.buf; td.input_ops[1] = b.ops;
        td.n_inputs = 2;
        td.outputs[0] = c.buf; td.output_ops[0] = c.ops;
        td.n_outputs = 1;
        td.affinity = NF_AFFINITY_GPU;
        td.push_constants_size = sizeof(pc);
        std::memcpy(td.push_constants, &pc, sizeof(pc));
        engine.add_task(gid, td);
        nf::PipelineEngine::Session sess(engine, gid);
        sess.step().get();
        vt.synchronize(prov);
        engine.destroy_graph(gid);
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    auto r = compute_stats(samples);
    double gflops = (2.0 * M * N * K) / (r.mean_ms * 1e6);
    std::printf("  %-14s %ux%ux%u: mean=%.3f ms, median=%.3f ms, p99=%.3f ms (%.1f GFLOPS)\n",
                op_name, M, N, K, r.mean_ms, r.median_ms, r.p99_ms, gflops);

    a.ops.release(a.buf); b.ops.release(b.buf); c.ops.release(c.buf);
}

static void bench_matmul(nf_provider prov, nf_provider_vtable& vt,
                          nf_provider_mem_vtable& mem_vt,
                          nf::PipelineEngine& engine, int iterations) {
    std::printf("\n--- MatMul Benchmark ---\n");
    const uint32_t sizes[][3] = {{256,256,256}, {512,512,512}, {1024,1024,1024}};
    const char* ops[] = {"linear_tiled", "linear"};
    for (auto& op : ops) {
        for (auto& s : sizes)
            bench_matmul_size(prov, vt, mem_vt, engine, op, s[0], s[1], s[2], iterations);
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    int iterations = 50;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
    }

    std::printf("=== NeuroFabric Benchmark Suite ===\n");
    std::printf("Iterations: %d\n", iterations);

    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    if (nf_plugin_register(&vt, &prov) != NF_OK ||
        nf_plugin_register_mem(&mem_vt, &prov) != NF_OK ||
        vt.init(prov) != NF_OK) {
        std::fprintf(stderr, "Metal init failed\n");
        return 1;
    }

    nf::PipelineEngine engine(4);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);

    bench_buffer_alloc(prov, mem_vt, iterations);
    bench_dispatch(prov, vt, mem_vt, engine, iterations);
    bench_matmul(prov, vt, mem_vt, engine, iterations);

    vt.shutdown(prov);
    std::printf("\n=== Benchmark complete ===\n");
    return 0;
}
