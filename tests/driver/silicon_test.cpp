/**
 * @file silicon_test.cpp
 * @brief Phase 9 — Real GPU Silicon Verification
 *
 * Exercises real Metal GPU compute through the full Neuro-Fabric stack:
 *   1. Load Metal plugin via nf_plugin_register (direct call)
 *   2. Init provider → creates real MTLDevice + shader pipelines
 *   3. Allocate buffers via provider vtable (real MTLBuffer)
 *   4. Build DAG via PipelineEngine, submit, wait on future
 *   5. Verify GPU-computed results with float tolerance
 *
 * Apple Silicon only — guarded by NF_PLUGIN_METAL in CMake.
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Always-execute check — works even with NDEBUG */
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

/* Declared by the Metal plugin — linked directly */
extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static constexpr size_t N_FLOATS = 1024;
static constexpr float  TOLERANCE = 1e-4f;

/* ================================================================== */
/*  Helper: make buffer_ops from the Metal plugin's mem vtable          */
/* ================================================================== */

static nf_provider          g_prov = nullptr;
static nf_provider_vtable   g_vt{};
static nf_provider_mem_vtable g_mem_vt{};

static nf_buffer alloc_buf(const nf_tensor_desc& desc, nf_buffer_ops* ops) {
    nf_buffer buf = nullptr;
    nf_buffer_alloc_request req{};
    req.desc = desc;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    nf_status st = g_mem_vt.alloc(g_prov, &req, ops, &buf);
    CHECK_OK(st);
    return buf;
}

static nf_tensor_desc make_desc(size_t n_floats) {
    nf_tensor_desc d{};
    d.dtype = NF_DTYPE_F32;
    d.ndim = 1;
    d.shape[0] = n_floats;
    d.size_bytes = n_floats * sizeof(float);
    return d;
}
/* ================================================================== */
/*  Test 1: metal_vector_add — real GPU dispatch                        */
/* ================================================================== */

static void test_metal_vector_add() {
    std::printf("  test_metal_vector_add...\n");

    nf_tensor_desc desc = make_desc(N_FLOATS);
    nf_buffer_ops a_ops{}, b_ops{}, out_ops{};
    nf_buffer a_buf   = alloc_buf(desc, &a_ops);
    nf_buffer b_buf   = alloc_buf(desc, &b_ops);
    nf_buffer out_buf = alloc_buf(desc, &out_ops);

    /* Fill A and B with deterministic patterns */
    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    CHECK_OK(a_ops.map(a_buf, &a_ptr));
    CHECK_OK(b_ops.map(b_buf, &b_ptr));
    auto* a_fp = static_cast<float*>(a_ptr);
    auto* b_fp = static_cast<float*>(b_ptr);
    for (size_t i = 0; i < N_FLOATS; ++i) {
        a_fp[i] = static_cast<float>(i) * 0.1f;
        b_fp[i] = static_cast<float>(i) * 0.2f;
    }
    CHECK_OK(a_ops.unmap(a_buf));
    CHECK_OK(b_ops.unmap(b_buf));

    /* Build DAG: single node metal_vector_add(A, B) → Out */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "metal_vector_add", NF_MAX_OP_NAME - 1);
    td.inputs[0]    = a_buf;
    td.input_ops[0] = a_ops;
    td.inputs[1]    = b_buf;
    td.input_ops[1] = b_ops;
    td.n_inputs     = 2;
    td.outputs[0]    = out_buf;
    td.output_ops[0] = out_ops;
    td.n_outputs     = 1;
    td.affinity      = NF_AFFINITY_GPU;

    engine.add_task(gid, td);
    auto future = engine.submit(gid);
    nf_status st = future.get();
    CHECK_OK(st);

    /* Wait for GPU fence via cache_sync */
    out_ops.cache_sync(out_buf, NF_CACHE_INVALIDATE, 0, 0);

    /* Verify results with float tolerance */
    void* out_ptr = nullptr;
    CHECK_OK(out_ops.map(out_buf, &out_ptr));
    auto* out_fp = static_cast<float*>(out_ptr);
    size_t mismatches = 0;
    for (size_t i = 0; i < N_FLOATS; ++i) {
        float a_val = static_cast<float>(i) * 0.1f;
        float b_val = static_cast<float>(i) * 0.2f;
        float expected = a_val + b_val;
        float diff = std::fabs(out_fp[i] - expected);
        float rel = (std::fabs(expected) > 1.0f) ? diff / std::fabs(expected) : diff;
        if (rel > TOLERANCE) {
            if (mismatches < 5) {
                std::printf("    MISMATCH [%zu]: got %.8f, expected %.8f\n",
                            i, out_fp[i], expected);
            }
            ++mismatches;
        }
    }
    CHECK_OK(out_ops.unmap(out_buf));
    CHECK(mismatches == 0);
    std::printf("    %zu floats verified (tolerance %.0e) ✓\n", N_FLOATS, TOLERANCE);

    /* Cleanup */
    out_ops.release(out_buf);
    b_ops.release(b_buf);
    a_ops.release(a_buf);
    engine.destroy_graph(gid);
}
/* ================================================================== */
/*  Test 2: attention_prefill — async GPU fence                         */
/* ================================================================== */

static void test_metal_attention_prefill() {
    std::printf("  test_metal_attention_prefill...\n");

    nf_tensor_desc desc = make_desc(N_FLOATS);
    nf_buffer_ops in_ops{}, k_ops{}, v_ops{};
    nf_buffer in_buf = alloc_buf(desc, &in_ops);
    nf_buffer k_buf  = alloc_buf(desc, &k_ops);
    nf_buffer v_buf  = alloc_buf(desc, &v_ops);

    /* Fill input */
    void* in_ptr = nullptr;
    CHECK_OK(in_ops.map(in_buf, &in_ptr));
    auto* in_fp = static_cast<float*>(in_ptr);
    for (size_t i = 0; i < N_FLOATS; ++i) {
        in_fp[i] = static_cast<float>(i) * 0.3f;
    }
    CHECK_OK(in_ops.unmap(in_buf));

    /* Build DAG: attention_prefill(input) → K, V */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);

    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "attention_prefill", NF_MAX_OP_NAME - 1);
    td.inputs[0]     = in_buf;
    td.input_ops[0]  = in_ops;
    td.n_inputs      = 1;
    td.outputs[0]    = k_buf;
    td.output_ops[0] = k_ops;
    td.outputs[1]    = v_buf;
    td.output_ops[1] = v_ops;
    td.n_outputs     = 2;
    td.affinity      = NF_AFFINITY_GPU;

    engine.add_task(gid, td);
    auto future = engine.submit(gid);
    nf_status st = future.get();
    CHECK_OK(st);

    /* Wait for GPU fences */
    k_ops.cache_sync(k_buf, NF_CACHE_INVALIDATE, 0, 0);
    v_ops.cache_sync(v_buf, NF_CACHE_INVALIDATE, 0, 0);

    /* Verify K = input * 0.5 */
    void* k_ptr = nullptr;
    CHECK_OK(k_ops.map(k_buf, &k_ptr));
    auto* k_fp = static_cast<float*>(k_ptr);
    size_t k_mismatches = 0;
    for (size_t i = 0; i < N_FLOATS; ++i) {
        float expected = static_cast<float>(i) * 0.3f * 0.5f;
        if (std::fabs(k_fp[i] - expected) > TOLERANCE) {
            if (k_mismatches < 5) {
                std::printf("    K MISMATCH [%zu]: got %.8f, expected %.8f\n",
                            i, k_fp[i], expected);
            }
            ++k_mismatches;
        }
    }
    CHECK_OK(k_ops.unmap(k_buf));
    CHECK(k_mismatches == 0);
    std::printf("    K: %zu floats verified ✓\n", N_FLOATS);

    /* Verify V = input * -0.25 */
    void* v_ptr = nullptr;
    CHECK_OK(v_ops.map(v_buf, &v_ptr));
    auto* v_fp = static_cast<float*>(v_ptr);
    size_t v_mismatches = 0;
    for (size_t i = 0; i < N_FLOATS; ++i) {
        float expected = static_cast<float>(i) * 0.3f * -0.25f;
        if (std::fabs(v_fp[i] - expected) > TOLERANCE) {
            if (v_mismatches < 5) {
                std::printf("    V MISMATCH [%zu]: got %.8f, expected %.8f\n",
                            i, v_fp[i], expected);
            }
            ++v_mismatches;
        }
    }
    CHECK_OK(v_ops.unmap(v_buf));
    CHECK(v_mismatches == 0);
    std::printf("    V: %zu floats verified ✓\n", N_FLOATS);

    /* Cleanup */
    v_ops.release(v_buf);
    k_ops.release(k_buf);
    in_ops.release(in_buf);
    engine.destroy_graph(gid);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("silicon_test: Phase 9 — Physical Silicon Ignition\n");

    /* Register Metal plugin */
    nf_status st = nf_plugin_register(&g_vt, &g_prov);
    CHECK_OK(st);
    st = nf_plugin_register_mem(&g_mem_vt);
    CHECK_OK(st);

    /* Init provider — creates real MTLDevice + shader pipelines */
    st = g_vt.init(g_prov);
    CHECK_OK(st);

    /* Print GPU device name for confirmation */
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    test_metal_vector_add();
    test_metal_attention_prefill();

    /* Shutdown */
    g_vt.shutdown(g_prov);

    std::printf("OK: all Phase 9 silicon tests passed\n");
    return 0;
}
