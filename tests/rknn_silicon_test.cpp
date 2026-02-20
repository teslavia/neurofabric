/**
 * @file rknn_silicon_test.cpp
 * @brief Phase 10 — Edge NPU Ignition & Sub-Graph Closure
 *
 * Two test paths:
 *   1. test_rknn_subgraph_simulation — always runs (all platforms)
 *      Exercises simulation dispatch: output = mean(input floats)
 *   2. test_rknn_subgraph_real_npu — only when NF_HAS_RKNN_SDK + model exists
 *      Exercises real NPU inference with .rknn model blob
 *
 * Uses CHECK() macro (not assert) for Release safety.
 * Links RKNN plugin directly (not dlopen) — same pattern as silicon_test.
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"
#include "neurofabric/PipelineEngine.hpp"
#include "graph/mmap_buffer.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef NF_HAS_RKNN_SDK
#include <rknn_api.h>
#endif

/* Always-execute check — works even with NDEBUG */
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

/* Declared by the RKNN plugin — linked directly */
extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static nf_provider              g_prov = nullptr;
static nf_provider_vtable       g_vt{};
static nf_provider_mem_vtable   g_mem_vt{};

static constexpr size_t N_FLOATS = 1024;
static constexpr float  TOLERANCE = 1e-5f;

/* ================================================================== */
/*  Helper: allocate buffer via RKNN provider mem vtable                */
/* ================================================================== */

static nf_buffer alloc_buf(const nf_tensor_desc& desc, nf_buffer_ops* ops) {
    nf_buffer buf = nullptr;
    nf_buffer_alloc_request req{};
    req.desc = desc;
    req.preferred = NF_MEM_DOMAIN_DMA_BUF;
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
/*  Helper: create a fake .rknn model blob as a temp file               */
/* ================================================================== */

static const char* FAKE_MODEL_PATH = "/tmp/nf_test_fake_model.rknn";

static void create_fake_model_file(size_t size) {
    int fd = ::open(FAKE_MODEL_PATH, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    CHECK(fd >= 0);
    /* Fill with a recognizable pattern */
    std::vector<uint8_t> data(size, 0xAB);
    /* Write "RKNN" magic at offset 0 for realism */
    if (size >= 4) {
        data[0] = 'R'; data[1] = 'K'; data[2] = 'N'; data[3] = 'N';
    }
    ssize_t written = ::write(fd, data.data(), data.size());
    CHECK(written == static_cast<ssize_t>(size));
    ::close(fd);
}

/* ================================================================== */
/*  Test 1: rknn_subgraph simulation — deterministic mean-reduction     */
/* ================================================================== */

static void test_rknn_subgraph_simulation() {
    std::printf("  test_rknn_subgraph_simulation...\n");

    /* 1. Create fake model blob file + mmap it */
    const size_t model_size = 4096;
    create_fake_model_file(model_size);

    int model_fd = ::open(FAKE_MODEL_PATH, O_RDONLY);
    CHECK(model_fd >= 0);

    nf_tensor_desc model_desc{};
    model_desc.dtype = NF_DTYPE_U8;
    model_desc.ndim = 1;
    model_desc.shape[0] = model_size;
    model_desc.size_bytes = model_size;

    nf_buffer_ops model_ops{};
    nf_buffer model_buf = nullptr;
    nf_status st = nf::mmap_buffer_create(model_fd, 0, model_size,
                                           model_desc, &model_ops, &model_buf);
    CHECK_OK(st);
    ::close(model_fd);

    /* 2. Allocate feature input buffer (1024 floats) */
    nf_tensor_desc feat_desc = make_desc(N_FLOATS);
    nf_buffer_ops feat_ops{};
    nf_buffer feat_buf = alloc_buf(feat_desc, &feat_ops);

    /* Fill with deterministic pattern: i * 0.1f */
    void* feat_ptr = nullptr;
    CHECK_OK(feat_ops.map(feat_buf, &feat_ptr));
    auto* fp = static_cast<float*>(feat_ptr);
    double expected_sum = 0.0;
    for (size_t i = 0; i < N_FLOATS; ++i) {
        fp[i] = static_cast<float>(i) * 0.1f;
        expected_sum += static_cast<double>(fp[i]);
    }
    CHECK_OK(feat_ops.unmap(feat_buf));
    float expected_mean = static_cast<float>(expected_sum / static_cast<double>(N_FLOATS));

    /* 3. Allocate output buffer */
    nf_tensor_desc out_desc = make_desc(N_FLOATS);
    nf_buffer_ops out_ops{};
    nf_buffer out_buf = alloc_buf(out_desc, &out_ops);

    /* 4. Build DAG: single "rknn_subgraph" node */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_NPU);

    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "rknn_subgraph", NF_MAX_OP_NAME - 1);

    /* inputs[0] = model blob, inputs[1] = feature data */
    td.inputs[0]    = model_buf;
    td.input_ops[0] = model_ops;
    td.inputs[1]    = feat_buf;
    td.input_ops[1] = feat_ops;
    td.n_inputs     = 2;

    td.outputs[0]    = out_buf;
    td.output_ops[0] = out_ops;
    td.n_outputs     = 1;
    td.affinity      = NF_AFFINITY_NPU;

    engine.add_task(gid, td);
    auto future = engine.submit(gid);
    st = future.get();
    CHECK_OK(st);

    /* 5. Verify output matches simulation formula: mean(input_floats) */
    void* out_ptr = nullptr;
    CHECK_OK(out_ops.map(out_buf, &out_ptr));
    auto* out_fp = static_cast<float*>(out_ptr);

    size_t mismatches = 0;
    for (size_t i = 0; i < N_FLOATS; ++i) {
        float diff = std::fabs(out_fp[i] - expected_mean);
        if (diff > TOLERANCE) {
            if (mismatches < 5) {
                std::printf("    MISMATCH [%zu]: got %.8f, expected %.8f\n",
                            i, out_fp[i], expected_mean);
            }
            ++mismatches;
        }
    }
    CHECK_OK(out_ops.unmap(out_buf));
    CHECK(mismatches == 0);
    std::printf("    %zu floats verified (mean=%.6f)\n", N_FLOATS, expected_mean);

    /* 6. Cleanup */
    out_ops.release(out_buf);
    feat_ops.release(feat_buf);
    model_ops.release(model_buf);
    engine.destroy_graph(gid);
    ::unlink(FAKE_MODEL_PATH);

    std::printf("  PASS: rknn_subgraph simulation\n");
}

/* ================================================================== */
/*  Test 2: rknn_subgraph via .nfir-style mmap (second simulation)      */
/*  Verifies model blob domain == NF_MEM_DOMAIN_MMAP                    */
/* ================================================================== */

static void test_rknn_subgraph_via_nfir() {
    std::printf("  test_rknn_subgraph_via_nfir...\n");

    /* Create a larger fake model to exercise mmap page alignment */
    const size_t model_size = 16384;
    create_fake_model_file(model_size);

    int model_fd = ::open(FAKE_MODEL_PATH, O_RDONLY);
    CHECK(model_fd >= 0);

    nf_tensor_desc model_desc{};
    model_desc.dtype = NF_DTYPE_U8;
    model_desc.ndim = 1;
    model_desc.shape[0] = model_size;
    model_desc.size_bytes = model_size;

    nf_buffer_ops model_ops{};
    nf_buffer model_buf = nullptr;
    nf_status st = nf::mmap_buffer_create(model_fd, 0, model_size,
                                           model_desc, &model_ops, &model_buf);
    CHECK_OK(st);
    ::close(model_fd);

    /* Verify mmap domain */
    nf_buffer_info m_info{};
    model_ops.get_info(model_buf, &m_info);
    CHECK(m_info.domain == NF_MEM_DOMAIN_MMAP);
    std::printf("    model domain = MMAP (%d)\n", m_info.domain);

    /* Verify model blob content is accessible */
    void* m_ptr = nullptr;
    CHECK_OK(model_ops.map(model_buf, &m_ptr));
    auto* bytes = static_cast<const uint8_t*>(m_ptr);
    CHECK(bytes[0] == 'R' && bytes[1] == 'K' && bytes[2] == 'N' && bytes[3] == 'N');
    CHECK_OK(model_ops.unmap(model_buf));

    /* Allocate feature input: 512 floats, all 2.0f */
    const size_t N2 = 512;
    nf_tensor_desc feat_desc = make_desc(N2);
    nf_buffer_ops feat_ops{};
    nf_buffer feat_buf = alloc_buf(feat_desc, &feat_ops);

    void* feat_ptr = nullptr;
    CHECK_OK(feat_ops.map(feat_buf, &feat_ptr));
    auto* fp = static_cast<float*>(feat_ptr);
    for (size_t i = 0; i < N2; ++i) fp[i] = 2.0f;
    CHECK_OK(feat_ops.unmap(feat_buf));

    /* Allocate output: 512 floats */
    nf_tensor_desc out_desc = make_desc(N2);
    nf_buffer_ops out_ops{};
    nf_buffer out_buf = alloc_buf(out_desc, &out_ops);

    /* Build DAG */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_NPU);

    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "rknn_subgraph", NF_MAX_OP_NAME - 1);
    td.inputs[0]    = model_buf;
    td.input_ops[0] = model_ops;
    td.inputs[1]    = feat_buf;
    td.input_ops[1] = feat_ops;
    td.n_inputs     = 2;
    td.outputs[0]    = out_buf;
    td.output_ops[0] = out_ops;
    td.n_outputs     = 1;
    td.affinity      = NF_AFFINITY_NPU;

    engine.add_task(gid, td);
    auto future = engine.submit(gid);
    st = future.get();
    CHECK_OK(st);

    /* Verify: all outputs should be 2.0f (mean of all-2.0 input) */
    void* out_ptr = nullptr;
    CHECK_OK(out_ops.map(out_buf, &out_ptr));
    auto* out_fp = static_cast<float*>(out_ptr);
    size_t mismatches = 0;
    for (size_t i = 0; i < N2; ++i) {
        if (std::fabs(out_fp[i] - 2.0f) > TOLERANCE) {
            if (mismatches < 5)
                std::printf("    MISMATCH [%zu]: got %.8f, expected 2.0\n",
                            i, out_fp[i]);
            ++mismatches;
        }
    }
    CHECK_OK(out_ops.unmap(out_buf));
    CHECK(mismatches == 0);
    std::printf("    %zu floats verified (mean=2.000000)\n", N2);

    /* Cleanup */
    out_ops.release(out_buf);
    feat_ops.release(feat_buf);
    model_ops.release(model_buf);
    engine.destroy_graph(gid);
    ::unlink(FAKE_MODEL_PATH);

    std::printf("  PASS: rknn_subgraph via nfir mmap\n");
}

/* ================================================================== */
/*  Test 3: Real NPU dispatch (NF_HAS_RKNN_SDK + model file guard)     */
/* ================================================================== */

#ifdef NF_HAS_RKNN_SDK
#include <chrono>

static void test_rknn_subgraph_real_npu() {
    std::printf("  test_rknn_subgraph_real_npu...\n");

    /* Check for model file — skip gracefully if not present */
    const char* model_path = "./yolov5s-640-640.rknn";
    struct stat sb;
    if (::stat(model_path, &sb) != 0) {
        std::printf("  [ SKIPPED ] model not found: %s\n", model_path);
        return;
    }
    std::printf("    model: %s (%.1f KB)\n", model_path,
                static_cast<double>(sb.st_size) / 1024.0);

    /* Mmap model blob */
    int model_fd = ::open(model_path, O_RDONLY);
    CHECK(model_fd >= 0);

    nf_tensor_desc model_desc{};
    model_desc.dtype = NF_DTYPE_U8;
    model_desc.ndim = 1;
    model_desc.shape[0] = static_cast<uint64_t>(sb.st_size);
    model_desc.size_bytes = static_cast<uint64_t>(sb.st_size);

    nf_buffer_ops model_ops{};
    nf_buffer model_buf = nullptr;
    nf_status st = nf::mmap_buffer_create(model_fd, 0,
                                           static_cast<uint64_t>(sb.st_size),
                                           model_desc, &model_ops, &model_buf);
    CHECK_OK(st);
    ::close(model_fd);

    /* Query model to determine input/output sizes */
    void* m_ptr = nullptr;
    CHECK_OK(model_ops.map(model_buf, &m_ptr));

    rknn_context probe_ctx = 0;
    int ret = rknn_init(&probe_ctx, m_ptr, static_cast<uint32_t>(sb.st_size), 0, nullptr);
    CHECK(ret == 0);

    rknn_input_output_num io_num{};
    rknn_query(probe_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    std::printf("    model inputs: %u, outputs: %u\n",
                io_num.n_input, io_num.n_output);

    /* Allocate input buffers matching model expectations */
    std::vector<nf_buffer> in_bufs;
    std::vector<nf_buffer_ops> in_ops_vec;
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        rknn_query(probe_ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        std::printf("    input[%u]: size=%u\n", i, attr.size);

        nf_tensor_desc d{};
        d.dtype = NF_DTYPE_U8;
        d.ndim = 1;
        d.shape[0] = attr.size;
        d.size_bytes = attr.size;

        nf_buffer_ops ops{};
        nf_buffer buf = alloc_buf(d, &ops);

        /* Fill with deterministic test pattern */
        void* ptr = nullptr;
        CHECK_OK(ops.map(buf, &ptr));
        std::memset(ptr, 128, attr.size);
        CHECK_OK(ops.unmap(buf));

        in_bufs.push_back(buf);
        in_ops_vec.push_back(ops);
    }

    /* Allocate output buffers */
    std::vector<nf_buffer> out_bufs;
    std::vector<nf_buffer_ops> out_ops_vec;
    for (uint32_t j = 0; j < io_num.n_output; ++j) {
        rknn_tensor_attr attr{};
        attr.index = j;
        rknn_query(probe_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        std::printf("    output[%u]: size=%u\n", j, attr.size);

        nf_tensor_desc d{};
        d.dtype = NF_DTYPE_U8;
        d.ndim = 1;
        d.shape[0] = attr.size;
        d.size_bytes = attr.size;

        nf_buffer_ops ops{};
        nf_buffer buf = alloc_buf(d, &ops);
        out_bufs.push_back(buf);
        out_ops_vec.push_back(ops);
    }

    rknn_destroy(probe_ctx);
    CHECK_OK(model_ops.unmap(model_buf));

    /* Build DAG: rknn_subgraph with model + inputs → outputs */
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_NPU);

    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "rknn_subgraph", NF_MAX_OP_NAME - 1);

    td.inputs[0]    = model_buf;
    td.input_ops[0] = model_ops;
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        td.inputs[i + 1]    = in_bufs[i];
        td.input_ops[i + 1] = in_ops_vec[i];
    }
    td.n_inputs = 1 + io_num.n_input;

    for (uint32_t j = 0; j < io_num.n_output; ++j) {
        td.outputs[j]    = out_bufs[j];
        td.output_ops[j] = out_ops_vec[j];
    }
    td.n_outputs = io_num.n_output;
    td.affinity  = NF_AFFINITY_NPU;

    engine.add_task(gid, td);

    auto t0 = std::chrono::steady_clock::now();
    auto future = engine.submit(gid);
    st = future.get();
    auto t1 = std::chrono::steady_clock::now();
    CHECK_OK(st);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("    NPU inference: %.2f ms\n", ms);

    /* Verify: outputs are non-zero (NPU actually computed something) */
    for (uint32_t j = 0; j < io_num.n_output; ++j) {
        void* ptr = nullptr;
        CHECK_OK(out_ops_vec[j].map(out_bufs[j], &ptr));
        auto* bytes = static_cast<const uint8_t*>(ptr);
        nf_buffer_info info{};
        out_ops_vec[j].get_info(out_bufs[j], &info);
        bool all_zero = true;
        for (size_t k = 0; k < info.desc.size_bytes && k < 1024; ++k) {
            if (bytes[k] != 0) { all_zero = false; break; }
        }
        CHECK_OK(out_ops_vec[j].unmap(out_bufs[j]));
        CHECK(!all_zero);
        std::printf("    output[%u]: non-zero verified\n", j);
    }

    /* Cleanup */
    for (uint32_t j = 0; j < io_num.n_output; ++j)
        out_ops_vec[j].release(out_bufs[j]);
    for (uint32_t i = 0; i < io_num.n_input; ++i)
        in_ops_vec[i].release(in_bufs[i]);
    model_ops.release(model_buf);
    engine.destroy_graph(gid);

    std::printf("  PASS: rknn_subgraph real NPU\n");
}
#else
static void test_rknn_subgraph_real_npu() {
    /* No SDK — nothing to run */
}
#endif

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("rknn_silicon_test: Phase 10 — Edge NPU Ignition\n");

    /* Register + init provider */
    nf_status st = nf_plugin_register(&g_vt, &g_prov);
    CHECK_OK(st);
    st = nf_plugin_register_mem(&g_mem_vt);
    CHECK_OK(st);
    st = g_vt.init(g_prov);
    CHECK_OK(st);

    std::printf("  Provider: %s\n", g_vt.get_name(g_prov));

    /* Simulation tests always run */
    test_rknn_subgraph_simulation();
    test_rknn_subgraph_via_nfir();

    /* Real NPU test — guarded by SDK + model file */
    test_rknn_subgraph_real_npu();

    g_vt.shutdown(g_prov);
    std::printf("OK: all Phase 10 RKNN tests passed\n");
    return 0;
}
