/**
 * @file nfir_e2e_test.cpp
 * @brief Phase 12 — End-to-End: AOT .nfir → GraphBuilder → PipelineEngine → verify
 *
 * Validates the full compiler→runtime pipeline:
 *   1. Generate a 4-tensor / 2-node .nfir (matches export_nfir.py demo graph)
 *   2. Load via GraphBuilder (mmap weights, allocate activations)
 *   3. Execute DAG: weighted_add → mock_relu
 *   4. Verify bit-exact output
 *   5. Confirm weight domain == NF_MEM_DOMAIN_MMAP
 *   6. Optionally load a Python-generated .nfir (--nfir=PATH)
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_ir_format.h"
#include "neurofabric/engine/PipelineEngine.hpp"
#include "neurofabric/engine/GraphBuilder.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)

/* ================================================================== */
/*  Host Buffer — malloc-backed activation allocator                   */
/* ================================================================== */

struct HostBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data = nullptr;
    nf_tensor_desc        desc{};
    bool                  mapped = false;
};

static uint32_t host_retain(nf_buffer self) {
    return reinterpret_cast<HostBuffer*>(self)->refcount.fetch_add(
        1, std::memory_order_relaxed) + 1;
}
static uint32_t host_release(nf_buffer self) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) { std::free(b->data); delete b; }
    return prev - 1;
}
static nf_status host_map(nf_buffer self, void** p) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    b->mapped = true; *p = b->data; return NF_OK;
}
static nf_status host_unmap(nf_buffer self) {
    reinterpret_cast<HostBuffer*>(self)->mapped = false; return NF_OK;
}
static nf_status host_cache(nf_buffer, nf_cache_op, uint64_t, uint64_t) {
    return NF_OK;
}
static nf_status host_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    info->desc = b->desc; info->domain = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0; info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0; return NF_OK;
}
static nf_status host_export(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_CPU; return NF_OK;
}

static nf_buffer_ops make_host_ops() {
    nf_buffer_ops ops{};
    ops.retain = host_retain; ops.release = host_release;
    ops.map = host_map; ops.unmap = host_unmap;
    ops.cache_sync = host_cache; ops.get_info = host_info;
    ops.export_handle = host_export; ops.import_handle = nullptr;
    return ops;
}

static nf_status host_alloc_fn(const nf_tensor_desc& desc,
                                nf_buffer_ops* ops, nf_buffer* buf) {
    auto* b = new HostBuffer;
    b->desc = desc;
    b->data = std::calloc(1, desc.size_bytes);
    if (!b->data) { delete b; return NF_ERROR_OUT_OF_MEMORY; }
    *ops = make_host_ops();
    *buf = reinterpret_cast<nf_buffer>(b);
    return NF_OK;
}

/* ================================================================== */
/*  Mock Provider — weighted_add + mock_relu via cross-dylib bridge    */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    auto* desc = reinterpret_cast<const nf_task_desc*>(
        reinterpret_cast<const char*>(inputs) -
        offsetof(nf_task_desc, inputs));

    if (std::strcmp(op_name, "weighted_add") == 0 && n_in >= 2 && n_out >= 1) {
        void *w_ptr, *a_ptr, *o_ptr;
        desc->input_ops[0].map(inputs[0], &w_ptr);
        desc->input_ops[1].map(inputs[1], &a_ptr);
        desc->output_ops[0].map(outputs[0], &o_ptr);

        auto* w = static_cast<const float*>(w_ptr);
        auto* a = static_cast<const float*>(a_ptr);
        auto* o = static_cast<float*>(o_ptr);

        nf_buffer_info info{};
        desc->output_ops[0].get_info(outputs[0], &info);
        size_t count = info.desc.size_bytes / sizeof(float);

        for (size_t i = 0; i < count; ++i)
            o[i] = w[i] + a[i];

        desc->input_ops[0].unmap(inputs[0]);
        desc->input_ops[1].unmap(inputs[1]);
        desc->output_ops[0].unmap(outputs[0]);
        return NF_OK;
    }

    if (std::strcmp(op_name, "mock_relu") == 0 && n_in >= 1 && n_out >= 1) {
        void *in_ptr, *out_ptr;
        desc->input_ops[0].map(inputs[0], &in_ptr);
        desc->output_ops[0].map(outputs[0], &out_ptr);

        nf_buffer_info info{};
        desc->output_ops[0].get_info(outputs[0], &info);
        size_t count = info.desc.size_bytes / sizeof(float);

        auto* src = static_cast<const float*>(in_ptr);
        auto* dst = static_cast<float*>(out_ptr);
        for (size_t i = 0; i < count; ++i)
            dst[i] = src[i] > 0.0f ? src[i] : 0.0f;

        desc->input_ops[0].unmap(inputs[0]);
        desc->output_ops[0].unmap(outputs[0]);
        return NF_OK;
    }

    return NF_ERROR_UNSUPPORTED_OP;
}

static const char* mock_name(nf_provider) { return "mock_cpu"; }
static uint32_t mock_abi(nf_provider) { return NF_ABI_VERSION; }
static nf_status mock_init(nf_provider) { return NF_OK; }
static void mock_shutdown(nf_provider) {}
static nf_status mock_sync(nf_provider) { return NF_OK; }

static nf_provider_vtable make_mock_vt() {
    nf_provider_vtable vt{};
    vt.get_name = mock_name; vt.get_abi_version = mock_abi;
    vt.init = mock_init; vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch; vt.synchronize = mock_sync;
    return vt;
}

/* ================================================================== */
/*  .nfir Generator — 4-tensor / 2-node graph (matches Python demo)    */
/*                                                                     */
/*  Tensor 0: weight_W  (WEIGHT, N floats, w[i] = i * 0.1f)           */
/*  Tensor 1: input_A   (ACTIVATION, N floats)                         */
/*  Tensor 2: mid_B     (ACTIVATION, N floats) — weighted_add output   */
/*  Tensor 3: output_C  (ACTIVATION, N floats) — relu output           */
/*  Node 0: weighted_add(T0, T1) → T2                                  */
/*  Node 1: mock_relu(T2) → T3                                         */
/* ================================================================== */

static constexpr uint32_t N_FLOATS = 1024;
static constexpr uint32_t TENSOR_BYTES = N_FLOATS * sizeof(float);

static void generate_nfir(const char* path) {
    const uint32_t NUM_TENSORS = 4;
    const uint32_t NUM_NODES   = 2;

    size_t metadata_size = sizeof(nf_ir_header)
                         + NUM_TENSORS * sizeof(nf_ir_tensor_desc)
                         + NUM_NODES   * sizeof(nf_ir_node_desc);
    uint64_t payload_offset = (metadata_size + NF_IR_PAYLOAD_ALIGN - 1)
                            & ~(uint64_t)(NF_IR_PAYLOAD_ALIGN - 1);

    /* Header */
    nf_ir_header hdr{};
    hdr.magic          = NF_IR_MAGIC;
    hdr.version        = NF_IR_VERSION;
    hdr.num_tensors    = NUM_TENSORS;
    hdr.num_nodes      = NUM_NODES;
    hdr.payload_offset = payload_offset;
    hdr.payload_size   = TENSOR_BYTES;
    hdr.header_crc32   = 0;
    hdr._pad0          = 0;
    hdr.header_crc32   = nf_ir_header_compute_crc(&hdr);

    /* Tensor descriptors */
    nf_ir_tensor_desc tensors[NUM_TENSORS]{};

    /* T0: weight */
    tensors[0].tensor_id     = 0;
    tensors[0].dtype         = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[0].ndim          = 1;
    tensors[0].usage         = NF_IR_USAGE_WEIGHT;
    tensors[0].shape[0]      = N_FLOATS;
    tensors[0].size_bytes    = TENSOR_BYTES;
    tensors[0].weight_offset = 0;

    /* T1: activation input */
    tensors[1].tensor_id  = 1;
    tensors[1].dtype      = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[1].ndim       = 1;
    tensors[1].usage      = NF_IR_USAGE_ACTIVATION;
    tensors[1].shape[0]   = N_FLOATS;
    tensors[1].size_bytes = TENSOR_BYTES;

    /* T2: mid (activation) */
    tensors[2].tensor_id  = 2;
    tensors[2].dtype      = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[2].ndim       = 1;
    tensors[2].usage      = NF_IR_USAGE_ACTIVATION;
    tensors[2].shape[0]   = N_FLOATS;
    tensors[2].size_bytes = TENSOR_BYTES;

    /* T3: output (activation) */
    tensors[3].tensor_id  = 3;
    tensors[3].dtype      = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[3].ndim       = 1;
    tensors[3].usage      = NF_IR_USAGE_ACTIVATION;
    tensors[3].shape[0]   = N_FLOATS;
    tensors[3].size_bytes = TENSOR_BYTES;

    /* Node descriptors */
    nf_ir_node_desc nodes[NUM_NODES]{};

    /* Node 0: weighted_add(T0, T1) → T2 */
    nodes[0].node_id = 0;
    std::strncpy(nodes[0].op_type, "weighted_add", NF_MAX_OP_NAME);
    nodes[0].n_inputs  = 2;
    nodes[0].n_outputs = 1;
    nodes[0].input_tensor_ids[0]  = 0;
    nodes[0].input_tensor_ids[1]  = 1;
    nodes[0].output_tensor_ids[0] = 2;
    nodes[0].task_flags = 0;

    /* Node 1: mock_relu(T2) → T3 */
    nodes[1].node_id = 1;
    std::strncpy(nodes[1].op_type, "mock_relu", NF_MAX_OP_NAME);
    nodes[1].n_inputs  = 1;
    nodes[1].n_outputs = 1;
    nodes[1].input_tensor_ids[0]  = 2;
    nodes[1].output_tensor_ids[0] = 3;
    nodes[1].task_flags = 0;

    /* Weight payload: w[i] = i * 0.1f */
    float weights[N_FLOATS];
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        weights[i] = static_cast<float>(i) * 0.1f;

    /* Write file */
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0);

    ::write(fd, &hdr, sizeof(hdr));
    ::write(fd, tensors, sizeof(tensors));
    ::write(fd, nodes, sizeof(nodes));

    size_t written = sizeof(hdr) + sizeof(tensors) + sizeof(nodes);
    if (written < payload_offset) {
        std::vector<uint8_t> pad(static_cast<size_t>(payload_offset) - written, 0);
        ::write(fd, pad.data(), pad.size());
    }

    ::write(fd, weights, sizeof(weights));
    ::close(fd);
    std::printf("    [gen] wrote %s (%zu + %u bytes)\n",
                path, static_cast<size_t>(payload_offset), TENSOR_BYTES);
}

/* ================================================================== */
/*  Test: generate .nfir → load → execute → verify                     */
/* ================================================================== */

static void test_nfir_e2e(const char* nfir_path) {
    const bool external = (nfir_path != nullptr);
    const char* path = external ? nfir_path : "/tmp/nfir_e2e_test.nfir";

    if (!external) {
        std::printf("  [1/6] Generating .nfir (C++ generator)\n");
        generate_nfir(path);
    } else {
        std::printf("  [1/6] Using external .nfir: %s\n", path);
    }

    /* 2. Create engine + provider */
    std::printf("  [2/6] Setting up PipelineEngine + mock provider\n");
    nf::PipelineEngine engine;
    nf_provider_vtable vt = make_mock_vt();
    engine.register_provider(nullptr, vt, NF_AFFINITY_ANY);

    /* 3. Load .nfir */
    std::printf("  [3/6] Loading .nfir via GraphBuilder\n");
    nf::GraphBuilder builder(engine, host_alloc_fn);
    nf_status st = builder.load(path);
    CHECK(st == NF_OK && "GraphBuilder::load failed");

    uint32_t graph_id;
    st = builder.build(&graph_id);
    CHECK(st == NF_OK && "GraphBuilder::build failed");
    std::printf("    graph_id=%u\n", graph_id);

    /* 4. Fill activation input (tensor 1) */
    std::printf("  [4/6] Filling activation input\n");
    nf_buffer input_buf = builder.get_tensor_buffer(1);
    nf_buffer_ops input_ops = builder.get_tensor_ops(1);
    CHECK(input_buf && "tensor 1 buffer is null");

    void* input_ptr = nullptr;
    input_ops.map(input_buf, &input_ptr);
    auto* inp = static_cast<float*>(input_ptr);
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        inp[i] = static_cast<float>(i) * -0.05f;
    input_ops.unmap(input_buf);

    /* 5. Execute */
    std::printf("  [5/6] Executing DAG\n");
    auto future = engine.submit(graph_id);
    st = future.get();
    CHECK(st == NF_OK && "DAG execution failed");

    /* 6. Verify bit-exact output */
    std::printf("  [6/6] Verifying output\n");

    /*
     * Graph: weighted_add(T0, T1) → T2, mock_relu(T2) → T3
     * T0[i] = i * 0.1f,  T1[i] = i * -0.05f
     * T2[i] = T0[i] + T1[i] = i * 0.05f
     * T3[i] = relu(T2[i]) = i * 0.05f  (all >= 0)
     */
    nf_buffer output_buf = builder.get_tensor_buffer(3);
    nf_buffer_ops output_ops = builder.get_tensor_ops(3);
    CHECK(output_buf && "tensor 3 buffer is null");

    void* output_ptr = nullptr;
    output_ops.map(output_buf, &output_ptr);
    auto* out = static_cast<const float*>(output_ptr);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < N_FLOATS; ++i) {
        float expected = static_cast<float>(i) * 0.05f;
        if (std::memcmp(&out[i], &expected, sizeof(float)) != 0) {
            if (mismatches < 5)
                std::printf("    MISMATCH [%u]: got %.6f, expected %.6f\n",
                            i, out[i], expected);
            ++mismatches;
        }
    }
    output_ops.unmap(output_buf);
    CHECK(mismatches == 0 && "output verification failed");
    std::printf("    [verify] %u floats bit-exact ✓\n", N_FLOATS);

    /* Verify mid tensor (T2) also correct */
    nf_buffer mid_buf = builder.get_tensor_buffer(2);
    nf_buffer_ops mid_ops = builder.get_tensor_ops(2);
    CHECK(mid_buf);

    void* mid_ptr = nullptr;
    mid_ops.map(mid_buf, &mid_ptr);
    auto* mid = static_cast<const float*>(mid_ptr);
    for (uint32_t i = 0; i < N_FLOATS; ++i) {
        float expected = static_cast<float>(i) * 0.05f;
        CHECK(std::memcmp(&mid[i], &expected, sizeof(float)) == 0);
    }
    mid_ops.unmap(mid_buf);
    std::printf("    [verify] mid tensor (T2) bit-exact ✓\n");

    /* Verify weight domain == MMAP */
    nf_buffer weight_buf = builder.get_tensor_buffer(0);
    nf_buffer_ops weight_ops = builder.get_tensor_ops(0);
    CHECK(weight_buf);

    nf_buffer_info w_info{};
    weight_ops.get_info(weight_buf, &w_info);
    CHECK(w_info.domain == NF_MEM_DOMAIN_MMAP);
    std::printf("    [verify] weight domain = MMAP (%d) ✓\n", w_info.domain);

    /* Cleanup */
    if (!external)
        ::unlink(path);

    std::printf("  PASS: nfir_e2e (%s)\n",
                external ? "Python-generated" : "C++-generated");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(int argc, char* argv[]) {
    std::printf("nfir_e2e_test: Phase 12 — AOT Compiler E2E\n");

    /* Parse optional --nfir=PATH for Python-generated file */
    const char* ext_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--nfir=", 7) == 0)
            ext_path = argv[i] + 7;
    }

    /* Test 1: C++-generated .nfir */
    std::printf("\n--- Test 1: C++ generated .nfir ---\n");
    test_nfir_e2e(nullptr);

    /* Test 2: Python-generated .nfir (if provided) */
    if (ext_path) {
        std::printf("\n--- Test 2: Python generated .nfir ---\n");
        test_nfir_e2e(ext_path);
    }

    std::printf("\nOK: all Phase 12 E2E tests passed\n");
    return 0;
}
