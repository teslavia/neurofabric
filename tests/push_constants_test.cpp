/**
 * @file push_constants_test.cpp
 * @brief Phase 16 — Push Constants injection & varying seq_len verification
 *
 * Verifies:
 *   1. Session::set_push_constants() injects data into named nodes
 *   2. Mock provider reads push_constants from nf_task_desc
 *   3. Different push_constants produce different outputs
 *   4. 10-step autoregressive loop with incrementing seq_len
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_ir_format.h"
#include "neurofabric/abi/neuro_scheduler_abi.h"
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

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #expr, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

static const char* NFIR_PATH = "/tmp/nf_push_constants_test.nfir";

/* ================================================================== */
/*  Push Constants Layout (matches Metal PushConstants struct)          */
/* ================================================================== */

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
    uint32_t max_seq_len;
    uint32_t window_size;
    uint32_t _pad1;
};

static_assert(sizeof(PushConstants) <= NF_MAX_PUSH_CONSTANTS,
              "PushConstants exceeds NF_MAX_PUSH_CONSTANTS");

/* ================================================================== */
/*  Host Buffer (same pattern as session_test)                         */
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

/* PLACEHOLDER_MOCK_PROVIDER */

/* ================================================================== */
/*  Mock Provider — reads push_constants via cross-dylib bridge        */
/* ================================================================== */

/* Last-seen push constants captured by mock dispatch */
static PushConstants g_last_pc{};
static bool g_pc_received = false;

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    /* Recover nf_task_desc via offsetof bridge */
    nf_task_desc* td = reinterpret_cast<nf_task_desc*>(
        reinterpret_cast<uint8_t*>(const_cast<nf_buffer*>(inputs))
        - offsetof(nf_task_desc, inputs));

    if (td->push_constants_size >= sizeof(PushConstants)) {
        std::memcpy(&g_last_pc, td->push_constants, sizeof(PushConstants));
        g_pc_received = true;
    }

    /* Scale output by seq_len so different push_constants → different output */
    if (n_in > 0 && n_out > 0 && outputs[0]) {
        /* Use last input (activation, not weights) as source */
        uint32_t src_idx = (n_in > 1) ? (n_in - 1) : 0;
        auto* ib = reinterpret_cast<HostBuffer*>(inputs[src_idx]);
        auto* ob = reinterpret_cast<HostBuffer*>(outputs[0]);
        uint64_t n_floats = std::min(ib->desc.size_bytes, ob->desc.size_bytes)
                            / sizeof(float);
        auto* in_fp  = static_cast<float*>(ib->data);
        auto* out_fp = static_cast<float*>(ob->data);
        float scale = (td->push_constants_size > 0)
            ? static_cast<float>(g_last_pc.seq_len + 1) : 1.0f;
        for (uint64_t i = 0; i < n_floats; ++i)
            out_fp[i] = in_fp[i] * scale;
    }
    return NF_OK;
}

static const char* mock_name(nf_provider) { return "pc_test_mock"; }
static uint32_t mock_abi(nf_provider) { return NF_ABI_VERSION; }
static nf_status mock_init(nf_provider) { return NF_OK; }
static void mock_shutdown(nf_provider) {}
static nf_status mock_sync(nf_provider) { return NF_OK; }

/* PLACEHOLDER_NFIR_GEN */

/* ================================================================== */
/*  .nfir Generation — simple 2-node graph for push_constants test     */
/* ================================================================== */

static constexpr uint32_t N_FLOATS = 256;
static constexpr uint32_t NUM_TENSORS = 4;
static constexpr uint32_t NUM_NODES = 2;

static bool generate_nfir() {
    /* Tensors:
     *   T0: weights  [256] WEIGHT
     *   T1: input    [256] ACTIVATION
     *   T2: hidden   [256] ACTIVATION
     *   T3: output   [256] ACTIVATION
     *
     * Nodes:
     *   N0: "scale_op"  (T0, T1) → T2
     *   N1: "reduce_op" (T2)     → T3
     */
    float weights[N_FLOATS];
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        weights[i] = static_cast<float>(i) * 0.01f;

    /* Header */
    uint32_t metadata_size = sizeof(nf_ir_header)
        + NUM_TENSORS * sizeof(nf_ir_tensor_desc)
        + NUM_NODES * sizeof(nf_ir_node_desc);
    uint64_t payload_offset = (metadata_size + 4095u) & ~4095u;
    uint64_t payload_size = sizeof(weights);

    nf_ir_header hdr{};
    hdr.magic = NF_IR_MAGIC;
    hdr.version = NF_IR_VERSION;
    hdr.num_tensors = NUM_TENSORS;
    hdr.num_nodes = NUM_NODES;
    hdr.payload_offset = payload_offset;
    hdr.payload_size = payload_size;
    hdr.header_crc32 = nf_ir_header_compute_crc(&hdr);

    /* Tensor descriptors */
    nf_ir_tensor_desc tensors[NUM_TENSORS]{};
    // T0: weights
    tensors[0].tensor_id = 0;
    tensors[0].dtype = NF_DTYPE_F32;
    tensors[0].ndim = 1;
    tensors[0].usage = NF_IR_USAGE_WEIGHT;
    tensors[0].shape[0] = N_FLOATS;
    tensors[0].size_bytes = sizeof(weights);
    tensors[0].weight_offset = 0;
    // T1-T3: activations
    for (uint32_t i = 1; i < NUM_TENSORS; ++i) {
        tensors[i].tensor_id = i;
        tensors[i].dtype = NF_DTYPE_F32;
        tensors[i].ndim = 1;
        tensors[i].usage = NF_IR_USAGE_ACTIVATION;
        tensors[i].shape[0] = N_FLOATS;
        tensors[i].size_bytes = N_FLOATS * sizeof(float);
        tensors[i].weight_offset = 0;
    }

    /* Node descriptors */
    nf_ir_node_desc nodes[NUM_NODES]{};
    // N0: scale_op(T0, T1) → T2
    nodes[0].node_id = 0;
    std::strncpy(nodes[0].op_type, "scale_op", NF_MAX_OP_NAME - 1);
    nodes[0].n_inputs = 2;
    nodes[0].n_outputs = 1;
    nodes[0].input_tensor_ids[0] = 0;
    nodes[0].input_tensor_ids[1] = 1;
    nodes[0].output_tensor_ids[0] = 2;
    // N1: reduce_op(T2) → T3
    nodes[1].node_id = 1;
    std::strncpy(nodes[1].op_type, "reduce_op", NF_MAX_OP_NAME - 1);
    nodes[1].n_inputs = 1;
    nodes[1].n_outputs = 1;
    nodes[1].input_tensor_ids[0] = 2;
    nodes[1].output_tensor_ids[0] = 3;

    /* Write file */
    int fd = ::open(NFIR_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    ::write(fd, &hdr, sizeof(hdr));
    ::write(fd, tensors, sizeof(tensors));
    ::write(fd, nodes, sizeof(nodes));
    /* Zero-pad to payload_offset */
    size_t written = sizeof(hdr) + sizeof(tensors) + sizeof(nodes);
    size_t pad = static_cast<size_t>(payload_offset) - written;
    std::vector<uint8_t> zeros(pad, 0);
    ::write(fd, zeros.data(), zeros.size());
    ::write(fd, weights, sizeof(weights));
    ::close(fd);
    return true;
}

/* PLACEHOLDER_TESTS */

/* ================================================================== */
/*  Test 1: Push constants injection + readback                        */
/* ================================================================== */

static void test_push_constants_basic() {
    std::printf("  test_push_constants_basic...\n");

    nf::PipelineEngine engine(2);
    int mock_tag = 0;
    nf_provider_vtable vt{};
    vt.get_name = mock_name;
    vt.get_abi_version = mock_abi;
    vt.init = mock_init;
    vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch;
    vt.synchronize = mock_sync;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_tag), vt, NF_AFFINITY_ANY);

    nf::GraphBuilder builder(engine, host_alloc_fn);
    REQUIRE(builder.load(NFIR_PATH) == NF_OK);
    uint32_t gid = 0;
    REQUIRE(builder.build(&gid) == NF_OK);

    nf::PipelineEngine::Session session(engine, gid);
    REQUIRE(session.valid());

    /* Inject push constants into "scale_op" node */
    PushConstants pc{};
    pc.seq_len = 42;
    pc.n_heads = 8;
    pc.head_dim = 64;
    pc.epsilon = 1e-5f;
    pc.step_idx = 0;

    nf_status st = session.set_push_constants(
        "scale_op", &pc, sizeof(pc));
    REQUIRE(st == NF_OK);

    /* Set input data */
    nf_buffer t1_buf = builder.get_tensor_buffer(1);
    nf_buffer_ops t1_ops = builder.get_tensor_ops(1);
    REQUIRE(t1_buf != nullptr);
    void* t1_ptr = nullptr;
    REQUIRE(t1_ops.map(t1_buf, &t1_ptr) == NF_OK);
    auto* inp = static_cast<float*>(t1_ptr);
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        inp[i] = 1.0f;
    t1_ops.unmap(t1_buf);

    /* Step */
    g_pc_received = false;
    auto fut = session.step();
    REQUIRE(fut.get() == NF_OK);

    /* Verify mock provider received push constants */
    REQUIRE(g_pc_received);
    REQUIRE(g_last_pc.seq_len == 42);
    REQUIRE(g_last_pc.n_heads == 8);
    REQUIRE(g_last_pc.head_dim == 64);
    REQUIRE(std::fabs(g_last_pc.epsilon - 1e-5f) < 1e-10f);

    std::printf("    push constants received: seq_len=%u, n_heads=%u ✓\n",
                g_last_pc.seq_len, g_last_pc.n_heads);

    /* Verify NOT_FOUND for unknown node */
    st = session.set_push_constants("nonexistent_op", &pc, sizeof(pc));
    REQUIRE(st == NF_ERROR_NOT_FOUND);

    /* Verify INVALID_ARG for oversized data */
    uint8_t big[NF_MAX_PUSH_CONSTANTS + 1];
    st = session.set_push_constants("scale_op", big, sizeof(big));
    REQUIRE(st == NF_ERROR_INVALID_ARG);

    std::printf("  PASS: test_push_constants_basic\n");
}

/* ================================================================== */
/*  Test 2: Different push_constants → different outputs               */
/* ================================================================== */

static void test_push_constants_varying() {
    std::printf("  test_push_constants_varying...\n");

    nf::PipelineEngine engine(2);
    int mock_tag = 0;
    nf_provider_vtable vt{};
    vt.get_name = mock_name;
    vt.get_abi_version = mock_abi;
    vt.init = mock_init;
    vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch;
    vt.synchronize = mock_sync;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_tag), vt, NF_AFFINITY_ANY);

    nf::GraphBuilder builder(engine, host_alloc_fn);
    REQUIRE(builder.load(NFIR_PATH) == NF_OK);
    uint32_t gid = 0;
    REQUIRE(builder.build(&gid) == NF_OK);

    nf::PipelineEngine::Session session(engine, gid);
    REQUIRE(session.valid());

    /* Set input */
    nf_buffer t1_buf = builder.get_tensor_buffer(1);
    nf_buffer_ops t1_ops = builder.get_tensor_ops(1);
    void* t1_ptr = nullptr;
    REQUIRE(t1_ops.map(t1_buf, &t1_ptr) == NF_OK);
    auto* inp = static_cast<float*>(t1_ptr);
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        inp[i] = 1.0f;
    t1_ops.unmap(t1_buf);

    nf_buffer t3_buf = builder.get_tensor_buffer(3);
    nf_buffer_ops t3_ops = builder.get_tensor_ops(3);

    float prev_val = 0.0f;
    for (uint32_t seq = 1; seq <= 5; ++seq) {
        PushConstants pc{};
        pc.seq_len = seq;
        REQUIRE(session.set_push_constants("scale_op", &pc, sizeof(pc)) == NF_OK);
        REQUIRE(session.set_push_constants("reduce_op", &pc, sizeof(pc)) == NF_OK);

        auto fut = session.step();
        REQUIRE(fut.get() == NF_OK);

        void* t3_ptr = nullptr;
        REQUIRE(t3_ops.map(t3_buf, &t3_ptr) == NF_OK);
        float val = static_cast<float*>(t3_ptr)[0];
        t3_ops.unmap(t3_buf);

        /* Each seq_len should produce a different output */
        if (seq > 1) {
            REQUIRE(val != prev_val);
        }
        prev_val = val;
        std::printf("    seq_len=%u → output[0]=%.4f\n", seq, val);
    }

    std::printf("  PASS: test_push_constants_varying\n");
}

/* ================================================================== */
/*  Test 3: 10-step autoregressive with incrementing seq_len           */
/* ================================================================== */

static void test_push_constants_autoregressive() {
    std::printf("  test_push_constants_autoregressive...\n");

    nf::PipelineEngine engine(2);
    int mock_tag = 0;
    nf_provider_vtable vt{};
    vt.get_name = mock_name;
    vt.get_abi_version = mock_abi;
    vt.init = mock_init;
    vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch;
    vt.synchronize = mock_sync;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_tag), vt, NF_AFFINITY_ANY);

    nf::GraphBuilder builder(engine, host_alloc_fn);
    REQUIRE(builder.load(NFIR_PATH) == NF_OK);
    uint32_t gid = 0;
    REQUIRE(builder.build(&gid) == NF_OK);

    nf::PipelineEngine::Session session(engine, gid);
    REQUIRE(session.valid());

    /* Set input */
    nf_buffer t1_buf = builder.get_tensor_buffer(1);
    nf_buffer_ops t1_ops = builder.get_tensor_ops(1);
    void* t1_ptr = nullptr;
    REQUIRE(t1_ops.map(t1_buf, &t1_ptr) == NF_OK);
    auto* inp = static_cast<float*>(t1_ptr);
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        inp[i] = 1.0f;
    t1_ops.unmap(t1_buf);

    nf_buffer t3_buf = builder.get_tensor_buffer(3);
    nf_buffer_ops t3_ops = builder.get_tensor_ops(3);

    std::vector<float> outputs;
    for (uint32_t step = 0; step < 10; ++step) {
        PushConstants pc{};
        pc.seq_len = 128 + step;  /* incrementing context length */
        pc.step_idx = step;
        pc.n_heads = 32;
        pc.head_dim = 128;
        pc.epsilon = 1e-5f;
        pc.theta = 10000.0f;

        REQUIRE(session.set_push_constants("scale_op", &pc, sizeof(pc)) == NF_OK);
        REQUIRE(session.set_push_constants("reduce_op", &pc, sizeof(pc)) == NF_OK);

        auto fut = session.step();
        REQUIRE(fut.get() == NF_OK);

        void* t3_ptr = nullptr;
        REQUIRE(t3_ops.map(t3_buf, &t3_ptr) == NF_OK);
        float val = static_cast<float*>(t3_ptr)[0];
        t3_ops.unmap(t3_buf);

        outputs.push_back(val);
    }

    /* Verify all 10 steps produced unique outputs (different seq_len) */
    for (size_t i = 1; i < outputs.size(); ++i) {
        REQUIRE(outputs[i] != outputs[i - 1]);
    }

    std::printf("    10 steps with seq_len 128..137: all unique outputs ✓\n");
    std::printf("  PASS: test_push_constants_autoregressive\n");
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main() {
    std::printf("=== push_constants_test ===\n");

    REQUIRE(generate_nfir());
    test_push_constants_basic();
    test_push_constants_varying();
    test_push_constants_autoregressive();

    ::unlink(NFIR_PATH);

    std::printf("=== ALL 3 TESTS PASSED ===\n");
    return 0;
}
