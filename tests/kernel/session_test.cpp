/**
 * @file session_test.cpp
 * @brief Phase 15 — Session pre-compiled execution plan test
 *
 * Verifies:
 *   1. Session construction from .nfir graph
 *   2. step() returns NF_OK, output is bit-exact across iterations
 *   3. Profiling data is valid after each step
 *   4. 10 autoregressive steps complete successfully
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_ir_format.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"
#include "neuralOS/kernel/GraphBuilder.hpp"

#include <atomic>
#include <chrono>
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

/* ================================================================== */
/*  Constants — same as split_llama_mock                               */
/* ================================================================== */

static constexpr uint32_t EMBED_ROWS  = 4096;
static constexpr uint32_t EMBED_COLS  = 512;
static constexpr uint32_t SEQ_LEN     = 128;
static constexpr uint32_t HIDDEN_DIM  = 4096;
static constexpr uint32_t N_HEADS     = 32;
static constexpr uint32_t HEAD_DIM    = 128;
static constexpr uint32_t VOCAB_SIZE  = 32000;
static constexpr uint32_t NUM_TENSORS = 6;
static constexpr uint32_t NUM_NODES   = 3;

static constexpr uint64_t T0_BYTES = EMBED_ROWS * EMBED_COLS * sizeof(float);
static constexpr uint64_t T1_BYTES = SEQ_LEN * sizeof(int32_t);
static constexpr uint64_t T2_BYTES = (uint64_t)SEQ_LEN * HIDDEN_DIM * sizeof(float);
static constexpr uint64_t T3_BYTES = (uint64_t)N_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float);
static constexpr uint64_t T4_BYTES = T2_BYTES;
static constexpr uint64_t T5_BYTES = (uint64_t)SEQ_LEN * VOCAB_SIZE * sizeof(float);

/* ================================================================== */
/*  Host Buffer                                                        */
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
/*  Mock Provider                                                      */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    if (std::strcmp(op_name, "attention_prefill") == 0 && n_in >= 2 && n_out >= 2) {
        auto* ib_tok = reinterpret_cast<HostBuffer*>(inputs[1]);
        auto* ob_hid = reinterpret_cast<HostBuffer*>(outputs[0]);
        auto* ob_kv  = reinterpret_cast<HostBuffer*>(outputs[1]);
        auto* tokens = static_cast<int32_t*>(ib_tok->data);
        auto* hidden = static_cast<float*>(ob_hid->data);
        auto* kv     = static_cast<float*>(ob_kv->data);
        size_t hid_count = T2_BYTES / sizeof(float);
        for (size_t i = 0; i < hid_count; ++i)
            hidden[i] = static_cast<float>(tokens[i % SEQ_LEN]) * 0.01f;
        size_t kv_count = T3_BYTES / sizeof(float);
        for (size_t i = 0; i < kv_count; ++i)
            kv[i] = 1.0f;
        return NF_OK;
    }
    if (std::strcmp(op_name, "network_relay") == 0 && n_in >= 1 && n_out >= 1) {
        auto* ib = reinterpret_cast<HostBuffer*>(inputs[0]);
        auto* ob = reinterpret_cast<HostBuffer*>(outputs[0]);
        std::memcpy(ob->data, ib->data, T2_BYTES);
        return NF_OK;
    }
    if (std::strcmp(op_name, "decode_step") == 0 && n_in >= 1 && n_out >= 1) {
        auto* ib = reinterpret_cast<HostBuffer*>(inputs[0]);
        auto* ob = reinterpret_cast<HostBuffer*>(outputs[0]);
        auto* relay  = static_cast<float*>(ib->data);
        auto* logits = static_cast<float*>(ob->data);
        size_t relay_count = T4_BYTES / sizeof(float);
        size_t logit_count = T5_BYTES / sizeof(float);
        for (size_t i = 0; i < logit_count; ++i)
            logits[i] = std::tanh(relay[i % relay_count]);
        return NF_OK;
    }
    return NF_ERROR_UNSUPPORTED_OP;
}

static const char* mock_name(nf_provider) { return "session_mock"; }
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
/*  .nfir Generator                                                    */
/* ================================================================== */

static const char* NFIR_PATH = "/tmp/session_test.nfir";

static void make_tensor(nf_ir_tensor_desc* td, uint32_t id,
                        uint8_t dtype, uint8_t ndim, uint8_t usage,
                        const uint64_t* shape, uint64_t size_bytes,
                        uint64_t weight_offset) {
    std::memset(td, 0, sizeof(*td));
    td->tensor_id = id; td->dtype = dtype; td->ndim = ndim;
    td->usage = usage;
    for (uint8_t i = 0; i < ndim; ++i) td->shape[i] = shape[i];
    td->size_bytes = size_bytes; td->weight_offset = weight_offset;
}

static void make_node(nf_ir_node_desc* nd, uint32_t id, const char* op,
                      const uint32_t* ins, uint32_t n_in,
                      const uint32_t* outs, uint32_t n_out,
                      uint32_t flags) {
    std::memset(nd, 0, sizeof(*nd));
    nd->node_id = id;
    std::strncpy(nd->op_type, op, NF_MAX_OP_NAME - 1);
    nd->n_inputs = n_in; nd->n_outputs = n_out;
    for (uint32_t i = 0; i < n_in; ++i)  nd->input_tensor_ids[i] = ins[i];
    for (uint32_t i = 0; i < n_out; ++i) nd->output_tensor_ids[i] = outs[i];
    nd->task_flags = flags;
}

static bool generate_nfir() {
    nf_ir_header hdr{};
    hdr.magic = NF_IR_MAGIC; hdr.version = NF_IR_VERSION;
    hdr.num_tensors = NUM_TENSORS; hdr.num_nodes = NUM_NODES;

    uint64_t meta_size = sizeof(nf_ir_header)
                       + NUM_TENSORS * sizeof(nf_ir_tensor_desc)
                       + NUM_NODES * sizeof(nf_ir_node_desc);
    hdr.payload_offset = (meta_size + NF_IR_PAYLOAD_ALIGN - 1)
                       & ~(uint64_t)(NF_IR_PAYLOAD_ALIGN - 1);
    hdr.payload_size = T0_BYTES;
    hdr.header_crc32 = nf_crc32c_update(0,
        reinterpret_cast<const uint8_t*>(&hdr),
        offsetof(nf_ir_header, header_crc32));

    nf_ir_tensor_desc tensors[NUM_TENSORS];
    uint64_t s0[] = {EMBED_ROWS, EMBED_COLS};
    make_tensor(&tensors[0], 0, NF_DTYPE_F32, 2, NF_IR_USAGE_WEIGHT,
                s0, T0_BYTES, 0);
    uint64_t s1[] = {SEQ_LEN};
    make_tensor(&tensors[1], 1, NF_DTYPE_I32, 1, NF_IR_USAGE_ACTIVATION,
                s1, T1_BYTES, 0);
    uint64_t s2[] = {SEQ_LEN, HIDDEN_DIM};
    make_tensor(&tensors[2], 2, NF_DTYPE_F32, 2, NF_IR_USAGE_ACTIVATION,
                s2, T2_BYTES, 0);
    uint64_t s3[] = {N_HEADS, SEQ_LEN, HEAD_DIM};
    make_tensor(&tensors[3], 3, NF_DTYPE_F32, 3, NF_IR_USAGE_ACTIVATION,
                s3, T3_BYTES, 0);
    uint64_t s4[] = {SEQ_LEN, HIDDEN_DIM};
    make_tensor(&tensors[4], 4, NF_DTYPE_F32, 2, NF_IR_USAGE_ACTIVATION,
                s4, T4_BYTES, 0);
    uint64_t s5[] = {SEQ_LEN, VOCAB_SIZE};
    make_tensor(&tensors[5], 5, NF_DTYPE_F32, 2, NF_IR_USAGE_ACTIVATION,
                s5, T5_BYTES, 0);

    nf_ir_node_desc nodes[NUM_NODES];
    uint32_t n0_in[] = {0, 1}, n0_out[] = {2, 3};
    make_node(&nodes[0], 0, "attention_prefill", n0_in, 2, n0_out, 2, 0);
    uint32_t n1_in[] = {2, 3}, n1_out[] = {4};
    make_node(&nodes[1], 1, "network_relay", n1_in, 2, n1_out, 1, NF_TASK_REMOTE);
    uint32_t n2_in[] = {4}, n2_out[] = {5};
    make_node(&nodes[2], 2, "decode_step", n2_in, 1, n2_out, 1, 0);

    int fd = ::open(NFIR_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    ::write(fd, &hdr, sizeof(hdr));
    ::write(fd, tensors, sizeof(tensors));
    ::write(fd, nodes, sizeof(nodes));

    uint64_t written = sizeof(hdr) + sizeof(tensors) + sizeof(nodes);
    if (written < hdr.payload_offset) {
        std::vector<uint8_t> pad(hdr.payload_offset - written, 0);
        ::write(fd, pad.data(), pad.size());
    }

    std::vector<float> weights(EMBED_ROWS * EMBED_COLS);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<float>(i) * 0.001f;
    ::write(fd, weights.data(), T0_BYTES);
    ::close(fd);
    return true;
}

/* ================================================================== */
/*  Test 1: Session construction + single step                         */
/* ================================================================== */

static void test_session_basic() {
    std::printf("  test_session_basic...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int mock_tag = 0;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_tag), vt, NF_AFFINITY_ANY);

    nf::GraphBuilder gb(engine, host_alloc_fn);
    REQUIRE(gb.load(NFIR_PATH) == NF_OK);
    uint32_t gid = 0;
    REQUIRE(gb.build(&gid) == NF_OK);

    /* Set input tokens */
    nf_buffer t1_buf = gb.get_tensor_buffer(1);
    nf_buffer_ops t1_ops = gb.get_tensor_ops(1);
    REQUIRE(t1_buf != nullptr);
    void* t1_ptr = nullptr;
    REQUIRE(t1_ops.map(t1_buf, &t1_ptr) == NF_OK);
    auto* tokens = static_cast<int32_t*>(t1_ptr);
    for (uint32_t i = 0; i < SEQ_LEN; ++i) tokens[i] = static_cast<int32_t>(i + 1);
    t1_ops.unmap(t1_buf);

    nf::PipelineEngine::Session session(engine, gid);
    REQUIRE(session.valid());
    REQUIRE(session.num_nodes() == NUM_NODES);

    auto fut = session.step();
    nf_status st = fut.get();
    REQUIRE(st == NF_OK);
    std::printf("    single step OK ✓\n");

    /* Verify output T5 is non-zero */
    nf_buffer t5_buf = gb.get_tensor_buffer(5);
    nf_buffer_ops t5_ops = gb.get_tensor_ops(5);
    void* t5_ptr = nullptr;
    REQUIRE(t5_ops.map(t5_buf, &t5_ptr) == NF_OK);
    auto* logits = static_cast<float*>(t5_ptr);
    bool any_nonzero = false;
    for (size_t i = 0; i < 100; ++i) {
        if (logits[i] != 0.0f) { any_nonzero = true; break; }
    }
    REQUIRE(any_nonzero);
    t5_ops.unmap(t5_buf);
    std::printf("    output T5 non-zero ✓\n");

    std::printf("  PASS: session_basic\n");
}
/* ================================================================== */
/*  Test 2: 10 autoregressive steps — bit-exact + profiling            */
/* ================================================================== */

static void test_session_autoregressive() {
    std::printf("  test_session_autoregressive...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int mock_tag = 1;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_tag), vt, NF_AFFINITY_ANY);

    nf::GraphBuilder gb(engine, host_alloc_fn);
    REQUIRE(gb.load(NFIR_PATH) == NF_OK);
    uint32_t gid = 0;
    REQUIRE(gb.build(&gid) == NF_OK);

    /* Set input tokens */
    nf_buffer t1_buf = gb.get_tensor_buffer(1);
    nf_buffer_ops t1_ops = gb.get_tensor_ops(1);
    void* t1_ptr = nullptr;
    REQUIRE(t1_ops.map(t1_buf, &t1_ptr) == NF_OK);
    auto* tokens = static_cast<int32_t*>(t1_ptr);
    for (uint32_t i = 0; i < SEQ_LEN; ++i) tokens[i] = static_cast<int32_t>(i + 1);
    t1_ops.unmap(t1_buf);

    nf::PipelineEngine::Session session(engine, gid);
    REQUIRE(session.valid());

    /* Capture reference output from first step */
    auto fut0 = session.step();
    REQUIRE(fut0.get() == NF_OK);

    nf_buffer t5_buf = gb.get_tensor_buffer(5);
    nf_buffer_ops t5_ops = gb.get_tensor_ops(5);
    void* t5_ptr = nullptr;
    REQUIRE(t5_ops.map(t5_buf, &t5_ptr) == NF_OK);

    /* Save first 64 floats as reference */
    float ref[64];
    std::memcpy(ref, t5_ptr, sizeof(ref));
    t5_ops.unmap(t5_buf);

    auto prof0 = session.last_profile();
    REQUIRE(prof0 != nullptr);
    REQUIRE(prof0->total_us() > 0.0);
    std::printf("    step 0: %.1f us, profile valid ✓\n", prof0->total_us());

    /* Run 9 more steps */
    double total_us = 0.0;
    for (int i = 1; i < 10; ++i) {
        auto fut = session.step();
        REQUIRE(fut.get() == NF_OK);

        auto prof = session.last_profile();
        REQUIRE(prof != nullptr);
        total_us += prof->total_us();

        /* Verify bit-exact output */
        REQUIRE(t5_ops.map(t5_buf, &t5_ptr) == NF_OK);
        auto* logits = static_cast<float*>(t5_ptr);
        for (int j = 0; j < 64; ++j)
            REQUIRE(logits[j] == ref[j]);
        t5_ops.unmap(t5_buf);
    }

    double avg_us = total_us / 9.0;
    double tps = 10.0 / ((prof0->total_us() + total_us) * 1e-6);
    std::printf("    10 steps: avg %.1f us/step, %.0f TPS ✓\n", avg_us, tps);
    std::printf("  PASS: session_autoregressive\n");
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main() {
    std::printf("=== session_test ===\n");

    REQUIRE(generate_nfir());
    test_session_basic();
    test_session_autoregressive();

    ::unlink(NFIR_PATH);

    std::printf("=== ALL 2 TESTS PASSED ===\n");
    return 0;
}
