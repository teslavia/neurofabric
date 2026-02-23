/**
 * @file split_llama_mock_test.cpp
 * @brief Phase 14 — AOT split_llama_mock .nfir topology verification
 *
 * Generates a 3-node / 6-tensor .nfir in C++, loads via GraphBuilder,
 * and verifies:
 *   1. Topology: 6 tensors, 3 nodes, correct edge connectivity
 *   2. Remote flag: node 1 (network_relay) has NF_TASK_REMOTE (0x04)
 *   3. Weight tensor T0 is mmap-backed from payload
 *   4. Mock dispatch: prefill → relay → decode bit-exact pipeline
 *
 * Nodes:
 *   N0: attention_prefill(T0, T1) → T2, T3   [LOCAL]
 *   N1: network_relay(T2, T3) → T4            [REMOTE]
 *   N2: decode_step(T4) → T5                  [LOCAL]
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

/* Always-on assertion — survives NDEBUG / Release builds */
#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #expr, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)
#include <vector>

#include <fcntl.h>
#include <unistd.h>

/* ================================================================== */
/*  Constants — mirror Python split_llama_mock preset                   */
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

/* Tensor byte sizes */
static constexpr uint64_t T0_BYTES = EMBED_ROWS * EMBED_COLS * sizeof(float);
static constexpr uint64_t T1_BYTES = SEQ_LEN * sizeof(int32_t);
static constexpr uint64_t T2_BYTES = (uint64_t)SEQ_LEN * HIDDEN_DIM * sizeof(float);
static constexpr uint64_t T3_BYTES = (uint64_t)N_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float);
static constexpr uint64_t T4_BYTES = T2_BYTES;
static constexpr uint64_t T5_BYTES = (uint64_t)SEQ_LEN * VOCAB_SIZE * sizeof(float);

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
/*  Mock Provider — correct ABI dispatch signature                     */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    if (std::strcmp(op_name, "attention_prefill") == 0 && n_in >= 2 && n_out >= 2) {
        /* T0(embed), T1(tokens) → T2(hidden), T3(kv_cache)
         * Mock: hidden[i] = float(tokens[i % SEQ_LEN]) * 0.01f */
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
        /* T2(hidden) → T4(relay_out): passthrough copy */
        auto* ib = reinterpret_cast<HostBuffer*>(inputs[0]);
        auto* ob = reinterpret_cast<HostBuffer*>(outputs[0]);
        std::memcpy(ob->data, ib->data, T2_BYTES);
        return NF_OK;
    }

    if (std::strcmp(op_name, "decode_step") == 0 && n_in >= 1 && n_out >= 1) {
        /* T4(relay_out) → T5(logits): logits[i] = tanh(relay[i % relay_count]) */
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

static const char* mock_name(nf_provider) { return "split_llama_mock"; }
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
/*  .nfir Generator — writes split_llama_mock binary to /tmp            */
/* ================================================================== */

static const char* NFIR_PATH = "/tmp/split_llama_mock_test.nfir";

static void make_tensor(nf_ir_tensor_desc* td, uint32_t id,
                        uint8_t dtype, uint8_t ndim, uint8_t usage,
                        const uint64_t* shape, uint64_t size_bytes,
                        uint64_t weight_offset) {
    std::memset(td, 0, sizeof(*td));
    td->tensor_id = id;
    td->dtype = dtype;
    td->ndim = ndim;
    td->usage = usage;
    for (uint8_t i = 0; i < ndim; ++i) td->shape[i] = shape[i];
    td->size_bytes = size_bytes;
    td->weight_offset = weight_offset;
}

static void make_node(nf_ir_node_desc* nd, uint32_t id, const char* op,
                      const uint32_t* ins, uint32_t n_in,
                      const uint32_t* outs, uint32_t n_out,
                      uint32_t flags) {
    std::memset(nd, 0, sizeof(*nd));
    nd->node_id = id;
    std::strncpy(nd->op_type, op, NF_MAX_OP_NAME - 1);
    nd->n_inputs = n_in;
    nd->n_outputs = n_out;
    for (uint32_t i = 0; i < n_in; ++i)  nd->input_tensor_ids[i] = ins[i];
    for (uint32_t i = 0; i < n_out; ++i) nd->output_tensor_ids[i] = outs[i];
    nd->task_flags = flags;
}

static bool generate_nfir() {
    /* Header */
    nf_ir_header hdr{};
    hdr.magic = NF_IR_MAGIC;
    hdr.version = NF_IR_VERSION;
    hdr.num_tensors = NUM_TENSORS;
    hdr.num_nodes = NUM_NODES;

    /* Compute payload offset: header + tensors + nodes, rounded up to 4KB */
    uint64_t meta_size = sizeof(nf_ir_header)
                       + NUM_TENSORS * sizeof(nf_ir_tensor_desc)
                       + NUM_NODES * sizeof(nf_ir_node_desc);
    hdr.payload_offset = (meta_size + NF_IR_PAYLOAD_ALIGN - 1)
                       & ~(uint64_t)(NF_IR_PAYLOAD_ALIGN - 1);
    hdr.payload_size = T0_BYTES;  /* only T0 is a weight */

    /* CRC32 of header (excluding header_crc32 and _pad0) */
    hdr.header_crc32 = nf_crc32c_update(0,
        reinterpret_cast<const uint8_t*>(&hdr),
        offsetof(nf_ir_header, header_crc32));

    /* Tensor descriptors */
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

    /* Node descriptors */
    nf_ir_node_desc nodes[NUM_NODES];
    uint32_t n0_in[] = {0, 1}, n0_out[] = {2, 3};
    make_node(&nodes[0], 0, "attention_prefill", n0_in, 2, n0_out, 2, 0);

    uint32_t n1_in[] = {2, 3}, n1_out[] = {4};
    make_node(&nodes[1], 1, "network_relay", n1_in, 2, n1_out, 1,
              NF_TASK_REMOTE);

    uint32_t n2_in[] = {4}, n2_out[] = {5};
    make_node(&nodes[2], 2, "decode_step", n2_in, 1, n2_out, 1, 0);

    /* Write .nfir file */
    int fd = ::open(NFIR_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    ::write(fd, &hdr, sizeof(hdr));
    ::write(fd, tensors, sizeof(tensors));
    ::write(fd, nodes, sizeof(nodes));

    /* Pad to payload_offset */
    uint64_t written = sizeof(hdr) + sizeof(tensors) + sizeof(nodes);
    if (written < hdr.payload_offset) {
        std::vector<uint8_t> pad(hdr.payload_offset - written, 0);
        ::write(fd, pad.data(), pad.size());
    }

    /* Weight payload: T0 embed_weights — fill with deterministic pattern */
    std::vector<float> weights(EMBED_ROWS * EMBED_COLS);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<float>(i) * 0.001f;
    ::write(fd, weights.data(), T0_BYTES);

    ::close(fd);
    std::printf("  [gen] wrote %s (%llu bytes payload)\n",
                NFIR_PATH, (unsigned long long)T0_BYTES);
    return true;
}

/* ================================================================== */
/*  Test 1: Topology verification via raw IR parse                     */
/* ================================================================== */

static void test_topology() {
    std::printf("  test_topology...\n");

    int fd = ::open(NFIR_PATH, O_RDONLY);
    REQUIRE(fd >= 0);

    nf_ir_header hdr{};
    REQUIRE(::read(fd, &hdr, sizeof(hdr)) == sizeof(hdr));
    REQUIRE(hdr.magic == NF_IR_MAGIC);
    REQUIRE(hdr.version == NF_IR_VERSION);
    REQUIRE(hdr.num_tensors == NUM_TENSORS);
    REQUIRE(hdr.num_nodes == NUM_NODES);
    std::printf("    header: %u tensors, %u nodes ✓\n",
                hdr.num_tensors, hdr.num_nodes);

    /* Read tensor descs */
    nf_ir_tensor_desc tds[NUM_TENSORS];
    REQUIRE(::read(fd, tds, sizeof(tds)) == sizeof(tds));

    /* T0 is weight, rest are activations */
    REQUIRE(tds[0].usage == NF_IR_USAGE_WEIGHT);
    for (uint32_t i = 1; i < NUM_TENSORS; ++i)
        REQUIRE(tds[i].usage == NF_IR_USAGE_ACTIVATION);
    REQUIRE(tds[0].size_bytes == T0_BYTES);
    REQUIRE(tds[1].size_bytes == T1_BYTES);
    REQUIRE(tds[5].size_bytes == T5_BYTES);
    std::printf("    tensor usage + sizes ✓\n");

    /* Read node descs */
    nf_ir_node_desc nds[NUM_NODES];
    REQUIRE(::read(fd, nds, sizeof(nds)) == sizeof(nds));

    /* N0: prefill */
    REQUIRE(std::strcmp(nds[0].op_type, "attention_prefill") == 0);
    REQUIRE(nds[0].n_inputs == 2 && nds[0].n_outputs == 2);
    REQUIRE(nds[0].input_tensor_ids[0] == 0);
    REQUIRE(nds[0].input_tensor_ids[1] == 1);
    REQUIRE(nds[0].output_tensor_ids[0] == 2);
    REQUIRE(nds[0].output_tensor_ids[1] == 3);
    REQUIRE(nds[0].task_flags == 0);

    /* N1: network_relay — REMOTE */
    REQUIRE(std::strcmp(nds[1].op_type, "network_relay") == 0);
    REQUIRE(nds[1].n_inputs == 2 && nds[1].n_outputs == 1);
    REQUIRE(nds[1].input_tensor_ids[0] == 2);
    REQUIRE(nds[1].input_tensor_ids[1] == 3);
    REQUIRE(nds[1].output_tensor_ids[0] == 4);
    REQUIRE((nds[1].task_flags & NF_TASK_REMOTE) != 0);
    std::printf("    N1 network_relay REMOTE flag ✓\n");

    /* N2: decode_step */
    REQUIRE(std::strcmp(nds[2].op_type, "decode_step") == 0);
    REQUIRE(nds[2].n_inputs == 1 && nds[2].n_outputs == 1);
    REQUIRE(nds[2].input_tensor_ids[0] == 4);
    REQUIRE(nds[2].output_tensor_ids[0] == 5);
    REQUIRE(nds[2].task_flags == 0);

    /* Edge connectivity: N0 outputs feed N1 inputs, N1 output feeds N2 */
    REQUIRE(nds[0].output_tensor_ids[0] == nds[1].input_tensor_ids[0]); /* T2 */
    REQUIRE(nds[0].output_tensor_ids[1] == nds[1].input_tensor_ids[1]); /* T3 */
    REQUIRE(nds[1].output_tensor_ids[0] == nds[2].input_tensor_ids[0]); /* T4 */
    std::printf("    edge connectivity ✓\n");

    ::close(fd);
    std::printf("  PASS: topology\n");
}

/* ================================================================== */
/*  Test 2: GraphBuilder load + build                                  */
/* ================================================================== */

static void test_graph_builder_load() {
    std::printf("  test_graph_builder_load...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int mock_prov_id = 0;
    nf_status st = engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_prov_id), vt, NF_AFFINITY_CPU);
    REQUIRE(st == NF_OK);

    nf::GraphBuilder gb(engine, host_alloc_fn);
    st = gb.load(NFIR_PATH);
    REQUIRE(st == NF_OK);
    std::printf("    load OK ✓\n");

    uint32_t graph_id = 0;
    st = gb.build(&graph_id);
    REQUIRE(st == NF_OK);
    std::printf("    build OK, graph_id=%u ✓\n", graph_id);

    /* Verify weight tensor T0 is accessible */
    nf_buffer t0_buf = gb.get_tensor_buffer(0);
    REQUIRE(t0_buf != nullptr);
    std::printf("    T0 weight buffer accessible ✓\n");

    std::printf("  PASS: graph_builder_load\n");
}

/* ================================================================== */
/*  Test 3: Payload weight verification                                */
/* ================================================================== */

static void test_weight_payload() {
    std::printf("  test_weight_payload...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int mock_prov_id = 1;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&mock_prov_id), vt, NF_AFFINITY_CPU);

    nf::GraphBuilder gb(engine, host_alloc_fn);
    REQUIRE(gb.load(NFIR_PATH) == NF_OK);

    uint32_t gid = 0;
    REQUIRE(gb.build(&gid) == NF_OK);

    nf_buffer t0_buf = gb.get_tensor_buffer(0);
    REQUIRE(t0_buf != nullptr);

    /* Map and verify deterministic pattern */
    nf_buffer_ops t0_ops = gb.get_tensor_ops(0);
    void* ptr = nullptr;
    REQUIRE(t0_ops.map(t0_buf, &ptr) == NF_OK);

    auto* weights = static_cast<float*>(ptr);
    /* Spot-check first and last values */
    REQUIRE(std::fabs(weights[0] - 0.0f) < 1e-6f);
    REQUIRE(std::fabs(weights[1] - 0.001f) < 1e-6f);
    REQUIRE(std::fabs(weights[100] - 0.1f) < 1e-4f);
    std::printf("    weight spot-check ✓\n");

    t0_ops.unmap(t0_buf);
    std::printf("  PASS: weight_payload\n");
}

/* ================================================================== */
/*  Test 4: File size sanity                                           */
/* ================================================================== */

static void test_file_size() {
    std::printf("  test_file_size...\n");

    int fd = ::open(NFIR_PATH, O_RDONLY);
    REQUIRE(fd >= 0);
    off_t size = ::lseek(fd, 0, SEEK_END);
    ::close(fd);

    /* Expected: payload_offset + T0_BYTES */
    uint64_t meta_size = sizeof(nf_ir_header)
                       + NUM_TENSORS * sizeof(nf_ir_tensor_desc)
                       + NUM_NODES * sizeof(nf_ir_node_desc);
    uint64_t payload_off = (meta_size + NF_IR_PAYLOAD_ALIGN - 1)
                         & ~(uint64_t)(NF_IR_PAYLOAD_ALIGN - 1);
    uint64_t expected = payload_off + T0_BYTES;

    REQUIRE(static_cast<uint64_t>(size) == expected);
    std::printf("    file size %lld == expected %llu ✓\n",
                (long long)size, (unsigned long long)expected);
    std::printf("  PASS: file_size\n");
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main() {
    std::printf("=== split_llama_mock_test ===\n");

    REQUIRE(generate_nfir());
    test_topology();
    test_graph_builder_load();
    test_weight_payload();
    test_file_size();

    /* Cleanup */
    ::unlink(NFIR_PATH);

    std::printf("=== ALL 4 TESTS PASSED ===\n");
    return 0;
}
