/**
 * @file ir_loader_test.cpp
 * @brief Phase 8 — IR Loader & MMap Weight Verification
 *
 * 1. Generates a .nfir file on disk with weights + graph topology
 * 2. Loads via GraphBuilder (mmap weights, allocate activations)
 * 3. Executes DAG: weighted_add → mock_relu
 * 4. Verifies bit-exact output
 * 5. Confirms weight buffer domain == NF_MEM_DOMAIN_MMAP
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_ir_format.h"
#include "neurofabric/PipelineEngine.hpp"
#include "neurofabric/GraphBuilder.hpp"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

/* ================================================================== */
/*  Host Buffer — simple malloc-backed activation allocator            */
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
/*  Mock Provider — uses cross-dylib bridge to access buffer ops       */
/*  PipelineEngine sets desc.user_data = &desc, so dispatch can        */
/*  recover input_ops/output_ops via pointer arithmetic.               */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    /*
     * Recover the full nf_task_desc via the cross-dylib bridge:
     * PipelineEngine stores user_data = &desc, and inputs == desc.inputs.
     * So: desc_ptr = (char*)inputs - offsetof(nf_task_desc, inputs)
     */
    auto* desc = reinterpret_cast<const nf_task_desc*>(
        reinterpret_cast<const char*>(inputs) -
        offsetof(nf_task_desc, inputs));

    if (std::strcmp(op_name, "weighted_add") == 0 && n_in >= 2 && n_out >= 1) {
        /* output[i] = weight[i] + input[i] */
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
        /* output[i] = max(0, input[i]) — in-place OK */
        void *i_ptr, *o_ptr;
        desc->input_ops[0].map(inputs[0], &i_ptr);
        desc->output_ops[0].map(outputs[0], &o_ptr);

        auto* src = static_cast<const float*>(i_ptr);
        auto* dst = static_cast<float*>(o_ptr);

        nf_buffer_info info{};
        desc->input_ops[0].get_info(inputs[0], &info);
        size_t count = info.desc.size_bytes / sizeof(float);

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
/*  .nfir File Generator                                               */
/* ================================================================== */

static constexpr uint32_t N_FLOATS = 64;
static constexpr uint32_t TENSOR_BYTES = N_FLOATS * sizeof(float);

/**
 * Generate a .nfir file at `path` with:
 *   Tensor 0: weight_W  (WEIGHT, 64 floats, pattern w[i] = i * 0.1f)
 *   Tensor 1: input_A   (ACTIVATION, 64 floats)
 *   Tensor 2: output_B  (ACTIVATION, 64 floats)
 *   Node 0: weighted_add — inputs=[0,1], outputs=[2]
 *   Node 1: mock_relu    — inputs=[2],   outputs=[2] (in-place)
 */
static void generate_nfir(const char* path) {
    /* Compute payload offset: header + 3 tensor descs + 2 node descs,
     * rounded up to 4KB boundary */
    size_t metadata_size = sizeof(nf_ir_header)
                         + 3 * sizeof(nf_ir_tensor_desc)
                         + 2 * sizeof(nf_ir_node_desc);
    uint64_t payload_offset = (metadata_size + NF_IR_PAYLOAD_ALIGN - 1)
                            & ~(uint64_t)(NF_IR_PAYLOAD_ALIGN - 1);

    /* Build header */
    nf_ir_header hdr{};
    hdr.magic          = NF_IR_MAGIC;
    hdr.version        = NF_IR_VERSION;
    hdr.num_tensors    = 3;
    hdr.num_nodes      = 2;
    hdr.payload_offset = payload_offset;
    hdr.payload_size   = TENSOR_BYTES;  /* one weight tensor */
    hdr.header_crc32   = 0;
    hdr._pad0          = 0;
    hdr.header_crc32   = nf_ir_header_compute_crc(&hdr);

    /* Build tensor descriptors */
    nf_ir_tensor_desc tensors[3]{};

    /* Tensor 0: weight_W */
    tensors[0].tensor_id     = 0;
    tensors[0].dtype         = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[0].ndim          = 1;
    tensors[0].usage         = NF_IR_USAGE_WEIGHT;
    tensors[0].shape[0]      = N_FLOATS;
    tensors[0].size_bytes    = TENSOR_BYTES;
    tensors[0].weight_offset = 0;  /* first (and only) weight at offset 0 */

    /* Tensor 1: input_A (activation) */
    tensors[1].tensor_id     = 1;
    tensors[1].dtype         = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[1].ndim          = 1;
    tensors[1].usage         = NF_IR_USAGE_ACTIVATION;
    tensors[1].shape[0]      = N_FLOATS;
    tensors[1].size_bytes    = TENSOR_BYTES;
    tensors[1].weight_offset = 0;

    /* Tensor 2: output_B (activation) */
    tensors[2].tensor_id     = 2;
    tensors[2].dtype         = static_cast<uint8_t>(NF_DTYPE_F32);
    tensors[2].ndim          = 1;
    tensors[2].usage         = NF_IR_USAGE_ACTIVATION;
    tensors[2].shape[0]      = N_FLOATS;
    tensors[2].size_bytes    = TENSOR_BYTES;
    tensors[2].weight_offset = 0;

    /* Build node descriptors */
    nf_ir_node_desc nodes[2]{};

    /* Node 0: weighted_add(weight_W, input_A) → output_B */
    nodes[0].node_id = 0;
    std::strncpy(nodes[0].op_type, "weighted_add", NF_MAX_OP_NAME);
    nodes[0].n_inputs  = 2;
    nodes[0].n_outputs = 1;
    nodes[0].input_tensor_ids[0]  = 0;  /* weight_W */
    nodes[0].input_tensor_ids[1]  = 1;  /* input_A  */
    nodes[0].output_tensor_ids[0] = 2;  /* output_B */
    nodes[0].task_flags = 0;

    /* Node 1: mock_relu(output_B) → output_B (in-place) */
    nodes[1].node_id = 1;
    std::strncpy(nodes[1].op_type, "mock_relu", NF_MAX_OP_NAME);
    nodes[1].n_inputs  = 1;
    nodes[1].n_outputs = 1;
    nodes[1].input_tensor_ids[0]  = 2;  /* output_B */
    nodes[1].output_tensor_ids[0] = 2;  /* output_B (in-place) */
    nodes[1].task_flags = 0;

    /* Weight payload: w[i] = i * 0.1f */
    float weights[N_FLOATS];
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        weights[i] = static_cast<float>(i) * 0.1f;

    /* Write file */
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    assert(fd >= 0);

    ::write(fd, &hdr, sizeof(hdr));
    ::write(fd, tensors, sizeof(tensors));
    ::write(fd, nodes, sizeof(nodes));

    /* Zero-pad to payload_offset */
    size_t written = sizeof(hdr) + sizeof(tensors) + sizeof(nodes);
    if (written < payload_offset) {
        std::vector<uint8_t> pad(static_cast<size_t>(payload_offset) - written, 0);
        ::write(fd, pad.data(), pad.size());
    }

    /* Write weight payload */
    ::write(fd, weights, sizeof(weights));

    ::close(fd);
}

/* ================================================================== */
/*  Test: Load .nfir, execute DAG, verify bit-exact                    */
/* ================================================================== */

static void test_ir_load_and_execute() {
    std::printf("  test_ir_load_and_execute...\n");

    /* 1. Generate .nfir file */
    const char* path = "/tmp/nf_test_phase8.nfir";
    generate_nfir(path);
    std::printf("    [gen] wrote %s\n", path);

    /* 2. Set up PipelineEngine with mock provider */
    nf::PipelineEngine engine(2);
    engine.register_provider(nullptr, make_mock_vt(), NF_AFFINITY_ANY);

    /* 3. Load and build via GraphBuilder */
    nf::GraphBuilder builder(engine, host_alloc_fn);
    nf_status st = builder.load(path);
    assert(st == NF_OK);
    std::printf("    [load] IR loaded OK\n");

    uint32_t graph_id;
    st = builder.build(&graph_id);
    assert(st == NF_OK);
    std::printf("    [build] DAG built, graph_id=%u\n", graph_id);

    /* 4. Fill activation input (tensor 1) with test pattern */
    nf_buffer input_buf = builder.get_tensor_buffer(1);
    nf_buffer_ops input_ops = builder.get_tensor_ops(1);
    assert(input_buf);

    void* input_ptr = nullptr;
    input_ops.map(input_buf, &input_ptr);
    auto* inp = static_cast<float*>(input_ptr);
    for (uint32_t i = 0; i < N_FLOATS; ++i)
        inp[i] = static_cast<float>(i) * -0.05f;
    input_ops.unmap(input_buf);

    /* 5. Submit and wait */
    auto future = engine.submit(graph_id);
    st = future.get();
    assert(st == NF_OK);
    std::printf("    [exec] DAG completed OK\n");

    /* 6. Verify bit-exact output */
    nf_buffer output_buf = builder.get_tensor_buffer(2);
    nf_buffer_ops output_ops = builder.get_tensor_ops(2);
    assert(output_buf);

    void* output_ptr = nullptr;
    output_ops.map(output_buf, &output_ptr);
    auto* out = static_cast<const float*>(output_ptr);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < N_FLOATS; ++i) {
        /* expected = max(0, weight[i] + input[i])
         *          = max(0, i*0.1 + i*(-0.05))
         *          = max(0, i*0.05)
         * For i >= 0: i*0.05 >= 0, so relu is identity */
        float expected = static_cast<float>(i) * 0.05f;
        if (std::memcmp(&out[i], &expected, sizeof(float)) != 0) {
            if (mismatches < 5)
                std::printf("    MISMATCH [%u]: got %.6f, expected %.6f\n",
                            i, out[i], expected);
            ++mismatches;
        }
    }
    output_ops.unmap(output_buf);
    assert(mismatches == 0);
    std::printf("    [verify] %u floats bit-exact match\n", N_FLOATS);

    /* 7. Verify mmap domain on weight buffer */
    nf_buffer weight_buf = builder.get_tensor_buffer(0);
    nf_buffer_ops weight_ops = builder.get_tensor_ops(0);
    assert(weight_buf);

    nf_buffer_info w_info{};
    weight_ops.get_info(weight_buf, &w_info);
    assert(w_info.domain == NF_MEM_DOMAIN_MMAP);
    std::printf("    [verify] weight domain = MMAP (%d)\n", w_info.domain);

    /* 8. Cleanup */
    ::unlink(path);
    std::printf("  PASS: IR load + execute (bit-exact)\n");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("ir_loader_test: Phase 8 — Zero-Copy Weight MMap & Graph IR\n");
    test_ir_load_and_execute();
    std::printf("OK: all Phase 8 IR loader tests passed\n");
    return 0;
}
