/**
 * @file onnx_e2e_test.cpp
 * @brief Phase 44C.3 — ONNX E2E test
 *
 * Hand-crafts a minimal ONNX protobuf binary, parses it,
 * converts to NfirHighGraph, and runs CompilerPipeline.
 */

#include "onnx/onnx_parser.hpp"
#include "onnx/onnx_to_nfir.hpp"
#include "neuralOS/compiler/compiler_pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); std::exit(1); } \
} while(0)

/* ---- Protobuf encoding helpers ---- */

static void pb_write_varint(std::vector<uint8_t>& buf, uint64_t val) {
    while (val > 0x7F) {
        buf.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    buf.push_back(static_cast<uint8_t>(val));
}

static void pb_write_tag(std::vector<uint8_t>& buf, uint32_t field, uint32_t wire_type) {
    pb_write_varint(buf, (field << 3) | wire_type);
}

static void pb_write_string(std::vector<uint8_t>& buf, uint32_t field, const char* s) {
    size_t len = std::strlen(s);
    pb_write_tag(buf, field, 2);  /* LENGTH */
    pb_write_varint(buf, len);
    buf.insert(buf.end(), s, s + len);
}

static void pb_write_bytes(std::vector<uint8_t>& buf, uint32_t field,
                            const std::vector<uint8_t>& data) {
    pb_write_tag(buf, field, 2);
    pb_write_varint(buf, data.size());
    buf.insert(buf.end(), data.begin(), data.end());
}

static void pb_write_submessage(std::vector<uint8_t>& buf, uint32_t field,
                                 const std::vector<uint8_t>& sub) {
    pb_write_tag(buf, field, 2);
    pb_write_varint(buf, sub.size());
    buf.insert(buf.end(), sub.begin(), sub.end());
}

/* Build a minimal ONNX NodeProto */
static std::vector<uint8_t> build_node(const char* op_type, const char* name,
                                        const std::vector<const char*>& inputs,
                                        const std::vector<const char*>& outputs) {
    std::vector<uint8_t> node;
    for (auto* inp : inputs)  pb_write_string(node, 1, inp);   /* input */
    for (auto* outp : outputs) pb_write_string(node, 2, outp);  /* output */
    pb_write_string(node, 3, name);     /* name */
    pb_write_string(node, 4, op_type);  /* op_type */
    return node;
}

/* Build a minimal TensorProto (initializer) */
static std::vector<uint8_t> build_tensor(const char* name, int32_t dtype,
                                          const std::vector<int64_t>& dims,
                                          const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> tensor;
    /* field 1: dims (packed varint) */
    if (!dims.empty()) {
        std::vector<uint8_t> packed;
        for (auto d : dims) pb_write_varint(packed, static_cast<uint64_t>(d));
        pb_write_bytes(tensor, 1, packed);
    }
    /* field 2: data_type */
    pb_write_tag(tensor, 2, 0);
    pb_write_varint(tensor, static_cast<uint64_t>(dtype));
    /* field 8: name */
    pb_write_string(tensor, 8, name);
    /* field 13: raw_data */
    if (!raw.empty()) pb_write_bytes(tensor, 13, raw);
    return tensor;
}

/* Build a minimal ONNX GraphProto */
static std::vector<uint8_t> build_graph(const char* name,
                                         const std::vector<std::vector<uint8_t>>& nodes,
                                         const std::vector<std::vector<uint8_t>>& initializers) {
    std::vector<uint8_t> graph;
    for (auto& n : nodes)        pb_write_submessage(graph, 1, n);  /* node */
    pb_write_string(graph, 2, name);                                 /* name */
    for (auto& init : initializers) pb_write_submessage(graph, 5, init);  /* initializer */
    return graph;
}

/* Build a minimal ONNX ModelProto */
static std::vector<uint8_t> build_model(int64_t ir_version, const char* producer,
                                         const std::vector<uint8_t>& graph) {
    std::vector<uint8_t> model;
    /* field 1: ir_version */
    pb_write_tag(model, 1, 0);
    pb_write_varint(model, static_cast<uint64_t>(ir_version));
    /* field 2: producer_name */
    pb_write_string(model, 2, producer);
    /* field 7: graph */
    pb_write_submessage(model, 7, graph);
    return model;
}

int main() {
    std::fprintf(stderr, "[onnx_e2e] starting...\n");

    /* ---- Build a small ONNX model: MatMul → Add → Relu ---- */

    auto matmul_node = build_node("MatMul", "matmul_0",
                                   {"input", "weight"}, {"mm_out"});
    auto add_node = build_node("Add", "add_0",
                                {"mm_out", "bias"}, {"add_out"});
    auto relu_node = build_node("Relu", "relu_0",
                                 {"add_out"}, {"output"});

    /* Weight initializer: 4x4 float32 = 64 bytes */
    std::vector<uint8_t> weight_data(64, 0);
    float one = 1.0f;
    for (int i = 0; i < 4; ++i)
        std::memcpy(weight_data.data() + i * 16 + i * 4, &one, 4);

    auto weight_tensor = build_tensor("weight", 1 /* FLOAT */,
                                       {4, 4}, weight_data);

    /* Bias initializer: 4 floats = 16 bytes */
    std::vector<uint8_t> bias_data(16, 0);
    auto bias_tensor = build_tensor("bias", 1 /* FLOAT */, {4}, bias_data);

    auto graph = build_graph("test_graph",
                              {matmul_node, add_node, relu_node},
                              {weight_tensor, bias_tensor});

    auto model_bytes = build_model(7, "nf_test", graph);

    std::fprintf(stderr, "[onnx_e2e] built model: %zu bytes\n", model_bytes.size());

    /* ---- Test 1: Parse the protobuf ---- */
    std::fprintf(stderr, "[onnx_e2e] test 1: parse protobuf\n");

    neuralOS::onnx::OnnxModel model;
    neuralOS::onnx::OnnxParser parser;
    bool ok = parser.parse(model_bytes.data(), model_bytes.size(), &model);
    CHECK(ok, "parse should succeed");
    CHECK(model.ir_version == 7, "ir_version should be 7");
    CHECK(model.producer_name == "nf_test", "producer should be nf_test");
    CHECK(model.graph.nodes.size() == 3, "should have 3 nodes");
    CHECK(model.graph.initializers.size() == 2, "should have 2 initializers");

    std::fprintf(stderr, "[onnx_e2e]   nodes: %zu, initializers: %zu\n",
                 model.graph.nodes.size(), model.graph.initializers.size());

    /* Verify node details */
    CHECK(model.graph.nodes[0].op_type == "MatMul", "node 0 should be MatMul");
    CHECK(model.graph.nodes[1].op_type == "Add", "node 1 should be Add");
    CHECK(model.graph.nodes[2].op_type == "Relu", "node 2 should be Relu");

    /* Verify initializer details */
    CHECK(model.graph.initializers[0].name == "weight", "init 0 should be weight");
    CHECK(model.graph.initializers[0].raw_data.size() == 64, "weight should be 64 bytes");
    CHECK(model.graph.initializers[0].dims.size() == 2, "weight should be 2D");
    CHECK(model.graph.initializers[1].name == "bias", "init 1 should be bias");

    /* ---- Test 2: Convert to NfirHighGraph ---- */
    std::fprintf(stderr, "[onnx_e2e] test 2: convert to NFIR\n");

    neuralOS::L1::NfirHighGraph nfir;
    ok = neuralOS::onnx::onnx_to_nfir(model.graph, &nfir);
    CHECK(ok, "onnx_to_nfir should succeed");
    CHECK(nfir.num_ops() == 3, "should have 3 ops");
    CHECK(nfir.num_tensors() >= 4, "should have at least 4 tensors");

    /* Verify op kinds */
    CHECK(nfir.ops[0].kind == neuralOS::L1::HighOpKind::MATMUL, "op 0 should be MATMUL");
    CHECK(nfir.ops[1].kind == neuralOS::L1::HighOpKind::ELEMENT_ADD, "op 1 should be ELEMENT_ADD");
    CHECK(nfir.ops[2].kind == neuralOS::L1::HighOpKind::SILU, "op 2 should be SILU (Relu mapped)");

    /* Verify weight tensor has size_bytes from initializer */
    bool found_weight = false;
    for (auto& t : nfir.tensors) {
        if (t.name == "weight") {
            CHECK(t.size_bytes == 64, "weight tensor should be 64 bytes");
            CHECK(t.dtype == 1, "weight dtype should be F32");
            CHECK(t.ndim == 2, "weight should be 2D");
            found_weight = true;
        }
    }
    CHECK(found_weight, "should find weight tensor");

    std::fprintf(stderr, "[onnx_e2e]   ops: %u, tensors: %u, edges: %zu\n",
                 nfir.num_ops(), nfir.num_tensors(), nfir.edges.size());

    /* ---- Test 3: Run CompilerPipeline ---- */
    std::fprintf(stderr, "[onnx_e2e] test 3: compiler pipeline\n");

    neuralOS::L1::CompilerPipeline compiler;
    auto cr = compiler.run(&nfir);
    std::fprintf(stderr, "[onnx_e2e]   removed=%u merged=%u shapes=%u fusions=%u\n",
                 cr.ops_removed, cr.ops_merged, cr.shapes_inferred, cr.fusions_found);
    /* Pipeline should run without crashing; specific counts depend on pass logic */
    CHECK(nfir.num_ops() >= 1, "graph should still have ops");

    /* ---- Done ---- */
    std::fprintf(stderr, "[onnx_e2e] PASS\n");
    return 0;
}
