/**
 * @file onnx_parser_test.cpp
 * @brief Phase 37.4 — ONNX parser + converter tests
 */

#include "onnx/onnx_parser.hpp"
#include "onnx/onnx_to_nfir.hpp"
#include "onnx/onnx_op_map.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== ONNX Parser Test ===\n");

    /* Test 1: Op map lookup */
    auto mm = neuralOS::compiler::onnx::lookup_op("MatMul");
    CHECK(mm.supported, "MatMul is supported");
    CHECK(mm.kind == neuralOS::compiler::HighOpKind::MATMUL, "MatMul → MATMUL");

    auto attn = neuralOS::compiler::onnx::lookup_op("Attention");
    CHECK(attn.kind == neuralOS::compiler::HighOpKind::ATTENTION, "Attention → ATTENTION");

    auto unknown = neuralOS::compiler::onnx::lookup_op("UnknownOp");
    CHECK(!unknown.supported, "UnknownOp not supported");
    CHECK(unknown.kind == neuralOS::compiler::HighOpKind::CUSTOM, "UnknownOp → CUSTOM");

    auto gelu = neuralOS::compiler::onnx::lookup_op("Gelu");
    CHECK(gelu.kind == neuralOS::compiler::HighOpKind::GELU, "Gelu → GELU");

    auto dq = neuralOS::compiler::onnx::lookup_op("DequantizeLinear");
    CHECK(dq.kind == neuralOS::compiler::HighOpKind::DEQUANT, "DequantizeLinear → DEQUANT");

    /* Test 2: Op map coverage */
    auto& map = neuralOS::compiler::onnx::get_op_map();
    CHECK(map.size() >= 40, "at least 40 ops mapped");

    /* Test 3: Build a synthetic OnnxGraph and convert to NFIR */
    neuralOS::compiler::onnx::OnnxGraph onnx_graph;
    onnx_graph.name = "test_model";

    neuralOS::compiler::onnx::OnnxNode n1;
    n1.op_type = "MatMul";
    n1.name = "matmul_0";
    n1.inputs = {"input_0", "weight_0"};
    n1.outputs = {"mm_out"};
    onnx_graph.nodes.push_back(n1);

    neuralOS::compiler::onnx::OnnxNode n2;
    n2.op_type = "Add";
    n2.name = "add_bias";
    n2.inputs = {"mm_out", "bias_0"};
    n2.outputs = {"add_out"};
    onnx_graph.nodes.push_back(n2);

    neuralOS::compiler::onnx::OnnxNode n3;
    n3.op_type = "Gelu";
    n3.name = "gelu_0";
    n3.inputs = {"add_out"};
    n3.outputs = {"gelu_out"};
    onnx_graph.nodes.push_back(n3);

    neuralOS::compiler::NfirHighGraph nfir;
    bool ok = neuralOS::compiler::onnx::onnx_to_nfir(onnx_graph, &nfir);
    CHECK(ok, "onnx_to_nfir succeeded");
    CHECK(nfir.num_ops() == 3, "3 ops converted");

    /* Verify op kinds */
    CHECK(nfir.ops[0].kind == neuralOS::compiler::HighOpKind::MATMUL, "op0 is MATMUL");
    CHECK(nfir.ops[1].kind == neuralOS::compiler::HighOpKind::ELEMENT_ADD, "op1 is ELEMENT_ADD");
    CHECK(nfir.ops[2].kind == neuralOS::compiler::HighOpKind::GELU, "op2 is GELU");

    /* Verify edges: matmul → add → gelu */
    CHECK(nfir.edges.count(0) > 0, "matmul has successors");
    bool has_edge_0_1 = false;
    for (auto s : nfir.edges[0]) if (s == 1) has_edge_0_1 = true;
    CHECK(has_edge_0_1, "edge matmul → add");

    bool has_edge_1_2 = false;
    for (auto s : nfir.edges[1]) if (s == 2) has_edge_1_2 = true;
    CHECK(has_edge_1_2, "edge add → gelu");

    /* Verify tensors created */
    CHECK(nfir.num_tensors() >= 5, "at least 5 tensors (input, weight, mm_out, bias, add_out, gelu_out)");

    /* Test 4: Null graph */
    CHECK(!neuralOS::compiler::onnx::onnx_to_nfir(onnx_graph, nullptr), "null output fails");

    /* Test 5: ProtobufReader basics */
    uint8_t varint_data[] = {0xAC, 0x02};  /* 300 in varint */
    neuralOS::compiler::onnx::ProtobufReader reader(varint_data, 2);
    uint64_t val = reader.read_varint();
    CHECK(val == 300, "varint decode: 300");

    /* Test 6: Empty parse */
    neuralOS::compiler::onnx::OnnxParser parser;
    neuralOS::compiler::onnx::OnnxModel model;
    CHECK(parser.parse(nullptr, 0, &model) == false, "null data fails");

    printf("PASS: all ONNX parser tests passed\n");
    return 0;
}
