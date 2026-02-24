/**
 * @file dag_to_nfir_test.cpp
 * @brief Phase 45B — DAG → NFIR lifting tests
 */

#include "neuralOS/compiler/dag_to_nfir.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

using namespace neuralOS::compiler;

static void test_map_op_name() {
    CHECK(map_op_name("rms_norm") == HighOpKind::RMS_NORM, "rms_norm mapping");
    CHECK(map_op_name("linear") == HighOpKind::MATMUL, "linear mapping");
    CHECK(map_op_name("linear_tiled") == HighOpKind::MATMUL, "linear_tiled mapping");
    CHECK(map_op_name("causal_attention") == HighOpKind::ATTENTION, "attention mapping");
    CHECK(map_op_name("silu") == HighOpKind::SILU, "silu mapping");
    CHECK(map_op_name("element_mul") == HighOpKind::ELEMENT_MUL, "element_mul mapping");
    CHECK(map_op_name("element_add") == HighOpKind::ELEMENT_ADD, "element_add mapping");
    CHECK(map_op_name("embedding_lookup") == HighOpKind::EMBEDDING, "embedding mapping");
    CHECK(map_op_name("softmax") == HighOpKind::SOFTMAX, "softmax mapping");
    CHECK(map_op_name("rope") == HighOpKind::ROPE, "rope mapping");
    CHECK(map_op_name("unknown_op") == HighOpKind::CUSTOM, "unknown → CUSTOM");
    CHECK(map_op_name(nullptr) == HighOpKind::CUSTOM, "nullptr → CUSTOM");
    fprintf(stderr, "  [PASS] test_map_op_name\n");
}

static void test_lift_nodes() {
    DagNodeInfo nodes[] = {
        {0, "embedding_lookup", 1, 1},
        {1, "rms_norm",         1, 1},
        {2, "linear",           2, 1},
        {3, "rope",             1, 1},
        {4, "causal_attention",  3, 1},
        {5, "linear_tiled",     2, 1},
        {6, "element_add",      2, 1},
        {7, "rms_norm",         1, 1},
        {8, "linear",           2, 1},
        {9, "silu",             1, 1},
        {10, "element_mul",     2, 1},
        {11, "linear",          2, 1},
        {12, "element_add",     2, 1},
    };
    uint32_t count = sizeof(nodes) / sizeof(nodes[0]);

    auto graph = lift_nodes_to_nfir(nodes, count);

    CHECK(graph.num_ops() == count, "op count matches");
    CHECK(graph.ops[0].kind == HighOpKind::EMBEDDING, "first op is EMBEDDING");
    CHECK(graph.ops[4].kind == HighOpKind::ATTENTION, "op4 is ATTENTION");
    CHECK(graph.ops[9].kind == HighOpKind::SILU, "op9 is SILU");

    /* Check edges: sequential chain */
    for (uint32_t i = 0; i + 1 < count; ++i) {
        auto eit = graph.edges.find(i);
        CHECK(eit != graph.edges.end(), "edge exists");
        CHECK(eit->second.size() == 1, "single successor");
        CHECK(eit->second[0] == i + 1, "correct successor");
    }

    /* Check dag_node_id attribute preserved */
    for (uint32_t i = 0; i < count; ++i) {
        auto it = graph.ops[i].attrs_i.find("dag_node_id");
        CHECK(it != graph.ops[i].attrs_i.end(), "dag_node_id present");
        CHECK(it->second == (int64_t)nodes[i].node_id, "dag_node_id correct");
    }

    fprintf(stderr, "  [PASS] test_lift_nodes\n");
}

static void test_lift_empty() {
    auto graph = lift_nodes_to_nfir(nullptr, 0);
    CHECK(graph.num_ops() == 0, "empty graph");
    fprintf(stderr, "  [PASS] test_lift_empty\n");
}

static void test_lift_step_graph_api() {
    uint32_t ids[] = {0, 1, 2};
    const char* names[] = {"rms_norm", "linear", "silu"};
    auto graph = lift_step_graph(ids, names, 3);
    CHECK(graph.num_ops() == 3, "3 ops");
    CHECK(graph.ops[0].kind == HighOpKind::RMS_NORM, "first is RMS_NORM");
    CHECK(graph.ops[1].kind == HighOpKind::MATMUL, "second is MATMUL");
    CHECK(graph.ops[2].kind == HighOpKind::SILU, "third is SILU");
    fprintf(stderr, "  [PASS] test_lift_step_graph_api\n");
}

int main() {
    fprintf(stderr, "[dag_to_nfir_test]\n");
    test_map_op_name();
    test_lift_nodes();
    test_lift_empty();
    test_lift_step_graph_api();
    fprintf(stderr, "[dag_to_nfir_test] ALL PASSED\n");
    return 0;
}
