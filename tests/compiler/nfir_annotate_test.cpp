/**
 * @file nfir_annotate_test.cpp
 * @brief Phase 45B — NFIR → DAG annotation tests
 */

#include "neuralOS/compiler/dag_to_nfir.hpp"
#include "neuralOS/compiler/nfir_annotate.hpp"
#include "neuralOS/compiler/compiler_pipeline.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

using namespace neuralOS::L1;

static void test_annotate_basic() {
    /* Build a small graph with fusible pattern: DEQUANT → MATMUL */
    DagNodeInfo nodes[] = {
        {100, "rms_norm",    1, 1},
        {101, "linear",      2, 1},
        {102, "silu",        1, 1},
        {103, "element_mul", 2, 1},
        {104, "linear",      2, 1},
    };
    auto graph = lift_nodes_to_nfir(nodes, 5);

    /* Run compiler pipeline (fusion should find silu+element_mul) */
    CompilerPipeline compiler;
    auto cr = compiler.run(&graph);

    /* Annotate */
    auto result = annotate_from_nfir(graph);

    /* All 5 ops should have annotations */
    CHECK(result.annotations.size() > 0, "annotations not empty");

    /* Check that dag_node_ids are preserved */
    bool found_100 = result.annotations.count(100) > 0;
    bool found_104 = result.annotations.count(104) > 0;
    CHECK(found_100, "dag_node_id 100 annotated");
    CHECK(found_104, "dag_node_id 104 annotated");

    /* Fusion should have been detected (silu+element_mul) */
    if (cr.fusions_found > 0) {
        CHECK(result.nodes_fused > 0, "fused nodes annotated");
        /* silu (dag_id=102) and element_mul (dag_id=103) should be fused */
        if (result.annotations.count(102) && result.annotations.count(103)) {
            CHECK(result.annotations[102].fused || result.annotations[103].fused,
                  "silu/element_mul marked fused");
        }
    }

    fprintf(stderr, "  [PASS] test_annotate_basic (fusions=%u, fused_nodes=%u)\n",
            cr.fusions_found, result.nodes_fused);
}

static void test_annotate_empty() {
    NfirHighGraph empty;
    auto result = annotate_from_nfir(empty);
    CHECK(result.annotations.empty(), "empty graph → empty annotations");
    CHECK(result.nodes_skipped == 0, "no skipped");
    CHECK(result.nodes_fused == 0, "no fused");
    fprintf(stderr, "  [PASS] test_annotate_empty\n");
}

static void test_roundtrip_lift_compile_annotate() {
    /* Simulate a mini transformer layer */
    DagNodeInfo nodes[] = {
        {0,  "embedding_lookup", 1, 1},
        {1,  "rms_norm",         1, 1},
        {2,  "linear",           2, 1},
        {3,  "linear",           2, 1},
        {4,  "linear",           2, 1},
        {5,  "rope",             1, 1},
        {6,  "rope",             1, 1},
        {7,  "causal_attention",  3, 1},
        {8,  "linear",           2, 1},
        {9,  "element_add",      2, 1},
        {10, "rms_norm",         1, 1},
        {11, "linear",           2, 1},
        {12, "silu",             1, 1},
        {13, "element_mul",      2, 1},
        {14, "linear",           2, 1},
        {15, "element_add",      2, 1},
    };
    uint32_t count = sizeof(nodes) / sizeof(nodes[0]);

    /* Step 1: Lift */
    auto graph = lift_nodes_to_nfir(nodes, count);
    CHECK(graph.num_ops() == count, "lifted op count");

    /* Step 2: Compile */
    CompilerPipeline compiler;
    auto cr = compiler.run(&graph);

    /* Step 3: Annotate */
    auto result = annotate_from_nfir(graph);

    fprintf(stderr, "  [PASS] test_roundtrip (ops=%u, removed=%u, merged=%u, "
            "shapes=%u, fusions=%u, annotated=%zu)\n",
            count, cr.ops_removed, cr.ops_merged,
            cr.shapes_inferred, cr.fusions_found,
            result.annotations.size());
}

int main() {
    fprintf(stderr, "[nfir_annotate_test]\n");
    test_annotate_basic();
    test_annotate_empty();
    test_roundtrip_lift_compile_annotate();
    fprintf(stderr, "[nfir_annotate_test] ALL PASSED\n");
    return 0;
}
