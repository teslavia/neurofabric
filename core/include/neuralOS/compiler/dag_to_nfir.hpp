/**
 * @file dag_to_nfir.hpp
 * @brief NeuralOS compiler — DAG -> NFIR Lifting (StepGraph -> NfirHighGraph)
 *
 * Phase 45B: Bridges the real inference path through the NFIR compiler.
 * Converts a PipelineEngine StepGraph into a NfirHighGraph so that
 * CompilerPipeline (DCE + CSE + Shape + Fusion) can optimize it.
 *
 * Header-only. Depends on nfir_high.hpp + PipelineEngine.hpp.
 */

#ifndef NEURALOS_COMPILER_DAG_TO_NFIR_HPP
#define NEURALOS_COMPILER_DAG_TO_NFIR_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

namespace neuralOS { namespace compiler {

/* ================================================================== */
/*  Op name → HighOpKind mapping                                       */
/* ================================================================== */

inline HighOpKind map_op_name(const char* op_name) {
    static const std::unordered_map<std::string, HighOpKind> table = {
        {"rms_norm",          HighOpKind::RMS_NORM},
        {"layer_norm",        HighOpKind::LAYER_NORM},
        {"linear",            HighOpKind::MATMUL},
        {"linear_tiled",      HighOpKind::MATMUL},
        {"causal_attention",  HighOpKind::ATTENTION},
        {"silu",              HighOpKind::SILU},
        {"gelu",              HighOpKind::GELU},
        {"element_mul",       HighOpKind::ELEMENT_MUL},
        {"element_add",       HighOpKind::ELEMENT_ADD},
        {"embedding_lookup",  HighOpKind::EMBEDDING},
        {"softmax",           HighOpKind::SOFTMAX},
        {"rope",              HighOpKind::ROPE},
        {"argmax",            HighOpKind::CUSTOM},
        {"metal_vector_add",  HighOpKind::ELEMENT_ADD},
    };
    auto it = table.find(op_name ? op_name : "");
    return (it != table.end()) ? it->second : HighOpKind::CUSTOM;
}

/* ================================================================== */
/*  lift_nodes_to_nfir — convert a list of (op_name, node_id) pairs    */
/* ================================================================== */

struct DagNodeInfo {
    uint32_t    node_id = 0;
    const char* op_name = nullptr;
    uint32_t    n_inputs  = 0;
    uint32_t    n_outputs = 0;
};

/**
 * Lift a sequence of DAG nodes into a NfirHighGraph.
 * Caller provides node info extracted from PipelineEngine.
 * Returns the populated graph with edges wired sequentially.
 */
inline NfirHighGraph lift_nodes_to_nfir(const DagNodeInfo* nodes, uint32_t count) {
    NfirHighGraph graph;
    if (!nodes || count == 0) return graph;

    /* Create synthetic tensors to wire ops (one output tensor per op) */
    uint32_t next_tensor = 0;

    /* Input tensor for the first op */
    {
        NfirTensorRef t;
        t.name = "input_0";
        t.ndim = 1;
        t.shape[0] = 1;
        next_tensor = graph.add_tensor(t) + 1;
    }

    for (uint32_t i = 0; i < count; ++i) {
        /* Output tensor for this op */
        NfirTensorRef t;
        t.name = "t_" + std::to_string(i);
        t.ndim = 1;
        t.shape[0] = 1;
        uint32_t out_tid = graph.add_tensor(t);

        NfirHighOp op;
        op.kind = map_op_name(nodes[i].op_name);
        op.name = nodes[i].op_name ? nodes[i].op_name : "unknown";
        op.attrs_i["dag_node_id"] = nodes[i].node_id;
        /* Wire: input from previous op's output, output to this tensor */
        if (i == 0) {
            op.input_ids = {0};  /* graph input tensor */
        } else {
            op.input_ids = {out_tid - 1};  /* previous op's output */
        }
        op.output_ids = {out_tid};
        graph.add_op(op);
        next_tensor = out_tid + 1;
    }

    /* Mark last tensor as graph output */
    if (next_tensor > 1) {
        graph.output_tensor_ids.push_back(next_tensor - 1);
    }

    /* Wire sequential edges */
    for (uint32_t i = 0; i + 1 < count; ++i) {
        graph.add_edge(i, i + 1);
    }

    return graph;
}

/**
 * Convenience: lift a StepGraph by extracting node info from layer structure.
 * Takes parallel arrays of node_ids and op_names (from PipelineEngine).
 */
inline NfirHighGraph lift_step_graph(const uint32_t* node_ids,
                                     const char* const* op_names,
                                     uint32_t count) {
    std::vector<DagNodeInfo> infos(count);
    for (uint32_t i = 0; i < count; ++i) {
        infos[i].node_id = node_ids[i];
        infos[i].op_name = op_names[i];
    }
    return lift_nodes_to_nfir(infos.data(), count);
}

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::map_op_name;
    using neuralOS::compiler::DagNodeInfo;
    using neuralOS::compiler::lift_nodes_to_nfir;
    using neuralOS::compiler::lift_step_graph;
}}

#endif // NEURALOS_COMPILER_DAG_TO_NFIR_HPP
