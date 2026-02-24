/**
 * @file nfir_annotate.hpp
 * @brief NeuralOS compiler — NFIR -> DAG Annotation (feedback compiler results)
 *
 * Phase 45B: After CompilerPipeline optimizes the NfirHighGraph,
 * annotate_dag_from_nfir() feeds results back to the DAG:
 *   - Marks DCE-removed nodes as skip
 *   - Marks fusion candidates for PipelineEngine dispatch merging
 *
 * Does NOT modify DAG structure — only adds annotations.
 * Header-only.
 */

#ifndef NEURALOS_COMPILER_NFIR_ANNOTATE_HPP
#define NEURALOS_COMPILER_NFIR_ANNOTATE_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neuralOS { namespace compiler {

/* ================================================================== */
/*  DagAnnotation — per-node optimization hints                        */
/* ================================================================== */

struct DagAnnotation {
    bool     skip       = false;   /* DCE: node output unused */
    bool     fused      = false;   /* Fusion: part of a fused group */
    uint32_t fuse_group = 0;       /* Fusion group ID (0 = none) */
};

/* ================================================================== */
/*  annotate_from_nfir — map compiler results back to DAG node IDs     */
/* ================================================================== */

struct AnnotationResult {
    std::unordered_map<uint32_t, DagAnnotation> annotations;  /* dag_node_id → annotation */
    uint32_t nodes_skipped = 0;
    uint32_t nodes_fused   = 0;
};

/**
 * Build annotations from an optimized NfirHighGraph.
 * Each NfirHighOp must have attrs_i["dag_node_id"] set by lift_nodes_to_nfir().
 */
inline AnnotationResult annotate_from_nfir(const NfirHighGraph& graph) {
    AnnotationResult result;

    /* Collect live op IDs (ops that survived DCE) */
    std::unordered_set<uint32_t> live_ops;
    for (auto& op : graph.ops) {
        live_ops.insert(op.id);
    }

    /* Mark fusion candidates */
    uint32_t fuse_group_id = 1;
    std::unordered_set<uint32_t> fused_op_ids;
    for (auto& fc : graph.fusion_candidates) {
        fused_op_ids.insert(fc.op_a);
        fused_op_ids.insert(fc.op_b);
    }

    /* Build annotations for each op */
    for (auto& op : graph.ops) {
        auto dag_it = op.attrs_i.find("dag_node_id");
        if (dag_it == op.attrs_i.end()) continue;
        uint32_t dag_id = static_cast<uint32_t>(dag_it->second);

        DagAnnotation ann;
        if (op.fusible || fused_op_ids.count(op.id)) {
            ann.fused = true;
            ann.fuse_group = fuse_group_id;  /* simplified: all fusions share group */
        }
        result.annotations[dag_id] = ann;
    }

    /* Count fused nodes */
    for (auto& [id, ann] : result.annotations) {
        if (ann.fused) result.nodes_fused++;
    }

    /* Assign fusion group IDs per fusion candidate pair */
    fuse_group_id = 1;
    for (auto& fc : graph.fusion_candidates) {
        /* Find dag_node_ids for op_a and op_b */
        uint32_t dag_a = UINT32_MAX, dag_b = UINT32_MAX;
        if (fc.op_a < graph.ops.size()) {
            auto it = graph.ops[fc.op_a].attrs_i.find("dag_node_id");
            if (it != graph.ops[fc.op_a].attrs_i.end())
                dag_a = static_cast<uint32_t>(it->second);
        }
        if (fc.op_b < graph.ops.size()) {
            auto it = graph.ops[fc.op_b].attrs_i.find("dag_node_id");
            if (it != graph.ops[fc.op_b].attrs_i.end())
                dag_b = static_cast<uint32_t>(it->second);
        }
        if (dag_a != UINT32_MAX) result.annotations[dag_a].fuse_group = fuse_group_id;
        if (dag_b != UINT32_MAX) result.annotations[dag_b].fuse_group = fuse_group_id;
        ++fuse_group_id;
    }

    return result;
}

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::DagAnnotation;
    using neuralOS::compiler::AnnotationResult;
    using neuralOS::compiler::annotate_from_nfir;
}}

#endif // NEURALOS_COMPILER_NFIR_ANNOTATE_HPP
