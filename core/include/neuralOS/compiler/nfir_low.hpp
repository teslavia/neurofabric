/**
 * @file nfir_low.hpp
 * @brief NeuralOS compiler — Low-Level NFIR (Tensor Instructions)
 *
 * Phase 37.1: Low-level IR mapping directly to nf_task_desc.
 *   - NfirLowOp: corresponds 1:1 with PipelineEngine tasks
 *   - NfirLowGraph: execution-ready DAG
 *   - lower(): High -> Low IR transformation
 */

#ifndef NEURALOS_COMPILER_NFIR_LOW_HPP
#define NEURALOS_COMPILER_NFIR_LOW_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace neuralOS { namespace compiler {

/* ================================================================== */
/*  NfirLowOp — low-level tensor instruction                          */
/* ================================================================== */

struct NfirLowOp {
    uint32_t    id   = 0;
    std::string op_name;    /* matches nf_task_desc.op_name */

    std::vector<uint32_t> input_tensor_ids;
    std::vector<uint32_t> output_tensor_ids;

    /* Push constants (serialized) */
    uint32_t push_constants_size = 0;
    uint8_t  push_constants[64] = {};

    /* Scheduling */
    uint32_t affinity = 0;  /* nf_affinity */
    uint32_t flags    = 0;  /* nf_task_flags */
};

/* ================================================================== */
/*  NfirLowGraph — execution-ready DAG                                 */
/* ================================================================== */

struct NfirLowGraph {
    std::vector<NfirLowOp>     ops;
    std::vector<NfirTensorRef> tensors;

    std::unordered_map<uint32_t, std::vector<uint32_t>> edges;

    uint32_t add_op(const NfirLowOp& op) {
        uint32_t id = static_cast<uint32_t>(ops.size());
        ops.push_back(op);
        ops.back().id = id;
        return id;
    }

    void add_edge(uint32_t from, uint32_t to) {
        edges[from].push_back(to);
    }

    uint32_t num_ops() const { return static_cast<uint32_t>(ops.size()); }
};

/* ================================================================== */
/*  lower() — High-Level IR → Low-Level IR transformation              */
/* ================================================================== */

/** Map HighOpKind to canonical op_name string */
inline const char* high_op_to_name(HighOpKind kind) {
    switch (kind) {
        case HighOpKind::MATMUL:      return "matmul";
        case HighOpKind::ATTENTION:   return "causal_attention";
        case HighOpKind::FFN_BLOCK:   return "ffn_block";
        case HighOpKind::MOE_BLOCK:   return "moe_block";
        case HighOpKind::RMS_NORM:    return "rms_norm";
        case HighOpKind::LAYER_NORM:  return "layer_norm";
        case HighOpKind::ROPE:        return "rope";
        case HighOpKind::SILU:        return "silu";
        case HighOpKind::GELU:        return "gelu";
        case HighOpKind::ELEMENT_MUL: return "element_mul";
        case HighOpKind::ELEMENT_ADD: return "element_add";
        case HighOpKind::SOFTMAX:     return "softmax";
        case HighOpKind::DEQUANT:     return "dequant";
        case HighOpKind::EMBEDDING:   return "embedding";
        case HighOpKind::LINEAR:      return "linear";
        case HighOpKind::QKV_PROJ:    return "qkv_proj";
        case HighOpKind::CUSTOM:      return "custom";
    }
    return "unknown";
}

/** Lower a NfirHighGraph to NfirLowGraph.
 *  Each high-level op becomes one or more low-level ops.
 *  Fused ops (from fusion_candidates) are merged. */
inline NfirLowGraph lower(const NfirHighGraph& high) {
    NfirLowGraph low;
    low.tensors = high.tensors;

    /* Build fusion map: op_a → fused_name (op_b gets absorbed) */
    std::unordered_map<uint32_t, std::string> fuse_map;
    std::unordered_map<uint32_t, uint32_t> absorbed;  /* op_b → op_a */
    for (auto& fc : high.fusion_candidates) {
        fuse_map[fc.op_a] = fc.fused_name;
        absorbed[fc.op_b] = fc.op_a;
    }

    /* Map high op id → low op id */
    std::unordered_map<uint32_t, uint32_t> id_map;

    for (auto& hop : high.ops) {
        /* Skip absorbed ops */
        if (absorbed.count(hop.id)) continue;

        NfirLowOp lop;
        auto fit = fuse_map.find(hop.id);
        if (fit != fuse_map.end()) {
            lop.op_name = fit->second;
            /* Merge inputs from both ops */
            lop.input_tensor_ids = hop.input_ids;
            /* Find the absorbed op and add its unique inputs */
            for (auto& [absorbed_id, parent_id] : absorbed) {
                if (parent_id == hop.id) {
                    auto* abs_op = high.find_op(absorbed_id);
                    if (abs_op) {
                        for (auto tid : abs_op->output_ids)
                            lop.output_tensor_ids.push_back(tid);
                    }
                }
            }
            if (lop.output_tensor_ids.empty())
                lop.output_tensor_ids = hop.output_ids;
        } else {
            lop.op_name = hop.name.empty() ? high_op_to_name(hop.kind) : hop.name;
            lop.input_tensor_ids = hop.input_ids;
            lop.output_tensor_ids = hop.output_ids;
        }

        uint32_t lid = low.add_op(lop);
        id_map[hop.id] = lid;
    }

    /* Remap edges */
    for (auto& [from, tos] : high.edges) {
        if (!id_map.count(from)) continue;
        for (auto to : tos) {
            /* If 'to' was absorbed, skip the edge */
            if (absorbed.count(to)) continue;
            if (!id_map.count(to)) continue;
            low.add_edge(id_map[from], id_map[to]);
        }
    }

    return low;
}

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::NfirLowOp;
    using neuralOS::compiler::NfirLowGraph;
    using neuralOS::compiler::high_op_to_name;
    using neuralOS::compiler::lower;
}}

#endif // NEURALOS_COMPILER_NFIR_LOW_HPP
