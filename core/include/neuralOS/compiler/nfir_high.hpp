/**
 * @file nfir_high.hpp
 * @brief NeuralOS L1 — High-Level NFIR (Semantic Compute Graph)
 *
 * Phase 37.1: High-level IR preserving semantic information.
 *   - NfirHighOp: matmul, attention, ffn_block, moe_block, etc.
 *   - NfirHighGraph: DAG of high-level ops with fusion annotations
 *   - FusionCandidate: marks fusible op pairs
 */

#ifndef NEURALOS_L1_NFIR_HIGH_HPP
#define NEURALOS_L1_NFIR_HIGH_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace neuralOS { namespace L1 {

/* ================================================================== */
/*  NfirHighOp — high-level semantic operator                          */
/* ================================================================== */

enum class HighOpKind : uint8_t {
    MATMUL,
    ATTENTION,
    FFN_BLOCK,
    MOE_BLOCK,
    RMS_NORM,
    LAYER_NORM,
    ROPE,
    SILU,
    GELU,
    ELEMENT_MUL,
    ELEMENT_ADD,
    SOFTMAX,
    DEQUANT,
    EMBEDDING,
    LINEAR,
    QKV_PROJ,
    CUSTOM
};

struct NfirTensorRef {
    uint32_t    id       = 0;
    uint8_t     dtype    = 0;   /* nf_dtype */
    uint8_t     ndim     = 0;
    uint64_t    shape[8] = {};
    uint64_t    size_bytes = 0;
    std::string name;
};

struct NfirHighOp {
    uint32_t    id   = 0;
    HighOpKind  kind = HighOpKind::CUSTOM;
    std::string name;

    std::vector<uint32_t> input_ids;
    std::vector<uint32_t> output_ids;

    /* Scheduling hints */
    uint32_t flops_estimate = 0;
    bool     fusible        = false;

    /* Op-specific attributes (key-value) */
    std::unordered_map<std::string, int64_t> attrs_i;
    std::unordered_map<std::string, float>   attrs_f;
    std::unordered_map<std::string, std::string> attrs_s;
};

/* ================================================================== */
/*  FusionCandidate — marks a pair of ops that can be fused            */
/* ================================================================== */

struct FusionCandidate {
    uint32_t    op_a     = 0;
    uint32_t    op_b     = 0;
    std::string fused_name;     /* e.g. "fused_dq_matmul" */
    float       speedup_est = 1.0f;
};

/* ================================================================== */
/*  NfirHighGraph — DAG of high-level ops                              */
/* ================================================================== */

struct NfirHighGraph {
    std::vector<NfirHighOp>    ops;
    std::vector<NfirTensorRef> tensors;
    std::vector<FusionCandidate> fusion_candidates;
    std::vector<uint32_t>      output_tensor_ids;  /* explicit graph outputs */

    /* Adjacency: op_id → successor op_ids */
    std::unordered_map<uint32_t, std::vector<uint32_t>> edges;

    uint32_t add_tensor(const NfirTensorRef& t) {
        uint32_t id = static_cast<uint32_t>(tensors.size());
        tensors.push_back(t);
        tensors.back().id = id;
        return id;
    }

    uint32_t add_op(const NfirHighOp& op) {
        uint32_t id = static_cast<uint32_t>(ops.size());
        ops.push_back(op);
        ops.back().id = id;
        return id;
    }

    void add_edge(uint32_t from_op, uint32_t to_op) {
        edges[from_op].push_back(to_op);
    }

    const NfirHighOp* find_op(uint32_t id) const {
        return (id < ops.size()) ? &ops[id] : nullptr;
    }

    uint32_t num_ops() const { return static_cast<uint32_t>(ops.size()); }
    uint32_t num_tensors() const { return static_cast<uint32_t>(tensors.size()); }
};

}} // namespace neuralOS::L1

#endif // NEURALOS_L1_NFIR_HIGH_HPP
