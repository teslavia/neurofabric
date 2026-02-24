/**
 * @file shape_inference_pass.hpp
 * @brief NeuralOS compiler — Shape Inference Pass
 *
 * Phase 42B.4: Propagates shapes through ops in NfirHighGraph.
 * Rules per HighOpKind for MATMUL, element-wise, norms, attention.
 */

#ifndef NEURALOS_COMPILER_SHAPE_INFERENCE_PASS_HPP
#define NEURALOS_COMPILER_SHAPE_INFERENCE_PASS_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <algorithm>
#include <cstdint>

namespace neuralOS { namespace compiler {

class ShapeInferencePass {
public:
    /** Run shape inference. Returns number of tensors with inferred shapes. */
    uint32_t run(NfirHighGraph* graph) {
        if (!graph) return 0;
        uint32_t inferred = 0;

        for (auto& op : graph->ops) {
            switch (op.kind) {
            case HighOpKind::MATMUL:
                inferred += infer_matmul(graph, op);
                break;
            case HighOpKind::ELEMENT_MUL:
            case HighOpKind::ELEMENT_ADD:
                inferred += infer_broadcast(graph, op);
                break;
            case HighOpKind::RMS_NORM:
            case HighOpKind::LAYER_NORM:
            case HighOpKind::SILU:
            case HighOpKind::GELU:
            case HighOpKind::SOFTMAX:
                inferred += infer_same_shape(graph, op);
                break;
            case HighOpKind::ATTENTION:
                inferred += infer_attention(graph, op);
                break;
            case HighOpKind::LINEAR:
                inferred += infer_linear(graph, op);
                break;
            default:
                break;
            }
        }
        return inferred;
    }

private:
    NfirTensorRef* get_tensor(NfirHighGraph* g, uint32_t id) {
        return (id < g->tensors.size()) ? &g->tensors[id] : nullptr;
    }

    /** MATMUL: [M,K] × [K,N] → [M,N] */
    uint32_t infer_matmul(NfirHighGraph* g, const NfirHighOp& op) {
        if (op.input_ids.size() < 2 || op.output_ids.empty()) return 0;
        auto* a = get_tensor(g, op.input_ids[0]);
        auto* b = get_tensor(g, op.input_ids[1]);
        auto* out = get_tensor(g, op.output_ids[0]);
        if (!a || !b || !out) return 0;
        if (a->ndim < 2 || b->ndim < 2) return 0;

        out->ndim = a->ndim;
        for (uint8_t i = 0; i < a->ndim - 2; ++i)
            out->shape[i] = a->shape[i];
        out->shape[a->ndim - 2] = a->shape[a->ndim - 2]; /* M */
        out->shape[a->ndim - 1] = b->shape[b->ndim - 1]; /* N */
        out->dtype = a->dtype;
        return 1;
    }

    /** Element-wise: broadcast rules */
    uint32_t infer_broadcast(NfirHighGraph* g, const NfirHighOp& op) {
        if (op.input_ids.size() < 2 || op.output_ids.empty()) return 0;
        auto* a = get_tensor(g, op.input_ids[0]);
        auto* b = get_tensor(g, op.input_ids[1]);
        auto* out = get_tensor(g, op.output_ids[0]);
        if (!a || !b || !out) return 0;

        out->ndim = std::max(a->ndim, b->ndim);
        for (uint8_t i = 0; i < out->ndim; ++i) {
            uint64_t da = (i < a->ndim) ? a->shape[i] : 1;
            uint64_t db = (i < b->ndim) ? b->shape[i] : 1;
            out->shape[i] = std::max(da, db);
        }
        out->dtype = a->dtype;
        return 1;
    }

    /** Same shape: output = input[0] shape */
    uint32_t infer_same_shape(NfirHighGraph* g, const NfirHighOp& op) {
        if (op.input_ids.empty() || op.output_ids.empty()) return 0;
        auto* in = get_tensor(g, op.input_ids[0]);
        auto* out = get_tensor(g, op.output_ids[0]);
        if (!in || !out) return 0;

        out->ndim = in->ndim;
        for (uint8_t i = 0; i < in->ndim; ++i)
            out->shape[i] = in->shape[i];
        out->dtype = in->dtype;
        return 1;
    }

    /** ATTENTION: [B,H,S,D] → [B,H,S,D] */
    uint32_t infer_attention(NfirHighGraph* g, const NfirHighOp& op) {
        if (op.input_ids.empty() || op.output_ids.empty()) return 0;
        auto* q = get_tensor(g, op.input_ids[0]);
        auto* out = get_tensor(g, op.output_ids[0]);
        if (!q || !out) return 0;

        out->ndim = q->ndim;
        for (uint8_t i = 0; i < q->ndim; ++i)
            out->shape[i] = q->shape[i];
        out->dtype = q->dtype;
        return 1;
    }

    /** LINEAR: [M,K] × [K,N] → [M,N] (same as matmul) */
    uint32_t infer_linear(NfirHighGraph* g, const NfirHighOp& op) {
        return infer_matmul(g, op);
    }
};

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::ShapeInferencePass;
}}

#endif // NEURALOS_COMPILER_SHAPE_INFERENCE_PASS_HPP