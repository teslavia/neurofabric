/**
 * @file onnx_to_nfir.hpp
 * @brief NeuralOS L1 — ONNX Graph → NfirHighGraph Converter
 *
 * Phase 37.4 / 44: Converts parsed OnnxGraph to NfirHighGraph.
 */

#ifndef NEURALOS_ONNX_TO_NFIR_HPP
#define NEURALOS_ONNX_TO_NFIR_HPP

#include "onnx_parser.hpp"
#include "onnx_op_map.hpp"
#include "neuralOS/compiler/nfir_high.hpp"

#include <string>
#include <unordered_map>

namespace neuralOS { namespace onnx {

/** Convert an OnnxGraph to NfirHighGraph.
 *  Returns true on success. */
inline bool onnx_to_nfir(const OnnxGraph& onnx, L1::NfirHighGraph* out) {
    if (!out) return false;

    /* Map ONNX tensor names → NFIR tensor IDs */
    std::unordered_map<std::string, uint32_t> tensor_map;

    auto get_or_create_tensor = [&](const std::string& name) -> uint32_t {
        auto it = tensor_map.find(name);
        if (it != tensor_map.end()) return it->second;
        L1::NfirTensorRef t;
        t.name = name;
        uint32_t id = out->add_tensor(t);
        tensor_map[name] = id;
        return id;
    };

    /* Process initializers (weights) → NfirTensorRef with size_bytes */
    for (auto& init : onnx.initializers) {
        auto& t = out->tensors[get_or_create_tensor(init.name)];
        t.size_bytes = init.raw_data.size();
        /* Map ONNX dtype → nf_dtype (simplified) */
        switch (init.data_type) {
            case ONNX_FLOAT:   t.dtype = 1; break;  /* NF_DTYPE_F32 */
            case ONNX_FLOAT16: t.dtype = 2; break;  /* NF_DTYPE_F16 */
            case ONNX_INT8:    t.dtype = 3; break;  /* NF_DTYPE_I8 */
            default:           t.dtype = 0; break;
        }
        t.ndim = static_cast<uint8_t>(init.dims.size());
        for (uint8_t d = 0; d < t.ndim && d < 8; ++d)
            t.shape[d] = static_cast<uint64_t>(init.dims[d] >= 0 ? init.dims[d] : 0);
    }

    /* Map ONNX node index → NFIR op ID */
    std::unordered_map<uint32_t, uint32_t> op_map;

    /* Convert nodes */
    for (uint32_t i = 0; i < onnx.nodes.size(); ++i) {
        auto& node = onnx.nodes[i];
        auto mapping = lookup_op(node.op_type);

        L1::NfirHighOp hop;
        hop.kind = mapping.kind;
        hop.name = node.name.empty() ? node.op_type : node.name;

        for (auto& inp : node.inputs)
            hop.input_ids.push_back(get_or_create_tensor(inp));
        for (auto& outp : node.outputs)
            hop.output_ids.push_back(get_or_create_tensor(outp));

        uint32_t op_id = out->add_op(hop);
        op_map[i] = op_id;
    }

    /* Build edges: if node B consumes an output of node A, add edge A→B */
    std::unordered_map<std::string, uint32_t> producer_map;
    for (uint32_t i = 0; i < onnx.nodes.size(); ++i) {
        for (auto& outp : onnx.nodes[i].outputs)
            producer_map[outp] = op_map[i];
    }

    for (uint32_t i = 0; i < onnx.nodes.size(); ++i) {
        for (auto& inp : onnx.nodes[i].inputs) {
            auto pit = producer_map.find(inp);
            if (pit != producer_map.end()) {
                out->add_edge(pit->second, op_map[i]);
            }
        }
    }

    return true;
}

}} // namespace neuralOS::onnx

#endif // NEURALOS_ONNX_TO_NFIR_HPP
