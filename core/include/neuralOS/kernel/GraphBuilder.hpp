/**
 * @file GraphBuilder.hpp
 * @brief IR-driven DAG construction — loads .nfir files into PipelineEngine
 *
 * INTERNAL TO CORE — never crosses a dynamic library boundary.
 */

#ifndef NF_GRAPH_BUILDER_HPP
#define NF_GRAPH_BUILDER_HPP

#include "neuralOS/ddi/neuro_ir_format.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace nf {

/** Callback for allocating activation buffers (provided by caller). */
using ActivationAllocFn = std::function<nf_status(
    const nf_tensor_desc& desc, nf_buffer_ops* ops, nf_buffer* buf)>;

class GraphBuilder {
public:
    explicit GraphBuilder(PipelineEngine& engine, ActivationAllocFn alloc_fn);
    ~GraphBuilder();

    /** Load .nfir file: parse header, mmap weights, allocate activations. */
    nf_status load(const char* nfir_path);

    /** Build DAG in PipelineEngine from loaded IR. Returns graph_id. */
    nf_status build(uint32_t* out_graph_id);

    /** Access loaded tensor buffer (for test verification). */
    nf_buffer     get_tensor_buffer(uint32_t tensor_id) const;
    nf_buffer_ops get_tensor_ops(uint32_t tensor_id) const;

private:
    struct TensorSlot {
        uint32_t      tensor_id = 0;
        nf_buffer     buf = nullptr;
        nf_buffer_ops ops{};
        bool          is_weight = false;
    };

    struct LoadedGraph {
        int                              fd = -1;
        nf_ir_header                     header{};
        std::vector<nf_ir_tensor_desc>   tensor_descs;
        std::vector<nf_ir_node_desc>     node_descs;
        std::vector<TensorSlot>          tensors;
    };

    nf_status validate_header(const nf_ir_header& hdr);
    nf_status load_tensors(LoadedGraph& lg);
    nf_status build_dag(LoadedGraph& lg, uint32_t* out_graph_id);

    PipelineEngine&   engine_;
    ActivationAllocFn alloc_fn_;
    LoadedGraph       loaded_;
};

} // namespace nf

#endif // NF_GRAPH_BUILDER_HPP
