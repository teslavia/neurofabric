/**
 * @file GraphBuilder.cpp
 * @brief IR parser, mmap weight loader, DAG instantiation
 */

#include "neuralOS/kernel/GraphBuilder.hpp"
#include "mmap_buffer.h"

#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>

namespace nf {

GraphBuilder::GraphBuilder(PipelineEngine& engine, ActivationAllocFn alloc_fn)
    : engine_(engine), alloc_fn_(std::move(alloc_fn)) {}

GraphBuilder::~GraphBuilder() {
    /* Release all tensor buffers we hold */
    for (auto& slot : loaded_.tensors) {
        if (slot.buf && slot.ops.release) {
            slot.ops.release(slot.buf);
            slot.buf = nullptr;
        }
    }
    if (loaded_.fd >= 0) {
        ::close(loaded_.fd);
        loaded_.fd = -1;
    }
}

nf_status GraphBuilder::validate_header(const nf_ir_header& hdr) {
    if (hdr.magic != NF_IR_MAGIC) return NF_ERROR_INVALID_ARG;
    if (hdr.version != NF_IR_VERSION) return NF_ERROR_ABI_MISMATCH;

    /* Verify CRC — covers everything except last 8 bytes */
    uint32_t expected_crc = hdr.header_crc32;
    uint32_t computed_crc = nf_ir_header_compute_crc(&hdr);
    if (expected_crc != computed_crc) return NF_ERROR_INVALID_ARG;

    return NF_OK;
}

nf_status GraphBuilder::load_tensors(LoadedGraph& lg) {
    for (uint32_t i = 0; i < lg.header.num_tensors; ++i) {
        const auto& td = lg.tensor_descs[i];
        TensorSlot slot;
        slot.tensor_id = td.tensor_id;

        /* Build nf_tensor_desc from IR tensor desc */
        nf_tensor_desc desc{};
        desc.dtype      = static_cast<nf_dtype>(td.dtype);
        desc.ndim       = td.ndim;
        for (uint32_t d = 0; d < td.ndim && d < NF_MAX_DIMS; ++d)
            desc.shape[d] = td.shape[d];
        desc.size_bytes = td.size_bytes;

        if (td.usage == NF_IR_USAGE_WEIGHT) {
            slot.is_weight = true;
            nf_status st = mmap_buffer_create(
                lg.fd,
                lg.header.payload_offset + td.weight_offset,
                td.size_bytes,
                desc, &slot.ops, &slot.buf);
            if (st != NF_OK) return st;
        } else {
            slot.is_weight = false;
            nf_status st = alloc_fn_(desc, &slot.ops, &slot.buf);
            if (st != NF_OK) return st;
        }

        lg.tensors.push_back(std::move(slot));
    }
    return NF_OK;
}

nf_status GraphBuilder::load(const char* nfir_path) {
    int fd = ::open(nfir_path, O_RDONLY);
    if (fd < 0) return NF_ERROR_NOT_FOUND;

    loaded_.fd = fd;

    /* Read header */
    nf_ir_header hdr{};
    if (::read(fd, &hdr, sizeof(hdr)) != static_cast<ssize_t>(sizeof(hdr))) {
        return NF_ERROR_INVALID_ARG;
    }

    nf_status st = validate_header(hdr);
    if (st != NF_OK) return st;
    loaded_.header = hdr;

    /* Read tensor descriptors */
    loaded_.tensor_descs.resize(hdr.num_tensors);
    size_t td_bytes = hdr.num_tensors * sizeof(nf_ir_tensor_desc);
    if (td_bytes > 0) {
        ssize_t r = ::read(fd, loaded_.tensor_descs.data(), td_bytes);
        if (r != static_cast<ssize_t>(td_bytes)) return NF_ERROR_INVALID_ARG;
    }

    /* Read node descriptors */
    loaded_.node_descs.resize(hdr.num_nodes);
    size_t nd_bytes = hdr.num_nodes * sizeof(nf_ir_node_desc);
    if (nd_bytes > 0) {
        ssize_t r = ::read(fd, loaded_.node_descs.data(), nd_bytes);
        if (r != static_cast<ssize_t>(nd_bytes)) return NF_ERROR_INVALID_ARG;
    }

    /* Load tensors: mmap weights, allocate activations */
    return load_tensors(loaded_);
}

nf_status GraphBuilder::build_dag(LoadedGraph& lg, uint32_t* out_graph_id) {
    uint32_t gid = engine_.create_graph();

    /* Map: tensor_id → producing node_id */
    std::unordered_map<uint32_t, uint32_t> tensor_producer;
    for (uint32_t ni = 0; ni < lg.header.num_nodes; ++ni) {
        const auto& nd = lg.node_descs[ni];
        for (uint32_t k = 0; k < nd.n_outputs; ++k) {
            tensor_producer[nd.output_tensor_ids[k]] = ni;
        }
    }

    /* Add tasks to graph */
    std::vector<uint32_t> node_to_task(lg.header.num_nodes);
    for (uint32_t ni = 0; ni < lg.header.num_nodes; ++ni) {
        const auto& nd = lg.node_descs[ni];

        nf_task_desc desc{};
        std::strncpy(desc.op_name, nd.op_type, NF_MAX_OP_NAME - 1);
        desc.op_name[NF_MAX_OP_NAME - 1] = '\0';

        /* Wire inputs */
        desc.n_inputs = nd.n_inputs;
        for (uint32_t j = 0; j < nd.n_inputs && j < NF_MAX_TASK_INPUTS; ++j) {
            uint32_t tid = nd.input_tensor_ids[j];
            if (tid < lg.tensors.size()) {
                desc.inputs[j]    = lg.tensors[tid].buf;
                desc.input_ops[j] = lg.tensors[tid].ops;
            }
        }

        /* Wire outputs */
        desc.n_outputs = nd.n_outputs;
        for (uint32_t j = 0; j < nd.n_outputs && j < NF_MAX_TASK_OUTPUTS; ++j) {
            uint32_t tid = nd.output_tensor_ids[j];
            if (tid < lg.tensors.size()) {
                desc.outputs[j]      = lg.tensors[tid].buf;
                desc.output_ops[j]   = lg.tensors[tid].ops;
            }
        }

        /* Affinity */
        if (nd.task_flags & NF_TASK_REMOTE)
            desc.affinity = NF_AFFINITY_REMOTE;
        else
            desc.affinity = NF_AFFINITY_ANY;

        desc.flags    = nd.task_flags;
        desc.priority = NF_PRIORITY_NORMAL;

        node_to_task[ni] = engine_.add_task(gid, desc);
    }

    /* Add dependency edges based on tensor producer→consumer */
    for (uint32_t ni = 0; ni < lg.header.num_nodes; ++ni) {
        const auto& nd = lg.node_descs[ni];
        for (uint32_t j = 0; j < nd.n_inputs; ++j) {
            uint32_t tid = nd.input_tensor_ids[j];
            auto it = tensor_producer.find(tid);
            if (it != tensor_producer.end() && it->second != ni) {
                engine_.add_edge(gid, node_to_task[it->second],
                                 node_to_task[ni]);
            }
        }
    }

    *out_graph_id = gid;
    return NF_OK;
}

nf_status GraphBuilder::build(uint32_t* out_graph_id) {
    return build_dag(loaded_, out_graph_id);
}

nf_buffer GraphBuilder::get_tensor_buffer(uint32_t tensor_id) const {
    if (tensor_id < loaded_.tensors.size())
        return loaded_.tensors[tensor_id].buf;
    return nullptr;
}

nf_buffer_ops GraphBuilder::get_tensor_ops(uint32_t tensor_id) const {
    if (tensor_id < loaded_.tensors.size())
        return loaded_.tensors[tensor_id].ops;
    return nf_buffer_ops{};
}

} // namespace nf
