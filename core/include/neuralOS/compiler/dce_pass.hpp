/**
 * @file dce_pass.hpp
 * @brief NeuralOS compiler — Dead Code Elimination Pass
 *
 * Phase 42B.1: Removes ops whose outputs are never consumed.
 * Algorithm: mark all ops reachable from graph outputs, remove unmarked.
 */

#ifndef NEURALOS_COMPILER_DCE_PASS_HPP
#define NEURALOS_COMPILER_DCE_PASS_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neuralOS { namespace compiler {

class DCEPass {
public:
    /** Run dead code elimination on a high-level graph.
     *  Returns number of ops removed. */
    uint32_t run(NfirHighGraph* graph) {
        if (!graph || graph->ops.empty()) return 0;

        /* Build reverse use-map: tensor_id → consuming op ids */
        std::unordered_map<uint32_t, std::vector<uint32_t>> tensor_consumers;
        for (auto& op : graph->ops) {
            for (uint32_t tid : op.input_ids)
                tensor_consumers[tid].push_back(op.id);
        }

        /* Find graph output tensors.
         * If explicit output_tensor_ids are set, use those.
         * Otherwise, infer: tensors produced by ops but not consumed by any op. */
        std::unordered_set<uint32_t> output_tensors;
        if (!graph->output_tensor_ids.empty()) {
            for (uint32_t tid : graph->output_tensor_ids)
                output_tensors.insert(tid);
        } else {
            for (auto& op : graph->ops) {
                for (uint32_t tid : op.output_ids) {
                    if (tensor_consumers.find(tid) == tensor_consumers.end() ||
                        tensor_consumers[tid].empty()) {
                        output_tensors.insert(tid);
                    }
                }
            }
        }

        /* If no outputs found, treat last op's outputs as graph outputs */
        if (output_tensors.empty() && !graph->ops.empty()) {
            for (uint32_t tid : graph->ops.back().output_ids)
                output_tensors.insert(tid);
        }

        /* Build tensor_id → producing op_id map */
        std::unordered_map<uint32_t, uint32_t> tensor_producer;
        for (auto& op : graph->ops) {
            for (uint32_t tid : op.output_ids)
                tensor_producer[tid] = op.id;
        }

        /* Walk backward from output tensors, marking reachable ops */
        std::unordered_set<uint32_t> live_ops;
        std::vector<uint32_t> worklist;
        for (uint32_t tid : output_tensors) {
            auto pit = tensor_producer.find(tid);
            if (pit != tensor_producer.end() && !live_ops.count(pit->second)) {
                live_ops.insert(pit->second);
                worklist.push_back(pit->second);
            }
        }

        while (!worklist.empty()) {
            uint32_t op_id = worklist.back();
            worklist.pop_back();
            if (op_id >= graph->ops.size()) continue;
            auto& op = graph->ops[op_id];
            for (uint32_t tid : op.input_ids) {
                auto pit = tensor_producer.find(tid);
                if (pit != tensor_producer.end() && !live_ops.count(pit->second)) {
                    live_ops.insert(pit->second);
                    worklist.push_back(pit->second);
                }
            }
        }

        /* Remove dead ops */
        uint32_t removed = 0;
        std::vector<NfirHighOp> live;
        for (auto& op : graph->ops) {
            if (live_ops.count(op.id)) {
                live.push_back(std::move(op));
            } else {
                /* Remove edges from/to this op */
                graph->edges.erase(op.id);
                ++removed;
            }
        }
        graph->ops = std::move(live);

        /* Clean up edges referencing removed ops */
        for (auto it = graph->edges.begin(); it != graph->edges.end(); ) {
            auto& succs = it->second;
            succs.erase(std::remove_if(succs.begin(), succs.end(),
                [&live_ops](uint32_t id) { return !live_ops.count(id); }),
                succs.end());
            if (succs.empty())
                it = graph->edges.erase(it);
            else
                ++it;
        }

        return removed;
    }
};

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::DCEPass;
}}

#endif // NEURALOS_COMPILER_DCE_PASS_HPP