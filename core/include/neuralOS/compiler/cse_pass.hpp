/**
 * @file cse_pass.hpp
 * @brief NeuralOS L1 — Common Subexpression Elimination Pass
 *
 * Phase 42B.2: Merges ops with identical (kind, input_ids, attrs).
 * Hash each op signature, detect collisions, redirect consumers.
 */

#ifndef NEURALOS_L1_CSE_PASS_HPP
#define NEURALOS_L1_CSE_PASS_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neuralOS { namespace L1 {

class CSEPass {
public:
    /** Run CSE on a high-level graph.
     *  Returns number of duplicate ops removed. */
    uint32_t run(NfirHighGraph* graph) {
        if (!graph || graph->ops.empty()) return 0;

        /* Build op signature → first op_id map */
        std::unordered_map<std::string, uint32_t> sig_to_op;
        /* Mapping: duplicate op_id → canonical op_id */
        std::unordered_map<uint32_t, uint32_t> redirect;

        for (auto& op : graph->ops) {
            std::string sig = make_signature(op);
            auto it = sig_to_op.find(sig);
            if (it != sig_to_op.end()) {
                /* Duplicate found — redirect outputs */
                redirect[op.id] = it->second;
            } else {
                sig_to_op[sig] = op.id;
            }
        }

        if (redirect.empty()) return 0;

        /* Build output tensor redirect: dup_output → canonical_output */
        std::unordered_map<uint32_t, uint32_t> tensor_redirect;
        for (auto& [dup_id, canon_id] : redirect) {
            auto* dup_op = graph->find_op(dup_id);
            auto* canon_op = graph->find_op(canon_id);
            if (!dup_op || !canon_op) continue;
            size_t n = std::min(dup_op->output_ids.size(),
                                canon_op->output_ids.size());
            for (size_t i = 0; i < n; ++i)
                tensor_redirect[dup_op->output_ids[i]] = canon_op->output_ids[i];
        }

        /* Redirect inputs of all ops that consume duplicate outputs */
        for (auto& op : graph->ops) {
            if (redirect.count(op.id)) continue;
            for (auto& tid : op.input_ids) {
                auto it = tensor_redirect.find(tid);
                if (it != tensor_redirect.end())
                    tid = it->second;
            }
        }

        /* Remove duplicate ops */
        std::unordered_set<uint32_t> dead(redirect.size());
        for (auto& [dup_id, _] : redirect)
            dead.insert(dup_id);

        std::vector<NfirHighOp> live;
        for (auto& op : graph->ops) {
            if (!dead.count(op.id))
                live.push_back(std::move(op));
        }
        uint32_t removed = static_cast<uint32_t>(graph->ops.size() - live.size());
        graph->ops = std::move(live);

        /* Clean up edges */
        for (auto it = graph->edges.begin(); it != graph->edges.end(); ) {
            if (dead.count(it->first)) {
                it = graph->edges.erase(it);
            } else {
                auto& succs = it->second;
                succs.erase(std::remove_if(succs.begin(), succs.end(),
                    [&dead](uint32_t id) { return dead.count(id) > 0; }),
                    succs.end());
                ++it;
            }
        }

        return removed;
    }

private:
    std::string make_signature(const NfirHighOp& op) const {
        std::string sig;
        sig += std::to_string(static_cast<uint8_t>(op.kind));
        sig += ":";
        for (uint32_t tid : op.input_ids) {
            sig += std::to_string(tid);
            sig += ",";
        }
        sig += "|";
        for (auto& [k, v] : op.attrs_i) {
            sig += k;
            sig += "=";
            sig += std::to_string(v);
            sig += ";";
        }
        for (auto& [k, v] : op.attrs_f) {
            sig += k;
            sig += "=";
            sig += std::to_string(v);
            sig += ";";
        }
        return sig;
    }
};

}} // namespace neuralOS::L1

#endif // NEURALOS_L1_CSE_PASS_HPP