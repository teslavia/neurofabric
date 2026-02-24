/**
 * @file fusion_pass.hpp
 * @brief NeuralOS L1 — Operator Fusion Pass
 *
 * Phase 37.2: Pattern-matching fusion on NfirHighGraph.
 *   - dequant + matmul → fused_dq_matmul
 *   - rms_norm + rope → fused_norm_rope
 *   - silu + element_mul → fused_swiglu
 *   - attention_q + attention_k + attention_v → fused_qkv_proj
 * Phase 42B.3: Chain fusion (multi-op greedy longest match).
 */

#ifndef NEURALOS_L1_FUSION_PASS_HPP
#define NEURALOS_L1_FUSION_PASS_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

namespace neuralOS { namespace L1 {

/* ================================================================== */
/*  FusionPattern — describes a fusible op pair                        */
/* ================================================================== */

struct FusionPattern {
    HighOpKind  first;
    HighOpKind  second;
    std::string fused_name;
    float       speedup_est;
};

/* ================================================================== */
/*  ChainPattern — describes a multi-op fusible chain                  */
/* ================================================================== */

struct ChainPattern {
    std::vector<HighOpKind> chain;
    std::string fused_name;
    float       speedup_est;
};

/* ================================================================== */
/*  FusionPass — identifies and marks fusible patterns                 */
/* ================================================================== */

class FusionPass {
public:
    FusionPass() {
        /* Built-in pair patterns */
        patterns_.push_back({HighOpKind::DEQUANT, HighOpKind::MATMUL,
                             "fused_dq_matmul", 1.8f});
        patterns_.push_back({HighOpKind::RMS_NORM, HighOpKind::ROPE,
                             "fused_norm_rope", 1.3f});
        patterns_.push_back({HighOpKind::SILU, HighOpKind::ELEMENT_MUL,
                             "fused_swiglu", 1.5f});

        /* Built-in chain patterns */
        chain_patterns_.push_back({
            {HighOpKind::RMS_NORM, HighOpKind::LINEAR, HighOpKind::SILU, HighOpKind::ELEMENT_MUL},
            "fused_ffn_block", 2.0f});
        chain_patterns_.push_back({
            {HighOpKind::MATMUL, HighOpKind::SOFTMAX, HighOpKind::MATMUL},
            "fused_attention_core", 1.6f});
    }

    /** Add a custom pair fusion pattern */
    void add_pattern(HighOpKind first, HighOpKind second,
                     const std::string& fused_name, float speedup = 1.0f) {
        patterns_.push_back({first, second, fused_name, speedup});
    }

    /** Add a custom chain fusion pattern */
    void add_chain_pattern(const std::vector<HighOpKind>& chain,
                           const std::string& fused_name, float speedup = 1.0f) {
        chain_patterns_.push_back({chain, fused_name, speedup});
    }

    /** Run fusion pass on a high-level graph.
     *  Tries chain patterns first (greedy longest match), then pair patterns.
     *  Returns number of fusions found. */
    uint32_t run(NfirHighGraph* graph) {
        if (!graph) return 0;
        graph->fusion_candidates.clear();

        std::unordered_set<uint32_t> fused;
        uint32_t count = 0;

        /* Phase 1: Chain fusion (greedy longest match) */
        for (auto& op : graph->ops) {
            if (fused.count(op.id)) continue;

            for (auto& cp : chain_patterns_) {
                if (cp.chain.empty()) continue;
                if (op.kind != cp.chain[0]) continue;

                /* Try to match the full chain */
                std::vector<uint32_t> chain_ids;
                chain_ids.push_back(op.id);
                uint32_t cur_id = op.id;
                bool matched = true;

                for (size_t ci = 1; ci < cp.chain.size(); ++ci) {
                    auto eit = graph->edges.find(cur_id);
                    if (eit == graph->edges.end() || eit->second.empty()) {
                        matched = false;
                        break;
                    }
                    /* Find a successor matching the next kind */
                    bool found = false;
                    for (uint32_t succ_id : eit->second) {
                        if (fused.count(succ_id)) continue;
                        if (succ_id >= graph->ops.size()) continue;
                        if (graph->ops[succ_id].kind == cp.chain[ci]) {
                            chain_ids.push_back(succ_id);
                            cur_id = succ_id;
                            found = true;
                            break;
                        }
                    }
                    if (!found) { matched = false; break; }
                }

                if (matched && chain_ids.size() == cp.chain.size()) {
                    /* Record as fusion candidate (first→last) */
                    FusionCandidate fc;
                    fc.op_a = chain_ids.front();
                    fc.op_b = chain_ids.back();
                    fc.fused_name = cp.fused_name;
                    fc.speedup_est = cp.speedup_est;
                    graph->fusion_candidates.push_back(fc);

                    for (uint32_t cid : chain_ids) {
                        graph->ops[cid].fusible = true;
                        fused.insert(cid);
                    }
                    ++count;
                    break;  /* matched a chain, move to next op */
                }
            }
        }

        /* Phase 2: Pair fusion (fallback for unchained ops) */
        for (auto& op : graph->ops) {
            if (fused.count(op.id)) continue;

            auto eit = graph->edges.find(op.id);
            if (eit == graph->edges.end()) continue;

            for (uint32_t succ_id : eit->second) {
                if (fused.count(succ_id)) continue;
                if (succ_id >= graph->ops.size()) continue;

                auto& succ = graph->ops[succ_id];

                for (auto& pat : patterns_) {
                    if (op.kind == pat.first && succ.kind == pat.second) {
                        FusionCandidate fc;
                        fc.op_a = op.id;
                        fc.op_b = succ_id;
                        fc.fused_name = pat.fused_name;
                        fc.speedup_est = pat.speedup_est;
                        graph->fusion_candidates.push_back(fc);

                        op.fusible = true;
                        succ.fusible = true;
                        fused.insert(op.id);
                        fused.insert(succ_id);
                        ++count;
                        break;
                    }
                }
            }
        }
        return count;
    }

    size_t num_patterns() const { return patterns_.size(); }
    size_t num_chain_patterns() const { return chain_patterns_.size(); }

private:
    std::vector<FusionPattern> patterns_;
    std::vector<ChainPattern>  chain_patterns_;
};

}} // namespace neuralOS::L1

#endif // NEURALOS_L1_FUSION_PASS_HPP
