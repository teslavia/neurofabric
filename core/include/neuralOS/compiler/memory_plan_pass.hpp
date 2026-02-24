/**
 * @file memory_plan_pass.hpp
 * @brief NeuralOS compiler — Static Memory Planning Pass
 *
 * Phase 37.3: Analyzes tensor lifetimes and computes optimal allocation.
 *   - Liveness analysis on NfirLowGraph
 *   - Buffer reuse via interval graph coloring
 *   - Peak memory estimation
 */

#ifndef NEURALOS_COMPILER_MEMORY_PLAN_PASS_HPP
#define NEURALOS_COMPILER_MEMORY_PLAN_PASS_HPP

#include "neuralOS/compiler/nfir_low.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace neuralOS { namespace compiler {

/* ================================================================== */
/*  MemoryPlan — output of the planning pass                           */
/* ================================================================== */

struct TensorAllocation {
    uint32_t tensor_id  = 0;
    uint64_t offset     = 0;    /* byte offset in unified buffer */
    uint64_t size_bytes = 0;
};

struct MemoryPlan {
    std::vector<TensorAllocation> allocations;
    uint64_t peak_bytes = 0;
    uint32_t num_reused = 0;  /* tensors sharing buffer space */
};

/* ================================================================== */
/*  LiveInterval — [first_use, last_use] for a tensor                  */
/* ================================================================== */

struct LiveInterval {
    uint32_t tensor_id = 0;
    uint32_t first_use = UINT32_MAX;
    uint32_t last_use  = 0;
    uint64_t size_bytes = 0;
};

/* ================================================================== */
/*  MemoryPlanPass — liveness analysis + buffer reuse                  */
/* ================================================================== */

class MemoryPlanPass {
public:
    /** Run memory planning on a low-level graph.
     *  Returns a MemoryPlan with tensor→offset mappings. */
    MemoryPlan run(const NfirLowGraph& graph) {
        /* Step 1: Compute live intervals */
        auto intervals = compute_liveness(graph);

        /* Step 2: Sort by size descending (large tensors first) */
        std::sort(intervals.begin(), intervals.end(),
            [](const LiveInterval& a, const LiveInterval& b) {
                return a.size_bytes > b.size_bytes;
            });

        /* Step 3: Greedy interval coloring (first-fit decreasing) */
        return greedy_allocate(intervals);
    }

private:
    std::vector<LiveInterval> compute_liveness(const NfirLowGraph& graph) {
        std::unordered_map<uint32_t, LiveInterval> intervals;

        /* Initialize from tensor refs */
        for (auto& t : graph.tensors) {
            intervals[t.id] = {t.id, UINT32_MAX, 0, t.size_bytes};
        }

        /* Scan ops in order to find first/last use */
        for (uint32_t i = 0; i < graph.ops.size(); ++i) {
            auto& op = graph.ops[i];
            for (auto tid : op.input_tensor_ids) {
                auto it = intervals.find(tid);
                if (it != intervals.end()) {
                    it->second.first_use = std::min(it->second.first_use, i);
                    it->second.last_use  = std::max(it->second.last_use, i);
                }
            }
            for (auto tid : op.output_tensor_ids) {
                auto it = intervals.find(tid);
                if (it != intervals.end()) {
                    it->second.first_use = std::min(it->second.first_use, i);
                    it->second.last_use  = std::max(it->second.last_use, i);
                }
            }
        }

        std::vector<LiveInterval> result;
        for (auto& [id, iv] : intervals) {
            if (iv.first_use != UINT32_MAX)
                result.push_back(iv);
        }
        return result;
    }

    MemoryPlan greedy_allocate(const std::vector<LiveInterval>& intervals) {
        MemoryPlan plan;

        /* Each "slot" is a (offset, size, last_use) tuple */
        struct Slot {
            uint64_t offset    = 0;
            uint64_t size      = 0;
            uint32_t last_use  = 0;
        };
        std::vector<Slot> slots;

        for (auto& iv : intervals) {
            bool reused = false;

            /* Try to reuse an expired slot */
            for (auto& slot : slots) {
                if (slot.last_use < iv.first_use && slot.size >= iv.size_bytes) {
                    /* Reuse this slot */
                    plan.allocations.push_back({iv.tensor_id, slot.offset, iv.size_bytes});
                    slot.last_use = iv.last_use;
                    ++plan.num_reused;
                    reused = true;
                    break;
                }
            }

            if (!reused) {
                /* Allocate new space at the end */
                uint64_t offset = plan.peak_bytes;
                plan.allocations.push_back({iv.tensor_id, offset, iv.size_bytes});
                plan.peak_bytes += iv.size_bytes;
                slots.push_back({offset, iv.size_bytes, iv.last_use});
            }
        }

        return plan;
    }
};

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::TensorAllocation;
    using neuralOS::compiler::MemoryPlan;
    using neuralOS::compiler::LiveInterval;
    using neuralOS::compiler::MemoryPlanPass;
}}

#endif // NEURALOS_COMPILER_MEMORY_PLAN_PASS_HPP
