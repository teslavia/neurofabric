/**
 * @file NeuralOSRuntime.hpp
 * @brief NeuralOS kernel — Unified Kernel Orchestrator
 *
 * Phase 43A.1: Composes vMMU, CFS, SpecEngine, ContextHub into a single runtime.
 * Wires cross-layer integration and provides unified submit/schedule/pressure API.
 *
 * Header-only. Wraps external PagedKVCache + RequestScheduler.
 */

#ifndef NEURALOS_KERNEL_RUNTIME_HPP
#define NEURALOS_KERNEL_RUNTIME_HPP

#include "neuralOS/kernel/vMMU.hpp"
#include "neuralOS/kernel/CFS.hpp"
#include "neuralOS/kernel/SpecEngine.hpp"
#include "neuralOS/kernel/ContextHub.hpp"
#include "neuralOS/kernel/kv_cache.hpp"
#include "neuralOS/kernel/request_scheduler.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace neuralOS { namespace kernel {

struct RuntimeConfig {
    vMMU::Config          vmmu_cfg;
    CFS::Config           cfs_cfg;
    bool                  enable_spec     = false;
    nf::SpeculativeConfig spec_cfg;
    uint64_t              context_budget  = 64 * 1024 * 1024;
    nf_eviction_policy    eviction_policy = NF_EVICT_LRU;
};

struct ScheduleResult {
    std::vector<uint32_t> selected_ids;
    std::vector<uint32_t> preempted_ids;
    uint32_t              total_tokens = 0;
};

class NeuralOSRuntime {
public:
    struct Stats {
        uint64_t total_submitted  = 0;
        uint64_t total_completed  = 0;
        uint64_t total_preempted  = 0;
        uint32_t pages_out        = 0;
        uint32_t pages_in         = 0;
        uint32_t prefix_hits      = 0;
    };

    NeuralOSRuntime(nf::PagedKVCache* kv, nf::RequestScheduler* sched, RuntimeConfig cfg = {})
        : kv_(kv), sched_(sched), cfg_(cfg)
    {
        hub_ = std::make_unique<nf::ContextHub>(cfg_.context_budget, cfg_.eviction_policy);
        vmmu_ = std::make_unique<vMMU>(kv_, hub_.get(), cfg_.vmmu_cfg);
        cfs_ = std::make_unique<CFS>(sched_, kv_, cfg_.cfs_cfg);
        cfs_->set_vmmu(vmmu_.get());

        if (cfg_.enable_spec) {
            spec_ = std::make_unique<SpecEngine>(cfg_.spec_cfg);
            spec_->set_vmmu(vmmu_.get(), kv_);
        }
        /* NOTE: We do NOT use vmmu_->set_pressure_callback() to avoid
         * deadlock (callback re-enters vMMU::mu_ via CFS::preempt→page_out).
         * Instead, check_pressure() does manual threshold check. */
    }

    uint32_t submit(nf::InferenceRequest req) {
        if (!req.prompt_tokens.empty()) {
            auto result = hub_->get(
                std::span<const int32_t>(req.prompt_tokens.data(),
                                         req.prompt_tokens.size()));
            if (result.found && result.match_len > 0)
                stats_.prefix_hits++;
        }
        uint32_t id = sched_->submit(std::move(req));
        if (id != UINT32_MAX) stats_.total_submitted++;
        return id;
    }

    ScheduleResult schedule_step() {
        auto rb = cfs_->rebalance();
        ScheduleResult result;
        result.selected_ids  = rb.selected_req_ids;
        result.preempted_ids = rb.preempted_req_ids;
        result.total_tokens  = rb.total_tokens;
        stats_.total_preempted += rb.preempted_req_ids.size();
        return result;
    }

    void check_pressure() {
        /* Manual pressure check — avoids deadlock from callback re-entry */
        uint32_t used  = kv_->allocator.num_used();
        uint32_t total = kv_->allocator.num_blocks;
        if (total == 0) return;
        float ratio = static_cast<float>(used) / static_cast<float>(total);
        if (ratio < cfg_.vmmu_cfg.pressure_threshold) return;

        /* Find highest vruntime active request to preempt */
        uint64_t max_vrt = 0;
        uint32_t victim_id = UINT32_MAX;
        for (uint32_t i = 0; i < sched_->num_requests; ++i) {
            auto& r = sched_->requests[i];
            if (r.state == nf::RequestState::DECODE ||
                r.state == nf::RequestState::PREFILL) {
                uint64_t vrt = cfs_->get_vruntime(r.req_id);
                if (vrt >= max_vrt) {
                    max_vrt = vrt;
                    victim_id = r.req_id;
                }
            }
        }
        if (victim_id != UINT32_MAX) {
            cfs_->preempt(victim_id);
            stats_.total_preempted++;
        }
    }

    uint32_t try_prefix_share(const std::vector<int32_t>& tokens) {
        if (tokens.empty()) return 0;
        auto result = hub_->get(
            std::span<const int32_t>(tokens.data(), tokens.size()));
        if (result.found && result.match_len > 0) {
            stats_.prefix_hits++;
            return result.match_len;
        }
        return 0;
    }

    void complete(uint32_t req_id) {
        for (uint32_t i = 0; i < sched_->num_requests; ++i) {
            if (sched_->requests[i].req_id == req_id) {
                sched_->requests[i].state = nf::RequestState::COMPLETE;
                stats_.total_completed++;
                break;
            }
        }
    }

    vMMU*           vmmu()         { return vmmu_.get(); }
    CFS*            cfs()          { return cfs_.get(); }
    nf::ContextHub* context_hub()  { return hub_.get(); }
    SpecEngine*     spec_engine()  { return spec_.get(); }
    const Stats&    stats() const  { return stats_; }

private:
    nf::PagedKVCache*      kv_;
    nf::RequestScheduler*  sched_;
    RuntimeConfig          cfg_;

    std::unique_ptr<nf::ContextHub> hub_;
    std::unique_ptr<vMMU>           vmmu_;
    std::unique_ptr<CFS>            cfs_;
    std::unique_ptr<SpecEngine>     spec_;

    Stats stats_{};
};

}} // namespace neuralOS::kernel

// Backward compatibility
namespace neuralOS { namespace L2 {
    using neuralOS::kernel::RuntimeConfig;
    using neuralOS::kernel::ScheduleResult;
    using neuralOS::kernel::NeuralOSRuntime;
}}

#endif // NEURALOS_KERNEL_RUNTIME_HPP
