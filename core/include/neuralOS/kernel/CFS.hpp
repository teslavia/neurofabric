/**
 * @file CFS.hpp
 * @brief NeuralOS kernel — Completely Fair Scheduler for LLM Inference
 *
 * Phase 36.3: VTC-based fair scheduling with preemption support.
 *   - VirtualTokenCounter tracks per-sequence virtual runtime
 *   - preempt() suspends low-priority tasks, links to vMMU page-out
 *   - resume() restores preempted tasks, links to vMMU page-in
 *   - rebalance() assigns compute budget based on VTC fairness
 *   - migrate_request() interface for cross-instance migration (Phase 38)
 *
 * Header-only. Operates on RequestScheduler + PagedKVCache from model_config.hpp.
 */

#ifndef NEURALOS_KERNEL_CFS_HPP
#define NEURALOS_KERNEL_CFS_HPP

#include "neuralOS/kernel/vMMU.hpp"
#include "neuralOS/kernel/kv_cache.hpp"
#include "neuralOS/kernel/request_scheduler.hpp"
#include "neuralOS/mesh/kv_migration.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace neuralOS { namespace kernel {

/* ================================================================== */
/*  VirtualTokenCounter — per-request fairness tracking                */
/* ================================================================== */

struct VirtualTokenCounter {
    uint64_t vruntime    = 0;   /* virtual runtime in token-units */
    uint64_t weight      = 1;   /* scheduling weight */
    uint64_t tokens_run  = 0;   /* actual tokens processed */

    /** Update vruntime after processing `n` tokens.
     *  vruntime += n / weight (higher weight = slower vruntime growth) */
    void account(uint32_t n_tokens) {
        tokens_run += n_tokens;
        vruntime += (weight > 0) ? (n_tokens * 1024 / weight) : n_tokens;
    }

    /** Reset counters */
    void reset() { vruntime = 0; weight = 1; tokens_run = 0; }
};

/* ================================================================== */
/*  CFS — Completely Fair Scheduler                                    */
/* ================================================================== */

class CFS {
public:
    struct Config {
        uint32_t max_batch_tokens   = 2048;  /* token budget per step */
        uint32_t preempt_threshold  = 4;     /* preempt if vruntime exceeds min by this factor */
        float    prefill_weight     = 2.0f;  /* prefill costs more than decode */
        float    decode_weight      = 1.0f;
    };

    CFS(nf::RequestScheduler* sched, nf::PagedKVCache* kv, Config cfg)
        : sched_(sched), kv_(kv), cfg_(cfg) {
        vtcs_.resize(nf::NF_MAX_REQUESTS);
    }

    CFS(nf::RequestScheduler* sched, nf::PagedKVCache* kv)
        : CFS(sched, kv, Config{}) {}

    /* ---- Cross-Layer Integration ---------------------------------- */

    void set_vmmu(vMMU* v) { vmmu_ = v; }
    void set_migrator(neuralOS::mesh::KVMigrator* m) { migrator_ = m; }

    vMMU* vmmu() const { return vmmu_; }
    uint32_t page_out_count() const { return page_out_count_; }
    uint32_t page_in_count() const { return page_in_count_; }

    /* ---- VTC Management ------------------------------------------- */

    /** Account tokens for a request after a step completes */
    void account_tokens(uint32_t req_id, uint32_t n_tokens, bool is_prefill) {
        std::lock_guard<std::mutex> lk(mu_);
        auto* req = find_request(req_id);
        if (!req) return;
        uint32_t idx = req_index(req_id);
        if (idx == UINT32_MAX) return;

        float cost = is_prefill ? cfg_.prefill_weight : cfg_.decode_weight;
        uint32_t weighted = static_cast<uint32_t>(n_tokens * cost);
        vtcs_[idx].weight = req->vtc_weight;
        vtcs_[idx].account(weighted);
        req->vtc_runtime = vtcs_[idx].vruntime;
    }

    /** Get the minimum vruntime across all active requests */
    uint64_t min_vruntime() const {
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t min_vrt = UINT64_MAX;
        for (uint32_t i = 0; i < sched_->num_requests; ++i) {
            auto& r = sched_->requests[i];
            if (r.state == nf::RequestState::DECODE ||
                r.state == nf::RequestState::PREFILL) {
                min_vrt = std::min(min_vrt, vtcs_[i].vruntime);
            }
        }
        return (min_vrt == UINT64_MAX) ? 0 : min_vrt;
    }

    /* ---- Preemption ----------------------------------------------- */

    /** Preempt a request: suspend it and optionally trigger vMMU page-out.
     *  Returns true if preemption succeeded. */
    bool preempt(uint32_t req_id) {
        std::lock_guard<std::mutex> lk(mu_);
        auto* req = find_request(req_id);
        if (!req) return false;
        if (req->state != nf::RequestState::DECODE &&
            req->state != nf::RequestState::PREFILL) return false;

        req->state = nf::RequestState::PREEMPTED;
        preempted_ids_.push_back(req_id);

        /* Cross-layer: page out KV blocks via vMMU */
        if (vmmu_ && req->seq_id < kv_->num_sequences) {
            uint32_t nblocks = kv_->sequences[req->seq_id].num_logical_blocks;
            if (nblocks > 0) {
                page_out_count_ += vmmu_->page_out(req->seq_id, 0, nblocks, nullptr, 0);
            }
        }
        return true;
    }

    /** Resume a preempted request. Returns true if resumed. */
    bool resume(uint32_t req_id) {
        std::lock_guard<std::mutex> lk(mu_);
        auto* req = find_request(req_id);
        if (!req) return false;
        if (req->state != nf::RequestState::PREEMPTED) return false;

        /* Cross-layer: page in KV blocks via vMMU */
        if (vmmu_ && req->seq_id < kv_->num_sequences) {
            uint32_t nblocks = kv_->sequences[req->seq_id].num_logical_blocks;
            if (nblocks > 0) {
                page_in_count_ += vmmu_->page_in(req->seq_id, 0, nblocks, nullptr, 0);
            }
        }

        /* Restore to DECODE (it was running before preemption) */
        req->state = nf::RequestState::DECODE;
        preempted_ids_.erase(
            std::remove(preempted_ids_.begin(), preempted_ids_.end(), req_id),
            preempted_ids_.end());
        return true;
    }

    /** Get list of currently preempted request IDs */
    std::vector<uint32_t> preempted_requests() const {
        std::lock_guard<std::mutex> lk(mu_);
        return preempted_ids_;
    }

    /* ---- Rebalance: VTC-based fair scheduling --------------------- */

    /** Rebalance: select requests for next batch based on VTC fairness.
     *  Picks requests with lowest vruntime first (most underserved).
     *  Preempts requests whose vruntime exceeds min by threshold. */
    struct ScheduleResult {
        std::vector<uint32_t> selected_req_ids;
        std::vector<uint32_t> preempted_req_ids;
        uint32_t total_tokens = 0;
    };

    ScheduleResult rebalance() {
        std::lock_guard<std::mutex> lk(mu_);
        ScheduleResult result;

        struct Candidate {
            uint32_t req_idx;
            uint32_t tokens;
            uint64_t vruntime;
        };
        std::vector<Candidate> candidates;

        for (uint32_t i = 0; i < sched_->num_requests; ++i) {
            auto& r = sched_->requests[i];
            if (r.state == nf::RequestState::DECODE) {
                candidates.push_back({i, 1, vtcs_[i].vruntime});
            } else if (r.state == nf::RequestState::PREFILL) {
                uint32_t rem = static_cast<uint32_t>(r.prompt_tokens.size())
                               - r.tokens_generated;
                candidates.push_back({i, rem, vtcs_[i].vruntime});
            } else if (r.state == nf::RequestState::QUEUED) {
                candidates.push_back({i, static_cast<uint32_t>(r.prompt_tokens.size()),
                                      vtcs_[i].vruntime});
            }
        }

        /* Sort by vruntime ascending (fairest first) */
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.vruntime < b.vruntime;
            });

        uint64_t min_vrt = candidates.empty() ? 0 : candidates[0].vruntime;
        uint32_t budget = cfg_.max_batch_tokens;

        for (auto& c : candidates) {
            /* Preempt if too far ahead */
            if (min_vrt > 0 && c.vruntime > min_vrt * cfg_.preempt_threshold) {
                auto& r = sched_->requests[c.req_idx];
                if (r.state == nf::RequestState::DECODE ||
                    r.state == nf::RequestState::PREFILL) {
                    r.state = nf::RequestState::PREEMPTED;
                    preempted_ids_.push_back(r.req_id);
                    result.preempted_req_ids.push_back(r.req_id);
                    /* Cross-layer: page out KV blocks via vMMU */
                    if (vmmu_ && r.seq_id < kv_->num_sequences) {
                        uint32_t nblocks = kv_->sequences[r.seq_id].num_logical_blocks;
                        if (nblocks > 0)
                            page_out_count_ += vmmu_->page_out(r.seq_id, 0, nblocks, nullptr, 0);
                    }
                }
                continue;
            }

            if (c.tokens <= budget) {
                result.selected_req_ids.push_back(
                    sched_->requests[c.req_idx].req_id);
                result.total_tokens += c.tokens;
                budget -= c.tokens;
            }
        }
        return result;
    }

    /* ---- Migration Interface (Phase 38) --------------------------- */

    /** Cross-instance request migration via KVMigrator.
     *  Serializes KV state, preempts locally, returns success. */
    bool migrate_request(uint32_t req_id, const char* target_node) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!migrator_) return false;
        auto* req = find_request(req_id);
        if (!req) return false;
        if (req->seq_id >= kv_->num_sequences) return false;

        /* Serialize KV state via migrator */
        last_migration_ = migrator_->serialize_kv(*kv_, req->seq_id, nullptr, 0);
        last_migration_target_ = target_node ? target_node : "";

        /* Preempt locally */
        if (req->state == nf::RequestState::DECODE ||
            req->state == nf::RequestState::PREFILL) {
            req->state = nf::RequestState::PREEMPTED;
            preempted_ids_.push_back(req_id);
        }
        ++migrations_;
        return true;
    }

    uint32_t num_migrations() const { return migrations_; }
    const std::string& last_migration_target() const { return last_migration_target_; }

    /* ---- Queries -------------------------------------------------- */

    uint64_t get_vruntime(uint32_t req_id) const {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t idx = req_index(req_id);
        return (idx != UINT32_MAX) ? vtcs_[idx].vruntime : 0;
    }

    uint32_t num_preempted() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(preempted_ids_.size());
    }

private:
    nf::InferenceRequest* find_request(uint32_t req_id) {
        for (uint32_t i = 0; i < sched_->num_requests; ++i)
            if (sched_->requests[i].req_id == req_id)
                return &sched_->requests[i];
        return nullptr;
    }

    uint32_t req_index(uint32_t req_id) const {
        for (uint32_t i = 0; i < sched_->num_requests; ++i)
            if (sched_->requests[i].req_id == req_id)
                return i;
        return UINT32_MAX;
    }

    nf::RequestScheduler*  sched_;
    nf::PagedKVCache*      kv_;
    Config                 cfg_;
    mutable std::mutex     mu_;

    vMMU*                            vmmu_ = nullptr;
    neuralOS::mesh::KVMigrator*        migrator_ = nullptr;
    uint32_t                         page_out_count_ = 0;
    uint32_t                         page_in_count_ = 0;
    uint32_t                         migrations_ = 0;
    std::string                      last_migration_target_;
    neuralOS::mesh::KVSerializedSequence last_migration_;

    std::vector<VirtualTokenCounter> vtcs_;
    std::vector<uint32_t>            preempted_ids_;
};

}} // namespace neuralOS::kernel

// Backward compatibility
namespace neuralOS { namespace L2 {
    using neuralOS::kernel::VirtualTokenCounter;
    using neuralOS::kernel::CFS;
}}

#endif // NEURALOS_KERNEL_CFS_HPP
