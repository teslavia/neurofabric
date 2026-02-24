/**
 * @file BatchInferenceLoop.hpp
 * @brief NeuralOS L2 — Continuous Batching Engine
 *
 * Phase 43B.1: Wraps NeuralOSRuntime for batched scheduling steps.
 * Thread-safe submit, step-based scheduling with VTC accounting.
 */

#ifndef NEURALOS_L2_BATCH_INFERENCE_LOOP_HPP
#define NEURALOS_L2_BATCH_INFERENCE_LOOP_HPP

#include "neuralOS/kernel/NeuralOSRuntime.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

namespace neuralOS { namespace L2 {

struct BatchStepResult {
    uint32_t num_selected  = 0;
    uint32_t num_preempted = 0;
    uint32_t total_tokens  = 0;
};

class BatchInferenceLoop {
public:
    BatchInferenceLoop(NeuralOSRuntime* runtime,
                       nf::PagedKVCache* kv,
                       nf::RequestScheduler* sched)
        : runtime_(runtime), kv_(kv), sched_(sched) {}

    uint32_t submit(nf::InferenceRequest req) {
        std::lock_guard<std::mutex> lk(mu_);
        return runtime_->submit(std::move(req));
    }

    BatchStepResult step() {
        std::lock_guard<std::mutex> lk(mu_);
        BatchStepResult result;

        auto sr = runtime_->schedule_step();
        result.num_selected  = static_cast<uint32_t>(sr.selected_ids.size());
        result.num_preempted = static_cast<uint32_t>(sr.preempted_ids.size());
        result.total_tokens  = sr.total_tokens;

        for (uint32_t req_id : sr.selected_ids) {
            auto* req = find_request(req_id);
            if (!req) continue;

            bool is_prefill = (req->state == nf::RequestState::PREFILL ||
                               req->state == nf::RequestState::QUEUED);
            uint32_t n_tokens = is_prefill
                ? static_cast<uint32_t>(req->prompt_tokens.size())
                : 1;

            runtime_->cfs()->account_tokens(req_id, n_tokens, is_prefill);

            if (req->state == nf::RequestState::QUEUED)
                req->state = nf::RequestState::PREFILL;
            else if (req->state == nf::RequestState::PREFILL)
                req->state = nf::RequestState::DECODE;
            else if (req->state == nf::RequestState::DECODE) {
                req->tokens_generated++;
                if (req->tokens_generated >= req->max_new_tokens)
                    runtime_->complete(req->req_id);
            }
        }

        runtime_->check_pressure();
        return result;
    }

    bool has_active() const {
        std::lock_guard<std::mutex> lk(mu_);
        for (uint32_t i = 0; i < sched_->num_requests; ++i) {
            auto s = sched_->requests[i].state;
            if (s != nf::RequestState::COMPLETE &&
                s != nf::RequestState::CANCELLED)
                return true;
        }
        return false;
    }

    std::vector<uint32_t> drain_completed() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<uint32_t> out;
        for (uint32_t i = 0; i < sched_->num_requests; ++i) {
            if (sched_->requests[i].state == nf::RequestState::COMPLETE)
                out.push_back(sched_->requests[i].req_id);
        }
        return out;
    }

private:
    nf::InferenceRequest* find_request(uint32_t req_id) {
        for (uint32_t i = 0; i < sched_->num_requests; ++i)
            if (sched_->requests[i].req_id == req_id)
                return &sched_->requests[i];
        return nullptr;
    }

    NeuralOSRuntime*       runtime_;
    nf::PagedKVCache*      kv_;
    nf::RequestScheduler*  sched_;
    mutable std::mutex     mu_;
};

}} // namespace neuralOS::L2

#endif // NEURALOS_L2_BATCH_INFERENCE_LOOP_HPP
