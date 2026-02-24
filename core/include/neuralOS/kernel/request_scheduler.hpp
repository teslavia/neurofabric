/**
 * @file request_scheduler.hpp
 * @brief NeuralOS L2 — RequestScheduler + BatchDescriptor + SpeculativeConfig
 *
 * Phase 45A: Extracted from model/model/model_config.hpp to eliminate
 * kernel→model reverse dependency. Kernel layer is now self-contained.
 *
 * Header-only. Depends only on kv_cache.hpp + STL.
 */

#ifndef NEURALOS_L2_REQUEST_SCHEDULER_HPP
#define NEURALOS_L2_REQUEST_SCHEDULER_HPP

#include "neuralOS/kernel/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace nf {

/* ================================================================== */
/*  RequestState + InferenceRequest                                    */
/* ================================================================== */

enum class RequestState : uint8_t {
    QUEUED, PREFILL, DECODE, COMPLETE, CANCELLED,
    PREEMPTED  /* Phase 36: suspended by CFS */
};

struct InferenceRequest {
    uint32_t     req_id       = 0;
    RequestState state        = RequestState::QUEUED;
    uint32_t     priority     = 0;
    uint64_t     submit_time_us = 0;

    /* Input */
    std::vector<int32_t> prompt_tokens;
    uint32_t max_new_tokens   = 32;
    float    temperature      = 0.8f;
    float    top_p            = 0.95f;

    /* State */
    uint32_t seq_id           = UINT32_MAX;
    uint32_t tokens_generated = 0;
    std::vector<int32_t> output_tokens;

    /* Callbacks (optional) */
    void (*on_token)(uint32_t req_id, int32_t token, void* user_data) = nullptr;
    void (*on_complete)(uint32_t req_id, void* user_data) = nullptr;
    void* user_data = nullptr;

    /* Phase 36: VTC (Virtual Token Counter) for CFS scheduling */
    uint64_t vtc_runtime     = 0;
    uint64_t vtc_weight      = 1;
};

/* ================================================================== */
/*  BatchDescriptor                                                    */
/* ================================================================== */

struct BatchDescriptor {
    uint32_t num_sequences = 0;
    uint32_t total_tokens  = 0;

    uint32_t seq_ids[NF_PAGED_MAX_SEQUENCES]   = {};
    uint32_t seq_lens[NF_PAGED_MAX_SEQUENCES]  = {};
    uint32_t step_idxs[NF_PAGED_MAX_SEQUENCES] = {};

    std::vector<uint32_t> merged_block_table;
    uint32_t max_blocks_per_seq = 0;
};

/* ================================================================== */
/*  RequestScheduler                                                   */
/* ================================================================== */

static constexpr uint32_t NF_MAX_REQUESTS = 256;

struct RequestScheduler {
    InferenceRequest requests[NF_MAX_REQUESTS];
    uint32_t num_requests = 0;
    uint32_t next_req_id  = 1;

    uint32_t submit(InferenceRequest req) {
        if (num_requests >= NF_MAX_REQUESTS) return UINT32_MAX;
        req.req_id = next_req_id++;
        req.state = RequestState::QUEUED;
        requests[num_requests++] = std::move(req);
        return req.req_id;
    }

    bool cancel(uint32_t req_id) {
        for (uint32_t i = 0; i < num_requests; ++i) {
            if (requests[i].req_id == req_id &&
                requests[i].state != RequestState::COMPLETE) {
                requests[i].state = RequestState::CANCELLED;
                return true;
            }
        }
        return false;
    }

    BatchDescriptor schedule_batch(uint32_t max_batch_tokens,
                                   PagedKVCache& kv_cache) {
        BatchDescriptor batch{};

        struct Candidate {
            uint32_t idx;
            uint32_t tokens;
            int      phase;
        };
        std::vector<Candidate> candidates;

        for (uint32_t i = 0; i < num_requests; ++i) {
            auto& r = requests[i];
            if (r.state == RequestState::DECODE) {
                candidates.push_back({i, 1, 0});
            } else if (r.state == RequestState::PREFILL) {
                uint32_t remaining = (uint32_t)r.prompt_tokens.size()
                                     - r.tokens_generated;
                candidates.push_back({i, remaining, 1});
            } else if (r.state == RequestState::QUEUED) {
                r.state = RequestState::PREFILL;
                r.seq_id = kv_cache.alloc_sequence();
                if (r.seq_id == UINT32_MAX) { r.state = RequestState::QUEUED; continue; }
                candidates.push_back({i, (uint32_t)r.prompt_tokens.size(), 1});
            }
        }

        std::sort(candidates.begin(), candidates.end(),
            [this](const Candidate& a, const Candidate& b) {
                if (a.phase != b.phase) return a.phase < b.phase;
                return requests[a.idx].priority > requests[b.idx].priority;
            });

        uint32_t budget = max_batch_tokens;
        for (auto& c : candidates) {
            if (budget == 0) break;
            if (c.tokens > budget) continue;
            auto& r = requests[c.idx];
            batch.seq_ids[batch.num_sequences]  = r.seq_id;
            batch.seq_lens[batch.num_sequences] = c.tokens;
            batch.step_idxs[batch.num_sequences] =
                (r.state == RequestState::DECODE)
                    ? (uint32_t)r.prompt_tokens.size() + r.tokens_generated
                    : 0;
            batch.total_tokens += c.tokens;
            batch.num_sequences++;
            budget -= c.tokens;
        }

        return batch;
    }

    void on_step_complete(const BatchDescriptor& batch,
                          const int32_t* sampled_tokens) {
        for (uint32_t i = 0; i < batch.num_sequences; ++i) {
            for (uint32_t j = 0; j < num_requests; ++j) {
                auto& r = requests[j];
                if (r.seq_id != batch.seq_ids[i]) continue;

                if (r.state == RequestState::PREFILL) {
                    r.state = RequestState::DECODE;
                } else if (r.state == RequestState::DECODE) {
                    int32_t tok = sampled_tokens[i];
                    r.output_tokens.push_back(tok);
                    r.tokens_generated++;
                    if (r.on_token)
                        r.on_token(r.req_id, tok, r.user_data);
                    if (r.tokens_generated >= r.max_new_tokens) {
                        r.state = RequestState::COMPLETE;
                        if (r.on_complete)
                            r.on_complete(r.req_id, r.user_data);
                    }
                }
                break;
            }
        }
    }
};

/* ================================================================== */
/*  SpeculativeConfig + speculative_accept                             */
/* ================================================================== */

struct SpeculativeConfig {
    uint32_t draft_layers   = 0;
    uint32_t num_speculative = 4;
    float    acceptance_threshold = 0.0f;

    uint32_t tree_width     = 1;
    uint32_t max_depth      = 8;
};

inline uint32_t speculative_accept(
    const float* draft_logits,
    const float* verify_logits,
    const int32_t* draft_tokens,
    uint32_t K, uint32_t vocab,
    uint32_t seed)
{
    (void)seed;
    for (uint32_t i = 0; i < K; ++i) {
        const float* v_row = verify_logits + i * vocab;
        int32_t v_argmax = 0;
        float v_max = v_row[0];
        for (uint32_t j = 1; j < vocab; ++j) {
            if (v_row[j] > v_max) { v_max = v_row[j]; v_argmax = (int32_t)j; }
        }
        if (v_argmax != draft_tokens[i]) return i;
    }
    return K;
}

} /* namespace nf */

#endif /* NEURALOS_L2_REQUEST_SCHEDULER_HPP */
