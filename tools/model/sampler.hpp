/**
 * @file sampler.hpp
 * @brief CPU-side token sampling — temperature, top-k, top-p, repetition penalty.
 *
 * Phase 23: Sampling Strategies.
 * Header-only, zero dependencies beyond <cstdint>, <cmath>, <algorithm>, <random>.
 */

#ifndef NF_SAMPLER_HPP
#define NF_SAMPLER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace nf {

struct SamplerParams {
    float    temperature      = 0.8f;   // 0 = greedy
    uint32_t top_k            = 40;     // 0 = disabled
    float    top_p            = 0.95f;  // 1.0 = disabled
    float    repeat_penalty   = 1.1f;   // 1.0 = disabled
    uint32_t repeat_window    = 64;
    uint64_t seed             = 0;      // 0 = random
    float    frequency_penalty = 0.0f;  // reserved Phase 25
    float    presence_penalty  = 0.0f;  // reserved Phase 25
};

inline int32_t sample_token(
    const float* logits, uint32_t vocab_size,
    const int32_t* recent_tokens, uint32_t n_recent,
    const SamplerParams& params,
    std::mt19937_64& rng)
{
    /* Greedy fast path */
    if (params.temperature <= 0.0f) {
        int32_t best = 0;
        float best_v = logits[0];
        for (uint32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = (int32_t)i; }
        }
        return best;
    }

    /* Copy logits into sortable pairs */
    struct TokenProb { int32_t id; float logit; };
    std::vector<TokenProb> candidates(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i)
        candidates[i] = {(int32_t)i, logits[i]};

    /* Repetition penalty */
    if (params.repeat_penalty != 1.0f && n_recent > 0) {
        uint32_t window = std::min(n_recent, params.repeat_window);
        for (uint32_t i = 0; i < window; ++i) {
            int32_t tid = recent_tokens[n_recent - 1 - i];
            if (tid >= 0 && (uint32_t)tid < vocab_size) {
                float& l = candidates[(uint32_t)tid].logit;
                l = (l > 0.0f) ? l / params.repeat_penalty
                               : l * params.repeat_penalty;
            }
        }
    }

    /* Temperature scaling */
    float inv_temp = 1.0f / params.temperature;
    for (auto& c : candidates) c.logit *= inv_temp;

    /* Top-k: partial sort to find k-th largest */
    uint32_t k = (params.top_k > 0 && params.top_k < vocab_size)
                 ? params.top_k : vocab_size;
    std::partial_sort(candidates.begin(), candidates.begin() + k,
                      candidates.end(),
                      [](const TokenProb& a, const TokenProb& b) {
                          return a.logit > b.logit;
                      });
    candidates.resize(k);

    /* Softmax */
    float max_logit = candidates[0].logit;
    float sum_exp = 0.0f;
    for (auto& c : candidates) {
        c.logit = std::exp(c.logit - max_logit);
        sum_exp += c.logit;
    }
    for (auto& c : candidates) c.logit /= sum_exp;

    /* Top-p (nucleus) */
    if (params.top_p < 1.0f) {
        float cumsum = 0.0f;
        uint32_t cutoff = (uint32_t)candidates.size();
        for (uint32_t i = 0; i < (uint32_t)candidates.size(); ++i) {
            cumsum += candidates[i].logit;
            if (cumsum >= params.top_p) { cutoff = i + 1; break; }
        }
        candidates.resize(cutoff);
        /* Re-normalize */
        float s = 0.0f;
        for (auto& c : candidates) s += c.logit;
        for (auto& c : candidates) c.logit /= s;
    }

    /* Weighted random sample */
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (auto& c : candidates) {
        cumsum += c.logit;
        if (r <= cumsum) return c.id;
    }
    return candidates.back().id;
}

/** Convenience: auto-seed version */
inline int32_t sample_token(
    const float* logits, uint32_t vocab_size,
    const int32_t* recent_tokens, uint32_t n_recent,
    const SamplerParams& params)
{
    std::mt19937_64 rng(params.seed ? params.seed
                        : std::random_device{}());
    return sample_token(logits, vocab_size, recent_tokens, n_recent,
                        params, rng);
}

} /* namespace nf */

#endif /* NF_SAMPLER_HPP */
