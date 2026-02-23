/**
 * @file model_config.hpp
 * @brief Phase 32: ModelConfig, PagedKVCache, BlockAllocator, RequestScheduler
 *
 * Collapses the 9-param create_llama_context() into a named-field struct.
 * Adds paged virtual memory for KV cache (PagedAttention) and multi-request
 * scheduling for continuous batching.
 *
 * All structures are POD-friendly with inline implementations.
 * No heap allocation in BlockAllocator/SequenceKV themselves — they use
 * fixed-size arrays sized for practical limits.
 */

#ifndef NF_MODEL_CONFIG_HPP
#define NF_MODEL_CONFIG_HPP

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/engine/PipelineEngine.hpp"
#include "kv_cache_policy.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

/* Forward declarations — avoid circular include with gguf_loader.hpp */
namespace nf { struct GGUFModel; }

namespace nf {

/* ================================================================== */
/*  1. ModelConfig                                                      */
/* ================================================================== */

struct ModelConfig {
    /* Engine & provider (required) */
    PipelineEngine*         engine   = nullptr;
    nf_provider             prov     = nullptr;
    nf_provider_vtable*     vt       = nullptr;
    nf_provider_mem_vtable* mem_vt   = nullptr;

    /* Model source (required) */
    const GGUFModel*        model    = nullptr;

    /* Sequence limits (required) */
    uint32_t max_seq         = 512;
    uint32_t max_prefill_seq = 0;     /* 0 = same as max_seq */

    /* Optional features (all default to off/null) */
    bool     use_fp16       = false;
    bool     use_paged_kv   = false;
    uint32_t kv_block_size  = 16;     /* tokens per physical block */
    uint32_t num_kv_blocks  = 0;      /* 0 = auto-calculate */

    const nf_kv_cache_config* kv_cfg = nullptr;
    const char* arch_override        = nullptr;  /* nullptr = auto-detect */
};

/* ================================================================== */
/*  2. PagedKVCache — Block Allocator + Per-Sequence Block Tables      */
/* ================================================================== */

static constexpr uint32_t NF_PAGED_MAX_SEQUENCES     = 64;
static constexpr uint32_t NF_PAGED_MAX_BLOCKS_PER_SEQ = 512;  /* 512 × 16 = 8192 tokens */
static constexpr uint32_t NF_PAGED_INVALID_BLOCK     = UINT32_MAX;

struct PagedKVBlock {
    uint32_t ref_count   = 0;   /* CoW: >1 means shared */
    uint32_t num_filled  = 0;   /* tokens written [0..block_size] */
};

struct BlockAllocator {
    uint32_t      num_blocks  = 0;
    uint32_t      block_size  = 16;
    uint32_t      free_top    = 0;     /* stack pointer */
    std::vector<uint32_t>      free_stack;
    std::vector<PagedKVBlock>  block_meta;

    void init(uint32_t n_blocks, uint32_t bsz) {
        num_blocks = n_blocks;
        block_size = bsz;
        free_stack.resize(n_blocks);
        block_meta.resize(n_blocks);
        reset();
    }

    void reset() {
        free_top = num_blocks;
        for (uint32_t i = 0; i < num_blocks; ++i) {
            free_stack[i] = num_blocks - 1 - i;  /* top of stack = block 0 */
            block_meta[i] = PagedKVBlock{};
        }
    }

    uint32_t alloc_block() {
        if (free_top == 0) return NF_PAGED_INVALID_BLOCK;
        uint32_t idx = free_stack[--free_top];
        block_meta[idx].ref_count = 1;
        block_meta[idx].num_filled = 0;
        return idx;
    }

    void free_block(uint32_t idx) {
        if (idx >= num_blocks) return;
        auto& m = block_meta[idx];
        if (m.ref_count > 0) --m.ref_count;
        if (m.ref_count == 0) {
            free_stack[free_top++] = idx;
            m.num_filled = 0;
        }
    }

    void ref(uint32_t idx) {
        if (idx < num_blocks) ++block_meta[idx].ref_count;
    }

    void unref(uint32_t idx) {
        if (idx < num_blocks) free_block(idx);
    }

    uint32_t num_free() const { return free_top; }
    uint32_t num_used() const { return num_blocks - free_top; }
};

struct SequenceKV {
    uint32_t seq_id          = 0;
    uint32_t num_tokens      = 0;
    uint32_t num_logical_blocks = 0;
    bool     active          = false;
    uint32_t block_table[NF_PAGED_MAX_BLOCKS_PER_SEQ] = {};

    void reset() {
        num_tokens = 0;
        num_logical_blocks = 0;
        active = false;
        std::memset(block_table, 0xFF, sizeof(block_table));  /* all INVALID */
    }
};

struct PagedKVCache {
    BlockAllocator allocator;

    /* Per-sequence state */
    SequenceKV sequences[NF_PAGED_MAX_SEQUENCES];
    uint32_t   num_sequences = 0;

    /* Dimensions */
    uint32_t n_layers   = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim   = 0;
    uint32_t block_size = 16;

    void init(uint32_t n_blocks, uint32_t bsz,
              uint32_t layers, uint32_t kv_heads, uint32_t hdim) {
        allocator.init(n_blocks, bsz);
        n_layers   = layers;
        n_kv_heads = kv_heads;
        head_dim   = hdim;
        block_size = bsz;
        num_sequences = 0;
        for (auto& s : sequences) s.reset();
    }

    /* Allocate a new sequence slot. Returns seq index or UINT32_MAX. */
    uint32_t alloc_sequence() {
        if (num_sequences >= NF_PAGED_MAX_SEQUENCES) return UINT32_MAX;
        uint32_t idx = num_sequences++;
        sequences[idx].seq_id = idx;
        sequences[idx].active = true;
        sequences[idx].num_tokens = 0;
        sequences[idx].num_logical_blocks = 0;
        std::memset(sequences[idx].block_table, 0xFF,
                    sizeof(sequences[idx].block_table));
        return idx;
    }

    /* Append a token to a sequence. Allocates new block if needed.
     * Returns the physical write offset (flat index into pool). */
    uint32_t append(uint32_t seq_idx) {
        if (seq_idx >= num_sequences) return UINT32_MAX;
        auto& seq = sequences[seq_idx];

        uint32_t logical_block = seq.num_tokens / block_size;
        uint32_t offset_in_block = seq.num_tokens % block_size;

        /* Need a new block? */
        if (logical_block >= seq.num_logical_blocks) {
            uint32_t phys = allocator.alloc_block();
            if (phys == NF_PAGED_INVALID_BLOCK) return UINT32_MAX;
            seq.block_table[logical_block] = phys;
            seq.num_logical_blocks = logical_block + 1;
        }

        uint32_t phys_block = seq.block_table[logical_block];
        allocator.block_meta[phys_block].num_filled = offset_in_block + 1;
        seq.num_tokens++;

        return phys_block * block_size + offset_in_block;
    }

    /* Free all blocks for a sequence. */
    void free_sequence(uint32_t seq_idx) {
        if (seq_idx >= num_sequences) return;
        auto& seq = sequences[seq_idx];
        for (uint32_t i = 0; i < seq.num_logical_blocks; ++i) {
            if (seq.block_table[i] != NF_PAGED_INVALID_BLOCK)
                allocator.free_block(seq.block_table[i]);
        }
        seq.reset();
    }

    /* Truncate a sequence to `new_len` tokens. Frees blocks beyond. */
    void truncate(uint32_t seq_idx, uint32_t new_len) {
        if (seq_idx >= num_sequences) return;
        auto& seq = sequences[seq_idx];
        if (new_len >= seq.num_tokens) return;

        uint32_t new_blocks = (new_len + block_size - 1) / block_size;
        if (new_len == 0) new_blocks = 0;

        for (uint32_t i = new_blocks; i < seq.num_logical_blocks; ++i) {
            if (seq.block_table[i] != NF_PAGED_INVALID_BLOCK) {
                allocator.free_block(seq.block_table[i]);
                seq.block_table[i] = NF_PAGED_INVALID_BLOCK;
            }
        }
        seq.num_logical_blocks = new_blocks;
        seq.num_tokens = new_len;

        /* Update last block's fill count */
        if (new_blocks > 0) {
            uint32_t last_phys = seq.block_table[new_blocks - 1];
            if (last_phys != NF_PAGED_INVALID_BLOCK) {
                uint32_t fill = new_len % block_size;
                allocator.block_meta[last_phys].num_filled =
                    (fill == 0) ? block_size : fill;
            }
        }
    }

    /* Get the flat block table for GPU upload (logical → physical). */
    void fill_block_table(uint32_t seq_idx, uint32_t* out,
                          uint32_t max_blocks) const {
        if (seq_idx >= num_sequences) return;
        const auto& seq = sequences[seq_idx];
        uint32_t n = std::min(seq.num_logical_blocks, max_blocks);
        for (uint32_t i = 0; i < n; ++i)
            out[i] = seq.block_table[i];
        for (uint32_t i = n; i < max_blocks; ++i)
            out[i] = NF_PAGED_INVALID_BLOCK;
    }
};

/* ================================================================== */
/*  3. RequestScheduler + BatchDescriptor (Phase 32 Step 3)            */
/* ================================================================== */

enum class RequestState : uint8_t {
    QUEUED, PREFILL, DECODE, COMPLETE, CANCELLED
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
};

struct BatchDescriptor {
    uint32_t num_sequences = 0;
    uint32_t total_tokens  = 0;

    /* Per-sequence info (parallel arrays) */
    uint32_t seq_ids[NF_PAGED_MAX_SEQUENCES]   = {};
    uint32_t seq_lens[NF_PAGED_MAX_SEQUENCES]  = {};
    uint32_t step_idxs[NF_PAGED_MAX_SEQUENCES] = {};

    /* Merged block table */
    std::vector<uint32_t> merged_block_table;
    uint32_t max_blocks_per_seq = 0;
};

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

    /**
     * Schedule a batch respecting token budget.
     * Prioritizes: DECODE > PREFILL > QUEUED (decode is cheaper).
     * Within same state, higher priority first, then FIFO by submit_time.
     */
    BatchDescriptor schedule_batch(uint32_t max_batch_tokens,
                                   PagedKVCache& kv_cache) {
        BatchDescriptor batch{};

        /* Collect eligible requests, sorted by scheduling priority */
        struct Candidate {
            uint32_t idx;
            uint32_t tokens;  /* 1 for decode, prompt_len for prefill */
            int      phase;   /* 0=decode, 1=prefill, 2=queued */
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
                candidates.push_back({i, (uint32_t)r.prompt_tokens.size(), 2});
            }
        }

        /* Sort: decode first, then prefill, then queued.
         * Within same phase: higher priority first, then earlier submit. */
        std::sort(candidates.begin(), candidates.end(),
            [&](const Candidate& a, const Candidate& b) {
                if (a.phase != b.phase) return a.phase < b.phase;
                auto& ra = requests[a.idx];
                auto& rb = requests[b.idx];
                if (ra.priority != rb.priority)
                    return ra.priority > rb.priority;
                return ra.submit_time_us < rb.submit_time_us;
            });

        uint32_t budget = max_batch_tokens;
        for (auto& c : candidates) {
            if (c.tokens > budget) continue;
            auto& r = requests[c.idx];

            /* Allocate sequence if needed */
            if (r.seq_id == UINT32_MAX) {
                r.seq_id = kv_cache.alloc_sequence();
                if (r.seq_id == UINT32_MAX) continue;
                r.state = RequestState::PREFILL;
            }

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
            /* Find request by seq_id */
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
/*  4. Speculative Decoding Config (Phase 32 Step 4)                   */
/* ================================================================== */

struct SpeculativeConfig {
    uint32_t draft_layers   = 0;   /* number of layers for draft model (0 = disabled) */
    uint32_t num_speculative = 4;  /* tokens to speculate per step */
    float    acceptance_threshold = 0.0f;  /* 0 = standard rejection sampling */
};

/**
 * Compare draft vs verify logits. Returns number of accepted tokens.
 * Uses standard speculative decoding rejection sampling:
 *   accept if p_verify(x) >= p_draft(x), else accept with prob p_verify/p_draft.
 */
inline uint32_t speculative_accept(
    const float* draft_logits,    /* [K × vocab] */
    const float* verify_logits,   /* [(K+1) × vocab] */
    const int32_t* draft_tokens,  /* [K] sampled draft tokens */
    uint32_t K, uint32_t vocab,
    uint32_t seed)
{
    /* Simple deterministic acceptance for now:
     * Accept prefix where argmax(verify) == draft_token */
    for (uint32_t i = 0; i < K; ++i) {
        const float* v_row = verify_logits + i * vocab;
        /* Find argmax of verify logits for position i */
        int32_t v_argmax = 0;
        float v_max = v_row[0];
        for (uint32_t j = 1; j < vocab; ++j) {
            if (v_row[j] > v_max) { v_max = v_row[j]; v_argmax = (int32_t)j; }
        }
        if (v_argmax != draft_tokens[i]) return i;
    }
    return K;  /* all accepted */
}

} /* namespace nf */

#endif /* NF_MODEL_CONFIG_HPP */
