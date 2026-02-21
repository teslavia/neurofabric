/**
 * @file kv_cache_policy.hpp
 * @brief KV Cache Policy Engine — pluggable eviction + optional INT8 quantization
 *
 * Phase 31: Multi-Architecture DAG & KV Cache Intelligence.
 *
 * Provides O(1) amortized eviction strategies for KV cache management:
 *   - NONE:    No eviction (original behavior, short context)
 *   - SLIDING: Ring buffer (Mistral-style sliding window)
 *   - LRU:     Doubly-linked list + hash map (general purpose)
 *
 * Optional INT8 quantization: quantize on write, dequantize on read.
 * Transparent to attention kernels (they always see FP16/FP32).
 *
 * All functions are pure, no heap allocation in the policy struct itself.
 * The LRU tracker is a separate utility for callers that need it.
 */

#ifndef NF_KV_CACHE_POLICY_HPP
#define NF_KV_CACHE_POLICY_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace nf {

/* ------------------------------------------------------------------ */
/*  1. Eviction Type & Configuration                                   */
/* ------------------------------------------------------------------ */

enum nf_kv_eviction_type : uint8_t {
    NF_KV_EVICT_NONE    = 0,
    NF_KV_EVICT_SLIDING = 1,
    NF_KV_EVICT_LRU     = 2,
    NF_KV_EVICT_PAGED   = 3,   /* Phase 32: paged virtual memory */
};

struct nf_kv_cache_config {
    nf_kv_eviction_type eviction;
    uint32_t window_size;       /* SLIDING: window size; LRU: max entries */
    bool     use_int8;          /* Enable INT8 quantization (2x capacity) */
    uint32_t max_seq_len;       /* Maximum sequence length */
};

/* ------------------------------------------------------------------ */
/*  2. Policy Interface (POD, no heap)                                 */
/* ------------------------------------------------------------------ */
struct nf_kv_cache_policy {
    nf_kv_cache_config config;

    /* Write offset: given current step, return KV cache write position.
     * Sliding: step % window_size
     * None:    step (direct append) */
    uint32_t (*write_offset)(const nf_kv_cache_policy* self, uint32_t step);

    /* Attention range: given current step, return [out_start, out_start+out_len).
     * Sliding: [max(0, step-window+1), step+1)
     * None:    [0, step+1) */
    void (*attn_range)(const nf_kv_cache_policy* self, uint32_t step,
                       uint32_t* out_start, uint32_t* out_len);

    /* Effective KV length for push constants (may differ from step+1 with eviction) */
    uint32_t (*effective_len)(const nf_kv_cache_policy* self, uint32_t step);
};

/* ------------------------------------------------------------------ */
/*  3. INT8 Quantization Utilities                                     */
/* ------------------------------------------------------------------ */

/* Per-block INT8: 32 elements per block, 1 float scale per block.
 * Layout: [scale(f32)][i8 x 32][scale(f32)][i8 x 32]...
 * This matches the granularity used by KV cache quantization in practice. */

static constexpr uint32_t NF_KV_Q8_BLOCK_SIZE = 32;

struct nf_kv_q8_block {
    float   scale;
    int8_t  qs[NF_KV_Q8_BLOCK_SIZE];
};

/* Quantize n_elements of FP32 → INT8 blocks.
 * n_elements must be a multiple of NF_KV_Q8_BLOCK_SIZE. */
inline void nf_kv_quantize_i8(const float* src, nf_kv_q8_block* dst,
                               uint32_t n_elements) {
    const uint32_t n_blocks = n_elements / NF_KV_Q8_BLOCK_SIZE;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const float* block_src = src + b * NF_KV_Q8_BLOCK_SIZE;
        nf_kv_q8_block& blk = dst[b];

        /* Find absmax for scale */
        float amax = 0.0f;
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            float av = std::fabs(block_src[i]);
            if (av > amax) amax = av;
        }

        blk.scale = amax / 127.0f;
        if (blk.scale == 0.0f) {
            std::memset(blk.qs, 0, NF_KV_Q8_BLOCK_SIZE);
            continue;
        }

        const float inv_scale = 127.0f / amax;
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            int32_t v = (int32_t)std::roundf(block_src[i] * inv_scale);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            blk.qs[i] = (int8_t)v;
        }
    }
}

/* Dequantize INT8 blocks → FP32. */
inline void nf_kv_dequantize_i8(const nf_kv_q8_block* src, float* dst,
                                 uint32_t n_elements) {
    const uint32_t n_blocks = n_elements / NF_KV_Q8_BLOCK_SIZE;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const nf_kv_q8_block& blk = src[b];
        float* block_dst = dst + b * NF_KV_Q8_BLOCK_SIZE;
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            block_dst[i] = blk.qs[i] * blk.scale;
        }
    }
}

/* FP16 variants */
inline void nf_kv_quantize_i8_f16(const uint16_t* src, nf_kv_q8_block* dst,
                                    uint32_t n_elements) {
    const uint32_t n_blocks = n_elements / NF_KV_Q8_BLOCK_SIZE;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint16_t* block_src = src + b * NF_KV_Q8_BLOCK_SIZE;
        nf_kv_q8_block& blk = dst[b];

        /* Decode F16 → F32 for absmax */
        float vals[NF_KV_Q8_BLOCK_SIZE];
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            uint16_t h = block_src[i];
            uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) f = sign;
            else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
            else f = sign | ((exp + 112) << 23) | (mant << 13);
            std::memcpy(&vals[i], &f, 4);
        }

        float amax = 0.0f;
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            float av = std::fabs(vals[i]);
            if (av > amax) amax = av;
        }

        blk.scale = amax / 127.0f;
        if (blk.scale == 0.0f) {
            std::memset(blk.qs, 0, NF_KV_Q8_BLOCK_SIZE);
            continue;
        }

        const float inv_scale = 127.0f / amax;
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            int32_t v = (int32_t)std::roundf(vals[i] * inv_scale);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            blk.qs[i] = (int8_t)v;
        }
    }
}

inline void nf_kv_dequantize_i8_f16(const nf_kv_q8_block* src, uint16_t* dst,
                                      uint32_t n_elements) {
    const uint32_t n_blocks = n_elements / NF_KV_Q8_BLOCK_SIZE;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const nf_kv_q8_block& blk = src[b];
        uint16_t* block_dst = dst + b * NF_KV_Q8_BLOCK_SIZE;
        for (uint32_t i = 0; i < NF_KV_Q8_BLOCK_SIZE; ++i) {
            float val = blk.qs[i] * blk.scale;
            /* F32 → F16 */
            uint32_t fb; std::memcpy(&fb, &val, 4);
            uint32_t sign = (fb >> 16) & 0x8000;
            int32_t exp = ((fb >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fb >> 13) & 0x3FF;
            if (exp <= 0) block_dst[i] = (uint16_t)sign;
            else if (exp >= 31) block_dst[i] = (uint16_t)(sign | 0x7C00);
            else block_dst[i] = (uint16_t)(sign | (exp << 10) | mant);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  4. Policy Implementations                                          */
/* ------------------------------------------------------------------ */

/* --- NONE: no eviction --- */

inline uint32_t kv_none_write_offset(const nf_kv_cache_policy*, uint32_t step) {
    return step;
}

inline void kv_none_attn_range(const nf_kv_cache_policy*, uint32_t step,
                                uint32_t* out_start, uint32_t* out_len) {
    *out_start = 0;
    *out_len = step + 1;
}

inline uint32_t kv_none_effective_len(const nf_kv_cache_policy*, uint32_t step) {
    return step + 1;
}

/* --- SLIDING: ring buffer --- */

inline uint32_t kv_sliding_write_offset(const nf_kv_cache_policy* self,
                                         uint32_t step) {
    return step % self->config.window_size;
}

inline void kv_sliding_attn_range(const nf_kv_cache_policy* self, uint32_t step,
                                   uint32_t* out_start, uint32_t* out_len) {
    uint32_t ws = self->config.window_size;
    if (step + 1 <= ws) {
        *out_start = 0;
        *out_len = step + 1;
    } else {
        *out_start = step - ws + 1;
        *out_len = ws;
    }
}

inline uint32_t kv_sliding_effective_len(const nf_kv_cache_policy* self,
                                          uint32_t step) {
    uint32_t ws = self->config.window_size;
    return (step + 1 < ws) ? (step + 1) : ws;
}

/* --- LRU: uses same write_offset as NONE, eviction tracked externally --- */
/* The LRU policy provides the same interface but callers use
 * nf_lru_tracker (below) to decide which positions to overwrite. */

inline uint32_t kv_lru_write_offset(const nf_kv_cache_policy*, uint32_t step) {
    return step; /* Caller overrides via LRU tracker when cache is full */
}

inline void kv_lru_attn_range(const nf_kv_cache_policy*, uint32_t step,
                               uint32_t* out_start, uint32_t* out_len) {
    *out_start = 0;
    *out_len = step + 1;
}

inline uint32_t kv_lru_effective_len(const nf_kv_cache_policy*, uint32_t step) {
    return step + 1;
}

/* ------------------------------------------------------------------ */
/*  5. Factory                                                         */
/* ------------------------------------------------------------------ */

inline nf_kv_cache_policy nf_create_kv_policy(const nf_kv_cache_config& cfg) {
    nf_kv_cache_policy p{};
    p.config = cfg;

    switch (cfg.eviction) {
    case NF_KV_EVICT_SLIDING:
        p.write_offset   = kv_sliding_write_offset;
        p.attn_range     = kv_sliding_attn_range;
        p.effective_len  = kv_sliding_effective_len;
        break;
    case NF_KV_EVICT_LRU:
        p.write_offset   = kv_lru_write_offset;
        p.attn_range     = kv_lru_attn_range;
        p.effective_len  = kv_lru_effective_len;
        break;
    case NF_KV_EVICT_NONE:
    default:
        p.write_offset   = kv_none_write_offset;
        p.attn_range     = kv_none_attn_range;
        p.effective_len  = kv_none_effective_len;
        break;
    }
    return p;
}

/* ------------------------------------------------------------------ */
/*  6. LRU Tracker (doubly-linked list + hash map, O(1) ops)           */
/* ------------------------------------------------------------------ */

/* Fixed-capacity LRU for KV cache position management.
 * Tracks which positions are "hot" (recently attended).
 * When cache is full, evict_lru() returns the coldest position. */

static constexpr uint32_t NF_LRU_INVALID = UINT32_MAX;

struct nf_lru_node {
    uint32_t pos;       /* KV cache position */
    uint32_t prev;      /* Index into nodes array */
    uint32_t next;
};

struct nf_lru_tracker {
    /* Nodes array: index = slot, nodes[slot].pos = KV position */
    nf_lru_node* nodes;
    uint32_t capacity;
    uint32_t count;

    /* Head = most recently used, tail = least recently used */
    uint32_t head;
    uint32_t tail;

    /* Position → slot lookup (flat array, indexed by position) */
    uint32_t* pos_to_slot;
    uint32_t max_positions;

    /* Free slot stack for reuse after external eviction */
    uint32_t* free_slots;
    uint32_t  free_count;
    uint32_t  next_fresh;  /* Next never-used slot index */
};

inline nf_lru_tracker nf_lru_create(uint32_t capacity, uint32_t max_positions) {
    nf_lru_tracker t{};
    t.capacity = capacity;
    t.count = 0;
    t.head = NF_LRU_INVALID;
    t.tail = NF_LRU_INVALID;
    t.max_positions = max_positions;
    t.next_fresh = 0;
    t.free_count = 0;
    t.nodes = new nf_lru_node[capacity];
    t.pos_to_slot = new uint32_t[max_positions];
    t.free_slots = new uint32_t[capacity];
    for (uint32_t i = 0; i < max_positions; ++i)
        t.pos_to_slot[i] = NF_LRU_INVALID;
    return t;
}

inline void nf_lru_destroy(nf_lru_tracker& t) {
    delete[] t.nodes;
    delete[] t.pos_to_slot;
    delete[] t.free_slots;
    t.nodes = nullptr;
    t.pos_to_slot = nullptr;
    t.free_slots = nullptr;
}

/* Remove a slot from the doubly-linked list (internal) */
inline void nf_lru_unlink(nf_lru_tracker& t, uint32_t slot) {
    nf_lru_node& n = t.nodes[slot];
    if (n.prev != NF_LRU_INVALID) t.nodes[n.prev].next = n.next;
    else t.head = n.next;
    if (n.next != NF_LRU_INVALID) t.nodes[n.next].prev = n.prev;
    else t.tail = n.prev;
    n.prev = n.next = NF_LRU_INVALID;
}

/* Push a slot to head (most recently used) */
inline void nf_lru_push_front(nf_lru_tracker& t, uint32_t slot) {
    nf_lru_node& n = t.nodes[slot];
    n.prev = NF_LRU_INVALID;
    n.next = t.head;
    if (t.head != NF_LRU_INVALID) t.nodes[t.head].prev = slot;
    t.head = slot;
    if (t.tail == NF_LRU_INVALID) t.tail = slot;
}

/* Touch a position: move to front (O(1)). If new, insert. */
inline void nf_lru_touch(nf_lru_tracker& t, uint32_t pos) {
    uint32_t slot = t.pos_to_slot[pos];
    if (slot != NF_LRU_INVALID) {
        /* Already tracked — move to front */
        nf_lru_unlink(t, slot);
        nf_lru_push_front(t, slot);
        return;
    }

    /* New position — acquire a slot */
    if (t.free_count > 0) {
        /* Reuse a previously freed slot */
        slot = t.free_slots[--t.free_count];
    } else if (t.next_fresh < t.capacity) {
        /* Use a never-used slot */
        slot = t.next_fresh++;
    } else {
        /* All slots in use — evict tail (LRU) to make room */
        slot = t.tail;
        t.pos_to_slot[t.nodes[slot].pos] = NF_LRU_INVALID;
        nf_lru_unlink(t, slot);
        --t.count;
    }

    t.nodes[slot].pos = pos;
    t.pos_to_slot[pos] = slot;
    nf_lru_push_front(t, slot);
    ++t.count;
}

/* Get the LRU (least recently used) position. Returns NF_LRU_INVALID if empty. */
inline uint32_t nf_lru_coldest(const nf_lru_tracker& t) {
    if (t.tail == NF_LRU_INVALID) return NF_LRU_INVALID;
    return t.nodes[t.tail].pos;
}

/* Evict the LRU position. Returns the evicted position. */
inline uint32_t nf_lru_evict(nf_lru_tracker& t) {
    if (t.tail == NF_LRU_INVALID) return NF_LRU_INVALID;
    uint32_t slot = t.tail;
    uint32_t pos = t.nodes[slot].pos;
    t.pos_to_slot[pos] = NF_LRU_INVALID;
    nf_lru_unlink(t, slot);
    t.free_slots[t.free_count++] = slot;
    --t.count;
    return pos;
}

/* Check if a position is tracked */
inline bool nf_lru_contains(const nf_lru_tracker& t, uint32_t pos) {
    return pos < t.max_positions && t.pos_to_slot[pos] != NF_LRU_INVALID;
}

} /* namespace nf */

#endif /* NF_KV_CACHE_POLICY_HPP */
