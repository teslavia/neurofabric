/**
 * @file metal_provider.mm
 * @brief Apple Metal Execution Provider — Real GPU Compute
 *
 * Phase 9: Physical Silicon Ignition.
 *
 * Replaces the Phase 6-7 behavioral simulation with real Metal API calls.
 * Data physically flows through M4 Pro GPU shader cores via MTLComputePipelineState.
 * Async addCompletedHandler drives the existing fence mechanism — no thread pool blocking.
 *
 * Embedded MSL shaders: vector_add, relu, attention_prefill_k, attention_prefill_v.
 * All compiled once at init via newLibraryWithSource:.
 *
 * ABI contract unchanged: extern "C" exports nf_plugin_register + nf_plugin_register_mem.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_scheduler_abi.h"

#include "metal_pso_registry.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

/* ================================================================== */
/*  Embedded MSL Shader Source                                          */
/* ================================================================== */

static NSString* const kShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

constant float NF_NEG_INF = -1e30f;
constant uint NF_MAX_HEAD_DIM = 128;

kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out     [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

kernel void relu(device const float* in [[buffer(0)]],
                 device float* out      [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    out[id] = in[id] > 0.0f ? in[id] : 0.0f;
}
kernel void attention_prefill_k(device const float* in [[buffer(0)]],
                                device float* out      [[buffer(1)]],
                                uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 0.5f;
}

kernel void attention_prefill_v(device const float* in [[buffer(0)]],
                                device float* out      [[buffer(1)]],
                                uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * -0.25f;
}

/* ---- Phase 16: LLM Compute Kernels ---- */

struct PushConstants {
    uint  seq_len;
    uint  n_heads;
    uint  head_dim;
    float epsilon;
    float theta;
    uint  M;
    uint  N;
    uint  K;
    uint  step_idx;
    uint  max_seq_len;  // KV cache max capacity
    uint  window_size;  // 0 = full causal, >0 = sliding window (Mistral)
    uint  _pad1;
};

kernel void rms_norm(device const float* in       [[buffer(0)]],
                     device float* out             [[buffer(1)]],
                     device const float* weights   [[buffer(2)]],
                     constant PushConstants& pc    [[buffer(15)]],
                     uint id [[thread_position_in_grid]]) {
    uint row = id / pc.head_dim;
    uint col = id % pc.head_dim;
    uint dim = pc.head_dim;

    /* Compute mean of squares for this row */
    float sum_sq = 0.0f;
    for (uint j = 0; j < dim; ++j) {
        float v = in[row * dim + j];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / float(dim) + pc.epsilon);
    out[id] = in[id] * rms * weights[col];
}

kernel void rope(device const float* in  [[buffer(0)]],
                 device float* out       [[buffer(1)]],
                 constant PushConstants& pc [[buffer(15)]],
                 uint id [[thread_position_in_grid]]) {
    uint half_dim = pc.head_dim / 2;
    uint pair = id % half_dim;
    uint base = id - pair;

    float freq = 1.0f / pow(pc.theta, float(2 * pair) / float(pc.head_dim));
    float angle = float(pc.seq_len + pc.step_idx) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = in[base + pair];
    float x1 = in[base + pair + half_dim];
    out[base + pair]            = x0 * cos_a - x1 * sin_a;
    out[base + pair + half_dim] = x0 * sin_a + x1 * cos_a;
}

/* ---- Phase 21: Position-aware RoPE for multi-token prefill ---- */

kernel void rope_batch(device const float* in  [[buffer(0)]],
                       device float* out       [[buffer(1)]],
                       constant PushConstants& pc [[buffer(15)]],
                       uint id [[thread_position_in_grid]]) {
    // Layout: [seq_len, n_heads, head_dim] flattened
    uint half_dim = pc.head_dim / 2;
    uint head_size = pc.head_dim;
    uint elem_in_head = id % head_size;
    uint pos = (id / (pc.n_heads * head_size)) + pc.step_idx;  // token position
    uint pair = elem_in_head % half_dim;

    float freq = 1.0f / pow(pc.theta, float(2 * pair) / float(pc.head_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint base = id - elem_in_head;
    float x0 = in[base + pair];
    float x1 = in[base + pair + half_dim];
    if (elem_in_head < half_dim)
        out[id] = x0 * cos_a - x1 * sin_a;
    else
        out[id] = x0 * sin_a + x1 * cos_a;
}

kernel void linear(device const float* A   [[buffer(0)]],
                   device const float* B   [[buffer(1)]],
                   device float* C         [[buffer(2)]],
                   constant PushConstants& pc [[buffer(15)]],
                   uint2 id [[thread_position_in_grid]]) {
    uint row = id.y;
    uint col = id.x;
    if (row >= pc.M || col >= pc.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < pc.K; ++k) {
        acc += A[row * pc.K + k] * B[k * pc.N + col];
    }
    C[row * pc.N + col] = acc;
}

kernel void causal_attention(device const float* Q [[buffer(0)]],
                             device const float* K [[buffer(1)]],
                             device const float* V [[buffer(2)]],
                             device float* out     [[buffer(3)]],
                             constant PushConstants& pc [[buffer(15)]],
                             uint2 id [[thread_position_in_grid]]) {
    uint head = id.y;
    uint q_pos = id.x;
    if (head >= pc.n_heads || q_pos >= pc.seq_len) return;

    uint dim = pc.head_dim;
    float scale = rsqrt(float(dim));

    /* Compute attention scores for this query position */
    float max_score = NF_NEG_INF;
    for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                 * K[head * pc.seq_len * dim + k_pos * dim + d];
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }

    /* Softmax + weighted sum of V */
    float sum_exp = 0.0f;
    for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                 * K[head * pc.seq_len * dim + k_pos * dim + d];
        sum_exp += exp(dot * scale - max_score);
    }

    for (uint d = 0; d < dim; ++d) {
        float val = 0.0f;
        for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint dd = 0; dd < dim; ++dd)
                dot += Q[head * pc.seq_len * dim + q_pos * dim + dd]
                     * K[head * pc.seq_len * dim + k_pos * dim + dd];
            float w = exp(dot * scale - max_score) / sum_exp;
            val += w * V[head * pc.seq_len * dim + k_pos * dim + d];
        }
        out[head * pc.seq_len * dim + q_pos * dim + d] = val;
    }
}

/* ---- Phase 19: KV Cache Attention (dual-mode: prefill + decode) ---- */
/* GQA support: pc.M carries n_kv_heads. If M==0 or M==n_heads, standard MHA. */

kernel void causal_attention_cached(
        device const float* Q       [[buffer(0)]],
        device const float* K_new   [[buffer(1)]],
        device const float* V_new   [[buffer(2)]],
        device float*       cache_k [[buffer(3)]],
        device float*       cache_v [[buffer(4)]],
        device float*       out     [[buffer(5)]],
        constant PushConstants& pc  [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint head  = id.y;
    uint q_pos = id.x;
    uint dim   = pc.head_dim;
    uint step  = pc.step_idx;
    uint n_kv  = (pc.M > 0 && pc.M < pc.n_heads) ? pc.M : pc.n_heads;
    uint kv_head = head * n_kv / pc.n_heads;  /* GQA: map Q head → KV head */

    if (head >= pc.n_heads) return;

    /* ---- Prefill mode (step == 0): full causal attention ---- */
    if (step == 0) {
        if (q_pos >= pc.seq_len) return;
        /* Copy K_new/V_new into cache (only KV-head threads write) */
        if (head < n_kv) {
            for (uint d = 0; d < dim; ++d) {
                uint src = head * pc.seq_len * dim + q_pos * dim + d;
                uint dst = head * pc.max_seq_len * dim + q_pos * dim + d;
                cache_k[dst] = K_new[src];
                cache_v[dst] = V_new[src];
            }
        }
        /* Sliding window: limit attention span when window_size > 0 */
        uint kv_start = 0;
        if (pc.window_size > 0 && q_pos >= pc.window_size)
            kv_start = q_pos - pc.window_size + 1;

        /* Causal attention with optional sliding window */
        float scale = rsqrt(float(dim));
        float max_score = NF_NEG_INF;
        for (uint k_pos = kv_start; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                     * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            dot *= scale;
            if (dot > max_score) max_score = dot;
        }
        float sum_exp = 0.0f;
        for (uint k_pos = kv_start; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                     * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            sum_exp += exp(dot * scale - max_score);
        }
        for (uint d = 0; d < dim; ++d) {
            float val = 0.0f;
            for (uint k_pos = kv_start; k_pos <= q_pos; ++k_pos) {
                float dot2 = 0.0f;
                for (uint dd = 0; dd < dim; ++dd)
                    dot2 += Q[head * pc.seq_len * dim + q_pos * dim + dd]
                          * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + dd];
                float w = exp(dot2 * scale - max_score) / sum_exp;
                val += w * cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            }
            out[head * pc.seq_len * dim + q_pos * dim + d] = val;
        }
        return;
    }

    /* ---- Decode mode (step > 0): single-token attention ---- */
    if (q_pos >= 1) return;  /* only 1 query token in decode */
    if (step >= pc.max_seq_len) return;  /* bounds check */

    /* Append new K/V to cache at position [step] (only KV-head threads write) */
    if (head < n_kv) {
        for (uint d = 0; d < dim; ++d) {
            cache_k[head * pc.max_seq_len * dim + step * dim + d] = K_new[head * dim + d];
            cache_v[head * pc.max_seq_len * dim + step * dim + d] = V_new[head * dim + d];
        }
    }

    /* Attention: Q (1 token) against cache_k[kv_start..step] */
    float scale = rsqrt(float(dim));
    uint kv_start = 0;
    if (pc.window_size > 0 && step >= pc.window_size)
        kv_start = step - pc.window_size + 1;

    float max_score = NF_NEG_INF;
    for (uint k_pos = kv_start; k_pos <= step; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * dim + d]
                 * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }
    float sum_exp = 0.0f;
    for (uint k_pos = kv_start; k_pos <= step; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * dim + d]
                 * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
        sum_exp += exp(dot * scale - max_score);
    }
    for (uint d = 0; d < dim; ++d) {
        float val = 0.0f;
        for (uint k_pos = kv_start; k_pos <= step; ++k_pos) {
            float dot2 = 0.0f;
            for (uint dd = 0; dd < dim; ++dd)
                dot2 += Q[head * dim + dd]
                      * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + dd];
            float w = exp(dot2 * scale - max_score) / sum_exp;
            val += w * cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
        }
        out[head * dim + d] = val;
    }
}

/* ---- Phase 26: Flash Attention (Tiled Online Softmax) ---- */
/* Same signature as causal_attention_cached but uses tiled KV access
   with online softmax — O(1) extra memory per head instead of O(seq²).
   Handles both prefill (step==0, seq_len tokens) and decode (step>0, 1 token).
   GQA: pc.M = n_kv_heads, same mapping as causal_attention_cached.
   Sliding window: pc.window_size > 0 limits attention span. */

constant uint FA_TILE_KV = 32;

kernel void flash_attention_tiled(
        device const float* Q       [[buffer(0)]],
        device const float* K_new   [[buffer(1)]],
        device const float* V_new   [[buffer(2)]],
        device float*       cache_k [[buffer(3)]],
        device float*       cache_v [[buffer(4)]],
        device float*       out     [[buffer(5)]],
        constant PushConstants& pc  [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint head  = id.y;
    uint q_pos = id.x;
    uint dim   = pc.head_dim;
    uint step  = pc.step_idx;
    uint n_kv  = (pc.M > 0 && pc.M < pc.n_heads) ? pc.M : pc.n_heads;
    uint kv_head = head * n_kv / pc.n_heads;

    if (head >= pc.n_heads) return;

    /* ---- Prefill mode (step == 0) ---- */
    if (step == 0) {
        if (q_pos >= pc.seq_len) return;

        /* Copy K_new/V_new into cache (KV-head threads only) */
        if (head < n_kv) {
            for (uint d = 0; d < dim; ++d) {
                uint src = head * pc.seq_len * dim + q_pos * dim + d;
                uint dst = head * pc.max_seq_len * dim + q_pos * dim + d;
                cache_k[dst] = K_new[src];
                cache_v[dst] = V_new[src];
            }
        }

        /* Online softmax attention over cache[0..q_pos] */
        float scale = rsqrt(float(dim));
        float m = NF_NEG_INF;  /* running max */
        float l = 0.0f;    /* running sum of exp */

        /* Accumulator for output (dim floats) — stored in registers */
        /* For head_dim up to 128, this fits in registers on Apple GPUs */
        float acc[NF_MAX_HEAD_DIM];
        for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

        /* Determine attention window */
        uint kv_start = 0;
        if (pc.window_size > 0 && q_pos >= pc.window_size)
            kv_start = q_pos - pc.window_size + 1;

        uint kv_end = q_pos;  /* inclusive, causal mask */

        /* Tile over KV positions */
        for (uint t_start = kv_start; t_start <= kv_end; t_start += FA_TILE_KV) {
            uint t_end = min(t_start + FA_TILE_KV - 1, kv_end);

            /* Compute scores for this tile */
            float tile_max = NF_NEG_INF;
            for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
                float dot = 0.0f;
                for (uint d = 0; d < dim; ++d)
                    dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                         * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
                dot *= scale;
                if (dot > tile_max) tile_max = dot;
            }

            /* Online softmax update */
            float m_new = max(m, tile_max);
            float correction = exp(m - m_new);

            /* Rescale existing accumulator */
            for (uint d = 0; d < dim; ++d)
                acc[d] *= correction;
            l *= correction;

            /* Accumulate this tile */
            for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
                float dot = 0.0f;
                for (uint d = 0; d < dim; ++d)
                    dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                         * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
                float w = exp(dot * scale - m_new);
                l += w;
                for (uint d = 0; d < dim; ++d)
                    acc[d] += w * cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            }
            m = m_new;
        }

        /* Normalize and write output */
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint d = 0; d < dim; ++d)
            out[head * pc.seq_len * dim + q_pos * dim + d] = acc[d] * inv_l;
        return;
    }

    /* ---- Decode mode (step > 0): single-token ---- */
    if (q_pos >= 1) return;
    if (step >= pc.max_seq_len) return;

    /* Append new K/V to cache */
    if (head < n_kv) {
        for (uint d = 0; d < dim; ++d) {
            cache_k[head * pc.max_seq_len * dim + step * dim + d] = K_new[head * dim + d];
            cache_v[head * pc.max_seq_len * dim + step * dim + d] = V_new[head * dim + d];
        }
    }

    /* Online softmax attention over cache[kv_start..step] */
    float scale = rsqrt(float(dim));
    float m_val = NF_NEG_INF;
    float l_val = 0.0f;
    float acc[NF_MAX_HEAD_DIM];
    for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

    uint kv_start = 0;
    if (pc.window_size > 0 && step >= pc.window_size)
        kv_start = step - pc.window_size + 1;

    for (uint t_start = kv_start; t_start <= step; t_start += FA_TILE_KV) {
        uint t_end = min(t_start + FA_TILE_KV - 1, step);

        float tile_max = NF_NEG_INF;
        for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * dim + d]
                     * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            dot *= scale;
            if (dot > tile_max) tile_max = dot;
        }

        float m_new = max(m_val, tile_max);
        float correction = exp(m_val - m_new);
        for (uint d = 0; d < dim; ++d)
            acc[d] *= correction;
        l_val *= correction;

        for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * dim + d]
                     * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            float w = exp(dot * scale - m_new);
            l_val += w;
            for (uint d = 0; d < dim; ++d)
                acc[d] += w * cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
        }
        m_val = m_new;
    }

    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    for (uint d = 0; d < dim; ++d)
        out[head * dim + d] = acc[d] * inv_l;
}

/* ---- Phase 17: Dequantization Kernels ---- */

struct block_q4_0 {
    half  d;           // scale factor
    uchar qs[16];      // 4-bit quants packed (2 per byte), 32 elements
};

struct block_q8_0 {
    half  d;           // scale factor
    char  qs[32];      // 8-bit quants, 32 elements
};

kernel void dequant_q4_0(device const block_q4_0* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    // ggml Q4_0: low nibble = elements 0..15, high nibble = elements 16..31
    // byte index = elem_idx % 16, low nibble for elem < 16, high for elem >= 16
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    out[id] = d * float(nibble - 8);
}

kernel void dequant_q8_0(device const block_q8_0* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    out[id] = d * float(blocks[block_idx].qs[elem_idx]);
}

/* ---- Phase 21: Q6_K Dequantization ---- */

struct block_q6_k {
    uchar ql[128];     // lower 4 bits of quants
    uchar qh[64];      // upper 2 bits of quants
    char  scales[16];  // per-sub-block scales
    half  d;           // super-block scale
};

kernel void dequant_q6_k(device const block_q6_k* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;

    float super_d = float(blocks[block_idx].d);

    // 256 elements split into 16 sub-blocks of 16 elements each
    uint sub_block = elem_idx / 16;   // 0..15
    uint sub_elem  = elem_idx % 16;   // 0..15

    // Scale for this sub-block
    float scale = float(blocks[block_idx].scales[sub_block]) * super_d;

    // Reconstruct 6-bit quant from ql (4 low bits) and qh (2 high bits)
    // ql: 128 bytes = 256 nibbles (low 4 bits per element)
    // Each byte of ql holds 2 elements: low nibble = even, high nibble = odd
    uint ql_byte_idx = elem_idx / 2;
    uchar ql_byte = blocks[block_idx].ql[ql_byte_idx];
    int ql_val = (elem_idx & 1) ? (ql_byte >> 4) : (ql_byte & 0xF);

    // qh: 64 bytes = 256 2-bit values
    // Each byte of qh holds 4 elements (2 bits each)
    uint qh_byte_idx = elem_idx / 4;
    uint qh_shift = (elem_idx % 4) * 2;
    int qh_val = (blocks[block_idx].qh[qh_byte_idx] >> qh_shift) & 0x3;

    // Combine: 6-bit value = (qh << 4) | ql, then subtract 32 for signed
    int quant = (qh_val << 4) | ql_val;
    quant -= 32;

    out[id] = scale * float(quant);
}

/* ---- Phase 25: K-Quant Dequantization Kernels ---- */

/* Q4_1: 32-element blocks, 20 bytes (half d, half m, uchar qs[16]) */
struct block_q4_1 { half d; half m; uchar qs[16]; };

kernel void dequant_q4_1(device const block_q4_1* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    float m = float(blocks[block_idx].m);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    out[id] = d * float(nibble) + m;
}

/* Q5_0: 32-element blocks, 22 bytes (half d, uchar qh[4], uchar qs[16]) */
struct block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };

kernel void dequant_q5_0(device const block_q5_0* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    /* 5th bit from qh */
    int bit = (blocks[block_idx].qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
    out[id] = d * float((nibble | (bit << 4)) - 16);
}

/* Q5_1: 32-element blocks, 24 bytes (half d, half m, uchar qh[4], uchar qs[16]) */
struct block_q5_1 { half d; half m; uchar qh[4]; uchar qs[16]; };

kernel void dequant_q5_1(device const block_q5_1* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    float m = float(blocks[block_idx].m);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    int bit = (blocks[block_idx].qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
    out[id] = d * float(nibble | (bit << 4)) + m;
}

/* Q2_K: 256-element blocks, 84 bytes */
struct block_q2_k { uchar scales[16]; uchar qs[64]; half d; half dmin; };

kernel void dequant_q2_k(device const block_q2_k* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d    = float(blocks[block_idx].d);
    float super_dmin = float(blocks[block_idx].dmin);
    /* 16 sub-blocks of 16 elements */
    uint sub = elem_idx / 16;
    uint sub_elem = elem_idx % 16;
    uchar sc_byte = blocks[block_idx].scales[sub];
    float sc  = float(sc_byte & 0xF) * super_d;
    float mn  = float(sc_byte >> 4) * super_dmin;
    /* 2-bit quants: 4 per byte */
    uint qs_idx = elem_idx / 4;
    uint qs_shift = (elem_idx % 4) * 2;
    int q2 = (blocks[block_idx].qs[qs_idx] >> qs_shift) & 0x3;
    out[id] = sc * float(q2) - mn;
}

/* Q3_K: 256-element blocks, 110 bytes */
struct block_q3_k { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; };

kernel void dequant_q3_k(device const block_q3_k* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d = float(blocks[block_idx].d);

    /* Decode 6-bit scale for this sub-block (16 sub-blocks of 16 elements) */
    uint sub = elem_idx / 16;
    int scale;
    if (sub < 8) {
        scale = int(blocks[block_idx].scales[sub] & 0xF)
              | ((int(blocks[block_idx].scales[8 + sub / 2] >> (4 * (sub % 2))) & 0x3) << 4);
    } else {
        uint s2 = sub - 8;
        scale = int(blocks[block_idx].scales[s2] >> 4)
              | ((int(blocks[block_idx].scales[8 + s2 / 2] >> (4 * (s2 % 2) + 2)) & 0x3) << 4);
    }
    scale -= 32;

    /* 3-bit quant: 2 low bits from qs, 1 high bit from hmask */
    uint qs_idx = elem_idx / 4;
    uint qs_shift = (elem_idx % 4) * 2;
    int q_lo = (blocks[block_idx].qs[qs_idx] >> qs_shift) & 0x3;
    int q_hi = (blocks[block_idx].hmask[elem_idx / 8] >> (elem_idx % 8)) & 1;
    int q3 = q_lo | (q_hi << 2);
    out[id] = super_d * float(scale) * float(q3 - 4);
}

/* Q4_K: 256-element blocks, 144 bytes */
struct block_q4_k { half d; half dmin; uchar scales[12]; uchar qs[128]; };

kernel void dequant_q4_k(device const block_q4_k* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d    = float(blocks[block_idx].d);
    float super_dmin = float(blocks[block_idx].dmin);

    /* 8 sub-blocks of 32 elements; 6-bit scale/min packed in 12 bytes */
    uint sub = elem_idx / 32;
    int sc, mn;
    if (sub < 4) {
        sc = int(blocks[block_idx].scales[sub] & 0x3F);
        mn = int(blocks[block_idx].scales[sub + 4] & 0x3F);
    } else {
        uint s2 = sub - 4;
        sc = int((blocks[block_idx].scales[s2]     >> 6) | ((blocks[block_idx].scales[s2 + 8] & 0x0F) << 2));
        mn = int((blocks[block_idx].scales[s2 + 4] >> 6) | ((blocks[block_idx].scales[s2 + 8] >> 4)   << 2));
    }

    /* 4-bit quant from qs */
    uint qs_idx = elem_idx / 2;
    int nibble = (elem_idx & 1) ? (blocks[block_idx].qs[qs_idx] >> 4) : (blocks[block_idx].qs[qs_idx] & 0xF);
    out[id] = super_d * float(sc) * float(nibble) - super_dmin * float(mn);
}

/* Q5_K: 256-element blocks, 176 bytes */
struct block_q5_k { half d; half dmin; uchar scales[12]; uchar qs[128]; uchar qh[32]; };

kernel void dequant_q5_k(device const block_q5_k* blocks [[buffer(0)]],
                          device float* out              [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d    = float(blocks[block_idx].d);
    float super_dmin = float(blocks[block_idx].dmin);

    /* Same scale packing as Q4_K */
    uint sub = elem_idx / 32;
    int sc, mn;
    if (sub < 4) {
        sc = int(blocks[block_idx].scales[sub] & 0x3F);
        mn = int(blocks[block_idx].scales[sub + 4] & 0x3F);
    } else {
        uint s2 = sub - 4;
        sc = int((blocks[block_idx].scales[s2]     >> 6) | ((blocks[block_idx].scales[s2 + 8] & 0x0F) << 2));
        mn = int((blocks[block_idx].scales[s2 + 4] >> 6) | ((blocks[block_idx].scales[s2 + 8] >> 4)   << 2));
    }

    /* 5-bit quant: 4 low bits from qs, 1 high bit from qh */
    uint qs_idx = elem_idx / 2;
    int nibble = (elem_idx & 1) ? (blocks[block_idx].qs[qs_idx] >> 4) : (blocks[block_idx].qs[qs_idx] & 0xF);
    int bit = (blocks[block_idx].qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
    out[id] = super_d * float(sc) * float(nibble | (bit << 4)) - super_dmin * float(mn);
}

/* ---- Phase 27: Dequantization → F16 output variants ---- */

kernel void dequant_q4_0_f16(device const block_q4_0* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    out[id] = half(d * float(nibble - 8));
}

kernel void dequant_q8_0_f16(device const block_q8_0* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    out[id] = half(d * float(blocks[block_idx].qs[elem_idx]));
}

kernel void dequant_q6_k_f16(device const block_q6_k* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d = float(blocks[block_idx].d);
    uint sub_block = elem_idx / 16;
    float scale = float(blocks[block_idx].scales[sub_block]) * super_d;
    uint ql_byte_idx = elem_idx / 2;
    uchar ql_byte = blocks[block_idx].ql[ql_byte_idx];
    int ql_val = (elem_idx & 1) ? (ql_byte >> 4) : (ql_byte & 0xF);
    uint qh_byte_idx = elem_idx / 4;
    uint qh_shift = (elem_idx % 4) * 2;
    int qh_val = (blocks[block_idx].qh[qh_byte_idx] >> qh_shift) & 0x3;
    int quant = (qh_val << 4) | ql_val;
    quant -= 32;
    out[id] = half(scale * float(quant));
}

kernel void dequant_q4_1_f16(device const block_q4_1* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    float m = float(blocks[block_idx].m);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    out[id] = half(d * float(nibble) + m);
}

kernel void dequant_q5_0_f16(device const block_q5_0* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    int bit = (blocks[block_idx].qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
    out[id] = half(d * float((nibble | (bit << 4)) - 16));
}

kernel void dequant_q5_1_f16(device const block_q5_1* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 32;
    uint elem_idx  = id % 32;
    float d = float(blocks[block_idx].d);
    float m = float(blocks[block_idx].m);
    uchar byte_val = blocks[block_idx].qs[elem_idx % 16];
    int nibble = (elem_idx < 16) ? (byte_val & 0xF) : (byte_val >> 4);
    int bit = (blocks[block_idx].qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
    out[id] = half(d * float(nibble | (bit << 4)) + m);
}

kernel void dequant_q2_k_f16(device const block_q2_k* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d    = float(blocks[block_idx].d);
    float super_dmin = float(blocks[block_idx].dmin);
    uint sub = elem_idx / 16;
    int sc = int(blocks[block_idx].scales[sub] & 0xF);
    int mn = int(blocks[block_idx].scales[sub] >> 4);
    uint qs_idx = elem_idx / 4;
    uint shift  = (elem_idx % 4) * 2;
    int q2 = (blocks[block_idx].qs[qs_idx] >> shift) & 0x3;
    out[id] = half(super_d * float(sc) * float(q2) - super_dmin * float(mn));
}

kernel void dequant_q3_k_f16(device const block_q3_k* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d = float(blocks[block_idx].d);
    uint sub = elem_idx / 16;
    int raw_scale = int(blocks[block_idx].scales[sub]);
    int scale = raw_scale - 32;
    uint qs_idx = elem_idx / 4;
    uint shift  = (elem_idx % 4) * 2;
    int q_lo = (blocks[block_idx].qs[qs_idx] >> shift) & 0x3;
    int q_hi = (blocks[block_idx].hmask[elem_idx / 8] >> (elem_idx % 8)) & 1;
    int q3 = q_lo | (q_hi << 2);
    out[id] = half(super_d * float(scale) * float(q3 - 4));
}

kernel void dequant_q4_k_f16(device const block_q4_k* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d    = float(blocks[block_idx].d);
    float super_dmin = float(blocks[block_idx].dmin);
    uint sub = elem_idx / 32;
    int sc, mn;
    if (sub < 4) {
        sc = int(blocks[block_idx].scales[sub] & 0x3F);
        mn = int(blocks[block_idx].scales[sub + 4] & 0x3F);
    } else {
        uint s2 = sub - 4;
        sc = int((blocks[block_idx].scales[s2]     >> 6) | ((blocks[block_idx].scales[s2 + 8] & 0x0F) << 2));
        mn = int((blocks[block_idx].scales[s2 + 4] >> 6) | ((blocks[block_idx].scales[s2 + 8] >> 4)   << 2));
    }
    uint qs_idx = elem_idx / 2;
    int nibble = (elem_idx & 1) ? (blocks[block_idx].qs[qs_idx] >> 4) : (blocks[block_idx].qs[qs_idx] & 0xF);
    out[id] = half(super_d * float(sc) * float(nibble) - super_dmin * float(mn));
}

kernel void dequant_q5_k_f16(device const block_q5_k* blocks [[buffer(0)]],
                              device half* out                [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint block_idx = id / 256;
    uint elem_idx  = id % 256;
    float super_d    = float(blocks[block_idx].d);
    float super_dmin = float(blocks[block_idx].dmin);
    uint sub = elem_idx / 32;
    int sc, mn;
    if (sub < 4) {
        sc = int(blocks[block_idx].scales[sub] & 0x3F);
        mn = int(blocks[block_idx].scales[sub + 4] & 0x3F);
    } else {
        uint s2 = sub - 4;
        sc = int((blocks[block_idx].scales[s2]     >> 6) | ((blocks[block_idx].scales[s2 + 8] & 0x0F) << 2));
        mn = int((blocks[block_idx].scales[s2 + 4] >> 6) | ((blocks[block_idx].scales[s2 + 8] >> 4)   << 2));
    }
    uint qs_idx = elem_idx / 2;
    int nibble = (elem_idx & 1) ? (blocks[block_idx].qs[qs_idx] >> 4) : (blocks[block_idx].qs[qs_idx] & 0xF);
    int bit = (blocks[block_idx].qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
    out[id] = half(super_d * float(sc) * float(nibble | (bit << 4)) - super_dmin * float(mn));
}

/* ---- Phase 17: Tiled MatMul (16x16 threadgroup tiles) ---- */

constant uint TILE_SIZE = 16;

kernel void linear_tiled(device const float* A   [[buffer(0)]],
                          device const float* B   [[buffer(1)]],
                          device float* C         [[buffer(2)]],
                          constant PushConstants& pc [[buffer(15)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    uint row = gid.y;
    uint col = gid.x;

    threadgroup float tileA[TILE_SIZE][TILE_SIZE + 1];  // +1 padding avoids bank conflicts
    threadgroup float tileB[TILE_SIZE][TILE_SIZE + 1];

    float acc = 0.0f;
    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * TILE_SIZE + lid.x;
        uint bRow = t * TILE_SIZE + lid.y;

        tileA[lid.y][lid.x] = (row < pc.M && aCol < pc.K)
            ? A[row * pc.K + aCol] : 0.0f;
        tileB[lid.y][lid.x] = (bRow < pc.K && col < pc.N)
            ? B[bRow * pc.N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; ++k)
            acc += tileA[lid.y][k] * tileB[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < pc.M && col < pc.N)
        C[row * pc.N + col] = acc;
}

/* ---- Phase 24: SIMD Group MatMul (8x8 simdgroup tiles) ---- */

kernel void linear_simd(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device float*       C          [[buffer(2)]],
    constant PushConstants& pc     [[buffer(15)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lid   [[thread_index_in_simdgroup]])
{
    /* Each threadgroup handles a 32x32 output tile.
       4x4 = 16 simdgroups, each computing an 8x8 sub-tile.
       16 simdgroups × 32 threads = 512 threads per threadgroup. */
    const uint sg_row = sgid / 4;
    const uint sg_col = sgid % 4;

    const uint row_base = tgid.y * 32 + sg_row * 8;
    const uint col_base = tgid.x * 32 + sg_col * 8;

    /* Early exit: entire 8x8 tile is out of bounds */
    if (row_base >= pc.M && col_base >= pc.N) return;

    simdgroup_float8x8 acc;
    simdgroup_float8x8 tileA;
    simdgroup_float8x8 tileB;

    /* Zero accumulator */
    acc = simdgroup_float8x8(0);

    /* Accumulate K dimension in tiles of 8 */
    for (uint k = 0; k < pc.K; k += 8) {
        /* Load 8x8 from A[row_base:row_base+8, k:k+8] */
        if (row_base + 8 <= pc.M && k + 8 <= pc.K) {
            simdgroup_load(tileA, A + row_base * pc.K + k, pc.K);
        } else {
            /* Bounds-safe load: zero-pad out-of-range elements */
            tileA = simdgroup_float8x8(0);
            if (row_base < pc.M && k < pc.K)
                simdgroup_load(tileA, A + row_base * pc.K + k, pc.K,
                               ulong2(min(pc.K - k, 8u), min(pc.M - row_base, 8u)));
        }

        /* Load 8x8 from B[k:k+8, col_base:col_base+8] */
        if (k + 8 <= pc.K && col_base + 8 <= pc.N) {
            simdgroup_load(tileB, B + k * pc.N + col_base, pc.N);
        } else {
            tileB = simdgroup_float8x8(0);
            if (k < pc.K && col_base < pc.N)
                simdgroup_load(tileB, B + k * pc.N + col_base, pc.N,
                               ulong2(min(pc.N - col_base, 8u), min(pc.K - k, 8u)));
        }

        simdgroup_multiply_accumulate(acc, tileA, tileB, acc);
    }

    /* Store 8x8 result to C[row_base:row_base+8, col_base:col_base+8] */
    if (row_base + 8 <= pc.M && col_base + 8 <= pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N);
    } else if (row_base < pc.M && col_base < pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N,
                        ulong2(min(pc.N - col_base, 8u), min(pc.M - row_base, 8u)));
    }
}

/* ---- Phase 18: Transformer Layer Primitives ---- */

kernel void softmax(device const float* in  [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    constant PushConstants& pc [[buffer(15)]],
                    uint id [[thread_position_in_grid]]) {
    /* One thread per row. pc.head_dim = row width, pc.seq_len = num rows */
    if (id >= pc.seq_len) return;
    uint dim = pc.head_dim;
    uint base = id * dim;

    /* Pass 1: find max for numerical stability */
    float max_val = in[base];
    for (uint j = 1; j < dim; ++j) {
        float v = in[base + j];
        if (v > max_val) max_val = v;
    }

    /* Pass 2: exp and sum */
    float sum_exp = 0.0f;
    for (uint j = 0; j < dim; ++j) {
        float e = exp(in[base + j] - max_val);
        out[base + j] = e;
        sum_exp += e;
    }

    /* Pass 3: normalize */
    float inv_sum = 1.0f / sum_exp;
    for (uint j = 0; j < dim; ++j)
        out[base + j] *= inv_sum;
}

kernel void silu(device const float* in  [[buffer(0)]],
                 device float* out       [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    float x = in[id];
    out[id] = x / (1.0f + exp(-x));
}

kernel void gelu(device const float* in  [[buffer(0)]],
                 device float* out       [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    float x = in[id];
    float x3 = x * x * x;
    out[id] = 0.5f * x * (1.0f + precise::tanh(0.7978845608f * (x + 0.044715f * x3)));
}

kernel void elementwise_mul(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* out     [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void embedding_lookup(device const float* weights   [[buffer(0)]],
                              device const int*   token_ids [[buffer(1)]],
                              device float* out             [[buffer(2)]],
                              constant PushConstants& pc    [[buffer(15)]],
                              uint id [[thread_position_in_grid]]) {
    /* One thread per output element. pc.head_dim = embed dim, pc.seq_len = num tokens */
    uint tok_idx = id / pc.head_dim;
    uint dim_idx = id % pc.head_dim;
    if (tok_idx >= pc.seq_len) return;
    int token = token_ids[tok_idx];
    out[id] = weights[token * pc.head_dim + dim_idx];
}

/* ---- Phase 27: FP16 Compute Kernels ---- */

kernel void rms_norm_f16(device const half* in       [[buffer(0)]],
                          device half* out             [[buffer(1)]],
                          device const float* weights  [[buffer(2)]],
                          constant PushConstants& pc   [[buffer(15)]],
                          uint id [[thread_position_in_grid]]) {
    uint row = id / pc.head_dim;
    uint col = id % pc.head_dim;
    uint dim = pc.head_dim;
    float sum_sq = 0.0f;
    for (uint j = 0; j < dim; ++j) {
        float v = float(in[row * dim + j]);
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / float(dim) + pc.epsilon);
    out[id] = half(float(in[id]) * rms * weights[col]);
}

kernel void rope_batch_f16(device const half* in  [[buffer(0)]],
                            device half* out       [[buffer(1)]],
                            constant PushConstants& pc [[buffer(15)]],
                            uint id [[thread_position_in_grid]]) {
    uint half_dim = pc.head_dim / 2;
    uint head_size = pc.head_dim;
    uint elem_in_head = id % head_size;
    uint pos = (id / (pc.n_heads * head_size)) + pc.step_idx;
    uint pair = elem_in_head % half_dim;
    float freq = 1.0f / pow(pc.theta, float(2 * pair) / float(pc.head_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    uint base = id - elem_in_head;
    float x0 = float(in[base + pair]);
    float x1 = float(in[base + pair + half_dim]);
    if (elem_in_head < half_dim)
        out[id] = half(x0 * cos_a - x1 * sin_a);
    else
        out[id] = half(x0 * sin_a + x1 * cos_a);
}

kernel void silu_f16(device const half* in  [[buffer(0)]],
                      device half* out       [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    float x = float(in[id]);
    out[id] = half(x / (1.0f + exp(-x)));
}

kernel void gelu_f16(device const half* in  [[buffer(0)]],
                      device half* out       [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    float x = float(in[id]);
    float x3 = x * x * x;
    out[id] = half(0.5f * x * (1.0f + precise::tanh(0.7978845608f * (x + 0.044715f * x3))));
}

kernel void elementwise_mul_f16(device const half* a [[buffer(0)]],
                                 device const half* b [[buffer(1)]],
                                 device half* out     [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void metal_vector_add_f16(device const half* a [[buffer(0)]],
                                  device const half* b [[buffer(1)]],
                                  device half* out     [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

kernel void embedding_lookup_f16(device const half* weights   [[buffer(0)]],
                                  device const int*  token_ids [[buffer(1)]],
                                  device half* out             [[buffer(2)]],
                                  constant PushConstants& pc   [[buffer(15)]],
                                  uint id [[thread_position_in_grid]]) {
    uint tok_idx = id / pc.head_dim;
    uint dim_idx = id % pc.head_dim;
    if (tok_idx >= pc.seq_len) return;
    int token = token_ids[tok_idx];
    out[id] = weights[token * pc.head_dim + dim_idx];
}

/* Phase 27: F16 tiled matmul */
constant uint TILE_SIZE_F16 = 16;

kernel void linear_tiled_f16(device const half* A   [[buffer(0)]],
                              device const half* B   [[buffer(1)]],
                              device half* C         [[buffer(2)]],
                              constant PushConstants& pc [[buffer(15)]],
                              uint2 gid [[thread_position_in_grid]],
                              uint2 lid [[thread_position_in_threadgroup]]) {
    uint row = gid.y;
    uint col = gid.x;
    threadgroup half tileA[16][16 + 1];  // +1 padding avoids bank conflicts
    threadgroup half tileB[16][16 + 1];
    float acc = 0.0f;
    uint numTiles = (pc.K + TILE_SIZE_F16 - 1) / TILE_SIZE_F16;
    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * TILE_SIZE_F16 + lid.x;
        uint bRow = t * TILE_SIZE_F16 + lid.y;
        tileA[lid.y][lid.x] = (row < pc.M && aCol < pc.K)
            ? A[row * pc.K + aCol] : half(0);
        tileB[lid.y][lid.x] = (bRow < pc.K && col < pc.N)
            ? B[bRow * pc.N + col] : half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE_F16; ++k)
            acc += float(tileA[lid.y][k]) * float(tileB[k][lid.x]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < pc.M && col < pc.N)
        C[row * pc.N + col] = half(acc);
}

/* Phase 27: F16 SIMD matmul — native half8x8 */
kernel void linear_simd_f16(
    device const half* A          [[buffer(0)]],
    device const half* B          [[buffer(1)]],
    device half*       C          [[buffer(2)]],
    constant PushConstants& pc    [[buffer(15)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lid   [[thread_index_in_simdgroup]])
{
    const uint sg_row = sgid / 4;
    const uint sg_col = sgid % 4;
    const uint row_base = tgid.y * 32 + sg_row * 8;
    const uint col_base = tgid.x * 32 + sg_col * 8;
    if (row_base >= pc.M && col_base >= pc.N) return;
    simdgroup_half8x8 acc;
    simdgroup_half8x8 tA;
    simdgroup_half8x8 tB;
    acc = simdgroup_half8x8(0);
    for (uint k = 0; k < pc.K; k += 8) {
        if (row_base + 8 <= pc.M && k + 8 <= pc.K) {
            simdgroup_load(tA, A + row_base * pc.K + k, pc.K);
        } else {
            tA = simdgroup_half8x8(0);
            if (row_base < pc.M && k < pc.K)
                simdgroup_load(tA, A + row_base * pc.K + k, pc.K,
                               ulong2(min(pc.K - k, 8u), min(pc.M - row_base, 8u)));
        }
        if (k + 8 <= pc.K && col_base + 8 <= pc.N) {
            simdgroup_load(tB, B + k * pc.N + col_base, pc.N);
        } else {
            tB = simdgroup_half8x8(0);
            if (k < pc.K && col_base < pc.N)
                simdgroup_load(tB, B + k * pc.N + col_base, pc.N,
                               ulong2(min(pc.N - col_base, 8u), min(pc.K - k, 8u)));
        }
        simdgroup_multiply_accumulate(acc, tA, tB, acc);
    }
    if (row_base + 8 <= pc.M && col_base + 8 <= pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N);
    } else if (row_base < pc.M && col_base < pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N,
                        ulong2(min(pc.N - col_base, 8u), min(pc.M - row_base, 8u)));
    }
}

/* Phase 27: F16 input → F32 output matmul (LM head) */
kernel void linear_f16_to_f32(
    device const half* A          [[buffer(0)]],
    device const half* B          [[buffer(1)]],
    device float*      C          [[buffer(2)]],
    constant PushConstants& pc    [[buffer(15)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lid   [[thread_index_in_simdgroup]])
{
    const uint sg_row = sgid / 4;
    const uint sg_col = sgid % 4;
    const uint row_base = tgid.y * 32 + sg_row * 8;
    const uint col_base = tgid.x * 32 + sg_col * 8;
    if (row_base >= pc.M && col_base >= pc.N) return;
    simdgroup_half8x8 tA;
    simdgroup_half8x8 tB;
    simdgroup_float8x8 acc;
    acc = simdgroup_float8x8(0);
    for (uint k = 0; k < pc.K; k += 8) {
        if (row_base + 8 <= pc.M && k + 8 <= pc.K) {
            simdgroup_load(tA, A + row_base * pc.K + k, pc.K);
        } else {
            tA = simdgroup_half8x8(0);
            if (row_base < pc.M && k < pc.K)
                simdgroup_load(tA, A + row_base * pc.K + k, pc.K,
                               ulong2(min(pc.K - k, 8u), min(pc.M - row_base, 8u)));
        }
        if (k + 8 <= pc.K && col_base + 8 <= pc.N) {
            simdgroup_load(tB, B + k * pc.N + col_base, pc.N);
        } else {
            tB = simdgroup_half8x8(0);
            if (k < pc.K && col_base < pc.N)
                simdgroup_load(tB, B + k * pc.N + col_base, pc.N,
                               ulong2(min(pc.N - col_base, 8u), min(pc.K - k, 8u)));
        }
        simdgroup_multiply_accumulate(acc, tA, tB, acc);
    }
    if (row_base + 8 <= pc.M && col_base + 8 <= pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N);
    } else if (row_base < pc.M && col_base < pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N,
                        ulong2(min(pc.N - col_base, 8u), min(pc.M - row_base, 8u)));
    }
}

/* Phase 27: Flash Attention F16 — half I/O, F32 accumulators */
kernel void flash_attention_tiled_f16(
        device const half* Q       [[buffer(0)]],
        device const half* K_new   [[buffer(1)]],
        device const half* V_new   [[buffer(2)]],
        device half*       cache_k [[buffer(3)]],
        device half*       cache_v [[buffer(4)]],
        device half*       out     [[buffer(5)]],
        constant PushConstants& pc [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint head  = id.y;
    uint q_pos = id.x;
    uint dim   = pc.head_dim;
    uint step  = pc.step_idx;
    uint n_kv  = (pc.M > 0 && pc.M < pc.n_heads) ? pc.M : pc.n_heads;
    uint kv_head = head * n_kv / pc.n_heads;
    if (head >= pc.n_heads) return;

    /* ---- Prefill mode (step == 0) ---- */
    if (step == 0) {
        if (q_pos >= pc.seq_len) return;
        if (head < n_kv) {
            for (uint d = 0; d < dim; ++d) {
                uint src = head * pc.seq_len * dim + q_pos * dim + d;
                uint dst = head * pc.max_seq_len * dim + q_pos * dim + d;
                cache_k[dst] = K_new[src];
                cache_v[dst] = V_new[src];
            }
        }
        float scale = rsqrt(float(dim));
        float m = NF_NEG_INF;
        float l = 0.0f;
        float acc[NF_MAX_HEAD_DIM];
        for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;
        uint kv_start = 0;
        if (pc.window_size > 0 && q_pos >= pc.window_size)
            kv_start = q_pos - pc.window_size + 1;
        uint kv_end = q_pos;
        for (uint t_start = kv_start; t_start <= kv_end; t_start += FA_TILE_KV) {
            uint t_end = min(t_start + FA_TILE_KV - 1, kv_end);
            float tile_max = NF_NEG_INF;
            for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
                float dot = 0.0f;
                for (uint d = 0; d < dim; ++d)
                    dot += float(Q[head * pc.seq_len * dim + q_pos * dim + d])
                         * float(cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
                dot *= scale;
                if (dot > tile_max) tile_max = dot;
            }
            float m_new = max(m, tile_max);
            float correction = exp(m - m_new);
            for (uint d = 0; d < dim; ++d) acc[d] *= correction;
            l *= correction;
            for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
                float dot = 0.0f;
                for (uint d = 0; d < dim; ++d)
                    dot += float(Q[head * pc.seq_len * dim + q_pos * dim + d])
                         * float(cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
                float w = exp(dot * scale - m_new);
                l += w;
                for (uint d = 0; d < dim; ++d)
                    acc[d] += w * float(cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
            }
            m = m_new;
        }
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint d = 0; d < dim; ++d)
            out[head * pc.seq_len * dim + q_pos * dim + d] = half(acc[d] * inv_l);
        return;
    }

    /* ---- Decode mode (step > 0) ---- */
    if (q_pos >= 1) return;
    if (step >= pc.max_seq_len) return;
    if (head < n_kv) {
        for (uint d = 0; d < dim; ++d) {
            cache_k[head * pc.max_seq_len * dim + step * dim + d] = K_new[head * dim + d];
            cache_v[head * pc.max_seq_len * dim + step * dim + d] = V_new[head * dim + d];
        }
    }
    float scale = rsqrt(float(dim));
    float m_val = NF_NEG_INF;
    float l_val = 0.0f;
    float acc[NF_MAX_HEAD_DIM];
    for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;
    uint kv_start = 0;
    if (pc.window_size > 0 && step >= pc.window_size)
        kv_start = step - pc.window_size + 1;
    for (uint t_start = kv_start; t_start <= step; t_start += FA_TILE_KV) {
        uint t_end = min(t_start + FA_TILE_KV - 1, step);
        float tile_max = NF_NEG_INF;
        for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += float(Q[head * dim + d])
                     * float(cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
            dot *= scale;
            if (dot > tile_max) tile_max = dot;
        }
        float m_new = max(m_val, tile_max);
        float correction = exp(m_val - m_new);
        for (uint d = 0; d < dim; ++d) acc[d] *= correction;
        l_val *= correction;
        for (uint k_pos = t_start; k_pos <= t_end; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += float(Q[head * dim + d])
                     * float(cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
            float w = exp(dot * scale - m_new);
            l_val += w;
            for (uint d = 0; d < dim; ++d)
                acc[d] += w * float(cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
        }
        m_val = m_new;
    }
    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    for (uint d = 0; d < dim; ++d)
        out[head * dim + d] = half(acc[d] * inv_l);
}

/* PLACEHOLDER_F16_ARGMAX_HALF is not needed — argmax always reads F32 logits */

/* ---- Phase 29: Fused Dequant-Q4_0 + Tiled MatMul ---- */

kernel void dequant_q4_0_linear_tiled(
    device const float*     A     [[buffer(0)]],
    device const block_q4_0* B_q  [[buffer(1)]],
    device float*           C     [[buffer(2)]],
    constant PushConstants& pc    [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint row = gid.y, col = gid.x;
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];
    float acc = 0.0f;
    for (uint t = 0; t < (pc.K + 15) / 16; ++t) {
        uint aCol = t * 16 + lid.x;
        tileA[lid.y][lid.x] = (row < pc.M && aCol < pc.K)
            ? A[row * pc.K + aCol] : 0.0f;
        uint bRow = t * 16 + lid.y;
        if (bRow < pc.K && col < pc.N) {
            uint elem = bRow * pc.N + col;
            uint blk = elem / 32;
            uint idx = elem % 32;
            float d = float(B_q[blk].d);
            uchar bv = B_q[blk].qs[idx % 16];
            int nib = (idx < 16) ? (bv & 0xF) : (bv >> 4);
            tileB[lid.y][lid.x] = d * float(nib - 8);
        } else {
            tileB[lid.y][lid.x] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k)
            acc += tileA[lid.y][k] * tileB[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < pc.M && col < pc.N)
        C[row * pc.N + col] = acc;
}

kernel void dequant_q4_0_linear_tiled_f16(
    device const half*      A     [[buffer(0)]],
    device const block_q4_0* B_q  [[buffer(1)]],
    device half*            C     [[buffer(2)]],
    constant PushConstants& pc    [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint row = gid.y, col = gid.x;
    threadgroup half tileA[16][16];
    threadgroup half tileB[16][16];
    float acc = 0.0f;
    for (uint t = 0; t < (pc.K + 15) / 16; ++t) {
        uint aCol = t * 16 + lid.x;
        tileA[lid.y][lid.x] = (row < pc.M && aCol < pc.K)
            ? A[row * pc.K + aCol] : half(0);
        uint bRow = t * 16 + lid.y;
        if (bRow < pc.K && col < pc.N) {
            uint elem = bRow * pc.N + col;
            uint blk = elem / 32;
            uint idx = elem % 32;
            float d = float(B_q[blk].d);
            uchar bv = B_q[blk].qs[idx % 16];
            int nib = (idx < 16) ? (bv & 0xF) : (bv >> 4);
            tileB[lid.y][lid.x] = half(d * float(nib - 8));
        } else {
            tileB[lid.y][lid.x] = half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k)
            acc += float(tileA[lid.y][k]) * float(tileB[k][lid.x]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < pc.M && col < pc.N)
        C[row * pc.N + col] = half(acc);
}

/* ---- Phase 33: Fused Dequant-Q4_0 + SIMD MatMul ---- */

kernel void dequant_q4_0_linear_simd(
    device const float*      A     [[buffer(0)]],
    device const block_q4_0* B_q   [[buffer(1)]],
    device float*            C     [[buffer(2)]],
    constant PushConstants&  pc    [[buffer(15)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lid   [[thread_index_in_simdgroup]])
{
    /* 32x32 output tile, 4x4 = 16 simdgroups, each 8x8 sub-tile */
    const uint sg_row = sgid / 4;
    const uint sg_col = sgid % 4;
    const uint row_base = tgid.y * 32 + sg_row * 8;
    const uint col_base = tgid.x * 32 + sg_col * 8;
    if (row_base >= pc.M && col_base >= pc.N) return;

    simdgroup_float8x8 acc;
    simdgroup_float8x8 tA;
    simdgroup_float8x8 tB;
    acc = simdgroup_float8x8(0);

    /* Per-simdgroup staging for dequantized B tile (8 rows x 8 cols) */
    threadgroup float B_stage[16][8][8 + 1];  /* [simdgroup][row][col+1] */

    for (uint k = 0; k < pc.K; k += 8) {
        /* Load A tile via simdgroup_load */
        if (row_base + 8 <= pc.M && k + 8 <= pc.K) {
            simdgroup_load(tA, A + row_base * pc.K + k, pc.K);
        } else {
            tA = simdgroup_float8x8(0);
            if (row_base < pc.M && k < pc.K)
                simdgroup_load(tA, A + row_base * pc.K + k, pc.K,
                               ulong2(min(pc.K - k, 8u), min(pc.M - row_base, 8u)));
        }

        /* Dequantize B[k:k+8, col_base:col_base+8] into per-simdgroup staging */
        for (uint i = lid; i < 64; i += 32) {
            uint bRow = k + (i / 8);
            uint bCol = col_base + (i % 8);
            float val = 0.0f;
            if (bRow < pc.K && bCol < pc.N) {
                uint elem = bRow * pc.N + bCol;
                uint blk = elem / 32;
                uint idx = elem % 32;
                float d = float(B_q[blk].d);
                uchar bv = B_q[blk].qs[idx % 16];
                int nib = (idx < 16) ? (bv & 0xF) : (bv >> 4);
                val = d * float(nib - 8);
            }
            B_stage[sgid][i / 8][i % 8] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(tB, &B_stage[sgid][0][0], 9);  /* stride = 8+1 */
        simdgroup_multiply_accumulate(acc, tA, tB, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_base + 8 <= pc.M && col_base + 8 <= pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N);
    } else if (row_base < pc.M && col_base < pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N,
                        ulong2(min(pc.N - col_base, 8u), min(pc.M - row_base, 8u)));
    }
}

kernel void dequant_q4_0_linear_simd_f16(
    device const half*       A     [[buffer(0)]],
    device const block_q4_0* B_q   [[buffer(1)]],
    device half*             C     [[buffer(2)]],
    constant PushConstants&  pc    [[buffer(15)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lid   [[thread_index_in_simdgroup]])
{
    const uint sg_row = sgid / 4;
    const uint sg_col = sgid % 4;
    const uint row_base = tgid.y * 32 + sg_row * 8;
    const uint col_base = tgid.x * 32 + sg_col * 8;
    if (row_base >= pc.M && col_base >= pc.N) return;

    simdgroup_half8x8 acc;
    simdgroup_half8x8 tA;
    simdgroup_half8x8 tB;
    acc = simdgroup_half8x8(0);

    /* Per-simdgroup staging for dequantized B tile (8 rows x 8 cols) */
    threadgroup half B_stage[16][8][8 + 1];  /* [simdgroup][row][col+1] */

    for (uint k = 0; k < pc.K; k += 8) {
        if (row_base + 8 <= pc.M && k + 8 <= pc.K) {
            simdgroup_load(tA, A + row_base * pc.K + k, pc.K);
        } else {
            tA = simdgroup_half8x8(0);
            if (row_base < pc.M && k < pc.K)
                simdgroup_load(tA, A + row_base * pc.K + k, pc.K,
                               ulong2(min(pc.K - k, 8u), min(pc.M - row_base, 8u)));
        }

        for (uint i = lid; i < 64; i += 32) {
            uint bRow = k + (i / 8);
            uint bCol = col_base + (i % 8);
            half val = half(0);
            if (bRow < pc.K && bCol < pc.N) {
                uint elem = bRow * pc.N + bCol;
                uint blk = elem / 32;
                uint idx = elem % 32;
                float d = float(B_q[blk].d);
                uchar bv = B_q[blk].qs[idx % 16];
                int nib = (idx < 16) ? (bv & 0xF) : (bv >> 4);
                val = half(d * float(nib - 8));
            }
            B_stage[sgid][i / 8][i % 8] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(tB, &B_stage[sgid][0][0], 9);  /* stride = 8+1 */
        simdgroup_multiply_accumulate(acc, tA, tB, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_base + 8 <= pc.M && col_base + 8 <= pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N);
    } else if (row_base < pc.M && col_base < pc.N) {
        simdgroup_store(acc, C + row_base * pc.N + col_base, pc.N,
                        ulong2(min(pc.N - col_base, 8u), min(pc.M - row_base, 8u)));
    }
}

kernel void argmax_rows(device const float* in  [[buffer(0)]],
                        device int* out         [[buffer(1)]],
                        constant PushConstants& pc [[buffer(15)]],
                        uint id [[thread_position_in_grid]]) {
    /* One thread per row. pc.N = row width, pc.seq_len = num rows */
    if (id >= pc.seq_len) return;
    uint dim = pc.N;
    uint base = id * dim;
    float best = in[base];
    int best_idx = 0;
    for (uint j = 1; j < dim; ++j) {
        if (in[base + j] > best) {
            best = in[base + j];
            best_idx = int(j);
        }
    }
    out[id] = best_idx;
}

/* ---- Phase 32: Paged Attention Kernel ---- */

kernel void flash_attention_paged(
        device const float* Q           [[buffer(0)]],
        device const float* K_pool      [[buffer(1)]],
        device const float* V_pool      [[buffer(2)]],
        device const uint*  block_table [[buffer(3)]],
        device float*       out         [[buffer(4)]],
        constant PushConstants& pc      [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint head    = id.y;
    uint q_pos   = id.x;
    uint dim     = pc.head_dim;
    uint n_kv    = (pc.M > 0 && pc.M < pc.n_heads) ? pc.M : pc.n_heads;
    uint kv_head = head * n_kv / pc.n_heads;
    uint bsz     = pc.window_size;  /* block_size passed via window_size field */

    if (head >= pc.n_heads) return;
    if (bsz == 0) bsz = 16;

    /* Decode mode: q_pos must be 0 for single-token */
    uint max_kv_pos = pc.step_idx;
    if (pc.step_idx == 0) {
        /* Prefill mode */
        if (q_pos >= pc.seq_len) return;
        max_kv_pos = q_pos;
    } else {
        if (q_pos >= 1) return;
    }

    float scale = rsqrt(float(dim));
    float m_prev = NF_NEG_INF;
    float l_prev = 0.0f;
    float acc[NF_MAX_HEAD_DIM];
    for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

    /* seq_offset stored in pc.K field for batched queries */
    uint q_base = (pc.K + q_pos) * pc.n_heads * dim + head * dim;
    /* For non-batched: K=0, so q_base = q_pos * n_heads * dim + head * dim */

    uint num_logical_blocks = (max_kv_pos + bsz) / bsz;
    if (num_logical_blocks > pc.N) num_logical_blocks = pc.N;
    /* pc.N = num_blocks allocated for this sequence */

    /* Block-by-block online softmax */
    for (uint lb = 0; lb < num_logical_blocks; ++lb) {
        uint phys = block_table[lb];
        /* K_pool layout: [phys_block][n_kv_heads][block_size][head_dim] */
        uint block_base = (phys * n_kv + kv_head) * bsz * dim;

        uint t_start = lb * bsz;
        uint t_end   = min(t_start + bsz - 1, max_kv_pos);
        uint t_count = t_end - t_start + 1;

        /* Tile: compute scores, find tile max */
        float tile_max = NF_NEG_INF;
        for (uint t = 0; t < t_count; ++t) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[q_base + d] * K_pool[block_base + t * dim + d];
            dot *= scale;
            if (dot > tile_max) tile_max = dot;
        }

        /* Online softmax merge */
        float m_new = max(m_prev, tile_max);
        float correction = exp(m_prev - m_new);
        l_prev *= correction;
        for (uint d = 0; d < dim; ++d) acc[d] *= correction;

        for (uint t = 0; t < t_count; ++t) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[q_base + d] * K_pool[block_base + t * dim + d];
            float w = exp(dot * scale - m_new);
            l_prev += w;
            for (uint d = 0; d < dim; ++d)
                acc[d] += w * V_pool[block_base + t * dim + d];
        }
        m_prev = m_new;
    }

    /* Normalize */
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    uint out_base = (pc.K + q_pos) * pc.n_heads * dim + head * dim;
    for (uint d = 0; d < dim; ++d)
        out[out_base + d] = acc[d] * inv_l;
}

/* ---- Phase 34-A: GQA Flash Attention (native n_kv_heads != n_heads) ---- */
/* Eliminates repeat-KV by computing Q→KV head mapping inline.
 * pc.M = n_kv_heads, pc.step_idx = current position, pc.max_seq_len = cache capacity.
 * Decode mode: single query token against KV cache.
 * Prefill mode (step_idx==0): full causal attention. */

kernel void flash_attention_gqa(
        device const float* Q       [[buffer(0)]],
        device const float* K_new   [[buffer(1)]],
        device const float* V_new   [[buffer(2)]],
        device float*       cache_k [[buffer(3)]],
        device float*       cache_v [[buffer(4)]],
        device float*       out     [[buffer(5)]],
        constant PushConstants& pc  [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint head  = id.y;
    uint q_pos = id.x;
    uint dim   = pc.head_dim;
    uint step  = pc.step_idx;
    uint n_kv  = (pc.M > 0 && pc.M < pc.n_heads) ? pc.M : pc.n_heads;
    uint kv_head = head * n_kv / pc.n_heads;

    if (head >= pc.n_heads) return;

    /* Prefill mode */
    if (step == 0) {
        if (q_pos >= pc.seq_len) return;
        if (head < n_kv) {
            for (uint d = 0; d < dim; ++d) {
                uint src = head * pc.seq_len * dim + q_pos * dim + d;
                uint dst = head * pc.max_seq_len * dim + q_pos * dim + d;
                cache_k[dst] = K_new[src];
                cache_v[dst] = V_new[src];
            }
        }
        float scale = rsqrt(float(dim));
        float m_prev = NF_NEG_INF;
        float l_prev = 0.0f;
        float acc[NF_MAX_HEAD_DIM];
        for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

        /* Tiled online softmax over causal positions */
        for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                     * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            dot *= scale;
            float m_new = max(m_prev, dot);
            float correction = exp(m_prev - m_new);
            l_prev = l_prev * correction + exp(dot - m_new);
            for (uint d = 0; d < dim; ++d) {
                acc[d] = acc[d] * correction
                       + exp(dot - m_new) * cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
            }
            m_prev = m_new;
        }
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint d = 0; d < dim; ++d)
            out[head * pc.seq_len * dim + q_pos * dim + d] = acc[d] * inv_l;
        return;
    }

    /* Decode mode */
    if (q_pos >= 1) return;
    if (step >= pc.max_seq_len) return;

    if (head < n_kv) {
        for (uint d = 0; d < dim; ++d) {
            cache_k[head * pc.max_seq_len * dim + step * dim + d] = K_new[head * dim + d];
            cache_v[head * pc.max_seq_len * dim + step * dim + d] = V_new[head * dim + d];
        }
    }

    float scale = rsqrt(float(dim));
    float m_prev = NF_NEG_INF;
    float l_prev = 0.0f;
    float acc[NF_MAX_HEAD_DIM];
    for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

    for (uint k_pos = 0; k_pos <= step; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * dim + d]
                 * cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
        dot *= scale;
        float m_new = max(m_prev, dot);
        float correction = exp(m_prev - m_new);
        l_prev = l_prev * correction + exp(dot - m_new);
        for (uint d = 0; d < dim; ++d) {
            acc[d] = acc[d] * correction
                   + exp(dot - m_new) * cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d];
        }
        m_prev = m_new;
    }
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint d = 0; d < dim; ++d)
        out[head * dim + d] = acc[d] * inv_l;
}

/* FP16 variant */
kernel void flash_attention_gqa_f16(
        device const half*  Q       [[buffer(0)]],
        device const half*  K_new   [[buffer(1)]],
        device const half*  V_new   [[buffer(2)]],
        device half*        cache_k [[buffer(3)]],
        device half*        cache_v [[buffer(4)]],
        device half*        out     [[buffer(5)]],
        constant PushConstants& pc  [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint head  = id.y;
    uint q_pos = id.x;
    uint dim   = pc.head_dim;
    uint step  = pc.step_idx;
    uint n_kv  = (pc.M > 0 && pc.M < pc.n_heads) ? pc.M : pc.n_heads;
    uint kv_head = head * n_kv / pc.n_heads;

    if (head >= pc.n_heads) return;

    if (step == 0) {
        if (q_pos >= pc.seq_len) return;
        if (head < n_kv) {
            for (uint d = 0; d < dim; ++d) {
                uint src = head * pc.seq_len * dim + q_pos * dim + d;
                uint dst = head * pc.max_seq_len * dim + q_pos * dim + d;
                cache_k[dst] = K_new[src];
                cache_v[dst] = V_new[src];
            }
        }
        float scale = rsqrt(float(dim));
        float m_prev = NF_NEG_INF;
        float l_prev = 0.0f;
        float acc[NF_MAX_HEAD_DIM];
        for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

        for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += float(Q[head * pc.seq_len * dim + q_pos * dim + d])
                     * float(cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
            dot *= scale;
            float m_new = max(m_prev, dot);
            float correction = exp(m_prev - m_new);
            l_prev = l_prev * correction + exp(dot - m_new);
            for (uint d = 0; d < dim; ++d) {
                acc[d] = acc[d] * correction
                       + exp(dot - m_new) * float(cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
            }
            m_prev = m_new;
        }
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint d = 0; d < dim; ++d)
            out[head * pc.seq_len * dim + q_pos * dim + d] = half(acc[d] * inv_l);
        return;
    }

    if (q_pos >= 1) return;
    if (step >= pc.max_seq_len) return;

    if (head < n_kv) {
        for (uint d = 0; d < dim; ++d) {
            cache_k[head * pc.max_seq_len * dim + step * dim + d] = K_new[head * dim + d];
            cache_v[head * pc.max_seq_len * dim + step * dim + d] = V_new[head * dim + d];
        }
    }

    float scale = rsqrt(float(dim));
    float m_prev = NF_NEG_INF;
    float l_prev = 0.0f;
    float acc[NF_MAX_HEAD_DIM];
    for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;

    for (uint k_pos = 0; k_pos <= step; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += float(Q[head * dim + d])
                 * float(cache_k[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
        dot *= scale;
        float m_new = max(m_prev, dot);
        float correction = exp(m_prev - m_new);
        l_prev = l_prev * correction + exp(dot - m_new);
        for (uint d = 0; d < dim; ++d) {
            acc[d] = acc[d] * correction
                   + exp(dot - m_new) * float(cache_v[kv_head * pc.max_seq_len * dim + k_pos * dim + d]);
        }
        m_prev = m_new;
    }
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint d = 0; d < dim; ++d)
        out[head * dim + d] = half(acc[d] * inv_l);
}

/* ---- Phase 34-B: MoE Top-K Gating Kernel ---- */
/* Softmax over n_experts, then selects top-K experts per token.
 * Input: gate_logits [batch, n_experts]
 * Output: expert_ids [batch, top_k] (int), expert_weights [batch, top_k] (float)
 * pc.M = n_experts, pc.N = top_k, pc.seq_len = batch_size */

kernel void moe_top_k_gating(
        device const float* gate_logits  [[buffer(0)]],
        device int*         expert_ids   [[buffer(1)]],
        device float*       expert_weights [[buffer(2)]],
        constant PushConstants& pc       [[buffer(15)]],
        uint id [[thread_position_in_grid]]) {
    uint batch_idx = id;
    if (batch_idx >= pc.seq_len) return;

    uint n_experts = pc.M;
    uint top_k = pc.N;
    if (top_k > 8) top_k = 8;

    device const float* logits = gate_logits + batch_idx * n_experts;
    device int*   out_ids = expert_ids + batch_idx * top_k;
    device float* out_w   = expert_weights + batch_idx * top_k;

    /* Softmax */
    float max_val = logits[0];
    for (uint i = 1; i < n_experts; ++i)
        if (logits[i] > max_val) max_val = logits[i];

    float probs[64];  /* max 64 experts */
    float sum = 0.0f;
    for (uint i = 0; i < n_experts && i < 64; ++i) {
        probs[i] = exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (uint i = 0; i < n_experts && i < 64; ++i) probs[i] /= sum;

    /* Top-K selection (simple insertion sort for small K) */
    int sel_ids[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    float sel_probs[8] = {0,0,0,0,0,0,0,0};

    for (uint i = 0; i < n_experts && i < 64; ++i) {
        /* Find insertion point */
        for (uint k = 0; k < top_k; ++k) {
            if (probs[i] > sel_probs[k]) {
                /* Shift down */
                for (uint j = top_k - 1; j > k; --j) {
                    sel_ids[j] = sel_ids[j-1];
                    sel_probs[j] = sel_probs[j-1];
                }
                sel_ids[k] = int(i);
                sel_probs[k] = probs[i];
                break;
            }
        }
    }

    /* Renormalize selected weights */
    float sel_sum = 0.0f;
    for (uint k = 0; k < top_k; ++k) sel_sum += sel_probs[k];
    for (uint k = 0; k < top_k; ++k) {
        out_ids[k] = sel_ids[k];
        out_w[k] = (sel_sum > 0.0f) ? (sel_probs[k] / sel_sum) : 0.0f;
    }
}

/* ---- Phase 34-B: MoE Scatter-Gather Kernel ---- */
/* Weighted sum of selected expert outputs.
 * expert_outputs: [n_experts, dim], expert_ids: [batch, top_k],
 * expert_weights: [batch, top_k], output: [batch, dim]
 * pc.M = n_experts, pc.N = top_k, pc.K = dim, pc.seq_len = batch_size */

kernel void moe_scatter_gather(
        device const float* expert_outputs [[buffer(0)]],
        device const int*   expert_ids     [[buffer(1)]],
        device const float* expert_weights [[buffer(2)]],
        device float*       output         [[buffer(3)]],
        constant PushConstants& pc         [[buffer(15)]],
        uint2 id [[thread_position_in_grid]]) {
    uint batch_idx = id.y;
    uint d = id.x;
    if (batch_idx >= pc.seq_len || d >= pc.K) return;

    uint top_k = pc.N;
    float val = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        int eid = expert_ids[batch_idx * top_k + k];
        if (eid < 0) continue;
        float w = expert_weights[batch_idx * top_k + k];
        val += w * expert_outputs[uint(eid) * pc.K + d];
    }
    output[batch_idx * pc.K + d] = val;
}
)";

/* ================================================================== */
/*  MetalBuffer — wraps real id<MTLBuffer>                              */
/* ================================================================== */

struct MetalBuffer {
    std::atomic<uint32_t>   refcount{1};
    id<MTLBuffer>           mtl_buffer = nil;   /**< Real GPU buffer (StorageModeShared) */
    nf_tensor_desc          desc{};
    nf_mem_domain           domain = NF_MEM_DOMAIN_UNIFIED;
    bool                    mapped = false;

    /* Phase 7 fence contract preserved */
    std::mutex              fence_mu;
    std::condition_variable fence_cv;
    std::atomic<bool>       gpu_done{true};
};

/* ================================================================== */
/*  nf_buffer_ops — Real Metal Unified Memory                           */
/* ================================================================== */

static uint32_t metal_buf_retain(nf_buffer self) {
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    return mb->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

static uint32_t metal_buf_release(nf_buffer self) {
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    uint32_t prev = mb->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        mb->mtl_buffer = nil;   /* ARC releases the MTLBuffer */
        delete mb;
    }
    return prev - 1;
}

static nf_status metal_buf_map(nf_buffer self, void** out_ptr) {
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    if (mb->mapped) return NF_ERROR_INVALID_ARG;
    mb->mapped = true;
    *out_ptr = [mb->mtl_buffer contents];   /* Zero-cost unified memory */
    return NF_OK;
}
static nf_status metal_buf_unmap(nf_buffer self) {
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    mb->mapped = false;
    return NF_OK;
}

static nf_status metal_buf_cache_sync(nf_buffer self, nf_cache_op,
                                      uint64_t, uint64_t) {
    /*
     * Apple Silicon unified memory is HARDWARE COHERENT for CPU caches.
     * However, GPU execution ordering is NOT automatic — a CPU thread
     * can read a buffer before the GPU command finishes writing it.
     *
     * Phase 7 contract: if GPU dispatch is in-flight on this buffer,
     * block until the fence signals completion via addCompletedHandler.
     */
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    if (!mb->gpu_done.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(mb->fence_mu);
        mb->fence_cv.wait(lk, [&] {
            return mb->gpu_done.load(std::memory_order_acquire);
        });
    }
    return NF_OK;
}

static nf_status metal_buf_get_info(nf_buffer self, nf_buffer_info* info) {
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    info->desc         = mb->desc;
    info->domain       = NF_MEM_DOMAIN_UNIFIED;
    info->offset_bytes = 0;
    info->share_token  = [mb->mtl_buffer gpuAddress];
    info->refcount     = mb->refcount.load(std::memory_order_relaxed);
    info->_reserved    = 0;
    return NF_OK;
}

static nf_status metal_buf_export(nf_buffer self, uint64_t* token,
                                  nf_mem_domain* domain) {
    auto* mb = reinterpret_cast<MetalBuffer*>(self);
    *token  = [mb->mtl_buffer gpuAddress];
    *domain = NF_MEM_DOMAIN_UNIFIED;
    return NF_OK;
}

static nf_buffer_ops make_metal_buf_ops() {
    nf_buffer_ops ops{};
    ops.retain        = metal_buf_retain;
    ops.release       = metal_buf_release;
    ops.map           = metal_buf_map;
    ops.unmap         = metal_buf_unmap;
    ops.cache_sync    = metal_buf_cache_sync;
    ops.get_info      = metal_buf_get_info;
    ops.export_handle = metal_buf_export;
    ops.import_handle = nullptr;
    return ops;
}

/* ================================================================== */
/*  Provider State — Real Metal Objects                                 */
/* ================================================================== */

struct nf_provider_metal {
    bool                        initialized = false;
    id<MTLDevice>               device  = nil;
    id<MTLCommandQueue>         queue   = nil;
    id<MTLLibrary>              library = nil;
    bool                        has_simd_matmul = false;
    id<MTLComputePipelineState> pso[PSO_COUNT] = {};
};

static nf_provider_metal s_instance;

/* Phase 29: Dispatch map types + forward declarations */
using DispatchHandler = nf_status(*)(nf_provider_metal*, const char*,
                                      const nf_buffer*, uint32_t,
                                      nf_buffer*, uint32_t);
struct DequantInfo { MetalPSO pso_idx; bool is_f16; };
static std::unordered_map<std::string, DequantInfo> s_dequant_map;
static std::unordered_map<std::string, DispatchHandler> s_dispatch_map;
static void init_dispatch_maps();

/* ================================================================== */
/*  Phase 23: Per-Kernel GPU Timestamp Profiling                        */
/* ================================================================== */

struct MetalDispatchTiming {
    char     op_name[64];
    double   gpu_start_s;
    double   gpu_end_s;
    uint8_t  dtype;          /* nf_dtype cast to uint8_t */
    uint32_t element_count;  /* elements processed */
};

static constexpr uint32_t kMaxTimings = 1024;
static MetalDispatchTiming s_timings[kMaxTimings];
static std::atomic<uint32_t> s_timing_head{0};
static std::atomic<bool> s_timing_enabled{false};

static void record_timing(id<MTLCommandBuffer> cb, const char* op,
                           uint8_t dtype = 0, uint32_t elem_count = 0) {
    if (!s_timing_enabled.load(std::memory_order_relaxed)) return;
    uint32_t idx = s_timing_head.fetch_add(1, std::memory_order_relaxed) % kMaxTimings;
    s_timings[idx].gpu_start_s    = cb.GPUStartTime;
    s_timings[idx].gpu_end_s      = cb.GPUEndTime;
    s_timings[idx].dtype          = dtype;
    s_timings[idx].element_count  = elem_count;
    std::strncpy(s_timings[idx].op_name, op, 63);
    s_timings[idx].op_name[63] = '\0';
}

extern "C" NF_API void nf_metal_enable_timing(bool enable) {
    s_timing_enabled.store(enable, std::memory_order_relaxed);
    if (enable) s_timing_head.store(0, std::memory_order_relaxed);
}

extern "C" NF_API uint32_t nf_metal_get_timing_count() {
    uint32_t head = s_timing_head.load(std::memory_order_relaxed);
    return (head > kMaxTimings) ? kMaxTimings : head;
}

extern "C" NF_API void nf_metal_get_timings(
    char (*op_names)[64], double* gpu_ms, uint32_t max_count)
{
    uint32_t head = s_timing_head.load(std::memory_order_relaxed);
    uint32_t count = (head > kMaxTimings) ? kMaxTimings : head;
    if (count > max_count) count = max_count;
    uint32_t start = (head > kMaxTimings) ? (head % kMaxTimings) : 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = (start + i) % kMaxTimings;
        std::strncpy(op_names[i], s_timings[idx].op_name, 64);
        gpu_ms[i] = (s_timings[idx].gpu_end_s - s_timings[idx].gpu_start_s) * 1000.0;
    }
}

extern "C" NF_API void nf_metal_get_timings_ext(
    char (*op_names)[64], double* gpu_ms,
    uint8_t* dtypes, uint32_t* elem_counts,
    uint32_t max_count)
{
    uint32_t head = s_timing_head.load(std::memory_order_relaxed);
    uint32_t count = (head > kMaxTimings) ? kMaxTimings : head;
    if (count > max_count) count = max_count;
    uint32_t start = (head > kMaxTimings) ? (head % kMaxTimings) : 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = (start + i) % kMaxTimings;
        std::strncpy(op_names[i], s_timings[idx].op_name, 64);
        gpu_ms[i] = (s_timings[idx].gpu_end_s - s_timings[idx].gpu_start_s) * 1000.0;
        if (dtypes)      dtypes[i]      = s_timings[idx].dtype;
        if (elem_counts) elem_counts[i] = s_timings[idx].element_count;
    }
}

/* ================================================================== */
/*  Provider VTable                                                    */
/* ================================================================== */

static const char* metal_get_name(nf_provider) { return "apple_metal"; }
static uint32_t    metal_get_abi_version(nf_provider) { return NF_ABI_VERSION; }
static nf_status   metal_synchronize(nf_provider self); /* forward decl */

static nf_status metal_init(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_metal*>(self);

    @autoreleasepool {
        p->device = MTLCreateSystemDefaultDevice();
        if (!p->device) return NF_ERROR_DEVICE_LOST;

        p->queue = [p->device newCommandQueue];
        if (!p->queue) return NF_ERROR_DEVICE_LOST;

        /* Compile MSL shaders */
        NSError* error = nil;
        p->library = [p->device newLibraryWithSource:kShaderSource
                                             options:nil
                                               error:&error];
        if (!p->library) {
            NSLog(@"[NF Metal] Shader compile error: %@", error);
            return NF_ERROR_INTERNAL;
        }

        /* Create pipeline states via registry table */
        auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [p->library newFunctionWithName:name];
            if (!fn) return nil;
            return [p->device newComputePipelineStateWithFunction:fn error:&error];
        };

        bool has_simd = [p->device supportsFamily:MTLGPUFamilyApple7];
        p->has_simd_matmul = has_simd;

        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            auto& reg = kPSOTable[i];
            if (reg.requires_simd && !has_simd) continue;
            p->pso[reg.index] = make_pso([NSString stringWithUTF8String:reg.msl_name]);
        }

        /* Validate: all non-SIMD-conditional PSOs must be non-nil */
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            auto& reg = kPSOTable[i];
            if (reg.requires_simd && !has_simd) continue;
            if (!p->pso[reg.index]) {
                NSLog(@"[NF Metal] PSO creation failed: %s", reg.msl_name);
                return NF_ERROR_INTERNAL;
            }
        }

        if (has_simd) p->has_simd_matmul = (p->pso[PSO_LINEAR_SIMD] != nil);
    }

    p->initialized = true;
    init_dispatch_maps();
    return NF_OK;
}

static void metal_shutdown(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_metal*>(self);
    metal_synchronize(self);
    for (uint16_t i = 0; i < PSO_COUNT; ++i) p->pso[i] = nil;
    p->has_simd_matmul = false;
    p->library       = nil;
    p->queue         = nil;
    p->device        = nil;
    p->initialized   = false;
    s_dispatch_map.clear();
    s_dequant_map.clear();
}

static nf_status metal_buffer_alloc(nf_provider self, const nf_tensor_desc* desc,
                                    nf_buffer* out) {
    auto* p = reinterpret_cast<nf_provider_metal*>(self);
    auto* mb = new MetalBuffer;
    mb->desc = *desc;
    mb->mtl_buffer = [p->device newBufferWithLength:desc->size_bytes
                                            options:MTLResourceStorageModeShared];
    if (!mb->mtl_buffer) {
        delete mb;
        return NF_ERROR_OUT_OF_MEMORY;
    }
    /* Zero-fill for deterministic behavior */
    std::memset([mb->mtl_buffer contents], 0, desc->size_bytes);
    *out = reinterpret_cast<nf_buffer>(mb);
    return NF_OK;
}

static void metal_buffer_free(nf_provider, nf_buffer buf) {
    if (!buf) return;
    metal_buf_release(buf);
}

static nf_status metal_buffer_map(nf_provider, nf_buffer buf, void** out) {
    return metal_buf_map(buf, out);
}

static nf_status metal_buffer_unmap(nf_provider, nf_buffer buf) {
    return metal_buf_unmap(buf);
}

/* ================================================================== */
/*  Dispatch — Real GPU Compute via MTLComputeCommandEncoder            */
/* ================================================================== */

/* Phase 29: File-scope push constants recovery */
static const uint8_t* get_push_constants(const nf_buffer* inputs) {
    nf_task_desc* td = reinterpret_cast<nf_task_desc*>(
        reinterpret_cast<uint8_t*>(const_cast<nf_buffer*>(inputs))
        - offsetof(nf_task_desc, inputs));
    if (td->push_constants_size == 0) return nullptr;
    return td->push_constants;
}

/**
 * Helper: encode a 1-input, 1-output unary compute kernel.
 * Marks output gpu_done=false, commits async, signals fence on completion.
 */
static nf_status dispatch_unary(nf_provider_metal* prov,
                                id<MTLComputePipelineState> pso,
                                MetalBuffer* in_mb, MetalBuffer* out_mb,
                                const char* op_name = nullptr) {
    out_mb->gpu_done.store(false, std::memory_order_release);

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];

        NSUInteger elem_sz = (out_mb->desc.dtype == NF_DTYPE_F16) ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            if (op_name) {
                NSUInteger esz = (out_mb->desc.dtype == NF_DTYPE_F16) ? 2 : 4;
                record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype,
                              (uint32_t)(out_mb->desc.size_bytes / esz));
            }
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];

        [cmdBuf commit];
    }
    return NF_OK;
}

/* ================================================================== */
/* ================================================================== */
/*  Phase 29: Extracted Dispatch Handlers                               */
/* ================================================================== */

static nf_status dispatch_metal_vector_add(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        auto pso = prov->pso[is_f16 ? PSO_VECTOR_ADD_F16 : PSO_VECTOR_ADD];
        [enc setComputePipelineState:pso];
        [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
        NSUInteger elem_sz = is_f16 ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_attention_prefill(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 2) return NF_ERROR_INVALID_ARG;
    auto* in_mb = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* k_mb  = reinterpret_cast<MetalBuffer*>(outputs[0]);
    auto* v_mb  = reinterpret_cast<MetalBuffer*>(outputs[1]);
    k_mb->gpu_done.store(false, std::memory_order_release);
    v_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->pso[PSO_ATTN_K]];
            [enc setBuffer:in_mb->mtl_buffer offset:0 atIndex:0];
            [enc setBuffer:k_mb->mtl_buffer  offset:0 atIndex:1];
            NSUInteger count = k_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->pso[PSO_ATTN_K].maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->pso[PSO_ATTN_V]];
            [enc setBuffer:in_mb->mtl_buffer offset:0 atIndex:0];
            [enc setBuffer:v_mb->mtl_buffer  offset:0 atIndex:1];
            NSUInteger count = v_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->pso[PSO_ATTN_V].maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }
        NSUInteger kcount = k_mb->desc.size_bytes / sizeof(float);
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)k_mb->desc.dtype, (uint32_t)kcount);
            k_mb->gpu_done.store(true, std::memory_order_release);
            k_mb->fence_cv.notify_all();
            v_mb->gpu_done.store(true, std::memory_order_release);
            v_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_mock_relu(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb = reinterpret_cast<MetalBuffer*>(inputs[0]);
    if (n_out >= 1 && outputs[0]) {
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        return dispatch_unary(prov, prov->pso[PSO_RELU], in_mb, out_mb, op_name);
    }
    return dispatch_unary(prov, prov->pso[PSO_RELU], in_mb, in_mb, op_name);
}

static nf_status dispatch_rms_norm(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* wt_mb  = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        auto pso = prov->pso[is_f16 ? PSO_RMS_NORM_F16 : PSO_RMS_NORM];
        [enc setComputePipelineState:pso];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
        [enc setBuffer:wt_mb->mtl_buffer  offset:0 atIndex:2];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        NSUInteger elem_sz = is_f16 ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_rope(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_ROPE]];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
        NSUInteger tpg = prov->pso[PSO_ROPE].maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_rope_batch(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        auto pso = prov->pso[is_f16 ? PSO_ROPE_BATCH_F16 : PSO_ROPE_BATCH];
        [enc setComputePipelineState:pso];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        NSUInteger elem_sz = is_f16 ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_linear(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool a_f16 = (a_mb->desc.dtype == NF_DTYPE_F16);
    bool out_f32 = (out_mb->desc.dtype == NF_DTYPE_F32);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t M = 1, N = 1;
        if (pc) {
            std::memcpy(&M, pc + 20, sizeof(uint32_t));
            std::memcpy(&N, pc + 24, sizeof(uint32_t));
        }
        if (a_f16 && out_f32 && prov->pso[PSO_LINEAR_F16_TO_F32] && M >= 8 && N >= 8) {
            [enc setComputePipelineState:prov->pso[PSO_LINEAR_F16_TO_F32]];
            NSUInteger tg_x = (N + 31) / 32;
            NSUInteger tg_y = (M + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        } else if (a_f16 && !out_f32 && prov->pso[PSO_LINEAR_SIMD_F16] && M >= 8 && N >= 8) {
            [enc setComputePipelineState:prov->pso[PSO_LINEAR_SIMD_F16]];
            NSUInteger tg_x = (N + 31) / 32;
            NSUInteger tg_y = (M + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        } else if (a_f16) {
            [enc setComputePipelineState:prov->pso[PSO_LINEAR_TILED_F16]];
            NSUInteger gridW = ((N + 15) / 16) * 16;
            NSUInteger gridH = ((M + 15) / 16) * 16;
            [enc dispatchThreads:MTLSizeMake(gridW, gridH, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        } else if (prov->has_simd_matmul && M >= 8 && N >= 8) {
            [enc setComputePipelineState:prov->pso[PSO_LINEAR_SIMD]];
            NSUInteger tg_x = (N + 31) / 32;
            NSUInteger tg_y = (M + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        } else {
            [enc setComputePipelineState:prov->pso[PSO_LINEAR_TILED]];
            NSUInteger gridW = ((N + 15) / 16) * 16;
            NSUInteger gridH = ((M + 15) / 16) * 16;
            [enc dispatchThreads:MTLSizeMake(gridW, gridH, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [enc endEncoding];
        uint32_t elem_count = M * N;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, elem_count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_causal_attention(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 3 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* q_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* k_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* v_mb   = reinterpret_cast<MetalBuffer*>(inputs[2]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_CAUSAL_ATTN]];
        [enc setBuffer:q_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:k_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:v_mb->mtl_buffer   offset:0 atIndex:2];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:3];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t seq_len = 1, n_heads = 1;
        if (pc) {
            std::memcpy(&seq_len, pc, sizeof(uint32_t));
            std::memcpy(&n_heads, pc + 4, sizeof(uint32_t));
        }
        [enc dispatchThreads:MTLSizeMake(seq_len, n_heads, 1)
       threadsPerThreadgroup:MTLSizeMake(
           MIN((NSUInteger)seq_len, (NSUInteger)16),
           MIN((NSUInteger)n_heads, (NSUInteger)16), 1)];
        [enc endEncoding];
        uint32_t elem_count = seq_len * n_heads;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, elem_count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_dequant_generic(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto it = s_dequant_map.find(op_name);
    if (it == s_dequant_map.end()) return NF_ERROR_UNSUPPORTED_OP;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    bool is_f16 = it->second.is_f16;
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[it->second.pso_idx]];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
        NSUInteger elem_sz = is_f16 ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = prov->pso[it->second.pso_idx].maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_linear_tiled(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_LINEAR_TILED]];
        [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t M = 1, N = 1;
        if (pc) { std::memcpy(&M, pc + 20, sizeof(uint32_t)); std::memcpy(&N, pc + 24, sizeof(uint32_t)); }
        NSUInteger gridW = ((N + 15) / 16) * 16, gridH = ((M + 15) / 16) * 16;
        [enc dispatchThreads:MTLSizeMake(gridW, gridH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        uint32_t ec = M * N;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_softmax(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_SOFTMAX]];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t rows = 1;
        if (pc) std::memcpy(&rows, pc, sizeof(uint32_t));
        NSUInteger tpg = prov->pso[PSO_SOFTMAX].maxTotalThreadsPerThreadgroup;
        if (tpg > rows) tpg = rows;
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, rows);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_silu(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    return dispatch_unary(prov, prov->pso[is_f16 ? PSO_SILU_F16 : PSO_SILU], in_mb, out_mb, op_name);
}

static nf_status dispatch_gelu(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    return dispatch_unary(prov, prov->pso[is_f16 ? PSO_GELU_F16 : PSO_GELU], in_mb, out_mb, op_name);
}

static nf_status dispatch_elementwise_mul(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        auto pso = prov->pso[is_f16 ? PSO_ELEM_MUL_F16 : PSO_ELEM_MUL];
        [enc setComputePipelineState:pso];
        [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
        NSUInteger elem_sz = is_f16 ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_embedding_lookup(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* w_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* t_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        auto pso = prov->pso[is_f16 ? PSO_EMBED_LOOKUP_F16 : PSO_EMBED_LOOKUP];
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:t_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        NSUInteger elem_sz = is_f16 ? sizeof(uint16_t) : sizeof(float);
        NSUInteger count = out_mb->desc.size_bytes / elem_sz;
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, (uint32_t)count);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_argmax(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_ARGMAX]];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t rows = 1;
        if (pc) std::memcpy(&rows, pc, sizeof(uint32_t));
        NSUInteger tpg = prov->pso[PSO_ARGMAX].maxTotalThreadsPerThreadgroup;
        if (tpg > rows) tpg = rows;
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)NF_DTYPE_I32, rows);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_causal_attn_cached(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 5 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* q_mb       = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* k_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* v_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[2]);
    auto* cache_k_mb = reinterpret_cast<MetalBuffer*>(inputs[3]);
    auto* cache_v_mb = reinterpret_cast<MetalBuffer*>(inputs[4]);
    auto* out_mb     = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_CAUSAL_ATTN_CACHED]];
        [enc setBuffer:q_mb->mtl_buffer       offset:0 atIndex:0];
        [enc setBuffer:k_new_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:v_new_mb->mtl_buffer   offset:0 atIndex:2];
        [enc setBuffer:cache_k_mb->mtl_buffer offset:0 atIndex:3];
        [enc setBuffer:cache_v_mb->mtl_buffer offset:0 atIndex:4];
        [enc setBuffer:out_mb->mtl_buffer     offset:0 atIndex:5];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t seq_len = 1, n_heads = 1, step_idx = 0;
        if (pc) {
            std::memcpy(&seq_len,  pc,      sizeof(uint32_t));
            std::memcpy(&n_heads,  pc + 4,  sizeof(uint32_t));
            std::memcpy(&step_idx, pc + 32, sizeof(uint32_t));
        }
        uint32_t grid_x = (step_idx == 0) ? seq_len : 1;
        [enc dispatchThreads:MTLSizeMake(grid_x, n_heads, 1)
       threadsPerThreadgroup:MTLSizeMake(
           MIN((NSUInteger)grid_x, (NSUInteger)16),
           MIN((NSUInteger)n_heads, (NSUInteger)16), 1)];
        [enc endEncoding];
        uint32_t ec = grid_x * n_heads;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status dispatch_flash_attn_cached(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 5 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* q_mb       = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* k_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* v_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[2]);
    auto* cache_k_mb = reinterpret_cast<MetalBuffer*>(inputs[3]);
    auto* cache_v_mb = reinterpret_cast<MetalBuffer*>(inputs[4]);
    auto* out_mb     = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[is_f16 ? PSO_FLASH_ATTN_F16 : PSO_FLASH_ATTN]];
        [enc setBuffer:q_mb->mtl_buffer       offset:0 atIndex:0];
        [enc setBuffer:k_new_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:v_new_mb->mtl_buffer   offset:0 atIndex:2];
        [enc setBuffer:cache_k_mb->mtl_buffer offset:0 atIndex:3];
        [enc setBuffer:cache_v_mb->mtl_buffer offset:0 atIndex:4];
        [enc setBuffer:out_mb->mtl_buffer     offset:0 atIndex:5];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t seq_len = 1, n_heads = 1, step_idx = 0;
        if (pc) {
            std::memcpy(&seq_len,  pc,      sizeof(uint32_t));
            std::memcpy(&n_heads,  pc + 4,  sizeof(uint32_t));
            std::memcpy(&step_idx, pc + 32, sizeof(uint32_t));
        }
        uint32_t grid_x = (step_idx == 0) ? seq_len : 1;
        [enc dispatchThreads:MTLSizeMake(grid_x, n_heads, 1)
       threadsPerThreadgroup:MTLSizeMake(
           MIN((NSUInteger)grid_x, (NSUInteger)16),
           MIN((NSUInteger)n_heads, (NSUInteger)16), 1)];
        [enc endEncoding];
        uint32_t ec = grid_x * n_heads;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

/* Phase 32: Paged attention dispatch */
static nf_status dispatch_flash_attn_paged(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 4 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* q_mb       = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* k_pool_mb  = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* v_pool_mb  = reinterpret_cast<MetalBuffer*>(inputs[2]);
    auto* bt_mb      = reinterpret_cast<MetalBuffer*>(inputs[3]);
    auto* out_mb     = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_FLASH_ATTN_PAGED]];
        [enc setBuffer:q_mb->mtl_buffer       offset:0 atIndex:0];
        [enc setBuffer:k_pool_mb->mtl_buffer  offset:0 atIndex:1];
        [enc setBuffer:v_pool_mb->mtl_buffer  offset:0 atIndex:2];
        [enc setBuffer:bt_mb->mtl_buffer      offset:0 atIndex:3];
        [enc setBuffer:out_mb->mtl_buffer     offset:0 atIndex:4];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t seq_len = 1, n_heads = 1, step_idx = 0;
        if (pc) {
            std::memcpy(&seq_len,  pc,      sizeof(uint32_t));
            std::memcpy(&n_heads,  pc + 4,  sizeof(uint32_t));
            std::memcpy(&step_idx, pc + 32, sizeof(uint32_t));
        }
        uint32_t grid_x = (step_idx == 0) ? seq_len : 1;
        [enc dispatchThreads:MTLSizeMake(grid_x, n_heads, 1)
       threadsPerThreadgroup:MTLSizeMake(
           MIN((NSUInteger)grid_x, (NSUInteger)16),
           MIN((NSUInteger)n_heads, (NSUInteger)16), 1)];
        [enc endEncoding];
        uint32_t ec = grid_x * n_heads;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

/* Phase 29: Fused dequant_q4_0 + linear dispatch */
static nf_status dispatch_fused_dq4_linear(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* bq_mb  = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
        [enc setBuffer:bq_mb->mtl_buffer  offset:0 atIndex:1];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t M = 1, N = 1;
        if (pc) { std::memcpy(&M, pc + 20, sizeof(uint32_t)); std::memcpy(&N, pc + 24, sizeof(uint32_t)); }
        /* SIMD path: use simdgroup matmul when GPU supports it and tiles are large enough */
        if (prov->has_simd_matmul && M >= 8 && N >= 8) {
            MetalPSO simd_pso = is_f16 ? PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD_F16
                                        : PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD;
            [enc setComputePipelineState:prov->pso[simd_pso]];
            NSUInteger tg_x = (N + 31) / 32;
            NSUInteger tg_y = (M + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        } else {
            [enc setComputePipelineState:prov->pso[is_f16 ? PSO_FUSED_DQ4_LINEAR_F16 : PSO_FUSED_DQ4_LINEAR]];
            NSUInteger gridW = ((N + 15) / 16) * 16, gridH = ((M + 15) / 16) * 16;
            [enc dispatchThreads:MTLSizeMake(gridW, gridH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [enc endEncoding];
        uint32_t ec = M * N;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

/* Phase 34-A: GQA flash attention dispatch */
static nf_status dispatch_flash_attn_gqa(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 5 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* q_mb       = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* k_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* v_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[2]);
    auto* cache_k_mb = reinterpret_cast<MetalBuffer*>(inputs[3]);
    auto* cache_v_mb = reinterpret_cast<MetalBuffer*>(inputs[4]);
    auto* out_mb     = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    bool is_f16 = (out_mb->desc.dtype == NF_DTYPE_F16);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[is_f16 ? PSO_FLASH_ATTN_GQA_F16 : PSO_FLASH_ATTN_GQA]];
        [enc setBuffer:q_mb->mtl_buffer       offset:0 atIndex:0];
        [enc setBuffer:k_new_mb->mtl_buffer   offset:0 atIndex:1];
        [enc setBuffer:v_new_mb->mtl_buffer   offset:0 atIndex:2];
        [enc setBuffer:cache_k_mb->mtl_buffer offset:0 atIndex:3];
        [enc setBuffer:cache_v_mb->mtl_buffer offset:0 atIndex:4];
        [enc setBuffer:out_mb->mtl_buffer     offset:0 atIndex:5];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t seq_len = 1, n_heads = 1, step_idx = 0;
        if (pc) {
            std::memcpy(&seq_len,  pc,      sizeof(uint32_t));
            std::memcpy(&n_heads,  pc + 4,  sizeof(uint32_t));
            std::memcpy(&step_idx, pc + 32, sizeof(uint32_t));
        }
        uint32_t grid_x = (step_idx == 0) ? seq_len : 1;
        [enc dispatchThreads:MTLSizeMake(grid_x, n_heads, 1)
       threadsPerThreadgroup:MTLSizeMake(
           MIN((NSUInteger)grid_x, (NSUInteger)16),
           MIN((NSUInteger)n_heads, (NSUInteger)16), 1)];
        [enc endEncoding];
        uint32_t ec = grid_x * n_heads;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

/* Phase 34-B: MoE top-K gating dispatch */
static nf_status dispatch_moe_gate(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 1 || n_out < 2) return NF_ERROR_INVALID_ARG;
    auto* logits_mb = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* ids_mb    = reinterpret_cast<MetalBuffer*>(outputs[0]);
    auto* weights_mb = reinterpret_cast<MetalBuffer*>(outputs[1]);
    const uint8_t* pc = get_push_constants(inputs);
    ids_mb->gpu_done.store(false, std::memory_order_release);
    weights_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_MOE_GATE]];
        [enc setBuffer:logits_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:ids_mb->mtl_buffer     offset:0 atIndex:1];
        [enc setBuffer:weights_mb->mtl_buffer offset:0 atIndex:2];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t batch = 1;
        if (pc) std::memcpy(&batch, pc, sizeof(uint32_t));
        NSUInteger tpg = prov->pso[PSO_MOE_GATE].maxTotalThreadsPerThreadgroup;
        if (tpg > batch) tpg = batch;
        [enc dispatchThreads:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)ids_mb->desc.dtype, batch);
            ids_mb->gpu_done.store(true, std::memory_order_release); ids_mb->fence_cv.notify_all();
            weights_mb->gpu_done.store(true, std::memory_order_release); weights_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

/* Phase 34-B: MoE scatter-gather dispatch */
static nf_status dispatch_moe_scatter(nf_provider_metal* prov, const char* op_name,
        const nf_buffer* inputs, uint32_t n_in, nf_buffer* outputs, uint32_t n_out) {
    if (n_in < 3 || n_out < 1) return NF_ERROR_INVALID_ARG;
    auto* expert_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
    auto* ids_mb     = reinterpret_cast<MetalBuffer*>(inputs[1]);
    auto* weights_mb = reinterpret_cast<MetalBuffer*>(inputs[2]);
    auto* out_mb     = reinterpret_cast<MetalBuffer*>(outputs[0]);
    const uint8_t* pc = get_push_constants(inputs);
    out_mb->gpu_done.store(false, std::memory_order_release);
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:prov->pso[PSO_MOE_SCATTER]];
        [enc setBuffer:expert_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:ids_mb->mtl_buffer     offset:0 atIndex:1];
        [enc setBuffer:weights_mb->mtl_buffer offset:0 atIndex:2];
        [enc setBuffer:out_mb->mtl_buffer     offset:0 atIndex:3];
        if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
        uint32_t batch = 1, dim = 1;
        if (pc) {
            std::memcpy(&batch, pc, sizeof(uint32_t));
            std::memcpy(&dim, pc + 28, sizeof(uint32_t));  /* pc.K = dim */
        }
        [enc dispatchThreads:MTLSizeMake(dim, batch, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)dim, (NSUInteger)256), 1, 1)];
        [enc endEncoding];
        uint32_t ec = batch * dim;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            record_timing(cb, op_name, (uint8_t)out_mb->desc.dtype, ec);
            out_mb->gpu_done.store(true, std::memory_order_release); out_mb->fence_cv.notify_all();
        }];
        [cmdBuf commit];
    }
    return NF_OK;
}

/* ================================================================== */
/*  Phase 29: Dispatch Map Initialization                               */
/* ================================================================== */

static void init_dispatch_maps() {
    /* Dequant map: op_name → {PSO index, is_f16} */
    s_dequant_map = {
        {"dequant_q4_0",     {PSO_DEQUANT_Q4_0,     false}},
        {"dequant_q8_0",     {PSO_DEQUANT_Q8_0,     false}},
        {"dequant_q6_k",     {PSO_DEQUANT_Q6_K,     false}},
        {"dequant_q4_1",     {PSO_DEQUANT_Q4_1,     false}},
        {"dequant_q5_0",     {PSO_DEQUANT_Q5_0,     false}},
        {"dequant_q5_1",     {PSO_DEQUANT_Q5_1,     false}},
        {"dequant_q2_k",     {PSO_DEQUANT_Q2_K,     false}},
        {"dequant_q3_k",     {PSO_DEQUANT_Q3_K,     false}},
        {"dequant_q4_k",     {PSO_DEQUANT_Q4_K,     false}},
        {"dequant_q5_k",     {PSO_DEQUANT_Q5_K,     false}},
        {"dequant_q4_0_f16", {PSO_DEQUANT_Q4_0_F16, true}},
        {"dequant_q8_0_f16", {PSO_DEQUANT_Q8_0_F16, true}},
        {"dequant_q6_k_f16", {PSO_DEQUANT_Q6_K_F16, true}},
        {"dequant_q4_1_f16", {PSO_DEQUANT_Q4_1_F16, true}},
        {"dequant_q5_0_f16", {PSO_DEQUANT_Q5_0_F16, true}},
        {"dequant_q5_1_f16", {PSO_DEQUANT_Q5_1_F16, true}},
        {"dequant_q2_k_f16", {PSO_DEQUANT_Q2_K_F16, true}},
        {"dequant_q3_k_f16", {PSO_DEQUANT_Q3_K_F16, true}},
        {"dequant_q4_k_f16", {PSO_DEQUANT_Q4_K_F16, true}},
        {"dequant_q5_k_f16", {PSO_DEQUANT_Q5_K_F16, true}},
    };

    /* Main dispatch map: op_name → handler function */
    s_dispatch_map = {
        {"metal_vector_add",          dispatch_metal_vector_add},
        {"attention_prefill",         dispatch_attention_prefill},
        {"mock_relu",                 dispatch_mock_relu},
        {"rms_norm",                  dispatch_rms_norm},
        {"rope",                      dispatch_rope},
        {"rope_batch",                dispatch_rope_batch},
        {"linear",                    dispatch_linear},
        {"causal_attention",          dispatch_causal_attention},
        {"linear_tiled",              dispatch_linear_tiled},
        {"softmax",                   dispatch_softmax},
        {"silu",                      dispatch_silu},
        {"gelu",                      dispatch_gelu},
        {"elementwise_mul",           dispatch_elementwise_mul},
        {"embedding_lookup",          dispatch_embedding_lookup},
        {"argmax",                    dispatch_argmax},
        {"causal_attention_cached",   dispatch_causal_attn_cached},
        {"flash_attention_cached",    dispatch_flash_attn_cached},
        {"dequant_q4_0_linear",       dispatch_fused_dq4_linear},
        {"dequant_q4_0_linear_f16",   dispatch_fused_dq4_linear},
        {"flash_attention_paged",     dispatch_flash_attn_paged},
        {"flash_attention_gqa",       dispatch_flash_attn_gqa},
        {"moe_top_k_gating",         dispatch_moe_gate},
        {"moe_scatter_gather",        dispatch_moe_scatter},
    };

    /* Register all dequant ops into the main dispatch map */
    for (auto& [name, info] : s_dequant_map) {
        s_dispatch_map[name] = dispatch_dequant_generic;
    }
}

/* ================================================================== */
/*  Phase 29: O(1) Hash Dispatch (replaces ~750-line strcmp chain)       */
/* ================================================================== */

static nf_status metal_dispatch(nf_provider self, const char* op_name,
                                const nf_buffer* inputs, uint32_t n_in,
                                nf_buffer* outputs, uint32_t n_out) {
    auto* prov = reinterpret_cast<nf_provider_metal*>(self);
    auto it = s_dispatch_map.find(op_name);
    if (it != s_dispatch_map.end())
        return it->second(prov, op_name, inputs, n_in, outputs, n_out);
    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status metal_synchronize(nf_provider self) {
    auto* prov = reinterpret_cast<nf_provider_metal*>(self);
    if (!prov->queue) return NF_OK;
    @autoreleasepool {
        /* Drain the command queue by committing an empty buffer and waiting */
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    return NF_OK;
}

/* ================================================================== */
/*  Memory Provider VTable (Phase 2 extension)                         */
/* ================================================================== */

static nf_status metal_mem_alloc(nf_provider self,
                                 const nf_buffer_alloc_request* req,
                                 nf_buffer_ops* ops,
                                 nf_buffer* buf) {
    nf_status st = metal_buffer_alloc(self, &req->desc, buf);
    if (st != NF_OK) return st;
    *ops = make_metal_buf_ops();
    return NF_OK;
}

static nf_status metal_mem_import(nf_provider, uint64_t, nf_mem_domain,
                                  const nf_tensor_desc*, nf_buffer_ops*,
                                  nf_buffer*) {
    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status metal_mem_can_import(nf_provider, nf_mem_domain domain) {
    return (domain == NF_MEM_DOMAIN_UNIFIED) ? NF_OK : NF_ERROR_UNSUPPORTED_OP;
}

/* ================================================================== */
/*  Plugin Entry Points                                                */
/* ================================================================== */

extern "C" NF_API nf_status nf_plugin_register(nf_provider_vtable* vt,
                                                nf_provider* out) {
    vt->get_name        = metal_get_name;
    vt->get_abi_version = metal_get_abi_version;
    vt->init            = metal_init;
    vt->shutdown        = metal_shutdown;
    vt->buffer_alloc    = metal_buffer_alloc;
    vt->buffer_free     = metal_buffer_free;
    vt->buffer_map      = metal_buffer_map;
    vt->buffer_unmap    = metal_buffer_unmap;
    vt->dispatch        = metal_dispatch;
    vt->synchronize     = metal_synchronize;

    *out = reinterpret_cast<nf_provider>(&s_instance);
    return NF_OK;
}

extern "C" NF_API nf_status nf_plugin_register_mem(
        nf_provider_mem_vtable* mem_vt) {
    mem_vt->alloc         = metal_mem_alloc;
    mem_vt->import_buffer = metal_mem_import;
    mem_vt->can_import    = metal_mem_can_import;
    return NF_OK;
}
