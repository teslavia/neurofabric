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

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"

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
    uint  max_seq_len;  // KV cache max capacity (was _pad0)
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
    float max_score = -1e30f;
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

    if (head >= pc.n_heads) return;

    /* ---- Prefill mode (step == 0): full causal attention ---- */
    if (step == 0) {
        if (q_pos >= pc.seq_len) return;
        /* Copy K_new/V_new into cache */
        for (uint d = 0; d < dim; ++d) {
            uint src = head * pc.seq_len * dim + q_pos * dim + d;
            uint dst = head * pc.max_seq_len * dim + q_pos * dim + d;
            cache_k[dst] = K_new[src];
            cache_v[dst] = V_new[src];
        }
        /* Full causal attention (same as causal_attention kernel) */
        float scale = rsqrt(float(dim));
        float max_score = -1e30f;
        for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                     * cache_k[head * pc.max_seq_len * dim + k_pos * dim + d];
            dot *= scale;
            if (dot > max_score) max_score = dot;
        }
        float sum_exp = 0.0f;
        for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
            float dot = 0.0f;
            for (uint d = 0; d < dim; ++d)
                dot += Q[head * pc.seq_len * dim + q_pos * dim + d]
                     * cache_k[head * pc.max_seq_len * dim + k_pos * dim + d];
            sum_exp += exp(dot * scale - max_score);
        }
        for (uint d = 0; d < dim; ++d) {
            float val = 0.0f;
            for (uint k_pos = 0; k_pos <= q_pos; ++k_pos) {
                float dot2 = 0.0f;
                for (uint dd = 0; dd < dim; ++dd)
                    dot2 += Q[head * pc.seq_len * dim + q_pos * dim + dd]
                          * cache_k[head * pc.max_seq_len * dim + k_pos * dim + dd];
                float w = exp(dot2 * scale - max_score) / sum_exp;
                val += w * cache_v[head * pc.max_seq_len * dim + k_pos * dim + d];
            }
            out[head * pc.seq_len * dim + q_pos * dim + d] = val;
        }
        return;
    }

    /* ---- Decode mode (step > 0): single-token attention ---- */
    if (q_pos >= 1) return;  /* only 1 query token in decode */
    if (step >= pc.max_seq_len) return;  /* bounds check */

    /* Append new K/V to cache at position [step] */
    for (uint d = 0; d < dim; ++d) {
        cache_k[head * pc.max_seq_len * dim + step * dim + d] = K_new[head * dim + d];
        cache_v[head * pc.max_seq_len * dim + step * dim + d] = V_new[head * dim + d];
    }

    /* Attention: Q (1 token) against cache_k[0..step] */
    float scale = rsqrt(float(dim));
    float max_score = -1e30f;
    for (uint k_pos = 0; k_pos <= step; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * dim + d]
                 * cache_k[head * pc.max_seq_len * dim + k_pos * dim + d];
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }
    float sum_exp = 0.0f;
    for (uint k_pos = 0; k_pos <= step; ++k_pos) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += Q[head * dim + d]
                 * cache_k[head * pc.max_seq_len * dim + k_pos * dim + d];
        sum_exp += exp(dot * scale - max_score);
    }
    for (uint d = 0; d < dim; ++d) {
        float val = 0.0f;
        for (uint k_pos = 0; k_pos <= step; ++k_pos) {
            float dot2 = 0.0f;
            for (uint dd = 0; dd < dim; ++dd)
                dot2 += Q[head * dim + dd]
                      * cache_k[head * pc.max_seq_len * dim + k_pos * dim + dd];
            float w = exp(dot2 * scale - max_score) / sum_exp;
            val += w * cache_v[head * pc.max_seq_len * dim + k_pos * dim + d];
        }
        out[head * dim + d] = val;
    }
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

    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

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
    id<MTLComputePipelineState> fn_vector_add = nil;
    id<MTLComputePipelineState> fn_relu       = nil;
    id<MTLComputePipelineState> fn_attn_k     = nil;
    id<MTLComputePipelineState> fn_attn_v     = nil;
    id<MTLComputePipelineState> fn_rms_norm   = nil;
    id<MTLComputePipelineState> fn_rope       = nil;
    id<MTLComputePipelineState> fn_linear     = nil;
    id<MTLComputePipelineState> fn_causal_attn = nil;
    id<MTLComputePipelineState> fn_dequant_q4_0 = nil;
    id<MTLComputePipelineState> fn_dequant_q8_0 = nil;
    id<MTLComputePipelineState> fn_linear_tiled = nil;
    id<MTLComputePipelineState> fn_softmax = nil;
    id<MTLComputePipelineState> fn_silu = nil;
    id<MTLComputePipelineState> fn_elem_mul = nil;
    id<MTLComputePipelineState> fn_embed_lookup = nil;
    id<MTLComputePipelineState> fn_argmax = nil;
    id<MTLComputePipelineState> fn_causal_attn_cached = nil;
};

static nf_provider_metal s_instance;

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

        /* Create pipeline states for each kernel */
        auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [p->library newFunctionWithName:name];
            if (!fn) return nil;
            return [p->device newComputePipelineStateWithFunction:fn error:&error];
        };

        p->fn_vector_add = make_pso(@"vector_add");
        p->fn_relu       = make_pso(@"relu");
        p->fn_attn_k     = make_pso(@"attention_prefill_k");
        p->fn_attn_v     = make_pso(@"attention_prefill_v");
        p->fn_rms_norm   = make_pso(@"rms_norm");
        p->fn_rope       = make_pso(@"rope");
        p->fn_linear     = make_pso(@"linear");
        p->fn_causal_attn = make_pso(@"causal_attention");
        p->fn_dequant_q4_0 = make_pso(@"dequant_q4_0");
        p->fn_dequant_q8_0 = make_pso(@"dequant_q8_0");
        p->fn_linear_tiled = make_pso(@"linear_tiled");
        p->fn_softmax      = make_pso(@"softmax");
        p->fn_silu         = make_pso(@"silu");
        p->fn_elem_mul     = make_pso(@"elementwise_mul");
        p->fn_embed_lookup = make_pso(@"embedding_lookup");
        p->fn_argmax       = make_pso(@"argmax_rows");
        p->fn_causal_attn_cached = make_pso(@"causal_attention_cached");

        if (!p->fn_vector_add || !p->fn_relu || !p->fn_attn_k || !p->fn_attn_v ||
            !p->fn_rms_norm || !p->fn_rope || !p->fn_linear || !p->fn_causal_attn ||
            !p->fn_dequant_q4_0 || !p->fn_dequant_q8_0 || !p->fn_linear_tiled ||
            !p->fn_softmax || !p->fn_silu || !p->fn_elem_mul ||
            !p->fn_embed_lookup || !p->fn_argmax || !p->fn_causal_attn_cached) {
            NSLog(@"[NF Metal] Pipeline state creation failed: %@", error);
            return NF_ERROR_INTERNAL;
        }
    }

    p->initialized = true;
    return NF_OK;
}

static void metal_shutdown(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_metal*>(self);
    metal_synchronize(self);
    p->fn_vector_add = nil;
    p->fn_relu       = nil;
    p->fn_attn_k     = nil;
    p->fn_attn_v     = nil;
    p->fn_rms_norm   = nil;
    p->fn_rope       = nil;
    p->fn_linear     = nil;
    p->fn_causal_attn = nil;
    p->fn_dequant_q4_0 = nil;
    p->fn_dequant_q8_0 = nil;
    p->fn_linear_tiled = nil;
    p->fn_softmax      = nil;
    p->fn_silu         = nil;
    p->fn_elem_mul     = nil;
    p->fn_embed_lookup = nil;
    p->fn_argmax       = nil;
    p->fn_causal_attn_cached = nil;
    p->library       = nil;
    p->queue         = nil;
    p->device        = nil;
    p->initialized   = false;
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

/**
 * Helper: encode a 1-input, 1-output unary compute kernel.
 * Marks output gpu_done=false, commits async, signals fence on completion.
 */
static nf_status dispatch_unary(nf_provider_metal* prov,
                                id<MTLComputePipelineState> pso,
                                MetalBuffer* in_mb, MetalBuffer* out_mb) {
    out_mb->gpu_done.store(false, std::memory_order_release);

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
        [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];

        NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
        if (tpg > count) tpg = count;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
            out_mb->gpu_done.store(true, std::memory_order_release);
            out_mb->fence_cv.notify_all();
        }];

        [cmdBuf commit];
    }
    return NF_OK;
}

static nf_status metal_dispatch(nf_provider self, const char* op_name,
                                const nf_buffer* inputs, uint32_t n_in,
                                nf_buffer* outputs, uint32_t n_out) {
    auto* prov = reinterpret_cast<nf_provider_metal*>(self);

    /* ---- vector_add: 2 inputs → 1 output ---- */
    if (std::strcmp(op_name, "metal_vector_add") == 0 && n_in >= 2 && n_out >= 1) {
        auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);

        out_mb->gpu_done.store(false, std::memory_order_release);

        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            [enc setComputePipelineState:prov->fn_vector_add];
            [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
            [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];

            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_vector_add.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];

            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];

            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- attention_prefill: 1 input → 2 outputs (K, V) ---- */
    if (std::strcmp(op_name, "attention_prefill") == 0 && n_in >= 1 && n_out >= 2) {
        auto* in_mb = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* k_mb  = reinterpret_cast<MetalBuffer*>(outputs[0]);
        auto* v_mb  = reinterpret_cast<MetalBuffer*>(outputs[1]);

        k_mb->gpu_done.store(false, std::memory_order_release);
        v_mb->gpu_done.store(false, std::memory_order_release);

        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];

            /* Encode K = input * 0.5 */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:prov->fn_attn_k];
                [enc setBuffer:in_mb->mtl_buffer offset:0 atIndex:0];
                [enc setBuffer:k_mb->mtl_buffer  offset:0 atIndex:1];
                NSUInteger count = k_mb->desc.size_bytes / sizeof(float);
                NSUInteger tpg = prov->fn_attn_k.maxTotalThreadsPerThreadgroup;
                if (tpg > count) tpg = count;
                [enc dispatchThreads:MTLSizeMake(count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                [enc endEncoding];
            }

            /* Encode V = input * -0.25 */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:prov->fn_attn_v];
                [enc setBuffer:in_mb->mtl_buffer offset:0 atIndex:0];
                [enc setBuffer:v_mb->mtl_buffer  offset:0 atIndex:1];
                NSUInteger count = v_mb->desc.size_bytes / sizeof(float);
                NSUInteger tpg = prov->fn_attn_v.maxTotalThreadsPerThreadgroup;
                if (tpg > count) tpg = count;
                [enc dispatchThreads:MTLSizeMake(count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                [enc endEncoding];
            }

            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                k_mb->gpu_done.store(true, std::memory_order_release);
                k_mb->fence_cv.notify_all();
                v_mb->gpu_done.store(true, std::memory_order_release);
                v_mb->fence_cv.notify_all();
            }];

            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- mock_relu: 1 input → 1 output (or in-place) ---- */
    if (std::strcmp(op_name, "mock_relu") == 0 && n_in >= 1) {
        auto* in_mb = reinterpret_cast<MetalBuffer*>(inputs[0]);

        if (n_out >= 1 && outputs[0]) {
            /* Out-of-place: input → relu → output */
            auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
            return dispatch_unary(prov, prov->fn_relu, in_mb, out_mb);
        } else {
            /* In-place: input → relu → input */
            return dispatch_unary(prov, prov->fn_relu, in_mb, in_mb);
        }
    }

    /* ---- Phase 16: LLM kernels with push_constants via user_data ---- */

    /* Recover push_constants from nf_task_desc via user_data pointer */
    auto get_push_constants = [&]() -> const uint8_t* {
        /* PipelineEngine sets user_data = &desc before dispatch */
        nf_task_desc* td = reinterpret_cast<nf_task_desc*>(
            reinterpret_cast<uint8_t*>(const_cast<nf_buffer*>(inputs))
            - offsetof(nf_task_desc, inputs));
        if (td->push_constants_size == 0) return nullptr;
        return td->push_constants;
    };

    /* ---- rms_norm: 2 inputs (data, weights) → 1 output ---- */
    if (std::strcmp(op_name, "rms_norm") == 0 && n_in >= 2 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* wt_mb  = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_rms_norm];
            [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
            [enc setBuffer:wt_mb->mtl_buffer  offset:0 atIndex:2];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_rms_norm.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- rope: 1 input → 1 output ---- */
    if (std::strcmp(op_name, "rope") == 0 && n_in >= 1 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_rope];
            [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_rope.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- linear: 2 inputs → 1 output (2D dispatch) ---- */
    if (std::strcmp(op_name, "linear") == 0 && n_in >= 2 && n_out >= 1) {
        auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_linear];
            [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
            [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            /* Recover M, N from push constants for 2D grid */
            uint32_t M = 1, N = 1;
            if (pc) {
                /* PushConstants layout: seq_len(4) n_heads(4) head_dim(4) epsilon(4) theta(4) M(4) N(4) K(4) */
                std::memcpy(&M, pc + 20, sizeof(uint32_t));
                std::memcpy(&N, pc + 24, sizeof(uint32_t));
            }
            [enc dispatchThreads:MTLSizeMake(N, M, 1)
           threadsPerThreadgroup:MTLSizeMake(
               MIN((NSUInteger)N, (NSUInteger)16),
               MIN((NSUInteger)M, (NSUInteger)16), 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- causal_attention: 3 inputs (Q,K,V) → 1 output (2D dispatch) ---- */
    if (std::strcmp(op_name, "causal_attention") == 0 && n_in >= 3 && n_out >= 1) {
        auto* q_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* k_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* v_mb   = reinterpret_cast<MetalBuffer*>(inputs[2]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_causal_attn];
            [enc setBuffer:q_mb->mtl_buffer   offset:0 atIndex:0];
            [enc setBuffer:k_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:v_mb->mtl_buffer   offset:0 atIndex:2];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:3];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            /* seq_len @ byte 0, n_heads @ byte 4 */
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
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- Phase 17: dequant_q4_0: 1 input (quantized) → 1 output (float) ---- */
    if (std::strcmp(op_name, "dequant_q4_0") == 0 && n_in >= 1 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_dequant_q4_0];
            [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
            /* n_elements = output size / sizeof(float) */
            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_dequant_q4_0.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- dequant_q8_0: 1 input (quantized) → 1 output (float) ---- */
    if (std::strcmp(op_name, "dequant_q8_0") == 0 && n_in >= 1 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_dequant_q8_0];
            [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_dequant_q8_0.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- linear_tiled: 2 inputs → 1 output (16x16 tiled matmul) ---- */
    if (std::strcmp(op_name, "linear_tiled") == 0 && n_in >= 2 && n_out >= 1) {
        auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_linear_tiled];
            [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
            [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            uint32_t M = 1, N = 1;
            if (pc) {
                std::memcpy(&M, pc + 20, sizeof(uint32_t));
                std::memcpy(&N, pc + 24, sizeof(uint32_t));
            }
            /* Round up grid to tile boundaries */
            NSUInteger gridW = ((N + 15) / 16) * 16;
            NSUInteger gridH = ((M + 15) / 16) * 16;
            [enc dispatchThreads:MTLSizeMake(gridW, gridH, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- Phase 18: Transformer layer primitives ---- */

    /* ---- softmax: 1 input → 1 output (1 thread per row) ---- */
    if (std::strcmp(op_name, "softmax") == 0 && n_in >= 1 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_softmax];
            [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            uint32_t rows = 1;
            if (pc) std::memcpy(&rows, pc, sizeof(uint32_t)); /* seq_len */
            NSUInteger tpg = prov->fn_softmax.maxTotalThreadsPerThreadgroup;
            if (tpg > rows) tpg = rows;
            [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- silu: 1 input → 1 output ---- */
    if (std::strcmp(op_name, "silu") == 0 && n_in >= 1 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        return dispatch_unary(prov, prov->fn_silu, in_mb, out_mb);
    }

    /* ---- elementwise_mul: 2 inputs → 1 output ---- */
    if (std::strcmp(op_name, "elementwise_mul") == 0 && n_in >= 2 && n_out >= 1) {
        auto* a_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* b_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_elem_mul];
            [enc setBuffer:a_mb->mtl_buffer   offset:0 atIndex:0];
            [enc setBuffer:b_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_elem_mul.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- embedding_lookup: 2 inputs (weights, token_ids) → 1 output ---- */
    if (std::strcmp(op_name, "embedding_lookup") == 0 && n_in >= 2 && n_out >= 1) {
        auto* w_mb   = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* t_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_embed_lookup];
            [enc setBuffer:w_mb->mtl_buffer   offset:0 atIndex:0];
            [enc setBuffer:t_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:2];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            NSUInteger count = out_mb->desc.size_bytes / sizeof(float);
            NSUInteger tpg = prov->fn_embed_lookup.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- argmax: 1 input → 1 output (int32, 1 thread per row) ---- */
    if (std::strcmp(op_name, "argmax") == 0 && n_in >= 1 && n_out >= 1) {
        auto* in_mb  = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* out_mb = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_argmax];
            [enc setBuffer:in_mb->mtl_buffer  offset:0 atIndex:0];
            [enc setBuffer:out_mb->mtl_buffer offset:0 atIndex:1];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];
            uint32_t rows = 1;
            if (pc) std::memcpy(&rows, pc, sizeof(uint32_t));
            NSUInteger tpg = prov->fn_argmax.maxTotalThreadsPerThreadgroup;
            if (tpg > rows) tpg = rows;
            [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

    /* ---- Phase 19: causal_attention_cached: 3 in (Q,K,V) + 2 in/out (cache_k,cache_v) → 1 out ---- */
    if (std::strcmp(op_name, "causal_attention_cached") == 0 && n_in >= 5 && n_out >= 1) {
        auto* q_mb       = reinterpret_cast<MetalBuffer*>(inputs[0]);
        auto* k_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[1]);
        auto* v_new_mb   = reinterpret_cast<MetalBuffer*>(inputs[2]);
        auto* cache_k_mb = reinterpret_cast<MetalBuffer*>(inputs[3]);
        auto* cache_v_mb = reinterpret_cast<MetalBuffer*>(inputs[4]);
        auto* out_mb     = reinterpret_cast<MetalBuffer*>(outputs[0]);
        const uint8_t* pc = get_push_constants();

        out_mb->gpu_done.store(false, std::memory_order_release);
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [prov->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:prov->fn_causal_attn_cached];
            [enc setBuffer:q_mb->mtl_buffer       offset:0 atIndex:0];
            [enc setBuffer:k_new_mb->mtl_buffer   offset:0 atIndex:1];
            [enc setBuffer:v_new_mb->mtl_buffer   offset:0 atIndex:2];
            [enc setBuffer:cache_k_mb->mtl_buffer offset:0 atIndex:3];
            [enc setBuffer:cache_v_mb->mtl_buffer offset:0 atIndex:4];
            [enc setBuffer:out_mb->mtl_buffer     offset:0 atIndex:5];
            if (pc) [enc setBytes:pc length:NF_MAX_PUSH_CONSTANTS atIndex:15];

            /* PushConstants layout (40B): seq_len(0) n_heads(4) head_dim(8)
               epsilon(12) theta(16) M(20) N(24) K(28) step_idx(32) max_seq_len(36) */
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
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
                out_mb->gpu_done.store(true, std::memory_order_release);
                out_mb->fence_cv.notify_all();
            }];
            [cmdBuf commit];
        }
        return NF_OK;
    }

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
