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

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>

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

        if (!p->fn_vector_add || !p->fn_relu || !p->fn_attn_k || !p->fn_attn_v) {
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
