/**
 * @file rknn_provider.cpp
 * @brief Rockchip RKNN Execution Provider — DMA-BUF Memory Allocator
 *
 * Phase 6: Real hardware-backed nf_buffer_ops implementation.
 *
 * On RK3588 (Rock 5B+), the NPU accesses memory via DMA-BUF file
 * descriptors allocated from the CMA (Contiguous Memory Allocator).
 * The CPU accesses the same memory via mmap() of the DMA-BUF fd.
 *
 * CRITICAL: RK3588 is NOT hardware cache-coherent between CPU and NPU.
 * After CPU writes (e.g. network recv), we MUST flush the CPU cache
 * to main memory before the NPU reads. After NPU writes, we MUST
 * invalidate the CPU cache before CPU reads.
 *
 * Real API calls (stubbed with behavioral simulation):
 *   - rknn_create_mem()  → allocates CMA DMA-BUF
 *   - rknn_destroy_mem() → frees DMA-BUF
 *   - DMA_BUF_IOCTL_SYNC → cache maintenance
 *   - mmap(fd)           → CPU-visible mapping
 *
 * When building on real RK3588 with RKNN SDK, replace stubs with:
 *   #include "rknn_api.h"
 *   #include <linux/dma-buf.h>
 *   #include <sys/ioctl.h>
 *   #include <sys/mman.h>
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

/* ================================================================== */
/*  DMA-BUF Simulation                                                 */
/*  On real RK3588, these would be actual fd + ioctl operations.       */
/* ================================================================== */

/** Simulated DMA-BUF fd counter (real: from /dev/dma_heap/). */
static std::atomic<int> s_next_fd{100};

/** Simulated cache dirty flag — tracks whether CPU cache is stale. */
struct RknnBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data        = nullptr;  /**< mmap'd VA        */
    nf_tensor_desc        desc{};
    int                   dma_buf_fd  = -1;       /**< DMA-BUF fd       */
    bool                  mapped      = false;
    bool                  cpu_dirty   = false;     /**< CPU wrote, needs flush */
    bool                  dev_dirty   = false;     /**< NPU wrote, needs inv  */
};

/* ================================================================== */
/*  nf_buffer_ops — RKNN DMA-BUF with explicit cache maintenance       */
/* ================================================================== */

static uint32_t rknn_buf_retain(nf_buffer self) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    return rb->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

static uint32_t rknn_buf_release(nf_buffer self) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    uint32_t prev = rb->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        /*
         * Real RK3588:
         *   munmap(rb->data, rb->desc.size_bytes);
         *   close(rb->dma_buf_fd);
         *   rknn_destroy_mem(ctx, rknn_mem);
         */
        std::free(rb->data);
        delete rb;
    }
    return prev - 1;
}

static nf_status rknn_buf_map(nf_buffer self, void** out_ptr) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    /*
     * Real RK3588:
     *   *out_ptr = mmap(NULL, size, PROT_READ|PROT_WRITE,
     *                   MAP_SHARED, rb->dma_buf_fd, 0);
     *
     * The mmap gives CPU access to the CMA DMA-BUF region.
     * This is NOT coherent with the NPU — cache ops are required.
     */
    if (rb->mapped) return NF_ERROR_INVALID_ARG;
    rb->mapped = true;
    *out_ptr = rb->data;
    return NF_OK;
}

static nf_status rknn_buf_unmap(nf_buffer self) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    /*
     * Real RK3588: munmap(rb->data, rb->desc.size_bytes);
     * Mark CPU as potentially dirty (wrote during map).
     */
    rb->mapped = false;
    rb->cpu_dirty = true;
    return NF_OK;
}

static nf_status rknn_buf_cache_sync(nf_buffer self, nf_cache_op op,
                                     uint64_t offset, uint64_t size) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    (void)offset; (void)size;

    /*
     * Real RK3588 — DMA_BUF_IOCTL_SYNC:
     *
     *   struct dma_buf_sync sync;
     *
     *   FLUSH (NF_CACHE_FLUSH):
     *     sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
     *     ioctl(rb->dma_buf_fd, DMA_BUF_IOCTL_SYNC, &sync);
     *     → Cleans CPU dirty cache lines to main memory.
     *     → MUST be called after CPU writes (e.g. network recv)
     *       and BEFORE NPU reads the buffer.
     *
     *   INVALIDATE (NF_CACHE_INVALIDATE):
     *     sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
     *     ioctl(rb->dma_buf_fd, DMA_BUF_IOCTL_SYNC, &sync);
     *     → Invalidates CPU cache lines so next CPU read fetches
     *       from main memory (which NPU may have written).
     *     → MUST be called after NPU writes and BEFORE CPU reads.
     */

    if (op & NF_CACHE_FLUSH) {
        /* Simulate: CPU dirty → main memory. NPU can now read. */
        rb->cpu_dirty = false;
    }
    if (op & NF_CACHE_INVALIDATE) {
        /* Simulate: discard CPU cache. CPU will re-read from DRAM. */
        rb->dev_dirty = false;
    }

    return NF_OK;
}

static nf_status rknn_buf_get_info(nf_buffer self, nf_buffer_info* info) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    info->desc         = rb->desc;
    info->domain       = NF_MEM_DOMAIN_DMA_BUF;
    info->offset_bytes = 0;
    info->share_token  = static_cast<uint64_t>(rb->dma_buf_fd);
    info->refcount     = rb->refcount.load(std::memory_order_relaxed);
    info->_reserved    = 0;
    return NF_OK;
}

static nf_status rknn_buf_export(nf_buffer self, uint64_t* token,
                                 nf_mem_domain* domain) {
    auto* rb = reinterpret_cast<RknnBuffer*>(self);
    /*
     * Real RK3588: dup(rb->dma_buf_fd) — caller owns the new fd.
     * Simulation: return the fd value directly.
     */
    *token  = static_cast<uint64_t>(rb->dma_buf_fd);
    *domain = NF_MEM_DOMAIN_DMA_BUF;
    return NF_OK;
}

static nf_buffer_ops make_rknn_buf_ops() {
    nf_buffer_ops ops{};
    ops.retain        = rknn_buf_retain;
    ops.release       = rknn_buf_release;
    ops.map           = rknn_buf_map;
    ops.unmap         = rknn_buf_unmap;
    ops.cache_sync    = rknn_buf_cache_sync;
    ops.get_info      = rknn_buf_get_info;
    ops.export_handle = rknn_buf_export;
    ops.import_handle = nullptr;
    return ops;
}

/* ================================================================== */
/*  NCHW → NHWC Strided Reorder (Phase 7)                              */
/*  RK3588 NPU expects NHWC layout; Metal sends NCHW.                  */
/* ================================================================== */

static void nchw_to_nhwc(const float* src, float* dst,
                          uint64_t N, uint64_t C, uint64_t H, uint64_t W) {
    for (uint64_t n = 0; n < N; ++n)
        for (uint64_t c = 0; c < C; ++c)
            for (uint64_t h = 0; h < H; ++h)
                for (uint64_t w = 0; w < W; ++w)
                    dst[n*H*W*C + h*W*C + w*C + c] =
                        src[n*C*H*W + c*H*W + h*W + w];
}

/* ================================================================== */
/*  Provider State                                                     */
/* ================================================================== */

struct nf_provider_rknn {
    bool initialized = false;
    /* Real RK3588: rknn_context ctx; */
};

static nf_provider_rknn s_instance;

/* ================================================================== */
/*  Provider VTable                                                    */
/* ================================================================== */

static const char* rknn_get_name(nf_provider) { return "rockchip_rknn"; }
static uint32_t    rknn_get_abi_version(nf_provider) { return NF_ABI_VERSION; }

static nf_status rknn_init(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_rknn*>(self);
    /* Real RK3588: rknn_init(&ctx, model_data, model_size, 0) */
    p->initialized = true;
    return NF_OK;
}

static void rknn_shutdown(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_rknn*>(self);
    p->initialized = false;
}

static nf_status rknn_buffer_alloc(nf_provider, const nf_tensor_desc* desc,
                                   nf_buffer* out) {
    /*
     * Real RK3588:
     *   rknn_tensor_mem* mem = rknn_create_mem(ctx, desc->size_bytes);
     *   int fd = mem->fd;  // DMA-BUF fd from CMA allocator
     *   void* va = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
     */
    auto* rb = new RknnBuffer;
    rb->desc = *desc;
    rb->data = std::calloc(1, desc->size_bytes);
    if (!rb->data) {
        delete rb;
        return NF_ERROR_OUT_OF_MEMORY;
    }
    rb->dma_buf_fd = s_next_fd.fetch_add(1, std::memory_order_relaxed);

    *out = reinterpret_cast<nf_buffer>(rb);
    return NF_OK;
}

static void rknn_buffer_free(nf_provider, nf_buffer buf) {
    if (buf) rknn_buf_release(buf);
}

static nf_status rknn_buffer_map(nf_provider, nf_buffer buf, void** out) {
    return rknn_buf_map(buf, out);
}

static nf_status rknn_buffer_unmap(nf_provider, nf_buffer buf) {
    return rknn_buf_unmap(buf);
}

/**
 * dispatch — executes operators on the RKNN NPU.
 *
 * Phase 6: supports "mock_relu" for E2E validation.
 * Phase 7: adds "decode_step" — simulated decode attention (tanh).
 * Real RK3588: rknn_inputs_set() + rknn_run() + rknn_outputs_get().
 */
static nf_status rknn_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    if (std::strcmp(op_name, "decode_step") == 0 && n_in >= 1 && n_out >= 1) {
        auto* rb = reinterpret_cast<RknnBuffer*>(inputs[0]);
        auto* out_rb = reinterpret_cast<RknnBuffer*>(outputs[0]);

        /* Pre-NPU fence: flush CPU cache if dirty */
        if (rb->cpu_dirty) {
            rknn_buf_cache_sync(inputs[0], NF_CACHE_FLUSH, 0, 0);
        }

        /* Simulated decode attention: output[i] = tanh(input[i]) */
        if (rb->desc.dtype == NF_DTYPE_F32) {
            auto* src = static_cast<float*>(rb->data);
            auto* dst = static_cast<float*>(out_rb->data);
            size_t count = rb->desc.size_bytes / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
                dst[i] = std::tanh(src[i]);
            }
        }

        /* Mark NPU as having written */
        out_rb->dev_dirty = true;
        return NF_OK;
    }

    if (std::strcmp(op_name, "mock_relu") == 0 && n_in >= 1) {
        auto* rb = reinterpret_cast<RknnBuffer*>(inputs[0]);

        /*
         * Before NPU reads: if CPU wrote this buffer (e.g. network recv),
         * the data is in CPU cache. We MUST flush to main memory.
         * The payload_serializer already calls cache_sync(FLUSH) after recv,
         * but defense-in-depth: check and flush again if needed.
         */
        if (rb->cpu_dirty) {
            rknn_buf_cache_sync(inputs[0], NF_CACHE_FLUSH, 0, 0);
        }

        /* Simulate NPU compute: ReLU in-place */
        if (rb->desc.dtype == NF_DTYPE_F32) {
            auto* fp = static_cast<float*>(rb->data);
            size_t count = rb->desc.size_bytes / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
                if (fp[i] < 0.0f) fp[i] = 0.0f;
            }
        }

        /* Mark device as having written (NPU output) */
        rb->dev_dirty = true;

        /* Copy to output if provided */
        if (n_out >= 1 && outputs[0]) {
            auto* out_rb = reinterpret_cast<RknnBuffer*>(outputs[0]);
            size_t copy_sz = rb->desc.size_bytes < out_rb->desc.size_bytes
                           ? rb->desc.size_bytes : out_rb->desc.size_bytes;
            std::memcpy(out_rb->data, rb->data, copy_sz);
            out_rb->dev_dirty = true;
        }

        return NF_OK;
    }

    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status rknn_synchronize(nf_provider) {
    /* Real RK3588: wait for NPU command queue to drain */
    return NF_OK;
}

/* ================================================================== */
/*  Memory Provider VTable                                             */
/* ================================================================== */

static nf_status rknn_mem_alloc(nf_provider self,
                                const nf_buffer_alloc_request* req,
                                nf_buffer_ops* ops,
                                nf_buffer* buf) {
    nf_status st = rknn_buffer_alloc(self, &req->desc, buf);
    if (st != NF_OK) return st;
    *ops = make_rknn_buf_ops();
    return NF_OK;
}

static nf_status rknn_mem_import(nf_provider, uint64_t, nf_mem_domain,
                                 const nf_tensor_desc*, nf_buffer_ops*,
                                 nf_buffer*) {
    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status rknn_mem_can_import(nf_provider, nf_mem_domain domain) {
    return (domain == NF_MEM_DOMAIN_DMA_BUF) ? NF_OK : NF_ERROR_UNSUPPORTED_OP;
}

/* ================================================================== */
/*  Plugin Entry Points                                                */
/* ================================================================== */

extern "C" NF_API nf_status nf_plugin_register(nf_provider_vtable* vt,
                                                nf_provider* out) {
    vt->get_name        = rknn_get_name;
    vt->get_abi_version = rknn_get_abi_version;
    vt->init            = rknn_init;
    vt->shutdown        = rknn_shutdown;
    vt->buffer_alloc    = rknn_buffer_alloc;
    vt->buffer_free     = rknn_buffer_free;
    vt->buffer_map      = rknn_buffer_map;
    vt->buffer_unmap    = rknn_buffer_unmap;
    vt->dispatch        = rknn_dispatch;
    vt->synchronize     = rknn_synchronize;

    *out = reinterpret_cast<nf_provider>(&s_instance);
    return NF_OK;
}

extern "C" NF_API nf_status nf_plugin_register_mem(
        nf_provider_mem_vtable* mem_vt) {
    mem_vt->alloc         = rknn_mem_alloc;
    mem_vt->import_buffer = rknn_mem_import;
    mem_vt->can_import    = rknn_mem_can_import;
    return NF_OK;
}
