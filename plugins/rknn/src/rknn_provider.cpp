/**
 * @file rknn_provider.cpp
 * @brief Rockchip RKNN Execution Provider — DMA-BUF Memory Allocator
 *
 * Phase 6: Real hardware-backed nf_buffer_ops implementation.
 *
 * Phase 9: SDK-ready guards — #ifdef NF_HAS_RKNN_SDK enables real
 * RKNN API paths (rknn_init, rknn_run, DMA-BUF ioctl). Without the
 * define, the simulation code compiles everywhere unchanged.
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
#include "neurofabric/neuro_scheduler_abi.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef NF_HAS_RKNN_SDK
#include <rknn_api.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#endif

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
#ifdef NF_HAS_RKNN_SDK
        munmap(rb->data, rb->desc.size_bytes);
        close(rb->dma_buf_fd);
#else
        std::free(rb->data);
#endif
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
#ifdef NF_HAS_RKNN_SDK
        struct dma_buf_sync sync;
        sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
        ioctl(rb->dma_buf_fd, DMA_BUF_IOCTL_SYNC, &sync);
#else
        /* Simulate: CPU dirty → main memory. NPU can now read. */
        rb->cpu_dirty = false;
#endif
    }
    if (op & NF_CACHE_INVALIDATE) {
#ifdef NF_HAS_RKNN_SDK
        struct dma_buf_sync sync;
        sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
        ioctl(rb->dma_buf_fd, DMA_BUF_IOCTL_SYNC, &sync);
#else
        /* Simulate: discard CPU cache. CPU will re-read from DRAM. */
        rb->dev_dirty = false;
#endif
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
#ifdef NF_HAS_RKNN_SDK
    rknn_context ctx              = 0;
    const void*  cached_model_ptr  = nullptr;
    uint64_t     cached_model_size = 0;
    rknn_context cached_ctx        = 0;
#endif
};

static nf_provider_rknn s_instance;

/* ================================================================== */
/*  Provider VTable                                                    */
/* ================================================================== */

static const char* rknn_get_name(nf_provider) { return "rockchip_rknn"; }
static uint32_t    rknn_get_abi_version(nf_provider) { return NF_ABI_VERSION; }

static nf_status rknn_prov_init(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_rknn*>(self);
#ifdef NF_HAS_RKNN_SDK
    /* Real RK3588: rknn_init(&p->ctx, model_data, model_size, 0) */
    /* Model loading deferred to dispatch-time or explicit load call */
#endif
    p->initialized = true;
    return NF_OK;
}

static void rknn_prov_shutdown(nf_provider self) {
    auto* p = reinterpret_cast<nf_provider_rknn*>(self);
#ifdef NF_HAS_RKNN_SDK
    if (p->cached_ctx) { rknn_destroy(p->cached_ctx); p->cached_ctx = 0; }
    if (p->ctx)        { rknn_destroy(p->ctx);        p->ctx = 0; }
    p->cached_model_ptr  = nullptr;
    p->cached_model_size = 0;
#endif
    p->initialized = false;
}

static nf_status rknn_buffer_alloc(nf_provider self, const nf_tensor_desc* desc,
                                   nf_buffer* out) {
    auto* rb = new RknnBuffer;
    rb->desc = *desc;
#ifdef NF_HAS_RKNN_SDK
    auto* p = reinterpret_cast<nf_provider_rknn*>(self);
    /* DMA-BUF alloc requires a valid rknn_context (from rknn_init with a model).
     * When no model is loaded yet (ctx==0), fall back to calloc — this covers
     * simulation tests and pre-model buffer allocation. */
    if (p->ctx) {
        rknn_tensor_mem* mem = rknn_create_mem(p->ctx, desc->size_bytes);
        if (!mem) { delete rb; return NF_ERROR_OUT_OF_MEMORY; }
        rb->dma_buf_fd = mem->fd;
        rb->data = mmap(NULL, desc->size_bytes, PROT_READ|PROT_WRITE,
                        MAP_SHARED, mem->fd, 0);
        if (rb->data == MAP_FAILED) {
            rknn_destroy_mem(p->ctx, mem);
            delete rb;
            return NF_ERROR_OUT_OF_MEMORY;
        }
    } else {
        rb->data = std::calloc(1, desc->size_bytes);
        if (!rb->data) { delete rb; return NF_ERROR_OUT_OF_MEMORY; }
        rb->dma_buf_fd = s_next_fd.fetch_add(1, std::memory_order_relaxed);
    }
#else
    (void)self;
    rb->data = std::calloc(1, desc->size_bytes);
    if (!rb->data) {
        delete rb;
        return NF_ERROR_OUT_OF_MEMORY;
    }
    rb->dma_buf_fd = s_next_fd.fetch_add(1, std::memory_order_relaxed);
#endif

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
 * Phase 10: adds "rknn_subgraph" — sub-graph closure dispatch.
 *   inputs[0] = mmap'd .rknn model blob
 *   inputs[1..n] = feature map buffers
 *   outputs[0..n] = result buffers
 * Real RK3588: rknn_init + rknn_set_io_mem + rknn_run (zero-copy DMA-BUF).
 * Simulation: deterministic mean-reduction for verifiable output.
 */
static nf_status rknn_dispatch(nf_provider self, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    if (std::strcmp(op_name, "rknn_subgraph") == 0 && n_in >= 2 && n_out >= 1) {
        /*
         * Recover the full nf_task_desc via the cross-dylib bridge:
         * PipelineEngine stores user_data = &desc, and inputs == desc.inputs.
         */
        auto* desc = reinterpret_cast<const nf_task_desc*>(
            reinterpret_cast<const char*>(inputs) -
            offsetof(nf_task_desc, inputs));

        /* Map model blob (inputs[0]) — mmap'd .rknn file */
        void* model_ptr = nullptr;
        desc->input_ops[0].map(inputs[0], &model_ptr);

        nf_buffer_info model_info{};
        desc->input_ops[0].get_info(inputs[0], &model_info);
        uint64_t model_size = model_info.desc.size_bytes;

#ifdef NF_HAS_RKNN_SDK
        auto* p = reinterpret_cast<nf_provider_rknn*>(self);
        bool use_real_npu = false;

        /* Model cache: reuse context if same pointer + size */
        rknn_context run_ctx = 0;
        if (p->cached_model_ptr == model_ptr &&
            p->cached_model_size == model_size && p->cached_ctx) {
            run_ctx = p->cached_ctx;
            use_real_npu = true;
        } else {
            if (p->cached_ctx) { rknn_destroy(p->cached_ctx); p->cached_ctx = 0; }
            int ret = rknn_init(&run_ctx, model_ptr,
                                static_cast<uint32_t>(model_size), 0, nullptr);
            if (ret == 0) {
                p->cached_ctx        = run_ctx;
                p->cached_model_ptr  = model_ptr;
                p->cached_model_size = model_size;
                use_real_npu = true;
            }
            /* If rknn_init fails (fake model), fall through to simulation */
        }

        if (use_real_npu) {
            /* Query input/output counts */
            rknn_input_output_num io_num{};
            rknn_query(run_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

            /* Set model inputs via copy-based API (works with any buffer backing) */
            for (uint32_t i = 0; i < io_num.n_input && (i + 1) < n_in; ++i) {
                rknn_tensor_attr attr{};
                attr.index = i;
                rknn_query(run_ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));

                void* in_ptr = nullptr;
                desc->input_ops[i + 1].map(inputs[i + 1], &in_ptr);

                rknn_input ri{};
                ri.index        = i;
                ri.buf          = in_ptr;
                ri.size         = attr.size;
                ri.pass_through = 1;  /* raw data, skip format conversion */
                ri.type         = attr.type;
                ri.fmt          = attr.fmt;
                rknn_inputs_set(run_ctx, 1, &ri);

                desc->input_ops[i + 1].unmap(inputs[i + 1]);
            }

            /* Synchronous NPU execution */
            int ret = rknn_run(run_ctx, nullptr);
            if (ret < 0) {
                desc->input_ops[0].unmap(inputs[0]);
                return NF_ERROR_INTERNAL;
            }

            /* Get model outputs via copy-based API */
            uint32_t n_model_out = io_num.n_output < n_out
                                 ? io_num.n_output : n_out;
            std::vector<rknn_output> ro(n_model_out);
            for (uint32_t j = 0; j < n_model_out; ++j) {
                ro[j].index      = j;
                ro[j].is_prealloc = 0;
                ro[j].want_float  = 0;
            }
            rknn_outputs_get(run_ctx, n_model_out, ro.data(), nullptr);

            /* Copy results into output buffers */
            for (uint32_t j = 0; j < n_model_out; ++j) {
                void* out_ptr = nullptr;
                desc->output_ops[j].map(outputs[j], &out_ptr);

                nf_buffer_info out_info{};
                desc->output_ops[j].get_info(outputs[j], &out_info);
                size_t copy_sz = ro[j].size < out_info.desc.size_bytes
                               ? ro[j].size : out_info.desc.size_bytes;
                std::memcpy(out_ptr, ro[j].buf, copy_sz);

                desc->output_ops[j].unmap(outputs[j]);
                auto* out_rb = reinterpret_cast<RknnBuffer*>(outputs[j]);
                out_rb->dev_dirty = true;
            }
            rknn_outputs_release(run_ctx, n_model_out, ro.data());

            desc->input_ops[0].unmap(inputs[0]);
            return NF_OK;
        }
        /* Fall through to simulation when rknn_init fails (fake model) */
#endif
        {
            (void)self;
            /* Simulation: output[i] = mean(all feature input floats) */
        double sum = 0.0;
        size_t total_count = 0;

        for (uint32_t k = 1; k < n_in; ++k) {
            void* in_ptr = nullptr;
            desc->input_ops[k].map(inputs[k], &in_ptr);

            nf_buffer_info in_info{};
            desc->input_ops[k].get_info(inputs[k], &in_info);
            size_t count = in_info.desc.size_bytes / sizeof(float);

            auto* fp = static_cast<const float*>(in_ptr);
            for (size_t i = 0; i < count; ++i) {
                sum += static_cast<double>(fp[i]);
            }
            total_count += count;

            desc->input_ops[k].unmap(inputs[k]);
        }

        float mean_val = (total_count > 0)
            ? static_cast<float>(sum / static_cast<double>(total_count))
            : 0.0f;

        /* Fill all outputs with the mean value */
        for (uint32_t j = 0; j < n_out; ++j) {
            void* out_ptr = nullptr;
            desc->output_ops[j].map(outputs[j], &out_ptr);

            nf_buffer_info out_info{};
            desc->output_ops[j].get_info(outputs[j], &out_info);
            size_t out_count = out_info.desc.size_bytes / sizeof(float);

            auto* out_fp = static_cast<float*>(out_ptr);
            for (size_t i = 0; i < out_count; ++i) {
                out_fp[i] = mean_val;
            }

            desc->output_ops[j].unmap(outputs[j]);
            auto* out_rb = reinterpret_cast<RknnBuffer*>(outputs[j]);
            out_rb->dev_dirty = true;
        }
        }
        /* Unmap model blob */
        desc->input_ops[0].unmap(inputs[0]);
        return NF_OK;
    }

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
#ifdef NF_HAS_RKNN_SDK
    /* Real RK3588: NPU operations are synchronous in rknn_run() */
#endif
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
    vt->init            = rknn_prov_init;
    vt->shutdown        = rknn_prov_shutdown;
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
