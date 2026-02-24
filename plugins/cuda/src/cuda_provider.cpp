/**
 * @file cuda_provider.cpp
 * @brief CUDA plugin — CPU fallback (stub) + optional real CUDA dispatch
 *
 * Phase 45E: Extension point for CUDA GPU acceleration.
 * When NF_HAS_CUDA is not defined, all ops execute on CPU as fallback.
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

/* ================================================================== */
/*  Internal buffer: simple malloc-backed                              */
/* ================================================================== */

struct CudaBuf {
    void*    ptr;
    uint64_t size;
    uint32_t refcount;
};

static uint32_t cuda_retain(nf_buffer self) {
    auto* b = reinterpret_cast<CudaBuf*>(self);
    return ++b->refcount;
}

static uint32_t cuda_release(nf_buffer self) {
    auto* b = reinterpret_cast<CudaBuf*>(self);
    if (--b->refcount == 0) {
        std::free(b->ptr);
        std::free(b);
        return 0;
    }
    return b->refcount;
}

static nf_status cuda_map(nf_buffer self, void** out) {
    auto* b = reinterpret_cast<CudaBuf*>(self);
    *out = b->ptr;
    return NF_OK;
}

static nf_status cuda_unmap(nf_buffer) { return NF_OK; }

static nf_status cuda_cache_sync(nf_buffer, nf_cache_op, uint64_t, uint64_t) {
    return NF_OK;
}

static nf_status cuda_query(nf_buffer self, nf_buffer_info* out) {
    if (!out) return NF_ERROR_INVALID_ARG;
    auto* b = reinterpret_cast<CudaBuf*>(self);
    std::memset(out, 0, sizeof(*out));
    out->desc.size_bytes = b->size;
    out->domain = NF_MEM_DOMAIN_CPU;
    return NF_OK;
}

static nf_buffer_ops g_cuda_buf_ops = {
    cuda_retain,
    cuda_release,
    cuda_map,
    cuda_unmap,
    cuda_cache_sync,
    cuda_query,
    nullptr,  /* export_handle */
    nullptr   /* import_handle */
};

/* ================================================================== */
/*  Provider vtable                                                    */
/* ================================================================== */

static const char* cuda_get_name(nf_provider) { return "cuda_cpu_fallback"; }
static uint32_t cuda_get_abi_version(nf_provider) { return NF_ABI_VERSION; }

static nf_status cuda_init(nf_provider) {
    std::fprintf(stderr, "[cuda_provider] init (CPU fallback mode)\n");
    return NF_OK;
}

static void cuda_shutdown(nf_provider) {
    std::fprintf(stderr, "[cuda_provider] shutdown\n");
}

static nf_status cuda_buffer_alloc(nf_provider, const nf_tensor_desc* desc, nf_buffer* out) {
    if (!desc || !out) return NF_ERROR_INVALID_ARG;
    auto* b = static_cast<CudaBuf*>(std::calloc(1, sizeof(CudaBuf)));
    if (!b) return NF_ERROR_OUT_OF_MEMORY;
    b->size = desc->size_bytes;
    b->ptr = std::calloc(1, desc->size_bytes > 0 ? desc->size_bytes : 1);
    b->refcount = 1;
    if (!b->ptr) { std::free(b); return NF_ERROR_OUT_OF_MEMORY; }
    *out = reinterpret_cast<nf_buffer>(b);
    return NF_OK;
}

static void cuda_buffer_free(nf_provider, nf_buffer buf) {
    if (buf) cuda_release(buf);
}

static nf_status cuda_buffer_map(nf_provider, nf_buffer buf, void** out) {
    return cuda_map(buf, out);
}

static nf_status cuda_buffer_unmap(nf_provider, nf_buffer) { return NF_OK; }

static nf_status cuda_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    if (!op_name) return NF_ERROR_INVALID_ARG;

    /* vector_add: out = in0 + in1 */
    if (std::strcmp(op_name, "vector_add") == 0 ||
        std::strcmp(op_name, "metal_vector_add") == 0) {
        if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
        auto* a = reinterpret_cast<CudaBuf*>(inputs[0]);
        auto* b = reinterpret_cast<CudaBuf*>(inputs[1]);
        auto* c = reinterpret_cast<CudaBuf*>(outputs[0]);
        uint32_t count = static_cast<uint32_t>(a->size / sizeof(float));
        const float* fa = static_cast<const float*>(a->ptr);
        const float* fb = static_cast<const float*>(b->ptr);
        float* fc = static_cast<float*>(c->ptr);
        for (uint32_t i = 0; i < count; ++i) fc[i] = fa[i] + fb[i];
        return NF_OK;
    }

    /* element_mul: out = in0 * in1 */
    if (std::strcmp(op_name, "element_mul") == 0) {
        if (n_in < 2 || n_out < 1) return NF_ERROR_INVALID_ARG;
        auto* a = reinterpret_cast<CudaBuf*>(inputs[0]);
        auto* b = reinterpret_cast<CudaBuf*>(inputs[1]);
        auto* c = reinterpret_cast<CudaBuf*>(outputs[0]);
        uint32_t count = static_cast<uint32_t>(a->size / sizeof(float));
        const float* fa = static_cast<const float*>(a->ptr);
        const float* fb = static_cast<const float*>(b->ptr);
        float* fc = static_cast<float*>(c->ptr);
        for (uint32_t i = 0; i < count; ++i) fc[i] = fa[i] * fb[i];
        return NF_OK;
    }

    /* linear: memcpy fallback */
    if (std::strcmp(op_name, "linear") == 0) {
        if (n_in < 1 || n_out < 1) return NF_ERROR_INVALID_ARG;
        auto* src = reinterpret_cast<CudaBuf*>(inputs[0]);
        auto* dst = reinterpret_cast<CudaBuf*>(outputs[0]);
        uint64_t sz = (src->size < dst->size) ? src->size : dst->size;
        std::memcpy(dst->ptr, src->ptr, sz);
        return NF_OK;
    }

    return NF_OK;
}

static nf_status cuda_synchronize(nf_provider) { return NF_OK; }

/* ================================================================== */
/*  Mem vtable                                                         */
/* ================================================================== */

static nf_status cuda_mem_alloc(nf_provider self,
                                const nf_buffer_alloc_request* req,
                                nf_buffer_ops* ops,
                                nf_buffer* buf) {
    if (!req || !ops || !buf) return NF_ERROR_INVALID_ARG;
    nf_status st = cuda_buffer_alloc(self, &req->desc, buf);
    if (st != NF_OK) return st;
    *ops = g_cuda_buf_ops;
    return NF_OK;
}

static nf_status cuda_mem_import(nf_provider, uint64_t, nf_mem_domain,
                                 const nf_tensor_desc*, nf_buffer_ops*, nf_buffer*) {
    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status cuda_mem_can_import(nf_provider, nf_mem_domain) {
    return NF_ERROR_UNSUPPORTED_OP;
}

/* ================================================================== */
/*  Plugin entry points                                                */
/* ================================================================== */

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov) {
    if (!vt || !prov) return NF_ERROR_INVALID_ARG;
    *prov = nullptr;
    vt->get_name        = cuda_get_name;
    vt->get_abi_version = cuda_get_abi_version;
    vt->init            = cuda_init;
    vt->shutdown        = cuda_shutdown;
    vt->buffer_alloc    = cuda_buffer_alloc;
    vt->buffer_free     = cuda_buffer_free;
    vt->buffer_map      = cuda_buffer_map;
    vt->buffer_unmap    = cuda_buffer_unmap;
    vt->dispatch        = cuda_dispatch;
    vt->synchronize     = cuda_synchronize;
    return NF_OK;
}

extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov) {
    if (!vt || !prov) return NF_ERROR_INVALID_ARG;
    *prov = nullptr;
    vt->alloc         = cuda_mem_alloc;
    vt->import_buffer = cuda_mem_import;
    vt->can_import    = cuda_mem_can_import;
    return NF_OK;
}
