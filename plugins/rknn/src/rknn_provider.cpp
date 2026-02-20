/**
 * @file rknn_provider.cpp
 * @brief Rockchip RKNN execution provider — plugin entry point stub.
 */

#include "neurofabric/neuro_fabric_abi.h"

struct nf_provider_rknn {
    // Phase 1 stub — will hold rknn_context, DMA-BUF fd pool, etc.
};

static nf_provider_rknn s_instance;

static const char* rknn_get_name(nf_provider) { return "rockchip_rknn"; }
static uint32_t    rknn_get_abi_version(nf_provider) { return NF_ABI_VERSION; }
static nf_status   rknn_init(nf_provider) { return NF_OK; }
static void        rknn_shutdown(nf_provider) {}

static nf_status rknn_buffer_alloc(nf_provider, const nf_tensor_desc*, nf_buffer*) {
    return NF_ERROR_UNSUPPORTED_OP;
}
static void      rknn_buffer_free(nf_provider, nf_buffer) {}
static nf_status rknn_buffer_map(nf_provider, nf_buffer, void**) {
    return NF_ERROR_UNSUPPORTED_OP;
}
static nf_status rknn_buffer_unmap(nf_provider, nf_buffer) {
    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status rknn_dispatch(nf_provider, const char*,
                               const nf_buffer*, uint32_t,
                               nf_buffer*, uint32_t) {
    return NF_ERROR_UNSUPPORTED_OP;
}
static nf_status rknn_synchronize(nf_provider) { return NF_OK; }

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
