/**
 * @file metal_provider.cpp
 * @brief Apple Metal execution provider — plugin entry point stub.
 */

#include "neurofabric/neuro_fabric_abi.h"

struct nf_provider_metal {
    // Phase 1 stub — will hold MTLDevice*, command queue, etc.
};

static nf_provider_metal s_instance;

static const char* metal_get_name(nf_provider) { return "apple_metal"; }
static uint32_t    metal_get_abi_version(nf_provider) { return NF_ABI_VERSION; }
static nf_status   metal_init(nf_provider) { return NF_OK; }
static void        metal_shutdown(nf_provider) {}

static nf_status metal_buffer_alloc(nf_provider, const nf_tensor_desc*, nf_buffer*) {
    return NF_ERROR_UNSUPPORTED_OP; // Phase 1 stub
}
static void      metal_buffer_free(nf_provider, nf_buffer) {}
static nf_status metal_buffer_map(nf_provider, nf_buffer, void**) {
    return NF_ERROR_UNSUPPORTED_OP;
}
static nf_status metal_buffer_unmap(nf_provider, nf_buffer) {
    return NF_ERROR_UNSUPPORTED_OP;
}

static nf_status metal_dispatch(nf_provider, const char*,
                                const nf_buffer*, uint32_t,
                                nf_buffer*, uint32_t) {
    return NF_ERROR_UNSUPPORTED_OP;
}
static nf_status metal_synchronize(nf_provider) { return NF_OK; }

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
