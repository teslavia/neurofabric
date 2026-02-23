/**
 * @file nf_c_api.cpp
 * @brief C API implementation — Session wrapper, mock provider, FFI gateway
 */

#include "neurofabric/abi/nf_c_api.h"
#include "neurofabric/engine/PipelineEngine.hpp"
#include "neurofabric/engine/GraphBuilder.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

/* ================================================================== */
/*  Host Buffer — malloc-backed activation allocator (same as tests)   */
/* ================================================================== */

namespace {

struct HostBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data = nullptr;
    nf_tensor_desc        desc{};
    bool                  mapped = false;
};

static uint32_t host_retain(nf_buffer self) {
    return reinterpret_cast<HostBuffer*>(self)->refcount.fetch_add(
        1, std::memory_order_relaxed) + 1;
}
static uint32_t host_release(nf_buffer self) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) { std::free(b->data); delete b; }
    return prev - 1;
}
static nf_status host_map(nf_buffer self, void** p) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    b->mapped = true; *p = b->data; return NF_OK;
}
static nf_status host_unmap(nf_buffer self) {
    reinterpret_cast<HostBuffer*>(self)->mapped = false; return NF_OK;
}
static nf_status host_cache(nf_buffer, nf_cache_op, uint64_t, uint64_t) {
    return NF_OK;
}
static nf_status host_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<HostBuffer*>(self);
    info->desc = b->desc; info->domain = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0; info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0; return NF_OK;
}
static nf_status host_export_h(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_CPU; return NF_OK;
}

static nf_buffer_ops make_host_ops() {
    nf_buffer_ops ops{};
    ops.retain = host_retain; ops.release = host_release;
    ops.map = host_map; ops.unmap = host_unmap;
    ops.cache_sync = host_cache; ops.get_info = host_info;
    ops.export_handle = host_export_h; ops.import_handle = nullptr;
    return ops;
}

static nf_status host_alloc_fn(const nf_tensor_desc& desc,
                                nf_buffer_ops* ops, nf_buffer* buf) {
    auto* b = new HostBuffer;
    b->desc = desc;
    b->data = std::calloc(1, desc.size_bytes);
    if (!b->data) { delete b; return NF_ERROR_OUT_OF_MEMORY; }
    *ops = make_host_ops();
    *buf = reinterpret_cast<nf_buffer>(b);
    return NF_OK;
}

/* ================================================================== */
/*  Mock Provider — dispatches all ops as no-op (passthrough)          */
/* ================================================================== */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer* inputs, uint32_t n_in,
                               nf_buffer* outputs, uint32_t n_out) {
    /* Copy first input to first output if sizes match, else zero-fill */
    if (n_in > 0 && n_out > 0 && inputs[0] && outputs[0]) {
        auto* ib = reinterpret_cast<HostBuffer*>(inputs[0]);
        auto* ob = reinterpret_cast<HostBuffer*>(outputs[0]);
        uint64_t copy_sz = std::min(ib->desc.size_bytes, ob->desc.size_bytes);
        if (ib->data && ob->data)
            std::memcpy(ob->data, ib->data, copy_sz);
    }
    return NF_OK;
}

static const char* mock_name(nf_provider) { return "c_api_mock"; }
static uint32_t mock_abi(nf_provider) { return NF_ABI_VERSION; }
static nf_status mock_init(nf_provider) { return NF_OK; }
static void mock_shutdown(nf_provider) {}
static nf_status mock_sync(nf_provider) { return NF_OK; }

static nf_provider_vtable make_mock_vt() {
    nf_provider_vtable vt{};
    vt.get_name = mock_name; vt.get_abi_version = mock_abi;
    vt.init = mock_init; vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch; vt.synchronize = mock_sync;
    return vt;
}

} // anonymous namespace

/* ================================================================== */
/*  Opaque Handle Structs                                              */
/* ================================================================== */

struct nf_engine_s {
    nf::PipelineEngine engine;
    int mock_prov_tag = 0;
};

struct nf_session_s {
    nf_engine_s*                        eng = nullptr;
    nf::GraphBuilder*                   builder = nullptr;
    nf::PipelineEngine::Session*        session = nullptr;
    uint32_t                            graph_id = 0;
    uint32_t                            num_tensors = 0;
    double                              last_step_us = 0.0;
};

/* ================================================================== */
/*  C API Implementation                                               */
/* ================================================================== */

extern "C" {

NF_C_API_EXPORT nf_engine_t nf_create_engine(uint32_t n_threads) {
    auto* e = new (std::nothrow) nf_engine_s{
        nf::PipelineEngine(n_threads == 0
            ? std::thread::hardware_concurrency() : n_threads)
    };
    if (!e) return nullptr;

    /* Register built-in mock provider */
    auto vt = make_mock_vt();
    e->engine.register_provider(
        reinterpret_cast<nf_provider>(&e->mock_prov_tag),
        vt, NF_AFFINITY_ANY);
    return e;
}

NF_C_API_EXPORT void nf_destroy_engine(nf_engine_t engine) {
    delete engine;
}

NF_C_API_EXPORT nf_session_t nf_create_session(nf_engine_t engine,
                                                const char* nfir_path) {
    if (!engine || !nfir_path) return nullptr;

    auto* s = new (std::nothrow) nf_session_s;
    if (!s) return nullptr;
    s->eng = engine;

    s->builder = new (std::nothrow) nf::GraphBuilder(engine->engine,
                                                      host_alloc_fn);
    if (!s->builder) { delete s; return nullptr; }

    if (s->builder->load(nfir_path) != NF_OK) {
        delete s->builder; delete s; return nullptr;
    }

    uint32_t gid = 0;
    if (s->builder->build(&gid) != NF_OK) {
        delete s->builder; delete s; return nullptr;
    }
    s->graph_id = gid;

    /* Count tensors by probing get_tensor_buffer */
    uint32_t count = 0;
    while (s->builder->get_tensor_buffer(count) != nullptr) ++count;
    s->num_tensors = count;

    /* Pre-compile execution plan */
    s->session = new (std::nothrow) nf::PipelineEngine::Session(
        engine->engine, gid);
    if (!s->session || !s->session->valid()) {
        delete s->session; delete s->builder; delete s; return nullptr;
    }

    return s;
}

NF_C_API_EXPORT void nf_destroy_session(nf_session_t session) {
    if (!session) return;
    delete session->session;
    delete session->builder;
    delete session;
}

NF_C_API_EXPORT int nf_session_set_input(nf_session_t s, uint32_t tensor_id,
                                          const void* data, uint64_t size) {
    if (!s || !data) return NF_ERROR_INVALID_ARG;

    nf_buffer buf = s->builder->get_tensor_buffer(tensor_id);
    nf_buffer_ops ops = s->builder->get_tensor_ops(tensor_id);
    if (!buf || !ops.map || !ops.unmap) return NF_ERROR_NOT_FOUND;

    void* ptr = nullptr;
    nf_status st = ops.map(buf, &ptr);
    if (st != NF_OK) return st;

    nf_buffer_info info{};
    if (ops.get_info) ops.get_info(buf, &info);
    uint64_t copy_sz = std::min(size, info.desc.size_bytes);
    std::memcpy(ptr, data, copy_sz);

    ops.unmap(buf);
    return NF_OK;
}

NF_C_API_EXPORT int nf_session_step(nf_session_t s) {
    if (!s || !s->session) return NF_ERROR_INVALID_ARG;

    auto t0 = std::chrono::steady_clock::now();
    auto fut = s->session->step();
    nf_status st = fut.get();
    auto t1 = std::chrono::steady_clock::now();

    s->last_step_us = std::chrono::duration<double, std::micro>(
        t1 - t0).count();
    return st;
}

NF_C_API_EXPORT int nf_session_get_output(nf_session_t s, uint32_t tensor_id,
                                           void* data, uint64_t size) {
    if (!s || !data) return NF_ERROR_INVALID_ARG;

    nf_buffer buf = s->builder->get_tensor_buffer(tensor_id);
    nf_buffer_ops ops = s->builder->get_tensor_ops(tensor_id);
    if (!buf || !ops.map || !ops.unmap) return NF_ERROR_NOT_FOUND;

    void* ptr = nullptr;
    nf_status st = ops.map(buf, &ptr);
    if (st != NF_OK) return st;

    nf_buffer_info info{};
    if (ops.get_info) ops.get_info(buf, &info);
    uint64_t copy_sz = std::min(size, info.desc.size_bytes);
    std::memcpy(data, ptr, copy_sz);

    ops.unmap(buf);
    return NF_OK;
}

NF_C_API_EXPORT double nf_session_get_last_step_us(nf_session_t s) {
    return s ? s->last_step_us : 0.0;
}

NF_C_API_EXPORT uint32_t nf_session_num_tensors(nf_session_t s) {
    return s ? s->num_tensors : 0;
}

NF_C_API_EXPORT uint32_t nf_session_num_nodes(nf_session_t s) {
    return (s && s->session) ?
        static_cast<uint32_t>(s->session->num_nodes()) : 0;
}

NF_C_API_EXPORT int nf_session_set_push_constants(nf_session_t s,
                                                    const char* node_name,
                                                    const void* data,
                                                    uint32_t size) {
    if (!s || !s->session || !node_name || !data)
        return NF_ERROR_INVALID_ARG;
    return s->session->set_push_constants(node_name, data, size);
}

} // extern "C"
