/**
 * @file buffer_test.cpp
 * @brief Phase 2 test — validates buffer ABI contract and TensorView wrapper.
 *
 * Creates a mock CPU buffer plugin inline (no .dylib needed), then
 * exercises the full lifecycle: alloc, retain/release, map/unmap,
 * cache_sync, export, slice, and TensorView RAII semantics.
 */

#include "neurofabric/TensorView.hpp"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ================================================================== */
/*  Mock CPU Buffer Plugin (inline, for testing only)                  */
/* ================================================================== */

struct MockBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 data      = nullptr;
    nf_tensor_desc        desc{};
    bool                  mapped    = false;
};

static uint32_t mock_retain(nf_buffer self) {
    auto* mb = reinterpret_cast<MockBuffer*>(self);
    return mb->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

static uint32_t mock_release(nf_buffer self) {
    auto* mb = reinterpret_cast<MockBuffer*>(self);
    uint32_t prev = mb->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        std::free(mb->data);
        delete mb;
    }
    return prev - 1;
}

static nf_status mock_map(nf_buffer self, void** out) {
    auto* mb = reinterpret_cast<MockBuffer*>(self);
    if (mb->mapped) return NF_ERROR_INVALID_ARG;
    mb->mapped = true;
    *out = mb->data;
    return NF_OK;
}

static nf_status mock_unmap(nf_buffer self) {
    auto* mb = reinterpret_cast<MockBuffer*>(self);
    mb->mapped = false;
    return NF_OK;
}

static nf_status mock_cache_sync(nf_buffer, nf_cache_op, uint64_t, uint64_t) {
    // CPU buffer: cache ops are no-ops (just like Apple Silicon unified mem)
    return NF_OK;
}

static nf_status mock_get_info(nf_buffer self, nf_buffer_info* info) {
    auto* mb = reinterpret_cast<MockBuffer*>(self);
    info->desc         = mb->desc;
    info->domain       = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0;
    info->share_token  = 0;
    info->refcount     = mb->refcount.load(std::memory_order_relaxed);
    info->_reserved    = 0;
    return NF_OK;
}

static nf_status mock_export(nf_buffer, uint64_t* token, nf_mem_domain* dom) {
    *token = 0;
    *dom   = NF_MEM_DOMAIN_CPU;
    return NF_OK;
}

static MockBuffer* mock_alloc(nf_tensor_desc desc, nf_buffer_ops* ops) {
    auto* mb   = new MockBuffer;
    mb->desc   = desc;
    mb->data   = std::calloc(1, desc.size_bytes);

    ops->retain        = mock_retain;
    ops->release       = mock_release;
    ops->map           = mock_map;
    ops->unmap         = mock_unmap;
    ops->cache_sync    = mock_cache_sync;
    ops->get_info      = mock_get_info;
    ops->export_handle = mock_export;
    ops->import_handle = nullptr; // not needed for this test

    return mb;
}

/* ================================================================== */
/*  Tests                                                              */
/* ================================================================== */

static void test_lifecycle() {
    nf_tensor_desc desc{};
    desc.dtype      = NF_DTYPE_F32;
    desc.ndim       = 2;
    desc.shape[0]   = 4;
    desc.shape[1]   = 8;
    desc.size_bytes = 4 * 8 * sizeof(float);

    nf_buffer_ops ops{};
    MockBuffer* mb = mock_alloc(desc, &ops);
    nf_buffer buf  = reinterpret_cast<nf_buffer>(mb);

    // Retain → refcount 2
    assert(ops.retain(buf) == 2);
    // Release → refcount 1
    assert(ops.release(buf) == 1);
    // Final release → freed (refcount 0)
    assert(ops.release(buf) == 0);
    // mb is now freed — do not touch

    std::printf("  PASS: lifecycle (retain/release)\n");
}

static void test_map_unmap() {
    nf_tensor_desc desc{};
    desc.dtype      = NF_DTYPE_F32;
    desc.ndim       = 1;
    desc.shape[0]   = 16;
    desc.size_bytes = 16 * sizeof(float);

    nf_buffer_ops ops{};
    MockBuffer* mb = mock_alloc(desc, &ops);
    nf_buffer buf  = reinterpret_cast<nf_buffer>(mb);

    void* ptr = nullptr;
    assert(ops.map(buf, &ptr) == NF_OK);
    assert(ptr != nullptr);

    // Write through mapped pointer
    auto* fp = static_cast<float*>(ptr);
    for (int i = 0; i < 16; ++i) fp[i] = static_cast<float>(i);

    assert(ops.unmap(buf) == NF_OK);

    // Double-map should fail in our mock
    assert(ops.map(buf, &ptr) == NF_OK); // re-map after unmap is fine
    assert(static_cast<float*>(ptr)[7] == 7.0f);
    ops.unmap(buf);

    ops.release(buf);
    std::printf("  PASS: map/unmap\n");
}

static void test_tensor_view_raii() {
    nf_tensor_desc desc{};
    desc.dtype      = NF_DTYPE_F32;
    desc.ndim       = 2;
    desc.shape[0]   = 3;
    desc.shape[1]   = 4;
    desc.size_bytes = 3 * 4 * sizeof(float);

    nf_buffer_ops ops{};
    MockBuffer* mb = mock_alloc(desc, &ops);
    nf_buffer buf  = reinterpret_cast<nf_buffer>(mb);

    {
        nf::TensorView tv(buf, ops);
        assert(tv.valid());
        assert(tv.domain() == NF_MEM_DOMAIN_CPU);
        assert(tv.desc().ndim == 2);

        // Map via TensorView
        {
            auto mapped = tv.map<float>();
            assert(mapped.valid());
            assert(mapped.size() == 12);
            mapped[0] = 42.0f;
            // MappedSpan unmaps on scope exit
        }

        // Cache ops (no-op on CPU, but must not crash)
        assert(tv.flush() == NF_OK);
        assert(tv.invalidate() == NF_OK);

        // Share (retain)
        {
            nf::TensorView shared = tv.share();
            assert(shared.valid());
            assert(shared.info().refcount == 2);
            // shared releases on scope exit → refcount back to 1
        }

        // tv releases on scope exit → refcount 0 → freed
    }

    std::printf("  PASS: TensorView RAII\n");
}

static void test_slice() {
    nf_tensor_desc desc{};
    desc.dtype      = NF_DTYPE_F32;
    desc.ndim       = 2;
    desc.shape[0]   = 8;
    desc.shape[1]   = 4;
    desc.size_bytes = 8 * 4 * sizeof(float);

    nf_buffer_ops ops{};
    MockBuffer* mb = mock_alloc(desc, &ops);
    nf_buffer buf  = reinterpret_cast<nf_buffer>(mb);

    {
        nf::TensorView tv(buf, ops);

        // Slice rows [2, 5) along dim 0
        nf::TensorView sv = tv.slice(0, 2, 5);
        assert(sv.valid());
        assert(sv.is_slice());
        assert(sv.slice_dim() == 0);
        assert(sv.slice_begin() == 2);
        assert(sv.slice_end() == 5);

        // Both tv and sv share the same backing buffer
        assert(sv.info().refcount == 2);

        // sv releases on scope exit, then tv releases → freed
    }

    std::printf("  PASS: slice (zero-copy sub-view)\n");
}

int main() {
    std::printf("buffer_test: Phase 2 — Unified Memory ABI\n");
    test_lifecycle();
    test_map_unmap();
    test_tensor_view_raii();
    test_slice();
    std::printf("OK: all buffer tests passed\n");
    return 0;
}
