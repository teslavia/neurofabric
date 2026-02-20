/**
 * @file mmap_buffer.cpp
 * @brief MMap buffer backend — nf_buffer_ops vtable for mmap'd weight files
 *
 * Weights stay mmap'd for the lifetime of the buffer. The OS pages them
 * in on demand, avoiding the malloc+read double-memory-pressure pattern.
 */

#include "mmap_buffer.h"

#include <sys/mman.h>
#include <unistd.h>
#include <new>

namespace nf {

/* ------------------------------------------------------------------ */
/*  nf_buffer_ops vtable functions                                     */
/* ------------------------------------------------------------------ */

static uint32_t mmap_retain(nf_buffer self) {
    auto* b = reinterpret_cast<MmapBuffer*>(self);
    return b->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

static uint32_t mmap_release(nf_buffer self) {
    auto* b = reinterpret_cast<MmapBuffer*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        if (b->mmap_base && b->mmap_base != MAP_FAILED) {
            ::munmap(b->mmap_base, b->mmap_len);
        }
        delete b;
    }
    return prev - 1;
}

static nf_status mmap_map(nf_buffer self, void** out_ptr) {
    auto* b = reinterpret_cast<MmapBuffer*>(self);
    b->is_mapped = true;
    *out_ptr = b->data_ptr;
    return NF_OK;
}

static nf_status mmap_unmap(nf_buffer self) {
    reinterpret_cast<MmapBuffer*>(self)->is_mapped = false;
    return NF_OK;
}

/* Cache sync is a no-op — OS manages page cache for mmap'd files. */
static nf_status mmap_cache_sync(nf_buffer, nf_cache_op, uint64_t, uint64_t) {
    return NF_OK;
}

static nf_status mmap_get_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<MmapBuffer*>(self);
    info->desc         = b->desc;
    info->domain       = NF_MEM_DOMAIN_MMAP;
    info->offset_bytes = 0;
    info->share_token  = 0;
    info->refcount     = b->refcount.load(std::memory_order_relaxed);
    info->_reserved    = 0;
    return NF_OK;
}

static nf_status mmap_export_handle(nf_buffer, uint64_t* token, nf_mem_domain* domain) {
    *token  = 0;  /* Not shareable cross-process */
    *domain = NF_MEM_DOMAIN_MMAP;
    return NF_OK;
}

static nf_buffer_ops make_mmap_ops() {
    nf_buffer_ops ops{};
    ops.retain        = mmap_retain;
    ops.release       = mmap_release;
    ops.map           = mmap_map;
    ops.unmap         = mmap_unmap;
    ops.cache_sync    = mmap_cache_sync;
    ops.get_info      = mmap_get_info;
    ops.export_handle = mmap_export_handle;
    ops.import_handle = nullptr;
    return ops;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

nf_status mmap_buffer_create(int fd, uint64_t offset, uint64_t size,
                              const nf_tensor_desc& desc,
                              nf_buffer_ops* out_ops, nf_buffer* out_buf) {
    /* Align offset down to system page boundary */
    long page_size = ::sysconf(_SC_PAGESIZE);
    uint64_t page_mask = static_cast<uint64_t>(page_size) - 1;
    uint64_t aligned_offset = offset & ~page_mask;
    uint64_t adjust = offset - aligned_offset;
    uint64_t map_size = size + adjust;

    void* base = ::mmap(nullptr, static_cast<size_t>(map_size),
                        PROT_READ, MAP_SHARED,
                        fd, static_cast<off_t>(aligned_offset));
    if (base == MAP_FAILED) {
        return NF_ERROR_OUT_OF_MEMORY;
    }

    auto* b = new (std::nothrow) MmapBuffer;
    if (!b) {
        ::munmap(base, static_cast<size_t>(map_size));
        return NF_ERROR_OUT_OF_MEMORY;
    }

    b->mmap_base = base;
    b->mmap_len  = static_cast<size_t>(map_size);
    b->data_ptr  = static_cast<uint8_t*>(base) + adjust;
    b->desc      = desc;
    b->is_mapped = false;

    *out_ops = make_mmap_ops();
    *out_buf = reinterpret_cast<nf_buffer>(b);
    return NF_OK;
}

} // namespace nf
