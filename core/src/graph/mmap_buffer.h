/**
 * @file mmap_buffer.h
 * @brief Internal header — MMap buffer backend for read-only weights
 */

#ifndef NF_MMAP_BUFFER_H
#define NF_MMAP_BUFFER_H

#include "neurofabric/neuro_buffer_abi.h"
#include <atomic>
#include <cstddef>

namespace nf {

struct MmapBuffer {
    std::atomic<uint32_t> refcount{1};
    void*                 mmap_base   = nullptr;  // actual mmap'd VA
    size_t                mmap_len    = 0;        // actual mmap region size
    void*                 data_ptr    = nullptr;  // mmap_base + page adjustment
    nf_tensor_desc        desc{};
    bool                  is_mapped   = false;    // map/unmap tracking
};

/**
 * Create an MmapBuffer from an fd + offset + size.
 * Handles page-alignment internally: rounds offset down to page boundary,
 * maps a larger region, and adjusts the data pointer.
 */
nf_status mmap_buffer_create(int fd, uint64_t offset, uint64_t size,
                              const nf_tensor_desc& desc,
                              nf_buffer_ops* out_ops, nf_buffer* out_buf);

} // namespace nf

#endif // NF_MMAP_BUFFER_H
