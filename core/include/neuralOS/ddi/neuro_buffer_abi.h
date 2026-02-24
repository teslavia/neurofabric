/**
 * @file neuro_buffer_abi.h
 * @brief Neuro-Fabric Unified Memory ABI — Zero-Copy Heterogeneous Buffers
 *
 * Extends the Phase 1 ABI with a complete memory management contract.
 * Design choice: Opaque Handle Pattern (方案 A) — the ONLY pattern that
 * simultaneously satisfies:
 *   1. Zero-vptr hourglass ABI stability
 *   2. Hardware-agnostic core (no Metal/DRM headers in core)
 *   3. Zero-copy tensor sharing across providers
 *
 * Memory model:
 *   Plugin allocates concrete backing (Metal MTLBuffer, DMA-BUF fd, malloc).
 *   Plugin wraps it behind nf_buffer (opaque handle) + nf_buffer_ops (C vtable).
 *   Core routes tensors between providers using ONLY these ops.
 *   Cache coherency is explicit — ARM DMA requires manual flush/invalidate.
 *
 * Includes neuro_fabric_abi.h for base types (nf_status, nf_tensor_desc, etc.)
 */

#ifndef NEUROFABRIC_BUFFER_ABI_H
#define NEUROFABRIC_BUFFER_ABI_H

#include "neuro_fabric_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  1. Memory Domain Tags                                              */
/*     Tells the core WHERE a buffer physically lives, enabling        */
/*     intelligent routing decisions without exposing hardware types.   */
/* ------------------------------------------------------------------ */

typedef enum nf_mem_domain {
    NF_MEM_DOMAIN_CPU         = 0,  /**< Plain host malloc / mmap          */
    NF_MEM_DOMAIN_UNIFIED     = 1,  /**< CPU+GPU coherent (Apple Silicon)  */
    NF_MEM_DOMAIN_DEVICE      = 2,  /**< Device-only (discrete GPU VRAM)   */
    NF_MEM_DOMAIN_DMA_BUF     = 3,  /**< Linux DMA-BUF fd (RK3588 NPU)    */
    NF_MEM_DOMAIN_EXTERNAL    = 4,  /**< Foreign import (Vulkan, EGL, etc) */
    NF_MEM_DOMAIN_MMAP        = 5,  /**< mmap'd read-only weight file      */
    NF_MEM_DOMAIN_CXL         = 6,  /**< Phase 38: CXL 3.0 GFAM           */
    NF_MEM_DOMAIN_CXL_SHARED  = 7   /**< Phase 38: CXL shared memory pool  */
} nf_mem_domain;

/* ------------------------------------------------------------------ */
/*  2. Cache Coherency Flags                                           */
/*     On ARM (RK3588 Cortex-A76), DMA-BUF transfers between CPU      */
/*     and NPU require EXPLICIT cache maintenance. Apple Silicon's     */
/*     unified memory is hardware-coherent, so these become no-ops     */
/*     in the Metal plugin — but the contract must exist for ARM.      */
/* ------------------------------------------------------------------ */

typedef enum nf_cache_op {
    /** Flush dirty CPU cache lines to main memory (before device read). */
    NF_CACHE_FLUSH            = 0x01,
    /** Invalidate CPU cache lines (before CPU read after device write). */
    NF_CACHE_INVALIDATE       = 0x02,
    /** Flush + Invalidate (full round-trip coherency). */
    NF_CACHE_FLUSH_INVALIDATE = 0x03
} nf_cache_op;

/* ------------------------------------------------------------------ */
/*  3. Buffer Info — POD descriptor returned by query                  */
/*     Safe to memcpy. Carries enough metadata for the core to make    */
/*     zero-copy routing decisions.                                    */
/* ------------------------------------------------------------------ */

typedef struct nf_buffer_info {
    nf_tensor_desc  desc;           /**< Shape, dtype, strides, size      */
    nf_mem_domain   domain;         /**< Where the backing store lives    */
    uint64_t        offset_bytes;   /**< Offset into backing allocation   */

    /**
     * Platform-specific shareable handle, encoded as uint64_t:
     *   - DMA-BUF:  the fd (file descriptor), cast to uint64_t
     *   - Metal:    MTLBuffer GPU address or IOSurface ID
     *   - CPU:      0 (not shareable across processes)
     *
     * This allows the core to pass the token to another plugin for
     * zero-copy import WITHOUT including any platform header.
     */
    uint64_t        share_token;

    /** Reference count snapshot (informational, not authoritative). */
    uint32_t        refcount;

    /** Padding for future fields without breaking ABI. */
    uint32_t        _reserved;
} nf_buffer_info;

#ifdef __cplusplus
static_assert(sizeof(nf_buffer_info) == 176,
    "ABI break: nf_buffer_info size changed");
#else
_Static_assert(sizeof(nf_buffer_info) == 176,
    "ABI break: nf_buffer_info size changed");
#endif

/* ------------------------------------------------------------------ */
/*  4. Buffer Operations VTable (Struct of Function Pointers)          */
/*     The plugin that OWNS a buffer fills this table at alloc time.   */
/*     Core (and other plugins doing zero-copy import) invoke the      */
/*     buffer ONLY through these pointers. Same hourglass discipline   */
/*     as nf_provider_vtable.                                          */
/* ------------------------------------------------------------------ */

typedef struct nf_buffer_ops {

    /* -- Lifecycle -------------------------------------------------- */

    /**
     * Increment reference count. Thread-safe (atomic).
     * Returns new refcount.
     */
    uint32_t  (*retain)(nf_buffer self);

    /**
     * Decrement reference count. When it hits zero, the plugin frees
     * the backing store. Thread-safe (atomic).
     * Returns new refcount (0 means freed).
     */
    uint32_t  (*release)(nf_buffer self);

    /* -- Mapping ---------------------------------------------------- */

    /**
     * Map buffer for CPU access. Returns a CPU-visible pointer.
     *   - Unified memory (Apple Silicon): zero-cost, returns GPU ptr directly.
     *   - DMA-BUF (RK3588): mmap() the fd, may trigger cache invalidate.
     *   - Discrete GPU: staging buffer copy.
     *
     * Caller MUST call unmap() when done.
     */
    nf_status (*map)(nf_buffer self, void** out_ptr);

    /** Unmap a previously mapped buffer. */
    nf_status (*unmap)(nf_buffer self);

    /* -- Cache Coherency -------------------------------------------- */

    /**
     * Explicit cache maintenance — the ARM DMA-BUF lifeline.
     *
     * @param op     Bitmask of nf_cache_op flags.
     * @param offset Byte offset into the buffer (0 for full range).
     * @param size   Byte length to operate on (0 = entire buffer).
     *
     * On Apple Silicon this is a no-op (hardware coherent).
     * On RK3588 this maps to DMA_BUF_IOCTL_SYNC.
     */
    nf_status (*cache_sync)(nf_buffer self,
                            nf_cache_op op,
                            uint64_t offset,
                            uint64_t size);

    /* -- Query ------------------------------------------------------ */

    /** Fill `info` with the buffer's metadata. */
    nf_status (*get_info)(nf_buffer self, nf_buffer_info* info);

    /* -- Zero-Copy Sharing ------------------------------------------ */

    /**
     * Export a shareable token for cross-plugin zero-copy import.
     *
     * The token semantics are domain-specific:
     *   - NF_MEM_DOMAIN_DMA_BUF:  returns the fd (dup'd; caller owns it)
     *   - NF_MEM_DOMAIN_UNIFIED:  returns IOSurface ID or GPU VA
     *   - NF_MEM_DOMAIN_DEVICE:   returns driver-specific export handle
     *
     * @param[out] token  The shareable handle.
     * @param[out] domain The memory domain of the export.
     */
    nf_status (*export_handle)(nf_buffer self,
                               uint64_t* token,
                               nf_mem_domain* domain);

    /**
     * Import a foreign buffer from a shareable token.
     * This is called on the IMPORTING plugin's buffer_ops, not the exporter's.
     *
     * @param token   The token obtained from export_handle.
     * @param domain  The memory domain of the token.
     * @param desc    Tensor descriptor (shape, dtype) for the import.
     * @param[out] out  Newly created buffer wrapping the imported memory.
     */
    nf_status (*import_handle)(nf_provider provider,
                               uint64_t token,
                               nf_mem_domain domain,
                               const nf_tensor_desc* desc,
                               nf_buffer* out);

} nf_buffer_ops;

/* ------------------------------------------------------------------ */
/*  5. Buffer Allocation Request (POD)                                 */
/*     Passed from core to plugin when requesting a new buffer.        */
/*     The plugin decides the actual backing domain.                   */
/* ------------------------------------------------------------------ */

typedef struct nf_buffer_alloc_request {
    nf_tensor_desc  desc;           /**< Desired shape and dtype          */
    nf_mem_domain   preferred;      /**< Hint: preferred memory domain    */
    uint32_t        flags;          /**< Reserved for alignment hints etc */
    uint32_t        _reserved;
} nf_buffer_alloc_request;

#ifdef __cplusplus
static_assert(sizeof(nf_buffer_alloc_request) == 160,
    "ABI break: nf_buffer_alloc_request size changed");
#else
_Static_assert(sizeof(nf_buffer_alloc_request) == 160,
    "ABI break: nf_buffer_alloc_request size changed");
#endif

/** Usage flag: buffer wraps mmap'd read-only weight data. */
#define NF_BUFFER_USAGE_WEIGHT_MMAP  0x01u

/* ------------------------------------------------------------------ */
/*  6. Extended Provider VTable for Memory                             */
/*     Plugins that support the buffer protocol fill this struct       */
/*     IN ADDITION to nf_provider_vtable. Queried by the core after   */
/*     successful nf_plugin_register via a capability probe.           */
/* ------------------------------------------------------------------ */

typedef struct nf_provider_mem_vtable {

    /**
     * Allocate a buffer on this provider's device/memory domain.
     * The plugin creates the backing store, fills `ops`, and returns
     * the opaque buffer handle.
     *
     * @param[in]  req   Allocation parameters.
     * @param[out] ops   Plugin fills this with its buffer function pointers.
     * @param[out] buf   Opaque handle to the new buffer.
     */
    nf_status (*alloc)(nf_provider self,
                       const nf_buffer_alloc_request* req,
                       nf_buffer_ops* ops,
                       nf_buffer* buf);

    /**
     * Import a foreign buffer via shareable token (zero-copy path).
     * See nf_buffer_ops::import_handle for semantics.
     */
    nf_status (*import_buffer)(nf_provider self,
                               uint64_t token,
                               nf_mem_domain domain,
                               const nf_tensor_desc* desc,
                               nf_buffer_ops* ops,
                               nf_buffer* buf);

    /**
     * Query whether this provider can zero-copy import from `domain`.
     * Returns NF_OK if yes, NF_ERROR_UNSUPPORTED_OP if no.
     */
    nf_status (*can_import)(nf_provider self, nf_mem_domain domain);

} nf_provider_mem_vtable;

/* ------------------------------------------------------------------ */
/*  7. Extended Plugin Entry Point                                     */
/*     Optional: plugins that support memory ops export this symbol    */
/*     IN ADDITION to nf_plugin_register.                              */
/* ------------------------------------------------------------------ */

typedef nf_status (*nf_plugin_register_mem_fn)(nf_provider_mem_vtable* mem_vt);

#define NF_PLUGIN_MEM_ENTRY_SYMBOL "nf_plugin_register_mem"

#ifdef __cplusplus
}
#endif

#endif /* NEUROFABRIC_BUFFER_ABI_H */
