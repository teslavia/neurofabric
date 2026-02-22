/**
 * @file neuro_fabric_abi.h
 * @brief Neuro-Fabric Core ABI Contract — The Hourglass Waist
 *
 * This header defines the ONLY legal interface for crossing dynamic library
 * boundaries in the Neuro-Fabric system. Every symbol here is:
 *   - Pure C linkage (extern "C")
 *   - POD or opaque-handle typed
 *   - Zero vtable, zero RTTI, zero exceptions across the boundary
 *
 * Design: Hourglass Pattern (Stefanus Du Toit, CppCon 2014)
 *   C++ (core) -> C ABI waist -> C++ (plugin)
 *
 * ABI version is encoded as a uint32_t monotonic counter. A plugin whose
 * compiled-against version differs from the runtime core version MUST be
 * rejected at load time.
 */

#ifndef NEUROFABRIC_ABI_H
#define NEUROFABRIC_ABI_H

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  1. Export / Visibility Macros                                      */
/* ------------------------------------------------------------------ */

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef NF_BUILDING_DLL
    #define NF_API __declspec(dllexport)
  #else
    #define NF_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define NF_API __attribute__((visibility("default")))
#else
  #define NF_API
#endif

/* ------------------------------------------------------------------ */
/*  2. ABI Version                                                     */
/* ------------------------------------------------------------------ */

#define NF_ABI_VERSION_MAJOR 0
#define NF_ABI_VERSION_MINOR 1
#define NF_ABI_VERSION_PATCH 1

/** Packed ABI version: 0x00MMNNPP */
#define NF_ABI_VERSION \
    (((uint32_t)NF_ABI_VERSION_MAJOR << 16) | \
     ((uint32_t)NF_ABI_VERSION_MINOR << 8)  | \
     ((uint32_t)NF_ABI_VERSION_PATCH))

/* ------------------------------------------------------------------ */
/*  3. Opaque Handles                                                  */
/*     All cross-boundary object references are type-safe void*        */
/*     wrappers. The concrete struct lives ONLY inside its owning      */
/*     translation unit (core or plugin). No header ever sees the      */
/*     layout — this is the "capability token" idiom.                  */
/* ------------------------------------------------------------------ */

/** Core scheduler context — owns the DAG and global allocator. */
typedef struct nf_context_t*       nf_context;

/** A single node in the compute DAG. */
typedef struct nf_node_t*          nf_node;

/** An execution provider instance (one per loaded plugin). */
typedef struct nf_provider_t*      nf_provider;

/** A device-side or unified-memory tensor buffer. */
typedef struct nf_buffer_t*        nf_buffer;

/* ------------------------------------------------------------------ */
/*  4. Status Codes                                                    */
/* ------------------------------------------------------------------ */

typedef enum nf_status {
    NF_OK                    = 0,
    NF_ERROR_INVALID_ARG     = 1,
    NF_ERROR_OUT_OF_MEMORY   = 2,
    NF_ERROR_NOT_FOUND       = 3,
    NF_ERROR_ABI_MISMATCH    = 4,
    NF_ERROR_PLUGIN_LOAD     = 5,
    NF_ERROR_DEVICE_LOST     = 6,
    NF_ERROR_UNSUPPORTED_OP  = 7,
    NF_ERROR_INTERNAL        = 255
} nf_status;

/* ------------------------------------------------------------------ */
/*  5. Data Types & Tensor Descriptor (POD)                            */
/* ------------------------------------------------------------------ */

typedef enum nf_dtype {
    NF_DTYPE_F32   = 0,
    NF_DTYPE_F16   = 1,
    NF_DTYPE_BF16  = 2,
    NF_DTYPE_I8    = 3,
    NF_DTYPE_I32   = 4,
    NF_DTYPE_U8    = 5,
    NF_DTYPE_Q4_0  = 6,   /**< ggml Q4_0: 18B/block, 32 elem/block */
    NF_DTYPE_Q8_0  = 7,   /**< ggml Q8_0: 34B/block, 32 elem/block */
    NF_DTYPE_Q4_1  = 8,   /**< ggml Q4_1: 20B/block, 32 elem/block */
    NF_DTYPE_Q5_0  = 9,   /**< ggml Q5_0: 22B/block, 32 elem/block */
    NF_DTYPE_Q5_1  = 10,  /**< ggml Q5_1: 24B/block, 32 elem/block */
    NF_DTYPE_Q2_K  = 11,  /**< ggml Q2_K: 84B/block, 256 elem/block */
    NF_DTYPE_Q3_K  = 12,  /**< ggml Q3_K: 110B/block, 256 elem/block */
    NF_DTYPE_Q4_K  = 13,  /**< ggml Q4_K: 144B/block, 256 elem/block */
    NF_DTYPE_Q5_K  = 14,  /**< ggml Q5_K: 176B/block, 256 elem/block */
    NF_DTYPE_Q6_K  = 15   /**< ggml Q6_K: 210B/block, 256 elem/block */
} nf_dtype;

#define NF_MAX_DIMS 8

/**
 * POD tensor descriptor — safe to memcpy across any boundary.
 * Does NOT own the data pointer; lifetime is managed by nf_buffer.
 */
typedef struct nf_tensor_desc {
    nf_dtype  dtype;
    uint32_t  ndim;
    uint64_t  shape[NF_MAX_DIMS];
    uint64_t  strides[NF_MAX_DIMS];   /**< byte strides, 0 = contiguous */
    uint64_t  size_bytes;
} nf_tensor_desc;

#ifdef __cplusplus
static_assert(sizeof(nf_tensor_desc) == 144,
    "ABI break: nf_tensor_desc size changed");
#else
_Static_assert(sizeof(nf_tensor_desc) == 144,
    "ABI break: nf_tensor_desc size changed");
#endif

/* ------------------------------------------------------------------ */
/*  6. Execution Provider VTable (Struct of Function Pointers)         */
/*     This IS the plugin contract. Every execution provider (Metal,   */
/*     RKNN, CUDA, …) fills this struct at load time. The core        */
/*     scheduler invokes providers ONLY through these pointers.        */
/*     Zero vtable. Zero indirect branch ambiguity. icache-friendly.   */
/* ------------------------------------------------------------------ */

typedef struct nf_provider_vtable {
    /** Human-readable provider name, e.g. "apple_metal", "rockchip_rknn". */
    const char* (*get_name)(nf_provider self);

    /** ABI version this plugin was compiled against. */
    uint32_t    (*get_abi_version)(nf_provider self);

    /* -- Lifecycle -------------------------------------------------- */

    /** One-time device init. Returns NF_OK or error. */
    nf_status   (*init)(nf_provider self);

    /** Tear down device resources. Idempotent. */
    void        (*shutdown)(nf_provider self);

    /* -- Memory ----------------------------------------------------- */

    /**
     * Allocate a device/unified buffer described by `desc`.
     * On unified-memory architectures (Apple Silicon, RK3588 CMA),
     * this MAY return a pointer directly accessible from CPU.
     */
    nf_status   (*buffer_alloc)(nf_provider self,
                                const nf_tensor_desc* desc,
                                nf_buffer* out);

    /** Release a buffer. NULL-safe. */
    void        (*buffer_free)(nf_provider self, nf_buffer buf);

    /**
     * Obtain a CPU-visible pointer. On discrete GPUs this implies a
     * staging copy; on unified memory it is zero-cost.
     */
    nf_status   (*buffer_map)(nf_provider self,
                              nf_buffer buf,
                              void** out_ptr);

    nf_status   (*buffer_unmap)(nf_provider self, nf_buffer buf);

    /* -- Compute ---------------------------------------------------- */

    /**
     * Submit a named kernel / operator for execution.
     * `op_name`   — canonical operator name (e.g. "matmul", "softmax").
     * `inputs`    — array of input buffers,  length `n_in`.
     * `outputs`   — array of output buffers, length `n_out`.
     *
     * Execution MAY be asynchronous; call `synchronize` to drain.
     */
    nf_status   (*dispatch)(nf_provider self,
                            const char* op_name,
                            const nf_buffer* inputs,  uint32_t n_in,
                            nf_buffer*       outputs, uint32_t n_out);

    /** Block until all dispatched work on this provider completes. */
    nf_status   (*synchronize)(nf_provider self);

} nf_provider_vtable;

/* ------------------------------------------------------------------ */
/*  7. Plugin Entry Point                                              */
/*     Every plugin .dylib/.so/.dll MUST export exactly ONE symbol:    */
/*       nf_plugin_register                                            */
/*     The core loader calls it, the plugin fills the vtable and       */
/*     returns its opaque provider handle.                             */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Signature that every plugin must export.
 *
 * @param[out] vtable   Plugin fills this struct with its function pointers.
 * @param[out] provider Plugin returns its opaque self-handle here.
 * @return NF_OK on success; any error aborts loading.
 */
typedef nf_status (*nf_plugin_register_fn)(nf_provider_vtable* vtable,
                                           nf_provider*        provider);

/** The well-known exported symbol name. */
#define NF_PLUGIN_ENTRY_SYMBOL "nf_plugin_register"

#ifdef __cplusplus
}
#endif

#endif /* NEUROFABRIC_ABI_H */
