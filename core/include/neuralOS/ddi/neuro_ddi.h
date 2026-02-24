/**
 * @file neuro_ddi.h
 * @brief Neuro-Fabric DDI — Async Completion Model (C11 ABI)
 *
 * Phase 37.5: Extends provider vtable with async dispatch.
 *   - nf_completion_token: opaque async completion handle
 *   - nf_ddi_vtable: dispatch_async, wait, poll, query_caps
 *   - nf_plugin_register_ddi: optional plugin export
 *
 * Additive extension — existing plugins work without changes.
 */

#ifndef NEUROFABRIC_DDI_H
#define NEUROFABRIC_DDI_H

#include "neuro_fabric_abi.h"
#include "neuro_buffer_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  1. Completion Token                                                */
/* ------------------------------------------------------------------ */

typedef struct nf_completion_token_t* nf_completion_token;

/* ------------------------------------------------------------------ */
/*  2. Driver Capabilities                                             */
/* ------------------------------------------------------------------ */

typedef struct nf_driver_caps {
    uint32_t max_concurrent;    /**< Max concurrent dispatches */
    uint64_t memory_bytes;      /**< Available device memory */
    uint64_t flops;             /**< Peak FLOPS */
    uint32_t supported_dtypes;  /**< Bitmask of nf_dtype */
    uint32_t flags;             /**< Capability flags (below) */
} nf_driver_caps;

#define NF_CAP_ASYNC    0x01    /**< Supports async dispatch */
#define NF_CAP_RDMA     0x02    /**< Supports RDMA transfers */
#define NF_CAP_FP16     0x04    /**< Native FP16 support */
#define NF_CAP_INT8     0x08    /**< Native INT8 support */
#define NF_CAP_PAGED    0x10    /**< Supports paged memory */

/* ------------------------------------------------------------------ */
/*  3. DDI VTable — Async Dispatch Extension                           */
/* ------------------------------------------------------------------ */

typedef struct nf_ddi_vtable {
    /**
     * Non-blocking dispatch. Returns a completion token.
     * Caller must wait_completion() or poll_completion() before
     * reading output buffers.
     */
    nf_status (*dispatch_async)(nf_provider self,
                                const char* op_name,
                                const nf_buffer* inputs,  uint32_t n_in,
                                nf_buffer*       outputs, uint32_t n_out,
                                nf_completion_token* token);

    /** Block until the given completion token resolves. */
    nf_status (*wait_completion)(nf_provider self,
                                 nf_completion_token token,
                                 uint64_t timeout_ms);

    /** Non-blocking poll. Returns NF_OK if complete, NF_ERROR_TIMEOUT if pending. */
    nf_status (*poll_completion)(nf_provider self,
                                 nf_completion_token token);

    /** Query driver capabilities. */
    nf_status (*query_caps)(nf_provider self,
                            nf_driver_caps* caps);

} nf_ddi_vtable;

/* ------------------------------------------------------------------ */
/*  4. Plugin Entry Point (Optional)                                   */
/* ------------------------------------------------------------------ */

typedef nf_status (*nf_plugin_register_ddi_fn)(nf_ddi_vtable* ddi_vt);

#define NF_PLUGIN_DDI_ENTRY_SYMBOL "nf_plugin_register_ddi"

/* ------------------------------------------------------------------ */
/*  5. Compile-time assertions                                         */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
static_assert(sizeof(nf_driver_caps) == 32,
    "ABI break: nf_driver_caps size changed");
#else
_Static_assert(sizeof(nf_driver_caps) == 32,
    "ABI break: nf_driver_caps size changed");
#endif

#ifdef __cplusplus
}
#endif

#endif /* NEUROFABRIC_DDI_H */
