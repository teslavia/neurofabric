/**
 * @file neuro_scheduler_abi.h
 * @brief Neuro-Fabric Scheduler ABI — DAG Task Graph & Async Futures
 *
 * Phase 3 of the hourglass ABI stack. Builds on:
 *   Phase 1: neuro_fabric_abi.h   (provider vtable, opaque handles)
 *   Phase 2: neuro_buffer_abi.h   (buffer ops, zero-copy, cache coherency)
 *
 * This header defines the C-ABI contract for:
 *   - Compute graph construction (nodes + edges)
 *   - Asynchronous task submission with opaque futures
 *   - Provider affinity hints for heterogeneous routing
 *
 * Evolution extension points:
 *   Step 1: Static local DAG (single node, CPU/NPU/GPU split)
 *   Step 2: nf_task_desc.flags gains NF_TASK_REMOTE → network plugin
 *   Step 3: nf_context_hub_* functions for agent-native state routing
 *
 * Same discipline: pure C linkage, POD structs, opaque handles,
 * zero vtable across the boundary.
 */

#ifndef NEUROFABRIC_SCHEDULER_ABI_H
#define NEUROFABRIC_SCHEDULER_ABI_H

#include "neuro_buffer_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  1. Opaque Handles                                                  */
/* ------------------------------------------------------------------ */

/** A compute graph — DAG of tasks with dependency edges. */
typedef struct nf_graph_t*    nf_graph;

/** A submitted task — one node in the execution DAG. */
typedef struct nf_task_t*     nf_task;

/** An async completion token — non-blocking wait primitive. */
typedef struct nf_future_t*   nf_future;

/** A context hub instance (Step 3 evolution). */
typedef struct nf_ctx_hub_t*  nf_ctx_hub;

/* ------------------------------------------------------------------ */
/*  2. Task Priority & Scheduling Hints                                */
/* ------------------------------------------------------------------ */

typedef enum nf_task_priority {
    NF_PRIORITY_LOW      = 0,
    NF_PRIORITY_NORMAL   = 1,
    NF_PRIORITY_HIGH     = 2,
    NF_PRIORITY_CRITICAL = 3   /**< Real-time path, preempts others */
} nf_task_priority;

typedef enum nf_task_flags {
    NF_TASK_NONE         = 0,
    NF_TASK_ASYNC        = 0x01,  /**< Non-blocking dispatch              */
    NF_TASK_FENCE        = 0x02,  /**< Barrier: all prior tasks must drain */
    NF_TASK_REMOTE       = 0x04,  /**< Step 2: eligible for network route  */
    NF_TASK_CACHEABLE    = 0x08   /**< Step 3: output may enter ContextHub */
} nf_task_flags;

/* ------------------------------------------------------------------ */
/*  3. Provider Affinity — routing hint for heterogeneous dispatch      */
/*     The scheduler uses this to decide WHICH provider executes a     */
/*     task. "ANY" lets the scheduler auto-select based on load.       */
/* ------------------------------------------------------------------ */

typedef enum nf_affinity {
    NF_AFFINITY_ANY      = 0,  /**< Scheduler picks best available      */
    NF_AFFINITY_CPU      = 1,
    NF_AFFINITY_GPU      = 2,
    NF_AFFINITY_NPU      = 3,
    NF_AFFINITY_REMOTE   = 4   /**< Step 2: force network dispatch      */
} nf_affinity;

/* ------------------------------------------------------------------ */
/*  4. Task Descriptor (POD) — submitted to the graph builder          */
/*     Fully describes one compute node before it enters the DAG.      */
/* ------------------------------------------------------------------ */

#define NF_MAX_TASK_INPUTS      16
#define NF_MAX_TASK_OUTPUTS     8
#define NF_MAX_OP_NAME          64
#define NF_MAX_PUSH_CONSTANTS   64

typedef struct nf_task_desc {
    /** Canonical operator name, e.g. "matmul", "conv2d", "softmax". */
    char            op_name[NF_MAX_OP_NAME];

    /** Input buffer handles (from prior tasks or external feed). */
    nf_buffer       inputs[NF_MAX_TASK_INPUTS];
    nf_buffer_ops   input_ops[NF_MAX_TASK_INPUTS];
    uint32_t        n_inputs;

    /** Output buffer handles (pre-allocated or scheduler-allocated). */
    nf_buffer       outputs[NF_MAX_TASK_OUTPUTS];
    nf_buffer_ops   output_ops[NF_MAX_TASK_OUTPUTS];
    uint32_t        n_outputs;

    /** Output tensor descriptors (for scheduler-side allocation). */
    nf_tensor_desc  output_descs[NF_MAX_TASK_OUTPUTS];

    /** Scheduling metadata. */
    nf_affinity     affinity;
    nf_task_priority priority;
    uint32_t        flags;       /**< Bitmask of nf_task_flags */

    /** Opaque user data pointer — passed through to dispatch. */
    void*           user_data;

    /** Push constants — small per-dispatch scalars (seq_len, temperature, etc.) */
    uint32_t        push_constants_size;
    uint8_t         push_constants[NF_MAX_PUSH_CONSTANTS];
} nf_task_desc;

#ifdef __cplusplus
static_assert(sizeof(nf_task_desc) == 3056,
    "nf_task_desc layout changed — offsetof bridge will break");
#else
_Static_assert(sizeof(nf_task_desc) == 3056,
    "nf_task_desc layout changed — offsetof bridge will break");
#endif

/* ------------------------------------------------------------------ */
/*  5. Future Status — polled or waited on by the caller               */
/* ------------------------------------------------------------------ */

typedef enum nf_future_status {
    NF_FUTURE_PENDING    = 0,
    NF_FUTURE_RUNNING    = 1,
    NF_FUTURE_COMPLETE   = 2,
    NF_FUTURE_ERROR      = 3
} nf_future_status;

/* ------------------------------------------------------------------ */
/*  6. Graph Construction & Execution API                              */
/*     Pure C functions exported from nf_core. The scheduler builds    */
/*     a DAG, resolves dependencies via topological sort, then         */
/*     dispatches ready tasks to providers.                            */
/* ------------------------------------------------------------------ */

/** Create an empty compute graph. */
NF_API nf_status nf_graph_create(nf_context ctx, nf_graph* out);

/** Destroy a graph and all its internal task nodes. */
NF_API void      nf_graph_destroy(nf_graph graph);

/**
 * Add a task node to the graph.
 * The task is described by `desc`. The returned `task` handle can be
 * used to wire dependency edges.
 */
NF_API nf_status nf_graph_add_task(nf_graph graph,
                                   const nf_task_desc* desc,
                                   nf_task* out);

/**
 * Declare a dependency edge: `task` cannot execute until `dependency`
 * completes. The scheduler uses these edges for topological ordering.
 */
NF_API nf_status nf_graph_add_edge(nf_graph graph,
                                   nf_task dependency,
                                   nf_task task);

/**
 * Submit the entire graph for execution.
 * The scheduler performs topological sort, resolves provider affinity,
 * and dispatches ready tasks. Returns a future for the whole graph.
 *
 * After submission the graph is frozen — no more add_task/add_edge.
 */
NF_API nf_status nf_graph_submit(nf_graph graph, nf_future* out);

/* ------------------------------------------------------------------ */
/*  7. Future / Async Wait Primitives                                  */
/*     Non-blocking poll + blocking wait with timeout.                 */
/* ------------------------------------------------------------------ */

/** Poll a future without blocking. */
NF_API nf_future_status nf_future_poll(nf_future future);

/**
 * Block until the future completes or `timeout_ms` elapses.
 * Pass 0 for non-blocking poll, UINT64_MAX for infinite wait.
 * Returns the final status.
 */
NF_API nf_future_status nf_future_wait(nf_future future,
                                       uint64_t timeout_ms);

/** Get the error status if future completed with NF_FUTURE_ERROR. */
NF_API nf_status nf_future_get_error(nf_future future);

/** Release the future handle. */
NF_API void      nf_future_destroy(nf_future future);

/* ------------------------------------------------------------------ */
/*  8. Context Hub — Step 3 Evolution Forward Declarations             */
/*     Agent-native state routing. These are ABI-stable entry points   */
/*     that will be implemented when the system evolves past static    */
/*     DAG scheduling into multi-agent territory.                      */
/*                                                                     */
/*     The key insight: a "context" is just a named TensorView with    */
/*     a TTL and a radix-tree prefix key for cache matching.           */
/* ------------------------------------------------------------------ */

#define NF_MAX_CTX_KEY_LEN 256

/** POD descriptor for a context entry in the hub. */
typedef struct nf_ctx_entry_desc {
    /** Hierarchical key, e.g. "agent/planner/kv_cache/layer_0". */
    char            key[NF_MAX_CTX_KEY_LEN];

    /** The tensor backing this context slice. */
    nf_buffer       buffer;
    nf_buffer_ops   buffer_ops;
    nf_tensor_desc  tensor_desc;

    /** Time-to-live in milliseconds. 0 = immortal (pinned). */
    uint64_t        ttl_ms;

    /** Sequence number for LRU / radix-tree prefix matching. */
    uint64_t        seq_id;

    /** Owning agent identifier (opaque string). */
    char            agent_id[64];

    uint32_t        _reserved;
} nf_ctx_entry_desc;

#ifdef __cplusplus
static_assert(sizeof(nf_ctx_entry_desc) == 560,
    "ABI break: nf_ctx_entry_desc size changed");
#else
_Static_assert(sizeof(nf_ctx_entry_desc) == 560,
    "ABI break: nf_ctx_entry_desc size changed");
#endif

/** Eviction policy for the context hub cache. */
typedef enum nf_eviction_policy {
    NF_EVICT_LRU           = 0,  /**< Least Recently Used               */
    NF_EVICT_TTL           = 1,  /**< Expire by time-to-live            */
    NF_EVICT_REFCOUNT      = 2,  /**< Evict when refcount drops to zero */
    NF_EVICT_RADIX_PREFIX  = 3   /**< Radix-tree prefix dedup (vLLM-style) */
} nf_eviction_policy;

/** Create a context hub with a memory budget (bytes). */
NF_API nf_status nf_ctx_hub_create(nf_context ctx,
                                   uint64_t budget_bytes,
                                   nf_eviction_policy policy,
                                   nf_ctx_hub* out);

NF_API void      nf_ctx_hub_destroy(nf_ctx_hub hub);

/** Insert or update a context entry. Retains the buffer. */
NF_API nf_status nf_ctx_hub_put(nf_ctx_hub hub,
                                const nf_ctx_entry_desc* entry);

/**
 * Lookup by key prefix (radix match).
 * Returns the longest-prefix-matching entry.
 * `out_buffer` and `out_ops` are filled if found.
 */
NF_API nf_status nf_ctx_hub_get(nf_ctx_hub hub,
                                const char* key_prefix,
                                nf_buffer* out_buffer,
                                nf_buffer_ops* out_ops);

/** Evict entries matching `key_prefix`. Pass "" to evict all. */
NF_API nf_status nf_ctx_hub_evict(nf_ctx_hub hub,
                                  const char* key_prefix);

/** Query current memory usage of the hub. */
NF_API nf_status nf_ctx_hub_stats(nf_ctx_hub hub,
                                  uint64_t* used_bytes,
                                  uint64_t* budget_bytes,
                                  uint32_t* entry_count);

#ifdef __cplusplus
}
#endif

#endif /* NEUROFABRIC_SCHEDULER_ABI_H */
