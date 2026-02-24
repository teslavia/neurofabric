/**
 * @file model_config.hpp
 * @brief ModelConfig — named-field struct for create_llama_context()
 *
 * Phase 45A: PagedKVCache and RequestScheduler extracted to kernel layer.
 * This file now only contains ModelConfig and re-exports the kernel headers
 * for backward compatibility.
 */

#ifndef NF_MODEL_CONFIG_HPP
#define NF_MODEL_CONFIG_HPP

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"
#include "kv_cache_policy.hpp"

/* Re-export kernel abstractions for backward compatibility */
#include "neuralOS/kernel/kv_cache.hpp"
#include "neuralOS/kernel/request_scheduler.hpp"

#include <cstdint>

/* Forward declarations — avoid circular include with gguf_loader.hpp */
namespace nf { struct GGUFModel; }

namespace nf {

/* ================================================================== */
/*  ModelConfig                                                        */
/* ================================================================== */

struct ModelConfig {
    /* Engine & provider (required) */
    PipelineEngine*         engine   = nullptr;
    nf_provider             prov     = nullptr;
    nf_provider_vtable*     vt       = nullptr;
    nf_provider_mem_vtable* mem_vt   = nullptr;

    /* Model source (required) */
    const GGUFModel*        model    = nullptr;

    /* Sequence limits (required) */
    uint32_t max_seq         = 512;
    uint32_t max_prefill_seq = 0;     /* 0 = same as max_seq */

    /* Optional features (all default to off/null) */
    bool     use_fp16       = false;
    bool     use_paged_kv   = false;
    uint32_t kv_block_size  = 16;     /* tokens per physical block */
    uint32_t num_kv_blocks  = 0;      /* 0 = auto-calculate */

    const nf_kv_cache_config* kv_cfg = nullptr;
    const char* arch_override        = nullptr;  /* nullptr = auto-detect */
    bool use_neuralOS                = false;    /* Phase 43: enable NeuralOS runtime */
};

} /* namespace nf */

#endif /* NF_MODEL_CONFIG_HPP */
