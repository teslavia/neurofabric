/**
 * @file metal_pso_registry.h
 * @brief PSO index enum and registration table for Metal compute kernels
 *
 * Phase 28: Table-driven PSO management. Replaces 48 named fields with
 * enum-indexed flat array. Included by metal_provider.mm and pso_registry_test.
 */

#ifndef NF_METAL_PSO_REGISTRY_H
#define NF_METAL_PSO_REGISTRY_H

#include <cstdint>

enum MetalPSO : uint16_t {
    /* Core (Phase 9-16) */
    PSO_VECTOR_ADD = 0, PSO_RELU, PSO_ATTN_K, PSO_ATTN_V,
    PSO_RMS_NORM, PSO_ROPE, PSO_ROPE_BATCH, PSO_LINEAR,
    PSO_CAUSAL_ATTN, PSO_DEQUANT_Q4_0, PSO_DEQUANT_Q8_0,
    PSO_DEQUANT_Q6_K, PSO_LINEAR_TILED, PSO_SOFTMAX, PSO_SILU,
    PSO_ELEM_MUL, PSO_EMBED_LOOKUP, PSO_ARGMAX, PSO_CAUSAL_ATTN_CACHED,
    /* Phase 24 */
    PSO_LINEAR_SIMD,
    /* Phase 25 */
    PSO_DEQUANT_Q4_1, PSO_DEQUANT_Q5_0, PSO_DEQUANT_Q5_1,
    PSO_DEQUANT_Q2_K, PSO_DEQUANT_Q3_K, PSO_DEQUANT_Q4_K,
    PSO_DEQUANT_Q5_K, PSO_FLASH_ATTN,
    /* Phase 27: FP16 compute */
    PSO_RMS_NORM_F16, PSO_ROPE_BATCH_F16, PSO_LINEAR_SIMD_F16,
    PSO_LINEAR_TILED_F16, PSO_LINEAR_F16_TO_F32, PSO_FLASH_ATTN_F16,
    PSO_SILU_F16, PSO_ELEM_MUL_F16, PSO_VECTOR_ADD_F16,
    PSO_EMBED_LOOKUP_F16,
    /* Phase 27: Dequant-to-F16 */
    PSO_DEQUANT_Q4_0_F16, PSO_DEQUANT_Q8_0_F16, PSO_DEQUANT_Q6_K_F16,
    PSO_DEQUANT_Q4_1_F16, PSO_DEQUANT_Q5_0_F16, PSO_DEQUANT_Q5_1_F16,
    PSO_DEQUANT_Q2_K_F16, PSO_DEQUANT_Q3_K_F16, PSO_DEQUANT_Q4_K_F16,
    PSO_DEQUANT_Q5_K_F16,
    /* Phase 29: Fused dequant+linear */
    PSO_FUSED_DQ4_LINEAR, PSO_FUSED_DQ4_LINEAR_F16,
    /* Phase 32: Paged attention */
    PSO_FLASH_ATTN_PAGED,
    /* Phase 33: GELU activation (Gemma) */
    PSO_GELU, PSO_GELU_F16,
    /* Phase 33: Fused dequant+SIMD matmul */
    PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD, PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD_F16,
    /* Phase 34: GQA flash attention */
    PSO_FLASH_ATTN_GQA, PSO_FLASH_ATTN_GQA_F16,
    /* Phase 34: MoE expert routing */
    PSO_MOE_GATE, PSO_MOE_SCATTER,
    /* Sentinel */
    PSO_COUNT
};

struct PSORegistration {
    MetalPSO    index;
    const char* msl_name;      /* MSL function name */
    bool        requires_simd; /* only init on GPU Family 7+ */
};

static constexpr PSORegistration kPSOTable[] = {
    /* Core (Phase 9-16) */
    { PSO_VECTOR_ADD,         "vector_add",                false },
    { PSO_RELU,               "relu",                      false },
    { PSO_ATTN_K,             "attention_prefill_k",       false },
    { PSO_ATTN_V,             "attention_prefill_v",       false },
    { PSO_RMS_NORM,           "rms_norm",                  false },
    { PSO_ROPE,               "rope",                      false },
    { PSO_ROPE_BATCH,         "rope_batch",                false },
    { PSO_LINEAR,             "linear",                    false },
    { PSO_CAUSAL_ATTN,        "causal_attention",          false },
    { PSO_DEQUANT_Q4_0,       "dequant_q4_0",              false },
    { PSO_DEQUANT_Q8_0,       "dequant_q8_0",              false },
    { PSO_DEQUANT_Q6_K,       "dequant_q6_k",              false },
    { PSO_LINEAR_TILED,       "linear_tiled",              false },
    { PSO_SOFTMAX,            "softmax",                   false },
    { PSO_SILU,               "silu",                      false },
    { PSO_ELEM_MUL,           "elementwise_mul",           false },
    { PSO_EMBED_LOOKUP,       "embedding_lookup",          false },
    { PSO_ARGMAX,             "argmax_rows",               false },
    { PSO_CAUSAL_ATTN_CACHED, "causal_attention_cached",   false },
    /* Phase 24 */
    { PSO_LINEAR_SIMD,        "linear_simd",               true  },
    /* Phase 25 */
    { PSO_DEQUANT_Q4_1,       "dequant_q4_1",              false },
    { PSO_DEQUANT_Q5_0,       "dequant_q5_0",              false },
    { PSO_DEQUANT_Q5_1,       "dequant_q5_1",              false },
    { PSO_DEQUANT_Q2_K,       "dequant_q2_k",              false },
    { PSO_DEQUANT_Q3_K,       "dequant_q3_k",              false },
    { PSO_DEQUANT_Q4_K,       "dequant_q4_k",              false },
    { PSO_DEQUANT_Q5_K,       "dequant_q5_k",              false },
    { PSO_FLASH_ATTN,         "flash_attention_tiled",     false },
    /* Phase 27: FP16 compute */
    { PSO_RMS_NORM_F16,       "rms_norm_f16",              false },
    { PSO_ROPE_BATCH_F16,     "rope_batch_f16",            false },
    { PSO_LINEAR_SIMD_F16,    "linear_simd_f16",           true  },
    { PSO_LINEAR_TILED_F16,   "linear_tiled_f16",          false },
    { PSO_LINEAR_F16_TO_F32,  "linear_f16_to_f32",         true  },
    { PSO_FLASH_ATTN_F16,     "flash_attention_tiled_f16", false },
    { PSO_SILU_F16,           "silu_f16",                  false },
    { PSO_ELEM_MUL_F16,       "elementwise_mul_f16",       false },
    { PSO_VECTOR_ADD_F16,     "metal_vector_add_f16",      false },
    { PSO_EMBED_LOOKUP_F16,   "embedding_lookup_f16",      false },
    /* Phase 27: Dequant-to-F16 */
    { PSO_DEQUANT_Q4_0_F16,   "dequant_q4_0_f16",         false },
    { PSO_DEQUANT_Q8_0_F16,   "dequant_q8_0_f16",         false },
    { PSO_DEQUANT_Q6_K_F16,   "dequant_q6_k_f16",         false },
    { PSO_DEQUANT_Q4_1_F16,   "dequant_q4_1_f16",         false },
    { PSO_DEQUANT_Q5_0_F16,   "dequant_q5_0_f16",         false },
    { PSO_DEQUANT_Q5_1_F16,   "dequant_q5_1_f16",         false },
    { PSO_DEQUANT_Q2_K_F16,   "dequant_q2_k_f16",         false },
    { PSO_DEQUANT_Q3_K_F16,   "dequant_q3_k_f16",         false },
    { PSO_DEQUANT_Q4_K_F16,   "dequant_q4_k_f16",         false },
    { PSO_DEQUANT_Q5_K_F16,   "dequant_q5_k_f16",         false },
    /* Phase 29: Fused dequant+linear */
    { PSO_FUSED_DQ4_LINEAR,     "dequant_q4_0_linear_tiled",     false },
    { PSO_FUSED_DQ4_LINEAR_F16, "dequant_q4_0_linear_tiled_f16", false },
    /* Phase 32: Paged attention */
    { PSO_FLASH_ATTN_PAGED,     "flash_attention_paged",         false },
    /* Phase 33: GELU activation (Gemma) */
    { PSO_GELU,                 "gelu",                          false },
    { PSO_GELU_F16,             "gelu_f16",                      false },
    /* Phase 33: Fused dequant+SIMD matmul */
    { PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD,     "dequant_q4_0_linear_simd",     true },
    { PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD_F16, "dequant_q4_0_linear_simd_f16", true },
    /* Phase 34: GQA flash attention */
    { PSO_FLASH_ATTN_GQA,     "flash_attention_gqa",     false },
    { PSO_FLASH_ATTN_GQA_F16, "flash_attention_gqa_f16", false },
    /* Phase 34: MoE expert routing */
    { PSO_MOE_GATE,    "moe_top_k_gating",   false },
    { PSO_MOE_SCATTER, "moe_scatter_gather",  false },
};

static constexpr uint16_t kPSOTableSize = sizeof(kPSOTable) / sizeof(kPSOTable[0]);

#endif /* NF_METAL_PSO_REGISTRY_H */
