/**
 * @file arch_registry.hpp
 * @brief Architecture registry — strategy pattern for multi-arch DAG building
 *
 * Phase 25: K-Quant Dequant · Entropy Reduction · Architecture Registry.
 * Phase 31: Multi-Architecture DAG & KV Cache Intelligence.
 *
 * Provides a registration mechanism for different model architectures
 * (LLaMA, Mistral, Phi, MoE, etc.) so the DAG builder can dispatch
 * to architecture-specific weight naming, attention, and FFN strategies.
 *
 * Phase 31 additions:
 *   - rope_op_name: RoPE kernel selection (full/partial)
 *   - ffn_op_name:  FFN activation function (silu/gelu)
 *   - norm_op_name: Normalization kernel (rms_norm/layer_norm)
 *   - reserved[2] + arch_ctx: forward-looking slots for Phase 32/33
 */

#ifndef NF_ARCH_REGISTRY_HPP
#define NF_ARCH_REGISTRY_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace nf {

struct nf_arch_config {
    const char* arch_name;          /* "llama", "mistral", "phi", "moe" */
    uint32_t n_layers;
    uint32_t dim, ff_dim, vocab_size;
    uint32_t n_heads, n_kv_heads, head_dim;
    float rope_theta, rms_norm_eps;

    /* Architecture-specific extensions */
    uint32_t sliding_window;        /* 0 = full causal (llama), >0 = sliding (mistral) */
    uint32_t partial_rope_dims;     /* 0 = full rope (llama), >0 = partial (phi) */
    uint32_t n_experts;             /* 0 = dense, >0 = MoE */
    uint32_t n_experts_used;        /* top-k for MoE routing */
};

/**
 * Weight naming strategy — returns tensor name for a given layer/component.
 * Writes into caller-provided buffer, returns pointer to buf.
 */
typedef const char* (*nf_weight_name_fn)(uint32_t layer, const char* component,
                                          char* buf, size_t buf_len);

/** Returns the attention op name for this architecture. */
typedef const char* (*nf_attn_op_name_fn)();

/** Returns the RoPE op name (with FP16 awareness). */
typedef const char* (*nf_rope_op_name_fn)(bool is_fp16);

/** Returns the FFN activation op name ("silu" or "gelu"). */
typedef const char* (*nf_ffn_op_name_fn)();

/** Returns the normalization op name ("rms_norm" or "layer_norm"). */
typedef const char* (*nf_norm_op_name_fn)();

/**
 * Architecture strategy — bundles all architecture-specific function pointers.
 *
 * All new fields (Phase 31) default to nullptr via aggregate init.
 * When nullptr, the DAG builder falls back to LLaMA defaults:
 *   rope_op_name=nullptr → "rope_batch" / "rope_batch_f16"
 *   ffn_op_name=nullptr  → "silu"
 *   norm_op_name=nullptr → "rms_norm"
 */
struct nf_arch_strategy {
    /* Phase 25 (original) */
    nf_weight_name_fn   weight_name;
    nf_attn_op_name_fn  attn_op_name;

    /* Phase 31: architecture-specific kernel selection */
    nf_rope_op_name_fn  rope_op_name;    /* nullptr → default rope_batch */
    nf_ffn_op_name_fn   ffn_op_name;     /* nullptr → default silu */
    nf_norm_op_name_fn  norm_op_name;    /* nullptr → default rms_norm */

    /* Phase 31: opaque architecture context (e.g. Phi3ArchCtx*) */
    void*               arch_ctx;

    /* Phase 32: paged attention op name (nullptr → not supported) */
    typedef const char* (*nf_paged_attn_op_name_fn)();
    nf_paged_attn_op_name_fn paged_attn_op_name;

    /* Phase 34: GQA attention op name (nullptr → use cached attention with pc.M) */
    typedef const char* (*nf_gqa_attn_op_name_fn)();
    nf_gqa_attn_op_name_fn gqa_attn_op_name;
};

/* ---- Global registry (fixed-size, no heap) ---- */

static constexpr int NF_MAX_ARCHS = 16;

struct ArchRegistryState {
    nf_arch_strategy strategies[NF_MAX_ARCHS];
    const char*      names[NF_MAX_ARCHS];
    int              count = 0;
};

inline ArchRegistryState& arch_registry_state() {
    static ArchRegistryState s;
    return s;
}

inline bool nf_register_arch(const char* name, nf_arch_strategy strat) {
    auto& s = arch_registry_state();
    if (s.count >= NF_MAX_ARCHS) return false;
    s.names[s.count] = name;
    s.strategies[s.count] = strat;
    ++s.count;
    return true;
}

inline const nf_arch_strategy* nf_find_arch(const char* name) {
    auto& s = arch_registry_state();
    for (int i = 0; i < s.count; ++i)
        if (std::strcmp(s.names[i], name) == 0) return &s.strategies[i];
    return nullptr;
}

inline int nf_arch_count() {
    return arch_registry_state().count;
}

inline void nf_clear_arch_registry() {
    auto& s = arch_registry_state();
    s.count = 0;
}

/* ---- Default LLaMA strategy functions ---- */

inline const char* llama_attn_op_name() { return "flash_attention_cached"; }
inline const char* llama_gqa_attn_op_name() { return "flash_attention_gqa"; }
inline const char* llama_paged_attn_op_name() { return "flash_attention_paged"; }
inline const char* mistral_attn_op_name() { return "flash_attention_cached"; }

inline const char* llama_rope_op_name(bool is_fp16) {
    return is_fp16 ? "rope_batch_f16" : "rope_batch";
}

inline const char* llama_ffn_op_name() { return "silu"; }
inline const char* llama_norm_op_name() { return "rms_norm"; }

inline const char* llama_weight_name(uint32_t layer, const char* component,
                                      char* buf, size_t buf_len) {
    if (std::strcmp(component, "token_embd") == 0) {
        std::snprintf(buf, buf_len, "token_embd.weight");
    } else if (std::strcmp(component, "output_norm") == 0) {
        std::snprintf(buf, buf_len, "output_norm.weight");
    } else if (std::strcmp(component, "output") == 0) {
        std::snprintf(buf, buf_len, "output.weight");
    } else {
        std::snprintf(buf, buf_len, "blk.%u.%s.weight", layer, component);
    }
    return buf;
}

inline void nf_register_llama() {
    nf_arch_strategy strat{};
    strat.weight_name  = llama_weight_name;
    strat.attn_op_name = llama_attn_op_name;
    strat.rope_op_name = llama_rope_op_name;
    strat.ffn_op_name  = llama_ffn_op_name;
    strat.norm_op_name = llama_norm_op_name;
    strat.paged_attn_op_name = llama_paged_attn_op_name;
    strat.gqa_attn_op_name = llama_gqa_attn_op_name;
    nf_register_arch("llama", strat);
}

/* Mistral uses same weight names as LLaMA; difference is sliding_window in config */
inline void nf_register_mistral() {
    nf_arch_strategy strat{};
    strat.weight_name  = llama_weight_name;
    strat.attn_op_name = mistral_attn_op_name;
    strat.rope_op_name = llama_rope_op_name;
    strat.ffn_op_name  = llama_ffn_op_name;
    strat.norm_op_name = llama_norm_op_name;
    strat.paged_attn_op_name = llama_paged_attn_op_name;
    strat.gqa_attn_op_name = llama_gqa_attn_op_name;
    nf_register_arch("mistral", strat);
}

/* ---- Phi-3 strategy ---- */

/* Phi-3 uses GGUF "blk.N.xxx" naming (same as LLaMA in gguf format).
 * Key differences:
 *   - partial RoPE (only first partial_rotary_factor * head_dim dims rotated)
 *   - SiLU activation (same as LLaMA)
 *   - May have fused QKV weight "blk.N.attn_qkv.weight"
 */

struct Phi3ArchCtx {
    float    partial_rotary_factor;  /* typically 0.5 */
    uint32_t rotary_dims;            /* = head_dim * partial_rotary_factor */
};

/* Phi-3 weight naming: same as LLaMA in GGUF format */
inline const char* phi3_weight_name(uint32_t layer, const char* component,
                                     char* buf, size_t buf_len) {
    return llama_weight_name(layer, component, buf, buf_len);
}

inline const char* phi3_attn_op_name() { return "flash_attention_cached"; }
inline const char* phi3_rope_op_name(bool is_fp16) {
    return is_fp16 ? "rope_batch_f16" : "rope_batch";
}
inline const char* phi3_ffn_op_name() { return "silu"; }
inline const char* phi3_norm_op_name() { return "rms_norm"; }

inline void nf_register_phi3(Phi3ArchCtx* ctx = nullptr) {
    nf_arch_strategy strat{};
    strat.weight_name  = phi3_weight_name;
    strat.attn_op_name = phi3_attn_op_name;
    strat.rope_op_name = phi3_rope_op_name;
    strat.ffn_op_name  = phi3_ffn_op_name;
    strat.norm_op_name = phi3_norm_op_name;
    strat.paged_attn_op_name = llama_paged_attn_op_name;
    strat.arch_ctx     = ctx;
    nf_register_arch("phi3", strat);
}

/* ---- Qwen2 strategy ---- */

/* Qwen2 uses HuggingFace naming: model.layers.{idx}.self_attn.{q,k,v,o}_proj.weight
 * In GGUF format, this maps to the standard blk.N.xxx naming (same as LLaMA).
 * Key differences from LLaMA: none structurally — Qwen2 is a LLaMA-family arch.
 */
inline const char* qwen2_attn_op_name() { return "flash_attention_cached"; }
inline const char* qwen2_rope_op_name(bool is_fp16) {
    return is_fp16 ? "rope_batch_f16" : "rope_batch";
}
inline const char* qwen2_ffn_op_name() { return "silu"; }
inline const char* qwen2_norm_op_name() { return "rms_norm"; }

inline void nf_register_qwen2() {
    nf_arch_strategy strat{};
    strat.weight_name  = llama_weight_name;
    strat.attn_op_name = qwen2_attn_op_name;
    strat.rope_op_name = qwen2_rope_op_name;
    strat.ffn_op_name  = qwen2_ffn_op_name;
    strat.norm_op_name = qwen2_norm_op_name;
    strat.paged_attn_op_name = llama_paged_attn_op_name;
    nf_register_arch("qwen2", strat);
}

/* ---- Gemma strategy ---- */

/* Gemma uses GELU activation instead of SiLU, and RMS norm with +1 offset.
 * Weight naming in GGUF follows the standard blk.N.xxx format.
 */
inline const char* gemma_attn_op_name() { return "flash_attention_cached"; }
inline const char* gemma_rope_op_name(bool is_fp16) {
    return is_fp16 ? "rope_batch_f16" : "rope_batch";
}
inline const char* gemma_ffn_op_name() { return "gelu"; }
inline const char* gemma_norm_op_name() { return "rms_norm"; }

inline void nf_register_gemma() {
    nf_arch_strategy strat{};
    strat.weight_name  = llama_weight_name;
    strat.attn_op_name = gemma_attn_op_name;
    strat.rope_op_name = gemma_rope_op_name;
    strat.ffn_op_name  = gemma_ffn_op_name;
    strat.norm_op_name = gemma_norm_op_name;
    strat.paged_attn_op_name = llama_paged_attn_op_name;
    nf_register_arch("gemma", strat);
}

/* ---- Mixtral strategy (MoE) ---- */

/* Mixtral is a Mixture-of-Experts variant of Mistral.
 * Uses sliding window attention (like Mistral) and SiLU activation.
 * MoE routing is handled at the DAG builder level via n_experts/n_experts_used
 * in nf_arch_config. Weight naming follows standard GGUF blk.N.xxx format.
 */
inline const char* mixtral_attn_op_name() { return "flash_attention_cached"; }
inline const char* mixtral_rope_op_name(bool is_fp16) {
    return is_fp16 ? "rope_batch_f16" : "rope_batch";
}
inline const char* mixtral_ffn_op_name() { return "silu"; }
inline const char* mixtral_norm_op_name() { return "rms_norm"; }

inline void nf_register_mixtral() {
    nf_arch_strategy strat{};
    strat.weight_name  = llama_weight_name;
    strat.attn_op_name = mixtral_attn_op_name;
    strat.rope_op_name = mixtral_rope_op_name;
    strat.ffn_op_name  = mixtral_ffn_op_name;
    strat.norm_op_name = mixtral_norm_op_name;
    strat.paged_attn_op_name = llama_paged_attn_op_name;
    nf_register_arch("mixtral", strat);
}

/* ---- Helper: resolve op name with fallback ---- */

inline const char* nf_resolve_rope_op(const nf_arch_strategy* strat, bool is_fp16) {
    if (strat && strat->rope_op_name) return strat->rope_op_name(is_fp16);
    return is_fp16 ? "rope_batch_f16" : "rope_batch";
}

inline const char* nf_resolve_ffn_op(const nf_arch_strategy* strat) {
    if (strat && strat->ffn_op_name) return strat->ffn_op_name();
    return "silu";
}

inline const char* nf_resolve_norm_op(const nf_arch_strategy* strat) {
    if (strat && strat->norm_op_name) return strat->norm_op_name();
    return "rms_norm";
}

inline const char* nf_resolve_attn_op(const nf_arch_strategy* strat) {
    if (strat && strat->attn_op_name) return strat->attn_op_name();
    return "flash_attention_cached";
}

inline const char* nf_resolve_paged_attn_op(const nf_arch_strategy* strat) {
    if (strat && strat->paged_attn_op_name) return strat->paged_attn_op_name();
    return "flash_attention_paged";
}

inline const char* nf_resolve_gqa_attn_op(const nf_arch_strategy* strat) {
    if (strat && strat->gqa_attn_op_name) return strat->gqa_attn_op_name();
    return "flash_attention_gqa";
}

} /* namespace nf */

#endif /* NF_ARCH_REGISTRY_HPP */
