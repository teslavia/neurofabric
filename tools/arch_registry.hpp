/**
 * @file arch_registry.hpp
 * @brief Architecture registry — strategy pattern for multi-arch DAG building
 *
 * Phase 25: K-Quant Dequant · Entropy Reduction · Architecture Registry.
 *
 * Provides a registration mechanism for different model architectures
 * (LLaMA, Mistral, Phi, MoE, etc.) so the DAG builder can dispatch
 * to architecture-specific weight naming, attention, and FFN strategies.
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

/**
 * Architecture strategy — bundles weight naming with optional metadata.
 * Attention and FFN build functions are Phase 26 scope (Mistral/Phi).
 */
struct nf_arch_strategy {
    nf_weight_name_fn weight_name;
};

/* ---- Global registry (fixed-size, no heap) ---- */

static constexpr int NF_MAX_ARCHS = 8;

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

/* ---- Default LLaMA weight naming strategy ---- */

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
    strat.weight_name = llama_weight_name;
    nf_register_arch("llama", strat);
}

/* Mistral uses same weight names as LLaMA; difference is sliding_window in config */
inline void nf_register_mistral() {
    nf_arch_strategy strat{};
    strat.weight_name = llama_weight_name;
    nf_register_arch("mistral", strat);
}

} /* namespace nf */

#endif /* NF_ARCH_REGISTRY_HPP */
