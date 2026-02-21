/**
 * @file quant_registry.hpp
 * @brief Centralized quantization dtype/op mapping
 *
 * Phase 25: K-Quant Dequant · Entropy Reduction · Architecture Registry.
 *
 * Replaces per-file lambdas with shared functions for dtype detection,
 * NF dtype mapping, and dequant op name resolution.
 */

#ifndef NF_QUANT_REGISTRY_HPP
#define NF_QUANT_REGISTRY_HPP

#include "gguf_loader.hpp"
#include "neurofabric/neuro_fabric_abi.h"

namespace nf {

inline bool nf_is_quantized(uint32_t gguf_dtype) {
    switch (gguf_dtype) {
        case GGUF_DTYPE_Q4_0: case GGUF_DTYPE_Q4_1:
        case GGUF_DTYPE_Q5_0: case GGUF_DTYPE_Q5_1:
        case GGUF_DTYPE_Q8_0:
        case GGUF_DTYPE_Q2_K: case GGUF_DTYPE_Q3_K:
        case GGUF_DTYPE_Q4_K: case GGUF_DTYPE_Q5_K:
        case GGUF_DTYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline nf_dtype nf_dtype_for_gguf(uint32_t gguf_dtype) {
    switch (gguf_dtype) {
        case GGUF_DTYPE_Q4_0: return NF_DTYPE_Q4_0;
        case GGUF_DTYPE_Q4_1: return NF_DTYPE_Q4_1;
        case GGUF_DTYPE_Q5_0: return NF_DTYPE_Q5_0;
        case GGUF_DTYPE_Q5_1: return NF_DTYPE_Q5_1;
        case GGUF_DTYPE_Q8_0: return NF_DTYPE_Q8_0;
        case GGUF_DTYPE_Q2_K: return NF_DTYPE_Q2_K;
        case GGUF_DTYPE_Q3_K: return NF_DTYPE_Q3_K;
        case GGUF_DTYPE_Q4_K: return NF_DTYPE_Q4_K;
        case GGUF_DTYPE_Q5_K: return NF_DTYPE_Q5_K;
        case GGUF_DTYPE_Q6_K: return NF_DTYPE_Q6_K;
        case GGUF_DTYPE_F16:  return NF_DTYPE_F16;
        default:              return NF_DTYPE_F32;
    }
}

inline const char* nf_dequant_op_name(uint32_t gguf_dtype) {
    switch (gguf_dtype) {
        case GGUF_DTYPE_Q4_0: return "dequant_q4_0";
        case GGUF_DTYPE_Q4_1: return "dequant_q4_1";
        case GGUF_DTYPE_Q5_0: return "dequant_q5_0";
        case GGUF_DTYPE_Q5_1: return "dequant_q5_1";
        case GGUF_DTYPE_Q8_0: return "dequant_q8_0";
        case GGUF_DTYPE_Q2_K: return "dequant_q2_k";
        case GGUF_DTYPE_Q3_K: return "dequant_q3_k";
        case GGUF_DTYPE_Q4_K: return "dequant_q4_k";
        case GGUF_DTYPE_Q5_K: return "dequant_q5_k";
        case GGUF_DTYPE_Q6_K: return "dequant_q6_k";
        default:              return nullptr;
    }
}

} /* namespace nf */

#endif /* NF_QUANT_REGISTRY_HPP */
