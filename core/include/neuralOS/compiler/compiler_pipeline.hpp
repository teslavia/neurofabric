/**
 * @file compiler_pipeline.hpp
 * @brief NeuralOS L1 — Compiler Pipeline
 *
 * Phase 43A.2: Chains DCE → CSE → ShapeInference → FusionPass.
 * Configurable: enable/disable individual passes.
 */

#ifndef NEURALOS_L1_COMPILER_PIPELINE_HPP
#define NEURALOS_L1_COMPILER_PIPELINE_HPP

#include "neuralOS/compiler/dce_pass.hpp"
#include "neuralOS/compiler/cse_pass.hpp"
#include "neuralOS/compiler/shape_inference_pass.hpp"
#include "neuralOS/compiler/fusion_pass.hpp"

#include <cstdint>

namespace neuralOS { namespace L1 {

struct CompileResult {
    uint32_t ops_removed    = 0;
    uint32_t ops_merged     = 0;
    uint32_t shapes_inferred = 0;
    uint32_t fusions_found  = 0;
};

class CompilerPipeline {
public:
    struct Config {
        bool enable_dce    = true;
        bool enable_cse    = true;
        bool enable_shape  = true;
        bool enable_fusion = true;
    };

    CompilerPipeline() = default;
    explicit CompilerPipeline(Config cfg) : cfg_(cfg) {}

    CompileResult run(NfirHighGraph* graph) {
        CompileResult result;
        if (!graph) return result;

        if (cfg_.enable_dce)
            result.ops_removed = dce_.run(graph);
        if (cfg_.enable_cse)
            result.ops_merged = cse_.run(graph);
        if (cfg_.enable_shape)
            result.shapes_inferred = shape_.run(graph);
        if (cfg_.enable_fusion)
            result.fusions_found = fusion_.run(graph);

        return result;
    }

    Config& config() { return cfg_; }
    const Config& config() const { return cfg_; }

private:
    Config              cfg_;
    DCEPass             dce_;
    CSEPass             cse_;
    ShapeInferencePass  shape_;
    FusionPass          fusion_;
};

}} // namespace neuralOS::L1

#endif // NEURALOS_L1_COMPILER_PIPELINE_HPP
