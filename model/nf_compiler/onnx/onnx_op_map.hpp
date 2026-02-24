/**
 * @file onnx_op_map.hpp
 * @brief NeuralOS L1 — ONNX Op → NF Op Mapping Table
 *
 * Phase 37.4: Maps ~40 common transformer ONNX ops to NF HighOpKind.
 */

#ifndef NEURALOS_COMPILER_ONNX_OP_MAP_HPP
#define NEURALOS_COMPILER_ONNX_OP_MAP_HPP

#include "neuralOS/compiler/nfir_high.hpp"

#include <string>
#include <unordered_map>

namespace neuralOS { namespace compiler { namespace onnx {

struct OpMapping {
    neuralOS::compiler::HighOpKind kind;
    bool supported;
};

inline const std::unordered_map<std::string, OpMapping>& get_op_map() {
    using K = neuralOS::compiler::HighOpKind;
    static const std::unordered_map<std::string, OpMapping> map = {
        {"MatMul",           {K::MATMUL,      true}},
        {"Gemm",             {K::MATMUL,      true}},
        {"MatMulInteger",    {K::MATMUL,      true}},
        {"Attention",        {K::ATTENTION,    true}},
        {"MultiHeadAttention", {K::ATTENTION,  true}},
        {"LayerNormalization", {K::LAYER_NORM, true}},
        {"SimplifiedLayerNormalization", {K::RMS_NORM, true}},
        {"RotaryEmbedding",  {K::ROPE,        true}},
        {"Relu",             {K::SILU,        true}},
        {"Sigmoid",          {K::SILU,        true}},
        {"Silu",             {K::SILU,        true}},
        {"Gelu",             {K::GELU,        true}},
        {"FastGelu",         {K::GELU,        true}},
        {"Mul",              {K::ELEMENT_MUL, true}},
        {"Add",              {K::ELEMENT_ADD, true}},
        {"Sub",              {K::ELEMENT_ADD, true}},
        {"Softmax",          {K::SOFTMAX,     true}},
        {"DequantizeLinear", {K::DEQUANT,     true}},
        {"QuantizeLinear",   {K::DEQUANT,     true}},
        {"Gather",           {K::EMBEDDING,   true}},
        {"Embed",            {K::EMBEDDING,   true}},
        {"Linear",           {K::LINEAR,      true}},
        {"Conv",             {K::CUSTOM,      true}},
        {"Reshape",          {K::CUSTOM,      true}},
        {"Transpose",        {K::CUSTOM,      true}},
        {"Squeeze",          {K::CUSTOM,      true}},
        {"Unsqueeze",        {K::CUSTOM,      true}},
        {"Concat",           {K::CUSTOM,      true}},
        {"Split",            {K::CUSTOM,      true}},
        {"Slice",            {K::CUSTOM,      true}},
        {"Cast",             {K::CUSTOM,      true}},
        {"Shape",            {K::CUSTOM,      true}},
        {"Expand",           {K::CUSTOM,      true}},
        {"Where",            {K::CUSTOM,      true}},
        {"Pow",              {K::CUSTOM,      true}},
        {"Sqrt",             {K::CUSTOM,      true}},
        {"Div",              {K::ELEMENT_MUL, true}},
        {"Tanh",             {K::CUSTOM,      true}},
        {"Erf",              {K::CUSTOM,      true}},
        {"ReduceMean",       {K::CUSTOM,      true}},
        {"Constant",         {K::CUSTOM,      true}},
    };
    return map;
}

inline OpMapping lookup_op(const std::string& onnx_op) {
    auto& m = get_op_map();
    auto it = m.find(onnx_op);
    if (it != m.end()) return it->second;
    return {neuralOS::compiler::HighOpKind::CUSTOM, false};
}

}}} // namespace neuralOS::compiler::onnx

/* ================================================================== */
/*  Backward-compat alias: neuralOS::onnx → neuralOS::compiler::onnx  */
/* ================================================================== */
namespace neuralOS { namespace onnx {
    using neuralOS::compiler::onnx::OpMapping;
    using neuralOS::compiler::onnx::get_op_map;
    using neuralOS::compiler::onnx::lookup_op;
}} // namespace neuralOS::onnx

#endif // NEURALOS_COMPILER_ONNX_OP_MAP_HPP
