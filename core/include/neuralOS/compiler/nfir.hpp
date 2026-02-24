/**
 * @file nfir.hpp
 * @brief NeuralOS compiler — NFIR Facade (re-exports neuro_ir_format.h + multi-level IR forward decls)
 *
 * Phase 36.1: Header-only facade. Does NOT move existing files.
 */

#ifndef NEURALOS_COMPILER_NFIR_HPP
#define NEURALOS_COMPILER_NFIR_HPP

#include "neuralOS/ddi/neuro_ir_format.h"

namespace neuralOS { namespace compiler {

/* Forward declarations for multi-level IR (Phase 37) */
struct NfirHighOp;
struct NfirHighGraph;
struct NfirLowOp;
struct NfirLowGraph;
struct FusionCandidate;

}} // namespace neuralOS::compiler

// Backward compatibility
namespace neuralOS { namespace L1 {
    using neuralOS::compiler::NfirHighOp;
    using neuralOS::compiler::NfirHighGraph;
    using neuralOS::compiler::NfirLowOp;
    using neuralOS::compiler::NfirLowGraph;
    using neuralOS::compiler::FusionCandidate;
}}

#endif // NEURALOS_COMPILER_NFIR_HPP
