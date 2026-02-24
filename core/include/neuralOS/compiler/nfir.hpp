/**
 * @file nfir.hpp
 * @brief NeuralOS L1 — NFIR Facade (re-exports neuro_ir_format.h + multi-level IR forward decls)
 *
 * Phase 36.1: Header-only facade. Does NOT move existing files.
 */

#ifndef NEURALOS_L1_NFIR_HPP
#define NEURALOS_L1_NFIR_HPP

#include "neuralOS/ddi/neuro_ir_format.h"

namespace neuralOS { namespace L1 {

/* Forward declarations for multi-level IR (Phase 37) */
struct NfirHighOp;
struct NfirHighGraph;
struct NfirLowOp;
struct NfirLowGraph;
struct FusionCandidate;

}} // namespace neuralOS::L1

#endif // NEURALOS_L1_NFIR_HPP
