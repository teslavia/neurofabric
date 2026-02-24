/**
 * @file ddi.hpp
 * @brief NeuralOS ddi — DDI Facade (re-exports ABI headers + async completion model)
 *
 * Phase 36.1: Header-only facade. Re-exports all 7 ABI headers through
 * a single include and forward-declares the async completion types (Phase 37).
 */

#ifndef NEURALOS_DDI_DDI_HPP
#define NEURALOS_DDI_DDI_HPP

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/ddi/neuro_network_protocol.h"
#include "neuralOS/ddi/neuro_ir_format.h"
#include "neuralOS/ddi/metrics.h"
#include "neuralOS/ddi/nf_c_api.h"

namespace neuralOS { namespace ddi {

/* Forward declarations for async completion model (Phase 37) */
struct CompletionToken;
struct DDIVtable;

}} // namespace neuralOS::ddi

// Backward compatibility
namespace neuralOS { namespace L3 {
    using neuralOS::ddi::CompletionToken;
    using neuralOS::ddi::DDIVtable;
}}

#endif // NEURALOS_DDI_DDI_HPP
