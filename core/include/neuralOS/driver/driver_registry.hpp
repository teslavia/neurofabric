/**
 * @file driver_registry.hpp
 * @brief NeuralOS L4 — Unified Driver Capability Query
 *
 * Phase 36.1: Provides a registry for querying provider capabilities
 * (max_concurrent, memory, flops, dtypes, async support, RDMA).
 */

#ifndef NEURALOS_L4_DRIVER_REGISTRY_HPP
#define NEURALOS_L4_DRIVER_REGISTRY_HPP

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"

#include <cstdint>
#include <string>
#include <vector>

namespace neuralOS { namespace L4 {

struct DriverCaps {
    std::string name;
    uint32_t    max_concurrent  = 1;
    uint64_t    memory_bytes    = 0;
    uint64_t    flops           = 0;
    uint32_t    supported_dtypes = 0;  /* bitmask of nf_dtype */
    bool        async_dispatch  = false;
    bool        rdma_capable    = false;
    nf_affinity affinity        = NF_AFFINITY_ANY;
};

class DriverRegistry {
public:
    void register_driver(nf_provider handle,
                         const nf_provider_vtable& vt,
                         const DriverCaps& caps) {
        entries_.push_back({handle, vt, caps});
    }

    const DriverCaps* query(const std::string& name) const {
        for (auto& e : entries_)
            if (e.caps.name == name) return &e.caps;
        return nullptr;
    }

    std::vector<const DriverCaps*> query_by_affinity(nf_affinity aff) const {
        std::vector<const DriverCaps*> result;
        for (auto& e : entries_)
            if (e.caps.affinity == aff || aff == NF_AFFINITY_ANY)
                result.push_back(&e.caps);
        return result;
    }

    size_t count() const { return entries_.size(); }

private:
    struct Entry {
        nf_provider        handle;
        nf_provider_vtable vtable;
        DriverCaps         caps;
    };
    std::vector<Entry> entries_;
};

}} // namespace neuralOS::L4

#endif // NEURALOS_L4_DRIVER_REGISTRY_HPP
