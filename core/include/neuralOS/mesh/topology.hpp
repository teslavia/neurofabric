/**
 * @file topology.hpp
 * @brief NeuralOS L5 — Topology Descriptor POD Structures
 *
 * Phase 36.1: Defines the node/edge topology descriptors used by
 * VirtualBus (L2) and MeshCoordinator (Phase 38).
 */

#ifndef NEURALOS_L5_TOPOLOGY_HPP
#define NEURALOS_L5_TOPOLOGY_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace neuralOS { namespace L5 {

struct NodeDescriptor {
    uint32_t    node_id     = 0;
    std::string name;
    std::string address;        /* ip:port */
    uint64_t    memory_bytes = 0;
    uint64_t    flops        = 0;
    bool        is_local     = false;
};

struct EdgeDescriptor {
    uint32_t src_node  = 0;
    uint32_t dst_node  = 0;
    double   bandwidth_gbps = 0.0;  /* GB/s */
    double   latency_us     = 0.0;  /* microseconds */
    uint32_t hops           = 1;
};

struct TopologyDescriptor {
    std::vector<NodeDescriptor> nodes;
    std::vector<EdgeDescriptor> edges;

    const NodeDescriptor* find_node(uint32_t id) const {
        for (auto& n : nodes)
            if (n.node_id == id) return &n;
        return nullptr;
    }
};

}} // namespace neuralOS::L5

#endif // NEURALOS_L5_TOPOLOGY_HPP
