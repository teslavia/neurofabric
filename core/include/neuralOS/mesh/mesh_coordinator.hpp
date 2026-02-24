/**
 * @file mesh_coordinator.hpp
 * @brief NeuralOS L5 — Global Mesh Coordinator
 *
 * Phase 38.1: Manages compute node registration, topology discovery,
 * subgraph assignment, and health monitoring.
 */

#ifndef NEURALOS_L5_MESH_COORDINATOR_HPP
#define NEURALOS_L5_MESH_COORDINATOR_HPP

#include "neuralOS/mesh/topology.hpp"
#include "neuralOS/kernel/VirtualBus.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuralOS { namespace L5 {

/* ================================================================== */
/*  NodeHealth — per-node health tracking                              */
/* ================================================================== */

struct NodeHealth {
    uint32_t node_id       = 0;
    bool     alive         = false;
    uint64_t last_heartbeat_ms = 0;
    double   latency_us    = 0.0;
    float    load_factor   = 0.0f;  /* 0.0 = idle, 1.0 = saturated */
};

/* ================================================================== */
/*  MeshCoordinator — global network mesh management                   */
/* ================================================================== */

class MeshCoordinator {
public:
    struct Config {
        uint64_t heartbeat_timeout_ms = 5000;
        uint64_t discovery_interval_ms = 10000;
    };

    explicit MeshCoordinator(Config cfg) : cfg_(cfg) {}
    MeshCoordinator() : MeshCoordinator(Config{}) {}

    /* ---- Node Registration ---------------------------------------- */

    bool register_node(uint32_t node_id, const std::string& name,
                       const std::string& address,
                       uint64_t memory_bytes, uint64_t flops,
                       bool is_local = false) {
        std::lock_guard<std::mutex> lk(mu_);
        if (nodes_.count(node_id)) return false;

        NodeDescriptor nd;
        nd.node_id      = node_id;
        nd.name         = name;
        nd.address      = address;
        nd.memory_bytes = memory_bytes;
        nd.flops        = flops;
        nd.is_local     = is_local;
        nodes_[node_id] = nd;

        NodeHealth h;
        h.node_id = node_id;
        h.alive = true;
        h.last_heartbeat_ms = now_ms();
        health_[node_id] = h;

        /* Register in VirtualBus */
        bus_.register_provider(node_id, name, address, memory_bytes, flops, is_local);
        return true;
    }

    bool unregister_node(uint32_t node_id) {
        std::lock_guard<std::mutex> lk(mu_);
        nodes_.erase(node_id);
        health_.erase(node_id);
        return true;
    }

    /* ---- Topology Discovery --------------------------------------- */

    /** Discover topology by measuring latency between nodes.
     *  In real implementation, this would ping each pair.
     *  Here we accept explicit link declarations. */
    void add_link(uint32_t src, uint32_t dst,
                  double bandwidth_gbps, double latency_us) {
        std::lock_guard<std::mutex> lk(mu_);
        bus_.add_link(src, dst, bandwidth_gbps, latency_us);
    }

    TopologyDescriptor discover_topology() const {
        std::lock_guard<std::mutex> lk(mu_);
        return bus_.graph().topo;
    }

    /* ---- Subgraph Assignment -------------------------------------- */

    /** Assign DAG tasks to optimal nodes using VirtualBus splitting */
    std::vector<L2::GraphPartition> assign_subgraph(
            const std::vector<uint32_t>& task_ids,
            const std::vector<uint64_t>& task_flops) {
        std::lock_guard<std::mutex> lk(mu_);
        return bus_.split_graph(task_ids, task_flops);
    }

    /* ---- Health Monitoring ---------------------------------------- */

    void heartbeat(uint32_t node_id, float load_factor = 0.0f) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = health_.find(node_id);
        if (it == health_.end()) return;
        it->second.alive = true;
        it->second.last_heartbeat_ms = now_ms();
        it->second.load_factor = load_factor;
    }

    /** Check for dead nodes (no heartbeat within timeout).
     *  Returns list of dead node IDs. */
    std::vector<uint32_t> monitor_health() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<uint32_t> dead;
        uint64_t now = now_ms();
        for (auto& [id, h] : health_) {
            if (h.alive && (now - h.last_heartbeat_ms) > cfg_.heartbeat_timeout_ms) {
                h.alive = false;
                dead.push_back(id);
            }
        }
        return dead;
    }

    /* ---- Queries -------------------------------------------------- */

    uint32_t num_nodes() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(nodes_.size());
    }

    uint32_t num_alive() const {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t count = 0;
        for (auto& [id, h] : health_)
            if (h.alive) ++count;
        return count;
    }

    const NodeHealth* get_health(uint32_t node_id) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = health_.find(node_id);
        return (it != health_.end()) ? &it->second : nullptr;
    }

    L2::RouteResult route(uint32_t src, uint32_t dst) const {
        std::lock_guard<std::mutex> lk(mu_);
        return bus_.route(src, dst);
    }

private:
    static uint64_t now_ms() {
        auto tp = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                tp.time_since_epoch()).count());
    }

    Config                                      cfg_;
    mutable std::mutex                          mu_;
    std::unordered_map<uint32_t, NodeDescriptor> nodes_;
    std::unordered_map<uint32_t, NodeHealth>    health_;
    L2::VirtualBus                              bus_;
};

}} // namespace neuralOS::L5

#endif // NEURALOS_L5_MESH_COORDINATOR_HPP
