/**
 * @file VirtualBus.hpp
 * @brief NeuralOS L2 — Virtual Interconnect Bus
 *
 * Phase 36.5: Topology graph with routing and DAG splitting.
 *   - TopologyGraph: nodes/edges with bandwidth, latency, hops
 *   - route(): Dijkstra shortest path (weight = latency + 1/bandwidth)
 *   - register_provider(): register local/remote providers
 *   - bandwidth_query(): query available bandwidth between nodes
 *   - split_graph(): partition DAG across providers (Phase 37 ILP interface)
 *
 * Header-only. Uses L5 topology descriptors.
 */

#ifndef NEURALOS_L2_VIRTUALBUS_HPP
#define NEURALOS_L2_VIRTUALBUS_HPP

#include "neuralOS/mesh/topology.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuralOS { namespace L2 {

/* ================================================================== */
/*  TopologyGraph — weighted directed graph of compute nodes           */
/* ================================================================== */

struct TopologyGraph {
    L5::TopologyDescriptor topo;

    void add_node(const L5::NodeDescriptor& node) {
        topo.nodes.push_back(node);
    }

    void add_edge(const L5::EdgeDescriptor& edge) {
        topo.edges.push_back(edge);
    }

    uint32_t num_nodes() const {
        return static_cast<uint32_t>(topo.nodes.size());
    }

    uint32_t num_edges() const {
        return static_cast<uint32_t>(topo.edges.size());
    }
};

/* ================================================================== */
/*  RouteResult — output of shortest-path routing                      */
/* ================================================================== */

struct RouteResult {
    std::vector<uint32_t> path;       /* node IDs from src to dst */
    double                cost = 0.0; /* total weighted cost */
    bool                  found = false;
};

/* ================================================================== */
/*  GraphPartition — output of DAG splitting                           */
/* ================================================================== */

struct GraphPartition {
    uint32_t              provider_node_id = 0;
    std::vector<uint32_t> task_ids;
};

/* ================================================================== */
/*  VirtualBus — topology-aware routing + DAG splitting                */
/* ================================================================== */

class VirtualBus {
public:
    VirtualBus() = default;

    /* ---- Provider Registration ------------------------------------ */

    void register_provider(uint32_t node_id, const std::string& name,
                           const std::string& address,
                           uint64_t memory_bytes, uint64_t flops,
                           bool is_local = false) {
        L5::NodeDescriptor nd;
        nd.node_id      = node_id;
        nd.name         = name;
        nd.address      = address;
        nd.memory_bytes = memory_bytes;
        nd.flops        = flops;
        nd.is_local     = is_local;
        graph_.add_node(nd);
    }

    void add_link(uint32_t src, uint32_t dst,
                  double bandwidth_gbps, double latency_us,
                  uint32_t hops = 1) {
        L5::EdgeDescriptor ed;
        ed.src_node      = src;
        ed.dst_node      = dst;
        ed.bandwidth_gbps = bandwidth_gbps;
        ed.latency_us    = latency_us;
        ed.hops          = hops;
        graph_.add_edge(ed);
    }

    /* ---- Routing: Dijkstra ---------------------------------------- */

    RouteResult route(uint32_t src, uint32_t dst) const {
        RouteResult result;
        if (src == dst) {
            result.path = {src};
            result.cost = 0.0;
            result.found = true;
            return result;
        }

        /* Build adjacency list */
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, double>>> adj;
        for (auto& e : graph_.topo.edges) {
            double w = e.latency_us + (e.bandwidth_gbps > 0
                       ? 1000.0 / e.bandwidth_gbps : 1e9);
            adj[e.src_node].push_back({e.dst_node, w});
        }

        /* Dijkstra */
        constexpr double INF = std::numeric_limits<double>::max();
        std::unordered_map<uint32_t, double> dist;
        std::unordered_map<uint32_t, uint32_t> prev;
        for (auto& n : graph_.topo.nodes) dist[n.node_id] = INF;
        dist[src] = 0.0;

        using PQ = std::pair<double, uint32_t>;
        std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
        pq.push({0.0, src});

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            if (u == dst) break;

            for (auto& [v, w] : adj[u]) {
                double nd = d + w;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    pq.push({nd, v});
                }
            }
        }

        if (dist.count(dst) == 0 || dist[dst] == INF) return result;

        /* Reconstruct path */
        result.cost = dist[dst];
        result.found = true;
        uint32_t cur = dst;
        while (cur != src) {
            result.path.push_back(cur);
            cur = prev[cur];
        }
        result.path.push_back(src);
        std::reverse(result.path.begin(), result.path.end());
        return result;
    }

    /* ---- Bandwidth Query ------------------------------------------ */

    double bandwidth_query(uint32_t src, uint32_t dst) const {
        for (auto& e : graph_.topo.edges) {
            if (e.src_node == src && e.dst_node == dst)
                return e.bandwidth_gbps;
        }
        return 0.0;
    }

    /* ---- DAG Splitting -------------------------------------------- */

    /** Split a set of tasks across available nodes.
     *  Greedy heuristic: assign tasks in topo order to node with most FLOPS.
     *  Phase 37 will add ILP solver. */
    std::vector<GraphPartition> split_graph(
            const std::vector<uint32_t>& task_ids,
            const std::vector<uint64_t>& task_flops) const {
        std::vector<GraphPartition> partitions;
        if (graph_.topo.nodes.empty() || task_ids.empty()) return partitions;

        /* Initialize one partition per node */
        for (auto& n : graph_.topo.nodes) {
            GraphPartition gp;
            gp.provider_node_id = n.node_id;
            partitions.push_back(gp);
        }

        /* Sort nodes by FLOPS descending */
        std::vector<size_t> node_order(graph_.topo.nodes.size());
        for (size_t i = 0; i < node_order.size(); ++i) node_order[i] = i;
        std::sort(node_order.begin(), node_order.end(),
            [&](size_t a, size_t b) {
                return graph_.topo.nodes[a].flops > graph_.topo.nodes[b].flops;
            });

        /* Greedy: assign each task to least-loaded node (by accumulated FLOPS) */
        std::vector<uint64_t> load(partitions.size(), 0);
        for (size_t t = 0; t < task_ids.size(); ++t) {
            size_t best = 0;
            uint64_t min_load = UINT64_MAX;
            for (size_t i = 0; i < partitions.size(); ++i) {
                if (load[i] < min_load) {
                    min_load = load[i];
                    best = i;
                }
            }
            uint64_t tf = (t < task_flops.size()) ? task_flops[t] : 1;
            partitions[best].task_ids.push_back(task_ids[t]);
            load[best] += tf;
        }

        return partitions;
    }

    /* ---- Queries -------------------------------------------------- */

    const TopologyGraph& graph() const { return graph_; }
    uint32_t num_nodes() const { return graph_.num_nodes(); }
    uint32_t num_edges() const { return graph_.num_edges(); }

private:
    TopologyGraph graph_;
};

}} // namespace neuralOS::L2

#endif // NEURALOS_L2_VIRTUALBUS_HPP
