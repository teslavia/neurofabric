/**
 * @file PipelineEngine.hpp
 * @brief Core DAG Scheduler — Topological Sort + Async Dispatch
 *
 * INTERNAL TO CORE — never crosses a dynamic library boundary.
 *
 * Architecture:
 *   1. Graph building: tasks are added with dependency edges.
 *   2. Submission: Kahn's algorithm produces a topological order.
 *      Tasks with zero in-degree enter the ready queue.
 *   3. Dispatch: a fixed-size thread pool dequeues ready tasks,
 *      resolves provider affinity, and calls the C-ABI dispatch.
 *   4. Completion: when a task finishes, its successors' in-degrees
 *      are decremented (atomic). Newly-ready tasks are enqueued.
 *   5. Future: the graph-level future completes when all tasks drain.
 *
 * Complexity: O(V + E) topological sort, O(1) amortized enqueue.
 *
 * Evolution hooks:
 *   Step 2: tasks with NF_TASK_REMOTE are routed to a network plugin
 *           instead of a local provider.
 *   Step 3: tasks with NF_TASK_CACHEABLE have their outputs registered
 *           in the ContextHub after completion.
 */

#ifndef NF_PIPELINE_ENGINE_HPP
#define NF_PIPELINE_ENGINE_HPP

#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/ddi/neuro_ddi.h"
#include "neuralOS/kernel/TensorView.hpp"
#include "neuralOS/kernel/ProfileTrace.hpp"
#include "neuralOS/driver/driver_registry.hpp"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nf {

/* ================================================================== */
/*  ThreadPool — fixed-size, lock-based work queue                     */
/*  Intentionally simple. No work-stealing yet (Step 2 evolution).     */
/* ================================================================== */

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads)
        : stop_(false) {
        workers_.reserve(n_threads);
        for (size_t i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F>
    std::future<nf_status> submit(F&& fn) {
        auto task = std::make_shared<std::packaged_task<nf_status()>>(
            std::forward<F>(fn));
        auto fut = task->get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.emplace_back([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            job();
        }
    }

    std::vector<std::thread>          workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    bool                              stop_;
};

/* ================================================================== */
/*  TaskNode — internal representation of one DAG vertex               */
/* ================================================================== */

struct TaskNode {
    uint32_t                id = 0;
    nf_task_desc            desc{};

    /* Adjacency: successors that depend on this node. */
    std::vector<uint32_t>   successors;

    /* Kahn's algorithm state */
    std::atomic<uint32_t>   in_degree{0};

    /* Result */
    std::atomic<nf_status>  result{NF_OK};
    bool                    dispatched = false;

    /* Profiling (always populated; near-zero cost — 3 clock reads per task) */
    TaskProfile             profile{};
};

/* ================================================================== */
/*  ProviderSlot — registered provider + its vtables                   */
/* ================================================================== */

struct ProviderSlot {
    nf_provider             handle   = nullptr;
    nf_provider_vtable      vtable{};
    nf_provider_mem_vtable  mem_vt{};
    nf_ddi_vtable           ddi_vt{};
    bool                    has_ddi  = false;
    nf_affinity             affinity = NF_AFFINITY_ANY;
    std::string             name;
};

/* ================================================================== */
/*  PipelineEngine — the core DAG scheduler                            */
/* ================================================================== */

class PipelineEngine {
public:
    explicit PipelineEngine(size_t n_threads = 0)
        : pool_(n_threads == 0 ? std::thread::hardware_concurrency() : n_threads)
    {}

    ~PipelineEngine() = default;

    PipelineEngine(const PipelineEngine&) = delete;
    PipelineEngine& operator=(const PipelineEngine&) = delete;

    /* -- Provider Registration -------------------------------------- */

    nf_status register_provider(nf_provider handle,
                                const nf_provider_vtable& vt,
                                nf_affinity affinity) {
        std::lock_guard<std::mutex> lk(mu_);
        ProviderSlot slot;
        slot.handle   = handle;
        slot.vtable   = vt;
        slot.affinity = affinity;
        slot.name     = vt.get_name ? vt.get_name(handle) : "unknown";
        providers_.push_back(std::move(slot));

        /* Auto-populate driver registry */
        neuralOS::L4::DriverCaps caps;
        caps.name = providers_.back().name;
        caps.affinity = affinity;
        registry_.register_driver(handle, vt, caps);

        return NF_OK;
    }

    void register_provider_mem(size_t idx, const nf_provider_mem_vtable& mem_vt) {
        std::lock_guard<std::mutex> lk(mu_);
        if (idx < providers_.size()) {
            providers_[idx].mem_vt = mem_vt;
        }
    }

    /** Register DDI async vtable for a provider slot.
     *  If query_caps is available, updates the driver registry. */
    void register_ddi(size_t idx, const nf_ddi_vtable& ddi_vt) {
        std::lock_guard<std::mutex> lk(mu_);
        if (idx >= providers_.size()) return;
        providers_[idx].ddi_vt = ddi_vt;
        providers_[idx].has_ddi = true;

        /* Query caps and update registry if available */
        if (ddi_vt.query_caps) {
            nf_driver_caps hw_caps{};
            if (ddi_vt.query_caps(providers_[idx].handle, &hw_caps) == NF_OK) {
                /* Update existing registry entry */
                auto& entries = registry_;
                neuralOS::L4::DriverCaps updated;
                updated.name = providers_[idx].name;
                updated.affinity = providers_[idx].affinity;
                updated.max_concurrent = hw_caps.max_concurrent;
                updated.memory_bytes = hw_caps.memory_bytes;
                updated.flops = hw_caps.flops;
                updated.supported_dtypes = hw_caps.supported_dtypes;
                updated.async_dispatch = (hw_caps.flags & NF_CAP_ASYNC) != 0;
                updated.rdma_capable = (hw_caps.flags & NF_CAP_RDMA) != 0;
                /* Re-register with updated caps */
                entries.register_driver(providers_[idx].handle,
                                        providers_[idx].vtable, updated);
            }
        }
    }

    uint32_t num_async_providers() const {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t count = 0;
        for (auto& p : providers_)
            if (p.has_ddi) ++count;
        return count;
    }

    const neuralOS::L4::DriverRegistry& registry() const { return registry_; }

    /* -- Profiling Access ------------------------------------------- */

    /**
     * Retrieve the profile for a submitted graph.
     * Returns nullptr if graph_id not found or not yet submitted.
     */
    std::shared_ptr<const GraphProfile> graph_profile(uint32_t gid) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = profiles_.find(gid);
        if (it == profiles_.end()) return nullptr;
        return it->second;
    }

    /* -- Graph Building --------------------------------------------- */

    /** Create a new empty graph. Returns graph ID. */
    uint32_t create_graph() {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t gid = next_graph_id_++;
        graphs_[gid] = GraphState{};
        return gid;
    }

    void destroy_graph(uint32_t gid) {
        std::lock_guard<std::mutex> lk(mu_);
        graphs_.erase(gid);
    }

    /** Add a task node. Returns task ID within the graph. */
    uint32_t add_task(uint32_t gid, const nf_task_desc& desc) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = graphs_.find(gid);
        if (it == graphs_.end()) return UINT32_MAX;

        auto& g = it->second;
        uint32_t tid = static_cast<uint32_t>(g.nodes.size());
        auto node = std::make_unique<TaskNode>();
        node->id   = tid;
        node->desc = desc;
        g.nodes.push_back(std::move(node));
        return tid;
    }

    /** Add dependency edge: `dep` must complete before `task`. */
    nf_status add_edge(uint32_t gid, uint32_t dep, uint32_t task) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = graphs_.find(gid);
        if (it == graphs_.end()) return NF_ERROR_NOT_FOUND;

        auto& g = it->second;
        if (dep >= g.nodes.size() || task >= g.nodes.size())
            return NF_ERROR_INVALID_ARG;

        g.nodes[dep]->successors.push_back(task);
        g.nodes[task]->in_degree.fetch_add(1, std::memory_order_relaxed);
        return NF_OK;
    }

    /* -- Submission: Kahn's Algorithm + Async Dispatch --------------- */

    /**
     * Submit a graph for execution.
     *
     * Algorithm (Kahn, 1962):
     *   1. Scan all nodes; those with in_degree == 0 enter ready queue.
     *   2. Dequeue a ready node, dispatch to matched provider.
     *   3. On completion, decrement successors' in_degree (atomic).
     *      If any successor reaches 0, enqueue it.
     *   4. Repeat until queue drains.
     *   5. If dispatched_count < total_nodes → cycle detected.
     *
     * Returns a future that resolves when the entire graph completes.
     */
    std::future<nf_status> submit(uint32_t gid) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = graphs_.find(gid);
        if (it == graphs_.end()) {
            std::promise<nf_status> p;
            p.set_value(NF_ERROR_NOT_FOUND);
            return p.get_future();
        }

        auto& g = it->second;
        size_t total = g.nodes.size();
        if (total == 0) {
            std::promise<nf_status> p;
            p.set_value(NF_OK);
            return p.get_future();
        }

        // Shared completion state
        auto remaining = std::make_shared<std::atomic<uint32_t>>(
            static_cast<uint32_t>(total));
        auto graph_error = std::make_shared<std::atomic<nf_status>>(NF_OK);
        auto graph_promise = std::make_shared<std::promise<nf_status>>();
        auto graph_future = graph_promise->get_future();

        // Profiling: shared GraphProfile for this submission
        auto profile = std::make_shared<GraphProfile>();
        profile->t_submit = SteadyClock::now();
        profile->tasks.resize(total);
        for (size_t i = 0; i < total; ++i) {
            profile->tasks[i].task_id = static_cast<uint32_t>(i);
        }

        // Capture node pointers (shared ownership for async lifetime)
        auto nodes = std::make_shared<std::vector<std::unique_ptr<TaskNode>>>(
            std::move(g.nodes));

        // Kahn's step 1: seed the ready queue with zero-in-degree nodes
        std::vector<uint32_t> ready;
        for (size_t i = 0; i < total; ++i) {
            if ((*nodes)[i]->in_degree.load(std::memory_order_relaxed) == 0) {
                ready.push_back(static_cast<uint32_t>(i));
            }
        }

        // Dispatch all initially-ready tasks
        auto now = SteadyClock::now();
        for (uint32_t tid : ready) {
            profile->tasks[tid].t_enqueue = now;
            dispatch_task(nodes, tid, remaining, graph_error,
                          graph_promise, profile);
        }

        // Store profile for later retrieval (already under mu_)
        profiles_[gid] = profile;

        return graph_future;
    }

private:
    /* -- Provider Matching ------------------------------------------ */

    ProviderSlot* find_provider(nf_affinity affinity, uint32_t task_flags = 0) {
        /*
         * NF_TASK_REMOTE override: if the task carries the remote flag,
         * force routing to the network proxy provider regardless of
         * the declared affinity. This is the Step 2 evolution hook —
         * the scheduler transparently bridges to a remote node.
         */
        if (task_flags & NF_TASK_REMOTE) {
            for (auto& p : providers_) {
                if (p.affinity == NF_AFFINITY_REMOTE) return &p;
            }
            /* No remote provider registered — fall through to local */
        }

        // Exact match first
        for (auto& p : providers_) {
            if (p.affinity == affinity) return &p;
        }
        // Fallback to ANY
        if (affinity != NF_AFFINITY_ANY) {
            for (auto& p : providers_) {
                if (p.affinity == NF_AFFINITY_ANY) return &p;
            }
        }
        // Last resort: first available
        return providers_.empty() ? nullptr : &providers_[0];
    }

    /* -- Async Task Dispatch ---------------------------------------- */

    void dispatch_task(
        std::shared_ptr<std::vector<std::unique_ptr<TaskNode>>> nodes,
        uint32_t tid,
        std::shared_ptr<std::atomic<uint32_t>> remaining,
        std::shared_ptr<std::atomic<nf_status>> graph_error,
        std::shared_ptr<std::promise<nf_status>> graph_promise,
        std::shared_ptr<GraphProfile> profile = nullptr)
    {
        auto& node = (*nodes)[tid];
        ProviderSlot* prov = find_provider(node->desc.affinity,
                                           node->desc.flags);

        pool_.submit([this, nodes, tid, remaining, graph_error,
                      graph_promise, prov, profile]() -> nf_status {
            auto& node = (*nodes)[tid];
            nf_status st = NF_OK;

            // Profiling: record op_name + dispatch start
            if (profile) {
                auto& tp = profile->tasks[tid];
                std::strncpy(tp.op_name, node->desc.op_name, 63);
                tp.op_name[63] = '\0';
                tp.t_start = SteadyClock::now();
            }

            if (prov && prov->vtable.dispatch) {
                node->desc.user_data = &node->desc;

                st = prov->vtable.dispatch(
                    prov->handle,
                    node->desc.op_name,
                    node->desc.inputs,  node->desc.n_inputs,
                    node->desc.outputs, node->desc.n_outputs);

                node->desc.user_data = nullptr;
            }

            // Profiling: record dispatch end
            if (profile) {
                profile->tasks[tid].t_end = SteadyClock::now();
            }

            node->result.store(st, std::memory_order_release);

            if (st != NF_OK) {
                graph_error->store(st, std::memory_order_relaxed);
            }

            // Kahn's step 3: propagate completion to successors
            for (uint32_t succ_id : node->successors) {
                auto& succ = (*nodes)[succ_id];
                uint32_t prev = succ->in_degree.fetch_sub(
                    1, std::memory_order_acq_rel);
                if (prev == 1) {
                    // Stamp successor enqueue time
                    if (profile) {
                        profile->tasks[succ_id].t_enqueue =
                            SteadyClock::now();
                    }
                    dispatch_task(nodes, succ_id, remaining,
                                  graph_error, graph_promise, profile);
                }
            }

            // Check if entire graph is done
            uint32_t left = remaining->fetch_sub(
                1, std::memory_order_acq_rel);
            if (left == 1) {
                // Last task completed
                graph_promise->set_value(
                    graph_error->load(std::memory_order_acquire));
            }

            return st;
        });
    }

public:
    /* -- Session: Pre-compiled Execution Plan ------------------------ */

    class Session {
    public:
        Session(PipelineEngine& engine, uint32_t gid)
            : engine_(engine)
        {
            std::lock_guard<std::mutex> lk(engine_.mu_);
            auto it = engine_.graphs_.find(gid);
            if (it == engine_.graphs_.end()) return;

            // Take ownership of nodes
            nodes_ = std::make_shared<std::vector<std::unique_ptr<TaskNode>>>(
                std::move(it->second.nodes));

            size_t n = nodes_->size();
            if (n == 0) return;

            // Store initial in-degrees and pre-resolve providers
            initial_in_degrees_.resize(n);
            plan_providers_.resize(n, nullptr);
            for (size_t i = 0; i < n; ++i) {
                auto& node = (*nodes_)[i];
                initial_in_degrees_[i] = node->in_degree.load(
                    std::memory_order_relaxed);
                plan_providers_[i] = engine_.find_provider(
                    node->desc.affinity, node->desc.flags);
                if (initial_in_degrees_[i] == 0) {
                    roots_.push_back(static_cast<uint32_t>(i));
                }
            }
            valid_ = true;

            // Build name→index map for O(1) push_constants injection
            for (size_t i = 0; i < n; ++i) {
                std::string name((*nodes_)[i]->desc.op_name);
                name_to_index_[name] = static_cast<uint32_t>(i);
            }
        }

        bool valid() const { return valid_; }
        size_t num_nodes() const { return nodes_ ? nodes_->size() : 0; }

        std::future<nf_status> step() {
            if (!valid_ || !nodes_ || nodes_->empty()) {
                std::promise<nf_status> p;
                p.set_value(NF_ERROR_NOT_FOUND);
                return p.get_future();
            }

            size_t n = nodes_->size();

            // Reset in-degrees from cached initial values
            for (size_t i = 0; i < n; ++i) {
                (*nodes_)[i]->in_degree.store(
                    initial_in_degrees_[i], std::memory_order_relaxed);
                (*nodes_)[i]->result.store(NF_OK, std::memory_order_relaxed);
            }

            // Shared completion state
            auto remaining = std::make_shared<std::atomic<uint32_t>>(
                static_cast<uint32_t>(n));
            auto graph_error =
                std::make_shared<std::atomic<nf_status>>(NF_OK);
            auto graph_promise =
                std::make_shared<std::promise<nf_status>>();
            auto graph_future = graph_promise->get_future();

            // Profiling
            auto profile = std::make_shared<GraphProfile>();
            profile->t_submit = SteadyClock::now();
            profile->tasks.resize(n);
            for (size_t i = 0; i < n; ++i)
                profile->tasks[i].task_id = static_cast<uint32_t>(i);

            last_profile_ = profile;

            // Dispatch pre-computed roots
            auto now = SteadyClock::now();
            for (uint32_t tid : roots_) {
                profile->tasks[tid].t_enqueue = now;
                dispatch_step(tid, remaining, graph_error,
                              graph_promise, profile);
            }

            return graph_future;
        }

        std::shared_ptr<const GraphProfile> last_profile() const {
            return last_profile_;
        }

        /**
         * Inject push constants into a named node's task descriptor.
         * Lock-free: caller is sole accessor between step() calls.
         */
        nf_status set_push_constants(const char* node_name,
                                     const void* data, uint32_t size) {
            if (!data || size > NF_MAX_PUSH_CONSTANTS)
                return NF_ERROR_INVALID_ARG;
            auto it = name_to_index_.find(std::string(node_name));
            if (it == name_to_index_.end()) return NF_ERROR_NOT_FOUND;
            auto& desc = (*nodes_)[it->second]->desc;
            std::memcpy(desc.push_constants, data, size);
            desc.push_constants_size = size;
            return NF_OK;
        }

        /**
         * Inject push constants by node ID (not name).
         * Required when multiple nodes share the same op_name.
         */
        nf_status set_push_constants_by_id(uint32_t node_id,
                                           const void* data, uint32_t size) {
            if (!data || size > NF_MAX_PUSH_CONSTANTS)
                return NF_ERROR_INVALID_ARG;
            if (!nodes_ || node_id >= nodes_->size())
                return NF_ERROR_NOT_FOUND;
            auto& desc = (*nodes_)[node_id]->desc;
            std::memcpy(desc.push_constants, data, size);
            desc.push_constants_size = size;
            return NF_OK;
        }

    private:
        void dispatch_step(
            uint32_t tid,
            std::shared_ptr<std::atomic<uint32_t>> remaining,
            std::shared_ptr<std::atomic<nf_status>> graph_error,
            std::shared_ptr<std::promise<nf_status>> graph_promise,
            std::shared_ptr<GraphProfile> profile)
        {
            auto nodes = nodes_;
            auto* prov = plan_providers_[tid];

            engine_.pool_.submit(
                [this, nodes, tid, remaining, graph_error,
                 graph_promise, prov, profile]() -> nf_status {
                auto& node = (*nodes)[tid];
                nf_status st = NF_OK;

                if (profile) {
                    auto& tp = profile->tasks[tid];
                    std::strncpy(tp.op_name, node->desc.op_name, 63);
                    tp.op_name[63] = '\0';
                    tp.t_start = SteadyClock::now();
                }

                if (prov && prov->vtable.dispatch) {
                    node->desc.user_data = &node->desc;
                    st = prov->vtable.dispatch(
                        prov->handle, node->desc.op_name,
                        node->desc.inputs, node->desc.n_inputs,
                        node->desc.outputs, node->desc.n_outputs);
                    node->desc.user_data = nullptr;
                }

                if (profile)
                    profile->tasks[tid].t_end = SteadyClock::now();

                node->result.store(st, std::memory_order_release);
                if (st != NF_OK)
                    graph_error->store(st, std::memory_order_relaxed);

                for (uint32_t succ_id : node->successors) {
                    auto& succ = (*nodes)[succ_id];
                    uint32_t prev = succ->in_degree.fetch_sub(
                        1, std::memory_order_acq_rel);
                    if (prev == 1) {
                        if (profile)
                            profile->tasks[succ_id].t_enqueue =
                                SteadyClock::now();
                        dispatch_step(succ_id, remaining,
                                      graph_error, graph_promise, profile);
                    }
                }

                uint32_t left = remaining->fetch_sub(
                    1, std::memory_order_acq_rel);
                if (left == 1)
                    graph_promise->set_value(
                        graph_error->load(std::memory_order_acquire));

                return st;
            });
        }

        PipelineEngine& engine_;
        std::shared_ptr<std::vector<std::unique_ptr<TaskNode>>> nodes_;
        std::vector<uint32_t> initial_in_degrees_;
        std::vector<ProviderSlot*> plan_providers_;
        std::vector<uint32_t> roots_;
        std::shared_ptr<GraphProfile> last_profile_;
        std::unordered_map<std::string, uint32_t> name_to_index_;
        bool valid_ = false;
    };

private:
    /* -- Internal State --------------------------------------------- */

    struct GraphState {
        std::vector<std::unique_ptr<TaskNode>> nodes;
    };

    ThreadPool                                  pool_;
    mutable std::mutex                          mu_;
    std::vector<ProviderSlot>                   providers_;
    std::unordered_map<uint32_t, GraphState>    graphs_;
    std::unordered_map<uint32_t,
        std::shared_ptr<GraphProfile>>          profiles_;
    uint32_t                                    next_graph_id_ = 0;
    neuralOS::L4::DriverRegistry                registry_;
};

} // namespace nf

#endif // NF_PIPELINE_ENGINE_HPP
