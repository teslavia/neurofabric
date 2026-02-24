/**
 * @file async_dataflow.hpp
 * @brief NeuralOS mesh — Pathways-Style Async Dataflow
 *
 * Phase 38.2: Control/data plane separation for distributed execution.
 *   - DataHandle: cross-node async data placeholder (Future-like)
 *   - ControlPlane: dispatch topology + DataHandles ahead of data
 *   - DataPlane: actual tensor transfer via TransportOps
 */

#ifndef NEURALOS_MESH_ASYNC_DATAFLOW_HPP
#define NEURALOS_MESH_ASYNC_DATAFLOW_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuralOS { namespace mesh {

/* ================================================================== */
/*  DataHandle — async data placeholder (cross-node Future)            */
/* ================================================================== */

struct DataHandle {
    uint64_t handle_id   = 0;
    uint32_t src_node    = 0;
    uint32_t dst_node    = 0;
    uint64_t size_bytes  = 0;
    std::string tensor_name;

    std::atomic<bool> resolved{false};
    void* data_ptr = nullptr;  /* set when resolved */

    using Callback = std::function<void(DataHandle*)>;
    Callback on_resolve;
};

/* ================================================================== */
/*  ControlPlane — pre-dispatch topology + DataHandles                 */
/* ================================================================== */

class ControlPlane {
public:
    /** Create a DataHandle for a future tensor transfer.
     *  Returns handle_id. */
    uint64_t dispatch_async(uint32_t src_node, uint32_t dst_node,
                            const std::string& tensor_name,
                            uint64_t size_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        auto h = std::make_shared<DataHandle>();
        h->handle_id = next_id_++;
        h->src_node = src_node;
        h->dst_node = dst_node;
        h->tensor_name = tensor_name;
        h->size_bytes = size_bytes;
        handles_[h->handle_id] = h;
        return h->handle_id;
    }

    /** Resolve a DataHandle (data is now available) */
    bool resolve(uint64_t handle_id, void* data_ptr) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = handles_.find(handle_id);
        if (it == handles_.end()) return false;
        auto& h = it->second;
        h->data_ptr = data_ptr;
        h->resolved.store(true, std::memory_order_release);
        if (h->on_resolve) h->on_resolve(h.get());
        return true;
    }

    /** Check if a handle is resolved */
    bool is_resolved(uint64_t handle_id) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = handles_.find(handle_id);
        if (it == handles_.end()) return false;
        return it->second->resolved.load(std::memory_order_acquire);
    }

    /** Gang schedule: wait until all handles in the set are resolved.
     *  Returns true if all resolved, false if any missing. */
    bool gang_schedule(const std::vector<uint64_t>& handle_ids) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto id : handle_ids) {
            auto it = handles_.find(id);
            if (it == handles_.end()) return false;
            if (!it->second->resolved.load(std::memory_order_acquire))
                return false;
        }
        return true;
    }

    /** Set callback for when a handle resolves */
    void on_resolve(uint64_t handle_id, DataHandle::Callback cb) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = handles_.find(handle_id);
        if (it != handles_.end())
            it->second->on_resolve = std::move(cb);
    }

    uint32_t num_pending() const {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t count = 0;
        for (auto& [id, h] : handles_)
            if (!h->resolved.load(std::memory_order_acquire)) ++count;
        return count;
    }

    uint32_t num_handles() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(handles_.size());
    }

private:
    mutable std::mutex mu_;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, std::shared_ptr<DataHandle>> handles_;
};

/* ================================================================== */
/*  DataPlane — actual tensor data transfer                            */
/* ================================================================== */

class DataPlane {
public:
    using TransportCallback = std::function<bool(uint32_t src, uint32_t dst,
                                                  const void* data, uint64_t size)>;
    using PrefetchCallback = std::function<void(uint32_t src, uint32_t dst,
                                                 const std::string& tensor_name)>;

    void set_transport(TransportCallback cb) { transport_cb_ = std::move(cb); }
    void set_prefetch(PrefetchCallback cb) { prefetch_cb_ = std::move(cb); }

    /** Transfer tensor data. Uses callback if set, else increments counter. */
    bool transfer(uint32_t src_node, uint32_t dst_node,
                  const void* data, uint64_t size_bytes) {
        if (transport_cb_) {
            bool ok = transport_cb_(src_node, dst_node, data, size_bytes);
            if (ok) ++transfers_;
            return ok;
        }
        ++transfers_;
        return true;
    }

    /** Prefetch: hint that data will be needed soon */
    void prefetch(uint32_t src_node, uint32_t dst_node,
                  const std::string& tensor_name) {
        if (prefetch_cb_) {
            prefetch_cb_(src_node, dst_node, tensor_name);
        }
        ++prefetches_;
    }

    uint32_t num_transfers() const { return transfers_; }
    uint32_t num_prefetches() const { return prefetches_; }

private:
    TransportCallback transport_cb_;
    PrefetchCallback  prefetch_cb_;
    uint32_t transfers_  = 0;
    uint32_t prefetches_ = 0;
};

}} // namespace neuralOS::mesh

// Backward compatibility
namespace neuralOS { namespace L5 {
    using neuralOS::mesh::DataHandle;
    using neuralOS::mesh::ControlPlane;
    using neuralOS::mesh::DataPlane;
}}

#endif // NEURALOS_MESH_ASYNC_DATAFLOW_HPP
