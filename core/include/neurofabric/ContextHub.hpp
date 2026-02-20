/**
 * @file ContextHub.hpp
 * @brief Agent-Native Context Hub — Global State Router & Cache
 *
 * INTERNAL TO CORE — never crosses a dynamic library boundary.
 *
 * The ContextHub is the Step 3 evolution of Neuro-Fabric: it transforms
 * the system from a "model inference engine" into a "multi-agent state
 * router". Key design principles:
 *
 *   1. Every context entry is a named TensorView with a hierarchical
 *      key (e.g. "agent/planner/kv_cache/layer_0/head_3").
 *
 *   2. Lookup uses longest-prefix matching on the key, inspired by
 *      vLLM's Radix Attention / SGLang's RadixTree. This enables
 *      automatic KV-cache reuse across agents sharing prompt prefixes.
 *
 *   3. Eviction is pluggable: LRU, TTL-based, refcount-based, or
 *      radix-prefix deduplication.
 *
 *   4. All tensor data is managed via Phase 2's TensorView (RAII,
 *      refcounted, zero-copy slicing). The hub never copies data —
 *      it only holds shared references.
 *
 * Memory budget enforcement:
 *   The hub tracks total size_bytes of all held TensorViews.
 *   When a put() would exceed the budget, the eviction policy runs
 *   until enough space is reclaimed or the put fails.
 */

#ifndef NF_CONTEXT_HUB_HPP
#define NF_CONTEXT_HUB_HPP

#include "neurofabric/TensorView.hpp"
#include "neurofabric/neuro_scheduler_abi.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nf {

/* ================================================================== */
/*  ContextEntry — one cached tensor with metadata                     */
/* ================================================================== */

struct ContextEntry {
    std::string     key;
    std::string     agent_id;
    TensorView      tensor;
    uint64_t        size_bytes  = 0;
    uint64_t        seq_id      = 0;
    uint64_t        ttl_ms      = 0;     /**< 0 = immortal (pinned) */

    using Clock = std::chrono::steady_clock;
    Clock::time_point created_at = Clock::now();
    Clock::time_point last_access = Clock::now();

    void touch() { last_access = Clock::now(); }

    bool expired() const {
        if (ttl_ms == 0) return false;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - created_at).count();
        return static_cast<uint64_t>(elapsed) >= ttl_ms;
    }
};

/* ================================================================== */
/*  RadixNode — trie node for hierarchical key prefix matching         */
/*                                                                     */
/*  Keys are split on '/' separators. Each node holds an optional      */
/*  ContextEntry. Lookup walks the trie and returns the deepest node   */
/*  that has an entry (longest prefix match).                          */
/*                                                                     */
/*  Example tree for keys:                                             */
/*    "agent/planner/kv/L0"                                            */
/*    "agent/planner/kv/L1"                                            */
/*    "agent/vision/frame"                                             */
/*                                                                     */
/*    root -> "agent" -> "planner" -> "kv" -> "L0" [entry]             */
/*                                         -> "L1" [entry]             */
/*                    -> "vision"  -> "frame" [entry]                  */
/* ================================================================== */

struct RadixNode {
    std::string segment;
    std::unordered_map<std::string, std::unique_ptr<RadixNode>> children;
    std::unique_ptr<ContextEntry> entry;
};

/* ================================================================== */
/*  ContextHub — the global agent state router                         */
/* ================================================================== */

class ContextHub {
public:
    ContextHub(uint64_t budget_bytes, nf_eviction_policy policy)
        : budget_(budget_bytes), used_(0), policy_(policy) {
        root_ = std::make_unique<RadixNode>();
    }

    ~ContextHub() = default;

    ContextHub(const ContextHub&) = delete;
    ContextHub& operator=(const ContextHub&) = delete;

    /* -- Put: insert or update a context entry ---------------------- */

    nf_status put(const std::string& key,
                  const std::string& agent_id,
                  TensorView tensor,
                  uint64_t ttl_ms = 0,
                  uint64_t seq_id = 0) {
        std::lock_guard<std::mutex> lk(mu_);

        uint64_t size = tensor.valid() ? tensor.desc().size_bytes : 0;

        // Evict until we have room
        while (used_ + size > budget_ && used_ > 0) {
            if (!evict_one()) {
                return NF_ERROR_OUT_OF_MEMORY;
            }
        }

        auto segments = split_key(key);
        RadixNode* node = root_.get();

        for (const auto& seg : segments) {
            auto it = node->children.find(seg);
            if (it == node->children.end()) {
                auto child = std::make_unique<RadixNode>();
                child->segment = seg;
                auto* raw = child.get();
                node->children[seg] = std::move(child);
                node = raw;
            } else {
                node = it->second.get();
            }
        }

        // If replacing, reclaim old size
        if (node->entry) {
            used_ -= node->entry->size_bytes;
            all_entries_.erase(
                std::remove(all_entries_.begin(), all_entries_.end(), node),
                all_entries_.end());
        }

        auto entry = std::make_unique<ContextEntry>();
        entry->key        = key;
        entry->agent_id   = agent_id;
        entry->tensor     = std::move(tensor);
        entry->size_bytes = size;
        entry->ttl_ms     = ttl_ms;
        entry->seq_id     = seq_id;
        node->entry = std::move(entry);

        all_entries_.push_back(node);
        used_ += size;
        return NF_OK;
    }

    /* -- Get: longest-prefix match lookup --------------------------- */

    struct LookupResult {
        TensorView  tensor;   /**< Shared ref (retained) */
        std::string matched_key;
        bool        found = false;
    };

    LookupResult get(const std::string& key_prefix) {
        std::lock_guard<std::mutex> lk(mu_);

        auto segments = split_key(key_prefix);
        RadixNode* node = root_.get();
        RadixNode* best = nullptr;

        for (const auto& seg : segments) {
            auto it = node->children.find(seg);
            if (it == node->children.end()) break;
            node = it->second.get();
            if (node->entry && !node->entry->expired()) {
                best = node;
            }
        }

        LookupResult result;
        if (best && best->entry) {
            best->entry->touch();
            result.tensor      = best->entry->tensor.share();
            result.matched_key = best->entry->key;
            result.found       = true;
        }
        return result;
    }

    /* -- Evict: remove entries by prefix ---------------------------- */

    nf_status evict(const std::string& key_prefix) {
        std::lock_guard<std::mutex> lk(mu_);

        if (key_prefix.empty()) {
            // Evict everything
            root_ = std::make_unique<RadixNode>();
            all_entries_.clear();
            used_ = 0;
            return NF_OK;
        }

        auto segments = split_key(key_prefix);
        RadixNode* node = root_.get();
        RadixNode* parent = nullptr;
        std::string last_seg;

        for (const auto& seg : segments) {
            auto it = node->children.find(seg);
            if (it == node->children.end()) return NF_ERROR_NOT_FOUND;
            parent = node;
            last_seg = seg;
            node = it->second.get();
        }

        // Reclaim all entries under this subtree
        reclaim_subtree(node);
        if (parent) {
            parent->children.erase(last_seg);
        }
        return NF_OK;
    }

    /* -- Stats ------------------------------------------------------ */

    struct Stats {
        uint64_t used_bytes   = 0;
        uint64_t budget_bytes = 0;
        uint32_t entry_count  = 0;
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        Stats s;
        s.used_bytes   = used_;
        s.budget_bytes = budget_;
        s.entry_count  = static_cast<uint32_t>(all_entries_.size());
        return s;
    }

private:
    /* -- Key splitting ---------------------------------------------- */

    static std::vector<std::string> split_key(const std::string& key) {
        std::vector<std::string> parts;
        size_t start = 0;
        while (start < key.size()) {
            size_t pos = key.find('/', start);
            if (pos == std::string::npos) {
                parts.push_back(key.substr(start));
                break;
            }
            if (pos > start) {
                parts.push_back(key.substr(start, pos - start));
            }
            start = pos + 1;
        }
        return parts;
    }

    /* -- Eviction strategies ---------------------------------------- */

    bool evict_one() {
        if (all_entries_.empty()) return false;

        RadixNode* victim = nullptr;

        switch (policy_) {
        case NF_EVICT_TTL:
            // Evict first expired entry
            for (auto* node : all_entries_) {
                if (node->entry && node->entry->expired()) {
                    victim = node;
                    break;
                }
            }
            if (!victim) {
                // Fall through to LRU if nothing expired
                [[fallthrough]];
            } else {
                break;
            }

        case NF_EVICT_LRU: {
            // Evict least recently accessed (skip pinned: ttl_ms == 0)
            auto oldest_time = ContextEntry::Clock::time_point::max();
            for (auto* node : all_entries_) {
                if (node->entry && node->entry->ttl_ms != 0 &&
                    node->entry->last_access < oldest_time) {
                    oldest_time = node->entry->last_access;
                    victim = node;
                }
            }
            // If all are pinned, evict the oldest anyway
            if (!victim && !all_entries_.empty()) {
                for (auto* node : all_entries_) {
                    if (node->entry && node->entry->last_access < oldest_time) {
                        oldest_time = node->entry->last_access;
                        victim = node;
                    }
                }
            }
            break;
        }

        case NF_EVICT_REFCOUNT:
            // Evict entries whose tensor has refcount == 1 (only hub holds it)
            for (auto* node : all_entries_) {
                if (node->entry && node->entry->tensor.valid()) {
                    auto bi = node->entry->tensor.info();
                    if (bi.refcount <= 1) {
                        victim = node;
                        break;
                    }
                }
            }
            break;

        case NF_EVICT_RADIX_PREFIX:
            // Evict the entry with the longest key (deepest in trie)
            // This preferentially removes fine-grained slices first
            {
                size_t max_depth = 0;
                for (auto* node : all_entries_) {
                    if (node->entry) {
                        size_t depth = std::count(
                            node->entry->key.begin(),
                            node->entry->key.end(), '/');
                        if (depth >= max_depth) {
                            max_depth = depth;
                            victim = node;
                        }
                    }
                }
            }
            break;
        }

        if (!victim || !victim->entry) return false;

        used_ -= victim->entry->size_bytes;
        victim->entry.reset();
        all_entries_.erase(
            std::remove(all_entries_.begin(), all_entries_.end(), victim),
            all_entries_.end());
        return true;
    }

    /* -- Subtree reclamation ---------------------------------------- */

    void reclaim_subtree(RadixNode* node) {
        if (!node) return;
        if (node->entry) {
            used_ -= node->entry->size_bytes;
            all_entries_.erase(
                std::remove(all_entries_.begin(), all_entries_.end(), node),
                all_entries_.end());
            node->entry.reset();
        }
        for (auto& [seg, child] : node->children) {
            reclaim_subtree(child.get());
        }
        node->children.clear();
    }

    /* -- State ------------------------------------------------------ */

    std::unique_ptr<RadixNode>      root_;
    std::vector<RadixNode*>         all_entries_;  /**< Flat index for eviction scan */
    uint64_t                        budget_;
    uint64_t                        used_;
    nf_eviction_policy              policy_;
    mutable std::mutex              mu_;
};

} // namespace nf

#endif // NF_CONTEXT_HUB_HPP
