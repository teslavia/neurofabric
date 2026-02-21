/**
 * @file ContextHub.hpp
 * @brief Phase 13 — Agent-Native Context Hub with Token-ID Radix Tree
 *
 * INTERNAL TO CORE — never crosses a dynamic library boundary.
 *
 * Compressed radix tree keyed by int32_t token-ID sequences.
 * Enables automatic KV-cache reuse across agents sharing prompt prefixes
 * (vLLM Radix Attention / SGLang style).
 *
 * Concurrency: std::shared_mutex (multiple readers, exclusive writers).
 * Eviction: LRU, TTL, refcount, radix-prefix (pluggable).
 * Memory: budget-enforced; hub holds TensorView shared refs (refcounted).
 */

#ifndef NF_CONTEXT_HUB_HPP
#define NF_CONTEXT_HUB_HPP

#include "neurofabric/TensorView.hpp"
#include "neurofabric/neuro_scheduler_abi.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace nf {

/* ================================================================== */
/*  ContextEntry — one cached tensor with metadata                     */
/* ================================================================== */

struct ContextEntry {
    using Clock = std::chrono::steady_clock;

    std::vector<int32_t> token_key;
    std::string          agent_id;
    TensorView           tensor;
    uint64_t             size_bytes  = 0;
    uint64_t             seq_id      = 0;
    uint64_t             ttl_ms      = 0;  /**< 0 = immortal (pinned) */
    Clock::time_point    created_at  = Clock::now();
    Clock::time_point    last_access = Clock::now();
    void touch() { last_access = Clock::now(); }

    bool expired() const {
        if (ttl_ms == 0) return false;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - created_at).count();
        return static_cast<uint64_t>(elapsed) >= ttl_ms;
    }
};

/* ================================================================== */
/*  RadixNode — compressed radix tree node for token-ID sequences      */
/*                                                                     */
/*  Each edge carries a span of token IDs (compressed path).           */
/*  Children are keyed by the first diverging token ID.                */
/*                                                                     */
/*  Example: insert [1,2,3,4,5] then [1,2,3,6,7]                      */
/*    root -> [1,2,3] (intermediate, no entry)                         */
/*              |-> 4: [4,5] (entry A)                                 */
/*              |-> 6: [6,7] (entry B)                                 */
/* ================================================================== */

struct RadixNode {
    std::vector<int32_t> segment;
    std::unordered_map<int32_t, std::unique_ptr<RadixNode>> children;
    std::unique_ptr<ContextEntry> entry;
};

/* ================================================================== */
/*  ContextHub — token-ID radix tree with budget-enforced caching      */
/* ================================================================== */

class ContextHub {
public:
    struct LookupResult {
        TensorView           tensor;
        std::vector<int32_t> matched_key;
        uint32_t             match_len = 0;
        bool                 found = false;
    };

    struct Stats {
        uint64_t used_bytes   = 0;
        uint64_t budget_bytes = 0;
        uint32_t entry_count  = 0;
    };

    ContextHub(uint64_t budget_bytes, nf_eviction_policy policy)
        : budget_(budget_bytes), used_(0), policy_(policy) {
        root_ = std::make_unique<RadixNode>();
    }

    ~ContextHub() = default;
    ContextHub(const ContextHub&) = delete;
    ContextHub& operator=(const ContextHub&) = delete;
    /* -- Put: insert or update a context entry ---------------------- */
    /*                                                                 */
    /* Phase 14: pre-allocate entry + leaf node OUTSIDE the exclusive  */
    /* lock to minimize critical section.  On Linux pthread_rwlock     */
    /* the writer can starve if readers continuously hold shared       */
    /* locks; keeping the exclusive window tight avoids this.          */

    nf_status put(std::span<const int32_t> token_key,
                  const std::string& agent_id,
                  TensorView tensor,
                  uint64_t ttl_ms = 0,
                  uint64_t seq_id = 0) {
        uint64_t size = tensor.valid() ? tensor.desc().size_bytes : 0;

        /* ---- Lock-free zone: heap allocations ---- */
        auto new_entry = std::make_unique<ContextEntry>();
        new_entry->token_key.assign(token_key.begin(), token_key.end());
        new_entry->agent_id   = agent_id;
        new_entry->tensor     = std::move(tensor);
        new_entry->size_bytes = size;
        new_entry->seq_id     = seq_id;
        new_entry->ttl_ms     = ttl_ms;

        /* Pre-build a candidate leaf (may not be used if path exists) */
        auto candidate_leaf = std::make_unique<RadixNode>();
        /* segment filled under lock once insertion point is known */

        /* ---- Exclusive lock: tree mutation only ---- */
        std::unique_lock<std::shared_mutex> lk(mu_);

        /* Evict until we have room */
        while (used_ + size > budget_ && used_ > 0) {
            if (!evict_one()) return NF_ERROR_OUT_OF_MEMORY;
        }

        /* Walk / build the compressed radix path */
        RadixNode* node = root_.get();
        size_t pos = 0;

        while (pos < token_key.size()) {
            int32_t next_tok = token_key[pos];
            auto it = node->children.find(next_tok);

            if (it == node->children.end()) {
                /* No child for this token — attach pre-allocated leaf */
                candidate_leaf->segment.assign(token_key.begin() + pos,
                                               token_key.end());
                RadixNode* raw = candidate_leaf.get();
                node->children[next_tok] = std::move(candidate_leaf);
                node = raw;
                pos = token_key.size();
                break;
            }

            RadixNode* child = it->second.get();
            auto& seg = child->segment;

            /* Compare child segment against remaining tokens */
            size_t remaining = token_key.size() - pos;
            size_t match_len = 0;
            size_t cmp_len = std::min(seg.size(), remaining);
            while (match_len < cmp_len &&
                   seg[match_len] == token_key[pos + match_len]) {
                ++match_len;
            }

            if (match_len == seg.size()) {
                /* Full segment match — descend */
                pos += match_len;
                node = child;
            } else {
                /* Partial match — split node at divergence point */
                split_node(node, next_tok, match_len);
                pos += match_len;
                /* After split, node->children[next_tok] is the new
                   intermediate; continue loop to create the new leaf */
                node = node->children[next_tok].get();
            }
        }

        /* Place entry at current node */
        if (node->entry) {
            /* Update existing — reclaim old size */
            used_ -= node->entry->size_bytes;
            all_entries_.erase(
                std::remove(all_entries_.begin(), all_entries_.end(), node),
                all_entries_.end());
        }

        node->entry = std::move(new_entry);
        all_entries_.push_back(node);
        used_ += size;

        return NF_OK;
    }
    /* -- Get: longest-prefix match lookup ----------------------------- */

    LookupResult get(std::span<const int32_t> token_prefix) {
        std::shared_lock<std::shared_mutex> lk(mu_);

        LookupResult result;
        RadixNode* node = root_.get();
        size_t pos = 0;
        RadixNode* best = nullptr;
        size_t best_pos = 0;

        /* Check root entry */
        if (node->entry && !node->entry->expired()) {
            best = node;
            best_pos = 0;
        }

        while (pos < token_prefix.size()) {
            int32_t next_tok = token_prefix[pos];
            auto it = node->children.find(next_tok);
            if (it == node->children.end()) break;

            RadixNode* child = it->second.get();
            auto& seg = child->segment;

            /* Compare segment against remaining tokens */
            size_t remaining = token_prefix.size() - pos;
            size_t cmp_len = std::min(seg.size(), remaining);
            size_t match_len = 0;
            while (match_len < cmp_len &&
                   seg[match_len] == token_prefix[pos + match_len]) {
                ++match_len;
            }

            if (match_len < seg.size()) {
                /* Partial segment match — can't descend further */
                break;
            }

            /* Full segment match */
            pos += match_len;
            node = child;

            if (node->entry && !node->entry->expired()) {
                best = node;
                best_pos = pos;
            }
        }

        if (best && best->entry) {
            best->entry->touch();
            result.tensor      = best->entry->tensor.share();
            result.matched_key = best->entry->token_key;
            result.match_len   = static_cast<uint32_t>(best_pos);
            result.found       = true;
        }
        return result;
    }

    /* -- Evict: remove entries by prefix or clear all ----------------- */

    nf_status evict(std::span<const int32_t> token_prefix = {}) {
        std::unique_lock<std::shared_mutex> lk(mu_);

        if (token_prefix.empty()) {
            /* Clear entire tree */
            reclaim_subtree(root_.get());
            root_ = std::make_unique<RadixNode>();
            return NF_OK;
        }

        /* Walk to the node matching the prefix */
        RadixNode* parent = nullptr;
        int32_t parent_key = 0;
        RadixNode* node = root_.get();
        size_t pos = 0;

        while (pos < token_prefix.size()) {
            int32_t next_tok = token_prefix[pos];
            auto it = node->children.find(next_tok);
            if (it == node->children.end()) return NF_ERROR_NOT_FOUND;

            RadixNode* child = it->second.get();
            auto& seg = child->segment;
            size_t remaining = token_prefix.size() - pos;
            size_t cmp_len = std::min(seg.size(), remaining);
            size_t match_len = 0;
            while (match_len < cmp_len &&
                   seg[match_len] == token_prefix[pos + match_len]) {
                ++match_len;
            }
            if (match_len < seg.size()) return NF_ERROR_NOT_FOUND;

            pos += match_len;
            parent = node;
            parent_key = next_tok;
            node = child;
        }

        /* Reclaim subtree and remove from parent */
        reclaim_subtree(node);
        if (parent) {
            parent->children.erase(parent_key);
        }
        return NF_OK;
    }

    /* -- Stats -------------------------------------------------------- */

    Stats stats() const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        Stats s;
        s.used_bytes   = used_;
        s.budget_bytes = budget_;
        s.entry_count  = static_cast<uint32_t>(all_entries_.size());
        return s;
    }
private:

    /* -- Split a child node at offset k within its segment ------------ */
    /*                                                                    */
    /*  Before: parent -> children[tok] -> child{seg=[A,B,C,D], ...}     */
    /*  After:  parent -> children[tok] -> mid{seg=[A,B]}                */
    /*                                       -> children[C] -> old{seg=[C,D], ...} */

    void split_node(RadixNode* parent, int32_t child_key, size_t k) {
        auto& child_ptr = parent->children[child_key];
        RadixNode* old_child = child_ptr.get();

        /* Create intermediate node with prefix segment */
        auto mid = std::make_unique<RadixNode>();
        mid->segment.assign(old_child->segment.begin(),
                            old_child->segment.begin() + k);

        /* Shorten old child's segment to suffix */
        int32_t diverge_tok = old_child->segment[k];
        old_child->segment.erase(old_child->segment.begin(),
                                  old_child->segment.begin() + k);

        /* Move old child under intermediate */
        mid->children[diverge_tok] = std::move(child_ptr);

        /* Replace parent's child with intermediate */
        child_ptr = std::move(mid);
    }

    /* -- Evict one entry according to policy -------------------------- */

    bool evict_one() {
        if (all_entries_.empty()) return false;

        RadixNode* victim = nullptr;

        switch (policy_) {
        case NF_EVICT_TTL: {
            for (auto* node : all_entries_) {
                if (node->entry && node->entry->expired()) {
                    victim = node; break;
                }
            }
            if (!victim) {
                /* Fall through to LRU if no expired entries */
                goto lru_fallback;
            }
            break;
        }
        lru_fallback:
        case NF_EVICT_LRU: {
            ContextEntry::Clock::time_point oldest =
                ContextEntry::Clock::time_point::max();
            for (auto* node : all_entries_) {
                if (node->entry && node->entry->last_access < oldest) {
                    oldest = node->entry->last_access;
                    victim = node;
                }
            }
            break;
        }
        case NF_EVICT_REFCOUNT: {
            for (auto* node : all_entries_) {
                if (node->entry && node->entry->tensor.valid()) {
                    nf_buffer_info bi = node->entry->tensor.info();
                    if (bi.refcount <= 1) {
                        victim = node; break;
                    }
                }
            }
            /* If all entries have external refs, evict LRU */
            if (!victim) goto lru_fallback;
            break;
        }
        case NF_EVICT_RADIX_PREFIX: {
            /* Evict entry with longest token sequence (deepest in tree) */
            size_t max_depth = 0;
            for (auto* node : all_entries_) {
                if (node->entry) {
                    size_t depth = node->entry->token_key.size();
                    if (depth >= max_depth) {
                        max_depth = depth;
                        victim = node;
                    }
                }
            }
            break;
        }
        }

        if (!victim || !victim->entry) return false;

        used_ -= victim->entry->size_bytes;
        victim->entry.reset();
        all_entries_.erase(
            std::remove(all_entries_.begin(), all_entries_.end(), victim),
            all_entries_.end());
        return true;
    }

    /* -- Subtree reclamation ------------------------------------------ */

    void reclaim_subtree(RadixNode* node) {
        if (!node) return;
        if (node->entry) {
            used_ -= node->entry->size_bytes;
            all_entries_.erase(
                std::remove(all_entries_.begin(), all_entries_.end(), node),
                all_entries_.end());
            node->entry.reset();
        }
        for (auto& [tok, child] : node->children) {
            reclaim_subtree(child.get());
        }
        node->children.clear();
    }

    /* -- State -------------------------------------------------------- */

    std::unique_ptr<RadixNode>      root_;
    std::vector<RadixNode*>         all_entries_;
    uint64_t                        budget_;
    uint64_t                        used_;
    nf_eviction_policy              policy_;
    mutable std::shared_mutex       mu_;
};

} // namespace nf

#endif // NF_CONTEXT_HUB_HPP
