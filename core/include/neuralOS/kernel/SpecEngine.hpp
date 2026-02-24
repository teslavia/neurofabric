/**
 * @file SpecEngine.hpp
 * @brief NeuralOS kernel — Speculative Execution Engine (Tree Search)
 *
 * Phase 36.4: Multi-branch speculation tree replacing linear chain.
 *   - SpecTree: N-ary tree of speculated token sequences
 *   - branch(): fork multiple execution paths at verification points
 *   - verify(): parallel verification, returns longest matching prefix
 *   - rollback(): discard uncommitted speculation blocks (links vMMU)
 *   - checkpoint() / restore(): state machine snapshots
 *
 * Header-only. Operates on SpeculativeConfig from model_config.hpp.
 */

#ifndef NEURALOS_KERNEL_SPECENGINE_HPP
#define NEURALOS_KERNEL_SPECENGINE_HPP

#include "neuralOS/kernel/vMMU.hpp"
#include "neuralOS/kernel/request_scheduler.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace neuralOS { namespace kernel {

/* ================================================================== */
/*  SpecNode — one node in the speculation tree                        */
/* ================================================================== */

struct SpecNode {
    int32_t  token_id    = -1;
    uint32_t depth       = 0;
    uint32_t seq_idx     = UINT32_MAX;  /* vMMU sequence for this branch */
    float    log_prob    = 0.0f;
    bool     verified    = false;
    bool     committed   = false;

    std::vector<std::unique_ptr<SpecNode>> children;
    SpecNode* parent = nullptr;

    uint32_t num_children() const {
        return static_cast<uint32_t>(children.size());
    }
};

/* ================================================================== */
/*  SpecTree — multi-branch speculation tree                           */
/* ================================================================== */

struct SpecTree {
    std::unique_ptr<SpecNode> root;
    uint32_t max_width = 1;
    uint32_t max_depth = 8;
    uint32_t total_nodes = 0;

    SpecTree(uint32_t width = 1, uint32_t depth = 8)
        : max_width(width), max_depth(depth) {
        root = std::make_unique<SpecNode>();
        root->depth = 0;
        root->verified = true;
        root->committed = true;
        total_nodes = 1;
    }

    /** Add a child token to a parent node. Returns the new node. */
    SpecNode* add_child(SpecNode* parent, int32_t token_id, float log_prob = 0.0f) {
        if (!parent) return nullptr;
        if (parent->depth >= max_depth) return nullptr;
        if (parent->num_children() >= max_width) return nullptr;

        auto child = std::make_unique<SpecNode>();
        child->token_id = token_id;
        child->depth = parent->depth + 1;
        child->log_prob = log_prob;
        child->parent = parent;
        SpecNode* raw = child.get();
        parent->children.push_back(std::move(child));
        ++total_nodes;
        return raw;
    }

    /** Collect all leaf nodes */
    void collect_leaves(SpecNode* node, std::vector<SpecNode*>& out) const {
        if (!node) return;
        if (node->children.empty()) {
            out.push_back(node);
            return;
        }
        for (auto& c : node->children)
            collect_leaves(c.get(), out);
    }

    /** Get path from root to a node (token sequence) */
    static std::vector<int32_t> path_to(const SpecNode* node) {
        std::vector<int32_t> path;
        const SpecNode* cur = node;
        while (cur && cur->token_id >= 0) {
            path.push_back(cur->token_id);
            cur = cur->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    /** Count total nodes in tree */
    static uint32_t count_nodes(const SpecNode* node) {
        if (!node) return 0;
        uint32_t c = 1;
        for (auto& ch : node->children)
            c += count_nodes(ch.get());
        return c;
    }

    /** Prune: remove all children of a node */
    static uint32_t prune_children(SpecNode* node) {
        if (!node) return 0;
        uint32_t removed = count_nodes(node) - 1;
        node->children.clear();
        return removed;
    }
};

/* ================================================================== */
/*  Checkpoint — serialized state for rollback                         */
/* ================================================================== */

struct SpecCheckpoint {
    uint32_t seq_idx       = UINT32_MAX;
    uint32_t num_tokens    = 0;
    uint32_t num_blocks    = 0;
    std::vector<int32_t> committed_tokens;
};

/* ================================================================== */
/*  SpecEngine — orchestrates tree-based speculative decoding          */
/* ================================================================== */

class SpecEngine {
public:
    SpecEngine(const nf::SpeculativeConfig& cfg)
        : cfg_(cfg)
        , tree_(cfg.tree_width, cfg.max_depth) {}

    /* ---- Cross-Layer Integration ---------------------------------- */

    void set_vmmu(neuralOS::kernel::vMMU* v, nf::PagedKVCache* kv) {
        vmmu_ = v;
        kv_for_cleanup_ = kv;
    }

    neuralOS::kernel::vMMU* vmmu() const { return vmmu_; }
    uint32_t freed_sequences() const { return freed_sequences_; }

    /* ---- Branch --------------------------------------------------- */

    /** At a verification point, branch into multiple speculation paths.
     *  Returns the new leaf nodes created. */
    std::vector<SpecNode*> branch(SpecNode* parent,
                                   const std::vector<int32_t>& candidate_tokens,
                                   const std::vector<float>& log_probs = {}) {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<SpecNode*> leaves;
        uint32_t n = std::min(static_cast<uint32_t>(candidate_tokens.size()),
                              cfg_.tree_width);
        for (uint32_t i = 0; i < n; ++i) {
            float lp = (i < log_probs.size()) ? log_probs[i] : 0.0f;
            auto* node = tree_.add_child(parent, candidate_tokens[i], lp);
            if (node) leaves.push_back(node);
        }
        return leaves;
    }

    /* ---- Verify --------------------------------------------------- */

    /** Verify speculated tokens against target model outputs.
     *  Returns the longest accepted prefix path. */
    struct VerifyResult {
        std::vector<int32_t> accepted_tokens;
        uint32_t             accepted_count = 0;
        SpecNode*            last_accepted  = nullptr;
    };

    VerifyResult verify(SpecNode* spec_root,
                        const int32_t* target_tokens,
                        uint32_t n_target) {
        std::lock_guard<std::mutex> lk(mu_);
        VerifyResult best;

        /* DFS: find the longest matching path */
        std::vector<SpecNode*> leaves;
        tree_.collect_leaves(spec_root, leaves);

        for (auto* leaf : leaves) {
            auto path = SpecTree::path_to(leaf);
            uint32_t match = 0;
            uint32_t limit = std::min(static_cast<uint32_t>(path.size()), n_target);
            while (match < limit && path[match] == target_tokens[match])
                ++match;

            if (match > best.accepted_count) {
                best.accepted_tokens.assign(path.begin(), path.begin() + match);
                best.accepted_count = match;
                /* Walk back to find last accepted node */
                SpecNode* cur = leaf;
                while (cur && cur->depth > match) cur = cur->parent;
                best.last_accepted = cur;
            }
        }

        /* Mark accepted nodes as verified */
        if (best.last_accepted) {
            SpecNode* cur = best.last_accepted;
            while (cur) {
                cur->verified = true;
                cur = cur->parent;
            }
        }
        return best;
    }

    /* ---- Rollback ------------------------------------------------- */

    /** Discard all unverified speculation branches.
     *  Returns number of nodes pruned. */
    uint32_t rollback(SpecNode* keep_until = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!keep_until) keep_until = tree_.root.get();

        /* Collect seq_idx from nodes about to be pruned */
        std::vector<uint32_t> pruned_seqs;
        collect_pruned_seqs(tree_.root.get(), pruned_seqs);

        uint32_t pruned = 0;
        /* Keep only the path to keep_until, prune everything else */
        prune_unverified(tree_.root.get(), pruned);
        tree_.total_nodes -= pruned;

        /* Cross-layer: free KV sequences for pruned branches */
        if (kv_for_cleanup_) {
            for (uint32_t seq : pruned_seqs) {
                if (seq != UINT32_MAX) {
                    kv_for_cleanup_->free_sequence(seq);
                    ++freed_sequences_;
                }
            }
        }
        return pruned;
    }

    /* ---- Checkpoint / Restore ------------------------------------- */

    /** Save current committed state */
    SpecCheckpoint checkpoint(uint32_t seq_idx, uint32_t num_tokens,
                               uint32_t num_blocks) {
        std::lock_guard<std::mutex> lk(mu_);
        SpecCheckpoint cp;
        cp.seq_idx = seq_idx;
        cp.num_tokens = num_tokens;
        cp.num_blocks = num_blocks;

        /* Collect committed token path */
        collect_committed(tree_.root.get(), cp.committed_tokens);
        checkpoints_.push_back(cp);
        return cp;
    }

    /** Restore from a checkpoint (returns the checkpoint data) */
    bool restore(SpecCheckpoint* out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (checkpoints_.empty()) return false;
        *out = checkpoints_.back();
        checkpoints_.pop_back();

        /* Reset tree to root */
        tree_.root->children.clear();
        tree_.total_nodes = 1;
        return true;
    }

    /* ---- Queries -------------------------------------------------- */

    SpecTree& tree() { return tree_; }
    const SpecTree& tree() const { return tree_; }

    uint32_t num_checkpoints() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(checkpoints_.size());
    }

    const nf::SpeculativeConfig& config() const { return cfg_; }

private:
    void prune_unverified(SpecNode* node, uint32_t& pruned) {
        if (!node) return;
        auto it = node->children.begin();
        while (it != node->children.end()) {
            if (!(*it)->verified) {
                pruned += SpecTree::count_nodes(it->get());
                it = node->children.erase(it);
            } else {
                prune_unverified(it->get(), pruned);
                ++it;
            }
        }
    }

    /** Collect seq_idx from unverified nodes (about to be pruned) */
    void collect_pruned_seqs(const SpecNode* node,
                             std::vector<uint32_t>& out) const {
        if (!node) return;
        for (auto& c : node->children) {
            if (!c->verified) {
                collect_seqs_recursive(c.get(), out);
            } else {
                collect_pruned_seqs(c.get(), out);
            }
        }
    }

    void collect_seqs_recursive(const SpecNode* node,
                                std::vector<uint32_t>& out) const {
        if (!node) return;
        if (node->seq_idx != UINT32_MAX)
            out.push_back(node->seq_idx);
        for (auto& c : node->children)
            collect_seqs_recursive(c.get(), out);
    }

    void collect_committed(const SpecNode* node,
                           std::vector<int32_t>& out) const {
        if (!node) return;
        if (node->token_id >= 0 && node->committed)
            out.push_back(node->token_id);
        for (auto& c : node->children)
            collect_committed(c.get(), out);
    }

    nf::SpeculativeConfig          cfg_;
    SpecTree                       tree_;
    mutable std::mutex             mu_;
    std::vector<SpecCheckpoint>    checkpoints_;

    neuralOS::kernel::vMMU*            vmmu_ = nullptr;
    nf::PagedKVCache*              kv_for_cleanup_ = nullptr;
    uint32_t                       freed_sequences_ = 0;
};

}} // namespace neuralOS::kernel

// Backward compatibility
namespace neuralOS { namespace L2 {
    using neuralOS::kernel::SpecNode;
    using neuralOS::kernel::SpecTree;
    using neuralOS::kernel::SpecCheckpoint;
    using neuralOS::kernel::SpecEngine;
}}

#endif // NEURALOS_KERNEL_SPECENGINE_HPP
