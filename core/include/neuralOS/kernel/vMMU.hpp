/**
 * @file vMMU.hpp
 * @brief NeuralOS L2 — Virtual Memory Management Unit
 *
 * Phase 36.2: Wraps PagedKVCache + ContextHub with:
 *   - CoW fork_sequence() for beam search / parallel sampling
 *   - cow_write() triggers physical copy on shared blocks
 *   - page_out() / page_in() for cold block eviction to host memory
 *   - radix_prefix_share() for ContextHub prefix → CoW sharing
 *   - PressureCallback for memory pressure notifications
 *
 * Header-only. All state lives in the wrapped PagedKVCache + ContextHub.
 */

#ifndef NEURALOS_L2_VMMU_HPP
#define NEURALOS_L2_VMMU_HPP

#include "neuralOS/kernel/ContextHub.hpp"
#include "neuralOS/kernel/kv_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neuralOS { namespace L2 {

/* ================================================================== */
/*  PressureCallback — invoked when memory usage exceeds threshold     */
/* ================================================================== */

using PressureCallback = std::function<void(uint64_t used_bytes,
                                            uint64_t budget_bytes)>;

/* ================================================================== */
/*  PageOutEntry — host-side storage for evicted KV blocks             */
/* ================================================================== */

struct PageOutEntry {
    uint32_t seq_id        = 0;
    uint32_t logical_block = 0;
    uint32_t original_phys = 0;
    uint32_t num_filled    = 0;
    std::vector<uint8_t> data;  /* host-side copy of KV block data */
};

/* ================================================================== */
/*  vMMU — Virtual Memory Management Unit                              */
/* ================================================================== */

class vMMU {
public:
    struct Config {
        float    pressure_threshold = 0.85f;  /* trigger callback at 85% usage */
        uint32_t page_out_batch     = 4;      /* blocks to evict per pressure event */
    };

    vMMU(nf::PagedKVCache* kv, nf::ContextHub* hub, Config cfg)
        : kv_(kv), hub_(hub), cfg_(cfg) {}

    vMMU(nf::PagedKVCache* kv, nf::ContextHub* hub)
        : vMMU(kv, hub, Config{}) {}

    /* ---- CoW Fork ------------------------------------------------- */

    /** Fork a sequence for beam search / parallel sampling.
     *  All blocks are shared (ref_count bumped). Returns new seq index. */
    uint32_t fork_sequence(uint32_t src_seq_idx) {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t dst = kv_->fork_sequence(src_seq_idx);
        if (dst != UINT32_MAX) {
            fork_parents_[dst] = src_seq_idx;
        }
        return dst;
    }

    /** Write to a block in a sequence. If shared, triggers CoW copy.
     *  Returns physical block index suitable for writing. */
    uint32_t cow_write(uint32_t seq_idx, uint32_t logical_block) {
        std::lock_guard<std::mutex> lk(mu_);
        return kv_->cow_write_block(seq_idx, logical_block);
    }

    /* ---- Page Out / Page In --------------------------------------- */

    /** Evict cold blocks from device to host memory.
     *  Returns number of blocks paged out. */
    uint32_t page_out(uint32_t seq_idx, uint32_t start_block,
                      uint32_t count, const uint8_t* block_data,
                      uint32_t bytes_per_block) {
        std::lock_guard<std::mutex> lk(mu_);
        if (seq_idx >= kv_->num_sequences) return 0;
        auto& seq = kv_->sequences[seq_idx];
        uint32_t paged = 0;

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t lb = start_block + i;
            if (lb >= seq.num_logical_blocks) break;
            uint32_t phys = seq.block_table[lb];
            if (phys == nf::NF_PAGED_INVALID_BLOCK) continue;

            /* Save to host */
            PageOutEntry entry;
            entry.seq_id        = seq_idx;
            entry.logical_block = lb;
            entry.original_phys = phys;
            entry.num_filled    = kv_->allocator.block_meta[phys].num_filled;
            if (block_data) {
                const uint8_t* src = block_data + phys * bytes_per_block;
                entry.data.assign(src, src + bytes_per_block);
            }
            paged_out_[make_page_key(seq_idx, lb)] = std::move(entry);

            /* Free device block */
            kv_->allocator.free_block(phys);
            seq.block_table[lb] = nf::NF_PAGED_INVALID_BLOCK;
            ++paged;
        }
        return paged;
    }

    /** Restore paged-out blocks back to device memory.
     *  Returns number of blocks paged in. Caller must copy data back. */
    uint32_t page_in(uint32_t seq_idx, uint32_t start_block,
                     uint32_t count, uint8_t* block_data,
                     uint32_t bytes_per_block) {
        std::lock_guard<std::mutex> lk(mu_);
        if (seq_idx >= kv_->num_sequences) return 0;
        auto& seq = kv_->sequences[seq_idx];
        uint32_t paged = 0;

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t lb = start_block + i;
            auto key = make_page_key(seq_idx, lb);
            auto it = paged_out_.find(key);
            if (it == paged_out_.end()) continue;

            uint32_t phys = kv_->allocator.alloc_block();
            if (phys == nf::NF_PAGED_INVALID_BLOCK) break;

            auto& entry = it->second;
            kv_->allocator.block_meta[phys].num_filled = entry.num_filled;
            seq.block_table[lb] = phys;

            /* Restore data */
            if (block_data && !entry.data.empty()) {
                uint8_t* dst = block_data + phys * bytes_per_block;
                std::memcpy(dst, entry.data.data(),
                            std::min((size_t)bytes_per_block, entry.data.size()));
            }
            paged_out_.erase(it);
            ++paged;
        }
        return paged;
    }

    /* ---- Radix Prefix Sharing ------------------------------------- */

    /** Match ContextHub prefixes against a token sequence.
     *  If a cached prefix matches, fork its sequence via CoW. */
    struct PrefixMatch {
        uint32_t match_len   = 0;
        uint64_t seq_id      = 0;
        bool     found       = false;
    };

    PrefixMatch radix_prefix_share(const std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lk(mu_);
        PrefixMatch best;
        if (!hub_) return best;

        auto refs = hub_->export_block_refs();
        for (auto& ref : refs) {
            /* Find longest common prefix */
            uint32_t common = 0;
            uint32_t limit = std::min((uint32_t)ref.token_key.size(),
                                      (uint32_t)tokens.size());
            while (common < limit && ref.token_key[common] == tokens[common])
                ++common;

            if (common > best.match_len) {
                best.match_len = common;
                best.seq_id    = ref.seq_id;
                best.found     = true;
            }
        }
        return best;
    }

    /* ---- Memory Pressure ------------------------------------------ */

    void set_pressure_callback(PressureCallback cb) {
        std::lock_guard<std::mutex> lk(mu_);
        pressure_cb_ = std::move(cb);
    }

    void check_pressure() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!pressure_cb_) return;
        uint32_t used  = kv_->allocator.num_used();
        uint32_t total = kv_->allocator.num_blocks;
        if (total == 0) return;
        float ratio = static_cast<float>(used) / static_cast<float>(total);
        if (ratio >= cfg_.pressure_threshold) {
            pressure_cb_(used, total);
        }
    }

    /* ---- Queries -------------------------------------------------- */

    uint32_t num_paged_out() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(paged_out_.size());
    }

    bool is_paged_out(uint32_t seq_idx, uint32_t logical_block) const {
        std::lock_guard<std::mutex> lk(mu_);
        return paged_out_.count(make_page_key(seq_idx, logical_block)) > 0;
    }

    bool has_fork_parent(uint32_t seq_idx) const {
        std::lock_guard<std::mutex> lk(mu_);
        return fork_parents_.count(seq_idx) > 0;
    }

private:
    static uint64_t make_page_key(uint32_t seq, uint32_t block) {
        return (static_cast<uint64_t>(seq) << 32) | block;
    }

    nf::PagedKVCache*  kv_;
    nf::ContextHub*    hub_;
    Config             cfg_;
    mutable std::mutex mu_;

    std::unordered_map<uint64_t, PageOutEntry>  paged_out_;
    std::unordered_map<uint32_t, uint32_t>      fork_parents_;  /* child → parent */
    PressureCallback                            pressure_cb_;
};

}} // namespace neuralOS::L2

#endif // NEURALOS_L2_VMMU_HPP
