/**
 * @file kv_cache.hpp
 * @brief NeuralOS L2 — PagedKVCache (Block Allocator + Per-Sequence Block Tables)
 *
 * Phase 45A: Extracted from model/model/model_config.hpp to eliminate
 * kernel→model reverse dependency. Kernel layer is now self-contained.
 *
 * Header-only. No external dependencies beyond STL.
 */

#ifndef NEURALOS_KERNEL_KV_CACHE_HPP
#define NEURALOS_KERNEL_KV_CACHE_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace neuralOS { namespace kernel {

/* ================================================================== */
/*  Constants                                                          */
/* ================================================================== */

static constexpr uint32_t NF_PAGED_MAX_SEQUENCES      = 64;
static constexpr uint32_t NF_PAGED_MAX_BLOCKS_PER_SEQ = 512;  /* 512 × 16 = 8192 tokens */
static constexpr uint32_t NF_PAGED_INVALID_BLOCK      = UINT32_MAX;

/* ================================================================== */
/*  PagedKVBlock                                                       */
/* ================================================================== */

struct PagedKVBlock {
    uint32_t ref_count   = 0;   /* CoW: >1 means shared */
    uint32_t num_filled  = 0;   /* tokens written [0..block_size] */
};

/* ================================================================== */
/*  BlockAllocator                                                     */
/* ================================================================== */

struct BlockAllocator {
    uint32_t      num_blocks  = 0;
    uint32_t      block_size  = 16;
    uint32_t      free_top    = 0;     /* stack pointer */
    std::vector<uint32_t>      free_stack;
    std::vector<PagedKVBlock>  block_meta;

    void init(uint32_t n_blocks, uint32_t bsz) {
        num_blocks = n_blocks;
        block_size = bsz;
        free_stack.resize(n_blocks);
        block_meta.resize(n_blocks);
        reset();
    }

    void reset() {
        free_top = num_blocks;
        for (uint32_t i = 0; i < num_blocks; ++i) {
            free_stack[i] = num_blocks - 1 - i;  /* top of stack = block 0 */
            block_meta[i] = PagedKVBlock{};
        }
    }

    uint32_t alloc_block() {
        if (free_top == 0) return NF_PAGED_INVALID_BLOCK;
        uint32_t idx = free_stack[--free_top];
        block_meta[idx].ref_count = 1;
        block_meta[idx].num_filled = 0;
        return idx;
    }

    void free_block(uint32_t idx) {
        if (idx >= num_blocks) return;
        auto& m = block_meta[idx];
        if (m.ref_count > 0) --m.ref_count;
        if (m.ref_count == 0) {
            free_stack[free_top++] = idx;
            m.num_filled = 0;
        }
    }

    void ref(uint32_t idx) {
        if (idx < num_blocks) ++block_meta[idx].ref_count;
    }

    void unref(uint32_t idx) {
        if (idx < num_blocks) free_block(idx);
    }

    uint32_t num_free() const { return free_top; }
    uint32_t num_used() const { return num_blocks - free_top; }

    uint32_t cow_copy_block(uint32_t src_idx) {
        if (src_idx >= num_blocks) return NF_PAGED_INVALID_BLOCK;
        uint32_t dst = alloc_block();
        if (dst == NF_PAGED_INVALID_BLOCK) return dst;
        block_meta[dst].num_filled = block_meta[src_idx].num_filled;
        return dst;
    }
};

/* ================================================================== */
/*  SequenceKV                                                         */
/* ================================================================== */

struct SequenceKV {
    uint32_t seq_id          = 0;
    uint32_t num_tokens      = 0;
    uint32_t num_logical_blocks = 0;
    bool     active          = false;
    uint32_t block_table[NF_PAGED_MAX_BLOCKS_PER_SEQ] = {};

    void reset() {
        num_tokens = 0;
        num_logical_blocks = 0;
        active = false;
        std::memset(block_table, 0xFF, sizeof(block_table));
    }
};

/* ================================================================== */
/*  PagedKVCache                                                       */
/* ================================================================== */

struct PagedKVCache {
    BlockAllocator allocator;

    SequenceKV sequences[NF_PAGED_MAX_SEQUENCES];
    uint32_t   num_sequences = 0;

    uint32_t n_layers   = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim   = 0;
    uint32_t block_size = 16;

    void init(uint32_t n_blocks, uint32_t bsz,
              uint32_t layers, uint32_t kv_heads, uint32_t hdim) {
        allocator.init(n_blocks, bsz);
        n_layers   = layers;
        n_kv_heads = kv_heads;
        head_dim   = hdim;
        block_size = bsz;
        num_sequences = 0;
        for (auto& s : sequences) s.reset();
    }

    uint32_t alloc_sequence() {
        if (num_sequences >= NF_PAGED_MAX_SEQUENCES) return UINT32_MAX;
        uint32_t idx = num_sequences++;
        sequences[idx].seq_id = idx;
        sequences[idx].active = true;
        sequences[idx].num_tokens = 0;
        sequences[idx].num_logical_blocks = 0;
        std::memset(sequences[idx].block_table, 0xFF,
                    sizeof(sequences[idx].block_table));
        return idx;
    }

    uint32_t append(uint32_t seq_idx) {
        if (seq_idx >= num_sequences) return UINT32_MAX;
        auto& seq = sequences[seq_idx];

        uint32_t logical_block = seq.num_tokens / block_size;
        uint32_t offset_in_block = seq.num_tokens % block_size;

        if (logical_block >= seq.num_logical_blocks) {
            uint32_t phys = allocator.alloc_block();
            if (phys == NF_PAGED_INVALID_BLOCK) return UINT32_MAX;
            seq.block_table[logical_block] = phys;
            seq.num_logical_blocks = logical_block + 1;
        }

        uint32_t phys_block = seq.block_table[logical_block];
        allocator.block_meta[phys_block].num_filled = offset_in_block + 1;
        seq.num_tokens++;

        return phys_block * block_size + offset_in_block;
    }

    void free_sequence(uint32_t seq_idx) {
        if (seq_idx >= num_sequences) return;
        auto& seq = sequences[seq_idx];
        for (uint32_t i = 0; i < seq.num_logical_blocks; ++i) {
            if (seq.block_table[i] != NF_PAGED_INVALID_BLOCK)
                allocator.free_block(seq.block_table[i]);
        }
        seq.reset();
    }

    void truncate(uint32_t seq_idx, uint32_t new_len) {
        if (seq_idx >= num_sequences) return;
        auto& seq = sequences[seq_idx];
        if (new_len >= seq.num_tokens) return;

        uint32_t new_blocks = (new_len + block_size - 1) / block_size;
        if (new_len == 0) new_blocks = 0;

        for (uint32_t i = new_blocks; i < seq.num_logical_blocks; ++i) {
            if (seq.block_table[i] != NF_PAGED_INVALID_BLOCK) {
                allocator.free_block(seq.block_table[i]);
                seq.block_table[i] = NF_PAGED_INVALID_BLOCK;
            }
        }
        seq.num_logical_blocks = new_blocks;
        seq.num_tokens = new_len;

        if (new_blocks > 0) {
            uint32_t last_phys = seq.block_table[new_blocks - 1];
            if (last_phys != NF_PAGED_INVALID_BLOCK) {
                uint32_t fill = new_len % block_size;
                allocator.block_meta[last_phys].num_filled =
                    (fill == 0) ? block_size : fill;
            }
        }
    }

    uint32_t fork_sequence(uint32_t src_seq_idx) {
        if (src_seq_idx >= num_sequences) return UINT32_MAX;
        uint32_t dst = alloc_sequence();
        if (dst == UINT32_MAX) return dst;
        auto& src = sequences[src_seq_idx];
        auto& d   = sequences[dst];
        d.num_tokens = src.num_tokens;
        d.num_logical_blocks = src.num_logical_blocks;
        for (uint32_t i = 0; i < src.num_logical_blocks; ++i) {
            uint32_t phys = src.block_table[i];
            if (phys != NF_PAGED_INVALID_BLOCK) {
                allocator.ref(phys);
            }
            d.block_table[i] = phys;
        }
        return dst;
    }

    uint32_t cow_write_block(uint32_t seq_idx, uint32_t logical_block) {
        if (seq_idx >= num_sequences) return NF_PAGED_INVALID_BLOCK;
        auto& seq = sequences[seq_idx];
        if (logical_block >= seq.num_logical_blocks) return NF_PAGED_INVALID_BLOCK;
        uint32_t phys = seq.block_table[logical_block];
        if (phys == NF_PAGED_INVALID_BLOCK) return NF_PAGED_INVALID_BLOCK;
        if (allocator.block_meta[phys].ref_count > 1) {
            uint32_t new_phys = allocator.cow_copy_block(phys);
            if (new_phys == NF_PAGED_INVALID_BLOCK) return new_phys;
            allocator.unref(phys);
            seq.block_table[logical_block] = new_phys;
            return new_phys;
        }
        return phys;
    }

    void fill_block_table(uint32_t seq_idx, uint32_t* out,
                          uint32_t max_blocks) const {
        if (seq_idx >= num_sequences) return;
        const auto& seq = sequences[seq_idx];
        uint32_t n = std::min(seq.num_logical_blocks, max_blocks);
        for (uint32_t i = 0; i < n; ++i)
            out[i] = seq.block_table[i];
        for (uint32_t i = n; i < max_blocks; ++i)
            out[i] = NF_PAGED_INVALID_BLOCK;
    }
};

}} // neuralOS::kernel

// Backward compatibility
namespace nf {
    using neuralOS::kernel::NF_PAGED_MAX_SEQUENCES;
    using neuralOS::kernel::NF_PAGED_MAX_BLOCKS_PER_SEQ;
    using neuralOS::kernel::NF_PAGED_INVALID_BLOCK;
    using neuralOS::kernel::PagedKVBlock;
    using neuralOS::kernel::BlockAllocator;
    using neuralOS::kernel::SequenceKV;
    using neuralOS::kernel::PagedKVCache;
}

#endif /* NEURALOS_KERNEL_KV_CACHE_HPP */
