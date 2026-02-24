/**
 * @file kv_migration.hpp
 * @brief NeuralOS L5 — KV Cache Cross-Node Hot Migration
 *
 * Phase 38.3: Live migration of KV cache sequences between nodes.
 *   - serialize_kv(): SequenceKV → wire format
 *   - deserialize_kv(): wire format → SequenceKV
 *   - live_migrate(): CoW fork → transfer → switch (no interruption)
 *
 * Links vMMU (CoW fork) + CFS (pause/resume) + TransportOps.
 */

#ifndef NEURALOS_L5_KV_MIGRATION_HPP
#define NEURALOS_L5_KV_MIGRATION_HPP

#include "neuralOS/kernel/kv_cache.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace neuralOS { namespace L5 {

/* ================================================================== */
/*  KVSerializedBlock — wire format for one KV block                   */
/* ================================================================== */

struct KVSerializedBlock {
    uint32_t logical_idx = 0;
    uint32_t num_filled  = 0;
    std::vector<uint8_t> data;
};

/* ================================================================== */
/*  KVSerializedSequence — wire format for a full sequence             */
/* ================================================================== */

struct KVSerializedSequence {
    uint32_t seq_id         = 0;
    uint32_t num_tokens     = 0;
    uint32_t num_blocks     = 0;
    uint32_t block_size     = 0;
    uint32_t bytes_per_block = 0;
    std::vector<KVSerializedBlock> blocks;
};

/* ================================================================== */
/*  KVMigrator — KV cache hot migration                                */
/* ================================================================== */

class KVMigrator {
public:
    /** Serialize a sequence's KV blocks into wire format.
     *  block_data points to the flat device memory pool. */
    KVSerializedSequence serialize_kv(const nf::PagedKVCache& kv,
                                      uint32_t seq_idx,
                                      const uint8_t* block_data,
                                      uint32_t bytes_per_block) {
        KVSerializedSequence out;
        if (seq_idx >= kv.num_sequences) return out;

        auto& seq = kv.sequences[seq_idx];
        out.seq_id = seq.seq_id;
        out.num_tokens = seq.num_tokens;
        out.num_blocks = seq.num_logical_blocks;
        out.block_size = kv.block_size;
        out.bytes_per_block = bytes_per_block;

        for (uint32_t i = 0; i < seq.num_logical_blocks; ++i) {
            uint32_t phys = seq.block_table[i];
            if (phys == nf::NF_PAGED_INVALID_BLOCK) continue;

            KVSerializedBlock sb;
            sb.logical_idx = i;
            sb.num_filled = kv.allocator.block_meta[phys].num_filled;
            if (block_data) {
                const uint8_t* src = block_data + phys * bytes_per_block;
                sb.data.assign(src, src + bytes_per_block);
            }
            out.blocks.push_back(std::move(sb));
        }
        return out;
    }

    /** Deserialize wire format into a target KV cache.
     *  Allocates new blocks in the target cache. Returns seq index. */
    uint32_t deserialize_kv(nf::PagedKVCache& kv,
                            const KVSerializedSequence& wire,
                            uint8_t* block_data,
                            uint32_t bytes_per_block) {
        uint32_t seq_idx = kv.alloc_sequence();
        if (seq_idx == UINT32_MAX) return seq_idx;

        auto& seq = kv.sequences[seq_idx];
        seq.num_tokens = wire.num_tokens;

        for (auto& sb : wire.blocks) {
            uint32_t phys = kv.allocator.alloc_block();
            if (phys == nf::NF_PAGED_INVALID_BLOCK) break;

            kv.allocator.block_meta[phys].num_filled = sb.num_filled;
            seq.block_table[sb.logical_idx] = phys;
            seq.num_logical_blocks = std::max(seq.num_logical_blocks,
                                               sb.logical_idx + 1);

            if (block_data && !sb.data.empty()) {
                uint8_t* dst = block_data + phys * bytes_per_block;
                std::memcpy(dst, sb.data.data(),
                            std::min((size_t)bytes_per_block, sb.data.size()));
            }
        }
        return seq_idx;
    }

    /** Live migration: fork → serialize → transfer → deserialize → switch.
     *  Returns the new sequence index on the target cache, or UINT32_MAX. */
    uint32_t live_migrate(nf::PagedKVCache& src_kv, uint32_t src_seq_idx,
                          nf::PagedKVCache& dst_kv,
                          const uint8_t* src_block_data,
                          uint8_t* dst_block_data,
                          uint32_t bytes_per_block) {
        /* Step 1: CoW fork on source (non-disruptive) */
        uint32_t fork_idx = src_kv.fork_sequence(src_seq_idx);
        if (fork_idx == UINT32_MAX) return UINT32_MAX;

        /* Step 2: Serialize the forked copy */
        auto wire = serialize_kv(src_kv, fork_idx, src_block_data, bytes_per_block);

        /* Step 3: Deserialize on target */
        uint32_t dst_idx = deserialize_kv(dst_kv, wire, dst_block_data, bytes_per_block);

        /* Step 4: Free the fork on source */
        src_kv.free_sequence(fork_idx);

        return dst_idx;
    }
};

}} // namespace neuralOS::L5

#endif // NEURALOS_L5_KV_MIGRATION_HPP
