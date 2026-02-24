/**
 * @file cxl_memory.hpp
 * @brief NeuralOS L5 — CXL 3.0 Memory Domain Abstraction
 *
 * Phase 38.4: CXL GFAM (Global Fabric Attached Memory) stub.
 * Maps to POSIX shared memory for simulation. Interface-first design.
 */

#ifndef NEURALOS_L5_CXL_MEMORY_HPP
#define NEURALOS_L5_CXL_MEMORY_HPP

#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

namespace neuralOS { namespace L5 {

/* ================================================================== */
/*  CXL Memory Domain IDs (extend neuro_buffer_abi.h)                  */
/* ================================================================== */

static constexpr uint32_t NF_MEM_DOMAIN_CXL        = 6;
static constexpr uint32_t NF_MEM_DOMAIN_CXL_SHARED = 7;

/* ================================================================== */
/*  CxlAllocation — tracks one CXL memory allocation                   */
/* ================================================================== */

struct CxlAllocation {
    uint64_t alloc_id    = 0;
    uint64_t size_bytes  = 0;
    uint32_t domain      = NF_MEM_DOMAIN_CXL;
    std::vector<uint8_t> backing;  /* simulated CXL memory */
};

/* ================================================================== */
/*  CxlMemoryPool — CXL memory pool abstraction                       */
/* ================================================================== */

class CxlMemoryPool {
public:
    struct Config {
        uint64_t pool_size_bytes = 1ULL * 1024 * 1024 * 1024;  /* 1 GB default */
        uint32_t simulated_latency_ns = 200;  /* ~200ns CXL load/store */
        bool     shared = false;  /* NF_MEM_DOMAIN_CXL_SHARED */
    };

    explicit CxlMemoryPool(Config cfg)
        : cfg_(cfg), used_(0) {}

    CxlMemoryPool() : CxlMemoryPool(Config{}) {}

    /** Allocate CXL memory. Returns alloc_id or 0 on failure. */
    uint64_t cxl_alloc(uint64_t size_bytes) {
        if (used_ + size_bytes > cfg_.pool_size_bytes) return 0;

        CxlAllocation alloc;
        alloc.alloc_id = next_id_++;
        alloc.size_bytes = size_bytes;
        alloc.domain = cfg_.shared ? NF_MEM_DOMAIN_CXL_SHARED : NF_MEM_DOMAIN_CXL;
        alloc.backing.resize(size_bytes, 0);

        uint64_t id = alloc.alloc_id;
        used_ += size_bytes;
        allocs_[id] = std::move(alloc);
        return id;
    }

    /** Free CXL memory */
    bool cxl_free(uint64_t alloc_id) {
        auto it = allocs_.find(alloc_id);
        if (it == allocs_.end()) return false;
        used_ -= it->second.size_bytes;
        allocs_.erase(it);
        return true;
    }

    /** Simulated load/store with CXL latency (~200ns) */
    bool cxl_load_store(uint64_t alloc_id, uint64_t offset,
                        void* dst, const void* src, uint64_t len,
                        bool is_store) {
        auto it = allocs_.find(alloc_id);
        if (it == allocs_.end()) return false;
        auto& alloc = it->second;
        if (offset + len > alloc.size_bytes) return false;

        /* Simulate CXL latency */
        if (cfg_.simulated_latency_ns > 0) {
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(cfg_.simulated_latency_ns));
        }

        if (is_store) {
            std::memcpy(alloc.backing.data() + offset, src, len);
        } else {
            std::memcpy(dst, alloc.backing.data() + offset, len);
        }
        return true;
    }

    /* Convenience wrappers */
    bool cxl_store(uint64_t alloc_id, uint64_t offset,
                   const void* src, uint64_t len) {
        return cxl_load_store(alloc_id, offset, nullptr, src, len, true);
    }

    bool cxl_load(uint64_t alloc_id, uint64_t offset,
                  void* dst, uint64_t len) {
        return cxl_load_store(alloc_id, offset, dst, nullptr, len, false);
    }

    /* ---- Queries -------------------------------------------------- */

    uint64_t used_bytes() const { return used_; }
    uint64_t free_bytes() const { return cfg_.pool_size_bytes - used_; }
    uint32_t num_allocs() const { return static_cast<uint32_t>(allocs_.size()); }
    bool is_shared() const { return cfg_.shared; }

private:
    Config   cfg_;
    uint64_t used_    = 0;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, CxlAllocation> allocs_;
};

}} // namespace neuralOS::L5

#endif // NEURALOS_L5_CXL_MEMORY_HPP
