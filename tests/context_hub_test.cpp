/**
 * @file context_hub_test.cpp
 * @brief Phase 13 — ContextHub Token-ID Radix Tree Stress Test
 *
 * 8 test functions:
 *   1. Basic insert/lookup
 *   2. Prefix match (longest prefix)
 *   3. Node splitting (compressed radix)
 *   4. LRU eviction under memory pressure
 *   5. TTL expiration
 *   6. Concurrent agents (3 threads, shared prefix)
 *   7. Memory leak on eviction (refcount tracking)
 *   8. Subtree eviction
 */

#include "neurofabric/ContextHub.hpp"
#include "neurofabric/TensorView.hpp"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

/* ================================================================== */
/*  CHECK macro — safe in Release builds (unlike assert)               */
/* ================================================================== */

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)

/* ================================================================== */
/*  MockBuf — minimal refcounted buffer for testing                    */
/* ================================================================== */

struct MockBuf {
    std::atomic<uint32_t> refcount{1};
    void*                 data = nullptr;
    nf_tensor_desc        desc{};
};
static uint32_t mb_retain(nf_buffer self) {
    return reinterpret_cast<MockBuf*>(self)->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}
static uint32_t mb_release(nf_buffer self) {
    auto* b = reinterpret_cast<MockBuf*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) { std::free(b->data); delete b; }
    return prev - 1;
}
static nf_status mb_map(nf_buffer self, void** p) {
    *p = reinterpret_cast<MockBuf*>(self)->data; return NF_OK;
}
static nf_status mb_unmap(nf_buffer) { return NF_OK; }
static nf_status mb_cache(nf_buffer, nf_cache_op, uint64_t, uint64_t) { return NF_OK; }
static nf_status mb_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<MockBuf*>(self);
    info->desc = b->desc; info->domain = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0; info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0; return NF_OK;
}
static nf_status mb_export(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_CPU; return NF_OK;
}

static nf_buffer_ops make_mock_ops() {
    nf_buffer_ops ops{};
    ops.retain = mb_retain; ops.release = mb_release;
    ops.map = mb_map; ops.unmap = mb_unmap;
    ops.cache_sync = mb_cache; ops.get_info = mb_info;
    ops.export_handle = mb_export; ops.import_handle = nullptr;
    return ops;
}

static std::pair<nf_buffer, nf_buffer_ops> make_tensor(uint64_t size_bytes) {
    auto* b = new MockBuf;
    b->desc.dtype = NF_DTYPE_F32;
    b->desc.ndim = 1;
    b->desc.shape[0] = size_bytes / sizeof(float);
    b->desc.size_bytes = size_bytes;
    b->data = std::calloc(1, size_bytes);
    return {reinterpret_cast<nf_buffer>(b), make_mock_ops()};
}

static uint32_t get_refcount(nf_buffer buf) {
    return reinterpret_cast<MockBuf*>(buf)->refcount.load(std::memory_order_relaxed);
}
/* ================================================================== */
/*  Test 1: Basic insert and lookup                                    */
/* ================================================================== */

static void test_basic_insert_lookup() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    auto [b1, o1] = make_tensor(4096);
    std::vector<int32_t> k1 = {10, 20, 30};
    CHECK(hub.put(k1, "agent_a", nf::TensorView(b1, o1)) == NF_OK);

    auto [b2, o2] = make_tensor(2048);
    std::vector<int32_t> k2 = {40, 50, 60};
    CHECK(hub.put(k2, "agent_b", nf::TensorView(b2, o2)) == NF_OK);

    auto [b3, o3] = make_tensor(1024);
    std::vector<int32_t> k3 = {70, 80};
    CHECK(hub.put(k3, "agent_c", nf::TensorView(b3, o3)) == NF_OK);

    auto r1 = hub.get(std::span<const int32_t>(k1));
    CHECK(r1.found);
    CHECK(r1.match_len == 3);

    auto r2 = hub.get(std::span<const int32_t>(k2));
    CHECK(r2.found);
    CHECK(r2.match_len == 3);

    auto r3 = hub.get(std::span<const int32_t>(k3));
    CHECK(r3.found);
    CHECK(r3.match_len == 2);

    auto s = hub.stats();
    CHECK(s.entry_count == 3);
    CHECK(s.used_bytes == 4096 + 2048 + 1024);

    std::printf("  PASS: basic insert/lookup\n");
}

/* ================================================================== */
/*  Test 2: Longest prefix match                                       */
/* ================================================================== */

static void test_prefix_match() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    auto [b1, o1] = make_tensor(4096);
    std::vector<int32_t> k1 = {1, 2, 3, 4, 5};
    CHECK(hub.put(k1, "a", nf::TensorView(b1, o1)) == NF_OK);

    // Longer query should match the 5-token entry
    std::vector<int32_t> longer = {1, 2, 3, 4, 5, 6, 7};
    auto r1 = hub.get(std::span<const int32_t>(longer));
    CHECK(r1.found);
    CHECK(r1.match_len == 5);

    // Shorter query — no entry at [1,2,3], so no match
    std::vector<int32_t> shorter = {1, 2, 3};
    auto r2 = hub.get(std::span<const int32_t>(shorter));
    CHECK(!r2.found);

    // Now insert at [1,2,3] too
    auto [b2, o2] = make_tensor(1024);
    CHECK(hub.put(shorter, "b", nf::TensorView(b2, o2)) == NF_OK);

    // Shorter query now matches
    auto r3 = hub.get(std::span<const int32_t>(shorter));
    CHECK(r3.found);
    CHECK(r3.match_len == 3);

    // Longer query still matches the deeper entry
    auto r4 = hub.get(std::span<const int32_t>(longer));
    CHECK(r4.found);
    CHECK(r4.match_len == 5);

    // Total miss
    std::vector<int32_t> miss = {99, 88};
    CHECK(!hub.get(std::span<const int32_t>(miss)).found);

    std::printf("  PASS: prefix match\n");
}
/* ================================================================== */
/*  Test 3: Node splitting (compressed radix)                          */
/* ================================================================== */

static void test_node_splitting() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    // Insert [1,2,3,4,5] — creates single edge from root
    auto [b1, o1] = make_tensor(1024);
    std::vector<int32_t> k1 = {1, 2, 3, 4, 5};
    CHECK(hub.put(k1, "a", nf::TensorView(b1, o1)) == NF_OK);

    // Insert [1,2,3,6,7] — forces split at position 3
    auto [b2, o2] = make_tensor(1024);
    std::vector<int32_t> k2 = {1, 2, 3, 6, 7};
    CHECK(hub.put(k2, "b", nf::TensorView(b2, o2)) == NF_OK);

    // Both entries retrievable
    auto r1 = hub.get(std::span<const int32_t>(k1));
    CHECK(r1.found);
    CHECK(r1.match_len == 5);

    auto r2 = hub.get(std::span<const int32_t>(k2));
    CHECK(r2.found);
    CHECK(r2.match_len == 5);

    // [1,2,3] has no entry (intermediate node)
    std::vector<int32_t> prefix = {1, 2, 3};
    CHECK(!hub.get(std::span<const int32_t>(prefix)).found);

    // Insert at [1,2,3] — all three coexist
    auto [b3, o3] = make_tensor(512);
    CHECK(hub.put(prefix, "c", nf::TensorView(b3, o3)) == NF_OK);

    CHECK(hub.get(std::span<const int32_t>(prefix)).found);
    CHECK(hub.get(std::span<const int32_t>(k1)).found);
    CHECK(hub.get(std::span<const int32_t>(k2)).found);
    CHECK(hub.stats().entry_count == 3);

    std::printf("  PASS: node splitting\n");
}

/* ================================================================== */
/*  Test 4: LRU eviction under memory pressure                        */
/* ================================================================== */

static void test_lru_eviction() {
    // Budget fits exactly 2 x 4096 entries
    nf::ContextHub hub(8192, NF_EVICT_LRU);

    for (int i = 0; i < 4; ++i) {
        auto [buf, ops] = make_tensor(4096);
        std::vector<int32_t> key = {100, static_cast<int32_t>(i)};
        CHECK(hub.put(key, "test", nf::TensorView(buf, ops), 1000) == NF_OK);
    }

    auto s = hub.stats();
    CHECK(s.entry_count == 2);
    CHECK(s.used_bytes <= 8192);

    // Earliest entries should have been evicted
    std::vector<int32_t> k0 = {100, 0};
    std::vector<int32_t> k1 = {100, 1};
    std::vector<int32_t> k3 = {100, 3};
    CHECK(!hub.get(std::span<const int32_t>(k0)).found);
    CHECK(!hub.get(std::span<const int32_t>(k1)).found);
    CHECK(hub.get(std::span<const int32_t>(k3)).found);

    std::printf("  PASS: LRU eviction\n");
}

/* ================================================================== */
/*  Test 5: TTL expiration                                             */
/* ================================================================== */

static void test_ttl_expiration() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_TTL);

    auto [buf, ops] = make_tensor(1024);
    std::vector<int32_t> key = {42, 43};
    CHECK(hub.put(key, "test", nf::TensorView(buf, ops), 50) == NF_OK);

    // Should be found immediately
    CHECK(hub.get(std::span<const int32_t>(key)).found);

    // Wait for expiration
    std::this_thread::sleep_for(std::chrono::milliseconds(70));

    // Expired — get() skips expired entries
    CHECK(!hub.get(std::span<const int32_t>(key)).found);

    std::printf("  PASS: TTL expiration\n");
}
/* ================================================================== */
/*  Test 6: Concurrent agents (3 threads, shared prefix)               */
/* ================================================================== */

static void test_concurrent_agents() {
    nf::ContextHub hub(16 * 1024 * 1024, NF_EVICT_LRU);

    // Shared system prompt prefix: [10, 20, 30, 40, 50]
    const std::vector<int32_t> sys_prefix = {10, 20, 30, 40, 50};

    std::barrier sync_point(3);
    std::atomic<int> errors{0};

    auto agent_fn = [&](int agent_id, int32_t suffix_base) {
        sync_point.arrive_and_wait();  // synchronized start

        // Each agent extends the shared prefix with unique tokens
        std::vector<int32_t> key = sys_prefix;
        key.push_back(suffix_base);
        key.push_back(suffix_base + 1);

        auto [buf, ops] = make_tensor(4096);
        std::string aid = "agent_" + std::to_string(agent_id);
        nf_status st = hub.put(key, aid, nf::TensorView(buf, ops));
        if (st != NF_OK) errors.fetch_add(1);

        // Verify own entry
        auto r = hub.get(std::span<const int32_t>(key));
        if (!r.found) errors.fetch_add(1);
        if (r.match_len != 7) errors.fetch_add(1);
    };

    std::thread t0(agent_fn, 0, 100);
    std::thread t1(agent_fn, 1, 200);
    std::thread t2(agent_fn, 2, 300);
    t0.join(); t1.join(); t2.join();

    CHECK(errors.load() == 0);

    // All 3 entries should exist
    CHECK(hub.stats().entry_count == 3);

    // Prefix query [10,20,30,40,50] should NOT match (no entry at prefix itself)
    auto prefix_r = hub.get(std::span<const int32_t>(sys_prefix));
    CHECK(!prefix_r.found);

    // Each agent's full key should still be retrievable
    for (int32_t base : {100, 200, 300}) {
        std::vector<int32_t> key = sys_prefix;
        key.push_back(base);
        key.push_back(base + 1);
        auto r = hub.get(std::span<const int32_t>(key));
        CHECK(r.found);
        CHECK(r.match_len == 7);
    }

    // Radix tree should have branched at token index 5
    // (shared [10,20,30,40,50] prefix, then 3 diverging children)
    CHECK(hub.stats().used_bytes == 3 * 4096);

    std::printf("  PASS: concurrent agents (3 threads)\n");
}

/* ================================================================== */
/*  Test 7: Memory leak on eviction (refcount tracking)                */
/* ================================================================== */

static void test_memory_leak_on_eviction() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    auto [buf, ops] = make_tensor(1024);
    nf_buffer raw_buf = buf;

    // Before insert: refcount = 1 (our local handle)
    CHECK(get_refcount(raw_buf) == 1);

    // share() increments refcount, hub takes ownership of the shared copy
    nf::TensorView tv(buf, ops);
    std::vector<int32_t> key = {1, 2, 3};
    CHECK(hub.put(key, "test", tv.share()) == NF_OK);

    // After insert: refcount = 2 (our tv + hub's copy)
    CHECK(get_refcount(raw_buf) == 2);

    // Evict — hub releases its TensorView
    hub.evict(std::span<const int32_t>(key));

    // After evict: refcount = 1 (only our tv)
    CHECK(get_refcount(raw_buf) == 1);
    CHECK(hub.stats().entry_count == 0);
    CHECK(hub.stats().used_bytes == 0);

    std::printf("  PASS: memory leak on eviction (refcount correct)\n");
}

/* ================================================================== */
/*  Test 8: Subtree eviction                                           */
/* ================================================================== */

static void test_evict_subtree() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    // Insert multiple entries under prefix [1,2,3]
    auto [b1, o1] = make_tensor(1024);
    std::vector<int32_t> k1 = {1, 2, 3, 4, 5};
    CHECK(hub.put(k1, "a", nf::TensorView(b1, o1)) == NF_OK);

    auto [b2, o2] = make_tensor(1024);
    std::vector<int32_t> k2 = {1, 2, 3, 6, 7};
    CHECK(hub.put(k2, "b", nf::TensorView(b2, o2)) == NF_OK);

    auto [b3, o3] = make_tensor(1024);
    std::vector<int32_t> k3 = {1, 2, 3};
    CHECK(hub.put(k3, "c", nf::TensorView(b3, o3)) == NF_OK);

    // Also insert outside the prefix
    auto [b4, o4] = make_tensor(1024);
    std::vector<int32_t> k4 = {9, 8, 7};
    CHECK(hub.put(k4, "d", nf::TensorView(b4, o4)) == NF_OK);

    CHECK(hub.stats().entry_count == 4);

    // Evict subtree [1,2,3] — should remove 3 entries
    std::vector<int32_t> prefix = {1, 2, 3};
    CHECK(hub.evict(std::span<const int32_t>(prefix)) == NF_OK);

    CHECK(hub.stats().entry_count == 1);
    CHECK(hub.stats().used_bytes == 1024);

    // [9,8,7] should survive
    std::vector<int32_t> survivor = {9, 8, 7};
    CHECK(hub.get(std::span<const int32_t>(survivor)).found);

    std::printf("  PASS: subtree eviction\n");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("context_hub_test: Phase 13 — Token-ID Radix Tree\n");
    test_basic_insert_lookup();
    test_prefix_match();
    test_node_splitting();
    test_lru_eviction();
    test_ttl_expiration();
    test_concurrent_agents();
    test_memory_leak_on_eviction();
    test_evict_subtree();
    std::printf("OK: all Phase 13 context hub tests passed\n");
    return 0;
}
