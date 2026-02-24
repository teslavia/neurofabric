/**
 * @file scheduler_test.cpp
 * @brief Phase 3 test — DAG scheduler, pipeline engine, and context hub.
 *
 * Exercises:
 *   1. PipelineEngine: graph build, Kahn's topo-sort, async dispatch
 *   2. ContextHub: put/get with radix prefix matching, eviction
 */

#include "neuralOS/kernel/PipelineEngine.hpp"
#include "neuralOS/kernel/ContextHub.hpp"
#include <cstdio>
#include <cstring>
#include <atomic>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)

/* ================================================================== */
/*  Mock Provider for DAG dispatch testing                             */
/* ================================================================== */

static std::atomic<int> g_dispatch_count{0};
static std::atomic<int> g_dispatch_order[8] = {};
static std::atomic<int> g_order_idx{0};

struct MockProvider {
    int id = 0;
};

static MockProvider g_mock_prov;

static const char* mock_get_name(nf_provider) { return "mock_cpu"; }
static uint32_t    mock_get_abi_version(nf_provider) { return NF_ABI_VERSION; }
static nf_status   mock_init(nf_provider) { return NF_OK; }
static void        mock_shutdown(nf_provider) {}

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer*, uint32_t,
                               nf_buffer*, uint32_t) {
    int idx = g_order_idx.fetch_add(1, std::memory_order_relaxed);
    if (idx < 8) {
        // Encode op_name's last char as order marker
        g_dispatch_order[idx].store(
            op_name[std::strlen(op_name) - 1] - '0',
            std::memory_order_relaxed);
    }
    g_dispatch_count.fetch_add(1, std::memory_order_relaxed);
    return NF_OK;
}

static nf_status mock_sync(nf_provider) { return NF_OK; }

static nf_provider_vtable make_mock_vtable() {
    nf_provider_vtable vt{};
    vt.get_name        = mock_get_name;
    vt.get_abi_version = mock_get_abi_version;
    vt.init            = mock_init;
    vt.shutdown        = mock_shutdown;
    vt.dispatch        = mock_dispatch;
    vt.synchronize     = mock_sync;
    return vt;
}

/* ================================================================== */
/*  Mock Buffer (reused from buffer_test)                              */
/* ================================================================== */

struct MockBuf {
    std::atomic<uint32_t> refcount{1};
    nf_tensor_desc desc{};
    uint64_t size = 0;
};

static uint32_t mbuf_retain(nf_buffer self) {
    return reinterpret_cast<MockBuf*>(self)->refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}
static uint32_t mbuf_release(nf_buffer self) {
    auto* b = reinterpret_cast<MockBuf*>(self);
    uint32_t prev = b->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) delete b;
    return prev - 1;
}
static nf_status mbuf_map(nf_buffer, void** p) { *p = nullptr; return NF_OK; }
static nf_status mbuf_unmap(nf_buffer) { return NF_OK; }
static nf_status mbuf_cache(nf_buffer, nf_cache_op, uint64_t, uint64_t) { return NF_OK; }
static nf_status mbuf_info(nf_buffer self, nf_buffer_info* info) {
    auto* b = reinterpret_cast<MockBuf*>(self);
    info->desc = b->desc;
    info->domain = NF_MEM_DOMAIN_CPU;
    info->offset_bytes = 0;
    info->share_token = 0;
    info->refcount = b->refcount.load(std::memory_order_relaxed);
    info->_reserved = 0;
    return NF_OK;
}
static nf_status mbuf_export(nf_buffer, uint64_t* t, nf_mem_domain* d) {
    *t = 0; *d = NF_MEM_DOMAIN_CPU; return NF_OK;
}

static nf_buffer_ops make_mock_buf_ops() {
    nf_buffer_ops ops{};
    ops.retain        = mbuf_retain;
    ops.release       = mbuf_release;
    ops.map           = mbuf_map;
    ops.unmap         = mbuf_unmap;
    ops.cache_sync    = mbuf_cache;
    ops.get_info      = mbuf_info;
    ops.export_handle = mbuf_export;
    ops.import_handle = nullptr;
    return ops;
}

static std::pair<nf_buffer, nf_buffer_ops> make_mock_tensor(uint64_t size) {
    auto* b = new MockBuf;
    b->desc.dtype = NF_DTYPE_F32;
    b->desc.ndim = 1;
    b->desc.shape[0] = size / sizeof(float);
    b->desc.size_bytes = size;
    b->size = size;
    return {reinterpret_cast<nf_buffer>(b), make_mock_buf_ops()};
}

/* ================================================================== */
/*  Test 1: DAG topological ordering                                   */
/* ================================================================== */

static void test_dag_topo_sort() {
    /*
     * Graph topology (diamond):
     *
     *       task_0
     *      /      \
     *   task_1   task_2
     *      \      /
     *       task_3
     *
     * Valid orderings: 0 before {1,2}, both before 3.
     */
    nf::PipelineEngine engine(2);

    auto vt = make_mock_vtable();
    engine.register_provider(
        reinterpret_cast<nf_provider>(&g_mock_prov),
        vt, NF_AFFINITY_CPU);

    uint32_t gid = engine.create_graph();

    nf_task_desc d0{}, d1{}, d2{}, d3{};
    std::strcpy(d0.op_name, "op_0");
    std::strcpy(d1.op_name, "op_1");
    std::strcpy(d2.op_name, "op_2");
    std::strcpy(d3.op_name, "op_3");

    uint32_t t0 = engine.add_task(gid, d0);
    uint32_t t1 = engine.add_task(gid, d1);
    uint32_t t2 = engine.add_task(gid, d2);
    uint32_t t3 = engine.add_task(gid, d3);

    engine.add_edge(gid, t0, t1);  // 0 → 1
    engine.add_edge(gid, t0, t2);  // 0 → 2
    engine.add_edge(gid, t1, t3);  // 1 → 3
    engine.add_edge(gid, t2, t3);  // 2 → 3

    // Reset counters
    g_dispatch_count.store(0);
    g_order_idx.store(0);

    auto fut = engine.submit(gid);
    nf_status result = fut.get();

    CHECK(result == NF_OK);
    CHECK(g_dispatch_count.load() == 4);

    // Verify ordering: task_0 must have dispatched before task_3
    // (task_0 is always first since it's the only root)
    int order[4];
    for (int i = 0; i < 4; ++i) {
        order[i] = g_dispatch_order[i].load(std::memory_order_relaxed);
    }
    CHECK(order[0] == 0);  // task_0 is always first

    // task_3 must be last
    CHECK(order[3] == 3);

    engine.destroy_graph(gid);
    std::printf("  PASS: DAG topological sort (diamond)\n");
}

/* ================================================================== */
/*  Test 2: Linear chain ordering                                      */
/* ================================================================== */

static void test_linear_chain() {
    nf::PipelineEngine engine(1);

    auto vt = make_mock_vtable();
    engine.register_provider(
        reinterpret_cast<nf_provider>(&g_mock_prov),
        vt, NF_AFFINITY_CPU);

    uint32_t gid = engine.create_graph();

    // Chain: A → B → C
    nf_task_desc da{}, db{}, dc{};
    std::strcpy(da.op_name, "op_0");
    std::strcpy(db.op_name, "op_1");
    std::strcpy(dc.op_name, "op_2");

    uint32_t a = engine.add_task(gid, da);
    uint32_t b = engine.add_task(gid, db);
    uint32_t c = engine.add_task(gid, dc);

    engine.add_edge(gid, a, b);
    engine.add_edge(gid, b, c);

    g_dispatch_count.store(0);
    g_order_idx.store(0);

    auto fut = engine.submit(gid);
    CHECK(fut.get() == NF_OK);
    CHECK(g_dispatch_count.load() == 3);

    // Strict order: 0, 1, 2
    for (int i = 0; i < 3; ++i) {
        CHECK(g_dispatch_order[i].load() == i);
    }

    engine.destroy_graph(gid);
    std::printf("  PASS: linear chain ordering\n");
}

/* ================================================================== */
/*  Test 3: ContextHub — put, get, prefix matching                     */
/* ================================================================== */

static void test_context_hub_basic() {
    // 1 MB budget, LRU eviction
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    auto [buf1, ops1] = make_mock_tensor(4096);
    nf::TensorView tv1(buf1, ops1);

    // Token keys: [10,20,30,40] and [10,20,30,50]
    std::vector<int32_t> key1 = {10, 20, 30, 40};
    auto st = hub.put(key1, "planner_agent", std::move(tv1), 0, 1);
    CHECK(st == NF_OK);

    auto [buf2, ops2] = make_mock_tensor(2048);
    nf::TensorView tv2(buf2, ops2);

    std::vector<int32_t> key2 = {10, 20, 30, 50};
    st = hub.put(key2, "planner_agent", std::move(tv2), 0, 2);
    CHECK(st == NF_OK);

    // Exact match
    auto r1 = hub.get(std::span<const int32_t>(key1));
    CHECK(r1.found);
    CHECK(r1.match_len == 4);

    // Prefix match: [10,20,30,50,60] should match key2 (len 4)
    std::vector<int32_t> longer = {10, 20, 30, 50, 60};
    auto r2 = hub.get(std::span<const int32_t>(longer));
    CHECK(r2.found);
    CHECK(r2.match_len == 4);

    // No match
    std::vector<int32_t> miss = {99, 88, 77};
    auto r3 = hub.get(std::span<const int32_t>(miss));
    CHECK(!r3.found);

    // Stats
    auto s = hub.stats();
    CHECK(s.entry_count == 2);
    CHECK(s.used_bytes == 4096 + 2048);

    std::printf("  PASS: ContextHub basic put/get/prefix\n");
}

/* ================================================================== */
/*  Test 4: ContextHub — eviction under memory pressure                */
/* ================================================================== */

static void test_context_hub_eviction() {
    // Tiny budget: 8 KB, LRU eviction
    nf::ContextHub hub(8192, NF_EVICT_LRU);

    // Fill with 4 x 4KB entries (only 2 fit)
    for (int i = 0; i < 4; ++i) {
        auto [buf, ops] = make_mock_tensor(4096);
        nf::TensorView tv(buf, ops);
        std::vector<int32_t> key = {100, static_cast<int32_t>(i)};
        // ttl_ms = 1000 so they're evictable (not pinned)
        auto st = hub.put(key, "test", std::move(tv), 1000, i);
        CHECK(st == NF_OK);
    }

    auto s = hub.stats();
    // Budget is 8192, each entry is 4096, so max 2 entries
    CHECK(s.entry_count == 2);
    CHECK(s.used_bytes <= 8192);

    std::printf("  PASS: ContextHub eviction under pressure\n");
}

/* ================================================================== */
/*  Test 5: ContextHub — explicit evict by prefix                      */
/* ================================================================== */

static void test_context_hub_evict_prefix() {
    nf::ContextHub hub(1024 * 1024, NF_EVICT_LRU);

    auto [b1, o1] = make_mock_tensor(1024);
    std::vector<int32_t> key_a = {1, 2, 3};
    hub.put(key_a, "a", nf::TensorView(b1, o1), 0, 0);

    auto [b2, o2] = make_mock_tensor(1024);
    std::vector<int32_t> key_b = {4, 5, 6};
    hub.put(key_b, "b", nf::TensorView(b2, o2), 0, 0);

    CHECK(hub.stats().entry_count == 2);

    // Evict only key_a subtree
    hub.evict(std::span<const int32_t>(key_a));
    CHECK(hub.stats().entry_count == 1);

    auto r = hub.get(std::span<const int32_t>(key_a));
    CHECK(!r.found);

    r = hub.get(std::span<const int32_t>(key_b));
    CHECK(r.found);

    std::printf("  PASS: ContextHub evict by prefix\n");
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main() {
    std::printf("scheduler_test: Phase 3 — Pipeline & ContextHub\n");
    test_dag_topo_sort();
    test_linear_chain();
    test_context_hub_basic();
    test_context_hub_eviction();
    test_context_hub_evict_prefix();
    std::printf("OK: all Phase 3 tests passed\n");
    return 0;
}
