/**
 * @file profiling_test.cpp
 * @brief Phase 15 — E2E Profiling: TTFT + TPS instrumentation verification
 *
 * Exercises:
 *   1. ProfileTrace types: TaskProfile, GraphProfile, SteadyClock
 *   2. PipelineEngine profiling integration: submit → profile retrieval
 *   3. Timing invariants: monotonicity, causality across DAG edges
 *   4. TTFT / TPS metric computation
 *   5. JSON dump format verification
 *   6. Multi-graph isolation: profiles don't leak between graphs
 *   7. Concurrent submission profiling
 *
 * DAG topology (mirrors split_llama_mock):
 *   N0: prefill(T0, T1) → T2, T3   [root]
 *   N1: relay(T2, T3) → T4          [depends on N0]
 *   N2: decode(T4) → T5             [depends on N1]
 */

#include "neurofabric/engine/ProfileTrace.hpp"
#include "neurofabric/engine/PipelineEngine.hpp"
#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_scheduler_abi.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <sstream>
#include <thread>
#include <vector>

/* Always-on assertion — survives NDEBUG / Release builds */
#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #expr, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

/* ================================================================== */
/*  Mock Provider — configurable latency per op                        */
/* ================================================================== */

static constexpr int PREFILL_US = 2000;  /* 2 ms */
static constexpr int RELAY_US   = 1000;  /* 1 ms */
static constexpr int DECODE_US  = 500;   /* 0.5 ms */

static nf_status mock_dispatch(nf_provider, const char* op_name,
                               const nf_buffer*, uint32_t,
                               nf_buffer*, uint32_t) {
    if (std::strcmp(op_name, "prefill") == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(PREFILL_US));
    else if (std::strcmp(op_name, "relay") == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(RELAY_US));
    else if (std::strcmp(op_name, "decode") == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(DECODE_US));
    return NF_OK;
}

static const char* mock_name(nf_provider) { return "profile_mock"; }
static uint32_t mock_abi(nf_provider)     { return NF_ABI_VERSION; }
static nf_status mock_init(nf_provider)   { return NF_OK; }
static void mock_shutdown(nf_provider)    {}
static nf_status mock_sync(nf_provider)   { return NF_OK; }

static nf_provider_vtable make_mock_vt() {
    nf_provider_vtable vt{};
    vt.get_name = mock_name; vt.get_abi_version = mock_abi;
    vt.init = mock_init; vt.shutdown = mock_shutdown;
    vt.dispatch = mock_dispatch; vt.synchronize = mock_sync;
    return vt;
}

/* ================================================================== */
/*  Helper: build 3-node linear DAG and submit                         */
/* ================================================================== */

struct GraphSetup {
    uint32_t gid;
    std::future<nf_status> future;
};

static GraphSetup build_and_submit(nf::PipelineEngine& engine) {
    uint32_t gid = engine.create_graph();

    nf_task_desc d0{}; std::strncpy(d0.op_name, "prefill", 63);
    d0.affinity = NF_AFFINITY_CPU;
    uint32_t t0 = engine.add_task(gid, d0);

    nf_task_desc d1{}; std::strncpy(d1.op_name, "relay", 63);
    d1.affinity = NF_AFFINITY_CPU;
    uint32_t t1 = engine.add_task(gid, d1);

    nf_task_desc d2{}; std::strncpy(d2.op_name, "decode", 63);
    d2.affinity = NF_AFFINITY_CPU;
    uint32_t t2 = engine.add_task(gid, d2);

    engine.add_edge(gid, t0, t1);  /* prefill → relay */
    engine.add_edge(gid, t1, t2);  /* relay → decode */

    auto fut = engine.submit(gid);
    return {gid, std::move(fut)};
}

/* ================================================================== */
/*  Test 1: Timing monotonicity — t_submit ≤ t_enqueue ≤ t_start ≤ t_end */
/* ================================================================== */

static void test_timing_monotonicity() {
    std::printf("  test_timing_monotonicity...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int prov_id = 0;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    auto [gid, fut] = build_and_submit(engine);
    nf_status st = fut.get();
    REQUIRE(st == NF_OK);

    auto prof = engine.graph_profile(gid);
    REQUIRE(prof != nullptr);
    REQUIRE(prof->tasks.size() == 3);

    /* t_submit must be before all task timestamps */
    for (size_t i = 0; i < 3; ++i) {
        auto& tp = prof->tasks[i];
        REQUIRE(prof->t_submit <= tp.t_enqueue);
        REQUIRE(tp.t_enqueue <= tp.t_start);
        REQUIRE(tp.t_start <= tp.t_end);
    }
    std::printf("    all 3 tasks: submit ≤ enqueue ≤ start ≤ end ✓\n");
    std::printf("  PASS: timing_monotonicity\n");
}

/* ================================================================== */
/*  Test 2: Causality — predecessor end ≤ successor enqueue            */
/* ================================================================== */

static void test_causality() {
    std::printf("  test_causality...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int prov_id = 1;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    auto [gid, fut] = build_and_submit(engine);
    REQUIRE(fut.get() == NF_OK);

    auto prof = engine.graph_profile(gid);
    REQUIRE(prof != nullptr);

    /* N0.end ≤ N1.enqueue (prefill completes before relay enqueued) */
    REQUIRE(prof->tasks[0].t_end <= prof->tasks[1].t_enqueue);
    /* N1.end ≤ N2.enqueue (relay completes before decode enqueued) */
    REQUIRE(prof->tasks[1].t_end <= prof->tasks[2].t_enqueue);

    std::printf("    prefill.end ≤ relay.enqueue ✓\n");
    std::printf("    relay.end ≤ decode.enqueue ✓\n");
    std::printf("  PASS: causality\n");
}

/* ================================================================== */
/*  Test 3: TTFT — time to first token (first task end - submit)       */
/* ================================================================== */

static void test_ttft() {
    std::printf("  test_ttft...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int prov_id = 2;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    auto [gid, fut] = build_and_submit(engine);
    REQUIRE(fut.get() == NF_OK);

    auto prof = engine.graph_profile(gid);
    REQUIRE(prof != nullptr);

    /* TTFT via free helper */
    auto ttft_raw = nf::duration_us(prof->t_submit, prof->tasks[0].t_end);
    std::printf("    TTFT (raw) = %.1f µs\n", ttft_raw);

    /* TTFT via GraphProfile::ttft_us() — finds first decode-class task */
    auto ttft = prof->ttft_us("decode");
    std::printf("    TTFT (decode) = %.1f µs\n", ttft);

    /* decode is the last task in the chain, so ttft_us("decode") should be
       >= total pipeline time (prefill + relay + decode) */
    double pipeline_min = (PREFILL_US + RELAY_US + DECODE_US) * 0.8;
    REQUIRE(ttft >= pipeline_min);
    REQUIRE(ttft < (PREFILL_US + RELAY_US + DECODE_US) * 10.0);

    /* first_task_latency_us() should be <= ttft since prefill ends first */
    auto first_lat = prof->first_task_latency_us();
    REQUIRE(first_lat >= PREFILL_US * 0.8);
    REQUIRE(first_lat <= ttft);

    std::printf("  PASS: ttft\n");
}

/* ================================================================== */
/*  Test 4: TPS — tokens per second (total pipeline throughput)        */
/* ================================================================== */

static void test_tps() {
    std::printf("  test_tps...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int prov_id = 3;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    auto [gid, fut] = build_and_submit(engine);
    REQUIRE(fut.get() == NF_OK);

    auto prof = engine.graph_profile(gid);
    REQUIRE(prof != nullptr);

    /* Total pipeline latency via GraphProfile::total_us() */
    auto total = prof->total_us();
    std::printf("    total pipeline = %.1f µs\n", total);

    /* Expected: ~3500 µs (2000 + 1000 + 500) + scheduling overhead */
    double expected_min = (PREFILL_US + RELAY_US + DECODE_US) * 0.8;
    REQUIRE(total >= expected_min);

    /* TPS via GraphProfile::tps() — uses decode dispatch time */
    constexpr uint32_t MOCK_TOKENS = 128;
    double tps_val = prof->tps(MOCK_TOKENS, "decode");
    std::printf("    TPS (128 tokens, decode) = %.0f tok/s\n", tps_val);
    REQUIRE(tps_val > 0.0);

    /* Cross-check: manual TPS from total pipeline time */
    double tps_total = MOCK_TOKENS / (total * 1e-6);
    std::printf("    TPS (128 tokens, total) = %.0f tok/s\n", tps_total);
    REQUIRE(tps_total > 0.0);

    /* decode-only TPS should be >= total TPS (decode is shorter than full pipeline) */
    REQUIRE(tps_val >= tps_total);

    std::printf("  PASS: tps\n");
}

/* ================================================================== */
/*  Test 5: JSON dump format                                           */
/* ================================================================== */

static void test_json_dump() {
    std::printf("  test_json_dump...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int prov_id = 4;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    auto [gid, fut] = build_and_submit(engine);
    REQUIRE(fut.get() == NF_OK);

    auto prof = engine.graph_profile(gid);
    REQUIRE(prof != nullptr);

    std::string json = prof->dump_json();
    REQUIRE(!json.empty());

    /* Verify key fields present */
    REQUIRE(json.find("\"submit_us\"") != std::string::npos);
    REQUIRE(json.find("\"tasks\"") != std::string::npos);
    REQUIRE(json.find("\"task_id\"") != std::string::npos);
    REQUIRE(json.find("\"enqueue_us\"") != std::string::npos);
    REQUIRE(json.find("\"start_us\"") != std::string::npos);
    REQUIRE(json.find("\"end_us\"") != std::string::npos);
    REQUIRE(json.find("\"dispatch_us\"") != std::string::npos);

    /* Should have 3 task entries */
    size_t count = 0;
    size_t pos = 0;
    while ((pos = json.find("\"task_id\"", pos)) != std::string::npos) {
        ++count; ++pos;
    }
    REQUIRE(count == 3);

    std::printf("    JSON length = %zu bytes, 3 task entries ✓\n", json.size());
    std::printf("  PASS: json_dump\n");
}

/* ================================================================== */
/*  Test 6: Multi-graph isolation                                      */
/* ================================================================== */

static void test_multi_graph_isolation() {
    std::printf("  test_multi_graph_isolation...\n");

    nf::PipelineEngine engine;
    auto vt = make_mock_vt();
    int prov_id = 5;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    /* Submit two graphs sequentially */
    auto [gid1, fut1] = build_and_submit(engine);
    REQUIRE(fut1.get() == NF_OK);

    auto [gid2, fut2] = build_and_submit(engine);
    REQUIRE(fut2.get() == NF_OK);

    REQUIRE(gid1 != gid2);

    auto p1 = engine.graph_profile(gid1);
    auto p2 = engine.graph_profile(gid2);
    REQUIRE(p1 != nullptr);
    REQUIRE(p2 != nullptr);
    REQUIRE(p1 != p2);  /* different profile objects */

    /* Graph 2 submitted after graph 1 completed */
    REQUIRE(p1->t_submit <= p2->t_submit);

    /* Each profile has its own 3 tasks */
    REQUIRE(p1->tasks.size() == 3);
    REQUIRE(p2->tasks.size() == 3);

    std::printf("    two graphs, independent profiles ✓\n");
    std::printf("  PASS: multi_graph_isolation\n");
}

/* ================================================================== */
/*  Test 7: Concurrent submission profiling                            */
/* ================================================================== */

static void test_concurrent_profiling() {
    std::printf("  test_concurrent_profiling...\n");

    nf::PipelineEngine engine(4);  /* 4 threads */
    auto vt = make_mock_vt();
    int prov_id = 6;
    engine.register_provider(
        reinterpret_cast<nf_provider>(&prov_id), vt, NF_AFFINITY_CPU);

    constexpr int N_GRAPHS = 4;
    std::vector<uint32_t> gids(N_GRAPHS);
    std::vector<std::future<nf_status>> futs;

    for (int i = 0; i < N_GRAPHS; ++i) {
        auto [gid, fut] = build_and_submit(engine);
        gids[i] = gid;
        futs.push_back(std::move(fut));
    }

    for (auto& f : futs)
        REQUIRE(f.get() == NF_OK);

    /* All profiles must exist and be valid */
    for (int i = 0; i < N_GRAPHS; ++i) {
        auto prof = engine.graph_profile(gids[i]);
        REQUIRE(prof != nullptr);
        REQUIRE(prof->tasks.size() == 3);

        /* Monotonicity within each graph */
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(prof->t_submit <= prof->tasks[j].t_enqueue);
            REQUIRE(prof->tasks[j].t_enqueue <= prof->tasks[j].t_start);
            REQUIRE(prof->tasks[j].t_start <= prof->tasks[j].t_end);
        }
    }

    std::printf("    %d concurrent graphs, all profiles valid ✓\n", N_GRAPHS);
    std::printf("  PASS: concurrent_profiling\n");
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main() {
    std::printf("=== profiling_test ===\n");

    test_timing_monotonicity();
    test_causality();
    test_ttft();
    test_tps();
    test_json_dump();
    test_multi_graph_isolation();
    test_concurrent_profiling();

    std::printf("=== ALL 7 TESTS PASSED ===\n");
    return 0;
}
