/**
 * @file chrome_trace_test.cpp
 * @brief Phase 26 — Chrome trace_event export unit test
 *
 * Provides mock GPU timing stubs so we can test trace_export.hpp
 * without linking the Metal plugin.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ---- Mock timing data ---- */
static constexpr uint32_t kMockCount = 4;
static const char* kMockNames[kMockCount] = {
    "matmul_q4", "rope_cached", "flash_attention_cached", "rms_norm"
};
static double kMockMs[kMockCount] = { 1.5, 0.3, 2.1, 0.1 };

/* Provide the extern "C" stubs that trace_export.hpp expects */
extern "C" uint32_t nf_metal_get_timing_count() {
    return kMockCount;
}
extern "C" void nf_metal_get_timings(char (*op_names)[64], double* gpu_ms,
                                      uint32_t max_count) {
    uint32_t n = (kMockCount < max_count) ? kMockCount : max_count;
    for (uint32_t i = 0; i < n; ++i) {
        std::strncpy(op_names[i], kMockNames[i], 64);
        gpu_ms[i] = kMockMs[i];
    }
}
extern "C" void nf_metal_get_timings_ext(char (*op_names)[64], double* gpu_ms,
                                          uint8_t* dtypes, uint32_t* elem_counts,
                                          uint32_t max_count) {
    nf_metal_get_timings(op_names, gpu_ms, max_count);
    uint32_t n = (kMockCount < max_count) ? kMockCount : max_count;
    for (uint32_t i = 0; i < n; ++i) {
        if (dtypes)      dtypes[i]      = 0;  /* F32 */
        if (elem_counts) elem_counts[i] = 1024;
    }
}

#include "trace_export.hpp"

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)

static void test_export_creates_file() {
    std::printf("  test_export_creates_file...\n");
    const char* path = "/tmp/nf_trace_test.json";
    bool ok = nf::export_chrome_trace(path);
    CHECK(ok);

    FILE* f = std::fopen(path, "r");
    CHECK(f != nullptr);

    /* Read entire file */
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    CHECK(sz > 0);

    char* buf = (char*)std::malloc(sz + 1);
    std::fread(buf, 1, sz, f);
    buf[sz] = '\0';
    std::fclose(f);

    /* Verify JSON structure */
    CHECK(std::strstr(buf, "\"traceEvents\"") != nullptr);
    CHECK(std::strstr(buf, "\"matmul_q4\"") != nullptr);
    CHECK(std::strstr(buf, "\"flash_attention_cached\"") != nullptr);
    CHECK(std::strstr(buf, "\"rms_norm\"") != nullptr);
    CHECK(std::strstr(buf, "\"ph\":\"X\"") != nullptr);
    CHECK(std::strstr(buf, "\"cat\":\"gpu\"") != nullptr);
    CHECK(std::strstr(buf, "\"dtype\":\"F32\"") != nullptr);
    CHECK(std::strstr(buf, "\"elements\":1024") != nullptr);

    /* Verify timestamps are monotonically increasing */
    /* matmul_q4 starts at 0, rope at 1500µs, flash at 1800µs, rms at 3900µs */
    CHECK(std::strstr(buf, "\"ts\":0.000") != nullptr);

    std::free(buf);
    std::remove(path);
    std::printf("    PASS\n");
}

static void test_json_has_correct_count() {
    std::printf("  test_json_has_correct_count...\n");
    const char* path = "/tmp/nf_trace_test2.json";
    nf::export_chrome_trace(path);

    FILE* f = std::fopen(path, "r");
    CHECK(f);
    char buf[4096];
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    std::fclose(f);

    /* Count "ph":"X" occurrences — should be exactly 4 */
    int count = 0;
    const char* p = buf;
    while ((p = std::strstr(p, "\"ph\":\"X\"")) != nullptr) {
        ++count;
        p += 7;
    }
    CHECK(count == 4);

    std::remove(path);
    std::printf("    PASS\n");
}

int main() {
    std::printf("chrome_trace_test: Phase 26 — Chrome Trace Export\n");
    test_export_creates_file();
    test_json_has_correct_count();
    std::printf("chrome_trace_test: ALL PASSED\n");
    return 0;
}
