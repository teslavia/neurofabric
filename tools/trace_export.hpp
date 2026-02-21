/**
 * @file trace_export.hpp
 * @brief Chrome trace_event JSON exporter for GPU profiling data
 *
 * Phase 26: Flash Attention · 7B Validation · Trace Export.
 *
 * Reads from the Metal GPU timing ring buffer and writes Chrome
 * trace_event format JSON (loadable in chrome://tracing or Perfetto).
 */

#ifndef NF_TRACE_EXPORT_HPP
#define NF_TRACE_EXPORT_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>

/* GPU timing API (defined in metal_provider.mm) */
extern "C" uint32_t nf_metal_get_timing_count();
extern "C" void     nf_metal_get_timings(char (*op_names)[64], double* gpu_ms,
                                          uint32_t max_count);

namespace nf {

inline bool export_chrome_trace(const char* path) {
    uint32_t count = nf_metal_get_timing_count();
    if (count == 0) return false;

    constexpr uint32_t MAX_ENTRIES = 1024;
    char op_names[MAX_ENTRIES][64];
    double gpu_ms[MAX_ENTRIES];
    if (count > MAX_ENTRIES) count = MAX_ENTRIES;
    nf_metal_get_timings(op_names, gpu_ms, count);

    FILE* f = std::fopen(path, "w");
    if (!f) return false;

    std::fprintf(f, "{\"traceEvents\":[\n");

    double ts = 0.0;  /* cumulative timestamp in microseconds */
    for (uint32_t i = 0; i < count; ++i) {
        double dur_us = gpu_ms[i] * 1000.0;  /* ms → µs */
        if (dur_us < 0.0) dur_us = 0.0;
        std::fprintf(f,
            "  {\"name\":\"%s\",\"cat\":\"gpu\",\"ph\":\"X\","
            "\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":0}%s\n",
            op_names[i], ts, dur_us,
            (i + 1 < count) ? "," : "");
        ts += dur_us;
    }

    std::fprintf(f, "]}\n");
    std::fclose(f);
    return true;
}

} /* namespace nf */

#endif /* NF_TRACE_EXPORT_HPP */
