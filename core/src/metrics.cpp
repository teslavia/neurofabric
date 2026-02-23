/**
 * @file metrics.cpp
 * @brief Phase 34-G: Metrics collector + structured logging implementation
 */

#include "neurofabric/abi/metrics.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

/* ---- Logging ---- */

static nf_log_fn      s_log_fn    = nullptr;
static void*          s_log_ud    = nullptr;
static nf_log_level   s_log_level = NF_LOG_INFO;
static std::mutex     s_log_mu;

void nf_log_set_callback(nf_log_fn fn, void* userdata) {
    std::lock_guard<std::mutex> lk(s_log_mu);
    s_log_fn = fn;
    s_log_ud = userdata;
}

void nf_log_set_level(nf_log_level level) {
    s_log_level = level;
}

nf_log_level nf_log_get_level(void) {
    return s_log_level;
}

static const char* level_str(nf_log_level l) {
    switch (l) {
        case NF_LOG_TRACE: return "TRACE";
        case NF_LOG_DEBUG: return "DEBUG";
        case NF_LOG_INFO:  return "INFO";
        case NF_LOG_WARN:  return "WARN";
        case NF_LOG_ERROR: return "ERROR";
        case NF_LOG_FATAL: return "FATAL";
        default:           return "?";
    }
}

void nf_log(nf_log_level level, const char* component, const char* fmt, ...) {
    if (level < s_log_level) return;

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    std::lock_guard<std::mutex> lk(s_log_mu);
    if (s_log_fn) {
        s_log_fn(level, component, buf, s_log_ud);
    } else {
        std::fprintf(stderr, "[%s] %s: %s\n", level_str(level), component, buf);
    }
}

/* ---- Metrics ---- */

static struct {
    std::atomic<uint64_t> prefill_us{0};
    std::atomic<uint64_t> decode_us{0};
    std::atomic<uint64_t> total_us{0};
    std::atomic<uint32_t> prefill_tokens{0};
    std::atomic<uint32_t> decode_tokens{0};
    std::atomic<uint64_t> gpu_mem_used{0};
    std::atomic<uint64_t> gpu_mem_peak{0};
    std::atomic<uint32_t> active_reqs{0};
    std::atomic<uint32_t> completed_reqs{0};
    std::atomic<uint32_t> failed_reqs{0};
} s_metrics;

void nf_metrics_reset(void) {
    s_metrics.prefill_us.store(0);
    s_metrics.decode_us.store(0);
    s_metrics.total_us.store(0);
    s_metrics.prefill_tokens.store(0);
    s_metrics.decode_tokens.store(0);
    s_metrics.gpu_mem_used.store(0);
    s_metrics.gpu_mem_peak.store(0);
    s_metrics.active_reqs.store(0);
    s_metrics.completed_reqs.store(0);
    s_metrics.failed_reqs.store(0);
}

void nf_metrics_record_prefill(uint64_t us, uint32_t tokens) {
    s_metrics.prefill_us.fetch_add(us);
    s_metrics.prefill_tokens.fetch_add(tokens);
    s_metrics.total_us.fetch_add(us);
}

void nf_metrics_record_decode(uint64_t us, uint32_t tokens) {
    s_metrics.decode_us.fetch_add(us);
    s_metrics.decode_tokens.fetch_add(tokens);
    s_metrics.total_us.fetch_add(us);
}

void nf_metrics_record_memory(uint64_t used, uint64_t peak) {
    s_metrics.gpu_mem_used.store(used);
    uint64_t cur_peak = s_metrics.gpu_mem_peak.load();
    while (peak > cur_peak && !s_metrics.gpu_mem_peak.compare_exchange_weak(cur_peak, peak));
}

void nf_metrics_record_request(int completed, int failed) {
    if (completed > 0) {
        s_metrics.completed_reqs.fetch_add((uint32_t)completed);
    }
    if (failed > 0) {
        s_metrics.failed_reqs.fetch_add((uint32_t)failed);
    }
}

void nf_metrics_snapshot_get(nf_metrics_snapshot* out) {
    out->prefill_us = s_metrics.prefill_us.load();
    out->decode_us = s_metrics.decode_us.load();
    out->total_us = s_metrics.total_us.load();
    out->tokens_generated = s_metrics.decode_tokens.load();
    double dec_s = (double)out->decode_us / 1e6;
    out->tokens_per_second = (dec_s > 0) ? (double)out->tokens_generated / dec_s : 0;
    out->gpu_memory_used = s_metrics.gpu_mem_used.load();
    out->gpu_memory_peak = s_metrics.gpu_mem_peak.load();
    out->kv_cache_bytes = 0; /* filled by caller if needed */
    out->active_requests = s_metrics.active_reqs.load();
    out->completed_requests = s_metrics.completed_reqs.load();
    out->failed_requests = s_metrics.failed_reqs.load();
}

size_t nf_metrics_to_json(char* buf, size_t buf_size) {
    nf_metrics_snapshot snap;
    nf_metrics_snapshot_get(&snap);
    int n = snprintf(buf, buf_size,
        "{\"prefill_us\":%llu,\"decode_us\":%llu,\"total_us\":%llu,"
        "\"tokens_generated\":%u,\"tokens_per_second\":%.2f,"
        "\"gpu_memory_used\":%llu,\"gpu_memory_peak\":%llu,"
        "\"active_requests\":%u,\"completed_requests\":%u,\"failed_requests\":%u}",
        (unsigned long long)snap.prefill_us, (unsigned long long)snap.decode_us,
        (unsigned long long)snap.total_us,
        snap.tokens_generated, snap.tokens_per_second,
        (unsigned long long)snap.gpu_memory_used, (unsigned long long)snap.gpu_memory_peak,
        snap.active_requests, snap.completed_requests, snap.failed_requests);
    return (n > 0) ? (size_t)n : 0;
}

size_t nf_metrics_to_prometheus(char* buf, size_t buf_size) {
    nf_metrics_snapshot snap;
    nf_metrics_snapshot_get(&snap);
    int n = snprintf(buf, buf_size,
        "# HELP nf_prefill_us Total prefill latency in microseconds\n"
        "# TYPE nf_prefill_us counter\n"
        "nf_prefill_us %llu\n"
        "# HELP nf_decode_us Total decode latency in microseconds\n"
        "# TYPE nf_decode_us counter\n"
        "nf_decode_us %llu\n"
        "# HELP nf_tokens_generated Total tokens generated\n"
        "# TYPE nf_tokens_generated counter\n"
        "nf_tokens_generated %u\n"
        "# HELP nf_tokens_per_second Current throughput\n"
        "# TYPE nf_tokens_per_second gauge\n"
        "nf_tokens_per_second %.2f\n"
        "# HELP nf_gpu_memory_bytes GPU memory usage\n"
        "# TYPE nf_gpu_memory_bytes gauge\n"
        "nf_gpu_memory_bytes %llu\n"
        "# HELP nf_requests_completed Total completed requests\n"
        "# TYPE nf_requests_completed counter\n"
        "nf_requests_completed %u\n"
        "# HELP nf_requests_failed Total failed requests\n"
        "# TYPE nf_requests_failed counter\n"
        "nf_requests_failed %u\n",
        (unsigned long long)snap.prefill_us, (unsigned long long)snap.decode_us,
        snap.tokens_generated, snap.tokens_per_second,
        (unsigned long long)snap.gpu_memory_used,
        snap.completed_requests, snap.failed_requests);
    return (n > 0) ? (size_t)n : 0;
}
