/**
 * @file metrics.h
 * @brief Phase 34-G: Structured logging & metrics collection
 *
 * C-ABI compatible metrics interface. Collects latency, throughput,
 * and memory usage. Supports JSON structured logging.
 */

#ifndef NF_METRICS_H
#define NF_METRICS_H

#include <stdint.h>
#include <stddef.h>

#include "neuro_fabric_abi.h" /* NF_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Log Levels ---- */
typedef enum nf_log_level {
    NF_LOG_TRACE = 0,
    NF_LOG_DEBUG = 1,
    NF_LOG_INFO  = 2,
    NF_LOG_WARN  = 3,
    NF_LOG_ERROR = 4,
    NF_LOG_FATAL = 5,
    NF_LOG_OFF   = 6
} nf_log_level;

/* ---- Log callback ---- */
typedef void (*nf_log_fn)(nf_log_level level, const char* component,
                           const char* message, void* userdata);

NF_API void nf_log_set_callback(nf_log_fn fn, void* userdata);
NF_API void nf_log_set_level(nf_log_level level);
NF_API nf_log_level nf_log_get_level(void);
NF_API void nf_log(nf_log_level level, const char* component, const char* fmt, ...);

/* ---- Metrics Collector ---- */

typedef struct nf_metrics_snapshot {
    uint64_t prefill_us;
    uint64_t decode_us;
    uint64_t total_us;
    uint32_t tokens_generated;
    double   tokens_per_second;
    uint64_t gpu_memory_used;
    uint64_t gpu_memory_peak;
    uint64_t kv_cache_bytes;
    uint32_t active_requests;
    uint32_t completed_requests;
    uint32_t failed_requests;
} nf_metrics_snapshot;

NF_API void nf_metrics_reset(void);
NF_API void nf_metrics_record_prefill(uint64_t us, uint32_t tokens);
NF_API void nf_metrics_record_decode(uint64_t us, uint32_t tokens);
NF_API void nf_metrics_record_memory(uint64_t used, uint64_t peak);
NF_API void nf_metrics_record_request(int completed, int failed);
NF_API void nf_metrics_snapshot_get(nf_metrics_snapshot* out);
NF_API size_t nf_metrics_to_json(char* buf, size_t buf_size);
NF_API size_t nf_metrics_to_prometheus(char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* NF_METRICS_H */
