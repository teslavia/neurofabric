/**
 * @file ProfileTrace.hpp
 * @brief Lightweight per-task profiling — TTFT, TPS, per-node latency
 *
 * Zero-overhead when NF_PROFILE is not defined (all probes compile to no-ops).
 * When enabled, records steady_clock timestamps at:
 *   - task enqueue (enters ready queue)
 *   - dispatch start (provider call begins)
 *   - dispatch end (provider call returns)
 *
 * Aggregation:
 *   - TTFT: time from graph submit to first decode-class task completion
 *   - TPS:  output_tokens / decode_total_seconds
 *   - Per-task: queue wait, dispatch latency, total latency
 */

#ifndef NEURALOS_KERNEL_PROFILE_TRACE_HPP
#define NEURALOS_KERNEL_PROFILE_TRACE_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace neuralOS { namespace kernel {

using SteadyClock = std::chrono::steady_clock;
using TimePoint   = SteadyClock::time_point;
using Duration    = std::chrono::duration<double, std::micro>;  // microseconds

/* ================================================================== */
/*  TaskProfile — per-task timing record                               */
/* ================================================================== */

struct TaskProfile {
    uint32_t    task_id     = 0;
    char        op_name[64] = {};

    TimePoint   t_enqueue   = {};   // entered ready queue
    TimePoint   t_start     = {};   // dispatch call begins
    TimePoint   t_end       = {};   // dispatch call returns

    /** Queue wait time (microseconds). */
    double queue_us() const {
        return std::chrono::duration_cast<Duration>(t_start - t_enqueue).count();
    }
    /** Dispatch latency (microseconds). */
    double dispatch_us() const {
        return std::chrono::duration_cast<Duration>(t_end - t_start).count();
    }

    /** Total latency from enqueue to completion (microseconds). */
    double total_us() const {
        return std::chrono::duration_cast<Duration>(t_end - t_enqueue).count();
    }
};

/* ================================================================== */
/*  GraphProfile — per-graph aggregation                               */
/* ================================================================== */

struct GraphProfile {
    TimePoint                   t_submit = {};   // graph submission time
    std::vector<TaskProfile>    tasks;

    /** Time from graph submit to first task completion (microseconds). */
    double first_task_latency_us() const {
        if (tasks.empty()) return 0.0;
        TimePoint earliest_end = tasks[0].t_end;
        for (auto& tp : tasks) {
            if (tp.t_end < earliest_end) earliest_end = tp.t_end;
        }
        return std::chrono::duration_cast<Duration>(earliest_end - t_submit).count();
    }

    /** Time from graph submit to last task completion (microseconds). */
    double total_us() const {
        if (tasks.empty()) return 0.0;
        TimePoint latest_end = tasks[0].t_end;
        for (auto& tp : tasks) {
            if (tp.t_end > latest_end) latest_end = tp.t_end;
        }
        return std::chrono::duration_cast<Duration>(latest_end - t_submit).count();
    }

    /**
     * TTFT — Time To First Token.
     * Defined as: submit → first decode-class task completion.
     * If no decode task found, falls back to first task completion.
     * @param decode_prefix  op_name prefix identifying decode tasks (e.g. "decode")
     */
    double ttft_us(const char* decode_prefix = "decode") const {
        TimePoint first_decode = {};
        bool found = false;
        for (auto& tp : tasks) {
            if (std::strncmp(tp.op_name, decode_prefix,
                             std::strlen(decode_prefix)) == 0) {
                if (!found || tp.t_end < first_decode) {
                    first_decode = tp.t_end;
                    found = true;
                }
            }
        }
        if (!found) return first_task_latency_us();
        return std::chrono::duration_cast<Duration>(first_decode - t_submit).count();
    }

    /**
     * TPS — Tokens Per Second.
     * @param n_tokens  number of output tokens produced by the decode stage
     * @param decode_prefix  op_name prefix identifying decode tasks
     */
    double tps(uint32_t n_tokens, const char* decode_prefix = "decode") const {
        double decode_total_us = 0.0;
        for (auto& tp : tasks) {
            if (std::strncmp(tp.op_name, decode_prefix,
                             std::strlen(decode_prefix)) == 0) {
                decode_total_us += tp.dispatch_us();
            }
        }
        if (decode_total_us <= 0.0) return 0.0;
        return static_cast<double>(n_tokens) / (decode_total_us * 1e-6);
    }

    /** Find profile for a specific task by ID. Returns nullptr if not found. */
    const TaskProfile* find_task(uint32_t task_id) const {
        for (auto& tp : tasks) {
            if (tp.task_id == task_id) return &tp;
        }
        return nullptr;
    }

    /**
     * Dump profile as JSON string.
     * Timestamps are relative to t_submit (microseconds).
     */
    std::string dump_json() const {
        std::string s;
        s.reserve(512);
        s += "{\n  \"submit_us\": 0.0,\n  \"tasks\": [\n";
        for (size_t i = 0; i < tasks.size(); ++i) {
            auto& tp = tasks[i];
            double enq = std::chrono::duration_cast<Duration>(
                tp.t_enqueue - t_submit).count();
            double st  = std::chrono::duration_cast<Duration>(
                tp.t_start - t_submit).count();
            double en  = std::chrono::duration_cast<Duration>(
                tp.t_end - t_submit).count();
            double disp = tp.dispatch_us();

            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "    {\"task_id\": %u, \"op_name\": \"%s\", "
                "\"enqueue_us\": %.1f, \"start_us\": %.1f, "
                "\"end_us\": %.1f, \"dispatch_us\": %.1f}",
                tp.task_id, tp.op_name, enq, st, en, disp);
            s += buf;
            if (i + 1 < tasks.size()) s += ",";
            s += "\n";
        }
        s += "  ]\n}";
        return s;
    }
};

/** Free helper: microseconds between two time points. */
inline double duration_us(TimePoint from, TimePoint to) {
    return std::chrono::duration_cast<Duration>(to - from).count();
}

}} // neuralOS::kernel

// Backward compatibility
namespace nf {
    using neuralOS::kernel::SteadyClock;
    using neuralOS::kernel::TimePoint;
    using neuralOS::kernel::Duration;
    using neuralOS::kernel::TaskProfile;
    using neuralOS::kernel::GraphProfile;
    using neuralOS::kernel::duration_us;
}

#endif // NEURALOS_KERNEL_PROFILE_TRACE_HPP
