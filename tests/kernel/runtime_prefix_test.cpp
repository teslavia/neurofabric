/**
 * @file runtime_prefix_test.cpp
 * @brief Phase 43 — NeuralOSRuntime prefix sharing tests
 */

#include "neuralOS/kernel/NeuralOSRuntime.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== NeuralOSRuntime Prefix Sharing Test ===\n");

    nf::PagedKVCache kv;
    kv.init(64, 4, 2, 4, 64);
    nf::RequestScheduler sched;

    neuralOS::kernel::NeuralOSRuntime runtime(&kv, &sched);

    auto* hub = runtime.context_hub();
    CHECK(hub != nullptr, "context_hub accessible");

    /* Insert a prefix into ContextHub */
    std::vector<int32_t> prefix = {10, 20, 30, 40, 50};
    nf::TensorView tv;
    hub->put(std::span<const int32_t>(prefix.data(), prefix.size()),
             "test_agent", std::move(tv), 0, 0);

    /* try_prefix_share with matching prefix — query is superset of stored */
    std::vector<int32_t> query = {10, 20, 30, 40, 50, 60};
    uint32_t match_len = runtime.try_prefix_share(query);
    CHECK(match_len == 5, "prefix match length == 5");
    CHECK(runtime.stats().prefix_hits == 1, "prefix_hits == 1");

    /* Insert a shorter prefix too */
    std::vector<int32_t> short_prefix = {10, 20, 30};
    nf::TensorView tv2;
    hub->put(std::span<const int32_t>(short_prefix.data(), short_prefix.size()),
             "test_agent2", std::move(tv2), 0, 1);

    /* try_prefix_share with query matching the shorter prefix */
    std::vector<int32_t> partial = {10, 20, 30, 99};
    uint32_t partial_len = runtime.try_prefix_share(partial);
    CHECK(partial_len >= 3, "partial prefix matched");
    CHECK(runtime.stats().prefix_hits >= 2, "prefix_hits incremented");

    /* try_prefix_share with no match */
    std::vector<int32_t> nomatch = {99, 98, 97};
    uint32_t no_len = runtime.try_prefix_share(nomatch);
    CHECK(no_len == 0, "no match returns 0");

    /* Submit with shared prefix */
    uint32_t hits_before = runtime.stats().prefix_hits;
    nf::InferenceRequest req;
    req.prompt_tokens = {10, 20, 30, 40, 50};
    req.max_new_tokens = 4;
    runtime.submit(req);
    CHECK(runtime.stats().prefix_hits > hits_before, "submit increments prefix_hits");

    printf("PASS: all NeuralOSRuntime prefix sharing tests passed\n");
    return 0;
}
