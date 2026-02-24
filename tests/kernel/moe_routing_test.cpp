/**
 * @file moe_routing_test.cpp
 * @brief Phase 34-B: MoE expert routing test (CPU reference)
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } \
} while(0)

/* CPU reference: softmax top-K gating */
struct GateResult {
    int32_t  expert_ids[8];   /* selected expert indices */
    float    weights[8];      /* normalized gate weights */
    uint32_t k;               /* number of selected experts */
};

static GateResult cpu_top_k_gate(const float* logits, uint32_t n_experts, uint32_t top_k) {
    GateResult r{};
    r.k = top_k;

    /* Softmax over all experts */
    std::vector<float> probs(n_experts);
    float max_val = *std::max_element(logits, logits + n_experts);
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_experts; ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < n_experts; ++i) probs[i] /= sum;

    /* Top-K selection */
    std::vector<uint32_t> indices(n_experts);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](uint32_t a, uint32_t b) { return probs[a] > probs[b]; });

    /* Renormalize selected weights */
    float sel_sum = 0.0f;
    for (uint32_t i = 0; i < top_k; ++i) sel_sum += probs[indices[i]];
    for (uint32_t i = 0; i < top_k; ++i) {
        r.expert_ids[i] = (int32_t)indices[i];
        r.weights[i] = probs[indices[i]] / sel_sum;
    }
    return r;
}

/* CPU reference: scatter-gather (weighted sum of expert outputs) */
static void cpu_scatter_gather(
    const float* expert_outputs,  /* [n_experts][dim] */
    const GateResult& gate,
    uint32_t dim,
    float* output)                /* [dim] */
{
    std::memset(output, 0, dim * sizeof(float));
    for (uint32_t i = 0; i < gate.k; ++i) {
        uint32_t eid = (uint32_t)gate.expert_ids[i];
        float w = gate.weights[i];
        for (uint32_t d = 0; d < dim; ++d)
            output[d] += w * expert_outputs[eid * dim + d];
    }
}

int main() {
    std::printf("=== Phase 34-B: MoE Routing Test ===\n");

    /* Test 1: Top-2 gating with 8 experts (Mixtral config) */
    {
        const uint32_t N_EXPERTS = 8, TOP_K = 2;
        float logits[8] = {1.0f, 3.0f, 0.5f, 2.0f, -1.0f, 0.0f, 4.0f, 1.5f};
        auto gate = cpu_top_k_gate(logits, N_EXPERTS, TOP_K);

        CHECK(gate.k == 2, "top-2 selected");
        CHECK(gate.expert_ids[0] == 6, "highest expert is 6 (logit=4.0)");
        CHECK(gate.expert_ids[1] == 1, "second expert is 1 (logit=3.0)");

        float wsum = gate.weights[0] + gate.weights[1];
        CHECK(std::fabs(wsum - 1.0f) < 1e-5f, "weights sum to 1.0");
        CHECK(gate.weights[0] > gate.weights[1], "weight[0] > weight[1]");
        std::printf("  [PASS] top-2 gating (8 experts)\n");
    }

    /* Test 2: Top-1 gating (greedy) */
    {
        float logits[4] = {0.1f, 0.9f, 0.3f, 0.2f};
        auto gate = cpu_top_k_gate(logits, 4, 1);
        CHECK(gate.k == 1, "top-1 selected");
        CHECK(gate.expert_ids[0] == 1, "expert 1 selected");
        CHECK(std::fabs(gate.weights[0] - 1.0f) < 1e-5f, "single weight = 1.0");
        std::printf("  [PASS] top-1 gating (greedy)\n");
    }

    /* Test 3: Scatter-gather weighted combination */
    {
        const uint32_t DIM = 4, N_EXPERTS = 3;
        float expert_out[3 * 4] = {
            1.0f, 0.0f, 0.0f, 0.0f,  /* expert 0 */
            0.0f, 1.0f, 0.0f, 0.0f,  /* expert 1 */
            0.0f, 0.0f, 1.0f, 0.0f,  /* expert 2 */
        };
        GateResult gate{};
        gate.k = 2;
        gate.expert_ids[0] = 0; gate.weights[0] = 0.7f;
        gate.expert_ids[1] = 2; gate.weights[1] = 0.3f;

        float output[4];
        cpu_scatter_gather(expert_out, gate, DIM, output);
        CHECK(std::fabs(output[0] - 0.7f) < 1e-5f, "dim 0 = 0.7");
        CHECK(std::fabs(output[1] - 0.0f) < 1e-5f, "dim 1 = 0.0");
        CHECK(std::fabs(output[2] - 0.3f) < 1e-5f, "dim 2 = 0.3");
        CHECK(std::fabs(output[3] - 0.0f) < 1e-5f, "dim 3 = 0.0");
        std::printf("  [PASS] scatter-gather weighted combination\n");
    }

    /* Test 4: Equal logits → uniform selection */
    {
        float logits[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        auto gate = cpu_top_k_gate(logits, 4, 2);
        CHECK(gate.k == 2, "top-2 from uniform");
        CHECK(std::fabs(gate.weights[0] - 0.5f) < 1e-5f, "uniform weight 0.5");
        CHECK(std::fabs(gate.weights[1] - 0.5f) < 1e-5f, "uniform weight 0.5");
        std::printf("  [PASS] uniform logits\n");
    }

    /* Test 5: Batch gating (multiple tokens) */
    {
        const uint32_t BATCH = 3, N_EXPERTS = 8, TOP_K = 2;
        float batch_logits[3][8] = {
            {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f},
            {1.0f, 1.0f, 5.0f, 1.0f, 1.0f, 5.0f, 1.0f, 1.0f},
        };
        for (uint32_t b = 0; b < BATCH; ++b) {
            auto gate = cpu_top_k_gate(batch_logits[b], N_EXPERTS, TOP_K);
            CHECK(gate.k == 2, "batch token top-2");
        }
        auto g0 = cpu_top_k_gate(batch_logits[0], N_EXPERTS, TOP_K);
        CHECK(g0.expert_ids[0] == 0, "batch[0] top expert = 0");
        auto g1 = cpu_top_k_gate(batch_logits[1], N_EXPERTS, TOP_K);
        CHECK(g1.expert_ids[0] == 7, "batch[1] top expert = 7");
        std::printf("  [PASS] batch gating (3 tokens)\n");
    }

    std::printf("=== All MoE routing tests passed ===\n");
    return 0;
}
