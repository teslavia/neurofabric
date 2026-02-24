/**
 * @file paged_attn_test.cpp
 * @brief Phase 32 Step 2: flash_attention_paged kernel correctness tests
 *
 * 3 sub-tests:
 *   1. PagedAttn_single_token_decode — 1 head, 4 dim, 8 cached tokens across 2 blocks
 *   2. PagedAttn_prefill_4_tokens — prefill mode, verify causal masking
 *   3. PagedAttn_gqa_2h_1kv — GQA with 2 Q heads sharing 1 KV head
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

struct BufPair {
    nf_buffer buf = nullptr;
    nf_buffer_ops ops{};
};

static BufPair alloc_buf(nf_provider prov, nf_provider_mem_vtable& mem_vt,
                          nf_dtype dtype, size_t size_bytes) {
    BufPair bp;
    nf_tensor_desc d{}; d.dtype = dtype; d.ndim = 1;
    d.shape[0] = size_bytes; d.size_bytes = size_bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    mem_vt.alloc(prov, &req, &bp.ops, &bp.buf);
    return bp;
}

static float* map_f(BufPair& bp) { void* p; bp.ops.map(bp.buf, &p); return (float*)p; }
static uint32_t* map_u(BufPair& bp) { void* p; bp.ops.map(bp.buf, &p); return (uint32_t*)p; }
static void unmap(BufPair& bp) { bp.ops.unmap(bp.buf); }
static void release(BufPair& bp) { if (bp.buf) bp.ops.release(bp.buf); bp.buf = nullptr; }

/* CPU reference: standard attention with paged KV layout */
static void ref_paged_attn(
    const float* Q, const float* K_pool, const float* V_pool,
    const uint32_t* block_table, float* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim,
    uint32_t n_kv_heads, uint32_t step_idx, uint32_t block_size,
    uint32_t num_blocks, uint32_t seq_offset)
{
    for (uint32_t h = 0; h < n_heads; ++h) {
        uint32_t kv_head = h * n_kv_heads / n_heads;
        float scale = 1.0f / std::sqrt((float)head_dim);

        for (uint32_t qp = 0; qp < seq_len; ++qp) {
            uint32_t max_kv = (step_idx == 0) ? qp : step_idx;
            uint32_t q_base = (seq_offset + qp) * n_heads * head_dim + h * head_dim;

            /* Compute all scores */
            std::vector<float> scores(max_kv + 1);
            float max_score = -1e30f;
            for (uint32_t kp = 0; kp <= max_kv; ++kp) {
                uint32_t lb = kp / block_size;
                uint32_t off = kp % block_size;
                uint32_t phys = block_table[lb];
                uint32_t k_base = (phys * n_kv_heads + kv_head) * block_size * head_dim
                                  + off * head_dim;
                float dot = 0;
                for (uint32_t d = 0; d < head_dim; ++d)
                    dot += Q[q_base + d] * K_pool[k_base + d];
                scores[kp] = dot * scale;
                if (scores[kp] > max_score) max_score = scores[kp];
            }

            /* Softmax */
            float sum_exp = 0;
            for (uint32_t kp = 0; kp <= max_kv; ++kp) {
                scores[kp] = std::exp(scores[kp] - max_score);
                sum_exp += scores[kp];
            }

            /* Weighted sum of V */
            uint32_t out_base = (seq_offset + qp) * n_heads * head_dim + h * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) out[out_base + d] = 0;
            for (uint32_t kp = 0; kp <= max_kv; ++kp) {
                float w = scores[kp] / sum_exp;
                uint32_t lb = kp / block_size;
                uint32_t off = kp % block_size;
                uint32_t phys = block_table[lb];
                uint32_t v_base = (phys * n_kv_heads + kv_head) * block_size * head_dim
                                  + off * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d)
                    out[out_base + d] += w * V_pool[v_base + d];
            }
        }
    }
}

static void run_paged_attn_gpu(
    nf_provider prov, nf_provider_vtable& vt, nf_provider_mem_vtable& mem_vt,
    nf::PipelineEngine& engine,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim,
    uint32_t n_kv_heads, uint32_t step_idx, uint32_t block_size,
    uint32_t num_blocks, uint32_t seq_offset,
    const float* Q_data, const float* K_pool_data, const float* V_pool_data,
    const uint32_t* bt_data, float* out_data,
    size_t q_bytes, size_t pool_bytes, size_t bt_bytes, size_t out_bytes)
{
    auto q_buf   = alloc_buf(prov, mem_vt, NF_DTYPE_F32, q_bytes);
    auto kp_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_F32, pool_bytes);
    auto vp_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_F32, pool_bytes);
    auto bt_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_I32, bt_bytes);
    auto out_buf = alloc_buf(prov, mem_vt, NF_DTYPE_F32, out_bytes);

    std::memcpy(map_f(q_buf), Q_data, q_bytes); unmap(q_buf);
    std::memcpy(map_f(kp_buf), K_pool_data, pool_bytes); unmap(kp_buf);
    std::memcpy(map_f(vp_buf), V_pool_data, pool_bytes); unmap(vp_buf);
    std::memcpy(map_u(bt_buf), bt_data, bt_bytes); unmap(bt_buf);
    std::memset(map_f(out_buf), 0, out_bytes); unmap(out_buf);

    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "flash_attention_paged", NF_MAX_OP_NAME - 1);
    td.inputs[0] = q_buf.buf;   td.input_ops[0] = q_buf.ops;
    td.inputs[1] = kp_buf.buf;  td.input_ops[1] = kp_buf.ops;
    td.inputs[2] = vp_buf.buf;  td.input_ops[2] = vp_buf.ops;
    td.inputs[3] = bt_buf.buf;  td.input_ops[3] = bt_buf.ops;
    td.n_inputs = 4;
    td.outputs[0] = out_buf.buf; td.output_ops[0] = out_buf.ops;
    td.n_outputs = 1;
    td.affinity = NF_AFFINITY_GPU;

    /* Push constants: seq_len, n_heads, head_dim, M(n_kv), step_idx, max_seq,
       window_size(=block_size), _pad, K(=seq_offset), N(=num_blocks) */
    struct PC {
        uint32_t seq_len, n_heads, head_dim;
        float epsilon, theta;
        uint32_t M, N, K;
        uint32_t step_idx, max_seq_len;
        uint32_t window_size, _pad;
    } pc{};
    pc.seq_len = seq_len;
    pc.n_heads = n_heads;
    pc.head_dim = head_dim;
    pc.M = n_kv_heads;
    pc.N = num_blocks;
    pc.K = seq_offset;
    pc.step_idx = step_idx;
    pc.max_seq_len = 512;
    pc.window_size = block_size;
    td.push_constants_size = sizeof(pc);
    std::memcpy(td.push_constants, &pc, sizeof(pc));

    engine.add_task(gid, td);
    nf::PipelineEngine::Session sess(engine, gid);
    nf_status st = sess.step().get();
    CHECK(st == NF_OK);

    /* Ensure GPU is done */
    vt.synchronize(prov);

    std::memcpy(out_data, map_f(out_buf), out_bytes); unmap(out_buf);

    engine.destroy_graph(gid);
    release(q_buf); release(kp_buf); release(vp_buf); release(bt_buf); release(out_buf);
}

static void test_single_token_decode(
    nf_provider prov, nf_provider_vtable& vt, nf_provider_mem_vtable& mem_vt,
    nf::PipelineEngine& engine)
{
    /* 1 head, 4 dim, 8 cached tokens across 2 blocks (bs=4) */
    const uint32_t NH = 1, HD = 4, NKV = 1, BS = 4;
    const uint32_t num_tokens = 8, step = 7;
    const uint32_t num_blocks = 2;

    /* Q: single token query [1 × NH × HD] */
    float Q[4]; for (int i = 0; i < 4; ++i) Q[i] = (float)(i + 1) * 0.1f;

    /* K/V pool: [num_blocks × NKV × BS × HD] */
    const uint32_t pool_elems = num_blocks * NKV * BS * HD;
    std::vector<float> K_pool(pool_elems), V_pool(pool_elems);
    for (uint32_t i = 0; i < pool_elems; ++i) {
        K_pool[i] = std::sin((float)i * 0.3f);
        V_pool[i] = std::cos((float)i * 0.2f);
    }

    /* Block table: logical 0→phys 0, logical 1→phys 1 */
    uint32_t bt[2] = {0, 1};

    /* CPU reference */
    float ref_out[4] = {};
    ref_paged_attn(Q, K_pool.data(), V_pool.data(), bt, ref_out,
                   1, NH, HD, NKV, step, BS, num_blocks, 0);

    /* GPU */
    float gpu_out[4] = {};
    run_paged_attn_gpu(prov, vt, mem_vt, engine,
        1, NH, HD, NKV, step, BS, num_blocks, 0,
        Q, K_pool.data(), V_pool.data(), bt, gpu_out,
        sizeof(Q), pool_elems * sizeof(float),
        sizeof(bt), sizeof(gpu_out));

    /* Compare */
    for (int d = 0; d < 4; ++d) {
        float diff = std::fabs(ref_out[d] - gpu_out[d]);
        CHECK(diff < 1e-3f);
    }
    std::printf("  [PASS] PagedAttn_single_token_decode\n");
}

static void test_prefill_4_tokens(
    nf_provider prov, nf_provider_vtable& vt, nf_provider_mem_vtable& mem_vt,
    nf::PipelineEngine& engine)
{
    /* 1 head, 4 dim, prefill 4 tokens, 1 block (bs=4) */
    const uint32_t NH = 1, HD = 4, NKV = 1, BS = 4;
    const uint32_t seq_len = 4, step = 0;
    const uint32_t num_blocks = 1;

    /* Q: [4 × 1 × 4] */
    float Q[16];
    for (int i = 0; i < 16; ++i) Q[i] = (float)(i + 1) * 0.05f;

    /* K/V pool: [1 × 1 × 4 × 4] — pre-filled with KV for positions 0..3 */
    float K_pool[16], V_pool[16];
    for (int i = 0; i < 16; ++i) {
        K_pool[i] = std::sin((float)i * 0.4f);
        V_pool[i] = std::cos((float)i * 0.3f);
    }

    uint32_t bt[1] = {0};

    /* CPU reference */
    float ref_out[16] = {};
    ref_paged_attn(Q, K_pool, V_pool, bt, ref_out,
                   seq_len, NH, HD, NKV, step, BS, num_blocks, 0);

    /* GPU */
    float gpu_out[16] = {};
    run_paged_attn_gpu(prov, vt, mem_vt, engine,
        seq_len, NH, HD, NKV, step, BS, num_blocks, 0,
        Q, K_pool, V_pool, bt, gpu_out,
        sizeof(Q), sizeof(K_pool), sizeof(bt), sizeof(gpu_out));

    for (int i = 0; i < 16; ++i) {
        float diff = std::fabs(ref_out[i] - gpu_out[i]);
        CHECK(diff < 1e-3f);
    }
    std::printf("  [PASS] PagedAttn_prefill_4_tokens\n");
}

static void test_gqa_2h_1kv(
    nf_provider prov, nf_provider_vtable& vt, nf_provider_mem_vtable& mem_vt,
    nf::PipelineEngine& engine)
{
    /* 2 Q heads, 1 KV head (GQA), 4 dim, 4 cached tokens, 1 block (bs=4) */
    const uint32_t NH = 2, HD = 4, NKV = 1, BS = 4;
    const uint32_t step = 3, num_blocks = 1;

    /* Q: [1 × 2 × 4] = 8 floats */
    float Q[8];
    for (int i = 0; i < 8; ++i) Q[i] = (float)(i + 1) * 0.1f;

    /* K/V pool: [1 × 1 × 4 × 4] = 16 floats (only 1 KV head) */
    float K_pool[16], V_pool[16];
    for (int i = 0; i < 16; ++i) {
        K_pool[i] = std::sin((float)i * 0.5f);
        V_pool[i] = std::cos((float)i * 0.4f);
    }

    uint32_t bt[1] = {0};

    /* CPU reference */
    float ref_out[8] = {};
    ref_paged_attn(Q, K_pool, V_pool, bt, ref_out,
                   1, NH, HD, NKV, step, BS, num_blocks, 0);

    /* GPU */
    float gpu_out[8] = {};
    run_paged_attn_gpu(prov, vt, mem_vt, engine,
        1, NH, HD, NKV, step, BS, num_blocks, 0,
        Q, K_pool, V_pool, bt, gpu_out,
        sizeof(Q), sizeof(K_pool), sizeof(bt), sizeof(gpu_out));

    for (int i = 0; i < 8; ++i) {
        float diff = std::fabs(ref_out[i] - gpu_out[i]);
        CHECK(diff < 1e-3f);
    }
    std::printf("  [PASS] PagedAttn_gqa_2h_1kv\n");
}

int main() {
    std::printf("=== paged_attn_test ===\n");

    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    CHECK(nf_plugin_register(&vt, &prov) == NF_OK);
    CHECK(nf_plugin_register_mem(&mem_vt, &prov) == NF_OK);
    CHECK(vt.init(prov) == NF_OK);

    nf::PipelineEngine engine(2);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);

    test_single_token_decode(prov, vt, mem_vt, engine);
    test_prefill_4_tokens(prov, vt, mem_vt, engine);
    test_gqa_2h_1kv(prov, vt, mem_vt, engine);

    vt.shutdown(prov);
    std::printf("All 3 paged_attn tests passed.\n");
    return 0;
}
