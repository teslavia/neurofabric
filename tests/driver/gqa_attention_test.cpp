/**
 * @file gqa_attention_test.cpp
 * @brief Phase 34-A: GQA Flash Attention kernel test
 *
 * Tests native grouped-query attention where n_kv_heads < n_heads.
 * Uses PipelineEngine for proper push constant injection.
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* prov);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* vt, nf_provider* prov);

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } \
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
static void unmap(BufPair& bp) { bp.ops.unmap(bp.buf); }
static void release(BufPair& bp) { if (bp.buf) bp.ops.release(bp.buf); bp.buf = nullptr; }

struct PC {
    uint32_t seq_len, n_heads, head_dim;
    float epsilon, theta;
    uint32_t M, N, K;
    uint32_t step_idx, max_seq_len;
    uint32_t window_size, _pad;
};

/* CPU reference: GQA decode attention (single query against KV cache) */
static void cpu_gqa_decode(
    const float* Q,       /* [n_heads * head_dim] */
    const float* K_cache, /* [n_kv_heads * max_seq * head_dim] */
    const float* V_cache, /* [n_kv_heads * max_seq * head_dim] */
    float* out,           /* [n_heads * head_dim] */
    uint32_t n_heads, uint32_t n_kv, uint32_t hd,
    uint32_t max_seq, uint32_t step)
{
    float scale = 1.0f / std::sqrt((float)hd);
    for (uint32_t h = 0; h < n_heads; ++h) {
        uint32_t kv_h = h * n_kv / n_heads;
        float max_s = -1e30f;
        std::vector<float> scores(step + 1);
        for (uint32_t t = 0; t <= step; ++t) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < hd; ++d)
                dot += Q[h * hd + d] * K_cache[kv_h * max_seq * hd + t * hd + d];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }
        float sum_exp = 0.0f;
        for (uint32_t t = 0; t <= step; ++t) {
            scores[t] = std::exp(scores[t] - max_s);
            sum_exp += scores[t];
        }
        for (uint32_t d = 0; d < hd; ++d) {
            float val = 0.0f;
            for (uint32_t t = 0; t <= step; ++t)
                val += (scores[t] / sum_exp) * V_cache[kv_h * max_seq * hd + t * hd + d];
            out[h * hd + d] = val;
        }
    }
}

int main() {
    std::printf("=== Phase 34-A: GQA Flash Attention Test ===\n");

    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;
    CHECK(nf_plugin_register(&vt, &prov) == NF_OK, "register");
    CHECK(nf_plugin_register_mem(&mem_vt, &prov) == NF_OK, "register_mem");
    CHECK(vt.init(prov) == NF_OK, "init");

    nf::PipelineEngine engine(2);
    engine.register_provider(prov, vt, NF_AFFINITY_GPU);

    const uint32_t NH = 8, NKV = 2, HD = 64, MAX_SEQ = 512;
    const uint32_t KV_LEN = 15;  /* step_idx = KV_LEN means KV_LEN+1 tokens in cache after append */

    /* Allocate GPU buffers */
    auto q_buf   = alloc_buf(prov, mem_vt, NF_DTYPE_F32, NH * HD * sizeof(float));
    auto kn_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_F32, NKV * HD * sizeof(float));
    auto vn_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_F32, NKV * HD * sizeof(float));
    auto kc_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_F32, NKV * MAX_SEQ * HD * sizeof(float));
    auto vc_buf  = alloc_buf(prov, mem_vt, NF_DTYPE_F32, NKV * MAX_SEQ * HD * sizeof(float));
    auto out_buf = alloc_buf(prov, mem_vt, NF_DTYPE_F32, NH * HD * sizeof(float));

    /* CPU-side data */
    std::vector<float> q_data(NH * HD);
    std::vector<float> kc_data(NKV * MAX_SEQ * HD, 0.0f);
    std::vector<float> vc_data(NKV * MAX_SEQ * HD, 0.0f);
    std::vector<float> kn_data(NKV * HD);
    std::vector<float> vn_data(NKV * HD);

    /* Fill Q — small values */
    for (uint32_t i = 0; i < NH * HD; ++i)
        q_data[i] = 0.01f * ((float)(i % 7) - 3.0f);

    /* Fill KV cache with existing tokens [0..KV_LEN-1] */
    for (uint32_t kv = 0; kv < NKV; ++kv)
        for (uint32_t t = 0; t < KV_LEN; ++t)
            for (uint32_t d = 0; d < HD; ++d) {
                kc_data[kv * MAX_SEQ * HD + t * HD + d] = 0.01f * ((float)((d + t + kv) % 11) - 5.0f);
                vc_data[kv * MAX_SEQ * HD + t * HD + d] = 0.01f * ((float)((d * 3 + t + kv) % 13) - 6.0f);
            }

    /* Fill K_new/V_new (will be appended at position KV_LEN) */
    for (uint32_t i = 0; i < NKV * HD; ++i) {
        kn_data[i] = 0.01f * ((float)(i % 9) - 4.0f);
        vn_data[i] = 0.01f * ((float)(i % 5) - 2.0f);
    }

    /* The kernel appends K_new/V_new at step_idx=KV_LEN, so update CPU cache too */
    for (uint32_t kv = 0; kv < NKV; ++kv)
        for (uint32_t d = 0; d < HD; ++d) {
            kc_data[kv * MAX_SEQ * HD + KV_LEN * HD + d] = kn_data[kv * HD + d];
            vc_data[kv * MAX_SEQ * HD + KV_LEN * HD + d] = vn_data[kv * HD + d];
        }

    /* CPU reference (after append, attend over [0..KV_LEN]) */
    std::vector<float> cpu_out(NH * HD);
    cpu_gqa_decode(q_data.data(), kc_data.data(), vc_data.data(),
                   cpu_out.data(), NH, NKV, HD, MAX_SEQ, KV_LEN);

    /* Upload to GPU */
    std::memcpy(map_f(q_buf), q_data.data(), q_data.size() * sizeof(float)); unmap(q_buf);
    std::memcpy(map_f(kn_buf), kn_data.data(), kn_data.size() * sizeof(float)); unmap(kn_buf);
    std::memcpy(map_f(vn_buf), vn_data.data(), vn_data.size() * sizeof(float)); unmap(vn_buf);
    /* Upload cache WITHOUT the appended token — kernel will append */
    {
        float* p = map_f(kc_buf);
        std::memset(p, 0, NKV * MAX_SEQ * HD * sizeof(float));
        for (uint32_t kv = 0; kv < NKV; ++kv)
            for (uint32_t t = 0; t < KV_LEN; ++t)
                for (uint32_t d = 0; d < HD; ++d)
                    p[kv * MAX_SEQ * HD + t * HD + d] = 0.01f * ((float)((d + t + kv) % 11) - 5.0f);
        unmap(kc_buf);
    }
    {
        float* p = map_f(vc_buf);
        std::memset(p, 0, NKV * MAX_SEQ * HD * sizeof(float));
        for (uint32_t kv = 0; kv < NKV; ++kv)
            for (uint32_t t = 0; t < KV_LEN; ++t)
                for (uint32_t d = 0; d < HD; ++d)
                    p[kv * MAX_SEQ * HD + t * HD + d] = 0.01f * ((float)((d * 3 + t + kv) % 13) - 6.0f);
        unmap(vc_buf);
    }
    std::memset(map_f(out_buf), 0, NH * HD * sizeof(float)); unmap(out_buf);

    /* Dispatch via PipelineEngine */
    uint32_t gid = engine.create_graph();
    nf_task_desc td{};
    std::strncpy(td.op_name, "flash_attention_gqa", NF_MAX_OP_NAME - 1);
    td.inputs[0] = q_buf.buf;   td.input_ops[0] = q_buf.ops;
    td.inputs[1] = kn_buf.buf;  td.input_ops[1] = kn_buf.ops;
    td.inputs[2] = vn_buf.buf;  td.input_ops[2] = vn_buf.ops;
    td.inputs[3] = kc_buf.buf;  td.input_ops[3] = kc_buf.ops;
    td.inputs[4] = vc_buf.buf;  td.input_ops[4] = vc_buf.ops;
    td.n_inputs = 5;
    td.outputs[0] = out_buf.buf; td.output_ops[0] = out_buf.ops;
    td.n_outputs = 1;
    td.affinity = NF_AFFINITY_GPU;

    PC pc{};
    pc.seq_len = 1;
    pc.n_heads = NH;
    pc.head_dim = HD;
    pc.M = NKV;
    pc.step_idx = KV_LEN;
    pc.max_seq_len = MAX_SEQ;
    td.push_constants_size = sizeof(pc);
    std::memcpy(td.push_constants, &pc, sizeof(pc));

    engine.add_task(gid, td);
    nf::PipelineEngine::Session sess(engine, gid);
    nf_status st = sess.step().get();
    CHECK(st == NF_OK, "dispatch GQA attention");
    vt.synchronize(prov);

    /* Compare GPU vs CPU */
    float* gpu = map_f(out_buf);
    float max_err = 0.0f;
    for (uint32_t i = 0; i < NH * HD; ++i) {
        float err = std::fabs(gpu[i] - cpu_out[i]);
        if (err > max_err) max_err = err;
    }
    unmap(out_buf);
    std::printf("  GQA max error: %.6f\n", max_err);
    CHECK(max_err < 0.01f, "GQA attention accuracy");
    std::printf("  [PASS] GQA decode attention (8 Q heads, 2 KV heads)\n");

    engine.destroy_graph(gid);
    release(q_buf); release(kn_buf); release(vn_buf);
    release(kc_buf); release(vc_buf); release(out_buf);

    vt.shutdown(prov);
    std::printf("=== All GQA attention tests passed ===\n");
    return 0;
}