/**
 * @file sliding_window_test.cpp
 * @brief Phase 30 — Sliding Window Attention Test
 *
 * Verifies causal_attention_cached with window_size > 0.
 */

#include "neurofabric/abi/neuro_fabric_abi.h"
#include "neurofabric/abi/neuro_buffer_abi.h"
#include "neurofabric/abi/neuro_scheduler_abi.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static nf_provider            g_prov = nullptr;
static nf_provider_vtable     g_vt{};
static nf_provider_mem_vtable g_mem{};

static nf_buffer alloc_buf(size_t bytes, nf_buffer_ops& ops) {
    nf_tensor_desc d{}; d.dtype = NF_DTYPE_F32; d.ndim = 1;
    d.shape[0] = bytes; d.size_bytes = bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    nf_buffer buf;
    CHECK_OK(g_mem.alloc(g_prov, &req, &ops, &buf));
    return buf;
}

static float* map_f(nf_buffer buf, nf_buffer_ops& ops) {
    void* p; ops.map(buf, &p); return (float*)p;
}

struct PushConstants {
    uint32_t seq_len; uint32_t n_heads; uint32_t head_dim;
    float epsilon; float theta;
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t step_idx; uint32_t max_seq_len;
    uint32_t window_size; uint32_t _pad1;
};
/* SLIDING_PART2 */

/* Dispatch attention op using the cross-dylib bridge pattern:
   push constants go into nf_task_desc, inputs pointer passed to dispatch */
static void dispatch_attn(const char* op,
                           nf_buffer in[5], nf_buffer out,
                           PushConstants& pc) {
    nf_task_desc td{};
    std::strncpy(td.op_name, op, NF_MAX_OP_NAME - 1);
    for (int i = 0; i < 5; ++i) td.inputs[i] = in[i];
    td.n_inputs = 5;
    td.outputs[0] = out;
    td.n_outputs = 1;
    td.push_constants_size = sizeof(pc);
    std::memcpy(td.push_constants, &pc, sizeof(pc));
    td.user_data = &td;
    CHECK_OK(g_vt.dispatch(g_prov, op, td.inputs, 5, &out, 1));
    g_vt.synchronize(g_prov);
}

/* Test 1: window_size=0 (full causal) — baseline correctness */
static void test_full_causal() {
    std::printf("  test_full_causal (window_size=0)...\n");
    const uint32_t NH = 1, HD = 4, SEQ = 4, MS = 8;
    size_t q_sz = NH * SEQ * HD * sizeof(float);
    size_t kv_sz = NH * MS * HD * sizeof(float);

    nf_buffer_ops qo, ko, vo, cko, cvo, oo;
    nf_buffer Q = alloc_buf(q_sz, qo), K = alloc_buf(q_sz, ko), V = alloc_buf(q_sz, vo);
    nf_buffer CK = alloc_buf(kv_sz, cko), CV = alloc_buf(kv_sz, cvo);
    nf_buffer OUT = alloc_buf(q_sz, oo);

    float* qp = map_f(Q, qo); float* kp = map_f(K, ko); float* vp = map_f(V, vo);
    for (uint32_t i = 0; i < NH * SEQ * HD; ++i) {
        qp[i] = 0.1f * (i % HD + 1);
        kp[i] = 0.1f * (i % HD + 1);
        vp[i] = (float)((i / HD) + 1);
    }
    qo.unmap(Q); ko.unmap(K); vo.unmap(V);

    float* ckp = map_f(CK, cko); float* cvp = map_f(CV, cvo);
    std::memset(ckp, 0, kv_sz); std::memset(cvp, 0, kv_sz);
    cko.unmap(CK); cvo.unmap(CV);

    PushConstants pc{};
    pc.seq_len = SEQ; pc.n_heads = NH; pc.head_dim = HD;
    pc.step_idx = 0; pc.max_seq_len = MS; pc.window_size = 0;

    nf_buffer ins[] = {Q, K, V, CK, CV};
    dispatch_attn("causal_attention_cached", ins, OUT, pc);

    float* outp = map_f(OUT, oo);
    for (uint32_t i = 0; i < NH * SEQ * HD; ++i)
        CHECK(std::isfinite(outp[i]));
    oo.unmap(OUT);

    qo.release(Q); ko.release(K); vo.release(V);
    cko.release(CK); cvo.release(CV); oo.release(OUT);
    std::printf("    full_causal verified ✓\n");
}
/* SLIDING_PART3 */

/* Test 2: window_size=2, seq_len=4 — output differs from full causal at pos 3 */
static void test_sliding_window() {
    std::printf("  test_sliding_window (window_size=2)...\n");
    const uint32_t NH = 1, HD = 4, SEQ = 4, MS = 8;
    size_t q_sz = NH * SEQ * HD * sizeof(float);
    size_t kv_sz = NH * MS * HD * sizeof(float);

    nf_buffer_ops qo, ko, vo, cko, cvo, oo_f, oo_w;
    nf_buffer Q = alloc_buf(q_sz, qo), K = alloc_buf(q_sz, ko), V = alloc_buf(q_sz, vo);
    nf_buffer CK = alloc_buf(kv_sz, cko), CV = alloc_buf(kv_sz, cvo);
    nf_buffer OF = alloc_buf(q_sz, oo_f), OW = alloc_buf(q_sz, oo_w);

    float* qp = map_f(Q, qo); float* kp = map_f(K, ko); float* vp = map_f(V, vo);
    for (uint32_t i = 0; i < NH * SEQ * HD; ++i) {
        qp[i] = 0.5f; kp[i] = 0.5f;
        vp[i] = (float)((i / HD) + 1);
    }
    qo.unmap(Q); ko.unmap(K); vo.unmap(V);

    /* Full causal */
    float* ckp = map_f(CK, cko); float* cvp = map_f(CV, cvo);
    std::memset(ckp, 0, kv_sz); std::memset(cvp, 0, kv_sz);
    cko.unmap(CK); cvo.unmap(CV);

    PushConstants pc{};
    pc.seq_len = SEQ; pc.n_heads = NH; pc.head_dim = HD;
    pc.step_idx = 0; pc.max_seq_len = MS; pc.window_size = 0;

    nf_buffer ins[] = {Q, K, V, CK, CV};
    dispatch_attn("causal_attention_cached", ins, OF, pc);

    /* Reset cache, sliding window=2 */
    ckp = map_f(CK, cko); cvp = map_f(CV, cvo);
    std::memset(ckp, 0, kv_sz); std::memset(cvp, 0, kv_sz);
    cko.unmap(CK); cvo.unmap(CV);

    pc.window_size = 2;
    dispatch_attn("causal_attention_cached", ins, OW, pc);

    float* fp = map_f(OF, oo_f); float* wp = map_f(OW, oo_w);

    /* Pos 0,1: window covers all → identical */
    for (uint32_t d = 0; d < HD; ++d) {
        CHECK(std::fabs(fp[0*HD+d] - wp[0*HD+d]) < 1e-5f);
        CHECK(std::fabs(fp[1*HD+d] - wp[1*HD+d]) < 1e-5f);
    }
    /* Pos 3: window=2 sees {2,3} vs full sees {0,1,2,3} → differs */
    bool differs = false;
    for (uint32_t d = 0; d < HD; ++d)
        if (std::fabs(fp[3*HD+d] - wp[3*HD+d]) > 1e-5f) differs = true;
    CHECK(differs);

    oo_f.unmap(OF); oo_w.unmap(OW);
    qo.release(Q); ko.release(K); vo.release(V);
    cko.release(CK); cvo.release(CV);
    oo_f.release(OF); oo_w.release(OW);
    std::printf("    sliding_window verified ✓\n");
}

/* Test 3: window_size=1 — each position only attends to itself */
static void test_window_size_one() {
    std::printf("  test_window_size_one...\n");
    const uint32_t NH = 1, HD = 4, SEQ = 3, MS = 8;
    size_t q_sz = NH * SEQ * HD * sizeof(float);
    size_t kv_sz = NH * MS * HD * sizeof(float);

    nf_buffer_ops qo, ko, vo, cko, cvo, oo;
    nf_buffer Q = alloc_buf(q_sz, qo), K = alloc_buf(q_sz, ko), V = alloc_buf(q_sz, vo);
    nf_buffer CK = alloc_buf(kv_sz, cko), CV = alloc_buf(kv_sz, cvo);
    nf_buffer OUT = alloc_buf(q_sz, oo);

    float* qp = map_f(Q, qo); float* kp = map_f(K, ko); float* vp = map_f(V, vo);
    for (uint32_t i = 0; i < NH * SEQ * HD; ++i) { qp[i] = 1.0f; kp[i] = 1.0f; }
    for (uint32_t s = 0; s < SEQ; ++s)
        for (uint32_t d = 0; d < HD; ++d)
            vp[s * HD + d] = (float)((s + 1) * 10);
    qo.unmap(Q); ko.unmap(K); vo.unmap(V);

    float* ckp = map_f(CK, cko); float* cvp = map_f(CV, cvo);
    std::memset(ckp, 0, kv_sz); std::memset(cvp, 0, kv_sz);
    cko.unmap(CK); cvo.unmap(CV);

    PushConstants pc{};
    pc.seq_len = SEQ; pc.n_heads = NH; pc.head_dim = HD;
    pc.step_idx = 0; pc.max_seq_len = MS; pc.window_size = 1;

    nf_buffer buf_ins[] = {Q, K, V, CK, CV};
    dispatch_attn("causal_attention_cached", buf_ins, OUT, pc);

    float* outp = map_f(OUT, oo);
    for (uint32_t s = 0; s < SEQ; ++s) {
        float expected = (float)((s + 1) * 10);
        for (uint32_t d = 0; d < HD; ++d)
            CHECK(std::fabs(outp[s * HD + d] - expected) < 1e-3f);
    }
    oo.unmap(OUT);

    qo.release(Q); ko.release(K); vo.release(V);
    cko.release(CK); cvo.release(CV); oo.release(OUT);
    std::printf("    window_size_one verified ✓\n");
}

int main() {
    std::printf("sliding_window_test: Phase 30 — Sliding Window Attention\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem));
    CHECK_OK(g_vt.init(g_prov));

    test_full_causal();
    test_sliding_window();
    test_window_size_one();

    g_vt.shutdown(g_prov);
    std::printf("OK: all Phase 30 sliding window tests passed\n");
    return 0;
}