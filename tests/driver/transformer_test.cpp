/**
 * @file transformer_test.cpp
 * @brief Phase 18 — Transformer Layer Primitives & Single-Layer Forward Pass
 *
 * Verifies on real Metal GPU via PipelineEngine:
 *   1. softmax: numerically stable, rows sum to 1
 *   2. silu: x * sigmoid(x) against CPU reference
 *   3. elementwise_mul: Hadamard product
 *   4. embedding_lookup: correct row extraction
 *   5. argmax: correct index of maximum per row
 *   6. Single-layer Transformer forward pass (embed→norm→attn→ffn→logits→argmax)
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

static constexpr float TOL = 1e-3f;

static nf_provider            g_prov = nullptr;
static nf_provider_vtable     g_vt{};
static nf_provider_mem_vtable g_mem_vt{};

struct PushConstants {
    uint32_t seq_len; uint32_t n_heads; uint32_t head_dim;
    float epsilon; float theta;
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t step_idx; uint32_t max_seq_len;
    uint32_t window_size; uint32_t _pad1;
};
/* --- Buffer + ops pair for tracking --- */
struct BufPair {
    nf_buffer buf = nullptr;
    nf_buffer_ops ops{};
};

static BufPair alloc_buf(nf_dtype dtype, size_t size_bytes) {
    BufPair bp;
    nf_tensor_desc d{}; d.dtype = dtype; d.ndim = 1;
    d.shape[0] = size_bytes; d.size_bytes = size_bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    CHECK_OK(g_mem_vt.alloc(g_prov, &req, &bp.ops, &bp.buf));
    return bp;
}

static float* map_f(BufPair& bp) {
    void* p; CHECK_OK(bp.ops.map(bp.buf, &p)); return (float*)p;
}
static int32_t* map_i(BufPair& bp) {
    void* p; CHECK_OK(bp.ops.map(bp.buf, &p)); return (int32_t*)p;
}
static void unmap(BufPair& bp) { bp.ops.unmap(bp.buf); }
static void release(BufPair& bp) { bp.ops.release(bp.buf); }
static void sync_read(BufPair& bp) {
    bp.ops.cache_sync(bp.buf, NF_CACHE_INVALIDATE, 0, 0);
}

/* --- Single-node GPU dispatch via PipelineEngine --- */
static void gpu_dispatch(const char* op,
                          BufPair* ins, uint32_t n_in,
                          BufPair* outs, uint32_t n_out,
                          const PushConstants* pc) {
    nf::PipelineEngine engine(2);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
    uint32_t gid = engine.create_graph();

    nf_task_desc td{};
    std::strncpy(td.op_name, op, NF_MAX_OP_NAME - 1);
    for (uint32_t i = 0; i < n_in; ++i) {
        td.inputs[i]    = ins[i].buf;
        td.input_ops[i] = ins[i].ops;
    }
    td.n_inputs = n_in;
    for (uint32_t i = 0; i < n_out; ++i) {
        td.outputs[i]    = outs[i].buf;
        td.output_ops[i] = outs[i].ops;
    }
    td.n_outputs = n_out;
    td.affinity  = NF_AFFINITY_GPU;
    if (pc) {
        std::memcpy(td.push_constants, pc, sizeof(PushConstants));
        td.push_constants_size = sizeof(PushConstants);
    }

    engine.add_task(gid, td);
    CHECK_OK(engine.submit(gid).get());

    for (uint32_t i = 0; i < n_out; ++i)
        sync_read(outs[i]);
    engine.destroy_graph(gid);
}

/* ================================================================== */
/*  Test 1: softmax correctness                                        */
/* ================================================================== */
static void test_softmax() {
    std::printf("  Test 1: softmax ... ");
    const uint32_t rows = 2, cols = 8;
    BufPair in  = alloc_buf(NF_DTYPE_F32, rows * cols * 4);
    BufPair out = alloc_buf(NF_DTYPE_F32, rows * cols * 4);

    float* p = map_f(in);
    for (uint32_t i = 0; i < rows * cols; ++i) p[i] = (float)i - 7.0f;
    unmap(in);

    PushConstants pc{}; pc.seq_len = rows; pc.head_dim = cols;
    gpu_dispatch("softmax", &in, 1, &out, 1, &pc);

    float* o = map_f(out);
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            CHECK(o[r * cols + c] >= 0.0f);
            sum += o[r * cols + c];
        }
        CHECK(std::fabs(sum - 1.0f) < TOL);
    }
    unmap(out);
    release(in); release(out);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 2: silu correctness                                           */
/* ================================================================== */
static void test_silu() {
    std::printf("  Test 2: silu ... ");
    const uint32_t N = 64;
    BufPair in  = alloc_buf(NF_DTYPE_F32, N * 4);
    BufPair out = alloc_buf(NF_DTYPE_F32, N * 4);

    float* p = map_f(in);
    for (uint32_t i = 0; i < N; ++i) p[i] = (float)i / 10.0f - 3.0f;
    unmap(in);

    gpu_dispatch("silu", &in, 1, &out, 1, nullptr);

    float* ip = map_f(in);
    float* op = map_f(out);
    for (uint32_t i = 0; i < N; ++i) {
        float x = ip[i];
        float expected = x / (1.0f + std::exp(-x));
        CHECK(std::fabs(op[i] - expected) < TOL);
    }
    unmap(in); unmap(out);
    release(in); release(out);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 3: elementwise_mul                                            */
/* ================================================================== */
static void test_elementwise_mul() {
    std::printf("  Test 3: elementwise_mul ... ");
    const uint32_t N = 32;
    BufPair a   = alloc_buf(NF_DTYPE_F32, N * 4);
    BufPair b   = alloc_buf(NF_DTYPE_F32, N * 4);
    BufPair out = alloc_buf(NF_DTYPE_F32, N * 4);

    float* ap = map_f(a); float* bp = map_f(b);
    for (uint32_t i = 0; i < N; ++i) { ap[i] = (float)i + 1.0f; bp[i] = 0.5f; }
    unmap(a); unmap(b);

    BufPair ins[] = {a, b};
    gpu_dispatch("elementwise_mul", ins, 2, &out, 1, nullptr);

    float* op = map_f(out);
    for (uint32_t i = 0; i < N; ++i)
        CHECK(std::fabs(op[i] - ((float)i + 1.0f) * 0.5f) < TOL);
    unmap(out);
    release(a); release(b); release(out);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 4: embedding_lookup                                           */
/* ================================================================== */
static void test_embedding_lookup() {
    std::printf("  Test 4: embedding_lookup ... ");
    const uint32_t vocab = 16, dim = 8, seq = 3;
    BufPair w   = alloc_buf(NF_DTYPE_F32, vocab * dim * 4);
    BufPair tok = alloc_buf(NF_DTYPE_I32, seq * 4);
    BufPair out = alloc_buf(NF_DTYPE_F32, seq * dim * 4);

    float* wp = map_f(w);
    for (uint32_t i = 0; i < vocab * dim; ++i) wp[i] = (float)i * 0.01f;
    unmap(w);

    int32_t* tp = map_i(tok);
    tp[0] = 0; tp[1] = 5; tp[2] = 15;
    unmap(tok);

    PushConstants pc{}; pc.seq_len = seq; pc.head_dim = dim;
    BufPair ins[] = {w, tok};
    gpu_dispatch("embedding_lookup", ins, 2, &out, 1, &pc);

    float* op = map_f(out);
    wp = map_f(w);
    int32_t tokens[] = {0, 5, 15};
    for (uint32_t s = 0; s < seq; ++s)
        for (uint32_t d = 0; d < dim; ++d)
            CHECK(std::fabs(op[s * dim + d] - wp[tokens[s] * dim + d]) < TOL);
    unmap(w); unmap(out);
    release(w); release(tok); release(out);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 5: argmax                                                     */
/* ================================================================== */
static void test_argmax() {
    std::printf("  Test 5: argmax ... ");
    const uint32_t rows = 4, cols = 8;
    BufPair in  = alloc_buf(NF_DTYPE_F32, rows * cols * 4);
    BufPair out = alloc_buf(NF_DTYPE_I32, rows * 4);

    float* p = map_f(in);
    /* Place max at different columns per row */
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < cols; ++c)
            p[r * cols + c] = (c == (r + 2) % cols) ? 100.0f : 0.0f;
    unmap(in);

    PushConstants pc{}; pc.seq_len = rows; pc.N = cols;
    gpu_dispatch("argmax", &in, 1, &out, 1, &pc);

    int32_t* op = map_i(out);
    for (uint32_t r = 0; r < rows; ++r)
        CHECK(op[r] == (int32_t)((r + 2) % cols));
    unmap(out);
    release(in); release(out);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 6b: causal_attention_cached (prefill + decode)                 */
/* ================================================================== */
static void test_causal_attention_cached() {
    std::printf("  Test 6b: causal_attention_cached ... ");
    const uint32_t n_heads = 2, head_dim = 4, max_seq = 16;
    const uint32_t dim = n_heads * head_dim;  // 8

    /* --- Prefill: seq_len=4, step_idx=0 --- */
    const uint32_t pf_seq = 4;
    BufPair q_pf   = alloc_buf(NF_DTYPE_F32, pf_seq * dim * 4);
    BufPair k_pf   = alloc_buf(NF_DTYPE_F32, pf_seq * dim * 4);
    BufPair v_pf   = alloc_buf(NF_DTYPE_F32, pf_seq * dim * 4);
    BufPair ck     = alloc_buf(NF_DTYPE_F32, max_seq * dim * 4);
    BufPair cv     = alloc_buf(NF_DTYPE_F32, max_seq * dim * 4);
    BufPair out_pf = alloc_buf(NF_DTYPE_F32, pf_seq * dim * 4);

    /* Fill Q/K/V with small deterministic values */
    auto fill = [](BufPair& bp, size_t n) {
        float* p = map_f(bp);
        for (size_t i = 0; i < n; ++i) p[i] = ((float)(i % 7) - 3.0f) * 0.1f;
        unmap(bp);
    };
    fill(q_pf, pf_seq * dim);
    fill(k_pf, pf_seq * dim);
    fill(v_pf, pf_seq * dim);
    /* Zero-init caches */
    { float* p = map_f(ck); std::memset(p, 0, max_seq * dim * 4); unmap(ck); }
    { float* p = map_f(cv); std::memset(p, 0, max_seq * dim * 4); unmap(cv); }

    PushConstants pc{};
    pc.seq_len = pf_seq; pc.n_heads = n_heads; pc.head_dim = head_dim;
    pc.step_idx = 0; pc.max_seq_len = max_seq;
    pc.epsilon = 1e-5f; pc.theta = 10000.0f;

    BufPair pf_ins[] = {q_pf, k_pf, v_pf, ck, cv};
    gpu_dispatch("causal_attention_cached", pf_ins, 5, &out_pf, 1, &pc);

    /* Verify prefill output: non-zero, finite */
    float* op = map_f(out_pf);
    bool any_nonzero = false;
    for (uint32_t i = 0; i < pf_seq * dim; ++i) {
        CHECK(std::isfinite(op[i]));
        if (op[i] != 0.0f) any_nonzero = true;
    }
    CHECK(any_nonzero);
    unmap(out_pf);

    /* --- Decode: seq_len=1, step_idx=4 (appends after prefill) --- */
    const uint32_t dc_seq = 1;
    BufPair q_dc   = alloc_buf(NF_DTYPE_F32, dc_seq * dim * 4);
    BufPair k_dc   = alloc_buf(NF_DTYPE_F32, dc_seq * dim * 4);
    BufPair v_dc   = alloc_buf(NF_DTYPE_F32, dc_seq * dim * 4);
    BufPair out_dc = alloc_buf(NF_DTYPE_F32, dc_seq * dim * 4);

    fill(q_dc, dc_seq * dim);
    fill(k_dc, dc_seq * dim);
    fill(v_dc, dc_seq * dim);

    pc.seq_len = dc_seq; pc.step_idx = pf_seq;  // decode at position 4
    BufPair dc_ins[] = {q_dc, k_dc, v_dc, ck, cv};
    gpu_dispatch("causal_attention_cached", dc_ins, 5, &out_dc, 1, &pc);

    /* Verify decode output: non-zero, finite */
    float* dp = map_f(out_dc);
    any_nonzero = false;
    for (uint32_t i = 0; i < dc_seq * dim; ++i) {
        CHECK(std::isfinite(dp[i]));
        if (dp[i] != 0.0f) any_nonzero = true;
    }
    CHECK(any_nonzero);
    unmap(out_dc);

    release(q_pf); release(k_pf); release(v_pf);
    release(ck); release(cv);
    release(out_pf); release(out_dc);
    release(q_dc); release(k_dc); release(v_dc);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 6: single-layer transformer forward pass                      */
/* ================================================================== */
static void test_single_layer_forward() {
    std::printf("  Test 6: single-layer forward ... ");
    /* Minimal dims: seq=2, dim=8, heads=2, head_dim=4, ff_dim=16, vocab=16 */
    const uint32_t seq = 2, dim = 8, heads = 2, hdim = 4;
    const uint32_t ff_dim = 16, vocab = 16;

    /* Allocate weight buffers */
    BufPair w_embed = alloc_buf(NF_DTYPE_F32, vocab * dim * 4);
    BufPair w_norm  = alloc_buf(NF_DTYPE_F32, dim * 4);
    BufPair w_q     = alloc_buf(NF_DTYPE_F32, dim * dim * 4);
    BufPair w_k     = alloc_buf(NF_DTYPE_F32, dim * dim * 4);
    BufPair w_v     = alloc_buf(NF_DTYPE_F32, dim * dim * 4);
    BufPair w_o     = alloc_buf(NF_DTYPE_F32, dim * dim * 4);
    BufPair w_gate  = alloc_buf(NF_DTYPE_F32, dim * ff_dim * 4);
    BufPair w_up    = alloc_buf(NF_DTYPE_F32, dim * ff_dim * 4);
    BufPair w_down  = alloc_buf(NF_DTYPE_F32, ff_dim * dim * 4);
    BufPair w_lm    = alloc_buf(NF_DTYPE_F32, dim * vocab * 4);

    /* Fill weights with small deterministic values */
    auto fill_weight = [](BufPair& bp, size_t n, float scale) {
        float* p = map_f(bp);
        for (size_t i = 0; i < n; ++i)
            p[i] = ((float)(i % 17) - 8.0f) * scale;
        unmap(bp);
    };
    fill_weight(w_embed, vocab * dim, 0.1f);
    { float* p = map_f(w_norm); for (uint32_t i = 0; i < dim; ++i) p[i] = 1.0f; unmap(w_norm); }
    fill_weight(w_q, dim * dim, 0.05f);
    fill_weight(w_k, dim * dim, 0.05f);
    fill_weight(w_v, dim * dim, 0.05f);
    fill_weight(w_o, dim * dim, 0.05f);
    fill_weight(w_gate, dim * ff_dim, 0.02f);
    fill_weight(w_up, dim * ff_dim, 0.02f);
    fill_weight(w_down, ff_dim * dim, 0.02f);
    fill_weight(w_lm, dim * vocab, 0.05f);

    /* Token input */
    BufPair tokens = alloc_buf(NF_DTYPE_I32, seq * 4);
    { int32_t* t = map_i(tokens); t[0] = 3; t[1] = 7; unmap(tokens); }

    /* Intermediate buffers */
    BufPair hidden   = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair normed   = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair q_buf    = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair k_buf    = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair v_buf    = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair attn_out = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair proj_out = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair resid1   = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair gate_out = alloc_buf(NF_DTYPE_F32, seq * ff_dim * 4);
    BufPair up_out   = alloc_buf(NF_DTYPE_F32, seq * ff_dim * 4);
    BufPair silu_out = alloc_buf(NF_DTYPE_F32, seq * ff_dim * 4);
    BufPair mul_out  = alloc_buf(NF_DTYPE_F32, seq * ff_dim * 4);
    BufPair down_out = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair resid2   = alloc_buf(NF_DTYPE_F32, seq * dim * 4);
    BufPair logits   = alloc_buf(NF_DTYPE_F32, seq * vocab * 4);
    BufPair argmax_o = alloc_buf(NF_DTYPE_I32, seq * 4);

    PushConstants pc{};
    pc.seq_len = seq; pc.n_heads = heads; pc.head_dim = hdim;
    pc.epsilon = 1e-5f; pc.theta = 10000.0f;

    /* 1. Embedding lookup */
    { BufPair ins[] = {w_embed, tokens};
      gpu_dispatch("embedding_lookup", ins, 2, &hidden, 1, &pc); }

    /* 2. RMS norm */
    { BufPair ins[] = {hidden, w_norm};
      gpu_dispatch("rms_norm", ins, 2, &normed, 1, &pc); }

    /* 3. Q/K/V projections (linear: input × weight) */
    pc.M = seq; pc.N = dim; pc.K = dim;
    { BufPair ins[] = {normed, w_q}; gpu_dispatch("linear", ins, 2, &q_buf, 1, &pc); }
    { BufPair ins[] = {normed, w_k}; gpu_dispatch("linear", ins, 2, &k_buf, 1, &pc); }
    { BufPair ins[] = {normed, w_v}; gpu_dispatch("linear", ins, 2, &v_buf, 1, &pc); }

    /* 4. Causal attention */
    { BufPair ins[] = {q_buf, k_buf, v_buf};
      gpu_dispatch("causal_attention", ins, 3, &attn_out, 1, &pc); }

    /* 5. Output projection */
    { BufPair ins[] = {attn_out, w_o};
      gpu_dispatch("linear", ins, 2, &proj_out, 1, &pc); }

    /* 6. Residual add: resid1 = hidden + proj_out */
    { BufPair ins[] = {hidden, proj_out};
      gpu_dispatch("metal_vector_add", ins, 2, &resid1, 1, nullptr); }

    /* 7. FFN: gate + up projections */
    pc.M = seq; pc.N = ff_dim; pc.K = dim;
    { BufPair ins[] = {resid1, w_gate}; gpu_dispatch("linear", ins, 2, &gate_out, 1, &pc); }
    { BufPair ins[] = {resid1, w_up};   gpu_dispatch("linear", ins, 2, &up_out, 1, &pc); }

    /* 8. SiLU(gate) * up */
    gpu_dispatch("silu", &gate_out, 1, &silu_out, 1, nullptr);
    { BufPair ins[] = {silu_out, up_out};
      gpu_dispatch("elementwise_mul", ins, 2, &mul_out, 1, nullptr); }

    /* 9. Down projection */
    pc.M = seq; pc.N = dim; pc.K = ff_dim;
    { BufPair ins[] = {mul_out, w_down}; gpu_dispatch("linear", ins, 2, &down_out, 1, &pc); }

    /* 10. Residual add: resid2 = resid1 + down_out */
    { BufPair ins[] = {resid1, down_out};
      gpu_dispatch("metal_vector_add", ins, 2, &resid2, 1, nullptr); }

    /* 11. LM head: logits = resid2 × w_lm */
    pc.M = seq; pc.N = vocab; pc.K = dim;
    { BufPair ins[] = {resid2, w_lm}; gpu_dispatch("linear", ins, 2, &logits, 1, &pc); }

    /* 12. Argmax over logits */
    PushConstants apc{}; apc.seq_len = seq; apc.head_dim = vocab;
    gpu_dispatch("argmax", &logits, 1, &argmax_o, 1, &apc);

    /* Verify: argmax produces valid token indices */
    int32_t* result = map_i(argmax_o);
    for (uint32_t s = 0; s < seq; ++s) {
        CHECK(result[s] >= 0 && result[s] < (int32_t)vocab);
    }
    unmap(argmax_o);

    /* Cleanup */
    release(tokens); release(hidden); release(normed);
    release(q_buf); release(k_buf); release(v_buf);
    release(attn_out); release(proj_out); release(resid1);
    release(gate_out); release(up_out); release(silu_out);
    release(mul_out); release(down_out); release(resid2);
    release(logits); release(argmax_o);
    release(w_embed); release(w_norm);
    release(w_q); release(w_k); release(w_v); release(w_o);
    release(w_gate); release(w_up); release(w_down); release(w_lm);

    std::printf("PASS\n");
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main() {
    std::printf("=== transformer_test (Phase 18) ===\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    test_softmax();
    test_silu();
    test_elementwise_mul();
    test_embedding_lookup();
    test_argmax();
    test_causal_attention_cached();
    test_single_layer_forward();

    if (g_vt.shutdown) g_vt.shutdown(g_prov);
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
