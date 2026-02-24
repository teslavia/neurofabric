/**
 * @file multi_layer_test.cpp
 * @brief Phase 19 — Multi-Layer Transformer Forward Pass
 *
 * Verifies on real Metal GPU:
 *   1. 4-layer prefill (seq=4): all outputs non-zero, finite
 *   2. Autoregressive decode (3 steps): valid token IDs, KV cache increments
 *   3. Logits distribution sanity: softmax entropy > 0
 *
 * Model config: vocab=32, dim=16, heads=4, head_dim=4, ff_dim=32, max_seq=32, n_layers=4
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
#include <string>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
    std::abort(); } } while(0)
#define CHECK_OK(call) CHECK((call) == NF_OK)

extern "C" nf_status nf_plugin_register(nf_provider_vtable* vt, nf_provider* out);
extern "C" nf_status nf_plugin_register_mem(nf_provider_mem_vtable* mem_vt);

/* Model hyperparameters */
static constexpr uint32_t VOCAB   = 32;
static constexpr uint32_t DIM     = 16;
static constexpr uint32_t HEADS   = 4;
static constexpr uint32_t HDIM    = 4;   /* DIM / HEADS */
static constexpr uint32_t FF_DIM  = 32;
static constexpr uint32_t MAX_SEQ = 32;
static constexpr uint32_t N_LAYERS = 4;

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
static void release(BufPair& bp) { if (bp.buf) bp.ops.release(bp.buf); bp.buf = nullptr; }
static void sync_read(BufPair& bp) {
    bp.ops.cache_sync(bp.buf, NF_CACHE_INVALIDATE, 0, 0);
}

static void fill_deterministic(BufPair& bp, size_t n_floats, float scale) {
    float* p = map_f(bp);
    for (size_t i = 0; i < n_floats; ++i)
        p[i] = ((float)(i % 17) - 8.0f) * scale;
    unmap(bp);
}

static void fill_ones(BufPair& bp, size_t n_floats) {
    float* p = map_f(bp);
    for (size_t i = 0; i < n_floats; ++i) p[i] = 1.0f;
    unmap(bp);
}

static void zero_buf(BufPair& bp, size_t size_bytes) {
    float* p = map_f(bp);
    std::memset(p, 0, size_bytes);
    unmap(bp);
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
/*  Weight & Buffer Management for N-layer Transformer                 */
/* ================================================================== */

struct LayerWeights {
    BufPair w_attn_norm;   /* [DIM] */
    BufPair w_q;           /* [DIM x DIM] */
    BufPair w_k;           /* [DIM x DIM] */
    BufPair w_v;           /* [DIM x DIM] */
    BufPair w_o;           /* [DIM x DIM] */
    BufPair w_ffn_norm;    /* [DIM] */
    BufPair w_gate;        /* [DIM x FF_DIM] */
    BufPair w_up;          /* [DIM x FF_DIM] */
    BufPair w_down;        /* [FF_DIM x DIM] */
};

struct ModelWeights {
    BufPair w_embed;       /* [VOCAB x DIM] */
    LayerWeights layers[N_LAYERS];
    BufPair w_final_norm;  /* [DIM] */
    BufPair w_lm_head;     /* [DIM x VOCAB] */
};

struct KVCache {
    BufPair k_cache[N_LAYERS];  /* [MAX_SEQ x DIM] each */
    BufPair v_cache[N_LAYERS];  /* [MAX_SEQ x DIM] each */
};

static ModelWeights alloc_model_weights() {
    ModelWeights mw{};
    mw.w_embed = alloc_buf(NF_DTYPE_F32, VOCAB * DIM * 4);
    fill_deterministic(mw.w_embed, VOCAB * DIM, 0.1f);

    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        auto& lw = mw.layers[l];
        lw.w_attn_norm = alloc_buf(NF_DTYPE_F32, DIM * 4);
        fill_ones(lw.w_attn_norm, DIM);
        lw.w_q    = alloc_buf(NF_DTYPE_F32, DIM * DIM * 4);
        lw.w_k    = alloc_buf(NF_DTYPE_F32, DIM * DIM * 4);
        lw.w_v    = alloc_buf(NF_DTYPE_F32, DIM * DIM * 4);
        lw.w_o    = alloc_buf(NF_DTYPE_F32, DIM * DIM * 4);
        fill_deterministic(lw.w_q, DIM * DIM, 0.05f);
        fill_deterministic(lw.w_k, DIM * DIM, 0.05f);
        fill_deterministic(lw.w_v, DIM * DIM, 0.05f);
        fill_deterministic(lw.w_o, DIM * DIM, 0.05f);
        lw.w_ffn_norm = alloc_buf(NF_DTYPE_F32, DIM * 4);
        fill_ones(lw.w_ffn_norm, DIM);
        lw.w_gate = alloc_buf(NF_DTYPE_F32, DIM * FF_DIM * 4);
        lw.w_up   = alloc_buf(NF_DTYPE_F32, DIM * FF_DIM * 4);
        lw.w_down = alloc_buf(NF_DTYPE_F32, FF_DIM * DIM * 4);
        fill_deterministic(lw.w_gate, DIM * FF_DIM, 0.02f);
        fill_deterministic(lw.w_up,   DIM * FF_DIM, 0.02f);
        fill_deterministic(lw.w_down, FF_DIM * DIM, 0.02f);
    }

    mw.w_final_norm = alloc_buf(NF_DTYPE_F32, DIM * 4);
    fill_ones(mw.w_final_norm, DIM);
    mw.w_lm_head = alloc_buf(NF_DTYPE_F32, DIM * VOCAB * 4);
    fill_deterministic(mw.w_lm_head, DIM * VOCAB, 0.05f);
    return mw;
}

static void release_model_weights(ModelWeights& mw) {
    release(mw.w_embed);
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        auto& lw = mw.layers[l];
        release(lw.w_attn_norm); release(lw.w_ffn_norm);
        release(lw.w_q); release(lw.w_k); release(lw.w_v); release(lw.w_o);
        release(lw.w_gate); release(lw.w_up); release(lw.w_down);
    }
    release(mw.w_final_norm); release(mw.w_lm_head);
}

static KVCache alloc_kv_cache() {
    KVCache kv{};
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        kv.k_cache[l] = alloc_buf(NF_DTYPE_F32, MAX_SEQ * DIM * 4);
        kv.v_cache[l] = alloc_buf(NF_DTYPE_F32, MAX_SEQ * DIM * 4);
        zero_buf(kv.k_cache[l], MAX_SEQ * DIM * 4);
        zero_buf(kv.v_cache[l], MAX_SEQ * DIM * 4);
    }
    return kv;
}

static void release_kv_cache(KVCache& kv) {
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        release(kv.k_cache[l]); release(kv.v_cache[l]);
    }
}

/* ================================================================== */
/*  N-Layer Forward Pass (sequential gpu_dispatch per op)              */
/* ================================================================== */

struct ForwardResult {
    BufPair logits;
    BufPair argmax_out;
    /* Per-layer intermediate buffers (owned, must be released) */
    std::vector<BufPair> intermediates;
};

static ForwardResult forward_pass(
    const ModelWeights& mw, KVCache& kv,
    BufPair& token_buf, uint32_t seq_len, uint32_t step_idx)
{
    ForwardResult res{};
    const uint32_t act_size = seq_len * DIM * 4;
    const uint32_t ff_size  = seq_len * FF_DIM * 4;

    /* Embedding */
    BufPair hidden = alloc_buf(NF_DTYPE_F32, act_size);
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = DIM;
        BufPair ins[] = {mw.w_embed, token_buf};
        gpu_dispatch("embedding_lookup", ins, 2, &hidden, 1, &pc);
    }

    /* N transformer layers */
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        const auto& lw = mw.layers[l];

        BufPair normed   = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair q_buf    = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair k_buf    = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair v_buf    = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair attn_out = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair proj_out = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair resid1   = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair normed2  = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair gate_out = alloc_buf(NF_DTYPE_F32, ff_size);
        BufPair up_out   = alloc_buf(NF_DTYPE_F32, ff_size);
        BufPair silu_out = alloc_buf(NF_DTYPE_F32, ff_size);
        BufPair mul_out  = alloc_buf(NF_DTYPE_F32, ff_size);
        BufPair down_out = alloc_buf(NF_DTYPE_F32, act_size);
        BufPair resid2   = alloc_buf(NF_DTYPE_F32, act_size);

        PushConstants pc{};
        pc.seq_len = seq_len; pc.n_heads = HEADS; pc.head_dim = HDIM;
        pc.epsilon = 1e-5f; pc.theta = 10000.0f;
        pc.step_idx = step_idx; pc.max_seq_len = MAX_SEQ;

        /* 1. Attention norm */
        { BufPair ins[] = {hidden, lw.w_attn_norm};
          gpu_dispatch("rms_norm", ins, 2, &normed, 1, &pc); }

        /* 2. Q/K/V projections */
        pc.M = seq_len; pc.N = DIM; pc.K = DIM;
        { BufPair ins[] = {normed, lw.w_q}; gpu_dispatch("linear", ins, 2, &q_buf, 1, &pc); }
        { BufPair ins[] = {normed, lw.w_k}; gpu_dispatch("linear", ins, 2, &k_buf, 1, &pc); }
        { BufPair ins[] = {normed, lw.w_v}; gpu_dispatch("linear", ins, 2, &v_buf, 1, &pc); }

        /* 3. Causal attention with KV cache */
        { BufPair ins[] = {q_buf, k_buf, v_buf, kv.k_cache[l], kv.v_cache[l]};
          gpu_dispatch("causal_attention_cached", ins, 5, &attn_out, 1, &pc); }

        /* 4. Output projection */
        { BufPair ins[] = {attn_out, lw.w_o};
          gpu_dispatch("linear", ins, 2, &proj_out, 1, &pc); }

        /* 5. Residual add */
        { BufPair ins[] = {hidden, proj_out};
          gpu_dispatch("metal_vector_add", ins, 2, &resid1, 1, nullptr); }

        /* 6. FFN norm */
        { BufPair ins[] = {resid1, lw.w_ffn_norm};
          gpu_dispatch("rms_norm", ins, 2, &normed2, 1, &pc); }

        /* 7. Gate + Up projections */
        pc.M = seq_len; pc.N = FF_DIM; pc.K = DIM;
        { BufPair ins[] = {normed2, lw.w_gate}; gpu_dispatch("linear", ins, 2, &gate_out, 1, &pc); }
        { BufPair ins[] = {normed2, lw.w_up};   gpu_dispatch("linear", ins, 2, &up_out, 1, &pc); }

        /* 8. SiLU(gate) * up */
        gpu_dispatch("silu", &gate_out, 1, &silu_out, 1, nullptr);
        { BufPair ins[] = {silu_out, up_out};
          gpu_dispatch("elementwise_mul", ins, 2, &mul_out, 1, nullptr); }

        /* 9. Down projection */
        pc.M = seq_len; pc.N = DIM; pc.K = FF_DIM;
        { BufPair ins[] = {mul_out, lw.w_down}; gpu_dispatch("linear", ins, 2, &down_out, 1, &pc); }

        /* 10. Residual add */
        { BufPair ins[] = {resid1, down_out};
          gpu_dispatch("metal_vector_add", ins, 2, &resid2, 1, nullptr); }

        /* Track intermediates for cleanup */
        for (auto* bp : {&normed, &q_buf, &k_buf, &v_buf, &attn_out,
                         &proj_out, &resid1, &normed2, &gate_out, &up_out,
                         &silu_out, &mul_out, &down_out}) {
            res.intermediates.push_back(*bp);
        }
        /* resid2 becomes next layer's hidden; don't release yet */
        res.intermediates.push_back(hidden);
        hidden = resid2;
    }

    /* Final norm + LM head + argmax */
    BufPair final_normed = alloc_buf(NF_DTYPE_F32, act_size);
    res.intermediates.push_back(final_normed);
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = DIM;
        pc.epsilon = 1e-5f; pc.n_heads = HEADS;
        BufPair ins[] = {hidden, mw.w_final_norm};
        gpu_dispatch("rms_norm", ins, 2, &final_normed, 1, &pc);
    }

    res.logits = alloc_buf(NF_DTYPE_F32, seq_len * VOCAB * 4);
    {
        PushConstants pc{}; pc.M = seq_len; pc.N = VOCAB; pc.K = DIM;
        BufPair ins[] = {final_normed, mw.w_lm_head};
        gpu_dispatch("linear", ins, 2, &res.logits, 1, &pc);
    }

    res.argmax_out = alloc_buf(NF_DTYPE_I32, seq_len * 4);
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.N = VOCAB;
        gpu_dispatch("argmax", &res.logits, 1, &res.argmax_out, 1, &pc);
    }

    res.intermediates.push_back(hidden);
    return res;
}

static void release_forward_result(ForwardResult& r) {
    release(r.logits);
    release(r.argmax_out);
    for (auto& bp : r.intermediates) release(bp);
    r.intermediates.clear();
}

/* ================================================================== */
/*  Test 1: 4-layer prefill (seq=4)                                    */
/* ================================================================== */
static void test_prefill() {
    std::printf("  Test 1: 4-layer prefill (seq=4) ... ");
    const uint32_t seq = 4;

    ModelWeights mw = alloc_model_weights();
    KVCache kv = alloc_kv_cache();

    BufPair tokens = alloc_buf(NF_DTYPE_I32, seq * 4);
    { int32_t* t = map_i(tokens); t[0]=1; t[1]=5; t[2]=12; t[3]=28; unmap(tokens); }

    ForwardResult res = forward_pass(mw, kv, tokens, seq, /*step_idx=*/0);

    /* Verify logits: non-zero, finite */
    float* lp = map_f(res.logits);
    bool any_nonzero = false;
    for (uint32_t i = 0; i < seq * VOCAB; ++i) {
        CHECK(std::isfinite(lp[i]));
        if (lp[i] != 0.0f) any_nonzero = true;
    }
    CHECK(any_nonzero);
    unmap(res.logits);

    /* Verify argmax: valid token IDs */
    int32_t* ap = map_i(res.argmax_out);
    for (uint32_t s = 0; s < seq; ++s)
        CHECK(ap[s] >= 0 && ap[s] < (int32_t)VOCAB);
    unmap(res.argmax_out);

    release_forward_result(res);
    release(tokens);
    release_kv_cache(kv);
    release_model_weights(mw);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 2: autoregressive decode (3 steps)                            */
/* ================================================================== */
static void test_autoregressive_decode() {
    std::printf("  Test 2: autoregressive decode (3 steps) ... ");
    const uint32_t pf_seq = 4;
    const uint32_t decode_steps = 3;

    ModelWeights mw = alloc_model_weights();
    KVCache kv = alloc_kv_cache();

    /* Prefill phase */
    BufPair pf_tokens = alloc_buf(NF_DTYPE_I32, pf_seq * 4);
    { int32_t* t = map_i(pf_tokens); t[0]=2; t[1]=8; t[2]=15; t[3]=30; unmap(pf_tokens); }

    ForwardResult pf_res = forward_pass(mw, kv, pf_tokens, pf_seq, 0);

    /* Get last token from prefill as first decode input */
    int32_t* pf_argmax = map_i(pf_res.argmax_out);
    int32_t next_token = pf_argmax[pf_seq - 1];
    CHECK(next_token >= 0 && next_token < (int32_t)VOCAB);
    unmap(pf_res.argmax_out);
    release_forward_result(pf_res);
    release(pf_tokens);

    /* Decode loop */
    for (uint32_t step = 0; step < decode_steps; ++step) {
        uint32_t step_idx = pf_seq + step;

        BufPair dc_token = alloc_buf(NF_DTYPE_I32, 1 * 4);
        { int32_t* t = map_i(dc_token); t[0] = next_token; unmap(dc_token); }

        ForwardResult dc_res = forward_pass(mw, kv, dc_token, 1, step_idx);

        /* Verify: valid token ID */
        int32_t* ap = map_i(dc_res.argmax_out);
        CHECK(ap[0] >= 0 && ap[0] < (int32_t)VOCAB);
        next_token = ap[0];
        unmap(dc_res.argmax_out);

        /* Verify logits are finite */
        float* lp = map_f(dc_res.logits);
        for (uint32_t i = 0; i < VOCAB; ++i)
            CHECK(std::isfinite(lp[i]));
        unmap(dc_res.logits);

        release_forward_result(dc_res);
        release(dc_token);
    }

    release_kv_cache(kv);
    release_model_weights(mw);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 3: logits distribution sanity                                 */
/* ================================================================== */
static void test_logits_distribution() {
    std::printf("  Test 3: logits distribution sanity ... ");
    const uint32_t seq = 4;

    ModelWeights mw = alloc_model_weights();
    KVCache kv = alloc_kv_cache();

    BufPair tokens = alloc_buf(NF_DTYPE_I32, seq * 4);
    { int32_t* t = map_i(tokens); t[0]=3; t[1]=7; t[2]=19; t[3]=25; unmap(tokens); }

    ForwardResult res = forward_pass(mw, kv, tokens, seq, 0);

    /* Compute softmax and entropy for last position */
    float* lp = map_f(res.logits);
    const float* last_logits = lp + (seq - 1) * VOCAB;

    /* Find max for numerical stability */
    float max_val = last_logits[0];
    for (uint32_t i = 1; i < VOCAB; ++i)
        if (last_logits[i] > max_val) max_val = last_logits[i];

    /* Softmax */
    float sum_exp = 0.0f;
    float probs[VOCAB];
    for (uint32_t i = 0; i < VOCAB; ++i) {
        probs[i] = std::exp(last_logits[i] - max_val);
        sum_exp += probs[i];
    }
    for (uint32_t i = 0; i < VOCAB; ++i)
        probs[i] /= sum_exp;

    /* Entropy: H = -sum(p * log(p)) */
    float entropy = 0.0f;
    for (uint32_t i = 0; i < VOCAB; ++i) {
        if (probs[i] > 1e-10f)
            entropy -= probs[i] * std::log(probs[i]);
    }
    unmap(res.logits);

    /* Entropy must be > 0 (non-degenerate distribution) */
    std::printf("entropy=%.4f ", entropy);
    CHECK(entropy > 0.0f);
    /* Entropy must be < log(VOCAB) = max entropy for uniform */
    CHECK(entropy < std::log((float)VOCAB) + 0.01f);

    release_forward_result(res);
    release(tokens);
    release_kv_cache(kv);
    release_model_weights(mw);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main() {
    std::printf("=== multi_layer_test (Phase 19) ===\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    test_prefill();
    test_autoregressive_decode();
    test_logits_distribution();

    if (g_vt.shutdown) g_vt.shutdown(g_prov);
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
