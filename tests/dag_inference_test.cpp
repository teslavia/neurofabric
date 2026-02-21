/**
 * @file dag_inference_test.cpp
 * @brief Phase 20 — DAG-Driven Multi-Layer Inference
 *
 * Builds the entire 4-layer Transformer forward pass as a single DAG graph,
 * driven by Session.step(). Validates bit-exact match against Phase 19's
 * sequential gpu_dispatch() approach.
 *
 * Model config: vocab=32, dim=16, heads=4, head_dim=4, ff_dim=32, max_seq=32, n_layers=4
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "neurofabric/neuro_buffer_abi.h"
#include "neurofabric/neuro_scheduler_abi.h"
#include "neurofabric/PipelineEngine.hpp"

#include <cassert>
#include <chrono>
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

/* Model hyperparameters (same as multi_layer_test) */
static constexpr uint32_t VOCAB   = 32;
static constexpr uint32_t DIM     = 16;
static constexpr uint32_t HEADS   = 4;
static constexpr uint32_t HDIM    = 4;
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

/* --- Single-node GPU dispatch (Phase 19 reference path) --- */
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
/*  Weight & Buffer Management (identical to multi_layer_test)          */
/* ================================================================== */

struct LayerWeights {
    BufPair w_attn_norm, w_q, w_k, w_v, w_o;
    BufPair w_ffn_norm, w_gate, w_up, w_down;
};

struct ModelWeights {
    BufPair w_embed;
    LayerWeights layers[N_LAYERS];
    BufPair w_final_norm, w_lm_head;
};

struct KVCache {
    BufPair k_cache[N_LAYERS];
    BufPair v_cache[N_LAYERS];
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
/*  Sequential forward_pass (Phase 19 reference — identical logic)      */
/* ================================================================== */

struct ForwardResult {
    BufPair logits;
    BufPair argmax_out;
    std::vector<BufPair> intermediates;
};

static ForwardResult forward_pass(
    const ModelWeights& mw, KVCache& kv,
    BufPair& token_buf, uint32_t seq_len, uint32_t step_idx)
{
    ForwardResult res{};
    const uint32_t act_size = seq_len * DIM * 4;
    const uint32_t ff_size  = seq_len * FF_DIM * 4;

    BufPair hidden = alloc_buf(NF_DTYPE_F32, act_size);
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = DIM;
        BufPair ins[] = {mw.w_embed, token_buf};
        gpu_dispatch("embedding_lookup", ins, 2, &hidden, 1, &pc);
    }

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

        { BufPair ins[] = {hidden, lw.w_attn_norm};
          gpu_dispatch("rms_norm", ins, 2, &normed, 1, &pc); }
        pc.M = seq_len; pc.N = DIM; pc.K = DIM;
        { BufPair ins[] = {normed, lw.w_q}; gpu_dispatch("linear", ins, 2, &q_buf, 1, &pc); }
        { BufPair ins[] = {normed, lw.w_k}; gpu_dispatch("linear", ins, 2, &k_buf, 1, &pc); }
        { BufPair ins[] = {normed, lw.w_v}; gpu_dispatch("linear", ins, 2, &v_buf, 1, &pc); }
        { BufPair ins[] = {q_buf, k_buf, v_buf, kv.k_cache[l], kv.v_cache[l]};
          gpu_dispatch("causal_attention_cached", ins, 5, &attn_out, 1, &pc); }
        { BufPair ins[] = {attn_out, lw.w_o};
          gpu_dispatch("linear", ins, 2, &proj_out, 1, &pc); }
        { BufPair ins[] = {hidden, proj_out};
          gpu_dispatch("metal_vector_add", ins, 2, &resid1, 1, nullptr); }
        { BufPair ins[] = {resid1, lw.w_ffn_norm};
          gpu_dispatch("rms_norm", ins, 2, &normed2, 1, &pc); }
        pc.M = seq_len; pc.N = FF_DIM; pc.K = DIM;
        { BufPair ins[] = {normed2, lw.w_gate}; gpu_dispatch("linear", ins, 2, &gate_out, 1, &pc); }
        { BufPair ins[] = {normed2, lw.w_up};   gpu_dispatch("linear", ins, 2, &up_out, 1, &pc); }
        gpu_dispatch("silu", &gate_out, 1, &silu_out, 1, nullptr);
        { BufPair ins[] = {silu_out, up_out};
          gpu_dispatch("elementwise_mul", ins, 2, &mul_out, 1, nullptr); }
        pc.M = seq_len; pc.N = DIM; pc.K = FF_DIM;
        { BufPair ins[] = {mul_out, lw.w_down}; gpu_dispatch("linear", ins, 2, &down_out, 1, &pc); }
        { BufPair ins[] = {resid1, down_out};
          gpu_dispatch("metal_vector_add", ins, 2, &resid2, 1, nullptr); }

        for (auto* bp : {&normed, &q_buf, &k_buf, &v_buf, &attn_out,
                         &proj_out, &resid1, &normed2, &gate_out, &up_out,
                         &silu_out, &mul_out, &down_out})
            res.intermediates.push_back(*bp);
        res.intermediates.push_back(hidden);
        hidden = resid2;
    }

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
    release(r.logits); release(r.argmax_out);
    for (auto& bp : r.intermediates) release(bp);
    r.intermediates.clear();
}

/* ================================================================== */
/*  DAG Builder — entire N-layer forward as a single graph              */
/* ================================================================== */

struct DAGActivations {
    BufPair token_buf;
    BufPair hidden[N_LAYERS + 1];  /* ping-pong chain */
    /* Shared per-layer intermediates (safe: DAG edges enforce ordering) */
    BufPair normed, q_buf, k_buf, v_buf, attn_out, proj_out, resid1;
    BufPair normed2, gate_out, up_out, silu_out, mul_out, down_out;
    BufPair final_normed, logits, argmax_out;
};

struct DAGNodeIDs {
    uint32_t embed;
    struct LayerIDs {
        uint32_t attn_norm, q_lin, k_lin, v_lin, cached_attn, o_lin;
        uint32_t resid_add1, ffn_norm, gate_lin, up_lin, silu, elem_mul;
        uint32_t down_lin, resid_add2;
    } layer[N_LAYERS];
    uint32_t final_norm, lm_head, argmax;
};

static uint32_t add_node(nf::PipelineEngine& engine, uint32_t gid,
                          const char* op,
                          BufPair* ins, uint32_t n_in,
                          BufPair* outs, uint32_t n_out) {
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
    return engine.add_task(gid, td);
}

static DAGNodeIDs build_dag(nf::PipelineEngine& engine, uint32_t gid,
                             const ModelWeights& mw, KVCache& kv,
                             DAGActivations& act, uint32_t seq_len) {
    DAGNodeIDs ids{};
    const uint32_t act_size = seq_len * DIM * 4;
    const uint32_t ff_size  = seq_len * FF_DIM * 4;

    /* Allocate activation buffers */
    for (uint32_t i = 0; i <= N_LAYERS; ++i)
        act.hidden[i] = alloc_buf(NF_DTYPE_F32, act_size);
    act.normed     = alloc_buf(NF_DTYPE_F32, act_size);
    act.q_buf      = alloc_buf(NF_DTYPE_F32, act_size);
    act.k_buf      = alloc_buf(NF_DTYPE_F32, act_size);
    act.v_buf      = alloc_buf(NF_DTYPE_F32, act_size);
    act.attn_out   = alloc_buf(NF_DTYPE_F32, act_size);
    act.proj_out   = alloc_buf(NF_DTYPE_F32, act_size);
    act.resid1     = alloc_buf(NF_DTYPE_F32, act_size);
    act.normed2    = alloc_buf(NF_DTYPE_F32, act_size);
    act.gate_out   = alloc_buf(NF_DTYPE_F32, ff_size);
    act.up_out     = alloc_buf(NF_DTYPE_F32, ff_size);
    act.silu_out   = alloc_buf(NF_DTYPE_F32, ff_size);
    act.mul_out    = alloc_buf(NF_DTYPE_F32, ff_size);
    act.down_out   = alloc_buf(NF_DTYPE_F32, act_size);
    act.final_normed = alloc_buf(NF_DTYPE_F32, act_size);
    act.logits     = alloc_buf(NF_DTYPE_F32, seq_len * VOCAB * 4);
    act.argmax_out = alloc_buf(NF_DTYPE_I32, seq_len * 4);

    /* Embed node */
    {
        BufPair ins[] = {mw.w_embed, act.token_buf};
        ids.embed = add_node(engine, gid, "embedding_lookup", ins, 2,
                             &act.hidden[0], 1);
    }

    uint32_t prev_last = ids.embed;

    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        const auto& lw = mw.layers[l];
        auto& lid = ids.layer[l];

        /* attn_norm: hidden[l] → normed */
        { BufPair ins[] = {act.hidden[l], lw.w_attn_norm};
          lid.attn_norm = add_node(engine, gid, "rms_norm", ins, 2,
                                   &act.normed, 1); }
        engine.add_edge(gid, prev_last, lid.attn_norm);

        /* Q/K/V linear: normed → q,k,v (parallel) */
        { BufPair ins[] = {act.normed, lw.w_q};
          lid.q_lin = add_node(engine, gid, "linear", ins, 2,
                               &act.q_buf, 1); }
        { BufPair ins[] = {act.normed, lw.w_k};
          lid.k_lin = add_node(engine, gid, "linear", ins, 2,
                               &act.k_buf, 1); }
        { BufPair ins[] = {act.normed, lw.w_v};
          lid.v_lin = add_node(engine, gid, "linear", ins, 2,
                               &act.v_buf, 1); }
        engine.add_edge(gid, lid.attn_norm, lid.q_lin);
        engine.add_edge(gid, lid.attn_norm, lid.k_lin);
        engine.add_edge(gid, lid.attn_norm, lid.v_lin);

        /* causal_attention_cached: q,k,v,kcache,vcache → attn_out */
        { BufPair ins[] = {act.q_buf, act.k_buf, act.v_buf,
                           kv.k_cache[l], kv.v_cache[l]};
          lid.cached_attn = add_node(engine, gid, "causal_attention_cached",
                                     ins, 5, &act.attn_out, 1); }
        engine.add_edge(gid, lid.q_lin, lid.cached_attn);
        engine.add_edge(gid, lid.k_lin, lid.cached_attn);
        engine.add_edge(gid, lid.v_lin, lid.cached_attn);

        /* output linear: attn_out → proj_out */
        { BufPair ins[] = {act.attn_out, lw.w_o};
          lid.o_lin = add_node(engine, gid, "linear", ins, 2,
                               &act.proj_out, 1); }
        engine.add_edge(gid, lid.cached_attn, lid.o_lin);

        /* residual add: hidden[l] + proj_out → resid1 */
        { BufPair ins[] = {act.hidden[l], act.proj_out};
          lid.resid_add1 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &act.resid1, 1); }
        engine.add_edge(gid, lid.o_lin, lid.resid_add1);
        /* Also depends on hidden[l] being ready (embed or prev layer) */
        engine.add_edge(gid, prev_last, lid.resid_add1);

        /* ffn_norm: resid1 → normed2 */
        { BufPair ins[] = {act.resid1, lw.w_ffn_norm};
          lid.ffn_norm = add_node(engine, gid, "rms_norm", ins, 2,
                                  &act.normed2, 1); }
        engine.add_edge(gid, lid.resid_add1, lid.ffn_norm);

        /* gate/up linear (parallel) */
        { BufPair ins[] = {act.normed2, lw.w_gate};
          lid.gate_lin = add_node(engine, gid, "linear", ins, 2,
                                  &act.gate_out, 1); }
        { BufPair ins[] = {act.normed2, lw.w_up};
          lid.up_lin = add_node(engine, gid, "linear", ins, 2,
                                &act.up_out, 1); }
        engine.add_edge(gid, lid.ffn_norm, lid.gate_lin);
        engine.add_edge(gid, lid.ffn_norm, lid.up_lin);

        /* silu(gate) */
        lid.silu = add_node(engine, gid, "silu", &act.gate_out, 1,
                            &act.silu_out, 1);
        engine.add_edge(gid, lid.gate_lin, lid.silu);

        /* elementwise_mul: silu_out * up_out → mul_out */
        { BufPair ins[] = {act.silu_out, act.up_out};
          lid.elem_mul = add_node(engine, gid, "elementwise_mul", ins, 2,
                                  &act.mul_out, 1); }
        engine.add_edge(gid, lid.silu, lid.elem_mul);
        engine.add_edge(gid, lid.up_lin, lid.elem_mul);

        /* down linear: mul_out → down_out */
        { BufPair ins[] = {act.mul_out, lw.w_down};
          lid.down_lin = add_node(engine, gid, "linear", ins, 2,
                                  &act.down_out, 1); }
        engine.add_edge(gid, lid.elem_mul, lid.down_lin);

        /* residual add: resid1 + down_out → hidden[l+1] */
        { BufPair ins[] = {act.resid1, act.down_out};
          lid.resid_add2 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &act.hidden[l + 1], 1); }
        engine.add_edge(gid, lid.down_lin, lid.resid_add2);
        engine.add_edge(gid, lid.resid_add1, lid.resid_add2);

        prev_last = lid.resid_add2;
    }

    /* final_norm → lm_head → argmax */
    { BufPair ins[] = {act.hidden[N_LAYERS], mw.w_final_norm};
      ids.final_norm = add_node(engine, gid, "rms_norm", ins, 2,
                                &act.final_normed, 1); }
    engine.add_edge(gid, prev_last, ids.final_norm);

    { BufPair ins[] = {act.final_normed, mw.w_lm_head};
      ids.lm_head = add_node(engine, gid, "linear", ins, 2,
                             &act.logits, 1); }
    engine.add_edge(gid, ids.final_norm, ids.lm_head);

    ids.argmax = add_node(engine, gid, "argmax", &act.logits, 1,
                          &act.argmax_out, 1);
    engine.add_edge(gid, ids.lm_head, ids.argmax);

    return ids;
}

/* Inject push constants into all DAG nodes for a given step */
static void inject_push_constants(nf::PipelineEngine::Session& sess,
                                   const DAGNodeIDs& ids,
                                   uint32_t seq_len, uint32_t step_idx) {
    /* Embed PC */
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = DIM;
        sess.set_push_constants_by_id(ids.embed, &pc, sizeof(pc));
    }

    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        const auto& lid = ids.layer[l];
        PushConstants pc{};
        pc.seq_len = seq_len; pc.n_heads = HEADS; pc.head_dim = HDIM;
        pc.epsilon = 1e-5f; pc.theta = 10000.0f;
        pc.step_idx = step_idx; pc.max_seq_len = MAX_SEQ;

        sess.set_push_constants_by_id(lid.attn_norm, &pc, sizeof(pc));

        /* Q/K/V + output linear */
        pc.M = seq_len; pc.N = DIM; pc.K = DIM;
        sess.set_push_constants_by_id(lid.q_lin, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.k_lin, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.v_lin, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.cached_attn, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.o_lin, &pc, sizeof(pc));

        /* resid_add1 and resid_add2 don't need PC (metal_vector_add) */

        /* FFN norm */
        sess.set_push_constants_by_id(lid.ffn_norm, &pc, sizeof(pc));

        /* gate/up linear */
        pc.M = seq_len; pc.N = FF_DIM; pc.K = DIM;
        sess.set_push_constants_by_id(lid.gate_lin, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.up_lin, &pc, sizeof(pc));

        /* silu, elem_mul don't need PC */

        /* down linear */
        pc.M = seq_len; pc.N = DIM; pc.K = FF_DIM;
        sess.set_push_constants_by_id(lid.down_lin, &pc, sizeof(pc));
    }

    /* Final norm */
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = DIM;
        pc.epsilon = 1e-5f; pc.n_heads = HEADS;
        sess.set_push_constants_by_id(ids.final_norm, &pc, sizeof(pc));
    }
    /* LM head */
    {
        PushConstants pc{}; pc.M = seq_len; pc.N = VOCAB; pc.K = DIM;
        sess.set_push_constants_by_id(ids.lm_head, &pc, sizeof(pc));
    }
    /* Argmax */
    {
        PushConstants pc{}; pc.seq_len = seq_len; pc.N = VOCAB;
        sess.set_push_constants_by_id(ids.argmax, &pc, sizeof(pc));
    }
}

static void release_dag_activations(DAGActivations& act) {
    for (uint32_t i = 0; i <= N_LAYERS; ++i) release(act.hidden[i]);
    release(act.normed); release(act.q_buf); release(act.k_buf);
    release(act.v_buf); release(act.attn_out); release(act.proj_out);
    release(act.resid1); release(act.normed2); release(act.gate_out);
    release(act.up_out); release(act.silu_out); release(act.mul_out);
    release(act.down_out); release(act.final_normed);
    release(act.logits); release(act.argmax_out);
}

/* ================================================================== */
/*  Test 1: DAG prefill vs sequential — bit-exact match                 */
/* ================================================================== */
static void test_dag_prefill_match() {
    std::printf("  Test 1: DAG prefill vs sequential — bit-exact match ... ");
    const uint32_t seq = 4;

    ModelWeights mw = alloc_model_weights();
    KVCache kv_seq = alloc_kv_cache();
    KVCache kv_dag = alloc_kv_cache();

    /* --- Sequential reference --- */
    BufPair seq_tokens = alloc_buf(NF_DTYPE_I32, seq * 4);
    { int32_t* t = map_i(seq_tokens); t[0]=1; t[1]=5; t[2]=12; t[3]=28; unmap(seq_tokens); }
    ForwardResult seq_res = forward_pass(mw, kv_seq, seq_tokens, seq, 0);

    /* --- DAG path --- */
    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
    uint32_t gid = engine.create_graph();

    DAGActivations act{};
    act.token_buf = alloc_buf(NF_DTYPE_I32, seq * 4);
    { int32_t* t = map_i(act.token_buf); t[0]=1; t[1]=5; t[2]=12; t[3]=28; unmap(act.token_buf); }

    DAGNodeIDs ids = build_dag(engine, gid, mw, kv_dag, act, seq);

    nf::PipelineEngine::Session sess(engine, gid);
    CHECK(sess.valid());

    inject_push_constants(sess, ids, seq, 0);
    CHECK_OK(sess.step().get());

    /* Sync all DAG outputs */
    sync_read(act.logits);
    sync_read(act.argmax_out);

    /* Compare logits element-by-element */
    float* seq_lp = map_f(seq_res.logits);
    float* dag_lp = map_f(act.logits);
    for (uint32_t i = 0; i < seq * VOCAB; ++i) {
        CHECK(std::isfinite(dag_lp[i]));
        CHECK(seq_lp[i] == dag_lp[i]);
    }
    unmap(seq_res.logits);
    unmap(act.logits);

    /* Compare argmax */
    int32_t* seq_ap = map_i(seq_res.argmax_out);
    int32_t* dag_ap = map_i(act.argmax_out);
    for (uint32_t s = 0; s < seq; ++s)
        CHECK(seq_ap[s] == dag_ap[s]);
    unmap(seq_res.argmax_out);
    unmap(act.argmax_out);

    release_forward_result(seq_res);
    release(seq_tokens);
    release(act.token_buf);
    release_dag_activations(act);
    release_kv_cache(kv_seq);
    release_kv_cache(kv_dag);
    release_model_weights(mw);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 2: DAG autoregressive decode — bit-exact match                 */
/* ================================================================== */
static void test_dag_autoregressive_match() {
    std::printf("  Test 2: DAG autoregressive decode (3 steps) — bit-exact match ... ");
    const uint32_t pf_seq = 4;
    const uint32_t decode_steps = 3;

    ModelWeights mw = alloc_model_weights();
    KVCache kv_seq = alloc_kv_cache();
    KVCache kv_dag = alloc_kv_cache();

    /* --- Prefill with both methods (same tokens) --- */
    BufPair seq_tokens = alloc_buf(NF_DTYPE_I32, pf_seq * 4);
    { int32_t* t = map_i(seq_tokens); t[0]=2; t[1]=8; t[2]=15; t[3]=30; unmap(seq_tokens); }
    ForwardResult pf_seq_res = forward_pass(mw, kv_seq, seq_tokens, pf_seq, 0);

    /* DAG prefill */
    nf::PipelineEngine engine(4);
    engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
    uint32_t gid_pf = engine.create_graph();

    DAGActivations act_pf{};
    act_pf.token_buf = alloc_buf(NF_DTYPE_I32, pf_seq * 4);
    { int32_t* t = map_i(act_pf.token_buf); t[0]=2; t[1]=8; t[2]=15; t[3]=30; unmap(act_pf.token_buf); }

    DAGNodeIDs ids_pf = build_dag(engine, gid_pf, mw, kv_dag, act_pf, pf_seq);
    nf::PipelineEngine::Session sess_pf(engine, gid_pf);
    CHECK(sess_pf.valid());
    inject_push_constants(sess_pf, ids_pf, pf_seq, 0);
    CHECK_OK(sess_pf.step().get());
    sync_read(act_pf.argmax_out);

    int32_t* pf_seq_ap = map_i(pf_seq_res.argmax_out);
    int32_t* pf_dag_ap = map_i(act_pf.argmax_out);
    int32_t seq_next = pf_seq_ap[pf_seq - 1];
    int32_t dag_next = pf_dag_ap[pf_seq - 1];
    CHECK(seq_next == dag_next);
    unmap(pf_seq_res.argmax_out);
    unmap(act_pf.argmax_out);

    release_forward_result(pf_seq_res);
    release(seq_tokens);
    release(act_pf.token_buf);
    release_dag_activations(act_pf);

    /* --- Decode loop --- */
    for (uint32_t step = 0; step < decode_steps; ++step) {
        uint32_t step_idx = pf_seq + step;

        /* Sequential decode */
        BufPair dc_seq_tok = alloc_buf(NF_DTYPE_I32, 1 * 4);
        { int32_t* t = map_i(dc_seq_tok); t[0] = seq_next; unmap(dc_seq_tok); }
        ForwardResult dc_seq_res = forward_pass(mw, kv_seq, dc_seq_tok, 1, step_idx);

        /* DAG decode — build fresh graph for seq_len=1 */
        uint32_t gid_dc = engine.create_graph();
        DAGActivations act_dc{};
        act_dc.token_buf = alloc_buf(NF_DTYPE_I32, 1 * 4);
        { int32_t* t = map_i(act_dc.token_buf); t[0] = dag_next; unmap(act_dc.token_buf); }

        DAGNodeIDs ids_dc = build_dag(engine, gid_dc, mw, kv_dag, act_dc, 1);
        nf::PipelineEngine::Session sess_dc(engine, gid_dc);
        CHECK(sess_dc.valid());
        inject_push_constants(sess_dc, ids_dc, 1, step_idx);
        CHECK_OK(sess_dc.step().get());
        sync_read(act_dc.argmax_out);

        /* Compare argmax tokens */
        int32_t* dc_seq_ap = map_i(dc_seq_res.argmax_out);
        int32_t* dc_dag_ap = map_i(act_dc.argmax_out);
        CHECK(dc_seq_ap[0] == dc_dag_ap[0]);
        seq_next = dc_seq_ap[0];
        dag_next = dc_dag_ap[0];
        unmap(dc_seq_res.argmax_out);
        unmap(act_dc.argmax_out);

        release_forward_result(dc_seq_res);
        release(dc_seq_tok);
        release(act_dc.token_buf);
        release_dag_activations(act_dc);
    }

    release_kv_cache(kv_seq);
    release_kv_cache(kv_dag);
    release_model_weights(mw);
    std::printf("PASS\n");
}

/* ================================================================== */
/*  Test 3: Performance comparison (observational)                      */
/* ================================================================== */
static void test_performance_comparison() {
    std::printf("  Test 3: performance comparison\n");
    const uint32_t seq = 4;
    const int ITERS = 10;

    ModelWeights mw = alloc_model_weights();

    /* --- Sequential timing --- */
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            KVCache kv = alloc_kv_cache();
            BufPair tokens = alloc_buf(NF_DTYPE_I32, seq * 4);
            { int32_t* t = map_i(tokens); t[0]=1; t[1]=5; t[2]=12; t[3]=28; unmap(tokens); }
            ForwardResult res = forward_pass(mw, kv, tokens, seq, 0);
            release_forward_result(res);
            release(tokens);
            release_kv_cache(kv);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("    Sequential: %.2f ms (%d iters)\n", ms, ITERS);
    }

    /* --- DAG Session timing --- */
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            KVCache kv = alloc_kv_cache();

            nf::PipelineEngine engine(4);
            engine.register_provider(g_prov, g_vt, NF_AFFINITY_GPU);
            uint32_t gid = engine.create_graph();

            DAGActivations act{};
            act.token_buf = alloc_buf(NF_DTYPE_I32, seq * 4);
            { int32_t* t = map_i(act.token_buf); t[0]=1; t[1]=5; t[2]=12; t[3]=28; unmap(act.token_buf); }

            DAGNodeIDs ids = build_dag(engine, gid, mw, kv, act, seq);
            nf::PipelineEngine::Session sess(engine, gid);
            inject_push_constants(sess, ids, seq, 0);
            CHECK_OK(sess.step().get());

            release(act.token_buf);
            release_dag_activations(act);
            release_kv_cache(kv);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("    DAG Session: %.2f ms (%d iters)\n", ms, ITERS);
    }

    release_model_weights(mw);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main() {
    std::printf("=== dag_inference_test (Phase 20) ===\n");

    CHECK_OK(nf_plugin_register(&g_vt, &g_prov));
    CHECK_OK(nf_plugin_register_mem(&g_mem_vt));
    CHECK_OK(g_vt.init(g_prov));
    std::printf("  GPU: %s\n", g_vt.get_name(g_prov));

    test_dag_prefill_match();
    test_dag_autoregressive_match();
    test_performance_comparison();

    if (g_vt.shutdown) g_vt.shutdown(g_prov);
    std::printf("=== ALL PASSED ===\n");
    return 0;
}
