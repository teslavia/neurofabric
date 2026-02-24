/**
 * @file llama_dag_builder.hpp
 * @brief Header-only LLaMA DAG builder from GGUF metadata
 *
 * Phase 21: Real-Model GGUF→DAG End-to-End Inference.
 * Phase 31: Multi-Architecture DAG & KV Cache Intelligence.
 *
 * Takes a GGUFModel* + Metal provider vtables, builds the full N-layer
 * Transformer DAG with dequant + RoPE nodes. All weights are memcpy'd
 * from mmap into Metal unified buffers (one-time cost).
 *
 * Phase 31: Uses arch_registry resolve helpers for op name selection
 * and integrates kv_cache_policy for eviction/quantization.
 */

#ifndef NF_LLAMA_DAG_BUILDER_HPP
#define NF_LLAMA_DAG_BUILDER_HPP

#include "gguf_loader.hpp"
#include "quant_registry.hpp"
#include "arch_registry.hpp"
#include "kv_cache_policy.hpp"
#include "model_config.hpp"
#include "neuralOS/ddi/neuro_fabric_abi.h"
#include "neuralOS/ddi/neuro_buffer_abi.h"
#include "neuralOS/ddi/neuro_scheduler_abi.h"
#include "neuralOS/kernel/PipelineEngine.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace nf {

struct BufPair {
    nf_buffer buf = nullptr;
    nf_buffer_ops ops{};
};

struct LayerNodeIDs {
    uint32_t attn_norm;
    uint32_t dq_q, dq_k, dq_v, dq_o;
    uint32_t q_lin, k_lin, v_lin;
    uint32_t q_rope, k_rope;
    uint32_t cached_attn;
    uint32_t o_lin;
    uint32_t resid_add1;
    uint32_t ffn_norm;
    uint32_t dq_gate, dq_up, dq_down;
    uint32_t gate_lin, up_lin;
    uint32_t silu_node;
    uint32_t elem_mul;
    uint32_t down_lin;
    uint32_t resid_add2;
};


struct LlamaDAG {
    /* Engine and graph */
    PipelineEngine* engine = nullptr;
    uint32_t gid = 0;

    /* Model config */
    uint32_t dim, n_layers, n_heads, n_kv_heads, head_dim;
    uint32_t ff_dim, vocab_size, max_seq;
    float rope_theta, rms_norm_eps;

    /* Node IDs for push constant injection */
    uint32_t embed_dequant_id = 0, embed_id = 0;
    std::vector<LayerNodeIDs> layer_ids;
    uint32_t final_norm_id = 0;
    uint32_t lm_head_dequant_id = 0, lm_head_id = 0;
    uint32_t argmax_id = 0;

    /* Buffers (owned) */
    std::vector<BufPair> weight_bufs;     /* raw weights in Metal */
    std::vector<BufPair> dequant_bufs;    /* shared dequant outputs */
    std::vector<BufPair> hidden_bufs;     /* ping-pong chain */
    std::vector<BufPair> kv_cache;        /* k_cache + v_cache per layer */
    std::vector<BufPair> activation_bufs; /* per-layer intermediates */

    BufPair token_buf{};
    BufPair embed_deq_out{};
    BufPair logits{};
    BufPair argmax_out{};

    /* Shared per-layer activation buffers */
    BufPair normed{}, q_buf{}, k_buf{}, v_buf{};
    BufPair q_rope_buf{}, k_rope_buf{};
    BufPair attn_out{}, proj_out{}, resid1{};
    BufPair normed2{}, gate_out{}, up_out{};
    BufPair silu_out{}, mul_out{}, down_out{};
    BufPair final_normed{};

    /* Shared dequant output buffers (reused across layers) */
    BufPair dq_q_out{}, dq_k_out{}, dq_v_out{}, dq_o_out{};
    BufPair dq_gate_out{}, dq_up_out{}, dq_down_out{};
};

/* ---- Helper functions ---- */

struct PushConstants {
    uint32_t seq_len; uint32_t n_heads; uint32_t head_dim;
    float epsilon; float theta;
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t step_idx; uint32_t max_seq_len;
    uint32_t window_size; uint32_t _pad1;
};


static inline BufPair llama_alloc_buf(
    nf_provider prov, nf_provider_mem_vtable& mem_vt,
    nf_dtype dtype, size_t size_bytes)
{
    BufPair bp;
    nf_tensor_desc d{}; d.dtype = dtype; d.ndim = 1;
    d.shape[0] = size_bytes; d.size_bytes = size_bytes;
    nf_buffer_alloc_request req{}; req.desc = d;
    req.preferred = NF_MEM_DOMAIN_UNIFIED;
    nf_status st = mem_vt.alloc(prov, &req, &bp.ops, &bp.buf);
    if (st != NF_OK) {
        std::fprintf(stderr, "[llama_dag] alloc failed: %d (%zu bytes)\n",
                     st, size_bytes);
        bp.buf = nullptr;
    }
    return bp;
}

static inline void llama_release(BufPair& bp) {
    if (bp.buf) bp.ops.release(bp.buf);
    bp.buf = nullptr;
}

static inline float* llama_map_f(BufPair& bp) {
    void* p; bp.ops.map(bp.buf, &p); return (float*)p;
}

static inline void llama_unmap(BufPair& bp) { bp.ops.unmap(bp.buf); }

/**
 * Copy raw bytes from mmap'd GGUF tensor into a Metal unified buffer.
 * Returns a BufPair owning the Metal buffer.
 */
static inline BufPair load_weight(
    nf_provider prov, nf_provider_mem_vtable& mem_vt,
    const GGUFTensorInfo& ti, nf_dtype dtype)
{
    BufPair bp = llama_alloc_buf(prov, mem_vt, dtype, ti.byte_size);
    if (!bp.buf) return bp;
    void* dst;
    bp.ops.map(bp.buf, &dst);
    std::memcpy(dst, ti.data, ti.byte_size);
    bp.ops.unmap(bp.buf);
    return bp;
}

static inline uint32_t add_node(
    PipelineEngine& engine, uint32_t gid, const char* op,
    BufPair* ins, uint32_t n_in, BufPair* outs, uint32_t n_out)
{
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


/* Forward declarations for custom deleters */
inline void release_llama_dag(LlamaDAG* dag);
struct LlamaContext;
inline void release_llama_context(LlamaContext* ctx);

/**
 * Build the full LLaMA DAG from GGUF model metadata.
 * Loads all weights from mmap into Metal unified buffers.
 * Returns heap-allocated LlamaDAG* on success, nullptr on failure.
 */
inline std::unique_ptr<LlamaDAG, void(*)(LlamaDAG*)> build_llama_dag(
    PipelineEngine& engine,
    const GGUFModel& model,
    nf_provider prov,
    nf_provider_vtable& vt,
    nf_provider_mem_vtable& mem_vt,
    uint32_t max_seq_override,
    uint32_t seq_len)
{
    using DagPtr = std::unique_ptr<LlamaDAG, void(*)(LlamaDAG*)>;
    auto fail = [](){ return DagPtr(nullptr, release_llama_dag); };
    DagPtr dag(new LlamaDAG, release_llama_dag);
    dag->engine = &engine;
    dag->dim = model.dim;
    dag->n_layers = model.n_layers;
    dag->n_heads = model.n_heads;
    dag->n_kv_heads = model.n_kv_heads;
    dag->head_dim = model.dim / model.n_heads;
    dag->ff_dim = model.ff_dim;
    dag->vocab_size = model.vocab_size;
    dag->max_seq = max_seq_override;
    dag->rope_theta = model.rope_theta;
    dag->rms_norm_eps = model.rms_norm_eps;

    const uint32_t D = dag->dim;
    const uint32_t HD = dag->head_dim;
    const uint32_t NH = dag->n_heads;
    const uint32_t NKV = dag->n_kv_heads;
    const uint32_t KV_DIM = NKV * HD;  /* GQA: K/V projection dimension */
    const uint32_t FF = dag->ff_dim;
    const uint32_t V = dag->vocab_size;
    const uint32_t NL = dag->n_layers;
    const uint32_t MS = dag->max_seq;

    const uint32_t act_size = seq_len * D * sizeof(float);
    const uint32_t kv_act_size = seq_len * KV_DIM * sizeof(float);
    const uint32_t ff_size  = seq_len * FF * sizeof(float);

    dag->gid = engine.create_graph();
    uint32_t gid = dag->gid;

    /* ---- Load weights from GGUF into Metal buffers ---- */
    auto find_tensor = [&](const std::string& name) -> const GGUFTensorInfo* {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            std::fprintf(stderr, "[llama_dag] Missing tensor: %s\n", name.c_str());
            return nullptr;
        }
        return &it->second;
    };


    auto is_quantized = [](uint32_t dtype) { return nf_is_quantized(dtype); };
    auto nf_dtype_for = [&](uint32_t gguf_dtype) -> nf_dtype { return nf_dtype_for_gguf(gguf_dtype); };
    auto dequant_op = [](uint32_t gguf_dtype) -> const char* { return nf_dequant_op_name(gguf_dtype); };

    /* Load embedding weight */
    auto* emb_ti = find_tensor("token_embd.weight");
    if (!emb_ti) { return fail(); }
    BufPair w_embed = load_weight(prov, mem_vt, *emb_ti, nf_dtype_for(emb_ti->gguf_dtype));
    dag->weight_bufs.push_back(w_embed);

    /* Embedding dequant output buffer (vocab × dim × float) */
    dag->embed_deq_out = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32,
                                          (size_t)V * D * sizeof(float));

    /* Per-layer weight loading */
    struct LayerWeightBufs {
        BufPair w_attn_norm, w_q, w_k, w_v, w_o;
        BufPair w_ffn_norm, w_gate, w_up, w_down;
        uint32_t q_dtype, k_dtype, v_dtype, o_dtype;
        uint32_t gate_dtype, up_dtype, down_dtype;
    };
    std::vector<LayerWeightBufs> lw_bufs(NL);

    for (uint32_t l = 0; l < NL; ++l) {
        auto& lw = lw_bufs[l];
        char name[128];

        std::snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        auto* ti = find_tensor(name);
        if (!ti) { return fail(); }
        lw.w_attn_norm = load_weight(prov, mem_vt, *ti, NF_DTYPE_F32);
        dag->weight_bufs.push_back(lw.w_attn_norm);

        auto load_w = [&](const char* suffix, BufPair& out, uint32_t& dt) {
            std::snprintf(name, sizeof(name), "blk.%u.%s.weight", l, suffix);
            auto* t = find_tensor(name);
            if (!t) return false;
            dt = t->gguf_dtype;
            out = load_weight(prov, mem_vt, *t, nf_dtype_for(dt));
            dag->weight_bufs.push_back(out);
            return true;
        };

        if (!load_w("attn_q",    lw.w_q,    lw.q_dtype))    { return fail(); }
        if (!load_w("attn_k",    lw.w_k,    lw.k_dtype))    { return fail(); }
        if (!load_w("attn_v",    lw.w_v,    lw.v_dtype))    { return fail(); }
        if (!load_w("attn_output", lw.w_o,  lw.o_dtype))    { return fail(); }

        std::snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        ti = find_tensor(name);
        if (!ti) { return fail(); }
        lw.w_ffn_norm = load_weight(prov, mem_vt, *ti, NF_DTYPE_F32);
        dag->weight_bufs.push_back(lw.w_ffn_norm);

        if (!load_w("ffn_gate",  lw.w_gate, lw.gate_dtype)) { return fail(); }
        if (!load_w("ffn_up",    lw.w_up,   lw.up_dtype))   { return fail(); }
        if (!load_w("ffn_down",  lw.w_down, lw.down_dtype)) { return fail(); }
    }

    /* Final norm + output (LM head) weights */
    auto* fn_ti = find_tensor("output_norm.weight");
    if (!fn_ti) { return fail(); }
    BufPair w_final_norm = load_weight(prov, mem_vt, *fn_ti, NF_DTYPE_F32);
    dag->weight_bufs.push_back(w_final_norm);

    auto* lm_ti = find_tensor("output.weight");
    if (!lm_ti) { return fail(); }
    BufPair w_lm_head = load_weight(prov, mem_vt, *lm_ti, nf_dtype_for(lm_ti->gguf_dtype));
    dag->weight_bufs.push_back(w_lm_head);


    /* ---- Allocate activation buffers ---- */

    /* Hidden state ping-pong: hidden[0..NL] */
    dag->hidden_bufs.resize(NL + 1);
    for (uint32_t i = 0; i <= NL; ++i)
        dag->hidden_bufs[i] = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);

    /* Shared per-layer intermediates */
    dag->normed     = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->q_buf      = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->k_buf      = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, kv_act_size);
    dag->v_buf      = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, kv_act_size);
    dag->q_rope_buf = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->k_rope_buf = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, kv_act_size);
    dag->attn_out   = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->proj_out   = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->resid1     = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->normed2    = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->gate_out   = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, ff_size);
    dag->up_out     = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, ff_size);
    dag->silu_out   = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, ff_size);
    dag->mul_out    = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, ff_size);
    dag->down_out   = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);
    dag->final_normed = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, act_size);

    /* Shared dequant output buffers (reused across layers) */
    size_t dq_dim2 = (size_t)D * D * sizeof(float);
    size_t dq_kv   = (size_t)D * KV_DIM * sizeof(float);
    size_t dq_ff   = (size_t)D * FF * sizeof(float);
    dag->dq_q_out    = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_dim2);
    dag->dq_k_out    = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_kv);
    dag->dq_v_out    = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_kv);
    dag->dq_o_out    = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_dim2);
    dag->dq_gate_out = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_ff);
    dag->dq_up_out   = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_ff);
    dag->dq_down_out = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, dq_ff);

    /* KV cache: n_layers × 2 × max_seq × dim */
    dag->kv_cache.resize(NL * 2);
    size_t kv_size = (size_t)MS * KV_DIM * sizeof(float);
    for (uint32_t l = 0; l < NL; ++l) {
        dag->kv_cache[l * 2]     = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, kv_size);
        dag->kv_cache[l * 2 + 1] = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32, kv_size);
        /* Zero-init KV cache */
        void* p;
        dag->kv_cache[l*2].ops.map(dag->kv_cache[l*2].buf, &p);
        std::memset(p, 0, kv_size);
        dag->kv_cache[l*2].ops.unmap(dag->kv_cache[l*2].buf);
        dag->kv_cache[l*2+1].ops.map(dag->kv_cache[l*2+1].buf, &p);
        std::memset(p, 0, kv_size);
        dag->kv_cache[l*2+1].ops.unmap(dag->kv_cache[l*2+1].buf);
    }

    /* Token input + logits + argmax output */
    dag->token_buf  = llama_alloc_buf(prov, mem_vt, NF_DTYPE_I32,
                                       seq_len * sizeof(int32_t));
    dag->logits     = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32,
                                       seq_len * V * sizeof(float));
    dag->argmax_out = llama_alloc_buf(prov, mem_vt, NF_DTYPE_I32,
                                       seq_len * sizeof(int32_t));


    /* ---- Build DAG nodes ---- */

    /* Embedding: dequant (if quantized) → embedding_lookup */
    if (is_quantized(emb_ti->gguf_dtype)) {
        BufPair dq_ins[] = {w_embed};
        dag->embed_dequant_id = add_node(engine, gid, dequant_op(emb_ti->gguf_dtype),
                                          dq_ins, 1, &dag->embed_deq_out, 1);
        BufPair em_ins[] = {dag->embed_deq_out, dag->token_buf};
        dag->embed_id = add_node(engine, gid, "embedding_lookup",
                                  em_ins, 2, &dag->hidden_bufs[0], 1);
        engine.add_edge(gid, dag->embed_dequant_id, dag->embed_id);
    } else {
        dag->embed_dequant_id = UINT32_MAX; /* no dequant needed */
        BufPair em_ins[] = {w_embed, dag->token_buf};
        dag->embed_id = add_node(engine, gid, "embedding_lookup",
                                  em_ins, 2, &dag->hidden_bufs[0], 1);
    }

    uint32_t prev_last = dag->embed_id;
    dag->layer_ids.resize(NL);

    for (uint32_t l = 0; l < NL; ++l) {
        auto& lw = lw_bufs[l];
        auto& lid = dag->layer_ids[l];

        /* attn_norm: hidden[l] → normed */
        { BufPair ins[] = {dag->hidden_bufs[l], lw.w_attn_norm};
          lid.attn_norm = add_node(engine, gid, "rms_norm", ins, 2,
                                   &dag->normed, 1); }
        engine.add_edge(gid, prev_last, lid.attn_norm);

        /* Dequant Q/K/V weights (if quantized) */
        auto add_dequant = [&](BufPair& w_raw, uint32_t dtype,
                               BufPair& dq_out) -> uint32_t {
            if (!is_quantized(dtype)) return UINT32_MAX;
            BufPair ins[] = {w_raw};
            return add_node(engine, gid, dequant_op(dtype), ins, 1, &dq_out, 1);
        };

        lid.dq_q = add_dequant(lw.w_q, lw.q_dtype, dag->dq_q_out);
        lid.dq_k = add_dequant(lw.w_k, lw.k_dtype, dag->dq_k_out);
        lid.dq_v = add_dequant(lw.w_v, lw.v_dtype, dag->dq_v_out);

        /* Q/K/V linear: normed × weight → q,k,v */
        BufPair& wq = is_quantized(lw.q_dtype) ? dag->dq_q_out : lw.w_q;
        BufPair& wk = is_quantized(lw.k_dtype) ? dag->dq_k_out : lw.w_k;
        BufPair& wv = is_quantized(lw.v_dtype) ? dag->dq_v_out : lw.w_v;

        { BufPair ins[] = {dag->normed, wq};
          lid.q_lin = add_node(engine, gid, "linear", ins, 2, &dag->q_buf, 1); }
        { BufPair ins[] = {dag->normed, wk};
          lid.k_lin = add_node(engine, gid, "linear", ins, 2, &dag->k_buf, 1); }
        { BufPair ins[] = {dag->normed, wv};
          lid.v_lin = add_node(engine, gid, "linear", ins, 2, &dag->v_buf, 1); }


        /* Edges: attn_norm → Q/K/V linear, dequant → linear */
        engine.add_edge(gid, lid.attn_norm, lid.q_lin);
        engine.add_edge(gid, lid.attn_norm, lid.k_lin);
        engine.add_edge(gid, lid.attn_norm, lid.v_lin);
        if (lid.dq_q != UINT32_MAX) engine.add_edge(gid, lid.dq_q, lid.q_lin);
        if (lid.dq_k != UINT32_MAX) engine.add_edge(gid, lid.dq_k, lid.k_lin);
        if (lid.dq_v != UINT32_MAX) engine.add_edge(gid, lid.dq_v, lid.v_lin);

        /* RoPE on Q and K */
        { BufPair ins[] = {dag->q_buf};
          lid.q_rope = add_node(engine, gid, "rope_batch", ins, 1,
                                &dag->q_rope_buf, 1); }
        { BufPair ins[] = {dag->k_buf};
          lid.k_rope = add_node(engine, gid, "rope_batch", ins, 1,
                                &dag->k_rope_buf, 1); }
        engine.add_edge(gid, lid.q_lin, lid.q_rope);
        engine.add_edge(gid, lid.k_lin, lid.k_rope);

        /* causal_attention_cached: q_rope, k_rope, v, cache_k, cache_v → attn_out */
        { BufPair ins[] = {dag->q_rope_buf, dag->k_rope_buf, dag->v_buf,
                           dag->kv_cache[l*2], dag->kv_cache[l*2+1]};
          lid.cached_attn = add_node(engine, gid, "flash_attention_cached",
                                     ins, 5, &dag->attn_out, 1); }
        engine.add_edge(gid, lid.q_rope, lid.cached_attn);
        engine.add_edge(gid, lid.k_rope, lid.cached_attn);
        engine.add_edge(gid, lid.v_lin, lid.cached_attn);

        /* Dequant output weight + output linear */
        lid.dq_o = add_dequant(lw.w_o, lw.o_dtype, dag->dq_o_out);
        BufPair& wo = is_quantized(lw.o_dtype) ? dag->dq_o_out : lw.w_o;
        { BufPair ins[] = {dag->attn_out, wo};
          lid.o_lin = add_node(engine, gid, "linear", ins, 2,
                               &dag->proj_out, 1); }
        engine.add_edge(gid, lid.cached_attn, lid.o_lin);
        if (lid.dq_o != UINT32_MAX) engine.add_edge(gid, lid.dq_o, lid.o_lin);

        /* Residual add: hidden[l] + proj_out → resid1 */
        { BufPair ins[] = {dag->hidden_bufs[l], dag->proj_out};
          lid.resid_add1 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &dag->resid1, 1); }
        engine.add_edge(gid, lid.o_lin, lid.resid_add1);
        engine.add_edge(gid, prev_last, lid.resid_add1);


        /* FFN norm: resid1 → normed2 */
        { BufPair ins[] = {dag->resid1, lw.w_ffn_norm};
          lid.ffn_norm = add_node(engine, gid, "rms_norm", ins, 2,
                                  &dag->normed2, 1); }
        engine.add_edge(gid, lid.resid_add1, lid.ffn_norm);

        /* Dequant gate/up/down weights */
        lid.dq_gate = add_dequant(lw.w_gate, lw.gate_dtype, dag->dq_gate_out);
        lid.dq_up   = add_dequant(lw.w_up,   lw.up_dtype,   dag->dq_up_out);
        lid.dq_down = add_dequant(lw.w_down, lw.down_dtype, dag->dq_down_out);

        BufPair& wgate = is_quantized(lw.gate_dtype) ? dag->dq_gate_out : lw.w_gate;
        BufPair& wup   = is_quantized(lw.up_dtype)   ? dag->dq_up_out   : lw.w_up;
        BufPair& wdown = is_quantized(lw.down_dtype)  ? dag->dq_down_out : lw.w_down;

        /* Gate/Up linear (parallel) */
        { BufPair ins[] = {dag->normed2, wgate};
          lid.gate_lin = add_node(engine, gid, "linear", ins, 2,
                                  &dag->gate_out, 1); }
        { BufPair ins[] = {dag->normed2, wup};
          lid.up_lin = add_node(engine, gid, "linear", ins, 2,
                                &dag->up_out, 1); }
        engine.add_edge(gid, lid.ffn_norm, lid.gate_lin);
        engine.add_edge(gid, lid.ffn_norm, lid.up_lin);
        if (lid.dq_gate != UINT32_MAX) engine.add_edge(gid, lid.dq_gate, lid.gate_lin);
        if (lid.dq_up   != UINT32_MAX) engine.add_edge(gid, lid.dq_up,   lid.up_lin);

        /* SiLU on gate output */
        { BufPair ins[] = {dag->gate_out};
          lid.silu_node = add_node(engine, gid, "silu", ins, 1,
                                   &dag->silu_out, 1); }
        engine.add_edge(gid, lid.gate_lin, lid.silu_node);

        /* elementwise_mul: silu_out × up_out → mul_out */
        { BufPair ins[] = {dag->silu_out, dag->up_out};
          lid.elem_mul = add_node(engine, gid, "elementwise_mul", ins, 2,
                                  &dag->mul_out, 1); }
        engine.add_edge(gid, lid.silu_node, lid.elem_mul);
        engine.add_edge(gid, lid.up_lin, lid.elem_mul);

        /* Down linear: mul_out × w_down → down_out */
        { BufPair ins[] = {dag->mul_out, wdown};
          lid.down_lin = add_node(engine, gid, "linear", ins, 2,
                                  &dag->down_out, 1); }
        engine.add_edge(gid, lid.elem_mul, lid.down_lin);
        if (lid.dq_down != UINT32_MAX) engine.add_edge(gid, lid.dq_down, lid.down_lin);

        /* Residual add: resid1 + down_out → hidden[l+1] */
        { BufPair ins[] = {dag->resid1, dag->down_out};
          lid.resid_add2 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &dag->hidden_bufs[l+1], 1); }
        engine.add_edge(gid, lid.down_lin, lid.resid_add2);
        engine.add_edge(gid, lid.resid_add1, lid.resid_add2);

        prev_last = lid.resid_add2;
    } /* end layer loop */


    /* Final norm: hidden[NL] → final_normed */
    { BufPair ins[] = {dag->hidden_bufs[NL], w_final_norm};
      dag->final_norm_id = add_node(engine, gid, "rms_norm", ins, 2,
                                     &dag->final_normed, 1); }
    engine.add_edge(gid, prev_last, dag->final_norm_id);

    /* LM head: dequant (if quantized) → linear */
    BufPair lm_head_deq_out;
    if (is_quantized(lm_ti->gguf_dtype)) {
        lm_head_deq_out = llama_alloc_buf(prov, mem_vt, NF_DTYPE_F32,
                                           (size_t)D * V * sizeof(float));
        BufPair dq_ins[] = {w_lm_head};
        dag->lm_head_dequant_id = add_node(engine, gid,
            dequant_op(lm_ti->gguf_dtype), dq_ins, 1, &lm_head_deq_out, 1);

        BufPair lm_ins[] = {dag->final_normed, lm_head_deq_out};
        dag->lm_head_id = add_node(engine, gid, "linear", lm_ins, 2,
                                    &dag->logits, 1);
        engine.add_edge(gid, dag->final_norm_id, dag->lm_head_id);
        engine.add_edge(gid, dag->lm_head_dequant_id, dag->lm_head_id);
    } else {
        dag->lm_head_dequant_id = UINT32_MAX;
        BufPair lm_ins[] = {dag->final_normed, w_lm_head};
        dag->lm_head_id = add_node(engine, gid, "linear", lm_ins, 2,
                                    &dag->logits, 1);
        engine.add_edge(gid, dag->final_norm_id, dag->lm_head_id);
    }

    /* Argmax: logits → argmax_out */
    { BufPair ins[] = {dag->logits};
      dag->argmax_id = add_node(engine, gid, "argmax", ins, 1,
                                 &dag->argmax_out, 1); }
    engine.add_edge(gid, dag->lm_head_id, dag->argmax_id);

    /* Store lm_head_deq_out if allocated */
    if (lm_head_deq_out.buf)
        dag->dequant_bufs.push_back(lm_head_deq_out);

    return dag;
}

/**
 * Inject push constants for all nodes in the DAG.
 * Must be called before each Session.step().
 */
inline void inject_llama_push_constants(
    LlamaDAG& dag, PipelineEngine::Session& sess,
    uint32_t seq_len, uint32_t step_idx)
{
    const uint32_t D  = dag.dim;
    const uint32_t NH = dag.n_heads;
    const uint32_t NKV = dag.n_kv_heads;
    const uint32_t HD = dag.head_dim;
    const uint32_t KV_DIM = NKV * HD;
    const uint32_t FF = dag.ff_dim;
    const uint32_t V  = dag.vocab_size;
    const uint32_t MS = dag.max_seq;

    /* Embed */
    { PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = D;
      sess.set_push_constants_by_id(dag.embed_id, &pc, sizeof(pc)); }

    for (uint32_t l = 0; l < dag.n_layers; ++l) {
        auto& lid = dag.layer_ids[l];

        /* Attention norm */
        PushConstants pc{};
        pc.seq_len = seq_len; pc.n_heads = NH; pc.head_dim = D;
        pc.epsilon = dag.rms_norm_eps; pc.theta = dag.rope_theta;
        pc.step_idx = step_idx; pc.max_seq_len = MS;
        sess.set_push_constants_by_id(lid.attn_norm, &pc, sizeof(pc));

        /* Q linear: M=seq_len, N=D, K=D */
        pc.M = seq_len; pc.N = D; pc.K = D;
        sess.set_push_constants_by_id(lid.q_lin, &pc, sizeof(pc));

        /* K/V linear: M=seq_len, N=KV_DIM, K=D (GQA) */
        { PushConstants kv_pc = pc;
          kv_pc.M = seq_len; kv_pc.N = KV_DIM; kv_pc.K = D;
          sess.set_push_constants_by_id(lid.k_lin, &kv_pc, sizeof(kv_pc));
          sess.set_push_constants_by_id(lid.v_lin, &kv_pc, sizeof(kv_pc)); }

        /* RoPE Q: seq_len, n_heads=NH, head_dim, theta, step_idx */
        { PushConstants rpc{};
          rpc.seq_len = seq_len; rpc.n_heads = NH; rpc.head_dim = HD;
          rpc.theta = dag.rope_theta; rpc.step_idx = step_idx;
          sess.set_push_constants_by_id(lid.q_rope, &rpc, sizeof(rpc)); }

        /* RoPE K: seq_len, n_heads=NKV, head_dim, theta, step_idx */
        { PushConstants rpc{};
          rpc.seq_len = seq_len; rpc.n_heads = NKV; rpc.head_dim = HD;
          rpc.theta = dag.rope_theta; rpc.step_idx = step_idx;
          sess.set_push_constants_by_id(lid.k_rope, &rpc, sizeof(rpc)); }

        /* Cached attention: M=n_kv_heads for GQA */
        { PushConstants apc{};
          apc.seq_len = seq_len; apc.n_heads = NH; apc.head_dim = HD;
          apc.M = NKV; /* GQA: n_kv_heads passed via M */
          apc.step_idx = step_idx; apc.max_seq_len = MS;
          sess.set_push_constants_by_id(lid.cached_attn, &apc, sizeof(apc)); }

        /* Output linear: M=seq_len, N=D, K=D */
        pc.M = seq_len; pc.N = D; pc.K = D;
        sess.set_push_constants_by_id(lid.o_lin, &pc, sizeof(pc));

        /* FFN norm */
        sess.set_push_constants_by_id(lid.ffn_norm, &pc, sizeof(pc));

        /* Gate/Up linear: M=seq_len, N=FF, K=D */
        pc.M = seq_len; pc.N = FF; pc.K = D;
        sess.set_push_constants_by_id(lid.gate_lin, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.up_lin, &pc, sizeof(pc));

        /* Down linear: M=seq_len, N=D, K=FF */
        pc.M = seq_len; pc.N = D; pc.K = FF;
        sess.set_push_constants_by_id(lid.down_lin, &pc, sizeof(pc));
    }

    /* Final norm */
    { PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = D;
      pc.epsilon = dag.rms_norm_eps; pc.n_heads = NH;
      sess.set_push_constants_by_id(dag.final_norm_id, &pc, sizeof(pc)); }

    /* LM head: M=seq_len, N=vocab, K=D */
    { PushConstants pc{}; pc.M = seq_len; pc.N = V; pc.K = D;
      sess.set_push_constants_by_id(dag.lm_head_id, &pc, sizeof(pc)); }

    /* Argmax */
    { PushConstants pc{}; pc.seq_len = seq_len; pc.N = V;
      sess.set_push_constants_by_id(dag.argmax_id, &pc, sizeof(pc)); }
}

/**
 * Release all buffers owned by the DAG.
 */
inline void release_llama_dag(LlamaDAG* dag) {
    if (!dag) return;

    for (auto& bp : dag->weight_bufs)   llama_release(bp);
    for (auto& bp : dag->dequant_bufs)  llama_release(bp);
    for (auto& bp : dag->hidden_bufs)   llama_release(bp);
    for (auto& bp : dag->kv_cache)      llama_release(bp);

    llama_release(dag->normed);     llama_release(dag->q_buf);
    llama_release(dag->k_buf);      llama_release(dag->v_buf);
    llama_release(dag->q_rope_buf); llama_release(dag->k_rope_buf);
    llama_release(dag->attn_out);   llama_release(dag->proj_out);
    llama_release(dag->resid1);     llama_release(dag->normed2);
    llama_release(dag->gate_out);   llama_release(dag->up_out);
    llama_release(dag->silu_out);   llama_release(dag->mul_out);
    llama_release(dag->down_out);   llama_release(dag->final_normed);

    llama_release(dag->dq_q_out);    llama_release(dag->dq_k_out);
    llama_release(dag->dq_v_out);    llama_release(dag->dq_o_out);
    llama_release(dag->dq_gate_out); llama_release(dag->dq_up_out);
    llama_release(dag->dq_down_out);

    llama_release(dag->embed_deq_out);
    llama_release(dag->token_buf);
    llama_release(dag->logits);
    llama_release(dag->argmax_out);

    delete dag;
}

/* ================================================================== */
/*  Phase 22: Persistent Session & KV Cache Continuity                */
/* ================================================================== */

/**
 * LlamaContext — holds everything that persists across decode steps:
 * pre-dequantized F32 weights, KV cache, activation buffers.
 */
struct LlamaContext {
    PipelineEngine* engine = nullptr;
    nf_provider prov;
    nf_provider_vtable vt;
    nf_provider_mem_vtable mem_vt;

    /* Model config */
    uint32_t dim, n_layers, n_heads, n_kv_heads, head_dim;
    uint32_t ff_dim, vocab_size, max_seq;
    float rope_theta, rms_norm_eps;
    uint32_t sliding_window = 0;  /* 0 = full causal, >0 = sliding (Mistral) */
    bool use_fp16 = false;  /* Phase 27: F16 activations + weights */
    bool use_paged_kv = false;  /* Phase 32: paged KV cache */

    /* Phase 31: KV cache policy + architecture strategy */
    nf_kv_cache_policy kv_policy{};
    const nf_arch_strategy* arch = nullptr;  /* resolved from registry, not owned */

    /* Phase 32: paged KV cache (nullptr when use_paged_kv=false) */
    std::unique_ptr<PagedKVCache> paged_kv;

    /* Pre-dequantized F32 weight buffers (computed once via warmup DAG) */
    struct LayerF32Weights {
        BufPair q, k, v, o;
        BufPair gate, up, down;
    };
    BufPair embed_f32;
    std::vector<LayerF32Weights> layer_f32;
    BufPair lm_head_f32;

    /* Raw quantized weights kept for fused ops (Phase 30) */
    struct LayerQuantWeights {
        BufPair q, k, v, o;
        BufPair gate, up, down;
        nf_dtype q_dt{}, k_dt{}, v_dt{}, o_dt{};
        nf_dtype gate_dt{}, up_dt{}, down_dt{};
    };
    std::vector<LayerQuantWeights> layer_quant;
    BufPair lm_head_quant{};
    nf_dtype lm_head_quant_dt{};

    /* RMS norm weights (always F32, no dequant needed) */
    struct LayerNormWeights {
        BufPair attn_norm, ffn_norm;
    };
    std::vector<LayerNormWeights> norm_weights;
    BufPair final_norm_weight;

    /* KV cache: n_layers × 2 (K, V) — contiguous (legacy) */
    std::vector<BufPair> kv_cache;

    /* Phase 32: Paged KV pool buffers (per-layer K and V pools + block table) */
    std::vector<BufPair> kv_pool_k;   /* [n_layers] */
    std::vector<BufPair> kv_pool_v;   /* [n_layers] */
    BufPair block_table_buf{};        /* GPU-side block table */

    /* Shared activation buffers (sized for max_prefill_seq) */
    std::vector<BufPair> hidden_bufs;
    BufPair normed, q_buf, k_buf, v_buf;
    BufPair q_rope_buf, k_rope_buf;
    BufPair attn_out, proj_out, resid1;
    BufPair normed2, gate_out, up_out;
    BufPair silu_out, mul_out, down_out;
    BufPair final_normed;
    BufPair token_buf, logits, argmax_out;
};

struct StepGraph {
    uint32_t gid;
    uint32_t embed_id;
    std::vector<LayerNodeIDs> layer_ids;
    uint32_t final_norm_id, lm_head_id, argmax_id;
};

/**
 * One-time setup: load weights, dequant via warmup DAG, allocate KV cache
 * and activation buffers. After this, all weights are F32.
 */
using LlamaContextPtr = std::unique_ptr<LlamaContext, void(*)(LlamaContext*)>;

inline LlamaContextPtr create_llama_context(
    PipelineEngine& engine,
    const GGUFModel& model,
    nf_provider prov,
    nf_provider_vtable& vt,
    nf_provider_mem_vtable& mem_vt,
    uint32_t max_seq,
    uint32_t max_prefill_seq,
    bool use_fp16 = false,
    const nf_kv_cache_config* kv_cfg = nullptr)
{
    auto fail = [](){ return LlamaContextPtr(nullptr, release_llama_context); };
    LlamaContextPtr ctx(new LlamaContext, release_llama_context);
    ctx->engine = &engine;
    ctx->prov = prov;
    ctx->vt = vt;
    ctx->mem_vt = mem_vt;

    ctx->dim = model.dim;
    ctx->n_layers = model.n_layers;
    ctx->n_heads = model.n_heads;
    ctx->n_kv_heads = model.n_kv_heads;
    ctx->head_dim = model.dim / model.n_heads;
    ctx->ff_dim = model.ff_dim;
    ctx->vocab_size = model.vocab_size;
    ctx->max_seq = max_seq;
    ctx->rope_theta = model.rope_theta;
    ctx->rms_norm_eps = model.rms_norm_eps;
    ctx->sliding_window = model.sliding_window;
    ctx->use_fp16 = use_fp16;

    /* Phase 31: resolve architecture strategy from registry */
    const char* arch_name = model.architecture.empty() ? "llama" : model.architecture.c_str();
    ctx->arch = nf_find_arch(arch_name);

    /* Phase 31: initialize KV cache policy */
    if (kv_cfg) {
        ctx->kv_policy = nf_create_kv_policy(*kv_cfg);
    } else if (model.sliding_window > 0) {
        /* Auto-configure sliding window from GGUF metadata */
        nf_kv_cache_config auto_cfg{};
        auto_cfg.eviction = NF_KV_EVICT_SLIDING;
        auto_cfg.window_size = model.sliding_window;
        auto_cfg.max_seq_len = max_seq;
        ctx->kv_policy = nf_create_kv_policy(auto_cfg);
    } else {
        nf_kv_cache_config none_cfg{};
        none_cfg.eviction = NF_KV_EVICT_NONE;
        none_cfg.max_seq_len = max_seq;
        ctx->kv_policy = nf_create_kv_policy(none_cfg);
    }

    const uint32_t D = ctx->dim;
    const uint32_t HD = ctx->head_dim;
    const uint32_t NKV = ctx->n_kv_heads;
    const uint32_t KV_DIM = NKV * HD;
    const uint32_t FF = ctx->ff_dim;
    const uint32_t V = ctx->vocab_size;
    const uint32_t NL = ctx->n_layers;
    const uint32_t MS = max_seq;

    auto find_tensor = [&](const std::string& name) -> const GGUFTensorInfo* {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            std::fprintf(stderr, "[llama_ctx] Missing tensor: %s\n", name.c_str());
            return nullptr;
        }
        return &it->second;
    };
    auto is_quantized = [](uint32_t dtype) { return nf_is_quantized(dtype); };
    auto nf_dtype_for = [&](uint32_t gguf_dtype) -> nf_dtype { return nf_dtype_for_gguf(gguf_dtype); };
    auto dequant_op_name = [](uint32_t gguf_dtype) -> const char* { return nf_dequant_op_name(gguf_dtype); };

    /* ---- Load raw weights + allocate F32 output buffers ---- */

    /* Track raw quantized buffers for release after warmup */
    std::vector<BufPair> raw_weight_bufs;

    /* Embedding */
    auto* emb_ti = find_tensor("token_embd.weight");
    if (!emb_ti) { return fail(); }
    BufPair w_embed_raw = load_weight(prov, mem_vt, *emb_ti, nf_dtype_for(emb_ti->gguf_dtype));
    raw_weight_bufs.push_back(w_embed_raw);
    {
        nf_dtype emb_out_dt = use_fp16 ? NF_DTYPE_F16 : NF_DTYPE_F32;
        size_t emb_elem = use_fp16 ? sizeof(uint16_t) : sizeof(float);
        ctx->embed_f32 = llama_alloc_buf(prov, mem_vt, emb_out_dt,
                                          (size_t)V * D * emb_elem);
    }
    uint32_t emb_gguf_dtype = emb_ti->gguf_dtype;

    /* Per-layer weights */
    struct RawLayerWeights {
        uint32_t q_dtype, k_dtype, v_dtype, o_dtype;
        uint32_t gate_dtype, up_dtype, down_dtype;
        BufPair w_q, w_k, w_v, w_o;
        BufPair w_gate, w_up, w_down;
    };
    std::vector<RawLayerWeights> raw_layers(NL);
    ctx->layer_f32.resize(NL);
    ctx->norm_weights.resize(NL);

    for (uint32_t l = 0; l < NL; ++l) {
        auto& rl = raw_layers[l];
        char name[128];

        /* Norm weights (always F32, kept permanently) */
        std::snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        auto* ti = find_tensor(name);
        if (!ti) { return fail(); }
        ctx->norm_weights[l].attn_norm = load_weight(prov, mem_vt, *ti, NF_DTYPE_F32);

        std::snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        ti = find_tensor(name);
        if (!ti) { return fail(); }
        ctx->norm_weights[l].ffn_norm = load_weight(prov, mem_vt, *ti, NF_DTYPE_F32);

/* PLACEHOLDER_LAYER_WEIGHTS */

        /* Load quantized weights into raw buffers */
        auto load_raw = [&](const char* suffix, BufPair& out, uint32_t& dt) {
            std::snprintf(name, sizeof(name), "blk.%u.%s.weight", l, suffix);
            auto* t = find_tensor(name);
            if (!t) return false;
            dt = t->gguf_dtype;
            out = load_weight(prov, mem_vt, *t, nf_dtype_for(dt));
            raw_weight_bufs.push_back(out);
            return true;
        };

        if (!load_raw("attn_q",      rl.w_q,    rl.q_dtype))    { return fail(); }
        if (!load_raw("attn_k",      rl.w_k,    rl.k_dtype))    { return fail(); }
        if (!load_raw("attn_v",      rl.w_v,    rl.v_dtype))    { return fail(); }
        if (!load_raw("attn_output", rl.w_o,    rl.o_dtype))    { return fail(); }
        if (!load_raw("ffn_gate",    rl.w_gate, rl.gate_dtype)) { return fail(); }
        if (!load_raw("ffn_up",      rl.w_up,   rl.up_dtype))   { return fail(); }
        if (!load_raw("ffn_down",    rl.w_down, rl.down_dtype)) { return fail(); }

        /* Allocate dequant output buffers (F32 or F16 depending on use_fp16) */
        size_t elem_sz = use_fp16 ? sizeof(uint16_t) : sizeof(float);
        nf_dtype out_dtype = use_fp16 ? NF_DTYPE_F16 : NF_DTYPE_F32;
        size_t dq_dim2 = (size_t)D * D * elem_sz;
        size_t dq_kv   = (size_t)D * KV_DIM * elem_sz;
        size_t dq_ff_sz = (size_t)D * FF * elem_sz;

        auto alloc_f32_or_copy = [&](BufPair& raw, uint32_t dtype,
                                      BufPair& f32_out, size_t out_size) {
            if (is_quantized(dtype) || use_fp16) {
                /* Quantized → dequant needed; or F16 mode → need new buffer even for F32 weights */
                f32_out = llama_alloc_buf(prov, mem_vt, out_dtype, out_size);
            } else {
                /* Already F32 and not F16 mode — just alias the raw buffer */
                f32_out = raw;
            }
        };

        alloc_f32_or_copy(rl.w_q,    rl.q_dtype,    ctx->layer_f32[l].q,    dq_dim2);
        alloc_f32_or_copy(rl.w_k,    rl.k_dtype,    ctx->layer_f32[l].k,    dq_kv);
        alloc_f32_or_copy(rl.w_v,    rl.v_dtype,    ctx->layer_f32[l].v,    dq_kv);
        alloc_f32_or_copy(rl.w_o,    rl.o_dtype,    ctx->layer_f32[l].o,    dq_dim2);
        alloc_f32_or_copy(rl.w_gate, rl.gate_dtype, ctx->layer_f32[l].gate, dq_ff_sz);
        alloc_f32_or_copy(rl.w_up,   rl.up_dtype,   ctx->layer_f32[l].up,   dq_ff_sz);
        alloc_f32_or_copy(rl.w_down, rl.down_dtype, ctx->layer_f32[l].down, dq_ff_sz);
    }

    /* Final norm (F32) + LM head */
    auto* fn_ti = find_tensor("output_norm.weight");
    if (!fn_ti) { return fail(); }
    ctx->final_norm_weight = load_weight(prov, mem_vt, *fn_ti, NF_DTYPE_F32);

    auto* lm_ti = find_tensor("output.weight");
    if (!lm_ti) { return fail(); }
    BufPair w_lm_raw = load_weight(prov, mem_vt, *lm_ti, nf_dtype_for(lm_ti->gguf_dtype));
    raw_weight_bufs.push_back(w_lm_raw);
    uint32_t lm_gguf_dtype = lm_ti->gguf_dtype;
    if (is_quantized(lm_gguf_dtype) || use_fp16) {
        nf_dtype lm_out_dt = use_fp16 ? NF_DTYPE_F16 : NF_DTYPE_F32;
        size_t lm_elem = use_fp16 ? sizeof(uint16_t) : sizeof(float);
        ctx->lm_head_f32 = llama_alloc_buf(prov, mem_vt, lm_out_dt,
                                             (size_t)V * D * lm_elem);
    } else {
        ctx->lm_head_f32 = w_lm_raw;
    }

/* PLACEHOLDER_WARMUP_DAG */

    /* ---- Build warmup DAG: dequant all quantized weights in one pass ---- */
    {
        uint32_t wgid = engine.create_graph();
        bool has_dequant = false;

        auto dq_op = [&](uint32_t dtype) -> const char* {
            return use_fp16 ? nf_dequant_op_name_f16(dtype) : dequant_op_name(dtype);
        };

        /* Embed dequant */
        if (is_quantized(emb_gguf_dtype)) {
            BufPair ins[] = {w_embed_raw};
            add_node(engine, wgid, dq_op(emb_gguf_dtype),
                     ins, 1, &ctx->embed_f32, 1);
            has_dequant = true;
        } else if (use_fp16) {
            /* F32 embed in FP16 mode — need GPU dequant_q4_0_f16 won't work, just memcpy + convert on CPU */
            void *src, *dst;
            w_embed_raw.ops.map(w_embed_raw.buf, &src);
            ctx->embed_f32.ops.map(ctx->embed_f32.buf, &dst);
            const float* sf = (const float*)src;
            uint16_t* dh = (uint16_t*)dst;
            size_t n = (size_t)V * D;
            for (size_t i = 0; i < n; ++i) {
                /* F32 → F16 via bit manipulation */
                uint32_t fb; std::memcpy(&fb, &sf[i], 4);
                uint32_t sign = (fb >> 16) & 0x8000;
                int32_t exp = ((fb >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (fb >> 13) & 0x3FF;
                if (exp <= 0) dh[i] = (uint16_t)sign;
                else if (exp >= 31) dh[i] = (uint16_t)(sign | 0x7C00);
                else dh[i] = (uint16_t)(sign | (exp << 10) | mant);
            }
            ctx->embed_f32.ops.unmap(ctx->embed_f32.buf);
            w_embed_raw.ops.unmap(w_embed_raw.buf);
        } else {
            /* F32 embed — copy raw into embed_f32 */
            void *src, *dst;
            w_embed_raw.ops.map(w_embed_raw.buf, &src);
            ctx->embed_f32.ops.map(ctx->embed_f32.buf, &dst);
            std::memcpy(dst, src, (size_t)V * D * sizeof(float));
            ctx->embed_f32.ops.unmap(ctx->embed_f32.buf);
            w_embed_raw.ops.unmap(w_embed_raw.buf);
        }

        /* Per-layer weight dequant (Phase 30: skip fused-eligible weights) */
        ctx->layer_quant.resize(NL);
        for (uint32_t l = 0; l < NL; ++l) {
            auto& rl = raw_layers[l];
            auto& lf = ctx->layer_f32[l];
            auto& lq = ctx->layer_quant[l];

            auto maybe_dq = [&](BufPair& raw, uint32_t dtype,
                                BufPair& f32_out, BufPair& quant_out, nf_dtype& quant_dt) {
                if (!is_quantized(dtype)) return;
                nf_dtype nf_dt = nf_dtype_for_gguf(dtype);
                if (nf_fused_linear_op_name(nf_dt, use_fp16) != nullptr) {
                    /* Fused path: keep raw quantized weight, skip dequant */
                    quant_out = raw;
                    quant_dt = nf_dt;
                } else {
                    /* Separate path: dequant in warmup */
                    BufPair ins[] = {raw};
                    add_node(engine, wgid, dq_op(dtype), ins, 1, &f32_out, 1);
                    has_dequant = true;
                }
            };

            maybe_dq(rl.w_q,    rl.q_dtype,    lf.q,    lq.q,    lq.q_dt);
            maybe_dq(rl.w_k,    rl.k_dtype,    lf.k,    lq.k,    lq.k_dt);
            maybe_dq(rl.w_v,    rl.v_dtype,    lf.v,    lq.v,    lq.v_dt);
            maybe_dq(rl.w_o,    rl.o_dtype,    lf.o,    lq.o,    lq.o_dt);
            maybe_dq(rl.w_gate, rl.gate_dtype, lf.gate, lq.gate, lq.gate_dt);
            maybe_dq(rl.w_up,   rl.up_dtype,   lf.up,   lq.up,   lq.up_dt);
            maybe_dq(rl.w_down, rl.down_dtype, lf.down, lq.down, lq.down_dt);
        }

        /* LM head dequant */
        if (is_quantized(lm_gguf_dtype)) {
            BufPair ins[] = {w_lm_raw};
            add_node(engine, wgid, dq_op(lm_gguf_dtype),
                     ins, 1, &ctx->lm_head_f32, 1);
            has_dequant = true;
        }

        /* Run warmup DAG */
        if (has_dequant) {
            PipelineEngine::Session warmup(engine, wgid);
            nf_status st = warmup.step().get();
            if (st != NF_OK) {
                std::fprintf(stderr, "[llama_ctx] warmup dequant failed: %d\n", st);
                return fail();
            }
        }
        engine.destroy_graph(wgid);
    }

    /* Release raw quantized weight buffers — F32 copies are permanent,
       fused-aliased buffers must be kept (Phase 30) */
    for (auto& bp : raw_weight_bufs) {
        /* Don't release if it was aliased into an F32 slot (non-quantized) */
        bool aliased = (bp.buf == ctx->embed_f32.buf)
                    || (bp.buf == ctx->lm_head_f32.buf)
                    || (bp.buf == ctx->lm_head_quant.buf);
        if (!aliased) {
            for (uint32_t l = 0; l < NL && !aliased; ++l) {
                auto& lf = ctx->layer_f32[l];
                if (bp.buf == lf.q.buf || bp.buf == lf.k.buf ||
                    bp.buf == lf.v.buf || bp.buf == lf.o.buf ||
                    bp.buf == lf.gate.buf || bp.buf == lf.up.buf ||
                    bp.buf == lf.down.buf) aliased = true;
                auto& lq = ctx->layer_quant[l];
                if (bp.buf == lq.q.buf || bp.buf == lq.k.buf ||
                    bp.buf == lq.v.buf || bp.buf == lq.o.buf ||
                    bp.buf == lq.gate.buf || bp.buf == lq.up.buf ||
                    bp.buf == lq.down.buf) aliased = true;
            }
        }
        if (!aliased) llama_release(bp);
    }

/* PLACEHOLDER_ALLOC_BUFS */

    /* ---- Allocate KV cache ---- */
    nf_dtype act_dt = use_fp16 ? NF_DTYPE_F16 : NF_DTYPE_F32;
    size_t act_elem = use_fp16 ? sizeof(uint16_t) : sizeof(float);
    ctx->kv_cache.resize(NL * 2);
    size_t kv_size = (size_t)MS * KV_DIM * act_elem;
    for (uint32_t l = 0; l < NL; ++l) {
        ctx->kv_cache[l * 2]     = llama_alloc_buf(prov, mem_vt, act_dt, kv_size);
        ctx->kv_cache[l * 2 + 1] = llama_alloc_buf(prov, mem_vt, act_dt, kv_size);
        void* p;
        ctx->kv_cache[l*2].ops.map(ctx->kv_cache[l*2].buf, &p);
        std::memset(p, 0, kv_size);
        ctx->kv_cache[l*2].ops.unmap(ctx->kv_cache[l*2].buf);
        ctx->kv_cache[l*2+1].ops.map(ctx->kv_cache[l*2+1].buf, &p);
        std::memset(p, 0, kv_size);
        ctx->kv_cache[l*2+1].ops.unmap(ctx->kv_cache[l*2+1].buf);
    }

    /* ---- Allocate activation buffers (sized for max_prefill_seq) ---- */
    uint32_t S = max_prefill_seq;
    uint32_t act_size  = S * D * act_elem;
    uint32_t kv_act_sz = S * KV_DIM * act_elem;
    uint32_t ff_size   = S * FF * act_elem;

    ctx->hidden_bufs.resize(NL + 1);
    for (uint32_t i = 0; i <= NL; ++i)
        ctx->hidden_bufs[i] = llama_alloc_buf(prov, mem_vt, act_dt, act_size);

    ctx->normed     = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->q_buf      = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->k_buf      = llama_alloc_buf(prov, mem_vt, act_dt, kv_act_sz);
    ctx->v_buf      = llama_alloc_buf(prov, mem_vt, act_dt, kv_act_sz);
    ctx->q_rope_buf = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->k_rope_buf = llama_alloc_buf(prov, mem_vt, act_dt, kv_act_sz);
    ctx->attn_out   = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->proj_out   = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->resid1     = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->normed2    = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->gate_out   = llama_alloc_buf(prov, mem_vt, act_dt, ff_size);
    ctx->up_out     = llama_alloc_buf(prov, mem_vt, act_dt, ff_size);
    ctx->silu_out   = llama_alloc_buf(prov, mem_vt, act_dt, ff_size);
    ctx->mul_out    = llama_alloc_buf(prov, mem_vt, act_dt, ff_size);
    ctx->down_out   = llama_alloc_buf(prov, mem_vt, act_dt, act_size);
    ctx->final_normed = llama_alloc_buf(prov, mem_vt, act_dt, act_size);

    ctx->token_buf  = llama_alloc_buf(prov, mem_vt, NF_DTYPE_I32,
                                       S * sizeof(int32_t));
    /* Logits: match activation dtype so linear kernel writes correct type */
    ctx->logits     = llama_alloc_buf(prov, mem_vt, act_dt,
                                       S * V * act_elem);
    ctx->argmax_out = llama_alloc_buf(prov, mem_vt, NF_DTYPE_I32,
                                       S * sizeof(int32_t));

    std::printf("[llama_ctx] context created: %u layers, %s, KV cache %zu MB\n",
                NL, use_fp16 ? "FP16" : "FP32", (NL * 2 * kv_size) >> 20);
    return ctx;
}

/* PLACEHOLDER_BUILD_STEP */

/**
 * Phase 32: ModelConfig-based context creation.
 * Delegates to the 9-param overload, adding paged KV setup when requested.
 */
inline LlamaContextPtr create_llama_context(const ModelConfig& cfg)
{
    uint32_t prefill = cfg.max_prefill_seq ? cfg.max_prefill_seq : cfg.max_seq;
    auto ctx = create_llama_context(
        *cfg.engine, *cfg.model, cfg.prov, *cfg.vt, *cfg.mem_vt,
        cfg.max_seq, prefill, cfg.use_fp16, cfg.kv_cfg);

    if (ctx && cfg.use_paged_kv) {
        /* Calculate number of blocks if auto */
        uint32_t n_blocks = cfg.num_kv_blocks;
        if (n_blocks == 0) {
            uint32_t blocks_per_seq = (cfg.max_seq + cfg.kv_block_size - 1)
                                      / cfg.kv_block_size;
            n_blocks = (uint32_t)(blocks_per_seq * ctx->n_layers * 1.2f);
            if (n_blocks < 64) n_blocks = 64;
        }
        ctx->paged_kv = std::make_unique<PagedKVCache>();
        ctx->paged_kv->init(n_blocks, cfg.kv_block_size,
                            ctx->n_layers, ctx->n_kv_heads, ctx->head_dim);
        ctx->use_paged_kv = true;

        /* Allocate per-layer K/V pool GPU buffers */
        nf_dtype act_dt = cfg.use_fp16 ? NF_DTYPE_F16 : NF_DTYPE_F32;
        size_t elem_sz = cfg.use_fp16 ? sizeof(uint16_t) : sizeof(float);
        size_t pool_size = (size_t)n_blocks * ctx->n_kv_heads
                         * cfg.kv_block_size * ctx->head_dim * elem_sz;
        ctx->kv_pool_k.resize(ctx->n_layers);
        ctx->kv_pool_v.resize(ctx->n_layers);
        for (uint32_t l = 0; l < ctx->n_layers; ++l) {
            ctx->kv_pool_k[l] = llama_alloc_buf(cfg.prov, *cfg.mem_vt, act_dt, pool_size);
            ctx->kv_pool_v[l] = llama_alloc_buf(cfg.prov, *cfg.mem_vt, act_dt, pool_size);
            /* Zero-init pools */
            void* p;
            ctx->kv_pool_k[l].ops.map(ctx->kv_pool_k[l].buf, &p);
            std::memset(p, 0, pool_size); ctx->kv_pool_k[l].ops.unmap(ctx->kv_pool_k[l].buf);
            ctx->kv_pool_v[l].ops.map(ctx->kv_pool_v[l].buf, &p);
            std::memset(p, 0, pool_size); ctx->kv_pool_v[l].ops.unmap(ctx->kv_pool_v[l].buf);
        }

        /* Block table GPU buffer: max_blocks_per_seq uint32s */
        uint32_t max_bt = NF_PAGED_MAX_BLOCKS_PER_SEQ;
        ctx->block_table_buf = llama_alloc_buf(cfg.prov, *cfg.mem_vt, NF_DTYPE_I32,
                                                max_bt * sizeof(uint32_t));

        std::printf("[llama_ctx] paged KV: %u blocks × %u tokens, pool %.1f MB/layer\n",
                    n_blocks, cfg.kv_block_size, pool_size / (1024.0 * 1024.0));
    }

    return ctx;
}

/* PLACEHOLDER_BUILD_STEP_2 */

/**
 * Phase 32 Step 4: Build a draft graph using only the first N layers.
 * Used for self-speculative decoding — draft model = first N layers of full model.
 * Returns a StepGraph with only draft_layers layers (no final norm/LM head).
 */
inline StepGraph build_speculative_draft_graph(
    LlamaContext& ctx, uint32_t seq_len, uint32_t draft_layers)
{
    PipelineEngine& engine = *ctx.engine;
    StepGraph sg{};
    sg.gid = engine.create_graph();
    uint32_t gid = sg.gid;

    const char* norm_op = nf_resolve_norm_op(ctx.arch);
    const char* rope_op = nf_resolve_rope_op(ctx.arch, ctx.use_fp16);
    const char* attn_op = ctx.use_paged_kv
        ? nf_resolve_paged_attn_op(ctx.arch)
        : nf_resolve_attn_op(ctx.arch);
    const char* ffn_act_op = nf_resolve_ffn_op(ctx.arch);

    uint32_t NL = draft_layers;
    if (NL > ctx.n_layers) NL = ctx.n_layers;

    (void)seq_len;

    /* Embedding */
    { BufPair ins[] = {ctx.embed_f32, ctx.token_buf};
      sg.embed_id = add_node(engine, gid, "embedding_lookup",
                              ins, 2, &ctx.hidden_bufs[0], 1); }

    uint32_t prev_last = sg.embed_id;
    sg.layer_ids.resize(NL);

    for (uint32_t l = 0; l < NL; ++l) {
        auto& lid = sg.layer_ids[l];
        auto& lf = ctx.layer_f32[l];
        auto& lq = ctx.layer_quant[l];
        auto& nw = ctx.norm_weights[l];

        lid.dq_q = lid.dq_k = lid.dq_v = lid.dq_o = UINT32_MAX;
        lid.dq_gate = lid.dq_up = lid.dq_down = UINT32_MAX;

        { BufPair ins[] = {ctx.hidden_bufs[l], nw.attn_norm};
          lid.attn_norm = add_node(engine, gid, norm_op, ins, 2,
                                   &ctx.normed, 1); }
        engine.add_edge(gid, prev_last, lid.attn_norm);

        auto fused_or_linear = [&](BufPair& activation, BufPair& f32_weight,
                                    BufPair& quant_weight, nf_dtype quant_dt,
                                    BufPair& output) -> uint32_t {
            const char* fused_op = nf_fused_linear_op_name(quant_dt, ctx.use_fp16);
            if (fused_op && quant_weight.buf) {
                BufPair ins[] = {activation, quant_weight};
                return add_node(engine, gid, fused_op, ins, 2, &output, 1);
            } else {
                BufPair ins[] = {activation, f32_weight};
                return add_node(engine, gid, "linear", ins, 2, &output, 1);
            }
        };

        lid.q_lin = fused_or_linear(ctx.normed, lf.q, lq.q, lq.q_dt, ctx.q_buf);
        lid.k_lin = fused_or_linear(ctx.normed, lf.k, lq.k, lq.k_dt, ctx.k_buf);
        lid.v_lin = fused_or_linear(ctx.normed, lf.v, lq.v, lq.v_dt, ctx.v_buf);
        engine.add_edge(gid, lid.attn_norm, lid.q_lin);
        engine.add_edge(gid, lid.attn_norm, lid.k_lin);
        engine.add_edge(gid, lid.attn_norm, lid.v_lin);

        { BufPair ins[] = {ctx.q_buf};
          lid.q_rope = add_node(engine, gid, rope_op, ins, 1, &ctx.q_rope_buf, 1); }
        { BufPair ins[] = {ctx.k_buf};
          lid.k_rope = add_node(engine, gid, rope_op, ins, 1, &ctx.k_rope_buf, 1); }
        engine.add_edge(gid, lid.q_lin, lid.q_rope);
        engine.add_edge(gid, lid.k_lin, lid.k_rope);

        if (ctx.use_paged_kv) {
            BufPair ins[] = {ctx.q_rope_buf, ctx.kv_pool_k[l], ctx.kv_pool_v[l],
                             ctx.block_table_buf};
            lid.cached_attn = add_node(engine, gid, attn_op, ins, 4, &ctx.attn_out, 1);
        } else {
            BufPair ins[] = {ctx.q_rope_buf, ctx.k_rope_buf, ctx.v_buf,
                             ctx.kv_cache[l*2], ctx.kv_cache[l*2+1]};
            lid.cached_attn = add_node(engine, gid, attn_op, ins, 5, &ctx.attn_out, 1);
        }
        engine.add_edge(gid, lid.q_rope, lid.cached_attn);
        engine.add_edge(gid, lid.k_rope, lid.cached_attn);
        engine.add_edge(gid, lid.v_lin, lid.cached_attn);

        lid.o_lin = fused_or_linear(ctx.attn_out, lf.o, lq.o, lq.o_dt, ctx.proj_out);
        engine.add_edge(gid, lid.cached_attn, lid.o_lin);

        { BufPair ins[] = {ctx.hidden_bufs[l], ctx.proj_out};
          lid.resid_add1 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &ctx.resid1, 1); }
        engine.add_edge(gid, lid.o_lin, lid.resid_add1);
        engine.add_edge(gid, prev_last, lid.resid_add1);

        { BufPair ins[] = {ctx.resid1, nw.ffn_norm};
          lid.ffn_norm = add_node(engine, gid, norm_op, ins, 2, &ctx.normed2, 1); }
        engine.add_edge(gid, lid.resid_add1, lid.ffn_norm);

        lid.gate_lin = fused_or_linear(ctx.normed2, lf.gate, lq.gate, lq.gate_dt, ctx.gate_out);
        lid.up_lin   = fused_or_linear(ctx.normed2, lf.up,   lq.up,   lq.up_dt,   ctx.up_out);
        engine.add_edge(gid, lid.ffn_norm, lid.gate_lin);
        engine.add_edge(gid, lid.ffn_norm, lid.up_lin);

        { BufPair ins[] = {ctx.gate_out};
          lid.silu_node = add_node(engine, gid, ffn_act_op, ins, 1, &ctx.silu_out, 1); }
        engine.add_edge(gid, lid.gate_lin, lid.silu_node);

        { BufPair ins[] = {ctx.silu_out, ctx.up_out};
          lid.elem_mul = add_node(engine, gid, "elementwise_mul", ins, 2, &ctx.mul_out, 1); }
        engine.add_edge(gid, lid.silu_node, lid.elem_mul);
        engine.add_edge(gid, lid.up_lin, lid.elem_mul);

        lid.down_lin = fused_or_linear(ctx.mul_out, lf.down, lq.down, lq.down_dt, ctx.down_out);
        engine.add_edge(gid, lid.elem_mul, lid.down_lin);

        { BufPair ins[] = {ctx.resid1, ctx.down_out};
          lid.resid_add2 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &ctx.hidden_bufs[l+1], 1); }
        engine.add_edge(gid, lid.down_lin, lid.resid_add2);
        engine.add_edge(gid, lid.resid_add1, lid.resid_add2);

        prev_last = lid.resid_add2;
    }

    /* Draft graph: add final norm + LM head + argmax using output of last draft layer */
    { BufPair ins[] = {ctx.hidden_bufs[NL], ctx.final_norm_weight};
      sg.final_norm_id = add_node(engine, gid, norm_op, ins, 2,
                                   &ctx.final_normed, 1); }
    engine.add_edge(gid, prev_last, sg.final_norm_id);

    { BufPair ins[] = {ctx.final_normed, ctx.lm_head_f32};
      sg.lm_head_id = add_node(engine, gid, "linear", ins, 2,
                                &ctx.logits, 1); }
    engine.add_edge(gid, sg.final_norm_id, sg.lm_head_id);

    { BufPair ins[] = {ctx.logits};
      sg.argmax_id = add_node(engine, gid, "argmax", ins, 1,
                               &ctx.argmax_out, 1); }
    engine.add_edge(gid, sg.lm_head_id, sg.argmax_id);

    return sg;
}

/**
 * Build a lightweight per-step graph using persistent F32 weights from ctx.
 * No weight loading, no dequant nodes. ~175 nodes vs ~330 with dequant.
 */
inline StepGraph build_llama_step_graph(LlamaContext& ctx, uint32_t seq_len)
{
    PipelineEngine& engine = *ctx.engine;
    StepGraph sg{};
    sg.gid = engine.create_graph();
    uint32_t gid = sg.gid;

    const uint32_t D = ctx.dim;
    const uint32_t HD = ctx.head_dim;
    const uint32_t NKV = ctx.n_kv_heads;
    const uint32_t KV_DIM = NKV * HD;
    const uint32_t FF = ctx.ff_dim;
    const uint32_t V = ctx.vocab_size;
    const uint32_t NL = ctx.n_layers;

    (void)seq_len; /* sizes come from push constants, buffers pre-allocated */

    /* Phase 31: resolve op names from architecture strategy */
    const char* norm_op = nf_resolve_norm_op(ctx.arch);
    const char* rope_op = nf_resolve_rope_op(ctx.arch, ctx.use_fp16);
    const char* attn_op = ctx.use_paged_kv
        ? nf_resolve_paged_attn_op(ctx.arch)
        : nf_resolve_attn_op(ctx.arch);
    const char* ffn_act_op = nf_resolve_ffn_op(ctx.arch);

    /* Embedding: embed_f32 + token_buf → hidden[0] */
    { BufPair ins[] = {ctx.embed_f32, ctx.token_buf};
      sg.embed_id = add_node(engine, gid, "embedding_lookup",
                              ins, 2, &ctx.hidden_bufs[0], 1); }

    uint32_t prev_last = sg.embed_id;
    sg.layer_ids.resize(NL);

    for (uint32_t l = 0; l < NL; ++l) {
        auto& lid = sg.layer_ids[l];
        auto& lf = ctx.layer_f32[l];
        auto& lq = ctx.layer_quant[l];
        auto& nw = ctx.norm_weights[l];

        /* Set dq_* fields to UINT32_MAX (unused — weights are pre-dequantized) */
        lid.dq_q = lid.dq_k = lid.dq_v = lid.dq_o = UINT32_MAX;
        lid.dq_gate = lid.dq_up = lid.dq_down = UINT32_MAX;

        /* attn_norm: hidden[l] → normed */
        { BufPair ins[] = {ctx.hidden_bufs[l], nw.attn_norm};
          lid.attn_norm = add_node(engine, gid, norm_op, ins, 2,
                                   &ctx.normed, 1); }
        engine.add_edge(gid, prev_last, lid.attn_norm);

        /* Phase 30: fused-aware linear helper */
        auto fused_or_linear = [&](BufPair& activation, BufPair& f32_weight,
                                    BufPair& quant_weight, nf_dtype quant_dt,
                                    BufPair& output) -> uint32_t {
            const char* fused_op = nf_fused_linear_op_name(quant_dt, ctx.use_fp16);
            if (fused_op && quant_weight.buf) {
                BufPair ins[] = {activation, quant_weight};
                return add_node(engine, gid, fused_op, ins, 2, &output, 1);
            } else {
                BufPair ins[] = {activation, f32_weight};
                return add_node(engine, gid, "linear", ins, 2, &output, 1);
            }
        };

        /* Q/K/V linear (parallel) — fused-aware */
        lid.q_lin = fused_or_linear(ctx.normed, lf.q, lq.q, lq.q_dt, ctx.q_buf);
        lid.k_lin = fused_or_linear(ctx.normed, lf.k, lq.k, lq.k_dt, ctx.k_buf);
        lid.v_lin = fused_or_linear(ctx.normed, lf.v, lq.v, lq.v_dt, ctx.v_buf);
        engine.add_edge(gid, lid.attn_norm, lid.q_lin);
        engine.add_edge(gid, lid.attn_norm, lid.k_lin);
        engine.add_edge(gid, lid.attn_norm, lid.v_lin);

/* PLACEHOLDER_STEP_LAYER_CONT */

        /* RoPE on Q and K */
        { BufPair ins[] = {ctx.q_buf};
          lid.q_rope = add_node(engine, gid, rope_op, ins, 1,
                                &ctx.q_rope_buf, 1); }
        { BufPair ins[] = {ctx.k_buf};
          lid.k_rope = add_node(engine, gid, rope_op, ins, 1,
                                &ctx.k_rope_buf, 1); }
        engine.add_edge(gid, lid.q_lin, lid.q_rope);
        engine.add_edge(gid, lid.k_lin, lid.k_rope);

        /* causal_attention_cached or flash_attention_paged */
        if (ctx.use_paged_kv) {
            /* Paged path: Q, K_pool, V_pool, block_table → attn_out */
            BufPair ins[] = {ctx.q_rope_buf, ctx.kv_pool_k[l], ctx.kv_pool_v[l],
                             ctx.block_table_buf};
            lid.cached_attn = add_node(engine, gid, attn_op,
                                       ins, 4, &ctx.attn_out, 1);
        } else {
            /* Legacy contiguous path */
            BufPair ins[] = {ctx.q_rope_buf, ctx.k_rope_buf, ctx.v_buf,
                             ctx.kv_cache[l*2], ctx.kv_cache[l*2+1]};
            lid.cached_attn = add_node(engine, gid, attn_op,
                                       ins, 5, &ctx.attn_out, 1);
        }
        engine.add_edge(gid, lid.q_rope, lid.cached_attn);
        engine.add_edge(gid, lid.k_rope, lid.cached_attn);
        engine.add_edge(gid, lid.v_lin, lid.cached_attn);

        /* Output linear — fused-aware */
        lid.o_lin = fused_or_linear(ctx.attn_out, lf.o, lq.o, lq.o_dt, ctx.proj_out);
        engine.add_edge(gid, lid.cached_attn, lid.o_lin);

        /* Residual add: hidden[l] + proj_out → resid1 */
        { BufPair ins[] = {ctx.hidden_bufs[l], ctx.proj_out};
          lid.resid_add1 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &ctx.resid1, 1); }
        engine.add_edge(gid, lid.o_lin, lid.resid_add1);
        engine.add_edge(gid, prev_last, lid.resid_add1);

        /* FFN norm: resid1 → normed2 */
        { BufPair ins[] = {ctx.resid1, nw.ffn_norm};
          lid.ffn_norm = add_node(engine, gid, norm_op, ins, 2,
                                  &ctx.normed2, 1); }
        engine.add_edge(gid, lid.resid_add1, lid.ffn_norm);

        /* Gate/Up linear (parallel) — fused-aware */
        lid.gate_lin = fused_or_linear(ctx.normed2, lf.gate, lq.gate, lq.gate_dt, ctx.gate_out);
        lid.up_lin   = fused_or_linear(ctx.normed2, lf.up,   lq.up,   lq.up_dt,   ctx.up_out);
        engine.add_edge(gid, lid.ffn_norm, lid.gate_lin);
        engine.add_edge(gid, lid.ffn_norm, lid.up_lin);

        /* SiLU on gate output */
        { BufPair ins[] = {ctx.gate_out};
          lid.silu_node = add_node(engine, gid, ffn_act_op, ins, 1,
                                   &ctx.silu_out, 1); }
        engine.add_edge(gid, lid.gate_lin, lid.silu_node);

        /* elementwise_mul: silu_out × up_out → mul_out */
        { BufPair ins[] = {ctx.silu_out, ctx.up_out};
          lid.elem_mul = add_node(engine, gid, "elementwise_mul", ins, 2,
                                  &ctx.mul_out, 1); }
        engine.add_edge(gid, lid.silu_node, lid.elem_mul);
        engine.add_edge(gid, lid.up_lin, lid.elem_mul);

        /* Down linear: mul_out × w_down → down_out — fused-aware */
        lid.down_lin = fused_or_linear(ctx.mul_out, lf.down, lq.down, lq.down_dt, ctx.down_out);
        engine.add_edge(gid, lid.elem_mul, lid.down_lin);

        /* Residual add: resid1 + down_out → hidden[l+1] */
        { BufPair ins[] = {ctx.resid1, ctx.down_out};
          lid.resid_add2 = add_node(engine, gid, "metal_vector_add", ins, 2,
                                    &ctx.hidden_bufs[l+1], 1); }
        engine.add_edge(gid, lid.down_lin, lid.resid_add2);
        engine.add_edge(gid, lid.resid_add1, lid.resid_add2);

        prev_last = lid.resid_add2;
    } /* end layer loop */

/* PLACEHOLDER_STEP_TAIL */

    /* Final norm: hidden[NL] → final_normed */
    { BufPair ins[] = {ctx.hidden_bufs[NL], ctx.final_norm_weight};
      sg.final_norm_id = add_node(engine, gid, norm_op, ins, 2,
                                   &ctx.final_normed, 1); }
    engine.add_edge(gid, prev_last, sg.final_norm_id);

    /* LM head: final_normed × lm_head_f32 → logits */
    { BufPair ins[] = {ctx.final_normed, ctx.lm_head_f32};
      sg.lm_head_id = add_node(engine, gid, "linear", ins, 2,
                                &ctx.logits, 1); }
    engine.add_edge(gid, sg.final_norm_id, sg.lm_head_id);

    /* Argmax: logits → argmax_out */
    { BufPair ins[] = {ctx.logits};
      sg.argmax_id = add_node(engine, gid, "argmax", ins, 1,
                               &ctx.argmax_out, 1); }
    engine.add_edge(gid, sg.lm_head_id, sg.argmax_id);

    return sg;
}

/**
 * Inject push constants for a step graph. Same logic as inject_llama_push_constants
 * but operates on LlamaContext + StepGraph (no dequant nodes to configure).
 */
inline void inject_step_push_constants(
    LlamaContext& ctx, StepGraph& sg, PipelineEngine::Session& sess,
    uint32_t seq_len, uint32_t step_idx)
{
    const uint32_t D  = ctx.dim;
    const uint32_t NH = ctx.n_heads;
    const uint32_t NKV = ctx.n_kv_heads;
    const uint32_t HD = ctx.head_dim;
    const uint32_t KV_DIM = NKV * HD;
    const uint32_t FF = ctx.ff_dim;
    const uint32_t V  = ctx.vocab_size;
    const uint32_t MS = ctx.max_seq;

    /* Embed */
    { PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = D;
      sess.set_push_constants_by_id(sg.embed_id, &pc, sizeof(pc)); }

    for (uint32_t l = 0; l < ctx.n_layers; ++l) {
        auto& lid = sg.layer_ids[l];

        PushConstants pc{};
        pc.seq_len = seq_len; pc.n_heads = NH; pc.head_dim = D;
        pc.epsilon = ctx.rms_norm_eps; pc.theta = ctx.rope_theta;
        pc.step_idx = step_idx; pc.max_seq_len = MS;
        sess.set_push_constants_by_id(lid.attn_norm, &pc, sizeof(pc));

        /* Q linear: M=seq_len, N=D, K=D */
        pc.M = seq_len; pc.N = D; pc.K = D;
        sess.set_push_constants_by_id(lid.q_lin, &pc, sizeof(pc));

        /* K/V linear: M=seq_len, N=KV_DIM, K=D */
        { PushConstants kv_pc = pc;
          kv_pc.M = seq_len; kv_pc.N = KV_DIM; kv_pc.K = D;
          sess.set_push_constants_by_id(lid.k_lin, &kv_pc, sizeof(kv_pc));
          sess.set_push_constants_by_id(lid.v_lin, &kv_pc, sizeof(kv_pc)); }

        /* RoPE Q */
        { PushConstants rpc{};
          rpc.seq_len = seq_len; rpc.n_heads = NH; rpc.head_dim = HD;
          rpc.theta = ctx.rope_theta; rpc.step_idx = step_idx;
          sess.set_push_constants_by_id(lid.q_rope, &rpc, sizeof(rpc)); }

        /* RoPE K */
        { PushConstants rpc{};
          rpc.seq_len = seq_len; rpc.n_heads = NKV; rpc.head_dim = HD;
          rpc.theta = ctx.rope_theta; rpc.step_idx = step_idx;
          sess.set_push_constants_by_id(lid.k_rope, &rpc, sizeof(rpc)); }

        /* Cached attention: M=n_kv_heads for GQA */
        { PushConstants apc{};
          apc.seq_len = seq_len; apc.n_heads = NH; apc.head_dim = HD;
          apc.M = NKV; apc.step_idx = step_idx; apc.max_seq_len = MS;
          if (ctx.use_paged_kv && ctx.paged_kv) {
              /* Paged: window_size=block_size, N=num_blocks, K=seq_offset(0) */
              apc.window_size = ctx.paged_kv->block_size;
              uint32_t seq_idx = 0;  /* single-sequence for now */
              apc.N = ctx.paged_kv->sequences[seq_idx].num_logical_blocks;
              apc.K = 0;  /* seq_offset for batched queries */
          } else {
              apc.window_size = ctx.kv_policy.config.window_size;
          }
          sess.set_push_constants_by_id(lid.cached_attn, &apc, sizeof(apc)); }

        /* Output linear: M=seq_len, N=D, K=D */
        pc.M = seq_len; pc.N = D; pc.K = D;
        sess.set_push_constants_by_id(lid.o_lin, &pc, sizeof(pc));

        /* FFN norm */
        sess.set_push_constants_by_id(lid.ffn_norm, &pc, sizeof(pc));

        /* Gate/Up linear: M=seq_len, N=FF, K=D */
        pc.M = seq_len; pc.N = FF; pc.K = D;
        sess.set_push_constants_by_id(lid.gate_lin, &pc, sizeof(pc));
        sess.set_push_constants_by_id(lid.up_lin, &pc, sizeof(pc));

        /* Down linear: M=seq_len, N=D, K=FF */
        pc.M = seq_len; pc.N = D; pc.K = FF;
        sess.set_push_constants_by_id(lid.down_lin, &pc, sizeof(pc));
    }

    /* Final norm */
    { PushConstants pc{}; pc.seq_len = seq_len; pc.head_dim = D;
      pc.epsilon = ctx.rms_norm_eps; pc.n_heads = NH;
      sess.set_push_constants_by_id(sg.final_norm_id, &pc, sizeof(pc)); }

    /* LM head: M=seq_len, N=vocab, K=D */
    { PushConstants pc{}; pc.M = seq_len; pc.N = V; pc.K = D;
      sess.set_push_constants_by_id(sg.lm_head_id, &pc, sizeof(pc)); }

    /* Argmax */
    { PushConstants pc{}; pc.seq_len = seq_len; pc.N = V;
      sess.set_push_constants_by_id(sg.argmax_id, &pc, sizeof(pc)); }
}

/**
 * Release all buffers owned by the context.
 */
inline void release_llama_context(LlamaContext* ctx) {
    if (!ctx) return;

    /* F32 weight buffers */
    llama_release(ctx->embed_f32);
    for (auto& lf : ctx->layer_f32) {
        llama_release(lf.q);  llama_release(lf.k);
        llama_release(lf.v);  llama_release(lf.o);
        llama_release(lf.gate); llama_release(lf.up);
        llama_release(lf.down);
    }
    llama_release(ctx->lm_head_f32);

    /* Fused quantized weight buffers (Phase 30) */
    for (auto& lq : ctx->layer_quant) {
        llama_release(lq.q);  llama_release(lq.k);
        llama_release(lq.v);  llama_release(lq.o);
        llama_release(lq.gate); llama_release(lq.up);
        llama_release(lq.down);
    }
    llama_release(ctx->lm_head_quant);

    /* Norm weights */
    for (auto& nw : ctx->norm_weights) {
        llama_release(nw.attn_norm);
        llama_release(nw.ffn_norm);
    }
    llama_release(ctx->final_norm_weight);

    /* KV cache */
    for (auto& bp : ctx->kv_cache) llama_release(bp);

    /* Phase 32: Paged KV pool buffers */
    for (auto& bp : ctx->kv_pool_k) llama_release(bp);
    for (auto& bp : ctx->kv_pool_v) llama_release(bp);
    llama_release(ctx->block_table_buf);

    /* Activation buffers */
    for (auto& bp : ctx->hidden_bufs) llama_release(bp);
    llama_release(ctx->normed);     llama_release(ctx->q_buf);
    llama_release(ctx->k_buf);      llama_release(ctx->v_buf);
    llama_release(ctx->q_rope_buf); llama_release(ctx->k_rope_buf);
    llama_release(ctx->attn_out);   llama_release(ctx->proj_out);
    llama_release(ctx->resid1);     llama_release(ctx->normed2);
    llama_release(ctx->gate_out);   llama_release(ctx->up_out);
    llama_release(ctx->silu_out);   llama_release(ctx->mul_out);
    llama_release(ctx->down_out);   llama_release(ctx->final_normed);

    /* I/O buffers */
    llama_release(ctx->token_buf);
    llama_release(ctx->logits);
    llama_release(ctx->argmax_out);

    delete ctx;
}

} /* namespace nf */

#endif /* NF_LLAMA_DAG_BUILDER_HPP */
