#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, 0);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, 0);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, 0);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, 0);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, 0);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, 0);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, 0);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, 0);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, 0);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, 0);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, 0);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, 0);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, 0);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, 0);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, 0);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

bool llama_model_qwen4exp::graph::hc_prescaled(const ggml_tensor * w) const {
    const bool folded = static_cast<const llama_model_qwen4exp &>(model).hc_prescaled_set.count(w) > 0;
    if (folded && loras) {
        for (const auto & [adapter, scale] : *loras) {
            if (scale != 0.0f && adapter->get_weight(const_cast<ggml_tensor *>(w))) {
                GGML_ABORT("LLAMA_QWEN4EXP_HC_FOLD=1 cannot be combined with a LoRA adapter on the hc weights");
            }
        }
    }
    return folded;
}

// Divides every scale of the tensor by 4 in place. Returns false, leaving the tensor untouched,
// if any scale would round, which can only happen when it lands in the subnormal range.
static bool qwen4exp_div4_exact(ggml_tensor * t) {
    if (!t || !t->data || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    if (t->type == GGML_TYPE_Q8_0) {
        const size_t bs = ggml_type_size(GGML_TYPE_Q8_0);
        for (size_t o = 0; o < buf.size(); o += bs) {
            ggml_fp16_t h;
            memcpy(&h, &buf[o], sizeof(h));
            const float q = ggml_fp16_to_fp32(h) * 0.25f;
            const ggml_fp16_t hq = ggml_fp32_to_fp16(q);
            // a matmul that dequantizes to F16 must not see the products leave the normal range
            if (ggml_fp16_to_fp32(hq) != q || (q != 0.0f && fabsf(q) < 6.103515625e-05f)) {
                return false;
            }
            memcpy(&buf[o], &hq, sizeof(hq));
        }
    } else if (t->type == GGML_TYPE_F32) {
        for (size_t o = 0; o < buf.size(); o += sizeof(float)) {
            float f;
            memcpy(&f, &buf[o], sizeof(f));
            const float q = f * 0.25f;
            if (q * 4.0f != f) {
                return false;
            }
            memcpy(&buf[o], &q, sizeof(q));
        }
    } else {
        return false;
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
    return true;
}

void llama_model_qwen4exp::hc_fold_impl() const {
    const char * e = getenv("LLAMA_QWEN4EXP_HC_FOLD");
    if (!e || !atoi(e) || hparams.dsv4_hc_mult != 4) {
        return;
    }
    std::vector<ggml_tensor *> ws = { hc_head_down };
    for (const auto & l : layers) {
        ws.insert(ws.end(), { l.hc_attn_down, l.hc_ffn_down, l.hc_attn_inject, l.hc_ffn_inject });
    }
    for (ggml_tensor * w : ws) {
        if (qwen4exp_div4_exact(w)) {
            hc_prescaled_set.insert(w);
        }
    }
    LLAMA_LOG_INFO("%s: folded 1/hc into %zu of %zu hc weights\n", __func__, hc_prescaled_set.size(), ws.size());
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    if (!hc_prescaled(w_down)) {
        lo = ggml_scale(ctx0, lo, 1.0f / (float) hc);
    }
    lo = ggml_silu(ctx0, lo);
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        for (int64_t c = 0; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = mixed ? ggml_add(ctx0, mixed, s) : s;
        }
        if (hc == 1) {
            mixed = ggml_cont(ctx0, mixed);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
  const ggml_tensor * w_inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, hc_prescaled(w_inject) ? inject : ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

struct llama_model_qwen4exp::moe_cache {
    ~moe_cache() { ggml_backend_sched_clear_staging_area(buf.get()); }

    int64_t n_slot   = 0;
    int64_t n_used   = 0;
    int64_t n_layer  = 0;
    int64_t n_expert = 0;
    int64_t interval = 32;
    int64_t window   = 1024;

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    std::vector<ggml_tensor *> up, gate, down, map_gpu, map_cpu;
    ggml_tensor * hist = nullptr;

    std::vector<std::vector<int32_t>> slot_of; // [n_layer][n_expert] -> slot or -1
    std::vector<std::vector<int32_t>> in_slot; // [n_layer][n_slot]   -> expert or -1
    std::vector<std::vector<int32_t>> counts;  // [n_layer][n_expert] over the window
    std::vector<int32_t> ring;                 // [window][n_layer][n_used]
    std::vector<int32_t> hist_host;

    int64_t n_rec = 0;
    int64_t since = 0;
    bool    hist_pending = false;
    int64_t warm      = 0;
    int64_t warm_left = 0;

    // the buffer doubles as the scheduler's staging area while no graph uses the cache
    uint64_t stage_epoch = 0;

    void reclaim(size_t extent) {
        const char * base = (const char *) ggml_backend_buffer_get_base(buf.get());
        auto clobbered = [&](const ggml_tensor * t) { return (size_t) ((const char *) t->data - base) < extent; };
        for (int64_t il = 0; il < n_layer; ++il) {
            if (clobbered(up[il]) || clobbered(gate[il]) || clobbered(down[il])) {
                std::fill(slot_of[il].begin(), slot_of[il].end(), -1);
                std::fill(in_slot[il].begin(), in_slot[il].end(), -1);
                for (ggml_tensor * t : { up[il], gate[il], down[il] }) {
                    ggml_backend_tensor_memset(t, 0, n_slot*t->nb[2], t->nb[2]);
                }
                upload_maps(il);
            } else if (clobbered(map_gpu[il]) || clobbered(map_cpu[il])) {
                upload_maps(il);
            }
        }
        since = interval;
    }

    void upload_maps(int64_t il) {
        std::vector<int32_t> g(n_expert), c(n_expert);
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t s = slot_of[il][e];
            g[e] = s >= 0 ? s : (int32_t) n_slot;
            c[e] = s >= 0 ? -1 : (int32_t) e;
        }
        ggml_backend_tensor_set(map_gpu[il], g.data(), 0, ggml_nbytes(map_gpu[il]));
        ggml_backend_tensor_set(map_cpu[il], c.data(), 0, ggml_nbytes(map_cpu[il]));
    }
};

// A miss maps to the shared zero slot, so ids repeat within a token, and the CPU branch carries -1.
// Only MMVQ tolerates the former and nothing on the device the latter, so stay below both the MMVQ
// ceiling (LLAMA_MMVQ_MMID_MAX) and the size at which CPU weights get offloaded.
static int64_t moe_cache_max_tok() {
    static const int64_t v = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_MAXTOK");
        const char * o = getenv("GGML_OP_OFFLOAD_MIN_BATCH");
        const char * m = getenv("LLAMA_MMVQ_MMID_MAX");
        const int64_t x   = e ? atoll(e) : 8;
        const int64_t off = o ? atoll(o) : 32;
        // 4 is the smallest built-in ceiling in any of the per-type tables, and the host cannot
        // read the one this device uses
        const int64_t mmq = m ? std::min<int64_t>(atoll(m), 32) : 4;
        const int64_t v   = std::min(std::min(x, mmq), off - 1);
        return v > 0 ? v : 0;
    }();
    return v;
}

llama_model_qwen4exp::moe_cache * llama_model_qwen4exp::moe_cache_get() const {
    if (mcache_tried) {
        return mcache.get();
    }
    mcache_tried = true;

    const char * e = getenv("LLAMA_MOE_CACHE");
    const int64_t n_slot = e ? atoll(e) : 0;
    if (n_slot < hparams.n_expert_used(0) || hparams.no_alloc || moe_cache_max_tok() <= 0 ||
            hparams.n_expert_groups > 1) {
        return nullptr;
    }

    const int64_t n_layer = hparams.n_layer();
    for (int64_t il = 0; il < n_layer; ++il) {
        const auto & l = layers[il];
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || l.ffn_gate_up_exps ||
                l.ffn_up_exps_s || l.ffn_gate_exps_s || l.ffn_down_exps_s ||
                !l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) ||
                !ggml_backend_buffer_is_host(l.ffn_gate_exps->buffer) || !ggml_backend_buffer_is_host(l.ffn_down_exps->buffer) ||
                hparams.swiglu_clamp_exp[il] > 1e-6f) {
            LLAMA_LOG_WARN("%s: expert cache disabled: layer %lld experts are not all host-side\n", __func__, (long long) il);
            return nullptr;
        }
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        return nullptr;
    }

    auto mc = std::make_shared<moe_cache>();
    mc->n_slot   = n_slot;
    mc->n_used   = hparams.n_expert_used(0);
    mc->n_layer  = n_layer;
    mc->n_expert = hparams.n_expert;
    if (const char * v = getenv("LLAMA_MOE_CACHE_INTERVAL")) { mc->interval = std::max<int64_t>(1, atoll(v)); }
    if (const char * v = getenv("LLAMA_MOE_CACHE_WINDOW"))   { mc->window   = std::max<int64_t>(1, atoll(v)); }

    ggml_init_params params = {
        /*.mem_size   =*/ size_t((8*n_layer + 4)*ggml_tensor_overhead()),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    mc->ctx.reset(ggml_init(params));

    auto slots_like = [&](const ggml_tensor * w) {
        return ggml_new_tensor_3d(mc->ctx.get(), w->type, w->ne[0], w->ne[1], n_slot + 1);
    };
    for (int64_t il = 0; il < n_layer; ++il) {
        mc->up     .push_back(slots_like(layers[il].ffn_up_exps));
        mc->gate   .push_back(slots_like(layers[il].ffn_gate_exps));
        mc->down   .push_back(slots_like(layers[il].ffn_down_exps));
        mc->map_gpu.push_back(ggml_new_tensor_2d(mc->ctx.get(), GGML_TYPE_I32, 1, mc->n_expert));
        mc->map_cpu.push_back(ggml_new_tensor_2d(mc->ctx.get(), GGML_TYPE_I32, 1, mc->n_expert));
    }
    mc->hist = ggml_new_tensor_2d(mc->ctx.get(), GGML_TYPE_I32, mc->n_used, n_layer);
    {
        const char * e = getenv("LLAMA_MOE_WARM");
        mc->warm = e ? std::max<int64_t>(0, atoll(e)) : 32;
    }

    mc->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mc->ctx.get(), ggml_backend_dev_buffer_type(dev)));
    if (!mc->buf) {
        LLAMA_LOG_WARN("%s: expert cache disabled: allocation failed\n", __func__);
        return nullptr;
    }
    ggml_backend_buffer_clear(mc->buf.get(), 0);
    ggml_backend_sched_set_staging_area(mc->buf.get(), ggml_backend_buffer_get_size(mc->buf.get()));
    mc->stage_epoch = ggml_backend_sched_staging_epoch(nullptr);
    LLAMA_LOG_INFO("%s: expert cache: %lld slots per layer, %.2f MiB\n", __func__, (long long) n_slot,
            ggml_backend_buffer_get_size(mc->buf.get())/1024.0/1024.0);

    mc->slot_of.assign(n_layer, std::vector<int32_t>(mc->n_expert, -1));
    mc->in_slot.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    mc->counts .assign(n_layer, std::vector<int32_t>(mc->n_expert, 0));
    mc->ring   .assign(mc->window*n_layer*mc->n_used, -1);
    mc->hist_host.assign(mc->n_used*n_layer, 0);
    for (int64_t il = 0; il < n_layer; ++il) {
        mc->upload_maps(il);
    }

    mcache = mc;
    return mcache.get();
}

class llama_model_qwen4exp::llm_graph_input_moe_cache : public llm_graph_input_i {
public:
    llm_graph_input_moe_cache(const llama_model_qwen4exp & model, bool active) : model(model), active(active) {}
    virtual ~llm_graph_input_moe_cache() = default;

    void set_input(const llama_ubatch * ubatch) override {
        moe_cache * mc = model.mcache.get();
        if (!mc) {
            return;
        }
        const int64_t L = mc->n_layer, k = mc->n_used;

        size_t extent = 0;
        const uint64_t epoch = ggml_backend_sched_staging_epoch(&extent);
        if (epoch != mc->stage_epoch) {
            ggml_backend_sched_staging_synchronize();
            mc->stage_epoch = epoch;
            mc->reclaim(extent);
        }

        if (mc->hist_pending) {
            ggml_backend_tensor_get(mc->hist, mc->hist_host.data(), 0, ggml_nbytes(mc->hist));
            const int64_t slot = mc->n_rec % mc->window;
            for (int64_t il = 0; il < L; ++il) {
                int32_t * r = &mc->ring[(slot*L + il)*k];
                for (int64_t j = 0; j < k; ++j) {
                    if (r[j] >= 0) {
                        mc->counts[il][r[j]]--;
                    }
                    const int32_t e = mc->hist_host[il*k + j];
                    r[j] = (e >= 0 && e < mc->n_expert) ? e : -1;
                    if (r[j] >= 0) {
                        mc->counts[il][r[j]]++;
                    }
                }
            }
            mc->n_rec++;
            mc->since++;
            mc->hist_pending = false;
        }

        const int64_t interval = mc->warm_left > 0 ? std::min<int64_t>(8, mc->interval) : mc->interval;
        if (mc->warm_left > 0) {
            mc->warm_left--;
        }
        if (active && mc->since >= interval) {
            refresh(*mc);
        }

        mc->hist_pending = active;
        // a prefill replaces the context, so let the window re-form quickly afterwards
        if (!active && ubatch != nullptr && ubatch->n_tokens > moe_cache_max_tok()) {
            mc->warm_left = mc->warm;
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return (params.ubatch.n_tokens <= moe_cache_max_tok()) == active;
    }

    const llama_model_qwen4exp & model;
    const bool active;

private:
    void refresh(moe_cache & mc) {
        std::vector<int32_t> order(mc.n_expert);
        for (int64_t il = 0; il < mc.n_layer; ++il) {
            const auto & cnt = mc.counts[il];
            for (int64_t e = 0; e < mc.n_expert; ++e) {
                order[e] = (int32_t) e;
            }
            const int64_t n_top = std::min(mc.n_slot, mc.n_expert);
            std::partial_sort(order.begin(), order.begin() + n_top, order.end(),
                    [&](int32_t a, int32_t b) { return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b; });

            std::vector<char> want(mc.n_expert, 0);
            for (int64_t i = 0; i < n_top; ++i) {
                if (cnt[order[i]] > 0) {
                    want[order[i]] = 1;
                }
            }

            std::vector<int32_t> free_slots;
            for (int64_t s = 0; s < mc.n_slot; ++s) {
                const int32_t e = mc.in_slot[il][s];
                if (e < 0 || !want[e]) {
                    if (e >= 0) {
                        mc.slot_of[il][e] = -1;
                    }
                    mc.in_slot[il][s] = -1;
                    free_slots.push_back((int32_t) s);
                }
            }

            bool changed = false;
            for (int64_t i = 0; i < n_top && !free_slots.empty(); ++i) {
                const int32_t e = order[i];
                if (!want[e] || mc.slot_of[il][e] >= 0) {
                    continue;
                }
                const int32_t s = free_slots.back();
                free_slots.pop_back();
                const auto & l = model.layers[il];
                for (auto [dst, src] : { std::make_pair(mc.up[il], l.ffn_up_exps), std::make_pair(mc.gate[il], l.ffn_gate_exps),
                                         std::make_pair(mc.down[il], l.ffn_down_exps) }) {
                    ggml_backend_tensor_set(dst, (const char *) src->data + e*src->nb[2], s*dst->nb[2], src->nb[2]);
                }
                mc.slot_of[il][e] = s;
                mc.in_slot[il][s] = e;
                changed = true;
            }
            if (changed || !free_slots.empty()) {
                mc.upload_maps(il);
            }
        }
        mc.since = 0;
    }
};


llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    static_cast<const llama_model_qwen4exp &>(model).hc_fold();

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    {
        const auto & m = static_cast<const llama_model_qwen4exp &>(model);
        if (m.moe_cache_get() != nullptr) {
            auto mi = std::make_unique<llm_graph_input_moe_cache>(m, n_tokens <= moe_cache_max_tok());
            moe_inp = mi.get();
            res->add_input(std::move(mi));
        }
    }

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, model.layers[il].hc_attn_inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il, inject);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, model.layers[il].hc_ffn_inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
        if (plan.use) {
            mctx->get_mem()->qsa_set_input(plan, upd_blk, upd_cells, upd_pos, ratio);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        if (res && sparse_path && mctx->get_mem() != nullptr) {
            const auto p = mctx->get_mem()->qsa_plan(params.ubatch, ratio);
            res &= p.use == plan.use && p.refresh == plan.refresh && p.n_upd == plan.n_upd;
            if (res) {
                plan = p;
            }
        }

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // block key cache, filled only on the sparse path
    bool sparse_path = false;
    llama_memory_hybrid_idx::qsa_cache_plan plan;
    ggml_tensor * upd_blk   = nullptr; // I32 [n_upd]
    ggml_tensor * upd_cells = nullptr; // I32 [ratio*n_upd]
    ggml_tensor * upd_pos   = nullptr; // I32 [4*n_upd]
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "indexer_k_pooled", il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks*n_stream, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
    pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);

    // the heads sit side by side on ne[1] and there are only a few of them
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                score->nb[2], score->nb[3], h*score->nb[1]);
        summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
    }

    score = summed;
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}


static bool qwen4exp_qsa_sparse_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_QSA_GATHER");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

static int64_t qwen4exp_qsa_chunk() {
    static const int64_t chunk = [] {
        const char * e = getenv("LLAMA_QSA_CHUNK");
        return e ? std::max<int64_t>(1, atoll(e)) : (int64_t) 32;
    }();
    return chunk;
}

// QSA with the top-k cells gathered per query instead of masking a dense attention.
// Every step is row-wise in the query, so the queries are processed in chunks: no tensor spans n_kv x n_tokens.
// get_rows writes F16 directly: the CUDA kernel dequantizes to float and rounds to half, exactly
// what get_rows to F32 followed by a cast would do, without the F32 round trip through memory.
static ggml_tensor * qwen4exp_get_rows_f16(ggml_context * ctx, ggml_tensor * rows, ggml_tensor * idx) {
    ggml_tensor * t = ggml_get_rows(ctx, rows, idx);
    // The F16 destination is a CUDA extension of get_rows, so it is only safe for a source that
    // is already on a device. A view carries no buffer until the allocator runs, so ask the tensor
    // it views: the K/V cache has one, a graph input such as the mask does not.
    const ggml_tensor * owner = rows->view_src ? rows->view_src : rows;
    const char * buft = owner->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(owner->buffer)) : nullptr;
    if (!buft || ggml_backend_buffer_is_host(owner->buffer) || strncmp(buft, "CUDA", 4) != 0) {
        return ggml_cast(ctx, t, GGML_TYPE_F16);
    }
    t->type  = GGML_TYPE_F16;
    t->nb[0] = ggml_type_size(GGML_TYPE_F16);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
    return t;
}

ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_sparse(
        llm_graph_input_attn_kv *               inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           q_cur,
        ggml_tensor *                           k_cur,
        ggml_tensor *                           v_cur,
        ggml_tensor *                           inp_pos,
        int *                                   sections,
        float                                   kq_scale,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    const auto * mctx_cur = inp->mctx;

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t r       = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv    = mctx_idx->get_n_kv();
    const int64_t n_blocks = (n_kv + r - 1)/r;
    const int64_t width   = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);
    static const bool qsa_pad = getenv("LLAMA_QSA_PAD") && atoi(getenv("LLAMA_QSA_PAD"));
    const int64_t width_fa = qsa_pad && width >= 256 ? GGML_PAD(width, 256) : width;

    ggml_tensor * kq_mask = inp->get_kq_mask();

    const bool blk_bias = kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tokens && kq_mask->ne[3] == 1 &&
        cparams.causal_attn && !hparams.use_alibi;

    llm_graph_input_qsa * qinp = nullptr;
    {
        const auto it = qsa_inps.find((uint32_t) r);
        if (it != qsa_inps.end()) {
            qinp = it->second;
        } else {
            auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);

            qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
            qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, 1);
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, 1);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks);
            qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tokens, 1);

            ggml_set_input(qsa->cell_blk);
            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
            ggml_set_input(qsa->bias);

            qsa->sparse_path = true;
            if (mctx_hyb->get_mem() != nullptr && mctx_hyb->get_mem()->qsa_pooled(il) != nullptr) {
                qsa->plan = mctx_hyb->get_mem()->qsa_plan(ubatch, (uint32_t) r);
                if (qsa->plan.use && !qsa->plan.refresh) {
                    qsa->upd_blk   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, qsa->plan.n_upd);
                    qsa->upd_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r*qsa->plan.n_upd);
                    qsa->upd_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*qsa->plan.n_upd);
                    ggml_set_input(qsa->upd_blk);
                    ggml_set_input(qsa->upd_cells);
                    ggml_set_input(qsa->upd_pos);
                }
            }

            qinp = qsa.get();
            res->add_input(std::move(qsa));
            qsa_inps.emplace((uint32_t) r, qinp);
        }
    }

    // indexer: pooled block keys and per-token queries, as in build_qsa_top_k
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, qinp->k_idxs, il));

    ggml_tensor * k_idx_all = mctx_idx->get_k(ctx0, il);
    k_idx_all = ggml_view_3d(ctx0, k_idx_all, idx_dim, n_kv, 1, k_idx_all->nb[2], k_idx_all->nb[3], 0);

    ggml_tensor * key_cache = qinp->plan.use && mctx_hyb->get_mem() ? mctx_hyb->get_mem()->qsa_pooled(il) : nullptr;
    const bool    upd_only  = key_cache != nullptr && !qinp->plan.refresh;

    const int64_t n_pool    = upd_only ? qinp->plan.n_upd : n_blocks;

    ggml_tensor * members = ggml_get_rows(ctx0, k_idx_all, upd_only ? qinp->upd_cells : qinp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_pool, 1);

    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_pool, 1,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_pool, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_pool);
    pooled = ggml_rope_multi(ctx0, pooled, upd_only ? qinp->upd_pos : qinp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    if (upd_only) {
        pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, n_pool);
        ggml_tensor * stored = ggml_set_rows(ctx0, key_cache, pooled, qinp->upd_blk);
        ggml_build_forward_expand(gf, stored);
        pooled = ggml_view_3d(ctx0, stored, idx_dim, n_blocks, 1, stored->nb[1], stored->nb[1]*n_blocks, 0);
    } else {
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, 1);
        if (key_cache != nullptr) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_2d(ctx0, pooled, idx_dim, n_blocks),
                    ggml_view_2d(ctx0, key_cache, idx_dim, n_blocks, key_cache->nb[1], 0)));
        }
    }
    cb(pooled, "indexer_k", il);

    ggml_tensor * q_idx = build_lora_mm(model.layers[il].index_q_proj, cur);
    q_idx = ggml_reshape_3d(ctx0, q_idx, idx_dim, n_idx_h, n_tokens);
    q_idx = build_norm(q_idx, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q_idx = ggml_rope_multi(ctx0, q_idx, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_idx, "indexer_q", il);

    // attention inputs, as in build_attn_qsa
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(k_all->ne[2] == n_kv && v_all->nb[1] <= v_all->nb[2]);

    ggml_tensor * k_rows = ggml_view_2d(ctx0, k_all, k_all->ne[0]*k_all->ne[1], n_kv, k_all->nb[2], 0);
    ggml_tensor * v_rows = ggml_view_2d(ctx0, v_all, v_all->ne[0]*v_all->ne[1], n_kv, v_all->nb[2], 0);

    const int64_t chunk = qwen4exp_qsa_chunk();
    const size_t  m_el  = ggml_element_size(kq_mask);

    ggml_tensor * out = nullptr;
    static const bool tree_concat = getenv("LLAMA_QSA_TREE_CONCAT") && atoi(getenv("LLAMA_QSA_TREE_CONCAT"));
    std::vector<ggml_tensor *> parts;

    for (int64_t i0 = 0; i0 < n_tokens; i0 += chunk) {
        const int64_t c = std::min(chunk, n_tokens - i0);

        ggml_tensor * q_idx_c = ggml_cont(ctx0, ggml_view_3d(ctx0, q_idx, idx_dim, n_idx_h, c,
                q_idx->nb[1], q_idx->nb[2], i0*q_idx->nb[2]));

        ggml_tensor * score = ggml_mul_mat(ctx0, pooled, ggml_reshape_3d(ctx0, q_idx_c, idx_dim, n_idx_h*c, 1));
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, c, 1);
        score = ggml_relu(ctx0, score);

        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, c, 1,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }
        score = summed;

        ggml_tensor * bias_c = ggml_view_3d(ctx0, qinp->bias, qinp->bias->ne[0], c, 1,
                qinp->bias->nb[1], qinp->bias->nb[2], i0*qinp->bias->nb[1]);

        if (blk_bias) {
            score = ggml_add(ctx0, score, bias_c);
        }

        ggml_tensor * expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), qinp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

        if (blk_bias) {
            ggml_tensor * mask_c = ggml_view_2d(ctx0, kq_mask, n_kv, c, kq_mask->nb[1], i0*kq_mask->nb[1]);
            mask_c = kq_mask->type == GGML_TYPE_F32 ? mask_c : ggml_cast(ctx0, mask_c, GGML_TYPE_F32);
            expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask_c, n_kv, c, 1));
        } else {
            expanded = ggml_add(ctx0, expanded, bias_c);
        }

        ggml_tensor * idx = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
        idx = ggml_reshape_2d(ctx0, idx, width, c);

        ggml_tensor * q_c = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], c,
                q_cur->nb[1], q_cur->nb[2], i0*q_cur->nb[2]);

        // FlashAttention only takes its GQA-batched kernels when the key count is a multiple of 256; pad with
        // repeats of selected cells and mask them out below, so the attended set is unchanged
        ggml_tensor * idx_kv = idx;
        if (width_fa > width) {
            idx_kv = ggml_concat(ctx0, idx, ggml_view_2d(ctx0, idx, width_fa - width, c, idx->nb[1], 0), 0);
        }
        ggml_tensor * idx_flat = ggml_reshape_1d(ctx0, idx_kv, width_fa*c);

        ggml_tensor * k_sel = ggml_reshape_4d(ctx0, qwen4exp_get_rows_f16(ctx0, k_rows, idx_flat), k_all->ne[0], k_all->ne[1], width_fa, c);
        ggml_tensor * v_sel = ggml_reshape_4d(ctx0, qwen4exp_get_rows_f16(ctx0, v_rows, idx_flat), v_all->ne[0], v_all->ne[1], width_fa, c);
        ggml_build_forward_expand(gf, k_sel);
        ggml_build_forward_expand(gf, v_sel);

        ggml_tensor * m_rows = ggml_view_3d(ctx0, kq_mask, 1, n_kv, c, m_el, kq_mask->nb[1], i0*kq_mask->nb[1]);
        ggml_tensor * m_sel;
        if (kq_mask->type == GGML_TYPE_F16) {
            m_sel = ggml_reshape_4d(ctx0, qwen4exp_get_rows_f16(ctx0, m_rows, idx), width, 1, 1, c);
        } else {
            m_sel = ggml_get_rows(ctx0, m_rows, idx);
            m_sel = ggml_cast(ctx0, ggml_reshape_4d(ctx0, m_sel, width, 1, 1, c), kq_mask->type);
        }
        if (width_fa > width) {
            ggml_tensor * m_pad = ggml_fill(ctx0, ggml_cont(ctx0, ggml_view_4d(ctx0, m_sel, width_fa - width, 1, 1, c,
                    m_sel->nb[1], m_sel->nb[2], m_sel->nb[3], 0)), -INFINITY);
            m_sel = ggml_concat(ctx0, m_sel, m_pad, 0);
        }

        ggml_tensor * att = build_attn_mha(q_c, k_sel, v_sel, nullptr, m_sel, nullptr, nullptr, 0, kq_scale, il);
        ggml_build_forward_expand(gf, att);

        if (tree_concat) {
            parts.push_back(att);
        } else {
            out = out ? ggml_concat(ctx0, out, att, 1) : att;
        }
    }
    // appending one chunk at a time recopies everything before it; pairing keeps it n log n
    while (parts.size() > 1) {
        std::vector<ggml_tensor *> next;
        for (size_t j = 0; j + 1 < parts.size(); j += 2) {
            next.push_back(ggml_concat(ctx0, parts[j], parts[j + 1], 1));
        }
        if (parts.size() % 2) {
            next.push_back(parts.back());
        }
        parts.swap(next);
    }
    if (!parts.empty()) {
        out = parts[0];
    }
    cb(out, "kqv_out", il);

    if (inp->self_v_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, inp->self_v_rot);
    }

    return out;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    const bool sparse = qsa && qwen4exp_qsa_sparse_enabled() && cparams.flash_attn &&
        mctx_hyb->get_n_stream() == 1 && inp->get_kq_mask()->ne[3] == 1 &&
        mctx_hyb->get_idx()->get_n_kv() >= 2*((int64_t) hparams.indexer_top_k + hparams.dsv4_compress_ratios[il] - 1);

    ggml_tensor * top_k = qsa && !sparse ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;
    ggml_tensor * qsa_in = cur;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (sparse) {
        cur = build_attn_qsa_sparse(inp, mctx_hyb, qsa_in, Qcur, Kcur, Vcur, inp_pos, sections, kq_scale, il);
    } else if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_moe_cached(ggml_tensor * cur, const int il, ggml_tensor * independent, ggml_tensor * independent2) {
    const auto & m  = static_cast<const llama_model_qwen4exp &>(model);
    const auto * mc = m.mcache.get();
    const auto & layer = model.layers[il];

    const int64_t n_embd_ = cur->ne[0];
    const int64_t n_tok   = cur->ne[1];
    const int64_t k       = n_expert_used;

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);
    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    cb(probs, "ffn_moe_probs", il);

    ggml_tensor * selected = ggml_argsort_top_k(ctx0, probs, k);
    cb(selected->src[0], "ffn_moe_argsort", il);
    cb(selected, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tok), selected);
    cb(weights, "ffn_moe_weights", il);
    weights = ggml_reshape_2d(ctx0, weights, k, n_tok);
    ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
    cb(weights_sum, "ffn_moe_weights_sum", il);
    weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
    cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
    weights = ggml_div(ctx0, weights, weights_sum);
    cb(weights, "ffn_moe_weights_norm", il);
    weights = ggml_reshape_3d(ctx0, weights, 1, k, n_tok);
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }
    ggml_build_forward_expand(gf, weights);

    if (n_tok > 0) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, selected, k, 1, selected->nb[1], 0)),
                ggml_view_2d(ctx0, mc->hist, k, 1, mc->hist->nb[1], il*mc->hist->nb[1])));
    }

    ggml_tensor * sel_flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, selected), k*n_tok);
    ggml_tensor * ids_gpu  = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, mc->map_gpu[il], sel_flat), k, n_tok);
    ggml_tensor * ids_cpu  = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, mc->map_cpu[il], sel_flat), k, n_tok);
    ggml_build_forward_expand(gf, ids_cpu);

    ggml_tensor * x = ggml_reshape_3d(ctx0, cur, n_embd_, 1, n_tok);

    auto branch = [&](ggml_tensor * w_up, ggml_tensor * w_gate, ggml_tensor * w_down, ggml_tensor * ids) {
        ggml_tensor * up = build_lora_mm_id(w_up, x, ids);
        cb(up, "ffn_moe_up", il);
        ggml_tensor * g = build_lora_mm_id(w_gate, x, ids);
        cb(g, "ffn_moe_gate", il);
        ggml_tensor * act = ggml_swiglu_split(ctx0, g, up);
        cb(act, "ffn_moe_swiglu", il);
        ggml_tensor * out = build_lora_mm_id(w_down, act, ids);
        cb(out, "ffn_moe_down", il);
        return out;
    };

    // device work queued between the CPU branch inputs and the CPU branch runs while the CPU computes
    ggml_tensor * on_gpu = branch(mc->up[il], mc->gate[il], mc->down[il], ids_gpu);
    ggml_build_forward_expand(gf, on_gpu);
    if (independent) {
        ggml_build_forward_expand(gf, independent);
    }
    if (independent2) {
        ggml_build_forward_expand(gf, independent2);
    }
    ggml_tensor * experts = ggml_add(ctx0, on_gpu,
            branch(layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, ids_cpu));
    experts = ggml_mul(ctx0, experts, weights);
    cb(experts, "ffn_moe_weighted", il);
    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
    for (int64_t i = 0; i < k; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd_, n_tok, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }
    ggml_tensor * moe_out = cur_experts[0];
    for (int64_t i = 1; i < k; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }
    return moe_out;
}

static bool inject_late() {
    static const bool v = getenv("LLAMA_INJECT_LATE") && atoi(getenv("LLAMA_INJECT_LATE"));
    return v;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il, ggml_tensor * independent) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * ffn_shexp = nullptr;
    if (model.layers[il].ffn_up_shexp != nullptr) {
        // shared experts, as in the Qwen3Next reference
        ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
    }

    ggml_tensor * moe_out = moe_inp && moe_inp->active && cur->ne[1] > 0 ? build_moe_cached(cur, il, ffn_shexp, inject_late() ? nullptr : independent) :
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    if (ffn_shexp) {
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
