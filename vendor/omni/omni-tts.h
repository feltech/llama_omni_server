#pragma once

#include <vector>
#include <string>
#include <llama.h>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <memory>
#include "token2wav-impl.h"

struct ggml_context;
struct gguf_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct common_params;

/*
 * TTS Context and Types
 * Ported from: llama.cpp-omni/tools/omni/omni.h
 * 
 * Key structures (matching omni.h):
 * - ProjectorSemanticWeights: omni.h:lines ~200-230 (TTS projector weights)
 * - TTSContext: omni.h:lines ~280-330 (TTS context wrapper)
 * - LlamaLayers: omni.h:lines ~340-370 (TTS transformer layers)
 * 
 * Ported functions:
 * - is_valid_tts_token: omni.cpp: lines ~150-160
 * - filter_special_tokens: omni.cpp: lines ~165-195
 * - init_tts_context_with_model: omni.cpp: lines ~197-210
 * - load_projector_semantic: omni.cpp: lines ~440-510
 * - normalize_l2_per_token: omni.cpp: lines ~1974-1990
 * - apply_projector_semantic: omni.cpp: lines ~417-465
 * - tts_transformer_forward: omni.cpp: lines ~168-273 (manual transformer pass)
 * - tts_emb_text: omni.cpp: lines ~1836-1850
 * - load_tts_weights: omni.cpp: lines ~291-415
 * - generate_audio_tokens_local_simplex: omni.cpp: lines ~4544-4890
 * - generate_audio_from_tokens: omni.cpp: lines ~667-730
 * - prefill_with_emb_tts: omni.cpp: lines ~2081-2120
 * - free_tts_context: omni.cpp: lines ~4107-4110
 */

namespace omni {
namespace tts {

struct ProjectorSemanticWeights {
    struct ggml_backend* backend = nullptr;
    struct ggml_backend_buffer* buf_w = nullptr;
    struct ggml_context* ctx_w = nullptr;
    struct Layers {
        struct ggml_tensor* linear1_weight = nullptr;
        struct ggml_tensor* linear1_bias = nullptr;
        struct ggml_tensor* linear2_weight = nullptr;
        struct ggml_tensor* linear2_bias = nullptr;
    } layer;
    struct HParams {
        int in_dim = 4096;
        int out_dim = 768;
    } hparams;
    struct ggml_context* ctx_meta = nullptr;
    struct gguf_context* ctx_gguf = nullptr;
    bool initialized = false;
};

struct TTSContext {
    llama_model* model_llm = nullptr;
    llama_context* ctx_llm = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_model* model_tts = nullptr;
    llama_context* ctx_tts = nullptr;
    float* head_code_weight = nullptr;
    float* emb_code_weight = nullptr;
    float* emb_text_weight = nullptr;
    ProjectorSemanticWeights projector_weights;
    bool projector_loaded = false;
    int head_code_hidden_size = 768;
    int head_code_num_audio_tokens = 6562;
    int emb_code_vocab_size = 6562;
    int emb_code_hidden_size = 768;
    int emb_text_vocab_size = 152064;
    int emb_text_hidden_size = 768;
    bool emb_code_stored_as_transposed = false;
    std::vector<float> tts_condition_embeddings;
    int tts_condition_length = 0;
    int tts_condition_n_embd = 0;
    std::vector<float> tts_history_embeddings;
    bool tts_condition_saved = false;
    std::vector<int> tts_all_generated_tokens;
    std::vector<int> tts_token_buffer;
    int tts_n_past_accumulated = 0;
    std::shared_ptr<omni::flow::Token2WavSession> token2wav_session;
    bool token2wav_initialized = false;
};

bool is_valid_tts_token(llama_token tid);
void filter_special_tokens(std::vector<llama_token>& token_ids, std::vector<float>& hidden_states, int n_embd);
bool init_tts_context_with_model(TTSContext& tts_ctx, llama_model* model, llama_context* ctx, const llama_vocab* vocab);
// tts_transformer_path: the transformer-only GGUF (182 standard tensors, from split_tts_gguf.py)
// tts_weights_path:     the custom-weights GGUF  (11 TTS-specific tensors, from split_tts_gguf.py)
bool load_tts_weights(TTSContext& tts_ctx, const char* projector_gguf,
                      const char* tts_transformer_path, const char* tts_weights_path,
                      const char* token2wav_model_dir, const char* device_token2mel = "gpu", const char* device_vocoder = "gpu", const char* custom_prompt_cache_path=nullptr);
bool load_projector_semantic(const char* gguf_path, ProjectorSemanticWeights& model);
void normalize_l2_per_token(float* embeddings, int n_tokens, int n_embd, float eps = 1e-8f);
void apply_projector_semantic(const ProjectorSemanticWeights& model, const std::vector<float>& input_data, int n_tokens, std::vector<float>& result);
bool tts_emb_text(const TTSContext& tts_ctx, llama_token token_id, float* embedding_out, int emb_size);
bool generate_audio_from_tokens(TTSContext& tts_ctx, const std::vector<int32_t>& audio_tokens, std::vector<float>& audio_samples, int sample_rate);
bool prefill_with_emb_tts(TTSContext& tts_ctx, struct common_params* params, float* embed, int n_pos, int n_batch, int* n_past_tts);
void free_tts_context(TTSContext& tts_ctx);

} // namespace tts
} // namespace omni
