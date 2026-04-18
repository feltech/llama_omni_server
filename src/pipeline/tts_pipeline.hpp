/**
 * @file   tts_pipeline.hpp
 * @brief  TTS synthesis pipeline: LLM hidden states → 24 kHz PCM.
 *
 * `TtsPipeline` owns the TTS transformer model, custom weights, and the
 * Token2Wav directory.  It is process-scoped — one instance per
 * `ModelManager`.  The duplex session calls `build_condition()` then
 * `synthesize()` at each `<|chunk_eos|>` boundary or turn end.
 *
 * ### Pipeline stages
 *
 * 1. **filter** — `filter_special_tokens` strips duplex control tokens.
 * 2. **project** — `apply_projector_semantic` maps 4096-d LLM hiddens → 768-d.
 *    `normalize_l2_per_token` normalises per token.
 * 3. **condition** — for each filtered token i:
 *    ```
 *    condition[i] = emb_text(token_ids[i]) + projected_normalized[i]
 *    ```
 *    then appends `text_eos` (151692) and `audio_bos` (151687) embeddings.
 * 4. **prefill** — clear TTS KV and decode condition into `ctx_tts_`.
 * 5. **sample** — compute `head_code` logits, apply temperature / rep-penalty /
 *    top-k / top-p, sample multinomial; feed `emb_code` back for next step.
 *    Stop at EOS or `kMaxAudioTokens`.
 * 6. **Token2Wav** — `generate_audio_from_tokens` → 24 kHz PCM.
 */

#pragma once

#include <memory>
#include <span>
#include <vector>

#include <llama.h>

#include "../config/config_loader.hpp"

namespace llama_omni_server
{

/**
 * @brief Owns the TTS model, weights, and Token2Wav pipeline.
 *
 * All public methods must be called on the GPU executor thread (`gpu_ex_`).
 * `synthesize()` is not re-entrant — it mutates the TTS KV state.
 */
class TtsPipeline
{
public:
	/**
	 * @brief Construct an unloaded TTS pipeline.
	 *
	 * The transformer model, custom weights, and Token2Wav backend are created
	 * later by `load()`.
	 */
	TtsPipeline();

	/**
	 * @brief Destroy the TTS pipeline and release all owned model resources.
	 *
	 * Frees the TTS transformer model/context, custom weight buffers, and any
	 * Token2Wav state held inside the private implementation object.
	 */
	~TtsPipeline();

	TtsPipeline(TtsPipeline const &) = delete;
	TtsPipeline & operator=(TtsPipeline const &) = delete;
	TtsPipeline(TtsPipeline &&) = delete;
	TtsPipeline & operator=(TtsPipeline &&) = delete;

	/**
	 * @brief Load TTS transformer, custom weights, and Token2Wav.
	 *
	 * Blocking GPU call.  Idempotent — no-op if already loaded.
	 *
	 * @param cfg        Application config.  Uses `model.tts_transformer_path`,
	 *                   `model.tts_weights_path`, `model.projector_path`,
	 *                   `model.token2wav_dir`, and the sampling values under `tts.*`.
	 * @param llm_model  Main LLM model; stored in `TTSContext.model_llm` for
	 *                   validation checks inside vendor functions.
	 * @param transformer_on_gpu  When `true`, place the TTS transformer on GPU.
	 * @param token2wav_on_gpu    When `true`, place Token2Wav on GPU.
	 */
	void load(
		AppConfig const & cfg,
		llama_model * llm_model,
		bool transformer_on_gpu,
		bool token2wav_on_gpu);

	/// @return `true` after a successful `load()`.
	[[nodiscard]] bool is_loaded() const noexcept;

	/**
	 * @brief Build TTS condition embeddings from LLM speaking-turn tokens.
	 *
	 * Filters special tokens, projects LLM hidden states to 768-d TTS space,
	 * and assembles the flat condition array.
	 *
	 * For each filtered token `i`:
	 * ```
	 * condition[i] = emb_text(token_ids[i]) + L2_norm(projector(hiddens[i]))
	 * ```
	 * then appends `text_eos` and `audio_bos`.
	 *
	 * @param token_ids   LLM token IDs from the speaking turn (will be filtered).
	 * @param hiddens     Flat `[n_tokens × n_llm_embd]` LLM hidden states.
	 * @param n_llm_embd  LLM embedding dimension (4096 for MiniCPM-o 4.5).
	 * @return Flat `[(n_filtered + 2) × 768]` condition embeddings, empty on
	 *         failure (no valid tokens after filtering).
	 */
	[[nodiscard]] std::vector<float> build_condition(
		std::vector<llama_token> token_ids, std::vector<float> hiddens, int n_llm_embd) const;

	/**
	 * @brief Synthesize audio from TTS condition embeddings.
	 *
	 * Clears the TTS KV, prefills `condition`, samples audio tokens until EOS
	 * or `kMaxAudioTokens`, runs Token2Wav, and returns 24 kHz mono PCM.
	 *
	 * @param condition  Output of `build_condition()`.  Must not be empty.
	 * @return 24 kHz mono float32 PCM, or empty on synthesis failure.
	 */
	[[nodiscard]] std::vector<float> synthesize(std::span<float const> condition);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace llama_omni_server
