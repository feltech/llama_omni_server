/**
 * @file   tts_pipeline.cpp
 * @brief  TtsPipeline implementation — TTS synthesis from LLM hidden states.
 *
 * Ported audio-token sampling logic from:
 * `tests/integration/vendor/test_tts.cpp` (lines 941–1246).
 *
 * ### Sampling loop
 *
 * At each step `t`:
 * 1. `llama_get_embeddings_ith(ctx_tts, -1)` → 768-d hidden state.
 * 2. Logits: `head_code_weight[tok_idx][dim] × hidden[dim]` (matmul).
 * 3. Temperature scaling, repetition penalty, softmax, top-k, top-p.
 * 4. Multinomial sample.
 * 5. Feed back `emb_code[selected_tok]` as the next-step embedding.
 */

#include "./tts_pipeline.hpp"
#include "./tts_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <experimental/mdspan>

#include <llama.h>
#include <spdlog/spdlog.h>

#include <omni/omni-tts.h>

namespace llama_omni_server
{

namespace
{

// ── TTS sampling parameters ────────────────────────────────────────────────────
// All values match omni.cpp:3617-3627 and test_tts.cpp:876-882.

/// Audio vocab size: relative IDs 0-6561, with 6561 = EOS.
constexpr int kNumAudioTokens = 6562;

/// Relative token index that signals end-of-speech.
constexpr int kEosRelativeIdx = kNumAudioTokens - 1;

/// Absolute token ID base: relative_idx + kAudioBosTokenId = absolute ID.
constexpr int kAudioBosTokenId = 151687;

/// text_eos token ID appended before audio_bos in condition.
constexpr llama_token kTextEosTokenId = 151692;

/// TTS embedding dimension (projector output / emb_text dimension).
constexpr int kTtsEmbDim = 768;

/// TTS transformer context window / batch size.
constexpr int kTtsNCtx = 4096;

/// Output PCM sample rate produced by Token2Wav.
constexpr int kOutputSampleRate = 24000;

// ── RAII wrappers ─────────────────────────────────────────────────────────────

/// Custom deleter for `llama_model*` allocated by `llama_model_load_from_file`.
struct TtsModelDeleter
{
	void operator()(llama_model * ptr) const noexcept
	{
		/// `llama_model_free` releases all TTS transformer weights from GPU/CPU.
		llama_model_free(ptr);
	}
};

/// Custom deleter for `llama_context*` allocated by `llama_init_from_model`.
struct TtsContextDeleter
{
	void operator()(llama_context * ptr) const noexcept
	{
		/// `llama_free` releases the TTS KV cache and compute buffers.
		llama_free(ptr);
	}
};

using TtsModelPtr = std::unique_ptr<llama_model, TtsModelDeleter>;
using TtsContextPtr = std::unique_ptr<llama_context, TtsContextDeleter>;

// ── Sampling helper functions ─────────────────────────────────────────────────

/**
 * @brief Decode condition embeddings into the TTS context KV cache.
 *
 * Splits the condition into batches of at most `kTtsNCtx` tokens and
 * runs `llama_decode` for each.  `n_past_tts` is updated in place.
 *
 * @return true on success, false if any `llama_decode` call fails.
 */
bool decode_tts_condition(
	llama_context * tts_ctx, std::span<float const> condition, int n_cond_tokens, int & n_past_tts)
{
	namespace stdex = std::experimental;
	auto cond_view =
		stdex::mdspan<float const, stdex::extents<int, stdex::dynamic_extent, kTtsEmbDim>>(
			condition.data(), n_cond_tokens);

	/// Allocate position vector once; re-fill with iota each batch iteration.
	std::vector<llama_pos> pos_vec;
	pos_vec.reserve(kTtsNCtx);

	for (int batch_start = 0; batch_start < n_cond_tokens; batch_start += kTtsNCtx)
	{
		int const n_eval = std::min(kTtsNCtx, n_cond_tokens - batch_start);

		/// Build position vector `[n_past_tts, n_past_tts + n_eval)`.
		pos_vec.resize(static_cast<std::size_t>(n_eval));
		std::ranges::iota(pos_vec, static_cast<llama_pos>(n_past_tts));

		/// Construct an embedding batch (batch.embd != nullptr, batch.token = nullptr).
		llama_batch cond_batch{};
		cond_batch.n_tokens = n_eval;
		/// The const_cast is required because the libllama C API takes `float*` (not `const
		/// float*`).
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		cond_batch.embd = const_cast<float *>(&cond_view[batch_start, 0]);
		cond_batch.pos = pos_vec.data();

		if (llama_decode(tts_ctx, cond_batch) != 0)
		{
			spdlog::error(
				"TtsPipeline: decode_tts_condition llama_decode failed at batch_start={}",
				batch_start);
			return false;
		}
		n_past_tts += n_eval;
	}
	return true;
}

}  // namespace

namespace tts_detail
{

/**
 * @brief Compute logits for the next audio token via head_code matmul.
 *
 * `logits[i] = sum_d( hidden[d] * head_code_weight[i * hidden_size + d] )`
 *
 * @param head_code_w  Pointer to head_code weight matrix `[n_audio_tokens × hidden_size]`.
 * @param hidden       Pointer to the current 768-d hidden state.
 * @param n_audio_tokens  Vocabulary size of audio tokens.
 * @param hidden_size  Dimension of hidden state (768).
 */
std::vector<float> compute_audio_logits(
	float const * head_code_w,	// NOLINT(bugprone-easily-swappable-parameters) — weight matrix,
								// distinct from hidden state
	float const * hidden,
	int n_audio_tokens,	 // NOLINT(bugprone-easily-swappable-parameters) — vocab count, distinct
						 // from embedding dim
	int hidden_size)
{
	using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using ConstMatrixMap = Eigen::Map<Matrix const>;
	using VectorMap = Eigen::Map<Eigen::VectorXf>;
	using ConstVectorMap = Eigen::Map<Eigen::VectorXf const>;

	std::vector<float> logits(static_cast<std::size_t>(n_audio_tokens));

	ConstMatrixMap const weights_as_mat(head_code_w, n_audio_tokens, hidden_size);
	ConstVectorMap const hidden_as_vec(hidden, hidden_size);
	VectorMap logits_as_vec(logits.data(), n_audio_tokens);

	logits_as_vec = weights_as_mat * hidden_as_vec;
	return logits;
}

/// Apply temperature scaling to logits in place.
void apply_temperature(std::span<float> logits, float temperature)
{
	Eigen::Map<Eigen::VectorXf> logits_map(logits.data(), static_cast<Eigen::Index>(logits.size()));
	logits_map /= temperature;
}

/**
 * @brief Apply repetition penalty to logits for recently generated tokens.
 *
 * Positive logits are divided by `rep_penalty`; negative logits are multiplied,
 * matching the omni.cpp reference implementation.
 */
void apply_rep_penalty(
	std::span<float> logits,
	std::span<int const> recent_relative,
	int penalty_last_n,	 // NOLINT(bugprone-easily-swappable-parameters) — window size, distinct
						 // roles from n_audio_tokens and rep_penalty
	int n_audio_tokens,
	float rep_penalty)
{
	int const start_int = std::max(0, static_cast<int>(recent_relative.size()) - penalty_last_n);
	std::vector<bool> occurred(static_cast<std::size_t>(n_audio_tokens), false);

	for (auto recent_idx = static_cast<std::size_t>(start_int); recent_idx < recent_relative.size();
		 ++recent_idx)
	{
		int const tok = recent_relative[recent_idx];
		if (tok >= 0 && tok < n_audio_tokens)
		{
			occurred[static_cast<std::size_t>(tok)] = true;
		}
	}

	for (int tok_idx = 0; tok_idx < n_audio_tokens; ++tok_idx)
	{
		if (occurred[static_cast<std::size_t>(tok_idx)])
		{
			float & logit = logits[static_cast<std::size_t>(tok_idx)];
			logit = (logit >= 0.0F) ? (logit / rep_penalty) : (logit * rep_penalty);
		}
	}
}

/// Convert logits to probabilities via softmax.
std::vector<float> softmax_probs(std::span<float const> logits)
{
	Eigen::Map<Eigen::VectorXf const> const logits_vec(
		logits.data(), static_cast<Eigen::Index>(logits.size()));
	float const max_logit = logits_vec.maxCoeff();
	Eigen::VectorXf exp_v = (logits_vec.array() - max_logit).exp();
	exp_v /= exp_v.sum();
	std::vector<float> probs(logits.size());
	Eigen::Map<Eigen::VectorXf>(probs.data(), static_cast<Eigen::Index>(probs.size())) = exp_v;
	return probs;
}

/// Apply top-k filtering: zero out probabilities outside the top-k.
void apply_top_k(std::span<float> probs, int top_k, int n_tokens)
{
	if (top_k <= 0 || top_k >= n_tokens)
	{
		return;
	}
	std::vector<float> sorted_probs(probs.begin(), probs.end());
	std::nth_element(
		sorted_probs.begin(), sorted_probs.begin() + (n_tokens - top_k), sorted_probs.end());
	float const threshold = sorted_probs[static_cast<std::size_t>(n_tokens - top_k)];
	for (float & prob : probs)
	{
		if (prob < threshold)
		{
			prob = 0.0F;
		}
	}
}

/// Apply top-p (nucleus) filtering: zero out tokens beyond the probability mass threshold.
void apply_top_p(
	std::span<float> probs,
	float const top_p,	// NOLINT(bugprone-easily-swappable-parameters) — probability threshold,
						// distinct from n_tokens
	int const n_tokens)
{
	if (top_p >= 1.0F)
	{
		return;
	}
	std::vector<int> prob_idx(static_cast<std::size_t>(n_tokens));
	std::ranges::iota(prob_idx, 0);
	std::ranges::sort(
		prob_idx,
		[&](int left, int right) noexcept
		{ return probs[static_cast<std::size_t>(left)] > probs[static_cast<std::size_t>(right)]; });

	float const prob_sum = std::accumulate(probs.cbegin(), probs.cend(), 0.0F);
	float cum_prob = 0.0F;
	bool cutoff_reached = false;
	for (int sorted_pos = 0; sorted_pos < n_tokens; ++sorted_pos)
	{
		int const idx = prob_idx[static_cast<std::size_t>(sorted_pos)];
		if (cutoff_reached)
		{
			probs[static_cast<std::size_t>(idx)] = 0.0F;
			continue;
		}
		cum_prob += probs[static_cast<std::size_t>(idx)] / prob_sum;
		if (cum_prob >= top_p)
		{
			cutoff_reached = true;
		}
	}
}

/// Re-normalise probabilities so they sum to 1.
void renormalize(std::span<float> probs)
{
	Eigen::Map<Eigen::VectorXf> probs_map(probs.data(), static_cast<Eigen::Index>(probs.size()));
	float const sum = probs_map.sum();
	if (sum > 0.0F)
	{
		probs_map /= sum;
	}
}

/**
 * @brief Draw one sample from a categorical distribution via linear scan.
 *
 * @return Relative token index in `[0, n_tokens)`.
 */
int multinomial_sample(std::span<float const> probs, std::mt19937 & rng, int n_tokens)
{
	std::uniform_real_distribution<float> dist(0.0F, 1.0F);
	float const rand_val = dist(rng);
	float cum_prob = 0.0F;
	for (int tok_idx = 0; tok_idx < n_tokens; ++tok_idx)
	{
		cum_prob += probs[static_cast<std::size_t>(tok_idx)];
		if (rand_val <= cum_prob)
		{
			return tok_idx;
		}
	}
	return n_tokens - 1;  ///< Fallback to last token.
}

}  // namespace tts_detail

// ── TtsPipeline::Impl ─────────────────────────────────────────────────────────

/**
 * @brief Implementation detail — keeps vendor headers out of the public API.
 */
struct TtsPipeline::Impl
{
	TtsModelPtr tts_model_;				   ///< Loaded TTS transformer weights.
	TtsContextPtr tts_ctx_;				   ///< TTS inference context (KV cache).
	omni::tts::TTSContext tts_weights_{};  ///< Projector, emb_code, emb_text, head_code, Token2Wav.
	std::mt19937 rng_{std::random_device{}()};	///< RNG for multinomial sampling.
	TtsConfig sample_cfg_{};  ///< Runtime sampling parameters copied from AppConfig.
	bool loaded_{false};
};

// ── TtsPipeline public API ────────────────────────────────────────────────────

TtsPipeline::TtsPipeline() : impl_{std::make_unique<Impl>()} {}

TtsPipeline::~TtsPipeline()
{
	if (impl_ && impl_->loaded_)
	{
		/// `free_tts_context` releases projector backend + emb_code/emb_text/head_code malloc'd
		/// arrays. The TTS model and context are released by their RAII handles above.
		omni::tts::free_tts_context(impl_->tts_weights_);
	}
}

void TtsPipeline::load(
	AppConfig const & cfg,
	llama_model * llm_model,
	bool const transformer_on_gpu,
	bool const token2wav_on_gpu)
{
	if (impl_->loaded_)
	{
		spdlog::debug("TtsPipeline::load: already loaded — skipping");
		return;
	}

	if (cfg.model.tts_transformer_path.empty())
	{
		spdlog::debug("TtsPipeline::load: tts_transformer_path empty — skipping");
		return;
	}

	spdlog::info("TtsPipeline: loading TTS transformer from '{}'", cfg.model.tts_transformer_path);

	/// `llama_model_load_from_file` loads the split TTS transformer GGUF (182 standard tensors).
	/// The original combined TTS GGUF (193 tensors) cannot be loaded because its 11 custom
	/// tensors (emb_code, emb_text, head_code, projector_*) are not recognised by libllama.
	auto model_params = llama_model_default_params();
	model_params.n_gpu_layers = transformer_on_gpu ? std::numeric_limits<int>::max() : 0;

	// NOLINTNEXTLINE(misc-const-correctness) — raw pointer, not const
	llama_model * raw_tts_model =
		llama_model_load_from_file(cfg.model.tts_transformer_path.c_str(), model_params);

	/// If GPU allocation fails (VRAM OOM), retry on CPU so that the TTS pipeline
	/// degrades gracefully rather than crashing — synthesis will be slower but correct.
	if (raw_tts_model == nullptr && model_params.n_gpu_layers != 0)
	{
		spdlog::warn("TtsPipeline: GPU allocation failed for TTS transformer — retrying on CPU");
		model_params.n_gpu_layers = 0;
		raw_tts_model =
			llama_model_load_from_file(cfg.model.tts_transformer_path.c_str(), model_params);
	}

	if (raw_tts_model == nullptr)
	{
		throw std::runtime_error{std::format(
			"TtsPipeline: failed to load TTS transformer from '{}'",
			cfg.model.tts_transformer_path)};
	}
	impl_->tts_model_.reset(raw_tts_model);
	impl_->sample_cfg_ = cfg.tts;

	/// `llama_init_from_model` allocates the TTS KV cache.
	/// `embeddings = true` ensures `llama_get_embeddings_ith` works during sampling.
	/// `n_ctx = n_batch = kTtsNCtx` accommodates the longest condition + audio buffer.
	auto tts_ctx_params = llama_context_default_params();
	tts_ctx_params.n_ctx = static_cast<std::uint32_t>(kTtsNCtx);
	tts_ctx_params.n_batch = static_cast<std::uint32_t>(kTtsNCtx);
	tts_ctx_params.n_ubatch = static_cast<std::uint32_t>(kTtsNCtx);
	tts_ctx_params.embeddings = true;

	// NOLINTNEXTLINE(misc-const-correctness) — raw pointer, not const
	llama_context * const raw_tts_ctx = llama_init_from_model(raw_tts_model, tts_ctx_params);
	if (raw_tts_ctx == nullptr)
	{
		throw std::runtime_error{"TtsPipeline: failed to create TTS inference context"};
	}
	impl_->tts_ctx_.reset(raw_tts_ctx);

	/// `init_tts_context_with_model` sets `model_llm` (main LLM) in TTSContext.
	/// The main LLM is only stored for vendor validation checks; TTS synthesis
	/// uses `tts_ctx_` (loaded above), not the LLM context.
	omni::tts::init_tts_context_with_model(
		impl_->tts_weights_,
		llm_model,
		nullptr,  ///< ctx_llm not needed — we own tts_ctx_ separately
		llm_model != nullptr ? llama_model_get_vocab(llm_model) : nullptr);

	spdlog::info("TtsPipeline: loading custom TTS weights from '{}'", cfg.model.tts_weights_path);

	/// `load_tts_weights` reads projector, emb_code, emb_text, head_code from GGUF files
	/// and initialises Token2Wav from the directory.  The transformer path is unused
	/// inside (we loaded it above); it's passed for reference only.
	bool const weights_loaded = omni::tts::load_tts_weights(
		impl_->tts_weights_,
		cfg.model.projector_path.c_str(),
		cfg.model.tts_transformer_path.c_str(),
		cfg.model.tts_weights_path.c_str(),
		cfg.model.token2wav_dir.empty() ? nullptr : cfg.model.token2wav_dir.c_str(),
		transformer_on_gpu ? "gpu" : "cpu",
		token2wav_on_gpu ? "gpu" : "cpu",
		cfg.voice.reference_prompt_cache.empty() ? nullptr : cfg.voice.reference_prompt_cache.c_str());

	if (!weights_loaded)
	{
		throw std::runtime_error{std::format(
			"TtsPipeline: load_tts_weights failed (projector='{}', weights='{}', token2wav='{}', "
			"prompt_cache='{}')",
			cfg.model.projector_path,
			cfg.model.tts_weights_path,
			cfg.model.token2wav_dir,
			cfg.voice.reference_prompt_cache)};
	}

	impl_->loaded_ = true;
	spdlog::info("TtsPipeline: loaded successfully");
}

bool TtsPipeline::is_loaded() const noexcept
{
	return impl_ && impl_->loaded_;
}

std::vector<float> TtsPipeline::build_condition(
	std::vector<llama_token> token_ids, std::vector<float> hiddens, int const n_llm_embd) const
{
	if (!impl_->loaded_)
	{
		spdlog::error("TtsPipeline::build_condition: not loaded");
		return {};
	}

	if (token_ids.empty())
	{
		return {};
	}

	/// `filter_special_tokens` removes duplex control tokens (e.g. `<|speak|>`,
	/// `<|chunk_eos|>`) and their corresponding hidden states.
	omni::tts::filter_special_tokens(token_ids, hiddens, n_llm_embd);

	if (token_ids.empty())
	{
		spdlog::debug("TtsPipeline::build_condition: no tokens after filtering");
		return {};
	}

	auto const n_tokens = static_cast<int>(token_ids.size());

	/// `apply_projector_semantic` runs the two-layer MLP (4096 → 768) on all hidden states.
	std::vector<float> projected;
	omni::tts::apply_projector_semantic(
		impl_->tts_weights_.projector_weights, hiddens, n_tokens, projected);

	/// `normalize_l2_per_token` applies L2 normalisation per 768-d token vector.
	omni::tts::normalize_l2_per_token(projected.data(), n_tokens, kTtsEmbDim);

	/// Build condition: `condition[i] = emb_text(token_ids[i]) + projected_normalized[i]`
	std::vector<float> text_embeddings(static_cast<std::size_t>(n_tokens) * kTtsEmbDim);
	namespace stdex = std::experimental;
	auto text_view = stdex::mdspan<float, stdex::extents<int, stdex::dynamic_extent, kTtsEmbDim>>(
		text_embeddings.data(), n_tokens);
	for (int tok_idx = 0; tok_idx < n_tokens; ++tok_idx)
	{
		/// `tts_emb_text` looks up token_ids[tok_idx] in the TTS text embedding table.
		/// This provides phoneme/semantic identity for the TTS model.
		if (!omni::tts::tts_emb_text(
				impl_->tts_weights_, token_ids[tok_idx], &text_view[tok_idx, 0], kTtsEmbDim))
		{
			spdlog::warn("TtsPipeline: tts_emb_text failed for token {}", token_ids[tok_idx]);
		}
	}

	std::vector<float> condition(static_cast<std::size_t>(n_tokens) * kTtsEmbDim);
	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf const> const text_map(text_embeddings.data(), n_tokens, kTtsEmbDim);
	Eigen::Map<RowMajorMatrixXf const> const projected_map(projected.data(), n_tokens, kTtsEmbDim);
	Eigen::Map<RowMajorMatrixXf> condition_map(condition.data(), n_tokens, kTtsEmbDim);
	condition_map = text_map + projected_map;

	/// Append `text_eos` embedding — marks end of text conditioning.
	/// `kTextEosTokenId = 151692` from Python: `tts_config.text_eos_token_id`.
	{
		std::vector<float> text_eos_emb(kTtsEmbDim);
		omni::tts::tts_emb_text(
			impl_->tts_weights_, kTextEosTokenId, text_eos_emb.data(), kTtsEmbDim);
		condition.insert(condition.end(), text_eos_emb.begin(), text_eos_emb.end());
	}

	/// Append `audio_bos` embedding — start-of-speech marker for the TTS model.
	/// `kAudioBosTokenId = 151687`.
	{
		std::vector<float> bos_emb(kTtsEmbDim);
		omni::tts::tts_emb_text(
			impl_->tts_weights_,
			static_cast<llama_token>(kAudioBosTokenId),
			bos_emb.data(),
			kTtsEmbDim);
		condition.insert(condition.end(), bos_emb.begin(), bos_emb.end());
	}

	return condition;
}

std::vector<float> TtsPipeline::synthesize(std::span<float const> condition)
{
	if (!impl_->loaded_)
	{
		spdlog::error("TtsPipeline::synthesize: not loaded");
		return {};
	}

	if (condition.empty())
	{
		return {};
	}

	llama_context * const tts_ctx = impl_->tts_ctx_.get();

	// ── Clear TTS KV cache ─────────────────────────────────────────────────────
	/// Clear any residual KV state from a previous synthesis call before prefilling.
	/// `llama_memory_seq_rm(mem, seq_id=0, p0=0, p1=-1)` removes all positions.
	// NOLINTNEXTLINE(misc-misplaced-const) — llama_memory_t is a pointer typedef
	llama_memory_t const mem = llama_get_memory(tts_ctx);
	if (mem != nullptr)
	{
		llama_memory_seq_rm(mem, 0, 0, -1);
	}

	// ── Prefill condition embeddings ───────────────────────────────────────────
	int n_past_tts = 0;
	auto const n_cond_tokens = static_cast<int>(condition.size()) / kTtsEmbDim;
	if (!decode_tts_condition(tts_ctx, condition, n_cond_tokens, n_past_tts))
	{
		return {};
	}

	// ── Audio token sampling loop ──────────────────────────────────────────────
	std::vector<int32_t> audio_tokens;
	audio_tokens.reserve(impl_->sample_cfg_.max_audio_tokens);
	std::vector<int> recent_relative;
	recent_relative.reserve(impl_->sample_cfg_.repeat_penalty_last_n);

	int const hidden_size = impl_->tts_weights_.head_code_hidden_size;	///< 768
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) — malloc'd vendor array
	float const * const head_code_w = impl_->tts_weights_.head_code_weight;
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) — malloc'd vendor array
	float const * const emb_code_w = impl_->tts_weights_.emb_code_weight;
	int const emb_code_size = impl_->tts_weights_.emb_code_hidden_size;	 ///< 768
	namespace stdex = std::experimental;
	auto emb_code_view = stdex::
		mdspan<float const, stdex::extents<int, stdex::dynamic_extent, stdex::dynamic_extent>>(
			emb_code_w, kNumAudioTokens, emb_code_size);

	int step = 0;
	for (; step < impl_->sample_cfg_.max_audio_tokens; ++step)
	{
		/// `llama_get_embeddings_ith(ctx, -1)` returns the 768-d activation of
		/// the last decoded token — the TTS model's representation used to
		/// predict the next audio token.
		float const * const hidden = llama_get_embeddings_ith(tts_ctx, -1);
		if (hidden == nullptr)
		{
			spdlog::error("TtsPipeline: llama_get_embeddings_ith returned null at step {}", step);
			break;
		}

		std::vector<float> logits =
			tts_detail::compute_audio_logits(head_code_w, hidden, kNumAudioTokens, hidden_size);
		tts_detail::apply_temperature(logits, impl_->sample_cfg_.temperature);
		tts_detail::apply_rep_penalty(
			logits,
			recent_relative,
			impl_->sample_cfg_.repeat_penalty_last_n,
			kNumAudioTokens,
			impl_->sample_cfg_.repeat_penalty);
		std::vector<float> probs = tts_detail::softmax_probs(logits);
		tts_detail::apply_top_k(probs, impl_->sample_cfg_.top_k_tokens, kNumAudioTokens);
		tts_detail::apply_top_p(probs, impl_->sample_cfg_.top_p_threshold, kNumAudioTokens);

		/// Apply min-p sampling: drop tokens with probability less than min_p * max_prob.
		if (auto const iter_to_max_prob = std::ranges::max_element(probs);
			iter_to_max_prob != probs.cend())
		{
			float const max_prob = *iter_to_max_prob;
			float const min_p_threshold = max_prob * impl_->sample_cfg_.min_p;
			std::ranges::transform(
				probs,
				begin(probs),
				[min_p_threshold](float const prob)
				{ return (prob < min_p_threshold) ? 0.0F : prob; });
			;
		}

		tts_detail::renormalize(probs);

		int const selected = tts_detail::multinomial_sample(probs, impl_->rng_, kNumAudioTokens);

		if (selected == kEosRelativeIdx)
		{
			spdlog::debug("TtsPipeline: EOS at step {}", step);
			break;
		}

		audio_tokens.push_back(static_cast<int32_t>(selected + kAudioBosTokenId));
		recent_relative.push_back(selected);

		// ── Feed emb_code embedding for next step ──────────────────────────────
		/// Look up `emb_code[selected]` (768-d) and decode it as the next position.
		std::vector<float> audio_emb(static_cast<std::size_t>(emb_code_size));
		Eigen::Map<Eigen::VectorXf const> const emb_row(&emb_code_view[selected, 0], emb_code_size);
		Eigen::Map<Eigen::VectorXf>(audio_emb.data(), emb_code_size) = emb_row;

		auto pos_val = static_cast<llama_pos>(n_past_tts);
		llama_batch audio_batch{};
		audio_batch.n_tokens = 1;
		audio_batch.embd = audio_emb.data();
		audio_batch.pos = &pos_val;

		if (llama_decode(tts_ctx, audio_batch) != 0)
		{
			spdlog::error("TtsPipeline: llama_decode failed for emb_code at step {}", step);
			break;
		}
		n_past_tts++;
	}

	if (audio_tokens.empty())
	{
		spdlog::warn("TtsPipeline: no audio tokens generated");
		return {};
	}

	if (step >= impl_->sample_cfg_.max_audio_tokens)
		spdlog::error("TtsPipeline: synthesis exceeded max audio tokens");

	spdlog::debug("TtsPipeline: synthesized {} audio tokens", audio_tokens.size());

	// ── Token2Wav ──────────────────────────────────────────────────────────────
	/// `generate_audio_from_tokens` runs the 4-stage Token2Wav pipeline
	/// (encoder → flow matching → flow extra → HiFi-GAN) and returns 24 kHz PCM.
	std::vector<float> pcm;
	if (!omni::tts::generate_audio_from_tokens(
			impl_->tts_weights_, audio_tokens, pcm, kOutputSampleRate))
	{
		spdlog::error("TtsPipeline: generate_audio_from_tokens failed");
		return {};
	}

	spdlog::debug(
		"TtsPipeline: Token2Wav produced {} samples ({:.2f} s)",
		pcm.size(),
		static_cast<double>(pcm.size()) / kOutputSampleRate);

	return pcm;
}

}  // namespace llama_omni_server
