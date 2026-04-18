/**
 * @file   audio_encoder.cpp
 * @brief  WAV-to-embedding implementation using the Whisper audio encoder.
 *
 * Ported from `tests/integration/vendor/test_duplex.cpp`:`load_and_encode_audio`
 * with the following adaptations:
 * - The `audition_ctx` is injected (already loaded) rather than created here.
 * - `std::cout` diagnostics replaced with `spdlog`.
 * - `encode_reference_voice` still returns an empty vector on failure, but
 *   `encode_pcm_audio` throws so streaming decode paths do not silently
 *   continue after an audio-encoder failure.
 */

#include "./audio_encoder.hpp"

#include <stdexcept>

#include <spdlog/spdlog.h>

#include "omni/audition.h"
#include "wav.hpp"

namespace llama_omni_server
{

std::vector<float> encode_reference_voice(
	audition_ctx * ctx, std::filesystem::path const & wav_path, int n_threads)
{
	if (ctx == nullptr)
	{
		spdlog::error("encode_reference_voice: audition_ctx is null");
		return {};
	}

	// ── Load WAV ──────────────────────────────────────────────────────────────

	std::vector<float> mono;
	try
	{
		mono = read_wav_mono(wav_path).pcm;
	}
	catch (std::runtime_error const & ex)
	{
		spdlog::error("encode_reference_voice: {}", ex.what());
		return {};
	}
	if (mono.size() % kInputAudioEmbeddingAlignmentSamples != 0)
	{
		std::size_t const aligned_size = kInputAudioEmbeddingAlignmentSamples *
			(1 + mono.size() / kInputAudioEmbeddingAlignmentSamples);
		spdlog::debug("encode_reference_voice: aligning {} to {}", mono.size(), aligned_size);
		mono.resize(aligned_size, 0);
	}

	// ── Whisper Mel spectrogram preprocessing ────────────────────────────────

	/// `audition_get_mel_filters` returns the 80-band Mel filter matrix used
	/// by the Whisper encoder.  The filters are stored in the loaded model.
	auto const filters = audition_get_mel_filters(ctx);

	/// `preprocess_audio` converts raw PCM to a Mel spectrogram.
	/// Chunked into 30-second windows (WHISPER_CHUNK_SIZE = 30); the first
	/// window is used for the reference voice (single-shot encoding).
	std::vector<whisper_preprocessor::whisper_mel> mel;
	whisper_preprocessor::preprocess_audio(mono.data(), mono.size(), filters, mel);
	if (mel.empty())
	{
		spdlog::error("encode_reference_voice: Mel preprocessing produced no output");
		return {};
	}

	// ── Audition audio encode ─────────────────────────────────────────────────

	/// `audition_audio_f32_init` allocates the C struct that wraps the Mel
	/// spectrogram buffer expected by the encoder.
	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel[0].n_len;
	mel_f32->ny = mel[0].n_mel;
	mel_f32->buf.assign(mel[0].data.begin(), mel[0].data.end());

	/// `audition_n_output_tokens` computes the number of embedding tokens the
	/// encoder will produce for this input length.
	int const n_tokens = audition_n_output_tokens(ctx, mel_f32);

	/// `audition_n_mmproj_embd` returns the embedding dimension expected by the
	/// LLM's multimodal projector — must match the LLM's hidden size (4096).
	int const n_embd = audition_n_mmproj_embd(ctx);

	spdlog::debug(
		"encode_reference_voice: n_tokens={} n_embd={} wav='{}'",
		n_tokens,
		n_embd,
		wav_path.string());

	/// The Whisper encoder runs in streaming mode by default, accumulating
	/// `iter` across calls.  The reference voice is a one-shot encode and
	/// must start from a clean KV state, otherwise position encoding offsets
	/// are wrong when multiple sessions have previously used the shared
	/// `audition_ctx`.  Clearing here guarantees reproducible embeddings.
	audition_whisper_clear_kv_cache(ctx);

	// Over-allocate by 2 tokens: `audition_n_output_tokens` can under-predict
	// by 1 ("token count mismatch within tolerance, diff=1"), which would
	// cause a heap overwrite if we allocated exactly n_tokens.
	std::vector<float> embeddings(static_cast<std::size_t>((n_tokens + 2) * n_embd));

	/// `audition_audio_encode` runs the Whisper encoder forward pass on `gpu_ex`
	/// (GGML/CUDA), writing `n_tokens * n_embd` floats into `embeddings.data()`.
	bool const encode_ok = audition_audio_encode(ctx, n_threads, mel_f32, embeddings.data());
	audition_audio_f32_free(mel_f32);

	if (!encode_ok)
	{
		spdlog::error("encode_reference_voice: audition_audio_encode failed");
		return {};
	}

	// Trim back to the predicted size (discard the 2-token over-allocation guard).
	// Cast each operand before multiplying to avoid signed overflow then widening.
	embeddings.resize(static_cast<std::size_t>(n_tokens) * static_cast<std::size_t>(n_embd));
	return embeddings;
}

// This definition implements the public header declaration consumed by Session,
// and `pcm` / `n_threads` are intentionally adjacent despite distinct roles.
// NOLINTNEXTLINE(misc-use-internal-linkage,bugprone-easily-swappable-parameters)
std::vector<float> encode_pcm_audio(
	audition_ctx * ctx, std::span<float const> pcm, int n_threads)
{
	if (ctx == nullptr)
	{
		spdlog::error("encode_pcm_audio: audition_ctx is null");
		throw std::runtime_error{"encode_pcm_audio: audition_ctx is null"};
	}

	if (pcm.empty())
	{
		spdlog::error("encode_pcm_audio: empty PCM buffer");
		throw std::runtime_error{"encode_pcm_audio: empty PCM buffer"};
	}

	// ── Whisper Mel spectrogram preprocessing ────────────────────────────────

	/// `audition_get_mel_filters` returns the 80-band Mel filter matrix used
	/// by the Whisper encoder.  The filters are stored in the loaded model.
	auto const filters = audition_get_mel_filters(ctx);

	/// `preprocess_audio` converts raw PCM to a Mel spectrogram.
	/// Chunked into 30-second windows; the first window is used.
	std::vector<whisper_preprocessor::whisper_mel> mel;
	whisper_preprocessor::preprocess_audio(pcm.data(), pcm.size(), filters, mel);
	if (mel.empty())
	{
		spdlog::error("encode_pcm_audio: Mel preprocessing produced no output");
		throw std::runtime_error{"encode_pcm_audio: Mel preprocessing produced no output"};
	}

	// ── Audition audio encode ─────────────────────────────────────────────────

	/// `audition_audio_f32_init` allocates the C struct wrapping the Mel buffer.
	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel[0].n_len;
	mel_f32->ny = mel[0].n_mel;
	mel_f32->buf.assign(mel[0].data.begin(), mel[0].data.end());

	/// `audition_n_output_tokens` computes the number of embedding tokens the
	/// encoder will produce for this input length.
	int const n_tokens = audition_n_output_tokens(ctx, mel_f32);

	/// `audition_n_mmproj_embd` returns the embedding dimension (4096).
	int const n_embd = audition_n_mmproj_embd(ctx);

	spdlog::debug(
		"encode_pcm_audio: n_tokens={} n_embd={} pcm_samples={}", n_tokens, n_embd, pcm.size());

	// Over-allocate by 2 tokens (same guard as encode_reference_voice).
	std::vector<float> embeddings(static_cast<std::size_t>((n_tokens + 2) * n_embd));

	/// `audition_audio_encode` runs the Whisper encoder forward pass on the GPU,
	/// writing `n_tokens * n_embd` floats into `embeddings.data()`.
	bool const encode_ok = audition_audio_encode(ctx, n_threads, mel_f32, embeddings.data());
	audition_audio_f32_free(mel_f32);

	if (!encode_ok)
	{
		spdlog::error("encode_pcm_audio: audition_audio_encode failed");
		throw std::runtime_error{"encode_pcm_audio: audition_audio_encode failed"};
	}

	// Trim back to the predicted size.
	embeddings.resize(static_cast<std::size_t>(n_tokens) * static_cast<std::size_t>(n_embd));
	return embeddings;
}

}  // namespace llama_omni_server
