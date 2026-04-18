/**
 * @file   audio_encoder.hpp
 * @brief  Audio-to-embedding helpers wrapping the Whisper audio encoder.
 *
 * Two public entry points:
 * - `encode_reference_voice`: loads a WAV file and encodes it (system prompt).
 * - `encode_pcm_audio`: encodes a raw PCM buffer (streaming duplex audio).
 *
 * Both functions must be called on the GPU executor because
 * `audition_audio_encode` dispatches GGML/CUDA compute.
 */

#pragma once

#include <filesystem>
#include <span>
#include <vector>

// Forward-declare opaque C struct — include omni/audition.h only in audio_encoder.cpp.
// NOLINTNEXTLINE(modernize-use-using,bugprone-forward-declaration-namespace)
struct audition_ctx;

/// Input audio should be padded to minimum/multiple of 1600 samples for Whisper encoder.
constexpr std::size_t kInputAudioEmbeddingAlignmentSamples{1600};

namespace llama_omni_server
{

/**
 * @brief Encode a 16 kHz WAV file to LLM-compatible float embeddings.
 *
 * Loads `wav_path` with libsndfile, down-mixes stereo to mono (first channel),
 * applies the Whisper Mel spectrogram preprocessor via `audition_get_mel_filters`
 * and `whisper_preprocessor::preprocess_audio`, then calls
 * `audition_audio_encode` to produce the final `[n_tokens × n_embd]` embeddings.
 *
 * On any failure the function logs an error and returns an empty vector.  The
 * caller must check `result.empty()` before using it.
 *
 * @param ctx       Loaded audition context (from `ModelManager::audio_ctx()`).
 *                  Must not be null.
 * @param wav_path  Path to a 16 kHz WAV file.  Stereo is silently down-mixed.
 * @param n_threads Number of CPU threads for the Whisper encoder forward pass.
 * @return Flat `[n_tokens * n_embd]` float32 buffer, or empty on failure.
 */
[[nodiscard]] std::vector<float> encode_reference_voice(
	audition_ctx * ctx, std::filesystem::path const & wav_path, int n_threads = 4);

/**
 * @brief Encode a raw mono 16 kHz PCM buffer to LLM-compatible float embeddings.
 *
 * The same Mel-spectrogram → Whisper encoder pipeline as `encode_reference_voice`,
 * but takes an already-decoded float buffer instead of a WAV file.  Used for
 * streaming duplex audio where PCM arrives over the audio input channel.
 *
 * On any failure the function logs an error and throws `std::runtime_error`.
 *
 * @param ctx       Loaded audition context (from `ModelManager::audio_ctx()`).
 *                  Must not be null.
 * @param pcm       Mono float32 PCM samples at 16 kHz.
 * @param n_threads Number of CPU threads for the Whisper encoder forward pass.
 * @return Flat `[n_tokens * n_embd]` float32 buffer.
 * @throws std::runtime_error if preprocessing or encoder inference fails.
 */
[[nodiscard]] std::vector<float> encode_pcm_audio(
	audition_ctx * ctx, std::span<float const> pcm, int n_threads = 4);

}  // namespace llama_omni_server
