/**
 * @file   tts_sampling.hpp
 * @brief  Internal pure helper functions for TTS token sampling.
 *
 * These helpers implement the model-independent math used by
 * `TtsPipeline::synthesize()`.  They are declared in a private header so that
 * unit tests can exercise them without loading any models.
 */

#pragma once

#include <random>
#include <span>
#include <vector>

namespace llama_omni_server::tts_detail
{

/**
 * @brief Compute logits for the next audio token via head_code matmul.
 *
 * @param head_code_w     Pointer to a row-major weight matrix
 *                        `[n_audio_tokens × hidden_size]`.
 * @param hidden          Pointer to the current hidden state `[hidden_size]`.
 * @param n_audio_tokens  Number of rows / output logits.
 * @param hidden_size     Width of the hidden state.
 * @return Logit vector of length `n_audio_tokens`.
 */
[[nodiscard]] std::vector<float> compute_audio_logits(
	float const * head_code_w, float const * hidden, int n_audio_tokens, int hidden_size);

/**
 * @brief Apply temperature scaling to logits in place.
 *
 * Divides each logit by `temperature`, leaving the vector unchanged when the
 * caller has already handled degenerate temperatures before calling.
 *
 * @param logits        Mutable logit vector to rescale.
 * @param temperature   Sampling temperature used by the TTS sampler.
 */
void apply_temperature(std::span<float> logits, float temperature);

/**
 * @brief Apply repetition penalty to logits for recently generated tokens.
 *
 * Positive logits are divided by `rep_penalty`; negative logits are multiplied,
 * matching the omni.cpp reference implementation.
 */
void apply_rep_penalty(
	std::span<float> logits,
	std::span<int const> recent_relative,
	int penalty_last_n,
	int n_audio_tokens,
	float rep_penalty);

/**
 * @brief Convert logits to probabilities via softmax.
 *
 * Computes a numerically stable softmax over the provided logits and returns
 * a new probability vector with one entry per input token.
 *
 * @param logits  Input logits for the current sampling step.
 * @return Probability vector with the same length as `logits`.
 */
[[nodiscard]] std::vector<float> softmax_probs(std::span<float const> logits);

/**
 * @brief Apply top-k filtering in place.
 *
 * Retains only the `top_k` highest-probability entries and zeroes all others.
 *
 * @param probs     Probability vector to filter in place.
 * @param top_k     Number of tokens to retain.
 * @param n_tokens  Logical token count to consider from `probs`.
 */
void apply_top_k(std::span<float> probs, int top_k, int n_tokens);

/**
 * @brief Apply top-p (nucleus) filtering in place.
 *
 * Sorts candidate tokens by descending probability and zeroes all entries
 * after the smallest prefix whose cumulative mass reaches `top_p`.
 *
 * @param probs      Probability vector to filter in place.
 * @param top_p      Cumulative probability threshold in `(0, 1]`.
 * @param n_tokens   Logical token count to consider from `probs`.
 */
void apply_top_p(std::span<float> probs, float top_p, int n_tokens);

/**
 * @brief Re-normalise probabilities so they sum to 1.
 *
 * Used after top-k / top-p filtering to restore a valid categorical
 * distribution before multinomial sampling.
 *
 * @param probs  Probability vector to renormalise in place.
 */
void renormalize(std::span<float> probs);

/**
 * @brief Draw one sample from a categorical distribution via linear scan.
 *
 * @return Relative token index in `[0, n_tokens)`.
 */
[[nodiscard]] int multinomial_sample(
	std::span<float const> probs, std::mt19937 & rng, int n_tokens);

}  // namespace llama_omni_server::tts_detail
