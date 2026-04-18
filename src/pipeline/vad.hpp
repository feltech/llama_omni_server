/**
 * @file   vad.hpp
 * @brief  Audio chunking and classification types.
 *
 * These types are consumed by `Session::loop_audio_in_ch_buffered()`,
 * which yields `AudioChunkEvent` driven from the `audio_in_ch_` channel.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace llama_omni_server
{

/**
 * @brief A raw audio chunk buffered from the client.
 */
// The PCM vector is empty by default via std::vector's default constructor.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
struct AudioChunkEvent
{
	std::vector<float> pcm;	   ///< Mono 16 kHz float32 samples.
	int duration_ms{0};		   ///< Approximate duration in milliseconds.
};

// ── RMS energy ────────────────────────────────────────────────────────────────

/**
 * @brief Compute the root-mean-square energy of a PCM buffer.
 *
 * Returns 0.0 for an empty buffer.  Used by `Session::vad_stream()` to
 * classify each incoming audio chunk against the VAD speech threshold.
 *
 * @param pcm  Mono float32 samples (any sample rate).
 * @return     RMS energy in the range [0.0, ∞).
 */
[[nodiscard]] constexpr float rms_energy(std::span<float const> pcm) noexcept
{
	if (pcm.empty())
	{
		return 0.0F;
	}
	float const sum_sq = std::inner_product(
		pcm.begin(), pcm.end(), pcm.begin(), 0.0F, std::plus<>{}, std::multiplies<>{});
	return std::sqrt(sum_sq / static_cast<float>(pcm.size()));
}

}  // namespace llama_omni_server
