/**
 * @file   wav.hpp
 * @brief  Shared WAV read/write helpers built on libsndfile.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace llama_omni_server
{

/**
 * @brief Basic metadata returned when inspecting a WAV file.
 */
struct WavInfo
{
	int sample_rate{0};
	int channels{0};
	std::int64_t frames{0};
};

/**
 * @brief Mono float32 PCM loaded from a WAV file.
 */
struct MonoWavData
{
	std::vector<float> pcm;
	WavInfo info{};
};

/**
 * @brief Load a WAV file and return the first channel as mono float32 PCM.
 *
 * @param path  Path to the WAV file.
 * @return PCM samples plus source metadata.
 *
 * @throws std::runtime_error if the file cannot be opened or read.
 */
[[nodiscard]] MonoWavData read_wav_mono(std::filesystem::path const & path);

/**
 * @brief Read only the metadata from a WAV file.
 *
 * @param path  Path to the WAV file.
 * @return Sample rate, channel count, and frame count.
 *
 * @throws std::runtime_error if the file cannot be opened.
 */
[[nodiscard]] WavInfo read_wav_info(std::filesystem::path const & path);

/**
 * @brief Write mono float32 PCM to a WAV file.
 *
 * @param path         Destination path.
 * @param pcm          Mono float32 PCM samples.
 * @param sample_rate  Sample rate in Hz.
 *
 * @throws std::runtime_error if the file cannot be opened for writing.
 */
void write_wav_mono(
	std::filesystem::path const & path, std::span<float const> pcm, int sample_rate);

}  // namespace llama_omni_server
