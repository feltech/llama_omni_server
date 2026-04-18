/**
 * @file   wav.cpp
 * @brief  Shared WAV read/write helpers built on libsndfile.
 */

#include "wav.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>

#include <sndfile.h>

namespace llama_omni_server
{

namespace
{

[[nodiscard]] SNDFILE * open_sndfile(
	std::filesystem::path const & path,
	int mode,
	SF_INFO & info,
	char const * operation)
{
	// libsndfile exposes a C vararg-style API and returns a mutable handle that
	// is later passed to sf_close / sf_readf_float / sf_writef_float.
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,misc-const-correctness)
	SNDFILE * const sndfile = sf_open(path.c_str(), mode, &info);
	if (sndfile == nullptr)
	{
		throw std::runtime_error{
			std::format(
				"wav: failed to {} '{}': {}",
				operation,
				path.string(),
				sf_strerror(nullptr))};
	}
	return sndfile;
}

}  // namespace

MonoWavData read_wav_mono(std::filesystem::path const & path)
{
	SF_INFO info{};
	SNDFILE * const sndfile = open_sndfile(path, SFM_READ, info, "open for reading");

	std::vector<float> interleaved(
		static_cast<std::size_t>(info.frames) * static_cast<std::size_t>(info.channels), 0.0F);
	sf_count_t const frames_read = sf_readf_float(sndfile, interleaved.data(), info.frames);
	sf_close(sndfile);

	if (frames_read != info.frames)
	{
		throw std::runtime_error{std::format(
			"wav: short read from '{}': expected {} frames, got {}",
			path.string(),
			static_cast<std::int64_t>(info.frames),
			static_cast<std::int64_t>(frames_read))};
	}

	std::vector<float> mono(static_cast<std::size_t>(info.frames), 0.0F);
	for (sf_count_t frame = 0; frame < info.frames; ++frame)
	{
		mono[static_cast<std::size_t>(frame)] =
			interleaved[static_cast<std::size_t>(frame) * static_cast<std::size_t>(info.channels)];
	}

	return MonoWavData{
		.pcm = std::move(mono),
		.info = WavInfo{
			.sample_rate = info.samplerate,
			.channels = info.channels,
			.frames = static_cast<std::int64_t>(info.frames)}};
}

WavInfo read_wav_info(std::filesystem::path const & path)
{
	SF_INFO info{};
	SNDFILE * const sndfile = open_sndfile(path, SFM_READ, info, "open for reading");
	sf_close(sndfile);
	return WavInfo{
		.sample_rate = info.samplerate,
		.channels = info.channels,
		.frames = static_cast<std::int64_t>(info.frames)};
}

void write_wav_mono(std::filesystem::path const & path, std::span<float const> pcm, int sample_rate)
{
	SF_INFO info{};
	info.samplerate = sample_rate;
	info.channels = 1;
	info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

	SNDFILE * const sndfile = open_sndfile(path, SFM_WRITE, info, "open for writing");
	sf_writef_float(sndfile, pcm.data(), static_cast<sf_count_t>(pcm.size()));
	sf_close(sndfile);
}

}  // namespace llama_omni_server
