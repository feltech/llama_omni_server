/*
 * Audio processing tests for MiniCPM-o-4_5 multimodal inference.
 *
 * This test file demonstrates audio input processing for the MiniCPM-o model
 * using only standard libraries (libsndfile, FFmpeg) and llama.cpp core API.
 *
 * Audio processing pipeline:
 * 1. Load audio file using libsndfile
 * 2. Resample to model's expected sample rate (typically 16kHz)
 * 3. Convert to model's expected format (typically float32)
 * 4. Process through audio encoder (future work)
 * 5. Submit audio embeddings to llama.cpp context
 *
 * Prerequisites:
 * - libsndfile for audio I/O
 * - FFmpeg for audio format conversion and resampling
 * - Sample audio files for testing
 */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <sndfile.h>
#include <string>
#include <vector>

#include "vendor_paths.hpp"

/* Test basic libsndfile functionality */
TEST_CASE("audio_libsndfile_init", "[audio][sndfile]")
{
	/* Query libsndfile version to verify it's available */
	char const * version = sf_version_string();
	REQUIRE(version != nullptr);
	std::cout << "libsndfile version: " << version << std::endl;
}

/* Test loading an audio file */
TEST_CASE("audio_load_wav_file", "[audio][load]")
{
	/* Use a sample audio file from the demo output directory */
	VendorPath audio_path{vp_test_data() + "/say_the_word_banana.wav"};

	std::cout << "=== Audio Loading Test ===" << std::endl;
	std::cout << "Loading: " << audio_path << std::endl;

	/* Step 1: Open audio file */
	SF_INFO sfinfo;
	sfinfo.format = 0;	// Must be set to 0 before sf_open

	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	if (sndfile == nullptr)
	{
		std::cerr << "  ✗ Failed to open audio file: " << sf_strerror(nullptr) << std::endl;
		FAIL("Failed to open audio file");
	}
	std::cout << "  ✓ Audio file opened" << std::endl;

	/* Step 2: Read audio metadata */
	std::cout << "Audio metadata:" << std::endl;
	std::cout << "  Sample rate: " << sfinfo.samplerate << " Hz" << std::endl;
	std::cout << "  Channels: " << sfinfo.channels << std::endl;
	std::cout << "  Frames: " << sfinfo.frames << std::endl;
	std::cout << "  Duration: " << (double)sfinfo.frames / sfinfo.samplerate << " seconds"
			  << std::endl;

	REQUIRE(sfinfo.samplerate > 0);
	REQUIRE(sfinfo.channels > 0);
	REQUIRE(sfinfo.frames > 0);

	/* Step 3: Close file */
	sf_close(sndfile);
	std::cout << "  ✓ Audio file closed" << std::endl;

	std::cout << "=== Audio loading test passed ===" << std::endl;
}

/* Test reading audio samples */
TEST_CASE("audio_read_samples", "[audio][samples]")
{
	VendorPath audio_path{vp_test_data() + "/say_the_word_banana.wav"};

	std::cout << "=== Audio Sample Reading Test ===" << std::endl;

	/* Step 1: Open audio file */
	SF_INFO sfinfo;
	sfinfo.format = 0;
	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	REQUIRE(sndfile != nullptr);
	std::cout << "  ✓ Audio file opened" << std::endl;

	/* Step 2: Allocate buffer for samples */
	// Read first 1 second of audio or entire file if shorter
	sf_count_t samples_to_read = std::min(
		(sf_count_t)(sfinfo.samplerate * sfinfo.channels), sfinfo.frames * sfinfo.channels);
	std::vector<float> samples(samples_to_read);

	std::cout << "  Reading " << samples_to_read << " samples..." << std::endl;

	/* Step 3: Read samples as float32 */
	// sf_readf_float reads frames (samples across all channels)
	sf_count_t frames_read =
		sf_readf_float(sndfile, samples.data(), samples_to_read / sfinfo.channels);

	if (frames_read <= 0)
	{
		std::cerr << "  ✗ Failed to read samples" << std::endl;
		sf_close(sndfile);
		FAIL("Failed to read samples");
	}

	std::cout << "  ✓ Read " << frames_read << " frames (" << frames_read * sfinfo.channels
			  << " samples)" << std::endl;

	/* Step 4: Analyze samples */
	// Calculate RMS (root mean square) to verify we have actual audio data
	double rms = 0.0;
	for (sf_count_t i = 0; i < frames_read * sfinfo.channels; i++)
	{
		rms += samples[i] * samples[i];
	}
	rms = std::sqrt(rms / (frames_read * sfinfo.channels));

	std::cout << "  RMS amplitude: " << rms << std::endl;
	REQUIRE(rms > 0.0);	 // Verify we have non-silent audio

	/* Step 5: Close file */
	sf_close(sndfile);
	std::cout << "  ✓ Audio file closed" << std::endl;

	std::cout << "=== Audio sample reading test passed ===" << std::endl;
}

/* Test converting audio to mono (if stereo) */
TEST_CASE("audio_convert_to_mono", "[audio][convert]")
{
	VendorPath audio_path{vp_test_data() + "/say_the_word_banana.wav"};

	std::cout << "=== Audio Mono Conversion Test ===" << std::endl;

	/* Step 1: Load audio */
	SF_INFO sfinfo;
	sfinfo.format = 0;
	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	REQUIRE(sndfile != nullptr);

	std::cout << "  Original channels: " << sfinfo.channels << std::endl;

	/* Step 2: Read all samples */
	std::vector<float> samples(sfinfo.frames * sfinfo.channels);
	sf_count_t frames_read = sf_readf_float(sndfile, samples.data(), sfinfo.frames);
	REQUIRE(
		frames_read > 0);  // Should read some frames (may be slightly less than metadata suggests)
	samples.resize(frames_read * sfinfo.channels);	// Adjust size to actual frames read

	/* Step 3: Convert to mono by averaging channels */
	std::vector<float> mono_samples(frames_read);

	if (sfinfo.channels == 1)
	{
		// Already mono
		mono_samples = samples;
		std::cout << "  ✓ Audio is already mono" << std::endl;
	}
	else
	{
		// Average all channels
		for (sf_count_t i = 0; i < frames_read; i++)
		{
			float sum = 0.0f;
			for (int ch = 0; ch < sfinfo.channels; ch++)
			{
				sum += samples[i * sfinfo.channels + ch];
			}
			mono_samples[i] = sum / sfinfo.channels;
		}
		std::cout << "  ✓ Converted " << sfinfo.channels << " channels to mono" << std::endl;
	}

	/* Step 4: Verify mono samples */
	REQUIRE(mono_samples.size() == (size_t)frames_read);

	// Calculate RMS of mono audio
	double rms = 0.0;
	for (size_t i = 0; i < mono_samples.size(); i++)
	{
		rms += mono_samples[i] * mono_samples[i];
	}
	rms = std::sqrt(rms / mono_samples.size());
	std::cout << "  Mono RMS amplitude: " << rms << std::endl;
	REQUIRE(rms > 0.0);

	/* Step 5: Close file */
	sf_close(sndfile);

	std::cout << "=== Mono conversion test passed ===" << std::endl;
}

/* Test audio normalization */
TEST_CASE("audio_normalize", "[audio][normalize]")
{
	VendorPath audio_path{vp_test_data() + "/say_the_word_banana.wav"};

	std::cout << "=== Audio Normalization Test ===" << std::endl;

	/* Step 1: Load audio */
	SF_INFO sfinfo;
	sfinfo.format = 0;
	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	REQUIRE(sndfile != nullptr);

	/* Step 2: Read samples */
	std::vector<float> samples(sfinfo.frames * sfinfo.channels);
	sf_count_t frames_read = sf_readf_float(sndfile, samples.data(), sfinfo.frames);
	REQUIRE(frames_read > 0);
	samples.resize(frames_read * sfinfo.channels);	// Adjust to actual size
	sf_close(sndfile);

	/* Step 3: Find peak amplitude */
	float peak = 0.0f;
	for (size_t i = 0; i < samples.size(); i++)
	{
		peak = std::max(peak, std::abs(samples[i]));
	}
	std::cout << "  Peak amplitude: " << peak << std::endl;
	REQUIRE(peak > 0.0f);

	/* Step 4: Normalize to [-1.0, 1.0] range */
	std::vector<float> normalized_samples(samples.size());
	for (size_t i = 0; i < samples.size(); i++)
	{
		normalized_samples[i] = samples[i] / peak;
	}
	std::cout << "  ✓ Samples normalized" << std::endl;

	/* Step 5: Verify normalization */
	float normalized_peak = 0.0f;
	for (size_t i = 0; i < normalized_samples.size(); i++)
	{
		normalized_peak = std::max(normalized_peak, std::abs(normalized_samples[i]));
	}
	std::cout << "  Normalized peak: " << normalized_peak << std::endl;
	REQUIRE(normalized_peak <= 1.0f);
	REQUIRE(normalized_peak >= 0.99f);	// Should be very close to 1.0

	std::cout << "=== Normalization test passed ===" << std::endl;
}

/* Test preparing audio for model input */
TEST_CASE("audio_prepare_for_model", "[audio][model]")
{
	VendorPath audio_path{vp_test_data() + "/say_the_word_banana.wav"};

	std::cout << "=== Audio Model Preparation Test ===" << std::endl;

	/*
	 * MiniCPM-o audio encoder typically expects:
	 * - Sample rate: 16000 Hz (16kHz)
	 * - Format: float32
	 * - Channels: 1 (mono)
	 * - Normalized: [-1.0, 1.0] range
	 */
	int const target_sample_rate = 16000;

	/* Step 1: Load audio */
	SF_INFO sfinfo;
	sfinfo.format = 0;
	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	REQUIRE(sndfile != nullptr);

	std::cout << "  Input audio:" << std::endl;
	std::cout << "    Sample rate: " << sfinfo.samplerate << " Hz" << std::endl;
	std::cout << "    Channels: " << sfinfo.channels << std::endl;
	std::cout << "    Frames: " << sfinfo.frames << std::endl;

	/* Step 2: Read all samples */
	std::vector<float> samples(sfinfo.frames * sfinfo.channels);
	sf_count_t frames_read = sf_readf_float(sndfile, samples.data(), sfinfo.frames);
	REQUIRE(
		frames_read > 0);  // Should read some frames (may be slightly less than metadata suggests)
	samples.resize(frames_read * sfinfo.channels);	// Adjust size to actual frames read
	sf_close(sndfile);

	/* Step 3: Convert to mono */
	std::vector<float> mono_samples(frames_read);
	if (sfinfo.channels == 1)
	{
		mono_samples = samples;
	}
	else
	{
		for (sf_count_t i = 0; i < frames_read; i++)
		{
			float sum = 0.0f;
			for (int ch = 0; ch < sfinfo.channels; ch++)
			{
				sum += samples[i * sfinfo.channels + ch];
			}
			mono_samples[i] = sum / sfinfo.channels;
		}
	}
	std::cout << "  ✓ Converted to mono" << std::endl;

	/* Step 4: Normalize */
	float peak = 0.0f;
	for (size_t i = 0; i < mono_samples.size(); i++)
	{
		peak = std::max(peak, std::abs(mono_samples[i]));
	}
	for (size_t i = 0; i < mono_samples.size(); i++)
	{
		mono_samples[i] /= peak;
	}
	std::cout << "  ✓ Normalized to [-1.0, 1.0]" << std::endl;

	/* Step 5: Resample if needed (note: actual resampling requires more complex code) */
	// For now, we'll just check if resampling is needed
	// TODO: Implement proper resampling using FFmpeg's libswresample
	if (sfinfo.samplerate != target_sample_rate)
	{
		std::cout << "  ⚠ Resampling from " << sfinfo.samplerate << " Hz to " << target_sample_rate
				  << " Hz would be needed" << std::endl;
		std::cout << "  (Resampling not yet implemented)" << std::endl;
	}
	else
	{
		std::cout << "  ✓ Sample rate already matches target (" << target_sample_rate << " Hz)"
				  << std::endl;
	}

	/* Step 6: Summary */
	std::cout << "  Model-ready audio:" << std::endl;
	std::cout << "    Samples: " << mono_samples.size() << std::endl;
	std::cout << "    Duration: " << (double)mono_samples.size() / sfinfo.samplerate << " seconds"
			  << std::endl;
	std::cout << "    Format: float32, mono, normalized" << std::endl;

	REQUIRE(mono_samples.size() > 0);

	std::cout << "=== Model preparation test passed ===" << std::endl;
}
