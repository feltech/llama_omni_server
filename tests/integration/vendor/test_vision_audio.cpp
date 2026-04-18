/*
 * Vision and Audio Encoder Integration Tests for MiniCPM-o-4_5.
 *
 * This test demonstrates complete multimodal encoding using:
 * - libmtmd CLIP for vision encoding (zero omni dependencies)
 * - vendor/omni audition for audio encoding (minimal omni extract)
 *
 * Prerequisites:
 * - Vision model: models/gguf/vision/MiniCPM-o-4_5-vision-F16.gguf
 * - Audio model: models/gguf/audio/MiniCPM-o-4_5-audio-F16.gguf
 * - LLM model: models/gguf/MiniCPM-o-4_5-Q4_K_M.gguf
 * - Test images: test_data/circle.png
 */

#include <catch2/catch_test_macros.hpp>
#include "llama.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <sndfile.h>

// Vision encoder from libmtmd (CLIP)
#include "clip.h"

// Audio encoder from vendor (minimal omni extract)
#include "audition.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vendor_paths.hpp"

/* Test: Vision encoder integration (using libmtmd CLIP) */
TEST_CASE("vision_encoder_integration", "[vision][encoder]")
{
	std::cout << "=== Vision Encoder Integration Test ===" << std::endl;

	/* Step 1: Initialize CLIP vision encoder */
	std::cout << "Step 1: Initializing vision encoder..." << std::endl;
	clip_context_params clip_params = {true, GGML_LOG_LEVEL_INFO};
	VendorPath vision_model_path{vp_model_root() + "/vision/MiniCPM-o-4_5-vision-F16.gguf"};

	clip_init_result result = clip_init(vision_model_path, clip_params);
	clip_ctx * ctx_clip = result.ctx_v;
	REQUIRE(ctx_clip != nullptr);
	std::cout << "  ✓ Vision encoder initialized" << std::endl;

	/* Step 2: Load and process test image */
	std::cout << "Step 2: Loading test image..." << std::endl;
	VendorPath image_path{vp_test_data() + "/circle.png"};

	// Load image using stb_image
	int width, height, channels;
	unsigned char * img_data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
	REQUIRE(img_data != nullptr);
	std::cout << "  ✓ Image loaded: " << width << "x" << height << "x" << channels << std::endl;

	/* Step 3: Create CLIP image structure */
	std::cout << "Step 3: Converting to CLIP format..." << std::endl;
	clip_image_u8 * img_u8 = clip_image_u8_init();
	clip_build_img_from_pixels(img_data, width, height, img_u8);

	/* Step 4: Preprocess image */
	std::cout << "Step 4: Preprocessing image..." << std::endl;
	clip_image_f32_batch * batch = clip_image_f32_batch_init();
	clip_image_preprocess(ctx_clip, img_u8, batch);
	size_t n_images = clip_image_f32_batch_n_images(batch);
	REQUIRE(n_images > 0);
	std::cout << "  ✓ Image preprocessed: " << n_images << " chunks" << std::endl;

	/* Step 5: Encode image to embeddings */
	std::cout << "Step 5: Encoding image to embeddings..." << std::endl;
	clip_image_f32 * first_img = clip_image_f32_get_img(batch, 0);
	int n_tokens = clip_n_output_tokens(ctx_clip, first_img);
	int n_embd = clip_n_mmproj_embd(ctx_clip);

	std::cout << "  Model info: n_tokens=" << n_tokens << std::endl;

	// Allocate buffer for embeddings
	int total_tokens = n_images * n_tokens;
	std::vector<float> embeddings(total_tokens * n_embd);

	bool encode_ok = clip_image_batch_encode(ctx_clip, 4, batch, embeddings.data());
	REQUIRE(encode_ok);
	std::cout << "  ✓ Encoded successfully" << std::endl;

	std::cout << "  ✓ Total embeddings: " << embeddings.size() << " floats" << std::endl;

	/* Step 6: Verify embedding format */
	std::cout << "Step 6: Verifying embedding format..." << std::endl;
	REQUIRE(total_tokens > 0);
	REQUIRE(embeddings.size() == (size_t)total_tokens * n_embd);

	// Check for NaN or inf values
	bool has_invalid = false;
	for (float val : embeddings)
	{
		if (std::isnan(val) || std::isinf(val))
		{
			has_invalid = true;
			break;
		}
	}
	REQUIRE(!has_invalid);
	std::cout << "  ✓ Embeddings are valid (no NaN/inf)" << std::endl;

	/* Step 7: Cleanup */
	std::cout << "Step 7: Cleaning up..." << std::endl;
	clip_image_f32_batch_free(batch);
	clip_image_u8_free(img_u8);
	clip_free(ctx_clip);
	stbi_image_free(img_data);

	std::cout << "=== Vision encoder integration test completed ===" << std::endl;
}

/* Test: Audio encoder integration */
TEST_CASE("audio_encoder_integration", "[audio][encoder]")
{
	std::cout << "=== Audio Encoder Integration Test ===" << std::endl;

	/* Step 1: Initialize audio encoder */
	std::cout << "Step 1: Initializing audio encoder..." << std::endl;
	audition_context_params audio_params = {true, GGML_LOG_LEVEL_INFO};
	VendorPath audio_model_path{vp_model_root() + "/audio/MiniCPM-o-4_5-audio-F16.gguf"};

	audition_ctx * ctx_audio = audition_init(audio_model_path, audio_params);
	REQUIRE(ctx_audio != nullptr);
	std::cout << "  ✓ Audio encoder initialized" << std::endl;

	/* Step 2: Load audio file */
	std::cout << "Step 2: Loading audio file..." << std::endl;
	VendorPath audio_path{vp_test_data() + "/say_the_word_banana.wav"};

	SF_INFO sfinfo;
	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	REQUIRE(sndfile != nullptr);
	std::cout << "  ✓ Audio file loaded" << std::endl;
	std::cout << "  Audio info: " << sfinfo.frames << " frames, " << sfinfo.channels
			  << " channels, " << sfinfo.samplerate << " Hz" << std::endl;

	/* Step 3: Read and convert audio samples */
	std::cout << "Step 3: Reading and converting audio..." << std::endl;

	// Read audio samples as float
	std::vector<float> samples(sfinfo.frames * sfinfo.channels);
	sf_count_t frames_read = sf_readf_float(sndfile, samples.data(), sfinfo.frames);
	REQUIRE(frames_read == sfinfo.frames);
	sf_close(sndfile);
	std::cout << "  ✓ Audio samples read: " << frames_read << " samples" << std::endl;

	// Convert to mono if needed
	std::vector<float> mono_samples;
	if (sfinfo.channels == 1)
	{
		mono_samples = std::move(samples);
	}
	else
	{
		mono_samples.resize(sfinfo.frames);
		for (size_t i = 0; i < mono_samples.size(); i++)
		{
			mono_samples[i] = samples[i * sfinfo.channels];	 // Use first channel
		}
	}

	/* Step 4: Preprocess audio */
	std::cout << "Step 4: Preprocessing audio..." << std::endl;

	// Use vendor audition's preprocessing
	auto filters = audition_get_mel_filters(ctx_audio);
	std::vector<whisper_preprocessor::whisper_mel> mel_output;
	whisper_preprocessor::preprocess_audio(
		mono_samples.data(), mono_samples.size(), filters, mel_output);

	REQUIRE(!mel_output.empty());
	std::cout << "  ✓ Mel spectrogram created: " << mel_output.size() << " segments" << std::endl;

	// Convert to audition format for encoding
	auto const & mel_segment = mel_output[0];
	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel_segment.n_len;
	mel_f32->ny = mel_segment.n_mel;
	mel_f32->buf.resize(mel_segment.data.size());
	std::copy(mel_segment.data.begin(), mel_segment.data.end(), mel_f32->buf.begin());

	std::cout << "  ✓ Mel spectrogram: " << mel_f32->nx << "x" << mel_f32->ny << std::endl;

	/* Step 5: Encode audio to embeddings */
	std::cout << "Step 5: Encoding audio to embeddings..." << std::endl;
	int n_tokens = audition_n_output_tokens(ctx_audio, mel_f32);
	int n_embd = audition_n_mmproj_embd(ctx_audio);

	std::cout << "  Model info: n_tokens=" << n_tokens << std::endl;

	// Allocate buffer for embeddings
	std::vector<float> embeddings(n_tokens * n_embd);

	bool encode_ok = audition_audio_encode(ctx_audio, 4, mel_f32, embeddings.data());
	REQUIRE(encode_ok);
	std::cout << "  ✓ Audio encoded: " << n_tokens << " tokens" << std::endl;
	std::cout << "  ✓ Total embeddings: " << embeddings.size() << " floats" << std::endl;

	/* Step 6: Verify embedding format */
	std::cout << "Step 6: Verifying embedding format..." << std::endl;

	// Check for NaN or inf values
	bool has_invalid = false;
	for (float val : embeddings)
	{
		if (std::isnan(val) || std::isinf(val))
		{
			has_invalid = true;
			break;
		}
	}
	REQUIRE(!has_invalid);
	std::cout << "  ✓ Embeddings are valid (no NaN/inf)" << std::endl;

	/* Step 7: Cleanup */
	std::cout << "Step 7: Cleaning up..." << std::endl;
	audition_audio_f32_free(mel_f32);
	audition_free(ctx_audio);

	std::cout << "=== Audio encoder integration test completed ===" << std::endl;
}
