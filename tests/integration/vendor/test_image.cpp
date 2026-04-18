/*
 * Image processing tests for MiniCPM-o-4_5 multimodal inference.
 *
 * This test file demonstrates image input processing for the MiniCPM-o model
 * using stb_image library and llama.cpp core API.
 *
 * Image processing pipeline:
 * 1. Load image file using stb_image
 * 2. Resize to model's expected dimensions (if needed)
 * 3. Convert to model's expected format (typically RGB float32)
 * 4. Normalize pixel values to expected range
 * 5. Process through vision encoder (future work)
 * 6. Submit image embeddings to llama.cpp context
 *
 * Prerequisites:
 * - stb_image for image loading
 * - Sample image files for testing
 */

#include <catch2/catch_test_macros.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "vendor_paths.hpp"

/* Test basic stb_image functionality */
TEST_CASE("image_stb_init", "[image][stb]")
{
	/* stb_image is header-only, so just verify we can use its version info */
	std::cout << "stb_image initialized (header-only library)" << std::endl;
	REQUIRE(true);
}

/* Test loading an image file */
TEST_CASE("image_load_file", "[image][load]")
{
	/* Use a sample image from the test directory */
	VendorPath image_path{vp_test_data() + "/vendor_common_image.png"};

	std::cout << "=== Image Loading Test ===" << std::endl;
	std::cout << "Loading: " << image_path << std::endl;

	/* Step 1: Load image */
	int width, height, channels;
	unsigned char * image_data = stbi_load(image_path, &width, &height, &channels, 0);

	if (image_data == nullptr)
	{
		std::cerr << "  ✗ Failed to load image: " << stbi_failure_reason() << std::endl;
		FAIL("Failed to load image");
	}
	std::cout << "  ✓ Image loaded" << std::endl;

	/* Step 2: Read image metadata */
	std::cout << "Image metadata:" << std::endl;
	std::cout << "  Width: " << width << " px" << std::endl;
	std::cout << "  Height: " << height << " px" << std::endl;
	std::cout << "  Channels: " << channels << std::endl;
	std::cout << "  Total pixels: " << width * height << std::endl;
	std::cout << "  Data size: " << width * height * channels << " bytes" << std::endl;

	REQUIRE(width > 0);
	REQUIRE(height > 0);
	REQUIRE(channels > 0);
	REQUIRE(image_data != nullptr);

	/* Step 3: Free image data */
	stbi_image_free(image_data);
	std::cout << "  ✓ Image data freed" << std::endl;

	std::cout << "=== Image loading test passed ===" << std::endl;
}

/* Test loading image with specific channel count */
TEST_CASE("image_load_rgb", "[image][load]")
{
	VendorPath image_path{vp_test_data() + "/vendor_common_image.png"};

	std::cout << "=== RGB Image Loading Test ===" << std::endl;

	/* Load image forcing RGB (3 channels) */
	int width, height, channels;
	unsigned char * image_data =
		stbi_load(image_path, &width, &height, &channels, 3);  // Force 3 channels (RGB)

	REQUIRE(image_data != nullptr);
	std::cout << "  ✓ Image loaded as RGB" << std::endl;
	std::cout << "  Original channels: " << channels << std::endl;
	std::cout << "  Loaded as: 3 channels (RGB)" << std::endl;

	REQUIRE(width > 0);
	REQUIRE(height > 0);

	stbi_image_free(image_data);
	std::cout << "=== RGB loading test passed ===" << std::endl;
}

/* Test converting image data to float32 */
TEST_CASE("image_convert_to_float", "[image][convert]")
{
	VendorPath image_path{vp_test_data() + "/vendor_common_image.png"};

	std::cout << "=== Float Conversion Test ===" << std::endl;

	/* Step 1: Load image as RGB */
	int width, height, channels;
	unsigned char * image_data = stbi_load(image_path, &width, &height, &channels, 3);
	REQUIRE(image_data != nullptr);

	/* Step 2: Convert uint8 [0-255] to float32 [0.0-1.0] */
	size_t pixel_count = width * height * 3;  // 3 channels (RGB)
	std::vector<float> float_data(pixel_count);

	for (size_t i = 0; i < pixel_count; i++)
	{
		float_data[i] = static_cast<float>(image_data[i]) / 255.0f;
	}
	std::cout << "  ✓ Converted to float32 [0.0-1.0]" << std::endl;

	/* Step 3: Verify conversion */
	float min_val = *std::min_element(float_data.begin(), float_data.end());
	float max_val = *std::max_element(float_data.begin(), float_data.end());

	std::cout << "  Float value range: [" << min_val << ", " << max_val << "]" << std::endl;

	REQUIRE(min_val >= 0.0f);
	REQUIRE(max_val <= 1.0f);

	stbi_image_free(image_data);
	std::cout << "=== Float conversion test passed ===" << std::endl;
}

/* Test normalizing image data for model input */
TEST_CASE("image_normalize_for_model", "[image][normalize]")
{
	VendorPath image_path{vp_test_data() + "/vendor_common_image.png"};

	std::cout << "=== Image Normalization Test ===" << std::endl;

	/*
	 * Many vision models expect normalized inputs:
	 * - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
	 * - Range: typically [-3, 3] after normalization
	 *
	 * For MiniCPM-o, we may need different normalization parameters.
	 * This test demonstrates standard ImageNet normalization.
	 */

	float const mean_rgb[3] = {0.485f, 0.456f, 0.406f};
	float const std_rgb[3] = {0.229f, 0.224f, 0.225f};

	/* Step 1: Load image as RGB */
	int width, height, channels;
	unsigned char * image_data = stbi_load(image_path, &width, &height, &channels, 3);
	REQUIRE(image_data != nullptr);

	/* Step 2: Convert to float [0.0-1.0] */
	size_t pixel_count = width * height;
	std::vector<float> normalized_data(pixel_count * 3);

	for (size_t i = 0; i < pixel_count; i++)
	{
		for (int c = 0; c < 3; c++)
		{
			// Convert to [0.0, 1.0]
			float pixel_val = static_cast<float>(image_data[i * 3 + c]) / 255.0f;

			// Normalize using ImageNet mean/std
			normalized_data[i * 3 + c] = (pixel_val - mean_rgb[c]) / std_rgb[c];
		}
	}
	std::cout << "  ✓ Normalized using ImageNet mean/std" << std::endl;

	/* Step 3: Verify normalization */
	float min_val = *std::min_element(normalized_data.begin(), normalized_data.end());
	float max_val = *std::max_element(normalized_data.begin(), normalized_data.end());

	std::cout << "  Normalized value range: [" << min_val << ", " << max_val << "]" << std::endl;

	// Typical range after ImageNet normalization
	REQUIRE(min_val >= -3.0f);
	REQUIRE(max_val <= 3.0f);

	stbi_image_free(image_data);
	std::cout << "=== Normalization test passed ===" << std::endl;
}

/* Test resizing image */
TEST_CASE("image_resize", "[image][resize]")
{
	VendorPath image_path{vp_test_data() + "/vendor_common_image.png"};

	std::cout << "=== Image Resize Test ===" << std::endl;

	/*
	 * Vision models typically expect fixed input dimensions.
	 * Common sizes: 224x224, 256x256, 384x384, 448x448, etc.
	 * MiniCPM-o may have its own preferred size.
	 */
	int const target_width = 384;
	int const target_height = 384;

	/* Step 1: Load image */
	int width, height, channels;
	unsigned char * image_data = stbi_load(image_path, &width, &height, &channels, 3);
	REQUIRE(image_data != nullptr);

	std::cout << "  Original size: " << width << "x" << height << std::endl;
	std::cout << "  Target size: " << target_width << "x" << target_height << std::endl;

	/* Step 2: Resize image */
	std::vector<unsigned char> resized_data(target_width * target_height * 3);

	// stbir_resize_uint8 resizes images (v1 API)
	// Parameters: input_pixels, input_w, input_h, input_stride_in_bytes,
	//             output_pixels, output_w, output_h, output_stride_in_bytes, num_channels
	int result = stbir_resize_uint8(
		image_data,
		width,
		height,
		0,	// source (0 = auto-calculate stride)
		resized_data.data(),
		target_width,
		target_height,
		0,	// destination
		3	// channels (RGB)
	);

	if (result == 0)
	{
		std::cerr << "  ✗ Resize failed" << std::endl;
		stbi_image_free(image_data);
		FAIL("Resize failed");
	}
	std::cout << "  ✓ Resized to " << target_width << "x" << target_height << std::endl;

	/* Step 3: Verify resized data */
	REQUIRE(resized_data.size() == (size_t)(target_width * target_height * 3));

	stbi_image_free(image_data);
	std::cout << "=== Resize test passed ===" << std::endl;
}

/* Test preparing image for model input */
TEST_CASE("image_prepare_for_model", "[image][model]")
{
	VendorPath image_path{vp_test_data() + "/vendor_common_image.png"};

	std::cout << "=== Image Model Preparation Test ===" << std::endl;

	/*
	 * Complete pipeline for preparing image for MiniCPM-o:
	 * 1. Load image
	 * 2. Resize to target dimensions
	 * 3. Convert to float32
	 * 4. Normalize (if required by model)
	 * 5. Arrange in model's expected layout (CHW or HWC)
	 */

	int const target_width = 384;
	int const target_height = 384;
	float const mean_rgb[3] = {0.485f, 0.456f, 0.406f};
	float const std_rgb[3] = {0.229f, 0.224f, 0.225f};

	/* Step 1: Load image */
	int width, height, channels;
	unsigned char * image_data = stbi_load(image_path, &width, &height, &channels, 3);
	REQUIRE(image_data != nullptr);
	std::cout << "  ✓ Image loaded: " << width << "x" << height << std::endl;

	/* Step 2: Resize */
	std::vector<unsigned char> resized_data(target_width * target_height * 3);
	int result = stbir_resize_uint8(
		image_data, width, height, 0, resized_data.data(), target_width, target_height, 0, 3);
	REQUIRE(result != 0);
	stbi_image_free(image_data);
	std::cout << "  ✓ Resized to " << target_width << "x" << target_height << std::endl;

	/* Step 3: Convert to float and normalize */
	size_t pixel_count = target_width * target_height;
	std::vector<float> model_input(pixel_count * 3);

	for (size_t i = 0; i < pixel_count; i++)
	{
		for (int c = 0; c < 3; c++)
		{
			float pixel_val = static_cast<float>(resized_data[i * 3 + c]) / 255.0f;
			model_input[i * 3 + c] = (pixel_val - mean_rgb[c]) / std_rgb[c];
		}
	}
	std::cout << "  ✓ Converted to float32 and normalized" << std::endl;

	/* Step 4: Optionally convert HWC to CHW layout (many models prefer CHW) */
	// For demonstration, we'll show how to convert to CHW layout
	std::vector<float> model_input_chw(pixel_count * 3);

	for (size_t i = 0; i < pixel_count; i++)
	{
		for (int c = 0; c < 3; c++)
		{
			// CHW layout: all R values, then all G values, then all B values
			model_input_chw[c * pixel_count + i] = model_input[i * 3 + c];
		}
	}
	std::cout << "  ✓ Converted from HWC to CHW layout" << std::endl;

	/* Step 5: Summary */
	std::cout << "  Model-ready image:" << std::endl;
	std::cout << "    Dimensions: " << target_width << "x" << target_height << std::endl;
	std::cout << "    Channels: 3 (RGB)" << std::endl;
	std::cout << "    Format: float32, normalized, CHW layout" << std::endl;
	std::cout << "    Size: " << model_input_chw.size() << " floats ("
			  << model_input_chw.size() * sizeof(float) << " bytes)" << std::endl;

	REQUIRE(model_input_chw.size() == (size_t)(target_width * target_height * 3));

	std::cout << "=== Model preparation test passed ===" << std::endl;
}

/* Test loading multiple images */
TEST_CASE("image_batch_loading", "[image][batch]")
{
	std::cout << "=== Batch Image Loading Test ===" << std::endl;

	/* Find multiple test images */
	// vendor_common_image.png — copied from pytorch-simple-demo tests/cases/common/images/image.png
	// vendor_batch_image.png  — copied from pytorch-simple-demo
	// tests/results/duplex/case_with_image/image_0.png
	static VendorPath const vp_common{vp_test_data() + "/vendor_common_image.png"};
	static VendorPath const vp_batch{vp_test_data() + "/vendor_batch_image.png"};
	std::vector<char const *> image_paths = {vp_common, vp_batch};

	std::vector<std::vector<float>> batch_images;

	for (auto const & image_path : image_paths)
	{
		int width, height, channels;
		unsigned char * image_data = stbi_load(image_path, &width, &height, &channels, 3);

		if (image_data == nullptr)
		{
			std::cout << "  ⚠ Skipping " << image_path << " (not found or invalid)" << std::endl;
			continue;
		}

		std::cout << "  ✓ Loaded: " << image_path << " (" << width << "x" << height << ")"
				  << std::endl;

		// Convert to float
		size_t pixel_count = width * height * 3;
		std::vector<float> float_image(pixel_count);
		for (size_t i = 0; i < pixel_count; i++)
		{
			float_image[i] = static_cast<float>(image_data[i]) / 255.0f;
		}

		batch_images.push_back(std::move(float_image));
		stbi_image_free(image_data);
	}

	std::cout << "  ✓ Loaded " << batch_images.size() << " images in batch" << std::endl;
	REQUIRE(batch_images.size() > 0);

	std::cout << "=== Batch loading test passed ===" << std::endl;
}
