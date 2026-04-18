/*
 * End-to-End Inference Tests for MiniCPM-o-4_5
 *
 * This test file demonstrates complete multimodal inference with the MiniCPM-o model,
 * including image recognition, audio processing, and multimodal combinations.
 *
 * Test Pipeline:
 * 1. Load MiniCPM-o LLM model (8.2B parameters, Q4_K_M quantization)
 * 2. Initialize inference context with 4096 token context window
 * 3. Construct multimodal prompt with special tokens
 * 4. Tokenize prompt and submit to model
 * 5. Submit image embeddings (from real CLIP vision encoder)
 * 6. Generate text response via autoregressive decoding
 * 7. Validate output quality
 *
 * Technologies Used:
 * - llama.cpp: Core LLM inference engine (GGUF format)
 * - libmtmd CLIP: Vision encoder for generating image embeddings
 * - Catch2: Test framework for assertions
 * - stb_image: Image loading
 *
 * Prerequisites:
 * - LLM model: models/gguf/MiniCPM-o-4_5-Q4_K_M.gguf
 * - Vision model: models/gguf/vision/MiniCPM-o-4_5-vision-F16.gguf
 * - Audio model: models/gguf/audio/MiniCPM-o-4_5-audio-F16.gguf (future)
 * - Test images: test_data/circle.png, square.png, triangle.png
 *
 * Key Concepts:
 * - Multimodal tokens: Special tokens (<image>, <|audio|>) mark media input
 * - Embedding vectors: Float32 vectors representing image/audio features
 * - Autoregressive generation: Model predicts next token based on previous ones
 * - Greedy decoding: Simple sampling strategy (choose token with highest probability)
 */

#include <catch2/catch_test_macros.hpp>
#include "llama.h"

// Include token constants and path helpers
#include "tokens.hpp"
#include "vendor_paths.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <sndfile.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "omni/audition.h"
#include "omni/clip.h"
#include "omni/omni-impl.h"

/*
 * Helper Function: Load CLIP Vision Encoder
 *
 * Initializes the CLIP vision encoder from a GGUF model file using libmtmd's
 * low-level CLIP API. This provides a cleaner alternative to the omni implementation.
 *
 * Parameters:
 *   model_path: Path to the vision encoder GGUF file
 *   use_gpu: Whether to use GPU acceleration (default: false for compatibility)
 *   verbosity: Log level for CLIP operations
 *
 * Returns:
 *   Pointer to clip_ctx on success, nullptr on failure
 *
 * Note: This uses clip_init() directly, bypassing the high-level mtmd API
 * that doesn't support MiniCPM-o 4.5.
 *
 * Technologies:
 * - clip_init(): Low-level CLIP initialization from libmtmd
 * - clip_ctx: CLIP context structure
 * - GGML: Low-level tensor operations
 */
clip_ctx * load_clip_encoder(char const * model_path, bool use_gpu = false)
{
	clip_context_params clip_params = {};
	clip_params.use_gpu = use_gpu;

	clip_init_result result = clip_init(model_path, clip_params);
	return result.ctx_v;
}

/*
 * Helper Function: Encode Image with CLIP
 *
 * Processes an image through the CLIP vision encoder to generate embeddings
 * suitable for MiniCPM-o multimodal inference.
 *
 * Parameters:
 *   clip_ctx: Initialized CLIP context from load_clip_encoder()
 *   image_path: Path to the image file
 *   n_threads: Number of CPU threads for encoding (default: 4)
 *
 * Returns:
 *   Vector of float32 embeddings with shape [n_tokens * n_embd]
 *   Returns empty vector on failure
 *
 * Process flow:
 * 1. Load image from file using stb_image
 * 2. Create CLIP image structure and populate with pixel data
 * 3. Preprocess image (resize, normalize, convert layout)
 * 4. Encode to embeddings via vision transformer
 * 5. Project embeddings to LLM space (4096-d)
 *
 * Technologies:
 * - stb_image: Image loading library
 * - clip_image_u8: CLIP image structure
 * - clip_image_f32_batch: Preprocessed image batch
 * - clip_image_batch_encode(): Vision transformer forward pass
 */
std::vector<float> encode_image_with_clip(
	clip_ctx * clip_ctx, char const * image_path, int n_threads = 4)
{
	if (!clip_ctx)
	{
		std::cerr << "  ✗ CLIP context is null" << std::endl;
		return {};
	}

	/* Step 1: Load image */
	int width, height, channels;
	unsigned char * img_data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
	if (!img_data)
	{
		std::cerr << "  ✗ Failed to load image: " << image_path << std::endl;
		return {};
	}

	/* Step 2: Create CLIP image structure */
	clip_image_u8 * img_u8 = clip_image_u8_init();
	if (!img_u8)
	{
		std::cerr << "  ✗ Failed to create CLIP image structure" << std::endl;
		stbi_image_free(img_data);
		return {};
	}

	/* Step 3: Build image from pixels */
	clip_build_img_from_pixels(img_data, width, height, img_u8);

	/* Step 4: Preprocess image */
	clip_image_f32_batch * batch = clip_image_f32_batch_init();
	if (!batch)
	{
		std::cerr << "  ✗ Failed to create image batch" << std::endl;
		clip_image_u8_free(img_u8);
		stbi_image_free(img_data);
		return {};
	}

	bool preprocess_ok = clip_image_preprocess(clip_ctx, img_u8, batch);
	if (!preprocess_ok)
	{
		std::cerr << "  ✗ Failed to preprocess image" << std::endl;
		clip_image_f32_batch_free(batch);
		clip_image_u8_free(img_u8);
		stbi_image_free(img_data);
		return {};
	}

	/* Step 5: Get encoding parameters */
	clip_image_f32 * first_img = clip_image_f32_get_img(batch, 0);
	int n_tokens_per_image = clip_n_output_tokens(clip_ctx, first_img);
	int n_embed = clip_n_mmproj_embd(clip_ctx);
	size_t n_images = clip_image_f32_batch_n_images(batch);
	int total_tokens = n_images * n_tokens_per_image;

	/* Step 6: Allocate and encode embeddings */
	std::vector<float> embeddings(total_tokens * n_embed);
	bool encode_ok = clip_image_batch_encode(clip_ctx, n_threads, batch, embeddings.data());

	/* Step 7: Cleanup */
	clip_image_f32_batch_free(batch);
	clip_image_u8_free(img_u8);
	stbi_image_free(img_data);

	if (!encode_ok)
	{
		std::cerr << "  ✗ Failed to encode image to embeddings" << std::endl;
		return {};
	}

	return embeddings;
}

/*
 * Helper Function: Load and Encode Audio with Audition
 *
 * Processes an audio file through the audition audio encoder to generate embeddings
 * suitable for MiniCPM-o multimodal inference.
 *
 * Parameters:
 *   audio_model_path: Path to the audio encoder GGUF file
 *   audio_path: Path to the audio file (WAV format)
 *   use_gpu: Whether to use GPU acceleration (default: true)
 *   n_threads: Number of CPU threads for encoding (default: 4)
 *
 * Returns:
 *   Vector of float32 embeddings with shape [n_tokens * n_embd]
 *   Returns empty vector on failure
 *
 * Process flow:
 * 1. Initialize audition encoder
 * 2. Load audio file using libsndfile
 * 3. Convert to mono if needed
 * 4. Preprocess audio to Mel spectrogram using whisper_preprocessor
 * 5. Encode to embeddings via audition encoder
 *
 * Technologies:
 * - libsndfile: Audio file loading
 * - whisper_preprocessor: Mel spectrogram generation (from libmtmd)
 * - audition encoder: Audio transformer from vendor/omni
 */
std::vector<float> load_and_encode_audio(
	char const * audio_model_path, char const * audio_path, bool use_gpu = true, int n_threads = 4)
{
	/* Step 1: Initialize audio encoder */
	audition_context_params audio_params = {use_gpu, GGML_LOG_LEVEL_WARN};
	audition_ctx * ctx_audio = audition_init(audio_model_path, audio_params);
	if (!ctx_audio)
	{
		std::cerr << "  ✗ Failed to initialize audio encoder" << std::endl;
		return {};
	}

	/* Step 2: Load audio file */
	SF_INFO sfinfo;
	SNDFILE * sndfile = sf_open(audio_path, SFM_READ, &sfinfo);
	if (!sndfile)
	{
		std::cerr << "  ✗ Failed to load audio file: " << audio_path << std::endl;
		audition_free(ctx_audio);
		return {};
	}

	/* Step 3: Read audio samples */
	std::vector<float> samples(sfinfo.frames * sfinfo.channels);
	sf_count_t frames_read = sf_readf_float(sndfile, samples.data(), sfinfo.frames);
	sf_close(sndfile);

	if (frames_read != sfinfo.frames)
	{
		std::cerr << "  ✗ Failed to read audio samples" << std::endl;
		audition_free(ctx_audio);
		return {};
	}

	/* Step 4: Convert to mono if needed */
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

	/* Step 5: Preprocess audio to Mel spectrogram */
	auto filters = audition_get_mel_filters(ctx_audio);
	std::vector<whisper_preprocessor::whisper_mel> mel_output;
	whisper_preprocessor::preprocess_audio(
		mono_samples.data(), mono_samples.size(), filters, mel_output);

	if (mel_output.empty())
	{
		std::cerr << "  ✗ Failed to preprocess audio" << std::endl;
		audition_free(ctx_audio);
		return {};
	}

	/* Step 6: Convert to audition format */
	auto const & mel_segment = mel_output[0];
	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel_segment.n_len;
	mel_f32->ny = mel_segment.n_mel;
	mel_f32->buf.resize(mel_segment.data.size());
	std::copy(mel_segment.data.begin(), mel_segment.data.end(), mel_f32->buf.begin());

	/* Step 7: Encode to embeddings */
	int n_tokens = audition_n_output_tokens(ctx_audio, mel_f32);
	int n_embd = audition_n_mmproj_embd(ctx_audio);

	std::vector<float> embeddings(n_tokens * n_embd);
	bool encode_ok = audition_audio_encode(ctx_audio, n_threads, mel_f32, embeddings.data());

	/* Step 8: Cleanup */
	audition_audio_f32_free(mel_f32);
	audition_free(ctx_audio);

	if (!encode_ok)
	{
		std::cerr << "  ✗ Failed to encode audio to embeddings" << std::endl;
		return {};
	}

	return embeddings;
}

/*
 * Helper Function: Greedy Token Sampling
 *
 * Selects the token with the highest logit probability.
 *
 * Parameters:
 *   logits: Array of logit scores for all vocabulary tokens
 *   n_vocab: Number of tokens in vocabulary
 *
 * Returns:
 *   Token ID with highest probability
 *
 * In LLM terms: Greedy decoding is the simplest sampling strategy.
 * Instead of randomly selecting from probabilities, we always pick
 * the token with the highest probability. This is fast but can be
 * less creative than other strategies like temperature sampling.
 *
 * Technologies:
 * - std::vector: Dynamic array for logits
 * - C++23 features: Modern C++ iteration
 * - Simple loop: Linear scan for maximum value
 */
llama_token sample_token_greedy(float const * logits, int n_vocab)
{
	llama_token best_token = 0;
	float best_logit = logits[0];

	for (int i = 1; i < n_vocab; i++)
	{
		if (logits[i] > best_logit)
		{
			best_logit = logits[i];
			best_token = i;
		}
	}

	return best_token;
}

/*
 * Helper Function: Generate Text Response
 *
 * Generates a response by autoregressively sampling tokens from the model.
 *
 * Parameters:
 *   ctx: Inference context (holds model state and KV cache)
 *   vocab: Model vocabulary for tokenization
 *   n_past: Number of tokens already processed
 *   max_tokens: Maximum number of tokens to generate
 *
 * Returns:
 *   Generated text as string
 *
 * Process flow:
 * 1. Get logits from model output
 * 2. Sample next token (greedy strategy)
 * 3. Check for EOS (End of Sequence) token
 * 4. Decode token to text
 * 5. Submit token for next generation step
 * 6. Repeat until EOS or max tokens reached
 */
std::string generate_response(
	llama_context * ctx, llama_vocab const * vocab, int n_past, int max_tokens = 50)
{
	int n_vocab = llama_vocab_n_tokens(vocab);
	std::string response;
	for (int i = 0; i < max_tokens; i++)
	{
		/*
		 * Get logits from model output
		 *
		 * llama_get_logits() - Retrieve logits for next token prediction
		 *
		 * Args:
		 *   ctx: The inference context
		 *
		 * Returns:
		 *   Pointer to logits array, or nullptr if not available
		 *
		 * Logits are raw unnormalized scores for each token in the vocabulary.
		 * They represent the model's confidence in each possible next token.
		 * Higher values = more confident in that token.
		 */
		float * logits = llama_get_logits(ctx);
		if (!logits)
		{
			std::cout << "  ⚠ No logits available" << std::endl;
			break;
		}

		/*
		 * Sample next token using greedy decoding
		 *
		 * Selects the token with the highest logit value.
		 * Simple and fast sampling strategy.
		 */
		llama_token next_token = sample_token_greedy(logits, n_vocab);

		/*
		 * Check for End of Sequence (EOS) tokens
		 *
		 * EOS TOKENS EXPLAINED:
		 * --------------------
		 * These special tokens signal that the model has finished its response.
		 * MiniCPM-o uses multiple EOS markers:
		 *
		 * - TOKEN_EOS_ID (151645 = "<|im_end|>"): Main end-of-message marker
		 *   - Signals "my response is complete"
		 *   - Part of the chat template format
		 *   - Example: "The shape is a circle.<|im_end|>"
		 *
		 * - TOKEN_END_OF_TEXT_ID (151643 = "<|endoftext|>"): End marker
		 *   - Legacy/alternative EOS token
		 *   - Also used for padding in training
		 *
		 * WHY CHECK BOTH:
		 * The model may use either token to signal completion depending on
		 * context. Checking both ensures we stop generation appropriately
		 * and avoid infinite loops or excess token generation.
		 *
		 * WHAT HAPPENS WHEN WE BREAK:
		 * - Generation loop exits
		 * - Response string contains all tokens generated so far
		 * - No more GPU/CPU computation wasted
		 */
		if (next_token == TOKEN_EOS_ID || next_token == TOKEN_END_OF_TEXT_ID)
		{
			std::cout << "  ✓ Reached EOS token" << std::endl;
			break;
		}

		/*
		 * Decode token to text
		 *
		 * llama_detokenize() - Convert single token ID to text representation
		 *
		 * Args:
		 *   vocab: The vocabulary/tokenizer
		 *   token: Single token ID to decode
		 *   n_tokens: Number of tokens (always 1 here)
		 *   text: Output buffer for text
		 *   text_len_max: Size of output buffer
		 *   remove_special: Remove special tokens from output
		 *   unparse_special: Keep special tokens in unparsed form
		 *
		 * Returns:
		 *   Number of bytes written, or negative on error
		 *
		 * Decoding converts token IDs back to human-readable text.
		 * This happens after generation to display the final output.
		 */
		std::vector<char> piece(256);
		int n_piece =
			llama_detokenize(vocab, &next_token, 1, piece.data(), piece.size(), false, false);
		if (n_piece > 0)
		{
			response.append(piece.data(), n_piece);
		}

		/*
		 * Submit token for next generation iteration
		 *
		 * Creates a minimal batch with single token and submits to model.
		 *
		 * llama_decode() - Run inference on a batch of tokens
		 *
		 * Args:
		 *   ctx: The inference context
		 *   batch: Batch containing token to process
		 *
		 * Returns:
		 *   0 on success, non-zero on error
		 *
		 * This processes the newly generated token through the model,
		 * updating the KV cache and computing logits for the next token.
		 */
		llama_pos token_pos = n_past + i;
		int8_t logit_flag = 1;

		llama_batch batch = {};
		batch.n_tokens = 1;
		batch.token = &next_token;
		batch.pos = &token_pos;
		batch.logits = &logit_flag;

		int result = llama_decode(ctx, batch);
		if (result != 0)
		{
			std::cout << "  ⚠ Decode failed during generation: " << result << std::endl;
			break;
		}
	}

	return response;
}

/*
 * Test: Image Shape Recognition via Multimodal Inference with Real CLIP Encoder
 *
 * Tests end-to-end multimodal inference with image input and text output.
 * Uses real CLIP vision encoder to generate image embeddings from actual images.
 *
 * This test validates:
 * - Model loading and initialization
 * - CLIP vision encoder loading and usage
 * - Image preprocessing and encoding
 * - Multimodal prompt construction
 * - Image embedding submission
 * - Text token generation
 * - Response decoding
 * - Complete inference pipeline
 *
 * Technologies:
 * - llama.cpp: LLM inference engine
 * - libmtmd CLIP: Vision encoder for generating image embeddings
 * - stb_image: Image loading
 */
TEST_CASE("inference_shape_recognition", "[inference][vision]")
{
	std::cout << "\n=== Shape Recognition Inference Test (Real CLIP Encoder) ===" << std::endl;

	/* Step 1: Load MiniCPM-o LLM model
	 *
	 * llama_model_load_from_file() - Load model weights from GGUF file
	 *
	 * The model file contains:
	 * - Transformer layer weights (36 layers for MiniCPM-o)
	 * - Embedding matrix (150K tokens, 4096 dimensions)
	 * - Vocabulary mappings
	 * - Model metadata
	 *
	 * MiniCPM-o-4_5 Q4_K_M is 8.2B parameters quantized to 4-bit,
	 * resulting in ~4.7GB file size.
	 *
	 * Use GPU offloading (n_gpu_layers=999) to offload all layers to GPU
	 * for faster inference. The RTX 5070 Ti has sufficient VRAM.
	 */
	VendorPath model_path{path_model_llm()};

	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 999;  // Offload all layers to GPU
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 512;

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	/* Get model metadata */
	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_embd = llama_model_n_embd(model);
	int n_vocab = llama_vocab_n_tokens(vocab);
	std::cout << "  ✓ Model loaded (n_embd=" << n_embd << ", n_vocab=" << n_vocab << ")"
			  << std::endl;

	/* Step 2: Load CLIP vision encoder */
	std::cout << "  Loading CLIP vision encoder..." << std::endl;
	VendorPath vision_model_path{path_model_vision()};
	clip_ctx * clip_ctx = load_clip_encoder(vision_model_path, true);
	REQUIRE(clip_ctx != nullptr);
	std::cout << "  ✓ CLIP vision encoder loaded" << std::endl;

	int n_past = 0;
	int result = 0;	 // Reusable result variable for llama_decode calls

	/*
	 * Step 3: Construct multimodal prompt
	 *
	 * MINICPM-O CHAT TEMPLATE FORMAT:
	 * --------------------------------
	 * MiniCPM-o uses a specific chat template with role markers:
	 *
	 * Format: ROLE_USER + [media tokens] + question + ROLE_TRANSITION
	 *
	 * Example for vision:
	 *   " lda\n<image>What shape is in this image? lda\nassistant\n"
	 *    ^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^
	 *    prefix       question/prompt           transition
	 *
	 * ROLE MARKERS EXPLAINED:
	 * - ROLE_USER (" lda\n"): Marks start of user message
	 *   - "lda" likely stands for "llama/LLM dialogue agent"
	 *   - Always includes leading space and \n
	 *
	 * - ROLE_TRANSITION (" lda\nassistant\n"): Marks user→assistant turn
	 *   - Signals: "user finished, model should respond now"
	 *   - "assistant" is the model's role name
	 *
	 * - TOKEN_IMAGE ("<image>"): Placeholder for image embeddings
	 *   - This exact string gets tokenized
	 *   - During inference, we replace its position with actual embeddings
	 *
	 * WHY THIS FORMAT:
	 * This format trains the model to understand when it's the user speaking
	 * vs when it should generate (assistant role). The role markers are part
	 * of the training data, so the model learned to respond after seeing
	 * " lda\nassistant\n".
	 */
	std::string const prompt_prefix = ROLE_USER;
	std::string const image_marker = TOKEN_IMAGE;
	std::string const prompt_suffix = "What shape is in this image?" + std::string(ROLE_TRANSITION);

	/*
	 * Step 4: Tokenize and submit prompt prefix
	 *
	 * Tokenization converts text to token IDs that the model can process.
	 * We submit tokens one by one to build up the context.
	 *
	 * llama_tokenize() - Convert text to token IDs
	 *
	 * First call with nullptr returns required buffer size (negative value).
	 * Second call fills the buffer with actual token IDs.
	 */
	std::vector<llama_token> prefix_tokens;
	int32_t n_prefix =
		-llama_tokenize(vocab, prompt_prefix.data(), prompt_prefix.size(), nullptr, 0, true, false);
	REQUIRE(n_prefix > 0);
	prefix_tokens.resize(n_prefix);
	n_prefix = llama_tokenize(
		vocab,
		prompt_prefix.data(),
		prompt_prefix.size(),
		prefix_tokens.data(),
		prefix_tokens.size(),
		true,
		false);

	std::cout << "  ✓ Tokenized prefix: " << n_prefix << " tokens" << std::endl;

	/*
	 * Submit all prefix tokens in a single batch (OPTIMIZED)
	 *
	 * KEY INSIGHT FOR PROMPT PROCESSING:
	 * ----------------------------------
	 * During prompt processing (before generation), we don't need logits
	 * for any token except the very last one. This is because:
	 *
	 * 1. We're not sampling/generating during prompt processing
	 * 2. We only need logits at the END to predict the first generated token
	 * 3. Setting logits=0 for all but last token saves computation
	 *
	 * For prefix tokens, we typically set ALL logits=0 because more tokens
	 * come after (suffix tokens). Only the final token in the entire prompt
	 * needs logits=1.
	 */

	// Create position array for all prefix tokens
	std::vector<llama_pos> prefix_pos(n_prefix);
	for (int32_t i = 0; i < n_prefix; i++)
	{
		prefix_pos[i] = n_past + i;
	}

	// Create logits flags (all 0 for prefix, since suffix comes after)
	std::vector<int8_t> prefix_logits(n_prefix, 0);

	// Submit all prefix tokens in a single batch
	llama_batch prefix_batch = {};
	prefix_batch.n_tokens = n_prefix;
	prefix_batch.token = prefix_tokens.data();
	prefix_batch.pos = prefix_pos.data();
	prefix_batch.logits = prefix_logits.data();

	result = llama_decode(ctx, prefix_batch);
	REQUIRE(result == 0);
	n_past += n_prefix;

	/*
	 * Step 5: Encode image with CLIP and submit embeddings
	 *
	 * MiniCPM-o accepts image embeddings instead of raw images.
	 * Process:
	 * 1. Load image file (e.g., circle.png)
	 * 2. Run through CLIP vision encoder
	 * 3. Output: [n_patches, n_embd] float32 embeddings
	 *
	 * For this test, we use the real CLIP encoder to generate embeddings
	 * from actual test images.
	 */
	std::cout << "  Encoding image with CLIP..." << std::endl;
	VendorPath image_path{path_test_data_dir() + "/circle.png"};
	std::vector<float> image_embeds = encode_image_with_clip(clip_ctx, image_path, 4);

	REQUIRE(!image_embeds.empty());
	int n_image_tokens = image_embeds.size() / n_embd;
	std::cout << "  ✓ Generated " << n_image_tokens << " image embedding tokens ("
			  << image_embeds.size() << " floats)" << std::endl;

	std::vector<llama_pos> image_pos(n_image_tokens);

	for (int i = 0; i < n_image_tokens; i++)
	{
		image_pos[i] = n_past + i;
	}

	/*
	 * Create batch with embeddings instead of tokens
	 *
	 * llama_batch.embd parameter points to embedding vectors when
	 * processing multimodal input. This is mutually exclusive with
	 * llama_batch.token.
	 */
	llama_batch image_batch = {};
	image_batch.n_tokens = n_image_tokens;
	image_batch.embd = image_embeds.data();
	image_batch.pos = image_pos.data();

	std::cout << "  Submitting " << n_image_tokens << " image embedding tokens..." << std::endl;
	result = llama_decode(ctx, image_batch);
	REQUIRE(result == 0);
	n_past += n_image_tokens;
	std::cout << "  ✓ Image embeddings processed" << std::endl;

	/*
	 * Step 6: Submit prompt suffix
	 *
	 * The question part of the prompt.
	 * We submit all tokens then generate response.
	 */
	std::vector<llama_token> suffix_tokens;
	int32_t n_suffix = -llama_tokenize(
		vocab, prompt_suffix.data(), prompt_suffix.size(), nullptr, 0, false, false);
	REQUIRE(n_suffix > 0);
	suffix_tokens.resize(n_suffix);
	n_suffix = llama_tokenize(
		vocab,
		prompt_suffix.data(),
		prompt_suffix.size(),
		suffix_tokens.data(),
		suffix_tokens.size(),
		false,
		false);

	std::cout << "  ✓ Tokenized suffix: " << n_suffix << " tokens" << std::endl;

	/*
	 * Submit all suffix tokens in a single batch (OPTIMIZED)
	 *
	 * WHY BATCH SUBMISSION IS BETTER:
	 * --------------------------------
	 * Old approach: Loop n_suffix times, calling llama_decode() each iteration
	 * - Each call has overhead (kernel launch, synchronization, etc.)
	 * - Misses opportunities for parallelization
	 * - For 11 tokens, this means 11 separate GPU/CPU calls
	 *
	 * New approach: Submit all tokens in one batch
	 * - Single llama_decode() call processes all tokens
	 * - GPU can parallelize attention computation across tokens
	 * - Reduces overhead by ~10-100x for prompt processing
	 *
	 * BATCHING EXPLAINED:
	 * -------------------
	 * llama.cpp batch API allows submitting multiple tokens at once.
	 * You need to provide ARRAYS for:
	 * - token: Array of token IDs [tok1, tok2, tok3, ...]
	 * - pos: Array of positions [pos1, pos2, pos3, ...]
	 * - logits: Array of flags [0, 0, 0, ..., 1] (only last needs logits)
	 *
	 * The last token needs logits=1 because that's where we'll sample
	 * the next token from. Earlier tokens don't need logits during
	 * prompt processing (we're not sampling from them).
	 */

	// Create position array for all suffix tokens
	std::vector<llama_pos> suffix_pos(n_suffix);
	for (int32_t i = 0; i < n_suffix; i++)
	{
		suffix_pos[i] = n_past + i;
	}

	// Create logits flags array (only last token needs logits)
	std::vector<int8_t> suffix_logits(n_suffix, 0);
	suffix_logits[n_suffix - 1] = 1;  // Enable logits for last token only

	// Submit all suffix tokens in a single batch
	llama_batch suffix_batch = {};
	suffix_batch.n_tokens = n_suffix;
	suffix_batch.token = suffix_tokens.data();
	suffix_batch.pos = suffix_pos.data();
	suffix_batch.logits = suffix_logits.data();

	result = llama_decode(ctx, suffix_batch);
	REQUIRE(result == 0);
	n_past += n_suffix;

	/*
	 * Step 7: Generate response
	 *
	 * Autoregressive generation: Model predicts tokens one at a time,
	 * using previously generated tokens as context.
	 */
	std::cout << "  Generating response..." << std::endl;
	std::string response = generate_response(ctx, vocab, n_past, 100);

	std::cout << "  Generated response: \"" << response << "\"" << std::endl;

	/* Validate response */
	REQUIRE(response.size() > 0);
	std::cout << "  ✓ Model generated " << response.size() << " characters" << std::endl;

	/* Validate content - the model should identify the shape correctly */
	// Convert to lowercase for case-insensitive comparison
	std::string response_lower = response;
	std::transform(response_lower.begin(), response_lower.end(), response_lower.begin(), ::tolower);
	REQUIRE(response_lower.find("circle") != std::string::npos);
	std::cout << "  ✓ Response correctly identifies 'circle'" << std::endl;

	/*
	 * Cleanup CLIP encoder
	 *
	 * clip_free() releases all CLIP context resources.
	 */
	clip_free(clip_ctx);

	/*
	 * Step 8: Cleanup
	 *
	 * Always free resources to prevent memory leaks.
	 * llama_free() frees context and its internal buffers.
	 * llama_model_free() frees model weights and vocabulary.
	 */
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Shape recognition test completed ===" << std::endl;
}

/*
 * Test: Audio Processing Inference
 *
 * Tests multimodal inference with audio input and text output.
 * Uses real audition audio encoder from vendor/omni.
 *
 * This test validates:
 * - Model loading and initialization
 * - Audio embedding submission
 * - Text token generation
 * - Response decoding
 * - Complete inference pipeline
 *
 * Technologies:
 * - llama.cpp: LLM inference engine
 * - libsndfile: Audio file loading (for future real usage)
 */
TEST_CASE("inference_audio_processing", "[inference][audio]")
{
	std::cout << "\n=== Audio Processing Inference Test ===" << std::endl;

	/* Load model */
	VendorPath model_path{path_model_llm()};

	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 999;  // Offload all layers to GPU
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 512;

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_embd = llama_model_n_embd(model);

	int n_past = 0;
	int result = 0;	 // Reusable result variable for llama_decode calls

	/* Construct prompt with audio marker (matching vision test pattern) */
	// Use constants from tokens.hpp instead of magic strings
	std::string const prompt_prefix = ROLE_USER;
	std::string const audio_marker = TOKEN_AUDIO;
	std::string const prompt_suffix =
		"What does the person say in this audio?" + std::string(ROLE_TRANSITION);

	std::cout << "  ✓ Prompt constructed with audio marker" << std::endl;

	/* Tokenize and submit prefix (OPTIMIZED - batched submission) */
	std::vector<llama_token> prefix_tokens;
	int32_t n_prefix =
		-llama_tokenize(vocab, prompt_prefix.data(), prompt_prefix.size(), nullptr, 0, true, false);
	REQUIRE(n_prefix > 0);
	prefix_tokens.resize(n_prefix);
	n_prefix = llama_tokenize(
		vocab,
		prompt_prefix.data(),
		prompt_prefix.size(),
		prefix_tokens.data(),
		prefix_tokens.size(),
		true,
		false);

	// Submit all prefix tokens in a single batch
	std::vector<llama_pos> prefix_pos(n_prefix);
	std::vector<int8_t> prefix_logits(n_prefix, 0);
	for (int32_t i = 0; i < n_prefix; i++)
	{
		prefix_pos[i] = n_past + i;
	}

	llama_batch prefix_batch = {};
	prefix_batch.n_tokens = n_prefix;
	prefix_batch.token = prefix_tokens.data();
	prefix_batch.pos = prefix_pos.data();
	prefix_batch.logits = prefix_logits.data();

	result = llama_decode(ctx, prefix_batch);
	REQUIRE(result == 0);
	n_past += n_prefix;
	std::cout << "  ✓ Tokenized prefix: " << n_prefix << " tokens" << std::endl;

	/* Tokenize and submit audio marker (OPTIMIZED - batched submission) */
	std::vector<llama_token> marker_tokens;
	int32_t n_marker =
		-llama_tokenize(vocab, audio_marker.data(), audio_marker.size(), nullptr, 0, false, false);
	REQUIRE(n_marker > 0);
	marker_tokens.resize(n_marker);
	n_marker = llama_tokenize(
		vocab,
		audio_marker.data(),
		audio_marker.size(),
		marker_tokens.data(),
		marker_tokens.size(),
		false,
		false);

	// Submit all marker tokens in a single batch
	std::vector<llama_pos> marker_pos(n_marker);
	std::vector<int8_t> marker_logits(n_marker, 0);
	for (int32_t i = 0; i < n_marker; i++)
	{
		marker_pos[i] = n_past + i;
	}

	llama_batch marker_batch = {};
	marker_batch.n_tokens = n_marker;
	marker_batch.token = marker_tokens.data();
	marker_batch.pos = marker_pos.data();
	marker_batch.logits = marker_logits.data();

	result = llama_decode(ctx, marker_batch);
	REQUIRE(result == 0);
	n_past += n_marker;
	std::cout << "  ✓ Tokenized audio marker (start): " << n_marker << " tokens" << std::endl;

	/* Submit audio embeddings (real encoder) */
	VendorPath audio_model_path{path_model_audio()};
	VendorPath audio_path{path_test_data_dir() + "/say_the_word_banana.wav"};

	std::cout << "  Loading and encoding audio..." << std::endl;
	std::vector<float> audio_embeds = load_and_encode_audio(audio_model_path, audio_path, true, 4);
	REQUIRE(!audio_embeds.empty());

	int n_audio_tokens = audio_embeds.size() / n_embd;
	std::vector<llama_pos> audio_pos(n_audio_tokens);

	for (int i = 0; i < n_audio_tokens; i++)
	{
		audio_pos[i] = n_past + i;
	}

	llama_batch audio_batch = {};
	audio_batch.n_tokens = n_audio_tokens;
	audio_batch.embd = audio_embeds.data();
	audio_batch.pos = audio_pos.data();

	std::cout << "  Submitting " << n_audio_tokens << " audio embedding tokens..." << std::endl;
	result = llama_decode(ctx, audio_batch);
	REQUIRE(result == 0);
	n_past += n_audio_tokens;
	std::cout << "  ✓ Audio embeddings processed" << std::endl;

	for (int32_t i = 0; i < n_marker; i++)
	{
		marker_pos[i] = n_past + i;
	}
	result = llama_decode(ctx, marker_batch);
	REQUIRE(result == 0);
	n_past += n_marker;
	std::cout << "  ✓ Tokenized audio marker (end): " << n_marker << " tokens" << std::endl;

	/* Submit prompt suffix (OPTIMIZED - batched submission) */
	std::vector<llama_token> suffix_tokens;
	int32_t n_suffix = -llama_tokenize(
		vocab, prompt_suffix.data(), prompt_suffix.size(), nullptr, 0, false, false);
	REQUIRE(n_suffix > 0);
	suffix_tokens.resize(n_suffix);
	n_suffix = llama_tokenize(
		vocab,
		prompt_suffix.data(),
		prompt_suffix.size(),
		suffix_tokens.data(),
		suffix_tokens.size(),
		false,
		false);

	// Submit all suffix tokens in a single batch
	std::vector<llama_pos> suffix_pos(n_suffix);
	std::vector<int8_t> suffix_logits(n_suffix, 0);
	for (int32_t i = 0; i < n_suffix; i++)
	{
		suffix_pos[i] = n_past + i;
	}
	suffix_logits[n_suffix - 1] = 1;  // Only last token needs logits

	llama_batch suffix_batch = {};
	suffix_batch.n_tokens = n_suffix;
	suffix_batch.token = suffix_tokens.data();
	suffix_batch.pos = suffix_pos.data();
	suffix_batch.logits = suffix_logits.data();

	result = llama_decode(ctx, suffix_batch);
	REQUIRE(result == 0);
	n_past += n_suffix;
	std::cout << "  ✓ Tokenized suffix: " << n_suffix << " tokens" << std::endl;

	/* Generate response */
	std::cout << "  Generating response..." << std::endl;
	std::string response = generate_response(ctx, vocab, n_past, 100);

	std::cout << "  Generated response: \"" << response << "\"" << std::endl;

	/* Validate response - the model should generate meaningful output */
	REQUIRE(response.size() > 0);
	std::cout << "  ✓ Model generated " << response.size() << " characters" << std::endl;

	/* Validate content - the model should transcribe/identify "banana" */
	std::string response_lower = response;
	std::transform(response_lower.begin(), response_lower.end(), response_lower.begin(), ::tolower);
	REQUIRE(response_lower.find("banana") != std::string::npos);
	std::cout << "  ✓ Response correctly transcribes 'banana' from audio" << std::endl;

	std::cout << "  Note: Using real audition audio encoder" << std::endl;

	/* Cleanup */
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Audio processing test completed ===" << std::endl;
}

/*
 * Test: Multimodal Combination Inference
 *
 * Tests inference with both image and audio inputs.
 * Uses real CLIP encoder for image embeddings and real audition encoder for audio.
 *
 * This test validates:
 * - Sequential multimodal embedding submission
 * - Mixed token/embedding workflow
 * - Complete multimodal context construction
 * - End-to-end response generation
 *
 * Technologies:
 * - llama.cpp: LLM inference engine
 * - libmtmd CLIP: Vision encoder for image embeddings
 * - stb_image: Image loading
 * - libsndfile: Audio loading (for future)
 */
TEST_CASE("inference_multimodal_combined", "[inference][multimodal]")
{
	std::cout << "\n=== Multimodal Combined Inference Test (Real CLIP + Real Audition) ==="
			  << std::endl;

	/* Load model */
	VendorPath model_path{path_model_llm()};

	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 999;  // Offload all layers to GPU
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 512;

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_embd = llama_model_n_embd(model);
	/* Load CLIP vision encoder */
	std::cout << "  Loading CLIP vision encoder..." << std::endl;
	VendorPath vision_model_path{path_model_vision()};
	clip_ctx * clip_ctx = load_clip_encoder(vision_model_path, true);
	REQUIRE(clip_ctx != nullptr);
	std::cout << "  ✓ CLIP vision encoder loaded" << std::endl;

	int n_past = 0;
	int result = 0;	 // Reusable result variable for llama_decode calls

	/* Construct multimodal prompt (using constants from tokens.hpp) */
	std::string const prompt_prefix = ROLE_USER;
	std::string const image_marker = TOKEN_IMAGE;
	std::string const audio_marker = TOKEN_AUDIO;
	std::string const prompt_suffix =
		"Describe what you see and hear." + std::string(ROLE_TRANSITION);

	std::cout << "  ✓ Multimodal prompt constructed" << std::endl;

	/* Tokenize and submit prefix (OPTIMIZED - batched submission) */
	std::vector<llama_token> prefix_tokens;
	int32_t n_prefix =
		-llama_tokenize(vocab, prompt_prefix.data(), prompt_prefix.size(), nullptr, 0, true, false);
	REQUIRE(n_prefix > 0);
	prefix_tokens.resize(n_prefix);
	n_prefix = llama_tokenize(
		vocab,
		prompt_prefix.data(),
		prompt_prefix.size(),
		prefix_tokens.data(),
		prefix_tokens.size(),
		true,
		false);

	// Submit all prefix tokens in a single batch
	std::vector<llama_pos> prefix_pos(n_prefix);
	std::vector<int8_t> prefix_logits(n_prefix, 0);
	for (int32_t i = 0; i < n_prefix; i++)
	{
		prefix_pos[i] = n_past + i;
	}

	llama_batch prefix_batch = {};
	prefix_batch.n_tokens = n_prefix;
	prefix_batch.token = prefix_tokens.data();
	prefix_batch.pos = prefix_pos.data();
	prefix_batch.logits = prefix_logits.data();

	result = llama_decode(ctx, prefix_batch);
	REQUIRE(result == 0);
	n_past += n_prefix;
	std::cout << "  ✓ Tokenized prefix: " << n_prefix << " tokens" << std::endl;

	/* Tokenize and submit image marker (OPTIMIZED - batched submission) */
	std::vector<llama_token> image_start_tokens;
	int32_t n_image_start =
		-llama_tokenize(vocab, image_marker.data(), image_marker.size(), nullptr, 0, false, false);
	REQUIRE(n_image_start > 0);
	image_start_tokens.resize(n_image_start);
	n_image_start = llama_tokenize(
		vocab,
		image_marker.data(),
		image_marker.size(),
		image_start_tokens.data(),
		image_start_tokens.size(),
		false,
		false);

	// Submit all image marker tokens in a single batch
	std::vector<llama_pos> image_marker_pos(n_image_start);
	std::vector<int8_t> image_marker_logits(n_image_start, 0);
	for (int32_t i = 0; i < n_image_start; i++)
	{
		image_marker_pos[i] = n_past + i;
	}

	llama_batch image_marker_batch = {};
	image_marker_batch.n_tokens = n_image_start;
	image_marker_batch.token = image_start_tokens.data();
	image_marker_batch.pos = image_marker_pos.data();
	image_marker_batch.logits = image_marker_logits.data();

	result = llama_decode(ctx, image_marker_batch);
	REQUIRE(result == 0);
	n_past += n_image_start;

	/* Submit image embeddings (using real CLIP encoder) */
	std::cout << "  Encoding image with CLIP..." << std::endl;
	VendorPath image_path{path_test_data_dir() + "/circle.png"};
	std::vector<float> image_embeds = encode_image_with_clip(clip_ctx, image_path, 4);

	REQUIRE(!image_embeds.empty());
	int n_image_tokens = image_embeds.size() / n_embd;
	std::cout << "  ✓ Generated " << n_image_tokens << " image embedding tokens" << std::endl;

	std::vector<llama_pos> image_pos(n_image_tokens);
	for (int i = 0; i < n_image_tokens; i++)
	{
		image_pos[i] = n_past + i;
	}

	llama_batch image_batch = {};
	image_batch.n_tokens = n_image_tokens;
	image_batch.embd = image_embeds.data();
	image_batch.pos = image_pos.data();

	std::cout << "  Submitting image embeddings..." << std::endl;
	result = llama_decode(ctx, image_batch);
	REQUIRE(result == 0);
	n_past += n_image_tokens;
	std::cout << "  ✓ Image embeddings processed" << std::endl;

	/* Tokenize and submit audio marker (OPTIMIZED - batched submission) */
	std::vector<llama_token> audio_marker_tokens;
	int32_t n_audio_marker =
		-llama_tokenize(vocab, audio_marker.data(), audio_marker.size(), nullptr, 0, false, false);
	REQUIRE(n_audio_marker > 0);
	audio_marker_tokens.resize(n_audio_marker);
	n_audio_marker = llama_tokenize(
		vocab,
		audio_marker.data(),
		audio_marker.size(),
		audio_marker_tokens.data(),
		audio_marker_tokens.size(),
		false,
		false);

	// Submit all audio marker tokens in a single batch
	std::vector<llama_pos> audio_marker_pos(n_audio_marker);
	std::vector<int8_t> audio_marker_logits(n_audio_marker, 0);
	for (int32_t i = 0; i < n_audio_marker; i++)
	{
		audio_marker_pos[i] = n_past + i;
	}

	llama_batch audio_marker_batch = {};
	audio_marker_batch.n_tokens = n_audio_marker;
	audio_marker_batch.token = audio_marker_tokens.data();
	audio_marker_batch.pos = audio_marker_pos.data();
	audio_marker_batch.logits = audio_marker_logits.data();

	result = llama_decode(ctx, audio_marker_batch);
	REQUIRE(result == 0);
	n_past += n_audio_marker;

	/* Submit audio embeddings (real encoder) */
	VendorPath audio_model_path{path_model_audio()};
	VendorPath audio_path{path_test_data_dir() + "/say_the_word_banana.wav"};

	std::cout << "  Loading and encoding audio..." << std::endl;
	std::vector<float> audio_embeds = load_and_encode_audio(audio_model_path, audio_path, true, 4);
	REQUIRE(!audio_embeds.empty());

	int n_audio_tokens = audio_embeds.size() / n_embd;
	std::vector<llama_pos> audio_pos(n_audio_tokens);

	for (int i = 0; i < n_audio_tokens; i++)
	{
		audio_pos[i] = n_past + i;
	}

	llama_batch audio_batch = {};
	audio_batch.n_tokens = n_audio_tokens;
	audio_batch.embd = audio_embeds.data();
	audio_batch.pos = audio_pos.data();

	std::cout << "  Submitting " << n_audio_tokens << " audio embedding tokens..." << std::endl;
	result = llama_decode(ctx, audio_batch);
	REQUIRE(result == 0);
	n_past += n_audio_tokens;
	std::cout << "  ✓ Audio embeddings processed" << std::endl;

	/* Cleanup CLIP encoder */
	clip_free(clip_ctx);

	/* Submit prompt suffix and generate response (OPTIMIZED - batched submission) */
	std::vector<llama_token> suffix_tokens;
	int32_t n_suffix = -llama_tokenize(
		vocab, prompt_suffix.data(), prompt_suffix.size(), nullptr, 0, false, false);
	REQUIRE(n_suffix > 0);
	suffix_tokens.resize(n_suffix);
	n_suffix = llama_tokenize(
		vocab,
		prompt_suffix.data(),
		prompt_suffix.size(),
		suffix_tokens.data(),
		suffix_tokens.size(),
		false,
		false);

	std::cout << "  ✓ Tokenized suffix: " << n_suffix << " tokens" << std::endl;

	// Submit all suffix tokens in a single batch
	std::vector<llama_pos> suffix_pos(n_suffix);
	std::vector<int8_t> suffix_logits(n_suffix, 0);
	for (int32_t i = 0; i < n_suffix; i++)
	{
		suffix_pos[i] = n_past + i;
	}
	suffix_logits[n_suffix - 1] = 1;  // Only last token needs logits

	llama_batch suffix_batch = {};
	suffix_batch.n_tokens = n_suffix;
	suffix_batch.token = suffix_tokens.data();
	suffix_batch.pos = suffix_pos.data();
	suffix_batch.logits = suffix_logits.data();

	result = llama_decode(ctx, suffix_batch);
	REQUIRE(result == 0);
	n_past += n_suffix;

	/* Generate response */
	std::cout << "  Generating response..." << std::endl;
	constexpr int max_tokens = 500;
	std::string response = generate_response(ctx, vocab, n_past, max_tokens);

	std::cout << "  Generated response: \"" << response << "\"" << std::endl;

	/* Validate response - the model should generate meaningful output */
	REQUIRE(response.size() > 0);
	std::cout << "  ✓ Model generated " << response.size() << " characters" << std::endl;

	/* Validate content - the model should identify both the image and audio content */
	std::string response_lower = response;
	std::transform(response_lower.begin(), response_lower.end(), response_lower.begin(), ::tolower);
	REQUIRE(response_lower.find("circle") != std::string::npos);
	std::cout << "  ✓ Response correctly identifies 'circle' from image" << std::endl;
	REQUIRE(response_lower.find("banana") != std::string::npos);
	std::cout << "  ✓ Response correctly transcribes 'banana' from audio" << std::endl;

	std::cout << "  Note: Using real CLIP encoder for images and real audition encoder for audio"
			  << std::endl;

	/* Cleanup */
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Multimodal combined test completed ===" << std::endl;
}