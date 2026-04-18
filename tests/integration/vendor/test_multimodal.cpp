/*
 * Multimodal input tests for MiniCPM-o-4_5 using llama.cpp embeddings.
 *
 * This test file demonstrates how to use llama.cpp's embedding API to submit
 * multimodal inputs (audio, image) alongside text tokens to the model.
 *
 * Key concepts:
 * - llama_batch can hold either tokens OR embeddings
 * - Setting batch.embd instead of batch.token allows custom embeddings
 * - Embeddings must match model's n_embd dimension
 * - Can mix token-based and embedding-based batches in sequence
 *
 * Prerequisites:
 * - llama.cpp with embedding support
 * - MiniCPM-o model loaded
 *
 * Note: This file uses dummy/random embeddings for testing llama.cpp's
 * embedding API. For real multimodal inference with actual images/audio,
 * use the encoder functions from test_inference.cpp:
 * - load_clip_encoder() to initialize CLIP vision encoder
 * - encode_image_with_clip() to generate real image embeddings
 */

#include <catch2/catch_test_macros.hpp>
#include "llama.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vendor_paths.hpp"

/* Test embedding batch creation */
TEST_CASE("multimodal_batch_with_embeddings", "[multimodal][batch]")
{
	std::cout << "=== Embedding Batch Creation Test ===" << std::endl;

	/* Step 1: Load model to get embedding dimension */
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model_params model_params = llama_model_default_params();
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	int32_t n_embd = llama_model_n_embd(model);
	std::cout << "  Model embedding dimension: " << n_embd << std::endl;
	REQUIRE(n_embd > 0);

	/*
	 * Step 2: Create batch with embedding space
	 *
	 * llama_batch_init() - Allocate a batch that can hold tokens OR embeddings
	 *
	 * Args:
	 *   n_tokens: Maximum number of tokens/embeddings the batch can hold
	 *   embd: Embedding dimension (if > 0, allocates space for embeddings)
	 *   n_seq_max: Maximum number of sequences per token
	 *
	 * Returns: llama_batch struct with allocated memory
	 *
	 * In LLM terms: A "batch" is a group of tokens to process together.
	 * Normally batches hold token IDs, but for multimodal models, we can
	 * also pass embedding vectors directly (e.g., from vision/audio encoders).
	 *
	 * Key fields in llama_batch:
	 * - n_tokens: How many tokens/embeddings are in this batch
	 * - token: Array of token IDs (used for text)
	 * - embd: Array of embedding vectors (used for images/audio)
	 * - pos: Array of positions in the sequence (0, 1, 2, ...)
	 * - seq_id: Which conversation/sequence each token belongs to
	 * - logits: Whether to output logits for each token (1) or not (0)
	 *
	 * When embd parameter > 0: Allocates embd array, token stays NULL
	 * When embd parameter == 0: Allocates token array, embd stays NULL
	 */
	int32_t n_tokens = 5;	// We'll process 5 embedding vectors
	int32_t n_seq_max = 1;	// Single sequence (one conversation)

	llama_batch batch = llama_batch_init(n_tokens, n_embd, n_seq_max);

	std::cout << "  Created batch:" << std::endl;
	std::cout << "    n_tokens: " << batch.n_tokens << std::endl;
	std::cout << "    embd ptr: " << (void *)batch.embd << std::endl;
	std::cout << "    token ptr: " << (void *)batch.token << std::endl;

	// When embd is allocated, token should be NULL (they're mutually exclusive)
	REQUIRE(batch.embd != nullptr);
	REQUIRE(batch.token == nullptr);

	/* Step 3: Fill batch with dummy embeddings */
	// For real multimodal inference, replace this with:
	// - load_clip_encoder() to load CLIP vision encoder
	// - encode_image_with_clip() to generate real image embeddings
	// - Or use libmtmd's whisper_preprocessor for audio embeddings
	batch.n_tokens = n_tokens;

	std::random_device rd;
	std::mt19937 gen(42);  // Fixed seed for reproducibility
	std::normal_distribution<float> dis(0.0f, 1.0f);

	for (int32_t i = 0; i < n_tokens; i++)
	{
		// Fill each embedding with random values
		for (int32_t j = 0; j < n_embd; j++)
		{
			batch.embd[i * n_embd + j] = dis(gen);
		}

		// Set position
		batch.pos[i] = i;

		// Set sequence ID (seq_id array already allocated by llama_batch_init)
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;	 // Sequence 0

		// Set logits flag (output logits for last token only)
		batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
	}

	std::cout << "  ✓ Filled batch with dummy embeddings" << std::endl;
	std::cout << "  Note: For real embeddings, use load_clip_encoder() + encode_image_with_clip()"
			  << std::endl;

	/* Step 4: Clean up */
	llama_batch_free(batch);
	llama_model_free(model);

	std::cout << "=== Embedding batch test passed ===" << std::endl;
}

/* Test submitting embeddings to model context */
TEST_CASE("multimodal_submit_embeddings", "[multimodal][context]")
{
	std::cout << "=== Embedding Submission Test ===" << std::endl;

	/* Step 1: Load model and create context */
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model_params model_params = llama_model_default_params();
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	/*
	 * Create inference context
	 *
	 * llama_init_from_model() - Create a context for inference
	 *
	 * Args:
	 *   model: The loaded model (weights)
	 *   params: Context parameters (context size, batch size, etc.)
	 *
	 * Returns: Pointer to llama_context, or nullptr on failure
	 *
	 * In LLM terms: If the model is the "brain" (neural network weights),
	 * the context is the "working memory" where actual thinking happens.
	 *
	 * The context includes:
	 * - KV cache: Stores attention keys/values for all processed tokens
	 *             (This is what makes generation efficient - we don't
	 *              recompute attention for previous tokens)
	 * - Computation buffers: Temporary memory for forward pass
	 * - Thread pool: For parallel computation
	 *
	 * Multiple contexts can share the same model (multiple conversations
	 * using the same neural network).
	 */
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 2048;	   // Max 2048 tokens in context window
	ctx_params.n_batch = 512;	   // Process up to 512 tokens at once
	ctx_params.embeddings = true;  // Enable embedding output (for similarity, etc.)

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);
	std::cout << "  ✓ Context created with embedding support" << std::endl;

	int32_t n_embd = llama_model_n_embd(model);

	/* Step 2: Create and fill embedding batch */
	int32_t n_embed_tokens = 3;	 // Simulate 3 audio/image embeddings
	llama_batch embed_batch = llama_batch_init(n_embed_tokens, n_embd, 1);
	embed_batch.n_tokens = n_embed_tokens;

	std::random_device rd;
	std::mt19937 gen(42);
	std::normal_distribution<float> dis(0.0f, 0.1f);  // Small values

	for (int32_t i = 0; i < n_embed_tokens; i++)
	{
		for (int32_t j = 0; j < n_embd; j++)
		{
			embed_batch.embd[i * n_embd + j] = dis(gen);
		}
		embed_batch.pos[i] = i;
		embed_batch.n_seq_id[i] = 1;
		embed_batch.seq_id[i][0] = 0;
		embed_batch.logits[i] = (i == n_embed_tokens - 1) ? 1 : 0;
	}

	std::cout << "  ✓ Created embedding batch" << std::endl;
	std::cout << "  Note: For real embeddings, use load_clip_encoder() + encode_image_with_clip()"
			  << std::endl;

	/*
	 * Step 3: Process embeddings through the model
	 *
	 * llama_decode() - Run inference on a batch of tokens/embeddings
	 *
	 * Args:
	 *   ctx: The inference context
	 *   batch: Batch of tokens or embeddings to process
	 *
	 * Returns: 0 on success, non-zero on error
	 *
	 * In LLM terms: This is the "forward pass" through the neural network.
	 * It processes the input (tokens or embeddings) through all transformer layers:
	 *
	 * 1. For tokens: Looks up embedding vectors from the embedding table
	 *    For embeddings: Uses them directly (bypasses embedding table)
	 *
	 * 2. Runs through each transformer layer:
	 *    - Multi-head self-attention (tokens attend to previous tokens)
	 *    - Feed-forward network (MLP)
	 *    - Layer normalization
	 *
	 * 3. Stores attention keys/values in KV cache for efficient generation
	 *
	 * 4. Outputs logits (next-token probabilities) or embeddings
	 *
	 * This is the computationally expensive part - running billions of
	 * floating-point operations through the neural network.
	 */
	int result = llama_decode(ctx, embed_batch);

	if (result == 0)
	{
		std::cout << "  ✓ Embeddings successfully processed!" << std::endl;

		/*
		 * llama_get_embeddings() - Get output embeddings from last decode
		 *
		 * Args:
		 *   ctx: The inference context
		 *
		 * Returns: Pointer to embeddings array, or nullptr if not available
		 *
		 * In LLM terms: After processing input, the model produces either:
		 * - Logits: Probability distribution over vocabulary (for generation)
		 * - Embeddings: Dense vector representation (for similarity/classification)
		 *
		 * Embeddings are useful for:
		 * - Semantic similarity (cosine similarity between sentence embeddings)
		 * - Classification (use as features for downstream tasks)
		 * - Retrieval (find similar documents)
		 *
		 * Must set ctx_params.embeddings = true to get these.
		 */
		float * output_embeddings = llama_get_embeddings(ctx);
		if (output_embeddings != nullptr)
		{
			std::cout << "  ✓ Output embeddings available" << std::endl;

			// Print first few values
			std::cout << "  First 5 output values: ";
			for (int i = 0; i < 5; i++)
			{
				std::cout << output_embeddings[i] << " ";
			}
			std::cout << std::endl;
		}
	}
	else
	{
		std::cout << "  ⚠ Decode returned: " << result << std::endl;
		std::cout << "  Note: This may fail if model doesn't support direct embedding input"
				  << std::endl;
		// Use CHECK (not REQUIRE) to document expectation but allow test to continue
		CHECK(result == 0);
	}

	/* Step 4: Clean up */
	llama_batch_free(embed_batch);
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Embedding submission test completed ===" << std::endl;
}

/* Test mixed token and embedding workflow - using omni.cpp minimal batch approach */
TEST_CASE("multimodal_mixed_tokens_embeddings", "[multimodal][mixed]")
{
	std::cout << "=== Mixed Token/Embedding Test (Minimal Batch Approach) ===" << std::endl;

	/* This test uses the minimal batch setup from llama.cpp-omni/tools/omni/omni.cpp:
	 * - Stack-allocated batches with zero initialization
	 * - Only set essential fields: n_tokens, embd/token, pos
	 * - Track position counter (n_past) manually
	 *
	 * Pattern from omni.cpp omni_eval_embed():
	 *   llama_batch batch = {};
	 *   batch.n_tokens = n_eval;
	 *   batch.embd = embed_data;
	 *   batch.pos = pos_array;
	 *   llama_decode(ctx, batch);
	 */

	/* Step 1: Setup */
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model_params model_params = llama_model_default_params();
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 2048;
	ctx_params.n_batch = 512;

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_embd = llama_model_n_embd(model);

	std::cout << "  ✓ Model and context initialized" << std::endl;

	// Track position counter like omni.cpp does
	int n_past = 0;

	/* Step 2: Submit multimodal embeddings using minimal batch */
	int32_t n_embed_tokens = 2;	 // Simulate 2 embedding vectors

	// Allocate embedding data
	std::vector<float> embed_data(n_embed_tokens * n_embd);
	std::vector<llama_pos> embed_pos(n_embed_tokens);

	// Fill with dummy data (for real usage, use encode_image_with_clip())
	std::random_device rd;
	std::mt19937 gen(42);
	std::normal_distribution<float> dis(0.0f, 0.1f);

	for (int32_t i = 0; i < n_embed_tokens; i++)
	{
		for (int32_t j = 0; j < n_embd; j++)
		{
			embed_data[i * n_embd + j] = dis(gen);
		}
		embed_pos[i] = n_past + i;
	}

	// Create minimal batch (like omni.cpp)
	llama_batch embed_batch = {};
	embed_batch.n_tokens = n_embed_tokens;
	embed_batch.embd = embed_data.data();
	embed_batch.pos = embed_pos.data();
	// Note: seq_id, n_seq_id, logits, token all remain NULL/0 from zero-init

	std::cout << "  Submitting " << n_embed_tokens << " embedding vectors..." << std::endl;
	int result = llama_decode(ctx, embed_batch);
	std::cout << "  Embedding decode result: " << result << std::endl;
	REQUIRE(result == 0);

	n_past += n_embed_tokens;
	std::cout << "  ✓ Embeddings processed, n_past now: " << n_past << std::endl;

	/* Step 3: Follow with text tokens */
	std::string const text_prompt = "Describe this.";

	std::vector<llama_token> tokens;
	int32_t n_tokens =
		-llama_tokenize(vocab, text_prompt.data(), text_prompt.size(), nullptr, 0, true, true);
	REQUIRE(n_tokens > 0);
	tokens.resize(n_tokens);
	n_tokens = llama_tokenize(
		vocab, text_prompt.data(), text_prompt.size(), tokens.data(), tokens.size(), true, true);

	std::cout << "  ✓ Tokenized prompt: " << n_tokens << " tokens" << std::endl;

	// Submit tokens one by one using minimal batch
	for (int32_t i = 0; i < n_tokens; i++)
	{
		llama_pos token_pos = n_past + i;

		// Minimal batch for single token
		llama_batch token_batch = {};
		token_batch.n_tokens = 1;
		token_batch.token = &tokens[i];
		token_batch.pos = &token_pos;
		// Output logits only for last token
		int8_t logit_flag = (i == n_tokens - 1) ? 1 : 0;
		token_batch.logits = &logit_flag;

		result = llama_decode(ctx, token_batch);
		if (result != 0)
		{
			std::cout << "  ✗ Token decode failed at position " << token_pos << ": " << result
					  << std::endl;
			REQUIRE(result == 0);
		}
	}

	n_past += n_tokens;
	std::cout << "  ✓ Text tokens processed, n_past now: " << n_past << std::endl;

	// Verify we can get logits
	float * logits = llama_get_logits(ctx);
	REQUIRE(logits != nullptr);
	std::cout << "  ✓ Logits available for generation" << std::endl;

	/* Step 4: Clean up */
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Mixed token/embedding test completed ===" << std::endl;
}

/* Test understanding model's expected embedding format */
TEST_CASE("multimodal_embedding_properties", "[multimodal][properties]")
{
	std::cout << "=== Embedding Properties Test ===" << std::endl;

	/* Query model for embedding-related properties */
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model_params model_params = llama_model_default_params();
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	/* Get various model dimensions */
	int32_t n_embd = llama_model_n_embd(model);
	int32_t n_layer = llama_model_n_layer(model);
	int32_t n_head = llama_model_n_head(model);
	int32_t n_ctx_train = llama_model_n_ctx_train(model);

	std::cout << "Model properties:" << std::endl;
	std::cout << "  Embedding dimension: " << n_embd << std::endl;
	std::cout << "  Number of layers: " << n_layer << std::endl;
	std::cout << "  Number of heads: " << n_head << std::endl;
	std::cout << "  Training context: " << n_ctx_train << std::endl;

	// For MiniCPM-o, we expect embeddings that match n_embd
	REQUIRE(n_embd > 0);
	REQUIRE(n_layer > 0);

	std::cout << "\nFor multimodal inputs with real encoders:" << std::endl;
	std::cout << "  - Audio embeddings should be: [n_audio_tokens, " << n_embd << "]" << std::endl;
	std::cout << "  - Image embeddings should be: [n_image_tokens, " << n_embd << "]" << std::endl;
	std::cout << "  - Each embedding vector must match n_embd dimension" << std::endl;
	std::cout << "\nTo use real encoders:" << std::endl;
	std::cout << "  1. Include clip.h header" << std::endl;
	std::cout << "  2. Call load_clip_encoder(vision_model_path) to initialize CLIP" << std::endl;
	std::cout << "  3. Call encode_image_with_clip(clip_ctx, image_path) to get embeddings"
			  << std::endl;

	llama_model_free(model);

	std::cout << "=== Embedding properties test passed ===" << std::endl;
}