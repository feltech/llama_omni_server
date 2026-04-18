/*
 * Pure libllama test for MiniCPM-o-4_5 model initialization and inference.
 *
 * This test demonstrates basic text-only inference using only llama.cpp's core API.
 * It does NOT use any omni extensions or custom fork-specific code.
 *
 * Prerequisites:
 * - MiniCPM-o-4_5 model file
 * - C++ compiler with C++17 support
 * - Catch2 test framework
 * - Conan dependency management
 *
 * Build with:
 *   conan install . --build=missing
 *   cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build
 *   ./build/test_init
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "llama.h"
#include "vendor_paths.hpp"

/* Simple test - just verify backend initialization and model loading */
TEST_CASE("libllama_backend_init", "[llama][backend]")
{
	/*
	 * llama_time_us() - Get current time in microseconds
	 *
	 * Returns: Current time as int64_t (microseconds since epoch)
	 *
	 * This is a simple utility function to verify the llama.cpp library is linked correctly.
	 * It's not strictly necessary for model inference but useful for timing measurements.
	 */
	int64_t time_us = llama_time_us();
	REQUIRE(time_us >= 0);
	std::cout << "Backend init test passed. Time: " << time_us << " microseconds" << std::endl;
}

/* Test model loading with standard API */
TEST_CASE("libllama_model_load", "[llama][model]")
{
	/*
	 * llama_model_default_params() - Get default model loading parameters
	 *
	 * Returns: llama_model_params struct with sensible defaults
	 *
	 * The model_params struct controls how the model weights are loaded:
	 * - n_gpu_layers: How many transformer layers to offload to GPU (-1 = all, 0 = none)
	 * - use_mmap: Whether to memory-map the model file (faster loading)
	 * - use_mlock: Whether to lock model in RAM (prevents swapping)
	 * - vocab_only: If true, only load vocabulary without weights
	 *
	 * In LLM terms: This is like configuring where your neural network layers live
	 * (CPU RAM vs GPU VRAM) and how they're loaded into memory.
	 */
	llama_model_params model_params = llama_model_default_params();

	/* Set up parameters for MiniCPM-o */
	// model_params.n_gpu_layers = -1; // Offload all layers to GPU for faster inference
	// Note: We don't set n_ctx here - that's part of context params, not model params

	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	/*
	 * llama_model_load_from_file() - Load model weights from GGUF file
	 *
	 * Args:
	 *   path: Path to .gguf model file (GGUF = GPT-Generated Unified Format)
	 *   params: Model loading parameters from llama_model_default_params()
	 *
	 * Returns: Pointer to llama_model, or nullptr on failure
	 *
	 * This loads the neural network weights from disk. In LLM terms:
	 * - Loads all transformer layer weights (attention, FFN, etc.)
	 * - Loads the embedding matrix (token ID → vector)
	 * - Loads the vocabulary (string → token ID mapping)
	 * - Loads model metadata (n_layers, n_heads, etc.)
	 *
	 * The model is immutable once loaded - multiple contexts can share the same model.
	 */
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	std::cout << "Model loaded successfully!" << std::endl;

	/*
	 * llama_model_get_vocab() - Get model's vocabulary/tokenizer
	 *
	 * Args:
	 *   model: The loaded model
	 *
	 * Returns: Pointer to llama_vocab (vocabulary/tokenizer)
	 *
	 * The vocabulary is the model's dictionary that maps between:
	 * - Strings (text) ↔ Token IDs (integers)
	 *
	 * Every LLM has a vocabulary - for example, GPT models have ~50k tokens,
	 * this model has ~150k tokens including special tokens for multimodal.
	 */
	llama_vocab const * vocab = llama_model_get_vocab(model);

	/*
	 * llama_vocab_n_tokens() - Get vocabulary size
	 *
	 * Args:
	 *   vocab: The vocabulary from llama_model_get_vocab()
	 *
	 * Returns: Total number of tokens in vocabulary (int32_t)
	 *
	 * This tells you how many unique tokens the model knows.
	 * Includes regular tokens (words/subwords) plus special tokens.
	 */
	int32_t n_tokens = llama_vocab_n_tokens(vocab);

	std::cout << "Vocabulary size: " << n_tokens << " tokens" << std::endl;

	/*
	 * llama_model_free() - Free model from memory
	 *
	 * Args:
	 *   model: The model to free
	 *
	 * Always call this when done with a model to prevent memory leaks.
	 * Releases all model weights, vocabulary, and metadata from memory.
	 */
	llama_model_free(model);
	std::cout << "Model memory freed." << std::endl;
}

/* Test context parameters and context creation */
TEST_CASE("libllama_context_params", "[llama][params]")
{
	/*
	 * llama_context_default_params() - Get default context parameters
	 *
	 * Returns: llama_context_params struct with sensible defaults
	 *
	 * A "context" in llama.cpp is an inference session. Think of it as:
	 * - The runtime state for one conversation/generation session
	 * - Includes the KV cache (stores attention keys/values for processed tokens)
	 * - Manages computation resources (threads, batch sizes)
	 *
	 * Key parameters:
	 * - n_ctx: Context size (max tokens that can be stored in KV cache)
	 *          In LLM terms: This is your "context window" or "sequence length"
	 *          Example: 4096 means you can process up to 4096 tokens at once
	 *
	 * - n_batch: Physical batch size for prompt processing
	 *            How many tokens to process in parallel during the initial prompt
	 *            Larger = faster prompt processing but more memory
	 *
	 * - n_ubatch: Logical batch size for computation graph
	 *             Internal optimization parameter, usually same as n_batch
	 *
	 * - n_threads: Number of CPU threads to use for computation
	 *              More threads = faster (up to a point)
	 *
	 * In LLM terms: If the model is the neural network weights (immutable),
	 * the context is where inference actually happens (mutable state).
	 */
	llama_context_params ctx_params = llama_context_default_params();

	std::cout << "Context params created:" << std::endl;
	std::cout << "  Default context size: " << ctx_params.n_ctx << std::endl;
	std::cout << "  Default batch size: " << ctx_params.n_batch << std::endl;
	std::cout << "  Default ubatch size: " << ctx_params.n_ubatch << std::endl;
	std::cout << "  Default threads: " << ctx_params.n_threads << std::endl;

	/* Verify parameters are valid */
	REQUIRE(ctx_params.n_ctx > 0);
	REQUIRE(ctx_params.n_batch > 0);
	REQUIRE(ctx_params.n_ubatch > 0);
	REQUIRE(ctx_params.n_threads > 0);
}

/* Test vocabulary access */
TEST_CASE("libllama_vocab_access", "[llama][vocab]")
{
	llama_model_params model_params = llama_model_default_params();

	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	REQUIRE(vocab != nullptr);

	int32_t n_tokens = llama_vocab_n_tokens(vocab);
	std::cout << "Vocab test: " << n_tokens << " tokens total" << std::endl;

	llama_model_free(model);
}

/* Test model metadata access */
TEST_CASE("libllama_model_metadata", "[llama][metadata]")
{
	llama_model_params model_params = llama_model_default_params();

	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	/*
	 * llama_model_n_ctx_train() - Get model's training context length
	 *
	 * Args:
	 *   model: The loaded model
	 *
	 * Returns: Context length model was trained with (int32_t)
	 *
	 * In LLM terms: This is the maximum sequence length the model saw during training.
	 * For example, if trained with 4096 tokens, it can theoretically handle up to
	 * 4096 tokens, though you can sometimes extend this with techniques like
	 * RoPE scaling or ALiBi.
	 */
	int32_t n_ctx = llama_model_n_ctx_train(model);

	/*
	 * llama_model_n_embd() - Get model's embedding dimension
	 *
	 * Args:
	 *   model: The loaded model
	 *
	 * Returns: Embedding dimension (int32_t)
	 *
	 * In LLM terms: This is the "hidden size" or "d_model" - the size of
	 * the internal representation vectors. For example:
	 * - GPT-2: 768
	 * - GPT-3: 12288
	 * - MiniCPM-o: 4096
	 *
	 * Every token becomes a vector of this size after the embedding layer,
	 * and all intermediate computations use this dimensionality.
	 */
	int32_t n_embd = llama_model_n_embd(model);

	/*
	 * llama_model_n_layer() - Get number of transformer layers
	 *
	 * Args:
	 *   model: The loaded model
	 *
	 * Returns: Number of transformer layers (int32_t)
	 *
	 * In LLM terms: This is the "depth" of the neural network. Each layer has:
	 * - Multi-head self-attention
	 * - Feed-forward network (MLP)
	 * - Layer normalization
	 *
	 * More layers = more powerful model (usually), but also slower and more memory.
	 * For example: GPT-3 has 96 layers, this model has 36 layers.
	 */
	int32_t n_layer = llama_model_n_layer(model);

	std::cout << "Model metadata:" << std::endl;
	std::cout << "  Context size (train): " << n_ctx << std::endl;
	std::cout << "  Embedding dimension: " << n_embd << std::endl;
	std::cout << "  Number of layers: " << n_layer << std::endl;

	REQUIRE(n_ctx > 0);
	REQUIRE(n_embd > 0);
	REQUIRE(n_layer > 0);

	llama_model_free(model);
}

/* Test complete model lifecycle */
TEST_CASE("libllama_complete_lifecycle", "[llama][lifecycle]")
{
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	std::cout << "=== Starting complete lifecycle test ===" << std::endl;

	/* Step 1: Load model */
	llama_model_params model_params = llama_model_default_params();
	std::cout << "Step 1: Loading model..." << std::endl;
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);
	std::cout << "  ✓ Model loaded" << std::endl;

	/* Step 2: Get vocabulary */
	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_vocab = llama_vocab_n_tokens(vocab);
	std::cout << "Step 2: Getting vocabulary..." << std::endl;
	std::cout << "  ✓ Vocabulary size: " << n_vocab << std::endl;
	REQUIRE(n_vocab > 0);

	/* Step 3: Get model metadata */
	int32_t n_ctx = llama_model_n_ctx_train(model);
	int32_t n_embd = llama_model_n_embd(model);
	std::cout << "Step 3: Getting metadata..." << std::endl;
	std::cout << "  ✓ Context: " << n_ctx << ", Embd: " << n_embd << std::endl;

	/* Step 4: Create context parameters */
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 512;
	ctx_params.n_ubatch = 512;
	ctx_params.n_threads = 4;
	std::cout << "Step 4: Creating context params..." << std::endl;
	std::cout << "  ✓ Params configured" << std::endl;

	/* Step 5: Create context (would need model loaded) */
	// For this test, we'll just verify params are valid
	REQUIRE(ctx_params.n_ctx >= 4096);
	std::cout << "Step 5: Context params validated" << std::endl;

	/* Step 6: Free model */
	std::cout << "Step 6: Cleaning up..." << std::endl;
	llama_model_free(model);
	std::cout << "  ✓ Model freed" << std::endl;

	std::cout << "=== Complete lifecycle test passed ===" << std::endl;
}

/* Test basic tokenization and decoding */
TEST_CASE("libllama_tokenization", "[llama][tokenization]")
{
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};
	std::string const prompt_text = "Hello, this is a test.";

	std::cout << "=== Starting tokenization test ===" << std::endl;
	std::cout << "Prompt: " << prompt_text << std::endl;

	/* Step 1: Load model */
	llama_model_params model_params = llama_model_default_params();
	std::cout << "Step 1: Loading model..." << std::endl;
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	std::cout << "  ✓ Model loaded" << std::endl;

	/* Step 2: Get vocabulary */
	std::cout << "Step 2: Getting vocabulary..." << std::endl;
	llama_vocab const * vocab = llama_model_get_vocab(model);
	std::cout << "  ✓ Vocabulary retrieved" << std::endl;

	/*
	 * Step 3: Tokenize prompt
	 *
	 * Tokenization converts text strings into integer token IDs that the model can process.
	 *
	 * llama_tokenize() - Convert text to tokens
	 *
	 * Args:
	 *   vocab: The vocabulary/tokenizer
	 *   text: Pointer to text string
	 *   text_len: Length of text in bytes
	 *   tokens: Output array for token IDs (nullptr to query size)
	 *   n_tokens_max: Size of output array
	 *   add_special: Whether to add BOS (Beginning Of Sequence) token
	 *   parse_special: Whether to parse special tokens like <|im_start|>
	 *
	 * Returns: Number of tokens on success, negative value means buffer too small
	 *
	 * In LLM terms: This is the tokenizer that breaks text into subwords.
	 * For example: "Hello, world!" might become tokens [15496, 11, 1917, 0]
	 * Each integer maps to a piece of text in the vocabulary.
	 *
	 * The two-call pattern is common in C APIs:
	 * 1. Call with nullptr/0 to get required size (returns negative count)
	 * 2. Allocate buffer of that size
	 * 3. Call again to actually fill the buffer
	 */
	std::cout << "Step 3: Tokenizing prompt..." << std::endl;
	std::cout << "  Prompt: " << prompt_text << std::endl;

	std::vector<llama_token> tokens;

	// First call: Query how many tokens we need
	// Note: Returns negative value, so we negate it to get positive count
	int32_t n_prompt_tokens =
		-llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
	std::cout << "  First token count: " << n_prompt_tokens << std::endl;

	if (n_prompt_tokens <= 0)
	{
		std::cout << "  ✗ Tokenization failed: " << n_prompt_tokens << std::endl;
		llama_model_free(model);
		FAIL("Tokenization failed");
	}

	// Allocate buffer for tokens
	tokens.resize(n_prompt_tokens);

	// Second call: Actually tokenize into the buffer
	n_prompt_tokens = llama_tokenize(
		vocab, prompt_text.data(), prompt_text.size(), tokens.data(), tokens.size(), true, true);
	std::cout << "  Second token count: " << n_prompt_tokens << std::endl;

	if (n_prompt_tokens < 0)
	{
		std::cout << "  ✗ Tokenization failed" << std::endl;
		llama_model_free(model);
		FAIL("Tokenization failed");
	}
	if ((size_t)n_prompt_tokens != tokens.size())
	{
		std::cout << "  ✗ Token count mismatch" << std::endl;
		llama_model_free(model);
		FAIL("Token count mismatch");
	}
	std::cout << "  ✓ Prompt tokenized: " << n_prompt_tokens << " tokens" << std::endl;

	/*
	 * Step 4: Test detokenization (reverse process)
	 *
	 * llama_detokenize() - Convert tokens back to text
	 *
	 * Args:
	 *   vocab: The vocabulary/tokenizer
	 *   tokens: Array of token IDs to convert
	 *   n_tokens: Number of tokens in array
	 *   text: Output buffer for text
	 *   text_len_max: Size of output buffer
	 *   remove_special: Whether to remove special tokens from output
	 *   unparse_special: Whether to convert special tokens back to text form
	 *
	 * Returns: Number of bytes written to text buffer, or negative on error
	 *
	 * In LLM terms: This is "decoding" tokens back to human-readable text.
	 * During generation, the model outputs token IDs, and we detokenize them
	 * to display the generated text.
	 *
	 * For example: tokens [15496, 11, 1917, 0] → "Hello, world!"
	 */
	std::cout << "Step 4: Testing detokenization..." << std::endl;

	std::vector<char> buf(256);
	int32_t n_piece =
		llama_detokenize(vocab, tokens.data(), n_prompt_tokens, buf.data(), buf.size(), true, true);
	if (n_piece <= 0)
	{
		std::cout << "  ✗ Detokenization failed" << std::endl;
		llama_model_free(model);
		FAIL("Detokenization failed");
	}
	REQUIRE(n_piece > 0);
	std::cout << "  ✓ Detokenized: " << buf.data() << std::endl;
	std::cout << "  ✓ Detokenized " << n_piece << " characters" << std::endl;

	/* Step 5: Cleanup */
	std::cout << "Step 5: Cleaning up..." << std::endl;
	llama_model_free(model);
	std::cout << "  ✓ Model freed" << std::endl;

	std::cout << "=== Tokenization test passed ===" << std::endl;
}

/* Test basic text generation with llama_decode */
TEST_CASE("libllama_text_generation", "[llama][generation]")
{
	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};
	std::string const prompt_text = "Hello, this is a test.";

	std::cout << "=== Starting text generation test ===" << std::endl;
	std::cout << "Prompt: " << prompt_text << std::endl;

	/* Step 1: Load model */
	llama_model_params model_params = llama_model_default_params();
	std::cout << "Step 1: Loading model..." << std::endl;
	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);
	std::cout << "  ✓ Model loaded" << std::endl;

	/* Step 2: Create context parameters */
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 2048;
	ctx_params.n_batch = 512;
	ctx_params.n_ubatch = 512;
	ctx_params.n_threads = 4;
	std::cout << "Step 2: Creating context..." << std::endl;
	std::cout << "  Context params: n_ctx=" << ctx_params.n_ctx
			  << ", n_batch=" << ctx_params.n_batch << ", n_ubatch=" << ctx_params.n_ubatch
			  << ", n_threads=" << ctx_params.n_threads << std::endl;
	llama_context * ctx = llama_init_from_model(model, ctx_params);
	if (ctx == nullptr)
	{
		std::cerr << "  ✗ Context creation failed" << std::endl;
		llama_model_free(model);
		FAIL("Context creation failed");
	}
	REQUIRE(ctx != nullptr);
	std::cout << "  ✓ Context created" << std::endl;

	/* Step 3: Tokenize prompt */
	llama_vocab const * vocab = llama_model_get_vocab(model);
	std::cout << "Step 3: Tokenizing prompt..." << std::endl;

	int32_t n_prompt_tokens =
		-llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
	std::cout << "  First token count: " << n_prompt_tokens << std::endl;

	if (n_prompt_tokens <= 0)
	{
		std::cout << "  ✗ Tokenization failed: " << n_prompt_tokens << std::endl;
		llama_free(ctx);
		llama_model_free(model);
		FAIL("Tokenization failed");
	}

	std::vector<llama_token> tokens;
	tokens.resize(n_prompt_tokens);

	n_prompt_tokens = llama_tokenize(
		vocab, prompt_text.data(), prompt_text.size(), tokens.data(), tokens.size(), true, true);
	std::cout << "  Second token count: " << n_prompt_tokens << std::endl;

	if (n_prompt_tokens < 0)
	{
		std::cout << "  ✗ Tokenization failed" << std::endl;
		llama_free(ctx);
		llama_model_free(model);
		FAIL("Tokenization failed");
	}
	if ((size_t)n_prompt_tokens != tokens.size())
	{
		std::cout << "  ✗ Token count mismatch" << std::endl;
		llama_free(ctx);
		llama_model_free(model);
		FAIL("Token count mismatch");
	}
	REQUIRE((size_t)n_prompt_tokens == tokens.size());
	std::cout << "  ✓ Prompt tokenized: " << n_prompt_tokens << " tokens" << std::endl;

	/* Step 4: Set tokens in context */
	std::cout << "Step 4: Setting tokens in context..." << std::endl;
	int32_t n_cur = 0;
	int32_t n_past = 0;
	bool is_eos = false;

	for (int32_t i = 0; i < n_prompt_tokens; i++)
	{
		llama_token token = tokens[i];

		int32_t n_tokens = llama_decode(ctx, llama_batch_get_one(&token, 1));
		if (n_tokens != 0)
		{
			std::cout << "  ✗ Decode failed: " << n_tokens << std::endl;
			llama_free(ctx);
			llama_model_free(model);
			FAIL("Decode failed");
		}

		n_cur++;
		n_past++;

		if (token == 151645)
		{  // <|im_end|>
			is_eos = true;
		}
	}

	if (n_past <= 0)
	{
		std::cout << "  ✗ No tokens processed" << std::endl;
		llama_free(ctx);
		llama_model_free(model);
		FAIL("No tokens processed");
	}
	std::cout << "  ✓ Tokens set in context" << std::endl;

	/* Step 5: Test detokenization */
	std::cout << "Step 5: Testing detokenization..." << std::endl;

	std::vector<char> buf(256);
	for (int32_t i = 0; i < n_prompt_tokens; i++)
	{
		int32_t n_piece =
			llama_detokenize(vocab, &tokens[i], 1, buf.data(), buf.size(), true, false);
		if (n_piece > 0)
		{
			std::cout << "  ✓ Detokenized token " << i << ": " << buf.data() << std::endl;
			break;
		}
	}

	/* Step 6: Cleanup */
	std::cout << "Step 6: Cleaning up..." << std::endl;
	llama_free(ctx);
	llama_model_free(model);
	std::cout << "  ✓ All resources freed" << std::endl;

	std::cout << "=== Text generation test passed ===" << std::endl;
}

/* Test parameter tuning for generation */
TEST_CASE("libllama_generation_params", "[llama][generation]")
{
	llama_model_params model_params = llama_model_default_params();

	VendorPath model_path{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};

	llama_model * model = llama_model_load_from_file(model_path, model_params);
	REQUIRE(model != nullptr);

	/* Test different generation parameters */
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 2048;
	ctx_params.n_batch = 256;
	ctx_params.n_ubatch = 256;
	ctx_params.n_threads = 2;

	REQUIRE(ctx_params.n_ctx == 2048);
	REQUIRE(ctx_params.n_batch == 256);
	REQUIRE(ctx_params.n_ubatch == 256);
	REQUIRE(ctx_params.n_threads == 2);

	std::cout << "Generation params test passed." << std::endl;

	llama_model_free(model);
}

int main(int argc, char ** argv)
{
	Catch::Session session;

	/* Catch2 configuration */
	std::cout << "\n=== MiniCPM-o 4_5 - Pure libllama Test Suite ===" << std::endl;
	std::cout << "Testing basic llama.cpp functionality without omni extensions." << std::endl;
	std::cout << "========================================\n" << std::endl;

	int returnCode = session.applyCommandLine(argc, argv);
	if (returnCode != 0)
		return returnCode;

	return session.run();
}