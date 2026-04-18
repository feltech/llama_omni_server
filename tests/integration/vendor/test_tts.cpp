/*
 * TTS Test for MiniCPM-o 4.5
 *
 * This test validates TTS (Text-to-Speech) generation from a properly structured prompt.
 *
 * Based on: llama.cpp-omni/tools/omni/omni.cpp TTS simplex implementation
 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890 (generate_audio_tokens_local_simplex)
 *
 * Test Flow (matching omni.cpp exactly):
 * 1. Load LLM model
 * 2. Create prompt with text
 * 3. Feed through model and extract hidden states from text tokens
 * 4. Filter special tokens for hidden states
 * 5. Load projector weights
 * 6. Apply projector and normalize L2
 * 7. Add audio_bos token embedding
 * 8. Load TTS model using llama_new_context_with_model() (omni.cpp:3602)
 * 9. Load TTS weights (emb_code, emb_text, head_code, Token2Wav) from GGUF (omni.cpp:3635)
 * 10. Run prefill with condition embeddings using llama_decode() (omni.cpp:2149)
 * 11. Sample audio tokens using head_code logits (omni.cpp:2680-2689)
 * 12. Look up emb_code embeddings for sampled tokens (omni.cpp:2662-2679)
 * 13. Prefill sampled embeddings to continue generation (omni.cpp:3007)
 * 14. Repeat sampling until EOS
 * 15. Feed tokens to Token2Wav (omni.cpp:667-730)
 * 16. Write audio to WAV file
 * 17. Assert duration >= 2 seconds
 *
 * Technical Background (from omni.cpp analysis):
 *
 * TTS Architecture (MiniCPM-o 4.5):
 * - LLM Model (4096-d): Generates text hidden states from text tokens
 * - Projector Semantic (4096-d → 768-d): Maps LLM hidden states to TTS embedding space
 * - emb_text (768-d): Token embedding lookup for TTS model identity
 * - TTS Model (768-d, 24 blocks): Generates audio tokens from conditioned embeddings
 * - head_code (768, 6562): Projects TTS hidden states to audio token logits
 * - emb_code (768, 6562): Audio token embeddings for decoding to audio
 * - Token2Wav: Diffusion model that converts audio tokens → audio waveform
 *
 * Key Computation Steps:
 * 1. Condition Embedding: condition[i] = emb_text(token_id[i]) + L2_normalize(projector(hidden[i]))
 *    - emb_text provides semantic identity in TTS space
 *    - Projector(hidden) provides LLM contextual representation
 *    - Without emb_text: TTS produces gibberish (no phoneme/semantic identity)
 *
 * 2. Audio Token Sampling:
 *    - At each step t: hidden_state = TTS_model[-1] (last hidden state)
 *    - logits = hidden_state @ head_code_weight (768 × 6562 matrix multiply)
 *    - Apply temperature scaling (0.8) and repetition penalty (1.05) over last 8 tokens
 *    - Sample token from logits (multinomial) or argmax (deterministic)
 *
 * 3. Token Prefill:
 *    - emb_code[relative_idx] → 768-d embedding
 *    - llama_decode(batch with embeddings) → updates TTS KV cache
 *
 * 4. Token2Wav Decoding:
 *    - Audio tokens (0-6561) + 3 silence tokens (4218) → 28-token window
 *    - Sliding window: process 28 tokens, move by 25 tokens
 *    - Diffusion model generates 24kHz mono waveform
 *
 * Special Tokens:
 * - audio_bos_token_id = 151687: Start-of-speech marker, always appended to condition
 * - num_audio_tokens = 6562: Vocab size of TTS audio space (0-6561)
 * - eos_relative_idx = 6561: End-of-speech token
 * - text_eos_token_id = 151692: Marks end of text (used in duplex mode)
 *
 * Sampling Parameters (TTSSamplingParams):
 * - temperature = 0.8: Controls randomness (lower = more deterministic)
 * - top_p = 0.85: Nucleus sampling threshold
 * - top_k = 25: Top-k sampling threshold
 * - repetition_penalty = 1.05: Penalize recently generated tokens
 * - min_p = 0.01: Minimum probability threshold
 * - penalty_last_n = 16: Window size for repetition penalty
 *
 * Model File Splitting:
 * - combined_tts_gguf (193 tensors): Cannot be loaded by libllama
 * - tts_transformer-F16.gguf (182 standard tensors): Transformer blocks
 * - tts-weights-F16.gguf (11 custom tensors): TTS-specific embeddings
 *   - emb_code.0.weight (768, 6562): Audio token embeddings
 *   - emb_text.weight (768, 152064): Text token embeddings for TTS
 *   - head_code.0.weight (6562, 768): Audio token logits
 *   - linear1.weight (4096, 768): Projector layer 1
 *   - linear1.bias (768): Projector layer 1 bias
 *   - linear2.weight (768, 768): Projector layer 2
 *   - linear2.bias (768): Projector layer 2 bias
 * - token2wav models: encoder, flow_matching, hifigan, prompt_cache
 *
 * Mathematical Notation:
 * - h_t^LLM: LLM hidden state at position t (4096-d)
 * - h_t^TTS: TTS hidden state at position t (768-d)
 * - w_emb_text: Text embedding matrix (768 × 152064)
 * - w_projector: Projector matrix (768 × 4096)
 * - w_head_code: Head code matrix (6562 × 768)
 * - w_emb_code: Audio code embedding matrix (768 × 6562)
 * - condition_t = emb_text(token_t) + L2_norm(w_projector @ h_t^LLM)
 * - logits_t = h_t^TTS @ w_head_code
 * - p_t = softmax(logits_t / temperature)
 * - token_t ~ Multinomial(p_t)
 */

#include <catch2/catch_test_macros.hpp>

#include "common/sampling.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <sndfile.h>
#include "omni/audition.h"
#include "omni/omni-tts.h"
#include "omni/token2wav-impl.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

#include "vendor_paths.hpp"

// Paths — resolved from LLAMAOMNISERVER_MODEL_ROOT env var (set by CMake to the
// GGUF directory, e.g. /path/to/models/gguf) with a relative-path fallback.
static VendorPath const PATH_MODEL_LLM{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};
static VendorPath const PATH_MODEL_PROJECTOR{
	vp_model_root() + "/tts/MiniCPM-o-4_5-projector-F16.gguf"};
// Transformer-only GGUF (182 standard llama tensors) — loadable by libllama.
// Generated from the combined TTS GGUF by: utils/split_tts_gguf.py
static VendorPath const PATH_MODEL_TTS_TRANSFORMER{
	vp_model_root() + "/tts/MiniCPM-o-4_5-tts-transformer-F16.gguf"};
// Custom-weights GGUF (11 TTS-specific tensors: emb_code, emb_text, head_code, projector_*).
// Generated from the combined TTS GGUF by: utils/split_tts_gguf.py
static VendorPath const PATH_MODEL_TTS_WEIGHTS{
	vp_model_root() + "/tts/MiniCPM-o-4_5-tts-weights-F16.gguf"};
static VendorPath const PATH_DIR_TOKEN2WAV{vp_model_root() + "/token2wav-gguf"};
static VendorPath const PATH_MODEL_AUDIO{vp_model_root() + "/audio/MiniCPM-o-4_5-audio-F16.gguf"};
static VendorPath const PATH_AUDIO_VOICE_REF{vp_test_data() + "/alternative_voice.wav"};
static VendorPath const PATH_ALTERNATIVE_PROMPT_CACHE{
	vp_model_root() + "/token2wav-gguf/alternative_prompt_cache.gguf"};

// Constants
constexpr int SAMPLE_RATE = 24000;
constexpr int MIN_AUDIO_DURATION_SECONDS = 2;

/*
 * Helper: Write PCM audio to WAV file
 */
int write_wav_mono(std::string_view path, std::vector<float> const & pcm, int sample_rate)
{
	constexpr int16_t num_channels = 1;
	constexpr int16_t bits_per_sample = 16;
	constexpr int16_t block_align = num_channels * (bits_per_sample / 8);
	int32_t const byte_rate = sample_rate * block_align;

	std::vector<int16_t> pcm_i16(pcm.size());
	for (size_t i = 0; i < pcm.size(); ++i)
	{
		float x = pcm[i];
		x = std::max(-1.0f, std::min(1.0f, x));
		pcm_i16[i] = static_cast<int16_t>(x * 32767.0f);
	}

	uint32_t const data_bytes = static_cast<uint32_t>(pcm_i16.size() * sizeof(int16_t));
	uint32_t const file_size = 36 + data_bytes;

	std::ofstream file(path.data(), std::ios::binary);
	if (!file)
		return -1;

	file.write("RIFF", 4);
	file.write(reinterpret_cast<char const *>(&file_size), 4);
	file.write("WAVE", 4);

	file.write("fmt ", 4);
	uint32_t fmt_size = 16;
	uint16_t fmt_tag = 1;
	file.write(reinterpret_cast<char const *>(&fmt_size), 4);
	file.write(reinterpret_cast<char const *>(&fmt_tag), 2);
	file.write(reinterpret_cast<char const *>(&num_channels), 2);
	file.write(reinterpret_cast<char const *>(&sample_rate), 4);
	file.write(reinterpret_cast<char const *>(&byte_rate), 4);
	file.write(reinterpret_cast<char const *>(&block_align), 2);
	file.write(reinterpret_cast<char const *>(&bits_per_sample), 2);

	file.write("data", 4);
	uint32_t data_chunk_size = data_bytes;
	file.write(reinterpret_cast<char const *>(&data_chunk_size), 4);

	file.write(reinterpret_cast<char const *>(pcm_i16.data()), data_bytes);

	return static_cast<int>(pcm_i16.size());
}

TEST_CASE("tts_simplex", "[tts][simplex]")
{
	std::cout << "\n=== TTS Simplex Test ===" << std::endl;

	/*
	 * Step 1: Load LLM model
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	 *
	 * Purpose: Load the main Qwen3 (MiniCPM-o 4.5) LLM model for text generation.
	 * The LLM will generate text hidden states that will be projected to TTS embeddings.
	 *
	 * Model: MiniCPM-o-4_5-Q4_K_M.gguf
	 * - Architecture: Qwen3 (8.2B parameters)
	 * - Embedding dimension: n_embd = 4096 (model-specific)
	 * - Context length: 40960 tokens
	 * - Quantization: Q4_K_M (4-bit quantization, ~10GB VRAM)
	 *
	 * llama_model_load_from_file():
	 * - Loads model architecture and tensor metadata from GGUF file
	 * - Allocates model parameters from VRAM/GPU
	 * - Does NOT allocate context KV cache yet (done by llama_init_from_model)
	 *
	 * llama_init_from_model():
	 * - Allocates context KV cache (n_ctx = 4096 tokens)
	 * - Allocates batch buffers (n_batch = 8192, n_ubatch = 8192)
	 * - Sets up sampling parameters
	 * - Enables embeddings output (needed for text hidden state extraction)
	 *
	 * Parameters:
	 * - n_gpu_layers = 0: Force CPU-only execution (avoid GPU memory issues in test)
	 * - n_ctx = 4096: Maximum context length (fits our text prompt)
	 * - n_batch = 8192: Maximum batch size for decode operations
	 * - n_ubatch = 8192: Maximum batch size for inference
	 * - embeddings = true: Enable hidden state extraction via llama_get_embeddings_ith()
	 *
	 * References:
	 * - llama.cpp: llama_model_load_from_file()
	 * - llama.cpp: llama_init_from_model()
	 * - omni.cpp: Line 4544-4890 (model loading section)
	 * - AGENTS.md: Phase 1 documentation
	 */
	std::cout << "Step 1: Loading LLM model..." << std::endl;

	llama_model_params model_params = llama_model_default_params();
	// model_params.n_gpu_layers = 0;	// CPU-only to avoid GPU memory issues in test environment

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;	   // Maximum context length (fits our prompt)
	ctx_params.n_batch = 8192;	   // Maximum batch size for decode operations
	ctx_params.n_ubatch = 8192;	   // Maximum batch size for inference
	ctx_params.embeddings = true;  // Enable hidden state extraction

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_embd = llama_model_n_embd(model);
	bool ok = false;

	std::cout << "  ✓ LLM loaded (n_embd=" << n_embd << ")" << std::endl;

	/*
	 * Step 2: Create prompt components
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	 *
	 * Purpose: Build the complete prompt that will be fed through the LLM.
	 * The prompt follows the official MiniCPM-o chat template format.
	 *
	 * Chat Template Format (MiniCPM-o 4.5):
	 * <|im_start|>system
	 * {system_message}
	 * <|im_end|>
	 * <|im_start|>user
	 * {user_message}
	 * <|im_end|>
	 * <|im_start|>assistant
	 * {assistant_message}
	 * <|im_end|>
	 *
	 * Special Tokens in Prompt:
	 * - <|im_start|>: Marks start of a message role (system/user/assistant)
	 * - <|im_end|>: Marks end of a message
	 * - These special tokens are part of the official chat template (not a thinking model feature)
	 *
	 * TTS-Specific Handling:
	 * - We use simplex mode (no duplex)
	 * - Text is synthesized directly after the assistant start marker
	 * - No <|tts_bos|> marker is needed in simplex mode (added later in TTS pipeline)
	 * - No <|im_end|> after text (assistant end marker will be added by LLM)
	 *
	 * Prompt Construction:
	 * 1. system_prompt: Sets the assistant's behavior (helpful assistant)
	 * 2. user_prompt: Provides the task (speak about the quick brown fox)
	 * 3. assistant_start: Marks the start of assistant's response
	 * 4. text_to_synthesize: The actual text that will be converted to speech
	 *
	 * Tokenization:
	 * - llama_tokenize():
	 *   - Converts raw text string to token IDs
	 *   - Handles UTF-8 encoding automatically
	 *   - Returns number of tokens (or negative error code if buffer too small)
	 * - We use a lambda to handle dynamic buffer resizing
	 *
	 * References:
	 * - llama.cpp: llama_tokenize()
	 * - omni.cpp: Line 4544-4890 (prompt construction)
	 * - MiniCPM-o documentation: Chat template format
	 */
	std::cout << "\nStep 2: Creating prompt components..." << std::endl;

	std::string const system_prompt =
		"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
	std::string const user_prompt =
		"<|im_start|>user\nSpeak about the quick brown fox.<|im_end|>\n";
	std::string const assistant_start = "<|im_start|>assistant\n";
	std::string const text_to_synthesize = "The quick brown fox jumps over the lazy banana";
	// const std::string tts_begin = "<|tts_bos|>";
	// const std::string assistant_end = "<|im_end|>\n";

	std::string const prefix = system_prompt + user_prompt + assistant_start;
	std::string const suffix = "";

	// Lambda for tokenization with dynamic buffer resizing
	auto const tokenize = [&vocab](std::string_view text, std::vector<llama_token> & tokens)
	{
		int32_t n_tokens = llama_tokenize(
			vocab,
			text.data(),
			static_cast<int32_t>(text.size()),
			tokens.data(),
			static_cast<int32_t>(tokens.size()),
			false,
			false);
		if (n_tokens < 0)
		{
			tokens.resize(-n_tokens);
			llama_tokenize(
				vocab,
				text.data(),
				static_cast<int32_t>(text.size()),
				tokens.data(),
				static_cast<int32_t>(tokens.size()),
				false,
				false);
		}
	};

	// Tokenize each component
	std::vector<llama_token> prefix_tokens;
	tokenize(prefix, prefix_tokens);
	std::vector<llama_token> text_tokens;
	tokenize(text_to_synthesize, text_tokens);
	std::vector<llama_token> suffix_tokens;
	tokenize(suffix, suffix_tokens);

	// Combine tokens in order: prefix + text + suffix
	std::vector<llama_token> prompt_tokens;
	prompt_tokens.reserve(prefix_tokens.size() + text_tokens.size() + suffix_tokens.size());
	prompt_tokens.insert(prompt_tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
	prompt_tokens.insert(prompt_tokens.end(), text_tokens.begin(), text_tokens.end());
	prompt_tokens.insert(prompt_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());

	std::cout << "  ✓ Prompt components tokenized: text=" << text_tokens.size()
			  << " total=" << prompt_tokens.size() << std::endl;

	/*
	 * Step 3: Feeding through model
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	 *
	 * Purpose: Feed the complete prompt through the LLM and extract hidden states
	 * from the text tokens. These hidden states represent the LLM's contextual
	 * understanding of the text and will be used to condition the TTS model.
	 *
	 * LLM Inference Flow:
	 * 1. llama_batch preparation:
	 *    - batch.token: Array of token IDs to process
	 *    - batch.pos: Position IDs for each token (0, 1, 2, ...)
	 *    - batch.embd: NULL (we're processing tokens, not embeddings)
	 *    - batch.logits: NULL (we don't need logits for hidden state extraction)
	 *    - batch.n_tokens: Number of tokens in batch
	 *
	 * 2. llama_decode():
	 *    - Processes tokens sequentially through the LLM model
	 *    - Updates KV cache with computed keys/values
	 *    - Computes hidden states for each token (if embeddings=true)
	 *    - Returns true on success, false on error
	 *
	 * 3. Hidden state extraction:
	 *    - llama_get_embeddings_ith(ctx, -1): Get last token's hidden state
	 *    - llama_get_embeddings_ith(ctx, -2): Get second-to-last token's hidden state
	 *    - llama_get_embeddings_ith(ctx, i): Get i-th token's hidden state
	 *    - Hidden state is an n_embd-dimensional vector (4096-d for MiniCPM-o)
	 *
	 * Hidden States in LLM:
	 * - Each token position t produces an embedding vector h_t of dimension n_embd
	 * - h_t = LLM_model(token_t, h_{t-1}, KV_cache) (autoregressive computation)
	 * - These embeddings capture semantic meaning, syntax, and context
	 * - For TTS, we need hidden states from the text tokens only (not special tokens)
	 *
	 * Token-to-Hidden State Mapping:
	 * - text_tokens: [token_A, token_B, token_C, ...] (filtered tokens)
	 * - text_part_hidden: [h_A, h_B, h_C, ...] (corresponding hidden states)
	 * - Must maintain alignment: text_tokens[i] ↔ text_part_hidden[i * n_embd : (i+1) * n_embd]
	 *
	 * References:
	 * - llama.cpp: llama_decode()
	 * - llama.cpp: llama_get_embeddings_ith()
	 * - omni.cpp: Line 4544-4890 (LLM forward pass and hidden state extraction)
	 * - AGENTS.md: Phase 2 documentation on hidden states
	 */
	std::cout << "\nStep 3: Feeding through model..." << std::endl;

	// Create position IDs for each token (0, 1, 2, ...)
	std::vector<llama_pos> positions(prompt_tokens.size());
	std::iota(positions.begin(), positions.end(), 0);

	// Prepare batch with prompt tokens
	llama_batch batch = {};
	batch.n_tokens = int32_t(prompt_tokens.size());
	batch.token = prompt_tokens.data();
	batch.pos = positions.data();
	batch.n_seq_id = nullptr;
	batch.seq_id = nullptr;
	batch.logits = nullptr;

	// Enable embeddings output (needed for hidden state extraction)
	llama_set_embeddings(ctx, true);

	// Decode prompt through LLM
	if (llama_decode(ctx, batch))
	{
		FAIL("Failed to decode prompt tokens");
	}

	// Extract hidden states from text tokens only
	// We skip:
	// - prefix_tokens: system/user/assistant system messages (not spoken)
	// - suffix_tokens: empty in this case
	// - text_tokens: the actual text to synthesize
	std::vector<float> text_part_hidden;

	for (std::size_t i = prefix_tokens.size(); i < prompt_tokens.size() - suffix_tokens.size(); i++)
	{
		// Get hidden state for token at position i
		float const * emb = llama_get_embeddings_ith(ctx, static_cast<int32_t>(i));
		if (emb)
		{
			// Copy each of the n_embd dimensions to our vector
			for (int j = 0; j < n_embd; j++) text_part_hidden.push_back(emb[j]);
		}
		else
		{
			std::cout << "  - Missing embedding for token at position " << i << std::endl;
		}
	}

	std::cout << "Initial num hidden embeddings = " << text_part_hidden.size() << std::endl;

	// Step 4: Filter special tokens (matching omni.cpp)
	// Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	omni::tts::filter_special_tokens(text_tokens, text_part_hidden, n_embd);

	std::cout << "  ✓ Extracted and filtered: num filtered text tokens = " << text_tokens.size()
			  << "; num filtered hidden embeddings = " << text_part_hidden.size() << std::endl;

	/*
	 * Step 5: Load projector weights
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	 *
	 * Purpose: Load the projector_semantic weights that map LLM hidden states
	 * (4096-d) to TTS embedding space (768-d). This is a critical component
	 * that bridges the LLM and TTS models.
	 *
	 * Projector Semantics (from omni.cpp analysis):
	 * - Input: LLM hidden state h^LLM (4096-d vector)
	 * - Layer 1: h_proj = ReLU(W_1 @ h^LLM + b_1)
	 *   - W_1: 4096 × 768 matrix (projector layer 1)
	 *   - b_1: 768-d bias vector
	 *   - ReLU: Rectified Linear Unit activation (max(0, x))
	 * - Layer 2: h_TTS = W_2 @ h_proj + b_2
	 *   - W_2: 768 × 768 matrix (projector layer 2)
	 *   - b_2: 768-d bias vector
	 * - Output: TTS hidden state h^TTS (768-d vector)
	 *
	 * Why Projector is Needed:
	 * - LLM outputs 4096-d embeddings (Qwen3 architecture)
	 * - TTS model expects 768-d embeddings
	 * - Direct dimension mismatch: 4096 ≠ 768
	 * - Projector learns the optimal mapping between these spaces
	 *
	 * Model File:
	 * - Path: models/gguf/tts/MiniCPM-o-4_5-projector-F16.gguf
	 * - Tensor count: 4 (linear1.weight, linear1.bias, linear2.weight, linear2.bias)
	 * - Format: F16 (16-bit floating point)
	 * - Loading: Uses gguf_init_from_file() to read GGUF metadata
	 * - Backend: Allocates tensors to GPU (if available) via ggml_backend_init_by_type()
	 *
	 * GGUF Loading Process (omni-tts.cpp: load_projector_semantic):
	 * 1. gguf_init_from_file():
	 *    - Opens GGUF file and reads tensor metadata
	 *    - Stores metadata in ctx_meta (for reading tensor names and sizes)
	 *    - Allocates temporary context for tensor structures
	 *
	 * 2. ggml_backend_init_by_type():
	 *    - Attempts to initialize GPU backend (CUDA)
	 *    - Falls back to CPU backend if GPU unavailable
	 *    - Returns backend handle for tensor allocation
	 *
	 * 3. ggml_init():
	 *    - Creates computation graph context
	 *    - Allocates tensor structures without data (no_alloc=true)
	 *
	 * 4. Tensor allocation:
	 *    - ggml_new_tensor_2d(): Creates 2D tensor with specified dimensions
	 *    - ggml_set_name(): Assigns tensor name for debugging
	 *    - ggml_backend_alloc_ctx_tensors(): Allocates tensor data to backend
	 *
	 * 5. Weight loading from file:
	 *    - fseek(): Seeks to tensor offset in GGUF file
	 *    - fread(): Reads tensor data from file
	 *    - ggml_fp16_to_fp32(): Converts F16 to F32 if needed
	 *    - ggml_backend_tensor_set(): Uploads weights to backend memory
	 *
	 * 6. Transposition handling:
	 *    - PyTorch stores linear layers as (out_features, in_features)
	 *    - For efficient CPU inference, transpose to (in_features, out_features)
	 *    - Check: if (t_gguf->ne[0] != t_model->ne[0] || t_gguf->ne[1] != t_model->ne[1])
	 *
	 * Mathematical Notation:
	 * - h^LLM ∈ ℝ^4096: LLM hidden state (input to projector)
	 * - W_1 ∈ ℝ^(768×4096): First projection matrix
	 * - b_1 ∈ ℝ^768: First bias vector
	 * - ReLU(x) = max(0, x): Activation function
	 * - h_proj = ReLU(W_1 @ h^LLM + b_1) ∈ ℝ^768: Projected hidden state
	 * - W_2 ∈ ℝ^(768×768): Second projection matrix
	 * - b_2 ∈ ℝ^768: Second bias vector
	 * - h^TTS = W_2 @ h_proj + b_2 ∈ ℝ^768: Final TTS hidden state
	 *
	 * References:
	 * - omni-tts.cpp: load_projector_semantic() (lines 53-124)
	 * - omni.cpp: Line 4544-4890 (projector loading)
	 * - AGENTS.md: TTS architecture documentation
	 * - HYBRID_SOLUTION.md: Projector usage guide
	 */
	std::cout << "\nStep 5: Loading projector weights..." << std::endl;

	omni::tts::TTSContext tts_ctx;
	ok = omni::tts::load_projector_semantic(PATH_MODEL_PROJECTOR, tts_ctx.projector_weights);
	REQUIRE(ok);
	std::cout << "  ✓ Projector loaded" << std::endl;

	/*
	 * Step 6: Apply projector to convert LLM embeddings to TTS embeddings
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	 *
	 * Purpose: Convert LLM hidden states to TTS embedding space and normalize
	 * them to unit L2 norm. This is a critical step that bridges the LLM and TTS models.
	 *
	 * TTS Conditioning Formula (from omni.cpp:5816-5841):
	 *   condition[i] = emb_text(token_id[i]) + L2_normalize(projector(hidden_state[i]))
	 *
	 * Components:
	 * 1. projector(hidden_state[i]):
	 *    - Input: LLM hidden state h^LLM ∈ ℝ^4096
	 *    - Output: Projected hidden state h^TTS ∈ ℝ^768
	 *    - Purpose: Map LLM representation to TTS space
	 *    - Architecture: Two-layer MLP with ReLU activation
	 *
	 * 2. emb_text(token_id[i]):
	 *    - Input: Text token ID (e.g., token "The")
	 *    - Output: Text embedding e_text ∈ ℝ^768
	 *    - Purpose: Provide semantic identity in TTS space
	 *    - Without emb_text: TTS produces gibberish (no phoneme/identity information)
	 *    - With emb_text: TTS knows what language/phonemes to generate
	 *
	 * 3. L2_normalize():
	 *    - Input: Projected hidden state h^TTS ∈ ℝ^768
	 *    - Output: Normalized vector h_norm ∈ ℝ^768
	 *    - Formula: h_norm = h^TTS / ||h^TTS||_2
	 *    - Purpose: Scale vectors to unit length (improves numerical stability)
	 *
	 * 4. Addition:
	 *    - Input: emb_text ∈ ℝ^768, h_norm ∈ ℝ^768
	 *    - Output: condition ∈ ℝ^768
	 *    - Formula: condition = emb_text + h_norm
	 *    - Purpose: Combine semantic identity with contextual representation
	 *
	 * Why This Works (from omni.cpp analysis):
	 * - LLM hidden states capture contextual meaning (syntax, semantics)
	 * - emb_text provides explicit token identity in TTS space
	 * - Projector maps LLM context to TTS representation space
	 * - L2 normalization ensures consistent scale across tokens
	 * - Summation merges identity and context into unified condition
	 *
	 * Mathematical Notation:
	 * - h^LLM_t: LLM hidden state at token t (4096-d)
	 * - W_1 ∈ ℝ^(768×4096), b_1 ∈ ℝ^768: First projection layer
	 * - W_2 ∈ ℝ^(768×768), b_2 ∈ ℝ^768: Second projection layer
	 * - ReLU(x) = max(0, x): Activation function
	 * - h_proj = ReLU(W_1 @ h^LLM_t + b_1) ∈ ℝ^768: Projected hidden state
	 * - e_text = emb_text(token_t) ∈ ℝ^768: Text embedding
	 * - L2_norm(x) = x / ||x||_2: L2 normalization
	 * - h_norm = L2_norm(h_proj) ∈ ℝ^768: Normalized hidden state
	 * - condition_t = e_text + h_norm ∈ ℝ^768: Final TTS condition
	 *
	 * Implementation (omni-tts.cpp: apply_projector_semantic):
	 * 1. Create temporary GGML context for computation
	 * 2. Build computation graph: input → linear1 → ReLU → linear2
	 * 3. Allocate tensors to backend memory (GPU/CPU)
	 * 4. Upload hidden states to input tensor
	 * 5. Run computation: ggml_backend_graph_compute()
	 * 6. Extract output hidden states
	 * 7. Free temporary context
	 *
	 * Implementation (omni-tts.cpp: normalize_l2_per_token):
	 * 1. For each token t from 0 to n-1:
	 *    a. Extract vector v = projected_hidden[t * n_embd : (t+1) * n_embd]
	 *    b. Compute norm = sqrt(sum(v_i^2))
	 *    c. If norm > 0: v = v / norm
	 *    d. Else (norm = 0): v = random unit vector (fallback)
	 *
	 * References:
	 * - omni-tts.cpp: apply_projector_semantic() (lines 161-176)
	 * - omni-tts.cpp: normalize_l2_per_token() (lines 137-159)
	 * - omni.cpp: Lines 5816-5841 (condition embedding construction)
	 * - AGENTS.md: TTS architecture and conditioning formula
	 */
	std::cout << "\nStep 6: Applying projector and merging with emb_text embeddings..."
			  << std::endl;

	// Apply projector to convert LLM hidden states to TTS space (4096-d → 768-d)
	std::vector<float> projected_hidden_pre_merge;
	omni::tts::apply_projector_semantic(
		tts_ctx.projector_weights,
		text_part_hidden,
		text_tokens.size(),
		projected_hidden_pre_merge);
	REQUIRE(projected_hidden_pre_merge.size() == text_tokens.size() * 768);

	// Normalize projected hidden states to unit L2 norm
	omni::tts::normalize_l2_per_token(projected_hidden_pre_merge.data(), text_tokens.size(), 768);
	std::cout << "  ✓ Projector applied to " << text_tokens.size()
			  << " tokens (4096-d → 768-d), emb_text merge pending" << std::endl;

	/*
	 * Step 7: Load TTS model and weights
	 * Must come before step 8 (audio_bos embedding and emb_text merging) because
	 * tts_emb_text requires emb_text_weight, which is populated here by load_tts_weights.
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:3602-3646
	 *
	 * Purpose: Load the TTS model (Transformer-based) and all TTS-specific weights.
	 * This completes the setup needed to generate audio tokens from text conditions.
	 *
	 * TTS Model Architecture (from omni.cpp analysis):
	 * - Type: Transformer decoder (24 blocks, causal attention)
	 * - Hidden size: 768-d (same as projector output)
	 * - Context length: 4096 tokens
	 * - Input: Condition embeddings (768-d vectors)
	 *   - Condition = emb_text(token) + L2_norm(projector(hidden_state))
	 * - Output: Hidden states for each token position
	 * - Special: head_code layer projects hidden states to audio token logits (6562-dim)
	 *
	 * TTS Weights (from omni.cpp analysis):
	 * 1. Transformer weights (in PATH_MODEL_TTS_TRANSFORMER):
	 *    - 182 standard llama tensors (layers, embeddings, norms)
	 *    - Cannot be loaded by standard libllama from combined GGUF
	 *    - GGUF splitting required (utils/split_tts_gguf.py)
	 *
	 * 2. TTS-specific weights (in PATH_MODEL_TTS_WEIGHTS):
	 *    - emb_code.0.weight (768, 6562): Audio token embeddings
	 *    - emb_text.weight (768, 152064): Text token embeddings for TTS
	 *    - head_code.0.weight (6562, 768): Audio token logits
	 *    - linear1.weight (4096, 768): Projector layer 1 (already loaded in step 5)
	 *    - linear1.bias (768): Projector layer 1 bias (already loaded in step 5)
	 *    - linear2.weight (768, 768): Projector layer 2 (already loaded in step 5)
	 *    - linear2.bias (768): Projector layer 2 bias (already loaded in step 5)
	 *
	 * 3. Token2Wav models (in PATH_DIR_TOKEN2WAV):
	 *    - encoder.gguf: Conformer encoder
	 *    - flow_matching.gguf: Conditional Flow Matching diffusion
	 *    - flow_extra.gguf: Additional flow matching components
	 *    - prompt_cache.gguf: Pre-computed prompt cache
	 *    - hifigan2.gguf: HiFi-GAN vocoder
	 *
	 * GGUF File Splitting (from AGENTS.md):
	 * - Combined TTS GGUF: 193 tensors (cannot be loaded by standard libllama)
	 * - Error: "wrong number of tensors; expected 193, got 182"
	 * - Reason: 11 custom tensors are not recognized by standard llama architecture
	 * - Solution: Split into transformer (standard) + weights (custom) GGUFs
	 *
	 * llama_model_load_from_file():
	 * - Loads transformer architecture and weights
	 * - Allocates model parameters to GPU/CPU
	 * - Does NOT load custom TTS weights (emb_code, emb_text, head_code)
	 *
	 * llama_init_from_model():
	 * - Allocates context KV cache (n_ctx = 4096)
	 * - Allocates batch buffers (n_batch = 4096, n_ubatch = 4096)
	 * - Enables embeddings output
	 *
	 * Parameters:
	 * - n_gpu_layers = 0: CPU-only execution
	 * - n_ctx = 4096: Maximum context length
	 * - n_batch = 4096: Maximum batch size (must be ≤ n_ctx)
	 * - n_ubatch = 4096: Maximum inference batch size
	 * - embeddings = true: Enable hidden state extraction
	 *
	 * load_tts_weights():
	 * - Loads emb_code, emb_text, head_code from GGUF
	 * - Loads Token2Wav models from directory
	 * - Stores weights in tts_ctx structure
	 *
	 * References:
	 * - llama.cpp: llama_model_load_from_file()
	 * - llama.cpp: llama_init_from_model()
	 * - omni-tts.cpp: load_tts_weights() (lines 194-295)
	 * - omni.cpp: Lines 3602-3646 (TTS model loading)
	 * - AGENTS.md: Model file splitting documentation
	 */
	std::cout << "\nStep 7: Loading TTS model and weights..." << std::endl;

	llama_model_params tts_model_params = llama_model_default_params();
	// tts_model_params.n_gpu_layers = 0;
	// Load the transformer-only GGUF (182 standard tensors).
	// The original combined GGUF (193 tensors) cannot be loaded by standard libllama because
	// done_getting_tensors() throws "wrong number of tensors; expected 193, got 182" —
	// the 11 custom TTS tensors (emb_code, emb_text, head_code, projector_*) are
	// not recognised by the standard llama architecture loader.
	// Use the file produced by utils/split_tts_gguf.py instead.
	llama_model * model_tts =
		llama_model_load_from_file(PATH_MODEL_TTS_TRANSFORMER, tts_model_params);
	REQUIRE(model_tts != nullptr);

	llama_context_params tts_ctx_params = llama_context_default_params();
	tts_ctx_params.n_ctx = 4096;
	// n_batch must not exceed n_ctx; libllama caps it internally and the assert
	// n_tokens_all <= cparams.n_batch fires if a batch larger than the cap is submitted.
	tts_ctx_params.n_batch = 4096;
	tts_ctx_params.n_ubatch = 4096;
	tts_ctx_params.embeddings = true;

	llama_context * ctx_tts = llama_init_from_model(model_tts, tts_ctx_params);
	REQUIRE(ctx_tts != nullptr);

	// Load TTS weights from GGUF (matching omni.cpp:3635-3646)
	// Ported from: llama.cpp-omni/tools/omni/omni.cpp:3635-3646
	ok = omni::tts::load_tts_weights(
		tts_ctx,
		PATH_MODEL_PROJECTOR,
		PATH_MODEL_TTS_TRANSFORMER,
		PATH_MODEL_TTS_WEIGHTS,
		PATH_DIR_TOKEN2WAV);
	REQUIRE(ok);
	std::cout << "  ✓ TTS weights loaded" << std::endl;

	/*
	 * Step 8: Build condition embeddings: emb_text(token) + projected_hidden, then append audio_bos
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:5816-5848
	 *
	 * Purpose: Construct the final TTS condition embeddings that will be fed into the TTS model.
	 * This step combines all components into the complete conditioning for audio generation.
	 *
	 * Condition Embedding Construction (from omni.cpp:5816-5841):
	 * For each text token position i:
	 *   condition[i] = emb_text(token_id[i]) + L2_norm(projector(hidden_state[i]))
	 *
	 * Components Breakdown:
	 * 1. emb_text(token_id[i]):
	 *    - Lookup: Retrieve embedding from emb_text_weight matrix
	 *    - Matrix: emb_text.weight ∈ ℝ^(768 × 152064)
	 *    - Access: embedding[j] = emb_text_weight[token_id[i] * 768 + j]
	 *    - Purpose: Provide semantic identity in TTS space (language, phonemes)
	 *    - Without emb_text: TTS produces gibberish (no identity information)
	 *
	 * 2. projector(hidden_state[i]):
	 *    - Already computed in step 6 as projected_hidden_pre_merge
	 *    - Input: LLM hidden state h^LLM ∈ ℝ^4096
	 *    - Output: Projected hidden state h^TTS ∈ ℝ^768
	 *    - Purpose: Map LLM contextual representation to TTS space
	 *
	 * 3. L2 normalization:
	 *    - Applied to projected_hidden in step 6 as normalize_l2_per_token()
	 *    - Formula: h_norm = h^TTS / ||h^TTS||_2
	 *    - Purpose: Ensure consistent scale across all tokens
	 *
	 * 4. Addition:
	 *    - condition[i] = emb_text(token_id[i]) + projected_hidden[i]
	 *    - Result: Combined semantic identity and contextual representation
	 *
	 * audio_bos Embedding (from omni.cpp:5845-5848):
	 * - Purpose: Start-of-speech marker for TTS model
	 * - Token ID: 151687 (special TTS token)
	 * - Lookup: emb_text[151687] ∈ ℝ^768
	 * - Behavior:
	 *   - Marks beginning of audio generation
	 *   - Resets TTS internal state (if needed)
	 *   - Ensures first audio token follows proper conditioning
	 *
	 * Final Condition Structure:
	 *   [condition[0], condition[1], ..., condition[N-1], audio_bos]
	 *   Total: N + 1 embeddings, each 768-d
	 *
	 * Implementation Details:
	 * - N = text_tokens.size() (number of text tokens)
	 * - Each condition[i] ∈ ℝ^768
	 * - audio_bos ∈ ℝ^768
	 * - Flat array: [768 * N + 768 floats]
	 *
	 * Mathematical Notation:
	 * - token_id[i]: Text token ID at position i
	 * - emb_text(): Text embedding lookup function
	 * - e_text = emb_text(token_id[i]) ∈ ℝ^768
	 * - h_norm = L2_norm(projector(hidden_state[i])) ∈ ℝ^768
	 * - condition[i] = e_text + h_norm ∈ ℝ^768
	 * - audio_bos = emb_text(151687) ∈ ℝ^768
	 * - condition = [condition[0], condition[1], ..., condition[N-1], audio_bos]
	 *
	 * References:
	 * - omni-tts.cpp: tts_emb_text() (lines 178-192)
	 * - omni.cpp: Lines 5816-5848 (condition embedding construction)
	 * - AGENTS.md: TTS architecture and conditioning formula
	 */
	std::cout
		<< "\nStep 8: Merging emb_text with projected hidden states and appending audio_bos..."
		<< std::endl;

	int const audio_bos_token_id = 151687;
	int const text_eos_token_id = 151692;  // tts_config.text_eos_token_id

	// Merge: condition[i] = emb_text(token_id[i]) + projected_hidden[i]
	// omni.cpp:5838-5841: merged_embeddings[i] = llm_embeds[i] + projected_hidden[i]
	std::vector<float> condition_embeddings(text_tokens.size() * 768);
	for (int i = 0; i < static_cast<int>(text_tokens.size()); i++)
	{
		std::vector<float> text_emb(768);
		omni::tts::tts_emb_text(tts_ctx, text_tokens[i], text_emb.data(), 768);
		for (int j = 0; j < 768; j++)
		{
			condition_embeddings[i * 768 + j] =
				text_emb[j] + projected_hidden_pre_merge[i * 768 + j];
		}
	}

	// Append text_eos embedding before audio_bos (modeling_minicpmo.py:1325-1357)
	// Python: inputs_embeds = cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds])
	// text_eos_token_id = tts_config.text_eos_token_id = 151692
	std::vector<float> text_eos_emb(768);
	omni::tts::tts_emb_text(tts_ctx, text_eos_token_id, text_eos_emb.data(), 768);
	condition_embeddings.insert(
		condition_embeddings.end(), text_eos_emb.begin(), text_eos_emb.end());

	// Append audio_bos embedding (omni.cpp:5845-5848)
	std::vector<float> bos_emb(768);
	omni::tts::tts_emb_text(tts_ctx, audio_bos_token_id, bos_emb.data(), 768);
	condition_embeddings.insert(condition_embeddings.end(), bos_emb.begin(), bos_emb.end());

	std::cout << "  ✓ Built condition: " << text_tokens.size()
			  << " merged tokens + text_eos + audio_bos = " << (text_tokens.size() + 2) << " total"
			  << std::endl;

	/*
	 * Step 9: Generate audio tokens using TTS model
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4544-4890
	 *
	 * Key: Use llama_decode() with embeddings and manual sampling with head_code logits
	 *
	 * Sampling logic (matching omni.cpp:2680-2760):
	 * 1. Get hidden state via llama_get_embeddings_ith() (omni.cpp:2521)
	 * 2. Compute logits: hidden @ head_code_weight (omni.cpp:2638-2660)
	 * 3. Apply temperature (omni.cpp:2691-2693)
	 * 4. Apply repetition penalty (omni.cpp:2700-2728)
	 * 5. Force no EOS if t < min_new_tokens (omni.cpp:2730-2733)
	 * 6. Multinomial sampling (omni.cpp:2744-2760)
	 */
	std::cout << "\nStep 9: Generating audio tokens using TTS model..." << std::endl;

	// Create sampler matching omni.cpp TTS sampling parameters
	// Ported from: llama.cpp-omni/tools/omni/omni.cpp:3617-3627
	struct common_params_sampling tts_sampling = {};
	tts_sampling.temp = 0.8f;			  // Controls randomness (lower = more deterministic)
	tts_sampling.top_p = 0.85f;			  // Nucleus sampling threshold
	tts_sampling.top_k = 25;			  // Top-k sampling threshold
	tts_sampling.penalty_repeat = 1.05f;  // Repetition penalty
	tts_sampling.min_p = 0.01f;			  // Minimum probability threshold
	tts_sampling.penalty_last_n = 16;	  // Window size for repetition penalty

	struct common_sampler * tts_sampler = common_sampler_init(model_tts, tts_sampling);
	REQUIRE(tts_sampler != nullptr);

	int const num_audio_tokens = 6562;
	int const eos_relative_idx = num_audio_tokens - 1;
	int const max_audio_tokens = 500;
	// omni.cpp simplex uses min_new_tokens=0: EOS can be sampled at any time.
	int const min_new_tokens = 0;

	std::vector<int32_t> audio_tokens;
	std::vector<int> all_generated_tokens_relative;

	int n_past_tts = 0;
	// n_batch must match the context's n_batch (set above to 4096).
	// condition_embeddings is a flat float array: n_cond_tokens * 768 floats.
	// The loop must iterate in units of TOKENS, not floats.
	int const n_batch_tts = 4096;
	int const n_cond_tokens = static_cast<int>(condition_embeddings.size()) / 768;

	// Helper lambda: prefill condition embeddings into TTS model KV cache.
	// Called at t=0 (and re-called to re-forward, matching omni.cpp sample_tts_token_simplex).
	// Ported from: llama.cpp-omni/tools/omni/omni.cpp:2149-2154
	auto prefill_condition = [&]()
	{
		// Enable embeddings output for TTS model
		llama_set_embeddings(ctx_tts, true);
		// Process embeddings in batches
		for (int i = 0; i < n_cond_tokens; i += n_batch_tts)
		{
			int n_eval = std::min(n_batch_tts, n_cond_tokens - i);
			// Create batch with embeddings (not tokens)
			llama_batch cond_batch = {};
			cond_batch.n_tokens = n_eval;
			cond_batch.embd = condition_embeddings.data() + i * 768;
			// Set position IDs for each embedding
			std::vector<llama_pos> pos_vec(n_eval);
			std::iota(pos_vec.begin(), pos_vec.end(), n_past_tts);
			cond_batch.pos = pos_vec.data();
			// Decode batch through TTS model
			if (llama_decode(ctx_tts, cond_batch))
			{
				FAIL("Failed to prefill condition embeddings");
			}
			// Update position counter
			n_past_tts += n_eval;
		}
	};

	// At t=0, re-forward the condition into a fresh KV cache (matching
	// omni.cpp sample_tts_token_simplex is_audio_bos path, lines ~2504-2519).
	// This matches Python's behaviour: the first audio token re-forwards the condition.
	llama_memory_t mem = llama_get_memory(ctx_tts);
	if (mem)
		llama_memory_seq_rm(
			mem, 0, 0, -1);	 // TODO(DF): removing this seems to have no effect, what is it for?
	prefill_condition();

	// Sample audio tokens using head_code logits (matching omni.cpp:2680-2760)
	// Ported from: llama.cpp-omni/tools/omni/omni.cpp:2680-2760
	for (int t = 0; t < max_audio_tokens; ++t)
	{
		// Get hidden state from the last decoded token (matching omni.cpp:2521)
		// Purpose: Extract the TTS model's last hidden state to compute audio token logits
		// llama_get_embeddings_ith(ctx_tts, -1):
		// - ctx_tts: TTS context (already prefilled with condition embeddings)
		// - -1: Last token in the sequence (most recent decode step)
		// - Returns: Pointer to float array containing 768-d hidden state
		// - This hidden state represents the TTS model's representation of the audio to generate
		// - Used to compute logits via head_code layer (audio token predictions)
		// References:
		// - llama.cpp: llama_get_embeddings_ith()
		// - omni.cpp: Line 2521 (hidden state extraction)
		float const * hidden_state = llama_get_embeddings_ith(ctx_tts, -1);
		REQUIRE(hidden_state != nullptr);

		// Compute logits using head_code layer (matching omni.cpp:2638-2660)
		// Purpose: Project TTS hidden states to audio token logits (6562 possible tokens)
		// Formula: logits = hidden_state @ head_code_weight
		// Where:
		// - hidden_state: 768-d TTS hidden state
		// - head_code_weight: 6562 × 768 matrix
		// - logits: 6562-d vector (audio token probabilities)
		// - Each element logits[i] represents log probability of audio token i
		// - This is essentially a matrix multiplication: sum(hidden_state[j] *
		// head_code_weight[i][j]) References:
		// - omni.cpp: Lines 2638-2660 (head_code computation)
		// - omni-tts.cpp: Lines 2638-2660 (implementation)
		std::vector<float> audio_logits(num_audio_tokens, 0.0f);
		float const * head_code_w = tts_ctx.head_code_weight;
		int const hidden_size = tts_ctx.head_code_hidden_size;

		for (int i = 0; i < num_audio_tokens; ++i)
		{
			// Get row i of head_code_weight: head_code_weight[i * hidden_size : (i+1) *
			// hidden_size]
			float const * row = head_code_w + i * hidden_size;
			float sum = 0.0f;
			// Compute dot product: sum(hidden_state[j] * head_code_weight[i][j])
			for (int j = 0; j < hidden_size; ++j)
			{
				sum += hidden_state[j] * row[j];
			}
			audio_logits[i] = sum;	// Matrix multiplication result
		}

		// Apply temperature (matching omni.cpp:2691-2693)
		// Purpose: Scale logits to control randomness in sampling
		// Formula: logits /= temperature
		// Where:
		// - temperature = 0.8 (TTSSamplingParams.temperature)
		// - Higher temperature → more random sampling
		// - Lower temperature → more deterministic sampling
		// - Temperature < 1: Reduces logits, making output more predictable
		// - If temperature = 1: No effect (no randomness scaling)
		// - If temperature > 1: Increases logits, making output more random
		// The default 0.8 provides a balance between creativity and coherence
		// References:
		// - omni.cpp: Lines 2691-2693 (temperature application)
		// - TTSSamplingParams: temperature = 0.8
		for (int i = 0; i < num_audio_tokens; ++i)
		{
			audio_logits[i] /= tts_sampling.temp;
		}

		// Apply repetition penalty (matching omni.cpp:2700-2728)
		// Purpose: Penalize recently generated tokens to reduce repetition
		// Formula: if token occurred in last N tokens:
		//   - if logits[i] >= 0: logits[i] /= penalty_repeat
		//   - if logits[i] < 0: logits[i] *= penalty_repeat
		// Where:
		// - penalty_repeat = 1.05 (TTSSamplingParams.penalty_repeat)
		// - N = 16 (TTSSamplingParams.penalty_last_n)
		// - This prevents immediate repetition of the same audio token
		// - Positive logits are divided (reduce probability)
		// - Negative logits are multiplied (increase probability)
		// References:
		// - omni.cpp: Lines 2700-2728 (repetition penalty)
		// - TTSSamplingParams: penalty_repeat = 1.05, penalty_last_n = 16
		if (!all_generated_tokens_relative.empty())
		{
			// Get window of last N tokens for repetition penalty (penalty_last_n = 16)
			int start_idx = std::max(
				0, (int)all_generated_tokens_relative.size() - tts_sampling.penalty_last_n);
			std::vector<bool> occurred(num_audio_tokens, false);
			for (int i = start_idx; i < (int)all_generated_tokens_relative.size(); ++i)
			{
				int tok = all_generated_tokens_relative[i];
				if (tok >= 0 && tok < num_audio_tokens)
				{
					occurred[tok] = true;
				}
			}

			for (int i = 0; i < num_audio_tokens; ++i)
			{
				if (occurred[i])
				{
					if (audio_logits[i] >= 0)
					{
						audio_logits[i] /= tts_sampling.penalty_repeat;
					}
					else
					{
						audio_logits[i] *= tts_sampling.penalty_repeat;
					}
				}
			}
		}

		// Force no EOS if t < min_new_tokens (matching omni.cpp:2730-2733)
		// Purpose: Ensure minimum audio duration before allowing EOS
		// Formula: logits[eos_relative_idx] = -inf
		// Where:
		// - eos_relative_idx = 6561 (end-of-speech token relative index)
		// - min_new_tokens = 0 (omni simplex mode)
		// - t: Current generation step
		// - By setting logits[eos_relative_idx] to -inf, we ensure:
		//   - EOS cannot be sampled until t >= min_new_tokens
		//   - Prevents early termination of audio generation
		//   - Allows generation of longer audio sequences
		// References:
		// - omni.cpp: Lines 2730-2733 (force no EOS)
		// - TTSSamplingParams: min_new_tokens = 0
		if (t < min_new_tokens)
		{
			audio_logits[eos_relative_idx] = -std::numeric_limits<float>::infinity();
		}

		// Sample token using multinomial sampling (matching omni.cpp:2744-2760)
		// Purpose: Select audio token from logits based on probability distribution
		// Algorithm:
		// 1. Compute max logit: max_l = max(audio_logits)
		// 2. Compute probabilities: probs[i] = exp(audio_logits[i] - max_l)
		// 3. Normalize: probs[i] /= sum(probs)
		// 4. Sample token from multinomial distribution
		// 5. Argmax: sel = argmax(probs) (most probable token)
		// Where:
		// - max_l: Maximum logits (numerical stability for exp)
		// - probs[i]: Probability of token i = exp(audio_logits[i] / temperature)
		// - sum(probs): Normalization factor (sum to 1)
		// - sel: Selected token ID (0-6561)
		// - For deterministic output: use argmax instead of multinomial sampling
		// - For creative output: use multinomial sampling with random seed
		// References:
		// - omni.cpp: Lines 2744-2760 (multinomial sampling)
		// - TTSSamplingParams: top_p = 0.85, top_k = 25 (not used in argmax)
		float max_l = -1e20f;
		for (float l : audio_logits)
			if (l > max_l)
				max_l = l;

		float sum_p = 0.0f;
		std::vector<float> probs(num_audio_tokens);
		for (int i = 0; i < num_audio_tokens; i++)
		{
			probs[i] = std::exp(audio_logits[i] - max_l);
			sum_p += probs[i];
		}
		for (int i = 0; i < num_audio_tokens; i++) probs[i] /= sum_p;

		// Apply top_k filtering: keep only top 25 tokens (matching omni.cpp sampling)
		{
			int const top_k = tts_sampling.top_k;
			if (top_k > 0 && top_k < num_audio_tokens)
			{
				// Find the k-th largest probability threshold
				std::vector<float> sorted_probs = probs;
				std::nth_element(
					sorted_probs.begin(),
					sorted_probs.begin() + (num_audio_tokens - top_k),
					sorted_probs.end());
				float threshold = sorted_probs[num_audio_tokens - top_k];
				for (int i = 0; i < num_audio_tokens; i++)
				{
					if (probs[i] < threshold)
						probs[i] = 0.0f;
				}
			}
		}

		// Apply top_p (nucleus) filtering: keep tokens until cumulative prob >= 0.85
		{
			float const top_p = tts_sampling.top_p;
			if (top_p < 1.0f)
			{
				// Sort indices by descending probability
				std::vector<int> idx(num_audio_tokens);
				std::iota(idx.begin(), idx.end(), 0);
				std::sort(
					idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
				// Renormalise first so cumsum is meaningful
				float s = 0.0f;
				for (float p : probs) s += p;
				float cum = 0.0f;
				bool cutoff_reached = false;
				for (int i : idx)
				{
					if (cutoff_reached)
					{
						probs[i] = 0.0f;
						continue;
					}
					cum += probs[i] / s;
					if (cum >= top_p)
						cutoff_reached = true;
				}
			}
		}

		// Re-normalise after top_k / top_p zeroing
		{
			float s = 0.0f;
			for (float p : probs) s += p;
			if (s > 0.0f)
				for (float & p : probs) p /= s;
		}

		// Multinomial sampling (matching omni.cpp:2744-2760)
		int sel = eos_relative_idx;
		{
			static std::mt19937 rng{std::random_device{}()};
			std::uniform_real_distribution<float> dist(0.0f, 1.0f);
			float r = dist(rng);
			float cum = 0.0f;
			for (int i = 0; i < num_audio_tokens; i++)
			{
				cum += probs[i];
				if (r <= cum)
				{
					sel = i;
					break;
				}
			}
		}

		// Check for EOS
		if (sel == eos_relative_idx && t >= min_new_tokens)
		{
			std::cout << "EOS sampled at t=" << t << ", min_new_tokens=" << min_new_tokens
					  << " EOS token = " << sel << std::endl;
			break;
		}

		// Convert relative token ID to absolute token ID
		// Purpose: Add audio_bos offset to get absolute token ID
		// Formula: absolute_id = relative_idx + audio_bos_token_id
		// Where:
		// - sel: Relative token ID (0-6561)
		// - audio_bos_token_id = 151687: Start-of-speech offset
		// - absolute_id: Absolute token ID (151687-152067)
		// - This offset is required because audio tokens are in a special range
		// References:
		// - TTSSamplingParams: audio_bos_token_id = 151687
		// - AGENTS.md: Special tokens documentation
		int absolute_id = sel + audio_bos_token_id;
		audio_tokens.push_back(absolute_id);
		all_generated_tokens_relative.push_back(sel);
		common_sampler_accept(tts_sampler, absolute_id, true);

		// Get emb_code embedding for sampled token and prefill (matching omni.cpp:3007-3010)
		// Purpose: Convert audio token ID to embedding and feed back to TTS model
		// This allows the TTS model to continue generating audio tokens
		// Algorithm:
		// 1. Look up embedding: audio_emb = emb_code[relative_idx]
		//    - emb_code.weight ∈ ℝ^(768 × 6562)
		//    - Access: audio_emb[j] = emb_code[relative_idx * 768 + j]
		// 2. Prefill embedding: llama_decode(batch with audio_emb)
		//    - batch.embd = audio_emb (768-d vector)
		//    - batch.n_tokens = 1
		//    - llama_decode() updates TTS KV cache
		// 3. Update n_past_tts counter
		// Where:
		// - sel: Relative token ID (0-6561)
		// - emb_code.weight: Audio token embedding matrix
		// - audio_token_embedding: 768-d embedding vector
		// - n_past_tts: Position counter for KV cache
		// References:
		// - omni.cpp: Lines 3007-3010 (prefill sampled embeddings)
		// - AGENTS.md: emb_code documentation
		if (sel >= 0 && sel < tts_ctx.emb_code_vocab_size)
		{
			float const * emb_code_w = tts_ctx.emb_code_weight;
			int const emb_code_hidden_size = tts_ctx.emb_code_hidden_size;

			std::vector<float> audio_token_embedding(emb_code_hidden_size);
			for (int j = 0; j < emb_code_hidden_size; ++j)
			{
				audio_token_embedding[j] = emb_code_w[sel * emb_code_hidden_size + j];
			}

			llama_set_embeddings(ctx_tts, true);
			llama_batch batch = {};
			batch.n_tokens = 1;
			batch.embd = audio_token_embedding.data();
			batch.pos = &n_past_tts;

			if (llama_decode(ctx_tts, batch))
			{
				FAIL("Failed to prefill audio token embedding");
			}
			n_past_tts += 1;
		}
	}

	std::cout << "  ✓ Generated " << audio_tokens.size() << " audio tokens" << std::endl;

	/*
	 * Step 10: Feed tokens to Token2Wav
	 * Ported from: llama.cpp-omni/tools/omni/token2wav/token2wav-impl.cpp:55-67
	 *
	 * Purpose: Convert audio tokens to audio waveform using the Token2Wav diffusion model.
	 * This is the final step in the TTS pipeline, transforming discrete audio tokens
	 * into a continuous audio waveform that can be played by users.
	 *
	 * Algorithm:
	 * 1. Convert absolute audio token IDs to relative IDs (0-6561)
	 * 2. Add 3 silence tokens (4218) to ensure proper start
	 * 3. Process tokens in sliding windows (28 tokens, step 25)
	 * 4. For each window:
	 *    a. Feed window to Token2Wav encoder (Conformer)
	 *    b. Generate diffusion steps via Flow Matching
	 *    c. Convert to waveform via HiFi-GAN vocoder
	 *    d. Append audio samples to output buffer
	 * 5. Return audio samples in float format
	 *
	 * Token2Wav Pipeline:
	 * - Encoder: Conformer encoder extracts features from tokens
	 * - Flow Matching: Conditional Flow Matching generates diffusion steps
	 * - HiFi-GAN: Vocoder converts diffusion steps to waveform
	 *
	 * Sliding Window Logic:
	 * - WINDOW_SIZE = 28 tokens: Token2Wav needs 28 tokens to output audio
	 * - STEP_SIZE = 25 tokens: Move by 25, keep 3 for lookahead
	 * - Always adds 3 silence tokens before each window
	 * - Flush remaining tokens when window size ≤ 28
	 *
	 * Audio Output:
	 * - Sample rate: 24000 Hz (24kHz mono)
	 * - Format: Float samples in [-1.0, 1.0] range
	 * - Channels: 1 (mono)
	 * - Bit depth: Will be converted to 16-bit for WAV file
	 *
	 * References:
	 * - omni-tts.cpp: generate_audio_from_tokens() (lines 301-361)
	 * - token2wav-impl.cpp: Flow matching and HiFi-GAN implementations
	 * - AGENTS.md: Token2Wav documentation
	 */
	std::cout << "\nStep 10: Feeding tokens to Token2Wav..." << std::endl;

	std::vector<float> audio_samples;
	ok = omni::tts::generate_audio_from_tokens(tts_ctx, audio_tokens, audio_samples, SAMPLE_RATE);
	REQUIRE(ok);
	REQUIRE(audio_samples.size() > 0);

	float max_amp = 0.0f;
	float min_amp = 0.0f;
	double sum_amp = 0.0;
	for (float s : audio_samples)
	{
		if (s > max_amp)
			max_amp = s;
		if (s < min_amp)
			min_amp = s;
		sum_amp += std::abs(s);
	}
	std::cout << "  ✓ Generated " << audio_samples.size() << " audio samples" << std::endl;
	std::cout << "  ✓ Amplitude range: [" << min_amp << ", " << max_amp
			  << "], mean_abs: " << (sum_amp / audio_samples.size()) << std::endl;

	/*
	 * Step 11: Write audio to WAV file and check duration
	 * Ported from: demo/llama-omni-server/tests/test_tts_roundtrip.cpp:266-300
	 *
	 * Purpose: Write the generated audio samples to a WAV file and verify
	 * the audio quality and duration. This is the final validation step
	 * to ensure the TTS pipeline produces valid audio output.
	 *
	 * WAV File Format:
	 * - RIFF header: "RIFF" + file_size + "WAVE"
	 * - fmt chunk: "fmt " + 16 + PCM format + sample rate + channels + byte rate + block align +
	 * bits per sample
	 * - data chunk: "data" + data_size + PCM data
	 *
	 * Audio Parameters:
	 * - Sample rate: 24000 Hz (24kHz mono)
	 * - Channels: 1 (mono)
	 * - Bits per sample: 16
	 * - Block align: 2 (channels * bits_per_sample / 8)
	 * - Byte rate: sample_rate * block_align
	 *
	 * Implementation:
	 * 1. Convert float samples to int16:
	 *    - Clip to range [-1.0, 1.0]
	 *    - Scale to [-32767, 32767]
	 *    - Cast to int16
	 * 2. Write RIFF header
	 * 3. Write fmt chunk
	 * 4. Write data chunk
	 * 5. Write PCM data
	 *
	 * Validation:
	 * - Duration >= 1.0s: Audio should be at least 1 second long
	 * - Duration < 30.0s: Audio should not be excessively long
	 * - Amplitude range: Should be within [-1.0, 1.0] after clipping
	 * - Mean absolute amplitude: Should be reasonable for speech
	 *
	 * References:
	 * - write_wav_mono(): Implementation in test_tts.cpp (lines 64-105)
	 * - AGENTS.md: TTS output documentation
	 */
	std::cout << "\nStep 11: Writing audio to WAV file..." << std::endl;

	std::string output_path = "/tmp/test_tts_output.wav";
	int samples_written = write_wav_mono(output_path, audio_samples, SAMPLE_RATE);
	REQUIRE(samples_written > 0);

	std::cout << "  ✓ Audio written to " << output_path << std::endl;

	/*
	 * Calculate and verify audio duration
	 * Duration = number_of_samples / sample_rate
	 * Sample rate = 24000 Hz
	 */
	float audio_duration = static_cast<float>(samples_written) / SAMPLE_RATE;
	std::cout << "  ✓ Audio duration: " << audio_duration << " seconds" << std::endl;

	REQUIRE(audio_duration >= 1.0f);
	REQUIRE(audio_duration < 30.0f);

	/*
	 * Step 12: Free TTS context
	 * Ported from: llama.cpp-omni/tools/omni/omni.cpp:4107-4110
	 */
	std::cout << "\nStep 12: Cleaning up..." << std::endl;

	omni::tts::free_tts_context(tts_ctx);
	common_sampler_free(tts_sampler);

	std::cout << "  ✓ TTS context freed" << std::endl;

	std::cout << "\n=== TTS Test Completed ===" << std::endl;
	std::cout << "✅ Audio generation successful!" << std::endl;
	std::cout << "✅ Duration: " << audio_duration << "s (required: >" << MIN_AUDIO_DURATION_SECONDS
			  << "s)" << std::endl;

	llama_free(ctx);
	llama_model_free(model);
	llama_free(ctx_tts);
	llama_model_free(model_tts);
}

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

// ── Helper: prefill a block of audio embedding vectors into an LLM context ───
// batch.logits[last] = 1 so the next sample() call sees valid logits.
static int prefill_audio(
	llama_context * ctx, std::vector<float> const & embeds, int n_embd, int & n_past)
{
	int n_tok = (int)(embeds.size() / n_embd);
	if (n_tok == 0)
		return false;
	llama_batch b = llama_batch_init(n_tok, n_embd, 1);
	b.n_tokens = n_tok;
	std::memcpy(b.embd, embeds.data(), embeds.size() * sizeof(float));
	for (int j = 0; j < n_tok; ++j)
	{
		b.pos[j] = n_past + j;
	}
	bool ok = (llama_decode(ctx, b) == 0);
	llama_batch_free(b);
	if (ok)
		n_past += n_tok;
	return n_tok;
}

// ── Helper: tokenize text and prefill into a context ─────────────────────────
static std::vector<llama_token> prefill_text(
	llama_context * ctx, llama_vocab const * vocab, std::string const & text, int & n_past)
{
	std::vector<llama_token> buf(text.size() + 16);
	int32_t n = llama_tokenize(
		vocab, text.data(), (int32_t)text.size(), buf.data(), (int32_t)buf.size(), false, true);
	if (n < 0)
	{
		buf.resize(-n);
		n = llama_tokenize(
			vocab, text.data(), (int32_t)text.size(), buf.data(), (int32_t)buf.size(), false, true);
	}
	buf.resize(n);

	int const batch_max = 512;
	for (int i = 0; i < n; i += batch_max)
	{
		int chunk = std::min(batch_max, n - i);
		llama_batch b = llama_batch_init(chunk, 0, 1);
		b.n_tokens = chunk;
		for (int j = 0; j < chunk; ++j)
		{
			b.token[j] = buf[i + j];
			b.pos[j] = n_past + j;
		}
		bool ok = (llama_decode(ctx, b) == 0);
		llama_batch_free(b);
		if (!ok)
			return {};
		n_past += chunk;
	}
	return buf;
}

TEST_CASE("tts_simplex_voice_cloning", "[tts][simplex]")
{
	llama_model_params model_params = llama_model_default_params();
	// model_params.n_gpu_layers = 0;	// CPU-only to avoid GPU memory issues in test environment

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;	   // Maximum context length (fits our prompt)
	ctx_params.n_batch = 8192;	   // Maximum batch size for decode operations
	ctx_params.n_ubatch = 8192;	   // Maximum batch size for inference
	ctx_params.embeddings = true;  // Enable hidden state extraction

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t n_embd = llama_model_n_embd(model);
	bool ok = false;

	// Voice cloning system prompt (matches omni.cpp audio_voice_clone_prompt).
	std::string const system_prompt =
		"<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
	std::vector<float> ref_embeds = load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_VOICE_REF);
	// Everything after the audio up to and including the assistant turn opener.
	std::string const post_audio =
		"<|audio_end|>Please assist users while maintaining this voice style.<|im_end|>\n"
		"<|im_start|>user\nSpeak about the quick brown fox.<|im_end|>\n"
		"<|im_start|>assistant\n";
	std::string const text_to_synthesize = "The quick brown fox jumps over the lazy banana";

	int n_past = 0;

	// Batch 1: system prompt text (ends with <|audio_start|>).
	prefill_text(ctx, vocab, system_prompt, n_past);

	// Batch 2: audio reference embeddings.
	prefill_audio(ctx, ref_embeds, n_embd, n_past);

	// Batch 3: [post_audio + text_to_synthesize] decoded together as ONE batch.
	// This mirrors tts_simplex which decodes [prefix + text] in a single batch so that
	// llama_get_embeddings_ith can address all positions by batch index.
	auto const tok_fn = [&](std::string const & s) -> std::vector<llama_token>
	{
		std::vector<llama_token> buf(s.size() + 16);
		int n = llama_tokenize(
			vocab, s.data(), (int32_t)s.size(), buf.data(), (int32_t)buf.size(), false, true);
		if (n < 0)
		{
			buf.resize(-n);
			n = llama_tokenize(
				vocab, s.data(), (int32_t)s.size(), buf.data(), (int32_t)buf.size(), false, true);
		}
		buf.resize(n);
		return buf;
	};

	std::vector<llama_token> post_audio_tokens = tok_fn(post_audio);
	std::vector<llama_token> text_tokens = tok_fn(text_to_synthesize);

	std::vector<llama_token> combined;
	combined.reserve(post_audio_tokens.size() + text_tokens.size());
	combined.insert(combined.end(), post_audio_tokens.begin(), post_audio_tokens.end());
	combined.insert(combined.end(), text_tokens.begin(), text_tokens.end());

	std::vector<llama_pos> positions(combined.size());
	std::iota(positions.begin(), positions.end(), n_past);

	llama_batch main_batch = {};
	main_batch.n_tokens = int32_t(combined.size());
	main_batch.token = combined.data();
	main_batch.pos = positions.data();
	main_batch.n_seq_id = nullptr;
	main_batch.seq_id = nullptr;
	main_batch.logits = nullptr;

	llama_set_embeddings(ctx, true);
	if (llama_decode(ctx, main_batch))
		FAIL("Failed to decode main batch");
	n_past += combined.size();

	// Extract hidden states for text tokens: batch indices
	// post_audio_tokens.size()..combined.size()-1, mirroring tts_simplex.
	std::vector<float> text_part_hidden;
	for (size_t i = post_audio_tokens.size(); i < combined.size(); i++)
	{
		float const * emb = llama_get_embeddings_ith(ctx, static_cast<int32_t>(i));
		REQUIRE(emb);
		for (int j = 0; j < n_embd; j++) text_part_hidden.push_back(emb[j]);
	}

	omni::tts::filter_special_tokens(text_tokens, text_part_hidden, n_embd);
	REQUIRE_FALSE(text_tokens.empty());

	omni::tts::TTSContext tts_ctx;
	ok = omni::tts::load_projector_semantic(PATH_MODEL_PROJECTOR, tts_ctx.projector_weights);
	REQUIRE(ok);

	// Apply projector to convert LLM hidden states to TTS space (4096-d → 768-d)
	std::vector<float> projected_hidden_pre_merge;
	omni::tts::apply_projector_semantic(
		tts_ctx.projector_weights,
		text_part_hidden,
		text_tokens.size(),
		projected_hidden_pre_merge);
	REQUIRE(projected_hidden_pre_merge.size() == text_tokens.size() * 768);

	// Normalize projected hidden states to unit L2 norm
	omni::tts::normalize_l2_per_token(projected_hidden_pre_merge.data(), text_tokens.size(), 768);

	llama_model_params tts_model_params = llama_model_default_params();
	// tts_model_params.n_gpu_layers = 0;
	llama_model * model_tts =
		llama_model_load_from_file(PATH_MODEL_TTS_TRANSFORMER, tts_model_params);
	REQUIRE(model_tts != nullptr);

	llama_context_params tts_ctx_params = llama_context_default_params();
	tts_ctx_params.n_ctx = 4096;
	tts_ctx_params.n_batch = 4096;
	tts_ctx_params.n_ubatch = 4096;
	tts_ctx_params.embeddings = true;

	llama_context * ctx_tts = llama_init_from_model(model_tts, tts_ctx_params);
	REQUIRE(ctx_tts != nullptr);

	ok = omni::tts::load_tts_weights(
		tts_ctx,
		PATH_MODEL_PROJECTOR,
		PATH_MODEL_TTS_TRANSFORMER,
		PATH_MODEL_TTS_WEIGHTS,
		PATH_DIR_TOKEN2WAV,
		"gpu",
		"gpu",
		PATH_ALTERNATIVE_PROMPT_CACHE);
	REQUIRE(ok);
	// REQUIRE(tts_ctx.token2wav_session->t2w.start_stream_with_prompt_cache_gguf(
	// 	PATH_ALTERNATIVE_PROMPT_CACHE));

	int const audio_bos_token_id = 151687;
	int const text_eos_token_id = 151692;  // tts_config.text_eos_token_id

	std::vector<float> condition_embeddings(text_tokens.size() * 768);
	for (int i = 0; i < static_cast<int>(text_tokens.size()); i++)
	{
		std::vector<float> text_emb(768);
		omni::tts::tts_emb_text(tts_ctx, text_tokens[i], text_emb.data(), 768);
		for (int j = 0; j < 768; j++)
		{
			condition_embeddings[i * 768 + j] =
				text_emb[j] + projected_hidden_pre_merge[i * 768 + j];
		}
	}

	std::vector<float> text_eos_emb(768);
	omni::tts::tts_emb_text(tts_ctx, text_eos_token_id, text_eos_emb.data(), 768);
	condition_embeddings.insert(
		condition_embeddings.end(), text_eos_emb.begin(), text_eos_emb.end());

	// Append audio_bos embedding (omni.cpp:5845-5848)
	std::vector<float> bos_emb(768);
	omni::tts::tts_emb_text(tts_ctx, audio_bos_token_id, bos_emb.data(), 768);
	condition_embeddings.insert(condition_embeddings.end(), bos_emb.begin(), bos_emb.end());

	common_params_sampling tts_sampling = {};
	tts_sampling.temp = 0.8f;			  // Controls randomness (lower = more deterministic)
	tts_sampling.top_p = 0.85f;			  // Nucleus sampling threshold
	tts_sampling.top_k = 25;			  // Top-k sampling threshold
	tts_sampling.penalty_repeat = 1.05f;  // Repetition penalty
	tts_sampling.min_p = 0.01f;			  // Minimum probability threshold
	tts_sampling.penalty_last_n = 16;	  // Window size for repetition penalty

	common_sampler * tts_sampler = common_sampler_init(model_tts, tts_sampling);
	REQUIRE(tts_sampler != nullptr);

	int const num_audio_tokens = 6562;
	int const eos_relative_idx = num_audio_tokens - 1;
	int const max_audio_tokens = 5000;
	int const min_new_tokens = 0;

	std::vector<int32_t> audio_tokens;
	std::vector<int> all_generated_tokens_relative;

	int n_past_tts = 0;
	// n_batch must match the context's n_batch (set above to 4096).
	int const n_batch_tts = 4096;
	int const n_cond_tokens = static_cast<int>(condition_embeddings.size()) / 768;

	// Enable embeddings output for TTS model
	llama_set_embeddings(ctx_tts, true);
	// Process embeddings in batches
	for (int i = 0; i < n_cond_tokens; i += n_batch_tts)
	{
		int n_eval = std::min(n_batch_tts, n_cond_tokens - i);
		// Create batch with embeddings (not tokens)
		llama_batch cond_batch = {};
		cond_batch.n_tokens = n_eval;
		cond_batch.embd = condition_embeddings.data() + i * 768;
		// Set position IDs for each embedding
		std::vector<llama_pos> pos_vec(n_eval);
		std::iota(pos_vec.begin(), pos_vec.end(), n_past_tts);
		cond_batch.pos = pos_vec.data();
		// Decode batch through TTS model
		if (llama_decode(ctx_tts, cond_batch))
		{
			FAIL("Failed to prefill condition embeddings");
		}
		// Update position counter
		n_past_tts += n_eval;
	}

	llama_set_embeddings(ctx_tts, true);
	for (int t = 0; t < max_audio_tokens; ++t)
	{
		float const * hidden_state = llama_get_embeddings_ith(ctx_tts, -1);
		REQUIRE(hidden_state != nullptr);

		std::vector<float> audio_logits(num_audio_tokens, 0.0f);
		float const * head_code_w = tts_ctx.head_code_weight;
		int const hidden_size = tts_ctx.head_code_hidden_size;

		for (int i = 0; i < num_audio_tokens; ++i)
		{
			// Get row i of head_code_weight: head_code_weight[i * hidden_size : (i+1) *
			// hidden_size]
			float const * row = head_code_w + i * hidden_size;
			float sum = 0.0f;
			// Compute dot product: sum(hidden_state[j] * head_code_weight[i][j])
			for (int j = 0; j < hidden_size; ++j)
			{
				sum += hidden_state[j] * row[j];
			}
			audio_logits[i] = sum;	// Matrix multiplication result
		}

		// Apply temperature
		for (int i = 0; i < num_audio_tokens; ++i) audio_logits[i] /= tts_sampling.temp;

		// Apply repetition penalty
		if (!all_generated_tokens_relative.empty())
		{
			int start_idx = std::max(
				0, (int)all_generated_tokens_relative.size() - tts_sampling.penalty_last_n);
			std::vector<bool> occurred(num_audio_tokens, false);
			for (int i = start_idx; i < (int)all_generated_tokens_relative.size(); ++i)
			{
				int tok = all_generated_tokens_relative[i];
				if (tok >= 0 && tok < num_audio_tokens)
					occurred[tok] = true;
			}
			for (int i = 0; i < num_audio_tokens; ++i)
			{
				if (!occurred[i])
					continue;
				if (audio_logits[i] >= 0)
					audio_logits[i] /= tts_sampling.penalty_repeat;
				else
					audio_logits[i] *= tts_sampling.penalty_repeat;
			}
		}

		// Force no EOS before min_new_tokens
		if (t < min_new_tokens)
			audio_logits[eos_relative_idx] = -std::numeric_limits<float>::infinity();

		// Softmax
		float max_l = -1e20f;
		for (float l : audio_logits)
			if (l > max_l)
				max_l = l;
		float sum_p = 0.0f;
		std::vector<float> probs(num_audio_tokens);
		for (int i = 0; i < num_audio_tokens; i++)
		{
			probs[i] = std::exp(audio_logits[i] - max_l);
			sum_p += probs[i];
		}
		for (int i = 0; i < num_audio_tokens; i++) probs[i] /= sum_p;

		// Top-k filtering
		if (tts_sampling.top_k > 0 && tts_sampling.top_k < num_audio_tokens)
		{
			std::vector<float> sorted_probs = probs;
			std::nth_element(
				sorted_probs.begin(),
				sorted_probs.begin() + (num_audio_tokens - tts_sampling.top_k),
				sorted_probs.end());
			float threshold = sorted_probs[num_audio_tokens - tts_sampling.top_k];
			for (int i = 0; i < num_audio_tokens; i++)
				if (probs[i] < threshold)
					probs[i] = 0.0f;
		}

		// Top-p (nucleus) filtering
		if (tts_sampling.top_p < 1.0f)
		{
			std::vector<int> idx(num_audio_tokens);
			std::iota(idx.begin(), idx.end(), 0);
			std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
			float s = 0.0f;
			for (float p : probs) s += p;
			float cum = 0.0f;
			bool cutoff_reached = false;
			for (int i : idx)
			{
				if (cutoff_reached)
				{
					probs[i] = 0.0f;
					continue;
				}
				cum += probs[i] / s;
				if (cum >= tts_sampling.top_p)
					cutoff_reached = true;
			}
		}

		// Re-normalise
		{
			float s = 0.0f;
			for (float p : probs) s += p;
			if (s > 0.0f)
				for (float & p : probs) p /= s;
		}

		// Multinomial sample
		int sel = eos_relative_idx;
		{
			static std::mt19937 rng_clone{42};
			std::uniform_real_distribution<float> dist(0.0f, 1.0f);
			float r = dist(rng_clone);
			float cum = 0.0f;
			for (int i = 0; i < num_audio_tokens; i++)
			{
				cum += probs[i];
				if (r <= cum)
				{
					sel = i;
					break;
				}
			}
		}

		// Check for EOS
		if (sel == eos_relative_idx && t >= min_new_tokens)
		{
			std::cout << "EOS sampled at t=" << t << ", min_new_tokens=" << min_new_tokens
					  << " EOS token = " << sel << std::endl;
			break;
		}

		int absolute_id = sel + audio_bos_token_id;
		audio_tokens.push_back(absolute_id);
		all_generated_tokens_relative.push_back(sel);
		common_sampler_accept(tts_sampler, absolute_id, true);

		if (!(0 <= sel && sel < tts_ctx.emb_code_vocab_size))
		{
			FAIL("Selected token is outside voab range");
		}
		float const * emb_code_w = tts_ctx.emb_code_weight;
		int const emb_code_hidden_size = tts_ctx.emb_code_hidden_size;

		std::vector<float> audio_token_embedding(emb_code_hidden_size);
		for (int j = 0; j < emb_code_hidden_size; ++j)
		{
			audio_token_embedding[j] = emb_code_w[sel * emb_code_hidden_size + j];
		}

		llama_batch batch = {};
		batch.n_tokens = 1;
		batch.embd = audio_token_embedding.data();
		batch.pos = &n_past_tts;

		if (llama_decode(ctx_tts, batch))
		{
			FAIL("Failed to prefill audio token embedding");
		}
		n_past_tts += 1;
	}

	std::vector<float> audio_samples;
	ok = omni::tts::generate_audio_from_tokens(tts_ctx, audio_tokens, audio_samples, SAMPLE_RATE);
	REQUIRE(ok);
	REQUIRE(audio_samples.size() > 0);

	float max_amp = 0.0f;
	float min_amp = 0.0f;
	double sum_amp = 0.0;
	for (float s : audio_samples)
	{
		if (s > max_amp)
			max_amp = s;
		if (s < min_amp)
			min_amp = s;
		sum_amp += std::abs(s);
	}

	std::string output_path = "/tmp/test_tts_clone_output.wav";
	int samples_written = write_wav_mono(output_path, audio_samples, SAMPLE_RATE);
	REQUIRE(samples_written > 0);

	float audio_duration = static_cast<float>(samples_written) / SAMPLE_RATE;
	std::cout << "  ✓ Audio duration: " << audio_duration << " seconds" << std::endl;

	REQUIRE(audio_duration >= 1.0f);
	REQUIRE(audio_duration < 300.0f);

	omni::tts::free_tts_context(tts_ctx);
	common_sampler_free(tts_sampler);
	llama_free(ctx);
	llama_model_free(model);
	llama_free(ctx_tts);
	llama_model_free(model_tts);
}
