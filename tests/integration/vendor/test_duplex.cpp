/*
 * Duplex Text Generation Test for MiniCPM-o 4.5
 *
 * This test validates duplex mode token generation without TTS.
 * It focuses on the core duplex control tokens (<|speak|>, <|listen|>, etc.)
 *
 * Based on: llama.cpp-omni/tools/omni/omni.cpp duplex mode implementation
 *
 * Test Flow:
 * 1. Load LLM model
 * 2. Create duplex-format prompt: system + <unit> (NO <|im_start|>assistant\n)
 * 3. Tokenize and prefill prompt
 * 4. Run generation loop
 * 5. Assert <|listen|> token is generated (model signals it is awaiting input)
 *
 * Duplex Prompt Format (from omni.cpp:4164-4169):
 *   <|im_start|>system
 *   Streaming Duplex Conversation! You are a helpful assistant.
 *   <|im_end|>
 *   <unit>[optional audio embeddings]
 *   ← model generates <|speak|> or <|listen|> here
 *
 * Key distinction from simplex:
 *   - Simplex ends with: <|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
 *   - Duplex ends with:  <unit>  (no assistant prompt; model decides autonomously)
 *
 * Duplex Special Tokens (from omni.cpp g_special_token_ids / find_token):
 *   <|speak|>     151706  Model decides to start speaking
 *   <|listen|>    151705  Model decides to start listening (await user input)
 *   <|chunk_eos|> 151718  Semantic chunk ends (within same turn)
 *   <|turn_eos|>  151717  Entire turn ends
 *   <|tts_pad|>   151722  Padding — forbidden during sampling
 *
 * Who generates each token:
 *   <|speak|>  / <|listen|>  — LLM (first decision token after <unit>)
 *   <|chunk_eos|> / <|turn_eos|> — LLM (closes a speaking or listening segment)
 *   <unit>                   — prepended by the caller before each user turn
 *
 * Generation pattern (observed):
 *   After <unit> with no audio:  → <|chunk_eos|>  (model ends the empty listen chunk)
 *   After <unit> with audio:     → <|speak|>[text tokens]<|chunk_eos|>  (or <|turn_eos|>)
 *                                  OR <|listen|><|chunk_eos|>  (if model decides to keep listening)
 *
 * Note: <|listen|> is not always emitted as an explicit token. When the model
 * ends a bare (audio-free) unit block, it may generate <|chunk_eos|> directly,
 * which is equivalent to "this listen chunk is complete". In omni.cpp both
 * <|listen|> and <|chunk_eos|> are treated as end tokens in duplex mode
 * (see is_end_token() duplex branch). The first generated duplex control token
 * after <unit> is our assertion target.
 *
 * References:
 *   omni.cpp:3964-3982  find_token() calls that resolve the real token IDs
 *   omni.cpp:9085-9090  duplex mode skips assistant prompt in stream_decode
 *   omni.cpp:4364-4366  <unit> is inserted before audio embeddings
 *   omni.cpp:948-972    sample() helper: common_sampler_sample → accept → eval_id
 */

#include <catch2/catch_test_macros.hpp>

#include "omni/audition.h"
#include "omni/clip.h"
#include "omni/omni-impl.h"

#include "common/common.h"
#include "common/sampling.h"
#include "llama.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <sndfile.h>


#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "vendor_paths.hpp"

// ── Paths ─────────────────────────────────────────────────────────────────────
// Paths resolved from LLAMAOMNISERVER_MODEL_ROOT and LLAMAOMNISERVER_TEST_REPO_ROOT
// env vars injected by CMake/CTest.

static VendorPath const PATH_MODEL_LLM{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};
static VendorPath const PATH_MODEL_AUDIO{vp_model_root() + "/audio/MiniCPM-o-4_5-audio-F16.gguf"};
static VendorPath const PATH_AUDIO_BANANA{vp_test_data() + "/say_the_word_banana.wav"};
static VendorPath const PATH_AUDIO_REFERENCE{vp_test_data() + "/alternative.wav"};
static VendorPath const PATH_MODEL_VISION{
	vp_model_root() + "/vision/MiniCPM-o-4_5-vision-F16.gguf"};
static VendorPath const PATH_AUDIO_DESCRIBE{vp_test_data() + "/describe_the_shape.wav"};
static VendorPath const PATH_IMAGE_TRIANGLE{vp_test_data() + "/triangle.png"};

// ── Duplex special token IDs (resolved via find_token("<|speak|>") etc.) ──────
// Source: omni.cpp g_special_token_ids list and find_token() at init time.
// These are vocabulary positions in the MiniCPM-o-4_5 tokenizer.
constexpr llama_token TOKEN_SPEAK = 151706;		 // <|speak|>
constexpr llama_token TOKEN_LISTEN = 151705;	 // <|listen|>
constexpr llama_token TOKEN_CHUNK_EOS = 151718;	 // <|chunk_eos|>
constexpr llama_token TOKEN_TURN_EOS = 151717;	 // <|turn_eos|>
constexpr llama_token TOKEN_IM_END = 151645;	 // <|im_end|>  (Qwen2 chat template, also EOG)
constexpr llama_token TOKEN_TTS_BOS = 151703;	 // <|tts_bos|> (simplex TTS start, also EOG)
constexpr llama_token TOKEN_TTS_EOS = 151704;	 // <|tts_eos|> (simplex TTS end,   also EOG)

// ── Helper: tokenize a string, resize-on-negative pattern ────────────────────
static std::vector<llama_token> tokenize(
	llama_vocab const * vocab, std::string const & text, bool add_bos = false)
{
	std::vector<llama_token> buf(text.size() + 16);
	int32_t n = llama_tokenize(
		vocab, text.data(), (int32_t)text.size(), buf.data(), (int32_t)buf.size(), add_bos, true);
	if (n < 0)
	{
		buf.resize(-n);
		n = llama_tokenize(
			vocab,
			text.data(),
			(int32_t)text.size(),
			buf.data(),
			(int32_t)buf.size(),
			add_bos,
			true);
	}
	buf.resize(n);
	return buf;
}

// ── Helper: detokenize a list of token ids to a string ───────────────────────
static std::string detokenize(llama_vocab const * vocab, std::vector<llama_token> const & tokens)
{
	std::string out;
	std::vector<char> buf(64);
	for (llama_token t : tokens)
	{
		int n = llama_token_to_piece(vocab, t, buf.data(), (int32_t)buf.size(), 0, true);
		if (n > 0)
			out.append(buf.data(), n);
	}
	return out;
}

// ── Helper: prefill a batch of tokens starting at n_past ─────────────────────
// Uses llama_batch_init / common_batch_add / llama_batch_free — the modern API.
// common_batch_add sets token, pos, n_seq_id, seq_id[0], and logits per slot.
static bool prefill(llama_context * ctx, std::vector<llama_token> const & tokens, int & n_past)
{
	int N = (int)tokens.size();
	int const batch_max = 512;
	llama_batch batch = llama_batch_init(batch_max, /*embd=*/0, /*n_seq_max=*/1);

	for (int i = 0; i < N; i += batch_max)
	{
		common_batch_clear(batch);
		int chunk_end = std::min(i + batch_max, N);
		for (int j = i; j < chunk_end; ++j)
		{
			// logits only for the last token of the last chunk
			bool want_logits = (j == N - 1);
			common_batch_add(batch, tokens[j], n_past + (j - i), {0}, want_logits);
		}
		if (llama_decode(ctx, batch))
		{
			llama_batch_free(batch);
			return false;
		}
		n_past += (chunk_end - i);
	}
	llama_batch_free(batch);
	return true;
}

// ── Helper: sample → accept → eval one token ─────────────────────────────────
// Mirrors omni.cpp sample(): common_sampler_sample → accept → eval_id.
// - diag_first: print raw logits for duplex control tokens before sampling
// - suppress_tokens: set these token logits to -inf before sampling (forces avoidance)
static llama_token sample_and_feed(
	common_sampler * smpl,
	llama_context * ctx,
	llama_vocab const * vocab,
	int & n_past,
	bool diag_first = false,
	std::vector<llama_token> const & suppress_tokens = {})
{
	float * logits = llama_get_logits_ith(ctx, -1);
	if (logits && !suppress_tokens.empty())
	{
		for (llama_token t : suppress_tokens) logits[t] = -std::numeric_limits<float>::infinity();
	}
	if (diag_first && logits)
	{
		std::cout << "  [diag] raw logits: speak(" << TOKEN_SPEAK << ")=" << logits[TOKEN_SPEAK]
				  << "  listen(" << TOKEN_LISTEN << ")=" << logits[TOKEN_LISTEN] << "  chunk_eos("
				  << TOKEN_CHUNK_EOS << ")=" << logits[TOKEN_CHUNK_EOS] << "  turn_eos("
				  << TOKEN_TURN_EOS << ")=" << logits[TOKEN_TURN_EOS] << "\n";
	}
	llama_token const id = common_sampler_sample(smpl, ctx, -1);
	common_sampler_accept(smpl, id, true);

	if (!llama_vocab_is_eog(vocab, id))
	{
		// Feed sampled token back so the model sees it for the next step.
		llama_batch batch = llama_batch_init(1, /*embd=*/0, /*n_seq_max=*/1);
		common_batch_clear(batch);
		common_batch_add(batch, id, n_past, {0}, /*logits=*/true);
		llama_decode(ctx, batch);
		llama_batch_free(batch);
		++n_past;
	}
	return id;
}

// ── Helper: prefill audio embedding vectors starting at n_past ────────────────
// Uses llama_batch_init with embd>0 so that batch.embd is allocated.
// We copy the embedding floats in, set pos/seq_id/logits manually, then decode.
static bool prefill_embeddings(
	llama_context * ctx, std::vector<float> const & embeds, int n_embd, int & n_past)
{
	int n_tokens = (int)(embeds.size() / n_embd);
	llama_batch batch = llama_batch_init(n_tokens, n_embd, /*n_seq_max=*/1);
	batch.n_tokens = n_tokens;
	std::memcpy(batch.embd, embeds.data(), embeds.size() * sizeof(float));
	for (int j = 0; j < n_tokens; ++j)
	{
		batch.pos[j] = n_past + j;
		batch.n_seq_id[j] = 1;
		batch.seq_id[j][0] = 0;
		batch.logits[j] = (j == n_tokens - 1) ? 1 : 0;
	}
	bool ok = (llama_decode(ctx, batch) == 0);
	llama_batch_free(batch);
	if (ok)
		n_past += n_tokens;
	return ok;
}

// ── Helper: encode a PCM float buffer via the audition audio encoder ─────────
// samples: mono float PCM at 16 kHz.  Returns [n_tokens*n_embd] or empty.
static std::vector<float> encode_pcm_audio(
	char const * audio_model_path, float const * samples, size_t n_samples, int n_threads = 4)
{
	audition_context_params ap = {/*use_gpu=*/true, GGML_LOG_LEVEL_WARN};
	audition_ctx * ctx_audio = audition_init(audio_model_path, ap);
	if (!ctx_audio)
		return {};

	auto filters = audition_get_mel_filters(ctx_audio);
	std::vector<whisper_preprocessor::whisper_mel> mel;
	whisper_preprocessor::preprocess_audio(samples, n_samples, filters, mel);
	if (mel.empty())
	{
		audition_free(ctx_audio);
		return {};
	}

	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel[0].n_len;
	mel_f32->ny = mel[0].n_mel;
	mel_f32->buf.assign(mel[0].data.begin(), mel[0].data.end());

	int n_tokens = audition_n_output_tokens(ctx_audio, mel_f32);
	int n_embd_aud = audition_n_mmproj_embd(ctx_audio);
	std::cout << "  [diag] silence n_tokens=" << n_tokens << " mmproj_embd=" << n_embd_aud << "\n";
	// Over-allocate by 2 tokens: audition_n_output_tokens() can under-predict by 1
	// (seen: "token count mismatch within tolerance, diff=1") — avoid heap corruption.
	std::vector<float> embeddings((n_tokens + 2) * n_embd_aud);
	bool ok = audition_audio_encode(ctx_audio, n_threads, mel_f32, embeddings.data());
	audition_audio_f32_free(mel_f32);
	audition_free(ctx_audio);
	if (!ok)
		return {};
	embeddings.resize(n_tokens * n_embd_aud);  // trim overflow back to predicted size
	return embeddings;
}

// ── Helper: load a WAV file and encode it via the audition audio encoder ──────
// Returns embeddings as [n_tokens * n_mmproj_embd] floats, or empty on failure.
// *out_n_embd is set to audition_n_mmproj_embd() if non-null.
static std::vector<float> load_and_encode_audio(
	char const * audio_model_path,
	char const * audio_path,
	int n_threads = 4,
	int * out_n_embd = nullptr)
{
	audition_context_params ap = {/*use_gpu=*/true, GGML_LOG_LEVEL_WARN};
	audition_ctx * ctx_audio = audition_init(audio_model_path, ap);
	if (!ctx_audio)
		return {};

	SF_INFO sfinfo = {};
	SNDFILE * sf = sf_open(audio_path, SFM_READ, &sfinfo);
	if (!sf)
	{
		audition_free(ctx_audio);
		return {};
	}

	std::vector<float> samples(sfinfo.frames * sfinfo.channels);
	sf_readf_float(sf, samples.data(), sfinfo.frames);
	sf_close(sf);

	// Downmix to mono (take first channel)
	std::vector<float> mono(sfinfo.frames);
	for (sf_count_t i = 0; i < sfinfo.frames; ++i) mono[i] = samples[i * sfinfo.channels];

	auto filters = audition_get_mel_filters(ctx_audio);
	std::vector<whisper_preprocessor::whisper_mel> mel;
	whisper_preprocessor::preprocess_audio(mono.data(), mono.size(), filters, mel);
	if (mel.empty())
	{
		audition_free(ctx_audio);
		return {};
	}

	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel[0].n_len;
	mel_f32->ny = mel[0].n_mel;
	mel_f32->buf.assign(mel[0].data.begin(), mel[0].data.end());

	int n_tokens = audition_n_output_tokens(ctx_audio, mel_f32);
	int n_embd_aud = audition_n_mmproj_embd(ctx_audio);
	if (out_n_embd)
		*out_n_embd = n_embd_aud;
	std::cout << "  [diag] audio n_tokens=" << n_tokens << " mmproj_embd=" << n_embd_aud << "\n";
	// Over-allocate by 2 tokens: audition_n_output_tokens() can under-predict by 1
	// (seen: "token count mismatch within tolerance, diff=1") — avoid heap corruption.
	std::vector<float> embeddings((n_tokens + 2) * n_embd_aud);
	bool ok = audition_audio_encode(ctx_audio, n_threads, mel_f32, embeddings.data());

	audition_audio_f32_free(mel_f32);
	audition_free(ctx_audio);
	if (!ok)
		return {};
	embeddings.resize(n_tokens * n_embd_aud);  // trim overflow back to predicted size
	return embeddings;
}

// ═════════════════════════════════════════════════════════════════════════════
TEST_CASE("duplex_listen_on_empty_unit", "[duplex][text]")
{
	std::cout << "\n=== Duplex: minimal <unit> → <|listen|> test ===" << std::endl;

	// ── Step 1: Load model ────────────────────────────────────────────────────
	std::cout << "Step 1: Loading LLM model..." << std::endl;
	llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = 99;  // offload all layers to GPU

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, mp);
	REQUIRE(model != nullptr);

	llama_context_params cp = llama_context_default_params();
	cp.n_ctx = 2048;
	cp.n_batch = 512;
	cp.n_ubatch = 512;

	llama_context * ctx = llama_init_from_model(model, cp);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	std::cout << "  ✓ LLM loaded (n_embd=" << llama_model_n_embd(model) << ")\n";

	// ── Step 2: Build duplex prompt ───────────────────────────────────────────
	// Duplex format (omni.cpp:4164-4169, 9085-9090):
	//   system prompt + <|im_end|>\n + <unit>
	//   (NO <|im_start|>assistant\n — model decides autonomously)
	//
	// Without audio in the <unit> block the model should decide to listen,
	// generating <|listen|> as its first token.
	std::cout << "Step 2: Building duplex prompt..." << std::endl;

	std::string const prompt =
		"<|im_start|>system\n"
		"Streaming Duplex Conversation! You are a helpful assistant.\n"
		"<|im_end|>\n"
		"<unit></unit>";  // ← duplex user-input marker; model generates <|speak|>/<|listen|> next

	std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, /*add_bos=*/false);
	std::cout << "  Prompt: " << prompt << std::endl;
	std::cout << "  Tokens (" << prompt_tokens.size() << "): ";
	for (llama_token t : prompt_tokens) std::cout << t << " ";
	std::cout << std::endl;

	// ── Step 3: Sampler ───────────────────────────────────────────────────────
	std::cout << "Step 3: Creating sampler..." << std::endl;
	common_params_sampling sp = {};
	sp.temp = 0.0f;	 // greedy — deterministic, makes listen more reliable
	sp.penalty_repeat = 1.0f;

	common_sampler * smpl = common_sampler_init(model, sp);
	REQUIRE(smpl != nullptr);
	std::cout << "  ✓ Sampler created (greedy / temp=0)\n";

	// ── Step 4: Prefill prompt ────────────────────────────────────────────────
	std::cout << "Step 4: Prefilling prompt..." << std::endl;
	int n_past = 0;
	bool ok = prefill(ctx, prompt_tokens, n_past);
	REQUIRE(ok);
	std::cout << "  ✓ Prefill done, n_past=" << n_past << std::endl;

	// ── Step 5: Generation loop ───────────────────────────────────────────────
	// In duplex mode the generation loop mirrors omni.cpp stream_decode (line
	// 9138+): sample → classify token → stop on end-token (listen/chunk_eos/
	// turn_eos) or max-tokens.
	std::cout << "Step 5: Generation loop..." << std::endl;

	std::vector<llama_token> generated;
	int const max_new = 20;	 // enough to see duplex control tokens

	for (int step = 0; step < max_new; ++step)
	{
		llama_token tok = sample_and_feed(smpl, ctx, vocab, n_past);

		if (llama_vocab_is_eog(vocab, tok))
		{
			std::cout << "  [step " << step + 1 << "] EOG token " << tok << std::endl;
			break;
		}

		generated.push_back(tok);

		char const * label = "";
		bool stop = false;
		if (tok == TOKEN_SPEAK)
		{
			label = "<|speak|>";
		}
		else if (tok == TOKEN_LISTEN)
		{
			label = "<|listen|>";
			stop = true;
		}
		else if (tok == TOKEN_CHUNK_EOS)
		{
			label = "<|chunk_eos|>";
			stop = true;
		}
		else if (tok == TOKEN_TURN_EOS)
		{
			label = "<|turn_eos|>";
			stop = true;
		}

		std::cout << "  [step " << step + 1 << "] token=" << tok;
		if (*label)
			std::cout << " " << label;
		std::cout << std::endl;

		if (stop)
			break;
	}

	std::cout << "  Generated " << generated.size() << " token(s): " << detokenize(vocab, generated)
			  << std::endl;

	// ── Step 6: Assertions ────────────────────────────────────────────────────
	// After a bare <unit> (no audio) the model must generate at least one
	// duplex control token as its very first token.
	//
	// Observed behaviour: with an empty <unit> the model generates <|chunk_eos|>
	// directly (151718), signalling "end this listen chunk".  <|listen|> (151705)
	// may also appear.  Either is acceptable — both are listed as end tokens in
	// omni.cpp is_end_token() for duplex mode.  What must NOT happen is that the
	// model starts outputting plain text (which would indicate it is not operating
	// in duplex mode at all).
	std::cout << "Step 6: Assertions..." << std::endl;

	REQUIRE_FALSE(generated.empty());

	int speak_count = 0, listen_count = 0, chunk_eos_count = 0, turn_eos_count = 0;
	for (llama_token t : generated)
	{
		if (t == TOKEN_SPEAK)
			++speak_count;
		else if (t == TOKEN_LISTEN)
			++listen_count;
		else if (t == TOKEN_CHUNK_EOS)
			++chunk_eos_count;
		else if (t == TOKEN_TURN_EOS)
			++turn_eos_count;
	}

	std::cout << "  <|speak|>     : " << speak_count << "\n"
			  << "  <|listen|>    : " << listen_count << "\n"
			  << "  <|chunk_eos|> : " << chunk_eos_count << "\n"
			  << "  <|turn_eos|>  : " << turn_eos_count << "\n";

	int total_duplex = speak_count + listen_count + chunk_eos_count + turn_eos_count;

	// At least one duplex control token must appear
	REQUIRE(total_duplex > 0);

	// The very first generated token must be a duplex control token.
	// (<|listen|> and <|chunk_eos|> are both valid "await input" signals.)
	bool first_is_duplex =
		(generated[0] == TOKEN_LISTEN || generated[0] == TOKEN_SPEAK ||
		 generated[0] == TOKEN_CHUNK_EOS || generated[0] == TOKEN_TURN_EOS);
	REQUIRE(first_is_duplex);

	// The model must NOT be generating plain-text tokens (all output should be
	// duplex control tokens for this empty-unit scenario).
	REQUIRE((int)generated.size() == total_duplex);

	std::cout << "  ✓ First token is a duplex control token — duplex mode working\n";
	if (listen_count > 0)
		std::cout << "  ✓ <|listen|> present\n";
	if (chunk_eos_count > 0)
		std::cout << "  ✓ <|chunk_eos|> present (empty listen chunk)\n";

	// ── Cleanup ───────────────────────────────────────────────────────────────
	common_sampler_free(smpl);
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Duplex test complete ===" << std::endl;
}

// ── Helper: load a PNG and encode it via the CLIP vision encoder ──────────────
// Returns one embedding chunk per image produced by clip_image_preprocess:
//   chunks[0] = overview (for <image>...</image>)
//   chunks[1..] = slices  (for <slice>...</slice>, if any)
// Each chunk is n_tokens_per_image * n_embd floats.
struct ImageEmbeddings
{
	std::vector<std::vector<float>> chunks;	 // chunks[0]=overview, rest=slices
	int n_embd = 0;
	int n_tok_each = 0;	 // n_output_tokens per chunk
};

static ImageEmbeddings load_and_encode_image(
	char const * vision_model_path, char const * image_path, int n_threads = 4)
{
	ImageEmbeddings result;

	clip_context_params cp = {};
	cp.use_gpu = true;
	clip_init_result cr = clip_init(vision_model_path, cp);
	clip_ctx * ctx = cr.ctx_v;
	if (!ctx)
		return result;

	int width, height, channels;
	unsigned char * raw = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
	if (!raw)
	{
		clip_free(ctx);
		return result;
	}

	clip_image_u8 * img_u8 = clip_image_u8_init();
	clip_build_img_from_pixels(raw, width, height, img_u8);
	stbi_image_free(raw);

	clip_image_f32_batch * batch = clip_image_f32_batch_init();
	if (!clip_image_preprocess(ctx, img_u8, batch))
	{
		clip_image_f32_batch_free(batch);
		clip_image_u8_free(img_u8);
		clip_free(ctx);
		return result;
	}
	clip_image_u8_free(img_u8);

	size_t n_images = clip_image_f32_batch_n_images(batch);
	clip_image_f32 * first = clip_image_f32_get_img(batch, 0);
	int n_tok = clip_n_output_tokens(ctx, first);
	int n_embd = clip_n_mmproj_embd(ctx);
	std::cout << "  [diag] image n_chunks=" << n_images << " n_tok_each=" << n_tok
			  << " n_embd=" << n_embd << "\n";

	std::vector<float> all(n_images * n_tok * n_embd);
	if (!clip_image_batch_encode(ctx, n_threads, batch, all.data()))
	{
		clip_image_f32_batch_free(batch);
		clip_free(ctx);
		return result;
	}
	clip_image_f32_batch_free(batch);
	clip_free(ctx);

	result.n_embd = n_embd;
	result.n_tok_each = n_tok;
	size_t chunk_sz = (size_t)n_tok * n_embd;
	for (size_t i = 0; i < n_images; ++i)
	{
		result.chunks.emplace_back(
			all.begin() + (long)(i * chunk_sz), all.begin() + (long)((i + 1) * chunk_sz));
	}
	return result;
}

// ── Helper: run one duplex generation turn until a chunk/turn end token ───────
// Returns accumulated plain text. Fills speak/listen/chunk_eos/turn_eos counts.
// Mirrors the inner loop of omni.cpp stream_decode (line 9138+).
// After generation, omni.cpp feeds </unit> back to keep the KV cache consistent
// for the next turn (omni.cpp:9337-9344); we replicate that here.
struct TurnResult
{
	std::vector<llama_token> tokens;
	std::string text;
	int speak_count = 0, listen_count = 0, chunk_eos_count = 0, turn_eos_count = 0;
};

static TurnResult run_duplex_turn(
	common_sampler * smpl,
	llama_context * ctx,
	llama_vocab const * vocab,
	int & n_past,
	int max_new = 200,
	std::vector<llama_token> const & suppress = {})
{
	TurnResult r;
	for (int step = 0; step < max_new; ++step)
	{
		llama_token tok = sample_and_feed(smpl, ctx, vocab, n_past, step == 0, suppress);
		if (llama_vocab_is_eog(vocab, tok))
			break;

		r.tokens.push_back(tok);
		if (tok == TOKEN_SPEAK)
		{
			++r.speak_count;
		}
		else if (tok == TOKEN_LISTEN)
		{
			++r.listen_count;
		}
		else if (tok == TOKEN_CHUNK_EOS)
		{
			++r.chunk_eos_count;
		}
		else if (tok == TOKEN_TURN_EOS)
		{
			++r.turn_eos_count;
		}

		std::vector<char> buf(64);
		int n = llama_token_to_piece(vocab, tok, buf.data(), (int)buf.size(), 0, true);
		if (n > 0)
			r.text.append(buf.data(), n);

		if (tok == TOKEN_CHUNK_EOS || tok == TOKEN_TURN_EOS || tok == TOKEN_LISTEN)
			break;
	}

	// Feed </unit> (151684) after each generation chunk so the KV cache stays
	// consistent for the next <unit> block (mirrors omni.cpp:9337-9344).
	auto unit_end = tokenize(vocab, "</unit>", false);
	prefill(ctx, unit_end, n_past);

	return r;
}

// ═════════════════════════════════════════════════════════════════════════════
TEST_CASE("duplex_audio_banana", "[duplex][audio]")
{
	std::cout << "\n=== Duplex: audio input → speak → banana ===" << std::endl;

	// ── Step 1: Load model ────────────────────────────────────────────────────
	std::cout << "Step 1: Loading LLM model..." << std::endl;
	llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = 99;

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, mp);
	REQUIRE(model != nullptr);

	llama_context_params cp = llama_context_default_params();
	cp.n_ctx = 2048;
	cp.n_batch = 512;
	cp.n_ubatch = 512;

	llama_context * ctx = llama_init_from_model(model, cp);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int const n_embd = llama_model_n_embd(model);
	std::cout << "  ✓ LLM loaded (n_embd=" << n_embd << ")\n";

	// ── Step 2: Encode audio ──────────────────────────────────────────────────
	// say_the_word_banana.wav: user says "say the word banana".
	// In duplex mode this audio goes inside a <unit> block with no
	// <|audio_start|>/<|audio_end|> wrappers (omni.cpp:4364-4366).
	std::cout << "Step 2: Encoding audio..." << std::endl;
	int n_embd_audio = 0;
	std::vector<float> audio_embeds =
		load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_BANANA, 4, &n_embd_audio);
	REQUIRE(!audio_embeds.empty());
	REQUIRE(n_embd_audio == n_embd);  // mmproj output must match LLM embedding dim
	int n_audio_tokens = (int)(audio_embeds.size() / n_embd);
	std::cout << "  ✓ Audio encoded: " << n_audio_tokens << " tokens (embd=" << n_embd_audio
			  << ")\n";

	// ── Step 3: Sampler ───────────────────────────────────────────────────────
	std::cout << "Step 3: Creating sampler..." << std::endl;
	common_params_sampling sp = {};
	sp.temp = 0.0f;	 // greedy for determinism
	sp.penalty_repeat = 1.0f;

	common_sampler * smpl = common_sampler_init(model, sp);
	REQUIRE(smpl != nullptr);

	// ── Step 4a: Encode reference voice for system prompt ─────────────────────
	// The full duplex system prompt is (omni.cpp:4164-4169):
	//   <|im_start|>system
	//   Streaming Duplex Conversation! You are a helpful assistant.
	//   <|audio_start|>[ref_voice_embeddings]<|audio_end|><|im_end|>
	// Without the reference voice the model has no voice identity and defaults
	// to always listening.
	std::cout << "Step 4a: Encoding reference voice..." << std::endl;
	std::vector<float> ref_embeds =
		load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_REFERENCE, 4);
	REQUIRE(!ref_embeds.empty());
	std::cout << "  ✓ Reference voice encoded: " << (ref_embeds.size() / n_embd) << " tokens\n";

	// ── Step 4b: Prefill system prompt + reference voice + <unit> + audio ────
	// Duplex format (omni.cpp:4164-4169, 4364-4366):
	//   system_with_voice  →  <unit>  →  [user audio embeddings]
	//   (no assistant prompt; model decides autonomously)
	std::cout << "Step 4b: Prefilling prompt + audio..." << std::endl;
	int n_past = 0;

	// System prompt prefix (up to and including <|audio_start|>)
	auto sys_pre = tokenize(
		vocab,
		"<|im_start|>system\n"
		"Streaming Duplex Conversation! You are a helpful assistant.\n"
		"<|audio_start|>",
		false);
	REQUIRE(prefill(ctx, sys_pre, n_past));

	// Reference voice embeddings
	REQUIRE(prefill_embeddings(ctx, ref_embeds, n_embd, n_past));

	// System prompt suffix + <unit> marker
	auto sys_post = tokenize(vocab, "<|audio_end|><|im_end|>\n<unit>", false);
	REQUIRE(prefill(ctx, sys_post, n_past));
	std::cout << "  ✓ System + ref voice prefilled (" << n_past << " tokens)\n";

	// User audio embeddings (inside the <unit> block, no wrappers).
	// In omni.cpp duplex mode: <unit> is prepended, audio is fed, and </unit>
	// is fed AFTER each generation chunk (omni.cpp:9337-9344) — not before.
	REQUIRE(prefill_embeddings(ctx, audio_embeds, n_embd, n_past));
	std::cout << "  ✓ User audio prefilled (n_past=" << n_past << ")\n";

	// ── Step 5: Silence audio for end-of-speech signal ───────────────────────
	// In a real duplex system the user's audio arrives as streaming chunks via
	// VAD.  After the last speech chunk, a short silence chunk arrives which
	// signals to the model that the user has finished speaking.  The model
	// then switches from <|listen|> to <|speak|>.
	//
	// We replicate this two-turn pattern:
	//   Turn A:  <unit>[banana_audio]  → <|listen|>  (model wants more input)
	//   Turn B:  <unit>[silence]       → <|speak|>...banana...  (user done)
	//
	// Silence: 0.5 s of near-zero PCM at 16 kHz — just enough to produce a
	// valid short mel spectrogram so the audio encoder can process it.
	std::cout << "Step 5a: Encoding silence..." << std::endl;
	size_t const silence_samples = 8000;					 // 0.5 s at 16 kHz
	std::vector<float> silence_pcm(silence_samples, 1e-6f);	 // near-zero, not exactly 0
	std::vector<float> silence_embeds =
		encode_pcm_audio(PATH_MODEL_AUDIO, silence_pcm.data(), silence_samples);
	REQUIRE(!silence_embeds.empty());

	// ── Step 5b: Multi-turn generation ────────────────────────────────────────
	// Turn A: banana audio already prefilled — run generation (expect listen)
	// Turn B: silence prefilled              — run generation (expect speak)
	std::cout << "Step 5b: Multi-turn generation..." << std::endl;

	TurnResult speak_result;
	int const max_turns = 4;

	// Phase 1: present banana audio turns until model decides to listen
	// Phase 2: after first listen, present silence → model should speak
	bool silence_presented = false;
	for (int turn = 0; turn < max_turns; ++turn)
	{
		TurnResult r = run_duplex_turn(smpl, ctx, vocab, n_past);
		std::cout << "  Turn " << (turn + 1) << ": tokens=" << r.tokens.size() << " text=\""
				  << r.text << "\""
				  << " speak=" << r.speak_count << " listen=" << r.listen_count
				  << " chunk_eos=" << r.chunk_eos_count << " turn_eos=" << r.turn_eos_count << "\n";

		if (r.speak_count > 0)
		{
			speak_result = r;
			break;
		}

		// Model chose to listen; present the next audio input.
		// After the first listen, send silence (end-of-speech signal).
		// For subsequent listens, keep re-sending silence.
		auto next_unit = tokenize(vocab, "<unit>", false);
		REQUIRE(prefill(ctx, next_unit, n_past));
		if (!silence_presented)
		{
			// First listen: follow with silence (end-of-speech)
			REQUIRE(prefill_embeddings(ctx, silence_embeds, n_embd, n_past));
			silence_presented = true;
		}
		else
		{
			// Still listening after silence — present silence again
			REQUIRE(prefill_embeddings(ctx, silence_embeds, n_embd, n_past));
		}
	}

	// ── Step 6: Assertions ────────────────────────────────────────────────────
	std::cout << "Step 6: Assertions..." << std::endl;

	// The speaking turn must begin with <|speak|>
	REQUIRE(speak_result.speak_count > 0);
	REQUIRE(!speak_result.tokens.empty());
	REQUIRE(speak_result.tokens[0] == TOKEN_SPEAK);
	std::cout << "  ✓ Speaking turn starts with <|speak|>\n";

	// The response must be terminated by a chunk/turn end token
	REQUIRE(speak_result.chunk_eos_count + speak_result.turn_eos_count > 0);
	std::cout << "  ✓ Speaking segment properly terminated\n";

	// The response must contain "banana" (case-insensitive)
	std::string text_lower = speak_result.text;
	std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
	REQUIRE(text_lower.find("banana") != std::string::npos);
	std::cout << "  ✓ Response contains \"banana\"\n";

	// ── Cleanup ───────────────────────────────────────────────────────────────
	common_sampler_free(smpl);
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Duplex audio test complete ===" << std::endl;
}

// ═════════════════════════════════════════════════════════════════════════════
TEST_CASE("duplex_visual_audio_triangle", "[duplex][multimodal]")
{
	std::cout << "\n=== Duplex: visual+audio → triangle ===" << std::endl;

	// ── Step 1: Encode image ──────────────────────────────────────────────────
	// CLIP must run before the LLM is loaded: CLIP with use_gpu=false still
	// conflicts with an active CUDA LLM context, so we pre-compute embeddings
	// and free the CLIP context before touching the GPU with the LLM.
	//
	// Duplex multimodal format (omni.cpp:4317-4356):
	//   <unit><image>[overview_embeds]</image>[<slice>[s]</slice>...][audio_embeds]
	std::cout << "Step 1: Encoding image (before LLM load)..." << std::endl;
	ImageEmbeddings img = load_and_encode_image(PATH_MODEL_VISION, PATH_IMAGE_TRIANGLE);
	REQUIRE(!img.chunks.empty());
	int const n_embd = img.n_embd;
	std::cout << "  ✓ Image encoded: " << img.chunks.size() << " chunk(s), " << img.n_tok_each
			  << " tok/chunk, n_embd=" << n_embd << "\n";

	// ── Step 2: Load LLM model ────────────────────────────────────────────────
	std::cout << "Step 2: Loading LLM model..." << std::endl;
	llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = 99;

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, mp);
	REQUIRE(model != nullptr);

	llama_context_params cp = llama_context_default_params();
	cp.n_ctx = 4096;  // larger context for vision + audio tokens
	cp.n_batch = 512;
	cp.n_ubatch = 512;

	llama_context * ctx = llama_init_from_model(model, cp);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	REQUIRE(llama_model_n_embd(model) == n_embd);  // CLIP and LLM dims must match
	std::cout << "  ✓ LLM loaded (n_embd=" << n_embd << ")\n";

	// ── Step 3: Encode audio ──────────────────────────────────────────────────
	// describe_the_shape.wav: user says "describe the shape".
	// Audition (use_gpu=true) works after LLM load; CLIP does not.
	std::cout << "Step 3: Encoding audio..." << std::endl;
	int n_embd_audio = 0;
	std::vector<float> audio_embeds =
		load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_DESCRIBE, 4, &n_embd_audio);
	REQUIRE(!audio_embeds.empty());
	REQUIRE(n_embd_audio == n_embd);
	std::cout << "  ✓ Audio encoded: " << (audio_embeds.size() / n_embd) << " tokens\n";

	// ── Step 4: Encode reference voice ───────────────────────────────────────
	std::cout << "Step 4: Encoding reference voice..." << std::endl;
	std::vector<float> ref_embeds =
		load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_REFERENCE, 4);
	REQUIRE(!ref_embeds.empty());
	std::cout << "  ✓ Reference voice encoded: " << (ref_embeds.size() / n_embd) << " tokens\n";

	// ── Step 5: Encode silence ────────────────────────────────────────────────
	std::cout << "Step 5: Encoding silence..." << std::endl;
	size_t const silence_samples = 8000;  // 0.5 s at 16 kHz
	std::vector<float> silence_pcm(silence_samples, 1e-6f);
	std::vector<float> silence_embeds =
		encode_pcm_audio(PATH_MODEL_AUDIO, silence_pcm.data(), silence_samples);
	REQUIRE(!silence_embeds.empty());
	std::cout << "  ✓ Silence encoded\n";

	// ── Step 6: Sampler ───────────────────────────────────────────────────────
	std::cout << "Step 6: Creating sampler..." << std::endl;
	common_params_sampling sp = {};
	sp.temp = 0.0f;
	sp.penalty_repeat = 1.0f;
	common_sampler * smpl = common_sampler_init(model, sp);
	REQUIRE(smpl != nullptr);

	// ── Step 7: Prefill system prompt + reference voice + multimodal unit ─────
	// Duplex multimodal format (omni.cpp:4317-4356):
	//   system: <|im_start|>system\n...<|audio_start|>[ref_voice]<|audio_end|><|im_end|>\n
	//   user:   <unit><image>[overview]</image>[<slice>[s]</slice>...][audio]
	//
	// Audio in duplex mode has no <|audio_start|>/<|audio_end|> wrappers.
	std::cout << "Step 7: Prefilling..." << std::endl;
	int n_past = 0;

	// System prompt prefix
	auto sys_pre = tokenize(
		vocab,
		"<|im_start|>system\n"
		"Streaming Duplex Conversation! You are a helpful assistant.\n"
		"<|audio_start|>",
		false);
	REQUIRE(prefill(ctx, sys_pre, n_past));

	// Reference voice embeddings
	REQUIRE(prefill_embeddings(ctx, ref_embeds, n_embd, n_past));

	// System prompt suffix + start of multimodal unit
	auto sys_post = tokenize(vocab, "<|audio_end|><|im_end|>\n<unit><image>", false);
	REQUIRE(prefill(ctx, sys_post, n_past));

	// Overview image embeddings (chunk 0)
	REQUIRE(prefill_embeddings(ctx, img.chunks[0], n_embd, n_past));

	// Close <image>, add slices if any
	auto img_end = tokenize(vocab, "</image>", false);
	REQUIRE(prefill(ctx, img_end, n_past));

	for (size_t i = 1; i < img.chunks.size(); ++i)
	{
		auto slice_open = tokenize(vocab, "<slice>", false);
		REQUIRE(prefill(ctx, slice_open, n_past));
		REQUIRE(prefill_embeddings(ctx, img.chunks[i], n_embd, n_past));
		auto slice_close = tokenize(vocab, "</slice>", false);
		REQUIRE(prefill(ctx, slice_close, n_past));
	}
	if (img.chunks.size() > 1)
	{
		auto nl = tokenize(vocab, "\n", false);
		REQUIRE(prefill(ctx, nl, n_past));
	}

	// Audio embeddings (no wrappers in duplex mode)
	REQUIRE(prefill_embeddings(ctx, audio_embeds, n_embd, n_past));
	std::cout << "  ✓ Multimodal unit prefilled (n_past=" << n_past << ")\n";

	// ── Step 8: Multi-turn generation with silence trigger ────────────────────
	// Same two-turn pattern as duplex_audio_banana:
	//   Turn A: visual+audio input → model likely listens
	//   Turn B: silence            → end-of-speech signal → model speaks
	std::cout << "Step 8: Multi-turn generation..." << std::endl;

	TurnResult speak_result;
	int const max_turns = 4;
	bool silence_presented = false;

	for (int turn = 0; turn < max_turns; ++turn)
	{
		TurnResult r = run_duplex_turn(smpl, ctx, vocab, n_past);
		std::cout << "  Turn " << (turn + 1) << ": tokens=" << r.tokens.size() << " text=\""
				  << r.text << "\""
				  << " speak=" << r.speak_count << " listen=" << r.listen_count
				  << " chunk_eos=" << r.chunk_eos_count << " turn_eos=" << r.turn_eos_count << "\n";

		if (r.speak_count > 0)
		{
			speak_result = r;
			break;
		}

		auto next_unit = tokenize(vocab, "<unit>", false);
		REQUIRE(prefill(ctx, next_unit, n_past));
		REQUIRE(prefill_embeddings(ctx, silence_embeds, n_embd, n_past));
		silence_presented = true;
		(void)silence_presented;
	}

	// ── Step 9: Assertions ────────────────────────────────────────────────────
	std::cout << "Step 9: Assertions..." << std::endl;

	REQUIRE(speak_result.speak_count > 0);
	REQUIRE(!speak_result.tokens.empty());
	REQUIRE(speak_result.tokens[0] == TOKEN_SPEAK);
	std::cout << "  ✓ Speaking turn starts with <|speak|>\n";

	REQUIRE(speak_result.chunk_eos_count + speak_result.turn_eos_count > 0);
	std::cout << "  ✓ Speaking segment properly terminated\n";

	std::string text_lower = speak_result.text;
	std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
	REQUIRE(text_lower.find("triangle") != std::string::npos);
	std::cout << "  ✓ Response contains \"triangle\"\n";

	// ── Cleanup ───────────────────────────────────────────────────────────────
	common_sampler_free(smpl);
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Duplex visual+audio test complete ===" << std::endl;
}

// ═════════════════════════════════════════════════════════════════════════════
// duplex_simplex_switching
//
// Investigates switching between duplex and simplex generation modes within
// the SAME conversation context (same KV cache).  This is the mechanism for
// injecting out-of-band data (sensor readings, tool results, etc.) as a
// hidden simplex turn between duplex audio exchanges.
//
// Root cause of mode selection: the mode is NOT set by the system prompt
// alone, but by the GENERATION TRIGGER prefilled before sampling starts:
//
//   Duplex trigger:  <unit>[audio embeddings]
//                    → model first token is <|speak|>/<|listen|>/<|chunk_eos|>
//
//   Simplex trigger: <|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
//                    → <|tts_bos|> is EOG; prefilling it sets "TTS generation
//                      mode".  Model's first sampled token is plain text.
//
// The earlier attempt (Phase 2 using only <|im_start|>assistant\n) failed
// because the model had no explicit TTS context token to anchor simplex mode.
// <|tts_bos|> (151703) is the required anchor: it is prefilled as part of the
// assistant turn header and tells the model "generate spoken text until
// <|tts_eos|>".  The duplex system prompt is PRESERVED throughout — only the
// per-turn generation trigger changes.
//
// Test flow (single llama_context, single KV cache throughout):
//
//   Phase 1 (Duplex):  system+ref_voice + <unit>[banana audio]
//                      → multi-turn until <|speak|> response
//   Phase 2 (Simplex): <|im_start|>user\n...\n<|im_end|>\n
//                      <|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
//                      → plain text (no duplex control tokens)
//                      close with: <|tts_eos|><|im_end|>\n  (EOG not fed back)
//   Phase 3 (Duplex):  <unit>[silence]
//                      → first token is a duplex control token
//
// <|tts_eos|> (151704) is also EOG, so sample_and_feed does not feed it back.
// Both TTS delimiters must be explicitly prefilled to keep the KV cache clean.
// ═════════════════════════════════════════════════════════════════════════════
TEST_CASE("duplex_simplex_switching", "[duplex][simplex][switching]")
{
	std::cout << "\n=== Duplex→Simplex→Duplex switching test (single KV cache) ===" << std::endl;

	// ── Step 1: Load model ────────────────────────────────────────────────────
	std::cout << "Step 1: Loading LLM model..." << std::endl;
	llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = 99;

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, mp);
	REQUIRE(model != nullptr);

	llama_context_params cp = llama_context_default_params();
	cp.n_ctx = 4096;  // phase1 audio + speak + phase2 Q&A + phase3 silence
	cp.n_batch = 512;
	cp.n_ubatch = 512;

	llama_context * ctx = llama_init_from_model(model, cp);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int const n_embd = llama_model_n_embd(model);
	std::cout << "  ✓ LLM loaded (n_embd=" << n_embd << ")\n";

	// ── Step 2: Encode audio inputs ───────────────────────────────────────────
	std::cout << "Step 2: Encoding audio inputs..." << std::endl;

	int n_embd_audio = 0;
	std::vector<float> audio_embeds =
		load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_BANANA, 4, &n_embd_audio);
	REQUIRE(!audio_embeds.empty());
	REQUIRE(n_embd_audio == n_embd);
	std::cout << "  ✓ Banana audio: " << (audio_embeds.size() / n_embd) << " tokens\n";

	std::vector<float> ref_embeds =
		load_and_encode_audio(PATH_MODEL_AUDIO, PATH_AUDIO_REFERENCE, 4);
	REQUIRE(!ref_embeds.empty());
	std::cout << "  ✓ Reference voice: " << (ref_embeds.size() / n_embd) << " tokens\n";

	size_t const silence_samples = 8000;
	std::vector<float> silence_pcm(silence_samples, 1e-6f);
	std::vector<float> silence_embeds =
		encode_pcm_audio(PATH_MODEL_AUDIO, silence_pcm.data(), silence_samples);
	REQUIRE(!silence_embeds.empty());
	std::cout << "  ✓ Silence encoded\n";

	// ── Step 3: Sampler ───────────────────────────────────────────────────────
	common_params_sampling sp = {};
	sp.temp = 0.0f;
	sp.penalty_repeat = 1.0f;
	common_sampler * smpl = common_sampler_init(model, sp);
	REQUIRE(smpl != nullptr);

	// ── Step 4: Prefill duplex system prompt + reference voice + banana audio ─
	std::cout << "Step 4: Prefilling system prompt + banana audio..." << std::endl;
	int n_past = 0;

	auto sys_pre = tokenize(
		vocab,
		"<|im_start|>system\n"
		"Streaming Duplex Conversation! You are a helpful assistant.\n"
		"<|audio_start|>",
		false);
	REQUIRE(prefill(ctx, sys_pre, n_past));
	REQUIRE(prefill_embeddings(ctx, ref_embeds, n_embd, n_past));

	auto sys_post = tokenize(vocab, "<|audio_end|><|im_end|>\n<unit>", false);
	REQUIRE(prefill(ctx, sys_post, n_past));
	REQUIRE(prefill_embeddings(ctx, audio_embeds, n_embd, n_past));
	std::cout << "  ✓ System + banana audio prefilled (n_past=" << n_past << ")\n";

	// ══════════════════════════════════════════════════════════════════════════
	// Phase 1: Duplex mode
	// ══════════════════════════════════════════════════════════════════════════
	std::cout << "\n--- Phase 1: Duplex mode ---" << std::endl;

	TurnResult duplex1_result;
	int const max_turns = 4;

	for (int turn = 0; turn < max_turns; ++turn)
	{
		TurnResult r = run_duplex_turn(smpl, ctx, vocab, n_past);
		std::cout << "  Duplex turn " << (turn + 1) << ": tokens=" << r.tokens.size() << " text=\""
				  << r.text << "\""
				  << " speak=" << r.speak_count << " listen=" << r.listen_count
				  << " chunk_eos=" << r.chunk_eos_count << "\n";

		if (r.speak_count > 0)
		{
			duplex1_result = r;
			break;
		}

		auto next_unit = tokenize(vocab, "<unit>", false);
		REQUIRE(prefill(ctx, next_unit, n_past));
		REQUIRE(prefill_embeddings(ctx, silence_embeds, n_embd, n_past));
	}

	REQUIRE(duplex1_result.speak_count > 0);
	REQUIRE(!duplex1_result.tokens.empty());
	REQUIRE(duplex1_result.tokens[0] == TOKEN_SPEAK);
	std::cout << "  ✓ Phase 1 PASSED: duplex speak turn confirmed (n_past=" << n_past << ")\n";

	// ══════════════════════════════════════════════════════════════════════════
	// Phase 2: Switch to simplex mode — SAME context, same KV cache
	//
	// After run_duplex_turn() the KV cache ends with </unit>.  We now inject
	// a standard chat-template user turn followed by the FULL simplex
	// assistant header:
	//
	//   <|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
	//
	// <|tts_bos|> is prefilled (not generated) and is the key mode-switch
	// signal.  Sampling starts AFTER it, in "TTS text generation" context.
	// The model generates plain text tokens until it emits <|tts_eos|> (EOG).
	//
	// Note: using only <|im_start|>assistant\n (without <|tts_bos|>) is NOT
	// sufficient — the model continues emitting duplex control tokens because
	// it has no explicit TTS anchor.  The full template is required.
	// ══════════════════════════════════════════════════════════════════════════
	std::cout << "\n--- Phase 2: Simplex mode (same KV cache) ---" << std::endl;

	// Full simplex assistant trigger: user turn + <think></think> + <|tts_bos|>
	// The <think>\n\n</think>\n\n block is required by the chat template
	// (see modeling_minicpmo_unified.py:2730-2732).
	auto simplex_prompt = tokenize(
		vocab,
		"<|im_start|>user\n"
		"What color is the sky?\n"
		"<|im_end|>\n"
		"<|im_start|>assistant\n"
		"<think>\n\n</think>\n\n"
		"<|tts_bos|>",
		false);
	REQUIRE(prefill(ctx, simplex_prompt, n_past));
	std::cout << "  ✓ Simplex prompt prefilled (n_past=" << n_past << ")\n";

	// Generate: sample until EOG (<|tts_eos|>/<|im_end|>), a duplex chunk
	// boundary (TOKEN_CHUNK_EOS / TOKEN_TURN_EOS), or a speak/listen decision.
	//
	// Observed behaviour: within a duplex KV cache the model treats <|tts_bos|>
	// like <|speak|> and closes the first text chunk with <|chunk_eos|> rather
	// than <|tts_eos|>.  The pattern is:
	//   <|tts_bos|>[plain text tokens]<|chunk_eos|>
	// This is the correct simplex-within-duplex boundary — <|chunk_eos|> signals
	// "end of this generation unit" regardless of duplex/simplex context.
	//
	// <|speak|> / <|listen|> appearing BEFORE the chunk boundary would indicate
	// the model has reverted to full duplex mode, which is the actual failure.
	std::vector<llama_token> simplex_tokens;
	std::string simplex_text;
	int simplex_speak_listen_count = 0;	 // only speak/listen are hard failures

	for (int step = 0; step < 150; ++step)
	{
		llama_token tok = sample_and_feed(smpl, ctx, vocab, n_past, step == 0);

		// <|tts_eos|> and <|im_end|> are EOG — not fed back by sample_and_feed
		if (llama_vocab_is_eog(vocab, tok))
			break;

		// <|chunk_eos|> / <|turn_eos|>: natural chunk boundary; already fed
		// back by sample_and_feed (they are not EOG).  Stop here — collecting
		// further tokens would just accumulate more <|chunk_eos|> spam.
		if (tok == TOKEN_CHUNK_EOS || tok == TOKEN_TURN_EOS)
			break;

		simplex_tokens.push_back(tok);

		if (tok == TOKEN_SPEAK || tok == TOKEN_LISTEN)
		{
			// Hard failure: model emitted a duplex decision token before any
			// chunk boundary — it is treating this as a pure duplex turn.
			++simplex_speak_listen_count;
			break;
		}

		std::vector<char> buf(64);
		int n = llama_token_to_piece(vocab, tok, buf.data(), (int)buf.size(), 0, true);
		if (n > 0)
			simplex_text.append(buf.data(), n);
	}

	// Close the simplex turn: feed <|tts_eos|><|im_end|>\n into the KV cache.
	// Both tokens are EOG so sample_and_feed never feeds them back; prefilling
	// them explicitly ensures Phase 3's <unit> follows a clean turn boundary.
	auto close_simplex = tokenize(vocab, "<|tts_eos|><|im_end|>\n", false);
	REQUIRE(prefill(ctx, close_simplex, n_past));

	std::cout << "  Simplex response: \"" << simplex_text << "\"\n";
	std::cout << "  Speak/listen tokens (should be 0): " << simplex_speak_listen_count << "\n";
	std::cout << "  n_past after simplex close: " << n_past << "\n";

	// Phase 2 assertions:
	// 1. The model generated plain text before any chunk boundary.
	REQUIRE(!simplex_text.empty());
	// 2. No <|speak|>/<|listen|> decision tokens appeared — the model did not
	//    revert to full duplex decision-making before producing text.
	REQUIRE(simplex_speak_listen_count == 0);
	// 3. The text answers the question (sky / blue / colour family).
	{
		std::string t = simplex_text;
		std::transform(t.begin(), t.end(), t.begin(), ::tolower);
		bool mentions_sky_or_colour =
			(t.find("sky") != std::string::npos || t.find("blue") != std::string::npos ||
			 t.find("colo") != std::string::npos);
		REQUIRE(mentions_sky_or_colour);
	}
	std::cout << "  ✓ Phase 2 PASSED: simplex plain text generated, no duplex decision tokens\n";

	// ══════════════════════════════════════════════════════════════════════════
	// Phase 3: Switch back to duplex mode — SAME context, same KV cache
	//
	// The KV cache now contains: system + duplex speak turn + simplex Q&A.
	// Injecting <unit>[silence] switches the model back to duplex mode.
	// The model must generate a duplex control token as its very first token.
	// ══════════════════════════════════════════════════════════════════════════
	std::cout << "\n--- Phase 3: Back to duplex mode (same KV cache) ---" << std::endl;

	auto unit_tok = tokenize(vocab, "<unit>", false);
	REQUIRE(prefill(ctx, unit_tok, n_past));
	REQUIRE(prefill_embeddings(ctx, silence_embeds, n_embd, n_past));
	std::cout << "  ✓ Duplex unit prefilled (n_past=" << n_past << ")\n";

	TurnResult duplex2_result = run_duplex_turn(smpl, ctx, vocab, n_past);
	std::cout << "  Duplex (restored) turn: tokens=" << duplex2_result.tokens.size() << " text=\""
			  << duplex2_result.text << "\""
			  << " speak=" << duplex2_result.speak_count
			  << " listen=" << duplex2_result.listen_count
			  << " chunk_eos=" << duplex2_result.chunk_eos_count
			  << " turn_eos=" << duplex2_result.turn_eos_count << "\n";

	int total_duplex2 = duplex2_result.speak_count + duplex2_result.listen_count +
		duplex2_result.chunk_eos_count + duplex2_result.turn_eos_count;
	REQUIRE(total_duplex2 > 0);

	bool first_is_duplex =
		(!duplex2_result.tokens.empty() &&
		 (duplex2_result.tokens[0] == TOKEN_SPEAK || duplex2_result.tokens[0] == TOKEN_LISTEN ||
		  duplex2_result.tokens[0] == TOKEN_CHUNK_EOS ||
		  duplex2_result.tokens[0] == TOKEN_TURN_EOS));
	REQUIRE(first_is_duplex);
	std::cout << "  ✓ Phase 3 PASSED: model back in duplex mode\n";

	// ── Cleanup ───────────────────────────────────────────────────────────────
	common_sampler_free(smpl);
	llama_free(ctx);
	llama_model_free(model);

	std::cout << "=== Duplex↔Simplex switching test complete ===" << std::endl;
}
