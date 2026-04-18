/*
 * Duplex TTS Test for MiniCPM-o 4.5
 *
 * Tests the TTS pipeline as it operates in duplex mode, where the model's
 * text response is interleaved with duplex special tokens.
 *
 * In real duplex operation the LLM generates tokens like:
 *
 *   <|speak|> The quick brown fox ... <|chunk_eos|>
 *
 * Those special tokens are then stripped by filter_special_tokens, and the
 * remaining text tokens — with their LLM hidden states — are used to
 * condition the TTS model.
 *
 * This test replicates that exact pipeline:
 *
 * 1.  Load LLM
 * 2.  Build a duplex-style prompt:
 *       system (duplex) + user + assistant\n + <|speak|> + text + <|chunk_eos|>
 *     The text is pre-supplied (matching what the model would generate) so
 *     we avoid autoregressive sampling instability with this specialised model.
 * 3.  Decode the full sequence in one batch
 * 4.  Extract hidden states for the assistant response section
 *     (speak token + text tokens + chunk_eos token)
 * 5.  filter_special_tokens removes <|speak|> and <|chunk_eos|>, keeping only
 *     text tokens — verifies duplex marker filtering works correctly
 * 6.  Load projector + TTS weights
 * 7.  Apply projector (4096→768), L2-normalise
 * 8.  Build TTS condition:
 *       condition[i] = emb_text(token[i]) + projected[i]
 *     then append emb_text(text_eos=151692) and emb_text(audio_bos=151687)
 * 9.  Clear TTS KV cache, prefill condition
 * 10. Sample audio tokens (top_k=25, top_p=0.85, temp=0.8, multinomial)
 * 11. Feed each sampled token back via emb_code
 * 12. Convert audio tokens → waveform via generate_audio_from_tokens
 * 13. Write WAV, assert duration >= 1 second
 *
 * Duplex special token IDs (verified from GGUF control-token listing):
 *   <|speak|>     = 151706
 *   <|listen|>    = 151705
 *   <|chunk_eos|> = 151718
 *   <|turn_eos|>  = 151717
 *
 * References:
 * - llama.cpp-omni/tools/omni/omni.cpp  (duplex TTS implementation)
 * - test_tts.cpp                         (simplex baseline)
 * - modeling_minicpmo.py:1299-1373      (_generate_speech_non_streaming)
 */

#include <catch2/catch_test_macros.hpp>

#include "common/sampling.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "omni/omni-tts.h"
#include "omni/token2wav-impl.h"

#include "omni/audition.h"

#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

#include "vendor_paths.hpp"

// Paths — resolved from LLAMAOMNISERVER_MODEL_ROOT env var (the GGUF directory).
static VendorPath const PATH_MODEL_LLM{vp_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf"};
static VendorPath const PATH_MODEL_PROJECTOR{
	vp_model_root() + "/tts/MiniCPM-o-4_5-projector-F16.gguf"};
static VendorPath const PATH_MODEL_TTS_TRANSFORMER{
	vp_model_root() + "/tts/MiniCPM-o-4_5-tts-transformer-F16.gguf"};
static VendorPath const PATH_MODEL_TTS_WEIGHTS{
	vp_model_root() + "/tts/MiniCPM-o-4_5-tts-weights-F16.gguf"};
static VendorPath const PATH_DIR_TOKEN2WAV{vp_model_root() + "/token2wav-gguf"};

// TTS special tokens (from tts_config.json)
constexpr llama_token TOKEN_AUDIO_BOS = 151687;
constexpr llama_token TOKEN_TEXT_EOS = 151692;

// Duplex special tokens (verified from GGUF control-token output)
constexpr llama_token TOKEN_SPEAK = 151706;		 // <|speak|>
constexpr llama_token TOKEN_LISTEN = 151705;	 // <|listen|>
constexpr llama_token TOKEN_CHUNK_EOS = 151718;	 // <|chunk_eos|>
constexpr llama_token TOKEN_TURN_EOS = 151717;	 // <|turn_eos|>

constexpr int SAMPLE_RATE = 24000;

// Additional paths for voice-clone test
static VendorPath const PATH_MODEL_AUDIO{vp_model_root() + "/audio/MiniCPM-o-4_5-audio-F16.gguf"};
static VendorPath const PATH_AUDIO_VOICE_REF{vp_test_data() + "/alternative.wav"};
static VendorPath const PATH_AUDIO_STORY_IN{vp_test_data() + "/tell_me_a_story.wav"};
static VendorPath const PATH_AUDIO_STOP{vp_test_data() + "/stop.wav"};
// Token2Wav-level voice clone: prompt cache built from the same reference audio
// (gen_prompt_cache.py --n-timesteps 5  alternative.wav
// alternative_prompt_cache.gguf)
static VendorPath const PATH_ALTERNATIVE_PROMPT_CACHE{
	vp_model_root() + "/token2wav-gguf/alternative_prompt_cache.gguf"};

static int write_wav_mono(std::string_view path, std::vector<float> const & pcm, int sample_rate)
{
	constexpr int16_t num_channels = 1;
	constexpr int16_t bits_per_sample = 16;
	constexpr int16_t block_align = num_channels * (bits_per_sample / 8);
	int32_t const byte_rate = sample_rate * block_align;

	std::vector<int16_t> pcm16(pcm.size());
	for (size_t i = 0; i < pcm.size(); ++i)
	{
		float x = std::max(-1.0f, std::min(1.0f, pcm[i]));
		pcm16[i] = static_cast<int16_t>(x * 32767.0f);
	}
	uint32_t const data_bytes = static_cast<uint32_t>(pcm16.size() * sizeof(int16_t));
	uint32_t const file_size = 36 + data_bytes;

	std::ofstream f(path.data(), std::ios::binary);
	if (!f)
		return -1;
	f.write("RIFF", 4);
	f.write(reinterpret_cast<char const *>(&file_size), 4);
	f.write("WAVE", 4);
	f.write("fmt ", 4);
	uint32_t fmt_size = 16;
	uint16_t fmt_tag = 1;
	f.write(reinterpret_cast<char const *>(&fmt_size), 4);
	f.write(reinterpret_cast<char const *>(&fmt_tag), 2);
	f.write(reinterpret_cast<char const *>(&num_channels), 2);
	f.write(reinterpret_cast<char const *>(&sample_rate), 4);
	f.write(reinterpret_cast<char const *>(&byte_rate), 4);
	f.write(reinterpret_cast<char const *>(&block_align), 2);
	f.write(reinterpret_cast<char const *>(&bits_per_sample), 2);
	f.write("data", 4);
	f.write(reinterpret_cast<char const *>(&data_bytes), 4);
	f.write(reinterpret_cast<char const *>(pcm16.data()), data_bytes);
	return static_cast<int>(pcm16.size());
}

// ── Helper: load WAV + preprocess + encode via audition ───────────────────────
// Returns [n_tokens * n_mmproj_embd] floats, or empty on failure.
// If out_n_embd is non-null it is set to audition_n_mmproj_embd().
static std::vector<float> load_and_encode_audio_vc(
	char const * audio_model_path,
	char const * audio_path,
	int n_threads = 4,
	int * out_n_embd = nullptr)
{
	audition_context_params ap = {/*use_gpu=*/true, GGML_LOG_LEVEL_WARN};
	audition_ctx * actx = audition_init(audio_model_path, ap);
	if (!actx)
		return {};

	SF_INFO sfinfo = {};
	SNDFILE * sf = sf_open(audio_path, SFM_READ, &sfinfo);
	if (!sf)
	{
		audition_free(actx);
		return {};
	}

	std::vector<float> raw(sfinfo.frames * sfinfo.channels);
	sf_readf_float(sf, raw.data(), sfinfo.frames);
	sf_close(sf);

	std::vector<float> mono(sfinfo.frames);
	for (sf_count_t i = 0; i < sfinfo.frames; ++i) mono[i] = raw[i * sfinfo.channels];

	auto filters = audition_get_mel_filters(actx);
	std::vector<whisper_preprocessor::whisper_mel> mel;
	whisper_preprocessor::preprocess_audio(mono.data(), mono.size(), filters, mel);
	if (mel.empty())
	{
		audition_free(actx);
		return {};
	}

	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel[0].n_len;
	mel_f32->ny = mel[0].n_mel;
	mel_f32->buf.assign(mel[0].data.begin(), mel[0].data.end());

	int n_tokens = audition_n_output_tokens(actx, mel_f32);
	int n_embd_aud = audition_n_mmproj_embd(actx);
	if (out_n_embd)
		*out_n_embd = n_embd_aud;

	std::vector<float> embeddings((n_tokens + 2) * n_embd_aud);
	bool ok = audition_audio_encode(actx, n_threads, mel_f32, embeddings.data());
	audition_audio_f32_free(mel_f32);
	audition_free(actx);
	if (!ok)
		return {};
	embeddings.resize(n_tokens * n_embd_aud);
	return embeddings;
}

// ── Helper: encode near-silence (0.5 s) as a duplex end-of-speech signal ──────
static std::vector<float> encode_silence_vc(
	char const * audio_model_path, size_t n_pcm_samples = 8000, int n_threads = 4)
{
	audition_context_params ap = {/*use_gpu=*/true, GGML_LOG_LEVEL_WARN};
	audition_ctx * actx = audition_init(audio_model_path, ap);
	if (!actx)
		return {};

	std::vector<float> pcm(n_pcm_samples, 1e-6f);
	auto filters = audition_get_mel_filters(actx);
	std::vector<whisper_preprocessor::whisper_mel> mel;
	whisper_preprocessor::preprocess_audio(pcm.data(), n_pcm_samples, filters, mel);
	if (mel.empty())
	{
		audition_free(actx);
		return {};
	}

	audition_audio_f32 * mel_f32 = audition_audio_f32_init();
	mel_f32->nx = mel[0].n_len;
	mel_f32->ny = mel[0].n_mel;
	mel_f32->buf.assign(mel[0].data.begin(), mel[0].data.end());

	int n_tokens = audition_n_output_tokens(actx, mel_f32);
	int n_embd_aud = audition_n_mmproj_embd(actx);

	std::vector<float> embeddings((n_tokens + 2) * n_embd_aud);
	bool ok = audition_audio_encode(actx, n_threads, mel_f32, embeddings.data());
	audition_audio_f32_free(mel_f32);
	audition_free(actx);
	if (!ok)
		return {};
	embeddings.resize(n_tokens * n_embd_aud);
	return embeddings;
}

// ── Helper: prefill a block of audio embedding vectors into an LLM context ───
// batch.logits[last] = 1 so the next sample() call sees valid logits.
static bool prefill_audio_vc(
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
		b.n_seq_id[j] = 1;
		b.seq_id[j][0] = 0;
		b.logits[j] = (j == n_tok - 1) ? 1 : 0;
	}
	bool ok = (llama_decode(ctx, b) == 0);
	llama_batch_free(b);
	if (ok)
		n_past += n_tok;
	return ok;
}

// ── Helper: tokenize text and prefill into a context ─────────────────────────
static bool prefill_text_vc(
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
			b.n_seq_id[j] = 1;
			b.seq_id[j][0] = 0;
			b.logits[j] = (j == chunk - 1) ? 1 : 0;
		}
		bool ok = (llama_decode(ctx, b) == 0);
		llama_batch_free(b);
		if (!ok)
			return false;
		n_past += chunk;
	}
	return true;
}

TEST_CASE("duplex_tts", "[tts][duplex]")
{
	std::cout << "\n=== Duplex TTS Test ===" << std::endl;

	// -------------------------------------------------------------------------
	// Step 1: Load LLM
	// -------------------------------------------------------------------------
	std::cout << "\nStep 1: Loading LLM..." << std::endl;

	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 0;

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 8192;
	ctx_params.n_ubatch = 8192;
	ctx_params.embeddings = true;

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t const n_embd = llama_model_n_embd(model);

	std::cout << "  ✓ LLM loaded (n_embd=" << n_embd << ")" << std::endl;

	// -------------------------------------------------------------------------
	// Step 2: Build duplex-style prompt
	//
	// In real duplex operation the model generates:
	//   <|speak|> The quick brown fox ... <|chunk_eos|>
	//
	// We pre-supply the text wrapped with those markers so the test is
	// deterministic.  filter_special_tokens will strip them in Step 4.
	// -------------------------------------------------------------------------
	std::cout << "\nStep 2: Building duplex prompt..." << std::endl;

	// parse_special=true is needed to tokenise <|speak|> and <|chunk_eos|>
	// as their single special-token IDs rather than as character sequences.
	auto const tokenize =
		[&](std::string_view text, std::vector<llama_token> & out, bool parse_special = false)
	{
		int32_t n = llama_tokenize(
			vocab,
			text.data(),
			static_cast<int32_t>(text.size()),
			out.data(),
			static_cast<int32_t>(out.size()),
			/*add_special=*/false,
			parse_special);
		if (n < 0)
		{
			out.resize(-n);
			n = llama_tokenize(
				vocab,
				text.data(),
				static_cast<int32_t>(text.size()),
				out.data(),
				static_cast<int32_t>(out.size()),
				false,
				parse_special);
		}
		else
		{
			out.resize(n);
		}
	};

	// Duplex system prompt (matching omni.cpp duplex mode)
	std::string const system_prompt =
		"<|im_start|>system\n"
		"Streaming Duplex Conversation! You are a helpful assistant."
		"<|im_end|>\n";
	std::string const user_prompt = "<|im_start|>user\nSay this sentence.<|im_end|>\n";
	std::string const assistant_start = "<|im_start|>assistant\n";

	std::string const text_to_synthesize = "The quick brown fox jumps over the lazy banana";

	std::vector<llama_token> prefix_tokens;	 // system + user + assistant\n
	tokenize(system_prompt + user_prompt + assistant_start, prefix_tokens);

	std::vector<llama_token> speak_tokens;	// <|speak|>
	tokenize("<|speak|>", speak_tokens, /*parse_special=*/true);
	REQUIRE(speak_tokens.size() == 1);
	REQUIRE(speak_tokens[0] == TOKEN_SPEAK);

	std::vector<llama_token> text_tokens;  // actual text to synthesize
	tokenize(text_to_synthesize, text_tokens);

	std::vector<llama_token> chunk_eos_tokens;	// <|chunk_eos|>
	tokenize("<|chunk_eos|>", chunk_eos_tokens, /*parse_special=*/true);
	REQUIRE(chunk_eos_tokens.size() == 1);
	REQUIRE(chunk_eos_tokens[0] == TOKEN_CHUNK_EOS);

	// Full sequence: prefix + <|speak|> + text + <|chunk_eos|>
	std::vector<llama_token> prompt_tokens;
	prompt_tokens.insert(prompt_tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
	prompt_tokens.insert(prompt_tokens.end(), speak_tokens.begin(), speak_tokens.end());
	prompt_tokens.insert(prompt_tokens.end(), text_tokens.begin(), text_tokens.end());
	prompt_tokens.insert(prompt_tokens.end(), chunk_eos_tokens.begin(), chunk_eos_tokens.end());

	std::cout << "  ✓ prefix=" << prefix_tokens.size() << " speak=" << speak_tokens.size()
			  << " text=" << text_tokens.size() << " chunk_eos=" << chunk_eos_tokens.size()
			  << " total=" << prompt_tokens.size() << std::endl;

	// -------------------------------------------------------------------------
	// Step 3: Decode full sequence, extract hidden states for response section
	//
	// We decode the entire prompt at once and extract hidden states for every
	// token in the assistant response (speak + text + chunk_eos).
	// -------------------------------------------------------------------------
	std::cout << "\nStep 3: Decoding sequence and extracting hidden states..." << std::endl;

	{
		std::vector<llama_pos> pos(prompt_tokens.size());
		std::iota(pos.begin(), pos.end(), 0);
		llama_batch b = {};
		b.n_tokens = static_cast<int32_t>(prompt_tokens.size());
		b.token = prompt_tokens.data();
		b.pos = pos.data();
		llama_set_embeddings(ctx, true);
		if (llama_decode(ctx, b))
			FAIL("Failed to decode prompt");
	}

	// Response section: speak_tokens + text_tokens + chunk_eos_tokens
	size_t const response_start = prefix_tokens.size();
	size_t const response_end = prompt_tokens.size();

	std::vector<llama_token> response_tokens;
	std::vector<float> response_hidden;
	response_tokens.reserve(response_end - response_start);

	for (size_t i = response_start; i < response_end; ++i)
	{
		response_tokens.push_back(prompt_tokens[i]);
		float const * emb = llama_get_embeddings_ith(ctx, static_cast<int32_t>(i));
		if (!emb)
			FAIL("Missing embedding for response token");
		for (int j = 0; j < n_embd; ++j) response_hidden.push_back(emb[j]);
	}

	std::cout << "  ✓ Extracted " << response_tokens.size() << " response tokens" << std::endl;

	// -------------------------------------------------------------------------
	// Step 4: Filter duplex special tokens
	//
	// filter_special_tokens removes <|speak|> (151706) and <|chunk_eos|> (151718)
	// because both are >= 150000.  Only the plain text tokens remain.
	// -------------------------------------------------------------------------
	std::cout << "\nStep 4: Filtering duplex special tokens..." << std::endl;

	omni::tts::filter_special_tokens(response_tokens, response_hidden, n_embd);

	std::cout << "  ✓ " << response_tokens.size() << " text tokens after filtering"
			  << " (expected " << text_tokens.size() << ")" << std::endl;

	// Duplex markers must have been stripped, text tokens must remain
	REQUIRE(response_tokens.size() == text_tokens.size());
	REQUIRE(response_tokens == text_tokens);

	// Verify content
	auto detokenize = [&](std::vector<llama_token> const & toks)
	{
		std::string out;
		std::vector<char> buf(32);
		for (llama_token t : toks)
		{
			int n = llama_token_to_piece(
				vocab, t, buf.data(), static_cast<int32_t>(buf.size()), 0, false);
			if (n > 0)
				out.append(buf.data(), n);
		}
		return out;
	};
	std::string filtered_text = detokenize(response_tokens);
	std::cout << "  ✓ Filtered text: \"" << filtered_text << "\"" << std::endl;
	REQUIRE(filtered_text.find("banana") != std::string::npos);

	// -------------------------------------------------------------------------
	// Step 5: Load TTS weights (projector, emb_code, emb_text, head_code, Token2Wav)
	// -------------------------------------------------------------------------
	std::cout << "\nStep 5: Loading TTS weights..." << std::endl;

	omni::tts::TTSContext tts_ctx;
	bool ok = omni::tts::load_tts_weights(
		tts_ctx,
		PATH_MODEL_PROJECTOR,
		PATH_MODEL_TTS_TRANSFORMER,
		PATH_MODEL_TTS_WEIGHTS,
		PATH_DIR_TOKEN2WAV);
	REQUIRE(ok);
	std::cout << "  ✓ TTS weights loaded" << std::endl;

	// -------------------------------------------------------------------------
	// Step 6: Load TTS transformer model and context
	// -------------------------------------------------------------------------
	std::cout << "\nStep 6: Loading TTS transformer..." << std::endl;

	llama_model_params tts_mparams = llama_model_default_params();
	tts_mparams.n_gpu_layers = 0;

	llama_model * model_tts = llama_model_load_from_file(PATH_MODEL_TTS_TRANSFORMER, tts_mparams);
	REQUIRE(model_tts != nullptr);

	llama_context_params tts_cparams = llama_context_default_params();
	tts_cparams.n_ctx = 4096;
	tts_cparams.n_batch = 4096;
	tts_cparams.n_ubatch = 4096;
	tts_cparams.embeddings = true;

	llama_context * ctx_tts = llama_init_from_model(model_tts, tts_cparams);
	REQUIRE(ctx_tts != nullptr);
	std::cout << "  ✓ TTS transformer loaded" << std::endl;

	// -------------------------------------------------------------------------
	// Step 7: Apply projector and build TTS condition embeddings
	//
	// condition[i] = emb_text(token[i]) + L2_norm(projector(hidden[i]))
	// then append emb_text(text_eos=151692) and emb_text(audio_bos=151687)
	// (modeling_minicpmo.py:1350-1357)
	// -------------------------------------------------------------------------
	std::cout << "\nStep 7: Building TTS condition embeddings..." << std::endl;

	int const n_text = static_cast<int>(response_tokens.size());

	// Project 4096-d → 768-d
	std::vector<float> projected;
	omni::tts::apply_projector_semantic(
		tts_ctx.projector_weights, response_hidden, n_text, projected);
	REQUIRE(static_cast<int>(projected.size()) == n_text * 768);

	// L2-normalise
	omni::tts::normalize_l2_per_token(projected.data(), n_text, 768);

	// Merge: condition[i] = emb_text(token[i]) + projected[i]
	std::vector<float> condition(n_text * 768);
	for (int i = 0; i < n_text; ++i)
	{
		std::vector<float> te(768);
		omni::tts::tts_emb_text(tts_ctx, response_tokens[i], te.data(), 768);
		for (int j = 0; j < 768; ++j) condition[i * 768 + j] = te[j] + projected[i * 768 + j];
	}

	// Append text_eos embedding
	{
		std::vector<float> te(768);
		omni::tts::tts_emb_text(tts_ctx, TOKEN_TEXT_EOS, te.data(), 768);
		condition.insert(condition.end(), te.begin(), te.end());
	}

	// Append audio_bos embedding
	{
		std::vector<float> te(768);
		omni::tts::tts_emb_text(tts_ctx, TOKEN_AUDIO_BOS, te.data(), 768);
		condition.insert(condition.end(), te.begin(), te.end());
	}

	int const n_cond = static_cast<int>(condition.size()) / 768;
	std::cout << "  ✓ Condition: " << n_text << " text + text_eos + audio_bos = " << n_cond
			  << " tokens" << std::endl;

	// -------------------------------------------------------------------------
	// Step 8: Clear TTS KV cache, prefill condition
	// -------------------------------------------------------------------------
	std::cout << "\nStep 8: Prefilling TTS condition..." << std::endl;

	{
		llama_memory_t mem = llama_get_memory(ctx_tts);
		if (mem)
			llama_memory_seq_rm(mem, 0, 0, -1);
	}

	int n_past_tts = 0;
	{
		int const n_batch_tts = 4096;
		llama_set_embeddings(ctx_tts, true);
		for (int i = 0; i < n_cond; i += n_batch_tts)
		{
			int n_eval = std::min(n_batch_tts, n_cond - i);
			llama_batch cb = {};
			cb.n_tokens = n_eval;
			cb.embd = condition.data() + i * 768;
			std::vector<llama_pos> pv(n_eval);
			std::iota(pv.begin(), pv.end(), n_past_tts);
			cb.pos = pv.data();
			if (llama_decode(ctx_tts, cb))
				FAIL("Failed to prefill condition");
			n_past_tts += n_eval;
		}
	}
	std::cout << "  ✓ Condition prefilled, n_past_tts=" << n_past_tts << std::endl;

	// -------------------------------------------------------------------------
	// Step 9: Sample audio tokens
	//
	// Same pipeline as test_tts.cpp (fixed):
	//   head_code matmul → temperature → rep-penalty → top_k → top_p
	//   → multinomial → emb_code feedback
	// -------------------------------------------------------------------------
	std::cout << "\nStep 9: Sampling audio tokens..." << std::endl;

	struct common_params_sampling tts_sp = {};
	tts_sp.temp = 0.8f;
	tts_sp.top_k = 25;
	tts_sp.top_p = 0.85f;
	tts_sp.penalty_repeat = 1.05f;
	tts_sp.penalty_last_n = 16;

	int const num_audio_tokens = 6562;
	int const eos_rel_idx = num_audio_tokens - 1;
	int const max_audio_tokens = 500;

	std::vector<int32_t> audio_tokens;
	std::vector<int> generated_relative;

	static std::mt19937 rng{std::random_device{}()};

	for (int t = 0; t < max_audio_tokens; ++t)
	{
		float const * hs = llama_get_embeddings_ith(ctx_tts, -1);
		REQUIRE(hs != nullptr);

		// head_code matmul
		std::vector<float> logits(num_audio_tokens, 0.0f);
		float const * hcw = tts_ctx.head_code_weight;
		int const hid = tts_ctx.head_code_hidden_size;
		for (int i = 0; i < num_audio_tokens; ++i)
		{
			float const * row = hcw + i * hid;
			float s = 0.0f;
			for (int j = 0; j < hid; ++j) s += hs[j] * row[j];
			logits[i] = s;
		}

		// Temperature
		for (float & l : logits) l /= tts_sp.temp;

		// Repetition penalty
		if (!generated_relative.empty())
		{
			int start = std::max(0, (int)generated_relative.size() - tts_sp.penalty_last_n);
			std::vector<bool> seen(num_audio_tokens, false);
			for (int k = start; k < (int)generated_relative.size(); ++k)
			{
				int id = generated_relative[k];
				if (id >= 0 && id < num_audio_tokens)
					seen[id] = true;
			}
			for (int i = 0; i < num_audio_tokens; ++i)
			{
				if (!seen[i])
					continue;
				if (logits[i] >= 0)
					logits[i] /= tts_sp.penalty_repeat;
				else
					logits[i] *= tts_sp.penalty_repeat;
			}
		}

		// Softmax
		float max_l = *std::max_element(logits.begin(), logits.end());
		std::vector<float> probs(num_audio_tokens);
		float sum = 0.0f;
		for (int i = 0; i < num_audio_tokens; ++i)
		{
			probs[i] = std::exp(logits[i] - max_l);
			sum += probs[i];
		}
		for (float & p : probs) p /= sum;

		// top_k
		if (tts_sp.top_k > 0 && tts_sp.top_k < num_audio_tokens)
		{
			std::vector<float> sp = probs;
			std::nth_element(sp.begin(), sp.begin() + (num_audio_tokens - tts_sp.top_k), sp.end());
			float thr = sp[num_audio_tokens - tts_sp.top_k];
			for (float & p : probs)
				if (p < thr)
					p = 0.0f;
		}

		// top_p
		if (tts_sp.top_p < 1.0f)
		{
			std::vector<int> idx(num_audio_tokens);
			std::iota(idx.begin(), idx.end(), 0);
			std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
			float s2 = 0.0f;
			for (float p : probs) s2 += p;
			float cum = 0.0f;
			bool cut = false;
			for (int id : idx)
			{
				if (cut)
				{
					probs[id] = 0.0f;
					continue;
				}
				cum += probs[id] / s2;
				if (cum >= tts_sp.top_p)
					cut = true;
			}
		}

		// Re-normalise
		{
			float s2 = 0.0f;
			for (float p : probs) s2 += p;
			if (s2 > 0.0f)
				for (float & p : probs) p /= s2;
		}

		// Multinomial sample
		int sel = eos_rel_idx;
		{
			std::uniform_real_distribution<float> dist(0.0f, 1.0f);
			float r = dist(rng), cum = 0.0f;
			for (int i = 0; i < num_audio_tokens; ++i)
			{
				cum += probs[i];
				if (r <= cum)
				{
					sel = i;
					break;
				}
			}
		}

		if (sel == eos_rel_idx)
		{
			std::cout << "  EOS at t=" << t << std::endl;
			break;
		}

		audio_tokens.push_back(TOKEN_AUDIO_BOS + sel);
		generated_relative.push_back(sel);

		// emb_code feedback
		{
			float const * ecw = tts_ctx.emb_code_weight;
			int const ecs = tts_ctx.emb_code_hidden_size;
			std::vector<float> emb(ecs);
			for (int j = 0; j < ecs; ++j) emb[j] = ecw[sel * ecs + j];

			llama_set_embeddings(ctx_tts, true);
			llama_batch eb = {};
			eb.n_tokens = 1;
			eb.embd = emb.data();
			eb.pos = &n_past_tts;
			if (llama_decode(ctx_tts, eb))
				FAIL("Failed to prefill emb_code");
			n_past_tts++;
		}
	}

	std::cout << "  ✓ Generated " << audio_tokens.size() << " audio tokens" << std::endl;
	REQUIRE(!audio_tokens.empty());

	// -------------------------------------------------------------------------
	// Step 10: Convert audio tokens → waveform
	// -------------------------------------------------------------------------
	std::cout << "\nStep 10: Converting to audio via Token2Wav..." << std::endl;

	std::vector<float> audio_samples;
	bool tok2wav_ok =
		omni::tts::generate_audio_from_tokens(tts_ctx, audio_tokens, audio_samples, SAMPLE_RATE);
	REQUIRE(tok2wav_ok);
	REQUIRE(!audio_samples.empty());

	std::cout << "  ✓ " << audio_samples.size() << " samples ("
			  << (float)audio_samples.size() / SAMPLE_RATE << " s)" << std::endl;

	// -------------------------------------------------------------------------
	// Step 11: Write WAV and verify minimum duration
	// -------------------------------------------------------------------------
	std::cout << "\nStep 11: Writing WAV..." << std::endl;

	std::string const out_path = "/tmp/duplex_tts_output.wav";
	int written = write_wav_mono(out_path, audio_samples, SAMPLE_RATE);
	REQUIRE(written > 0);
	REQUIRE(audio_samples.size() >= static_cast<size_t>(SAMPLE_RATE));	// >= 1 second

	std::cout << "  ✓ " << out_path << " — " << (float)written / SAMPLE_RATE << " s" << std::endl;
	std::cout << "\n=== Duplex TTS Test Complete ===" << std::endl;
}

// =============================================================================
// duplex_story_voice_clone
//
// Full duplex pipeline with LLM-level voice cloning:
//   1. Voice reference (alternative.wav) is encoded with the
//      audio encoder and injected into the system prompt:
//        <|im_start|>system
//        Streaming Duplex Conversation! You are a helpful assistant.
//        <|audio_start|>[ref_embeds]<|audio_end|><|im_end|>
//      This primes the LLM to match the reference speaker's style/prosody.
//
//   2. User audio (tell_me_a_story.wav) is encoded and placed in <unit> as
//      the user turn.  A silence chunk follows if the model first listens,
//      replicating the two-turn VAD pattern used in real duplex operation.
//
//   3. The model generates a story autoregressively.  Hidden states are
//      captured token-by-token during the <|speak|> phase and fed into the
//      same TTS pipeline as duplex_tts (projector → L2-norm → condition
//      embeddings → TTS transformer → Token2Wav).
//
//   4. The synthesised audio is written to /tmp/duplex_story_voice_clone.wav
//      and must be at least 2 seconds long.
// =============================================================================
TEST_CASE("duplex_story_voice_clone", "[tts][duplex][voice_clone]")
{
	std::cout << "\n=== Duplex TTS: Story with Voice Clone ===" << std::endl;

	// -------------------------------------------------------------------------
	// Step 1: Load LLM (embeddings=true for hidden state capture)
	// -------------------------------------------------------------------------
	std::cout << "\nStep 1: Loading LLM..." << std::endl;

	llama_model_params model_params = llama_model_default_params();
	// model_params.n_gpu_layers = 0;

	llama_model * model = llama_model_load_from_file(PATH_MODEL_LLM, model_params);
	REQUIRE(model != nullptr);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 512;
	ctx_params.n_ubatch = 512;
	ctx_params.embeddings = true;  // required: hidden states captured during generation

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	REQUIRE(ctx != nullptr);

	llama_vocab const * vocab = llama_model_get_vocab(model);
	int32_t const n_embd = llama_model_n_embd(model);
	std::cout << "  ✓ LLM loaded (n_embd=" << n_embd << ")" << std::endl;

	// -------------------------------------------------------------------------
	// Step 2: Load TTS weights
	// -------------------------------------------------------------------------
	std::cout << "\nStep 2: Loading TTS weights..." << std::endl;

	omni::tts::TTSContext tts_ctx;
	bool ok = omni::tts::load_tts_weights(
		tts_ctx,
		PATH_MODEL_PROJECTOR,
		PATH_MODEL_TTS_TRANSFORMER,
		PATH_MODEL_TTS_WEIGHTS,
		PATH_DIR_TOKEN2WAV);
	REQUIRE(ok);
	std::cout << "  ✓ TTS weights loaded" << std::endl;

	// Switch Token2Wav to the alternative voice cache so Token2Wav-level
	// timbre (spk_bc) matches the reference audio, not the default speaker.
	if (tts_ctx.token2wav_session)
	{
		bool cache_ok = tts_ctx.token2wav_session->t2w.start_stream_with_prompt_cache_gguf(
			PATH_ALTERNATIVE_PROMPT_CACHE);
		REQUIRE(cache_ok);
		std::cout << "  ✓ Token2Wav voice cache switched to alternative voice" << std::endl;
	}

	// -------------------------------------------------------------------------
	// Step 3: Load TTS transformer
	// -------------------------------------------------------------------------
	std::cout << "\nStep 3: Loading TTS transformer..." << std::endl;

	llama_model_params tts_mparams = llama_model_default_params();
	// tts_mparams.n_gpu_layers = 0;
	llama_model * model_tts = llama_model_load_from_file(PATH_MODEL_TTS_TRANSFORMER, tts_mparams);
	REQUIRE(model_tts != nullptr);

	llama_context_params tts_cparams = llama_context_default_params();
	tts_cparams.n_ctx = 4096;
	tts_cparams.n_batch = 4096;
	tts_cparams.n_ubatch = 4096;
	tts_cparams.embeddings = true;
	llama_context * ctx_tts = llama_init_from_model(model_tts, tts_cparams);
	REQUIRE(ctx_tts != nullptr);
	std::cout << "  ✓ TTS transformer loaded" << std::endl;

	// -------------------------------------------------------------------------
	// Step 4: Encode voice reference
	//
	// Embeds are injected into system prompt as <|audio_start|>[embeds]<|audio_end|>
	// giving the LLM a voice identity to imitate (prosody/style only at this level;
	// Token2Wav timbre comes from the alternative spk_bc in alternative_prompt_cache.gguf).
	// -------------------------------------------------------------------------
	std::cout << "\nStep 4: Encoding voice reference..." << std::endl;

	int n_embd_audio = 0;
	std::vector<float> ref_embeds =
		load_and_encode_audio_vc(PATH_MODEL_AUDIO, PATH_AUDIO_VOICE_REF, 4, &n_embd_audio);
	REQUIRE(!ref_embeds.empty());
	REQUIRE(n_embd_audio == n_embd);
	int n_ref_tokens = (int)(ref_embeds.size() / n_embd);
	std::cout << "  ✓ Voice reference: " << n_ref_tokens << " tokens" << std::endl;

	// -------------------------------------------------------------------------
	// Step 5: Encode user audio ("tell me a story")
	// -------------------------------------------------------------------------
	std::cout << "\nStep 5: Encoding user audio..." << std::endl;

	std::vector<float> user_embeds =
		load_and_encode_audio_vc(PATH_MODEL_AUDIO, PATH_AUDIO_STORY_IN, 4);
	REQUIRE(!user_embeds.empty());
	int n_user_tokens = (int)(user_embeds.size() / n_embd);
	std::cout << "  ✓ User audio: " << n_user_tokens << " tokens" << std::endl;

	std::vector<float> stop_embeds = load_and_encode_audio_vc(PATH_MODEL_AUDIO, PATH_AUDIO_STOP, 4);
	REQUIRE(!user_embeds.empty());
	int n_stop_voice_tokens = (int)(stop_embeds.size() / n_embd);
	std::cout << "  ✓ Stop audio: " << n_stop_voice_tokens << " tokens" << std::endl;

	// -------------------------------------------------------------------------
	// Step 6: Encode silence (end-of-speech signal for second duplex turn)
	// -------------------------------------------------------------------------
	std::cout << "\nStep 6: Encoding silence..." << std::endl;

	std::vector<float> silence_embeds = encode_silence_vc(PATH_MODEL_AUDIO);
	REQUIRE(!silence_embeds.empty());
	std::cout << "  ✓ Silence: " << (silence_embeds.size() / n_embd) << " tokens" << std::endl;

	// -------------------------------------------------------------------------
	// Step 7: Create sampler (greedy for deterministic speak/listen decision)
	// -------------------------------------------------------------------------
	common_params_sampling sp_llm = {};
	sp_llm.temp = 0.0f;
	sp_llm.penalty_repeat = 1.0f;
	common_sampler * smpl = common_sampler_init(model, sp_llm);
	REQUIRE(smpl != nullptr);

	// -------------------------------------------------------------------------
	// Step 8: Prefill system prompt (with voice reference) + <unit> + user audio
	//
	// Duplex voice-clone format (omni.cpp:4164-4169, TTS_PIPELINE.md):
	//   <|im_start|>system
	//   Streaming Duplex Conversation! You are a helpful assistant.
	//   <|audio_start|>[ref_embeds]<|audio_end|><|im_end|>
	//   <unit>[user_audio_embeds]
	// -------------------------------------------------------------------------
	std::cout << "\nStep 8: Prefilling prompt..." << std::endl;

	int n_past = 0;

	REQUIRE(prefill_text_vc(
		ctx,
		vocab,
		"<|im_start|>system\n"
		"Streaming Duplex Conversation! You are a helpful assistant.\n"
		"<|audio_start|>",
		n_past));

	REQUIRE(prefill_audio_vc(ctx, ref_embeds, n_embd, n_past));

	REQUIRE(prefill_text_vc(ctx, vocab, "<|audio_end|><|im_end|>\n<unit>", n_past));
	std::cout << "  ✓ System + ref voice prefilled (" << n_past << " tokens)" << std::endl;

	REQUIRE(prefill_audio_vc(ctx, user_embeds, n_embd, n_past));
	std::cout << "  ✓ User audio prefilled (n_past=" << n_past << ")" << std::endl;

	// REQUIRE(prefill_text_vc(ctx, vocab, "</unit>", n_past));
	// -------------------------------------------------------------------------
	// Step 9: Multi-turn autoregressive generation with hidden state capture
	//
	// Turn A: user audio → model generates <|listen|> or <|chunk_eos|>
	//         (needs more signal → we feed silence as Turn B)
	// Turn B: silence → model generates <|speak|> + story text + <|chunk_eos|>
	//
	// During the speak phase every generated token is decoded one-by-one and
	// its hidden state is captured via llama_get_embeddings_ith(ctx, -1).
	// Special tokens (<|speak|>, <|chunk_eos|>) are kept and later stripped by
	// filter_special_tokens, mirroring the pre-supplied approach in duplex_tts.
	// -------------------------------------------------------------------------
	std::cout << "\nStep 9: Multi-turn generation with hidden state capture..." << std::endl;

	// Single-token decode: advances n_past and makes logits + embeddings available.
	auto const decode_one = [&](llama_token tok) -> bool
	{
		llama_batch b = llama_batch_init(1, 0, 1);
		b.n_tokens = 1;
		b.token[0] = tok;
		b.pos[0] = n_past;
		b.n_seq_id[0] = 1;
		b.seq_id[0][0] = 0;
		b.logits[0] = 1;
		llama_set_embeddings(ctx, true);
		bool r = (llama_decode(ctx, b) == 0);
		llama_batch_free(b);
		if (r)
			++n_past;
		return r;
	};

	std::vector<llama_token> speak_tokens;	// tokens in speak phase (incl. markers)
	std::vector<float> speak_hidden;		// hidden states, n_embd per token

	bool in_speak_phase = false;
	bool generation_done = false;

	// Each duplex "chunk" is only a few words.  We continue presenting silence turns
	// and collecting tokens until <|turn_eos|> or the total speak-token budget is hit.
	int const max_turns = 30;		 // enough for a multi-sentence story
	int const max_speak_toks = 300;	 // total pre-filter speak tokens (incl. markers)
	int const stop_at_n_speak_toks = 50;

	for (int turn = 0; turn < max_turns && !generation_done; ++turn)
	{
		// Per-turn cap: short for listen turns, generous for speak turns
		int const max_this_turn = in_speak_phase ? max_speak_toks : 20;
		std::cout << "  Turn " << (turn + 1) << (in_speak_phase ? " (speak)" : " (listen)") << "..."
				  << std::endl;

		for (int step = 0; step < max_this_turn && !generation_done; ++step)
		{
			// Hard stop on total collected speak tokens
			if ((int)speak_tokens.size() >= max_speak_toks)
			{
				generation_done = true;
				break;
			}

			llama_token tok = common_sampler_sample(smpl, ctx, -1);
			common_sampler_accept(smpl, tok, true);

			if (llama_vocab_is_eog(vocab, tok))
			{
				generation_done = true;
				break;
			}

			// Mark speak phase BEFORE decode so hidden state is captured below.
			if (tok == TOKEN_SPEAK)
				in_speak_phase = true;

			if (!decode_one(tok))
				FAIL("LLM decode failed");

			if (in_speak_phase)
			{
				speak_tokens.push_back(tok);
				float const * hs = llama_get_embeddings_ith(ctx, -1);
				REQUIRE(hs != nullptr);
				for (int j = 0; j < n_embd; ++j) speak_hidden.push_back(hs[j]);
			}

			// Diagnostics for control tokens and beginning of each turn
			if (step < 3 || tok == TOKEN_SPEAK || tok == TOKEN_LISTEN || tok == TOKEN_CHUNK_EOS ||
				tok == TOKEN_TURN_EOS)
			{
				std::vector<char> piece(32);
				int plen = llama_token_to_piece(vocab, tok, piece.data(), 32, 0, true);
				std::string ps = (plen > 0) ? std::string(piece.data(), plen) : "?";
				std::cout << "    [" << step << "] tok=" << tok << " \"" << ps << "\"" << std::endl;
			}

			// <|turn_eos|> → model is done speaking for this turn
			if (in_speak_phase && tok == TOKEN_TURN_EOS)
			{
				generation_done = true;
				break;
			}

			if (in_speak_phase && speak_tokens.size() == stop_at_n_speak_toks)
			{
				std::cout << "Saying 'Stop'" << std::endl;
				REQUIRE(prefill_text_vc(ctx, vocab, "</unit>", n_past));
				REQUIRE(prefill_text_vc(ctx, vocab, "<unit>", n_past));
				REQUIRE(prefill_audio_vc(ctx, stop_embeds, n_embd, n_past));
			}

			// <|chunk_eos|> in speak phase → end of one speech chunk.
			// Present another silence unit to let the model continue the story.
			// if (in_speak_phase && tok == TOKEN_CHUNK_EOS) {
			//     REQUIRE(prefill_text_vc(ctx, vocab, "</unit>", n_past));
			//     REQUIRE(prefill_text_vc(ctx, vocab, "<unit>", n_past));
			//     REQUIRE(prefill_audio_vc(ctx, silence_embeds, n_embd, n_past));
			//     break;  // break inner → outer continues with next chunk
			// }

			// End of listen turn → present silence to trigger speak
			// if (!in_speak_phase &&
			//     (tok == TOKEN_LISTEN || tok == TOKEN_CHUNK_EOS || tok == TOKEN_TURN_EOS)) {
			//     REQUIRE(prefill_text_vc(ctx, vocab, "</unit>", n_past));
			//     REQUIRE(prefill_text_vc(ctx, vocab, "<unit>", n_past));
			//     REQUIRE(prefill_audio_vc(ctx, silence_embeds, n_embd, n_past));
			//     break;  // break inner → outer loop advances to next turn
			// }
		}
	}

	REQUIRE_FALSE(speak_tokens.empty());
	std::cout << "  ✓ Collected " << speak_tokens.size() << " speak tokens (before filtering)"
			  << std::endl;

	common_sampler_free(smpl);

	// -------------------------------------------------------------------------
	// Step 10: Filter duplex special tokens from speak sequence
	//
	// Removes <|speak|> (151706) and <|chunk_eos|> (151718) from both the token
	// list and the aligned hidden state matrix.
	// -------------------------------------------------------------------------
	std::cout << "\nStep 10: Filtering special tokens..." << std::endl;

	omni::tts::filter_special_tokens(speak_tokens, speak_hidden, n_embd);
	REQUIRE(!speak_tokens.empty());
	std::cout << "  ✓ " << speak_tokens.size() << " text tokens after filtering" << std::endl;

	// -------------------------------------------------------------------------
	// Step 11: Apply projector and build TTS condition embeddings
	//
	// condition[i] = emb_text(token[i]) + L2_norm(projector(hidden[i]))
	// then append emb_text(text_eos=151692) and emb_text(audio_bos=151687)
	// -------------------------------------------------------------------------
	std::cout << "\nStep 11: Building TTS condition embeddings..." << std::endl;

	int const n_text = static_cast<int>(speak_tokens.size());

	std::vector<float> projected;
	omni::tts::apply_projector_semantic(tts_ctx.projector_weights, speak_hidden, n_text, projected);
	REQUIRE(static_cast<int>(projected.size()) == n_text * 768);
	omni::tts::normalize_l2_per_token(projected.data(), n_text, 768);

	std::vector<float> condition(n_text * 768);
	for (int i = 0; i < n_text; ++i)
	{
		std::vector<float> te(768);
		omni::tts::tts_emb_text(tts_ctx, speak_tokens[i], te.data(), 768);
		for (int j = 0; j < 768; ++j) condition[i * 768 + j] = te[j] + projected[i * 768 + j];
	}
	{
		std::vector<float> te(768);
		omni::tts::tts_emb_text(tts_ctx, TOKEN_TEXT_EOS, te.data(), 768);
		condition.insert(condition.end(), te.begin(), te.end());
	}
	{
		std::vector<float> te(768);
		omni::tts::tts_emb_text(tts_ctx, TOKEN_AUDIO_BOS, te.data(), 768);
		condition.insert(condition.end(), te.begin(), te.end());
	}

	int const n_cond = static_cast<int>(condition.size()) / 768;
	std::cout << "  ✓ Condition: " << n_text << " text + eos + bos = " << n_cond << " tokens"
			  << std::endl;

	// -------------------------------------------------------------------------
	// Step 12: Clear TTS KV cache and prefill condition
	// -------------------------------------------------------------------------
	std::cout << "\nStep 12: Prefilling TTS condition..." << std::endl;

	{
		llama_memory_t mem = llama_get_memory(ctx_tts);
		if (mem)
			llama_memory_seq_rm(mem, 0, 0, -1);
	}

	int n_past_tts = 0;
	{
		llama_set_embeddings(ctx_tts, true);
		int const n_batch_tts = 4096;
		for (int i = 0; i < n_cond; i += n_batch_tts)
		{
			int n_eval = std::min(n_batch_tts, n_cond - i);
			llama_batch cb = {};
			cb.n_tokens = n_eval;
			cb.embd = condition.data() + i * 768;
			std::vector<llama_pos> pv(n_eval);
			std::iota(pv.begin(), pv.end(), n_past_tts);
			cb.pos = pv.data();
			if (llama_decode(ctx_tts, cb))
				FAIL("Failed to prefill TTS condition");
			n_past_tts += n_eval;
		}
	}
	std::cout << "  ✓ Condition prefilled, n_past_tts=" << n_past_tts << std::endl;

	// -------------------------------------------------------------------------
	// Step 13: Sample audio tokens (same pipeline as duplex_tts)
	// -------------------------------------------------------------------------
	std::cout << "\nStep 13: Sampling audio tokens..." << std::endl;

	struct common_params_sampling tts_sp = {};
	tts_sp.temp = 0.8f;
	tts_sp.top_k = 25;
	tts_sp.top_p = 0.85f;
	tts_sp.penalty_repeat = 1.05f;
	tts_sp.penalty_last_n = 16;

	int const num_audio_tokens = 6562;
	int const eos_rel_idx = num_audio_tokens - 1;
	// 300 text tokens → roughly 30–60s of speech → ~6000 audio tokens needed
	int const max_audio_tokens = 6000;

	std::vector<int32_t> audio_tokens;
	std::vector<int> generated_relative;
	static std::mt19937 rng_vc{42};

	for (int t = 0; t < max_audio_tokens; ++t)
	{
		float const * hs = llama_get_embeddings_ith(ctx_tts, -1);
		REQUIRE(hs != nullptr);

		std::vector<float> logits(num_audio_tokens, 0.0f);
		float const * hcw = tts_ctx.head_code_weight;
		int const hid = tts_ctx.head_code_hidden_size;
		for (int i = 0; i < num_audio_tokens; ++i)
		{
			float const * row = hcw + i * hid;
			float s = 0.0f;
			for (int j = 0; j < hid; ++j) s += hs[j] * row[j];
			logits[i] = s;
		}

		for (float & l : logits) l /= tts_sp.temp;

		if (!generated_relative.empty())
		{
			int start = std::max(0, (int)generated_relative.size() - tts_sp.penalty_last_n);
			std::vector<bool> seen(num_audio_tokens, false);
			for (int k = start; k < (int)generated_relative.size(); ++k)
			{
				int id = generated_relative[k];
				if (id >= 0 && id < num_audio_tokens)
					seen[id] = true;
			}
			for (int i = 0; i < num_audio_tokens; ++i)
			{
				if (!seen[i])
					continue;
				if (logits[i] >= 0)
					logits[i] /= tts_sp.penalty_repeat;
				else
					logits[i] *= tts_sp.penalty_repeat;
			}
		}

		float max_l = *std::max_element(logits.begin(), logits.end());
		std::vector<float> probs(num_audio_tokens);
		float sum = 0.0f;
		for (int i = 0; i < num_audio_tokens; ++i)
		{
			probs[i] = std::exp(logits[i] - max_l);
			sum += probs[i];
		}
		for (float & p : probs) p /= sum;

		if (tts_sp.top_k > 0 && tts_sp.top_k < num_audio_tokens)
		{
			std::vector<float> sp2 = probs;
			std::nth_element(
				sp2.begin(), sp2.begin() + (num_audio_tokens - tts_sp.top_k), sp2.end());
			float thr = sp2[num_audio_tokens - tts_sp.top_k];
			for (float & p : probs)
				if (p < thr)
					p = 0.0f;
		}

		if (tts_sp.top_p < 1.0f)
		{
			std::vector<int> idx(num_audio_tokens);
			std::iota(idx.begin(), idx.end(), 0);
			std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
			float s2 = 0.0f;
			for (float p : probs) s2 += p;
			float cum = 0.0f;
			bool cut = false;
			for (int id : idx)
			{
				if (cut)
				{
					probs[id] = 0.0f;
					continue;
				}
				cum += probs[id] / s2;
				if (cum >= tts_sp.top_p)
					cut = true;
			}
		}

		{
			float s2 = 0.0f;
			for (float p : probs) s2 += p;
			if (s2 > 0.0f)
				for (float & p : probs) p /= s2;
		}

		int sel = eos_rel_idx;
		{
			std::uniform_real_distribution<float> dist(0.0f, 1.0f);
			float r = dist(rng_vc), cum = 0.0f;
			for (int i = 0; i < num_audio_tokens; ++i)
			{
				cum += probs[i];
				if (r <= cum)
				{
					sel = i;
					break;
				}
			}
		}

		if (sel == eos_rel_idx)
		{
			std::cout << "  EOS at t=" << t << std::endl;
			break;
		}

		audio_tokens.push_back(TOKEN_AUDIO_BOS + sel);
		generated_relative.push_back(sel);

		{
			float const * ecw = tts_ctx.emb_code_weight;
			int const ecs = tts_ctx.emb_code_hidden_size;
			std::vector<float> emb(ecs);
			for (int j = 0; j < ecs; ++j) emb[j] = ecw[sel * ecs + j];

			llama_set_embeddings(ctx_tts, true);
			llama_batch eb = {};
			eb.n_tokens = 1;
			eb.embd = emb.data();
			eb.pos = &n_past_tts;
			if (llama_decode(ctx_tts, eb))
				FAIL("Failed to feed emb_code");
			n_past_tts++;
		}
	}

	std::cout << "  ✓ Generated " << audio_tokens.size() << " audio tokens" << std::endl;
	REQUIRE(!audio_tokens.empty());

	// -------------------------------------------------------------------------
	// Step 14: Convert audio tokens → waveform via Token2Wav
	// -------------------------------------------------------------------------
	std::cout << "\nStep 14: Converting to audio via Token2Wav..." << std::endl;

	std::vector<float> audio_samples;
	bool tok2wav_ok =
		omni::tts::generate_audio_from_tokens(tts_ctx, audio_tokens, audio_samples, SAMPLE_RATE);
	REQUIRE(tok2wav_ok);
	REQUIRE(!audio_samples.empty());
	REQUIRE((float)audio_samples.size() / SAMPLE_RATE > 5.0f);

	std::cout << "  ✓ " << audio_samples.size() << " samples ("
			  << (float)audio_samples.size() / SAMPLE_RATE << " s)" << std::endl;

	// -------------------------------------------------------------------------
	// Step 15: Write WAV and assert >= 2 seconds
	// -------------------------------------------------------------------------
	std::cout << "\nStep 15: Writing WAV..." << std::endl;

	std::string const out_path_vc = "/tmp/duplex_story_voice_clone.wav";
	int written_vc = write_wav_mono(out_path_vc, audio_samples, SAMPLE_RATE);
	REQUIRE(written_vc > 0);
	REQUIRE(audio_samples.size() >= static_cast<size_t>(1 * SAMPLE_RATE));	// >= 1 second

	std::cout << "  ✓ " << out_path_vc << " — " << (float)written_vc / SAMPLE_RATE << " s"
			  << std::endl;
	std::cout << "\n=== Duplex Story Voice Clone Test Complete ===" << std::endl;

	llama_free(ctx_tts);
	llama_model_free(model_tts);
	llama_free(ctx);
	llama_model_free(model);
}
