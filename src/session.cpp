/**
 * @file   session.cpp
 * @brief  Session implementation — duplex session setup, event loop, and GPU helpers.
 *
 * ### GPU task pattern
 *
 * Every blocking llama.cpp call is wrapped in a `cobalt::task<T>` and
 * dispatched to `gpu_ex_` via `cobalt::spawn`.  Coroutines that run on
 * the main executor and are directly co_awaited use `cobalt::promise`;
 * `cobalt::task` is reserved for coroutines that are `spawn`ed (on GPU
 * or with `detached`).  The main cobalt executor suspends while GPU work
 * runs; there is no concurrent access to session members so no mutex is
 * needed.
 *
 * ### Duplex system prompt
 *
 * `init_duplex_system_prompt()` is called once at session startup (after
 * `init_gpu_resources()`).  It encodes the reference voice WAV via
 * `audition_audio_encode`, then batch-decodes the assembled system prompt
 * into the KV cache in three sub-batches:
 * ```
 * <|im_start|>system\n{duplex_system}\n<|audio_start|>   ← text tokens
 * [n_tokens × 4096-d float embeddings]                   ← audio embeddings
 * <|audio_end|><|im_end|>\n                              ← text tokens
 * ```
 * The KV state is saved to `cfg_.voice.prompt_cache_path` if configured, and
 * loaded on subsequent sessions to skip the expensive encode step.
 *
 * ### Buffered audio + duplex loop
 *
 * `run_duplex_loop()` merges buffered audio and control traffic into one
 * duplex event stream. Input PCM is coalesced into fixed-size audio units;
 * each unit explicitly opens `<unit>`, decodes optional video then audio,
 * returns one generated chunk boundary, and finalizes with `</unit>`.
 * `run_duplex_turn()` samples until
 * `<|listen|>`/`<|chunk_eos|>`/`<|chunk_tts_eos|>` and returns a structured
 * result; `<|turn_eos|>` marks end-of-turn but does not itself force the
 * return boundary. Typed text input is handled as a text-only duplex-context
 * chat turn inside the same outer loop.
 */

#include "session.hpp"

#include <atomic>
#include <chrono>
#include <compare>
#include <cstring>
#include <filesystem>
#include <format>
#include <generator>
#include <iostream>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <experimental/mdspan>

#include <gsl/util>

#include <boost/asio/detached.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/cobalt/join.hpp>
#include <boost/cobalt/spawn.hpp>
#include <boost/cobalt/wait_group.hpp>

#include <spdlog/spdlog.h>

// Vendored 3rd party utilities
#include "omni/audition.h"
#include "omni/clip.h"
#include "pipeline/audio_encoder.hpp"
#include "pipeline/tts_pipeline.hpp"

// Internal utilities
#include "wav.hpp"

namespace llama_omni_server
{

namespace
{

// ── Named constants ───────────────────────────────────────────────────────────

/// Logit flag: produce logit output at this token position (needed for sampling).
constexpr int8_t kLogitEnabled{1};

/// Logit flag: no logit output (prompt positions we do not sample from).
constexpr int8_t kLogitDisabled{0};

/// Output PCM sample rate produced by the TTS Token2Wav pipeline.
constexpr int kTtsOutputSampleRate{24000};

/// Input PCM sample rate produced by the browser microphone path.
constexpr int kInputAudioSampleRateHz{16000};

/// Millisecond conversion factor used for PCM duration calculations.
constexpr int kMillisPerSecond{1000};

[[nodiscard]] bool is_session_shutdown_error(boost::system::error_code const & code) noexcept
{
	return code == boost::asio::error::operation_aborted || code == boost::asio::error::broken_pipe;
}

[[noreturn]] void throw_session_shutdown_requested()
{
	throw boost::system::system_error{boost::asio::error::operation_aborted};
}

// ── MiniCPM-o 4.5 duplex special token IDs ────────────────────────────────────
// Source: AGENTS.md §"Duplex control tokens".

/// `<unit>` — caller prepends before each user-input block.
constexpr llama_token kTokenUnit{151683};

/// `</unit>` — caller appends when finalizing one user-input block.
constexpr llama_token kTokenUnitEnd{151684};

/// `<|speak|>` — model output: model decides to start speaking.
constexpr llama_token kTokenSpeak{151706};

/// `<|listen|>` — model output: model decides to keep listening; ends generation.
constexpr llama_token kTokenListen{151705};

/// `<|chunk_eos|>` — model output: audio chunk boundary (soft-interrupt point).
constexpr llama_token kTokenChunkEos{151718};

/// `<|turn_eos|>` — model output: full turn complete marker.
constexpr llama_token kTokenTurnEos{151717};

/// `<|chunk_tts_eos|>` — model output: alternate chunk boundary used upstream.
constexpr llama_token kTokenChunkTtsEos{151721};

/// `<|tts_pad|>` — forbidden during sampling; skip without emitting.
constexpr llama_token kTokenTtsPad{151722};

/// `<image>` — opens an image block inside a `<unit>`.
constexpr llama_token kTokenImage{151669};

/// `</image>` — closes an image block; logits enabled here for sampling.
constexpr llama_token kTokenImageEnd{151670};

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * @brief Tokenise `text` into llama_token IDs.
 *
 * Optimistic single-call pattern: allocate a buffer sized to `text.size()`
 * (tokens cover multiple characters so this is usually large enough).  If
 * the buffer is too small, `llama_tokenize` returns a negative value whose
 * absolute is the required count; resize and retry.
 *
 * `add_special = false` because we build the ChatML wrap manually.
 * `parse_special = true` so `<|im_start|>` etc. tokenise as single IDs.
 *
 * @throws std::runtime_error on tokenisation failure.
 */
std::vector<llama_token> do_tokenize(llama_vocab const * vocab, std::string_view text)
{
	// Optimistic allocation: token count < character count in practice.
	auto n_max = static_cast<int32_t>(text.size());
	std::vector<llama_token> tokens(static_cast<std::size_t>(n_max));

	int result = llama_tokenize(
		vocab,
		text.data(),
		static_cast<int32_t>(text.size()),
		tokens.data(),
		n_max,
		/*add_special=*/false,
		/*parse_special=*/true);

	if (result >= 0)
	{
		// Fitted in the initial buffer; trim to actual count.
		tokens.resize(static_cast<std::size_t>(result));
		return tokens;
	}

	// Buffer was too small; -result is the exact required count.
	n_max = -result;
	tokens.resize(static_cast<std::size_t>(n_max));

	result = llama_tokenize(
		vocab,
		text.data(),
		static_cast<int32_t>(text.size()),
		tokens.data(),
		n_max,
		/*add_special=*/false,
		/*parse_special=*/true);

	if (result < 0 || !std::cmp_equal(result, n_max))
	{
		throw std::runtime_error{
			std::format("do_tokenize: second call returned {} (expected {})", result, n_max)};
	}

	return tokens;
}

/**
 * @brief Convert one token ID to its UTF-8 text piece.
 *
 * `llama_token_to_piece` writes the piece into `buf`.  Returns empty string
 * if the token has no printable representation (e.g. most special tokens).
 *
 * `special = false` → don't render angle-bracket form for special tokens.
 */
std::string token_piece(
	llama_vocab const * vocab, llama_token token, bool should_include_special = false)
{
	constexpr int kBufSize{256};
	// Initialise with null bytes; llama_token_to_piece writes into the buffer
	// and returns the number of bytes written.  We then trim to that length.
	std::string buf(kBufSize, '\0');
	int const n_chars = llama_token_to_piece(
		vocab,
		token,
		buf.data(),
		kBufSize,
		/*lstrip=*/0,
		/*special=*/should_include_special);
	if (n_chars <= 0)
	{
		return {};
	}
	buf.resize(static_cast<std::size_t>(n_chars));
	return buf;
}

void llama_spdlog_callback(ggml_log_level level, char const * text, void * /*user_data*/)
{
	if (text == nullptr)
	{
		return;
	}

	std::string_view message{text};
	while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
	{
		message.remove_suffix(1);
	}

	if (message.empty())
	{
		return;
	}

	switch (level)
	{
		case GGML_LOG_LEVEL_DEBUG:
			spdlog::trace("libllama: {}", message);
			break;
		case GGML_LOG_LEVEL_INFO:
			spdlog::debug("libllama: {}", message);
			break;
		case GGML_LOG_LEVEL_WARN:
			spdlog::warn("libllama: {}", message);
			break;
		case GGML_LOG_LEVEL_ERROR:
			spdlog::error("libllama: {}", message);
			break;
		case GGML_LOG_LEVEL_CONT:
		case GGML_LOG_LEVEL_NONE:
		default:
			spdlog::trace("libllama: {}", message);
			break;
	}
}

[[nodiscard]] std::string next_session_id()
{
	static std::atomic_uint64_t next_id{1};
	return std::format("session_{:04}", next_id.fetch_add(1));
}

}  // namespace

// ── RAII deleters ─────────────────────────────────────────────────────────────

void LlamaSamplerDeleter::operator()(llama_sampler * ptr) const noexcept
{
	/// `llama_sampler_free` releases the sampler chain and all sub-samplers.
	llama_sampler_free(ptr);
}

// ── Session ───────────────────────────────────────────────────────────────────

Session::Session(
	AppConfig const & cfg,
	ModelManager & models,
	boost::asio::thread_pool & gpu_ex,
	boost::cobalt::channel<OutFrame> & output_ch,
	boost::cobalt::channel<std::vector<float>> & audio_in_ch,
	boost::cobalt::channel<VideoFrame> & video_in_ch,
	boost::cobalt::channel<TextInput> & text_in_ch,
	boost::cobalt::channel<SpeechChunkAckEvent> & speech_ack_ch)
	: cfg_{cfg},
	  models_{models},
	  gpu_ex_{gpu_ex},
	  output_ch_{output_ch},
	  audio_in_ch_{audio_in_ch},
	  video_in_ch_{video_in_ch},
	  text_in_ch_{text_in_ch},
	  speech_ack_ch_{speech_ack_ch}
{
	conv_.session_id = next_session_id();
}

Session::~Session() = default;

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void Session::note_audio_received(std::size_t sample_count) noexcept
{
	num_pending_audio_samples_ += sample_count;
}

std::uint32_t Session::video_last_ms() const noexcept
{
	if (!last_video_decode_at_.has_value())
	{
		return 0U;
	}

	auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - *last_video_decode_at_);
	return static_cast<std::uint32_t>(std::max<std::int64_t>(elapsed.count(), 0));
}

bool Session::listening() const noexcept
{
	return listening_;
}

bool Session::startup_complete() const noexcept
{
	return startup_complete_;
}

void Session::request_shutdown() noexcept
{
	shutdown_requested_.store(true, std::memory_order_relaxed);
}

bool Session::shutdown_requested() const noexcept
{
	return shutdown_requested_.load(std::memory_order_relaxed);
}

void Session::throw_if_shutdown_requested() const
{
	if (shutdown_requested())
	{
		throw_session_shutdown_requested();
	}
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: coroutine
// accesses member variables (gpu_ex_, control_ch_, etc.)
boost::cobalt::task<void> Session::run()
{
	spdlog::info("session: starting");
	std::optional<std::string> fatal_error;
	try
	{
		/// Route libllama / ggml logging through the project's spdlog sink before
		/// any model or context API calls happen in this session.
		llama_log_set(llama_spdlog_callback, nullptr);

		// Allocate the llama_context and sampler on the GPU thread before
		// accepting any control events.
		co_await boost::cobalt::spawn(
			gpu_ex_.get_executor(), init_gpu_resources(), boost::cobalt::use_op);

		spdlog::info("session: GPU resources ready");

		// Prefill the duplex system prompt.
		co_await init_duplex_system_prompt();
		startup_complete_ = true;

		// Enter the buffered-audio duplex loop.
		co_await loop_duplex_events();
	}
	catch (boost::system::system_error const & ex)
	{
		if (is_session_shutdown_error(ex.code()))
		{
			spdlog::debug("session: run exiting during shutdown ({})", ex.what());
		}
		else
		{
			fatal_error = ex.what();
		}
	}
	catch (std::exception const & ex)
	{
		fatal_error = ex.what();
	}

	if (fatal_error.has_value())
	{
		co_await fail_session(std::move(*fatal_error));
	}

	if (ctx_ != nullptr || sampler_ != nullptr)
	{
		// Release GPU resources on the GPU thread before destruction.
		co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
			gpu_ex_.get_executor(),
			free_gpu_resources(),
			boost::cobalt::use_op);
	}

	spdlog::info("session: stopped");
}

boost::cobalt::promise<void> Session::fail_session(std::string message)
{
	spdlog::error("session: fatal error: {}", message);

	if (shutdown_requested())
	{
		spdlog::debug("session: suppressing terminal error output during shutdown");
	}
	else
	{
		try
		{
			co_await output_ch_.write(OutFrame{ErrorOutFrame{.message = message}});
		}
		catch (boost::system::system_error const & ex)
		{
			spdlog::debug("session: failed to emit ErrorOutFrame ({})", ex.what());
		}

		try
		{
			co_await output_ch_.write(OutFrame{DoneOutFrame{.end_of_turn = false}});
		}
		catch (boost::system::system_error const & ex)
		{
			spdlog::debug("session: failed to emit terminal DoneOutFrame ({})", ex.what());
		}
	}

	if (text_in_ch_.is_open())
	{
		text_in_ch_.close();
	}
	if (speech_ack_ch_.is_open())
	{
		speech_ack_ch_.close();
	}
	if (audio_in_ch_.is_open())
	{
		audio_in_ch_.close();
	}
	if (video_in_ch_.is_open())
	{
		video_in_ch_.close();
	}

	co_return;
}

// The duplex loop merges audio and text events via a shared channel (two
// feeder tasks), drives generation turns, and relies on a per-turn drainer
// coroutine to relay speech chunks to the client and await playback acks.
// cobalt::race cannot be used here due to a GCC 15.2 ICE (make_decl_rtl,
// varasm.cc:1459) triggered by any channel type inside a race expression.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
boost::cobalt::promise<void> Session::loop_duplex_events()
{
	spdlog::info("session: entering duplex loop");

	// Cobalt exposes the current coroutine executor through `this_coro::executor`.
	// NOLINTNEXTLINE(cppcoreguidelines-init-variables,readability-static-accessed-through-instance)
	auto const exec = co_await boost::cobalt::this_coro::executor;
	static constexpr std::size_t kMergedChanCap = 1U;
	auto merged_ch = std::make_shared<boost::cobalt::channel<DuplexEvent>>(kMergedChanCap, exec);
	boost::cobalt::wait_group feeders;

	/// Feeder 1: coalesces raw PCM from audio_in_ch_ into fixed-size chunks.
	feeders.push_back(loop_audio_in_ch(merged_ch));

	/// Feeder 2: forwards TextInput events from speech_ack_ch_.
	feeders.push_back(loop_text_in_ch(merged_ch));

	/// Feeder 3: forwards SpeechChunkAckEvent events from text_in_ch_.
	feeders.push_back(loop_speech_ack_ch(merged_ch));

	/// Feeder 4: CLIP encodes VideoFrame events from video_in_ch_ and forwards as
	/// PendingClipResult.
	feeders.push_back(loop_video_in_ch(merged_ch));

	/// Shared state for the latest CLIP encode result.
	std::optional<PendingClipResult> pending_clip;
	// After a speech chunk has been sent to the client, how many generation turns are allowed
	// whilst we await an ack from the client (i.e. to have some data ready as soon as ack is
	// received, but balance that against moving faster than the client can play audio).
	static constexpr uint32_t kNumTurnsAllowedWithoutAck = 1;
	// How many speech chunks can be sent in total before we must block waiting for an ack before
	// we can continue generating.
	static constexpr uint32_t kNumPendingAcksAllowed = 1;

	// Current turn number.
	uint32_t current_turn = 0;
	// Limit turn number to turns below this value.
	uint32_t allowed_max_turn = std::numeric_limits<uint32_t>::max();
	// Track last time speech was acked.
	uint32_t last_speech_seq = 0;
	uint32_t last_speech_ack_seq = 0;

	std::exception_ptr loop_error;
	try
	{
		while (true)
		{
			// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
			DuplexEvent event = co_await merged_ch->read();

			if (auto * text_ev = std::get_if<TextInput>(&event))
			{
				co_await handle_text_input_in_duplex(std::move(text_ev->text));
				continue;
			}

			if (auto * ack_ev = std::get_if<PendingClipResult>(&event))
			{
				pending_clip = std::move(*ack_ev);
				continue;
			}

			if (auto * ack_ev = std::get_if<SpeechChunkAckEvent>(&event))
			{
				last_speech_ack_seq = ack_ev->seq;
				if (last_speech_ack_seq + kNumPendingAcksAllowed >= last_speech_seq)
					allowed_max_turn = std::numeric_limits<uint32_t>::max();
				continue;
			}

			if (current_turn >= allowed_max_turn)
			{
				listening_ = false;
				spdlog::debug(
					"session: too many turns without a speech ack; dropping pending audio and "
					"skipping turn");
				continue;
			}

			listening_ = true;

			// At this point we must have an audio event. A conversation chunk from the llm is a
			// "reaction" to input audio, even if the audio is silence. So logically we have a
			// constant stream of audio (notwithstanding allowed_max_turn), which we decode chunk
			// by chunk, pairing each chunk with the llm's response.

			// AudioChunkEvent — decode audio (+ pending CLIP image), then run
			// a full generation turn with the per-turn chunk drainer.
			auto & audio_event = std::get<AudioChunkEvent>(event);
			spdlog::debug(
				"session: input audio chunk {}ms → encode + generate", audio_event.duration_ms);

			// Open turn.
			throw_if_shutdown_requested();
			co_await open_duplex_unit();

			// Decode audio chunk.
			co_await boost::cobalt::spawn(	// NOLINT(readability-static-accessed-through-instance)
				gpu_ex_.get_executor(),
				batch_decode_audio_on_gpu(
					std::move(audio_event.pcm), /*need_logits=*/!pending_clip.has_value()),
				boost::cobalt::use_op);
			spdlog::debug("session: audio decode finished for current duplex chunk");

			throw_if_shutdown_requested();

			// Save off the latest available CLIP-encoded video frame (so we have a stable
			// reference to VideoAckOutFrame, below).
			std::optional<PendingClipResult> consumed_clip = std::move(pending_clip);
			pending_clip.reset();

			// Decode latest available video frame.
			if (consumed_clip)
			{
				co_await boost::cobalt::spawn(	// NOLINT(readability-static-accessed-through-instance)
					gpu_ex_.get_executor(),
					batch_decode_image_on_gpu(std::move(consumed_clip->clip)),
					boost::cobalt::use_op);
				spdlog::debug("session: image decode finished for current duplex chunk");

				throw_if_shutdown_requested();
			}

			DuplexTurnResult turn_result;
			// Attempt to generate a single chunk of speech (audio + text).
			spdlog::debug("session: starting duplex chunk generation");
			turn_result = co_await generate_duplex_chunk();
			spdlog::debug("session: duplex chunk generation returned");
			throw_if_shutdown_requested();

			// If speech was generated, send to the client and flag that we need to await ack.
			if (turn_result.speech_chunk)
			{
				last_speech_seq = turn_result.speech_chunk->seq;
				// Allow generation to continue for a little bit whilst client plays audio.
				allowed_max_turn = current_turn + kNumTurnsAllowedWithoutAck + 1;
				// Send the speech chunk to the client
				co_await output_ch_.write(OutFrame{*turn_result.speech_chunk});
			}

			throw_if_shutdown_requested();
			co_await finalize_duplex_turn(turn_result);
			throw_if_shutdown_requested();

			// Send the consumed video frame to the client for rendering, so the user can see
			// what the llm saw.
			if (consumed_clip)
			{
				co_await output_ch_.write(
					OutFrame{VideoAckOutFrame{
						.seq = consumed_clip->seq, .raw = std::move(consumed_clip->raw)}});
			}

			++current_turn;
		}
	}
	catch (boost::system::system_error const & ex)
	{
		if (is_session_shutdown_error(ex.code()))
		{
			spdlog::debug("session: duplex loop exiting during shutdown ({})", ex.what());
		}
		else
		{
			loop_error = std::current_exception();
		}
	}
	catch (...)
	{
		loop_error = std::current_exception();
	}

	// Cleanup: close channels to wake any remaining blocked tasks.
	merged_ch->close();
	if (audio_in_ch_.is_open())
	{
		audio_in_ch_.close();
	}
	if (video_in_ch_.is_open())
	{
		video_in_ch_.close();
	}
	if (text_in_ch_.is_open())
	{
		text_in_ch_.close();
	}
	if (speech_ack_ch_.is_open())
	{
		speech_ack_ch_.close();
	}

	try
	{
		co_await feeders.await_exit(loop_error);
	}
	catch (...)
	{
		if (loop_error == nullptr)
		{
			loop_error = std::current_exception();
		}
	}

	if (current_unit_open_ && !shutdown_requested())
	{
		co_await close_duplex_unit();
	}

	if (loop_error != nullptr)
	{
		std::rethrow_exception(loop_error);
	}

	spdlog::info("session: duplex loop exited");
}

boost::cobalt::promise<void> Session::open_duplex_unit()
{
	throw_if_shutdown_requested();

	if (current_unit_open_)
	{
		co_return;
	}

	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_prompt_on_gpu({kTokenUnit}),
		boost::cobalt::use_op);
	throw_if_shutdown_requested();
	current_unit_start_ = conv_.n_past;
	current_unit_open_ = true;
}

boost::cobalt::promise<void> Session::close_duplex_unit()
{
	if (!current_unit_open_)
	{
		co_return;
	}

	int const block_end = conv_.n_past;

	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_prompt_on_gpu({kTokenUnitEnd}),
		boost::cobalt::use_op);
	throw_if_shutdown_requested();

	conv_.unit_history.push_back(UnitBlock{.pos_start = current_unit_start_, .pos_end = block_end});
	current_unit_open_ = false;

	// Check if KV cache is approaching capacity; evict if needed.
	int const remaining = cfg_.inference.n_ctx - conv_.n_past;
	if (remaining < cfg_.inference.overflow_reserve)
	{
		bool const eviction_ok = co_await evict_oldest_units();
		if (!eviction_ok)
		{
			spdlog::error("session: KV eviction failed; context may overflow");
		}
	}
}

boost::cobalt::promise<void> Session::finalize_duplex_turn(DuplexTurnResult result)
{
	throw_if_shutdown_requested();
	co_await output_ch_.write(OutFrame{DoneOutFrame{.end_of_turn = result.end_of_turn}});
	throw_if_shutdown_requested();
	co_await close_duplex_unit();

	if (result.end_of_turn)
	{
		throw_if_shutdown_requested();
		co_await boost::cobalt::spawn(
			gpu_ex_.get_executor(), clear_audio_encoder_kv_on_gpu(), boost::cobalt::use_op);
	}
}

boost::cobalt::promise<void> Session::handle_text_input_in_duplex(std::string text)
{
	spdlog::debug("session: entering duplex-context text turn");
	throw_if_shutdown_requested();

	if (current_unit_open_)
	{
		co_await close_duplex_unit();
	}

	bool const text_turn_completed = co_await run_simplex_turn_in_duplex(std::move(text));

	spdlog::debug(
		"session: duplex-context text turn - writing done to output_ch (end_of_turn={})",
		text_turn_completed);

	throw_if_shutdown_requested();
	co_await output_ch_.write(OutFrame{DoneOutFrame{.end_of_turn = text_turn_completed}});

	spdlog::debug("session: duplex-context text turn complete; duplex resumed");
}

// This coroutine implements the duplex token state machine end-to-end; the
// branch count comes from explicit model-token handling at each boundary.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
boost::cobalt::promise<DuplexTurnResult> Session::generate_duplex_chunk()
{
	auto const * vocab = &models_.vocab();

	spdlog::debug("session: duplex turn start, n_past={}", conv_.n_past);

	bool has_sampled_speak_token = false;  ///< True after `<|speak|>` token has been seen.
	debug_turn_audio_.clear();

	/// Accumulate speaking tokens + their LLM hidden states for TTS conditioning.
	/// Flushed to TTS at each `<|chunk_eos|>` and at turn end.
	std::vector<llama_token> tts_token_buf;
	std::vector<float> tts_hidden_buf;	///< Flat [n_tokens × n_embd].
	std::string chunk_text;
	std::string tokens_debugging;
	int const max_duplex_tokens = cfg_.inference.max_duplex_tokens;
	DuplexTurnResult result;
	bool is_end_of_turn_seen = false;

	int step = 0;
	for (; step < max_duplex_tokens; ++step)
	{
		throw_if_shutdown_requested();

		// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
		SampledOutput const sampled =
			co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
				gpu_ex_.get_executor(),
				sample_and_decode_on_gpu(),
				boost::cobalt::use_op);
		throw_if_shutdown_requested();

		llama_token const tok = sampled.id;

		if (spdlog::should_log(spdlog::level::debug))
		{
			tokens_debugging += token_piece(vocab, tok, true);
		}

		if (llama_vocab_is_eog(vocab, tok))
		{
			spdlog::debug("session: duplex turn — EOG at step {}", step);
			break;
		}

		// ── Duplex control token classification ───────────────────────────

		if (tok == kTokenSpeak)
		{
			/// `<|speak|>` — model chose to speak; subsequent tokens are audio codec tokens.
			has_sampled_speak_token = true;
			conv_.n_past++;
			continue;
		}

		if (tok == kTokenTurnEos)
		{
			/// `<|turn_eos|>` marks end-of-turn but does not itself force the
			/// returned chunk boundary.
			spdlog::debug(
				"session: duplex turn — turn_eos at step {} (n_past={}, chunk_text_bytes={}, "
				"tts_tokens={})",
				step,
				conv_.n_past,
				chunk_text.size(),
				tts_token_buf.size());
			is_end_of_turn_seen = true;
			conv_.n_past++;
			continue;
		}

		if (tok == kTokenListen)
		{
			/// `<|listen|>` ends the returned chunk. Any prior `<|turn_eos|>`
			/// state propagates to the DoneOutFrame.
			spdlog::debug(
				"session: duplex turn — listen at step {} (n_past={}, chunk_text_bytes={}, "
				"tts_tokens={})",
				step,
				conv_.n_past,
				chunk_text.size(),
				tts_token_buf.size());
			conv_.n_past++;
			result = DuplexTurnResult{
				.end_of_turn = is_end_of_turn_seen,
				.ended_with_listen = true,
				.speech_chunk = co_await synthesize_speech_chunk(
					std::move(chunk_text), std::move(tts_token_buf), std::move(tts_hidden_buf))};
			chunk_text = {};
			break;
		}

		if (tok == kTokenChunkEos || tok == kTokenChunkTtsEos)
		{
			/// `<|chunk_eos|>` and `<|chunk_tts_eos|>` end the returned chunk.
			spdlog::debug(
				"session: duplex turn — chunk boundary token {} at step {} "
				"(chunk_text_bytes={}, tts_tokens={})",
				tok,
				step,
				chunk_text.size(),
				tts_token_buf.size());
			conv_.n_past++;
			result = DuplexTurnResult{
				.end_of_turn = is_end_of_turn_seen,
				.ended_with_listen = false,
				.speech_chunk = co_await synthesize_speech_chunk(
					std::move(chunk_text), std::move(tts_token_buf), std::move(tts_hidden_buf))};
			chunk_text = {};
			break;
		}

		if (tok == kTokenTtsPad)
		{
			/// `<|tts_pad|>` — padding token; skip without emitting.
			conv_.n_past++;
			continue;
		}

		// ── Visible text token ────────────────────────────────────────────

		if (has_sampled_speak_token)
		{
			/// Accumulate this token + hidden state for TTS conditioning.
			tts_token_buf.push_back(tok);
			tts_hidden_buf.insert(
				tts_hidden_buf.end(), sampled.hidden.begin(), sampled.hidden.end());

			/// Buffer text until the matching audio chunk is synthesized.
			std::string const piece = token_piece(vocab, tok);
			if (!piece.empty())
				chunk_text += piece;
		}

		conv_.n_past++;
	}

	if (step == max_duplex_tokens)
	{
		std::size_t const chunk_text_bytes = chunk_text.size();
		spdlog::error(
			"session: duplex turn hit max_duplex_tokens ({}) without end token "
			"(chunk_text_bytes={}, tokens='{}')",
			max_duplex_tokens,
			chunk_text_bytes,
			tokens_debugging);
	}

	spdlog::debug("session: duplex turn tokens={}", tokens_debugging);
	flush_debug_audio_wav();

	spdlog::debug("session: duplex turn done, n_past={}", conv_.n_past);
	co_return result;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
boost::cobalt::promise<bool> Session::run_simplex_turn_in_duplex(std::string text)
{
	auto const * vocab = &models_.vocab();

	std::string const prompt =
		std::format("<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", text);
	std::vector<llama_token> prompt_tokens = do_tokenize(vocab, prompt);

	if (prompt_tokens.empty())
	{
		spdlog::warn(
			"session: duplex-context text prompt tokenised to zero tokens — skipping turn");
		co_return true;
	}

	spdlog::debug(
		"session: duplex-context text turn prompt={} tokens, n_past_start={}",
		prompt_tokens.size(),
		conv_.n_past);

	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_prompt_on_gpu(std::move(prompt_tokens)),
		boost::cobalt::use_op);
	throw_if_shutdown_requested();

	std::string tokens_debugging;
	bool emitted_visible_text = false;
	int leading_control_tokens = 0;
	static constexpr int kMaxLeadingDuplexControlTokens = 32;

	int gen_idx = 0;
	int const max_generation_tokens = cfg_.inference.max_generation_tokens;
	for (; gen_idx < max_generation_tokens; ++gen_idx)
	{
		throw_if_shutdown_requested();
		SampledOutput const sampled =
			co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
				gpu_ex_.get_executor(),
				sample_and_decode_on_gpu(),
				boost::cobalt::use_op);
		throw_if_shutdown_requested();

		llama_token const tok = sampled.id;

		if (spdlog::should_log(spdlog::level::debug))
		{
			tokens_debugging += token_piece(vocab, tok, true);
		}

		if (llama_vocab_is_eog(vocab, tok))
		{
			spdlog::debug("session: duplex-context text turn — EOG at generation step {}", gen_idx);
			break;
		}

		if (tok == kTokenChunkEos)
		{
			if (!emitted_visible_text)
			{
				leading_control_tokens++;
				spdlog::debug(
					"session: duplex-context text turn — skipping leading chunk boundary token {} "
					"at generation step {}",
					tok,
					gen_idx);
				conv_.n_past++;
				if (leading_control_tokens >= kMaxLeadingDuplexControlTokens)
				{
					spdlog::warn(
						"session: duplex-context text turn — aborting after {} leading chunk "
						"boundary tokens without visible text",
						leading_control_tokens);
					break;
				}
				continue;
			}
			spdlog::debug(
				"session: duplex-context text turn — chunk boundary token {} at generation step {}",
				tok,
				gen_idx);
			conv_.n_past++;
			break;
		}

		if (tok == kTokenTurnEos)
		{
			if (!emitted_visible_text)
			{
				leading_control_tokens++;
				spdlog::debug(
					"session: duplex-context text turn — skipping leading turn boundary token {} "
					"at generation step {}",
					tok,
					gen_idx);
				conv_.n_past++;
				if (leading_control_tokens >= kMaxLeadingDuplexControlTokens)
				{
					spdlog::warn(
						"session: duplex-context text turn — aborting after {} leading turn "
						"boundary tokens without visible text",
						leading_control_tokens);
					break;
				}
				continue;
			}
			spdlog::debug(
				"session: duplex-context text turn — turn boundary token {} at generation step {}",
				tok,
				gen_idx);
			conv_.n_past++;
			break;
		}

		if (tok == kTokenListen)
		{
			if (!emitted_visible_text)
			{
				leading_control_tokens++;
				spdlog::debug(
					"session: duplex-context text turn — skipping leading listen token {} at "
					"generation step {}",
					tok,
					gen_idx);
				conv_.n_past++;
				if (leading_control_tokens >= kMaxLeadingDuplexControlTokens)
				{
					spdlog::warn(
						"session: duplex-context text turn — aborting after {} leading listen "
						"tokens without visible text",
						leading_control_tokens);
					break;
				}
				continue;
			}
			spdlog::debug(
				"session: duplex-context text turn — listen token {} at generation step {}",
				tok,
				gen_idx);
			conv_.n_past++;
			break;
		}

		if (tok == kTokenSpeak)
		{
			if (!emitted_visible_text)
			{
				leading_control_tokens++;
				spdlog::debug(
					"session: duplex-context text turn — skipping leading speak token {} at "
					"generation step {}",
					tok,
					gen_idx);
				conv_.n_past++;
				if (leading_control_tokens >= kMaxLeadingDuplexControlTokens)
				{
					spdlog::warn(
						"session: duplex-context text turn — aborting after {} leading speak "
						"tokens without visible text",
						leading_control_tokens);
					break;
				}
				continue;
			}
			spdlog::error(
				"session: duplex-context text turn - speak token {} at "
				"generation step {}",
				tok,
				gen_idx);
			conv_.n_past++;
			break;
		}

		if (tok == kTokenTtsPad)
		{
			if (!emitted_visible_text)
			{
				leading_control_tokens++;
				if (leading_control_tokens >= kMaxLeadingDuplexControlTokens)
				{
					spdlog::warn(
						"session: duplex-context text turn — aborting after {} leading TTS "
						"pad tokens without visible text",
						leading_control_tokens);
					break;
				}
			}
			conv_.n_past++;
			continue;
		}

		std::string piece = token_piece(vocab, tok);
		if (!piece.empty())
		{
			emitted_visible_text = true;
			throw_if_shutdown_requested();
			co_await output_ch_.write(OutFrame{TextOutFrame{std::move(piece)}});
		}

		conv_.n_past++;
	}

	auto close_tokens = do_tokenize(vocab, "<|im_end|>\n");
	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_prompt_on_gpu(std::move(close_tokens)),
		boost::cobalt::use_op);
	throw_if_shutdown_requested();

	spdlog::debug("session: duplex-context text turn tokens={}", tokens_debugging);

	co_return gen_idx < max_generation_tokens;
}

boost::cobalt::promise<void> Session::loop_video_in_ch(
	std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch)
{
	spdlog::debug("run_clip_loop: starting");

	static constexpr auto kMinEncodeInterval = std::chrono::seconds(1);
	auto last_encode_at = std::chrono::steady_clock::time_point{};

	try
	{
		while (true)
		{
			throw_if_shutdown_requested();
			/// Block until the next video frame arrives or the input channel closes.
			// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
			VideoFrame frame = co_await video_in_ch_.read();

			auto const now = std::chrono::steady_clock::now();
			if (now - last_encode_at < kMinEncodeInterval)
			{
				spdlog::debug(
					"run_clip_loop: throttling frame seq={} (within {}ms of last encode)",
					frame.seq,
					std::chrono::duration_cast<std::chrono::milliseconds>(now - last_encode_at)
						.count());
				continue;
			}

			std::uint32_t const seq = frame.seq;

			std::vector<std::byte> raw_for_ack = frame.raw;

			/// Run CLIP encode on the GPU executor (synchronous GGML/CUDA work).
			// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
			ClipResult result =
				co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
					gpu_ex_.get_executor(),
					encode_clip_on_gpu(seq, std::move(frame.raw)),
					boost::cobalt::use_op);
			throw_if_shutdown_requested();

			last_encode_at = std::chrono::steady_clock::now();

			/// Store the latest successful encode so the generation loop can
			/// consume it at the next audio-triggered turn (latest-wins).
			if (!result.embeddings.empty())
			{
				co_await merged_ch->write(
					DuplexEvent{PendingClipResult{
						.clip = std::move(result), .seq = seq, .raw = std::move(raw_for_ack)}});
			}
		}
	}
	catch (boost::system::system_error const & ex)
	{
		spdlog::debug("run_clip_loop: channel error ({}), exiting", ex.what());
	}

	// Close merged_ch so the main loop wakes if no other feeder did so first.
	if (merged_ch->is_open())
	{
		merged_ch->close();
	}
}

boost::cobalt::promise<bool> Session::evict_oldest_units()
{
	int const used = conv_.n_past;
	int const total = cfg_.inference.n_ctx;
	int const free_now = total - used;
	int const target_free = cfg_.inference.overflow_reserve;

	if (free_now >= target_free)
	{
		co_return true;	 // Already enough space.
	}

	int need_to_free = target_free - free_now;
	int evict_end = conv_.sys_prompt_end;
	int blocks_evicted = 0;

	// Walk unit_history from oldest to newest, accumulating blocks to evict.
	for (std::size_t idx = 0; idx < conv_.unit_history.size() && need_to_free > 0; ++idx)
	{
		auto const & block = conv_.unit_history[idx];
		int const block_size = block.pos_end - block.pos_start;
		evict_end = block.pos_end;
		need_to_free -= block_size;
		++blocks_evicted;
	}

	if (blocks_evicted == 0)
	{
		spdlog::warn("session: KV cache full but no completed unit blocks to evict");
		co_return false;
	}

	int const evict_start = conv_.sys_prompt_end;
	int const delta = evict_end - evict_start;

	spdlog::info(
		"session: evicting {} unit blocks, freeing {} KV positions [{}, {})",
		blocks_evicted,
		delta,
		evict_start,
		evict_end);

	// Perform KV cache surgery on the GPU executor.
	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		evict_kv_on_gpu(evict_start, evict_end, delta),
		boost::cobalt::use_op);

	// 3. Update bookkeeping.
	conv_.n_past -= delta;
	current_unit_start_ -= delta;

	// 4. Remove evicted blocks from history and adjust remaining block positions.
	conv_.unit_history.erase(
		conv_.unit_history.begin(), conv_.unit_history.begin() + blocks_evicted);
	for (auto & block : conv_.unit_history)
	{
		block.pos_start -= delta;
		block.pos_end -= delta;
	}

	// sys_prompt_end does not change — we only evicted AFTER it.

	spdlog::info(
		"session: after eviction: n_past={}, unit_history.size()={}, free={}",
		conv_.n_past,
		conv_.unit_history.size(),
		total - conv_.n_past);

	co_return(total - conv_.n_past) >= target_free;
}

boost::cobalt::task<void> Session::evict_kv_on_gpu(int evict_start, int evict_end, int delta)
{
	// Runs on gpu_ex — synchronous GGML/CUDA memory operations.
	auto * mem = llama_get_memory(ctx_.get());

	// 1. Remove the evicted range.
	bool const removal_ok = llama_memory_seq_rm(mem, 0, evict_start, evict_end);
	if (!removal_ok)
	{
		spdlog::error("session: llama_memory_seq_rm failed for [{}, {})", evict_start, evict_end);
		co_return;
	}

	// 2. Shift all remaining positions down by delta.
	//    The range [evict_end, -1) means "from evict_end to infinity".
	llama_memory_seq_add(mem, 0, evict_end, -1, -delta);

	co_return;
}

boost::cobalt::task<void> Session::init_gpu_resources()
{
	// Runs on gpu_ex — llama calls are synchronous GPU operations.
	ctx_ = models_.create_context(cfg_.inference.n_ctx);
	if (ctx_ == nullptr)
	{
		throw std::runtime_error{"Session::init_gpu_resources: create_context failed"};
	}

	/// `llama_sampler_chain_init` creates an empty sampler chain.
	/// Sub-samplers are added in evaluation order.
	auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

	/// Penalize recently sampled tokens to reduce degenerate loops such as
	/// repeated `<|listen|>` decisions across consecutive duplex turns.
	/// Keep this active for both greedy and temperature-based sampling so
	/// deterministic test configs still exercise the same anti-loop guard.
	llama_sampler_chain_add(
		chain,
		llama_sampler_init_penalties(
			cfg_.inference.repeat_penalty_last_n,
			cfg_.inference.repeat_penalty,
			/*penalty_freq=*/0.0F,
			/*penalty_present=*/0.0F));

	if (cfg_.inference.temperature <= 0.0)
	{
		/// Greedy sampler: always selects the single highest-probability token.
		/// Used when `temperature ≤ 0` to produce deterministic output — useful for
		/// tests that need reliable model behaviour regardless of random seed.
		llama_sampler_chain_add(chain, llama_sampler_init_greedy());
	}
	else
	{
		/// Temperature: scale logits by 1/temperature before softmax.
		/// Values < 1.0 → sharper/more deterministic; > 1.0 → more random.
		llama_sampler_chain_add(chain, llama_sampler_init_temp(cfg_.inference.temperature));

		/// Distribution sampler: draw from the softmax probabilities.
		/// `LLAMA_DEFAULT_SEED` provides a fixed seed for reproducibility.
		llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
	}

	sampler_.reset(chain);
	co_return;
}

boost::cobalt::task<void> Session::free_gpu_resources()
{
	sampler_.reset();
	ctx_.reset();
	co_return;
}

// ── Duplex system prompt ──────────────────────────────────────────────────────

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses ctx_
// and conv_
boost::cobalt::task<void> Session::load_prompt_cache_on_gpu(std::filesystem::path path)
{
	// Runs on gpu_ex.
	if (ctx_ == nullptr)
	{
		throw std::logic_error{"load_prompt_cache_on_gpu: ctx_ is null"};
	}

	size_t n_token_count = 0;

	/// `llama_state_load_file` restores the full KV cache state from a file
	/// written by `llama_state_save_file`.  `tokens_out = nullptr` / capacity 0
	/// tells libllama to skip re-building the token list (positions come from
	/// the KV cache query below).
	bool const load_ok =
		llama_state_load_file(ctx_.get(), path.c_str(), nullptr, 0, &n_token_count);

	if (!load_ok)
	{
		throw std::runtime_error{std::format(
			"load_prompt_cache_on_gpu: llama_state_load_file failed for '{}'", path.string())};
	}

	/// `llama_memory_seq_pos_max` returns the highest KV position occupied by
	/// sequence 0 after the state is restored.  `n_past = max_pos + 1` because
	/// positions are zero-based and the next token would go at that position.
	llama_pos const max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_.get()), /*seq_id=*/0);

	if (max_pos < 0)
	{
		throw std::runtime_error{
			"load_prompt_cache_on_gpu: llama_memory_seq_pos_max returned -1 "
			"(KV cache empty after load?)"};
	}

	conv_.n_past = static_cast<int>(max_pos) + 1;
	conv_.sys_prompt_end = conv_.n_past;
	co_return;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses ctx_
boost::cobalt::task<void> Session::save_prompt_cache_on_gpu(std::filesystem::path path)
{
	// Runs on gpu_ex.
	if (ctx_ == nullptr)
	{
		throw std::logic_error{"save_prompt_cache_on_gpu: ctx_ is null"};
	}

	/// `llama_state_save_file` serialises the full KV cache to disk.
	/// `tokens = nullptr` / count 0: we do not store the token list alongside
	/// the KV state (the session prompt is deterministic from the config).
	bool const save_ok = llama_state_save_file(ctx_.get(), path.c_str(), nullptr, 0);
	if (!save_ok)
	{
		// Non-fatal: warn and continue without a cache.
		spdlog::warn(
			"save_prompt_cache_on_gpu: llama_state_save_file failed for '{}'", path.string());
	}
	co_return;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses
// models_
boost::cobalt::task<std::vector<float>> Session::encode_voice_on_gpu(std::filesystem::path wav_path)
{
	// Runs on gpu_ex — audition_audio_encode dispatches GGML/CUDA work.
	co_return encode_reference_voice(&models_.audio_ctx(), wav_path);
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses
// models_
boost::cobalt::task<void> Session::clear_audio_encoder_kv_on_gpu()
{
	if (cfg_.model.audio_path.empty())
	{
		co_return;
	}

	spdlog::debug("session: clearing shared audio encoder KV state");
	audition_whisper_clear_kv_cache(&models_.audio_ctx());
	co_return;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses ctx_
// and conv_
boost::cobalt::task<void> Session::decode_embeddings_on_gpu(
	std::vector<float> embeddings,
	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- semantically distinct
	int n_embd,
	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- semantically distinct
	int n_past_start,
	bool need_logits)
{
	// Runs on gpu_ex.
	if (ctx_ == nullptr)
	{
		throw std::logic_error{"decode_embeddings_on_gpu: ctx_ is null"};
	}

	int const n_tokens = static_cast<int>(embeddings.size()) / n_embd;

	/// `llama_batch_init(n_tokens, embd, n_seq_max)` — when `embd > 0` it
	/// allocates `batch.embd` (n_tokens * embd floats) instead of `batch.token`
	/// (int32 IDs).  The seq_id arrays are also allocated and must be filled.
	llama_batch batch = llama_batch_init(n_tokens, n_embd, /*n_seq_max=*/1);
	batch.n_tokens = n_tokens;

	/// Copy the embedding float buffer into the batch.  `memcpy` is safe here:
	/// both src and dst are plain float arrays of identical byte size.
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- batch.embd is a raw C
	// array from llama_batch_init
	std::memcpy(batch.embd, embeddings.data(), embeddings.size() * sizeof(float));

	for (int j = 0; j < n_tokens; ++j)
	{
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.pos[j] = static_cast<llama_pos>(n_past_start + j);
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.n_seq_id[j] = 1;
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.seq_id[j][0] = 0;
		/// Enable logits only at the last position, and only when the caller
		/// is about to sample immediately after this decode.
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.logits[j] = (need_logits && j == n_tokens - 1) ? kLogitEnabled : kLogitDisabled;
	}

	/// Forward pass: writes audio embeddings into the KV cache at positions
	/// [n_past_start, n_past_start + n_tokens).
	int const ret = llama_decode(ctx_.get(), batch);
	llama_batch_free(batch);

	if (ret != 0)
	{
		throw std::runtime_error{
			std::format("decode_embeddings_on_gpu: llama_decode returned {}", ret)};
	}

	conv_.n_past = n_past_start + n_tokens;
	co_return;
}

boost::cobalt::promise<void> Session::init_duplex_system_prompt()
{
	// Guard: skip entirely if audio encoder or reference WAV is not configured.
	if (cfg_.model.audio_path.empty() || cfg_.voice.reference_wav.empty())
	{
		throw std::logic_error{"session: audio encoder or reference WAV not configured"};
	}

	// Cache hit: restore KV state from disk and skip the expensive encode step.
	std::filesystem::path const cache_path{cfg_.prompts.system_prompt_cache_path};
	if (!cache_path.empty() && std::filesystem::exists(cache_path))
	{
		spdlog::info("session: loading prompt cache from '{}'", cache_path.string());
		co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
			gpu_ex_.get_executor(),
			load_prompt_cache_on_gpu(cache_path),
			boost::cobalt::use_op);
		spdlog::info("session: prompt cache loaded, sys_prompt_end={}", conv_.sys_prompt_end);
		co_return;
	}

	// Full encode path ─────────────────────────────────────────────────────────

	spdlog::info("session: encoding reference voice from '{}'", cfg_.voice.reference_wav);

	// NOLINTNEXTLINE(cppcoreguidelines-init-variables) -- co_await result cannot use brace-init
	std::vector<float> embeddings =
		co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
			gpu_ex_.get_executor(),
			encode_voice_on_gpu(cfg_.voice.reference_wav),
			boost::cobalt::use_op);

	if (embeddings.empty())
	{
		throw std::logic_error{std::format(
			"session: failed to encode reference voice from '{}'", cfg_.voice.reference_wav)};
	}

	// `encode_reference_voice` clears the shared Whisper KV before encoding, but
	// the encode itself advances the streaming state. Clear again so the first
	// live user utterance starts from a fresh audio-encoder position instead of
	// continuing from the reference voice.
	co_await boost::cobalt::spawn(
		gpu_ex_.get_executor(), clear_audio_encoder_kv_on_gpu(), boost::cobalt::use_op);

	/// `audition_n_mmproj_embd` reads a constant from the loaded model params;
	/// safe to call from the main executor (read-only, no CUDA dispatch).
	int const n_embd = audition_n_mmproj_embd(&models_.audio_ctx());

	// Prefill text prefix: "<|im_start|>system\n{duplex_system}\n<|audio_start|>"
	std::string const prefix =
		std::format("<|im_start|>system\n{}\n<|audio_start|>", cfg_.prompts.duplex_system);
	auto prefix_tokens = do_tokenize(&models_.vocab(), prefix);

	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_prompt_on_gpu(std::move(prefix_tokens)),
		boost::cobalt::use_op);

	// Prefill audio embeddings (n_past updated by decode_embeddings_on_gpu).
	int const n_past_before_audio = conv_.n_past;
	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_embeddings_on_gpu(std::move(embeddings), n_embd, n_past_before_audio),
		boost::cobalt::use_op);

	// Prefill text suffix: "<|audio_end|><|im_end|>\n"
	auto suffix_tokens = do_tokenize(&models_.vocab(), "<|audio_end|><|im_end|>\n");
	co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
		gpu_ex_.get_executor(),
		decode_prompt_on_gpu(std::move(suffix_tokens)),
		boost::cobalt::use_op);

	co_await seed_listen_turns();

	// Record the first evictable KV position (system prompt is never removed).
	conv_.sys_prompt_end = conv_.n_past;
	spdlog::info(
		"session: duplex system prompt prefilled, sys_prompt_end={}", conv_.sys_prompt_end);

	// Save KV state to disk for fast startup next time.
	if (!cache_path.empty())
	{
		spdlog::info("session: saving prompt cache to '{}'", cache_path.string());
		co_await boost::cobalt::spawn(  // NOLINT(readability-static-accessed-through-instance)
			gpu_ex_.get_executor(),
			save_prompt_cache_on_gpu(cache_path),
			boost::cobalt::use_op);
	}
}

boost::cobalt::promise<void> Session::seed_listen_turns()
{
	spdlog::debug(
		"session: seeding {} empty listen turns.", cfg_.inference.duplex_force_listen_count);

	static constexpr float kNearZeroAmplitude = 1.0e-6F;
	static std::vector<float> const short_silence(/* 1 sec = */ AudioConfig::input_sample_rate,
												  kNearZeroAmplitude);

	for (int i = 0; i < cfg_.inference.duplex_force_listen_count; ++i)
	{
		co_await boost::cobalt::spawn(
			gpu_ex_.get_executor(), decode_prompt_on_gpu({kTokenUnit}), boost::cobalt::use_op);

		co_await boost::cobalt::spawn(
			gpu_ex_.get_executor(),
			batch_decode_audio_on_gpu(short_silence, false),
			boost::cobalt::use_op);

		co_await boost::cobalt::spawn(
			gpu_ex_.get_executor(),
			decode_prompt_on_gpu({kTokenListen, kTokenUnitEnd}),
			boost::cobalt::use_op);
	}
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: coroutine
// accesses ctx_ and conv_ members
boost::cobalt::task<void> Session::decode_prompt_on_gpu(std::vector<llama_token> tokens)
{
	// Runs on gpu_ex — all llama calls are synchronous.
	if (ctx_ == nullptr)
	{
		throw std::logic_error{"decode_prompt_on_gpu: ctx_ is null"};
	}

	auto const n_toks = static_cast<int32_t>(tokens.size());

	// KV positions: prompt tokens occupy [n_past, n_past + n_toks).
	std::vector<llama_pos> positions(tokens.size());
	std::ranges::iota(positions, conv_.n_past);

	// Request logits only from the last token — we sample from that position.
	std::vector logits_flags(tokens.size(), kLogitDisabled);
	logits_flags.back() = kLogitEnabled;

	/// `llama_batch` is a plain C struct; we set all fields manually.
	/// `n_seq_id` / `seq_id` left as nullptr → libllama defaults to sequence 0.
	/// `embd = nullptr` → tokens are IDs, not float embeddings.
	llama_batch batch{};
	batch.n_tokens = n_toks;
	batch.token = tokens.data();
	batch.embd = nullptr;
	batch.pos = positions.data();
	batch.n_seq_id = nullptr;
	batch.seq_id = nullptr;
	batch.logits = logits_flags.data();

	/// Forward pass: updates the KV cache for all `n_toks` positions and
	/// writes output logits at the last position.  Returns 0 on success.
	int const ret = llama_decode(ctx_.get(), batch);
	if (ret != 0)
	{
		throw std::runtime_error{
			std::format("decode_prompt_on_gpu: llama_decode returned {}", ret)};
	}

	conv_.n_past += static_cast<int>(tokens.size());
	co_return;
}

boost::cobalt::task<SampledOutput> Session::sample_and_decode_on_gpu()
{
	// Runs on gpu_ex — synchronous GPU calls.
	if (ctx_ == nullptr || sampler_ == nullptr)
	{
		throw std::logic_error{"sample_and_decode_on_gpu: ctx_ or sampler_ is null"};
	}

	spdlog::debug("sample_and_decode_on_gpu: starting (n_past={})", conv_.n_past);

	/// Sample from the logits at the last decoded position (idx = -1).
	/// The sampler chain applies temperature scaling then draws from the
	/// resulting distribution.
	float * const logits_data = llama_get_logits_ith(ctx_.get(), -1);
	if (logits_data != nullptr)
	{
		std::size_t const logits_size = llama_vocab_n_tokens(&models_.vocab());
		std::span const logits(logits_data, logits_size);
		if (cfg_.inference.listen_prob_scale != 1.0F)
		{
			gsl::at(logits, kTokenListen) *= cfg_.inference.listen_prob_scale;
		}
		if (cfg_.inference.turn_eos_prob_scale != 1.0F)
		{
			gsl::at(logits, kTokenTurnEos) *= cfg_.inference.turn_eos_prob_scale;
		}
		if (cfg_.inference.chunk_eos_prob_scale != 1.0F)
		{
			gsl::at(logits, kTokenChunkEos) *= cfg_.inference.chunk_eos_prob_scale;
		}
	}

	llama_token const sampled_tok = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);

	/// Record the chosen token in the sampler's internal state (e.g. for
	/// repetition-penalty tracking).
	llama_sampler_accept(sampler_.get(), sampled_tok);

	// EOS: no further decode step needed — return with empty hidden state.
	if (llama_vocab_is_eog(&models_.vocab(), sampled_tok))
	{
		co_return SampledOutput{.id = sampled_tok, .hidden = {}};
	}

	// Decode the sampled token to update the KV cache for the next step.
	// We need a non-const lvalue for batch.token (llama C API takes llama_token*).
	llama_token mutable_tok{sampled_tok};
	llama_pos tok_pos{static_cast<llama_pos>(conv_.n_past)};
	int8_t logit_flag{kLogitEnabled};

	llama_batch batch{};
	batch.n_tokens = 1;
	batch.token = &mutable_tok;
	batch.embd = nullptr;
	batch.pos = &tok_pos;
	batch.n_seq_id = nullptr;
	batch.seq_id = nullptr;
	batch.logits = &logit_flag;

	int const ret = llama_decode(ctx_.get(), batch);
	if (ret != 0)
	{
		throw std::runtime_error{
			std::format("sample_and_decode_on_gpu: llama_decode returned {}", ret)};
	}

	/// Extract the hidden state at the last decoded position (index -1).
	/// `ctx_params.embeddings = true` was set at context creation, so this
	/// pointer is always valid after a successful `llama_decode`.  The data
	/// is owned by the context and must be copied before the next decode.
	int const n_embd = llama_model_n_embd(&models_.model());
	std::vector<float> hidden;
	if (float const * const emb_ptr = llama_get_embeddings_ith(ctx_.get(), -1); emb_ptr != nullptr)
	{
		std::span<float const> const emb_span{emb_ptr, static_cast<std::size_t>(n_embd)};
		hidden.assign(emb_span.begin(), emb_span.end());
	}

	spdlog::debug(
		"sample_and_decode_on_gpu: finished token={} (n_past={})", sampled_tok, conv_.n_past);

	co_return SampledOutput{.id = sampled_tok, .hidden = std::move(hidden)};
}

boost::cobalt::task<std::vector<float>> Session::synthesize_speech_on_gpu(
	std::vector<llama_token> token_ids, std::vector<float> hiddens)
{
	// Runs on gpu_ex — synchronous GPU calls.
	if (cfg_.model.tts_transformer_path.empty())
	{
		co_return std::vector<float>{};
	}
	TtsPipeline & tts = models_.tts_pipeline();
	if (!tts.is_loaded())
	{
		co_return std::vector<float>{};
	}

	/// Build TTS condition: filter special tokens, project hiddens, merge emb_text.
	int const n_embd = llama_model_n_embd(&models_.model());
	std::vector<float> const condition =
		tts.build_condition(std::move(token_ids), std::move(hiddens), n_embd);

	if (condition.empty())
	{
		spdlog::debug("synthesize_speech_on_gpu: empty condition after filtering");
		co_return std::vector<float>{};
	}

	/// Synthesize: clear TTS KV, prefill condition, sample audio tokens, run Token2Wav.
	co_return tts.synthesize(condition);
}

boost::cobalt::promise<std::optional<SpeechChunkOutFrame>> Session::synthesize_speech_chunk(
	std::string text, std::vector<llama_token> token_buf, std::vector<float> hidden_buf)
{
	if (text.empty())
	{
		co_return std::nullopt;
	}

	if (token_buf.empty())
	{
		co_return std::nullopt;
	}

	auto pcm = co_await boost::cobalt::spawn(
		gpu_ex_.get_executor(),
		synthesize_speech_on_gpu(std::move(token_buf), std::move(hidden_buf)),
		boost::cobalt::use_op);

	if (pcm.empty())
	{
		spdlog::warn("emit_speech_chunk: TTS returned no PCM for text '{}'", text);
		co_return std::nullopt;
	}

	if (!cfg_.debug.audio_output_wav.empty())
	{
		debug_turn_audio_.insert(debug_turn_audio_.end(), pcm.begin(), pcm.end());
	}

	std::uint32_t const seq = next_speech_chunk_seq_++;

	spdlog::debug("session: created speech chunk seq={}", seq);
	SpeechChunkOutFrame chunk{.seq = seq, .text = std::move(text), .pcm = std::move(pcm)};
	co_return chunk;
}

void Session::flush_debug_audio_wav()
{
	if (cfg_.debug.audio_output_wav.empty() || debug_turn_audio_.empty())
	{
		return;
	}

	std::filesystem::path const prefix_path{cfg_.debug.audio_output_wav};
	std::filesystem::path const parent =
		prefix_path.has_parent_path() ? prefix_path.parent_path() : std::filesystem::current_path();
	std::string const stem =
		prefix_path.stem().empty() ? prefix_path.filename().string() : prefix_path.stem().string();
	std::filesystem::path const out_path = parent /
		std::format("{}_{}_turn_{:04}.wav",
					stem.empty() ? "tts" : stem,
					conv_.session_id,
					++debug_turn_index_);

	try
	{
		std::filesystem::create_directories(parent);
		write_wav_mono(out_path, debug_turn_audio_, kTtsOutputSampleRate);
		spdlog::debug(
			"session: wrote {} debug audio samples to '{}'",
			debug_turn_audio_.size(),
			out_path.string());
	}
	catch (std::runtime_error const & ex)
	{
		spdlog::warn("session: failed to write debug WAV '{}': {}", out_path.string(), ex.what());
	}
	debug_turn_audio_.clear();
}

// ── Buffered audio + duplex loop ──────────────────────────────────────────────

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
boost::cobalt::promise<void> Session::loop_audio_in_ch(
	std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch)
{
	try
	{
		std::deque<float> pending_pcm;
		std::size_t const input_audio_chunk_samples =
			(static_cast<std::size_t>(kInputAudioSampleRateHz) *
			 static_cast<std::size_t>(cfg_.inference.speech_chunk_ms)) /
			static_cast<std::size_t>(kMillisPerSecond);

		while (true)
		{
			/// Read the next PCM chunk from the audio input channel. The channel
			/// carries mono float32 samples at 16 kHz; the buffered duplex unit
			/// size is configurable and defaults to 1000 ms.
			std::vector<float> pcm = co_await audio_in_ch_.read();
			num_pending_audio_samples_ = num_pending_audio_samples_ > pcm.size()
				? num_pending_audio_samples_ - pcm.size()
				: 0U;

			pending_pcm.append_range(std::move(pcm));

			while (pending_pcm.size() >= input_audio_chunk_samples)
			{
				std::vector<float> chunk;
				chunk.reserve(input_audio_chunk_samples);
				for (std::size_t idx = 0; idx < input_audio_chunk_samples; ++idx)
				{
					chunk.push_back(pending_pcm.front());
					pending_pcm.pop_front();
				}

				int const duration_ms =
					(static_cast<int>(chunk.size()) * kMillisPerSecond) / kInputAudioSampleRateHz;

				spdlog::debug(
					"session: buffering {}ms input audio for unconditional duplex generate",
					duration_ms);
				co_await merged_ch->write(
					DuplexEvent{
						AudioChunkEvent{.pcm = std::move(chunk), .duration_ms = duration_ms}});
			}
		}
	}
	catch (boost::system::system_error const & ex)
	{
		// audio_in_ch_ was closed — session is shutting down.
		spdlog::debug("loop_audio_in_ch_buffered: channel closed ({}), exiting", ex.what());
	}
	// Close merged_ch so the main loop wakes from its read() and exits.
	// This is a coordinated shutdown: the last feeder to exit closes the channel.
	// The other feeder will get broken_pipe on its next write and also exit.
	if (merged_ch->is_open())
	{
		merged_ch->close();
	}
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
boost::cobalt::promise<void> Session::loop_text_in_ch(
	std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch)
{
	try
	{
		while (true)
		{
			/// Read the next text input from text_in_ch_ and forward into merged_ch.
			TextInput text_ev = co_await text_in_ch_.read();
			spdlog::debug("loop_text_in_ch: forwarding TextInput into duplex loop");
			co_await merged_ch->write(DuplexEvent{std::move(text_ev)});
		}
	}
	catch (boost::system::system_error const & ex)
	{
		// text_in_ch_ or merged_ch closed — exit cleanly (normal shutdown path).
		spdlog::debug("loop_text_in_ch: channel closed ({}), exiting", ex.what());
	}
	// Close merged_ch so the main loop wakes if no other feeder did so first.
	if (merged_ch->is_open())
	{
		merged_ch->close();
	}
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
boost::cobalt::promise<void> Session::loop_speech_ack_ch(
	std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch)
{
	try
	{
		while (true)
		{
			/// Read the next text input from text_in_ch_ and forward into merged_ch.
			SpeechChunkAckEvent const event = co_await speech_ack_ch_.read();
			spdlog::debug("loop_speech_ack_ch: forwarding SpeechChunkAckEvent into duplex loop");
			co_await merged_ch->write(DuplexEvent{event});
		}
	}
	catch (boost::system::system_error const & ex)
	{
		// text_in_ch_ or merged_ch closed — exit cleanly (normal shutdown path).
		spdlog::debug("loop_text_in_ch: channel closed ({}), exiting", ex.what());
	}
	// Close merged_ch so the main loop wakes if no other feeder did so first.
	if (merged_ch->is_open())
	{
		merged_ch->close();
	}
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses
// models_, ctx_, conv_
boost::cobalt::task<void> Session::batch_decode_audio_on_gpu(
	std::vector<float> pcm, bool need_logits)
{
	// Runs on gpu_ex.
	throw_if_shutdown_requested();
	if (cfg_.model.audio_path.empty())
	{
		spdlog::warn("batch_decode_audio_on_gpu: no audio context — skipping");
		co_return;
	}

	// Align samples to 1600-sample boundary for Whisper.
	if (pcm.size() % kInputAudioEmbeddingAlignmentSamples != 0)
	{
		std::size_t const aligned_size = kInputAudioEmbeddingAlignmentSamples *
			(1 + pcm.size() / kInputAudioEmbeddingAlignmentSamples);
		spdlog::debug("batch_decode_audio_on_gpu: aligning {} to {}", pcm.size(), aligned_size);
		pcm.resize(aligned_size, 0);
	}

	spdlog::debug("batch_decode_audio_on_gpu: starting audio encoder for {} samples", pcm.size());
	/// `encode_pcm_audio` runs the Whisper Mel preprocessor and encoder on the
	/// GPU, producing `[n_tokens × n_embd]` float embeddings from raw PCM.
	std::vector<float> embeddings = encode_pcm_audio(&models_.audio_ctx(), pcm);
	spdlog::debug(
		"batch_decode_audio_on_gpu: audio encoder finished with {} floats", embeddings.size());
	throw_if_shutdown_requested();

	/// `audition_n_mmproj_embd` returns the embedding dimension (4096) from
	/// the loaded audio model; must match the LLM's hidden size.
	int const n_embd = audition_n_mmproj_embd(&models_.audio_ctx());
	int const n_tokens = static_cast<int>(embeddings.size()) / n_embd;

	spdlog::debug("batch_decode_audio_on_gpu: n_tokens={} n_past_start={}", n_tokens, conv_.n_past);

	/// `llama_batch_init(n_tokens, embd, n_seq_max)` — `embd > 0` allocates
	/// `batch.embd` for float embedding input instead of integer token IDs.
	llama_batch batch = llama_batch_init(n_tokens, n_embd, /*n_seq_max=*/1);
	batch.n_tokens = n_tokens;

	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	std::memcpy(batch.embd, embeddings.data(), embeddings.size() * sizeof(float));

	for (int j = 0; j < n_tokens; ++j)
	{
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.pos[j] = static_cast<llama_pos>(conv_.n_past + j);
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.n_seq_id[j] = 1;
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.seq_id[j][0] = 0;
		/// Enable logits only at the last position, and only when the caller
		/// is about to sample immediately after this audio chunk. Requesting
		/// logits on pure prefill chunks wastes GPU bandwidth without benefit.
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		batch.logits[j] = (need_logits && j == n_tokens - 1) ? kLogitEnabled : kLogitDisabled;
	}

	spdlog::debug("batch_decode_audio_on_gpu: entering llama_decode for {} tokens", n_tokens);
	int const ret = llama_decode(ctx_.get(), batch);
	llama_batch_free(batch);
	spdlog::debug("batch_decode_audio_on_gpu: llama_decode returned {}", ret);
	throw_if_shutdown_requested();

	if (ret != 0)
	{
		throw std::runtime_error{
			std::format("batch_decode_audio_on_gpu: llama_decode returned {}", ret)};
	}

	conv_.n_past += n_tokens;
	conv_.last_heard_token_pos = conv_.n_past;
	spdlog::debug("batch_decode_audio_on_gpu: finished (n_past={})", conv_.n_past);
	co_return;
}

// ── CLIP video encode loop ────────────────────────────────────────────────────

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses
// models_
boost::cobalt::task<ClipResult> Session::encode_clip_on_gpu(
	std::uint32_t seq, std::vector<std::byte> raw)
{
	// Runs on gpu_ex.
	if (cfg_.model.vision_path.empty())
	{
		spdlog::warn("encode_clip_on_gpu: no vision model configured — skipping frame seq={}", seq);
		co_return ClipResult{};
	}
	clip_ctx & vision_ctx = models_.vision_ctx();

	spdlog::debug("encode_clip_on_gpu: encoding frame seq={}", seq);

	/// `clip_image_u8_init` allocates an empty pixel buffer.  We fill it via
	/// `clip_build_img_from_pixels` which copies the RGB data and sets nx/ny.
	clip_image_u8 * const img_u8 = clip_image_u8_init();

	/// Input frames are pre-resized by the client to 448×448 (the CLIP model's
	/// native resolution for MiniCPM-o 4.5) so we do not resize here.
	static constexpr int kClipImageSize = 448;

	/// `clip_build_img_from_pixels` takes `unsigned char*`; our buffer is `std::byte*`.
	/// Both types are required by the standard to have the same representation and size.
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
	auto const * rgb_pixels = reinterpret_cast<unsigned char const *>(raw.data());
	clip_build_img_from_pixels(rgb_pixels, kClipImageSize, kClipImageSize, img_u8);

	/// `clip_image_preprocess` normalises pixel values, tiles the image (MiniCPM
	/// uses dynamic high-resolution tiling), and writes f32 tile batches into
	/// `f32_batch`.  Returns false if the model cannot process this image.
	clip_image_f32_batch * const f32_batch = clip_image_f32_batch_init();

	bool const preprocess_ok = clip_image_preprocess(&vision_ctx, img_u8, f32_batch);
	clip_image_u8_free(img_u8);

	if (!preprocess_ok)
	{
		clip_image_f32_batch_free(f32_batch);
		spdlog::error("encode_clip_on_gpu: clip_image_preprocess failed for seq={}", seq);
		co_return ClipResult{};
	}

	/// Sum output tokens across all tiles — tiled models produce more tokens
	/// than non-tiled (MiniCPM-o 4.5 uses 64 tokens per tile).
	int n_tokens = 0;
	int const n_tiles = static_cast<int>(clip_image_f32_batch_n_images(f32_batch));
	for (int i = 0; i < n_tiles; ++i)
	{
		n_tokens += clip_n_output_tokens(&vision_ctx, clip_image_f32_get_img(f32_batch, i));
	}

	/// `clip_n_mmproj_embd` returns the projection output dimension, which must
	/// match the LLM's hidden size (4096 for MiniCPM-o 4.5).
	int const n_embd = clip_n_mmproj_embd(&vision_ctx);

	std::vector<float> embeddings(static_cast<std::size_t>(n_tokens * n_embd));

	/// `clip_image_batch_encode` runs the CLIP vision Transformer for all tiles
	/// and writes the projected [n_tokens × n_embd] float embeddings into `vec`.
	bool const encode_ok = clip_image_batch_encode(
		&vision_ctx, cfg_.inference.clip_num_threads, f32_batch, embeddings.data());
	clip_image_f32_batch_free(f32_batch);

	if (!encode_ok)
	{
		spdlog::error("encode_clip_on_gpu: clip_image_batch_encode failed for seq={}", seq);
		co_return ClipResult{};
	}

	spdlog::debug(
		"encode_clip_on_gpu: seq={} n_tiles={} n_tokens={} n_embd={}",
		seq,
		n_tiles,
		n_tokens,
		n_embd);

	co_return ClipResult{.seq = seq, .n_embd = n_embd, .embeddings = std::move(embeddings)};
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static) -- false positive: accesses
// ctx_ and conv_
boost::cobalt::task<void> Session::batch_decode_image_on_gpu(ClipResult clip)
{
	// Runs on gpu_ex.
	if (ctx_ == nullptr)
	{
		throw std::logic_error{"batch_decode_image_on_gpu: ctx_ is null"};
	}

	int const n_tokens = static_cast<int>(clip.embeddings.size()) / clip.n_embd;
	spdlog::debug(
		"batch_decode_image_on_gpu: seq={} n_tokens={} n_past_start={}",
		clip.seq,
		n_tokens,
		conv_.n_past);

	// ── Step 1: decode <image> token (no logits) ─────────────────────────────
	{
		/// Single-token batch for the `<image>` marker.  `n_seq_id = nullptr` /
		/// `seq_id = nullptr` → libllama assigns sequence ID 0 by default.
		llama_token mutable_tok{kTokenImage};
		llama_pos tok_pos{static_cast<llama_pos>(conv_.n_past)};
		int8_t logit_flag{kLogitDisabled};

		llama_batch batch{};
		batch.n_tokens = 1;
		batch.token = &mutable_tok;
		batch.embd = nullptr;
		batch.pos = &tok_pos;
		batch.n_seq_id = nullptr;
		batch.seq_id = nullptr;
		batch.logits = &logit_flag;

		int const ret = llama_decode(ctx_.get(), batch);
		if (ret != 0)
		{
			throw std::runtime_error{
				std::format("batch_decode_image_on_gpu: <image> decode returned {}", ret)};
		}
		conv_.n_past++;
	}

	// ── Step 2: decode CLIP embeddings (no logits) ───────────────────────────
	{
		/// Embedding batch: `embd > 0` tells libllama to use `batch.embd` (float
		/// vectors) rather than `batch.token` (integer IDs).
		llama_batch batch = llama_batch_init(n_tokens, clip.n_embd, /*n_seq_max=*/1);
		batch.n_tokens = n_tokens;

		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		std::memcpy(batch.embd, clip.embeddings.data(), clip.embeddings.size() * sizeof(float));

		for (int j = 0; j < n_tokens; ++j)
		{
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			batch.pos[j] = static_cast<llama_pos>(conv_.n_past + j);
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			batch.n_seq_id[j] = 1;
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			batch.seq_id[j][0] = 0;
			/// No logits needed for embedding positions — we sample only after `</image>`.
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			batch.logits[j] = kLogitDisabled;
		}

		int const ret = llama_decode(ctx_.get(), batch);
		llama_batch_free(batch);

		if (ret != 0)
		{
			throw std::runtime_error{
				std::format("batch_decode_image_on_gpu: embeddings decode returned {}", ret)};
		}

		conv_.n_past += n_tokens;
	}

	// ── Step 3: decode </image> token (logits enabled for immediate sampling) ──
	{
		/// Logits enabled at `</image>` so that `run_duplex_turn()` can sample
		/// the first token without an extra decode step.
		llama_token mutable_tok{kTokenImageEnd};
		llama_pos tok_pos{static_cast<llama_pos>(conv_.n_past)};
		int8_t logit_flag{kLogitEnabled};

		llama_batch batch{};
		batch.n_tokens = 1;
		batch.token = &mutable_tok;
		batch.embd = nullptr;
		batch.pos = &tok_pos;
		batch.n_seq_id = nullptr;
		batch.seq_id = nullptr;
		batch.logits = &logit_flag;

		int const ret = llama_decode(ctx_.get(), batch);
		if (ret != 0)
		{
			throw std::runtime_error{
				std::format("batch_decode_image_on_gpu: </image> decode returned {}", ret)};
		}
		conv_.n_past++;
	}

	last_video_decode_at_ = std::chrono::steady_clock::now();
	co_return;
}

}  // namespace llama_omni_server
