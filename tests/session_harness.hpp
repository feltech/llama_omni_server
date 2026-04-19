/**
 * @file   session_harness.hpp
 * @brief  Integration-test helpers: SharedModels singleton and SessionHarness.
 *
 * `SharedModels` loads the LLM once per process and returns a reference to
 * the shared `ModelManager`.  This avoids the ~30 s model-load cost per test.
 *
 * `SessionHarness` runs a `Session` inside a proper cobalt executor context
 * established by `cobalt::run()`.  Call `run(test_coro)` to drive the event
 * loop until the test coroutine and session both complete.
 *
 * ### Usage
 *
 * ```cpp
 * SCENARIO("text question → text answer") {
 *     SessionHarness harness{SharedModels::get()};
 *     std::vector<std::string> tokens;
 *     harness.run(collect_text_response(harness, tokens));
 *     CHECK(!tokens.empty());
 * }
 * ```
 *
 * ### Important: channel access timing
 *
 * Channels are created inside `cobalt::run()`, so they only exist while `run()`
 * is executing.  Test coroutines MUST access channels via `harness.audio_in_ch()`
 * etc. inside the coroutine body — never at lambda capture time.  Since
 * `cobalt::task` is lazy, the body runs after channels are created.
 *
 * ```cpp
 * // GOOD: channel accessed inside coroutine body (lazy)
 * harness.run([&harness]() -> task<void> {
 *     harness.audio_in_ch().close();
 *     co_return;
 * }());
 *
 * // BAD: channel accessed at lambda construction time
 * harness.run([&ch = harness.audio_in_ch()]() -> task<void> { ... }());
 * ```
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/run.hpp>
#include <boost/cobalt/spawn.hpp>
#include <boost/cobalt/task.hpp>
#include <boost/cobalt/this_coro.hpp>
#include <gsl/pointers>
#include <spdlog/spdlog.h>

#include "config/config_loader.hpp"
#include "pipeline/model_manager.hpp"
#include "session.hpp"
#include "wav.hpp"

/// Context window for integration tests — 2048 keeps KV cache ≤ 288 MB per session,
/// leaving headroom for audio encoder, vision encoder, and TTS alongside the main
/// LLM in 16 GB VRAM.  Tests run sequentially so only one KV cache is live at a time.
static constexpr int kTestNCtx = 2048;

namespace llama_omni_server::test
{

[[nodiscard]] inline std::filesystem::path repo_root()
{
	// NOLINTNEXTLINE(concurrency-mt-unsafe) — single-threaded test process
	char const * repo_root_env = std::getenv("LLAMAOMNISERVER_TEST_REPO_ROOT");
	if (repo_root_env != nullptr)
	{
		return std::filesystem::path{repo_root_env};
	}
	return std::filesystem::current_path();
}

[[nodiscard]] inline std::filesystem::path example_config_path()
{
	return repo_root() / "config.example.yaml";
}

[[nodiscard]] inline AppConfig load_example_config()
{
	return load_config(example_config_path());
}

// ── Shared models singletons ──────────────────────────────────────────────────

/**
 * @brief Loads the models exactly once per process and caches the result.
 *
 * Model loading is expensive (~30 s for a 10 GB Q4 model).  Calling
 * `SharedModels::get()` multiple times returns the same `ModelManager`
 * reference, loading the model only on the first call.
 *
 * The model is loaded synchronously on the calling thread.  Call this from
 * a test `main` or `REQUIRE`-guarded fixture before spawning any cobalt tasks.
 */
struct SharedModels
{
	/// @return Reference to the process-wide `ModelManager` (loaded on first call).
	static ModelManager & get()
	{
		// Raw pointer, not a value-type static: a POD static has no destructor,
		// so no atexit is registered at initialisation time.  We register our own
		// atexit AFTER model loading (= after CUDA has registered its own
		// driver-shutdown atexit).  Because atexit is LIFO, our delete runs BEFORE
		// CUDA's handler, so llama_model_free sees a live CUDA context.
		//
		// Using a value-type "static ModelManager instance" causes the reverse
		// ordering — CUDA shuts down first, then ~ModelManager calls
		// llama_model_free against a dead driver ("CUDA error: driver shutting
		// down").
		// This is intentionally a mutable process-wide singleton pointer.
		// A value-type static would destruct after CUDA shutdown, which breaks
		// llama model cleanup during test-process exit.
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
		static gsl::owner<ModelManager *> instance = nullptr;

		if (instance == nullptr)
		{
			AppConfig cfg = load_example_config();
			cfg.inference.n_ctx = kTestNCtx;

			// The singleton is heap-allocated and released from the atexit callback
			// below so its destruction runs before CUDA's own shutdown handler.
			// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
			instance = new ModelManager(cfg);

			// Registered after CUDA init so it runs before CUDA's own atexit handler.
			// We explicitly check the return code because std::atexit can fail.
			int const atexit_rc = std::atexit(
				[]() noexcept
				{
					// This statement deletes the singleton allocated above during orderly
					// process shutdown, while the CUDA driver is still alive.
					// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
					delete std::exchange(instance, nullptr);
				});
			if (atexit_rc != 0)
			{
				throw std::runtime_error{"SharedModels::get: std::atexit registration failed"};
			}
		}

		return *instance;
	}
};

// ── Session harness ───────────────────────────────────────────────────────────

/// Channel capacity for text input (client → session).
static constexpr std::size_t kTextInChannelCap = 8U;
/// Channel capacity for speech chunk acks (client → session).
static constexpr std::size_t kSpeechAckChannelCap = 4U;

/// Channel capacity for output frames (session → client).
static constexpr std::size_t kOutputChannelCap = 16U;

/// Channel capacity for audio input chunks (test → session VAD).
static constexpr std::size_t kAudioChannelCap = 32U;

/// Channel capacity for video input frames (test → session CLIP loop).
static constexpr std::size_t kVideoChannelCap = 8U;

/**
 * @brief RAII fixture that runs a `Session` inside a cobalt executor context.
 *
 * Uses `cobalt::run()` to establish the cobalt thread-local executor state
 * required by `cobalt::promise` coroutines.  Channels and the session are
 * created inside the cobalt context and destroyed when `run()` returns.
 *
 * The test coroutine MUST close `audio_in_ch()` before returning so that the
 * session's `run()` coroutine exits and the event loop drains cleanly.
 *
 * Non-copyable and non-movable.
 */
class SessionHarness
{
public:
	/**
	 * @brief Construct the harness with the default text-only config.
	 *
	 * @param models  Shared model manager (loaded via `SharedModels::get()`).
	 */
	explicit SessionHarness(ModelManager & models) : cfg_{make_test_config()}, models_{models} {}

	/**
	 * @brief Construct the harness with a custom config (e.g. with audio).
	 *
	 * @param models  Shared model manager constructed from a config compatible
	 *                with this session's required model components.
	 * @param cfg     Complete application config for this session.
	 */
	SessionHarness(ModelManager & models, AppConfig cfg) : cfg_{std::move(cfg)}, models_{models} {}

	SessionHarness(SessionHarness const &) = delete;
	SessionHarness & operator=(SessionHarness const &) = delete;
	SessionHarness(SessionHarness &&) = delete;
	SessionHarness & operator=(SessionHarness &&) = delete;

	~SessionHarness() = default;

	/**
	 * @brief Run the session and test coroutine inside a cobalt executor context.
	 *
	 * `cobalt::run()` establishes the thread-local cobalt state needed by
	 * `promise` coroutines.  Channels and the session are created inside
	 * this context.
	 *
	 * @param test_coro  Test coroutine that drives the session via channels.
	 *                   Must access channels via harness accessors inside its
	 *                   body (lazy), not at capture time.
	 */
	void run(boost::cobalt::task<void> test_coro)
	{
		boost::cobalt::run(driver(std::move(test_coro)));
	}

	/// Direct access to the text input channel (write `TextInput` from tests).
	/// Only valid inside a `run()` call.
	// NOLINTBEGIN(bugprone-unchecked-optional-access) — .value() throws on empty by design.
	[[nodiscard]] boost::cobalt::channel<TextInput> & text_in_ch()
	{
		return text_in_ch_.value();
	}

	/// Direct access to the speech ack channel (write `SpeechChunkAckEvent` from tests).
	/// Only valid inside a `run()` call.
	[[nodiscard]] boost::cobalt::channel<SpeechChunkAckEvent> & speech_ack_ch()
	{
		return speech_ack_ch_.value();
	}

	/// Direct access to the output channel (read `OutFrame`s in tests).
	/// Only valid inside a `run()` call.
	[[nodiscard]] boost::cobalt::channel<OutFrame> & output_ch()
	{
		return output_ch_.value();
	}

	/// Direct access to the audio input channel (write PCM chunks from tests).
	/// Only valid inside a `run()` call.
	[[nodiscard]] boost::cobalt::channel<std::vector<float>> & audio_in_ch()
	{
		return audio_in_ch_.value();
	}

	/// Direct access to the video input channel (write VideoFrames from tests).
	/// Only valid inside a `run()` call.
	[[nodiscard]] boost::cobalt::channel<VideoFrame> & video_in_ch()
	{
		return video_in_ch_.value();
	}
	// NOLINTEND(bugprone-unchecked-optional-access)

	/// Read-only access to the session's conversation state.
	/// Valid to call after `run()` returns (snapshot is stored).
	[[nodiscard]] ConversationState const & session_conv() const noexcept
	{
		return conv_snapshot_;
	}

	/// Read-only access to the harness config.
	[[nodiscard]] AppConfig const & config() const noexcept
	{
		return cfg_;
	}

	/// Build a config with text-only inference (no audio encoder or reference WAV).
	static AppConfig make_test_config()
	{
		AppConfig cfg = load_example_config();
		cfg.inference.n_ctx = kTestNCtx;
		cfg.inference.temperature = 0.0;
		cfg.inference.duplex_force_listen_count = 0;
		cfg.debug.audio_output_wav = (std::filesystem::temp_directory_path() /
									  "llama_omni_server_test_integration_audio" / "turn_audio.wav")
										 .string();
		return cfg;
	}

private:
	/**
	 * @brief Driver coroutine that creates channels/session inside the cobalt context.
	 *
	 * This task is passed to `cobalt::run()`, which establishes the thread-local
	 * cobalt executor state.  Channels created here inherit that executor via
	 * the default `this_thread::get_executor()`.
	 */
	boost::cobalt::task<void> driver(boost::cobalt::task<void> test_coro)
	{
		// Cobalt exposes the current coroutine executor through `this_coro::executor`.
		// NOLINTNEXTLINE(readability-static-accessed-through-instance)
		auto exec = co_await boost::cobalt::this_coro::executor;

		// Create channels on the cobalt executor.
		text_in_ch_.emplace(kTextInChannelCap, exec);
		speech_ack_ch_.emplace(kSpeechAckChannelCap, exec);
		output_ch_.emplace(kOutputChannelCap, exec);
		audio_in_ch_.emplace(kAudioChannelCap, exec);
		video_in_ch_.emplace(kVideoChannelCap, exec);

		// Create session with the live channels.
		Session session{
			cfg_,
			models_,
			gpu_ex_,
			*output_ch_,
			*audio_in_ch_,
			*video_in_ch_,
			*text_in_ch_,
			*speech_ack_ch_};
		std::optional<boost::cobalt::promise<void>> session_worker;
		session_worker.emplace(
			[&session]() -> boost::cobalt::promise<void> { co_await session.run(); }());

		std::exception_ptr pending_error;
		try
		{
			co_await std::move(test_coro);
		}
		catch (...)
		{
			pending_error = std::current_exception();
		}

		if (audio_in_ch_->is_open())
		{
			audio_in_ch_->close();
		}
		if (video_in_ch_->is_open())
		{
			video_in_ch_->close();
		}
		if (text_in_ch_->is_open())
		{
			text_in_ch_->close();
		}
		if (speech_ack_ch_->is_open())
		{
			speech_ack_ch_->close();
		}

		try
		{
			if (session_worker.has_value())
			{
				co_await std::move(*session_worker);
			}
		}
		catch (...)
		{
			if (pending_error == nullptr)
			{
				pending_error = std::current_exception();
			}
		}

		// Snapshot conversation state before session is destroyed.
		conv_snapshot_ = session.conv();

		// Destroy channels before the cobalt executor context goes away.
		text_in_ch_.reset();
		speech_ack_ch_.reset();
		output_ch_.reset();
		audio_in_ch_.reset();
		video_in_ch_.reset();

		if (pending_error != nullptr)
		{
			std::rethrow_exception(pending_error);
		}
	}

	AppConfig cfg_;
	ModelManager & models_;
	boost::asio::thread_pool gpu_ex_{1};

	// Channels — emplaced inside `driver()` on the cobalt executor, reset before it returns.
	std::optional<boost::cobalt::channel<TextInput>> text_in_ch_;
	std::optional<boost::cobalt::channel<SpeechChunkAckEvent>> speech_ack_ch_;
	std::optional<boost::cobalt::channel<OutFrame>> output_ch_;
	std::optional<boost::cobalt::channel<std::vector<float>>> audio_in_ch_;
	std::optional<boost::cobalt::channel<VideoFrame>> video_in_ch_;

	/// Conversation state snapshot taken at the end of `run()`.
	ConversationState conv_snapshot_;
};

// ── Shared test helpers ───────────────────────────────────────────────────────

/// Half-second of silence at 16 kHz.
static constexpr std::size_t kHalfSecondSamples = 8000U;
/// Default duplex resume chunk size derived from the example config.
static constexpr std::size_t kResumeChunkSamples =
	(static_cast<std::size_t>(AudioConfig::input_sample_rate) *
	 static_cast<std::size_t>(InferenceConfig::default_speech_chunk_ms)) /
	static_cast<std::size_t>(1000U);
/// Number of full silence units to enqueue after a prerecorded utterance.
/// One queued silence keeps the harness close to the browser path while
/// avoiding overfilling the merged duplex-event channel with three immediate
/// 1-second units at once.
static constexpr int kQueuedEndOfSpeechSilenceUnits = 1;
constexpr float kNearZeroAmplitude = 1.0e-6F;
constexpr auto kPostAckPause = std::chrono::milliseconds{20};

/// Safety cap for drain_until_done to prevent infinite loops.
static constexpr int kDrainMaxFrames = 200;

/// Returns the test_data root directory from the CTest-injected env variable.
[[nodiscard]] inline std::filesystem::path test_data_dir()
{
	return repo_root() / "test_data";
}

/// Accumulated output from one turn of duplex or text generation.
struct TurnResult
{
	std::vector<TextOutFrame> text_frames;
	std::vector<SpeechChunkOutFrame> speech_chunks;
	bool got_end_of_turn{false};
	int resumable_done_count{0};
};

/// Join a vector of TextOutFrames into a single lowercase string.
[[nodiscard]] inline std::string joined_lower(std::vector<TextOutFrame> const & frames)
{
	std::string joined;
	for (auto const & frame : frames)
	{
		joined += frame.token;
	}
	std::string lower = joined;
	std::ranges::transform(
		lower,
		lower.begin(),
		[](unsigned char chr) { return static_cast<char>(std::tolower(chr)); });
	return lower;
}

/// Join the text fields of SpeechChunkOutFrames into a single lowercase string.
[[nodiscard]] inline std::string speech_text_joined_lower(
	std::vector<SpeechChunkOutFrame> const & chunks)
{
	std::string joined;
	for (auto const & chunk : chunks)
	{
		joined += chunk.text;
	}
	std::string lower = joined;
	std::ranges::transform(
		lower,
		lower.begin(),
		[](unsigned char chr) { return static_cast<char>(std::tolower(chr)); });
	return lower;
}

/**
 * @brief Drain output frames until an end-of-turn DoneOutFrame arrives,
 *        auto-acknowledging speech chunks and feeding continuation silence.
 */
[[nodiscard]] inline boost::cobalt::task<void> drain_until_done(
	boost::cobalt::channel<std::vector<float>> & audio_ch,
	boost::cobalt::channel<SpeechChunkAckEvent> & ack_ch,
	boost::cobalt::channel<OutFrame> & out_ch,
	TurnResult & result,
	bool auto_ack_speech = true,
	bool continue_after_resumable_done = false,
	bool stop_after_output_on_resumable_done = false,
	bool feed_silence_on_resumable_done = true,
	int max_frames = kDrainMaxFrames)
{
	for (int i = 0; i < max_frames; ++i)
	{
		// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
		OutFrame out_frame = co_await out_ch.read();

		if (auto const * text = std::get_if<TextOutFrame>(&out_frame))
		{
			result.text_frames.push_back(*text);
		}
		else if (auto const * speech = std::get_if<SpeechChunkOutFrame>(&out_frame))
		{
			result.speech_chunks.push_back(*speech);
			if (auto_ack_speech)
			{
				// Ack the chunk so the drainer can emit the next one.
				co_await ack_ch.write(SpeechChunkAckEvent{.seq = speech->seq});
				// After acking, send one configured-sized silence chunk so the
				// session can resume duplex generation promptly.
				std::vector<float> const resume_silence(kResumeChunkSamples, kNearZeroAmplitude);
				co_await audio_ch.write(resume_silence);
			}
		}
		else if (auto const * done = std::get_if<DoneOutFrame>(&out_frame))
		{
			if (done->end_of_turn)
			{
				result.got_end_of_turn = true;
				co_return;
			}

			++result.resumable_done_count;

			bool const has_useful_output =
				!result.text_frames.empty() || !result.speech_chunks.empty();
			if (!continue_after_resumable_done ||
				(stop_after_output_on_resumable_done && has_useful_output))
			{
				co_return;
			}

			// Same configured-sized minimum chunk — the session needs one
			// complete audio chunk to resume generation.  When the session is
			// already driven by buffered audio (e.g. send_audio_and_drain), skip
			// this to avoid an unbounded feedback loop.
			if (feed_silence_on_resumable_done)
			{
				std::vector<float> const resume_silence(kResumeChunkSamples, kNearZeroAmplitude);
				co_await audio_ch.write(resume_silence);
			}
		}
		// VideoAckOutFrame, ErrorOutFrame: silently skipped.
	}
}

/**
 * @brief Send a speech WAV file as audio input and drain output until the
 *        model produces a complete turn.
 */
[[nodiscard]] inline boost::cobalt::task<void> send_audio_and_drain(
	std::filesystem::path const & wav_path,
	boost::cobalt::channel<std::vector<float>> & audio_ch,
	boost::cobalt::channel<SpeechChunkAckEvent> & ack_ch,
	boost::cobalt::channel<OutFrame> & out_ch,
	TurnResult & result,
	int max_frames = kDrainMaxFrames)
{
	auto const pcm = read_wav_mono(wav_path).pcm;
	co_await audio_ch.write(pcm);

	std::vector<float> const silence(kResumeChunkSamples, kNearZeroAmplitude);
	for (int i = 0; i < kQueuedEndOfSpeechSilenceUnits; ++i)
	{
		co_await audio_ch.write(silence);
	}

	// The browser path keeps streaming silence for a bounded period after the
	// user stops talking; it does not inject unbounded continuation silence
	// immediately after every resumable DoneOutFrame. Mirror that here by
	// driving one listen-resume step at a time and capping how many extra
	// silence turns we provide before giving up.
	static constexpr int kMaxContinuationSilenceTurns = 4;
	for (int turn = 0; turn < kMaxContinuationSilenceTurns; ++turn)
	{
		TurnResult turn_result;
		co_await drain_until_done(
			audio_ch,
			ack_ch,
			out_ch,
			turn_result,
			/*auto_ack_speech=*/true,
			/*continue_after_resumable_done=*/false,
			/*stop_after_output_on_resumable_done=*/false,
			/*feed_silence_on_resumable_done=*/false,
			max_frames);

		result.text_frames.insert(
			result.text_frames.end(),
			turn_result.text_frames.begin(),
			turn_result.text_frames.end());
		result.speech_chunks.insert(
			result.speech_chunks.end(),
			turn_result.speech_chunks.begin(),
			turn_result.speech_chunks.end());
		result.got_end_of_turn = result.got_end_of_turn || turn_result.got_end_of_turn;
		result.resumable_done_count += turn_result.resumable_done_count;

		if (turn_result.got_end_of_turn)
		{
			co_return;
		}

		if (turn_result.resumable_done_count == 0)
		{
			co_return;
		}

		co_await audio_ch.write(silence);
	}
}

/**
 * @brief Send a TextInput and drain output until a DoneOutFrame.
 *
 * Stale audio events from the previous voice drain's silence ping-pong
 * may still be queued in the merged channel.  These produce DoneOutFrames
 * before the text turn's output arrives.  This function skips those stale
 * DoneOutFrames, waiting for the first TextOutFrame (which must come from
 * the text turn), then drains normally until a DoneOutFrame.
 */
[[nodiscard]] inline boost::cobalt::task<void> send_text_and_drain(
	std::string_view text,
	boost::cobalt::channel<std::vector<float>> & audio_ch,
	boost::cobalt::channel<TextInput> & text_ch,
	boost::cobalt::channel<SpeechChunkAckEvent> & ack_ch,
	boost::cobalt::channel<OutFrame> & out_ch,
	TurnResult & result)
{
	co_await text_ch.write(TextInput{.text = std::string{text}});

	// Phase 1: skip stale DoneOutFrames until we see a TextOutFrame.
	for (int i = 0; i < kDrainMaxFrames; ++i)
	{
		// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
		OutFrame out_frame = co_await out_ch.read();

		if (auto const * text_frame = std::get_if<TextOutFrame>(&out_frame))
		{
			result.text_frames.push_back(*text_frame);
			// Phase 2: drain remaining text turn output.
			co_await drain_until_done(
				audio_ch,
				ack_ch,
				out_ch,
				result,
				/*auto_ack_speech=*/true,
				/*continue_after_resumable_done=*/false,
				/*stop_after_output_on_resumable_done=*/false,
				/*feed_silence_on_resumable_done=*/false);
			co_return;
		}

		if (auto const * done = std::get_if<DoneOutFrame>(&out_frame))
		{
			// Stale audio DoneOutFrame — skip and keep waiting for text output.
			spdlog::debug(
				"send_text_and_drain: skipping stale DoneOutFrame (end_of_turn={})",
				done->end_of_turn);
			continue;
		}
		// Other frame types (StatusOutFrame, VideoAckOutFrame, etc.) — skip.
	}
	// Fell through without seeing text output — text turn produced no visible text.
	// Record as end-of-turn so the test can detect this case.
	result.got_end_of_turn = true;
}

}  // namespace llama_omni_server::test
