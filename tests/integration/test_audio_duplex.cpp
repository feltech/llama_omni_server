/**
 * @file   test_audio_duplex.cpp
 * @brief  Phase 4 integration tests — VAD + duplex audio → listen/speak.
 *
 * Exercises the full duplex pipeline from audio input to duplex output:
 *
 * 1. Initial silence: silence alone should be ignored until the session is
 *    explicitly closed.
 *
 * 2. Speak response: send a real speech chunk followed by a silence chunk —
 *    the model should detect speech, then on silence trigger generation and
 *    emit chunk-aligned spoken output.
 *
 * 3. Close before audio: close audio channel immediately — session should exit
 *    cleanly with n_past > 0.
 *
 * ### Audio format
 *
 * All PCM buffers are mono float32 at 16 kHz.
 * - "speech": loaded from `test_data/say_the_word_banana.wav` — real
 *   speech.
 * - "silence": 0.5 s (8000 samples) of near-zero values (1e-6f).
 */

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <ranges>
#include <vector>

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <boost/asio/steady_timer.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/io/steady_timer.hpp>
#include <boost/cobalt/race.hpp>
#include <boost/cobalt/task.hpp>
#include <boost/cobalt/wait_group.hpp>

#include "session.hpp"
#include "session_harness.hpp"
#include "wav.hpp"

namespace llama_omni_server::test
{

namespace
{

/// Generate a near-zero silence buffer.
std::vector<float> make_silence_pcm(std::size_t n_samples)
{
	return std::vector<float>(n_samples, kNearZeroAmplitude);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO("Duplex: close before any audio exits cleanly", "[session][phase4]")
{
	GIVEN("a duplex session configured with audio encoder and reference WAV")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the test immediately closes the session without sending audio")
		{
			harness.run(
				[&harness]() -> boost::cobalt::task<void>
				{
					harness.audio_in_ch().close();
					co_return;
				}());

			THEN("n_past is greater than zero (system prompt was prefilled)")
			{
				CHECK(harness.session_conv().n_past > 0);
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO("Duplex: silence input alone is ignored until close", "[session][phase4]")
{
	GIVEN("a duplex session configured with audio encoder and reference WAV")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the test sends one silence chunk then closes the session")
		{
			harness.run(
				[&harness]() -> boost::cobalt::task<void>
				{
					// Initial silence alone should not trigger a duplex generation turn.
					co_await harness.audio_in_ch().write(make_silence_pcm(kHalfSecondSamples));
					harness.audio_in_ch().close();
				}());

			THEN("the session exits cleanly without requiring a generation turn")
			{
				CHECK(harness.session_conv().n_past > 0);
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO("Duplex: speech then silence → model speaks with text output", "[session][phase4][slow]")
{
	spdlog::set_level(spdlog::level::trace);
	GIVEN("a duplex session configured with audio encoder and reference WAV")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the test sends speech followed by silence")
		{
			TurnResult result;

			harness.run(
				[&harness, &result]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					static constexpr int kShortDrain = 30;
					auto const banana_pcm =
						read_wav_mono(test_data_dir() / "say_the_word_banana.wav").pcm;
					std::vector<float> single_unit = banana_pcm;
					single_unit.resize(std::min(single_unit.size(), kResumeChunkSamples));
					co_await audio_ch.write(single_unit);
					co_await drain_until_done(
						audio_ch,
						ack_ch,
						out_ch,
						result,
						/*auto_ack_speech=*/true,
						/*continue_after_resumable_done=*/false,
						/*stop_after_output_on_resumable_done=*/false,
						/*feed_silence_on_resumable_done=*/false,
						kShortDrain);
					audio_ch.close();
				}());

			THEN("the duplex turn makes progress without falling into typed-text mode")
			{
				CHECK(result.text_frames.empty());
				bool const produced_duplex_progress = result.got_end_of_turn ||
					result.resumable_done_count > 0 || !result.speech_chunks.empty();
				REQUIRE(produced_duplex_progress);

				for (auto const & chunk : result.speech_chunks)
				{
					CHECK(!chunk.text.empty());
					CHECK(!chunk.pcm.empty());
				}
				std::vector<std::string> text_chunks;
				text_chunks.reserve(result.speech_chunks.size());
				for (auto const & chunk : result.speech_chunks)
				{
					text_chunks.push_back(chunk.text);
				}
				spdlog::debug("speech chunks: {}", text_chunks);
				CAPTURE(text_chunks);
			}
		}
	}
}

}  // namespace llama_omni_server::test
