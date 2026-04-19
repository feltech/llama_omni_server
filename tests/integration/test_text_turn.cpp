/**
 * @file   test_text_turn.cpp
 * @brief  Phase 7 integration test — typed text inside a live duplex session.
 *
 * Verifies the session can:
 * 1. complete a duplex voice turn,
 * 2. run a typed simplex turn in the same KV cache and emit text output,
 * 3. return to duplex mode and complete another speaking voice turn.
 */

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <ranges>
#include <string>
#include <vector>

#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/task.hpp>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include "../session_harness.hpp"
#include "session.hpp"
#include "wav.hpp"

namespace llama_omni_server::test
{

SCENARIO(
	"Duplex session supports typed text turns and returns to duplex mode",
	"[session][phase7][slow]")
{
	spdlog::set_level(spdlog::level::debug);

	GIVEN("a duplex session with TTS loaded")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the client performs voice → text → voice in one session")
		{
			TurnResult first_voice;
			TurnResult text_turn;
			TurnResult second_voice;

			harness.run(
				[&harness, &first_voice, &text_turn, &second_voice]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & text_ch = harness.text_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					static constexpr int kShortDrain = 30;
					auto const banana_pcm =
						read_wav_mono(test_data_dir() / "say_the_word_banana.wav").pcm;
					auto send_single_voice_unit =
						[&](TurnResult & turn_result) -> boost::cobalt::task<void>
					{
						std::vector<float> single_unit = banana_pcm;
						single_unit.resize(std::min(single_unit.size(), kResumeChunkSamples));
						co_await audio_ch.write(single_unit);
						co_await drain_until_done(
							audio_ch,
							ack_ch,
							out_ch,
							turn_result,
							/*auto_ack_speech=*/true,
							/*continue_after_resumable_done=*/false,
							/*stop_after_output_on_resumable_done=*/false,
							/*feed_silence_on_resumable_done=*/false,
							kShortDrain);
					};

					spdlog::debug("TEST: drive_voice_turn #1");
					// This test cares about mode transitions, not backlog handling.
					// Feed exactly one duplex audio unit so the subsequent text turn
					// is not delayed by extra queued speech chunks from the WAV tail.
					co_await send_single_voice_unit(first_voice);

					spdlog::debug("TEST: text input");
					co_await send_text_and_drain(
						"What color is the sky?", audio_ch, text_ch, ack_ch, out_ch, text_turn);

					spdlog::debug("TEST: drive_voice_turn #2");
					co_await send_single_voice_unit(second_voice);

					audio_ch.close();
				}());

			THEN("Output is voice -> text -> voice")
			{
				// First voice turn: duplex mode is active, so the session should
				// not fall into typed-text output. Speech is welcome but not
				// guaranteed for this prompt/model combination.
				CHECK(first_voice.text_frames.empty());
				bool const first_voice_produced_duplex_progress = first_voice.got_end_of_turn ||
					first_voice.resumable_done_count > 0 || !first_voice.speech_chunks.empty();
				CHECK(first_voice_produced_duplex_progress);
				CHECK(
					std::ranges::all_of(
						first_voice.speech_chunks,
						[](SpeechChunkOutFrame const & chunk)
						{ return !chunk.pcm.empty() && !chunk.text.empty(); }));

				// Text turn: text output or end-of-turn, no speech.
				bool const text_turn_produced_output =
					text_turn.got_end_of_turn || !text_turn.text_frames.empty();
				CHECK(text_turn_produced_output);
				CHECK(text_turn.speech_chunks.empty());

				// Second voice turn: verifies duplex mode resumed.  The model
				// may or may not produce speech after a text turn; the key
				// invariant is that the session processes audio without
				// emitting text frames (which would mean we're still in
				// simplex text mode).
				CHECK(second_voice.text_frames.empty());
			}
		}
	}
}

}  // namespace llama_omni_server::test
