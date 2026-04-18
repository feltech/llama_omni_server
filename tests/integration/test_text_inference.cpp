/**
 * @file   test_text_inference.cpp
 * @brief  Phase 2 integration test — text-only simplex inference.
 *
 * Requires: `config.example.yaml` to point at a valid LLM on an NVIDIA GPU.
 * Does NOT require audio/video models or WebSocket.
 *
 * The tests drive `Session::run_simplex_turn` through the `SessionHarness`
 * channel pair.  The LLM is loaded once (via `SharedModels::get()`) and
 * shared across all test cases.
 */

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include "session.hpp"
#include "session_harness.hpp"

namespace llama_omni_server::test
{

// NOLINTNEXTLINE(readability-function-cognitive-complexity) -- Catch2 BDD macros
SCENARIO("text question produces a non-empty text answer")
{
	GIVEN("a session connected to the loaded LLM")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the client sends a simple arithmetic question")
		{
			TurnResult result;

			harness.run(
				[&harness, &result]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & text_ch = harness.text_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					co_await send_text_and_drain(
						"What is 1+1?", audio_ch, text_ch, ack_ch, out_ch, result);
					audio_ch.close();
				}());

			THEN("the server streams at least one text token and the answer is non-empty")
			{
				CHECK(!result.text_frames.empty());
				auto const answer = joined_lower(result.text_frames);
				CHECK(!answer.empty());
				spdlog::info("Answer to 'What is 1+1?': '{}'", answer);
			}
		}
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) -- Catch2 BDD macros
SCENARIO("two sequential questions produce two non-empty answers")
{
	spdlog::set_level(spdlog::level::debug);

	GIVEN("a session connected to the loaded LLM")
	{
		SessionHarness harness{SharedModels::get()};
		TurnResult first;
		TurnResult second;

		WHEN("the client sends two questions in sequence")
		{
			harness.run(
				[&harness, &first, &second]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & text_ch = harness.text_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					co_await send_text_and_drain(
						"What is 2+2?", audio_ch, text_ch, ack_ch, out_ch, first);
					co_await send_text_and_drain(
						"What is the capital of France?",
						audio_ch,
						text_ch,
						ack_ch,
						out_ch,
						second);
					audio_ch.close();
				}());

			THEN("both responses are non-empty and contain expected answers")
			{
				CHECK(joined_lower(first.text_frames).contains("4"));
				CHECK(joined_lower(second.text_frames).contains("paris"));
			}
		}
	}
}

}  // namespace llama_omni_server::test
