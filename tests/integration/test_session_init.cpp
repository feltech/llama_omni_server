/**
 * @file   test_session_init.cpp
 * @brief  Phase 3 integration tests — voice reference + duplex system prompt.
 *
 * Verifies that `Session::init_duplex_system_prompt()` correctly encodes the
 * reference voice WAV and prefills the duplex system prompt into the LLM KV
 * cache.  Two scenarios are tested:
 *
 * 1. Cold start: encode the reference WAV from scratch.
 *    - `conv_.n_past > 0` after session init.
 *    - `conv_.sys_prompt_end == conv_.n_past` (all decoded so far is the prompt).
 *
 * 2. Cache hit: a prompt cache file is written on the first session, then a
 *    second session loads it instead of re-encoding.
 *    - Both sessions produce the same `n_past` value.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <ranges>

#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/spawn.hpp>
#include <boost/cobalt/task.hpp>

#include "../session_harness.hpp"
#include "session.hpp"

namespace llama_omni_server::test
{

namespace
{

}  // namespace

SCENARIO("Session init with reference WAV prefills KV cache", "[session][phase3]")
{
	GIVEN("a session configured with the audio encoder and reference WAV")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the session initialises and immediately receives a close event")
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

			THEN("sys_prompt_end is positive and n_past has advanced beyond it")
			{
				// sys_prompt_end marks the first KV position after the system prompt.
				// Phase 4 run_duplex_loop() decodes <unit> on entry and </unit> on exit,
				// so n_past > sys_prompt_end even when no audio is sent.
				CHECK(harness.session_conv().sys_prompt_end > 0);
				CHECK(harness.session_conv().n_past >= harness.session_conv().sys_prompt_end);
			}
		}
	}
}

SCENARIO("Session init loads prompt from cache on second run", "[session][phase3][cache]")
{
	// Use a temporary file under test_data/integration/ for the prompt cache.
	// The file is removed before and after the test to ensure a clean run.
	std::filesystem::path const cache_path =
		test_data_dir() / "integration" / "test_session_init_prompt_cache.bin";

	// Ensure no stale cache from a previous (possibly interrupted) run.
	std::filesystem::remove(cache_path);
	std::filesystem::create_directories(cache_path.parent_path());

	GIVEN("a first session that encodes the reference voice and saves a prompt cache")
	{
		int n_past_first = 0;

		{
			AppConfig cfg = SessionHarness::make_test_config();
			cfg.prompts.system_prompt_cache_path = cache_path.string();
			SessionHarness harness{SharedModels::get(), std::move(cfg)};

			harness.run(
				[&harness]() -> boost::cobalt::task<void>
				{
					harness.audio_in_ch().close();
					co_return;
				}());

			n_past_first = harness.session_conv().n_past;
		}

		REQUIRE(n_past_first > 0);
		REQUIRE(std::filesystem::exists(cache_path));

		WHEN("a second session starts with the same cache path")
		{
			int n_past_second = 0;

			{
				AppConfig cfg = SessionHarness::make_test_config();
				cfg.prompts.system_prompt_cache_path = cache_path.string();
				SessionHarness harness{SharedModels::get(), std::move(cfg)};

				harness.run(
					[&harness]() -> boost::cobalt::task<void>
					{
						harness.audio_in_ch().close();
						co_return;
					}());

				n_past_second = harness.session_conv().n_past;
			}

			THEN("n_past equals the first session's value (loaded from cache)")
			{
				CHECK(n_past_second > 0);
				CHECK(n_past_second == n_past_first);
			}
		}

		std::filesystem::remove(cache_path);
	}
}

SCENARIO("Session init failure emits error and exits cleanly", "[session][phase3]")
{
	GIVEN("a session configured with an invalid reference WAV path")
	{
		AppConfig cfg = SessionHarness::make_test_config();
		cfg.voice.reference_wav = "/definitely/missing/reference.wav";
		SessionHarness harness{SharedModels::get(), std::move(cfg)};

		WHEN("the harness waits for terminal output")
		{
			std::vector<OutFrame> received;

			harness.run(
				[&harness, &received]() -> boost::cobalt::task<void>
				{
					auto & out_ch = harness.output_ch();
					while (true)
					{
						OutFrame const frame = co_await out_ch.read();
						received.push_back(frame);
						if (std::holds_alternative<DoneOutFrame>(frame))
						{
							break;
						}
					}
				}());

			THEN("the session reports an error and still emits terminal done")
			{
				auto const error_it = std::ranges::find_if(
					received,
					[](OutFrame const & frame)
					{ return std::holds_alternative<ErrorOutFrame>(frame); });
				REQUIRE(error_it != received.end());

				auto const done_it = std::ranges::find_if(
					received,
					[](OutFrame const & frame)
					{ return std::holds_alternative<DoneOutFrame>(frame); });
				REQUIRE(done_it != received.end());
				CHECK_FALSE(std::get<DoneOutFrame>(*done_it).end_of_turn);
				auto const & err = std::get<ErrorOutFrame>(*error_it);
				CHECK_FALSE(err.message.empty());
				// The error should mention the missing reference WAV.
				// NOLINTNEXTLINE(abseil-string-find-str-contains) -- absl is not a project dep
				CHECK(err.message.find("reference") != std::string::npos);
			}
		}
	}
}

}  // namespace llama_omni_server::test
