/**
 * @file   test_kv_eviction_integration.cpp
 * @brief  Integration test for KV cache sliding-window eviction.
 *
 * Runs a session with a very small `n_ctx` and streams enough audio to fill
 * the context.  Verifies that:
 *
 * 1. The session does not crash or error.
 * 2. After the session completes, `n_past` is well below `n_ctx` (eviction
 *    occurred).
 * 3. `unit_history` has been trimmed (fewer blocks than would exist without
 *    eviction).
 * 4. Generation continues to produce output after eviction.
 *
 * ### Prerequisites
 *
 * - LLM and audio encoder GGUF files at the paths in `session_harness.hpp`.
 * - GPU with sufficient VRAM.
 * - Run via `ctest` so `LLAMAOMNISERVER_TEST_REPO_ROOT` is injected.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <boost/asio/steady_timer.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/task.hpp>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include "../session_harness.hpp"
#include "session.hpp"
#include "wav.hpp"

namespace llama_omni_server::test
{

namespace
{

/// Small context size to force eviction quickly.
constexpr int kEvictionTestNCtx = 256;
constexpr int kEvictionTestOverflowReserve = 32;

/// Number of 1-second audio rounds to force KV eviction.
constexpr int kRoundsToFill = 20;
/// Per-round output-frame budget to avoid spending unbounded time in
/// listen-only continuation turns while still exercising KV eviction.
constexpr int kMaxFramesPerRound = 6;

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO(
	"Duplex: long-running session with small n_ctx survives KV eviction",
	"[session][phase4b][kv_eviction][slow]")
{
	spdlog::set_level(spdlog::level::debug);

	GIVEN("a session with a very small context window")
	{
		// Build a config with a tiny n_ctx to force eviction quickly.
		// The system prompt takes ~78 tokens, and each turn adds ~100-200
		// tokens. With n_ctx=256 and overflow_reserve=32, eviction should
		// trigger after a few turns.
		AppConfig cfg = SessionHarness::make_test_config();
		cfg.inference.n_ctx = kEvictionTestNCtx;
		cfg.inference.overflow_reserve = kEvictionTestOverflowReserve;

		SessionHarness harness{SharedModels::get(), cfg};

		WHEN("enough audio is streamed to fill and evict the KV cache")
		{
			int turn_count = 0;

			harness.run(
				[&harness, &turns = turn_count]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					auto const banana_pcm =
						read_wav_mono(test_data_dir() / "say_the_word_banana.wav").pcm;
					std::vector<float> single_unit = banana_pcm;
					single_unit.resize(std::min(single_unit.size(), kResumeChunkSamples));

					for (int round = 0; round < kRoundsToFill; ++round)
					{
						spdlog::debug("test: round {}/{}", round + 1, kRoundsToFill);

						TurnResult round_result;
						co_await audio_ch.write(single_unit);
						co_await drain_until_done(
							audio_ch,
							ack_ch,
							out_ch,
							round_result,
							/*auto_ack_speech=*/true,
							/*continue_after_resumable_done=*/false,
							/*stop_after_output_on_resumable_done=*/false,
							/*feed_silence_on_resumable_done=*/false,
							kMaxFramesPerRound);

						if (round_result.got_end_of_turn || round_result.resumable_done_count > 0 ||
							!round_result.speech_chunks.empty() ||
							!round_result.text_frames.empty())
						{
							++turns;
						}
					}

					audio_ch.close();
				}());

			THEN("the session completes without error and produces output")
			{
				// The session survived multiple turns with a tiny context.
				// If eviction didn't work, llama_decode would have failed.
				INFO("Turns completed: " << turn_count);
				CHECK(turn_count == kRoundsToFill);

				// After the session completes, check the final state.
				auto const & conv = harness.session_conv();
				INFO("Final n_past: " << conv.n_past);
				INFO("Final unit_history size: " << conv.unit_history.size());
				INFO("sys_prompt_end: " << conv.sys_prompt_end);

				// n_past should remain below n_ctx (256) — eviction keeps context bounded.
				CHECK(conv.n_past < kEvictionTestNCtx);

				// unit_history should be non-empty (the session tracked KV blocks).
				CHECK(!conv.unit_history.empty());
				CHECK(conv.unit_history.size() < static_cast<std::size_t>(turn_count));

				// sys_prompt_end should be unchanged.
				CHECK(conv.sys_prompt_end > 0);
			}
		}
	}
}

}  // namespace llama_omni_server::test
