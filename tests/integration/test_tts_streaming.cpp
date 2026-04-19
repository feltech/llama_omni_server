/**
 * @file   test_tts_streaming.cpp
 * @brief  Phase 6 integration tests — TTS audio output from duplex session.
 *
 * Validates that after a `<|speak|>` turn the session emits chunk-aligned
 * speech output: one or more `SpeechChunkOutFrame`s containing both text and
 * the matching 24 kHz PCM.
 *
 * ### Test scenarios
 *
 * 1. **Silence → speak → chunked speech**: send a silence chunk, wait for the
 *    model to choose to speak (or listen), and when it speaks assert at least
 *    one `SpeechChunkOutFrame` is emitted.
 *
 * 2. **Speech → silence → speak → chunked speech**: send a real speech chunk
 *    followed by silence to trigger a generation turn; assert both text and
 *    audio output arrive in the same chunk frame.
 *
 * 3. **Chunk playback ack gating**: after the first speech chunk is emitted,
 *    the server may make limited forward progress, but it must eventually
 *    stall with queued audio until the client acknowledges playback.
 */

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <vector>

#include <boost/asio/steady_timer.hpp>
#include <spdlog/spdlog.h>

#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/task.hpp>

#include "../session_harness.hpp"
#include "session.hpp"
#include "wav.hpp"

namespace llama_omni_server::test
{

namespace
{

/// Output PCM sample rate produced by Token2Wav (matches tts_pipeline.cpp).
constexpr int kTtsOutputSampleRate = 24000;

/// Total time budget for observing pre-ack behavior after queueing more audio.
constexpr auto kPreAckObservationDeadline = std::chrono::seconds{6};

/// Sustained quiet period that indicates ack-gated backpressure took effect.
constexpr auto kPreAckQuietPeriod = std::chrono::milliseconds{1500};

/// Time budget for observing that output resumes once the ack is delivered.
constexpr auto kPostAckObservationDeadline = std::chrono::seconds{3};

/// Short polling interval for observing collector state without blocking forever.
constexpr auto kStatePollInterval = std::chrono::milliseconds{20};

/// Generate a near-zero silence buffer with RMS ≈ 1e-6.
std::vector<float> make_silence_pcm(std::size_t n_samples)
{
	return std::vector<float>(n_samples, kNearZeroAmplitude);
}

/// Assert that at least 1 WAV file is available in the directory.
void assert_debug_wav_files_available(std::filesystem::path const & dir)
{
	std::vector<std::filesystem::path> wavs;
	for (auto const & entry : std::filesystem::directory_iterator(dir))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".wav")
		{
			wavs.push_back(entry.path());
		}
	}

	REQUIRE(!wavs.empty());
	auto const debug_wav = wavs.front();
	auto const debug_info = read_wav_info(debug_wav);
	CHECK(debug_info.channels == 1);
	CHECK(debug_info.sample_rate == kTtsOutputSampleRate);
	CHECK(debug_info.frames > 0);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO(
	"TTS: speech then silence → model speaks → chunk-aligned text and audio output",
	"[session][phase6][slow]")
{
	spdlog::set_level(spdlog::level::debug);
	GIVEN("a duplex session with TTS pipeline loaded")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the test sends speech then silence to trigger a speaking turn")
		{
			TurnResult result;

			harness.run(
				[&harness, &result]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					co_await send_audio_and_drain(
						test_data_dir() / "say_the_word_banana.wav",
						audio_ch,
						ack_ch,
						out_ch,
						result);
					audio_ch.close();
				}());

			THEN("the session completed the turn with chunk-aligned text and audio output")
			{
				CHECK(result.got_end_of_turn);
				REQUIRE(!result.speech_chunks.empty());
				for (auto const & chunk : result.speech_chunks)
				{
					CHECK(!chunk.text.empty());
					CHECK(!chunk.pcm.empty());
				}

				auto const joined = speech_text_joined_lower(result.speech_chunks);
				// NOLINTNEXTLINE(abseil-string-find-str-contains) — absl is not a project dep
				CHECK(joined.find("banana") != std::string::npos);

				spdlog::debug(
					"TTS test: {} speech chunks, text='{}'", result.speech_chunks.size(), joined);

				/// Write generated PCM to a WAV file for manual inspection.
				std::vector<float> audio_pcm;
				for (auto const & chunk : result.speech_chunks)
				{
					audio_pcm.insert(audio_pcm.end(), chunk.pcm.begin(), chunk.pcm.end());
				}
				if (!audio_pcm.empty())
				{
					write_wav_mono(
						std::filesystem::temp_directory_path() / "tts_test_output.wav",
						audio_pcm,
						kTtsOutputSampleRate);
				}

				assert_debug_wav_files_available(
					std::filesystem::path{harness.config().debug.audio_output_wav}.parent_path());
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO(
	"TTS: generation pauses at chunk boundary until playback is acknowledged",
	"[session][phase6][slow]")
{
	spdlog::set_level(spdlog::level::debug);
	GIVEN("a duplex session with TTS pipeline loaded")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the first speech chunk is emitted but not yet acknowledged")
		{
			bool saw_pre_ack_stall = false;
			bool saw_post_ack_progress = false;
			std::optional<SpeechChunkOutFrame> first_chunk;
			std::vector<SpeechChunkOutFrame> extra_pre_ack_chunks;
			int pre_ack_done_count = 0;

			harness.run(
				[&harness,
				 &saw_pre_ack_stall,
				 &saw_post_ack_progress,
				 &first_chunk,
				 &extra_pre_ack_chunks,
				 &pre_ack_done_count]() -> boost::cobalt::task<void>
				{
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					auto const speech_pcm =
						read_wav_mono(test_data_dir() / "say_the_word_banana.wav").pcm;
					co_await audio_ch.write(speech_pcm);
					co_await audio_ch.write(make_silence_pcm(kResumeChunkSamples));
					auto try_read_output = [&]() -> std::optional<OutFrame>
					{
						auto read_op = out_ch.read();
						if (!read_op.await_ready())
						{
							return std::nullopt;
						}
						return read_op.await_resume();
					};

					static constexpr int kMaxFramesBeforeSpeech = 64;
					for (int read_idx = 0;
						 read_idx < kMaxFramesBeforeSpeech && !first_chunk.has_value();
						 ++read_idx)
					{
						OutFrame frame = co_await out_ch.read();
						if (auto * speech_chunk = std::get_if<SpeechChunkOutFrame>(&frame))
						{
							first_chunk = *speech_chunk;
						}
						else if (auto * done = std::get_if<DoneOutFrame>(&frame))
						{
							if (!done->end_of_turn)
							{
								co_await audio_ch.write(make_silence_pcm(kHalfSecondSamples));
							}
						}
					}

					REQUIRE(first_chunk.has_value());
					CHECK(!first_chunk->text.empty());
					CHECK(!first_chunk->pcm.empty());

					static constexpr int kQueuedSilenceTurnsWithoutAck = 4;
					for (int silence_idx = 0; silence_idx < kQueuedSilenceTurnsWithoutAck;
						 ++silence_idx)
					{
						co_await audio_ch.write(make_silence_pcm(kResumeChunkSamples));
					}

					auto const pre_ack_deadline =
						std::chrono::steady_clock::now() + kPreAckObservationDeadline;
					auto quiet_since = std::chrono::steady_clock::now();
					while (std::chrono::steady_clock::now() < pre_ack_deadline)
					{
						if (auto frame = try_read_output(); frame.has_value())
						{
							if (auto const * speech_chunk =
									std::get_if<SpeechChunkOutFrame>(&*frame))
							{
								extra_pre_ack_chunks.push_back(*speech_chunk);
							}
							else if (auto const * done = std::get_if<DoneOutFrame>(&*frame))
							{
								static_cast<void>(done);
								++pre_ack_done_count;
							}
							quiet_since = std::chrono::steady_clock::now();
						}
						else
						{
							if (std::chrono::steady_clock::now() - quiet_since >=
								kPreAckQuietPeriod)
							{
								saw_pre_ack_stall = true;
								break;
							}
							boost::asio::steady_timer timer{out_ch.get_executor()};
							timer.expires_after(kStatePollInterval);
							co_await timer.async_wait(boost::cobalt::use_op);
						}
					}

					uint32_t latest_seen_seq = first_chunk->seq;
					for (auto const & speech_chunk : extra_pre_ack_chunks)
					{
						latest_seen_seq = speech_chunk.seq;
					}

					co_await ack_ch.write(SpeechChunkAckEvent{.seq = latest_seen_seq});
					co_await audio_ch.write(make_silence_pcm(kResumeChunkSamples));

					auto const post_ack_deadline =
						std::chrono::steady_clock::now() + kPostAckObservationDeadline;
					while (std::chrono::steady_clock::now() < post_ack_deadline &&
						   !saw_post_ack_progress)
					{
						if (auto frame = try_read_output(); frame.has_value())
						{
							if (auto const * speech_chunk =
									std::get_if<SpeechChunkOutFrame>(&*frame))
							{
								saw_post_ack_progress = true;
								co_await ack_ch.write(
									SpeechChunkAckEvent{.seq = speech_chunk->seq});
							}
							else if (std::holds_alternative<DoneOutFrame>(*frame))
							{
								saw_post_ack_progress = true;
							}
						}
						else
						{
							boost::asio::steady_timer timer{out_ch.get_executor()};
							timer.expires_after(kStatePollInterval);
							co_await timer.async_wait(boost::cobalt::use_op);
						}
					}

					audio_ch.close();
				}());

			THEN("generation eventually stalls with queued audio and resumes after the ack")
			{
				REQUIRE(first_chunk.has_value());
				CHECK(saw_pre_ack_stall);
				CHECK(extra_pre_ack_chunks.size() <= 1);
				CHECK(pre_ack_done_count <= 2);
				CHECK(saw_post_ack_progress);
			}
		}
	}
}

}  // namespace llama_omni_server::test
