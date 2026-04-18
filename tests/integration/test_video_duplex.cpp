/**
 * @file   test_video_duplex.cpp
 * @brief  Phase 4b integration tests: CLIP video encode loop + VideoAckOutFrame.
 *
 * These tests verify that `run_clip_loop()` encodes incoming VideoFrames and
 * always echoes a matching `VideoAckOutFrame` back through the output channel,
 * and that the model correctly incorporates the visual context when generating
 * a response (e.g. naming a shape shown in an image).
 *
 * ### Prerequisites
 *
 * - LLM, audio encoder, and vision encoder GGUF files at the paths in
 *   `session_harness.hpp`.
 * - GPU with sufficient VRAM (~16 GB for all three encoders loaded together).
 * - Run via `ctest` so `LLAMAOMNISERVER_TEST_REPO_ROOT` is injected.
 */

// STB_IMAGE_IMPLEMENTATION must be defined in exactly one TU per binary.
// Defines the stbi_load/stbir_resize_uint8 function bodies here.
// NOLINTBEGIN(*)  -- third-party macro-heavy headers; do not lint
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
// NOLINTEND(*)

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

#include <boost/asio/steady_timer.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/task.hpp>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include "session.hpp"
#include "session_harness.hpp"
#include "wav.hpp"

namespace llama_omni_server::test
{

namespace
{

/// Side length of the 448×448 frame expected by the CLIP vision encoder.
constexpr int kClipFrameSide = 448;

/// Total byte count for one 448×448 RGB frame (3 bytes per pixel).
constexpr std::size_t kFrameBytes =
	static_cast<std::size_t>(kClipFrameSide) * static_cast<std::size_t>(kClipFrameSide) * 3UZ;

/// RGB fill value used for the test frame's red channel.
constexpr std::byte kFillR{0xAA};

/// RGB fill value used for the test frame's green channel.
constexpr std::byte kFillG{0xBB};

/// RGB fill value used for the test frame's blue channel.
constexpr std::byte kFillB{0xCC};

/**
 * @brief Load a PNG from `path`, resize to 448×448 RGB, and return as a VideoFrame.
 *
 * Uses `stbi_load` (3-channel RGB output) then `stbir_resize_uint8` to bring
 * the image to the CLIP encoder's expected input size.
 *
 * @param path  Path to a PNG (or JPEG / BMP / …).
 * @param seq   Sequence number to embed in the returned VideoFrame.
 */
VideoFrame load_png_as_video_frame(std::filesystem::path const & path, std::uint32_t seq)
{
	int width = 0;
	int height = 0;
	int channels_in_file = 0;

	/// `stbi_load` decodes the image to 8-bit-per-channel interleaved RGB
	/// (forced to 3 channels via the last argument).
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
	unsigned char * raw =
		stbi_load(path.c_str(), &width, &height, &channels_in_file, /*desired_channels=*/3);
	REQUIRE(raw != nullptr);

	/// Resize to 448×448 using bilinear interpolation.  `stbir_resize_uint8`
	/// writes into a caller-supplied output buffer.
	std::vector<std::byte> resized(kFrameBytes);
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- std::byte* → unsigned char*
	auto * resized_ptr = reinterpret_cast<unsigned char *>(resized.data());
	int const resize_ok = stbir_resize_uint8(
		raw, width, height, 0, resized_ptr, kClipFrameSide, kClipFrameSide, 0, /*num_channels=*/3);
	stbi_image_free(raw);
	REQUIRE(resize_ok != 0);

	return VideoFrame{.seq = seq, .raw = std::move(resized)};
}

/**
 * @brief Load mono float32 PCM from a 16 kHz WAV file.
 *
 * @param path  Path to the WAV file.
 */
std::vector<float> load_wav_pcm(std::filesystem::path const & path)
{
	return read_wav_mono(path).pcm;
}

/**
 * @brief Build a synthetic 448×448 RGB `VideoFrame` filled with a solid colour.
 *
 * @param seq  Client-assigned sequence number.
 */
VideoFrame make_test_frame(std::uint32_t seq)
{
	std::vector<std::byte> raw(kFrameBytes);
	for (std::size_t idx = 0; idx < kFrameBytes; idx += 3)
	{
		raw[idx + 0] = kFillR;
		raw[idx + 1] = kFillG;
		raw[idx + 2] = kFillB;
	}
	return VideoFrame{.seq = seq, .raw = std::move(raw)};
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO(
	"Duplex: VideoAckOutFrame arrives after audio-triggered generation, not at CLIP encode time",
	"[session][phase4b][video_ack_timing]")
{
	spdlog::set_level(spdlog::level::debug);

	GIVEN("a session with audio + vision encoder loaded")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("a video frame is sent, then audio triggers a generation turn")
		{
			static constexpr std::uint32_t kTestSeq = 7U;
			std::vector<OutFrame> received;
			bool saw_generation_before_ack = false;

			harness.run(
				[&harness, &received, &saw_generation_before_ack]() -> boost::cobalt::task<void>
				{
					auto & video_ch = harness.video_in_ch();
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					auto & frames = received;
					auto & generation_before_ack = saw_generation_before_ack;
					// Send the video frame (CLIP encode will run, but ack should NOT
					// be emitted yet — it must wait for LLM consumption).
					co_await video_ch.write(make_test_frame(kTestSeq));
					spdlog::debug("test: sent VideoFrame seq={}", kTestSeq);

					// Send speech + silence to trigger a generation turn that
					// consumes the CLIP result.
					auto speech = load_wav_pcm(test_data_dir() / "say_the_word_banana.wav");
					auto const mid =
						speech.begin() + static_cast<std::ptrdiff_t>(speech.size() / 2);
					std::vector<float> const silence(kResumeChunkSamples, kNearZeroAmplitude);

					co_await audio_ch.write(std::vector<float>(speech.begin(), mid));
					co_await audio_ch.write(std::vector<float>(mid, speech.end()));

					// Drain output frames. The ack must appear only AFTER the
					// model has consumed the CLIP result (i.e. after at least one
					// SpeechChunkOutFrame or DoneOutFrame from a generation turn).
					bool ack_seen = false;
					bool generation_seen = false;
					static constexpr int kMaxFrames = 200;
					for (int i = 0; i < kMaxFrames && !ack_seen; ++i)
					{
						// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
						OutFrame out_frame = co_await out_ch.read();

						if (std::holds_alternative<SpeechChunkOutFrame>(out_frame) ||
							std::holds_alternative<DoneOutFrame>(out_frame))
						{
							generation_seen = true;

							if (auto const * speech_chunk =
									std::get_if<SpeechChunkOutFrame>(&out_frame))
							{
								co_await ack_ch.write(
									SpeechChunkAckEvent{.seq = speech_chunk->seq});
								// NOLINTNEXTLINE(readability-static-accessed-through-instance)
								auto const exec = co_await boost::cobalt::this_coro::executor;
								boost::asio::steady_timer timer{exec};
								timer.expires_after(kPostAckPause);
								co_await timer.async_wait(boost::cobalt::use_op);
								co_await audio_ch.write(silence);
							}

							if (auto const * done = std::get_if<DoneOutFrame>(&out_frame))
							{
								if (!done->end_of_turn)
								{
									co_await audio_ch.write(silence);
								}
							}
						}

						if (std::get_if<VideoAckOutFrame>(&out_frame) != nullptr)
						{
							ack_seen = true;
							generation_before_ack = generation_seen;
						}

						frames.push_back(std::move(out_frame));
					}

					audio_ch.close();
				}());

			THEN("a VideoAckOutFrame is received with correct seq/size, after generation")
			{
				auto ack_it = std::ranges::find_if(
					received,
					[](OutFrame const & frm)
					{ return std::holds_alternative<VideoAckOutFrame>(frm); });
				REQUIRE(ack_it != received.end());
				auto const & ack = std::get<VideoAckOutFrame>(*ack_it);
				CHECK(ack.seq == kTestSeq);
				CHECK(ack.raw.size() == kFrameBytes);

				CHECK(saw_generation_before_ack);
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO(
	"Duplex: show triangle image + ask 'describe the shape' → response contains 'triangle'",
	"[session][phase4b][slow]")
{
	spdlog::set_level(spdlog::level::debug);

	GIVEN("a session with audio + vision encoder loaded")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("the session receives a triangle image and the 'describe the shape' audio")
		{
			TurnResult result;

			harness.run(
				[&harness, &result]() -> boost::cobalt::task<void>
				{
					auto & video_ch = harness.video_in_ch();
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					// Send video frame first so CLIP encodes it before audio triggers generation.
					co_await video_ch.write(
						load_png_as_video_frame(test_data_dir() / "triangle.png", /*seq=*/0U));

					// Voice turn: speech describing the image.
					auto const pcm = read_wav_mono(test_data_dir() / "describe_the_shape.wav").pcm;
					co_await audio_ch.write(pcm);
					co_await drain_until_done(
						audio_ch,
						ack_ch,
						out_ch,
						result,
						/*auto_ack_speech=*/true,
						/*continue_after_resumable_done=*/true,
						/*stop_after_output_on_resumable_done=*/true,
						/*feed_silence_on_resumable_done=*/true);

					audio_ch.close();
				}());

			THEN("the response contains 'triangle'")
			{
				auto const lower = speech_text_joined_lower(result.speech_chunks);
				INFO("Full response: " << lower);
				// NOLINTNEXTLINE(abseil-string-find-str-contains) -- absl is not a project dep
				CHECK(lower.find("triangle") != std::string::npos);
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO(
	"Duplex: rapid video frames are throttled to at most one CLIP encode per second",
	"[session][phase4b][workstream1][slow]")
{
	spdlog::set_level(spdlog::level::debug);

	GIVEN("a session with audio + vision encoder loaded")
	{
		SessionHarness harness{SharedModels::get()};

		WHEN("five VideoFrames are sent in rapid succession, then audio triggers generation")
		{
			static constexpr int kNFrames = 5;
			std::vector<std::uint32_t> acked_seqs;
			std::vector<OutFrame> all_frames;

			harness.run(
				[&harness, &acked_seqs, &all_frames]() -> boost::cobalt::task<void>
				{
					auto & video_ch = harness.video_in_ch();
					auto & audio_ch = harness.audio_in_ch();
					auto & ack_ch = harness.speech_ack_ch();
					auto & out_ch = harness.output_ch();
					auto & out_seqs = acked_seqs;
					auto & frames = all_frames;
					// Send all 5 frames rapidly — only the first should be
					// CLIP-encoded due to the 1 FPS throttle.
					for (std::uint32_t seq = 0U; seq < static_cast<std::uint32_t>(kNFrames); ++seq)
					{
						co_await video_ch.write(make_test_frame(seq));
					}
					spdlog::debug("test: sent {} VideoFrames rapidly", kNFrames);

					// Send silence to trigger a generation turn that consumes
					// the single CLIP result and emits one VideoAckOutFrame.
					std::vector<float> const silence(kResumeChunkSamples, kNearZeroAmplitude);
					co_await audio_ch.write(silence);
					co_await audio_ch.write(silence);

					// Read output until a generation turn completes.  In the new
					// design VideoAckOutFrame is emitted BEFORE DoneOutFrame, so
					// we break immediately when we see DoneOutFrame.
					static constexpr int kMaxReads = 200;
					for (int i = 0; i < kMaxReads; ++i)
					{
						OutFrame out_frame = co_await out_ch.read();
						if (auto const * vid_ack = std::get_if<VideoAckOutFrame>(&out_frame))
						{
							out_seqs.push_back(vid_ack->seq);
						}
						if (auto const * speech_chunk =
								std::get_if<SpeechChunkOutFrame>(&out_frame))
						{
							co_await ack_ch.write(SpeechChunkAckEvent{.seq = speech_chunk->seq});
							// NOLINTNEXTLINE(readability-static-accessed-through-instance)
							auto const exec = co_await boost::cobalt::this_coro::executor;
							boost::asio::steady_timer timer{exec};
							timer.expires_after(kPostAckPause);
							co_await timer.async_wait(boost::cobalt::use_op);
							co_await audio_ch.write(silence);
						}
						frames.push_back(std::move(out_frame));
						if (std::holds_alternative<DoneOutFrame>(frames.back()))
						{
							break;
						}
					}

					audio_ch.close();
				}());

			THEN("at most one VideoAckOutFrame is received (throttled to 1 FPS)")
			{
				INFO("Acked seqs: " << acked_seqs.size());
				CHECK(acked_seqs.size() <= 1U);
			}
		}
	}
}

}  // namespace llama_omni_server::test
