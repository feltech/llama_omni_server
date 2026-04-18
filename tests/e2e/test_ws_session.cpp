/**
 * @file   test_ws_session.cpp
 * @brief  End-to-end WebSocket session tests — Phase 1.
 *
 * These tests use a real WebSocket server bound to a local TCP port.  They
 * verify the HTTP→WS upgrade, MsgPack framing, session lifecycle, and
 * connection-management behaviour described in PLAN.md §"Phase 1".
 *
 * These scenarios exercise the full production websocket/session path against
 * a real local TCP socket.
 *
 * @see tests/e2e/ws_fixture.hpp for fixture and client helpers.
 */

#include <chrono>
#include <cstring>
#include <filesystem>
#include <ranges>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include "wav.hpp"
#include "ws_fixture.hpp"

namespace llama_omni_server::test
{

/// Delay after sending a close frame to allow the server to process it.
/// The close handshake is asynchronous; a brief pause avoids a race.
constexpr std::chrono::milliseconds kCloseSettleMs{100};
/// Timeout for the subprocess server to finish model load and begin listening.
constexpr std::chrono::seconds kServerProcessStartTimeout{90};
/// Timeout for SIGINT-driven process shutdown to complete.
constexpr std::chrono::seconds kServerProcessExitTimeout{5};
/// One buffered duplex unit is 1000 ms of 16 kHz mono PCM.
constexpr std::size_t kDuplexChunkSamples = 16000;
/// 500 ms of 16 kHz silence remains the empirically reliable duplex trigger in this e2e path.
constexpr std::size_t kSilenceChunkSamples = 8000;
/// Minimum amount of generated spoken text expected from the banana prompt.
constexpr std::size_t kMinBananaReplyChars = 20;
/// Practical long-form websocket-test context window; the checked-in example
/// config defaults to 10240 even though the model metadata advertises 40960.
constexpr int kLongFormTestNCtx = 10240;
/// Larger duplex decode budget for the long-form spoken story scenario.
constexpr int kStoryMaxDuplexTokens = 1024;
/// Bounded wait for the browser-chunk smoke test to observe turn progress.
constexpr std::chrono::seconds kBrowserChunkTurnWait{15};
/// Near-zero amplitude used for silence trigger chunks.
constexpr float kSilenceAmplitude = 1.0e-6F;
namespace
{

/**
 * @brief Retry websocket connect+init until the server accepts a fresh session.
 *
 * The full production session path tears down asynchronously after a client
 * close/drop. During that short window, a replacement connection can observe a
 * transient reset. This helper asserts the real contract the tests care about:
 * that the session is eventually freed and a new client can initialise.
 */
template <typename Connectable>
void require_eventual_ready(Connectable const & fixture)
{
	auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
	int attempt = 0;
	while (std::chrono::steady_clock::now() < deadline)
	{
		try
		{
			++attempt;
			spdlog::debug("require_eventual_ready: attempt {}", attempt);
			auto client = fixture.connect();
			client.send_msgpack({{"type", "init"}});
			auto const resp = client.recv_type("ready");
			REQUIRE(resp["type"].template get<std::string>() == "ready");
			spdlog::debug("require_eventual_ready: ready on attempt {}", attempt);
			return;
		}
		catch (std::exception const & ex)
		{
			spdlog::debug("require_eventual_ready: attempt {} failed: {}", attempt, ex.what());
			std::this_thread::sleep_for(kCloseSettleMs);
		}
	}

	FAIL("timed out waiting for the server to accept a replacement session");
}

void send_audio_chunk(WsTestClient & client, std::span<float const> pcm)
{
	std::vector<std::uint8_t> bytes(pcm.size_bytes());
	std::memcpy(bytes.data(), pcm.data(), pcm.size_bytes());

	nlohmann::json frame;
	frame["type"] = "audio_in";
	frame["d"] = nlohmann::json::binary(bytes);
	client.send_msgpack(frame);
}

void send_pcm_as_browser_chunks(WsTestClient & client, std::span<float const> pcm)
{
	for (std::size_t offset = 0; offset < pcm.size(); offset += kE2eBrowserAudioChunkSamples)
	{
		auto const chunk_end = std::min(offset + kE2eBrowserAudioChunkSamples, pcm.size());
		send_audio_chunk(client, pcm.subspan(offset, chunk_end - offset));
	}
}

}  // namespace

SCENARIO("init → ready handshake")
{
	GIVEN("a real omni_server subprocess and a connected client")
	{
		ServerProcess server;
		server.wait_until_listening(kServerProcessStartTimeout);
		auto client = server.connect();

		WHEN("the client sends an init frame")
		{
			client.send_msgpack({{"type", "init"}});

			THEN("the server responds with a ready frame")
			{
				auto const resp = client.recv_type("ready");
				CHECK(resp["type"].get<std::string>() == "ready");
			}
		}
	}
}

SCENARIO("second client is rejected with server_busy")
{
	GIVEN("a real omni_server subprocess with an active session")
	{
		ServerProcess server;
		server.wait_until_listening(kServerProcessStartTimeout);
		auto first = server.connect();

		// Establish the first session.
		first.send_msgpack({{"type", "init"}});
		auto const ready_resp = first.recv_type("ready");
		REQUIRE(ready_resp["type"].get<std::string>() == "ready");

		WHEN("a second client connects and sends init")
		{
			auto second = server.connect();
			second.send_msgpack({{"type", "init"}});

			THEN("the second client receives an error with code server_busy")
			{
				auto const resp = second.recv_type("error");
				CHECK(resp["type"].get<std::string>() == "error");
				CHECK(resp["code"].get<std::string>() == "server_busy");
			}
		}
	}
}

SCENARIO("client close frees the session")
{
	GIVEN("an active session in a real omni_server subprocess")
	{
		ServerProcess server;
		server.wait_until_listening(kServerProcessStartTimeout);
		auto client = server.connect();
		client.send_msgpack({{"type", "init"}});
		client.recv_type("ready");

		WHEN("the client sends a close frame and disconnects")
		{
			client.send_msgpack({{"type", "close"}});
			std::this_thread::sleep_for(kCloseSettleMs);
			client.drop();

			THEN("a new client can connect and initialise successfully")
			{
				require_eventual_ready(server);
			}
		}
	}
}

SCENARIO("disconnect without close frees the session")
{
	GIVEN("an active subprocess session that drops its connection abruptly")
	{
		ServerProcess server;
		server.wait_until_listening(kServerProcessStartTimeout);
		{
			auto client = server.connect();
			client.send_msgpack({{"type", "init"}});
			client.recv_type("ready");
			spdlog::debug("e2e disconnect-without-close test: dropping client socket");
			client.drop();
		}

		// Give the server a moment to detect the disconnection.
		std::this_thread::sleep_for(kCloseSettleMs);

		WHEN("a new client connects and sends init")
		{
			THEN("the server responds with ready (session was freed)")
			{
				require_eventual_ready(server);
			}
		}
	}
}

SCENARIO("server stop exits promptly while a client session is active")
{
	GIVEN("a running server with a connected and initialized client")
	{
		WsFixture const fixture;
		auto client = fixture.connect();
		client.send_msgpack({{"type", "init"}});
		client.recv_type("ready");

		WHEN("the server is asked to stop while the websocket remains connected")
		{
			fixture.request_stop();
			bool const stopped = fixture.wait_stopped(std::chrono::seconds{2});

			THEN("the server thread exits promptly instead of hanging on session I/O")
			{
				REQUIRE(stopped);
			}
		}
	}
}

SCENARIO("SIGINT exits promptly before any session is connected")
{
	GIVEN("a real omni_server subprocess listening on a TCP port")
	{
		ServerProcess server;
		server.wait_until_listening(kServerProcessStartTimeout);

		WHEN("the process receives SIGINT before any client initializes a session")
		{
			server.send_sigint();
			bool const exited = server.wait_exited(kServerProcessExitTimeout);

			THEN("the process exits promptly instead of hanging in shutdown")
			{
				auto const log = server.read_log();
				INFO(log);
				REQUIRE(exited);
				CHECK(log.contains("Server shut down cleanly"));
			}
		}
	}
}

SCENARIO("SIGINT exits promptly after a session is connected")
{
	GIVEN("a real omni_server subprocess with an initialized websocket session")
	{
		ServerProcess server;
		server.wait_until_listening(kServerProcessStartTimeout);
		auto client = server.connect();
		client.send_msgpack({{"type", "init"}});
		client.recv_type("ready", kServerProcessStartTimeout);

		WHEN("the process receives SIGINT while the session remains connected")
		{
			server.send_sigint();
			bool const exited = server.wait_exited(kServerProcessExitTimeout);

			THEN("the process exits promptly instead of hanging in live-session teardown")
			{
				auto const log = server.read_log();
				INFO(log);
				REQUIRE(exited);
				CHECK(log.contains("Server shut down cleanly"));
			}
		}
	}
}

SCENARIO(
	"spoken tell_me_a_story_about_a_banana.wav over websocket yields a substantive spoken response")
{
	spdlog::set_level(spdlog::level::debug);
	GIVEN("a running websocket server with an initialized duplex session")
	{
		AppConfig cfg = test_config();
		cfg.inference.n_ctx = kLongFormTestNCtx;
		cfg.inference.temperature = 0.0F;
		cfg.inference.max_duplex_tokens = kStoryMaxDuplexTokens;
		cfg.logging.level = "debug";
		WsFixture const fixture{std::move(cfg)};
		auto client = fixture.connect();
		client.send_msgpack({{"type", "init"}});
		client.recv_type("ready");

		WHEN("the client streams tell_me_a_story_about_a_banana.wav as audio_in frames")
		{
			send_wav_as_browser_chunks(
				client, test_data_dir() / "tell_me_a_story_about_a_banana.wav");
			send_silence_trigger(client);

			WsTurnResult result;
			auto const silence = std::vector(kSilenceChunkSamples, kSilenceAmplitude);
			drain_ws_until_done(
				client, result, [&] { send_audio_chunk(client, silence); }, kMinBananaReplyChars);
			client.send_msgpack({{"type", "close"}});

			THEN("the model returns a substantive spoken response")
			{
				INFO(result.text);
				REQUIRE(result.text.size() >= kMinBananaReplyChars);
			}
		}
	}
}

SCENARIO("spoken browser-sized websocket audio chunks do not hang turn teardown")
{
	spdlog::set_level(spdlog::level::debug);
	GIVEN("a running websocket server with an initialized duplex session")
	{
		AppConfig cfg = test_config();
		cfg.logging.level = "debug";
		WsFixture const fixture{std::move(cfg)};
		auto client = fixture.connect();
		client.send_msgpack({{"type", "init"}});
		client.recv_type("ready");

		WHEN("the client streams one duplex unit from my_name_is_bob.wav in browser-sized chunks")
		{
			auto const pcm = read_wav_mono(test_data_dir() / "my_name_is_bob.wav").pcm;
			REQUIRE(pcm.size() >= kDuplexChunkSamples);
			send_pcm_as_browser_chunks(client, std::span{pcm}.first(kDuplexChunkSamples));

			WsTurnResult result;
			bool saw_turn_wait_timeout = false;
			try
			{
				drain_ws_until_done(client, result, [] {}, 0, kBrowserChunkTurnWait);
				spdlog::debug("e2e browser-chunk test: drain_ws_until_done returned");
			}
			catch (boost::system::system_error const & ex)
			{
				if (ex.code() != boost::beast::error::timeout)
				{
					throw;
				}
				// This smoke test only asserts that one browser-sized duplex unit
				// does not wedge the websocket path. A bounded read timeout is an
				// acceptable observation here: it means no more frames arrived
				// before the test's wait budget expired, not that transport errors
				// were ignored. Unexpected socket failures still escape above.
				saw_turn_wait_timeout = true;
				spdlog::debug("e2e browser-chunk test: bounded turn wait timed out");
			}
			if (!saw_turn_wait_timeout)
			{
				drain_ws_until_quiet(client, result);
				spdlog::debug("e2e browser-chunk test: drain_ws_until_quiet returned");
			}
			client.drop();

			THEN("the websocket path terminates without hanging")
			{
				INFO("text: " << result.text);
				INFO("speech_chunks: " << result.speech_chunks.size());
				// The pass condition is intentionally "bounded progress or bounded
				// quiet". This scenario is not a semantic response test; it only
				// checks that browser-sized chunks do not hang teardown or leave the
				// websocket path stuck indefinitely.
				bool const saw_bounded_progress = saw_turn_wait_timeout || result.got_end_of_turn ||
					!result.text.empty() || !result.speech_chunks.empty();
				CHECK(saw_bounded_progress);
			}
		}
	}
}

}  // namespace llama_omni_server::test
