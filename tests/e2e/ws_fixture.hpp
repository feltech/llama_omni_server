/**
 * @file   ws_fixture.hpp
 * @brief  WebSocket end-to-end test fixture and client helpers.
 *
 * Provides `WsFixture` (starts a real WebSocket server bound to a local port)
 * and `WsTestClient` (connects to it and exchanges MsgPack frames).  These
 * helpers cover the Level-3 e2e tests described in PLAN.md §"Integration Test
 * Architecture".
 *
 * ### Usage
 *
 * ```cpp
 * WsFixture const fixture;
 * auto client = fixture.connect();
 * client.send_msgpack({{\"type\", \"init\"}});
 * auto resp = client.recv_type(\"ready\");
 * REQUIRE(resp[\"type\"] == \"ready\");
 * ```
 *
 * ### Shutdown
 *
 * `WsFixture::~WsFixture()` calls `server_->stop()`, which posts acceptor
 * cancellation onto the cobalt executor so the pending `async_accept` wakes
 * immediately. The destructor then joins the server thread cleanly without
 * requiring a dummy TCP connection.
 */

#pragma once

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include <boost/asio/connect.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/websocket/error.hpp>
#include <boost/beast/websocket/stream.hpp>
#include <boost/cobalt/run.hpp>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "config/config_loader.hpp"
#include "protocol.hpp"
#include "server.hpp"
#include "session_harness.hpp"

namespace llama_omni_server::test
{

// ── Test configuration ────────────────────────────────────────────────────────

/// Polling interval used while waiting for the server to start binding.
constexpr std::chrono::milliseconds kServerStartPollMs{10};
/// Exit code used when the subprocess fails to exec the server binary.
constexpr int kExecFailureExitCode = 127;

/// Build the real application config for websocket tests. Port 0 asks the OS
/// for a free ephemeral port; `port()` returns the actual bound port after
/// `run()` starts.
inline AppConfig test_config()
{
	AppConfig cfg = load_example_config();
	cfg.server.host = "127.0.0.1";
	cfg.server.port = 0;  // OS assigns a free port
	cfg.inference.n_ctx = kTestNCtx;
	cfg.inference.duplex_force_listen_count = 0;
	cfg.logging.status_period_ms = 0;
	return cfg;
}

[[nodiscard]] inline std::filesystem::path build_dir()
{
	// NOLINTNEXTLINE(concurrency-mt-unsafe) — single-threaded test process
	char const * exe_dir = std::getenv("LLAMAOMNISERVER_TEST_EXE_DIR");
	if (exe_dir != nullptr)
	{
		return std::filesystem::path{exe_dir};
	}
	return std::filesystem::path{LLAMAOMNISERVER_TEST_SERVER_EXE}.parent_path();
}

[[nodiscard]] inline std::filesystem::path server_binary_path()
{
	return std::filesystem::path{LLAMAOMNISERVER_TEST_SERVER_EXE};
}

[[nodiscard]] inline std::uint16_t reserve_loopback_port()
{
	boost::asio::io_context ioc;
	boost::asio::ip::tcp::acceptor acceptor{ioc};
	boost::asio::ip::tcp::endpoint const endpoint{
		boost::asio::ip::make_address("127.0.0.1"), static_cast<std::uint16_t>(0)};
	acceptor.open(endpoint.protocol());
	acceptor.bind(endpoint);
	return acceptor.local_endpoint().port();
}

inline void write_process_config(std::filesystem::path const & config_path, std::uint16_t port)
{
	AppConfig cfg = load_example_config();
	cfg.server.host = "127.0.0.1";
	cfg.server.port = static_cast<int>(port);
	cfg.inference.n_ctx = kTestNCtx;
	cfg.logging.level = "debug";
	cfg.logging.status_period_ms = 0;
	cfg.model.vision_path.clear();
	cfg.model.tts_transformer_path.clear();
	cfg.model.tts_weights_path.clear();
	cfg.model.projector_path.clear();
	cfg.model.token2wav_dir.clear();
	cfg.vram.tts_transformer_gpu = false;
	cfg.vram.token2wav_gpu = false;

	YAML::Emitter emitter;
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "server" << YAML::Value << YAML::BeginMap;
	emitter << YAML::Key << "host" << YAML::Value << cfg.server.host;
	emitter << YAML::Key << "port" << YAML::Value << cfg.server.port;
	emitter << YAML::EndMap;
	emitter << YAML::Key << "model" << YAML::Value << YAML::BeginMap;
	emitter << YAML::Key << "llm_path" << YAML::Value << cfg.model.llm_path;
	emitter << YAML::Key << "vision_path" << YAML::Value << "";
	emitter << YAML::Key << "audio_path" << YAML::Value << cfg.model.audio_path;
	emitter << YAML::Key << "tts_transformer_path" << YAML::Value << "";
	emitter << YAML::Key << "tts_weights_path" << YAML::Value << "";
	emitter << YAML::Key << "projector_path" << YAML::Value << "";
	emitter << YAML::Key << "token2wav_dir" << YAML::Value << "";
	emitter << YAML::EndMap;
	emitter << YAML::Key << "inference" << YAML::Value << YAML::BeginMap;
	emitter << YAML::Key << "n_ctx" << YAML::Value << cfg.inference.n_ctx;
	emitter << YAML::Key << "duplex_force_listen_count" << YAML::Value << 0;
	emitter << YAML::EndMap;
	emitter << YAML::Key << "voice" << YAML::Value << YAML::BeginMap;
	emitter << YAML::Key << "reference_wav" << YAML::Value << cfg.voice.reference_wav;
	emitter << YAML::EndMap;
	emitter << YAML::Key << "logging" << YAML::Value << YAML::BeginMap;
	emitter << YAML::Key << "level" << YAML::Value << cfg.logging.level;
	emitter << YAML::Key << "status_period_ms" << YAML::Value << cfg.logging.status_period_ms;
	emitter << YAML::EndMap;
	emitter << YAML::Key << "vram" << YAML::Value << YAML::BeginMap;
	emitter << YAML::Key << "tts_transformer_gpu" << YAML::Value << false;
	emitter << YAML::Key << "token2wav_gpu" << YAML::Value << false;
	emitter << YAML::EndMap;
	emitter << YAML::EndMap;

	std::ofstream out{config_path};
	if (!out)
	{
		throw std::runtime_error{"failed to create subprocess config file"};
	}
	out << emitter.c_str();
}

struct ServerProcessFiles
{
	std::filesystem::path config_path;
	std::filesystem::path log_path;
};

[[nodiscard]] inline pid_t spawn_server_process(ServerProcessFiles const & files)
{
	pid_t const child_pid = fork();
	if (child_pid != 0)
	{
		return child_pid;
	}

	int const log_fd = ::creat(files.log_path.c_str(), 0644);
	if (log_fd < 0)
	{
		::_exit(kExecFailureExitCode);
	}

	::dup2(log_fd, STDOUT_FILENO);
	::dup2(log_fd, STDERR_FILENO);
	::close(log_fd);

	std::array<std::string, 2> args{server_binary_path().string(), files.config_path.string()};
	std::array<char *, 3> argv{args[0].data(), args[1].data(), nullptr};
	::execv(args[0].c_str(), argv.data());
	::_exit(kExecFailureExitCode);
}

// ── WsTestClient ──────────────────────────────────────────────────────────────

/**
 * @brief Synchronous WebSocket client for e2e tests.
 *
 * Wraps a Boost.Beast WebSocket stream connected to the test server.  All
 * operations are synchronous (blocking) for test simplicity.
 *
 * Non-copyable and non-movable: `boost::asio::io_context` is neither.
 */
class WsTestClient
{
public:
	using WsStream = boost::beast::websocket::stream<boost::beast::tcp_stream>;

	/**
	 * @brief Connect to the given host and port.
	 *
	 * @throws boost::system::system_error on connection failure.
	 */
	WsTestClient(std::string const & host, std::uint16_t port)
	{
		boost::asio::ip::tcp::resolver resolver{ioc_};
		auto const endpoints = resolver.resolve(host, std::to_string(port));
		boost::beast::tcp_stream stream{ioc_};
		stream.connect(endpoints);
		ws_ = std::make_unique<WsStream>(std::move(stream));
		ws_->binary(true);
		boost::beast::get_lowest_layer(*ws_).expires_after(kDefaultWsOpTimeout);
		// Perform the WebSocket handshake (client side).
		ws_->handshake(host, "/");
		boost::beast::get_lowest_layer(*ws_).expires_never();
		ws_->set_option(
			boost::beast::websocket::stream_base::timeout::suggested(
				boost::beast::role_type::client));
	}

	WsTestClient(WsTestClient const &) = delete;
	WsTestClient & operator=(WsTestClient const &) = delete;
	WsTestClient(WsTestClient &&) = delete;
	WsTestClient & operator=(WsTestClient &&) = delete;

	/**
	 * @brief Send a JSON value encoded as MsgPack.
	 *
	 * @param msg  JSON object to encode and send.
	 */
	void send_msgpack(nlohmann::json const & msg)
	{
		auto const bytes = nlohmann::json::to_msgpack(msg);
		boost::beast::get_lowest_layer(*ws_).expires_after(kDefaultWsOpTimeout);
		ws_->write(boost::asio::buffer(bytes.data(), bytes.size()));
		boost::beast::get_lowest_layer(*ws_).expires_never();
	}

	/**
	 * @brief Receive one MsgPack message and decode it as JSON.
	 *
	 * @return Decoded JSON object.
	 * @throws boost::system::system_error on read failure.
	 */
	nlohmann::json recv_msgpack(std::chrono::milliseconds timeout = kDefaultWsOpTimeout)
	{
		boost::beast::flat_buffer buf;
		boost::asio::steady_timer timer{ioc_};
		boost::system::error_code read_error;
		bool timed_out = false;

		ws_->async_read(
			buf,
			[&](boost::system::error_code const error_code,
				[[maybe_unused]] std::size_t const transferred)
			{
				read_error = error_code;
				[[maybe_unused]] auto const cancelled = timer.cancel();
			});

		timer.expires_after(timeout);
		timer.async_wait(
			[&](boost::system::error_code const error_code)
			{
				if (error_code == boost::asio::error::operation_aborted)
				{
					return;
				}
				timed_out = true;
				boost::system::error_code ignored_error;
				[[maybe_unused]] auto const cancelled =
					boost::beast::get_lowest_layer(*ws_).socket().cancel(ignored_error);
			});

		ioc_.restart();
		ioc_.run();

		if (timed_out && (read_error == boost::asio::error::operation_aborted || !read_error))
		{
			throw boost::system::system_error{boost::beast::error::timeout};
		}
		if (read_error)
		{
			throw boost::system::system_error{read_error};
		}
		// Use std::span to avoid raw pointer arithmetic.
		auto const data_view = buf.data();
		auto const bytes = std::span<std::uint8_t const>{
			static_cast<std::uint8_t const *>(data_view.data()), data_view.size()};
		return nlohmann::json::from_msgpack(bytes.begin(), bytes.end());
	}

	/**
	 * @brief Receive frames until one with the given type arrives.
	 *
	 * Skips frames whose "type" field does not match.  Throws if no matching
	 * frame arrives within `timeout`.
	 *
	 * @param type     Expected "type" string value.
	 * @param timeout  Maximum time to wait across all skipped frames.
	 * @return The first matching JSON frame.
	 * @throws std::runtime_error on timeout.
	 */
	nlohmann::json recv_type(
		std::string_view type, std::chrono::milliseconds timeout = std::chrono::seconds{30})
	{
		auto const deadline = std::chrono::steady_clock::now() + timeout;
		while (std::chrono::steady_clock::now() < deadline)
		{
			auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
				deadline - std::chrono::steady_clock::now());
			auto frame = recv_msgpack(remaining);
			if (frame.contains("type") && frame["type"].get<std::string>() == type)
			{
				return frame;
			}
		}
		throw std::runtime_error{
			std::string{"recv_type: timed out waiting for frame type \""} + std::string{type} +
			"\""};
	}

	/// Close the WebSocket connection gracefully.
	void close()
	{
		if (ws_ && ws_->is_open())
		{
			boost::beast::websocket::close_reason const close_reason{
				boost::beast::websocket::close_code::normal};
			boost::asio::steady_timer timer{ioc_};
			boost::system::error_code error_code;
			bool timed_out = false;

			ws_->async_close(
				close_reason,
				[&](boost::system::error_code const close_error)
				{
					error_code = close_error;
					[[maybe_unused]] auto const cancelled = timer.cancel();
				});

			timer.expires_after(kDefaultWsOpTimeout);
			timer.async_wait(
				[&](boost::system::error_code const timer_error)
				{
					if (timer_error == boost::asio::error::operation_aborted)
					{
						return;
					}
					timed_out = true;
					boost::system::error_code ignored_error;
					auto const cancelled =
						boost::beast::get_lowest_layer(*ws_).socket().cancel(ignored_error);
					(void)cancelled;
				});

			ioc_.restart();
			ioc_.run();

			if (timed_out && (error_code == boost::asio::error::operation_aborted || !error_code))
			{
				error_code = boost::beast::error_code{boost::beast::error::timeout};
			}
			if (error_code && error_code != boost::beast::websocket::error::closed &&
				error_code != boost::beast::error::timeout &&
				error_code != boost::asio::error::connection_reset &&
				error_code != boost::asio::error::eof &&
				error_code != boost::asio::error::operation_aborted)
			{
				throw boost::system::system_error{error_code};
			}
			if (error_code)
			{
				spdlog::debug(
					"WsTestClient::close ignored websocket close error: {}", error_code.message());
			}
			ws_.reset();
		}
	}

	/// Drop the underlying TCP connection without a websocket close handshake.
	void drop() noexcept
	{
		if (ws_)
		{
			boost::system::error_code error_code;
			[[maybe_unused]] auto const close_result =
				boost::beast::get_lowest_layer(*ws_).socket().close(error_code);
			ws_.reset();
		}
	}

	~WsTestClient() noexcept
	{
		// Destructor must not propagate exceptions; swallow any close errors.
		try
		{
			close();
		}
		catch (...)	 // NOLINT(bugprone-empty-catch) -- noexcept dtor must swallow
		{
		}
	}

private:
	static constexpr std::chrono::milliseconds kDefaultWsOpTimeout{15000};

	boost::asio::io_context ioc_;
	std::unique_ptr<WsStream> ws_;
};

/**
 * @brief Spawn the real `omni_server` as a subprocess for SIGINT e2e tests.
 *
 * This helper launches the built server binary with a temporary config file,
 * redirects stdout/stderr to a log file, and provides timed waiting for both
 * socket readiness and process exit after signals.
 */
class ServerProcess
{
public:
	ServerProcess()
		: port_{reserve_loopback_port()},
		  config_path_{repo_root() / std::format("test_server_{}.yaml", port_)},
		  log_path_{repo_root() / std::format("test_server_{}.log", port_)},
		  pid_{[this]
			   {
				   write_process_config(config_path_, port_);
				   return spawn_server_process(
					   ServerProcessFiles{.config_path = config_path_, .log_path = log_path_});
			   }()}
	{
		if (pid_ < 0)
		{
			throw std::runtime_error{"fork failed"};
		}
	}

	ServerProcess(ServerProcess const &) = delete;
	ServerProcess & operator=(ServerProcess const &) = delete;
	ServerProcess(ServerProcess &&) = delete;
	ServerProcess & operator=(ServerProcess &&) = delete;

	~ServerProcess()
	{
		if (!exited_)
		{
			::kill(pid_, SIGKILL);
			int status = 0;
			[[maybe_unused]] auto const wait_result = ::waitpid(pid_, &status, 0);
		}
		std::filesystem::remove(config_path_);
		std::filesystem::remove(log_path_);
	}

	void wait_until_listening(std::chrono::milliseconds timeout)
	{
		auto const deadline = std::chrono::steady_clock::now() + timeout;
		while (std::chrono::steady_clock::now() < deadline)
		{
			require_running();
			try
			{
				boost::asio::io_context ioc;
				boost::asio::ip::tcp::socket socket{ioc};
				boost::asio::ip::tcp::endpoint const endpoint{
					boost::asio::ip::make_address("127.0.0.1"), port_};
				socket.connect(endpoint);
				boost::system::error_code error_code;
				[[maybe_unused]] auto const close_result = socket.close(error_code);
				return;
			}
			catch (std::exception const &)
			{
				std::this_thread::sleep_for(kServerStartPollMs);
			}
		}
		throw std::runtime_error{"server subprocess did not start listening before timeout"};
	}

	void send_sigint() const
	{
		if (::kill(pid_, SIGINT) != 0)
		{
			throw std::runtime_error{"failed to send SIGINT to server subprocess"};
		}
	}

	[[nodiscard]] bool wait_exited(std::chrono::milliseconds timeout)
	{
		if (exited_)
		{
			return true;
		}

		auto const deadline = std::chrono::steady_clock::now() + timeout;
		while (std::chrono::steady_clock::now() < deadline)
		{
			int status = 0;
			pid_t const wait_result = ::waitpid(pid_, &status, WNOHANG);
			if (wait_result == pid_)
			{
				exited_ = true;
				exit_status_ = status;
				return true;
			}
			if (wait_result < 0)
			{
				throw std::runtime_error{"waitpid failed for server subprocess"};
			}
			std::this_thread::sleep_for(kServerStartPollMs);
		}

		return false;
	}

	[[nodiscard]] WsTestClient connect() const
	{
		return WsTestClient{"127.0.0.1", port_};
	}

	[[nodiscard]] std::string read_log() const
	{
		std::ifstream log_stream{log_path_};
		return std::string{
			std::istreambuf_iterator<char>{log_stream}, std::istreambuf_iterator<char>{}};
	}

	[[nodiscard]] int exit_status() const
	{
		if (!exited_)
		{
			throw std::logic_error{"exit_status requested before process exit"};
		}
		return exit_status_;
	}

private:
	void require_running()
	{
		int status = 0;
		pid_t const wait_result = ::waitpid(pid_, &status, WNOHANG);
		if (wait_result == pid_)
		{
			exited_ = true;
			exit_status_ = status;
			throw std::runtime_error{std::format(
				"server subprocess exited unexpectedly before listening (status={}):\n{}",
				exit_status_,
				read_log())};
		}
		if (wait_result < 0)
		{
			throw std::runtime_error{"waitpid failed while polling server subprocess"};
		}
	}

	std::uint16_t port_;
	std::filesystem::path config_path_;
	std::filesystem::path log_path_;
	pid_t pid_{-1};
	bool exited_{false};
	int exit_status_{0};
};

// ── WsFixture ─────────────────────────────────────────────────────────────────

/**
 * @brief RAII fixture that starts a real WebSocket server on a local port.
 *
 * The server runs on a background thread driving its own cobalt io_context.
 * Construct a `WsFixture` at the start of a test and let it destruct when the
 * test ends — the destructor calls `stop()` and joins the thread cleanly.
 *
 * Declare instances `const` — all useful methods (`connect`, `port`) are const.
 *
 * Non-copyable and non-movable: `std::jthread` and `WebSocketServer` are not
 * copyable; move semantics would leave the fixture in an unusable state.
 */
class WsFixture
{
public:
	/// Start the server in a background thread.  Retries connecting until the
	/// server is accepting connections (up to 1 second).
	explicit WsFixture(AppConfig cfg = test_config())
		: runtime_{std::make_shared<AppRuntime>(AppRuntime{
			  .cfg = std::move(cfg),
			  .models =
				  std::shared_ptr<ModelManager>{
					  &SharedModels::get(), [](ModelManager *) noexcept {}},
			  .gpu_ex = std::make_shared<boost::asio::thread_pool>(1)})},
		  server_{std::make_unique<WebSocketServer>(runtime_)}
	{
		// Run the cobalt server task on a background thread.
		// cobalt::run() creates its own io_context and blocks until the task ends.
		server_thread_ =
			std::thread{[this]
						{
							try
							{
								boost::cobalt::run(server_->run());
								server_stopped_.set_value();
							}
							catch (...)
							{
								server_stopped_.set_exception(std::current_exception());
							}
						}};

		try
		{
			// Retry until the server starts listening (port() returns non-zero).
			auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
			while (server_->port() == 0)
			{
				if (server_stopped_future_.wait_for(std::chrono::milliseconds{0}) ==
					std::future_status::ready)
				{
					server_stopped_future_.get();
					throw std::runtime_error{"WsFixture: server stopped before binding a port"};
				}
				if (std::chrono::steady_clock::now() >= deadline)
				{
					throw std::runtime_error{"WsFixture: server did not start within 5 seconds"};
				}
				std::this_thread::sleep_for(kServerStartPollMs);
			}
		}
		catch (...)
		{
			request_stop();
			if (server_thread_.joinable())
			{
				server_thread_.join();
			}
			throw;
		}
	}

	WsFixture(WsFixture const &) = delete;
	WsFixture & operator=(WsFixture const &) = delete;
	WsFixture(WsFixture &&) = delete;
	WsFixture & operator=(WsFixture &&) = delete;

	/// Stop the server and join the background thread.
	~WsFixture()
	{
		// stop() posts acceptor cancellation onto the cobalt executor — no
		// dummy TCP connection required.
		request_stop();
		if (server_thread_.joinable())
		{
			server_thread_.join();
		}
	}

	/// Request server shutdown without joining the background thread.
	void request_stop() const
	{
		server_->stop();
	}

	/// Wait for the background server thread to exit.
	[[nodiscard]] bool wait_stopped(std::chrono::milliseconds timeout) const
	{
		return server_stopped_future_.wait_for(timeout) == std::future_status::ready;
	}

	/// Connect a new test client to the running server.
	[[nodiscard]] WsTestClient connect() const
	{
		return WsTestClient{"127.0.0.1", server_->port()};
	}

	/// Return the actual bound port.
	[[nodiscard]] std::uint16_t port() const noexcept
	{
		return server_->port();
	}

private:
	std::shared_ptr<AppRuntime> runtime_;
	std::unique_ptr<WebSocketServer> server_;
	std::promise<void> server_stopped_;
	std::future<void> server_stopped_future_{server_stopped_.get_future()};
	std::thread server_thread_;
};

// ── E2E shared helpers ────────────────────────────────────────────────────────

/// Accumulated output from one e2e WebSocket turn.
struct WsTurnResult
{
	std::string text;
	std::vector<nlohmann::json> speech_chunks;
	bool got_end_of_turn{false};
};

/// Maximum number of non-end_of_turn `done` frames before the drain loop
/// gives up. Prevents infinite silence-feed loops when the model keeps
/// producing `<|listen|>` without ever reaching `<|turn_eos|>`.
constexpr int kMaxDrainDoneRounds = 30;
/// Wall-clock cap for draining a websocket turn before the test returns.
constexpr std::chrono::seconds kDrainWsMaxWait{45};

/**
 * @brief Drain websocket frames until an end-of-turn done frame arrives.
 *
 * Returns early (with `got_end_of_turn = false`) if `kMaxDrainDoneRounds`
 * resumable done frames arrive without `end_of_turn`, or when `min_text_len`
 * characters of text have been collected.
 */
inline void drain_ws_until_done(
	WsTestClient & client,
	WsTurnResult & result,
	std::function<void()> const & send_silence,
	std::size_t min_text_len = 0,
	std::chrono::milliseconds max_wait = kDrainWsMaxWait)
{
	auto const deadline = std::chrono::steady_clock::now() + max_wait;
	int done_rounds = 0;
	while (std::chrono::steady_clock::now() < deadline)
	{
		auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
			deadline - std::chrono::steady_clock::now());
		auto const frame = client.recv_msgpack(remaining);
		if (!frame.contains("type"))
		{
			continue;
		}
		auto const & type_str = frame["type"].get<std::string>();

		if (type_str == "speech_chunk")
		{
			if (frame.contains("text"))
			{
				result.text += frame["text"].get<std::string>();
			}
			result.speech_chunks.push_back(frame);
			if (frame.contains("seq"))
			{
				nlohmann::json ack;
				ack["type"] = "speech_chunk_played";
				ack["seq"] = frame["seq"].get<std::uint32_t>();
				client.send_msgpack(ack);
			}
			send_silence();
		}
		else if (type_str == "text")
		{
			if (frame.contains("text"))
			{
				result.text += frame["text"].get<std::string>();
			}
		}
		else if (type_str == "done")
		{
			if (frame.contains("end_of_turn") && frame["end_of_turn"].get<bool>())
			{
				result.got_end_of_turn = true;
				return;
			}
			if (++done_rounds >= kMaxDrainDoneRounds)
			{
				spdlog::warn(
					"drain_ws_until_done: hit {} resumable done rounds without end_of_turn, "
					"returning with text.size()={}",
					done_rounds,
					result.text.size());
				return;
			}
			send_silence();
		}
		else if (type_str == "error")
		{
			auto const msg =
				frame.contains("message") ? frame["message"].get<std::string>() : "unknown";
			FAIL("WebSocket error frame: " << msg);
		}
		// "status", "video_ack", "ready", etc.: silently skipped.

		if (min_text_len > 0 && result.text.size() >= min_text_len)
		{
			return;
		}
	}
}

/**
 * @brief Drain any already-in-flight websocket frames until the connection goes quiet.
 *
 * This is used by tests that intentionally stop feeding continuation audio but
 * still want one last server-side decode already in progress to finish before
 * they close the websocket.
 */
inline void drain_ws_until_quiet(
	WsTestClient & client,
	WsTurnResult & result,
	std::chrono::milliseconds quiet_timeout = std::chrono::seconds{2})
{
	auto deadline = std::chrono::steady_clock::now() + quiet_timeout;
	while (std::chrono::steady_clock::now() < deadline)
	{
		try
		{
			auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
				deadline - std::chrono::steady_clock::now());
			auto const frame = client.recv_msgpack(remaining);
			if (!frame.contains("type"))
			{
				continue;
			}
			auto const & type_str = frame["type"].get<std::string>();
			if (type_str == "speech_chunk")
			{
				deadline = std::chrono::steady_clock::now() + quiet_timeout;
				if (frame.contains("text"))
				{
					result.text += frame["text"].get<std::string>();
				}
				result.speech_chunks.push_back(frame);
				if (frame.contains("seq"))
				{
					nlohmann::json ack;
					ack["type"] = "speech_chunk_played";
					ack["seq"] = frame["seq"].get<std::uint32_t>();
					client.send_msgpack(ack);
				}
			}
			else if (type_str == "text")
			{
				deadline = std::chrono::steady_clock::now() + quiet_timeout;
				if (frame.contains("text"))
				{
					result.text += frame["text"].get<std::string>();
				}
			}
			else if (type_str == "done")
			{
				deadline = std::chrono::steady_clock::now() + quiet_timeout;
				if (frame.contains("end_of_turn") && frame["end_of_turn"].get<bool>())
				{
					result.got_end_of_turn = true;
				}
			}
			else if (type_str == "error")
			{
				auto const msg =
					frame.contains("message") ? frame["message"].get<std::string>() : "unknown";
				FAIL("WebSocket error frame: " << msg);
			}
		}
		catch (boost::system::system_error const & ex)
		{
			if (ex.code() == boost::beast::error::timeout)
			{
				// This helper is specifically used to observe that the websocket has
				// gone quiet after any already-in-flight server work finishes. In
				// that narrow context, a read timeout means "no further frames
				// arrived within the quiet window", which is the expected stopping
				// condition. Other socket errors still propagate to fail the test.
				return;
			}
			throw;
		}
	}
}

/// Approximate browser microphone chunk size after 48 kHz → 16 kHz downsampling.
static constexpr std::size_t kE2eBrowserAudioChunkSamples = 1365;

/**
 * @brief Send a WAV file as browser-sized audio_in chunks over WebSocket.
 */
inline void send_wav_as_browser_chunks(
	WsTestClient & client, std::filesystem::path const & wav_path)
{
	auto const pcm = read_wav_mono(wav_path).pcm;
	for (std::size_t offset = 0; offset < pcm.size(); offset += kE2eBrowserAudioChunkSamples)
	{
		auto const chunk_end = std::min(offset + kE2eBrowserAudioChunkSamples, pcm.size());
		auto const span = std::span{pcm}.subspan(offset, chunk_end - offset);
		std::vector<std::uint8_t> bytes(span.size_bytes());
		std::memcpy(bytes.data(), span.data(), span.size_bytes());
		nlohmann::json frame;
		frame["type"] = "audio_in";
		frame["d"] = nlohmann::json::binary(std::move(bytes));
		client.send_msgpack(frame);
	}
}

/// Number of resume-sized silence chunks that reliably trigger generation.
static constexpr int kE2eTriggerSilenceChunkCount = 2;

/**
 * @brief Send N silence chunks to trigger/resume generation.
 */
inline void send_silence_trigger(WsTestClient & client, int count = kE2eTriggerSilenceChunkCount)
{
	std::vector<float> const silence(
		kResumeChunkSamples, llama_omni_server::test::kNearZeroAmplitude);
	for (int i = 0; i < count; ++i)
	{
		std::vector<std::uint8_t> bytes(silence.size() * sizeof(float));
		std::memcpy(bytes.data(), silence.data(), bytes.size());
		nlohmann::json frame;
		frame["type"] = "audio_in";
		frame["d"] = nlohmann::json::binary(std::move(bytes));
		client.send_msgpack(frame);
	}
}

}  // namespace llama_omni_server::test
