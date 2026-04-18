/**
 * @file   server.hpp
 * @brief  WebSocket server class declaration for llama-omni-server.
 *
 * `WebSocketServer` accepts TCP connections, performs the HTTP→WebSocket
 * upgrade via Boost.Beast, and manages the single-session lifecycle.
 * At most one session is active at any time; a second connection receives
 * an `ErrorFrame{server_busy}` and is immediately closed.
 *
 * The server is designed to run as a `boost::cobalt::task<void>` on the
 * cobalt main executor.  All GPU-bound work is dispatched to `gpu_ex` via
 * `cobalt::spawn`.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

#include <boost/asio/thread_pool.hpp>
#include <boost/cobalt/promise.hpp>
#include <boost/cobalt/task.hpp>

#include "config/config_loader.hpp"

namespace llama_omni_server
{

class ModelManager;

/**
 * @brief Shared ownership root for long-lived server dependencies.
 *
 * `WebSocketServer` and all accepted sessions retain a `shared_ptr` to this
 * object so configuration, loaded models, and the GPU executor outlive every
 * asynchronous coroutine that depends on them.
 */
struct AppRuntime
{
	/// Loaded application configuration used by the server and every session.
	AppConfig cfg;

	/// Shared loaded model manager used to create live sessions.
	std::shared_ptr<ModelManager> models;

	/// Shared single-thread GPU executor used by session/model work.
	std::shared_ptr<boost::asio::thread_pool> gpu_ex;
};

/**
 * @brief WebSocket server that accepts connections and manages the session lifecycle.
 *
 * Exactly one session is permitted at a time.  When a second client connects
 * while a session is active it receives `{"type":"error","code":"server_busy"}`
 * and the connection is closed.
 *
 * ### Lifecycle
 *
 * 1. Construct an `AppRuntime`, then `WebSocketServer server{runtime};`
 * 2. Run: `co_await server.run();` — suspends until `stop()` is called.
 * 3. Shutdown: call `server.stop()` from a signal handler or another task.
 *
 * `run()` is a `cobalt::task<void>` and must be `co_await`-ed from another
 * cobalt coroutine (e.g. `co_main`).  For testing, wrap it with
 * `cobalt::run(server.run())`.
 *
 * @note All coroutine work executes on the cobalt executor (single thread).
 *       `stop()` is the only method safe to call from another thread.
 */
class WebSocketServer
{
public:
	/**
	 * @brief Construct the server with injected dependencies.
	 *
	 * @param runtime  Shared ownership root for config, models, and GPU executor.
	 */
	explicit WebSocketServer(std::shared_ptr<AppRuntime> runtime);

	/**
	 * @brief Run the TCP accept loop.
	 *
	 * Binds to `cfg.server.host:cfg.server.port`, starts listening, and
	 * accepts connections in a loop until `stop()` is called.  Each accepted
	 * connection is handled inline before the next `accept` call.
	 *
	 * @return A cobalt task that completes when the server stops.
	 */
	// Kept as task (not promise) because cobalt::run() requires task.
	[[nodiscard]] boost::cobalt::task<void> run();

	/**
	 * @brief Signal the server to stop accepting new connections.
	 *
	 * Thread-safe. Posts an acceptor cancel/close onto the cobalt executor so a
	 * pending `async_accept` wakes immediately without requiring a dummy TCP
	 * connection.
	 */
	void stop();

	/**
	 * @brief Return the actual bound port.
	 *
	 * When `cfg.server.port == 0` the OS assigns a free ephemeral port; this
	 * method returns that port.  Only valid after `run()` has started and the
	 * acceptor is open.
	 *
	 * @return The bound port number, or 0 if `run()` has not yet started.
	 */
	[[nodiscard]] std::uint16_t port() const noexcept;

private:
	/// Shared ownership root retained for the entire server lifetime.
	std::shared_ptr<AppRuntime> runtime_;

	/// Set by stop() to keep cross-thread shutdown requests idempotent.
	std::atomic<bool> stop_requested_{false};

	/// Bound port stored after the acceptor is opened; read by port().
	/// Atomic because port() is called from the test/main thread while run()
	/// executes on the cobalt thread (same cross-thread access as stop()).
	std::atomic<std::uint16_t> bound_port_{0};

	/// Protects `stop_accept_` because `run()` installs it and `stop()` invokes it
	/// from another thread.
	mutable std::mutex stop_accept_mu_;

	/// Cross-thread callback that posts acceptor cancellation onto the cobalt
	/// executor while `run()` owns a live acceptor.
	std::function<void()> stop_accept_;
};

}  // namespace llama_omni_server
