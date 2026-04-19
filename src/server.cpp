/**
 * @file   server.cpp
 * @brief  WebSocket server implementation for llama-omni-server.
 */

#include "server.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/asio/error.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/websocket/stream.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/op.hpp>
#include <boost/cobalt/promise.hpp>
#include <boost/cobalt/task.hpp>
#include <boost/cobalt/this_coro.hpp>
#include <boost/cobalt/wait_group.hpp>
#include <boost/cobalt/with.hpp>
#include <spdlog/spdlog.h>

#include "protocol.hpp"
#include "session.hpp"

namespace llama_omni_server
{

namespace
{

/** @brief Short alias for the public wire-protocol namespace. */
namespace wire = llama_omni_server::wire;

/** @brief Executor type used by cobalt `use_op` coroutines in this file. */
using cobalt_executor_t = boost::cobalt::use_op_t::executor_with_default<boost::cobalt::executor>;
/** @brief TCP acceptor rebound onto the cobalt executor. */
using tcp_acceptor_t = boost::asio::ip::tcp::acceptor::rebind_executor<cobalt_executor_t>::other;
/** @brief TCP socket rebound onto the cobalt executor. */
using tcp_socket_t = boost::asio::ip::tcp::socket::rebind_executor<cobalt_executor_t>::other;
/** @brief Beast websocket stream layered on the cobalt-aware TCP socket. */
using ws_stream_t = boost::beast::websocket::stream<tcp_socket_t>;
/** @brief Steady timer rebound onto the cobalt executor. */
using steady_timer_t = boost::asio::steady_timer::rebind_executor<cobalt_executor_t>::other;

/// Capacity of the websocket outbound queue bridging session frames to the socket.
constexpr std::size_t kWsOutChanCap = 1U;
/// Capacity of the text-input queue sent from websocket input into `Session`.
constexpr std::size_t kTextInChanCap = 1U;
/// Capacity of the speech-chunk ack queue from websocket input into `Session`.
constexpr std::size_t kSpeechAckChanCap = 2U;
/// Capacity of the internal `Session` outbound frame queue.
constexpr std::size_t kSessionOutChanCap = 1U;
/// Capacity of buffered input audio chunks awaiting VAD/session consumption.
/// Must be large enough to hold a full burst of browser-sized audio chunks
/// without blocking the WS read loop, which also delivers speech ack frames.
/// A browser sends ~35 chunks per second of audio at 1365 samples/chunk;
/// 256 entries allows ~7 seconds of buffered audio.
constexpr std::size_t kAudioInChanCap = 1U;
/// Capacity of buffered video frames awaiting CLIP processing.
constexpr std::size_t kVideoInChanCap = 1U;

/**
 * @brief Wrap a contiguous byte span as an Asio const buffer.
 *
 * @param data  Encoded frame bytes that remain alive for the duration of the
 *              asynchronous write call.
 * @return Buffer view suitable for Beast/Asio write operations.
 */
boost::asio::const_buffer to_asio_buffer(std::span<std::uint8_t const> data) noexcept
{
	return boost::asio::buffer(data.data(), data.size());
}

/**
 * @brief Decode a websocket audio payload into float PCM samples.
 *
 * The wire protocol transports microphone audio as little-endian IEEE-754
 * floats packed into a byte vector. This helper validates that the payload is
 * sample-aligned, then copies the bytes into a `std::vector<float>`.
 *
 * @param raw  Raw audio bytes from a `wire::AudioInFrame`.
 * @return PCM samples ready for the session audio input channel.
 * @throws std::runtime_error if the payload size is not divisible by
 *         `sizeof(float)`.
 */
std::vector<float> decode_audio_bytes(std::vector<std::byte> const & raw)
{
	if (raw.size() % sizeof(float) != 0U)
	{
		throw std::runtime_error{"audio_in payload size is not divisible by sizeof(float)"};
	}

	std::vector<float> pcm(raw.size() / sizeof(float), 0.0F);
	std::memcpy(pcm.data(), raw.data(), raw.size());
	return pcm;
}

/**
 * @brief Encode float PCM samples into the websocket wire format.
 *
 * @param pcm  32-bit float PCM samples to serialise.
 * @return Byte vector suitable for `wire::AudioOutFrame` or
 *         `wire::SpeechChunkFrame`.
 */
std::vector<std::byte> encode_audio_bytes(std::span<float const> pcm)
{
	std::vector<std::byte> raw(pcm.size() * sizeof(float));
	std::memcpy(raw.data(), pcm.data(), raw.size());
	return raw;
}

// This visitor is intentionally a conversion table from internal frames to wire
// frames, so the repeated "return converted frame" branches are expected.
// NOLINTBEGIN(bugprone-branch-clone)
/**
 * @brief Convert one internal session output frame into the websocket wire format.
 *
 * @param frame  Session-level frame produced by `Session::run()`.
 * @return The equivalent websocket-visible frame.
 */
wire::OutFrame translate_session_frame_to_wire_frame(OutFrame frame)
{
	return std::visit(
		[](auto & frm) -> wire::OutFrame
		{
			using T = std::decay_t<decltype(frm)>;

			if constexpr (std::is_same_v<T, TextOutFrame>)
			{
				return wire::TextOutFrame{.text = std::move(frm.token)};
			}
			else if constexpr (std::is_same_v<T, DoneOutFrame>)
			{
				return wire::DoneFrame{.end_of_turn = frm.end_of_turn};
			}
			else if constexpr (std::is_same_v<T, ErrorOutFrame>)
			{
				return wire::ErrorFrame{.code = "model_error", .message = std::move(frm.message)};
			}
			else if constexpr (std::is_same_v<T, VideoAckOutFrame>)
			{
				return wire::VideoAckFrame{.seq = frm.seq, .raw = std::move(frm.raw)};
			}
			else if constexpr (std::is_same_v<T, SpeechChunkOutFrame>)
			{
				return wire::SpeechChunkFrame{
					.seq = frm.seq,
					.text = std::move(frm.text),
					.raw = encode_audio_bytes(frm.pcm)};
			}
			else
			{
				static_assert(sizeof(T) == 0, "Unhandled session out-frame type");
			}
			std::unreachable();
		},
		frame);
}
// NOLINTEND(bugprone-branch-clone)

/**
 * @brief Runtime state shared by the helper coroutines that implement one live websocket session.
 *
 * The websocket send loop, websocket receive loop, session run loop, session
 * output forwarder, and optional status loop all operate on the same set of
 * channels and the same `Session` instance. This struct centralises those
 * shared objects so they can be passed around as one reference-counted object.
 */
struct SessionRuntime
{
	// These members are fully initialised in the initializer list; the Session
	// depends on the earlier channels being constructed first.
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	SessionRuntime(std::shared_ptr<AppRuntime> app_runtime_in, cobalt_executor_t const & exec)
		: app_runtime{std::move(app_runtime_in)},
		  ws_out_ch{kWsOutChanCap, exec},
		  text_in_ch{kTextInChanCap, exec},
		  speech_ack_ch{kSpeechAckChanCap, exec},
		  session_out_ch{kSessionOutChanCap, exec},
		  audio_in_ch{kAudioInChanCap, exec},
		  video_in_ch{kVideoInChanCap, exec},
		  session{
			  app_runtime->cfg,
			  *app_runtime->models,
			  *app_runtime->gpu_ex,
			  session_out_ch,
			  audio_in_ch,
			  video_in_ch,
			  text_in_ch,
			  speech_ack_ch}
	{
	}

	/// Shared root that owns config, models, and GPU executor for this session.
	std::shared_ptr<AppRuntime> app_runtime;
	/// Frames awaiting websocket serialisation and socket write.
	boost::cobalt::channel<wire::OutFrame> ws_out_ch;
	/// Text typed by the client, forwarded into `Session`.
	boost::cobalt::channel<TextInput> text_in_ch;
	/// Speech chunk playback acknowledgements from the client.
	boost::cobalt::channel<SpeechChunkAckEvent> speech_ack_ch;
	/// Raw `Session` output frames awaiting protocol translation.
	boost::cobalt::channel<OutFrame> session_out_ch;
	/// Microphone PCM chunks awaiting `Session` consumption.
	boost::cobalt::channel<std::vector<float>> audio_in_ch;
	/// Incoming video frames awaiting CLIP/session consumption.
	boost::cobalt::channel<VideoFrame> video_in_ch;
	/// Owned live session instance backing this websocket connection.
	Session session;
};

/**
 * @brief Structured scope state for one websocket session.
 *
 * `with()` owns this object so the teardown path can access both the live
 * runtime and the child wait group regardless of how the main receive loop exits.
 */
struct SessionScope
{
	/// Child promises running for the current websocket session.
	boost::cobalt::wait_group workers;
	/// Runtime allocated after a valid init frame starts the live session.
	std::shared_ptr<SessionRuntime> runtime;
};

/**
 * @brief Structured scope state for the top-level server accept loop.
 *
 * The server owns at most one live session promise plus a dynamic wait group
 * of transient busy-reject tasks.
 */
struct ServerRunScope
{
	/// The single currently active live session, if any.
	std::optional<boost::cobalt::promise<void>> live_session;
	/// Short-lived tasks used to reject additional clients while busy.
	boost::cobalt::wait_group rejected_connections;
};

/// Poll interval while waiting for session startup to finish before emitting `ReadyFrame`.
constexpr std::chrono::milliseconds kSessionStartupPollMs{10};

/**
 * @brief Close a cobalt channel if it is still open.
 *
 * @param channel  Channel to close.
 */
template <typename T>
void close_channel_if_open(boost::cobalt::channel<T> & channel)
{
	if (channel.is_open())
	{
		channel.close();
	}
}

/**
 * @brief Decode one websocket MessagePack payload into a typed wire frame.
 *
 * @param buf  Beast flat buffer containing a complete websocket message.
 * @return Parsed `wire::InFrame`.
 */
wire::InFrame decode_ws_frame(boost::beast::flat_buffer const & buf)
{
	auto const data = buf.data();
	auto const bytes =
		std::span<std::uint8_t const>{static_cast<std::uint8_t const *>(data.data()), data.size()};
	return wire::decode_frame(bytes);
}

/**
 * @brief Encode and send one wire frame on the websocket.
 *
 * @param websocket  Accepted websocket connection.
 * @param frame      Outbound frame to serialise and write.
 */
boost::cobalt::promise<void> send_wire_frame(
	std::shared_ptr<ws_stream_t> websocket, wire::OutFrame frame)
{
	auto const bytes = wire::encode_frame(std::move(frame));
	co_await websocket->async_write(to_asio_buffer(bytes), boost::cobalt::use_op);
}

/**
 * @brief Read and decode one websocket message with cooperative cancellation.
 *
 * The underlying Beast `async_read` uses the current coroutine cancellation
 * state via `boost::cobalt::use_op`, so callers only need to ensure an
 * appropriate cancellation source has been installed for the coroutine.
 *
 * @param websocket  Accepted websocket connection.
 * @return Next decoded websocket input frame.
 */
boost::cobalt::promise<wire::InFrame> read_ws_frame(std::shared_ptr<ws_stream_t> websocket)
{
	boost::beast::flat_buffer buffer;
	co_await websocket->async_read(buffer, boost::cobalt::use_op);
	co_return decode_ws_frame(buffer);
}

/**
 * @brief Accept a second client long enough to report `server_busy`, then close it.
 *
 * @param socket  Newly accepted TCP socket rejected because another session is active.
 */
boost::cobalt::promise<void> busy_reject(tcp_socket_t socket)
{
	ws_stream_t websocket{std::move(socket)};
	try
	{
		spdlog::debug("server: busy_reject starting websocket upgrade");
		co_await websocket.async_accept(boost::cobalt::use_op);
		websocket.binary(true);
		co_await send_wire_frame(
			std::make_shared<ws_stream_t>(std::move(websocket)), wire::ServerBusyFrame{});
		spdlog::debug("server: busy_reject sent server_busy response");
	}
	catch (std::exception const & err)
	{
		spdlog::warn("server: busy_reject failed ({})", err.what());
	}
}

/**
 * @brief Drain translated outbound frames to the websocket socket.
 *
 * @param websocket  Accepted websocket connection.
 * @param runtime    Shared runtime state for the active session.
 */
boost::cobalt::promise<void> loop_ws_out_ch_to_wire(
	std::shared_ptr<ws_stream_t> websocket, std::shared_ptr<SessionRuntime> runtime)
{
	spdlog::debug("server: loop_ws_out_ch_to_wire starting");
	try
	{
		while (true)
		{
			wire::OutFrame frame = co_await runtime->ws_out_ch.read();
			auto const bytes = wire::encode_frame(std::move(frame));
			co_await websocket->async_write(to_asio_buffer(bytes), boost::cobalt::use_op);
		}
	}
	catch (boost::system::system_error const & err)
	{
		spdlog::debug("server: loop_ws_out_ch_to_wire exiting ({})", err.what());
	}
}

/**
 * @brief Forward raw `Session` output frames into websocket wire frames.
 *
 * @param runtime  Shared runtime state for the active session.
 */
boost::cobalt::promise<void> loop_session_out_to_ws_out_ch(std::shared_ptr<SessionRuntime> runtime)
{
	spdlog::debug("server: loop_session_out_to_ws_out_ch starting");
	try
	{
		while (true)
		{
			OutFrame frame = co_await runtime->session_out_ch.read();
			co_await runtime->ws_out_ch.write(
				translate_session_frame_to_wire_frame(std::move(frame)));
		}
	}
	catch (boost::system::system_error const & err)
	{
		spdlog::debug("server: loop_session_out_to_ws_out_ch exiting ({})", err.what());
	}

	if (runtime->ws_out_ch.is_open())
	{
		runtime->ws_out_ch.close();
	}
}

/**
 * @brief Run the core `Session` coroutine and publish its completion.
 *
 * This helper owns teardown of the session-facing channels so shutdown
 * initiation and shutdown completion do not overlap responsibilities.
 *
 * @param runtime  Shared runtime state for the active session.
 */
boost::cobalt::promise<void> run_session_then_cleanup(std::shared_ptr<SessionRuntime> runtime)
{
	spdlog::debug("server: run_session_then_cleanup starting");
	try
	{
		co_await runtime->session.run();
	}
	catch (std::exception const & err)
	{
		spdlog::error("server: run_session_then_cleanup caught exception: {}", err.what());
	}

	close_channel_if_open(runtime->text_in_ch);
	close_channel_if_open(runtime->speech_ack_ch);
	close_channel_if_open(runtime->audio_in_ch);
	close_channel_if_open(runtime->video_in_ch);
	close_channel_if_open(runtime->session_out_ch);
	spdlog::debug("server: run_session_then_cleanup cleanup complete");
}

/**
 * @brief Wait until a live session has completed startup or terminated early.
 *
 * `ReadyFrame` should only be sent once the `Session` has finished GPU/context
 * setup and system-prompt prefill. If the session exits first, startup failed
 * or was cancelled and no ready frame should be emitted.
 *
 * @param runtime  Shared runtime state for the active session.
 * @return `true` if startup completed, `false` if the session ended first.
 */
boost::cobalt::promise<bool> wait_for_session_startup(std::shared_ptr<SessionRuntime> runtime)
{
	// Cobalt/Asio expose the current executor through `this_coro::executor`.
	// NOLINTNEXTLINE(readability-static-accessed-through-instance)
	auto const exec = co_await boost::asio::this_coro::executor;
	steady_timer_t timer{cobalt_executor_t{exec}};

	while (true)
	{
		if (runtime->session.startup_complete())
		{
			co_return true;
		}
		if (!runtime->session_out_ch.is_open())
		{
			co_return false;
		}

		timer.expires_after(kSessionStartupPollMs);
		co_await timer.async_wait(boost::cobalt::use_op);
	}
}

// Cobalt/Asio expose the current executor as `this_coro::executor`; that API
// necessarily looks like static access through an instance.
// NOLINTBEGIN(readability-static-accessed-through-instance)
/**
 * @brief Periodically publish live session status frames.
 *
 * @param runtime           Shared runtime state for the active session.
 * @param status_period_ms  Interval between status frames in milliseconds.
 */
boost::cobalt::promise<void> loop_periodic_status_to_ws_out_ch(
	std::shared_ptr<SessionRuntime> runtime, int status_period_ms)
{
	auto const exec = co_await boost::asio::this_coro::executor;
	steady_timer_t timer{cobalt_executor_t{exec}};
	spdlog::debug(
		"server: loop_periodic_status_to_ws_out_ch starting with status_period_ms={}",
		status_period_ms);

	try
	{
		while (true)
		{
			timer.expires_after(std::chrono::milliseconds{status_period_ms});
			co_await timer.async_wait(boost::cobalt::use_op);

			co_await runtime->ws_out_ch.write(
				wire::OutFrame{wire::StatusFrame{
					.listening = runtime->session.listening(),
					.video_last_ms = runtime->session.video_last_ms()}});
		}
	}
	catch (boost::system::system_error const & err)
	{
		spdlog::debug("server: loop_periodic_status_to_ws_out_ch exiting ({})", err.what());
	}
}
// NOLINTEND(readability-static-accessed-through-instance)

/**
 * @brief Drive the websocket receive loop for one live session scope.
 *
 * @param scope        Structured scope state shared with the teardown path.
 * @param websocket    Accepted websocket connection.
 * @param app_runtime  Shared ownership root for config, models, and GPU executor.
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
boost::cobalt::task<void> loop_ws_receive_frame_and_dispach(
	std::shared_ptr<SessionScope> scope,
	std::shared_ptr<ws_stream_t> websocket,
	std::shared_ptr<AppRuntime> app_runtime)
{
	bool initialized = false;
	spdlog::debug("server: loop_ws_receive_frame_and_dispach starting");

	while (true)
	{
		wire::InFrame const decoded = co_await read_ws_frame(websocket);

		if (!initialized)
		{
			if (std::holds_alternative<wire::CloseFrame>(decoded))
			{
				break;
			}

			if (auto const * bad_frame = std::get_if<wire::BadFrame>(&decoded);
				bad_frame != nullptr)
			{
				co_await send_wire_frame(
					websocket, wire::ErrorFrame{.code = "bad_frame", .message = bad_frame->reason});
				continue;
			}

			if (!std::holds_alternative<wire::InitFrame>(decoded))
			{
				co_await send_wire_frame(
					websocket,
					wire::ErrorFrame{
						.code = "bad_frame", .message = "expected init before session input"});
				continue;
			}

			// Cobalt exposes the current coroutine executor through `this_coro::executor`.
			// NOLINTNEXTLINE(readability-static-accessed-through-instance)
			auto const exec = co_await boost::cobalt::this_coro::executor;
			scope->runtime = std::make_shared<SessionRuntime>(app_runtime, cobalt_executor_t{exec});
			spdlog::debug(
				"server: loop_ws_receive_frame_and_dispach init received; starting live session "
				"workers");
			scope->workers.push_back(run_session_then_cleanup(scope->runtime));
			scope->workers.push_back(loop_ws_out_ch_to_wire(websocket, scope->runtime));
			scope->workers.push_back(loop_session_out_to_ws_out_ch(scope->runtime));
			if (app_runtime->cfg.logging.status_period_ms > 0)
			{
				scope->workers.push_back(loop_periodic_status_to_ws_out_ch(
					scope->runtime, app_runtime->cfg.logging.status_period_ms));
			}

			initialized = true;
			if (co_await wait_for_session_startup(scope->runtime))
			{
				co_await scope->runtime->ws_out_ch.write(wire::OutFrame{wire::ReadyFrame{}});
				spdlog::debug("server: loop_ws_receive_frame_and_dispach ready frame queued");
			}
			else
			{
				spdlog::debug(
					"server: loop_ws_receive_frame_and_dispach session ended before startup "
					"completed");
			}
			continue;
		}

		if (std::holds_alternative<wire::InitFrame>(decoded))
		{
			co_await scope->runtime->ws_out_ch.write(
				wire::OutFrame{wire::ErrorFrame{
					.code = "bad_frame", .message = "session already initialised"}});
		}
		else if (std::holds_alternative<wire::CloseFrame>(decoded))
		{
			spdlog::debug("server: loop_ws_receive_frame_and_dispach CloseFrame");
			break;
		}
		else if (auto const * text_frame = std::get_if<wire::TextInFrame>(&decoded);
				 text_frame != nullptr)
		{
			co_await scope->runtime->text_in_ch.write(TextInput{.text = text_frame->text});
		}
		else if (auto const * ack_frame = std::get_if<wire::SpeechChunkPlayedFrame>(&decoded);
				 ack_frame != nullptr)
		{
			spdlog::debug(
				"server: loop_ws_receive_frame_and_dispach SpeechChunkPlayedFrame seq={}",
				ack_frame->seq);
			co_await scope->runtime->speech_ack_ch.write(
				SpeechChunkAckEvent{.seq = ack_frame->seq});
		}
		else if (auto const * audio_frame = std::get_if<wire::AudioInFrame>(&decoded);
				 audio_frame != nullptr)
		{
			std::vector<float> pcm = decode_audio_bytes(audio_frame->raw);
			scope->runtime->session.note_audio_received(pcm.size());
			co_await scope->runtime->audio_in_ch.write(std::move(pcm));
		}
		else if (auto const * video_frame = std::get_if<wire::VideoInFrame>(&decoded);
				 video_frame != nullptr)
		{
			co_await scope->runtime->video_in_ch.write(
				VideoFrame{.seq = video_frame->seq, .raw = video_frame->raw});
		}
		else if (auto const * bad_frame = std::get_if<wire::BadFrame>(&decoded);
				 bad_frame != nullptr)
		{
			co_await scope->runtime->ws_out_ch.write(
				wire::OutFrame{
					wire::ErrorFrame{.code = "bad_frame", .message = bad_frame->reason}});
		}
	}

	spdlog::debug("server: loop_ws_receive_frame_and_dispach closing websocket");
	co_await websocket->async_close(
		boost::beast::websocket::close_code::normal, boost::cobalt::use_op);
	spdlog::debug("server: loop_ws_receive_frame_and_dispach finished");
}

/**
 * @brief Tear down one live session scope after the receive loop exits.
 *
 * @param scope            Structured scope state shared with the session body.
 * @param current_error    Exception captured from the body, if any.
 */
boost::cobalt::task<void> teardown_session(
	std::shared_ptr<SessionScope> scope, std::exception_ptr current_error)
{
	spdlog::debug(
		"server: teardown_session starting (has_runtime={}, worker_count={})",
		scope->runtime != nullptr,
		scope->workers.size());
	if (scope->runtime != nullptr)
	{
		spdlog::debug("server: teardown_session requesting cooperative session shutdown");
		scope->runtime->session.request_shutdown();
		spdlog::debug("server: teardown_session closing session input channels");
		// Close all input channels; feeder tasks wake with broken_pipe and exit,
		// causing merged_ch to close and loop_duplex_events() to exit cleanly.
		close_channel_if_open(scope->runtime->audio_in_ch);
		close_channel_if_open(scope->runtime->text_in_ch);
		close_channel_if_open(scope->runtime->speech_ack_ch);
		close_channel_if_open(scope->runtime->video_in_ch);
		close_channel_if_open(scope->runtime->session_out_ch);
		close_channel_if_open(scope->runtime->ws_out_ch);
	}
	if (scope->workers.size() > 0U)
	{
		spdlog::debug("server: teardown_session awaiting worker exit");
		co_await scope->workers.await_exit(current_error);
	}
	spdlog::debug("server: teardown_session finished");
}

// This coroutine implements the full websocket session state machine, so the
// branching is deliberate and closely follows the wire protocol.
// `this_coro::executor` is also the intended Cobalt API for obtaining the
// current executor inside the coroutine.
// NOLINTBEGIN(readability-function-cognitive-complexity,bugprone-branch-clone,readability-static-accessed-through-instance)
/**
 * @brief Run one full websocket session using the production `Session` implementation.
 *
 * @param websocket    Accepted websocket connection.
 * @param app_runtime  Shared ownership root for config, models, and GPU executor.
 */
boost::cobalt::promise<void> run_session(
	std::shared_ptr<ws_stream_t> websocket, std::shared_ptr<AppRuntime> app_runtime)
{
	spdlog::debug("server: run_session starting");
	try
	{
		auto scope = std::make_shared<SessionScope>();
		co_await boost::cobalt::with(
			std::move(scope),
			[websocket = std::move(websocket),
			 app_runtime = std::move(app_runtime)](std::shared_ptr<SessionScope> scope_arg)
			{
				return loop_ws_receive_frame_and_dispach(
					std::move(scope_arg), websocket, app_runtime);
			},
			[](std::shared_ptr<SessionScope> scope_arg, std::exception_ptr current_error)
			{ return teardown_session(std::move(scope_arg), std::move(current_error)); });
	}
	catch (boost::system::system_error const & err)
	{
		if (err.code() == boost::asio::error::operation_aborted)
		{
			spdlog::debug("server: run_session ended with cancellation ({})", err.what());
		}
		else
		{
			spdlog::warn("server: run_session ended with error ({})", err.what());
		}
	}
	catch (std::exception const & err)
	{
		spdlog::error("server: run_session caught unexpected exception: {}", err.what());
	}
	spdlog::debug("server: run_session finished");
}
// NOLINTEND(readability-function-cognitive-complexity,bugprone-branch-clone,readability-static-accessed-through-instance)

/**
 * @brief Accept and run one websocket session from the already-accepted TCP socket.
 *
 * @param socket          Accepted TCP socket.
 * @param app_runtime     Shared ownership root for config, models, and GPU executor.
 */
boost::cobalt::promise<void> ws_accept_then_run_session(
	tcp_socket_t socket, std::shared_ptr<AppRuntime> app_runtime)
{
	try
	{
		spdlog::debug("server: ws_accept_then_run_session starting websocket handshake");
		auto websocket = std::make_shared<ws_stream_t>(std::move(socket));
		co_await websocket->async_accept(boost::cobalt::use_op);
		websocket->binary(true);
		spdlog::debug("server: ws_accept_then_run_session websocket handshake complete");
		co_await run_session(std::move(websocket), std::move(app_runtime));
	}
	catch (std::exception const & err)
	{
		spdlog::error("server: ws_accept_then_run_session caught exception: {}", err.what());
	}
}

/**
 * @brief Drive the server accept loop inside a structured `with()` scope.
 *
 * @param scope        Structured scope owning the live session promise and reject tasks.
 * @param acceptor     Shared acceptor listening for new TCP clients.
 * @param app_runtime  Shared ownership root for config, models, and GPU executor.
 */
boost::cobalt::task<void> loop_tcp_accept(
	std::shared_ptr<ServerRunScope> scope,
	std::shared_ptr<tcp_acceptor_t> acceptor,
	std::shared_ptr<AppRuntime> app_runtime)
{
	spdlog::debug("server: loop_tcp_accept starting");
	while (true)
	{
		if (scope->live_session && scope->live_session->ready())
		{
			spdlog::debug("server: loop_tcp_accept live session promise is ready; clearing slot");
			scope->live_session.reset();
		}
		scope->rejected_connections.reap();
		spdlog::debug(
			"server: loop_tcp_accept waiting for next TCP connection (has_live_session={}, "
			"rejected_pending={})",
			scope->live_session.has_value(),
			scope->rejected_connections.size());

		auto socket = co_await acceptor->async_accept(boost::cobalt::use_op);
		spdlog::debug("server: loop_tcp_accept TCP accept completed");

		if (scope->live_session && !scope->live_session->ready())
		{
			spdlog::debug("server: loop_tcp_accept server busy; rejecting extra connection");
			scope->rejected_connections.push_back(busy_reject(std::move(socket)));
		}
		else
		{
			spdlog::debug("server: loop_tcp_accept starting new live session");
			scope->live_session.emplace(ws_accept_then_run_session(std::move(socket), app_runtime));
		}
	}
}

/**
 * @brief Cancel and drain all top-level server child promises.
 *
 * @param scope          Structured accept-loop scope to tear down.
 * @param current_error  Exception captured from the accept-loop body, if any.
 */
boost::cobalt::task<void> teardown_loop_tcp_accept(
	std::shared_ptr<ServerRunScope> scope, std::exception_ptr current_error)
{
	spdlog::debug(
		"server: teardown_loop_tcp_accept starting (has_live_session={}, rejected_pending={})",
		scope->live_session.has_value(),
		scope->rejected_connections.size());
	if (scope->live_session)
	{
		spdlog::debug("server: teardown_loop_tcp_accept cancelling live session");
		scope->live_session->cancel(boost::asio::cancellation_type::all);
		spdlog::debug("server: teardown_loop_tcp_accept awaiting live session exit");
		co_await std::move(*scope->live_session);
		scope->live_session.reset();
	}

	spdlog::debug("server: teardown_loop_tcp_accept cancelling rejected connections");
	scope->rejected_connections.cancel(boost::asio::cancellation_type::all);
	spdlog::debug("server: teardown_loop_tcp_accept awaiting rejected connections exit");
	co_await scope->rejected_connections.await_exit(current_error);
	spdlog::debug("server: teardown_loop_tcp_accept finished");
}

}  // namespace

WebSocketServer::WebSocketServer(std::shared_ptr<AppRuntime> runtime) : runtime_{std::move(runtime)}
{
}

boost::cobalt::task<void> WebSocketServer::run()
{
	spdlog::debug("server: WebSocketServer::run starting");
	// Cobalt exposes the current coroutine executor through `this_coro::executor`.
	// NOLINTNEXTLINE(readability-static-accessed-through-instance)
	auto const cobalt_exec = co_await boost::cobalt::this_coro::executor;
	cobalt_executor_t const cobalt_ex{cobalt_exec};

	auto const address = boost::asio::ip::make_address(runtime_->cfg.server.host);
	boost::asio::ip::tcp::endpoint const endpoint{
		address, static_cast<std::uint16_t>(runtime_->cfg.server.port)};

	tcp_acceptor_t acceptor{cobalt_ex};
	acceptor.open(endpoint.protocol());
	acceptor.set_option(boost::asio::socket_base::reuse_address{true});
	acceptor.bind(endpoint);
	acceptor.listen(boost::asio::socket_base::max_listen_connections);

	bound_port_.store(acceptor.local_endpoint().port(), std::memory_order_release);
	stop_requested_.store(false, std::memory_order_release);
	auto acceptor_ptr = std::make_shared<tcp_acceptor_t>(std::move(acceptor));
	{
		std::scoped_lock const lock{stop_accept_mu_};
		stop_accept_ = [acceptor = acceptor_ptr]
		{
			boost::asio::post(
				acceptor->get_executor(),
				[acceptor]
				{
					spdlog::debug(
						"server: WebSocketServer::stop handler cancelling/closing acceptor");
					boost::system::error_code cancel_ec;
					auto const cancel_result = acceptor->cancel(cancel_ec);
					(void)cancel_result;
					if (cancel_ec)
					{
						spdlog::debug(
							"server: WebSocketServer::stop acceptor cancel returned '{}'",
							cancel_ec.message());
					}

					boost::system::error_code close_ec;
					auto const close_result = acceptor->close(close_ec);
					(void)close_result;
					if (close_ec && close_ec != boost::asio::error::bad_descriptor)
					{
						spdlog::debug(
							"server: WebSocketServer::stop acceptor close returned '{}'",
							close_ec.message());
					}
				});
		};
	}
	spdlog::debug(
		"server: WebSocketServer::run acceptor listening on {}:{}",
		runtime_->cfg.server.host,
		bound_port_.load(std::memory_order_acquire));

	try
	{
		auto scope = std::make_shared<ServerRunScope>();
		co_await boost::cobalt::with(
			std::move(scope),
			[acceptor_arg = acceptor_ptr,
			 app_runtime = runtime_](std::shared_ptr<ServerRunScope> scope_arg)
			{ return loop_tcp_accept(std::move(scope_arg), acceptor_arg, app_runtime); },
			[](std::shared_ptr<ServerRunScope> scope_arg, std::exception_ptr current_error)
			{ return teardown_loop_tcp_accept(std::move(scope_arg), std::move(current_error)); });
	}
	catch (boost::system::system_error const & err)
	{
		spdlog::debug(
			"server: WebSocketServer::run caught system_error during accept loop ({})", err.what());
		if (err.code() != boost::asio::error::operation_aborted)
		{
			spdlog::error("server: accept error: {}", err.what());
		}
	}

	{
		std::scoped_lock const lock{stop_accept_mu_};
		stop_accept_ = {};
	}

	boost::system::error_code close_ec;
	auto const close_result = acceptor_ptr->close(close_ec);
	(void)close_result;
	if (close_ec)
	{
		spdlog::warn("server: WebSocketServer::run error closing acceptor: {}", close_ec.message());
	}
	else
	{
		spdlog::debug("server: WebSocketServer::run acceptor closed");
	}
	spdlog::debug("server: WebSocketServer::run finished");
}

void WebSocketServer::stop()
{
	if (stop_requested_.exchange(true, std::memory_order_acq_rel))
	{
		spdlog::debug("server: WebSocketServer::stop ignoring duplicate stop request");
		return;
	}

	spdlog::debug("server: WebSocketServer::stop requested");

	std::function<void()> stop_accept;
	{
		std::scoped_lock const lock{stop_accept_mu_};
		stop_accept = stop_accept_;
	}

	if (!stop_accept)
	{
		spdlog::debug("server: WebSocketServer::stop found no active stop callback");
		return;
	}

	stop_accept();
}

std::uint16_t WebSocketServer::port() const noexcept
{
	return bound_port_.load(std::memory_order_acquire);
}

}  // namespace llama_omni_server
