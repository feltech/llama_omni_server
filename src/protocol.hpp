/**
 * @file   protocol.hpp
 * @brief  MsgPack wire protocol frame types for llama-omni-server.
 *
 * Defines the `InFrame` and `OutFrame` variant types that represent all
 * client→server and server→client messages in the WebSocket protocol.
 * Each frame serialises as a MsgPack map with a string `"type"` field.
 * Binary payloads use the `"d"` key and are encoded as a MsgPack `bin`
 * blob via `nlohmann::json::binary()` — NOT as a plain array — so that
 * float32/RGB data is transferred at its native byte size.
 *
 * `encode_frame` and `decode_frame` are implemented inline in this header
 * so that no separate translation unit is required for the protocol layer.
 *
 * @see PLAN.md §"Wire Protocol" for the full message table.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace llama_omni_server::wire
{

// ── Client → Server frame types ───────────────────────────────────────────────

/**
 * @brief Requests that the server start a new duplex session.
 *
 * Sent once by the client after the WebSocket upgrade. The server responds
 * with `ReadyFrame` if no session is active, or `ErrorFrame{server_busy}`
 * if one is already running.
 */
struct InitFrame
{
};

/**
 * @brief Requests that the server close the current session cleanly.
 */
struct CloseFrame
{
};

/**
 * @brief Carries a chunk of raw audio captured by the browser.
 *
 * @note `raw` contains little-endian float32 samples at 16 kHz mono.
 *       The byte count must be divisible by 4.  Values are expected in
 *       [-1.0, 1.0]; the server warns (but does not reject) out-of-range data.
 */
struct AudioInFrame
{
	std::vector<std::byte> raw;	 ///< Raw float32 LE bytes, 16 kHz mono.
};

/**
 * @brief Carries a single video frame captured by the browser.
 *
 * The client pre-resizes the frame to 448×448 RGB before sending.
 *
 * @note `raw` must be exactly 448×448×3 = 602 112 bytes.
 *       `seq` is a monotonically increasing sequence number; the server
 *       echoes it back in the corresponding `VideoAckFrame`.
 */
struct VideoInFrame
{
	std::vector<std::byte> raw;	 ///< RGB uint8 bytes, 448×448×3.
	std::uint32_t seq;			 ///< Client-assigned sequence number.
};

/**
 * @brief Carries typed text input for a simplex turn.
 *
 * When the user types instead of speaking, the client sends this frame.
 * The server starts a simplex inference turn using `text` as the prompt.
 */
struct TextInFrame
{
	std::string text;  ///< UTF-8 text prompt.
};

/**
 * @brief Acknowledges that the client has finished playing a speech chunk.
 *
 * `seq` must match a previously emitted `SpeechChunkFrame.seq`.
 */
struct SpeechChunkPlayedFrame
{
	std::uint32_t seq;	///< Sequence number from the corresponding `SpeechChunkFrame`.
};

/**
 * @brief Produced by `decode_frame()` when a MsgPack message cannot be parsed.
 *
 * This is NOT a real wire type — it is never transmitted. It is returned
 * as a sentinel to allow the session loop to log and skip malformed frames
 * without throwing.
 */
struct BadFrame
{
	std::string reason;	 ///< Human-readable description of the decode error.
};

/** @brief Discriminated union of all client→server message types. */
using InFrame = std::variant<
	InitFrame,
	CloseFrame,
	AudioInFrame,
	VideoInFrame,
	TextInFrame,
	SpeechChunkPlayedFrame,
	BadFrame>;

// ── Server → Client frame types ───────────────────────────────────────────────

/**
 * @brief Sent to the client after a successful `InitFrame`.
 *
 * Signals that the session is active and the server is ready to receive audio
 * and video frames.
 */
struct ReadyFrame
{
};

/**
 * @brief Sent when a second client tries to connect while a session is active.
 *
 * Encoded on the wire as `{"type":"error","code":"server_busy","message":"..."}`.
 */
struct ServerBusyFrame
{
};

/**
 * @brief Generic error notification from the server.
 *
 * @note Well-known values of `code`: `"server_busy"`, `"model_error"`,
 *       `"bad_frame"`.
 */
struct ErrorFrame
{
	std::string code;	  ///< Machine-readable error code.
	std::string message;  ///< Human-readable description.
};

/**
 * @brief Carries a chunk of TTS-synthesised audio to play on the client.
 *
 * @note `raw` contains little-endian float32 samples at 24 kHz mono.
 */
struct AudioOutFrame
{
	std::vector<std::byte> raw;	 ///< Raw float32 LE bytes, 24 kHz mono.
};

/**
 * @brief Chunk-aligned spoken output from the server.
 *
 * The text and PCM correspond to the same `<|chunk_eos|>`-delimited speech
 * chunk. The client must acknowledge playback with `SpeechChunkPlayedFrame`
 * before the server continues generation.
 */
struct SpeechChunkFrame
{
	std::uint32_t seq;			 ///< Server-assigned monotonic speech-chunk sequence number.
	std::string text;			 ///< Visible text aligned to this audio chunk.
	std::vector<std::byte> raw;	 ///< Raw float32 LE bytes, 24 kHz mono.
};

/**
 * @brief Streaming text token from the LLM.
 *
 * One frame is sent per text token as the model generates them, enabling
 * progressive display in the browser UI.
 */
struct TextOutFrame
{
	std::string text;  ///< Decoded text for the current token.
};

/**
 * @brief Signals that the current generation turn has ended.
 *
 * Sent after the last `TextOutFrame` / `AudioOutFrame` for a turn.
 */
struct DoneFrame
{
	bool end_of_turn;  ///< True for `<|turn_eos|>`, false for `<|listen|>`.
};

/**
 * @brief Periodic status snapshot sent to the client.
 *
 * Sent on a configurable timer (`logging.status_period_ms`, default 200 ms).
 */
struct StatusFrame
{
	bool listening;				  ///< True when incoming audio is currently accepted.
	std::uint32_t video_last_ms;  ///< Age of the last processed video frame (ms).
};

/**
 * @brief Echoes back the processed video frame to the client.
 *
 * Sent immediately after CLIP encoding completes so the browser can display
 * the model's view of the video.
 */
struct VideoAckFrame
{
	std::uint32_t seq;			 ///< Sequence number from the corresponding `VideoInFrame`.
	std::vector<std::byte> raw;	 ///< Processed RGB uint8 bytes, 448×448×3.
};

/** @brief Discriminated union of all server→client message types. */
using OutFrame = std::variant<
	ReadyFrame,
	ServerBusyFrame,
	ErrorFrame,
	AudioOutFrame,
	SpeechChunkFrame,
	TextOutFrame,
	DoneFrame,
	StatusFrame,
	VideoAckFrame>;

// ── Encode / decode ───────────────────────────────────────────────────────────

/**
 * @brief Encode an `OutFrame` to a MsgPack byte buffer.
 *
 * Each frame type serialises as a MsgPack map.  The `"type"` key always
 * carries a string value used for branching on the client side.  Binary
 * payloads are stored under the `"d"` key as a MsgPack `bin` blob (not a
 * JSON array) to avoid the 8× overhead of encoding float/byte values as
 * 64-bit floats.
 *
 * @param frame  The frame to encode.
 * @return       Owned vector of MsgPack bytes ready for `ws.async_write`.
 */
[[nodiscard]] inline std::vector<std::uint8_t> encode_frame(OutFrame frame)
{
	nlohmann::json obj;

	/// Helper: copy std::byte vector into a uint8 vector for json::binary().
	auto bytes_to_uint8 = [](std::vector<std::byte> const & src)
	{
		std::vector<std::uint8_t> out(src.size());
		for (std::size_t i = 0; i < src.size(); ++i)
		{
			out[i] = static_cast<std::uint8_t>(src[i]);
		}
		return out;
	};

	std::visit(
		[&](auto & frm)
		{
			using T = std::decay_t<decltype(frm)>;

			if constexpr (std::is_same_v<T, ReadyFrame>)
			{
				obj["type"] = "ready";
			}
			else if constexpr (std::is_same_v<T, ServerBusyFrame>)
			{
				// ServerBusy is a specialised error frame on the wire.
				obj["type"] = "error";
				obj["code"] = "server_busy";
				obj["message"] = "A session is already active";
			}
			else if constexpr (std::is_same_v<T, ErrorFrame>)
			{
				obj["type"] = "error";
				obj["code"] = frm.code;
				obj["message"] = frm.message;
			}
			else if constexpr (std::is_same_v<T, AudioOutFrame>)
			{
				obj["type"] = "audio_out";
				// Binary blob: avoids per-byte float64 encoding overhead.
				obj["d"] = nlohmann::json::binary(bytes_to_uint8(frm.raw));
			}
			else if constexpr (std::is_same_v<T, SpeechChunkFrame>)
			{
				obj["type"] = "speech_chunk";
				obj["seq"] = frm.seq;
				obj["text"] = frm.text;
				obj["d"] = nlohmann::json::binary(bytes_to_uint8(frm.raw));
			}
			else if constexpr (std::is_same_v<T, TextOutFrame>)
			{
				obj["type"] = "text";
				obj["text"] = frm.text;
			}
			else if constexpr (std::is_same_v<T, DoneFrame>)
			{
				obj["type"] = "done";
				obj["end_of_turn"] = frm.end_of_turn;
			}
			else if constexpr (std::is_same_v<T, StatusFrame>)
			{
				obj["type"] = "status";
				obj["listening"] = frm.listening;
				obj["video_last_ms"] = frm.video_last_ms;
			}
			else if constexpr (std::is_same_v<T, VideoAckFrame>)
			{
				obj["type"] = "video_ack";
				obj["seq"] = frm.seq;
				// Binary blob for the RGB bytes — same rationale as audio_out.
				obj["d"] = nlohmann::json::binary(bytes_to_uint8(frm.raw));
			}
		},
		frame);

	return nlohmann::json::to_msgpack(obj);
}

/**
 * @brief Decode a MsgPack byte buffer into an `InFrame`.
 *
 * Parses the `"type"` string field and dispatches to the appropriate struct.
 * Binary payloads are expected as MsgPack `bin` blobs (decoded by nlohmann
 * as `json::binary_t` containing `std::vector<std::uint8_t>`).
 *
 * @param data  View over the raw MsgPack bytes from the WebSocket receive buffer.
 * @return      The decoded frame, or `BadFrame{reason}` on any parse error.
 */
// The wire protocol is intentionally parsed in one place so every frame type
// stays visible in a single dispatch table.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] inline InFrame decode_frame(std::span<std::uint8_t const> data)
{
	nlohmann::json obj;
	try
	{
		// from_msgpack deserialises the binary buffer; strict=true rejects
		// trailing bytes that would indicate a framing error.
		obj = nlohmann::json::from_msgpack(
			data.begin(),
			data.end(),
			/*strict=*/true,
			/*allow_exceptions=*/true);
	}
	catch (std::exception const & ex)
	{
		return BadFrame{std::string{"msgpack decode error: "} + ex.what()};
	}

	if (!obj.is_object())
	{
		return BadFrame{"frame is not a MsgPack map"};
	}

	auto type_it = obj.find("type");
	if (type_it == obj.end() || !type_it->is_string())
	{
		return BadFrame{"missing or non-string \"type\" field"};
	}

	std::string const type = type_it->get<std::string>();

	/// Helper: extract a binary_t payload under key "d" as std::byte vector.
	auto get_binary = [&](std::string const & key) -> std::vector<std::byte>
	{
		auto iter = obj.find(key);
		if (iter == obj.end() || !iter->is_binary())
		{
			return {};
		}
		auto const & bin = iter->get_binary();
		std::vector<std::byte> out(bin.size());
		for (std::size_t i = 0; i < bin.size(); ++i)
		{
			out[i] = static_cast<std::byte>(bin[i]);
		}
		return out;
	};

	try
	{
		if (type == "init")
		{
			return InitFrame{};
		}
		if (type == "close")
		{
			return CloseFrame{};
		}
		if (type == "audio_in")
		{
			return AudioInFrame{get_binary("d")};
		}
		if (type == "video_in")
		{
			auto seq_it = obj.find("seq");
			if (seq_it == obj.end() || !seq_it->is_number_unsigned())
			{
				return BadFrame{"video_in missing or invalid \"seq\" field"};
			}
			return VideoInFrame{.raw = get_binary("d"), .seq = seq_it->get<std::uint32_t>()};
		}
		if (type == "text")
		{
			auto text_it = obj.find("text");
			if (text_it == obj.end() || !text_it->is_string())
			{
				return BadFrame{"text frame missing or invalid \"text\" field"};
			}
			return TextInFrame{text_it->get<std::string>()};
		}
		if (type == "speech_chunk_played")
		{
			auto seq_it = obj.find("seq");
			if (seq_it == obj.end() || !seq_it->is_number_unsigned())
			{
				return BadFrame{"speech_chunk_played missing or invalid \"seq\" field"};
			}
			return SpeechChunkPlayedFrame{.seq = seq_it->get<std::uint32_t>()};
		}
	}
	catch (std::exception const & ex)
	{
		return BadFrame{std::string{"frame field extraction error: "} + ex.what()};
	}

	return BadFrame{std::string{"unknown frame type: "} + type};
}

}  // namespace llama_omni_server::wire
