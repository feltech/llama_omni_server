/**
 * @file   test_protocol.cpp
 * @brief  Unit tests for MsgPack frame encode/decode (no GPU, no network).
 *
 * Tests the `encode_frame` and `decode_frame` functions defined in
 * `src/protocol.hpp`.  Each test encodes a frame to MsgPack bytes and
 * round-trips it through decode, verifying field values.
 *
 * Binary payloads are verified to survive the binary→bytes→binary round-trip
 * without becoming JSON arrays (which would cause a 8× size explosion).
 */

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "protocol.hpp"

namespace llama_omni_server::test
{

namespace
{

namespace wire = llama_omni_server::wire;

// ── Test data constants ───────────────────────────────────────────────────────

/// Distinct byte values for audio_in round-trip tests.
constexpr std::uint8_t kAudioByte0 = 0xAA;
constexpr std::uint8_t kAudioByte1 = 0xBB;
constexpr std::uint8_t kAudioByte2 = 0xCC;
constexpr std::uint8_t kAudioByte3 = 0xDD;

/// Sequence number for video_in tests.
constexpr std::uint32_t kVideoInSeq = 7U;

/// Sequence number for video_ack tests.
constexpr std::uint32_t kVideoAckSeq = 42U;

/// Sequence number for speech chunk tests.
constexpr std::uint32_t kSpeechChunkSeq = 9U;

/// Byte values for video_ack round-trip tests.
constexpr std::uint8_t kVidByte0 = 0xFF;
constexpr std::uint8_t kVidByte1 = 0x80;
constexpr std::uint8_t kVidByte2 = 0x40;

/// Timing value for StatusFrame tests.
constexpr std::uint32_t kVideoLastMs = 500U;

/// Garbage bytes that do not form a valid MsgPack document.
constexpr std::uint8_t kGarbageByte0 = 0xFF;
constexpr std::uint8_t kGarbageByte1 = 0xFE;
constexpr std::uint8_t kGarbageByte2 = 0xFD;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Make a std::vector<std::byte> from an initialiser list of uint8 values.
std::vector<std::byte> make_bytes(std::initializer_list<std::uint8_t> vals)
{
	std::vector<std::byte> out;
	out.reserve(vals.size());
	for (auto val : vals)
	{
		out.push_back(static_cast<std::byte>(val));
	}
	return out;
}

/// Round-trip an OutFrame through encode_frame → from_msgpack → InFrame decode.
/// Returns the top-level nlohmann::json object for field inspection.
nlohmann::json encode_then_parse(wire::OutFrame const & frame)
{
	auto const bytes = wire::encode_frame(frame);
	return nlohmann::json::from_msgpack(bytes.begin(), bytes.end());
}

/// Encode a JSON object as MsgPack and pass it to decode_frame.
wire::InFrame decode_json(nlohmann::json const & msg)
{
	auto const bytes = nlohmann::json::to_msgpack(msg);
	return wire::decode_frame(std::span<std::uint8_t const>{bytes.data(), bytes.size()});
}

}  // namespace

// ── encode_frame tests ────────────────────────────────────────────────────────

SCENARIO("encode_frame produces valid MsgPack maps")
{
	GIVEN("a ReadyFrame")
	{
		auto const obj = encode_then_parse(wire::ReadyFrame{});
		THEN("type field is \"ready\"")
		{
			REQUIRE(obj["type"].get<std::string>() == "ready");
		}
	}

	GIVEN("a ServerBusyFrame")
	{
		auto const obj = encode_then_parse(wire::ServerBusyFrame{});
		THEN("type is \"error\" and code is \"server_busy\"")
		{
			REQUIRE(obj["type"].get<std::string>() == "error");
			REQUIRE(obj["code"].get<std::string>() == "server_busy");
			CHECK_FALSE(obj["message"].get<std::string>().empty());
		}
	}

	GIVEN("an ErrorFrame with custom code and message")
	{
		auto const obj =
			encode_then_parse(wire::ErrorFrame{.code = "model_error", .message = "out of memory"});
		THEN("type is \"error\" and fields match")
		{
			REQUIRE(obj["type"].get<std::string>() == "error");
			REQUIRE(obj["code"].get<std::string>() == "model_error");
			REQUIRE(obj["message"].get<std::string>() == "out of memory");
		}
	}

	GIVEN("an AudioOutFrame with raw bytes")
	{
		auto const raw = make_bytes({kAudioByte0, kAudioByte1, kAudioByte2, kAudioByte3});
		auto const obj = encode_then_parse(wire::AudioOutFrame{raw});
		THEN("type is \"audio_out\" and d is a binary blob")
		{
			REQUIRE(obj["type"].get<std::string>() == "audio_out");
			// nlohmann decodes MsgPack bin as json::binary_t (not an array).
			REQUIRE(obj["d"].is_binary());
			auto const & bin = obj["d"].get_binary();
			REQUIRE(bin.size() == 4U);
			CHECK(bin[0] == kAudioByte0);
			CHECK(bin[3] == kAudioByte3);
		}
	}

	GIVEN("a SpeechChunkFrame with text, seq, and raw bytes")
	{
		auto const raw = make_bytes({kAudioByte0, kAudioByte1, kAudioByte2, kAudioByte3});
		auto const obj = encode_then_parse(
			wire::SpeechChunkFrame{.seq = kSpeechChunkSeq, .text = "hello", .raw = raw});
		THEN("type is \"speech_chunk\" and all fields match")
		{
			REQUIRE(obj["type"].get<std::string>() == "speech_chunk");
			REQUIRE(obj["seq"].get<std::uint32_t>() == kSpeechChunkSeq);
			REQUIRE(obj["text"].get<std::string>() == "hello");
			REQUIRE(obj["d"].is_binary());
			auto const & bin = obj["d"].get_binary();
			REQUIRE(bin.size() == 4U);
			CHECK(bin[0] == kAudioByte0);
			CHECK(bin[3] == kAudioByte3);
		}
	}

	GIVEN("a TextOutFrame")
	{
		auto const obj = encode_then_parse(wire::TextOutFrame{"hello"});
		THEN("type is \"text\" and text field matches")
		{
			REQUIRE(obj["type"].get<std::string>() == "text");
			REQUIRE(obj["text"].get<std::string>() == "hello");
		}
	}

	GIVEN("a DoneFrame with end_of_turn=true")
	{
		auto const obj = encode_then_parse(wire::DoneFrame{true});
		THEN("type is \"done\" and end_of_turn is true")
		{
			REQUIRE(obj["type"].get<std::string>() == "done");
			REQUIRE(obj["end_of_turn"].get<bool>() == true);
		}
	}

	GIVEN("a StatusFrame")
	{
		auto const obj =
			encode_then_parse(wire::StatusFrame{.listening = true, .video_last_ms = kVideoLastMs});
		THEN("all fields are present and correct")
		{
			REQUIRE(obj["type"].get<std::string>() == "status");
			REQUIRE(obj["listening"].get<bool>() == true);
			REQUIRE(obj["video_last_ms"].get<std::uint32_t>() == kVideoLastMs);
		}
	}

	GIVEN("a VideoAckFrame")
	{
		auto const raw = make_bytes({kVidByte0, kVidByte1, kVidByte2});
		auto const obj = encode_then_parse(wire::VideoAckFrame{.seq = kVideoAckSeq, .raw = raw});
		THEN("type is \"video_ack\", seq is correct, and d is a binary blob")
		{
			REQUIRE(obj["type"].get<std::string>() == "video_ack");
			REQUIRE(obj["seq"].get<std::uint32_t>() == kVideoAckSeq);
			REQUIRE(obj["d"].is_binary());
			auto const & bin = obj["d"].get_binary();
			REQUIRE(bin.size() == 3U);
			CHECK(bin[0] == kVidByte0);
		}
	}
}

// ── decode_frame tests ────────────────────────────────────────────────────────

SCENARIO("decode_frame recognises all client→server types")
{
	GIVEN("an init message")
	{
		auto const result = decode_json({{"type", "init"}});
		THEN("result is InitFrame")
		{
			REQUIRE(std::holds_alternative<wire::InitFrame>(result));
		}
	}

	GIVEN("a close message")
	{
		auto const result = decode_json({{"type", "close"}});
		THEN("result is CloseFrame")
		{
			REQUIRE(std::holds_alternative<wire::CloseFrame>(result));
		}
	}

	GIVEN("an audio_in message with binary payload")
	{
		// Build a MsgPack message with a bin-type "d" field.
		nlohmann::json msg;
		msg["type"] = "audio_in";
		msg["d"] = nlohmann::json::binary({kAudioByte0, kAudioByte1, kAudioByte2, kAudioByte3});

		auto const result = decode_json(msg);
		THEN("result is AudioInFrame with correct raw bytes")
		{
			REQUIRE(std::holds_alternative<wire::AudioInFrame>(result));
			auto const & audio_frame = std::get<wire::AudioInFrame>(result);
			REQUIRE(audio_frame.raw.size() == 4U);
			CHECK(audio_frame.raw[0] == std::byte{kAudioByte0});
			CHECK(audio_frame.raw[3] == std::byte{kAudioByte3});
		}
	}

	GIVEN("a video_in message with seq and binary payload")
	{
		nlohmann::json msg;
		msg["type"] = "video_in";
		msg["seq"] = kVideoInSeq;
		msg["d"] = nlohmann::json::binary({0x01, 0x02});

		auto const result = decode_json(msg);
		THEN("result is VideoInFrame with seq=7 and correct bytes")
		{
			REQUIRE(std::holds_alternative<wire::VideoInFrame>(result));
			auto const & video_frame = std::get<wire::VideoInFrame>(result);
			CHECK(video_frame.seq == kVideoInSeq);
			REQUIRE(video_frame.raw.size() == 2U);
		}
	}

	GIVEN("a text message")
	{
		auto const result = decode_json({{"type", "text"}, {"text", "hello world"}});
		THEN("result is TextInFrame with correct text")
		{
			REQUIRE(std::holds_alternative<wire::TextInFrame>(result));
			REQUIRE(std::get<wire::TextInFrame>(result).text == "hello world");
		}
	}

	GIVEN("a speech_chunk_played message")
	{
		auto const result =
			decode_json({{"type", "speech_chunk_played"}, {"seq", kSpeechChunkSeq}});
		THEN("result is SpeechChunkPlayedFrame with matching seq")
		{
			REQUIRE(std::holds_alternative<wire::SpeechChunkPlayedFrame>(result));
			REQUIRE(std::get<wire::SpeechChunkPlayedFrame>(result).seq == kSpeechChunkSeq);
		}
	}
}

SCENARIO("decode_frame returns BadFrame on malformed input")
{
	GIVEN("completely invalid bytes")
	{
		std::array<std::uint8_t, 3> garbage{kGarbageByte0, kGarbageByte1, kGarbageByte2};
		auto const result = wire::decode_frame(std::span<std::uint8_t const>{garbage});
		THEN("result is BadFrame")
		{
			REQUIRE(std::holds_alternative<wire::BadFrame>(result));
		}
	}

	GIVEN("a MsgPack map with no type field")
	{
		auto const result = decode_json({{"foo", "bar"}});
		THEN("result is BadFrame")
		{
			REQUIRE(std::holds_alternative<wire::BadFrame>(result));
		}
	}

	GIVEN("an unknown type string")
	{
		auto const result = decode_json({{"type", "unknown_msg_type"}});
		THEN("result is BadFrame")
		{
			REQUIRE(std::holds_alternative<wire::BadFrame>(result));
		}
	}

	GIVEN("a video_in message missing seq field")
	{
		nlohmann::json msg;
		msg["type"] = "video_in";
		msg["d"] = nlohmann::json::binary({0x01});
		// Note: seq is deliberately omitted.
		auto const result = decode_json(msg);
		THEN("result is BadFrame")
		{
			REQUIRE(std::holds_alternative<wire::BadFrame>(result));
		}
	}

	GIVEN("a text message missing text field")
	{
		auto const result = decode_json({{"type", "text"}});
		THEN("result is BadFrame")
		{
			REQUIRE(std::holds_alternative<wire::BadFrame>(result));
		}
	}

	GIVEN("a speech_chunk_played message missing seq field")
	{
		auto const result = decode_json({{"type", "speech_chunk_played"}});
		THEN("result is BadFrame")
		{
			REQUIRE(std::holds_alternative<wire::BadFrame>(result));
		}
	}
}

}  // namespace llama_omni_server::test
