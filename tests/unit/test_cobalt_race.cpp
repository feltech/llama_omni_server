/**
 * @file test_cobalt_race.cpp
 * @brief ICE-safe race() utility for cobalt::channel on GCC 15.
 *
 * GCC 15.2 ICEs in make_decl_rtl (varasm.cc:1459) on any use of
 * cobalt::race — even with trivial int channels. This file implements
 * and tests a drop-in race() that uses a one-shot result channel
 * internally, keeping cobalt::channel as the public API.
 */

#include <catch2/catch_test_macros.hpp>

#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/spawn.hpp>
#include <boost/cobalt/task.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace
{

// ── Types mirroring session.hpp ──────────────────────────────────────────────

struct TextInput
{
	std::string text;
};
struct CloseEvent
{
};
struct SpeechChunkAckEvent
{
	std::uint32_t seq{0};
};
using ControlEvent = std::variant<TextInput, CloseEvent, SpeechChunkAckEvent>;

// ── ICE-safe race utility ────────────────────────────────────────────────────
//
// Spawns two reader tasks that race to deliver the first result into a shared
// one-shot cobalt::channel. The losing reader stays suspended on its source
// channel until data arrives or the channel closes — at which point it tries
// to write to the (already-closed) result channel, gets an error, and exits.
//
// This is safe because:
// - cobalt is single-threaded cooperative: no data races
// - the result channel capacity is 1: second writer fails immediately
// - source channels are not modified or closed by race()
// - in a loop, old loser tasks drain naturally before new readers queue

/// Race two cobalt channels of different types. Returns variant<T1, T2>.
template <typename T1, typename T2>
boost::cobalt::task<std::variant<T1, T2>> race(
	boost::cobalt::channel<T1> & ch1, boost::cobalt::channel<T2> & ch2)
{
	using Result = std::variant<T1, T2>;

	auto exec = co_await boost::asio::this_coro::executor;
	auto result_ch = std::make_shared<boost::cobalt::channel<Result>>(1, exec);

	// Reader 1: ch1 → result_ch
	// NOLINTBEGIN(cppcoreguidelines-avoid-capturing-lambda-coroutines,bugprone-empty-catch)
	boost::cobalt::spawn(
		exec,
		[&ch1, result_ch]() -> boost::cobalt::task<void> {
			try
			{
				auto val = co_await ch1.read();
				co_await result_ch->write(Result{std::in_place_index<0>, std::move(val)});
			}
			catch (...)
			{
				// Expected: channel closed or result_ch full (loser path).
			}
		}(),
		boost::asio::detached);

	// Reader 2: ch2 → result_ch
	boost::cobalt::spawn(
		exec,
		[&ch2, result_ch]() -> boost::cobalt::task<void> {
			try
			{
				auto val = co_await ch2.read();
				co_await result_ch->write(Result{std::in_place_index<1>, std::move(val)});
			}
			catch (...)
			{
				// Expected: channel closed or result_ch full (loser path).
			}
		}(),
		boost::asio::detached);
	// NOLINTEND(cppcoreguidelines-avoid-capturing-lambda-coroutines,bugprone-empty-catch)

	auto result = co_await result_ch->read();
	result_ch->close();
	co_return result;
}

/// Race two cobalt channels of the same type. Returns the winning value.
template <typename T>
boost::cobalt::task<T> race(boost::cobalt::channel<T> & ch1, boost::cobalt::channel<T> & ch2)
{
	auto exec = co_await boost::asio::this_coro::executor;
	auto result_ch = std::make_shared<boost::cobalt::channel<T>>(1, exec);

	// NOLINTBEGIN(cppcoreguidelines-avoid-capturing-lambda-coroutines,bugprone-empty-catch)
	boost::cobalt::spawn(
		exec,
		[&ch1, result_ch]() -> boost::cobalt::task<void> {
			try
			{
				auto val = co_await ch1.read();
				co_await result_ch->write(std::move(val));
			}
			catch (...)
			{
				// Expected: channel closed or result_ch full (loser path).
			}
		}(),
		boost::asio::detached);

	boost::cobalt::spawn(
		exec,
		[&ch2, result_ch]() -> boost::cobalt::task<void> {
			try
			{
				auto val = co_await ch2.read();
				co_await result_ch->write(std::move(val));
			}
			catch (...)
			{
				// Expected: channel closed or result_ch full (loser path).
			}
		}(),
		boost::asio::detached);
	// NOLINTEND(cppcoreguidelines-avoid-capturing-lambda-coroutines,bugprone-empty-catch)

	auto result = co_await result_ch->read();
	result_ch->close();
	co_return result;
}

// ── Tests ────────────────────────────────────────────────────────────────────

boost::cobalt::task<void> test_race_same_type(boost::asio::io_context & ioc)
{
	boost::cobalt::channel<int> ch1(1, ioc.get_executor());
	boost::cobalt::channel<int> ch2(1, ioc.get_executor());

	static constexpr int kTestValue = 42;
	co_await ch1.write(kTestValue);

	auto result = co_await race(ch1, ch2);
	REQUIRE(result == kTestValue);
}

boost::cobalt::task<void> test_race_audio_wins(boost::asio::io_context & ioc)
{
	boost::cobalt::channel<std::vector<float>> audio_ch(1, ioc.get_executor());
	boost::cobalt::channel<ControlEvent> control_ch(1, ioc.get_executor());

	co_await audio_ch.write(std::vector<float>{1.0F});

	auto result = co_await race(audio_ch, control_ch);
	REQUIRE(std::holds_alternative<std::vector<float>>(result));
	REQUIRE(std::get<std::vector<float>>(result).size() == 1);
}

boost::cobalt::task<void> test_race_control_wins(boost::asio::io_context & ioc)
{
	boost::cobalt::channel<std::vector<float>> audio_ch(1, ioc.get_executor());
	boost::cobalt::channel<ControlEvent> control_ch(1, ioc.get_executor());

	co_await control_ch.write(ControlEvent{CloseEvent{}});

	auto result = co_await race(audio_ch, control_ch);
	REQUIRE(std::holds_alternative<ControlEvent>(result));
	REQUIRE(std::holds_alternative<CloseEvent>(std::get<ControlEvent>(result)));
}

boost::cobalt::task<void> test_race_loop(boost::asio::io_context & ioc)
{
	boost::cobalt::channel<std::vector<float>> audio_ch(4, ioc.get_executor());
	boost::cobalt::channel<ControlEvent> control_ch(4, ioc.get_executor());

	static constexpr float kSample1 = 1.0F;
	static constexpr float kSample2 = 2.0F;
	co_await audio_ch.write(std::vector<float>{kSample1});
	co_await audio_ch.write(std::vector<float>{kSample2});
	co_await control_ch.write(ControlEvent{CloseEvent{}});

	int audio_count = 0;
	bool got_close = false;

	static constexpr int kMaxIterations = 10;
	for (int iter = 0; iter < kMaxIterations && !got_close; ++iter)
	{
		auto evt = co_await race(audio_ch, control_ch);

		if (auto * pcm = std::get_if<std::vector<float>>(&evt))
		{
			(void)pcm;
			++audio_count;
		}
		else if (auto * ctrl = std::get_if<ControlEvent>(&evt))
		{
			if (std::holds_alternative<CloseEvent>(*ctrl))
			{
				got_close = true;
			}
		}
	}

	REQUIRE(audio_count == 2);
	REQUIRE(got_close);
}

}  // namespace

TEST_CASE("race() same-type cobalt channels", "[cobalt][race-workaround]")
{
	boost::asio::io_context ioc;
	boost::cobalt::spawn(ioc, test_race_same_type(ioc), boost::asio::detached);
	ioc.run();
}

TEST_CASE("race() audio wins over control", "[cobalt][race-workaround]")
{
	boost::asio::io_context ioc;
	boost::cobalt::spawn(ioc, test_race_audio_wins(ioc), boost::asio::detached);
	ioc.run();
}

TEST_CASE("race() control wins over audio", "[cobalt][race-workaround]")
{
	boost::asio::io_context ioc;
	boost::cobalt::spawn(ioc, test_race_control_wins(ioc), boost::asio::detached);
	ioc.run();
}

TEST_CASE("race() in a loop simulating duplex event loop", "[cobalt][race-workaround]")
{
	boost::asio::io_context ioc;
	boost::cobalt::spawn(ioc, test_race_loop(ioc), boost::asio::detached);
	ioc.run();
}
