/**
 * @file   test_kv_eviction.cpp
 * @brief  Unit tests for KV cache eviction bookkeeping logic.
 *
 * These tests verify the position arithmetic of the sliding-window KV
 * eviction without needing a real llama context or GPU.  They simulate
 * the bookkeeping that `evict_oldest_units()` performs.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "session.hpp"

#include <catch2/catch_test_macros.hpp>

namespace llama_omni_server::test
{

namespace
{

// Named constants to avoid magic-number warnings.
constexpr int kTotalCtxLarge = 10240;
constexpr int kTotalCtxSmall = 2048;
constexpr int kTotalCtxMedium = 4096;
constexpr int kSysPromptEnd = 100;
constexpr int kBlockSizeA = 400;
constexpr int kBlockSizeB = 300;
constexpr int kBlockSizeC = 500;
constexpr int kBlockSizeD = 600;
constexpr int kBlockSizeE = 200;
constexpr int kNumBlocksLarge = 10;

// Named n_past values for test scenarios.
constexpr int kNPastNoEviction = 5000;
constexpr int kNPastTenBlocks = 4100;
constexpr int kNPastThreeBlocks = 1500;
constexpr int kNPastNoBlocks = 2000;
constexpr int kNPastSixBlocks = 1900;
constexpr int kNPastThreeBlocksA = 1300;

/// Parameters for `make_conv_with_blocks` — grouped to avoid swappable-param warning.
struct ConvBuilder
{
	int n_past;
	int sys_prompt_end;
	std::vector<int> block_sizes;
};

/**
 * @brief Simulate the eviction bookkeeping that evict_oldest_units() does.
 *
 * Pure-C++ simulation of the position arithmetic — no llama calls.
 */
bool simulate_eviction(
	ConversationState & conv, int & current_start, int total_ctx, int target_free)
{
	int const used = conv.n_past;
	int const free_now = total_ctx - used;

	if (free_now >= target_free)
	{
		return true;
	}

	int need_to_free = target_free - free_now;
	int evict_end = conv.sys_prompt_end;
	int blocks_evicted = 0;

	for (auto idx = std::size_t{0}; idx < conv.unit_history.size() && need_to_free > 0; ++idx)
	{
		auto const & block = conv.unit_history[idx];
		int const block_size = block.pos_end - block.pos_start;
		evict_end = block.pos_end;
		need_to_free -= block_size;
		++blocks_evicted;
	}

	if (blocks_evicted == 0)
	{
		return false;
	}

	int const evict_start = conv.sys_prompt_end;
	int const delta = evict_end - evict_start;

	conv.n_past -= delta;
	current_start -= delta;

	conv.unit_history.erase(conv.unit_history.begin(), conv.unit_history.begin() + blocks_evicted);
	for (auto & block : conv.unit_history)
	{
		block.pos_start -= delta;
		block.pos_end -= delta;
	}

	return (total_ctx - conv.n_past) >= target_free;
}

ConversationState make_conv_with_blocks(ConvBuilder const & builder)
{
	ConversationState conv;
	conv.sys_prompt_end = builder.sys_prompt_end;
	conv.n_past = builder.n_past;

	int pos = builder.sys_prompt_end;
	for (int const size : builder.block_sizes)
	{
		conv.unit_history.push_back(UnitBlock{.pos_start = pos, .pos_end = pos + size});
		pos += size;
	}

	return conv;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

SCENARIO("Eviction bookkeeping: no eviction needed when space is sufficient", "[kv][unit]")
{
	GIVEN("a conversation state with plenty of free space")
	{
		auto conv = make_conv_with_blocks(
			{.n_past = kNPastNoEviction,
			 .sys_prompt_end = kSysPromptEnd,
			 .block_sizes = {kBlockSizeA, kBlockSizeA, kBlockSizeA}});
		int current_start = kNPastNoEviction;

		WHEN("we request eviction with a small target_free")
		{
			bool const eviction_succeeded =
				simulate_eviction(conv, current_start, kTotalCtxLarge, /*target_free=*/5240);

			THEN("no eviction occurs and the function returns true")
			{
				CHECK(eviction_succeeded);
				CHECK(conv.n_past == kNPastNoEviction);
				CHECK(conv.sys_prompt_end == kSysPromptEnd);
				CHECK(conv.unit_history.size() == 3U);
				CHECK(current_start == kNPastNoEviction);
			}
		}
	}
}

SCENARIO("Eviction bookkeeping: evicts oldest blocks to free space", "[kv][unit]")
{
	GIVEN("a conversation state with 10 blocks of 400 tokens each")
	{
		std::vector<int> const block_sizes(kNumBlocksLarge, kBlockSizeA);
		auto conv = make_conv_with_blocks(
			{.n_past = kNPastTenBlocks,
			 .sys_prompt_end = kSysPromptEnd,
			 .block_sizes = block_sizes});
		int const initial_start = kNPastTenBlocks;
		int current_start = initial_start;

		WHEN("we need 2000 free tokens but only have 6140")
		{
			bool const eviction_succeeded =
				simulate_eviction(conv, current_start, kTotalCtxLarge, /*target_free=*/8140);

			THEN("5 oldest blocks are evicted and positions are shifted")
			{
				REQUIRE(eviction_succeeded);
				CHECK(conv.unit_history.size() == 5U);
				CHECK(conv.n_past == kNPastTenBlocks - 2000);
				CHECK(current_start == initial_start - 2000);
				CHECK(conv.sys_prompt_end == kSysPromptEnd);

				CHECK(conv.unit_history[0].pos_start == 100);
				CHECK(conv.unit_history[0].pos_end == 500);
				CHECK(conv.unit_history[4].pos_start == 1700);
				CHECK(conv.unit_history[4].pos_end == 2100);
			}
		}
	}
}

SCENARIO("Eviction bookkeeping: evicts more blocks than needed", "[kv][unit]")
{
	GIVEN("a conversation state with 3 blocks of varying sizes")
	{
		auto conv = make_conv_with_blocks(
			{.n_past = kNPastThreeBlocks,
			 .sys_prompt_end = kSysPromptEnd,
			 .block_sizes = {kBlockSizeB, kBlockSizeC, kBlockSizeD}});
		int const initial_start = kNPastThreeBlocks;
		int current_start = initial_start;

		WHEN("we need 400 free tokens but the first block only frees 300")
		{
			bool const eviction_succeeded =
				simulate_eviction(conv, current_start, kTotalCtxSmall, /*target_free=*/948);

			THEN("2 blocks are evicted (overshooting the exact need)")
			{
				REQUIRE(eviction_succeeded);
				CHECK(conv.unit_history.size() == 1U);
				CHECK(conv.n_past == kNPastThreeBlocks - 800);
				CHECK(current_start == initial_start - 800);
				CHECK(conv.sys_prompt_end == kSysPromptEnd);

				CHECK(conv.unit_history[0].pos_start == 100);
				CHECK(conv.unit_history[0].pos_end == 700);
			}
		}
	}
}

SCENARIO("Eviction bookkeeping: returns false when no blocks to evict", "[kv][unit]")
{
	GIVEN("a conversation state with no completed unit blocks")
	{
		ConversationState conv;
		conv.sys_prompt_end = kSysPromptEnd;
		conv.n_past = kNPastNoBlocks;
		int current_start = kSysPromptEnd;

		WHEN("we request eviction but unit_history is empty")
		{
			bool const eviction_succeeded =
				simulate_eviction(conv, current_start, kTotalCtxSmall, /*target_free=*/500);

			THEN("eviction fails gracefully and state is unchanged")
			{
				CHECK_FALSE(eviction_succeeded);
				CHECK(conv.n_past == kNPastNoBlocks);
				CHECK(conv.unit_history.empty());
				CHECK(conv.sys_prompt_end == kSysPromptEnd);
				CHECK(current_start == kSysPromptEnd);
			}
		}
	}
}

SCENARIO("Eviction bookkeeping: multiple sequential evictions", "[kv][unit]")
{
	GIVEN("a conversation state with 6 blocks of 300 tokens each")
	{
		auto conv = make_conv_with_blocks(
			{.n_past = kNPastSixBlocks,
			 .sys_prompt_end = kSysPromptEnd,
			 .block_sizes = {
				 kBlockSizeB, kBlockSizeB, kBlockSizeB, kBlockSizeB, kBlockSizeB, kBlockSizeB}});
		int current_start = kNPastSixBlocks;

		WHEN("we trigger two sequential evictions")
		{
			bool const eviction_succeeded_1 =
				simulate_eviction(conv, current_start, kTotalCtxMedium, /*target_free=*/2200);
			REQUIRE(eviction_succeeded_1);

			int const old_n_past = conv.n_past;
			conv.unit_history.push_back(
				UnitBlock{.pos_start = old_n_past, .pos_end = old_n_past + kBlockSizeB});
			conv.n_past = old_n_past + kBlockSizeB;
			current_start = old_n_past;

			bool const eviction_succeeded_2 =
				simulate_eviction(conv, current_start, kTotalCtxMedium, /*target_free=*/2200);

			THEN("both evictions succeed and state remains consistent")
			{
				REQUIRE(eviction_succeeded_2);
				CHECK(conv.unit_history.size() == 5U);
				CHECK(conv.sys_prompt_end == kSysPromptEnd);
				for (auto i = std::size_t{0}; i < conv.unit_history.size(); ++i)
				{
					CHECK(conv.unit_history[i].pos_start >= conv.sys_prompt_end);
					CHECK(conv.unit_history[i].pos_end > conv.unit_history[i].pos_start);
				}
			}
		}
	}
}

SCENARIO("Eviction bookkeeping: current_unit_start tracks correctly across evictions", "[kv][unit]")
{
	GIVEN("a conversation state with blocks and an open unit")
	{
		auto conv = make_conv_with_blocks(
			{.n_past = kNPastThreeBlocksA,
			 .sys_prompt_end = kSysPromptEnd,
			 .block_sizes = {kBlockSizeA, kBlockSizeA, kBlockSizeA}});
		int current_start = kNPastThreeBlocksA;

		WHEN("eviction occurs and then a new block is completed")
		{
			bool const eviction_succeeded =
				simulate_eviction(conv, current_start, kTotalCtxSmall, /*target_free=*/1148);
			REQUIRE(eviction_succeeded);

			CHECK(current_start == 900);

			conv.unit_history.push_back(
				UnitBlock{.pos_start = current_start, .pos_end = current_start + kBlockSizeE});
			conv.n_past = current_start + kBlockSizeE;
			current_start = conv.n_past;

			THEN("the newly completed block has correct positions")
			{
				CHECK(conv.unit_history.back().pos_start == 900);
				CHECK(conv.unit_history.back().pos_end == 1100);
				CHECK(current_start == 1100);
			}
		}
	}
}

}  // namespace llama_omni_server::test
