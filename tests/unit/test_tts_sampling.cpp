/**
 * @file   test_tts_sampling.cpp
 * @brief  Unit tests for pure TTS sampling helpers.
 *
 * These tests cover the model-independent math used by the TTS sampling loop.
 * They do not load any GGUFs or touch the GPU.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <random>
#include <vector>

#include "pipeline/tts_sampling.hpp"

namespace llama_omni_server::tts_detail
{

namespace
{

constexpr float kEpsilon = 1.0e-5F;
constexpr float kHalf = 0.5F;
constexpr float kZero = 0.0F;
constexpr float kOne = 1.0F;
constexpr float kTwo = 2.0F;
constexpr float kThree = 3.0F;
constexpr float kFour = 4.0F;
constexpr float kFive = 5.0F;
constexpr float kSix = 6.0F;
constexpr float kExpectedLogit0 = 4.5F;
constexpr float kExpectedLogit1 = 9.0F;
constexpr float kTempInput0 = 2.0F;
constexpr float kTempInput1 = -4.0F;
constexpr float kTempInput2 = 6.0F;
constexpr float kTemperature = 2.0F;
constexpr float kPenaltyInput0 = 1.0F;
constexpr float kPenaltyInput1 = -2.0F;
constexpr float kPenaltyInput2 = 3.0F;
constexpr float kPenaltyInput3 = -4.0F;
constexpr float kRepPenalty = 1.5F;
constexpr float kExpectedPenalty1 = -3.0F;
constexpr float kExpectedPenalty3 = -6.0F;
constexpr float kTopKProb0 = 0.10F;
constexpr float kTopKProb1 = 0.40F;
constexpr float kTopKProb2 = 0.20F;
constexpr float kTopKProb3 = 0.30F;
constexpr float kTopPProb0 = 0.50F;
constexpr float kTopPProb1 = 0.30F;
constexpr float kTopPProb2 = 0.15F;
constexpr float kTopPProb3 = 0.05F;
constexpr float kTopPThreshold = 0.8F;
constexpr int kNumTopTokens = 2;
constexpr int kNumTokens = 4;
constexpr int kPenaltyLastN = 2;
constexpr int kSampleIters = 8;
constexpr int kNumLogitRows = 2;
constexpr int kHiddenSize = 3;
constexpr int kSampleTokenCount = 3;
constexpr int kExpectedSampleIdx = 1;
constexpr std::size_t kFirstIdx = 0U;
constexpr std::size_t kSecondIdx = 1U;
constexpr std::size_t kThirdIdx = 2U;
constexpr std::size_t kFourthIdx = 3U;

}  // namespace

SCENARIO("compute_audio_logits multiplies head_code weights by the hidden state")
{
	std::vector<float> const head_code_w = {kOne, kTwo, kThree, kFour, kFive, kSix};
	std::vector<float> const hidden = {kHalf, -kOne, kTwo};

	std::vector<float> const logits =
		compute_audio_logits(head_code_w.data(), hidden.data(), kNumLogitRows, kHiddenSize);

	REQUIRE(logits.size() == static_cast<std::size_t>(kNumLogitRows));
	CHECK(logits[kFirstIdx] == Catch::Approx(kExpectedLogit0).margin(kEpsilon));
	CHECK(logits[kSecondIdx] == Catch::Approx(kExpectedLogit1).margin(kEpsilon));
}

SCENARIO("apply_temperature scales logits in place")
{
	std::vector<float> logits = {kTempInput0, kTempInput1, kTempInput2};

	apply_temperature(logits, kTemperature);

	CHECK(logits[kFirstIdx] == Catch::Approx(kOne).margin(kEpsilon));
	CHECK(logits[kSecondIdx] == Catch::Approx(kPenaltyInput1).margin(kEpsilon));
	CHECK(logits[kThirdIdx] == Catch::Approx(kPenaltyInput2).margin(kEpsilon));
}

SCENARIO("apply_rep_penalty only affects recent tokens and preserves sign semantics")
{
	std::vector<float> logits = {kPenaltyInput0, kPenaltyInput1, kPenaltyInput2, kPenaltyInput3};
	std::vector<int> const recent_relative = {0, 3, 1};

	apply_rep_penalty(logits, recent_relative, kPenaltyLastN, kNumTokens, kRepPenalty);

	CHECK(logits[kFirstIdx] == Catch::Approx(kPenaltyInput0).margin(kEpsilon));
	CHECK(logits[kSecondIdx] == Catch::Approx(kExpectedPenalty1).margin(kEpsilon));
	CHECK(logits[kThirdIdx] == Catch::Approx(kPenaltyInput2).margin(kEpsilon));
	CHECK(logits[kFourthIdx] == Catch::Approx(kExpectedPenalty3).margin(kEpsilon));
}

SCENARIO("softmax and renormalize produce a valid probability distribution")
{
	std::vector<float> const logits = {kOne, kTwo, kThree};

	std::vector<float> probs = softmax_probs(logits);

	REQUIRE(probs.size() == logits.size());
	CHECK(probs[kThirdIdx] > probs[kSecondIdx]);
	CHECK(probs[kSecondIdx] > probs[kFirstIdx]);
	CHECK(
		std::accumulate(probs.cbegin(), probs.cend(), kZero) ==
		Catch::Approx(kOne).margin(kEpsilon));

	probs[kFirstIdx] = kZero;
	renormalize(probs);

	CHECK(
		std::accumulate(probs.cbegin(), probs.cend(), kZero) ==
		Catch::Approx(kOne).margin(kEpsilon));
	CHECK(probs[kFirstIdx] == Catch::Approx(kZero).margin(kEpsilon));
}

SCENARIO("top-k and top-p filtering zero out low-probability tokens")
{
	std::vector<float> top_k_probs = {kTopKProb0, kTopKProb1, kTopKProb2, kTopKProb3};

	apply_top_k(top_k_probs, kNumTopTokens, kNumTokens);

	CHECK(top_k_probs[kFirstIdx] == Catch::Approx(kZero).margin(kEpsilon));
	CHECK(top_k_probs[kSecondIdx] == Catch::Approx(kTopKProb1).margin(kEpsilon));
	CHECK(top_k_probs[kThirdIdx] == Catch::Approx(kZero).margin(kEpsilon));
	CHECK(top_k_probs[kFourthIdx] == Catch::Approx(kTopKProb3).margin(kEpsilon));

	std::vector<float> top_p_probs = {kTopPProb0, kTopPProb1, kTopPProb2, kTopPProb3};
	apply_top_p(top_p_probs, kTopPThreshold, kNumTokens);

	CHECK(top_p_probs[kFirstIdx] == Catch::Approx(kTopPProb0).margin(kEpsilon));
	CHECK(top_p_probs[kSecondIdx] == Catch::Approx(kTopPProb1).margin(kEpsilon));
	CHECK(top_p_probs[kThirdIdx] == Catch::Approx(kZero).margin(kEpsilon));
	CHECK(top_p_probs[kFourthIdx] == Catch::Approx(kZero).margin(kEpsilon));
}

SCENARIO("multinomial_sample returns the only token with non-zero probability")
{
	std::vector<float> const probs = {kZero, kOne, kZero};
	std::random_device seed_source;
	std::mt19937 rng{seed_source()};

	for (int idx = 0; idx < kSampleIters; ++idx)
	{
		CHECK(multinomial_sample(probs, rng, kSampleTokenCount) == kExpectedSampleIdx);
	}
}

}  // namespace llama_omni_server::tts_detail
