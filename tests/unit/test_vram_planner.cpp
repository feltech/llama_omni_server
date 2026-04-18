#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <config/config_loader.hpp>
#include <pipeline/vram_planner.hpp>

namespace llama_omni_server::test
{

namespace
{

constexpr std::uint64_t kBytesPerMib = 1024ULL * 1024ULL;
constexpr std::uint64_t kBytesPerGib = 1024ULL * kBytesPerMib;
constexpr int kTestNCtx = 2048;
constexpr int kForcedGpuLayers = 7;
constexpr std::uint64_t kTtsOnlyBudgetBytes = 2000ULL * kBytesPerMib;
constexpr std::uint64_t kMediumBudgetBytes = 8ULL * kBytesPerGib;
constexpr std::uint64_t kKvComparisonBudgetBytes = 7800ULL * kBytesPerMib;

AppConfig make_config()
{
	AppConfig cfg;
	cfg.model.llm_path = "llm.gguf";
	cfg.model.audio_path = "audio.gguf";
	cfg.model.vision_path = "vision.gguf";
	cfg.model.tts_transformer_path = "tts-transformer.gguf";
	cfg.model.tts_weights_path = "tts-weights.gguf";
	cfg.model.projector_path = "projector.gguf";
	cfg.model.token2wav_dir = "token2wav";
	cfg.inference.n_ctx = kTestNCtx;
	return cfg;
}

}  // namespace

SCENARIO("VramPlanner prioritises latency-critical components before LLM layers", "[vram][unit]")
{
	GIVEN("a full multimodal config and a limited VRAM budget")
	{
		AppConfig const cfg = make_config();

		WHEN("the budget only fits TTS and Token2Wav after headroom")
		{
			std::uint64_t const available_vram = kTtsOnlyBudgetBytes;

			THEN("the latency-critical TTS stages stay on GPU while audio, CLIP, and LLM spill")
			{
				ModelPlacement const placement =
					VramPlanner::compute_placement(cfg, available_vram);
				CHECK(placement.token2wav_on_gpu);
				CHECK(placement.tts_transformer_on_gpu);
				CHECK_FALSE(placement.audio_on_gpu);
				CHECK_FALSE(placement.clip_on_gpu);
				CHECK(placement.n_llm_gpu_layers == 0);
			}
		}
	}
}

SCENARIO("VramPlanner can split TTS transformer and Token2Wav placement", "[vram][unit]")
{
	GIVEN("a config that disables only Token2Wav GPU placement")
	{
		AppConfig cfg = make_config();
		cfg.vram.token2wav_gpu = false;

		WHEN("planning placement")
		{
			ModelPlacement const placement =
				VramPlanner::compute_placement(cfg, kMediumBudgetBytes);

			THEN("the transformer can remain on GPU while Token2Wav is forced to CPU")
			{
				CHECK(placement.tts_transformer_on_gpu);
				CHECK_FALSE(placement.token2wav_on_gpu);
			}
		}
	}
}

SCENARIO("VramPlanner honours the explicit VRAM override config", "[vram][unit]")
{
	GIVEN("a config with force_llm_gpu_layers set")
	{
		AppConfig cfg = make_config();
		cfg.vram.force_llm_gpu_layers = kForcedGpuLayers;

		WHEN("planning placement")
		{
			ModelPlacement const placement =
				VramPlanner::compute_placement(cfg, kMediumBudgetBytes);

			THEN("the forced layer count is applied")
			{
				CHECK(placement.n_llm_gpu_layers == kForcedGpuLayers);
			}
		}
	}
}

SCENARIO("VramPlanner reserves less VRAM when KV cache is forced off GPU", "[vram][unit]")
{
	GIVEN("two otherwise identical configs")
	{
		AppConfig const cfg_gpu_kv = make_config();
		AppConfig cfg_cpu_kv = cfg_gpu_kv;
		cfg_cpu_kv.vram.kv_cache_gpu = false;
		std::uint64_t const available_vram = kKvComparisonBudgetBytes;

		WHEN("planning with GPU KV and CPU KV")
		{
			ModelPlacement const gpu_kv =
				VramPlanner::compute_placement(cfg_gpu_kv, available_vram);
			ModelPlacement const cpu_kv =
				VramPlanner::compute_placement(cfg_cpu_kv, available_vram);

			THEN("CPU KV reserves less VRAM and allows at least as many LLM layers")
			{
				CHECK_FALSE(cpu_kv.kv_cache_on_gpu);
				CHECK(cpu_kv.reserved_vram_bytes < gpu_kv.reserved_vram_bytes);
				CHECK(cpu_kv.n_llm_gpu_layers >= gpu_kv.n_llm_gpu_layers);
			}
		}
	}
}

}  // namespace llama_omni_server::test
