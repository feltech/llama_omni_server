#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>

#include <config/config_loader.hpp>

namespace
{

constexpr float kConfiguredInferenceTemperature = 0.2F;
constexpr float kConfiguredInferenceRepeatPenalty = 1.2F;
constexpr float kConfiguredInferenceTopP = 0.5F;
constexpr float kConfiguredTtsTemperature = 0.6F;
constexpr float kConfiguredTtsRepeatPenalty = 1.1F;
constexpr float kConfiguredTtsTopP = 0.7F;

std::filesystem::path repo_root()
{
	// NOLINTNEXTLINE(concurrency-mt-unsafe) — single-threaded test process
	char const * dir = std::getenv("LLAMAOMNISERVER_TEST_REPO_ROOT");
	REQUIRE(dir != nullptr);
	return {dir};
}

std::filesystem::path test_data_dir()
{
	return repo_root() / "test_data";
}

std::filesystem::path unit_data(char const * filename)
{
	return test_data_dir() / "unit" / filename;
}

std::filesystem::path example_config_path()
{
	return repo_root() / "config.example.yaml";
}

}  // namespace

SCENARIO("Config Loader loads the repository example config", "[config][bdd]")
{
	GIVEN("The checked-in config.example.yaml")
	{
		WHEN("Loading configuration")
		{
			auto const config = llama_omni_server::load_config(example_config_path());

			THEN("Server settings should match defaults")
			{
				REQUIRE(config.server.host == llama_omni_server::ServerConfig::default_host);
				REQUIRE(config.server.port == llama_omni_server::ServerConfig::default_port);
			}

			THEN("Relative model paths should be resolved against the config directory")
			{
				REQUIRE(
					config.model.llm_path ==
					(repo_root() / "../../models/gguf/MiniCPM-o-4_5-Q4_K_M.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.model.vision_path ==
					(repo_root() / "../../models/gguf/vision/MiniCPM-o-4_5-vision-F16.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.model.audio_path ==
					(repo_root() / "../../models/gguf/audio/MiniCPM-o-4_5-audio-F16.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.model.tts_transformer_path ==
					(repo_root() / "../../models/gguf/tts/MiniCPM-o-4_5-tts-transformer-F16.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.model.tts_weights_path ==
					(repo_root() / "../../models/gguf/tts/MiniCPM-o-4_5-tts-weights-F16.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.model.projector_path ==
					(repo_root() / "../../models/gguf/tts/MiniCPM-o-4_5-projector-F16.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.model.token2wav_dir ==
					(repo_root() / "../../models/gguf/token2wav-gguf").lexically_normal().string());
			}

			THEN("Voice, prompts and debug paths should also resolve relative to the config")
			{
				REQUIRE(
					config.voice.reference_wav ==
					(repo_root() / "test_data/default_ref_audio.wav").lexically_normal().string());
				REQUIRE(
					config.voice.reference_prompt_cache ==
					(repo_root() / "../../models/gguf/token2wav-gguf/prompt_cache.gguf")
						.lexically_normal()
						.string());
				REQUIRE(config.prompts.system_prompt_cache_path.empty());
				REQUIRE(config.debug.audio_output_wav.empty());
			}

			THEN("Runtime inference settings should match the checked-in example config")
			{
				constexpr auto kExpectedTemp = 0.7F;
				REQUIRE(
					config.inference.n_ctx == llama_omni_server::InferenceConfig::default_n_ctx);
				REQUIRE(
					config.inference.overflow_reserve ==
					llama_omni_server::InferenceConfig::default_overflow_reserve);
				REQUIRE(config.inference.temperature == Catch::Approx(kExpectedTemp));
				REQUIRE(
					config.inference.repeat_penalty ==
					Catch::Approx(llama_omni_server::InferenceConfig::default_repeat_penalty));
				REQUIRE(
					config.inference.repeat_penalty_last_n ==
					llama_omni_server::InferenceConfig::default_repeat_penalty_last_n);
				REQUIRE(
					config.inference.top_k == llama_omni_server::InferenceConfig::default_top_k);
				REQUIRE(
					config.inference.top_p ==
					Catch::Approx(llama_omni_server::InferenceConfig::default_top_p));
				REQUIRE(
					config.inference.min_p ==
					Catch::Approx(llama_omni_server::InferenceConfig::default_min_p));
				REQUIRE(
					config.inference.listen_prob_scale ==
					Catch::Approx(llama_omni_server::InferenceConfig::default_listen_prob_scale));
				REQUIRE(
					config.inference.turn_eos_prob_scale ==
					Catch::Approx(llama_omni_server::InferenceConfig::default_turn_eos_prob_scale));
				REQUIRE(
					config.inference.duplex_force_listen_count ==
					llama_omni_server::InferenceConfig::default_duplex_force_listen_count);
				REQUIRE(
					config.inference.speech_chunk_ms ==
					llama_omni_server::InferenceConfig::default_speech_chunk_ms);
				REQUIRE(
					config.inference.max_generation_tokens ==
					llama_omni_server::InferenceConfig::default_max_generation_tokens);
				REQUIRE(
					config.inference.max_duplex_tokens ==
					llama_omni_server::InferenceConfig::default_max_duplex_tokens);
				REQUIRE(
					config.inference.clip_num_threads ==
					llama_omni_server::InferenceConfig::default_clip_num_threads);
			}

			THEN("TTS sampling defaults should be loaded")
			{
				constexpr int kExpectedMaxAudioTokens = 700;
				REQUIRE(config.tts.max_audio_tokens == kExpectedMaxAudioTokens);
				REQUIRE(
					config.tts.temperature ==
					Catch::Approx(llama_omni_server::TtsConfig::default_temperature));
				REQUIRE(
					config.tts.repeat_penalty ==
					Catch::Approx(llama_omni_server::TtsConfig::default_repeat_penalty));
				REQUIRE(
					config.tts.top_k_tokens == llama_omni_server::TtsConfig::default_top_k_tokens);
				REQUIRE(
					config.tts.top_p_threshold ==
					Catch::Approx(llama_omni_server::TtsConfig::default_top_p_threshold));
				REQUIRE(
					config.tts.min_p == Catch::Approx(llama_omni_server::TtsConfig::default_min_p));
				REQUIRE(
					config.tts.repeat_penalty_last_n ==
					llama_omni_server::TtsConfig::default_repeat_penalty_last_n);
			}

			THEN("Other runtime-backed sections should load successfully")
			{
				REQUIRE(config.logging.level == "debug");
				REQUIRE(
					config.logging.status_period_ms ==
					llama_omni_server::LoggingConfig::default_status_period_ms);
				REQUIRE(
					config.vram.force_llm_gpu_layers ==
					llama_omni_server::VramConfig::default_force_llm_gpu_layers);
				REQUIRE(
					config.vram.headroom_mib ==
					llama_omni_server::VramConfig::default_headroom_mib);
				REQUIRE(config.vram.kv_cache_gpu);
				REQUIRE(config.vram.tts_transformer_gpu);
				REQUIRE(config.vram.token2wav_gpu);
			}
		}
	}
}

SCENARIO("Config Loader applies defaults for omitted sections", "[config][bdd]")
{
	GIVEN("A minimal config YAML file")
	{
		WHEN("Loading configuration")
		{
			auto const config = llama_omni_server::load_config(unit_data("config_minimal.yaml"));

			THEN("Default values should be applied")
			{
				REQUIRE(config.server.host == llama_omni_server::ServerConfig::default_host);
				REQUIRE(config.server.port == llama_omni_server::ServerConfig::default_port);
				REQUIRE(
					config.inference.n_ctx == llama_omni_server::InferenceConfig::default_n_ctx);
				REQUIRE(
					config.inference.max_generation_tokens ==
					llama_omni_server::InferenceConfig::default_max_generation_tokens);
				REQUIRE(
					config.tts.max_audio_tokens ==
					llama_omni_server::TtsConfig::default_max_audio_tokens);
				REQUIRE(
					config.inference.speech_chunk_ms ==
					llama_omni_server::InferenceConfig::default_speech_chunk_ms);
				REQUIRE(
					config.logging.status_period_ms ==
					llama_omni_server::LoggingConfig::default_status_period_ms);
			}
		}
	}
}

SCENARIO("Config Loader resolves relative path options against the YAML directory", "[config][bdd]")
{
	GIVEN("A config file with relative path options")
	{
		WHEN("Loading configuration")
		{
			auto const config = llama_omni_server::load_config(unit_data("config_valid.yaml"));

			THEN("All configured paths should be rewritten relative to that file")
			{
				auto const unit_dir = unit_data(".").lexically_normal();

				REQUIRE(
					config.model.llm_path ==
					(unit_dir / "../../../models/gguf/MiniCPM-o-4_5-Q4_K_M.gguf")
						.lexically_normal()
						.string());
				REQUIRE(
					config.voice.reference_wav ==
					(unit_dir / "../../default_ref_audio.wav").lexically_normal().string());
				REQUIRE(
					config.voice.reference_prompt_cache ==
					(unit_dir / "../../tmp/reference_prompt_cache.bin")
						.lexically_normal()
						.string());
				REQUIRE(
					config.prompts.system_prompt_cache_path ==
					(unit_dir / "../../tmp/system_prompt_cache.bin").lexically_normal().string());
				REQUIRE(
					config.debug.audio_output_wav ==
					(unit_dir / "../../tmp/output.wav").lexically_normal().string());
			}
		}
	}
}

SCENARIO("Config Loader rejects invalid files", "[config][bdd]")
{
	WHEN("The config file does not exist")
	{
		THEN("Loading should fail")
		{
			REQUIRE_THROWS(llama_omni_server::load_config(unit_data("config_missing.yaml")));
		}
	}

	WHEN("The config file contains invalid YAML")
	{
		THEN("Loading should fail")
		{
			REQUIRE_THROWS(llama_omni_server::load_config(unit_data("config_invalid.yaml")));
		}
	}

	WHEN("The config file contains an invalid port")
	{
		THEN("Validation should fail")
		{
			REQUIRE_THROWS(llama_omni_server::load_config(unit_data("config_invalid_port.yaml")));
		}
	}
}

SCENARIO("Config Loader parses configured scalar values", "[config][bdd]")
{
	GIVEN("A config file overriding several scalar fields")
	{
		WHEN("Loading configuration")
		{
			auto const config = llama_omni_server::load_config(unit_data("config_types.yaml"));

			THEN("The overridden values should be preserved")
			{
				REQUIRE(config.server.port == 9001);
				REQUIRE(config.inference.n_ctx == 8192);
				REQUIRE(config.inference.repeat_penalty_last_n == 32);
				REQUIRE(config.inference.top_k == 10);
				REQUIRE(config.inference.top_p == Catch::Approx(kConfiguredInferenceTopP));
				REQUIRE(config.inference.max_generation_tokens == 256);
				REQUIRE(config.inference.max_duplex_tokens == 128);
				REQUIRE(config.inference.clip_num_threads == 2);
				REQUIRE(config.tts.max_audio_tokens == 128);
				REQUIRE(config.tts.top_k_tokens == 12);
				REQUIRE(config.tts.repeat_penalty_last_n == 8);
				REQUIRE(config.logging.level == "debug");
				REQUIRE(config.logging.status_period_ms == 50);
				REQUIRE(config.vram.kv_cache_gpu == false);
				REQUIRE(config.vram.tts_transformer_gpu == false);
				REQUIRE(config.vram.token2wav_gpu == false);
				REQUIRE(
					config.inference.temperature == Catch::Approx(kConfiguredInferenceTemperature));
				REQUIRE(
					config.inference.repeat_penalty ==
					Catch::Approx(kConfiguredInferenceRepeatPenalty));
				REQUIRE(config.tts.temperature == Catch::Approx(kConfiguredTtsTemperature));
				REQUIRE(config.tts.repeat_penalty == Catch::Approx(kConfiguredTtsRepeatPenalty));
				REQUIRE(config.tts.top_p_threshold == Catch::Approx(kConfiguredTtsTopP));
			}
		}
	}
}
