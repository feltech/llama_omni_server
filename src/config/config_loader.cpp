#include "config_loader.hpp"

#include <array>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace llama_omni_server
{

namespace
{

constexpr int min_valid_port = 1;
constexpr int max_valid_port = 65535;

template <typename T>
T yaml_get(YAML::Node const & node, char const * key, T default_value)
{
	if (auto node_val = node[key]; node_val.IsDefined() && !node_val.IsNull())
	{
		return node_val.as<T>();
	}
	return default_value;
}

bool validate_config(AppConfig const & config)
{
	if (config.server.port < min_valid_port || config.server.port > max_valid_port)
	{
		spdlog::error(
			"Invalid port: {} (must be {}–{})", config.server.port, min_valid_port, max_valid_port);
		return false;
	}
	// input_sample_rate is fixed at 16000 (static constexpr); no validation needed.
	return true;
}

std::string resolve_config_path(
	std::filesystem::path const & config_dir, std::string const & configured_path)
{
	if (configured_path.empty())
	{
		return configured_path;
	}

	std::filesystem::path path{configured_path};
	if (path.is_relative())
	{
		path = config_dir / path;
	}
	return path.lexically_normal().string();
}

void resolve_relative_paths(std::filesystem::path const & config_dir, AppConfig & config)
{
	for (std::string * path : std::array{
			 &config.model.llm_path,
			 &config.model.vision_path,
			 &config.model.audio_path,
			 &config.model.tts_transformer_path,
			 &config.model.tts_weights_path,
			 &config.model.projector_path,
			 &config.model.token2wav_dir,
			 &config.voice.reference_wav,
			 &config.voice.reference_prompt_cache,
			 &config.prompts.system_prompt_cache_path,
			 &config.debug.audio_output_wav})
	{
		*path = resolve_config_path(config_dir, *path);
	}

	// Default reference_prompt_cache to token2wav_dir/prompt_cache.gguf if not set.
	// This runs after token2wav_dir has been resolved.
	if (config.voice.reference_prompt_cache.empty() && !config.model.token2wav_dir.empty())
	{
		config.voice.reference_prompt_cache =
			(std::filesystem::path{config.model.token2wav_dir} / "prompt_cache.gguf").string();
	}
}

void load_server_config(YAML::Node const & root, AppConfig & config)
{
	if (auto srv_node = root["server"]; srv_node.IsDefined() && srv_node.IsMap())
	{
		config.server.host = yaml_get<std::string>(srv_node, "host", config.server.host);
		config.server.port = yaml_get<int>(srv_node, "port", config.server.port);
	}
}

void load_model_config(YAML::Node const & root, AppConfig & config)
{
	if (auto mdl_node = root["model"]; mdl_node.IsDefined() && mdl_node.IsMap())
	{
		config.model.llm_path = yaml_get<std::string>(mdl_node, "llm_path", config.model.llm_path);
		config.model.vision_path =
			yaml_get<std::string>(mdl_node, "vision_path", config.model.vision_path);
		config.model.audio_path =
			yaml_get<std::string>(mdl_node, "audio_path", config.model.audio_path);
		config.model.tts_transformer_path = yaml_get<std::string>(
			mdl_node, "tts_transformer_path", config.model.tts_transformer_path);
		config.model.tts_weights_path =
			yaml_get<std::string>(mdl_node, "tts_weights_path", config.model.tts_weights_path);
		config.model.projector_path =
			yaml_get<std::string>(mdl_node, "projector_path", config.model.projector_path);
		config.model.token2wav_dir =
			yaml_get<std::string>(mdl_node, "token2wav_dir", config.model.token2wav_dir);
	}
}

void load_inference_config(YAML::Node const & root, AppConfig & config)
{
	if (auto inf_node = root["inference"]; inf_node.IsDefined() && inf_node.IsMap())
	{
		config.inference.n_ctx = yaml_get<int>(inf_node, "n_ctx", config.inference.n_ctx);
		config.inference.overflow_reserve =
			yaml_get<int>(inf_node, "overflow_reserve", config.inference.overflow_reserve);
		config.inference.temperature =
			yaml_get<float>(inf_node, "temperature", config.inference.temperature);
		config.inference.repeat_penalty =
			yaml_get<float>(inf_node, "repeat_penalty", config.inference.repeat_penalty);
		config.inference.repeat_penalty_last_n = yaml_get<int>(
			inf_node, "repeat_penalty_last_n", config.inference.repeat_penalty_last_n);
		config.inference.top_k = yaml_get<int>(inf_node, "top_k", config.inference.top_k);
		config.inference.top_p = yaml_get<float>(inf_node, "top_p", config.inference.top_p);
		config.inference.min_p = yaml_get<float>(inf_node, "min_p", config.inference.min_p);
		config.inference.listen_prob_scale =
			yaml_get<float>(inf_node, "listen_prob_scale", config.inference.listen_prob_scale);
		config.inference.turn_eos_prob_scale =
			yaml_get<float>(inf_node, "turn_eos_prob_scale", config.inference.turn_eos_prob_scale);
		config.inference.chunk_eos_prob_scale = yaml_get<float>(
			inf_node, "chunk_eos_prob_scale", config.inference.chunk_eos_prob_scale);
		config.inference.duplex_force_listen_count = yaml_get<int>(
			inf_node, "duplex_force_listen_count", config.inference.duplex_force_listen_count);
		config.inference.speech_chunk_ms =
			yaml_get<int>(inf_node, "speech_chunk_ms", config.inference.speech_chunk_ms);
		config.inference.max_generation_tokens = yaml_get<int>(
			inf_node, "max_generation_tokens", config.inference.max_generation_tokens);
		config.inference.max_duplex_tokens =
			yaml_get<int>(inf_node, "max_duplex_tokens", config.inference.max_duplex_tokens);
		config.inference.clip_num_threads =
			yaml_get<int>(inf_node, "clip_num_threads", config.inference.clip_num_threads);
	}
}

void load_voice_config(YAML::Node const & root, AppConfig & config)
{
	if (auto voice_node = root["voice"]; voice_node.IsDefined() && voice_node.IsMap())
	{
		config.voice.reference_wav =
			yaml_get<std::string>(voice_node, "reference_wav", config.voice.reference_wav);
		config.voice.reference_prompt_cache = yaml_get<std::string>(
			voice_node, "reference_prompt_cache", config.voice.reference_prompt_cache);
	}
}

void load_prompts_config(YAML::Node const & root, AppConfig & config)
{
	if (auto pmt_node = root["prompts"]; pmt_node.IsDefined() && pmt_node.IsMap())
	{
		config.prompts.duplex_system =
			yaml_get<std::string>(pmt_node, "duplex_system", config.prompts.duplex_system);
		config.prompts.system_prompt_cache_path = yaml_get<std::string>(
			pmt_node, "system_prompt_cache_path", config.prompts.system_prompt_cache_path);
	}
}

void load_tts_config(YAML::Node const & root, AppConfig & config)
{
	if (auto tts_node = root["tts"]; tts_node.IsDefined() && tts_node.IsMap())
	{
		config.tts.max_audio_tokens =
			yaml_get<int>(tts_node, "max_audio_tokens", config.tts.max_audio_tokens);
		config.tts.temperature = yaml_get<float>(tts_node, "temperature", config.tts.temperature);
		config.tts.repeat_penalty =
			yaml_get<float>(tts_node, "repeat_penalty", config.tts.repeat_penalty);
		config.tts.top_k_tokens = yaml_get<int>(tts_node, "top_k_tokens", config.tts.top_k_tokens);
		config.tts.top_p_threshold =
			yaml_get<float>(tts_node, "top_p_threshold", config.tts.top_p_threshold);
		config.tts.min_p = yaml_get<float>(tts_node, "min_p", config.tts.min_p);
		config.tts.repeat_penalty_last_n =
			yaml_get<int>(tts_node, "repeat_penalty_last_n", config.tts.repeat_penalty_last_n);
	}
}

void load_debug_config(YAML::Node const & root, AppConfig & config)
{
	if (auto dbg_node = root["debug"]; dbg_node.IsDefined() && dbg_node.IsMap())
	{
		config.debug.audio_output_wav =
			yaml_get<std::string>(dbg_node, "audio_output_wav", config.debug.audio_output_wav);
	}
}

void load_logging_config(YAML::Node const & root, AppConfig & config)
{
	if (auto log_node = root["logging"]; log_node.IsDefined() && log_node.IsMap())
	{
		config.logging.level = yaml_get<std::string>(log_node, "level", config.logging.level);
		config.logging.status_period_ms =
			yaml_get<int>(log_node, "status_period_ms", config.logging.status_period_ms);
	}
}

void load_vram_config(YAML::Node const & root, AppConfig & config)
{
	if (auto vram_node = root["vram"]; vram_node.IsDefined() && vram_node.IsMap())
	{
		config.vram.force_llm_gpu_layers =
			yaml_get<int>(vram_node, "force_llm_gpu_layers", config.vram.force_llm_gpu_layers);
		config.vram.headroom_mib =
			yaml_get<int>(vram_node, "headroom_mib", config.vram.headroom_mib);
		config.vram.kv_cache_gpu =
			yaml_get<bool>(vram_node, "kv_cache_gpu", config.vram.kv_cache_gpu);
		config.vram.tts_transformer_gpu =
			yaml_get<bool>(vram_node, "tts_transformer_gpu", config.vram.tts_transformer_gpu);
		config.vram.token2wav_gpu =
			yaml_get<bool>(vram_node, "token2wav_gpu", config.vram.token2wav_gpu);
	}
}

}  // namespace

AppConfig load_config(std::filesystem::path const & config_path)
{
	if (!std::filesystem::exists(config_path))
	{
		spdlog::error("Config file does not exist: {}", config_path.string());
		throw std::runtime_error{"Config file not found"};
	}

	YAML::Node root;
	try
	{
		root = YAML::LoadFile(config_path.string());
	}
	catch (YAML::Exception const & exc)
	{
		spdlog::error("Failed to parse YAML '{}': {}", config_path.string(), exc.what());
		throw std::runtime_error{std::string("Invalid YAML: ") + exc.what()};
	}

	AppConfig config;
	load_server_config(root, config);
	load_model_config(root, config);
	load_inference_config(root, config);
	load_voice_config(root, config);
	load_prompts_config(root, config);
	load_tts_config(root, config);
	load_debug_config(root, config);
	load_logging_config(root, config);
	load_vram_config(root, config);
	resolve_relative_paths(config_path.parent_path(), config);

	if (!validate_config(config))
	{
		throw std::runtime_error{"Configuration validation failed"};
	}

	spdlog::info("Configuration loaded from: {}", config_path.string());
	return config;
}

}  // namespace llama_omni_server
