#pragma once

#include <filesystem>
#include <string>

namespace llama_omni_server
{

struct ServerConfig
{
	static constexpr std::string_view default_host = "0.0.0.0";
	static constexpr int default_port = 8080;

	std::string host{default_host};
	int port = default_port;
};

struct ModelConfig
{
	std::string llm_path;
	std::string vision_path;
	std::string audio_path;
	std::string tts_transformer_path;
	std::string tts_weights_path;
	std::string projector_path;
	std::string token2wav_dir;
};

struct InferenceConfig
{
	/// Practical default for this app on the target 16 GiB GPU. The GGUF/model
	/// metadata advertises a theoretical maximum context of 40960 tokens.
	static constexpr int default_n_ctx = 10240;
	static constexpr int default_overflow_reserve = 512;
	static constexpr float default_temperature = 0.8F;
	static constexpr float default_repeat_penalty = 1.05F;
	static constexpr int default_repeat_penalty_last_n = 64;
	static constexpr int default_top_k = 40;
	static constexpr float default_top_p = 0.95F;
	static constexpr float default_min_p = 0.05F;
	static constexpr float default_listen_prob_scale = 1.0F;
	static constexpr float default_turn_eos_prob_scale = 1.0F;
	static constexpr float default_chunk_eos_prob_scale = 1.0F;
	static constexpr int default_duplex_force_listen_count = 5;
	static constexpr int default_speech_chunk_ms = 1000;
	static constexpr int default_max_generation_tokens = 1024;
	static constexpr int default_max_duplex_tokens = 512;
	static constexpr int default_clip_num_threads = 4;

	int n_ctx = default_n_ctx;
	int overflow_reserve = default_overflow_reserve;
	float temperature = default_temperature;
	float repeat_penalty = default_repeat_penalty;
	int repeat_penalty_last_n = default_repeat_penalty_last_n;
	int top_k = default_top_k;
	float top_p = default_top_p;
	float min_p = default_min_p;
	float listen_prob_scale = default_listen_prob_scale;
	float turn_eos_prob_scale = default_turn_eos_prob_scale;
	float chunk_eos_prob_scale = default_chunk_eos_prob_scale;
	int duplex_force_listen_count = default_duplex_force_listen_count;
	int speech_chunk_ms = default_speech_chunk_ms;
	int max_generation_tokens = default_max_generation_tokens;
	int max_duplex_tokens = default_max_duplex_tokens;
	int clip_num_threads = default_clip_num_threads;
};

struct VoiceConfig
{
	std::string reference_wav;
	std::string reference_prompt_cache;
};

struct PromptsConfig
{
	std::string duplex_system = "Streaming Duplex Conversation! You are a helpful assistant.";
	std::string system_prompt_cache_path;
};

struct AudioConfig
{
	/// Input sample rate is fixed at 16 kHz — the only rate accepted by the
	/// model's Whisper audio encoder.  It is not configurable.
	static constexpr int input_sample_rate = 16000;
};

struct TtsConfig
{
	static constexpr int default_max_audio_tokens = 500;
	static constexpr float default_temperature = 0.8F;
	static constexpr float default_repeat_penalty = 1.05F;
	static constexpr int default_top_k_tokens = 25;
	static constexpr float default_top_p_threshold = 0.85F;
	static constexpr float default_min_p = 0.01F;
	static constexpr int default_repeat_penalty_last_n = 16;

	int max_audio_tokens = default_max_audio_tokens;
	float temperature = default_temperature;
	float repeat_penalty = default_repeat_penalty;
	int top_k_tokens = default_top_k_tokens;
	float top_p_threshold = default_top_p_threshold;
	float min_p = default_min_p;
	int repeat_penalty_last_n = default_repeat_penalty_last_n;
};

struct DebugConfig
{
	std::string audio_output_wav;
};

struct LoggingConfig
{
	static constexpr int default_status_period_ms = 200;

	std::string level = "debug";
	/// Interval at which "status" frames are sent to the client
	int status_period_ms = default_status_period_ms;
};

struct VramConfig
{
	static constexpr int default_force_llm_gpu_layers = -1;
	static constexpr int default_headroom_mib = 512;
	static constexpr bool default_tts_transformer_gpu = true;
	static constexpr bool default_token2wav_gpu = true;

	int force_llm_gpu_layers = default_force_llm_gpu_layers;
	int headroom_mib = default_headroom_mib;
	bool kv_cache_gpu = true;
	bool tts_transformer_gpu = default_tts_transformer_gpu;
	bool token2wav_gpu = default_token2wav_gpu;
};

struct AppConfig
{
	ServerConfig server;
	ModelConfig model;
	InferenceConfig inference;
	VoiceConfig voice;
	PromptsConfig prompts;
	AudioConfig audio;
	TtsConfig tts;
	DebugConfig debug;
	LoggingConfig logging;
	VramConfig vram;
};

/**
 * @brief Load and validate the application config from a YAML file.
 *
 * Parses the file at `config_path`, applies defaults for omitted keys, and
 * validates required fields and value ranges. The returned `AppConfig` is
 * ready to pass directly into the server, model manager, or test harness.
 *
 * @param config_path  Path to the YAML config file on disk.
 * @return Loaded config.
 *
 * @throws std::runtime_error if the file is missing, cannot be parsed as YAML,
 *         or fails validation.
 */
AppConfig load_config(std::filesystem::path const & config_path);

}  // namespace llama_omni_server
