/**
 * @file   vram_planner.cpp
 * @brief  VRAM budgeting and model-placement decisions.
 */

#include "./vram_planner.hpp"

#include <algorithm>
#include <ggml-backend.h>
#include <limits>

namespace llama_omni_server
{

namespace
{

constexpr std::uint64_t kBytesPerMib = 1024ULL * 1024ULL;

constexpr std::uint64_t mib_to_bytes(std::uint64_t value) noexcept
{
	return value * kBytesPerMib;
}

constexpr std::uint64_t saturating_subtract(std::uint64_t lhs, std::uint64_t rhs) noexcept
{
	return (lhs > rhs) ? (lhs - rhs) : 0ULL;
}

constexpr std::uint64_t reserve_if_enabled(
	bool enabled, bool & on_gpu, std::uint64_t bytes, std::uint64_t & remaining) noexcept
{
	if (!enabled)
	{
		on_gpu = false;
		return 0ULL;
	}

	if (remaining < bytes)
	{
		on_gpu = false;
		return 0ULL;
	}

	on_gpu = true;
	remaining -= bytes;
	return bytes;
}

constexpr std::uint64_t kv_cache_bytes(int n_ctx) noexcept
{
	// From PLAN.md: n_ctx=2048 consumes ~288 MiB, so each token reserves
	// roughly 147 456 bytes of KV cache.
	static constexpr std::uint64_t kKvBytesPerToken = 147456ULL;
	return static_cast<std::uint64_t>(std::max(n_ctx, 0)) * kKvBytesPerToken;
}

[[nodiscard]] constexpr bool tts_transformer_enabled(AppConfig const & cfg) noexcept
{
	return !cfg.model.tts_transformer_path.empty();
}

[[nodiscard]] constexpr bool token2wav_enabled(AppConfig const & cfg) noexcept
{
	return !cfg.model.tts_transformer_path.empty() && !cfg.model.tts_weights_path.empty() &&
		!cfg.model.projector_path.empty() && !cfg.model.token2wav_dir.empty();
}

}  // namespace

std::optional<std::uint64_t> VramPlanner::probe_available_vram_bytes() noexcept
{
	ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
	if (gpu_device == nullptr)
	{
		return std::nullopt;
	}

	std::size_t free_bytes = 0;
	std::size_t total_bytes = 0;
	ggml_backend_dev_memory(gpu_device, &free_bytes, &total_bytes);
	if (total_bytes == 0U)
	{
		return std::nullopt;
	}

	return static_cast<std::uint64_t>(free_bytes);
}

ModelPlacement VramPlanner::compute_placement(
	AppConfig const & cfg, std::uint64_t available_vram_bytes) noexcept
{
	// Conservative fixed estimates from PLAN.md.
	static constexpr std::uint64_t kLlmLayerBytes = mib_to_bytes(156ULL);
	static constexpr std::uint64_t kAudioEncoderBytes = mib_to_bytes(500ULL);
	static constexpr std::uint64_t kClipEncoderBytes = mib_to_bytes(350ULL);
	static constexpr std::uint64_t kTtsTransformerBytes = mib_to_bytes(1000ULL);
	static constexpr std::uint64_t kToken2WavBytes = mib_to_bytes(300ULL);
	static constexpr int kMaxLlmLayers = 36;

	ModelPlacement placement{};
	placement.available_vram_bytes = available_vram_bytes;
	placement.kv_cache_on_gpu = cfg.vram.kv_cache_gpu;

	std::uint64_t const headroom_bytes =
		static_cast<std::uint64_t>(std::max(cfg.vram.headroom_mib, 0)) * kBytesPerMib;
	std::uint64_t remaining = saturating_subtract(available_vram_bytes, headroom_bytes);
	placement.usable_vram_bytes = remaining;

	placement.reserved_vram_bytes += reserve_if_enabled(
		token2wav_enabled(cfg) && cfg.vram.token2wav_gpu,
		placement.token2wav_on_gpu,
		kToken2WavBytes,
		remaining);
	placement.reserved_vram_bytes += reserve_if_enabled(
		tts_transformer_enabled(cfg) && cfg.vram.tts_transformer_gpu,
		placement.tts_transformer_on_gpu,
		kTtsTransformerBytes,
		remaining);
	placement.reserved_vram_bytes += reserve_if_enabled(
		!cfg.model.vision_path.empty(), placement.clip_on_gpu, kClipEncoderBytes, remaining);
	placement.reserved_vram_bytes += reserve_if_enabled(
		!cfg.model.audio_path.empty(), placement.audio_on_gpu, kAudioEncoderBytes, remaining);

	if (placement.kv_cache_on_gpu)
	{
		std::uint64_t const kv_bytes = kv_cache_bytes(cfg.inference.n_ctx);
		std::uint64_t const reserved_kv = std::min(remaining, kv_bytes);
		placement.reserved_vram_bytes += reserved_kv;
		remaining -= reserved_kv;
	}

	if (cfg.vram.force_llm_gpu_layers >= 0)
	{
		placement.n_llm_gpu_layers = cfg.vram.force_llm_gpu_layers;
		return placement;
	}

	placement.n_llm_gpu_layers = static_cast<int>(remaining / kLlmLayerBytes);
	placement.n_llm_gpu_layers = std::clamp(placement.n_llm_gpu_layers, 0, kMaxLlmLayers);
	return placement;
}

}  // namespace llama_omni_server
