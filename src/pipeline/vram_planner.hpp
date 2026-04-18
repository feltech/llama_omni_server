/**
 * @file   vram_planner.hpp
 * @brief  VRAM budgeting and model-placement decisions.
 *
 * `VramPlanner` computes a concrete `ModelPlacement` from the application
 * config and an observed amount of free VRAM.  The result is consumed by
 * `ModelManager` before loading the LLM, audio encoder, CLIP encoder, and TTS
 * pipeline so GPU placement is explicit and logged rather than left implicit
 * in the underlying libraries.
 */

#pragma once

#include <cstdint>
#include <optional>

#include "../config/config_loader.hpp"

namespace llama_omni_server
{

/**
 * @brief  Concrete GPU/CPU placement decisions for all major model components.
 */
struct ModelPlacement
{
	int n_llm_gpu_layers{0};
	bool audio_on_gpu{true};
	bool clip_on_gpu{true};
	bool tts_transformer_on_gpu{true};
	bool token2wav_on_gpu{true};
	bool kv_cache_on_gpu{true};
	std::uint64_t available_vram_bytes{0};
	std::uint64_t usable_vram_bytes{0};
	std::uint64_t reserved_vram_bytes{0};
};

/**
 * @brief  Computes placement decisions from a VRAM budget.
 *
 * The planner is intentionally approximate: it uses conservative fixed-size
 * estimates from `PLAN.md` to choose an auditable placement policy before any
 * model is loaded.
 */
class VramPlanner
{
public:
	/**
	 * @brief  Probe free VRAM from the CUDA runtime, if available.
	 *
	 * @return Free VRAM in bytes, or `std::nullopt` when CUDA runtime probing is
	 *         unavailable or fails.
	 */
	[[nodiscard]] static std::optional<std::uint64_t> probe_available_vram_bytes() noexcept;

	/**
	 * @brief  Compute placement decisions for the current config and VRAM budget.
	 *
	 * @param cfg                    Full application config.
	 * @param available_vram_bytes   Free VRAM in bytes.
	 * @return Placement decisions for all GPU-sensitive model components.
	 */
	[[nodiscard]] static ModelPlacement compute_placement(
		AppConfig const & cfg, std::uint64_t available_vram_bytes) noexcept;
};

}  // namespace llama_omni_server
