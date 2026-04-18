/**
 * @file   model_manager.hpp
 * @brief  Loads and owns the LLM weights; provides context construction for sessions.
 *
 * `ModelManager` is a process-scoped singleton: construct it once at startup
 * on the GPU executor, then pass a reference to every `Session`. Sessions
 * each create their own `llama_context` (KV cache) via `create_context()`.
 *
 * ### Opaque llama pointers
 *
 * `llama_model`, `llama_context`, and `llama_vocab` are opaque C structs.  We
 * forward-declare them here and include `llama.h` only in the implementation
 * so downstream TUs are not forced to pull in the full llama.h header tree.
 */

#pragma once

#include "../config/config_loader.hpp"
#include "./tts_pipeline.hpp"
#include "./vram_planner.hpp"

#include <memory>

// Forward-declare opaque C structs — do NOT include llama.h, audition.h, or clip.h here.
// NOLINTBEGIN(modernize-use-using,bugprone-forward-declaration-namespace)
// modernize-use-using: these are C structs; `struct Foo;` style is intentional.
// bugprone-forward-declaration-namespace: C structs have global linkage and are correctly
// declared at global scope, not inside any namespace.
struct llama_model;
struct llama_context;
struct llama_vocab;
struct audition_ctx;
struct clip_ctx;
// NOLINTEND(modernize-use-using,bugprone-forward-declaration-namespace)

namespace llama_omni_server
{

// ── RAII wrappers for opaque model pointers ───────────────────────────────────

/**
 * @brief Custom deleter calling `llama_model_free()`.
 *
 * Defined in model_manager.cpp where llama.h is included; keeps the header
 * clean of the llama.h transitive include.
 */
struct LlamaModelDeleter
{
	/// Free a llama_model loaded by `llama_model_load_from_file`.
	void operator()(llama_model * ptr) const noexcept;
};

/// Owning, move-only handle for a `llama_model`.
using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;

/**
 * @brief Custom deleter calling `llama_free()` for a `llama_context`.
 *
 * Defined in model_manager.cpp where llama.h is included.
 */
struct LlamaContextDeleter
{
	/// Free a llama_context created by `llama_init_from_model`.
	void operator()(llama_context * ptr) const noexcept;
};

/// Owning handle for a `llama_context` (inference context / KV cache).
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

/**
 * @brief Custom deleter calling `audition_free()`.
 *
 * Defined in model_manager.cpp where audition.h is included.
 */
struct AuditionCtxDeleter
{
	/// Free an audition_ctx loaded by `audition_init`.
	void operator()(audition_ctx * ptr) const noexcept;
};

/// Owning handle for an `audition_ctx` (Whisper audio encoder).
using AuditionCtxPtr = std::unique_ptr<audition_ctx, AuditionCtxDeleter>;

/**
 * @brief Custom deleter calling `clip_free()`.
 *
 * Defined in model_manager.cpp where clip.h is included.
 */
struct ClipCtxDeleter
{
	/// Free a clip_ctx loaded by `clip_init`.
	void operator()(clip_ctx * ptr) const noexcept;
};

/// Owning handle for a `clip_ctx` (CLIP vision encoder).
using ClipCtxPtr = std::unique_ptr<clip_ctx, ClipCtxDeleter>;

// ── ModelManager ─────────────────────────────────────────────────────────────

/**
 * @brief Owns the loaded LLM weights and constructs per-session contexts.
 *
 * ### Lifecycle
 *
 * 1. Construct with `ModelManager(cfg)` from the GPU executor (blocking call).
 * 2. Pass a reference to each `Session`; sessions call `create_context(n_ctx)`
 *    to get their own inference context (`llama_context`).
 *
 * ### Thread safety
 *
 * Construction and `create_context()` are blocking GPU calls and must be
 * called on the single-threaded `gpu_ex` pool. After construction, `model()`
 * and `vocab()` are read-only and safe to call from any thread.
 *
 * ### Non-copyable and non-movable
 *
 * `ModelManager` owns GPU resources via `LlamaModelPtr` (a `unique_ptr` with a
 * custom deleter that calls `llama_model_free`).  Copying would require
 * duplicating the entire model in GPU memory, which is not supported by the
 * libllama API.  Move is suppressed because `Session` objects hold a raw
 * non-owning reference to the manager; if the manager were moved the
 * reference would dangle.
 */
// Default-initialised smart pointers/optionals are null/empty; clang-tidy
// misfires here because this class intentionally relies on those defaults.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
class ModelManager
{
public:
	/**
	 * @brief Construct and load all configured model components.
	 *
	 * Loads the LLM unconditionally, then loads the optional audio, vision, and
	 * TTS components if their configured paths are non-empty.
	 *
	 * @param cfg  Application config describing model paths and placement.
	 */
	explicit ModelManager(AppConfig const & cfg);

	/// Destructor defined in .cpp so `LlamaModelPtr` can call `llama_model_free`
	/// without requiring consumers to include llama.h.
	~ModelManager();

	ModelManager(ModelManager const &) = delete;
	ModelManager & operator=(ModelManager const &) = delete;
	ModelManager(ModelManager &&) = delete;
	ModelManager & operator=(ModelManager &&) = delete;

	/**
	 * @brief Create an inference context for one session.
	 *
	 * Blocking call — intended to run on the GPU executor.
	 *
	 * @param n_ctx  Context window size for the per-session KV cache.
	 * @return     New owning inference context handle, or null on failure.
	 */
	[[nodiscard]] LlamaContextPtr create_context(int n_ctx) const;

	/// @return The loaded model; guaranteed available after successful construction.
	[[nodiscard]] llama_model & model() const;

	/// @return The model's vocabulary; guaranteed available after successful construction.
	[[nodiscard]] llama_vocab const & vocab() const;

	/// @return The loaded Whisper audio encoder context.
	/// @throws std::logic_error if audio support was not configured.
	[[nodiscard]] audition_ctx & audio_ctx() const;

	/// @return The loaded CLIP vision encoder context.
	/// @throws std::logic_error if vision support was not configured.
	[[nodiscard]] clip_ctx & vision_ctx() const;

	/// @return The loaded TTS pipeline.
	/// @throws std::logic_error if TTS support was not configured.
	[[nodiscard]] TtsPipeline & tts_pipeline() const;

private:
	/**
	 * @brief Load the LLM from the path in `cfg.model`.
	 *
	 * Blocking helper used by the constructor.
	 *
	 * @param cfg  Application config; uses `cfg.model.llm_path` and `cfg.vram`.
	 * @param placement  Fixed placement decision computed during construction.
	 */
	void load_llm(AppConfig const & cfg, ModelPlacement const & placement);

	/**
	 * @brief Load the Whisper audio encoder from the path in `cfg.model`.
	 *
	 * Blocking helper used by the constructor.
	 *
	 * @param cfg  Application config; uses `cfg.model.audio_path`.
	 * @param placement  Fixed placement decision computed during construction.
	 */
	void load_audio(AppConfig const & cfg, ModelPlacement const & placement);

	/**
	 * @brief Load the CLIP vision encoder from the path in `cfg.model`.
	 *
	 * Blocking helper used by the constructor.
	 *
	 * @param cfg  Application config; uses `cfg.model.vision_path`.
	 * @param placement  Fixed placement decision computed during construction.
	 */
	void load_vision(AppConfig const & cfg, ModelPlacement const & placement);

	/**
	 * @brief Load the TTS pipeline (transformer, weights, Token2Wav) from config.
	 *
	 * Blocking helper used by the constructor.
	 *
	 * @param cfg  Application config; uses `cfg.model.tts_transformer_path`,
	 *             `cfg.model.tts_weights_path`, `cfg.model.projector_path`, and
	 *             `cfg.model.token2wav_dir`.
	 * @param placement  Fixed placement decision computed during construction.
	 */
	void load_tts(AppConfig const & cfg, ModelPlacement const & placement);

	LlamaModelPtr model_;		///< Owns the loaded LLM weights.
	AuditionCtxPtr audio_ctx_;	///< Owns the Whisper audio encoder (optional).
	ClipCtxPtr vision_ctx_;		///< Owns the CLIP vision encoder (optional).
	// Default-initialised unique_ptr is null; no explicit {} needed here.
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	std::unique_ptr<TtsPipeline> tts_pipeline_;	 ///< Owns the TTS pipeline (optional).
	bool kv_cache_on_gpu_ = true;				 ///< Session KV placement derived at construction.
};

}  // namespace llama_omni_server
