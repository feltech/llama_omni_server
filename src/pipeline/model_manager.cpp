/**
 * @file   model_manager.cpp
 * @brief  ModelManager implementation — loads LLM, audio encoder, and vision encoder.
 */

#include "./model_manager.hpp"
#include "./tts_pipeline.hpp"

#include <format>
#include <limits>
#include <stdexcept>

// llama.h exposes the full libllama C API (model loading, context init, vocab).
// audition.h exposes the Whisper audio encoder API (audition_init, audition_free, etc.).
// clip.h exposes the CLIP vision encoder API (clip_init, clip_free, etc.).
// All included here (not in the header) to keep downstream TUs clean.
#include <llama.h>
#include <spdlog/spdlog.h>

#include "omni/audition.h"
#include "omni/clip.h"

namespace llama_omni_server
{

namespace
{

constexpr std::uint64_t kBytesPerMib = 1024ULL * 1024ULL;

}  // namespace

// ── LlamaModelDeleter ─────────────────────────────────────────────────────────

void LlamaModelDeleter::operator()(llama_model * ptr) const noexcept
{
	/// `llama_model_free` releases all model weights, vocabulary, and metadata
	/// allocated by `llama_model_load_from_file`.
	llama_model_free(ptr);
}

void LlamaContextDeleter::operator()(llama_context * ptr) const noexcept
{
	/// `llama_free` releases the KV cache and all GPU buffers for this context.
	/// Does NOT free the model weights (model is owned by ModelManager).
	llama_free(ptr);
}

// ── AuditionCtxDeleter ────────────────────────────────────────────────────────

void AuditionCtxDeleter::operator()(audition_ctx * ptr) const noexcept
{
	/// `audition_free` releases all Whisper encoder tensors and GGML backends
	/// allocated by `audition_init`.
	audition_free(ptr);
}

// ── ClipCtxDeleter ────────────────────────────────────────────────────────────

void ClipCtxDeleter::operator()(clip_ctx * ptr) const noexcept
{
	/// `clip_free` releases all CLIP vision encoder tensors and GGML backends
	/// allocated by `clip_init`.
	clip_free(ptr);
}

// ── ModelManager ─────────────────────────────────────────────────────────────

ModelManager::~ModelManager() = default;

ModelManager::ModelManager(AppConfig const & cfg)
{
	std::optional<std::uint64_t> const probed_vram = VramPlanner::probe_available_vram_bytes();
	if (!probed_vram.has_value())
	{
		spdlog::warn(
			"ModelManager: VRAM probe unavailable — assuming ample GPU budget for "
			"placement planning");
	}

	ModelPlacement const placement = VramPlanner::compute_placement(
		cfg, probed_vram.value_or(std::numeric_limits<std::uint64_t>::max()));
	spdlog::info(
		"ModelManager: placement llm_gpu_layers={} audio_on_gpu={} clip_on_gpu={} "
		"tts_transformer_on_gpu={} token2wav_on_gpu={} kv_cache_on_gpu={} "
		"usable_vram_mib={} reserved_vram_mib={}",
		placement.n_llm_gpu_layers,
		placement.audio_on_gpu,
		placement.clip_on_gpu,
		placement.tts_transformer_on_gpu,
		placement.token2wav_on_gpu,
		placement.kv_cache_on_gpu,
		placement.usable_vram_bytes / kBytesPerMib,
		placement.reserved_vram_bytes / kBytesPerMib);

	kv_cache_on_gpu_ = placement.kv_cache_on_gpu;
	load_llm(cfg, placement);
	load_audio(cfg, placement);
	load_vision(cfg, placement);
	load_tts(cfg, placement);
}

void ModelManager::load_llm(AppConfig const & cfg, ModelPlacement const & placement)
{
	if (cfg.model.llm_path.empty())
	{
		throw std::runtime_error{"ModelManager::load_llm: cfg.model.llm_path is empty"};
	}

	spdlog::info("ModelManager: loading LLM from '{}'", cfg.model.llm_path);

	/// `llama_model_default_params()` returns a struct with sensible defaults:
	/// - `n_gpu_layers = 0` (all CPU), `use_mmap = true`, `use_mlock = false`.
	/// We override `n_gpu_layers` from the placement plan.  The plan convention
	/// uses -1 to mean "all layers on GPU", but libllama does not accept -1
	/// directly, so we convert it to INT_MAX.
	auto params = llama_model_default_params();
	params.n_gpu_layers = (placement.n_llm_gpu_layers < 0) ? std::numeric_limits<int>::max()
														   : placement.n_llm_gpu_layers;

	/// `llama_model_load_from_file` reads the GGUF file, maps the weights into
	/// memory, and uploads the requested number of layers to the GPU via CUDA.
	/// Returns nullptr on any error (bad path, OOM, incompatible format).
	llama_model * raw_model = llama_model_load_from_file(cfg.model.llm_path.c_str(), params);

	if (raw_model == nullptr)
	{
		throw std::runtime_error{
			std::format("ModelManager: failed to load model from '{}'", cfg.model.llm_path)};
	}

	model_.reset(raw_model);
	spdlog::info("ModelManager: LLM loaded successfully");
}

void ModelManager::load_audio(AppConfig const & cfg, ModelPlacement const & placement)
{
	if (cfg.model.audio_path.empty())
	{
		spdlog::debug("ModelManager::load_audio: cfg.model.audio_path is empty — skipping");
		return;
	}

	spdlog::info("ModelManager: loading audio encoder from '{}'", cfg.model.audio_path);

	/// `audition_init` loads the Whisper encoder weights from a GGUF file and
	/// builds a GGML compute graph for the mel-spectrogram → embedding pipeline.
	/// `use_gpu = true` offloads the encoder to CUDA; `GGML_LOG_LEVEL_WARN`
	/// suppresses verbose tensor-loading messages.
	bool const use_gpu = placement.audio_on_gpu;
	audition_context_params const audio_params{
		.use_gpu = use_gpu, .verbosity = GGML_LOG_LEVEL_WARN};
	audition_ctx * raw_ctx = audition_init(cfg.model.audio_path.c_str(), audio_params);

	if (raw_ctx == nullptr)
	{
		throw std::runtime_error{std::format(
			"ModelManager: failed to load audio encoder from '{}'", cfg.model.audio_path)};
	}

	audio_ctx_.reset(raw_ctx);
	spdlog::info("ModelManager: audio encoder loaded successfully");
}

void ModelManager::load_vision(AppConfig const & cfg, ModelPlacement const & placement)
{
	if (cfg.model.vision_path.empty())
	{
		spdlog::debug("ModelManager::load_vision: cfg.model.vision_path is empty — skipping");
		return;
	}

	spdlog::info("ModelManager: loading vision encoder from '{}'", cfg.model.vision_path);
	/// `clip_init` loads the CLIP vision encoder weights from a GGUF file.
	/// `ctx_v` is the vision encoder context; `ctx_a` is the audio context
	/// (which we do not use — the Whisper encoder is loaded via audition.h).
	bool const use_gpu = placement.clip_on_gpu;
	clip_context_params const vision_params{.use_gpu = use_gpu, .verbosity = GGML_LOG_LEVEL_WARN};
	clip_init_result const result = clip_init(cfg.model.vision_path.c_str(), vision_params);

	if (result.ctx_v == nullptr)
	{
		throw std::runtime_error{std::format(
			"ModelManager: failed to load vision encoder from '{}'", cfg.model.vision_path)};
	}

	vision_ctx_.reset(result.ctx_v);

	// clip_init also provides an audio context (ctx_a) that we don't use — free it.
	if (result.ctx_a != nullptr)
	{
		clip_free(result.ctx_a);
	}

	spdlog::info("ModelManager: vision encoder loaded successfully");
}

void ModelManager::load_tts(AppConfig const & cfg, ModelPlacement const & placement)
{
	if (cfg.model.tts_transformer_path.empty())
	{
		spdlog::debug("ModelManager::load_tts: tts_transformer_path empty — skipping");
		return;
	}

	/// Create a fresh TtsPipeline and load it.  Pass the main LLM model so the
	/// vendor code can set `TTSContext.model_llm` for internal validation.
	tts_pipeline_ = std::make_unique<TtsPipeline>();
	tts_pipeline_->load(
		cfg, &model(), placement.tts_transformer_on_gpu, placement.token2wav_on_gpu);
	spdlog::info("ModelManager: TTS pipeline loaded");
}

LlamaContextPtr ModelManager::create_context(int const n_ctx) const
{
	/// `llama_context_default_params()` returns defaults:
	/// - `n_ctx = 512`, `n_batch = 2048`, `n_ubatch = 512`, `n_threads = nproc/2`.
	/// We override `n_ctx` from the caller; larger context uses more VRAM.
	/// `embeddings = true` enables `llama_get_embeddings_ith` so that
	/// `run_duplex_turn` can extract hidden states for TTS conditioning.
	/// `offload_kqv = false` keeps the KV cache and attention K/Q/V ops on CPU,
	/// matching the `vram.kv_cache_gpu = false` escape hatch for VRAM-constrained setups.
	auto ctx_params = llama_context_default_params();
	ctx_params.n_ctx = static_cast<std::uint32_t>(n_ctx);
	ctx_params.embeddings = true;
	ctx_params.offload_kqv = kv_cache_on_gpu_;

	/// `llama_init_from_model` allocates the KV cache and sets up compute graphs
	/// for the given context size.  One context per session; they share the model
	/// weights (read-only) but each has its own KV state.
	// `llama_context` is an opaque C handle whose pointee mutability is controlled
	// entirely by libllama; adding pointee const here would be a lie.
	// NOLINTNEXTLINE(misc-const-correctness)
	llama_context * const ctx = llama_init_from_model(&model(), ctx_params);

	if (ctx == nullptr)
	{
		spdlog::error("ModelManager: failed to create context (n_ctx={})", n_ctx);
	}

	return LlamaContextPtr{ctx};
}

llama_model & ModelManager::model() const
{
	return *model_;
}

llama_vocab const & ModelManager::vocab() const
{
	/// `llama_model_get_vocab` returns the model's vocabulary — the mapping
	/// between string tokens and integer token IDs.  The pointer is owned by
	/// the model and is valid for the model's lifetime.
	auto const * const vocab = llama_model_get_vocab(&model());
	if (vocab == nullptr)
	{
		throw std::logic_error{"ModelManager::vocab returned null"};
	}
	return *vocab;
}

audition_ctx & ModelManager::audio_ctx() const
{
	return *audio_ctx_;
}

clip_ctx & ModelManager::vision_ctx() const
{
	return *vision_ctx_;
}

TtsPipeline & ModelManager::tts_pipeline() const
{
	return *tts_pipeline_;
}

}  // namespace llama_omni_server
