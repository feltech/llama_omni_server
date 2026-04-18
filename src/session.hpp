/**
 * @file   session.hpp
 * @brief  Session class, conversation state, and channel message types.
 *
 * A `Session` owns one llama_context (KV cache) and drives inference on the
 * `gpu_ex` thread pool.  It reads audio from `audio_in_ch_`, text from
 * `text_in_ch_`, video from `video_in_ch_`, and speech acks from
 * `speech_ack_ch_`.  Generated speech chunks are written to `output_ch_`.
 *
 * ### Design: channel-based duplex event loop
 *
 * The outer duplex loop reads merged audio, text, and speech-ack events
 * from `merged_ch`. Audio-triggered generation returns at one duplex chunk
 * boundary (`<|listen|>`, `<|chunk_eos|>`, or `<|chunk_tts_eos|>`), emits
 * at most one `SpeechChunkOutFrame`, and records ack progress via merged
 * `SpeechChunkAckEvent`s before later audio units are allowed to continue.
 *
 * ### Video / CLIP scope
 *
 * `run_clip_loop()` runs concurrently on the cobalt main executor, draining
 * `video_in_ch_`, encoding frames via CLIP on `gpu_ex_`, and storing
 * `clip_result_` for insertion at the next audio-triggered generation.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <boost/asio/thread_pool.hpp>
#include <boost/cobalt/channel.hpp>
#include <boost/cobalt/task.hpp>
#include <deque>

#include <boost/cobalt/promise.hpp>
#include <llama.h>

#include "config/config_loader.hpp"
#include "pipeline/model_manager.hpp"
#include "pipeline/vad.hpp"

namespace llama_omni_server
{

// ── KV-cache position tracking ────────────────────────────────────────────────

/// Start and end positions of one `<unit>` block in the LLM KV cache.
///
/// Used for sliding-window context eviction: the oldest `UnitBlock` can be
/// removed from the KV cache to free space while preserving the system prompt.
struct UnitBlock
{
	int pos_start{0};  ///< First KV position of this unit block.
	int pos_end{0};	   ///< One-past-end KV position of this unit block.
};

/// Conversation state tracking KV cache positions and unit block history.
///
/// All fields default to "empty / not started".
struct ConversationState
{
	/// Unique identifier for this session.
	std::string session_id;
	/// KV ranges per completed `<unit>` block.
	std::deque<UnitBlock> unit_history;
	/// Number of tokens in the LLM KV cache.
	int n_past{0};
	/// First KV position eligible for eviction (end of system prompt).
	int sys_prompt_end{0};
	/// TTS context position.
	int tts_n_past{0};
	/// Last KV position of heard audio.
	int last_heard_token_pos{0};
};

// ── Channel message types ─────────────────────────────────────────────────────

/// Text typed by the client; handled as a duplex-context text-only chat turn.
struct TextInput
{
	std::string text;
};

/// Acknowledges that the client has finished playing a speech chunk.
struct SpeechChunkAckEvent
{
	std::uint32_t seq{0};
};

/// Inbound video frame from client: pre-resized 448×448 RGB uint8 pixels.
struct VideoFrame
{
	std::uint32_t seq{0};		 ///< Client-assigned monotonic sequence number.
	std::vector<std::byte> raw;	 ///< RGB uint8 bytes, 448×448×3 = 602 112 bytes.
};

/// One streamed text token from the LLM.
struct TextOutFrame
{
	std::string token;
};

/// Turn-complete signal.
struct DoneOutFrame
{
	bool end_of_turn{false};
};

/// Session-level error notification.
struct ErrorOutFrame
{
	std::string message;
};

/// Chunk-aligned spoken output from the TTS pipeline.
///
/// The text and PCM correspond to the same `<|chunk_eos|>`-delimited speech
/// chunk. The client must acknowledge playback with `SpeechChunkAckEvent`
/// before generation proceeds to the next chunk.
struct SpeechChunkOutFrame
{
	std::uint32_t seq{0};	 ///< Server-assigned monotonic speech chunk sequence number.
	std::string text;		 ///< Visible text aligned to `pcm`.
	std::vector<float> pcm;	 ///< 24 kHz mono float32 samples.
};

/// Video acknowledge — echoes the processed frame back to the client.
///
/// Sent after the CLIP-encoded frame is consumed by the LLM during a
/// generation turn, so the browser can display the model's view.
struct VideoAckOutFrame
{
	std::uint32_t seq{0};		 ///< Sequence number from the `VideoFrame`.
	std::vector<std::byte> raw;	 ///< Echoed RGB uint8 bytes, 448×448×3.
};

using OutFrame =
	std::variant<TextOutFrame, DoneOutFrame, ErrorOutFrame, VideoAckOutFrame, SpeechChunkOutFrame>;

/// Result of a CLIP vision encode.
///
/// Produced by `encode_clip_on_gpu()` and stored as `PendingClipResult`.
/// Consumed when inserting an `<image>` block into a `<unit>` before
/// a generation turn.
struct ClipResult
{
	std::uint32_t seq{0};			///< Sequence number from the originating `VideoFrame`.
	int n_embd{0};					///< Embedding dimension from `clip_n_mmproj_embd`.
	std::vector<float> embeddings;	///< Flat `[n_tokens × n_embd]` float embeddings.
};

/// CLIP encode result bundled with the original frame data needed for the
/// deferred `VideoAckOutFrame`.
struct PendingClipResult
{
	ClipResult clip;			 ///< Encoded embeddings.
	std::uint32_t seq{0};		 ///< Frame sequence number.
	std::vector<std::byte> raw;	 ///< Original RGB bytes for the ack echo.
};

/// Merged event type for the duplex input loop.
/// Audio from `loop_audio_in_ch()` and text from `loop_text_in_ch()`.
using DuplexEvent =
	std::variant<AudioChunkEvent, TextInput, SpeechChunkAckEvent, PendingClipResult>;

// ── Internal LLM output types ─────────────────────────────────────────────────

/// A regular text token from the LLM (word-piece string).
struct TextToken
{
	llama_token id{};	///< Token ID from `llama_sampler_sample`.
	std::string piece;	///< UTF-8 text piece (may be empty for special tokens).
};

/// An audio codec token from the LLM (fed to TTS pipeline).
struct AudioToken
{
	llama_token id{};  ///< Token ID.
};

/// `<|chunk_eos|>` — soft-interrupt boundary within a speaking turn.
struct ChunkEosToken
{
};

/// Result of one returned duplex chunk.
struct DuplexTurnResult
{
	bool end_of_turn{false};		///< True if `<|turn_eos|>` occurred before the return boundary.
	bool ended_with_listen{false};	///< True if the returned boundary was `<|listen|>`.
	std::optional<SpeechChunkOutFrame> speech_chunk;
};

/// Return value of `sample_and_decode_on_gpu()`.
struct SampledOutput
{
	llama_token id{};			///< Sampled token ID.
	std::vector<float> hidden;	///< LLM hidden state at this token position.
};

/// Custom deleter calling `llama_sampler_free()`.
struct LlamaSamplerDeleter
{
	void operator()(llama_sampler * ptr) const noexcept;
};

/// Owning handle for a `llama_sampler` chain.
using LlamaSamplerPtr = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;

// ── Session ───────────────────────────────────────────────────────────────────

/// Manages one active client session: inference context, KV state,
/// and the audio/text/video/output channel set.
///
/// ### Lifecycle
///
/// 1. Construct with injected dependencies.
/// 2. Spawn `run()` as a detached cobalt task on the main executor.
/// 3. Write audio to `audio_in_ch_`, text to `text_in_ch_`, video to
///    `video_in_ch_`; read `OutFrame`s from `output_ch_`.
/// 4. Close channels to end the session; `run()` returns.
///
/// ### Thread model
///
/// `run()` and all methods it calls run on the cobalt main executor (single
/// thread, cooperative).  GPU work is dispatched to `gpu_ex_` via
/// `cobalt::spawn`; the main coroutine suspends until the GPU work returns.
class Session
{
public:
	/**
	 * @brief Construct a session.
	 *
	 * @param cfg          Application configuration.
	 * @param models       Loaded model manager.
	 * @param gpu_ex       Single-thread GPU executor.
	 * @param output_ch    Channel to send `OutFrame` messages.
	 * @param audio_in_ch  Channel to receive raw PCM audio (16 kHz mono).
	 * @param video_in_ch  Channel to receive 448×448 RGB video frames.
	 * @param text_in_ch   Channel to receive `TextInput` messages.
	 * @param speech_ack_ch  Channel to receive `SpeechChunkAckEvent` messages.
	 */
	Session(
		AppConfig const & cfg,
		ModelManager & models,
		boost::asio::thread_pool & gpu_ex,
		boost::cobalt::channel<OutFrame> & output_ch,
		boost::cobalt::channel<std::vector<float>> & audio_in_ch,
		boost::cobalt::channel<VideoFrame> & video_in_ch,
		boost::cobalt::channel<TextInput> & text_in_ch,
		boost::cobalt::channel<SpeechChunkAckEvent> & speech_ack_ch);

	/// Destructor — defined in .cpp to avoid requiring `llama.h` in consumers.
	~Session();

	Session(Session const &) = delete;
	Session & operator=(Session const &) = delete;
	Session(Session &&) = delete;
	Session & operator=(Session &&) = delete;

	/// Run the session event loop.
	///
	/// Initialises GPU resources, prefills the duplex system prompt, then
	/// enters the duplex event loop. Returns when channels are closed.
	[[nodiscard]] boost::cobalt::task<void> run();

	/// Record that raw audio was accepted from the client.
	///
	/// @param sample_count  Number of mono float32 samples accepted at 16 kHz.
	void note_audio_received(std::size_t sample_count) noexcept;

	/// Return the age of the last video frame decoded into the LLM KV (ms),
	/// or 0 if no video frame has been consumed yet.
	[[nodiscard]] std::uint32_t video_last_ms() const noexcept;

	/// Return true when incoming audio is currently accepted.
	[[nodiscard]] bool listening() const noexcept;

	/// Return true once startup finished and the duplex loop is ready for input.
	[[nodiscard]] bool startup_complete() const noexcept;

	/// @return Read-only reference to the conversation state.
	[[nodiscard]] ConversationState const & conv() const noexcept
	{
		return conv_;
	}

private:
	/// Allocate the `llama_context` and sampler on the GPU executor.
	[[nodiscard]] boost::cobalt::task<void> init_gpu_resources();

	/// Release the `llama_context` and sampler on the GPU executor.
	[[nodiscard]] boost::cobalt::task<void> free_gpu_resources();

	/// Run a typed text turn while the session is otherwise in duplex mode.
	///
	/// @param text  User message text.
	/// @return      `true` if the turn ended naturally, `false` if it hit the cap.
	[[nodiscard]] boost::cobalt::promise<bool> run_simplex_turn_in_duplex(std::string text);

	/// Prefill the duplex system prompt (voice reference + text) into the KV cache.
	[[nodiscard]] boost::cobalt::promise<void> init_duplex_system_prompt();

	/// Simulate a brief conversation with empty inputs and listen tokens
	/// to bias the model against premature speaking on startup.
	[[nodiscard]] boost::cobalt::promise<void> seed_listen_turns();

	/// Run the duplex outer loop (merged audio + text event dispatch).
	[[nodiscard]] boost::cobalt::promise<void> loop_duplex_events();

	/// Concurrent CLIP video encode loop. Reads `VideoFrame`s from
	/// `video_in_ch_`, encodes via CLIP on `gpu_ex_`, and stores the latest
	/// result in `pending_clip` for consumption by `loop_duplex_events()`.
	[[nodiscard]] boost::cobalt::task<void> loop_video_in_ch(
		std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch);

	/// Encode a raw RGB frame with the CLIP vision encoder on the GPU.
	///
	/// @param seq  Sequence number from the source `VideoFrame`.
	/// @param raw  RGB uint8 bytes (448×448×3).
	[[nodiscard]] boost::cobalt::task<ClipResult> encode_clip_on_gpu(
		std::uint32_t seq, std::vector<std::byte> raw);

	/// Decode `<image>[clip_embeddings]</image>` into the open `<unit>`.
	///
	/// @param clip  CLIP encode result (seq, n_embd, embeddings).
	[[nodiscard]] boost::cobalt::task<void> batch_decode_image_on_gpu(ClipResult clip);

	/// Buffer incoming PCM into fixed-size audio chunks, writing to `merged_ch`.
	[[nodiscard]] boost::cobalt::task<void> loop_audio_in_ch(
		std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch);

	/// Forward `TextInput` from `text_in_ch_` to `merged_ch`.
	[[nodiscard]] boost::cobalt::task<void> loop_text_in_ch(
		std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch);
	boost::cobalt::task<void> loop_speech_ack_ch(
		std::shared_ptr<boost::cobalt::channel<DuplexEvent>> merged_ch);

	/// Run one duplex generation turn, writing speech chunks to `chunk_ch`.
	///
	/// Samples tokens from the LLM until a returned chunk boundary (`<|listen|>`,
	/// `<|chunk_eos|>`, or `<|chunk_tts_eos|>`) or the hard token cap. A prior
	/// `<|turn_eos|>` marks `end_of_turn` but does not itself force the return.
	[[nodiscard]] boost::cobalt::promise<DuplexTurnResult> generate_duplex_chunk();

	/// Encode a raw PCM buffer and decode the embeddings into the KV cache.
	///
	/// @param pcm         Mono 16 kHz float32 samples.
	/// @param need_logits If true, enables logit output at the last position.
	[[nodiscard]] boost::cobalt::task<void> batch_decode_audio_on_gpu(
		std::vector<float> pcm, bool need_logits);

	/// Load a saved KV state from `path` on the GPU executor.
	[[nodiscard]] boost::cobalt::task<void> load_prompt_cache_on_gpu(std::filesystem::path path);

	/// Save the current KV state to `path` on the GPU executor.
	[[nodiscard]] boost::cobalt::task<void> save_prompt_cache_on_gpu(std::filesystem::path path);

	/// Encode the reference WAV on the GPU executor.
	///
	/// @param wav_path  Path to the 16 kHz reference WAV file.
	/// @return `[n_tokens * n_embd]` float32 embeddings, or empty on failure.
	[[nodiscard]] boost::cobalt::task<std::vector<float>> encode_voice_on_gpu(
		std::filesystem::path wav_path);

	/// Reset the shared audio encoder's streaming KV/cache state.
	[[nodiscard]] boost::cobalt::task<void> clear_audio_encoder_kv_on_gpu();

	/// Decode a block of float embeddings directly on the GPU executor.
	///
	/// @param embeddings   Flat `[n_tokens * n_embd]` float32 buffer.
	/// @param n_embd       Embedding dimension.
	/// @param n_past_start KV position of the first embedding token.
	/// @param need_logits  If true, enable logit output at the last position.
	[[nodiscard]] boost::cobalt::task<void> decode_embeddings_on_gpu(
		std::vector<float> embeddings, int n_embd, int n_past_start, bool need_logits = false);

	/// Decode all prompt tokens in one batch on the GPU executor.
	/// Reads and updates `conv_.n_past` internally.
	[[nodiscard]] boost::cobalt::task<void> decode_prompt_on_gpu(std::vector<llama_token> tokens);

	/// Sample one token, decode it on the GPU, and capture hidden state.
	///
	/// @return `SampledOutput` with the token ID and the hidden state
	///         at the decoded position (needed for TTS conditioning).
	[[nodiscard]] boost::cobalt::task<SampledOutput> sample_and_decode_on_gpu();

	/// Build TTS condition and synthesize PCM on `gpu_ex_`.
	[[nodiscard]] boost::cobalt::task<std::vector<float>> synthesize_speech_on_gpu(
		std::vector<llama_token> token_ids, std::vector<float> hiddens);

	/// Synthesize PCM from accumulated spoken tokens via the GPU executor.
	[[nodiscard]] boost::cobalt::promise<std::vector<float>> synthesize_speech_pcm(
		std::vector<llama_token> token_buf, std::vector<float> hidden_buf);

	/// Synthesize one speech chunk from buffered tokens and hidden states.
	[[nodiscard]] boost::cobalt::promise<std::optional<SpeechChunkOutFrame>>
	synthesize_speech_chunk(
		std::string text, std::vector<llama_token> token_buf, std::vector<float> hidden_buf);

	/// Open a new duplex `<unit>` block when real input for that unit arrives.
	[[nodiscard]] boost::cobalt::promise<void> open_duplex_unit();

	/// Close the current duplex `</unit>` block and record its KV range.
	[[nodiscard]] boost::cobalt::promise<void> close_duplex_unit();

	/// Finalize one duplex unit: emit `DoneOutFrame`, close `</unit>`, and
	/// clear the audio encoder when the model marked end-of-turn.
	[[nodiscard]] boost::cobalt::promise<void> finalize_duplex_turn(DuplexTurnResult result);

	/// Handle one typed text turn while inside the duplex event loop.
	[[nodiscard]] boost::cobalt::promise<void> handle_text_input_in_duplex(std::string text);

	/// Emit a terminal error frame and close inbound channels.
	[[nodiscard]] boost::cobalt::promise<void> fail_session(std::string message);

	/// Write accumulated per-turn TTS PCM to a debug WAV if configured.
	void flush_debug_audio_wav();

	/// Evict the oldest unit blocks from the KV cache until overflow headroom is restored.
	[[nodiscard]] boost::cobalt::promise<bool> evict_oldest_units();

	/// Perform KV cache surgery: remove a range and shift remaining positions.
	///
	/// @param evict_start  First KV position to remove (inclusive).
	/// @param evict_end    One-past-end KV position to remove.
	/// @param delta        Amount to shift remaining positions down by.
	[[nodiscard]] boost::cobalt::task<void> evict_kv_on_gpu(
		int evict_start, int evict_end, int delta);

	AppConfig const & cfg_;
	ModelManager & models_;
	boost::asio::thread_pool & gpu_ex_;
	boost::cobalt::channel<OutFrame> & output_ch_;
	boost::cobalt::channel<std::vector<float>> & audio_in_ch_;
	boost::cobalt::channel<VideoFrame> & video_in_ch_;
	boost::cobalt::channel<TextInput> & text_in_ch_;
	boost::cobalt::channel<SpeechChunkAckEvent> & speech_ack_ch_;

	/// KV cache; initialised in `init_gpu_resources()`.
	LlamaContextPtr ctx_;
	/// Temperature sampler; initialised in `init_gpu_resources()`.
	LlamaSamplerPtr sampler_;
	/// KV position tracking.
	ConversationState conv_;

	/// Accumulated TTS PCM for the current turn (debug output).
	std::vector<float> debug_turn_audio_;
	/// Monotonic per-session turn counter for debug WAVs.
	std::uint64_t debug_turn_index_{0};
	/// Monotonic speech-chunk sequence number.
	std::uint32_t next_speech_chunk_seq_{1};
	/// Raw client audio accepted but not yet processed into buffered chunks.
	std::size_t num_pending_audio_samples_{0};
	/// Last time a CLIP frame was decoded into the LLM.
	std::optional<std::chrono::steady_clock::time_point> last_video_decode_at_;
	/// True while the duplex loop is accepting audio instead of dropping it.
	bool listening_{true};
	/// True once GPU init and prompt prefill completed successfully.
	bool startup_complete_{false};
	/// KV position where the current open `<unit>` block began.
	int current_unit_start_{0};
	/// True when an explicit duplex `<unit>` is currently open in KV.
	bool current_unit_open_{false};
};

}  // namespace llama_omni_server
