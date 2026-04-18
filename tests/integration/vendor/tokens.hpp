#pragma once

#include <cstdlib>
#include <string>

/**
 * @file tokens.hpp
 * @brief Special token constants for MiniCPM-o 4.5
 *
 * This file defines all special tokens used by MiniCPM-o 4.5 model.
 * Tokens are extracted from the tokenizer.json file in the model directory.
 *
 * Token types:
 * - Basic tokens (IDs < 151643): Standard vocabulary tokens
 * - Special tokens (IDs >= 151643): Control and multimodal tokens
 *
 * Multimodal token categories:
 * 1. Vision tokens: <image>, </image>, <ref>, <box>, <quad>, etc.
 * 2. Audio tokens: <|audio|>, <|audio_start|>, <|audio_end|>
 * 3. TTS tokens: <|tts_bos|>, <|tts_eos|>, <|tts_pad|>
 * 4. Spoken tokens: <|spk|>, <|spk_bos|>, <|spk_eos|>
 * 5. Interaction tokens: <|listen|>, <|speak|>, <|interrupt|>
 * 6. VAD tokens: <|vad_start|>, <|vad_end|>
 * 7. Emotion tokens: <|emotion_start|>, <|emotion_end|>
 * 8. Prosody tokens: <|speed_start|>, <|speed_end|>, <|pitch_start|>, <|pitch_end|>
 * 9. Turn tokens: <|turn_bos|>, <|turn_eos|>, <|chunk_bos|>, <|chunk_eos|>
 * 10. Timbre tokens: <|timbre_7|> through <|timbre_31|>
 */

/**
 * @name Basic Special Tokens
 * @brief Standard control tokens
 * @{
 */

/// Unknown token - used for out-of-vocabulary words
constexpr char const * TOKEN_UNK = "<unk>";
constexpr int TOKEN_UNK_ID = 128244;

/// End of text token
constexpr char const * TOKEN_END_OF_TEXT = "<|end_of_text|>";
constexpr int TOKEN_END_OF_TEXT_ID = 151643;

/// Start of message (BOS) token
constexpr char const * TOKEN_BOS = "<|im_start|>";
constexpr int TOKEN_BOS_ID = 151644;

/// End of message (EOS) token
constexpr char const * TOKEN_EOS = "<|im_end|>";
constexpr int TOKEN_EOS_ID = 151645;

///@}

/**
 * @name Vision Tokens
 * @brief Image-related multimodal tokens
 * @{
 */

/// Object reference start
constexpr char const * TOKEN_OBJECT_REF_START = "<|object_ref_start|>";
constexpr int TOKEN_OBJECT_REF_START_ID = 151646;

/// Object reference end
constexpr char const * TOKEN_OBJECT_REF_END = "<|object_ref_end|>";
constexpr int TOKEN_OBJECT_REF_END_ID = 151647;

/// Box start
constexpr char const * TOKEN_BOX_START = "<|box_start|>";
constexpr int TOKEN_BOX_START_ID = 151648;

/// Box end
constexpr char const * TOKEN_BOX_END = "<|box_end|>";
constexpr int TOKEN_BOX_END_ID = 151649;

/// Quad start
constexpr char const * TOKEN_QUAD_START = "<|quad_start|>";
constexpr int TOKEN_QUAD_START_ID = 151650;

/// Quad end
constexpr char const * TOKEN_QUAD_END = "<|quad_end|>";
constexpr int TOKEN_QUAD_END_ID = 151651;

/// Vision start
constexpr char const * TOKEN_VISION_START = "<|vision_start|>";
constexpr int TOKEN_VISION_START_ID = 151652;

/// Vision end
constexpr char const * TOKEN_VISION_END = "<|vision_end|>";
constexpr int TOKEN_VISION_END_ID = 151653;

/// Vision pad
constexpr char const * TOKEN_VISION_PAD = "<|vision_pad|>";
constexpr int TOKEN_VISION_PAD_ID = 151654;

/// Image pad
constexpr char const * TOKEN_IMAGE_PAD = "<|image_pad|>";
constexpr int TOKEN_IMAGE_PAD_ID = 151655;

/// Video pad
constexpr char const * TOKEN_VIDEO_PAD = "<|video_pad|>";
constexpr int TOKEN_VIDEO_PAD_ID = 151656;

/// Image marker (actual image content)
constexpr char const * TOKEN_IMAGE = "<image>";
constexpr int TOKEN_IMAGE_ID = 151669;

/// End of image marker
constexpr char const * TOKEN_IMAGE_END = "</image>";
constexpr int TOKEN_IMAGE_END_ID = 151670;

/// Reference marker
constexpr char const * TOKEN_REF = "<ref>";
constexpr int TOKEN_REF_ID = 151671;

/// End of reference marker
constexpr char const * TOKEN_REF_END = "</ref>";
constexpr int TOKEN_REF_END_ID = 151672;

/// Box marker
constexpr char const * TOKEN_BOX = "<box>";
constexpr int TOKEN_BOX_ID = 151673;

/// End of box marker
constexpr char const * TOKEN_BOX_END_TOKEN = "</box>";
constexpr int TOKEN_BOX_END_TOKEN_ID = 151674;

/// Quad marker
constexpr char const * TOKEN_QUAD = "<quad>";
constexpr int TOKEN_QUAD_ID = 151675;

/// End of quad marker
constexpr char const * TOKEN_QUAD_END_TOKEN = "</quad>";
constexpr int TOKEN_QUAD_END_TOKEN_ID = 151676;

/// Point marker
constexpr char const * TOKEN_POINT = "<point>";
constexpr int TOKEN_POINT_ID = 151677;

/// End of point marker
constexpr char const * TOKEN_POINT_END = "</point>";
constexpr int TOKEN_POINT_END_ID = 151678;

/// Slice marker
constexpr char const * TOKEN_SLICE = "<slice>";
constexpr int TOKEN_SLICE_ID = 151679;

/// End of slice marker
constexpr char const * TOKEN_SLICE_END = "</slice>";
constexpr int TOKEN_SLICE_END_ID = 151680;

/// Image ID marker
constexpr char const * TOKEN_IMAGE_ID_START = "<image_id>";
constexpr int TOKEN_IMAGE_ID_START_ID = 151681;

/// End of image ID marker
constexpr char const * TOKEN_IMAGE_ID_END = "</image_id>";
constexpr int TOKEN_IMAGE_ID_END_ID = 151682;

/// Unit marker
constexpr char const * TOKEN_UNIT = "<unit>";
constexpr int TOKEN_UNIT_ID = 151683;

/// End of unit marker
constexpr char const * TOKEN_UNIT_END = "</unit>";
constexpr int TOKEN_UNIT_END_ID = 151684;

/// Answer marker
constexpr char const * TOKEN_ANSWER = "<answer>";
constexpr int TOKEN_ANSWER_ID = 151685;

/// End of answer marker
constexpr char const * TOKEN_ANSWER_END = "</answer>";
constexpr int TOKEN_ANSWER_END_ID = 151686;

/// Focus marker
constexpr char const * TOKEN_FOCUS = "<focus>";
constexpr int TOKEN_FOCUS_ID = 151687;

/// End of focus marker
constexpr char const * TOKEN_FOCUS_END = "</focus>";
constexpr int TOKEN_FOCUS_END_ID = 151688;

/// Line marker
constexpr char const * TOKEN_LINE = "<line>";
constexpr int TOKEN_LINE_ID = 151689;

/// End of line marker
constexpr char const * TOKEN_LINE_END = "</line>";
constexpr int TOKEN_LINE_END_ID = 151690;

/// Perception marker
constexpr char const * TOKEN_PERCEPTION = "<perception>";
constexpr int TOKEN_PERCEPTION_ID = 151691;

/// End of perception marker
constexpr char const * TOKEN_PERCEPTION_END = "</perception>";
constexpr int TOKEN_PERCEPTION_END_ID = 151692;

/// Source image marker
constexpr char const * TOKEN_SOURCE_IMAGE = "<source_image>";
constexpr int TOKEN_SOURCE_IMAGE_ID = 151693;

/// End of source image marker
constexpr char const * TOKEN_SOURCE_IMAGE_END = "</source_image>";
constexpr int TOKEN_SOURCE_IMAGE_END_ID = 151694;

/// Image save marker
constexpr char const * TOKEN_IMAGE_SAVE = "<image_save_to>";
constexpr int TOKEN_IMAGE_SAVE_ID = 151695;

/// End of image save marker
constexpr char const * TOKEN_IMAGE_SAVE_END = "</image_save_to>";
constexpr int TOKEN_IMAGE_SAVE_END_ID = 151696;

///@}

/**
 * @name Audio Input Tokens
 * @brief Audio input processing tokens
 * @{
 */

/// Start of audio input
/// Example: "<|audio_start|>[audio_embeddings]<|audio_end|>"
constexpr char const * TOKEN_AUDIO_START = "<|audio_start|>";
constexpr int TOKEN_AUDIO_START_ID = 151697;

/// Audio input marker (short form)
/// Example: "<|audio|>[audio_embeddings]"
constexpr char const * TOKEN_AUDIO = "<|audio|>";
constexpr int TOKEN_AUDIO_ID = 151698;

/// End of audio input
constexpr char const * TOKEN_AUDIO_END = "<|audio_end|>";
constexpr int TOKEN_AUDIO_END_ID = 151699;

///@}

/**
 * @name Speech Tokens (Speaker/Spk)
 * @brief Spoken audio tokens for speaker identification
 * @{
 */

/// Start of speaker turn
constexpr char const * TOKEN_SPK_BOS = "<|spk_bos|>";
constexpr int TOKEN_SPK_BOS_ID = 151700;

/// Speaker marker
constexpr char const * TOKEN_SPK = "<|spk|>";
constexpr int TOKEN_SPK_ID = 151701;

/// End of speaker turn
constexpr char const * TOKEN_SPK_END = "<|spk_eos|>";
constexpr int TOKEN_SPK_END_ID = 151702;

///@}

/**
 * @name TTS (Text-to-Speech) Tokens
 * @brief Tokens for TTS output generation
 * @{
 */

/// Start of TTS sequence
constexpr char const * TOKEN_TTS_BOS = "<|tts_bos|>";
constexpr int TOKEN_TTS_BOS_ID = 151703;

/// End of TTS sequence
constexpr char const * TOKEN_TTS_END = "<|tts_eos|>";
constexpr int TOKEN_TTS_END_ID = 151704;

///@}

/**
 * @name Interaction/Dialogue Tokens
 * @brief Conversation and turn-taking tokens
 * @{
 */

/// Listen mode - model is listening for user input
/// Example: "<|listen|><|audio_start|>..."
constexpr char const * TOKEN_LISTEN = "<|listen|>";
constexpr int TOKEN_LISTEN_ID = 151705;

/// Speak mode - model is generating speech output
/// Example: "<|speak|>..."
constexpr char const * TOKEN_SPEAK = "<|speak|>";
constexpr int TOKEN_SPEAK_ID = 151706;

/// Interrupt flag - user interrupted the model
constexpr char const * TOKEN_INTERRUPT = "<|interrupt|>";
constexpr int TOKEN_INTERRUPT_ID = 151707;

///@}

/**
 * @name VAD (Voice Activity Detection) Tokens
 * @brief Speech detection tokens
 * @{
 */

/// Start of voice activity
constexpr char const * TOKEN_VAD_START = "<|vad_start|>";
constexpr int TOKEN_VAD_START_ID = 151708;

/// End of voice activity
constexpr char const * TOKEN_VAD_END = "<|vad_end|>";
constexpr int TOKEN_VAD_END_ID = 151709;

///@}

/**
 * @name Emotion Tokens
 * @brief Emotion expression tokens
 * @{
 */

/// Start of emotion expression
constexpr char const * TOKEN_EMOTION_START = "<|emotion_start|>";
constexpr int TOKEN_EMOTION_START_ID = 151710;

/// End of emotion expression
constexpr char const * TOKEN_EMOTION_END = "<|emotion_end|>";
constexpr int TOKEN_EMOTION_END_ID = 151711;

///@}

/**
 * @name Prosody Tokens (Speaking Style)
 * @brief Speech prosody and style tokens
 * @{
 */

/// Start of speed control
constexpr char const * TOKEN_SPEED_START = "<|speed_start|>";
constexpr int TOKEN_SPEED_START_ID = 151712;

/// End of speed control
constexpr char const * TOKEN_SPEED_END = "<|speed_end|>";
constexpr int TOKEN_SPEED_END_ID = 151713;

/// Start of pitch control
constexpr char const * TOKEN_PITCH_START = "<|pitch_start|>";
constexpr int TOKEN_PITCH_START_ID = 151714;

/// End of pitch control
constexpr char const * TOKEN_PITCH_END = "<|pitch_end|>";
constexpr int TOKEN_PITCH_END_ID = 151715;

///@}

/**
 * @name Turn/Chunk Tokens (Full-Duplex)
 * @brief Conversation turn and streaming chunk tokens
 * @{
 */

/// Start of conversation turn
constexpr char const * TOKEN_TURN_BOS = "<|turn_bos|>";
constexpr int TOKEN_TURN_BOS_ID = 151716;

/// End of conversation turn
constexpr char const * TOKEN_TURN_END = "<|turn_eos|>";
constexpr int TOKEN_TURN_END_ID = 151717;

/// End of streaming chunk
constexpr char const * TOKEN_CHUNK_END = "<|chunk_eos|>";
constexpr int TOKEN_CHUNK_END_ID = 151718;

/// Start of streaming chunk
constexpr char const * TOKEN_CHUNK_BOS = "<|chunk_bos|>";
constexpr int TOKEN_CHUNK_BOS_ID = 151719;

/// Start of chunk TTS
constexpr char const * TOKEN_CHUNK_TTS_BOS = "<|chunk_tts_bos|>";
constexpr int TOKEN_CHUNK_TTS_BOS_ID = 151720;

/// End of chunk TTS
constexpr char const * TOKEN_CHUNK_TTS_END = "<|chunk_tts_eos|>";
constexpr int TOKEN_CHUNK_TTS_END_ID = 151721;

/// TTS padding token
constexpr char const * TOKEN_TTS_PAD = "<|tts_pad|>";
constexpr int TOKEN_TTS_PAD_ID = 151722;

///@}

/**
 * @name Timbre Tokens (Voice Cloning)
 * @brief Voice timbre/style tokens (timbre_7 through timbre_31)
 * @{
 */

/// Timbre 7 token
constexpr char const * TOKEN_TIMBRE_7 = "<|timbre_7|>";
constexpr int TOKEN_TIMBRE_7_ID = 151723;

/// Timbre 8 token
constexpr char const * TOKEN_TIMBRE_8 = "<|timbre_8|>";
constexpr int TOKEN_TIMBRE_8_ID = 151724;

/// Timbre 9 token
constexpr char const * TOKEN_TIMBRE_9 = "<|timbre_9|>";
constexpr int TOKEN_TIMBRE_9_ID = 151725;

/// Timbre 10 token
constexpr char const * TOKEN_TIMBRE_10 = "<|timbre_10|>";
constexpr int TOKEN_TIMBRE_10_ID = 151726;

/// Timbre 11 token
constexpr char const * TOKEN_TIMBRE_11 = "<|timbre_11|>";
constexpr int TOKEN_TIMBRE_11_ID = 151727;

/// Timbre 12 token
constexpr char const * TOKEN_TIMBRE_12 = "<|timbre_12|>";
constexpr int TOKEN_TIMBRE_12_ID = 151728;

/// Timbre 13 token
constexpr char const * TOKEN_TIMBRE_13 = "<|timbre_13|>";
constexpr int TOKEN_TIMBRE_13_ID = 151729;

/// Timbre 14 token
constexpr char const * TOKEN_TIMBRE_14 = "<|timbre_14|>";
constexpr int TOKEN_TIMBRE_14_ID = 151730;

/// Timbre 15 token
constexpr char const * TOKEN_TIMBRE_15 = "<|timbre_15|>";
constexpr int TOKEN_TIMBRE_15_ID = 151731;

/// Timbre 16 token
constexpr char const * TOKEN_TIMBRE_16 = "<|timbre_16|>";
constexpr int TOKEN_TIMBRE_16_ID = 151732;

/// Timbre 17 token
constexpr char const * TOKEN_TIMBRE_17 = "<|timbre_17|>";
constexpr int TOKEN_TIMBRE_17_ID = 151733;

/// Timbre 18 token
constexpr char const * TOKEN_TIMBRE_18 = "<|timbre_18|>";
constexpr int TOKEN_TIMBRE_18_ID = 151734;

/// Timbre 19 token
constexpr char const * TOKEN_TIMBRE_19 = "<|timbre_19|>";
constexpr int TOKEN_TIMBRE_19_ID = 151735;

/// Timbre 20 token
constexpr char const * TOKEN_TIMBRE_20 = "<|timbre_20|>";
constexpr int TOKEN_TIMBRE_20_ID = 151736;

/// Timbre 21 token
constexpr char const * TOKEN_TIMBRE_21 = "<|timbre_21|>";
constexpr int TOKEN_TIMBRE_21_ID = 151737;

/// Timbre 22 token
constexpr char const * TOKEN_TIMBRE_22 = "<|timbre_22|>";
constexpr int TOKEN_TIMBRE_22_ID = 151738;

/// Timbre 23 token
constexpr char const * TOKEN_TIMBRE_23 = "<|timbre_23|>";
constexpr int TOKEN_TIMBRE_23_ID = 151739;

/// Timbre 24 token
constexpr char const * TOKEN_TIMBRE_24 = "<|timbre_24|>";
constexpr int TOKEN_TIMBRE_24_ID = 151740;

/// Timbre 25 token
constexpr char const * TOKEN_TIMBRE_25 = "<|timbre_25|>";
constexpr int TOKEN_TIMBRE_25_ID = 151741;

/// Timbre 26 token
constexpr char const * TOKEN_TIMBRE_26 = "<|timbre_26|>";
constexpr int TOKEN_TIMBRE_26_ID = 151742;

/// Timbre 27 token
constexpr char const * TOKEN_TIMBRE_27 = "<|timbre_27|>";
constexpr int TOKEN_TIMBRE_27_ID = 151743;

/// Timbre 28 token
constexpr char const * TOKEN_TIMBRE_28 = "<|timbre_28|>";
constexpr int TOKEN_TIMBRE_28_ID = 151744;

/// Timbre 29 token
constexpr char const * TOKEN_TIMBRE_29 = "<|timbre_29|>";
constexpr int TOKEN_TIMBRE_29_ID = 151745;

/// Timbre 30 token
constexpr char const * TOKEN_TIMBRE_30 = "<|timbre_30|>";
constexpr int TOKEN_TIMBRE_30_ID = 151746;

/// Timbre 31 token
constexpr char const * TOKEN_TIMBRE_31 = "<|timbre_31|>";
constexpr int TOKEN_TIMBRE_31_ID = 151747;

///@}

/**
 * @name Chat Template Constants
 * @brief Role markers and formatting strings for chat templates
 *
 * MiniCPM-o uses a specific chat template format for multimodal conversations:
 *
 * Basic format:
 *   " <role>\n<content> <role>\nassistant\n<response>"
 *
 * Example with vision:
 *   " lda\n<image>What's in this image? lda\nassistant\nI see a circle."
 *
 * Example with audio:
 *   " lda\n<|audio|>Transcribe this audio. lda\nassistant\nThe audio says..."
 *
 * The role "lda" appears to be derived from "llama" or "LLM" and acts as the user role.
 * The "assistant" role marks where the model should start generating its response.
 *
 * @{
 */

/// User role marker - appears to be short for "llama/LLM/dialogue agent"
/// Used at the start and end of user messages
/// Example: " lda\n<content> lda\nassistant\n"
constexpr char const * ROLE_USER = " lda\n";

/// Assistant role marker - marks where model starts generating
/// Always preceded by user role and followed by newline
/// Example: "...user content lda\nassistant\n<model generates here>"
constexpr char const * ROLE_ASSISTANT = "assistant\n";

/// Complete role transition - user to assistant
/// This is the pattern that signals "user finished, model should respond"
/// Example: " lda\n<user question> lda\nassistant\n"
constexpr char const * ROLE_TRANSITION = " lda\nassistant\n";

///@}

/**
 * @name Model File Paths
 * @brief Path helpers for MiniCPM-o model files and test data.
 *
 * Paths are resolved from the LLAMAOMNISERVER_MODEL_ROOT and
 * LLAMAOMNISERVER_TEST_REPO_ROOT environment variables injected by CMake/CTest.
 * Fallbacks are relative paths suitable for manual invocation from the repo root.
 *
 * Standard model directory layout:
 *   models/
 *     gguf/
 *       MiniCPM-o-4_5-Q4_K_M.gguf              (main LLM, 4.7 GB)
 *       vision/MiniCPM-o-4_5-vision-F16.gguf   (CLIP encoder, 1 GB)
 *       audio/MiniCPM-o-4_5-audio-F16.gguf     (audio encoder, 629 MB)
 *       tts/MiniCPM-o-4_5-tts-transformer-F16.gguf
 *       tts/MiniCPM-o-4_5-tts-weights-F16.gguf
 *       tts/MiniCPM-o-4_5-projector-F16.gguf
 *       token2wav-gguf/                         (Token2Wav vocoder directory)
 *
 * @{
 */

// NOLINTNEXTLINE(concurrency-mt-unsafe) — single-threaded test process
/// Returns the root directory of MiniCPM-o model files.
inline std::string path_model_root()
{
	// NOLINTNEXTLINE(concurrency-mt-unsafe)
	char const * env = std::getenv("LLAMAOMNISERVER_MODEL_ROOT");
	return env != nullptr ? env : "../../models/gguf";
}

/// Returns the test_data directory within the repo.
inline std::string path_test_data_dir()
{
	// NOLINTNEXTLINE(concurrency-mt-unsafe)
	char const * env = std::getenv("LLAMAOMNISERVER_TEST_REPO_ROOT");
	return std::string{env != nullptr ? env : "."} + "/test_data";
}

/// Path to main LLM model (8.2B parameters, Q4_K_M quantization)
inline std::string path_model_llm()
{
	return path_model_root() + "/MiniCPM-o-4_5-Q4_K_M.gguf";
}

/// Path to CLIP vision encoder (F16 precision)
inline std::string path_model_vision()
{
	return path_model_root() + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
}

/// Path to audio encoder (F16 precision)
inline std::string path_model_audio()
{
	return path_model_root() + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
}

/// Path to TTS projector model (projector_semantic, F16 precision)
inline std::string path_model_projector()
{
	return path_model_root() + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
}

/// Path to TTS transformer model (standard llama tensors, F16 precision)
inline std::string path_model_tts_transformer()
{
	return path_model_root() + "/tts/MiniCPM-o-4_5-tts-transformer-F16.gguf";
}

/// Path to TTS weights model (custom tensors: emb_code, emb_text, head_code)
inline std::string path_model_tts_weights()
{
	return path_model_root() + "/tts/MiniCPM-o-4_5-tts-weights-F16.gguf";
}

/// Path to Token2Wav vocoder directory
inline std::string path_token2wav_dir()
{
	return path_model_root() + "/token2wav-gguf";
}

///@}

/**
 * @name Prompt Template Examples
 * @brief Common prompt patterns for different modalities
 *
 * These examples show the correct format for constructing prompts.
 * All prompts should follow the pattern:
 *   ROLE_USER + [multimodal_marker] + question + ROLE_TRANSITION
 *
 * @{
 */

/**
 * @brief Vision-only prompt template
 *
 * Format: " lda\n<image>What shape is in this image? lda\nassistant\n"
 *
 * Steps:
 * 1. Start with ROLE_USER (" lda\n")
 * 2. Insert TOKEN_IMAGE ("<image>") as placeholder
 * 3. Add your question text
 * 4. End with ROLE_TRANSITION (" lda\nassistant\n")
 *
 * During inference:
 * - Tokenize the prefix (before <image>)
 * - Submit image embeddings (replacing <image>)
 * - Tokenize the suffix (after <image> to assistant\n)
 * - Generate response
 */
constexpr char const * EXAMPLE_PROMPT_VISION =
	" lda\n<image>What shape is in this image? lda\nassistant\n";

/**
 * @brief Audio-only prompt template
 *
 * Format: " lda\n<|audio|>What does the person say? lda\nassistant\n"
 *
 * Steps:
 * 1. Start with ROLE_USER (" lda\n")
 * 2. Insert TOKEN_AUDIO ("<|audio|>") as placeholder
 * 3. Add your question text
 * 4. End with ROLE_TRANSITION (" lda\nassistant\n")
 *
 * During inference:
 * - Tokenize the prefix (before <|audio|>)
 * - Tokenize the audio marker separately
 * - Submit audio embeddings
 * - Tokenize the suffix (after <|audio|> to assistant\n)
 * - Generate response
 */
constexpr char const * EXAMPLE_PROMPT_AUDIO =
	" lda\n<|audio|>What does the person say in this audio? lda\nassistant\n";

/**
 * @brief Multimodal prompt template (vision + audio)
 *
 * Format: " lda\n<image><|audio|>Describe what you see and hear. lda\nassistant\n"
 *
 * Steps:
 * 1. Start with ROLE_USER (" lda\n")
 * 2. Insert TOKEN_IMAGE ("<image>")
 * 3. Insert TOKEN_AUDIO ("<|audio|>")
 * 4. Add your question text
 * 5. End with ROLE_TRANSITION (" lda\nassistant\n")
 *
 * During inference:
 * - Tokenize the prefix (before <image>)
 * - Submit image embeddings
 * - Tokenize middle part (between <image> and <|audio|>)
 * - Submit audio embeddings
 * - Tokenize the suffix (after <|audio|> to assistant\n)
 * - Generate response
 */
constexpr char const * EXAMPLE_PROMPT_MULTIMODAL =
	" lda\n<image><|audio|>Describe what you see and hear. lda\nassistant\n";

///@}

/**
 * @name Token Utility Functions
 * @{
 */

/// Check if token is a special multimodal token
inline bool is_multimodal_token(int token_id)
{
	return (token_id >= 151646 && token_id <= 151696) || (token_id >= 151697 && token_id <= 151747);
}

/// Check if token marks multimodal content start
inline bool is_multimodal_start(int token_id)
{
	switch (token_id)
	{
		case TOKEN_IMAGE_ID:
		case TOKEN_AUDIO_START_ID:
		case TOKEN_AUDIO_ID:
		case TOKEN_SPK_BOS_ID:
		case TOKEN_TTS_BOS_ID:
		case TOKEN_LISTEN_ID:
		case TOKEN_SPEAK_ID:
		case TOKEN_VAD_START_ID:
		case TOKEN_EMOTION_START_ID:
		case TOKEN_SPEED_START_ID:
		case TOKEN_PITCH_START_ID:
		case TOKEN_TURN_BOS_ID:
		case TOKEN_CHUNK_BOS_ID:
		case TOKEN_CHUNK_TTS_BOS_ID:
			return true;
		default:
			return false;
	}
}

/// Check if token marks multimodal content end
inline bool is_multimodal_end(int token_id)
{
	switch (token_id)
	{
		case TOKEN_IMAGE_END_ID:
		case TOKEN_AUDIO_END_ID:
		case TOKEN_SPK_END_ID:
		case TOKEN_TTS_END_ID:
		case TOKEN_INTERRUPT_ID:
		case TOKEN_VAD_END_ID:
		case TOKEN_EMOTION_END_ID:
		case TOKEN_SPEED_END_ID:
		case TOKEN_PITCH_END_ID:
		case TOKEN_TURN_END_ID:
		case TOKEN_CHUNK_END_ID:
		case TOKEN_CHUNK_TTS_END_ID:
			return true;
		default:
			return false;
	}
}

/// Get timbre token string by ID
inline char const * get_timbre_token(int timbre_id)
{
	switch (timbre_id)
	{
		case 7:
			return TOKEN_TIMBRE_7;
		case 8:
			return TOKEN_TIMBRE_8;
		case 9:
			return TOKEN_TIMBRE_9;
		case 10:
			return TOKEN_TIMBRE_10;
		case 11:
			return TOKEN_TIMBRE_11;
		case 12:
			return TOKEN_TIMBRE_12;
		case 13:
			return TOKEN_TIMBRE_13;
		case 14:
			return TOKEN_TIMBRE_14;
		case 15:
			return TOKEN_TIMBRE_15;
		case 16:
			return TOKEN_TIMBRE_16;
		case 17:
			return TOKEN_TIMBRE_17;
		case 18:
			return TOKEN_TIMBRE_18;
		case 19:
			return TOKEN_TIMBRE_19;
		case 20:
			return TOKEN_TIMBRE_20;
		case 21:
			return TOKEN_TIMBRE_21;
		case 22:
			return TOKEN_TIMBRE_22;
		case 23:
			return TOKEN_TIMBRE_23;
		case 24:
			return TOKEN_TIMBRE_24;
		case 25:
			return TOKEN_TIMBRE_25;
		case 26:
			return TOKEN_TIMBRE_26;
		case 27:
			return TOKEN_TIMBRE_27;
		case 28:
			return TOKEN_TIMBRE_28;
		case 29:
			return TOKEN_TIMBRE_29;
		case 30:
			return TOKEN_TIMBRE_30;
		case 31:
			return TOKEN_TIMBRE_31;
		default:
			return nullptr;
	}
}

/// Get timbre token ID by timbre number (7-31)
inline int get_timbre_token_id(int timbre_id)
{
	switch (timbre_id)
	{
		case 7:
			return TOKEN_TIMBRE_7_ID;
		case 8:
			return TOKEN_TIMBRE_8_ID;
		case 9:
			return TOKEN_TIMBRE_9_ID;
		case 10:
			return TOKEN_TIMBRE_10_ID;
		case 11:
			return TOKEN_TIMBRE_11_ID;
		case 12:
			return TOKEN_TIMBRE_12_ID;
		case 13:
			return TOKEN_TIMBRE_13_ID;
		case 14:
			return TOKEN_TIMBRE_14_ID;
		case 15:
			return TOKEN_TIMBRE_15_ID;
		case 16:
			return TOKEN_TIMBRE_16_ID;
		case 17:
			return TOKEN_TIMBRE_17_ID;
		case 18:
			return TOKEN_TIMBRE_18_ID;
		case 19:
			return TOKEN_TIMBRE_19_ID;
		case 20:
			return TOKEN_TIMBRE_20_ID;
		case 21:
			return TOKEN_TIMBRE_21_ID;
		case 22:
			return TOKEN_TIMBRE_22_ID;
		case 23:
			return TOKEN_TIMBRE_23_ID;
		case 24:
			return TOKEN_TIMBRE_24_ID;
		case 25:
			return TOKEN_TIMBRE_25_ID;
		case 26:
			return TOKEN_TIMBRE_26_ID;
		case 27:
			return TOKEN_TIMBRE_27_ID;
		case 28:
			return TOKEN_TIMBRE_28_ID;
		case 29:
			return TOKEN_TIMBRE_29_ID;
		case 30:
			return TOKEN_TIMBRE_30_ID;
		case 31:
			return TOKEN_TIMBRE_31_ID;
		default:
			return -1;
	}
}

///@}