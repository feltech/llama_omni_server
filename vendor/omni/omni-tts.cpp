#include "omni-tts.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "omni-impl.h"

#include <algorithm>
#include <cmath>
#include <common/common.h>
#include <cstring>
#include <random>

// ============================================================================
// GLOBAL LOGGER STATE
// ============================================================================

/*
 * omni_logger_state: Global logger configuration for TTS operations
 *
 * Purpose: Centralized logging configuration for TTS-related operations
 * across the omni-tts module. Provides consistent error reporting and debugging.
 *
 * Fields:
 * - verbosity_thold: Minimum log level threshold
 *   - GGML_LOG_LEVEL_ERROR (0): Only critical errors
 *   - GGML_LOG_LEVEL_WARN (1): Warnings and errors
 *   - GGML_LOG_LEVEL_INFO (2): Informational messages
 *   - GGML_LOG_LEVEL_DEBUG (3): Debug messages (default)
 *
 * - log_callback: Function pointer for custom log output
 *   - Default: omni_log_callback_default() from omni-impl.h
 *   - Receives log level and message string
 *
 * - log_callback_user_data: User data passed to log_callback
 *   - Can be NULL or point to application-specific context
 *
 * Usage:
 * - Set verbosity_thold to control log verbosity
 * - Override log_callback for custom logging behavior
 * - Access via g_logger_state in TTS operations
 *
 * References:
 * - llama.cpp: ggml_log_callback
 * - omni-impl.h: omni_log_callback_default()
 */
// Weak definition: when linked with audition.cpp (which also defines g_logger_state
// with external linkage), the linker uses audition.cpp's strong definition.
__attribute__((weak)) struct omni_logger_state g_logger_state = {
    .verbosity_thold = GGML_LOG_LEVEL_ERROR,
    .log_callback = omni_log_callback_default,
    .log_callback_user_data = nullptr
};

namespace omni {
namespace tts {

/*
 * is_valid_tts_token: Check if a token is valid for TTS processing
 *
 * Purpose: Filter out special tokens that should not be included in TTS audio generation.
 * These tokens are special markers (e.g., <|im_start|>, <|im_end|>) or out-of-range tokens.
 *
 * Functionality:
 * 1. Check against known special token IDs (llama.cpp-omni internal tokens)
 * 2. Check if token >= 150000 (likely special token range)
 * 3. Return true if token is valid, false if it should be filtered
 *
 * Valid TTS tokens should be in range:
 * - 0-151686: Regular text tokens
 * - 151687: audio_bos token (valid, but handled specially)
 * - 151688-152063: Regular text tokens
 * - 152064-152065: TTS special tokens (valid)
 * - 152066-152067: TTS special tokens (valid)
 * - 151692: text_eos token (valid, duplex mode only)
 *
 * Invalid tokens (filtered out):
 * - 151667: <|im_end|>
 * - 151668: <|im_start|>
 * - 151704: <|im_end|>
 * - 151706: <|im_start|>
 * - 151705: <|im_end|>
 * - 151718: <|im_start|>
 * - 151721: <|im_end|>
 * - 151717: <|im_start|>
 * - 271: Special token
 * - >= 150000: Likely special token (LLM special tokens)
 *
 * References:
 * - llama.cpp-omni/tools/omni/omni.cpp: Lines 4540-4538 (filter_special_tokens)
 * - AGENTS.md: TTS special tokens documentation
 *
 * Algorithm:
 * 1. Iterate through known special token IDs
 * 2. If token matches any special ID, return false (invalid)
 * 3. If token >= 150000, return false (likely special token)
 * 4. Otherwise, return true (valid token)
 *
 * Mathematical Notation:
 * - S = {151667, 151668, 151704, 151706, 151705, 151718, 151721, 151717, 271} (special IDs)
 * - T = {t | t ≥ 150000} (special token range)
 * - is_valid(t) = !(t ∈ S) && !(t ∈ T)
 *
 * Complexity: O(1) - constant time lookup against 9 special tokens
 *
 * Example Usage:
 *   llama_token token = 151687;  // audio_bos
 *   if (is_valid_tts_token(token)) {
 *       // Token is valid, proceed with TTS processing
 *   }
 *
 * Note: This function is used in filter_special_tokens() to filter both token IDs
 * and hidden states before TTS conditioning.
 */
bool is_valid_tts_token(llama_token tid) {
    // Known special token IDs that should be filtered out
    static const std::vector<llama_token> g_special_token_ids = {
        151667, 151668, 151704, 151706, 151705, 151718, 151721, 151717, 271
    };
    // Check if token matches any known special ID
    for (llama_token sid : g_special_token_ids) if (tid == sid) return false;
    // Check if token is in the special token range (>= 150000)
    if (tid >= 150000) return false;
    // Token is valid for TTS processing
    return true;
}

/*
 * filter_special_tokens: Filter special tokens from token IDs and corresponding hidden states
 *
 * Purpose: Remove special tokens (e.g., <|im_start|>, <|im_end|>) and their associated
 * hidden states before TTS processing. These tokens are not part of the spoken text
 * and should not be converted to audio.
 *
 * Functionality:
 * 1. Validate alignment between token_ids and hidden_states
 * 2. Iterate through tokens
 * 3. Check if token is valid using is_valid_tts_token()
 * 4. If valid, copy both token and hidden state to filtered vectors
 * 5. If invalid, skip (do not copy)
 * 6. Replace original vectors with filtered versions
 *
 * Alignment Check:
 * - hidden_states.size() must equal token_ids.size() * n_embd
 * - Each token has exactly n_embd hidden state dimensions
 * - If alignment is broken, function returns without updating (safety check)
 *
 * Algorithm:
 * 1. if (hidden_states.size() != token_ids.size() * n_embd) return (alignment check)
 * 2. filtered_token_ids.clear(), filtered_hidden_states.clear()
 * 3. for i in range(0, token_ids.size()):
 *       if is_valid_tts_token(token_ids[i]):
 *           filtered_token_ids.append(token_ids[i])
 *           for j in range(0, n_embd):
 *               filtered_hidden_states.append(hidden_states[i * n_embd + j])
 * 4. token_ids = filtered_token_ids
 * 5. hidden_states = filtered_hidden_states
 *
 * Complexity:
 * - Time: O(n_tokens * n_embd) - linear in the number of tokens and embedding dimensions
 * - Space: O(n_tokens * n_embd) - filtered vectors of same size as input
 *
 * Mathematical Notation:
 * - S = {t | is_valid_tts_token(t) = false} (special tokens to filter)
 * - hidden_states = [h_0, h_1, ..., h_{n-1}], where h_i ∈ ℝ^n_embd
 * - token_ids = [t_0, t_1, ..., t_{n-1}]
 * - filtered_token_ids = [t_i | t_i ∉ S]
 * - filtered_hidden_states = [h_i | t_i ∉ S]
 *
 * Example Usage:
 *   std::vector<llama_token> tokens = {151667, 100, 151668, 101};
 *   std::vector<float> hidden_states = {...};  // 4 * n_embd floats
 *   filter_special_tokens(tokens, hidden_states, n_embd);
 *   // tokens = {100, 101}
 *   // hidden_states = [hidden_state_for_100, hidden_state_for_101]
 *
 * References:
 * - llama.cpp-omni/tools/omni/omni.cpp: Lines 4540-4538 (filter_special_tokens)
 * - omni-tts.cpp: is_valid_tts_token() (lines 24-31)
 * - AGENTS.md: TTS special tokens documentation
 *
 * Safety Considerations:
 * - Alignment check prevents silent failures due to mismatched dimensions
 * - If alignment is broken, function returns without updating (safety check)
 * - This ensures we never process mismatched token-hidden state pairs
 *
 * Note: This function operates on copies of vectors, so original vectors are not modified
 * until the final assignment. This prevents partial updates in case of early return.
 */
void filter_special_tokens(std::vector<llama_token>& token_ids, std::vector<float>& hidden_states, int n_embd) {
    // Validate alignment between token IDs and hidden states
    // Each token should have exactly n_embd hidden state dimensions
    if (hidden_states.size() != token_ids.size() * n_embd) return;

    // Initialize filtered vectors
    std::vector<llama_token> filtered_token_ids;
    std::vector<float> filtered_hidden_states;

    // Iterate through tokens and filter out special tokens
    for (size_t i = 0; i < token_ids.size(); ++i) {
        // Check if token is valid for TTS processing
        if (!is_valid_tts_token(token_ids[i])) continue;

        // Copy valid token
        filtered_token_ids.push_back(token_ids[i]);

        // Copy corresponding hidden state (n_embd dimensions)
        for (int j = 0; j < n_embd; ++j) {
            filtered_hidden_states.push_back(hidden_states[i * n_embd + j]);
        }
    }

    // Replace original vectors with filtered versions
    token_ids = filtered_token_ids;
    hidden_states = filtered_hidden_states;
}

bool init_tts_context_with_model(TTSContext& tts_ctx, llama_model* model, llama_context* ctx, const llama_vocab* vocab) {
    tts_ctx.model_llm = model;
    tts_ctx.ctx_llm = ctx;
    tts_ctx.vocab = vocab;
    return true;
}

/*
 * load_projector_semantic: Load projector semantic weights from GGUF file
 *
 * Purpose: Load the projector semantic weights from a GGUF file and initialize
 * the ProjectorSemanticWeights structure. The projector maps LLM hidden states
 * (4096-d) to TTS embeddings (768-d) using a two-layer MLP.
 *
 * Functionality:
 * 1. Load GGUF file using gguf_init_from_file()
 * 2. Initialize backend for tensor allocation (GPU or CPU)
 * 3. Create computation graph context
 * 4. Extract and create tensor structures for each weight
 * 5. Allocate tensors to backend memory
 * 6. Load weight data from file to backend tensors
 * 7. Handle transposition if needed
 * 8. Free GGUF context and return initialized model
 *
 * GGUF File Structure:
 * - File path: path to projector GGUF file
 * - Contains: 4 tensors (linear1.weight, linear1.bias, linear2.weight, linear2.bias)
 * - Format: F16 (16-bit floating point) or F32 (32-bit floating point)
 * - Cannot be loaded by standard llama.cpp (not recognized as transformer tensors)
 *
 * Projector Weights:
 * 1. linear1.weight (4096 × 768):
 *    - Purpose: First projection matrix
 *    - Formula: h_proj = W_1 @ h^LLM + b_1
 *    - Access: h_proj[j] = sum(W_1[j][i] * h^LLM[i])
 *    - Layout: [4096 rows of 768 cols] = standard row-major
 *    - Transposition: No transpose needed (standard row-major)
 *
 * 2. linear1.bias (768):
 *    - Purpose: First bias vector
 *    - Adds to output of linear1: h_proj = W_1 @ h^LLM + b_1
 *    - Layout: 768-dimensional vector
 *    - Transposition: No transpose needed (1D vector)
 *
 * 3. linear2.weight (768 × 768):
 *    - Purpose: Second projection matrix
 *    - Formula: h_TTS = W_2 @ h_proj + b_2
 *    - Access: h_TTS[j] = sum(W_2[j][i] * h_proj[i])
 *    - Layout: [768 rows of 768 cols] = standard row-major
 *    - Transposition: No transpose needed (standard row-major)
 *
 * 4. linear2.bias (768):
 *    - Purpose: Second bias vector
 *    - Adds to output of linear2: h_TTS = W_2 @ h_proj + b_2
 *    - Layout: 768-dimensional vector
 *    - Transposition: No transpose needed (1D vector)
 *
 * Backend Initialization:
 * - ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL):
 *   - Attempts to initialize GPU backend (CUDA)
 *   - Returns NULL if GPU unavailable
 * - ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL):
 *   - Falls back to CPU backend if GPU unavailable
 * - If both fail, returns error
 *
 * Tensor Allocation:
 * - ggml_new_tensor_2d(ctx, type, 4096, 768): Creates 2D tensor
 *   - type: GGML_TYPE_F32 or GGML_TYPE_F16
 *   - 4096: Input dimension (LLM hidden size)
 *   - 768: Output dimension (TTS hidden size)
 * - ggml_new_tensor_1d(ctx, type, 768): Creates 1D tensor
 *   - type: GGML_TYPE_F32 or GGML_TYPE_F16
 *   - 768: Dimension (bias vector)
 * - ggml_set_name(tensor, name): Assigns tensor name for debugging
 * - ggml_backend_alloc_ctx_tensors(ctx, backend): Allocates tensor data to backend
 *   - Allocates memory on GPU or CPU based on backend type
 *   - Returns buffer handle
 *
 * Weight Loading from File:
 * - fseek(f, offset, SEEK_SET): Seeks to tensor offset in GGUF file
 *   - offset = gguf_get_data_offset(model.ctx_gguf) + gguf_get_tensor_offset(model.ctx_gguf, i)
 *   - gguf_get_data_offset(): Base offset of tensor data in file
 *   - gguf_get_tensor_offset(): Offset of specific tensor in data section
 * - fread(data_f32.data(), sizeof(float), ne, f): Reads tensor data from file
 *   - data_f32: Buffer to store float data
 *   - sizeof(float): Size of each float (4 bytes)
 *   - ne: Number of elements to read
 *   - f: File pointer
 * - If F16: Convert to F32 using ggml_fp16_to_fp32()
 * - ggml_backend_tensor_set(t_model, data_f32.data(), 0, ne * sizeof(float)):
 *   - Uploads weight data to backend tensor
 *   - t_model: Model tensor (allocated to backend)
 *   - data_f32.data(): Source data
 *   - 0: Offset in source data
 *   - ne * sizeof(float): Number of bytes to copy
 *
 * Transposition Handling:
 * - Check: if (ggml_n_dims(t_gguf) == 2 && (t_gguf->ne[0] != t_model->ne[0] || t_gguf->ne[1] != t_model->ne[1]))
 * - If transposition needed:
 *   - PyTorch stores as (out, in): t_gguf->ne[0] = out, t_gguf->ne[1] = in
 *   - CPU needs (in, out): t_model->ne[0] = in, t_model->ne[1] = out
 *   - Transpose: transposed[c * out_dim + r] = data_f32[r * in_dim + c]
 *   - Upload transposed data to backend
 *
 * Algorithm:
 * 1. params = { .no_alloc = false, .ctx = &model.ctx_meta }
 * 2. model.ctx_gguf = gguf_init_from_file(gguf_path, params)
 * 3. model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL)
 * 4. if (!model.backend): model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL)
 * 5. if (!model.backend): { gguf_free(model.ctx_gguf); return false }
 * 6. n_tensors = gguf_get_n_tensors(model.ctx_gguf)
 * 7. ctx_size = ggml_tensor_overhead() * n_tensors
 * 8. ctx_params = { .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true }
 * 9. model.ctx_w = ggml_init(ctx_params)
 * 10. for i in range(0, n_tensors):
 *       a. name = gguf_get_tensor_name(model.ctx_gguf, i)
 *       b. type = gguf_get_tensor_type(model.ctx_gguf, i)
 *       c. Create tensor: linear1.weight (4096 × 768), linear1.bias (768), etc.
 *       d. Assign tensor to appropriate layer
 *       e. ggml_set_name(tensor, name)
 * 11. model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend)
 * 12. f = fopen(gguf_path, "rb")
 * 13. for i in range(0, n_tensors):
 *       a. name = gguf_get_tensor_name(model.ctx_gguf, i)
 *       b. t_gguf = ggml_get_tensor(model.ctx_meta, name)
 *       c. t_model = model.layer.linear1_weight (or linear2, bias)
 *       d. offset = gguf_get_data_offset(model.ctx_gguf) + gguf_get_tensor_offset(model.ctx_gguf, i)
 *       e. fseek(f, offset, SEEK_SET)
 *       f. Read data from file (F32 or F16)
 *       g. Transpose if needed
 *       h. ggml_backend_tensor_set(t_model, data_f32.data(), 0, ne * sizeof(float))
 * 14. fclose(f)
 * 15. gguf_free(model.ctx_gguf)
 * 16. model.initialized = true
 * 17. return true
 *
 * Complexity:
 * - Time: O(n_tensors * total_elements) - linear in total number of elements
 * - Space: O(total_elements) - memory for tensors and data
 *
 * Mathematical Notation:
 * - W_1 ∈ ℝ^(768 × 4096): First projection matrix
 * - b_1 ∈ ℝ^768: First bias vector
 * - W_2 ∈ ℝ^(768 × 768): Second projection matrix
 * - b_2 ∈ ℝ^768: Second bias vector
 * - h_proj = ReLU(W_1 @ h^LLM + b_1)
 * - h_TTS = W_2 @ h_proj + b_2
 *
 * Example Usage:
 *   ProjectorSemanticWeights model;
 *   bool success = load_projector_semantic("/path/to/projector.gguf", model);
 *   if (success) {
 *       // model.layer.linear1_weight, linear2_weight, biases are loaded
 *       // model.initialized = true
 *       // Ready to apply projector to hidden states
 *   }
 *
 * References:
 * - omni-tts.cpp: Lines 53-124 (load_projector_semantic implementation)
 * - AGENTS.md: Projector architecture documentation
 * - HYBRID_SOLUTION.md: Projector usage guide
 *
 * Error Handling:
 * - Returns false if GGUF file cannot be opened
 * - Returns false if backend initialization fails
 * - Returns false if tensor creation fails
 * - Returns false if weight loading fails
 * - Prints error messages to stderr
 *
 * Safety Checks:
 * - Validates GGUF file path
 * - Validates backend initialization
 * - Validates tensor creation
 * - Validates weight loading
 * - Handles transposition correctly
 *
 * Note: The backend allocation is critical for performance:
 * - GPU backend: Much faster for large models (4096 × 768 matrices)
 * - CPU backend: Slower but works on all platforms
 * - If GPU unavailable, falls back to CPU
 *
 * Note: Transposition is critical for linear layers:
 * - PyTorch stores as (out, in): [768, 4096] = [4096 rows of 768 cols]
 * - CPU needs (in, out): [4096, 768] = [768 rows of 4096 cols]
 * - Without transposition, matrix multiplication would be incorrect
 *
 * Note: This function assumes the GGUF file contains only the 4 projector tensors.
 * If the file contains other tensors, they will be ignored.
 *
 * Note: The computation graph is built separately in apply_projector_semantic().
 * This function only loads the weights.
 */
bool load_projector_semantic(const char* gguf_path, ProjectorSemanticWeights& model) {
    struct gguf_init_params params = { .no_alloc = false, .ctx = &model.ctx_meta };
    model.ctx_gguf = gguf_init_from_file(gguf_path, params);
    if (!model.ctx_gguf) return false;

    // Initialize backend for tensor allocation (GPU or CPU)
    // Attempts to use GPU backend first, falls back to CPU if unavailable
    model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (!model.backend) model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!model.backend) { gguf_free(model.ctx_gguf); return false; }

    const int64_t n_tensors = gguf_get_n_tensors(model.ctx_gguf);
    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    struct ggml_init_params ctx_params = { .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true };
    model.ctx_w = ggml_init(ctx_params);

    // Create tensor structures for each weight
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(model.ctx_gguf, i);
        enum ggml_type type = gguf_get_tensor_type(model.ctx_gguf, i);
        struct ggml_tensor* tensor = nullptr;
        
        // Create appropriate tensor based on tensor name
        if (strcmp(name, "linear1.weight") == 0) {
            // linear1.weight: 4096 × 768 matrix (first projection)
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 4096, 768);
            model.layer.linear1_weight = tensor;
        } else if (strcmp(name, "linear1.bias") == 0) {
            // linear1.bias: 768-d bias vector
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear1_bias = tensor;
        } else if (strcmp(name, "linear2.weight") == 0) {
            // linear2.weight: 768 × 768 matrix (second projection)
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 768, 768);
            model.layer.linear2_weight = tensor;
        } else if (strcmp(name, "linear2.bias") == 0) {
            // linear2.bias: 768-d bias vector
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear2_bias = tensor;
        }
        
        // Assign tensor name for debugging
        if (tensor) ggml_set_name(tensor, name);
    }

    // Allocate tensors to backend memory (GPU or CPU)
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);

    // Load weight data from GGUF file to backend tensors
    FILE* f = fopen(gguf_path, "rb");
    if (f) {
        for (int64_t i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(model.ctx_gguf, i);
            struct ggml_tensor* t_gguf = ggml_get_tensor(model.ctx_meta, name);
            struct ggml_tensor* t_model = nullptr;
            
            // Find corresponding model tensor
            if (strcmp(name, "linear1.weight") == 0) t_model = model.layer.linear1_weight;
            else if (strcmp(name, "linear1.bias") == 0) t_model = model.layer.linear1_bias;
            else if (strcmp(name, "linear2.weight") == 0) t_model = model.layer.linear2_weight;
            else if (strcmp(name, "linear2.bias") == 0) t_model = model.layer.linear2_bias;
            
            // Skip if tensor not found
            if (!t_model) continue;
            
            // Calculate offset in file
            size_t offset = gguf_get_data_offset(model.ctx_gguf) + gguf_get_tensor_offset(model.ctx_gguf, i);
            fseek(f, offset, SEEK_SET);
            
            // Read tensor data from file
            int64_t ne = ggml_nelements(t_gguf);
            std::vector<float> data_f32(ne);
            if (t_gguf->type == GGML_TYPE_F32) {
                // Direct read for F32
                if (fread(data_f32.data(), sizeof(float), ne, f) != (size_t)ne) continue;
            } else if (t_gguf->type == GGML_TYPE_F16) {
                // Convert F16 to F32
                std::vector<ggml_fp16_t> data_f16(ne);
                if (fread(data_f16.data(), sizeof(ggml_fp16_t), ne, f) != (size_t)ne) continue;
                for (int64_t j = 0; j < ne; ++j) data_f32[j] = ggml_fp16_to_fp32(data_f16[j]);
            }

            // Handle transposition if needed
            // Transposition is required because PyTorch stores as (out, in) but CPU needs (in, out)
            if (ggml_n_dims(t_gguf) == 2 && (t_gguf->ne[0] != t_model->ne[0] || t_gguf->ne[1] != t_model->ne[1])) {
                // Transpose: t_gguf is [out, in], t_model is [in, out]
                int64_t out_dim = t_gguf->ne[0];
                int64_t in_dim = t_gguf->ne[1];
                std::vector<float> transposed(ne);
                for (int64_t r = 0; r < out_dim; ++r) {
                    for (int64_t c = 0; c < in_dim; ++c) {
                        transposed[c * out_dim + r] = data_f32[r * in_dim + c];
                    }
                }
                ggml_backend_tensor_set(t_model, transposed.data(), 0, ne * sizeof(float));
            } else {
                // No transpose needed: upload data directly
                ggml_backend_tensor_set(t_model, data_f32.data(), 0, ne * sizeof(float));
            }
        }
        fclose(f);
    }
    gguf_free(model.ctx_gguf);
    model.initialized = true;
    return true;
}

/*
 * projector_build_graph: Build computation graph for projector layer
 *
 * Purpose: Construct a computation graph for the two-layer MLP projector
 * that transforms LLM hidden states (4096-d) to TTS embeddings (768-d).
 *
 * Functionality:
 * 1. Create computation graph context
 * 2. Build projection layers:
 *    - Linear1: hidden = W_1 @ input + b_1
 *    - ReLU activation: hidden = ReLU(hidden)
 *    - Linear2: output = W_2 @ hidden + b_2
 * 3. Build graph and return
 *
 * Projector Architecture:
 * - Input: h^LLM ∈ ℝ^(4096 × n_tokens) (LLM hidden states)
 * - Linear1: h_proj = ReLU(W_1 @ h^LLM + b_1)
 *   - W_1 ∈ ℝ^(768 × 4096): First projection matrix
 *   - b_1 ∈ ℝ^768: First bias vector
 *   - ReLU: Rectified Linear Unit (max(0, x))
 * - Linear2: h_TTS = W_2 @ h_proj + b_2
 *   - W_2 ∈ ℝ^(768 × 768): Second projection matrix
 *   - b_2 ∈ ℝ^768: Second bias vector
 * - Output: h_TTS ∈ ℝ^(768 × n_tokens) (TTS hidden states)
 *
 * Computation Graph:
 * - ggml_new_graph(ctx): Creates empty computation graph
 * - ggml_mul_mat(ctx, W_1, input): Matrix multiplication
 *   - Computes: hidden = W_1 @ input
 *   - W_1: 768 × 4096 matrix
 *   - input: 4096 × n_tokens tensor
 *   - output: 768 × n_tokens tensor
 * - ggml_add(ctx, hidden, b_1): Add bias
 *   - Computes: hidden = hidden + b_1
 *   - b_1: 768 × 1 bias vector
 *   - output: 768 × n_tokens tensor
 * - ggml_relu(ctx, hidden): Apply ReLU activation
 *   - Computes: hidden = max(0, hidden)
 *   - output: 768 × n_tokens tensor
 * - ggml_mul_mat(ctx, W_2, hidden): Matrix multiplication
 *   - Computes: output = W_2 @ hidden
 *   - W_2: 768 × 768 matrix
 *   - hidden: 768 × n_tokens tensor
 *   - output: 768 × n_tokens tensor
 * - ggml_add(ctx, output, b_2): Add bias
 *   - Computes: output = output + b_2
 *   - b_2: 768 × 1 bias vector
 *   - output: 768 × n_tokens tensor
 * - ggml_build_forward_expand(gf, output): Build graph with output as root
 *
 * Algorithm:
 * 1. gf = ggml_new_graph(ctx)
 * 2. hidden = ggml_mul_mat(W_1, input)  // W_1 @ input
 * 3. hidden = ggml_add(hidden, b_1)      // + b_1
 * 4. hidden = ggml_relu(hidden)          // ReLU(max(0, x))
 * 5. output = ggml_mul_mat(W_2, hidden)  // W_2 @ hidden
 * 6. output = ggml_add(output, b_2)      // + b_2
 * 7. ggml_build_forward_expand(gf, output)
 * 8. return gf
 *
 * Complexity:
 * - Time: O(n_tokens * in_dim * out_dim) - linear in number of tokens
 * - Space: O(n_tokens * (in_dim + out_dim)) - temporary buffers
 *
 * Mathematical Notation:
 * - input = h^LLM ∈ ℝ^(4096 × n_tokens): LLM hidden states (input)
 * - W_1 ∈ ℝ^(768 × 4096): First projection matrix
 * - b_1 ∈ ℝ^768: First bias vector
 * - h_proj = W_1 @ input + b_1 ∈ ℝ^(768 × n_tokens)
 * - h_proj = ReLU(h_proj) ∈ ℝ^(768 × n_tokens)
 * - W_2 ∈ ℝ^(768 × 768): Second projection matrix
 * - b_2 ∈ ℝ^768: Second bias vector
 * - output = W_2 @ h_proj + b_2 ∈ ℝ^(768 × n_tokens): Final output
 *
 * Example Usage:
 *   ProjectorSemanticWeights model;
 *   model.layer.linear1_weight = ...;  // 768 × 4096
 *   model.layer.linear2_weight = ...;  // 768 × 768
 *   struct ggml_context* ctx = ggml_init({size, buffer, no_alloc});
 *   struct ggml_tensor* input = ggml_new_tensor_2d(ctx, F32, 4096, 10);
 *   struct ggml_cgraph* gf = projector_build_graph(model, ctx, input);
 *   // gf now contains computation graph for projector layer
 *
 * References:
 * - omni-tts.cpp: Lines 126-135 (projector_build_graph implementation)
 * - omni-tts.cpp: apply_projector_semantic() (lines 161-176)
 * - AGENTS.md: Projector architecture documentation
 *
 * Error Handling:
 * - Assumes model is properly initialized
 * - Assumes input tensor is valid
 * - Assumes model weights are loaded
 * - No error checking for tensor allocation
 *
 * Note: This function creates a computation graph, not executes it.
 * The graph is executed by ggml_backend_graph_compute() in apply_projector_semantic().
 *
 * Note: The computation graph is reusable for multiple input batches.
 * Only the input tensor data changes between calls.
 *
 * Note: The graph structure is fixed (linear1 → ReLU → linear2).
 * This allows efficient execution via ggml_backend_graph_compute().
 */
static struct ggml_cgraph* projector_build_graph(const ProjectorSemanticWeights& model, struct ggml_context* ctx, struct ggml_tensor* input) {
    // Create computation graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    
    // Linear1: hidden = W_1 @ input + b_1
    struct ggml_tensor* hidden = ggml_mul_mat(ctx, model.layer.linear1_weight, input);
    hidden = ggml_add(ctx, hidden, model.layer.linear1_bias);
    
    // ReLU activation: hidden = max(0, hidden)
    hidden = ggml_relu(ctx, hidden);
    
    // Linear2: output = W_2 @ hidden + b_2
    struct ggml_tensor* output = ggml_mul_mat(ctx, model.layer.linear2_weight, hidden);
    output = ggml_add(ctx, output, model.layer.linear2_bias);
    
    // Build forward computation graph with output as root
    ggml_build_forward_expand(gf, output);
    
    return gf;
}

/*
 * normalize_l2_per_token: Normalize vectors to unit L2 norm
 *
 * Purpose: Scale each embedding vector to have unit L2 norm (length 1).
 * This is a critical preprocessing step for TTS to ensure consistent scale
 * across all tokens, improving numerical stability and gradient flow.
 *
 * Functionality:
 * 1. For each token position t from 0 to n_tokens-1:
 *    a. Extract vector v = embeddings[t * n_embd : (t+1) * n_embd]
 *    b. Compute squared norm: norm_sq = sum(v_i²)
 *    c. Compute norm: norm = sqrt(norm_sq + eps)
 *    d. If norm > 0: scale vector by 1/norm
 *    e. If norm = 0: set to random unit vector (fallback)
 *
 * L2 Normalization Formula:
 * - L2_norm(x) = x / ||x||_2
 * - ||x||_2 = sqrt(sum(x_i²))
 * - eps: Small epsilon to prevent division by zero
 *
 * Why L2 Normalization is Critical:
 * 1. Ensures consistent scale across all tokens
 *    - Without normalization: Some tokens may dominate, others may be too small
 *    - With normalization: All tokens have consistent magnitude
 *
 * 2. Improves numerical stability
 *    - Prevents overflow/underflow from large/small values
 *    - Helps with gradient flow during training
 *    - Reduces variance in dot product computations
 *
 * 3. Standard practice in deep learning
 *    - Common in attention mechanisms (e.g., scaled dot-product attention)
 *    - Used in embedding spaces for consistency
 *    - Helps with distance-based comparisons
 *
 * Algorithm Details:
 * - Iterates over tokens sequentially: O(n_tokens)
 * - For each token, computes dot product of its components: O(n_embd)
 * - Overall complexity: O(n_tokens * n_embd)
 * - Space complexity: O(1) - operates in-place on input array
 *
 * Edge Cases:
 * - If norm = 0 (vector is zero), set to random unit vector
 *   This shouldn't happen in practice (model outputs are non-zero)
 *   Fallback ensures numerical stability
 * - If n_tokens = 0, function returns immediately
 * - If n_embd = 0, function returns immediately
 * - If eps = 0 and vector is zero, division by zero would occur
 *   So eps should be small but non-zero (e.g., 1e-12)
 *
 * Mathematical Notation:
 * - v ∈ ℝ^n_embd: Input vector
 * - ||v||_2 = sqrt(∑_{i=0}^{n_embd-1} v_i²): L2 norm
 * - v_norm = v / ||v||_2: Normalized vector
 * - eps: Small constant to prevent division by zero
 *
 * Example Usage:
 *   float embeddings[3 * 768] = {
 *       1.0f, 2.0f, 3.0f,  // Token 0 (norm = sqrt(14) ≈ 3.74)
 *       0.5f, 0.5f, 0.5f,  // Token 1 (norm = sqrt(0.75) ≈ 0.87)
 *       0.0f, 0.0f, 0.0f   // Token 2 (norm = 0)
 *   };
 *   normalize_l2_per_token(embeddings, 3, 768, 1e-12);
 *   // After normalization:
 *   // Token 0: [0.267, 0.534, 0.802] (norm = 1)
 *   // Token 1: [0.577, 0.577, 0.577] (norm = 1)
 *   // Token 2: [0.577, 0.577, 0.577] (norm = 1, fallback)
 *
 * References:
 * - AGENTS.md: L2 normalization in TTS conditioning
 * - omni-tts.cpp: Lines 137-159 (normalize_l2_per_token implementation)
 * - PyTorch: F.normalize()
 *
 * Importance in TTS Pipeline:
 * - Applied to projector outputs (4096-d → 768-d)
 * - Ensures all tokens have consistent scale in TTS space
 * - Critical for stable audio generation
 * - Without normalization: Audio quality degrades (inconsistent volume, artifacts)
 *
 * Debugging:
 * - Verify norm > 0 for all tokens (should always be true)
 * - Check that all normalized vectors have norm ≈ 1
 * - Compare with Python implementation: F.normalize(projector(hidden), p=2)
 *
 * Note: This function operates in-place on the input array.
 * The input array is modified to contain normalized vectors.
 * No new array is allocated.
 *
 * Note: The eps parameter should be small (e.g., 1e-12) to prevent
 * division by zero while maintaining numerical stability.
 */
void normalize_l2_per_token(float* embeddings, int n_tokens, int n_embd, float eps) {
    // Iterate over each token position
    for (int t = 0; t < n_tokens; t++) {
        // Extract vector for token t
        float* vec = embeddings + t * n_embd;
        
        // Compute squared norm of the vector
        // ||v||² = ∑(v_i²)
        float norm_sq = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            norm_sq += vec[i] * vec[i];
        }
        
        // PyTorch F.normalize formula:
        // norm = sqrt(sum(x²) + eps), then x = x / norm
        // The eps parameter prevents division by zero when norm is very small
        float norm = std::sqrt(norm_sq + eps);
        
        // Normalize the vector to unit length
        if (norm > 0.0f) {
            // Standard case: scale vector by 1/norm
            float inv_norm = 1.0f / norm;
            for (int i = 0; i < n_embd; i++) {
                vec[i] *= inv_norm;
            }
        } else {
            // Edge case: norm is zero (vector is all zeros)
            // This shouldn't happen in practice, but we handle it for safety
            // Set to random unit vector (fallback)
            float inv_sqrt_n = 1.0f / std::sqrt((float)n_embd);
            for (int i = 0; i < n_embd; i++) {
                vec[i] = inv_sqrt_n;
            }
        }
    }
}

/*
 * apply_projector_semantic: Apply projector layer to hidden states
 *
 * Purpose: Apply the two-layer MLP projector to transform LLM hidden states
 * from 4096-d to 768-d, matching the TTS model's input dimension.
 *
 * Functionality:
 * 1. Validate projector is initialized
 * 2. Create temporary GGML context for computation
 * 3. Create input tensor with shape [in_dim, n_tokens]
 * 4. Build computation graph: input → linear1 → ReLU → linear2
 * 5. Allocate tensors to backend memory (GPU/CPU)
 * 6. Upload input hidden states to tensor
 * 7. Run computation via ggml_backend_graph_compute()
 * 8. Extract output hidden states
 * 9. Free temporary context
 *
 * Projector Architecture:
 * - Input: h^LLM ∈ ℝ^(4096 × n_tokens) (LLM hidden states)
 * - Linear1: h_proj = ReLU(W_1 @ h^LLM + b_1)
 *   - W_1 ∈ ℝ^(768 × 4096): First projection matrix
 *   - b_1 ∈ ℝ^768: First bias vector
 *   - ReLU: Rectified Linear Unit (max(0, x))
 * - Linear2: h_TTS = W_2 @ h_proj + b_2
 *   - W_2 ∈ ℝ^(768 × 768): Second projection matrix
 *   - b_2 ∈ ℝ^768: Second bias vector
 * - Output: h_TTS ∈ ℝ^(768 × n_tokens) (TTS hidden states)
 *
 * Implementation Details:
 * - Uses ggml_new_tensor_2d() to create tensor with shape [in_dim, n_tokens]
 *   where in_dim is 4096 (LLM hidden size) and n_tokens is the number of tokens
 * - Uses ggml_set_input() to mark tensor as input (not output)
 * - Uses projector_build_graph() to build computation graph
 *   - Creates linear1 layer: h_proj = ReLU(W_1 @ h + b_1)
 *   - Creates linear2 layer: h_TTS = W_2 @ h_proj + b_2
 *   - Uses ggml_mul_mat() for matrix multiplication
 *   - Uses ggml_add() for bias addition
 *   - Uses ggml_relu() for ReLU activation
 * - Uses ggml_backend_alloc_ctx_tensors() to allocate tensors to backend
 *   - Attempts GPU backend first (if available)
 *   - Falls back to CPU backend if GPU unavailable
 * - Uses ggml_backend_tensor_set() to upload input data to tensor
 * - Uses ggml_backend_graph_compute() to run computation
 * - Uses ggml_backend_tensor_get() to extract output data
 * - Uses ggml_backend_buffer_free() to free backend buffer
 * - Uses ggml_free() to free temporary context
 *
 * Algorithm:
 * 1. if (!model.initialized): return (nothing to do)
 * 2. ctx = ggml_init({mem_size, mem_buffer, no_alloc=true})
 * 3. input = ggml_new_tensor_2d(ctx, F32, in_dim, n_tokens)
 * 4. ggml_set_input(input)
 * 5. gf = projector_build_graph(model, ctx, input)
 *    - linear1 = ggml_mul_mat(W_1, input) + b_1
 *    - linear1 = ggml_relu(linear1)
 *    - output = ggml_mul_mat(W_2, linear1) + b_2
 * 6. buf_compute = ggml_backend_alloc_ctx_tensors(ctx, model.backend)
 * 7. ggml_backend_tensor_set(input, input_data, 0, size)
 * 8. ggml_backend_graph_compute(model.backend, gf)
 * 9. output_tensor = ggml_graph_node(gf, num_nodes - 1)
 * 10. result.resize(n_tokens * out_dim)
 * 11. ggml_backend_tensor_get(output_tensor, result, 0, size)
 * 12. ggml_backend_buffer_free(buf_compute)
 * 13. ggml_free(ctx)
 *
 * Complexity:
 * - Time: O(n_tokens * in_dim * out_dim) - linear in number of tokens and dimensions
 * - Space: O(n_tokens * (in_dim + out_dim)) - temporary buffers
 *
 * Mathematical Notation:
 * - h^LLM ∈ ℝ^(4096 × n_tokens): LLM hidden states (input)
 * - W_1 ∈ ℝ^(768 × 4096): First projection matrix
 * - b_1 ∈ ℝ^768: First bias vector
 * - h_proj = ReLU(W_1 @ h^LLM + b_1) ∈ ℝ^(768 × n_tokens)
 * - W_2 ∈ ℝ^(768 × 768): Second projection matrix
 * - b_2 ∈ ℝ^768: Second bias vector
 * - h_TTS = W_2 @ h_proj + b_2 ∈ ℝ^(768 × n_tokens) (output)
 *
 * Example Usage:
 *   ProjectorSemanticWeights model;
 *   model.initialized = true;
 *   model.backend = gpu_backend;
 *   model.layer.linear1_weight = ...;  // 4096 × 768
 *   model.layer.linear2_weight = ...;  // 768 × 768
 *   std::vector<float> input(4096 * 10);  // 10 tokens, 4096-d each
 *   std::vector<float> output(768 * 10);  // 10 tokens, 768-d each
 *   apply_projector_semantic(model, input, 10, output);
 *   // output now contains projected hidden states
 *
 * References:
 * - omni-tts.cpp: Lines 161-176 (apply_projector_semantic implementation)
 * - omni-tts.cpp: Lines 126-135 (projector_build_graph helper)
 * - AGENTS.md: Projector architecture documentation
 *
 * Error Handling:
 * - Returns immediately if model not initialized
 * - No error checking for tensor allocation (assumes successful)
 * - No error checking for backend allocation (assumes successful)
 *
 * Note: This function uses temporary GGML context and buffers.
 * All resources are freed before return.
 *
 * Note: The projector is applied to each token independently:
 * - Each token's hidden state is processed through the same MLP
 * - No cross-token interaction (independent processing)
 * - Output is just a linear transformation of input
 *
 * Note: Backend allocation is critical for performance:
 * - GPU backend: Much faster for large models
 * - CPU backend: Slower but works on all platforms
 * - If GPU unavailable, falls back to CPU
 *
 * Note: Computation graph is built once and reused:
 * - Graph structure is fixed (linear1 → ReLU → linear2)
 * - Only input data changes between calls
 * - Efficient for multiple token batches
 */
void apply_projector_semantic(const ProjectorSemanticWeights& model, const std::vector<float>& input_data, int n_tokens, std::vector<float>& result) {
    if (!model.initialized) return;
    struct ggml_init_params params = { .mem_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead(), .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context* ctx = ggml_init(params);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.in_dim, n_tokens);
    ggml_set_input(input);
    struct ggml_cgraph* gf = projector_build_graph(model, ctx, input);
    struct ggml_backend_buffer* buf_compute = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    ggml_backend_tensor_set(input, input_data.data(), 0, n_tokens * model.hparams.in_dim * sizeof(float));
    ggml_backend_graph_compute(model.backend, gf);
    struct ggml_tensor* output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    result.resize(n_tokens * model.hparams.out_dim);
    ggml_backend_tensor_get(output, result.data(), 0, n_tokens * model.hparams.out_dim * sizeof(float));
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx);
}

/*
 * tts_emb_text: Look up text embedding for a given token ID
 *
 * Purpose: Retrieve the TTS text embedding for a specific token ID from the
 * emb_text_weight matrix. This embedding provides semantic identity information
 * in the TTS space, which is critical for generating intelligible speech.
 *
 * Functionality:
 * 1. Validate that emb_text_weight matrix is loaded
 * 2. Validate token_id is within valid range [0, emb_text_vocab_size)
 * 3. Extract embedding from emb_text_weight matrix using token_id as row index
 * 4. Copy embedding to output buffer (emb_size dimensions)
 *
 * Matrix Structure:
 * - emb_text.weight ∈ ℝ^(768 × 152064)
 * - Stored as row-major array: emb_text_weight[152064 * 768]
 * - Access pattern: embedding[j] = emb_text_weight[token_id * 768 + j]
 * - Each token ID maps to a 768-dimensional embedding vector
 *
 * Why emb_text is Critical:
 * - Without emb_text: TTS produces gibberish (no phoneme/semantic identity)
 * - With emb_text: TTS knows what language/phonemes to generate
 * - Provides token-specific identity in TTS semantic space
 * - Combines with projector's contextual information for final conditioning
 *
 * Token ID Mapping:
 * - 0-151686: Regular text tokens (alphabet, punctuation, etc.)
 * - 151687: audio_bos token (start-of-speech marker)
 * - 151688-152063: Regular text tokens
 * - 152064-152065: TTS special tokens
 * - 151692: text_eos token (end-of-text marker, duplex mode)
 * - 152066-152067: TTS special tokens
 *
 * Mathematical Notation:
 * - emb_text.weight ∈ ℝ^(768 × V), where V = 152064 (text vocabulary size)
 * - token_id ∈ {0, 1, ..., V-1}: Token ID to lookup
 * - e = emb_text.weight[token_id] ∈ ℝ^768: Text embedding
 * - embedding_out = e: Output buffer
 *
 * Algorithm:
 * 1. if (tts_ctx.emb_text_weight == NULL): return false (error)
 * 2. if (token_id < 0 || token_id >= emb_text_vocab_size): return false (error)
 * 3. for j in range(0, emb_size):
 *       embedding_out[j] = emb_text_weight[token_id * 768 + j]
 * 4. return true
 *
 * Complexity:
 * - Time: O(emb_size) - linear in embedding dimension (768)
 * - Space: O(1) - output buffer provided by caller
 *
 * Example Usage:
 *   TTSContext tts_ctx;
 *   float embedding[768];
 *   llama_token token_id = 1000;
 *   if (tts_emb_text(tts_ctx, token_id, embedding, 768)) {
 *       // embedding now contains the 768-d text embedding for token_id
 *   }
 *
 * References:
 * - llama.cpp-omni/tools/omni/omni.cpp: Lines 5838-5841 (emb_text lookup in condition)
 * - AGENTS.md: TTS architecture and emb_text importance
 * - HYBRID_SOLUTION.md: Projector usage guide
 *
 * Error Handling:
 * - Returns false if emb_text_weight is NULL (not loaded)
 * - Returns false if token_id is out of range
 * - Prints error message to stderr in case of failure
 *
 * Note: This function assumes emb_text_weight is stored in row-major order
 * (standard for GGUF files). The matrix is accessed as:
 *   embedding[j] = emb_text_weight[token_id * 768 + j]
 * This means token_id is the row index, and j is the column index.
 *
 * Alternative Access Patterns:
 * - Some libraries store embeddings in column-major order
 * - In that case, access would be: embedding[j] = emb_text_weight[j * vocab_size + token_id]
 * - GGUF uses row-major, so our access pattern is correct
 *
 * Debugging:
 * - Check emb_text_vocab_size to ensure token_id is valid
 * - Verify embedding dimensions (should be 768 for MiniCPM-o)
 * - Compare with Python implementation: emb_text.weight[token_id]
 */
bool tts_emb_text(const TTSContext& tts_ctx, llama_token token_id, float* embedding_out, int emb_size) {
    // Validate that emb_text_weight matrix is loaded
    if (!tts_ctx.emb_text_weight) {
        std::cerr << "  ✗ tts_emb_text: emb_text_weight is NULL" << std::endl;
        return false;
    }
    // Validate token_id is within valid range [0, emb_text_vocab_size)
    // emb_text_vocab_size should be 152064 (text vocabulary size)
    if (token_id < 0 || token_id >= tts_ctx.emb_text_vocab_size) {
        std::cerr << "  ✗ tts_emb_text: token_id " << token_id << " out of range [0, " << tts_ctx.emb_text_vocab_size << ")" << std::endl;
        return false;
    }
    // Extract embedding from emb_text_weight matrix
    // Matrix structure: [152064 rows of 768 columns] = 768 × 152064
    // Access pattern: embedding[j] = emb_text_weight[token_id * 768 + j]
    // This means token_id is the row index, and j is the column index
    for (int i = 0; i < emb_size; i++) {
        embedding_out[i] = tts_ctx.emb_text_weight[token_id * 768 + i];
    }
    return true;
}

/*
 * load_tts_weights: Load TTS-specific weights from GGUF files
 *
 * Purpose: Load all TTS-specific embeddings and weights from the tts-weights GGUF file.
 * This includes emb_code, emb_text, head_code, and initializes Token2Wav models.
 * The projector weights are loaded separately via load_projector_semantic().
 *
 * Functionality:
 * 1. Load projector weights (already done before calling this function)
 * 2. Load tts-weights GGUF file using gguf_init_from_file()
 * 3. Extract and load TTS-specific tensors:
 *    - emb_code.0.weight: Audio token embeddings (768 × 6562)
 *    - emb_text.weight: Text token embeddings (768 × 152064)
 *    - head_code.0.weight: Audio token logits (6562 × 768, transposed)
 * 4. Initialize Token2Wav models from directory
 * 5. Free GGUF context
 *
 * GGUF File Structure:
 * - tts-weights-F16.gguf: Contains 11 custom TTS tensors
 * - Cannot be loaded by standard llama.cpp (not recognized as transformer tensors)
 * - Must be loaded via gguf_init_from_file() and manual tensor extraction
 *
 * TTS-Specific Tensors:
 * 1. emb_code.0.weight (768 × 6562):
 *    - Purpose: Convert audio token IDs (0-6561) to embeddings
 *    - Access: embedding[j] = emb_code[relative_idx * 768 + j]
 *    - Layout: [6562 rows of 768 cols] = standard row-major
 *
 * 2. emb_text.weight (768 × 152064):
 *    - Purpose: Provide semantic identity in TTS space
 *    - Access: embedding[j] = emb_text[text_token_id * 768 + j]
 *    - Layout: [152064 rows of 768 cols] = standard row-major
 *    - Critical: Without emb_text, TTS produces gibberish
 *
 * 3. head_code.0.weight (6562 × 768):
 *    - Purpose: Project TTS hidden states to audio token logits
 *    - Access: logits[i] = hidden_state @ head_code_weight[i]
 *    - Layout: [768 rows of 6562 cols] = transposed for CPU efficiency
 *    - Transposition: PyTorch stores as (out, in), but CPU needs (in, out)
 *
 * Transposition Handling:
 * - PyTorch stores linear layers as (out_features, in_features)
 * - For efficient CPU inference with ggml_mul_mat, transpose to (in_features, out_features)
 * - Check: if (t_gguf->ne[0] != t_model->ne[0] || t_gguf->ne[1] != t_model->ne[1])
 * - Transposed access: embedding[j] = emb_code[j * vocab_size + relative_idx]
 *
 * Token2Wav Initialization:
 * - Loads encoder, flow_matching, flow_extra, prompt_cache, hifigan models
 * - Initializes diffusion model for audio generation
 * - Uses prompt_cache for efficient sampling (n_timesteps = 5)
 * - If initialization fails, TTS can still work (audio generation will produce silence)
 *
 * Algorithm:
 * 1. Call load_projector_semantic() (already done before this function)
 * 2. Load GGUF file: ctx_weights_gguf = gguf_init_from_file(tts_weights_path, params)
 * 3. Define lambda load_tensor_data() to extract named tensor
 * 4. Load emb_code.0.weight (no transpose)
 * 5. Load emb_text.weight (no transpose)
 * 6. Load head_code.0.weight (transpose: 6562×768 → 768×6562)
 * 7. Free GGUF context: gguf_free(ctx_weights_gguf)
 * 8. Initialize Token2Wav models (if directory provided)
 * 9. Return true if all tensors loaded successfully
 *
 * Complexity:
 * - Time: O(total_tensor_elements) - linear in total number of elements
 * - Space: O(total_tensor_elements) - allocated for each tensor
 *
 * Mathematical Notation:
 * - W_emb_code ∈ ℝ^(768 × 6562): Audio code embedding matrix
 * - W_emb_text ∈ ℝ^(768 × 152064): Text embedding matrix
 * - W_head_code ∈ ℝ^(6562 × 768): Head code matrix (transposed)
 * - emb_code(token) = W_emb_code[token] ∈ ℝ^768
 * - emb_text(token) = W_emb_text[token] ∈ ℝ^768
 * - logits = hidden_state @ W_head_code
 *
 * Example Usage:
 *   TTSContext tts_ctx;
 *   bool success = load_tts_weights(tts_ctx,
 *       "/path/to/projector.gguf",
 *       "/path/to/tts-transformer.gguf",
 *       "/path/to/tts-weights.gguf",
 *       "/path/to/token2wav");
 *   if (success) {
 *       // TTS weights loaded successfully
 *       // tts_ctx.emb_code_weight, tts_ctx.emb_text_weight, tts_ctx.head_code_weight are ready
 *   }
 *
 * References:
 * - llama.cpp-omni/tools/omni/omni.cpp: Lines 3635-3646 (TTS weights loading)
 * - omni-tts.cpp: load_projector_semantic() (lines 53-124)
 * - AGENTS.md: TTS weights documentation
 *
 * Error Handling:
 * - Returns false if GGUF file cannot be opened
 * - Returns false if any tensor cannot be loaded
 * - Prints error messages to stderr for debugging
 * - Token2Wav initialization failure is non-fatal (audio generation produces silence)
 *
 * Note: The transformer GGUF is loaded by the caller via llama_model_load_from_file()
 * in test_tts.cpp. This function only loads the custom TTS weights.
 *
 * Note: Transposition is critical for head_code because:
 * - PyTorch stores as (6562, 768) = [768 rows of 6562 cols]
 * - CPU sampling needs (768, 6562) = [6562 rows of 768 cols] for efficient matrix multiplication
 * - Transposed access: logits[i] = hidden_state[j] * head_code_weight[i * 768 + j]
 *
 * Note: emb_code and emb_text use no transpose because they are embeddings (not linear layers)
 * - Embeddings are accessed as lookup tables, not used in matrix multiplication
 * - Standard row-major layout is optimal for embedding lookup
 */
bool load_tts_weights(TTSContext& tts_ctx, const char* projector_gguf,
                      const char* tts_transformer_path, const char* tts_weights_path,
                      const char* token2wav_model_dir, const char* device_token2mel, const char* device_vocoder, const char* custom_prompt_cache_path) {
    // Load projector weights (already done before calling this function)
    if (!load_projector_semantic(projector_gguf, tts_ctx.projector_weights)) return false;

    // Open the weights GGUF file containing TTS-specific tensors
    // This file contains 11 custom tensors that are not recognized by standard llama.cpp
    // The file is produced by utils/split_tts_gguf.py to separate custom tensors from standard transformer tensors
    struct ggml_context* ctx_weights_meta = nullptr;
    struct gguf_init_params weights_params = { .no_alloc = false, .ctx = &ctx_weights_meta };
    struct gguf_context* ctx_weights_gguf = gguf_init_from_file(tts_weights_path, weights_params);
    if (!ctx_weights_gguf) {
        std::cerr << "  ✗ load_tts_weights: failed to open weights GGUF: " << tts_weights_path << std::endl;
        return false;
    }

    // Helper lambda to extract a named tensor from GGUF and copy to malloc'd float array
    // Purpose: Simplify tensor extraction with automatic transposition handling
    // d0/d1 receive ne[0]/ne[1] of the tensor as stored in the file (GGUF layout)
    // When transpose=true, the layout is transposed for CPU dot-product loops
    // CPU loops use [row * hidden_size + col] indexing, which expects transposed layout
    auto load_tensor_data = [&](const char* name, float*& target_ptr, int& d0, int& d1, bool transpose = false) {
        struct ggml_tensor* t = ggml_get_tensor(ctx_weights_meta, name);
        if (!t) return false;
        int64_t n = ggml_nelements(t);
        target_ptr = (float*)malloc(n * sizeof(float));
        d0 = (int)t->ne[0]; d1 = (ggml_n_dims(t) > 1) ? (int)t->ne[1] : 1;
        
        std::vector<float> tmp(n);
        if (t->type == GGML_TYPE_F32) memcpy(tmp.data(), t->data, n * sizeof(float));
        else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t* s = (const ggml_fp16_t*)t->data;
            for (int64_t j = 0; j < n; ++j) tmp[j] = ggml_fp16_to_fp32(s[j]);
        }
        
        // Final layout handling:
        // - ne[0] is columns (inner dimension), ne[1] is rows (outer dimension)
        // - Access pattern: target_ptr[row_idx * ne[0] + col_idx]
        if (transpose && ggml_n_dims(t) == 2) {
            // Transpose for CPU efficiency: (ne[1], ne[0]) → (ne[0], ne[1])
            int64_t rows = t->ne[1];  // Original rows
            int64_t cols = t->ne[0];  // Original columns
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < cols; ++c) {
                    target_ptr[c * rows + r] = tmp[r * cols + c];
                }
            }
            d0 = (int)rows; d1 = (int)cols;
        } else {
            // No transpose needed: copy directly
            memcpy(target_ptr, tmp.data(), n * sizeof(float));
        }
        return true;
    };

    // Load emb_code.0.weight (audio token embeddings)
    // Matrix: [768 × 6562] = [6562 rows of 768 cols]
    // This maps audio token IDs (0-6561) to 768-dimensional embeddings
    // No transpose needed (embeddings are accessed as lookup tables, not used in matrix multiplication)
    load_tensor_data("emb_code.0.weight", tts_ctx.emb_code_weight, tts_ctx.emb_code_hidden_size, tts_ctx.emb_code_vocab_size, false);
    if (tts_ctx.emb_code_weight) std::cout << "  ✓ Loaded emb_code.0.weight: " << tts_ctx.emb_code_hidden_size << "x" << tts_ctx.emb_code_vocab_size << std::endl;
    else std::cerr << "  ✗ Failed to load emb_code.0.weight" << std::endl;

    // Load emb_text.weight (text token embeddings for TTS)
    // Matrix: [768 × 152064] = [152064 rows of 768 cols]
    // This maps text token IDs (0-151686) to 768-dimensional embeddings
    // Critical for TTS: provides semantic identity in TTS space
    // Without emb_text, TTS produces gibberish (no phoneme/identity information)
    load_tensor_data("emb_text.weight", tts_ctx.emb_text_weight, tts_ctx.emb_text_hidden_size, tts_ctx.emb_text_vocab_size, false);
    if (tts_ctx.emb_text_weight) std::cout << "  ✓ Loaded emb_text.weight: " << tts_ctx.emb_text_hidden_size << "x" << tts_ctx.emb_text_vocab_size << std::endl;
    else std::cerr << "  ✗ Failed to load emb_text.weight" << std::endl;
    
    // Load head_code.0.weight (audio token logits)
    // Matrix: [6562 × 768] = [768 rows of 6562 cols] (transposed)
    // This maps 768-d TTS hidden states to 6562-d audio token logits
    // Transpose is required because:
    // - PyTorch stores as (6562, 768) = [768 rows of 6562 cols]
    // - CPU sampling needs (768, 6562) = [6562 rows of 768 cols] for efficient matrix multiplication
    // - Access pattern: logits[i] = hidden_state[j] * head_code_weight[i * 768 + j]
    load_tensor_data("head_code.0.weight", tts_ctx.head_code_weight, tts_ctx.head_code_hidden_size, tts_ctx.head_code_num_audio_tokens, true);
    if (tts_ctx.head_code_weight) std::cout << "  ✓ Loaded head_code.0.weight (transposed): " << tts_ctx.head_code_hidden_size << "x" << tts_ctx.head_code_num_audio_tokens << std::endl;
    else std::cerr << "  ✗ Failed to load head_code.0.weight" << std::endl;

    // Done with the weights GGUF file — release it
    gguf_free(ctx_weights_gguf);
    ggml_free(ctx_weights_meta);

    // The transformer GGUF is loaded by the caller via libllama (llama_model_load_from_file)
    // This function only loads the custom TTS weights, not the transformer architecture
    // tts_transformer_path is unused in this function (passed for reference)
    (void)tts_transformer_path;

    // Initialize Token2Wav models for audio waveform generation
    // These models are not part of the TTS model itself, but are used to convert audio tokens to audio
    if (token2wav_model_dir) {
        tts_ctx.token2wav_session = std::make_shared<omni::flow::Token2WavSession>();
        std::string model_dir = token2wav_model_dir;
        // Initialize Token2Wav with all required models
        // encoder.gguf: Conformer encoder for feature extraction
        // flow_matching.gguf: Conditional Flow Matching diffusion model
        // flow_extra.gguf: Additional flow matching components
        // prompt_cache.gguf: Pre-computed prompt cache for efficient sampling
        // hifigan2.gguf: HiFi-GAN vocoder for waveform generation
        bool ok = tts_ctx.token2wav_session->init_from_prompt_cache_gguf(
            model_dir + "/encoder.gguf",
            model_dir + "/flow_matching.gguf",
            model_dir + "/flow_extra.gguf",
            custom_prompt_cache_path == nullptr ? model_dir + "/prompt_cache.gguf": custom_prompt_cache_path,
            model_dir + "/hifigan2.gguf",
            device_token2mel, device_vocoder
        );
        if (ok) {
            tts_ctx.token2wav_initialized = true;
            std::cout << "  ✓ Token2Wav initialized" << std::endl;
        } else {
            std::cerr << "  ✗ Token2Wav initialization failed" << std::endl;
        }
    }

    return true;
}


// generate_audio_tokens_local_simplex and generate_audio_tokens were removed.
// They used tts_transformer_forward (the manual ggml pass), which is superseded
// by the libllama-based inference loop in test_tts.cpp.

/*
 * generate_audio_from_tokens: Convert audio tokens to audio waveform using Token2Wav
 *
 * Purpose: Take a sequence of audio tokens (0-6561) and convert them to audio samples
 * using the Token2Wav diffusion model pipeline. This is the final step in the TTS pipeline,
 * transforming discrete audio tokens into a continuous audio waveform.
 *
 * Functionality:
 * 1. Validate Token2Wav is initialized
 * 2. Convert absolute audio token IDs to relative IDs (0-6561)
 * 3. Add 3 silence tokens (4218) to ensure proper start
 * 4. Process tokens in sliding windows (28 tokens, step 25)
 * 5. Feed each window to Token2Wav encoder, flow matching, and HiFi-GAN
 * 6. Accumulate audio samples into output buffer
 *
 * Audio Token Handling:
 * - Audio tokens are in absolute ID range: 151687 + relative_idx
 * - Convert to relative: relative_idx = absolute - 151687
 * - Silence tokens: 4218 (add 3 for proper start)
 * - eos token: 6561 (relative index)
 *
 * Sliding Window Logic:
 * - WINDOW_SIZE = 28 tokens: Token2Wav needs 28 tokens to output audio
 * - STEP_SIZE = 25 tokens: Move by 25, keep 3 for lookahead
 * - Always adds 3 silence tokens before each window
 * - Flush remaining tokens when window size ≤ 28
 *
 * Token2Wav Pipeline:
 * 1. Encoder (Conformer): Extracts features from audio tokens
 * 2. Flow Matching (CFM): Generates diffusion steps
 * 3. HiFi-GAN Vocoder: Converts diffusion steps to waveform
 *
 * Algorithm:
 * 1. if (!token2wav_initialized): return false (generate silence)
 * 2. Convert tokens to relative IDs (0-6561)
 * 3. Add 3 silence tokens (4218) to buffer
 * 4. While buffer not empty:
 *    a. window = first 28 tokens
 *    b. is_last_window = (buffer.size() ≤ 28)
 *    c. Feed window to Token2Wav
 *    d. Append chunk_wav to audio_samples
 *    e. If is_last_window: clear buffer
 *    f. Else: remove first 25 tokens (keep 3)
 * 5. Return true
 *
 * Complexity:
 * - Time: O(n_tokens) - linear in number of audio tokens
 * - Space: O(audio_samples) - output buffer
 *
 * Mathematical Notation:
 * - tokens: Absolute audio token IDs ∈ {151687, ..., 151687+6561}
 * - tokens_rel: Relative audio token IDs ∈ {0, 1, ..., 6561}
 * - silence = 4218: Silence token
 * - window: Sequence of 28 tokens
 * - audio_samples: Output waveform samples ∈ ℝ
 *
 * Example Usage:
 *   std::vector<int32_t> audio_tokens = {151688, 151689, ..., 151687+6561};
 *   std::vector<float> audio_samples;
 *   bool success = generate_audio_from_tokens(tts_ctx, audio_tokens, audio_samples, 24000);
 *   if (success) {
 *       // audio_samples contains 24kHz mono audio
 *   }
 *
 * References:
 * - llama.cpp-omni/tools/omni/omni.cpp: Lines 667-730 (Token2Wav integration)
 * - omni.cpp: Lines 4544-4890 (TTS simplex implementation)
 * - token2wav-impl.cpp: Flow matching and HiFi-GAN implementations
 * - AGENTS.md: Token2Wav documentation
 *
 * Error Handling:
 * - Returns false if Token2Wav not initialized
 * - Returns false if feed_window fails
 * - Prints error messages to stderr
 * - If Token2Wav fails, returns silence (error handling)
 *
 * Note: The sliding window ensures smooth audio output:
 * - Each window produces a chunk of audio
 * - Overlap ensures no gaps between chunks
 * - Lookahead tokens maintain context
 *
 * Note: Adding 3 silence tokens is critical:
 * - Token2Wav needs at least 25 tokens to produce meaningful audio
 * - 3 silence tokens provide padding for the first window
 * - Ensures proper start of audio generation
 *
 * Note: Sample rate is 24000 Hz (24kHz mono)
 * - Standard for TTS (matches MiniCPM-o specification)
 * - Output is float samples in [-1.0, 1.0] range
 * - Convert to int16 for WAV file: int16 = float * 32767
 */
bool generate_audio_from_tokens(TTSContext& tts_ctx, const std::vector<int32_t>& audio_tokens, std::vector<float>& audio_samples, int sample_rate) {
    if (!tts_ctx.token2wav_initialized || !tts_ctx.token2wav_session) {
        std::cerr << "  ✗ Token2Wav not initialized, generating silence" << std::endl;
        audio_samples.resize(sample_rate); 
        return true;
    }

    // Convert absolute audio token IDs to relative IDs (0-6561)
    // Audio tokens are in range [151687, 151687+6561]
    // Convert to range [0, 6561] for Token2Wav
    std::vector<int32_t> relative_tokens;
    for (int32_t t : audio_tokens) {
        // Tokens are typically in range 151687+ relative_idx
        // Convert to relative: relative_idx = t - 151687
        // If token is already in relative range (unlikely), keep as-is
        if (t >= 151687) relative_tokens.push_back(t - 151687);
        else relative_tokens.push_back(t);
    }

    // Add 3 silence tokens (4218) as the Python implementation does
    // Silence tokens provide padding for Token2Wav to produce meaningful audio
    std::vector<int32_t> token_buffer = {4218, 4218, 4218};
    token_buffer.insert(token_buffer.begin(), relative_tokens.begin(), relative_tokens.end());

    // 🚀 Sliding window logic from omni.cpp
    // Token2Wav needs 28 tokens to output audio (25 chunk + 3 lookahead)
    // We process windows of 28 tokens and move by 25 tokens.
    // This ensures smooth audio output with minimal overlap.
    constexpr size_t WINDOW_SIZE = 28;
    constexpr size_t STEP_SIZE = 25;
    
    // Check if the Token2Wav model is initialized with correct n_timesteps
    // Based on previous findings, n_timesteps=5 is expected for prompt_cache.gguf
    // This controls the number of diffusion steps for audio generation

    // Process tokens in sliding windows
    while (!token_buffer.empty()) {
        // Determine how many tokens to process in this window
        size_t process_size = std::min(token_buffer.size(), WINDOW_SIZE);
        
        // is_last_window is true if we have 28 or fewer tokens and we want to flush
        // In omni.cpp, is_last_window (is_final) triggers a full flush
        bool is_last_window = (token_buffer.size() <= WINDOW_SIZE);
        
        // Extract window of tokens
        std::vector<int32_t> window(token_buffer.begin(), token_buffer.begin() + process_size);
        std::vector<float> chunk_wav;
        
        // Debug output for the first few windows
        if (token_buffer.size() > relative_tokens.size()) {
             // This is the first window (includes silence tokens)
        }

        // Feed window to Token2Wav pipeline
        if (tts_ctx.token2wav_session->feed_window(window, is_last_window, chunk_wav)) {
            // Accumulate audio samples from this chunk
            if (!chunk_wav.empty()) {
                audio_samples.insert(audio_samples.end(), chunk_wav.begin(), chunk_wav.end());
            }
        } else {
            std::cerr << "  ✗ Token2Wav feed_window failed" << std::endl;
            return false;
        }
        
        // Update token buffer for next iteration
        if (is_last_window) {
            // Last window: clear all remaining tokens
            token_buffer.clear();
        } else {
            // Remove 25 tokens, keep 3 for the next window's lookahead
            // This creates overlap between windows for smooth audio
            token_buffer.erase(token_buffer.begin(), token_buffer.begin() + STEP_SIZE);
        }
    }

    return true;
}
/*
 * prefill_with_emb_tts: Feed condition embeddings into TTS model
 *
 * Purpose: Feed sequence of embeddings into the TTS model instead of tokens.
 * This updates the TTS KV cache with condition embeddings, which are then used
 * to generate audio tokens. This is critical for TTS conditioning.
 *
 * Functionality:
 * 1. Validate input parameters (n_pos, n_embd, models loaded)
 * 2. Save condition embeddings for first audio token re-forward
 * 3. Split embeddings into batches of size n_batch
 * 4. For each batch:
 *    a. Create batch with embeddings (not tokens)
 *    b. Set position IDs for each embedding
 *    c. Enable embeddings output
 *    d. Call llama_decode() to update KV cache
 *    e. Extract hidden states (for debugging)
 * 5. Update n_past_tts counter
 *
 * Condition Embeddings:
 * - Input: embeddings array [n_pos * n_embd floats]
 * - Format: Flat array of 768-d embeddings (for MiniCPM-o)
 * - Contains: emb_text(token) + L2_norm(projector(hidden_state))
 * - Purpose: Condition TTS model for audio generation
 *
 * Position IDs:
 * - Python: pos_ids = torch.arange(text_start_pos, text_start_pos + n_pos)
 * - C++: batch.pos[j] = text_start_pos + i + j (where i is batch offset)
 * - Purpose: Ensure correct RoPE (Rotary Positional Embedding) computation
 * - Ensures embeddings are placed at correct positions in KV cache
 *
 * KV Cache Management:
 * - First call: Clears KV cache (if tts_condition_saved = false)
 * - Subsequent calls: Appends to KV cache
 * - This allows re-forwarding condition for first audio token
 * - Matches Python's behavior: past_key_values=None on first forward
 *
 * Batch Processing:
 * - Embeddings are split into batches of size n_batch
 * - llama_decode() processes each batch sequentially
 * - Allows processing long sequences without memory overflow
 * - n_batch should match context's n_batch (typically 4096)
 *
 * Hidden State Extraction (for debugging):
 * - llama_get_embeddings_ith(ctx, -1): Last token's hidden state
 * - llama_get_embeddings_ith(ctx, -2): Second-to-last token's hidden state
 * - llama_get_embeddings_ith(ctx, i): i-th token's hidden state
 * - Purpose: Extract TTS hidden states for analysis
 * - Used in: sample_tts_token() for head_code logits
 *
 * Algorithm:
 * 1. if (n_pos <= 0): return false (error)
 * 2. if (n_pos > 10000): return false (error, likely corruption)
 * 3. if (!model_llm || !model_tts): return false (error)
 * 4. n_embd = llama_model_n_embd(model)
 * 5. if (n_embd <= 0 || n_embd > 10000): return false (error)
 * 6. if (n_pos > INT_MAX / n_embd): return false (overflow check)
 * 7. if (!tts_condition_saved && n_pos > 0):
 *      - Save embeddings to tts_condition_embeddings
 *      - Set tts_condition_length = n_pos
 *      - Set tts_condition_n_embd = n_embd
 *      - Set tts_condition_saved = true
 * 8. text_start_pos = *n_past_tts
 * 9. for i in range(0, n_pos, n_batch):
 *      a. n_eval = min(n_batch, n_pos - i)
 *      b. batch.n_tokens = n_eval
 *      c. batch.embd = embeddings[i * n_embd : (i + n_eval) * n_embd]
 *      d. batch.pos = [text_start_pos + i + j for j in range(n_eval)]
 *      e. llama_set_embeddings(ctx_llm, true)
 *      f. llama_decode(ctx_llm, batch)
 *      g. Extract hidden states (if debugging)
 * 10. *n_past_tts = text_start_pos + n_pos
 * 11. return true
 *
 * Complexity:
 * - Time: O(n_pos * n_batch) - linear in number of embeddings
 * - Space: O(n_batch * n_embd) - batch buffer
 *
 * Mathematical Notation:
 * - embed ∈ ℝ^(n_pos × n_embd): Condition embeddings
 * - batch.embd ∈ ℝ^(n_eval × n_embd): Batch embeddings
 * - batch.pos ∈ {0, 1, ..., n_pos-1}: Position IDs
 * - h^TTS_t ∈ ℝ^n_embd: TTS hidden state at position t
 * - KV_cache += {K_t, V_t}: Updated KV cache
 *
 * Example Usage:
 *   TTSContext tts_ctx;
 *   float embeddings[768 * 10];  // 10 condition embeddings
 *   int n_past_tts = 0;
 *   bool success = prefill_with_emb_tts(tts_ctx, params, embeddings, 10, 4, &n_past_tts);
 *   if (success) {
 *       // TTS KV cache updated with condition embeddings
 *       // n_past_tts = 10
 *   }
 *
 * References:
 * - llama.cpp: llama_decode()
 * - llama.cpp: llama_set_embeddings()
 * - llama.cpp: llama_get_embeddings_ith()
 * - omni-tts.cpp: Lines 360-469 (prefill implementation)
 * - omni.cpp: Lines 2149-2154 (prefill in simplex mode)
 * - AGENTS.md: Phase 3 documentation on multimodal embeddings
 *
 * Error Handling:
 * - Returns false if n_pos is invalid
 * - Returns false if n_embd is invalid
 * - Returns false if models not loaded
 * - Returns false if llama_decode() fails
 * - Prints error messages to stderr
 *
 * Safety Checks:
 * - Validates n_pos against INT_MAX to prevent overflow
 * - Validates n_embd against reasonable range
 * - Checks for model loading before processing
 * - Ensures embeddings are properly aligned
 *
 * Note: batch.embd and batch.token are mutually exclusive.
 * We use batch.embd for condition embeddings, not batch.token.
 *
 * Note: Position IDs are critical for RoPE (Rotary Positional Embeddings)
 * - Ensures embeddings are placed at correct positions
 * - Prevents position drift in KV cache
 * - Matches Python's torch.arange behavior
 *
 * Note: This function enables embeddings output via llama_set_embeddings()
 * - Required for hidden state extraction (for debugging)
 * - Required for head_code logits computation in sample_tts_token()
 */
bool prefill_with_emb_tts(TTSContext& tts_ctx, common_params* params, float* embed, int n_pos, int n_batch, int* n_past_tts) {
    // 🔧 [安全检查] 验证输入参数
    if (n_pos <= 0) {
        LOG_ERR("%s: invalid n_pos=%d, skipping\n", __func__, n_pos);
        return false;
    }
    if (n_pos > 10000) {
        LOG_ERR("%s: n_pos=%d seems too large, likely data corruption\n", __func__, n_pos);
        return false;
    }
    if (!tts_ctx.model_llm || !tts_ctx.model_tts) {
        LOG_ERR("%s: TTS model not loaded\n", __func__);
        return false;
    }

    int n_embd = llama_model_n_embd(llama_get_model(tts_ctx.ctx_llm));

    // 🔧 [安全检查] 验证 n_embd 是合理值
    if (n_embd <= 0 || n_embd > 10000) {
        LOG_ERR("%s: invalid n_embd=%d from TTS model, likely model corruption\n", __func__, n_embd);
        return false;
    }

    // 🔧 [安全检查] 检查乘法溢出
    if (n_pos > (INT_MAX / n_embd)) {
        LOG_ERR("%s: n_pos=%d * n_embd=%d would overflow\n", __func__, n_pos, n_embd);
        return false;
    }

    // Save condition embeddings for first audio token re-forward (if not already saved)
    // This is needed to match Python's behavior: first audio token re-forwards the condition
    // In simplex mode, condition is re-forwarded at t=0 before first audio token
    if (!tts_ctx.tts_condition_saved && n_pos > 0) {
        tts_ctx.tts_condition_embeddings.resize(n_pos * n_embd);
        std::memcpy(tts_ctx.tts_condition_embeddings.data(), embed, n_pos * n_embd * sizeof(float));
        tts_ctx.tts_condition_length = n_pos;
        tts_ctx.tts_condition_n_embd = n_embd;
        tts_ctx.tts_condition_saved = true;
    }

    // Save the starting position before the loop
    int text_start_pos = *n_past_tts;

    // Check if we need to save all hidden states (for alignment testing)
    const char* save_hidden_states_dir = getenv("TTS_SAVE_HIDDEN_STATES_DIR");
    bool save_all_hidden_states = (save_hidden_states_dir != nullptr);

    // Process embeddings in batches
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        // Create batch with embeddings (not tokens)
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd = (embed + i * n_embd);  // 使用embedding作为输入

        // 设置pos值以确保正确的KV cache位置
        // Python: pos_ids = torch.arange(text_start_pos, text_start_pos + condition_length)
        // C++: batch.pos[j] = text_start_pos + i + j (where i is the offset within the current batch)
        std::vector<llama_pos> pos_vec(n_eval);
        batch.pos = pos_vec.data();
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = text_start_pos + i + j;  // Fix: use text_start_pos + i + j instead of *n_past_tts + j
        }

        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(tts_ctx.ctx_llm, true);

        if (llama_decode(tts_ctx.ctx_llm, batch)) {
            LOG_ERR("%s : failed to eval TTS embeddings. pos %d/%d (batch size %d, n_past %d)\n",
                    __func__, i, n_pos, n_batch, *n_past_tts);
            llama_set_embeddings(tts_ctx.ctx_llm, false);
            return false;
        }

        // Save hidden states for each token in the batch (for alignment testing)
        // Note: llama_get_embeddings_ith uses negative indices relative to the end of the batch
        // For a batch of n_eval tokens: -1 is last, -2 is second-to-last, ..., -n_eval is first
        if (save_all_hidden_states) {
            for (int j = 0; j < n_eval; j++) {
                int token_idx = text_start_pos + i + j;
                // Get j-th token in current batch: j=0 -> -n_eval (first), j=n_eval-1 -> -1 (last)
                int llama_idx = j - n_eval;
                const float* hidden_state = llama_get_embeddings_ith(tts_ctx.ctx_llm, llama_idx);
                if (hidden_state) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/hidden_states_%03d.bin", save_hidden_states_dir, token_idx);
                    FILE* f = fopen(filepath, "wb");
                    if (f) {
                        fwrite(&token_idx, sizeof(int32_t), 1, f);
                        fwrite(&n_embd, sizeof(int32_t), 1, f);
                        fwrite(hidden_state, sizeof(float), n_embd, f);
                        fclose(f);
                    }
                } else {
                    LOG_WRN("TTS: Failed to get hidden state for token %d (llama_idx=%d)\n", token_idx, llama_idx);
                }
            }
        }

        // Keep embeddings enabled for sample_tts_token to use
    }

    // Update n_past_tts after all tokens are processed
    *n_past_tts = text_start_pos + n_pos;

    return true;
}

void free_tts_context(TTSContext& tts_ctx) {
    if (tts_ctx.head_code_weight) free(tts_ctx.head_code_weight);
    if (tts_ctx.emb_code_weight) free(tts_ctx.emb_code_weight);
    if (tts_ctx.emb_text_weight) free(tts_ctx.emb_text_weight);
    if (tts_ctx.projector_weights.buf_w) ggml_backend_buffer_free(tts_ctx.projector_weights.buf_w);
    if (tts_ctx.projector_weights.ctx_w) ggml_free(tts_ctx.projector_weights.ctx_w);
    if (tts_ctx.projector_weights.backend) ggml_backend_free(tts_ctx.projector_weights.backend);
    // tts_buf_w / tts_ctx_w / tts_backend / ctx_meta / ctx_gguf were used by the removed
    // tts_transformer_forward manual pass.  The transformer is now owned by libllama.
}

} // namespace tts
} // namespace omni
