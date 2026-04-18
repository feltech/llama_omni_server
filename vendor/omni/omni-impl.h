/**
 * Minimal omni-impl.h for audition encoder
 *
 * This file contains only the minimal definitions needed to build the audition
 * audio encoder from llama.cpp-omni. Most functionality uses libmtmd instead.
 *
 * Source: llama.cpp-omni/tools/omni/omni-impl.h (minimal extract)
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <map>

//
// GGUF metadata keys
//
#define KEY_FTYPE               "general.file_type"
#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_MODEL_TYPE          "general.model_type"
#define KEY_HAS_AUDIO_ENC       "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC      "clip.has_vision_encoder"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"

#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"

// Audio-specific keys
#define KEY_A_NUM_MEL_BINS      "clip.audio.num_mel_bins"
#define KEY_A_PROJ_STACK_FACTOR "clip.audio.projector.stack_factor"

//
// Tensor name patterns
//
#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s"
#define TN_LN_2            "%s.blk.%d.ln2.%s"
#define TN_LS_1            "%s.blk.%d.ls1.%s"
#define TN_LS_2            "%s.blk.%d.ls2.%s"
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"

// Alignment utility
#define VISION_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

//
// Model type enum
//
enum omni_model_type {
    MiniCPM_o,
    OMNI_MODEL_TYPE_UNKNOWN,
};

static std::map<omni_model_type, std::string> OMNI_MODEL_TYPE_NAMES = {
    { MiniCPM_o, "MiniCPM-o" },
};

static omni_model_type omni_model_type_from_string(const std::string & str) {
    for (const auto & pair : OMNI_MODEL_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return OMNI_MODEL_TYPE_UNKNOWN;
}

//
// Logging infrastructure
//
struct omni_logger_state {
    ggml_log_level verbosity_thold;
    ggml_log_callback log_callback;
    void * log_callback_user_data;
};

// Global logger state (defined in audition.cpp)
extern struct omni_logger_state g_logger_state;

static void omni_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

static void omni_log_internal_v(enum ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

static void omni_log_internal(enum ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    omni_log_internal_v(level, format, args);
    va_end(args);
}

#define LOG_TMPL(level, ...) \
    do { \
        if ((level) >= g_logger_state.verbosity_thold) { \
            omni_log_internal((level), __VA_ARGS__); \
        } \
    } while (0)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  __VA_ARGS__)

//
// Utility functions
//
static std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

//
// Smart pointers for GGML resources
// NOTE: These are already defined in ggml-cpp.h, so we just include that header
//
#include "ggml-cpp.h"
