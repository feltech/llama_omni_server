cmake_minimum_required(VERSION 3.23)
project(llama_omni_server CXX)

# ── Model root — override to point at model files on any machine ───────────────
# Defaults to the standard sibling-repo GGUF directory layout.
# Override with -DLLAMAOMNISERVER_MODEL_ROOT=/path/to/your/gguf/dir
# Exposed to test processes via the LLAMAOMNISERVER_MODEL_ROOT env var.
set(LLAMAOMNISERVER_MODEL_ROOT "${PROJECT_SOURCE_DIR}/../../models/gguf"
    CACHE PATH "Directory containing MiniCPM-o GGUF model files (parent of vision/, audio/, tts/, token2wav-gguf/)")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build options ─────────────────────────────────────────────────────────────
option(
    LLAMAOMNISERVER_ENABLE_TESTS "Enable unit/integration/e2e tests" OFF
)
option(
    LLAMAOMNISERVER_ENABLE_TEST_VENDOR
    "Build exploratory vendor integration tests (require GPU and model files)"
    OFF)

# ── Compiler warnings — all project targets inherit these ─────────────────────
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# ── clang-tidy — applied to project targets only, not vendor code ─────────────
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND TRUE)  # TODO(DF): remove TRUE/FALSE - just added for speeding up prototype builds.
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if (CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        # No --config-file: use directory-based .clang-tidy discovery so that
        # tests/.clang-tidy (InheritParentConfig: true) can override checks for
        # test targets without affecting src/ targets.
        set(PROJECT_CXX_CLANG_TIDY
            "${CLANG_TIDY_EXE}"
            "--use-color"
            "--allow-no-checks"
        )
    else ()
        message(WARNING "clang-tidy not found — static analysis disabled")
    endif ()
endif ()

find_package(Catch2 REQUIRED)
find_package(llama-cpp REQUIRED)
find_package(SndFile REQUIRED)
find_package(stb REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(mdspan REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Microsoft.GSL REQUIRED)
# Boost: Boost::headers provides all header-only libs (asio, beast, cobalt headers).
# Boost::cobalt is the compiled cobalt runtime (coroutines, channels, generators).
find_package(Boost REQUIRED COMPONENTS headers cobalt)

# ── shared source library (config, protocol, server) ─────────────────────────
add_library(omni_server_lib STATIC
    src/config/config_loader.cpp
    src/server.cpp
)
target_include_directories(omni_server_lib PRIVATE src)
target_include_directories(omni_server_lib SYSTEM PRIVATE vendor)
target_link_libraries(omni_server_lib PUBLIC
    yaml-cpp::yaml-cpp
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    llama-cpp::llama-cpp
    Boost::cobalt
    Boost::headers
    Microsoft.GSL::GSL
    std::mdspan
)
if (PROJECT_CXX_CLANG_TIDY)
    set_target_properties(omni_server_lib PROPERTIES CXX_CLANG_TIDY "${PROJECT_CXX_CLANG_TIDY}")
endif ()

# ── GPU / model library (session, model_manager, audio encoder; requires llama-cpp) ──
add_library(omni_model_lib STATIC
    src/wav.cpp
    src/pipeline/model_manager.cpp
    src/pipeline/vram_planner.cpp
    src/pipeline/audio_encoder.cpp
    src/pipeline/tts_pipeline.cpp
    src/session.cpp
    vendor/omni/audition.cpp
    vendor/omni/omni-tts.cpp
    vendor/omni/token2wav.cpp
    vendor/omni/token2wav-impl.cpp
)
target_include_directories(omni_model_lib PRIVATE src)
target_include_directories(omni_model_lib SYSTEM PRIVATE vendor)
target_link_libraries(omni_model_lib PUBLIC
    llama-cpp::llama-cpp
    SndFile::sndfile
    spdlog::spdlog
    Boost::cobalt
    Boost::headers
    Microsoft.GSL::GSL
    std::mdspan
    Eigen3::Eigen
    mtmd
)
if (CUDAToolkit_FOUND)
    target_link_libraries(omni_model_lib PUBLIC CUDA::cudart)
endif ()
# vendor/omni/audition.cpp is ported external code — suppress warnings and exclude
# from clang-tidy.  The SYSTEM vendor include path suppresses warnings from vendor
# headers when included by project code, but audition.cpp includes its sibling
# headers via relative paths (not via -isystem), so needs -w for its own TU.
foreach (vendor_src
    vendor/omni/audition.cpp
    vendor/omni/omni-tts.cpp
    vendor/omni/token2wav.cpp
    vendor/omni/token2wav-impl.cpp)
    set_source_files_properties(${vendor_src} PROPERTIES
        COMPILE_OPTIONS "-w"
        CXX_CLANG_TIDY ""
    )
endforeach ()
if (PROJECT_CXX_CLANG_TIDY)
    set_target_properties(omni_model_lib PROPERTIES CXX_CLANG_TIDY "${PROJECT_CXX_CLANG_TIDY}")
endif ()

# ── server executable ─────────────────────────────────────────────────────────
add_executable(omni_server
    src/main.cpp
)
target_link_libraries(omni_server PRIVATE omni_server_lib omni_model_lib)
if (PROJECT_CXX_CLANG_TIDY)
    set_target_properties(omni_server PROPERTIES CXX_CLANG_TIDY "${PROJECT_CXX_CLANG_TIDY}")
endif ()

if (LLAMAOMNISERVER_ENABLE_TESTS)

    # ── unit tests (no GPU) ───────────────────────────────────────────────────────
    add_executable(test_unit
        tests/unit/test_config_loader.cpp
        tests/unit/test_protocol.cpp
        tests/unit/test_vram_planner.cpp
        tests/unit/test_tts_sampling.cpp
        tests/unit/test_kv_eviction.cpp
    )
    target_include_directories(test_unit PRIVATE src)
    target_link_libraries(test_unit PRIVATE Catch2::Catch2WithMain omni_server_lib omni_model_lib)
    if (PROJECT_CXX_CLANG_TIDY)
        set_target_properties(test_unit PROPERTIES CXX_CLANG_TIDY "${PROJECT_CXX_CLANG_TIDY}")
    endif ()

    # ── e2e tests (GPU + model files required); real WebSocket over TCP) ──────────
    add_executable(test_e2e
        tests/e2e/test_ws_session.cpp
    )
    target_include_directories(test_e2e PRIVATE src)
    target_compile_definitions(test_e2e PRIVATE LLAMAOMNISERVER_TEST_SERVER_EXE="$<TARGET_FILE:omni_server>")
    target_link_libraries(test_e2e PRIVATE Catch2::Catch2WithMain omni_server_lib)
    target_link_libraries(test_e2e PRIVATE omni_model_lib)
    if (PROJECT_CXX_CLANG_TIDY)
        set_target_properties(test_e2e PROPERTIES CXX_CLANG_TIDY "${PROJECT_CXX_CLANG_TIDY}")
    endif ()

    # ── integration tests (GPU + model files required) ────────────────────────────
    add_executable(test_integration
        tests/integration/test_text_inference.cpp
        tests/integration/test_session_init.cpp
        tests/integration/test_audio_duplex.cpp
        tests/integration/test_video_duplex.cpp
        tests/integration/test_tts_streaming.cpp
        tests/integration/test_text_turn.cpp
        tests/integration/test_kv_eviction_integration.cpp
    )
    target_include_directories(test_integration PRIVATE src tests/integration)
    target_link_libraries(test_integration PRIVATE
        Catch2::Catch2WithMain
        omni_server_lib
        omni_model_lib
        SndFile::sndfile
        stb::stb
    )
    if (PROJECT_CXX_CLANG_TIDY)
        set_target_properties(test_integration PROPERTIES CXX_CLANG_TIDY "${PROJECT_CXX_CLANG_TIDY}")
    endif ()

    # ── exploratory vendor integration tests (GPU + model files required) ─────────
    # clang-tidy is NOT applied to vendor targets — the code is ported from an
    # external source and is not expected to comply with project style rules.
    if (LLAMAOMNISERVER_ENABLE_TEST_VENDOR)
        add_executable(test_tts
            tests/integration/vendor/test_tts.cpp
            vendor/omni/omni-tts.cpp
            vendor/omni/token2wav-impl.cpp
            vendor/omni/token2wav.cpp
            vendor/omni/audition.cpp
        )
        target_link_libraries(test_tts PRIVATE Catch2::Catch2WithMain llama-cpp::llama-cpp SndFile::sndfile spdlog::spdlog)
        target_include_directories(test_tts SYSTEM PRIVATE vendor)

        add_executable(test_duplex_tts
            tests/integration/vendor/test_duplex_tts.cpp
            vendor/omni/omni-tts.cpp
            vendor/omni/token2wav-impl.cpp
            vendor/omni/token2wav.cpp
            vendor/omni/audition.cpp
        )
        target_link_libraries(test_duplex_tts PRIVATE Catch2::Catch2WithMain llama-cpp::llama-cpp SndFile::sndfile spdlog::spdlog)
        target_include_directories(test_duplex_tts SYSTEM PRIVATE vendor)

        add_executable(test_duplex
            tests/integration/vendor/test_duplex.cpp
            vendor/omni/audition.cpp
        )
        target_link_libraries(test_duplex PRIVATE Catch2::Catch2WithMain llama-cpp::llama-cpp SndFile::sndfile mtmd stb::stb spdlog::spdlog)
        target_include_directories(test_duplex SYSTEM PRIVATE vendor)

        add_executable(test_inference
            tests/integration/vendor/test_inference.cpp
            vendor/omni/audition.cpp
        )
        target_link_libraries(test_inference PRIVATE Catch2::Catch2WithMain llama-cpp::llama-cpp SndFile::sndfile mtmd stb::stb spdlog::spdlog)
        target_include_directories(test_inference SYSTEM PRIVATE vendor)
        target_include_directories(test_inference PRIVATE tests)
    endif ()

    # ── Enable testing ────────────────────────────────────────────────────────────
    enable_testing()

    add_test(NAME test_unit COMMAND test_unit)
    set_tests_properties(test_unit PROPERTIES
        ENVIRONMENT "LLAMAOMNISERVER_TEST_REPO_ROOT=${CMAKE_SOURCE_DIR};LLAMAOMNISERVER_MODEL_ROOT=${LLAMAOMNISERVER_MODEL_ROOT}"
    )

    add_test(NAME test_e2e COMMAND test_e2e)
    set_tests_properties(test_e2e PROPERTIES
        ENVIRONMENT "LLAMAOMNISERVER_TEST_REPO_ROOT=${CMAKE_SOURCE_DIR};LLAMAOMNISERVER_TEST_EXE_DIR=${CMAKE_BINARY_DIR};LLAMAOMNISERVER_MODEL_ROOT=${LLAMAOMNISERVER_MODEL_ROOT}"
    )

    add_test(NAME test_integration COMMAND test_integration --verbosity high --durations yes)
    set_tests_properties(test_integration PROPERTIES
        ENVIRONMENT "LLAMAOMNISERVER_TEST_REPO_ROOT=${CMAKE_SOURCE_DIR};LLAMAOMNISERVER_MODEL_ROOT=${LLAMAOMNISERVER_MODEL_ROOT}"
    )

    if (LLAMAOMNISERVER_ENABLE_TEST_VENDOR)
        add_test(NAME test_tts COMMAND test_tts)
        add_test(NAME test_duplex_tts COMMAND test_duplex_tts)
        add_test(NAME test_duplex COMMAND test_duplex)
        add_test(NAME test_inference COMMAND test_inference)
        set_tests_properties(test_tts test_duplex_tts test_duplex test_inference PROPERTIES
            ENVIRONMENT "LLAMAOMNISERVER_TEST_REPO_ROOT=${CMAKE_SOURCE_DIR};LLAMAOMNISERVER_MODEL_ROOT=${LLAMAOMNISERVER_MODEL_ROOT}"
        )
    endif ()
endif ()
