#pragma once
/**
 * @file vendor_paths.hpp
 * @brief Path helpers for exploratory vendor integration tests.
 *
 * Reads model root and repo root from CMake-injected environment variables so
 * that test binaries are portable across machines and build directories.
 *
 *   LLAMAOMNISERVER_MODEL_ROOT    — root of the GGUF model files tree
 *                                   (the gguf/ directory itself).
 *                                   Set by the CMake cache variable of the same
 *                                   name; defaults to PROJECT_SOURCE_DIR/../../models.
 *   LLAMAOMNISERVER_TEST_REPO_ROOT — absolute path to the repo root (where
 *                                    CMakeLists.txt lives); set to CMAKE_SOURCE_DIR
 *                                    by CTest.
 *
 * Default fallbacks (used when running test binaries manually without CTest) are
 * paths relative to the working directory that work when tests are run from the
 * repo root.
 */

// NOLINTBEGIN — vendor test helper, clang-tidy disabled for vendor targets

#include <cstdlib>
#include <string>

/// Returns the model root directory (the gguf/ directory, containing model files).
inline std::string vp_model_root()
{
	char const * env = std::getenv("LLAMAOMNISERVER_MODEL_ROOT");  // NOLINT(concurrency-mt-unsafe)
	return env != nullptr ? env : "../../models/gguf";
}

/// Returns the test_data directory inside the repo.
inline std::string vp_test_data()
{
	char const * env =
		std::getenv("LLAMAOMNISERVER_TEST_REPO_ROOT");	// NOLINT(concurrency-mt-unsafe)
	return std::string{env != nullptr ? env : "."} + "/test_data";
}

/**
 * @brief String wrapper with implicit const char* conversion for C API compatibility.
 *
 * Stores the path as a std::string and provides an implicit conversion to
 * const char* so that VendorPath values can be passed directly to llama.cpp
 * and libsndfile functions without explicit .c_str() calls.
 *
 * Rvalue overloads of all pointer-returning members are deleted so that passing
 * a temporary VendorPath to a function is a compile-time error rather than a
 * silent dangling-pointer bug.
 */
struct VendorPath
{
	std::string value;

	explicit VendorPath(std::string str) : value{std::move(str)} {}

	// Implicit conversion to const char* for C API calls (e.g. llama_model_load_from_file).
	operator char const *() const & noexcept
	{
		return value.c_str();
	}  // NOLINT(google-explicit-constructor)
	operator char const *() const && = delete;

	// Implicit conversion to std::string const& for string operations.
	operator std::string const &() const & noexcept
	{
		return value;
	}  // NOLINT(google-explicit-constructor)
	operator std::string const &() const && = delete;

	[[nodiscard]] char const * c_str() const & noexcept
	{
		return value.c_str();
	}
	[[nodiscard]] char const * c_str() const && = delete;
};

// NOLINTEND
