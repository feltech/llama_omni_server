/**
 * @file   main.cpp
 * @brief  Entry point for llama-omni-server using Boost.Cobalt.
 *
 * Uses `boost::cobalt::main` as the coroutine entry point so that `co_main`
 * runs on a cobalt-managed executor. SIGINT / SIGTERM are handled by Cobalt's
 * `co_main` machinery, which cancels the awaited coroutine tree. The server
 * accept loop and live-session teardown therefore rely on propagated coroutine
 * cancellation rather than a separate signal thread.
 *
 * ### Shutdown flow
 *
 * 1. User presses Ctrl-C or sends SIGTERM.
 * 2. Cobalt cancels the awaited `co_main` coroutine tree.
 * 3. `server.run()` unwinds through its structured teardown scopes.
 * 4. `gpu_ex.join()` waits for any in-flight GPU work to finish.
 * 5. `co_return 0` exits the process cleanly.
 */

#include <filesystem>
#include <format>
#include <span>

#include <boost/asio/thread_pool.hpp>
#include <boost/cobalt/main.hpp>

#include <gsl/util>
#include <spdlog/spdlog.h>

#include "config/config_loader.hpp"
#include "pipeline/model_manager.hpp"
#include "server.hpp"

// `co_main` is the cobalt entry point — declared by <boost/cobalt/main.hpp>
// and called from the cobalt-generated `main()`.  The signature must match
// exactly: returning `boost::cobalt::main` and accepting `(int, char*[])`.
// The `argv` C-style array is required by the cobalt framework's main contract.
// Suppress the C-array warning for the entire function because Boost.Cobalt
// requires the exact `co_main(int, char*[])` signature.
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
boost::cobalt::main co_main(int argc, char * argv[])
{
	std::span const args{argv, static_cast<std::size_t>(argc)};

	if (args.size() < 2)
	{
		spdlog::error("Usage: {} <config.yaml>", args.front());
		// NOLINTNEXTLINE(*-avoid-c-arrays) - false positive.
		co_return 1;
	}

	// Load and validate configuration from the YAML file.
	auto const config_path = std::filesystem::path{gsl::at(args, 1)};
	spdlog::info("Loading config from: {}", config_path.string());

	llama_omni_server::AppConfig cfg;
	try
	{
		cfg = llama_omni_server::load_config(config_path);
	}
	catch (std::exception const & err)
	{
		spdlog::error("Failed to load config: {}", err.what());
		co_return 1;
	}

	spdlog::set_level(spdlog::level::from_str(cfg.logging.level));

	std::shared_ptr<llama_omni_server::ModelManager> model_manager;
	try
	{
		model_manager = std::make_shared<llama_omni_server::ModelManager>(cfg);
	}
	catch (std::exception const & err)
	{
		spdlog::error("Failed to load models: {}", err.what());
		co_return 1;
	}

	auto runtime = std::make_shared<llama_omni_server::AppRuntime>(
		std::move(cfg), std::move(model_manager), std::make_shared<boost::asio::thread_pool>(1));

	llama_omni_server::WebSocketServer server{runtime};

	spdlog::info("Starting server on {}:{}", runtime->cfg.server.host, runtime->cfg.server.port);

	// Run the accept loop until Cobalt cancellation or an unrecoverable error
	// tears down the awaited coroutine tree.
	co_await server.run();

	// Wait for all in-flight GPU work to finish before exiting.
	runtime->gpu_ex->join();

	spdlog::info("Server shut down cleanly");
	co_return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
