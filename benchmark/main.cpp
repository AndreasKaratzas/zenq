#include <benchmark/benchmark.h>

#ifdef HPC_LOGGING_ENABLED
    #include "common/logging.hpp"
#endif

// This main file will use the automatic main function from Google Benchmark
// which is linked with benchmark::benchmark_main

int main(int argc, char** argv) {
#ifdef HPC_LOGGING_ENABLED
    // Initialize logging if enabled
    hpc::logging::OptimizationLogger::init(hpc::logging::LogLevel::Info, true);
    hpc::logging::OptimizationLogger::enableOptimizationLogging(true);
    LOG_INFO("Starting benchmarks...");
#endif

    // Run benchmarks
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

#ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Benchmarks completed");
#endif

    return 0;
}