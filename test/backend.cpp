#include "kernel_compute/common/config.hpp"
#include "kernel_compute/compute/engine.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace kernel_compute;

// Test configuration
struct TestConfig {
    // Kernel configurations to test
    std::vector<KernelConfig> kernel_configs = {
        // Default config
        {.batch_size  = 1,
         .channels    = 3,
         .height      = 224,
         .width       = 224,
         .kernel_size = 3,
         .stride      = 1,
         .padding     = 1,
         .type        = KernelType::Dense},
        // Custom config example
        {.batch_size  = 16,
         .channels    = 64,
         .height      = 112,
         .width       = 112,
         .kernel_size = 5,
         .stride      = 2,
         .padding     = 2,
         .type        = KernelType::Dense}};

    // Compute configurations to test
    std::vector<ComputeConfig> compute_configs = {
        // CPU config
        {.device           = DeviceType::CPU,
         .data_type        = DataType::Float32,
         .enable_profiling = true,
         .cpu              = {.num_threads = 8, .use_avx512 = true}},
        // CUDA config
        {.device           = DeviceType::CUDA,
         .data_type        = DataType::Float32,
         .enable_profiling = true,
         .cuda             = {.device_id = 0, .max_threads_per_block = 256}}};

    // Test parameters
    int  num_warmup_runs = 5;
    int  num_test_runs   = 100;
    bool verify_results  = true;
};

// Utility functions
template <typename T>
void fill_random(std::span<T> data) {
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (auto& val : data) {
        val = static_cast<T>(dis(gen));
    }
}

template <typename T>
float compute_mse(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size())
        return std::numeric_limits<float>::infinity();

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum / a.size());
}

struct TestResults {
    float avg_time_ms{0.0f};
    float min_time_ms{0.0f};
    float max_time_ms{0.0f};
    float std_dev_ms{0.0f};
    float mse{0.0f};
};

TestResults run_single_test(Engine&             engine,
                            const KernelConfig& kernel_config,
                            const TestConfig&   test_config) {
    // Prepare input and weight tensors
    Shape input_shape = {kernel_config.batch_size,
                         kernel_config.channels,
                         kernel_config.height,
                         kernel_config.width};

    Shape weight_shape = {kernel_config.channels,
                          kernel_config.channels,
                          kernel_config.kernel_size,
                          kernel_config.kernel_size};

    Tensor input(input_shape, DataType::Float32);
    Tensor weights(weight_shape, DataType::Float32);

    // Initialize with random data
    fill_random(input.data<float>());
    fill_random(weights.data<float>());

    // Warmup runs
    for (int i = 0; i < test_config.num_warmup_runs; ++i) {
        engine.compute(input, weights, kernel_config);
        engine.synchronize();
    }

    // Benchmark runs
    std::vector<float> timings;
    timings.reserve(test_config.num_test_runs);

    Tensor reference_output;
    for (int i = 0; i < test_config.num_test_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        Tensor output = engine.compute(input, weights, kernel_config);
        engine.synchronize();

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(end - start);
        timings.push_back(duration.count());

        // Store first output as reference
        if (i == 0) {
            reference_output = output.clone();
        }
        // Verify results if enabled
        else if (test_config.verify_results) {
            float mse = compute_mse(output.data<float>(), reference_output.data<float>());
            if (mse > 1e-6f) {
                std::cerr << "Warning: Run " << i << " produced different results (MSE: " << mse
                          << ")\n";
            }
        }
    }

    // Calculate statistics
    TestResults results;
    results.avg_time_ms = std::accumulate(timings.begin(), timings.end(), 0.0f) / timings.size();
    results.min_time_ms = *std::min_element(timings.begin(), timings.end());
    results.max_time_ms = *std::max_element(timings.begin(), timings.end());

    float variance = 0.0f;
    for (float t : timings) {
        float diff = t - results.avg_time_ms;
        variance += diff * diff;
    }
    results.std_dev_ms = std::sqrt(variance / timings.size());

    return results;
}

void run_tests(const TestConfig& test_config) {
    for (const auto& compute_config : test_config.compute_configs) {
        std::cout << "\nTesting " << (compute_config.device == DeviceType::CUDA ? "CUDA" : "CPU")
                  << " implementation\n";
        std::cout << "----------------------------------------\n";

        try {
            auto engine = Engine::create(compute_config.device);
            engine->initialize(compute_config);

            for (const auto& kernel_config : test_config.kernel_configs) {
                std::cout << "\nKernel configuration:\n"
                          << "  Batch size: " << kernel_config.batch_size << "\n"
                          << "  Channels: " << kernel_config.channels << "\n"
                          << "  Input size: " << kernel_config.height << "x" << kernel_config.width
                          << "\n"
                          << "  Kernel size: " << kernel_config.kernel_size << "\n"
                          << "  Stride: " << kernel_config.stride << "\n"
                          << "  Padding: " << kernel_config.padding << "\n"
                          << "  Type: "
                          << (kernel_config.type == KernelType::Dense ? "Dense" : "Sparse") << "\n";

                auto results = run_single_test(*engine, kernel_config, test_config);

                std::cout << "\nPerformance results:\n"
                          << "  Average time: " << results.avg_time_ms << " ms\n"
                          << "  Min time: " << results.min_time_ms << " ms\n"
                          << "  Max time: " << results.max_time_ms << " ms\n"
                          << "  Std deviation: " << results.std_dev_ms << " ms\n";
            }

            engine->reset();

        } catch (const std::exception& e) {
            std::cerr << "Error testing "
                      << (compute_config.device == DeviceType::CUDA ? "CUDA" : "CPU")
                      << " implementation: " << e.what() << "\n";
            continue;
        }
    }
}

int main() {
    TestConfig config;
    // Modify config here if needed

    try {
        run_tests(config);
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}