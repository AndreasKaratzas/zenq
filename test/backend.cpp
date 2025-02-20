#include "backend/compute/engine.hpp"
#include <gtest/gtest.h>
#include <random>

namespace {

class BackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize engines for both CPU and CUDA
        cpu_engine_ = std::make_unique<hpc::compute::Engine>(hpc::core::DeviceType::CPU);
#ifdef CUDA_ENABLED
        cuda_engine_ = std::make_unique<hpc::compute::Engine>(hpc::core::DeviceType::CUDA);
#endif
    }

    void fill_random(hpc::compute::Tensor& tensor) {
        auto                                  span = tensor.data_span<float>();
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (float& val : span) {
            val = dis(gen);
        }
    }

    bool tensors_equal(const hpc::compute::Tensor& a,
                       const hpc::compute::Tensor& b,
                       float                       tolerance = 1e-5f) {
        if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
            return false;
        }

        auto span_a = a.data_span<float>();
        auto span_b = b.data_span<float>();

        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(span_a[i] - span_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<hpc::compute::Engine> cpu_engine_;
#ifdef CUDA_ENABLED
    std::unique_ptr<hpc::compute::Engine> cuda_engine_;
#endif
};

TEST_F(BackendTest, BasicConvolution) {
    // Create input tensor (N=1, C=3, H=32, W=32)
    hpc::compute::Tensor input({1, 3, 32, 32}, hpc::core::DataType::FLOAT32);

    // Create weights tensor (Out_C=16, In_C=3, K=3, K=3)
    hpc::compute::Tensor weights({16, 3, 3, 3}, hpc::core::DataType::FLOAT32);

    // Fill tensors with random data
    fill_random(input);
    fill_random(weights);

    // Configure convolution
    hpc::config::ConvConfig config{.kernel_size  = 3,
                                   .stride       = 1,
                                   .padding      = 1,
                                   .in_channels  = 3,
                                   .out_channels = 16,
                                   .use_bias     = false};

    // Run on CPU
    auto cpu_output = cpu_engine_->conv2d(input, weights, config);

#ifdef CUDA_ENABLED
    // Run on CUDA
    auto cuda_output = cuda_engine_->conv2d(input, weights, config);

    // Compare results
    EXPECT_TRUE(tensors_equal(cpu_output, cuda_output));
#endif

    // Verify output shape
    std::vector<size_t> expected_shape = {1, 16, 32, 32}; // Same padding
    EXPECT_EQ(cpu_output.shape(), expected_shape);
}

TEST_F(BackendTest, StridedConvolution) {
    // Create input tensor (N=1, C=3, H=32, W=32)
    hpc::compute::Tensor input({1, 3, 32, 32}, hpc::core::DataType::FLOAT32);

    // Create weights tensor (Out_C=16, In_C=3, K=3, K=3)
    hpc::compute::Tensor weights({16, 3, 3, 3}, hpc::core::DataType::FLOAT32);

    fill_random(input);
    fill_random(weights);

    // Configure strided convolution
    hpc::config::ConvConfig config{.kernel_size  = 3,
                                   .stride       = 2,
                                   .padding      = 1,
                                   .in_channels  = 3,
                                   .out_channels = 16,
                                   .use_bias     = false};

    auto cpu_output = cpu_engine_->conv2d(input, weights, config);

#ifdef CUDA_ENABLED
    auto cuda_output = cuda_engine_->conv2d(input, weights, config);
    EXPECT_TRUE(tensors_equal(cpu_output, cuda_output));
#endif

    // Verify output shape for strided convolution
    std::vector<size_t> expected_shape = {1, 16, 16, 16}; // Half size due to stride=2
    EXPECT_EQ(cpu_output.shape(), expected_shape);
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}