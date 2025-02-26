#include "compute/cpp/kernels/conv2d.hpp"
#include "compute/cpp/tensor.hpp"
#include "compute/cpp/view.hpp"
#ifdef CUDA_ENABLED
    #include "compute/cuda/wrapper.hpp"
#endif

#include <chrono> // For std::chrono::high_resolution_clock
#include <cmath>  // For std::isnan, std::isinf
#include <gtest/gtest.h>
#include <memory>
#include <numeric> // For std::iota, std::fill, std::abs
#include <type_traits>
#include <vector>

using namespace hpc::compute;

template <typename T>
class Conv2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 3x3 kernel descriptor
        kernel = make_conv2d<T>(3, // kernel_height
                                3, // kernel_width
                                2, // in_channels
                                2, // out_channels
                                1, // stride
                                1  // padding
        );

        // Initialize kernel weights with known values
        std::vector<size_t> weight_dims = {2, 2, 3, 3};
        Tensor<T>           weights(weight_dims);
        T*                  w_data = weights.data();

        // Initialize with zeros first
        std::fill(w_data, w_data + weights.size(), static_cast<T>(0));

        // Set very simple weight patterns to reduce numerical errors
        // First filter: simple 1.0 at center
        weights(0, 0, 1, 1) = static_cast<T>(1.0);
        // Second filter: simple 0.1 at all positions
        for (size_t kh = 0; kh < 3; ++kh) {
            for (size_t kw = 0; kw < 3; ++kw) {
                weights(1, 1, kh, kw) = static_cast<T>(0.1);
            }
        }

        kernel->load_weights(std::move(weights));
    }

    // Corrected reference convolution implementation
    Tensor<T> reference_convolution(const Tensor<T>& input,
                                    const Tensor<T>& weights,
                                    size_t           stride,
                                    size_t           padding) {
        size_t batch_size  = input.shape()[0];
        size_t in_channels = input.shape()[1];
        size_t in_height   = input.shape()[2];
        size_t in_width    = input.shape()[3];

        size_t out_channels  = weights.shape()[0];
        size_t kernel_height = weights.shape()[2];
        size_t kernel_width  = weights.shape()[3];

        size_t out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        size_t out_width  = (in_width + 2 * padding - kernel_width) / stride + 1;

        Tensor<T> output({batch_size, out_channels, out_height, out_width});
        std::fill(output.data(), output.data() + output.size(), static_cast<T>(0));

        // Create padded input
        std::vector<T> padded_input_data((in_height + 2 * padding) * (in_width + 2 * padding) *
                                             in_channels * batch_size,
                                         static_cast<T>(0));

        // Copy input data to padded input
        for (size_t n = 0; n < batch_size; ++n) {
            for (size_t c = 0; c < in_channels; ++c) {
                for (size_t h = 0; h < in_height; ++h) {
                    for (size_t w = 0; w < in_width; ++w) {
                        size_t padded_idx =
                            ((n * in_channels + c) * (in_height + 2 * padding) + (h + padding)) *
                                (in_width + 2 * padding) +
                            (w + padding);
                        size_t input_idx = ((n * in_channels + c) * in_height + h) * in_width + w;
                        padded_input_data[padded_idx] = input.data()[input_idx];
                    }
                }
            }
        }

        // Perform direct convolution
        for (size_t n = 0; n < batch_size; ++n) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        T sum = static_cast<T>(0);
                        for (size_t ic = 0; ic < in_channels; ++ic) {
                            for (size_t kh = 0; kh < kernel_height; ++kh) {
                                for (size_t kw = 0; kw < kernel_width; ++kw) {
                                    size_t ih = oh * stride + kh;
                                    size_t iw = ow * stride + kw;

                                    size_t padded_idx =
                                        ((n * in_channels + ic) * (in_height + 2 * padding) + ih) *
                                            (in_width + 2 * padding) +
                                        iw;

                                    T input_val  = padded_input_data[padded_idx];
                                    T weight_val = weights(oc, ic, kh, kw);
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                        size_t out_idx =
                            ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                        output.data()[out_idx] = sum;
                    }
                }
            }
        }

        return output;
    }

    std::unique_ptr<BaseKernel<T>> kernel;
};

using ConvTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Conv2DTest, ConvTypes);

TYPED_TEST(Conv2DTest, BasicForwardPass) {
    // Create input tensor with known values
    std::vector<size_t> input_dims = {1, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims);

    // Use simple patterns to reduce numerical issues
    for (size_t c = 0; c < 2; ++c) {
        for (size_t h = 0; h < 5; ++h) {
            for (size_t w = 0; w < 5; ++w) {
                // First channel: increasing values
                // Second channel: constant values
                if (c == 0) {
                    input(0, c, h, w) = static_cast<TypeParam>(h * 5 + w + 1);
                } else {
                    input(0, c, h, w) = static_cast<TypeParam>(10);
                }
            }
        }
    }

    // Perform forward pass
    Tensor<TypeParam> output = this->kernel->forward(input);

    // Verify output dimensions
    const auto& out_shape = output.shape();
    EXPECT_EQ(out_shape[0], 1u);
    EXPECT_EQ(out_shape[1], 2u);
    EXPECT_EQ(out_shape[2], 5u);
    EXPECT_EQ(out_shape[3], 5u);

    // Check output validity
    const TypeParam* output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i])) << "Output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output_data[i])) << "Output contains Inf at index " << i;
    }

    // Verify key point manually (center of filter at center of image)
    // With our weight setup, this should match the input value
    EXPECT_NEAR(output(0, 0, 2, 2), input(0, 0, 2, 2), static_cast<TypeParam>(1e-4))
        << "Center output pixel doesn't match expected value";

    // Second channel should be approximately sum of surrounding values * 0.1
    EXPECT_GT(output(0, 1, 2, 2), static_cast<TypeParam>(0))
        << "Second channel output should be positive";
}

TYPED_TEST(Conv2DTest, EnhancedNumericalForwardTest) {
    // Create a specialized kernel with known weights for numerical validation
    auto num_kernel = make_conv2d<TypeParam>(3, // kernel_height
                                             3, // kernel_width
                                             2, // in_channels
                                             2, // out_channels
                                             1, // stride
                                             1  // padding
    );

    // Create carefully designed weights for arithmetic validation
    std::vector<size_t> weight_dims = {2, 2, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);
    std::fill(weights.data(), weights.data() + weights.size(), static_cast<TypeParam>(0));

    // Set specific weight values for precise calculation validation
    // First output channel, first input channel: cross pattern
    weights(0, 0, 0, 1) = static_cast<TypeParam>(1.0); // top middle
    weights(0, 0, 1, 0) = static_cast<TypeParam>(2.0); // middle left
    weights(0, 0, 1, 1) = static_cast<TypeParam>(5.0); // center
    weights(0, 0, 1, 2) = static_cast<TypeParam>(3.0); // middle right
    weights(0, 0, 2, 1) = static_cast<TypeParam>(4.0); // bottom middle

    // First output channel, second input channel: corners only
    weights(0, 1, 0, 0) = static_cast<TypeParam>(0.5); // top left
    weights(0, 1, 0, 2) = static_cast<TypeParam>(0.5); // top right
    weights(0, 1, 2, 0) = static_cast<TypeParam>(0.5); // bottom left
    weights(0, 1, 2, 2) = static_cast<TypeParam>(0.5); // bottom right

    // Second output channel: simple values in first row
    weights(1, 0, 0, 0) = static_cast<TypeParam>(1.0);
    weights(1, 0, 0, 1) = static_cast<TypeParam>(1.0);
    weights(1, 0, 0, 2) = static_cast<TypeParam>(1.0);

    // Second output channel: simple values in second input channel
    weights(1, 1, 1, 1) = static_cast<TypeParam>(2.0); // only center

    num_kernel->load_weights(std::move(weights));

    // Create input with precise known values
    std::vector<size_t> input_dims = {1, 2, 4, 4};
    Tensor<TypeParam>   input(input_dims);

    // First channel: grid of increasing values
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            input(0, 0, h, w) = static_cast<TypeParam>(h * 4 + w + 1);
        }
    }

    // Second channel: constant value of 10
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            input(0, 1, h, w) = static_cast<TypeParam>(10.0);
        }
    }

    // Print input values for first channel
    std::cout << "\nInput Channel 0:\n";
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            std::cout << input(0, 0, h, w) << " ";
        }
        std::cout << "\n";
    }

    // Print input values for second channel
    std::cout << "\nInput Channel 1:\n";
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            std::cout << input(0, 1, h, w) << " ";
        }
        std::cout << "\n";
    }

    // Perform forward pass
    Tensor<TypeParam> output = num_kernel->forward(input);

    // Verify output dimensions
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 1u);
    ASSERT_EQ(out_shape[1], 2u);
    ASSERT_EQ(out_shape[2], 4u);
    ASSERT_EQ(out_shape[3], 4u);

    // Print the output values for debugging
    std::cout << "\nOutput Channel 0:\n";
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            std::cout << output(0, 0, h, w) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nOutput Channel 1:\n";
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            std::cout << output(0, 1, h, w) << " ";
        }
        std::cout << "\n";
    }

    // ======= Manually calculate expected values for specific positions =======

    // Position (1,1) - middle of image
    // For output channel 0:
    //   From input channel 0:
    //     weights(0,0,0,1) * input(0,0,0,1) = 1.0 * 2 = 2
    //     weights(0,0,1,0) * input(0,0,1,0) = 2.0 * 5 = 10
    //     weights(0,0,1,1) * input(0,0,1,1) = 5.0 * 6 = 30
    //     weights(0,0,1,2) * input(0,0,1,2) = 3.0 * 7 = 21
    //     weights(0,0,2,1) * input(0,0,2,1) = 4.0 * 10 = 40
    //   From input channel 1:
    //     weights(0,1,0,0) * input(0,1,0,0) = 0.5 * 10 = 5
    //     weights(0,1,0,2) * input(0,1,0,2) = 0.5 * 10 = 5
    //     weights(0,1,2,0) * input(0,1,2,0) = 0.5 * 10 = 5
    //     weights(0,1,2,2) * input(0,1,2,2) = 0.5 * 10 = 5
    //   Total: 2 + 10 + 30 + 21 + 40 + 5 + 5 + 5 + 5 = 123
    TypeParam expected_0_1_1 = static_cast<TypeParam>(123.0);

    // Position (2,2) - bottom right quadrant
    // For output channel 0:
    //   From input channel 0:
    //     weights(0,0,0,1) * input(0,0,1,2) = 1.0 * 7 = 7
    //     weights(0,0,1,0) * input(0,0,2,1) = 2.0 * 10 = 20
    //     weights(0,0,1,1) * input(0,0,2,2) = 5.0 * 11 = 55
    //     weights(0,0,1,2) * input(0,0,2,3) = 3.0 * 12 = 36
    //     weights(0,0,2,1) * input(0,0,3,2) = 4.0 * 15 = 60
    //   From input channel 1:
    //     weights(0,1,0,0) * input(0,1,1,1) = 0.5 * 10 = 5
    //     weights(0,1,0,2) * input(0,1,1,3) = 0.5 * 10 = 5
    //     weights(0,1,2,0) * input(0,1,3,1) = 0.5 * 10 = 5
    //     weights(0,1,2,2) * input(0,1,3,3) = 0.5 * 10 = 5
    //   Total: 7 + 20 + 55 + 36 + 60 + 5 + 5 + 5 + 5 = 198
    TypeParam expected_0_2_2 = static_cast<TypeParam>(198.0);

    // Position (0,0) - top left with padding
    // For output channel 1:
    //   From input channel 0:
    //     weights(1,0,0,0) * 0 (padding) = 1.0 * 0 = 0
    //     weights(1,0,0,1) * 0 (padding) = 1.0 * 0 = 0
    //     weights(1,0,0,2) * 0 (padding) = 1.0 * 0 = 0
    //   From input channel 1:
    //     weights(1,1,1,1) * input(0,1,0,0) = 2.0 * 10 = 20
    //   Total: 0 + 0 + 0 + 20 = 20
    TypeParam expected_1_0_0 = static_cast<TypeParam>(20.0);

    // Position (0,2) - top edge with some real inputs for channel 1
    // For output channel 1:
    //   From input channel 0:
    //     weights(1,0,0,0) * 0 (padding) = 1.0 * 0 = 0
    //     weights(1,0,0,1) * 0 (padding) = 1.0 * 0 = 0
    //     weights(1,0,0,2) * 0 (padding) = 1.0 * 0 = 0
    //   From input channel 1:
    //     weights(1,1,1,1) * input(0,1,0,2) = 2.0 * 10 = 20
    //   Total: 0 + 0 + 0 + 20 = 20
    TypeParam expected_1_0_2 = static_cast<TypeParam>(20.0);

    // Check specific manually calculated points
    const TypeParam tolerance = std::is_same<TypeParam, float>::value
                                    ? static_cast<TypeParam>(1e-4)
                                    : static_cast<TypeParam>(1e-10);

    EXPECT_NEAR(output(0, 0, 1, 1), expected_0_1_1, tolerance)
        << "Validation failed at position (0,0,1,1)";

    EXPECT_NEAR(output(0, 0, 2, 2), expected_0_2_2, tolerance)
        << "Validation failed at position (0,0,2,2)";

    EXPECT_NEAR(output(0, 1, 0, 0), expected_1_0_0, tolerance)
        << "Validation failed at position (0,1,0,0)";

    EXPECT_NEAR(output(0, 1, 0, 2), expected_1_0_2, tolerance)
        << "Validation failed at position (0,1,0,2)";
}

TYPED_TEST(Conv2DTest, ZeroPaddingConvolution) {
    // Create a specialized kernel with padding=0
    auto no_pad_kernel = make_conv2d<TypeParam>(3, // kernel_height
                                                3, // kernel_width
                                                2, // in_channels
                                                2, // out_channels
                                                1, // stride
                                                0  // padding = 0
    );

    // Create weights with simple, easy-to-trace values
    std::vector<size_t> weight_dims = {2, 2, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);
    std::fill(weights.data(), weights.data() + weights.size(), static_cast<TypeParam>(0));

    // Output channel 0, input channel 0: Simple identity filter
    weights(0, 0, 0, 0) = static_cast<TypeParam>(0.0);
    weights(0, 0, 0, 1) = static_cast<TypeParam>(0.0);
    weights(0, 0, 0, 2) = static_cast<TypeParam>(0.0);
    weights(0, 0, 1, 0) = static_cast<TypeParam>(0.0);
    weights(0, 0, 1, 1) = static_cast<TypeParam>(1.0); // center only
    weights(0, 0, 1, 2) = static_cast<TypeParam>(0.0);
    weights(0, 0, 2, 0) = static_cast<TypeParam>(0.0);
    weights(0, 0, 2, 1) = static_cast<TypeParam>(0.0);
    weights(0, 0, 2, 2) = static_cast<TypeParam>(0.0);

    // Output channel 0, input channel 1: Uniform value
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            weights(0, 1, h, w) = static_cast<TypeParam>(0.1); // all positions 0.1
        }
    }

    // Output channel 1, input channel 0: Horizontal edge detector
    weights(1, 0, 0, 0) = static_cast<TypeParam>(1.0);
    weights(1, 0, 0, 1) = static_cast<TypeParam>(2.0);
    weights(1, 0, 0, 2) = static_cast<TypeParam>(1.0);
    weights(1, 0, 1, 0) = static_cast<TypeParam>(0.0);
    weights(1, 0, 1, 1) = static_cast<TypeParam>(0.0);
    weights(1, 0, 1, 2) = static_cast<TypeParam>(0.0);
    weights(1, 0, 2, 0) = static_cast<TypeParam>(-1.0);
    weights(1, 0, 2, 1) = static_cast<TypeParam>(-2.0);
    weights(1, 0, 2, 2) = static_cast<TypeParam>(-1.0);

    // Output channel 1, input channel 1: Zero
    std::fill(&weights(1, 1, 0, 0), &weights(1, 1, 0, 0) + 9, static_cast<TypeParam>(0));

    no_pad_kernel->load_weights(std::move(weights));

    // Create a 5x5 input (larger to see boundary effects with no padding)
    std::vector<size_t> input_dims = {1, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims);

    // First channel: grid of increasing values
    for (size_t h = 0; h < 5; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            input(0, 0, h, w) = static_cast<TypeParam>(h * 5 + w + 1);
        }
    }

    // Second channel: constant value of 5
    for (size_t h = 0; h < 5; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            input(0, 1, h, w) = static_cast<TypeParam>(5.0);
        }
    }

    // Print input values for first channel
    std::cout << "\nInput Channel 0:\n";
    for (size_t h = 0; h < 5; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            std::cout << input(0, 0, h, w) << " ";
        }
        std::cout << "\n";
    }

    // Print input values for second channel
    std::cout << "\nInput Channel 1:\n";
    for (size_t h = 0; h < 5; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            std::cout << input(0, 1, h, w) << " ";
        }
        std::cout << "\n";
    }

    // Perform forward pass
    Tensor<TypeParam> output = no_pad_kernel->forward(input);

    // Verify output dimensions - with no padding, output should be (input_size - kernel_size + 1)
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 1u);
    ASSERT_EQ(out_shape[1], 2u);
    ASSERT_EQ(out_shape[2], 3u); // 5 - 3 + 1 = 3
    ASSERT_EQ(out_shape[3], 3u); // 5 - 3 + 1 = 3

    // Print the output values for debugging
    std::cout << "\nOutput Channel 0:\n";
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            std::cout << output(0, 0, h, w) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nOutput Channel 1:\n";
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            std::cout << output(0, 1, h, w) << " ";
        }
        std::cout << "\n";
    }

    // ======= Manually calculate expected values for specific positions =======

    // Position (0,0) - top-left of output
    // For output channel 0:
    //   From input channel 0:
    //     Only center weight is non-zero:
    //     weights(0,0,1,1) * input(0,0,1,1) = 1.0 * 7 = 7
    //   From input channel 1:
    //     All weights are 0.1:
    //     Sum of (0.1 * 5) for 9 positions = 0.1 * 5 * 9 = 4.5
    //   Total: 7 + 4.5 = 11.5
    TypeParam expected_0_0_0 = static_cast<TypeParam>(11.5);

    // Position (1,1) - center of output
    // For output channel 0:
    //   From input channel 0:
    //     weights(0,0,1,1) * input(0,0,2,2) = 1.0 * 13 = 13
    //   From input channel 1:
    //     weights(0,1,*,*) * input(0,1,*,*) = 0.1 * 5 * 9 = 4.5
    //   Total: 13 + 4.5 = 17.5
    TypeParam expected_0_1_1 = static_cast<TypeParam>(17.5);

    // Position (2,2) - bottom-right of output
    // For output channel 0:
    //   From input channel 0:
    //     weights(0,0,1,1) * input(0,0,3,3) = 1.0 * 19 = 19
    //   From input channel 1:
    //     weights(0,1,*,*) * input(0,1,*,*) = 0.1 * 5 * 9 = 4.5
    //   Total: 19 + 4.5 = 23.5
    TypeParam expected_0_2_2 = static_cast<TypeParam>(23.5);

    // Position (0,0) - top-left of output
    // For output channel 1 (horizontal edge detector):
    //   From input channel 0:
    //     Top row: weights(1,0,0,*) * input(0,0,0,*)
    //       = 1.0*1 + 2.0*2 + 1.0*3 = 1 + 4 + 3 = 8
    //     Bottom row: weights(1,0,2,*) * input(0,0,2,*)
    //       = -1.0*11 + -2.0*12 + -1.0*13 = -11 - 24 - 13 = -48
    //   Total: 8 - 48 = -40
    TypeParam expected_1_0_0 = static_cast<TypeParam>(-40.0);

    // Check specific manually calculated points
    const TypeParam tolerance = std::is_same<TypeParam, float>::value
                                    ? static_cast<TypeParam>(1e-4)
                                    : static_cast<TypeParam>(1e-10);

    EXPECT_NEAR(output(0, 0, 0, 0), expected_0_0_0, tolerance)
        << "Validation failed at position (0,0,0,0)";

    EXPECT_NEAR(output(0, 0, 1, 1), expected_0_1_1, tolerance)
        << "Validation failed at position (0,0,1,1)";

    EXPECT_NEAR(output(0, 0, 2, 2), expected_0_2_2, tolerance)
        << "Validation failed at position (0,0,2,2)";

    EXPECT_NEAR(output(0, 1, 0, 0), expected_1_0_0, tolerance)
        << "Validation failed at position (0,1,0,0)";
}

TYPED_TEST(Conv2DTest, SimpleConvolutionWithSobel) {
    // Create Sobel kernel with padding=1
    auto sobel_kernel = make_conv2d<TypeParam>(3, 3, 1, 1, 1, 1);

    // Initialize Sobel operator weights (horizontal edge detector)
    std::vector<size_t> weight_dims = {1, 1, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);

    // Horizontal Sobel operator
    const TypeParam sobel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // Load weights directly into tensor
    for (size_t i = 0; i < 9; ++i) {
        weights.data()[i] = sobel[i];
    }

    sobel_kernel->load_weights(std::move(weights));

    // Create a simple gradient test pattern
    std::vector<size_t> input_dims = {1, 1, 5, 5};
    Tensor<TypeParam>   input(input_dims);

    // Fill with a vertical gradient (rows of identical values)
    for (size_t h = 0; h < 5; ++h) {
        TypeParam row_value = static_cast<TypeParam>(h + 1);
        for (size_t w = 0; w < 5; ++w) {
            input(0, 0, h, w) = row_value;
        }
    }

    // Perform convolution
    Tensor<TypeParam> output = sobel_kernel->forward(input);

    // Verify dimensions
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 1u);
    ASSERT_EQ(out_shape[1], 1u);
    ASSERT_EQ(out_shape[2], 5u);
    ASSERT_EQ(out_shape[3], 5u);

    // For a vertical gradient with constant rows, a horizontal Sobel
    // should produce a consistent value of 8 for the middle section
    // (Each row has a difference of 2 from row+1 and row-1, so sum = 8)
    for (size_t w = 1; w < 4; ++w) {
        EXPECT_NEAR(output(0, 0, 2, w), static_cast<TypeParam>(8.0), static_cast<TypeParam>(1e-4))
            << "Horizontal Sobel should produce value of 8 for vertical gradient at position (2,"
            << w << ")";
    }
}

TYPED_TEST(Conv2DTest, ZeroInput) {
    // Create zero-filled input
    std::vector<size_t> input_dims = {1, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims);
    std::fill(input.data(), input.data() + input.size(), static_cast<TypeParam>(0));

    // Forward pass
    Tensor<TypeParam> output = this->kernel->forward(input);

    // Verify zero output
    const TypeParam* output_data = output.data();
    const TypeParam  tolerance   = std::is_same<TypeParam, float>::value
                                       ? static_cast<TypeParam>(1e-5)
                                       : static_cast<TypeParam>(1e-10);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output_data[i], TypeParam(0), tolerance) << "Non-zero output at index " << i;
    }
}

TYPED_TEST(Conv2DTest, BatchProcessing) {
    // Create multi-batch input with simpler patterns to reduce numerical issues
    std::vector<size_t> input_dims = {2, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims);

    // Initialize with different patterns for each batch
    for (size_t b = 0; b < 2; ++b) {
        for (size_t c = 0; c < 2; ++c) {
            for (size_t h = 0; h < 5; ++h) {
                for (size_t w = 0; w < 5; ++w) {
                    // First batch: simple increasing pattern
                    // Second batch: constant pattern
                    if (b == 0) {
                        input(b, c, h, w) = static_cast<TypeParam>((c + 1) * (h + 1));
                    } else {
                        input(b, c, h, w) = static_cast<TypeParam>((c + 1) * 5.0);
                    }
                }
            }
        }
    }

    // Forward pass
    Tensor<TypeParam> output = this->kernel->forward(input);

    // Verify batch dimension
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 2u);
    ASSERT_EQ(out_shape[1], 2u);

    // Verify the outputs are different between batches - just checking a few key points
    EXPECT_NE(output(0, 0, 2, 2), output(1, 0, 2, 2))
        << "Output for different batches should be different at center point";

    // Also check corner point
    EXPECT_NE(output(0, 0, 0, 0), output(1, 0, 0, 0))
        << "Output for different batches should be different at corner point";
}

TYPED_TEST(Conv2DTest, InvalidInput) {
    // Test invalid dimension count
    std::vector<size_t> invalid_dims = {5, 5, 2};
    Tensor<TypeParam>   invalid_rank(invalid_dims);
    EXPECT_THROW(this->kernel->forward(invalid_rank), std::runtime_error);

    // Test invalid channel count
    std::vector<size_t> invalid_channels = {1, 3, 5, 5};
    Tensor<TypeParam>   wrong_channels(invalid_channels);
    EXPECT_THROW(this->kernel->forward(wrong_channels), std::runtime_error);

    // Test invalid spatial dimensions
    std::vector<size_t> invalid_spatial = {1, 2, 1, 1};
    Tensor<TypeParam>   small_spatial(invalid_spatial);
    EXPECT_THROW(this->kernel->forward(small_spatial), std::runtime_error);
}

TYPED_TEST(Conv2DTest, PaddingBehavior) {
    // Create a kernel with padding=1
    auto conv2d = make_conv2d<TypeParam>(3, // kernel_height
                                         3, // kernel_width
                                         1, // in_channels
                                         1, // out_channels
                                         1, // stride
                                         1  // padding
    );

    // Create asymmetric kernel weights to properly test padding
    std::vector<size_t> weight_dims = {1, 1, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);

    // Initialize weights to be non-zero only in one corner
    // This ensures padding effects are visible
    std::fill(weights.data(), weights.data() + weights.size(), static_cast<TypeParam>(0));
    weights(0, 0, 0, 0) = static_cast<TypeParam>(1.0); // Only top-left is 1, rest are 0

    conv2d->load_weights(std::move(weights));

    // Create a 3x3 input with distinct values
    Tensor<TypeParam> input({1, 1, 3, 3});

    // Fill with increasing values
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            input(0, 0, h, w) = static_cast<TypeParam>(h * 3 + w + 1);
        }
    }

    // Perform convolution
    Tensor<TypeParam> output = conv2d->forward(input);

    // Verify dimensions - with padding, output should be same size as input
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 1u);
    ASSERT_EQ(out_shape[1], 1u);
    ASSERT_EQ(out_shape[2], 3u);
    ASSERT_EQ(out_shape[3], 3u);

    // With our corner-only kernel (weight at 0,0):
    // - When kernel center is at (2,2), the 0,0 weight aligns with input position (1,1)
    // - So output(0,0,2,2) should match input(0,0,1,1) which is 5
    EXPECT_NEAR(output(0, 0, 2, 2), input(0, 0, 1, 1), static_cast<TypeParam>(1e-5))
        << "Bottom-right output should match middle input with this kernel";

    // Top-left output should have zeros due to padding
    EXPECT_NEAR(output(0, 0, 0, 0), static_cast<TypeParam>(0), static_cast<TypeParam>(1e-5))
        << "Top-left output should be approximately zero due to padding";
}

TYPED_TEST(Conv2DTest, StrideTest) {
    // Create a kernel with stride=2
    auto stride_kernel = make_conv2d<TypeParam>(3, // kernel_height
                                                3, // kernel_width
                                                1, // in_channels
                                                1, // out_channels
                                                2, // stride
                                                1  // padding
    );

    // Create identity filter
    std::vector<size_t> weight_dims = {1, 1, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);

    // Initialize weights as identity kernel
    std::fill(weights.data(), weights.data() + weights.size(), static_cast<TypeParam>(0));
    weights(0, 0, 1, 1) = static_cast<TypeParam>(1); // Only center element is 1

    stride_kernel->load_weights(std::move(weights));

    // Create a simple 5x5 input
    Tensor<TypeParam> input({1, 1, 5, 5});

    // Fill input with unique values
    for (size_t h = 0; h < 5; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            input(0, 0, h, w) = static_cast<TypeParam>(h * 5 + w + 1);
        }
    }

    // Perform convolution
    Tensor<TypeParam> output = stride_kernel->forward(input);

    // Verify output dimensions with stride=2
    // Should be ceil((5 + 2*1 - 3)/2 + 1) = ceil(2.5) = 3
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 1u);
    ASSERT_EQ(out_shape[1], 1u);
    ASSERT_EQ(out_shape[2], 3u);
    ASSERT_EQ(out_shape[3], 3u);

    // With identity kernel and stride=2, output should match strided input
    EXPECT_NEAR(output(0, 0, 1, 1), input(0, 0, 2, 2), static_cast<TypeParam>(1e-5))
        << "Stride sampling should match input at center";
}

TYPED_TEST(Conv2DTest, PerformanceTest) {
    // Create larger tensors to test performance
    std::vector<size_t> input_dims = {2, 4, 16, 16};
    Tensor<TypeParam>   input(input_dims);

    // Initialize with simple pattern for deterministic testing
    for (size_t b = 0; b < 2; ++b) {
        for (size_t c = 0; c < 4; ++c) {
            for (size_t h = 0; h < 16; ++h) {
                for (size_t w = 0; w < 16; ++w) {
                    input(b, c, h, w) = static_cast<TypeParam>((c + 1) * (h + w + 1) * 0.1);
                }
            }
        }
    }

    // Create a medium kernel
    auto perf_kernel = make_conv2d<TypeParam>(3, // kernel_height
                                              3, // kernel_width
                                              4, // in_channels
                                              8, // out_channels
                                              1, // stride
                                              1  // padding
    );

    // Initialize with simple pattern
    std::vector<size_t> weight_dims = {8, 4, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);

    // Use small values to avoid numerical issues
    std::fill(weights.data(), weights.data() + weights.size(), static_cast<TypeParam>(0.01));

    perf_kernel->load_weights(std::move(weights));

    // Measure execution time (simple approach)
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform forward pass
    Tensor<TypeParam> output = perf_kernel->forward(input);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Verify output dimensions
    const auto& out_shape = output.shape();
    ASSERT_EQ(out_shape[0], 2u);
    ASSERT_EQ(out_shape[1], 8u);
    ASSERT_EQ(out_shape[2], 16u);
    ASSERT_EQ(out_shape[3], 16u);

    // Verify the output is valid
    const TypeParam* output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i])) << "Output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output_data[i])) << "Output contains Inf at index " << i;
    }

    // Performance information (not part of actual test)
    std::cout << "Conv2D Performance Test completed in " << duration << " ms" << std::endl;
}

// Test fixture for both CPU and CUDA tests
template <typename T>
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, int, unsigned int>;
TYPED_TEST_SUITE(TensorTest, TestTypes);

TYPED_TEST(TensorTest, DefaultConstruction){// CPU Test
                                            {Tensor<TypeParam> t;
EXPECT_EQ(t.rank(), 0);
EXPECT_EQ(t.size(), 0);
EXPECT_EQ(t.data(), nullptr);
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    cuda::TensorWrapper<TypeParam> t; // Use default constructor
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.tensor_data().data, nullptr);
}
#endif
}

TYPED_TEST(TensorTest, SimpleConstruction){// CPU Test
                                           {Tensor<TypeParam> t({2, 3, 4});
EXPECT_EQ(t.rank(), 3);
EXPECT_EQ(t.size(), 24);
EXPECT_NE(t.data(), nullptr);
EXPECT_TRUE(t.is_contiguous());
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    cuda::TensorWrapper<TypeParam> t({2, 3, 4}); // Now works with initializer list
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 24);
    EXPECT_NE(t.tensor_data().data, nullptr);
    EXPECT_TRUE(t.is_contiguous());
}
#endif
}

TYPED_TEST(TensorTest, AccessOperators){// CPU Test
                                        {Tensor<TypeParam> t({2, 3});
t(0, 0) = TypeParam(1);
t(0, 1) = TypeParam(2);
t(0, 2) = TypeParam(3);
t(1, 0) = TypeParam(4);
t(1, 1) = TypeParam(5);
t(1, 2) = TypeParam(6);

EXPECT_EQ(t(0, 0), TypeParam(1));
EXPECT_EQ(t(0, 1), TypeParam(2));
EXPECT_EQ(t(0, 2), TypeParam(3));
EXPECT_EQ(t(1, 0), TypeParam(4));
EXPECT_EQ(t(1, 1), TypeParam(5));
EXPECT_EQ(t(1, 2), TypeParam(6));
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    Tensor<TypeParam> host_t({2, 3});
    host_t(0, 0) = TypeParam(1);
    host_t(0, 1) = TypeParam(2);
    host_t(0, 2) = TypeParam(3);
    host_t(1, 0) = TypeParam(4);
    host_t(1, 1) = TypeParam(5);
    host_t(1, 2) = TypeParam(6);

    cuda::TensorWrapper<TypeParam> device_t(host_t);
    Tensor<TypeParam>              result({2, 3});
    device_t.copy_to_host(result);

    EXPECT_EQ(result(0, 0), TypeParam(1));
    EXPECT_EQ(result(0, 1), TypeParam(2));
    EXPECT_EQ(result(0, 2), TypeParam(3));
    EXPECT_EQ(result(1, 0), TypeParam(4));
    EXPECT_EQ(result(1, 1), TypeParam(5));
    EXPECT_EQ(result(1, 2), TypeParam(6));
}
#endif
}

TYPED_TEST(TensorTest, MemoryLayouts){// CPU Test
                                      {Tensor<TypeParam> row_major({2, 3}, MemoryLayout::RowMajor);
Tensor<TypeParam> col_major({2, 3}, MemoryLayout::ColumnMajor);

EXPECT_EQ(row_major.strides()[0], 3);
EXPECT_EQ(row_major.strides()[1], 1);

EXPECT_EQ(col_major.strides()[0], 1);
EXPECT_EQ(col_major.strides()[1], 2);
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    cuda::TensorWrapper<TypeParam> row_major({2, 3}, cuda::MemoryLayout::RowMajor);
    cuda::TensorWrapper<TypeParam> col_major({2, 3}, cuda::MemoryLayout::ColumnMajor);

    EXPECT_EQ(row_major.tensor_data().strides[0], 3);
    EXPECT_EQ(row_major.tensor_data().strides[1], 1);

    EXPECT_EQ(col_major.tensor_data().strides[0], 1);
    EXPECT_EQ(col_major.tensor_data().strides[1], 2);
}
#endif
}

TYPED_TEST(TensorTest, ZeroOperation){// CPU Test
                                      {Tensor<TypeParam> t({2, 2});
t(0, 0) = TypeParam(1);
t(0, 1) = TypeParam(2);
t(1, 0) = TypeParam(3);
t(1, 1) = TypeParam(4);

t.zero();

EXPECT_EQ(t(0, 0), TypeParam(0));
EXPECT_EQ(t(0, 1), TypeParam(0));
EXPECT_EQ(t(1, 0), TypeParam(0));
EXPECT_EQ(t(1, 1), TypeParam(0));
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    Tensor<TypeParam> host_t({2, 2});
    host_t(0, 0) = TypeParam(1);
    host_t(0, 1) = TypeParam(2);
    host_t(1, 0) = TypeParam(3);
    host_t(1, 1) = TypeParam(4);

    cuda::TensorWrapper<TypeParam> device_t(host_t);
    device_t.zero();

    Tensor<TypeParam> result({2, 2});
    device_t.copy_to_host(result);

    EXPECT_EQ(result(0, 0), TypeParam(0));
    EXPECT_EQ(result(0, 1), TypeParam(0));
    EXPECT_EQ(result(1, 0), TypeParam(0));
    EXPECT_EQ(result(1, 1), TypeParam(0));
}
#endif
}

TYPED_TEST(TensorTest, MoveOperations){// CPU Test
                                       {Tensor<TypeParam> t1({2, 2});
t1(0, 0) = TypeParam(1);
t1(0, 1) = TypeParam(2);
t1(1, 0) = TypeParam(3);
t1(1, 1) = TypeParam(4);

Tensor<TypeParam> t2(std::move(t1));
EXPECT_EQ(t1.data(), nullptr);
EXPECT_EQ(t1.size(), 0);
EXPECT_EQ(t2(0, 0), TypeParam(1));
EXPECT_EQ(t2(1, 1), TypeParam(4));
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    Tensor<TypeParam> host_t({2, 2});
    host_t(0, 0) = TypeParam(1);
    host_t(0, 1) = TypeParam(2);
    host_t(1, 0) = TypeParam(3);
    host_t(1, 1) = TypeParam(4);

    cuda::TensorWrapper<TypeParam> device_t1(host_t);
    cuda::TensorWrapper<TypeParam> device_t2(std::move(device_t1));

    EXPECT_EQ(device_t1.tensor_data().data, nullptr);
    EXPECT_EQ(device_t1.size(), 0);

    Tensor<TypeParam> result({2, 2});
    device_t2.copy_to_host(result);
    EXPECT_EQ(result(0, 0), TypeParam(1));
    EXPECT_EQ(result(1, 1), TypeParam(4));
}
#endif
}

// Additional CUDA-specific tests
#ifdef CUDA_ENABLED
TYPED_TEST(TensorTest, StreamOperations) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor<TypeParam> host_t({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            host_t(i, j) = TypeParam(i * 3 + j);
        }
    }

    cuda::TensorWrapper<TypeParam> cuda_wrapper(host_t);
    Tensor<TypeParam>              result({2, 3});

    // Perform stream operations
    cuda_wrapper.zero();
    cuda_wrapper.copy_from_host(host_t);
    cuda_wrapper.copy_to_host(result);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    // Verify results
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(result(i, j), host_t(i, j));
        }
    }
}

TYPED_TEST(TensorTest, ConcurrentOperations) {
    constexpr size_t                            num_tensors = 4;
    std::vector<cuda::TensorWrapper<TypeParam>> device_tensors;
    std::vector<Tensor<TypeParam>>              host_tensors;
    std::vector<Tensor<TypeParam>>              results;

    // Initialize tensors
    for (size_t i = 0; i < num_tensors; ++i) {
        host_tensors.emplace_back(std::vector<size_t>{2, 2});
        results.emplace_back(std::vector<size_t>{2, 2});
        device_tensors.emplace_back(host_tensors[i]);

        for (size_t j = 0; j < 4; ++j) {
            host_tensors[i].data()[j] = TypeParam(i * 4 + j);
        }
    }

    // Perform concurrent operations
    #pragma omp parallel for
    for (size_t i = 0; i < num_tensors; ++i) {
        device_tensors[i].copy_from_host(host_tensors[i]);
        device_tensors[i].zero();
        device_tensors[i].copy_from_host(host_tensors[i]);
        device_tensors[i].copy_to_host(results[i]);
    }

    // Verify results
    for (size_t i = 0; i < num_tensors; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(results[i].data()[j], host_tensors[i].data()[j]);
        }
    }
}

TYPED_TEST(TensorTest, LayoutConversion) {
    Tensor<TypeParam> host_t({3, 3}, MemoryLayout::RowMajor);
    for (size_t i = 0; i < 9; ++i) {
        host_t.data()[i] = TypeParam(i);
    }

    // Create column-major device tensor
    cuda::TensorWrapper<TypeParam> device_t({3, 3}, cuda::MemoryLayout::ColumnMajor);
    device_t.copy_from_host(host_t);

    Tensor<TypeParam> result({3, 3}, MemoryLayout::ColumnMajor);
    device_t.copy_to_host(result);

    // Verify data matches despite different layouts
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(result(i, j), host_t(i, j));
        }
    }
}
#endif

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}