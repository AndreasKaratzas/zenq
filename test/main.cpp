#include "compute/cpp/kernels/conv2d.hpp"
#include "compute/cpp/tensor.hpp"
#include "compute/cpp/view.hpp"
#ifdef CUDA_ENABLED
    #include "compute/cuda/wrapper.hpp"
#endif

#include <cmath> // For std::isnan, std::isinf
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
        KernelDescriptor desc(KernelType::Convolution2D);
        desc.set_param("kernel_height",
                       static_cast<size_t>(3)); // Use string literals and static_cast
        desc.set_param("kernel_width",
                       static_cast<size_t>(3)); // Use string literals and static_cast
        desc.set_param("stride", static_cast<size_t>(1));
        desc.set_param("padding", static_cast<size_t>(1));
        desc.set_param("in_channels", static_cast<size_t>(2));
        desc.set_param("out_channels", static_cast<size_t>(2));

        // Create kernel instance using make_conv2d
        kernel = make_conv2d<T>(3, 3, 2, 2, 1, 1); // Use make_conv2d factory function

        // Initialize kernel weights with known values
        std::vector<size_t> weight_dims = {2, 2, 3, 3};
        Tensor<T>           weights(weight_dims); // Using vector for dims now
        T*                  w_data = weights.data();
        for (size_t i = 0; i < weights.size(); ++i) {
            w_data[i] = static_cast<T>(i + 1) *
                        static_cast<T>(0.1); // static_cast for mixed type operations
        }
        kernel->load_weights(std::move(weights));
    }

    std::unique_ptr<BaseKernel<T>> kernel;
};

using ConvTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Conv2DTest, ConvTypes);

TYPED_TEST(Conv2DTest, BasicForwardPass) {
    // Create input tensor with known values
    std::vector<size_t> input_dims = {1, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims); // Using vector for dims
    TypeParam*          input_data = input.data();
    for (size_t i = 0; i < input.size(); ++i) {
        input_data[i] = static_cast<TypeParam>(i) *
                        static_cast<TypeParam>(0.1); // static_cast for mixed type operations
    }

    // Perform forward pass
    Tensor<TypeParam> output = this->kernel->forward(input);

    // Verify output dimensions
    const auto&         out_dims_array = output.dims();
    std::vector<size_t> out_dims(out_dims_array.begin(), out_dims_array.begin() + output.rank());

    EXPECT_EQ(out_dims[0], 1); // batch size
    EXPECT_EQ(out_dims[1], 2); // output channels
    EXPECT_EQ(out_dims[2], 5); // height
    EXPECT_EQ(out_dims[3], 5); // width

    // Check output validity
    const TypeParam* output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i])) << "Output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output_data[i])) << "Output contains Inf at index " << i;
    }
}

TYPED_TEST(Conv2DTest, SimpleConvolutionWithSobel) {
    // Create Sobel kernel descriptor
    KernelDescriptor desc(KernelType::Convolution2D);
    desc.set_param("kernel_height", static_cast<size_t>(3));
    desc.set_param("kernel_width", static_cast<size_t>(3));
    desc.set_param("stride", static_cast<size_t>(1));
    desc.set_param("padding", static_cast<size_t>(1));
    desc.set_param("in_channels", static_cast<size_t>(1));
    desc.set_param("out_channels", static_cast<size_t>(1));

    // Create kernel instance
    auto sobel_kernel = make_conv2d<TypeParam>(3, 3, 1, 1, 1, 1);

    // Initialize Sobel operator weights
    std::vector<size_t> weight_dims = {1, 1, 3, 3};
    Tensor<TypeParam>   weights(weight_dims);
    TypeParam*          w_data  = weights.data();
    const TypeParam     sobel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1}; // Horizontal Sobel
    std::copy(sobel, sobel + 9, w_data);
    sobel_kernel->load_weights(std::move(weights));

    // Create input tensor
    std::vector<size_t> input_dims = {1, 1, 3, 3};
    Tensor<TypeParam>   input(input_dims);
    TypeParam*          input_data = input.data();
    for (size_t i = 0; i < input.size(); ++i) {
        input_data[i] = static_cast<TypeParam>(i + 1);
    }

    // Perform convolution
    Tensor<TypeParam> output = sobel_kernel->forward(input);

    // Verify dimensions
    const auto&         out_dims_array = output.dims();
    std::vector<size_t> out_dims(out_dims_array.begin(), out_dims_array.begin() + output.rank());
    ASSERT_EQ(out_dims[0], 1);
    ASSERT_EQ(out_dims[1], 1);
    ASSERT_EQ(out_dims[2], 3);
    ASSERT_EQ(out_dims[3], 3);

    // Verify Sobel operator results
    const TypeParam  expected[]  = {-9, -6, 9, -20, -8, 20, -21, -6, 21}; // Updated expected values
    const TypeParam* output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected[i], TypeParam(1e-5)) << "Mismatch at index " << i;
    }
}

TYPED_TEST(Conv2DTest, ZeroInput) {
    // Create zero-filled input
    std::vector<size_t> input_dims = {1, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims);
    input.zero();

    // Forward pass
    Tensor<TypeParam> output = this->kernel->forward(input);

    // Verify zero output
    const TypeParam* output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output_data[i], TypeParam(0), TypeParam(1e-6))
            << "Non-zero output at index " << i;
    }
}

TYPED_TEST(Conv2DTest, BatchProcessing) {
    // Create multi-batch input
    std::vector<size_t> input_dims = {2, 2, 5, 5};
    Tensor<TypeParam>   input(input_dims);
    TypeParam*          input_data = input.data();
    for (size_t i = 0; i < input.size(); ++i) {
        input_data[i] = static_cast<TypeParam>(i) * static_cast<TypeParam>(0.1);
    }

    // Forward pass
    Tensor<TypeParam> output = this->kernel->forward(input);

    // Verify batch dimension
    const auto&         out_dims_array = output.dims();
    std::vector<size_t> out_dims(out_dims_array.begin(), out_dims_array.begin() + output.rank());
    ASSERT_EQ(out_dims[0], 2);

    // Check batch independence
    const TypeParam* output_data  = output.data();
    const size_t     batch_stride = out_dims[1] * out_dims[2] * out_dims[3];

    // First elements of each batch should differ
    EXPECT_NE(output_data[0], output_data[batch_stride]);

    // Check corresponding elements differ between batches
    for (size_t i = 0; i < batch_stride; ++i) {
        EXPECT_NE(output_data[i], output_data[batch_stride + i])
            << "Batch elements identical at offset " << i;
    }
}

TYPED_TEST(Conv2DTest, InvalidInput) {
    // Test invalid dimension count
    std::vector<size_t> invalid_dims = {5, 5, 2};
    Tensor<TypeParam>   invalid_rank(invalid_dims);
    EXPECT_THROW(this->kernel->forward(invalid_rank),
                 std::runtime_error); // Expect runtime_error now

    // Test invalid channel count
    std::vector<size_t> invalid_channels = {1, 3, 5, 5};
    Tensor<TypeParam>   wrong_channels(invalid_channels);
    EXPECT_THROW(this->kernel->forward(wrong_channels),
                 std::runtime_error); // Expect runtime_error now

    // Test invalid spatial dimensions (not directly tested by validate_input anymore, but can keep
    // for logical coverage if needed)
    std::vector<size_t> invalid_spatial = {
        1, 2, 1, 1}; // Input smaller than kernel - removed zero dim test
    Tensor<TypeParam> small_spatial(invalid_spatial);
    EXPECT_THROW(this->kernel->forward(small_spatial),
                 std::runtime_error); // Expect runtime_error now
}

TYPED_TEST(Conv2DTest, PaddingBehavior) {
    using TypeParam = TypeParam;
    // Test case setup
    KernelDescriptor desc(KernelType::Convolution2D);
    desc.set_param("kernel_height", size_t(3));
    desc.set_param("kernel_width", size_t(3));
    desc.set_param("in_channels", size_t(2));
    desc.set_param("out_channels", size_t(2));
    desc.set_param("stride", size_t(1));
    desc.set_param("padding", size_t(1));

    auto conv2d = make_conv2d<TypeParam>(desc.get_param<size_t>("kernel_height"),
                                         desc.get_param<size_t>("kernel_width"),
                                         desc.get_param<size_t>("in_channels"),
                                         desc.get_param<size_t>("out_channels"),
                                         desc.get_param<size_t>("stride"),
                                         desc.get_param<size_t>("padding"));

    Tensor<TypeParam> input({1, 2, 3, 3}, MemoryLayout::RowMajor);
    // Initialize input with different values for each channel
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            input(0, 0, h, w) = 1.0; // Channel 0 values are 1
            input(0, 1, h, w) = 2.0; // Channel 1 values are 2
        }
    }

    std::cout << "\n--- Input Tensor in PaddingBehavior Test (Before Forward) ---" << std::endl;
    std::cout << "Input Shape: ";
    for (size_t dim : input.shape())
        std::cout << dim << " ";
    std::cout << std::endl;
    std::cout << "Channel 0 (Slice 0-2, 0-2):" << std::endl;
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            std::cout << input(0, 0, h, w) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Channel 1 (Slice 0-2, 0-2):" << std::endl;
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            std::cout << input(0, 1, h, w) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--- End Input Tensor Print ---" << std::endl;

    Tensor<TypeParam> output = conv2d->forward(input);

    // ... (rest of the test - dimension checks, index calculation, EXPECT_GT) ...
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