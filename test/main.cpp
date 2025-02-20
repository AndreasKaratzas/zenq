#include "compute/cpp/tensor.hpp"
#include "compute/cpp/view.hpp"
#ifdef CUDA_ENABLED
    #include "compute/cuda/wrapper.hpp"
#endif

#include <gtest/gtest.h>
#include <type_traits>

using namespace hpc::compute;

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