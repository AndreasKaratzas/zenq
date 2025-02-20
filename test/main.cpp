#include "compute/cpp/tensor.hpp"

#ifdef CUDA_ENABLED
    #include "compute/cuda/wrapper.hpp"
#endif

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

namespace {

using namespace hpc::compute;

// Test fixture
template <typename T>
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, int, unsigned int>;
TYPED_TEST_SUITE(TensorTest, TestTypes);

// Basic construction tests
TYPED_TEST(TensorTest, DefaultConstruction){// CPU Test
                                            {Tensor<TypeParam> t;
EXPECT_EQ(t.rank(), 0);
EXPECT_EQ(t.size(), 0);
EXPECT_EQ(t.data(), nullptr);
} // namespace

#ifdef CUDA_ENABLED
// CUDA Test
{
    std::vector<std::size_t>       empty;
    cuda::TensorWrapper<TypeParam> t(empty);
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
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    std::vector<std::size_t>       dims{2, 3, 4};
    cuda::TensorWrapper<TypeParam> t(dims);
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 24);
    EXPECT_NE(t.tensor_data().data, nullptr);
}
#endif
}

// Data transfer tests
#ifdef CUDA_ENABLED
TYPED_TEST(TensorTest, DataTransfer) {
    Tensor<TypeParam> host_t({2, 2});
    for (std::size_t i = 0; i < 4; ++i) {
        host_t.data()[i] = static_cast<TypeParam>(i + 1);
    }

    cuda::TensorWrapper<TypeParam> device_t(host_t);
    Tensor<TypeParam>              result({2, 2});
    device_t.copy_to_host(result);

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result.data()[i], static_cast<TypeParam>(i + 1));
    }
}
#endif

// Zero operation tests
TYPED_TEST(TensorTest, ZeroOperation){// CPU Test
                                      {Tensor<TypeParam> t({2, 2});
for (std::size_t i = 0; i < 4; ++i) {
    t.data()[i] = static_cast<TypeParam>(i + 1);
}
t.zero();
for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(t.data()[i], TypeParam(0));
}
}

#ifdef CUDA_ENABLED
// CUDA Test
{
    Tensor<TypeParam> host_t({2, 2});
    for (std::size_t i = 0; i < 4; ++i) {
        host_t.data()[i] = static_cast<TypeParam>(i + 1);
    }

    cuda::TensorWrapper<TypeParam> device_t(host_t);
    device_t.zero();

    Tensor<TypeParam> result({2, 2});
    device_t.copy_to_host(result);

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result.data()[i], TypeParam(0));
    }
}
#endif
}

// Move operation tests
TYPED_TEST(TensorTest, MoveOperations) {
    // CPU Test
    {
        Tensor<TypeParam> t1({2, 2});
        for (std::size_t i = 0; i < 4; ++i) {
            t1.data()[i] = static_cast<TypeParam>(i + 1);
        }

        Tensor<TypeParam> t2(std::move(t1));
        EXPECT_EQ(t1.data(), nullptr);
        EXPECT_EQ(t1.size(), 0);
        EXPECT_EQ(t2.data()[3], TypeParam(4));
    }

#ifdef CUDA_ENABLED
    // CUDA Test
    {
        Tensor<TypeParam> host_t({2, 2});
        for (std::size_t i = 0; i < 4; ++i) {
            host_t.data()[i] = static_cast<TypeParam>(i + 1);
        }

        cuda::TensorWrapper<TypeParam> device_t1(host_t);
        cuda::TensorWrapper<TypeParam> device_t2(std::move(device_t1));

        EXPECT_EQ(device_t1.tensor_data().data, nullptr);
        EXPECT_EQ(device_t1.size(), 0);

        Tensor<TypeParam> result({2, 2});
        device_t2.copy_to_host(result);
        EXPECT_EQ(result.data()[3], TypeParam(4));
    }
#endif
}

#ifdef CUDA_ENABLED
// CUDA-specific concurrent operations test
TYPED_TEST(TensorTest, ConcurrentOperations) {
    const std::size_t                           num_tensors = 4;
    std::vector<Tensor<TypeParam>>              host_tensors;
    std::vector<cuda::TensorWrapper<TypeParam>> device_tensors;
    std::vector<Tensor<TypeParam>>              results;

    // Initialize tensors
    for (std::size_t i = 0; i < num_tensors; ++i) {
        host_tensors.emplace_back(std::vector<std::size_t>{2, 2});
        for (std::size_t j = 0; j < 4; ++j) {
            host_tensors[i].data()[j] = static_cast<TypeParam>(i * 4 + j);
        }
        device_tensors.emplace_back(host_tensors[i]);
        results.emplace_back(std::vector<std::size_t>{2, 2});
    }

    // Perform concurrent operations
    #pragma omp parallel for
    for (std::size_t i = 0; i < num_tensors; ++i) {
        device_tensors[i].copy_from_host(host_tensors[i]);
        device_tensors[i].zero();
        device_tensors[i].copy_from_host(host_tensors[i]);
        device_tensors[i].copy_to_host(results[i]);
    }

    // Verify results
    for (std::size_t i = 0; i < num_tensors; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(results[i].data()[j], host_tensors[i].data()[j]);
        }
    }
}
#endif

} // anonymous namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}