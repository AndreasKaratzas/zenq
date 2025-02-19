#include "backend/common/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace hpc;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code
    }
};

TEST_F(TensorTest, ConstructorTest) {
    std::vector<size_t> shape = {2, 3, 4};
    Tensor              tensor(shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.device(), DeviceType::CPU);
}

TEST_F(TensorTest, CopyConstructorTest) {
    std::vector<size_t> shape = {2, 3};
    Tensor              original(shape);

    // Modify original data
    float* data = static_cast<float*>(original.data());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    Tensor copy(original);

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());

    // Verify data was copied correctly
    float* copy_data = static_cast<float*>(copy.data());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(copy_data[i], static_cast<float>(i));
    }
}

TEST_F(TensorTest, MoveConstructorTest) {
    std::vector<size_t> shape = {2, 3};
    Tensor              original(shape);
    void*               original_data = original.data();

    Tensor moved(std::move(original));

    EXPECT_EQ(moved.data(), original_data);
    EXPECT_EQ(original.data(), nullptr);
}

TEST_F(TensorTest, AlignmentTest) {
    std::vector<size_t> shape = {16};
    Tensor              tensor(shape);

    // Check if the memory is properly aligned
    uintptr_t addr = reinterpret_cast<uintptr_t>(tensor.data());
    EXPECT_EQ(addr % ALIGNMENT, 0);
}

TEST_F(TensorTest, AVXAdditionTest) {
    std::vector<size_t> shape = {16}; // Multiple of AVX512 register size
    Tensor              a(shape);
    Tensor              b(shape);

    // Initialize data
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());

    for (size_t i = 0; i < 16; ++i) {
        a_data[i] = static_cast<float>(i);
        b_data[i] = static_cast<float>(i * 2);
    }

    a.add_avx(b);

    // Verify results
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(a_data[i], static_cast<float>(i + i * 2));
    }
}

TEST_F(TensorTest, ZeroTest) {
    std::vector<size_t> shape = {2, 3};
    Tensor              tensor(shape);

    // Initialize with non-zero values
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    tensor.zero_();

    // Verify all elements are zero
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_F(TensorTest, ResizeTest) {
    std::vector<size_t> shape = {2, 3};
    Tensor              tensor(shape);

    std::vector<size_t> new_shape = {3, 4};
    tensor.resize(new_shape);

    EXPECT_EQ(tensor.shape(), new_shape);
    EXPECT_EQ(tensor.size(), 12);
}

TEST_F(TensorTest, MapTest) {
    std::vector<size_t> shape = {2, 3};
    Tensor              tensor(shape);

    // Initialize data
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Apply square function using map
    auto squared = tensor.map([](float x) { return x * x; });

    // Verify results
    float* result_data = static_cast<float*>(squared.data());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], static_cast<float>(i * i));
    }
}

#ifdef CUDA_ENABLED
TEST_F(TensorTest, CUDATransferTest) {
    std::vector<size_t> shape = {2, 3};
    Tensor              tensor(shape);

    // Initialize data
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Transfer to GPU
    tensor.to_cuda();
    EXPECT_EQ(tensor.device(), DeviceType::CUDA);

    // Transfer back to CPU
    tensor.to_cpu();
    EXPECT_EQ(tensor.device(), DeviceType::CPU);

    // Verify data is preserved
    float* cpu_data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(cpu_data[i], static_cast<float>(i));
    }
}
#endif

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}