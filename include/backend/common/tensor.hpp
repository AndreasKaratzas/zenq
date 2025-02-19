#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CUDA_ENABLED
    #include <cuda_runtime.h>
#endif

#ifdef __AVX512F__
    #include <immintrin.h>
    #define ALIGNMENT 64
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define ALIGNMENT 32
#else
    #define ALIGNMENT 16
#endif

namespace hpc {

enum class DeviceType {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
};

class Tensor {
public:
    // Constructors
    Tensor(const std::vector<size_t>& shape,
           DataType                   dtype  = DataType::FLOAT32,
           DeviceType                 device = DeviceType::CPU);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();

    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // Element-wise operations
    template <typename Func>
    Tensor map(Func f) const;

    // SIMD optimized operations
    void add_avx(const Tensor& other);
    void multiply_avx(const Tensor& other);

// CUDA operations
#ifdef CUDA_ENABLED
    void to_cuda();
    void to_cpu();
#endif

    // Utility functions
    size_t size() const {
        return total_size_;
    }
    const std::vector<size_t>& shape() const {
        return shape_;
    }
    DataType dtype() const {
        return dtype_;
    }
    DeviceType device() const {
        return device_;
    }
    void* data() {
        return data_;
    }
    const void* data() const {
        return data_;
    }

    // Memory management
    void resize(const std::vector<size_t>& new_shape);
    void zero_();

private:
    // Helper functions
    void   allocate_memory();
    void   deallocate_memory();
    size_t calculate_size() const;
    size_t calculate_stride(size_t dim) const;

    // Memory layout
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t              total_size_;
    void*               data_;

    // Properties
    DataType   dtype_;
    DeviceType device_;
    bool       owns_data_;

    // Memory alignment helpers
    static size_t aligned_size(size_t size);
    static void*  aligned_malloc(size_t size, size_t alignment);
    static void   aligned_free(void* ptr);
};

// Template implementations
template <typename Func>
Tensor Tensor::map(Func f) const {
    Tensor result(shape_, dtype_, device_);

    if (device_ == DeviceType::CUDA) {
#ifdef CUDA_ENABLED
        // TODO: Implement CUDA kernel for map operation
        throw std::runtime_error("CUDA map not yet implemented");
#else
        throw std::runtime_error("CUDA support not enabled");
#endif
    }

    // CPU implementation with SIMD optimizations where possible
    switch (dtype_) {
    case DataType::FLOAT32: {
        float* src = static_cast<float*>(data_);
        float* dst = static_cast<float*>(result.data_);

#ifdef __AVX512F__
        // Process 16 elements at a time with AVX-512
        const size_t simd_size = 16;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m512 v = _mm512_load_ps(&src[i]);
            // Note: Some functions might not have AVX-512 equivalents
            // In such cases, we need to process elements individually
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#elif defined(__AVX2__)
        // Process 8 elements at a time with AVX2
        const size_t simd_size = 8;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m256 v = _mm256_load_ps(&src[i]);
            // Note: Some functions might not have AVX2 equivalents
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#else
        // Scalar implementation
        for (size_t i = 0; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#endif
        break;
    }
    case DataType::FLOAT64: {
        double* src = static_cast<double*>(data_);
        double* dst = static_cast<double*>(result.data_);

#ifdef __AVX512F__
        // Process 8 elements at a time with AVX-512 (double precision)
        const size_t simd_size = 8;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m512d v = _mm512_load_pd(&src[i]);
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#elif defined(__AVX2__)
        // Process 4 elements at a time with AVX2 (double precision)
        const size_t simd_size = 4;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m256d v = _mm256_load_pd(&src[i]);
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#else
        // Scalar implementation
        for (size_t i = 0; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#endif
        break;
    }
    case DataType::INT32: {
        int32_t* src = static_cast<int32_t*>(data_);
        int32_t* dst = static_cast<int32_t*>(result.data_);

#ifdef __AVX512F__
        // Process 16 elements at a time with AVX-512
        const size_t simd_size = 16;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m512i v = _mm512_load_epi32(&src[i]);
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#elif defined(__AVX2__)
        // Process 8 elements at a time with AVX2
        const size_t simd_size = 8;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(&src[i]));
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#else
        // Scalar implementation
        for (size_t i = 0; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#endif
        break;
    }
    case DataType::INT64: {
        int64_t* src = static_cast<int64_t*>(data_);
        int64_t* dst = static_cast<int64_t*>(result.data_);

#ifdef __AVX512F__
        // Process 8 elements at a time with AVX-512
        const size_t simd_size = 8;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m512i v = _mm512_load_epi64(&src[i]);
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#elif defined(__AVX2__)
        // Process 4 elements at a time with AVX2
        const size_t simd_size = 4;
        size_t       i         = 0;
        for (; i + simd_size <= total_size_; i += simd_size) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(&src[i]));
            for (size_t j = 0; j < simd_size; ++j) {
                dst[i + j] = f(src[i + j]);
            }
        }
        // Process remaining elements
        for (; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#else
        // Scalar implementation
        for (size_t i = 0; i < total_size_; ++i) {
            dst[i] = f(src[i]);
        }
#endif
        break;
    }
    default:
        throw std::runtime_error("Unsupported data type for map operation");
    }

    return result;
}

} // namespace hpc