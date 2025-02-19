#include "backend/common/tensor.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>

namespace hpc {

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device), owns_data_(true) {
    total_size_ = calculate_size();
    strides_    = std::vector<size_t>(shape.size());

    // Calculate strides
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape[i];
    }

    allocate_memory();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), total_size_(other.total_size_),
      dtype_(other.dtype_), device_(other.device_), owns_data_(true) {
    allocate_memory();
    std::memcpy(data_, other.data_, total_size_ * sizeof(float));
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      total_size_(other.total_size_), data_(other.data_), dtype_(other.dtype_),
      device_(other.device_), owns_data_(other.owns_data_) {
    other.data_      = nullptr;
    other.owns_data_ = false;
}

Tensor::~Tensor() {
    if (owns_data_ && data_ != nullptr) {
        deallocate_memory();
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (owns_data_) {
            deallocate_memory();
        }

        shape_      = other.shape_;
        strides_    = other.strides_;
        total_size_ = other.total_size_;
        dtype_      = other.dtype_;
        device_     = other.device_;
        owns_data_  = true;

        allocate_memory();
        std::memcpy(data_, other.data_, total_size_ * sizeof(float));
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_) {
            deallocate_memory();
        }

        shape_      = std::move(other.shape_);
        strides_    = std::move(other.strides_);
        total_size_ = other.total_size_;
        data_       = other.data_;
        dtype_      = other.dtype_;
        device_     = other.device_;
        owns_data_  = other.owns_data_;

        other.data_      = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

void Tensor::allocate_memory() {
    size_t element_size;
    switch (dtype_) {
    case DataType::FLOAT32:
        element_size = sizeof(float);
        break;
    case DataType::FLOAT64:
        element_size = sizeof(double);
        break;
    case DataType::INT32:
        element_size = sizeof(int32_t);
        break;
    case DataType::INT64:
        element_size = sizeof(int64_t);
        break;
    default:
        throw std::runtime_error("Unsupported data type");
    }

    size_t total_bytes = aligned_size(total_size_ * element_size);
    data_              = aligned_malloc(total_bytes, ALIGNMENT);

    if (data_ == nullptr) {
        throw std::bad_alloc();
    }
}

void Tensor::deallocate_memory() {
    if (data_ != nullptr) {
        aligned_free(data_);
        data_ = nullptr;
    }
}

size_t Tensor::calculate_size() const {
    return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::calculate_stride(size_t dim) const {
    if (dim >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return strides_[dim];
}

void Tensor::zero_() {
    if (device_ == DeviceType::CPU) {
        std::memset(data_, 0, total_size_ * sizeof(float));
    }
#ifdef CUDA_ENABLED
    else {
        // CUDA implementation
    }
#endif
}

void Tensor::resize(const std::vector<size_t>& new_shape) {
    // Store old data if we need to copy it
    void*               old_data  = data_;
    size_t              old_size  = total_size_;
    std::vector<size_t> old_shape = shape_;

    // Update shape and recalculate size
    shape_      = new_shape;
    total_size_ = calculate_size();

    // Recalculate strides
    strides_      = std::vector<size_t>(shape_.size());
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }

    // Allocate new memory
    allocate_memory();

    // Copy old data if possible
    if (old_data != nullptr) {
        size_t copy_size = std::min(old_size, total_size_) * sizeof(float);
        std::memcpy(data_, old_data, copy_size);

        // Free old memory
        if (owns_data_) {
            aligned_free(old_data);
        }
    }
}

void Tensor::to_cuda() {
    if (device_ == DeviceType::CUDA) {
        return; // Already on CUDA
    }

    // Allocate CUDA memory
    void*  cuda_data = nullptr;
    size_t bytes     = total_size_ * sizeof(float);
    cudaMalloc(&cuda_data, bytes);

    if (cuda_data == nullptr) {
        throw std::runtime_error("Failed to allocate CUDA memory");
    }

    // Copy data to GPU
    cudaMemcpy(cuda_data, data_, bytes, cudaMemcpyHostToDevice);

    // Free CPU memory
    if (owns_data_) {
        aligned_free(data_);
    }

    data_   = cuda_data;
    device_ = DeviceType::CUDA;
}

void Tensor::to_cpu() {
    if (device_ == DeviceType::CPU) {
        return; // Already on CPU
    }

    // Allocate CPU memory
    void* cpu_data = aligned_malloc(total_size_ * sizeof(float), ALIGNMENT);

    if (cpu_data == nullptr) {
        throw std::runtime_error("Failed to allocate CPU memory");
    }

    // Copy data to CPU
    cudaMemcpy(cpu_data, data_, total_size_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(data_);

    data_   = cpu_data;
    device_ = DeviceType::CPU;
}

// AVX operations
void Tensor::add_avx(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for addition");
    }

#ifdef __AVX512F__
    if (dtype_ == DataType::FLOAT32) {
        float* src1 = static_cast<float*>(data_);
        float* src2 = static_cast<float*>(other.data_);

        for (size_t i = 0; i < total_size_; i += 16) {
            __m512 a      = _mm512_load_ps(&src1[i]);
            __m512 b      = _mm512_load_ps(&src2[i]);
            __m512 result = _mm512_add_ps(a, b);
            _mm512_store_ps(&src1[i], result);
        }
    }
#elif defined(__AVX2__)
    if (dtype_ == DataType::FLOAT32) {
        float* src1 = static_cast<float*>(data_);
        float* src2 = static_cast<float*>(other.data_);

        for (size_t i = 0; i < total_size_; i += 8) {
            __m256 a      = _mm256_load_ps(&src1[i]);
            __m256 b      = _mm256_load_ps(&src2[i]);
            __m256 result = _mm256_add_ps(a, b);
            _mm256_store_ps(&src1[i], result);
        }
    }
#endif
}

// Memory alignment helpers
size_t Tensor::aligned_size(size_t size) {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

void* Tensor::aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void Tensor::aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace hpc