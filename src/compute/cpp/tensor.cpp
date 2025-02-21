#include "compute/cpp/tensor.hpp"
#include <cstdlib> // For aligned_alloc and free

namespace hpc::compute {

// --- Tensor Implementation ---

template <Numeric T>
Tensor<T>::Tensor() noexcept
    : data_(nullptr), rank_(0), size_(0), layout_(MemoryLayout::RowMajor),
      order_(MemoryOrder::Native) {
    dims_.fill(0);
    strides_.fill(0);
}

template <Numeric T>
Tensor<T>::Tensor(std::initializer_list<size_type> dims, MemoryLayout layout, MemoryOrder order)
    : rank_(dims.size()), size_(1), layout_(layout), order_(order) {
    if (rank_ > MAX_DIMS) {
        throw std::runtime_error("Too many dimensions");
    }

    dims_.fill(0); // Initialize with zeros
    size_t i = 0;
    for (size_type dim : dims) {
        dims_[i++] = dim;
        size_ *= dim;
    }
    allocate();
    compute_strides();
}

template <Numeric T>
Tensor<T>::Tensor(const std::vector<size_type>& dims, MemoryLayout layout, MemoryOrder order)
    : rank_(dims.size()), size_(1), layout_(layout), order_(order) {
    if (rank_ > MAX_DIMS) {
        throw std::runtime_error("Too many dimensions");
    }

    dims_.fill(0);
    for (size_t i = 0; i < rank_; ++i) {
        dims_[i] = dims[i];
        size_ *= dims[i];
    }
    allocate();
    compute_strides();
}

template <Numeric T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : data_(other.data_), dims_(other.dims_), strides_(other.strides_), rank_(other.rank_),
      size_(other.size_), layout_(other.layout_), order_(other.order_) {
    other.data_ = nullptr;
    other.rank_ = 0;
    other.size_ = 0;
    // No need to clear other.dims_ and other.strides_
}

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate(); // Deallocate current resources

        data_    = other.data_;
        dims_    = other.dims_;
        strides_ = other.strides_;
        rank_    = other.rank_;
        size_    = other.size_;
        layout_  = other.layout_;
        order_   = other.order_;

        other.data_ = nullptr;
        other.rank_ = 0;
        other.size_ = 0;
    }
    return *this;
}

// Copy constructor
template <Numeric T>
Tensor<T>::Tensor(const Tensor& other)
    : rank_(other.rank_), size_(other.size_), layout_(other.layout_), order_(other.order_) {
    dims_    = other.dims_;
    strides_ = other.strides_;
    allocate();                                         // Allocate memory
    std::copy(other.data_, other.data_ + size_, data_); // Copy data
}

// Copy assignment operator
template <Numeric T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this != &other) {
        // Allocate memory *before* deallocating, to handle self-assignment
        // and ensure exception safety.
        Tensor temp(other);     // Use copy constructor to create a temporary
        std::swap(*this, temp); // Swap resources with the temporary.
    }
    return *this;
}

template <Numeric T>
Tensor<T>::~Tensor() {
    deallocate();
}

template <Numeric T>
typename Tensor<T>::reference Tensor<T>::at(const std::array<size_type, MAX_DIMS>& indices) {
    return data_[compute_offset(indices)];
}

template <Numeric T>
typename Tensor<T>::const_reference Tensor<T>::at(
    const std::array<size_type, MAX_DIMS>& indices) const {
    return data_[compute_offset(indices)];
}

template <Numeric T>
void Tensor<T>::zero() noexcept {
    std::fill(data_, data_ + size_, static_cast<T>(0));
}

template <Numeric T>
void Tensor<T>::reshape(const std::vector<size_type>& new_dims) {
    // Calculate new size
    size_type new_size = 1;
    for (size_type dim : new_dims) {
        new_size *= dim;
    }

    // If the sizes dont match you cant reshape
    if (new_size != size_) {
        throw std::runtime_error("New dimensions are not compatible with the current size.");
    }

    // Update Dimensions and rank
    if (new_dims.size() > MAX_DIMS) {
        throw std::runtime_error("New dimensions exceed maximum number of dimensions.");
    }
    rank_ = new_dims.size();
    dims_.fill(0);
    for (size_t i = 0; i < rank_; i++) {
        dims_[i] = new_dims[i];
    }

    // Update Strides
    compute_strides();
}

template <Numeric T>
bool Tensor<T>::is_contiguous() const noexcept {
    // A tensor is contiguous if its strides match the expected strides for
    // its layout.
    std::array<size_type, MAX_DIMS> expected_strides;
    if (layout_ == MemoryLayout::RowMajor) {
        expected_strides[rank_ - 1] = 1;
        for (long long i = rank_ - 2; i >= 0; --i) {
            expected_strides[i] = expected_strides[i + 1] * dims_[i + 1];
        }
    } else { // ColumnMajor
        expected_strides[0] = 1;
        for (size_type i = 1; i < rank_; ++i) {
            expected_strides[i] = expected_strides[i - 1] * dims_[i - 1];
        }
    }

    for (size_type i = 0; i < rank_; ++i) {
        if (strides_[i] != expected_strides[i]) {
            return false;
        }
    }
    return true;
}

template <Numeric T>
void Tensor<T>::make_contiguous() {
    if (is_contiguous()) {
        return; // Already contiguous
    }

    Tensor<T> temp(shape(), layout_, order_); // Create a new contiguous tensor

    // Basic copy
    for (size_t i = 0; i < size_; ++i) {
        temp.data_[i] = data_[i];
    }
    *this = std::move(temp); // Move the data to this tensor.
}

template <Numeric T>
void Tensor<T>::allocate() {
    if (size_ == 0) {
        data_ = nullptr; // Handle empty tensors
        return;
    }
    data_ = allocate_aligned(size_, alignment());
    if (!data_) {
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
}

template <Numeric T>
void Tensor<T>::deallocate() noexcept {
    if (data_) {
        deallocate_aligned(data_, order_, alignment());
        data_ = nullptr; // Set to nullptr after deallocation
    }
}

template <Numeric T>
void Tensor<T>::compute_strides() {
    strides_.fill(0); // Initialize with zeros

    if (layout_ == MemoryLayout::RowMajor) {
        strides_[rank_ - 1] = 1;
        for (long long i = rank_ - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * dims_[i + 1];
        }
    } else { // ColumnMajor
        strides_[0] = 1;
        for (size_type i = 1; i < rank_; ++i) {
            strides_[i] = strides_[i - 1] * dims_[i - 1];
        }
    }
}

template <Numeric T>
typename Tensor<T>::size_type Tensor<T>::compute_offset(
    const std::array<size_type, MAX_DIMS>& indices) const {
    if (rank_ == 0) { // Handle scalar case.
        return 0;
    }

    size_type offset = 0;
    for (size_type i = 0; i < rank_; ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        offset += indices[i] * strides_[i];
    }
    return offset;
}

template <Numeric T>
typename Tensor<T>::pointer Tensor<T>::allocate_aligned(std::size_t size, std::size_t alignment) {
    if (alignment == 0) { // Handle potential division by zero
        return new T[size];
    }
#ifdef _MSC_VER
    return static_cast<pointer>(_aligned_malloc(size * sizeof(T), alignment));
#else
    void* ptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
        return nullptr; // Allocation failed
    }
    return static_cast<pointer>(ptr);
#endif
}

template <Numeric T>
void Tensor<T>::deallocate_aligned(pointer p, MemoryOrder order, std::size_t align) noexcept {
    if (!p)
        return;

    if (order == MemoryOrder::Aligned64 || order == MemoryOrder::Aligned32) {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p); // Use standard free for posix_memalign
#endif
    } else {
        // For Native or Packed
        delete[] p;
    }
}

template <Numeric T>
typename Tensor<T>::size_type Tensor<T>::alignment() const noexcept {
    switch (order_) {
    case MemoryOrder::Aligned64:
        return 64;
    case MemoryOrder::Aligned32:
        return 32;
    case MemoryOrder::Packed: // Packed and Native both use natural alignment.
    case MemoryOrder::Native:
    default:
        return std::alignment_of_v<T>; // Natural alignment of the data type T
    }
}

// Explicit instantiations for common types
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;
template class Tensor<unsigned int>;
template class Tensor<long>;
template class Tensor<unsigned long>;

} // namespace hpc::compute