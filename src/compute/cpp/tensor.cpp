#include "compute/cpp/tensor.hpp"
#include "compute/cpp/view.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace hpc::compute {

template <Numeric T>
Tensor<T>::Tensor() noexcept
    : data_(nullptr), dims_{}, strides_{}, rank_(0), size_(0), layout_(MemoryLayout::RowMajor),
      order_(MemoryOrder::Aligned64) {}

template <Numeric T>
Tensor<T>::Tensor(std::initializer_list<std::size_t> dims, MemoryLayout layout, MemoryOrder order)
    : data_(nullptr), rank_(dims.size()), layout_(layout), order_(order) {
    if (rank_ > MAX_DIMS) {
        throw std::invalid_argument("Number of dimensions exceeds maximum allowed");
    }

    std::copy(dims.begin(), dims.end(), dims_.begin());
    std::fill(dims_.begin() + rank_, dims_.end(), 1);

    size_ = 1;
    for (std::size_t i = 0; i < rank_; ++i) {
        size_ *= dims_[i];
    }

    compute_strides();
    allocate();
}

template <Numeric T>
Tensor<T>::Tensor(const std::vector<std::size_t>& dims, MemoryLayout layout, MemoryOrder order)
    : data_(nullptr), rank_(dims.size()), layout_(layout), order_(order) {
    if (rank_ > MAX_DIMS) {
        throw std::invalid_argument("Number of dimensions exceeds maximum allowed");
    }

    std::copy(dims.begin(), dims.end(), dims_.begin());
    std::fill(dims_.begin() + rank_, dims_.end(), 1);

    size_ = 1;
    for (std::size_t i = 0; i < rank_; ++i) {
        size_ *= dims_[i];
    }

    compute_strides();
    allocate();
}

template <Numeric T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : data_(other.data_), dims_(other.dims_), strides_(other.strides_), rank_(other.rank_),
      size_(other.size_), layout_(other.layout_), order_(other.order_) {
    other.data_ = nullptr;
    other.rank_ = 0;
    other.size_ = 0;
}

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();

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

template <Numeric T>
Tensor<T>::~Tensor() {
    deallocate();
}

template <Numeric T>
typename Tensor<T>::reference Tensor<T>::at(const std::array<std::size_t, MAX_DIMS>& indices) {
    for (std::size_t i = 0; i < rank_; ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    return data_[compute_offset(indices)];
}

template <Numeric T>
typename Tensor<T>::const_reference Tensor<T>::at(
    const std::array<std::size_t, MAX_DIMS>& indices) const {
    for (std::size_t i = 0; i < rank_; ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    return data_[compute_offset(indices)];
}

template <Numeric T>
TensorView<T> Tensor<T>::view(std::initializer_list<std::pair<std::size_t, std::size_t>> ranges) {
    if (ranges.size() > rank_) {
        throw std::invalid_argument("Too many ranges specified for tensor rank");
    }

    std::array<std::size_t, MAX_DIMS> new_dims = dims_;
    std::array<std::size_t, MAX_DIMS> offsets{};

    std::size_t range_idx = 0;
    for (const auto& range : ranges) {
        if (range.first >= dims_[range_idx] || range.second > dims_[range_idx] ||
            range.first >= range.second) {
            throw std::out_of_range("Invalid range specified");
        }

        new_dims[range_idx] = range.second - range.first;
        offsets[range_idx]  = range.first;
        range_idx++;
    }

    std::size_t base_offset = compute_offset(offsets);
    return TensorView<T>(data_ + base_offset, new_dims, strides_, rank_, layout_);
}

template <Numeric T>
TensorView<T> Tensor<T>::slice(std::size_t dim, std::size_t index) {
    if (dim >= rank_) {
        throw std::invalid_argument("Dimension index out of range");
    }
    if (index >= dims_[dim]) {
        throw std::out_of_range("Slice index out of range");
    }

    std::array<std::size_t, MAX_DIMS> new_dims;
    std::array<std::size_t, MAX_DIMS> new_strides;
    std::size_t                       new_rank = rank_ - 1;

    std::array<std::size_t, MAX_DIMS> offset_indices{};
    offset_indices[dim]     = index;
    std::size_t base_offset = compute_offset(offset_indices);

    std::size_t j = 0;
    for (std::size_t i = 0; i < rank_; ++i) {
        if (i != dim) {
            new_dims[j]    = dims_[i];
            new_strides[j] = strides_[i];
            ++j;
        }
    }

    // Fill remaining dimensions with 1
    for (; j < MAX_DIMS; ++j) {
        new_dims[j]    = 1;
        new_strides[j] = 1;
    }

    // Create and return the view
    return TensorView<T>(data_ + base_offset, new_dims, new_strides, new_rank, layout_);
}

template <Numeric T>
std::size_t Tensor<T>::alignment() const noexcept {
    switch (order_) {
    case MemoryOrder::Aligned64:
        return 64;
    case MemoryOrder::Aligned32:
        return 32;
    case MemoryOrder::Native:
        return alignof(T);
    case MemoryOrder::Packed:
        return alignof(T);
    default:
        return alignof(T);
    }
}

template <Numeric T>
void Tensor<T>::zero() noexcept {
    if (data_) {
        std::memset(data_, 0, size_ * sizeof(T));
    }
}

template <Numeric T>
bool Tensor<T>::is_contiguous() const noexcept {
    if (rank_ <= 1)
        return true;

    std::size_t expected_stride = 1;
    if (layout_ == MemoryLayout::RowMajor) {
        for (std::size_t i = rank_; i > 0; --i) {
            if (strides_[i - 1] != expected_stride)
                return false;
            expected_stride *= dims_[i - 1];
        }
    } else { // ColumnMajor
        for (std::size_t i = 0; i < rank_; ++i) {
            if (strides_[i] != expected_stride)
                return false;
            expected_stride *= dims_[i];
        }
    }
    return true;
}

template <Numeric T>
void Tensor<T>::make_contiguous() {
    if (is_contiguous())
        return;

    pointer new_data = allocate_aligned(size_, alignment());

    std::array<std::size_t, MAX_DIMS> indices{};
    for (std::size_t i = 0; i < size_; ++i) {
        std::size_t old_offset = compute_offset(indices);
        new_data[i]            = data_[old_offset];

        for (std::size_t d = rank_; d > 0; --d) {
            if (++indices[d - 1] < dims_[d - 1])
                break;
            indices[d - 1] = 0;
        }
    }

    deallocate();
    data_ = new_data;
    compute_strides();
}

template <Numeric T>
void Tensor<T>::allocate() {
    if (size_ > 0) {
        data_ = allocate_aligned(size_, alignment());
    }
}

template <Numeric T>
void Tensor<T>::deallocate() noexcept {
    if (data_) {
        deallocate_aligned(data_, order_, alignment());
        data_ = nullptr;
    }
}

template <Numeric T>
void Tensor<T>::compute_strides() {
    if (layout_ == MemoryLayout::RowMajor) {
        strides_[rank_ - 1] = 1;
        for (std::size_t i = rank_ - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * dims_[i];
        }
    } else { // ColumnMajor
        strides_[0] = 1;
        for (std::size_t i = 1; i < rank_; ++i) {
            strides_[i] = strides_[i - 1] * dims_[i - 1];
        }
    }
}

template <Numeric T>
std::size_t Tensor<T>::compute_offset(const std::array<std::size_t, MAX_DIMS>& indices) const {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < rank_; ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}

template <Numeric T>
typename Tensor<T>::pointer Tensor<T>::allocate_aligned(std::size_t size, std::size_t alignment) {
    void*       ptr        = nullptr;
    std::size_t alloc_size = size * sizeof(T);

    if (alignment > alignof(std::max_align_t)) {
#if defined(_MSC_VER)
        ptr = _aligned_malloc(alloc_size, alignment);
        if (!ptr)
            throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, alignment, alloc_size) != 0) {
            throw std::bad_alloc();
        }
#endif
    } else {
        ptr = ::operator new (alloc_size, std::align_val_t{alignment});
    }

    return static_cast<pointer>(ptr);
}

template <Numeric T>
void Tensor<T>::deallocate_aligned(pointer p, MemoryOrder order, std::size_t align) noexcept {
    if (!p)
        return;

    if (order == MemoryOrder::Aligned64 || order == MemoryOrder::Aligned32) {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    } else {
        ::operator delete (p, std::align_val_t{align});
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