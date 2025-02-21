#include "compute/cpp/view.hpp"
#include <stdexcept>

namespace hpc::compute {

template <Numeric T>
TensorView<T>::TensorView(pointer                                data,
                          const std::array<size_type, MAX_DIMS>& dims,
                          const std::array<size_type, MAX_DIMS>& strides,
                          size_type                              rank,
                          MemoryLayout                           layout) noexcept
    : data_(data), dims_(dims), strides_(strides), rank_(rank), layout_(layout) {}

template <Numeric T>
typename TensorView<T>::reference TensorView<T>::at(
    const std::array<size_type, MAX_DIMS>& indices) {
    for (size_type i = 0; i < rank_; ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    return data_[compute_offset(indices)];
}

template <Numeric T>
typename TensorView<T>::const_reference TensorView<T>::at(
    const std::array<size_type, MAX_DIMS>& indices) const {
    for (size_type i = 0; i < rank_; ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    return data_[compute_offset(indices)];
}
template <Numeric T>
bool TensorView<T>::is_contiguous() const noexcept {
    if (rank_ <= 1)
        return true;

    size_type expected_stride = 1;
    if (layout_ == MemoryLayout::RowMajor) {
        for (size_type i = rank_; i > 0; --i) {
            if (strides_[i - 1] != expected_stride)
                return false;
            expected_stride *= dims_[i - 1];
        }
    } else { // ColumnMajor
        for (size_type i = 0; i < rank_; ++i) {
            if (strides_[i] != expected_stride)
                return false;
            expected_stride *= dims_[i];
        }
    }
    return true;
}

template <Numeric T>
TensorView<T> TensorView<T>::view(std::initializer_list<std::pair<size_type, size_type>> ranges) {
    if (ranges.size() > rank_) {
        throw std::invalid_argument("Too many ranges specified for tensor rank");
    }

    // Create new dimensions and validate ranges
    std::array<size_type, MAX_DIMS> new_dims = dims_;
    std::array<size_type, MAX_DIMS> offsets{};

    size_type range_idx = 0;
    for (const auto& range : ranges) {
        if (range.first >= dims_[range_idx] || range.second > dims_[range_idx] ||
            range.first >= range.second) {
            throw std::out_of_range("Invalid range specified");
        }

        new_dims[range_idx] = range.second - range.first;
        offsets[range_idx]  = range.first;
        range_idx++;
    }

    // Calculate the starting data pointer based on offsets
    size_type base_offset = compute_offset(offsets);
    pointer   view_data   = data_ + base_offset;

    // Create and return the view
    return TensorView<T>(view_data, new_dims, strides_, rank_, layout_);
}

template <Numeric T>
TensorView<T> TensorView<T>::slice(size_type dim, size_type index) {
    if (dim >= rank_) {
        throw std::invalid_argument("Dimension index out of range");
    }
    if (index >= dims_[dim]) {
        throw std::out_of_range("Slice index out of range");
    }

    // Create new dimensions for the slice (reduce rank by 1)
    std::array<size_type, MAX_DIMS> new_dims;
    std::array<size_type, MAX_DIMS> new_strides;
    size_type                       new_rank = rank_ - 1;

    // Calculate offset to the start of the slice
    std::array<size_type, MAX_DIMS> offset_indices{};
    offset_indices[dim]   = index;
    size_type base_offset = compute_offset(offset_indices);

    // Copy dimensions and strides, skipping the sliced dimension
    size_type j = 0;
    for (size_type i = 0; i < rank_; ++i) {
        if (i != dim) {
            new_dims[j]    = dims_[i];
            new_strides[j] = strides_[i];
            ++j;
        }
    }

    // Fill remaining dimensions with 1
    for (; j < MAX_DIMS; ++j) {
        new_dims[j]    = 1;
        new_strides[j] = 1; // Or any other default value
    }

    // Create and return the view
    return TensorView<T>(data_ + base_offset, new_dims, new_strides, new_rank, layout_);
}

template <Numeric T>
typename TensorView<T>::size_type TensorView<T>::compute_offset(
    const std::array<size_type, MAX_DIMS>& indices) const {
    size_type offset = 0;
    for (size_type i = 0; i < rank_; ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}

// Explicit instantiations for common types
template class TensorView<float>;
template class TensorView<double>;
template class TensorView<int>;
template class TensorView<unsigned int>;
template class TensorView<long>;
template class TensorView<unsigned long>;

} // namespace hpc::compute