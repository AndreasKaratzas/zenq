#pragma once

#include <concepts>
#include <cstdint>
#include <vector>

namespace hpc {
namespace core {

using Shape = std::vector<std::size_t>;

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

#if defined(__CUDACC__)
template <typename T>
struct is_tensor_data_type {
    static constexpr bool value = std::is_same<T, float>::value || std::is_same<T, double>::value ||
                                  std::is_same<T, int32_t>::value ||
                                  std::is_same<T, int64_t>::value;
};

template <typename T>
inline constexpr bool is_tensor_data_type_v = is_tensor_data_type<T>::value;

    #define TENSOR_DATA_TYPE typename
#else
    // CPU code can use concepts
    #include <concepts>

template <typename T>
concept TensorDataType = std::same_as<T, float> || std::same_as<T, double> ||
                         std::same_as<T, int32_t> || std::same_as<T, int64_t>;

    #define TENSOR_DATA_TYPE TensorDataType
#endif

} // namespace core
} // namespace hpc