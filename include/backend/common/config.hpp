#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hpc {
namespace config {

struct ConvConfig {
    size_t kernel_size;
    size_t stride{1};
    size_t padding{0};
    size_t dilation{1};
    size_t groups{1};
    size_t in_channels;
    size_t out_channels;
    bool   use_bias{true};
};

} // namespace config
} // namespace hpc