#include "opq_rabitq.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace opengauss_demo {

OpqProjector::OpqProjector(const std::size_t dim)
    : dim_(dim), rotation_(dim, std::vector<float>(dim, 0.0F)) {
    for (std::size_t idx = 0; idx < dim_; ++idx) {
        rotation_[idx][idx] = 1.0F;
    }
}

void OpqProjector::SetRotationMatrix(const std::vector<std::vector<float>>& matrix) {
    if (matrix.size() != dim_) {
        throw std::invalid_argument("OPQ matrix row mismatch");
    }
    for (const auto& row : matrix) {
        if (row.size() != dim_) {
            throw std::invalid_argument("OPQ matrix column mismatch");
        }
    }
    rotation_ = matrix;
}

std::vector<float> OpqProjector::Transform(const std::vector<float>& input) const {
    if (input.size() != dim_) {
        throw std::invalid_argument("OPQ input dim mismatch");
    }

    std::vector<float> output(dim_, 0.0F);
    for (std::size_t row = 0; row < dim_; ++row) {
        float value = 0.0F;
        for (std::size_t col = 0; col < dim_; ++col) {
            value += rotation_[row][col] * input[col];
        }
        output[row] = value;
    }
    return output;
}

RabitQCodec::RabitQCodec(const std::uint8_t bits) : bits_(bits), dim_(0) {
    if (bits_ < 4U || bits_ > 7U) {
        throw std::invalid_argument("RabitQ bits must be in [4, 7]");
    }
}

void RabitQCodec::Fit(const std::vector<std::vector<float>>& training_vectors) {
    if (training_vectors.empty()) {
        throw std::invalid_argument("RabitQ Fit requires non-empty training set");
    }

    dim_ = training_vectors.front().size();
    if (dim_ == 0) {
        throw std::invalid_argument("RabitQ Fit requires non-zero dimension");
    }

    min_per_dim_.assign(dim_, std::numeric_limits<float>::max());
    std::vector<float> max_per_dim(dim_, std::numeric_limits<float>::lowest());

    for (const auto& vector : training_vectors) {
        if (vector.size() != dim_) {
            throw std::invalid_argument("RabitQ Fit dimension mismatch");
        }
        for (std::size_t idx = 0; idx < dim_; ++idx) {
            min_per_dim_[idx] = std::min(min_per_dim_[idx], vector[idx]);
            max_per_dim[idx] = std::max(max_per_dim[idx], vector[idx]);
        }
    }

    const float levels = static_cast<float>((1U << bits_) - 1U);
    scale_per_dim_.assign(dim_, 1.0F);
    for (std::size_t idx = 0; idx < dim_; ++idx) {
        const float span = std::max(1e-6F, max_per_dim[idx] - min_per_dim_[idx]);
        scale_per_dim_[idx] = levels / span;
    }
}

std::vector<std::uint8_t> RabitQCodec::Encode(const std::vector<float>& vector) const {
    if (dim_ == 0 || min_per_dim_.empty() || scale_per_dim_.empty()) {
        throw std::logic_error("RabitQ codec is not fitted");
    }
    if (vector.size() != dim_) {
        throw std::invalid_argument("RabitQ Encode dim mismatch");
    }

    const float levels = static_cast<float>((1U << bits_) - 1U);
    std::vector<std::uint8_t> code(dim_, 0U);
    for (std::size_t idx = 0; idx < dim_; ++idx) {
        const float normalized = (vector[idx] - min_per_dim_[idx]) * scale_per_dim_[idx];
        const float clamped = std::clamp(normalized, 0.0F, levels);
        code[idx] = static_cast<std::uint8_t>(std::lround(clamped));
    }
    return code;
}

std::vector<float> RabitQCodec::Decode(const std::vector<std::uint8_t>& code) const {
    if (dim_ == 0 || code.size() != dim_) {
        throw std::invalid_argument("RabitQ Decode dim mismatch");
    }

    std::vector<float> vector(dim_, 0.0F);
    for (std::size_t idx = 0; idx < dim_; ++idx) {
        vector[idx] = static_cast<float>(code[idx]) / scale_per_dim_[idx] + min_per_dim_[idx];
    }
    return vector;
}

std::size_t RabitQCodec::Dim() const {
    return dim_;
}

std::uint8_t RabitQCodec::Bits() const {
    return bits_;
}

}  // namespace opengauss_demo
