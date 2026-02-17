#ifndef OPENGAUSS_VECTOR_ENGINE_OPQ_RABITQ_H_
#define OPENGAUSS_VECTOR_ENGINE_OPQ_RABITQ_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace opengauss_demo {

class OpqProjector {
public:
    explicit OpqProjector(std::size_t dim);

    void SetRotationMatrix(const std::vector<std::vector<float>>& matrix);
    std::vector<float> Transform(const std::vector<float>& input) const;

private:
    std::size_t dim_;
    std::vector<std::vector<float>> rotation_;
};

class RabitQCodec {
public:
    explicit RabitQCodec(std::uint8_t bits = 6);

    void Fit(const std::vector<std::vector<float>>& training_vectors);
    std::vector<std::uint8_t> Encode(const std::vector<float>& vector) const;
    std::vector<float> Decode(const std::vector<std::uint8_t>& code) const;

    std::size_t Dim() const;
    std::uint8_t Bits() const;

private:
    std::uint8_t bits_;
    std::size_t dim_;
    std::vector<float> min_per_dim_;
    std::vector<float> scale_per_dim_;
};

}  // namespace opengauss_demo

#endif  // OPENGAUSS_VECTOR_ENGINE_OPQ_RABITQ_H_
