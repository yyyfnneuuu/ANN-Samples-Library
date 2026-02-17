#ifndef OPENGAUSS_VECTOR_ENGINE_DUAL_ENGINE_INDEX_H_
#define OPENGAUSS_VECTOR_ENGINE_DUAL_ENGINE_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace opengauss_demo {

struct SearchHit {
    std::uint32_t id{};
    float distance{0.0F};
};

struct EvaluationMetrics {
    double recall_at_k{0.0};
    std::uint64_t memory_p95_us{0};
    std::uint64_t disk_p95_us{0};
};

class DualEngineIndex {
public:
    explicit DualEngineIndex(std::size_t dim, std::uint8_t bits = 6);

    void Build(const std::vector<std::vector<float>>& vectors, std::size_t block_size = 64);

    std::vector<SearchHit> SearchMemory(const std::vector<float>& query, std::size_t top_k) const;

    std::vector<SearchHit> SearchDisk(
        const std::vector<float>& query,
        std::size_t top_k,
        std::size_t rerank_k = 64) const;

    EvaluationMetrics Evaluate(
        const std::vector<std::vector<float>>& queries,
        std::size_t top_k,
        std::size_t rerank_k = 64) const;

private:
    std::size_t dim_;
    std::size_t block_size_;
    std::uint8_t bits_;
    std::vector<std::vector<float>> vectors_;
    std::vector<std::vector<float>> decoded_vectors_;
    std::vector<std::vector<std::uint8_t>> quant_codes_;
    std::vector<std::uint64_t> block_ids_;
};

}  // namespace opengauss_demo

#endif  // OPENGAUSS_VECTOR_ENGINE_DUAL_ENGINE_INDEX_H_
