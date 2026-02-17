#include "dual_engine_index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "diskann_scheduler.h"
#include "opq_rabitq.h"

namespace opengauss_demo {

namespace {

float L2(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        return std::numeric_limits<float>::max();
    }
    float sum = 0.0F;
    for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
        const float diff = lhs[idx] - rhs[idx];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<SearchHit> TopKFromDistances(const std::vector<float>& distances, std::size_t top_k) {
    std::vector<SearchHit> hits;
    hits.reserve(distances.size());
    for (std::size_t idx = 0; idx < distances.size(); ++idx) {
        hits.push_back(SearchHit{.id = static_cast<std::uint32_t>(idx), .distance = distances[idx]});
    }
    if (hits.size() > top_k) {
        std::nth_element(
            hits.begin(),
            hits.begin() + static_cast<long>(top_k),
            hits.end(),
            [](const SearchHit& lhs, const SearchHit& rhs) { return lhs.distance < rhs.distance; });
        hits.resize(top_k);
    }
    std::sort(
        hits.begin(),
        hits.end(),
        [](const SearchHit& lhs, const SearchHit& rhs) { return lhs.distance < rhs.distance; });
    return hits;
}

std::uint64_t P95(std::vector<std::uint64_t> values) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>((values.size() - 1) * 0.95);
    return values[idx];
}

}  // namespace

DualEngineIndex::DualEngineIndex(const std::size_t dim, const std::uint8_t bits)
    : dim_(dim), block_size_(64), bits_(bits) {}

void DualEngineIndex::Build(const std::vector<std::vector<float>>& vectors, const std::size_t block_size) {
    block_size_ = std::max<std::size_t>(1, block_size);
    vectors_ = vectors;
    decoded_vectors_.clear();
    quant_codes_.clear();
    block_ids_.clear();

    if (vectors_.empty()) {
        return;
    }

    OpqProjector projector(dim_);
    // Use identity rotation by default. In production this matrix is learned offline.

    std::vector<std::vector<float>> projected;
    projected.reserve(vectors_.size());
    for (const auto& vector : vectors_) {
        projected.push_back(projector.Transform(vector));
    }

    RabitQCodec codec(bits_);
    codec.Fit(projected);

    quant_codes_.reserve(projected.size());
    decoded_vectors_.reserve(projected.size());
    block_ids_.reserve(projected.size());
    for (std::size_t idx = 0; idx < projected.size(); ++idx) {
        const auto code = codec.Encode(projected[idx]);
        quant_codes_.push_back(code);
        decoded_vectors_.push_back(codec.Decode(code));
        block_ids_.push_back(idx / block_size_);
    }
}

std::vector<SearchHit> DualEngineIndex::SearchMemory(const std::vector<float>& query, const std::size_t top_k) const {
    std::vector<float> distances(vectors_.size(), std::numeric_limits<float>::max());
    for (std::size_t idx = 0; idx < vectors_.size(); ++idx) {
        distances[idx] = L2(query, vectors_[idx]);
    }
    return TopKFromDistances(distances, top_k);
}

std::vector<SearchHit> DualEngineIndex::SearchDisk(
    const std::vector<float>& query,
    const std::size_t top_k,
    const std::size_t rerank_k) const {
    if (vectors_.empty()) {
        return {};
    }

    std::vector<IoRequest> requests;
    requests.reserve(vectors_.size());
    for (std::size_t idx = 0; idx < vectors_.size(); ++idx) {
        requests.push_back(IoRequest{.node_id = static_cast<std::uint32_t>(idx), .block_id = block_ids_[idx]});
    }

    DiskIoBatchScheduler scheduler(/*max_batch_size=*/16);
    const std::vector<IoRequest> ordered = scheduler.Execute(requests);

    std::vector<float> coarse_dist(vectors_.size(), std::numeric_limits<float>::max());
    for (const auto& request : ordered) {
        const std::uint32_t id = request.node_id;
        coarse_dist[id] = L2(query, decoded_vectors_[id]);
    }

    const auto coarse_top = TopKFromDistances(coarse_dist, std::max(top_k, rerank_k));
    std::vector<SearchHit> reranked;
    reranked.reserve(coarse_top.size());
    for (const auto& hit : coarse_top) {
        reranked.push_back(SearchHit{.id = hit.id, .distance = L2(query, vectors_[hit.id])});
    }

    std::sort(
        reranked.begin(),
        reranked.end(),
        [](const SearchHit& lhs, const SearchHit& rhs) { return lhs.distance < rhs.distance; });
    if (reranked.size() > top_k) {
        reranked.resize(top_k);
    }
    return reranked;
}

EvaluationMetrics DualEngineIndex::Evaluate(
    const std::vector<std::vector<float>>& queries,
    const std::size_t top_k,
    const std::size_t rerank_k) const {
    EvaluationMetrics metrics;
    if (queries.empty() || vectors_.empty()) {
        return metrics;
    }

    std::vector<std::uint64_t> memory_latency;
    std::vector<std::uint64_t> disk_latency;
    memory_latency.reserve(queries.size());
    disk_latency.reserve(queries.size());

    double recall_sum = 0.0;

    for (const auto& query : queries) {
        const auto start_mem = std::chrono::steady_clock::now();
        const auto exact = SearchMemory(query, top_k);
        const auto end_mem = std::chrono::steady_clock::now();

        const auto start_disk = std::chrono::steady_clock::now();
        const auto approx = SearchDisk(query, top_k, rerank_k);
        const auto end_disk = std::chrono::steady_clock::now();

        memory_latency.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end_mem - start_mem).count());
        disk_latency.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end_disk - start_disk).count());

        std::unordered_set<std::uint32_t> exact_ids;
        for (const auto& hit : exact) {
            exact_ids.insert(hit.id);
        }

        std::size_t overlap = 0;
        for (const auto& hit : approx) {
            if (exact_ids.count(hit.id) > 0) {
                ++overlap;
            }
        }
        recall_sum += static_cast<double>(overlap) / static_cast<double>(top_k);
    }

    metrics.recall_at_k = recall_sum / static_cast<double>(queries.size());
    metrics.memory_p95_us = P95(memory_latency);
    metrics.disk_p95_us = P95(disk_latency);
    return metrics;
}

}  // namespace opengauss_demo
