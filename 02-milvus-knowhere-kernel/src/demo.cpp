#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "async_graph_searcher.h"

namespace {

using knowhere_demo::GraphNode;
using knowhere_demo::NodeId;

std::vector<float> RandomEmbedding(std::mt19937* rng, std::size_t dim) {
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> vec(dim, 0.0F);
    for (float& value : vec) {
        value = dist(*rng);
    }
    return vec;
}

std::vector<GraphNode> BuildRandomGraph(std::size_t n, std::size_t dim, std::size_t degree, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> id_dist(0, static_cast<uint32_t>(n - 1));

    std::vector<GraphNode> graph;
    graph.reserve(n);
    for (NodeId id = 0; id < n; ++id) {
        std::vector<NodeId> neighbors;
        neighbors.reserve(degree);
        while (neighbors.size() < degree) {
            NodeId next = id_dist(rng);
            if (next != id) {
                neighbors.push_back(next);
            }
        }
        graph.push_back(GraphNode{
            .id = id,
            .embedding = RandomEmbedding(&rng, dim),
            .neighbors = std::move(neighbors),
        });
    }
    return graph;
}

std::uint64_t Percentile(std::vector<std::uint64_t> values, double p) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>((values.size() - 1) * p);
    return values[idx];
}

}  // namespace

int main() {
    using knowhere_demo::AsyncGraphSearcher;
    using knowhere_demo::SearchRequest;
    using knowhere_demo::SearchStats;

    constexpr std::size_t kNodeCount = 5000;
    constexpr std::size_t kDim = 128;
    constexpr std::size_t kDegree = 16;
    constexpr std::size_t kRounds = 60;

    AsyncGraphSearcher searcher(BuildRandomGraph(kNodeCount, kDim, kDegree));

    SearchRequest request;
    request.query = std::vector<float>(kDim, 0.45F);
    request.top_k = 10;
    request.filter_bitmap = std::vector<std::uint8_t>(kNodeCount, 1U);
    for (std::size_t i = 0; i < kNodeCount; i += 11) {
        request.filter_bitmap[i] = 0U;
    }

    std::vector<std::uint64_t> baseline_latency;
    std::vector<std::uint64_t> optimized_latency;
    baseline_latency.reserve(kRounds);
    optimized_latency.reserve(kRounds);

    SearchStats baseline_stats;
    SearchStats optimized_stats;

    for (std::size_t round = 0; round < kRounds; ++round) {
        const auto start_base = std::chrono::steady_clock::now();
        auto base_res = searcher.SearchBaseline(request, /*entrypoint=*/round % 100, /*max_visit=*/700, &baseline_stats);
        (void)base_res;
        const auto end_base = std::chrono::steady_clock::now();

        const auto start_opt = std::chrono::steady_clock::now();
        auto opt_res = searcher.SearchOptimized(
            request,
            /*entrypoint=*/round % 100,
            /*max_visit=*/700,
            /*batch_size=*/64,
            &optimized_stats);
        (void)opt_res;
        const auto end_opt = std::chrono::steady_clock::now();

        baseline_latency.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count());
        optimized_latency.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count());
    }

    const auto base_p50 = Percentile(baseline_latency, 0.5);
    const auto base_p95 = Percentile(baseline_latency, 0.95);
    const auto opt_p50 = Percentile(optimized_latency, 0.5);
    const auto opt_p95 = Percentile(optimized_latency, 0.95);

    const double p95_speedup = base_p95 > 0 ? static_cast<double>(base_p95) / static_cast<double>(opt_p95) : 0.0;

    std::cout << "Baseline latency(us): p50=" << base_p50 << " p95=" << base_p95 << "\n";
    std::cout << "Optimized latency(us): p50=" << opt_p50 << " p95=" << opt_p95 << "\n";
    std::cout << "P95 speedup=" << std::fixed << std::setprecision(2) << p95_speedup << "x\n";
    std::cout << "Optimized stats: visited=" << optimized_stats.visited
              << " filtered=" << optimized_stats.filtered_nodes
              << " prefetch_us=" << optimized_stats.prefetch_us
              << " compute_us=" << optimized_stats.compute_us << "\n";

    return 0;
}
