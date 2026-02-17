#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dual_engine_index.h"
#include "versioned_graph.h"

namespace {

std::vector<float> RandomVector(std::mt19937* rng, std::size_t dim) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> vector(dim, 0.0F);
    for (float& value : vector) {
        value = dist(*rng);
    }
    return vector;
}

void PrintPath(const std::vector<std::uint32_t>& path, const std::string& title) {
    std::cout << title << ": ";
    for (const auto node : path) {
        std::cout << node << " ";
    }
    std::cout << "\n";
}

}  // namespace

int main() {
    using opengauss_demo::DualEngineIndex;
    using opengauss_demo::VersionedGraph;

    constexpr std::size_t kDim = 96;
    constexpr std::size_t kDataSize = 4000;
    constexpr std::size_t kQuerySize = 120;
    constexpr std::size_t kTopK = 10;

    std::mt19937 rng(42);
    std::vector<std::vector<float>> vectors;
    vectors.reserve(kDataSize);
    for (std::size_t idx = 0; idx < kDataSize; ++idx) {
        vectors.push_back(RandomVector(&rng, kDim));
    }

    std::vector<std::vector<float>> queries;
    queries.reserve(kQuerySize);
    for (std::size_t idx = 0; idx < kQuerySize; ++idx) {
        queries.push_back(RandomVector(&rng, kDim));
    }

    DualEngineIndex index(kDim, /*bits=*/6);
    index.Build(vectors, /*block_size=*/64);
    const auto metrics = index.Evaluate(queries, kTopK, /*rerank_k=*/32);

    std::cout << "DualEngine evaluate:\n";
    std::cout << "  Recall@" << kTopK << "=" << std::fixed << std::setprecision(4)
              << metrics.recall_at_k << "\n";
    std::cout << "  Memory p95(us)=" << metrics.memory_p95_us << "\n";
    std::cout << "  Disk p95(us)=" << metrics.disk_p95_us << "\n";
    if (metrics.disk_p95_us > 0) {
        const double ratio = static_cast<double>(metrics.memory_p95_us) /
                             static_cast<double>(metrics.disk_p95_us);
        std::cout << "  Memory/Disk p95 ratio=" << std::setprecision(3) << ratio << "\n";
    }

    VersionedGraph graph(/*node_count=*/6);
    graph.SetNeighbors(0, {1, 2});
    graph.SetNeighbors(1, {3});
    graph.SetNeighbors(2, {4});
    graph.SetNeighbors(3, {});
    graph.SetNeighbors(4, {});

    PrintPath(graph.TraverseWithOcc(/*entrypoint=*/0, /*max_steps=*/6), "OCC before update");

    graph.SetNeighbors(2, {4, 5});
    graph.SetNeighbors(3, {5});
    PrintPath(graph.TraverseWithOcc(/*entrypoint=*/0, /*max_steps=*/6), "OCC after update");

    return 0;
}
