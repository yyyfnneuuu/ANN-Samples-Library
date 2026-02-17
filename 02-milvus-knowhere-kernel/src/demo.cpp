#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "async_graph_searcher.h"

namespace {

using knowhere_demo::GraphNode;
using knowhere_demo::NodeId;

std::vector<float> MakeEmbedding(float seed) {
    return {seed, seed + 0.1F, seed + 0.2F, seed + 0.3F};
}

std::vector<GraphNode> BuildToyGraph() {
    std::vector<GraphNode> graph;
    graph.reserve(12);

    for (NodeId id = 0; id < 12; ++id) {
        graph.push_back(GraphNode{
            .id = id,
            .embedding = MakeEmbedding(static_cast<float>(id) * 0.15F),
            .neighbors = {
                static_cast<NodeId>((id + 1) % 12),
                static_cast<NodeId>((id + 3) % 12),
                static_cast<NodeId>((id + 5) % 12),
            },
        });
    }
    return graph;
}

}  // namespace

int main() {
    using knowhere_demo::AsyncGraphSearcher;
    using knowhere_demo::SearchRequest;

    AsyncGraphSearcher searcher(BuildToyGraph());

    SearchRequest request;
    request.query = {0.45F, 0.55F, 0.65F, 0.75F};
    request.top_k = 5;
    request.filter_bitmap = std::vector<std::uint8_t>(12, 1U);

    // Simulate a strong filter: filtered nodes are still used as graph bridges.
    request.filter_bitmap[3] = 0U;
    request.filter_bitmap[4] = 0U;
    request.filter_bitmap[7] = 0U;

    const auto start = std::chrono::steady_clock::now();
    const auto results = searcher.Search(request, /*entrypoint=*/0, /*max_visit=*/128, /*batch_size=*/16);
    const auto end = std::chrono::steady_clock::now();

    std::cout << "TopK Results (distance asc):\n";
    for (const auto& candidate : results) {
        std::cout << "  node=" << std::setw(2) << candidate.id << " dist=" << std::fixed
                  << std::setprecision(4) << candidate.distance << "\n";
    }

    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Elapsed: " << elapsed_us << " us\n";
    return 0;
}
