#include "async_graph_searcher.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <queue>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "topk_reducer.h"

namespace knowhere_demo {

AsyncGraphSearcher::AsyncGraphSearcher(std::vector<GraphNode> graph) : graph_(std::move(graph)) {}

std::vector<Candidate> AsyncGraphSearcher::Search(
    const SearchRequest& request,
    const NodeId entrypoint,
    const std::size_t max_visit,
    const std::size_t batch_size) const {
    if (graph_.empty() || entrypoint >= graph_.size() || request.query.empty()) {
        return {};
    }

    TopKReducer reducer(request.top_k);
    std::queue<NodeId> frontier;
    std::unordered_set<NodeId> visited;
    std::vector<Candidate> local_batch;
    local_batch.reserve(batch_size);

    frontier.push(entrypoint);
    visited.insert(entrypoint);

    std::size_t expanded = 0;
    while (!frontier.empty() && expanded < max_visit) {
        const NodeId current = frontier.front();
        frontier.pop();

        std::future<std::vector<NodeId>> prefetch_future =
            std::async(std::launch::async, &AsyncGraphSearcher::PrefetchNeighbors, this, current);

        const GraphNode& node = graph_[current];
        const float distance = L2Distance(request.query, node.embedding);
        local_batch.push_back(Candidate{
            .id = node.id,
            .distance = distance,
            .passed_filter = PassFilter(node.id, request),
        });

        const std::vector<NodeId> neighbors = prefetch_future.get();
        for (const NodeId neighbor : neighbors) {
            if (neighbor >= graph_.size()) {
                continue;
            }
            if (visited.insert(neighbor).second) {
                // Keep traversal connectivity even when current node is filtered out.
                frontier.push(neighbor);
            }
        }

        ++expanded;
        if (local_batch.size() >= batch_size) {
            reducer.AbsorbBatch(local_batch);
            local_batch.clear();
        }
    }

    reducer.AbsorbBatch(local_batch);
    return reducer.Finalize();
}

float AsyncGraphSearcher::L2Distance(const std::vector<float>& lhs, const std::vector<float>& rhs) const {
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

bool AsyncGraphSearcher::PassFilter(const NodeId node_id, const SearchRequest& request) const {
    if (request.filter_bitmap.empty() || node_id >= request.filter_bitmap.size()) {
        return true;
    }
    return request.filter_bitmap[node_id] == 1U;
}

std::vector<NodeId> AsyncGraphSearcher::PrefetchNeighbors(const NodeId node_id) const {
    std::this_thread::sleep_for(std::chrono::microseconds(15));
    return graph_[node_id].neighbors;
}

}  // namespace knowhere_demo
