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
    const std::size_t batch_size,
    SearchStats* stats) const {
    return SearchOptimized(request, entrypoint, max_visit, batch_size, stats);
}

std::vector<Candidate> AsyncGraphSearcher::SearchBaseline(
    const SearchRequest& request,
    const NodeId entrypoint,
    const std::size_t max_visit,
    SearchStats* stats) const {
    if (graph_.empty() || entrypoint >= graph_.size() || request.query.empty()) {
        return {};
    }

    SearchStats local_stats;
    TopKReducer reducer(request.top_k);
    std::queue<NodeId> frontier;
    std::unordered_set<NodeId> visited;
    frontier.push(entrypoint);
    visited.insert(entrypoint);

    while (!frontier.empty() && local_stats.visited < max_visit) {
        const NodeId current = frontier.front();
        frontier.pop();

        const auto compute_start = std::chrono::steady_clock::now();
        const GraphNode& node = graph_[current];
        const bool passed = PassFilter(node.id, request);
        const float distance = L2Distance(request.query, node.embedding);
        const auto compute_end = std::chrono::steady_clock::now();

        local_stats.compute_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start).count();
        local_stats.filtered_nodes += passed ? 0 : 1;

        reducer.AbsorbBatch({Candidate{.id = node.id, .distance = distance, .passed_filter = passed}});

        const auto prefetch_start = std::chrono::steady_clock::now();
        const std::vector<NodeId> neighbors = PrefetchNeighbors(current);
        const auto prefetch_end = std::chrono::steady_clock::now();
        local_stats.prefetch_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(prefetch_end - prefetch_start).count();

        for (const NodeId neighbor : neighbors) {
            if (neighbor >= graph_.size()) {
                continue;
            }
            if (visited.insert(neighbor).second) {
                frontier.push(neighbor);
            }
        }

        ++local_stats.visited;
    }

    if (stats) {
        *stats = local_stats;
    }
    return reducer.Finalize();
}

std::vector<Candidate> AsyncGraphSearcher::SearchOptimized(
    const SearchRequest& request,
    const NodeId entrypoint,
    const std::size_t max_visit,
    const std::size_t batch_size,
    SearchStats* stats) const {
    if (graph_.empty() || entrypoint >= graph_.size() || request.query.empty()) {
        return {};
    }

    SearchStats local_stats;
    TopKReducer reducer(request.top_k);
    std::queue<NodeId> frontier;
    std::unordered_set<NodeId> visited;
    std::vector<Candidate> local_batch;
    local_batch.reserve(batch_size);

    frontier.push(entrypoint);
    visited.insert(entrypoint);

    while (!frontier.empty() && local_stats.visited < max_visit) {
        std::vector<NodeId> stage_nodes;
        stage_nodes.reserve(batch_size);
        while (!frontier.empty() && stage_nodes.size() < batch_size && local_stats.visited + stage_nodes.size() < max_visit) {
            stage_nodes.push_back(frontier.front());
            frontier.pop();
        }

        std::vector<std::future<std::vector<NodeId>>> prefetch_jobs;
        prefetch_jobs.reserve(stage_nodes.size());

        const auto prefetch_start = std::chrono::steady_clock::now();
        for (const NodeId node : stage_nodes) {
            prefetch_jobs.push_back(
                std::async(std::launch::async, &AsyncGraphSearcher::PrefetchNeighbors, this, node));
        }
        const auto prefetch_launch_end = std::chrono::steady_clock::now();

        const auto compute_start = std::chrono::steady_clock::now();
        for (const NodeId node_id : stage_nodes) {
            const GraphNode& node = graph_[node_id];
            const bool passed = PassFilter(node.id, request);
            const float distance = L2Distance(request.query, node.embedding);
            local_batch.push_back(Candidate{.id = node.id, .distance = distance, .passed_filter = passed});
            local_stats.filtered_nodes += passed ? 0 : 1;
        }
        const auto compute_end = std::chrono::steady_clock::now();

        for (std::size_t idx = 0; idx < prefetch_jobs.size(); ++idx) {
            const std::vector<NodeId> neighbors = prefetch_jobs[idx].get();
            for (const NodeId neighbor : neighbors) {
                if (neighbor >= graph_.size()) {
                    continue;
                }
                if (visited.insert(neighbor).second) {
                    frontier.push(neighbor);
                }
            }
        }
        const auto prefetch_end = std::chrono::steady_clock::now();

        reducer.AbsorbBatch(local_batch);
        local_batch.clear();

        local_stats.visited += stage_nodes.size();
        local_stats.compute_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start).count();
        // include async launch + waiting + merge as prefetch stage cost.
        local_stats.prefetch_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(prefetch_end - prefetch_start).count();
        (void)prefetch_launch_end;
    }

    if (stats) {
        *stats = local_stats;
    }
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
