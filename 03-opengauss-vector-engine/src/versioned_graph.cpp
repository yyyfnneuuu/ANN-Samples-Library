#include "versioned_graph.h"

#include <mutex>
#include <queue>
#include <unordered_set>

namespace opengauss_demo {

VersionedGraph::VersionedGraph(const std::size_t node_count)
    : neighbors_(node_count), versions_(node_count, 0) {}

void VersionedGraph::SetNeighbors(std::uint32_t node_id, std::vector<std::uint32_t> neighbors) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (node_id >= neighbors_.size()) {
        return;
    }
    neighbors_[node_id] = std::move(neighbors);
    ++versions_[node_id];
}

void VersionedGraph::BumpVersion(const std::uint32_t node_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (node_id >= versions_.size()) {
        return;
    }
    ++versions_[node_id];
}

std::vector<std::uint32_t> VersionedGraph::TraverseWithOcc(
    const std::uint32_t entrypoint,
    const std::size_t max_steps,
    const std::size_t max_retries) const {
    if (entrypoint >= neighbors_.size()) {
        return {};
    }

    std::vector<std::uint32_t> visited_order;
    visited_order.reserve(max_steps);

    std::queue<std::uint32_t> frontier;
    std::unordered_set<std::uint32_t> dedup;
    frontier.push(entrypoint);
    dedup.insert(entrypoint);

    while (!frontier.empty() && visited_order.size() < max_steps) {
        const std::uint32_t node = frontier.front();
        frontier.pop();

        std::vector<std::uint32_t> neighbors;
        bool read_ok = false;
        for (std::size_t retry = 0; retry < max_retries; ++retry) {
            if (TryReadNeighbors(node, &neighbors)) {
                read_ok = true;
                break;
            }
        }
        if (!read_ok) {
            continue;
        }

        visited_order.push_back(node);
        for (const std::uint32_t next : neighbors) {
            if (next >= neighbors_.size()) {
                continue;
            }
            if (dedup.insert(next).second) {
                frontier.push(next);
            }
        }
    }

    return visited_order;
}

bool VersionedGraph::TryReadNeighbors(const std::uint32_t node_id, std::vector<std::uint32_t>* neighbors) const {
    std::uint64_t begin_version = 0;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (node_id >= neighbors_.size()) {
            return false;
        }
        begin_version = versions_[node_id];
        *neighbors = neighbors_[node_id];
    }

    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (node_id >= versions_.size()) {
            return false;
        }
        return versions_[node_id] == begin_version;
    }
}

}  // namespace opengauss_demo
