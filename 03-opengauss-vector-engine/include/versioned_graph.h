#ifndef OPENGAUSS_VECTOR_ENGINE_VERSIONED_GRAPH_H_
#define OPENGAUSS_VECTOR_ENGINE_VERSIONED_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <vector>

namespace opengauss_demo {

class VersionedGraph {
public:
    explicit VersionedGraph(std::size_t node_count);

    void SetNeighbors(std::uint32_t node_id, std::vector<std::uint32_t> neighbors);
    void BumpVersion(std::uint32_t node_id);

    std::vector<std::uint32_t> TraverseWithOcc(
        std::uint32_t entrypoint,
        std::size_t max_steps,
        std::size_t max_retries = 3) const;

private:
    bool TryReadNeighbors(std::uint32_t node_id, std::vector<std::uint32_t>* neighbors) const;

    mutable std::shared_mutex mutex_;
    std::vector<std::vector<std::uint32_t>> neighbors_;
    std::vector<std::uint64_t> versions_;
};

}  // namespace opengauss_demo

#endif  // OPENGAUSS_VECTOR_ENGINE_VERSIONED_GRAPH_H_
