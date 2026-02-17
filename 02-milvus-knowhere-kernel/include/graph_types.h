#ifndef KNOWHERE_KERNEL_GRAPH_TYPES_H_
#define KNOWHERE_KERNEL_GRAPH_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace knowhere_demo {

using NodeId = std::uint32_t;

struct GraphNode {
    NodeId id{};
    std::vector<float> embedding;
    std::vector<NodeId> neighbors;
};

struct SearchRequest {
    std::vector<float> query;
    std::size_t top_k{10};
    // 1 means the node can be returned, 0 means filtered out.
    std::vector<std::uint8_t> filter_bitmap;
};

struct Candidate {
    NodeId id{};
    float distance{0.0F};
    bool passed_filter{true};
};

struct SearchStats {
    std::size_t visited{0};
    std::size_t filtered_nodes{0};
    std::uint64_t prefetch_us{0};
    std::uint64_t compute_us{0};
};

}  // namespace knowhere_demo

#endif  // KNOWHERE_KERNEL_GRAPH_TYPES_H_
