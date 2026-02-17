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

}  // namespace knowhere_demo

#endif  // KNOWHERE_KERNEL_GRAPH_TYPES_H_
