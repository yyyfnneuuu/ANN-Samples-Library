#ifndef KNOWHERE_KERNEL_ASYNC_GRAPH_SEARCHER_H_
#define KNOWHERE_KERNEL_ASYNC_GRAPH_SEARCHER_H_

#include <cstddef>
#include <vector>

#include "graph_types.h"

namespace knowhere_demo {

class AsyncGraphSearcher {
public:
    explicit AsyncGraphSearcher(std::vector<GraphNode> graph);

    std::vector<Candidate> SearchBaseline(
        const SearchRequest& request,
        NodeId entrypoint,
        std::size_t max_visit = 256,
        SearchStats* stats = nullptr) const;

    std::vector<Candidate> SearchOptimized(
        const SearchRequest& request,
        NodeId entrypoint,
        std::size_t max_visit = 256,
        std::size_t batch_size = 32,
        SearchStats* stats = nullptr) const;

    std::vector<Candidate> Search(
        const SearchRequest& request,
        NodeId entrypoint,
        std::size_t max_visit = 256,
        std::size_t batch_size = 32,
        SearchStats* stats = nullptr) const;

private:
    float L2Distance(const std::vector<float>& lhs, const std::vector<float>& rhs) const;
    bool PassFilter(NodeId node_id, const SearchRequest& request) const;
    std::vector<NodeId> PrefetchNeighbors(NodeId node_id) const;

    std::vector<GraphNode> graph_;
};

}  // namespace knowhere_demo

#endif  // KNOWHERE_KERNEL_ASYNC_GRAPH_SEARCHER_H_
