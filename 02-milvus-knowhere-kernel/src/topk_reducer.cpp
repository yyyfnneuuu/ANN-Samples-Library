#include "topk_reducer.h"

#include <algorithm>

namespace knowhere_demo {

TopKReducer::TopKReducer(std::size_t top_k) : top_k_(top_k) {
    heap_.reserve(top_k);
}

bool TopKReducer::MaxHeapCmp(const Candidate& left, const Candidate& right) {
    // Max-heap by distance. Root is the current worst in TopK.
    return left.distance < right.distance;
}

void TopKReducer::AbsorbBatch(const std::vector<Candidate>& batch) {
    for (const Candidate& candidate : batch) {
        if (!candidate.passed_filter) {
            continue;
        }

        if (heap_.size() < top_k_) {
            heap_.push_back(candidate);
            std::push_heap(heap_.begin(), heap_.end(), MaxHeapCmp);
            continue;
        }

        if (candidate.distance >= heap_.front().distance) {
            continue;
        }

        std::pop_heap(heap_.begin(), heap_.end(), MaxHeapCmp);
        heap_.back() = candidate;
        std::push_heap(heap_.begin(), heap_.end(), MaxHeapCmp);
    }
}

std::vector<Candidate> TopKReducer::Finalize() const {
    std::vector<Candidate> sorted = heap_;
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const Candidate& left, const Candidate& right) { return left.distance < right.distance; });
    return sorted;
}

}  // namespace knowhere_demo
