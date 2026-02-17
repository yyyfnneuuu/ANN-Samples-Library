#ifndef KNOWHERE_KERNEL_TOPK_REDUCER_H_
#define KNOWHERE_KERNEL_TOPK_REDUCER_H_

#include <cstddef>
#include <vector>

#include "graph_types.h"

namespace knowhere_demo {

// Bounded TopK reducer that avoids global full sort on every batch.
class TopKReducer {
public:
    explicit TopKReducer(std::size_t top_k);

    void AbsorbBatch(const std::vector<Candidate>& batch);
    std::vector<Candidate> Finalize() const;

private:
    static bool MaxHeapCmp(const Candidate& left, const Candidate& right);

    std::size_t top_k_;
    std::vector<Candidate> heap_;
};

}  // namespace knowhere_demo

#endif  // KNOWHERE_KERNEL_TOPK_REDUCER_H_
