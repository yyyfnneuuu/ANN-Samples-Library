#include "diskann_scheduler.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace opengauss_demo {

DiskIoBatchScheduler::DiskIoBatchScheduler(const std::size_t max_batch_size)
    : max_batch_size_(max_batch_size) {}

std::vector<IoRequest> DiskIoBatchScheduler::Execute(const std::vector<IoRequest>& requests) const {
    std::vector<IoRequest> ordered = requests;
    std::sort(ordered.begin(), ordered.end(), [](const IoRequest& lhs, const IoRequest& rhs) {
        if (lhs.block_id == rhs.block_id) {
            return lhs.node_id < rhs.node_id;
        }
        return lhs.block_id < rhs.block_id;
    });

    // Simulate async batched I/O submission.
    for (std::size_t start = 0; start < ordered.size(); start += max_batch_size_) {
        std::this_thread::sleep_for(std::chrono::microseconds(80));
    }
    return ordered;
}

std::size_t DiskIoBatchScheduler::EstimateMergedOps(const std::vector<IoRequest>& ordered) const {
    if (ordered.empty()) {
        return 0;
    }

    std::size_t merged_ops = 1;
    std::uint64_t prev_block = ordered.front().block_id;
    for (std::size_t idx = 1; idx < ordered.size(); ++idx) {
        if (ordered[idx].block_id > prev_block + 1) {
            ++merged_ops;
        }
        prev_block = ordered[idx].block_id;
    }
    return merged_ops;
}

}  // namespace opengauss_demo
