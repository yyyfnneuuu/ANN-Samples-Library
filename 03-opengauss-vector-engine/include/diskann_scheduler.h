#ifndef OPENGAUSS_VECTOR_ENGINE_DISKANN_SCHEDULER_H_
#define OPENGAUSS_VECTOR_ENGINE_DISKANN_SCHEDULER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace opengauss_demo {

struct IoRequest {
    std::uint32_t node_id{};
    std::uint64_t block_id{};
};

class DiskIoBatchScheduler {
public:
    explicit DiskIoBatchScheduler(std::size_t max_batch_size = 16);

    std::vector<IoRequest> Execute(const std::vector<IoRequest>& requests) const;
    std::size_t EstimateMergedOps(const std::vector<IoRequest>& ordered) const;

private:
    std::size_t max_batch_size_;
};

}  // namespace opengauss_demo

#endif  // OPENGAUSS_VECTOR_ENGINE_DISKANN_SCHEDULER_H_
