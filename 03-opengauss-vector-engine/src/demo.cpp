#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "diskann_scheduler.h"
#include "opq_rabitq.h"
#include "versioned_graph.h"

namespace {

std::vector<std::vector<float>> BuildTrainingSet() {
    std::vector<std::vector<float>> data;
    data.reserve(32);
    for (int row = 0; row < 32; ++row) {
        data.push_back({
            0.10F * row,
            0.08F * row + 0.02F,
            0.06F * row + 0.03F,
            0.05F * row + 0.01F,
            0.04F * row + 0.04F,
            0.03F * row + 0.05F,
            0.02F * row + 0.06F,
            0.01F * row + 0.07F,
        });
    }
    return data;
}

float L2(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float sum = 0.0F;
    for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
        const float diff = lhs[idx] - rhs[idx];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void PrintPath(const std::vector<std::uint32_t>& path, const std::string& title) {
    std::cout << title << ": ";
    for (const auto node : path) {
        std::cout << node << " ";
    }
    std::cout << "\n";
}

}  // namespace

int main() {
    using opengauss_demo::DiskIoBatchScheduler;
    using opengauss_demo::IoRequest;
    using opengauss_demo::OpqProjector;
    using opengauss_demo::RabitQCodec;
    using opengauss_demo::VersionedGraph;

    // 1) OPQ + RabitQ sample
    auto training = BuildTrainingSet();
    OpqProjector opq(/*dim=*/8);
    opq.SetRotationMatrix({
        {0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F},
    });

    std::vector<std::vector<float>> projected;
    projected.reserve(training.size());
    for (const auto& vec : training) {
        projected.push_back(opq.Transform(vec));
    }

    RabitQCodec codec(/*bits=*/6);
    codec.Fit(projected);

    const std::vector<float> query = opq.Transform({1.70F, 1.36F, 1.08F, 0.86F, 0.72F, 0.56F, 0.40F, 0.24F});
    const std::vector<std::uint8_t> code = codec.Encode(query);
    const std::vector<float> decoded = codec.Decode(code);

    std::cout << "[OPQ+RabitQ] bits=" << static_cast<int>(codec.Bits())
              << " reconstruction_l2=" << std::fixed << std::setprecision(6) << L2(query, decoded) << "\n";

    // 2) DiskANN I/O scheduling sample
    DiskIoBatchScheduler scheduler(/*max_batch_size=*/4);
    const std::vector<IoRequest> requests = {
        {10, 31}, {2, 12}, {3, 13}, {8, 30}, {5, 29}, {11, 34}, {7, 35}, {1, 11},
    };
    const std::vector<IoRequest> ordered = scheduler.Execute(requests);
    const std::size_t merged_ops = scheduler.EstimateMergedOps(ordered);

    std::cout << "[DiskANN Scheduler] merged_ops=" << merged_ops << " ordered_blocks=";
    for (const auto& req : ordered) {
        std::cout << req.block_id << " ";
    }
    std::cout << "\n";

    // 3) OCC traversal sample
    VersionedGraph graph(/*node_count=*/6);
    graph.SetNeighbors(0, {1, 2});
    graph.SetNeighbors(1, {3});
    graph.SetNeighbors(2, {4});
    graph.SetNeighbors(3, {});
    graph.SetNeighbors(4, {});

    PrintPath(graph.TraverseWithOcc(/*entrypoint=*/0, /*max_steps=*/6), "[OCC] before_update");

    graph.SetNeighbors(2, {4, 5});
    graph.SetNeighbors(3, {5});
    PrintPath(graph.TraverseWithOcc(/*entrypoint=*/0, /*max_steps=*/6), "[OCC] after_update");

    return 0;
}
