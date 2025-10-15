NPU-Accelerated Vector Search (IVF-HSP)

1. 概述
本项目提供了一个基于 NPU (神经网络处理单元) 的向量相似度检索库。它专门为在海量向量数据集中执行低延迟、高吞吐的检索而设计。
该实现的核心是 IVF-HSP (Inverted File with Hierarchical Subspace Pursuit) 算法，这是一种分层搜索策略，能够有效利用 NPU 的Cube计算能力，在保证高召回率的同时，大幅减少搜索范围和计算量。
代码将复杂的搜索流程拆分为独立的、可维护的模块，从接收查询到返回结果的完整端到端流水线。

2. 核心特性
NPU硬件加速: 充分利用昇腾Cube硬件的强大算力，为距离计算和排序等密集型任务提供加速。
分层搜索算法: 通过 L1-L2-L3 三级流水线，逐步缩小搜索空间，实现了效率和精度的平衡。
多种搜索模式:
单索引/多索引检索: 支持对单个或多个索引同时进行检索。
结果合并: 在多索引检索时，可选择将多个来源的结果进行归并排序，返回全局 Top-K。
掩码(Mask)支持: 支持在检索时动态过滤部分向量，实现更灵活的搜索策略。
高效内存管理: 自动处理大数据量查询的分页（Paged）机制，避免因单次请求数据过大而导致的性能问题。

3. 核心算法：三级搜索流水线
NpuIndexIVFHSP 的核心是一个三级倒排文件搜索流水线。它将一次复杂的全局搜索分解为三个更简单的阶段：
Stage 1: L1 粗粒度搜索 (Coarse Search)
目标: 快速定位到可能包含最近邻结果的大致区域（聚类）。
Stage 2: L2 中粒度筛选 (Refined Search)
目标: 在 L1 阶段选出的候选聚类中，进一步筛选出更精确的子区域（子聚类）。
过程: 将查询向量与 L2 码本（Sub-Centroids）进行比较，从 L1 的结果中进一步筛选出 nProbeL2 个最相似的子聚类。这一步为 L3 阶段准备了最终的、极小范围的候选集。
Stage 3: L3 精准距离计算与排序 (Precise Ranking)
目标: 在 L2 阶段确定的极小候选集内，对每个向量进行精确的距离计算，并找出最终的 Top-K 结果。
距离计算过程:
从设备内存中读取 X 阶段筛选出的子聚类所包含的所有向量的编码。
使用 Cube进行矩阵乘，通过投影矩阵的能量计算查询向量与这些候选向量编码之间的精确距离。
对计算出的距离进行全局排序，返回距离最小的 Top-K 个向量的标签和距离。

4. 代码文件结构
NpuIndexIVFHSP.h:
职责: 定义 NpuIndexIVFHSP 类的公共接口和私有成员。这是外部与本库交互的入口。

NpuIndexIVFHSP_Search.cpp:
职责: 实现 Search 系列公共API。主要负责参数校验、分页逻辑处理，并将请求分派给内部的 SearchImpl 函数。

NpuIndexIVFHSP_SearchImpl.cpp:
职责: 实现 SearchImpl 系列函数。负责准备设备数据（如H2D内存拷贝），并以批处理（Batch）的方式驱动核心搜索流程。

NpuIndexIVFHSP_SearchBatch.cpp:
职责: 实现 SearchBatchImpl 系列函数。这是搜索的核心调度器，负责编排 L1-L2-L3 三级流水线，处理单索引、多索引、带掩码等不同场景的复杂逻辑。

NpuIndexIVFHSP_Pipeline.cpp:
职责: 实现 L1, L2, L3 各个流水线阶段的具体逻辑（SearchBatchImplL1, SearchBatchImplL2, SearchBatchImplL3）。这部分代码与 NPU 算子（Operator）直接交互，是核心算法的最终执行层。

common_types.h:
职责: 存放项目共享的类型定义、常量、宏和第三方库的占位符声明。

5. API 使用示例
#include "NpuIndexIVFHSP.h"
#include <vector>

NpuIndexIVFHSP* index = new NpuIndexIVFHSP();

// 1. 基础搜索 (std::vector)
void basic_search_example() {
int nq = 10; // 查询向量的数量
int topK = 5; // 返回最相似的5个结果
int dim = 128; // 向量维度

std::vector<float> queries(nq * dim, 0.1f);
std::vector<float> distances(nq * topK);
std::vector<int64_t> labels(nq * topK);

APP_ERROR ret = index->Search(queries, topK, distances, labels);
if (ret == APP_ERR_OK) {
    // 处理结果
}
}

// 2. 带掩码的搜索
void masked_search_example() {
size_t nq = 10;
int topK = 5;
int dim = 128;
int ntotal = index->ntotal; // 索引中的总向量数

float* queryData = new float[nq * dim];
uint8_t* mask = new uint8_t[nq * ((ntotal + 7) / 8)];
float* dists = new float[nq * topK];
int64_t* labels = new int64_t[nq * topK];
// 填充 queryData 和 mask ...
index->Search(nq, mask, queryData, topK, dists, labels);
// 清理内存
}

// 3. 多索引搜索并合并结果
void multi_index_search_example() {
std::vector<NpuIndexIVFHSP*> indexes = {index1, index2};
size_t nq = 10;
int topK = 5;

float* queryData = new float[nq * dim];
float* dists = new float[nq * topK];
int64_t* labels = new int64_t[nq * topK];

NpuIndexIVFHSP::Search(indexes, nq, queryData, topK, dists, labels, true);
}

6. 依赖
Huawei Ascend CANN