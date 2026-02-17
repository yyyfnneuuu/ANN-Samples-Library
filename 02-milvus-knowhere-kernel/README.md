# 项目二：Milvus/Knowhere 高吞吐检索链路优化

该目录实现图检索执行链路的可编译原型，重点对应简历项目二中的三类优化：

- 异步流水线：邻居预取与距离计算分离并批量并行
- TopK 规约算子：使用 bounded heap 增量维护候选集
- 过滤前移：过滤节点不进入结果集，但保留图连通扩展

## 目录

- `include/graph_types.h`：图节点、查询请求、运行统计结构
- `include/async_graph_searcher.h`：Baseline / Optimized 双路径检索接口
- `src/async_graph_searcher.cpp`：异步预取 + 批处理执行实现
- `src/topk_reducer.cpp`：候选集规约算子
- `src/demo.cpp`：基准对照入口（P50/P95、速度提升）

## 编译与运行

```bash
cd 02-milvus-knowhere-kernel
cmake -S . -B build
cmake --build build -j
./build/knowhere_kernel_demo_app
```

输出会包含 baseline 与 optimized 的时延分位数和 speedup。
