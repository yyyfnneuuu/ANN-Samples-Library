# 项目二：Milvus/Knowhere 内核增强（样例）

该目录对应简历项目二，展示三类核心优化思路：

- 检索链路异步流水线：`邻居预取` 与 `距离计算` 并行
- TopK 候选集规约：避免每轮全量排序，使用 bounded heap 增量维护
- 过滤前移与连通性补偿：过滤节点不入结果，但可作为桥接节点继续扩展

## 目录

- `include/graph_types.h`：图节点与检索请求定义
- `include/topk_reducer.h` + `src/topk_reducer.cpp`：TopK 规约算子
- `include/async_graph_searcher.h` + `src/async_graph_searcher.cpp`：异步流水线检索器
- `src/demo.cpp`：可执行演示

## 编译运行

```bash
cd 02-milvus-knowhere-kernel
cmake -S . -B build
cmake --build build -j
./build/knowhere_kernel_demo_app
```

## 与简历项目对应关系

- 异步 Pipeline：`src/async_graph_searcher.cpp`
- TopK 线性规约思路：`src/topk_reducer.cpp`
- 结构化过滤前移：`src/async_graph_searcher.cpp` 的 `PassFilter` + 邻居扩展逻辑
