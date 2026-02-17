# 项目三：OpenGauss 内核级向量检索引擎

该目录实现面向数据库内核场景的向量检索核心能力原型，覆盖简历项目三的关键技术路线：

- 内存/磁盘双路径检索（Dual Engine）
- OPQ + RabitQ 量化编码与回表重排
- DiskANN 风格的批量 I/O 调度
- OCC 版本校验并发读路径

## 目录

- `include/opq_rabitq.h` + `src/opq_rabitq.cpp`：OPQ 变换与 RabitQ 编解码
- `include/diskann_scheduler.h` + `src/diskann_scheduler.cpp`：批量 I/O 调度器
- `include/dual_engine_index.h` + `src/dual_engine_index.cpp`：内存/磁盘双引擎检索与评估
- `include/versioned_graph.h` + `src/versioned_graph.cpp`：OCC 版本化图读路径
- `src/demo.cpp`：端到端评估入口（Recall@K + p95）

## 编译与运行

```bash
cd 03-opengauss-vector-engine
cmake -S . -B build
cmake --build build -j
./build/opengauss_vector_demo
```

输出包含双引擎检索的 Recall 与时延指标，以及 OCC 更新前后的遍历路径。
