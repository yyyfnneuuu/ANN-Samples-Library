# 项目三：OpenGauss 内核级向量检索引擎（样例）

该目录对应简历项目三，覆盖三个核心模块：

- `HNSW/DiskANN` 双引擎场景中的关键能力抽象
- `OPQ + RabitQ` 高压缩量化路径（4-7bit）
- 磁盘随机读批量调度 + OCC 版本化读路径

## 目录

- `include/opq_rabitq.h` + `src/opq_rabitq.cpp`：OPQ 变换与 RabitQ 量化编码
- `include/diskann_scheduler.h` + `src/diskann_scheduler.cpp`：磁盘 I/O 批量调度
- `include/versioned_graph.h` + `src/versioned_graph.cpp`：版本号校验的 OCC 图读取
- `src/demo.cpp`：端到端演示

## 编译运行

```bash
cd 03-opengauss-vector-engine
cmake -S . -B build
cmake --build build -j
./build/opengauss_vector_demo
```

## 与简历项目对应关系

- OPQ / RabitQ：`src/opq_rabitq.cpp`
- DiskANN I/O 优化：`src/diskann_scheduler.cpp`
- OCC 并发控制：`src/versioned_graph.cpp`
