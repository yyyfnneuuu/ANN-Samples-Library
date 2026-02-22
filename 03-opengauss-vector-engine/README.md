# 项目三：OpenGauss 内核级向量检索引擎

该目录实现面向数据库内核场景的向量检索核心能力原型：

- 内存/磁盘双路径检索
- OPQ + RabitQ 量化编码与回表重排
- DiskANN 批量 I/O 调度
- OCC 版本校验并发读路径

## 目录

- `include/opq_rabitq.h` + `src/opq_rabitq.cpp`：OPQ 变换与 RabitQ 编解码
- `include/diskann_scheduler.h` + `src/diskann_scheduler.cpp`：批量 I/O 调度器
- `include/dual_engine_index.h` + `src/dual_engine_index.cpp`：内存/磁盘双引擎检索与评估
- `include/versioned_graph.h` + `src/versioned_graph.cpp`：OCC 版本化图读路径
- `src/demo.cpp`：入口

## 编译与运行

```bash
cd 03-opengauss-vector-engine
cmake -S . -B build
cmake --build build -j
./build/opengauss_vector_demo
```