# ANN-Samples-Library

该仓库已按简历三段核心项目重构为可展示、可运行的样例代码。

## 项目结构

- `01-codemate-agentic-rag/`：项目一，CodeMate Agentic RAG 代码搜索系统（Python）
- `02-milvus-knowhere-kernel/`：项目二，Milvus/Knowhere 内核吞吐优化（C++17）
- `03-opengauss-vector-engine/`：项目三，OpenGauss 向量引擎核心能力（C++17）
- `legacy/`：原始历史样例代码归档

## 快速开始

### 项目一（Python）

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 examples/run_demo.py
PYTHONPATH=. python3 -m pytest -q
```

### 项目二（C++17）

```bash
cd 02-milvus-knowhere-kernel
cmake -S . -B build
cmake --build build -j
./build/knowhere_kernel_demo_app
```

### 项目三（C++17）

```bash
cd 03-opengauss-vector-engine
cmake -S . -B build
cmake --build build -j
./build/opengauss_vector_demo
```

## 与简历映射

1. 华为 CodeMate — 基于 Milvus 的 Agentic RAG 代码搜索系统
- 目录：`01-codemate-agentic-rag/`
- 体现：多路检索规划、RRF 融合、结构化过滤、双版本索引切换、EvalCase 评估

2. Milvus/Knowhere 内核增强—自研高吞吐索引优化
- 目录：`02-milvus-knowhere-kernel/`
- 体现：异步流水线、TopK 规约、过滤前移与连通性补偿

3. OpenGauss 内核级向量检索引擎研发
- 目录：`03-opengauss-vector-engine/`
- 体现：OPQ/RabitQ、DiskANN I/O 调度、OCC 版本化并发读

## 说明

- 代码为简历展示型样例，关注设计与关键算法思想，不包含任何商业机密实现。
- 可作为技术面试讲解材料，也可继续按需要扩展为完整工程。
