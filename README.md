# ANN-Samples-Library

该仓库包含三条与简历项目对应的工程代码线，分别覆盖 Agentic RAG 检索系统、向量检索内核优化和数据库内核向量引擎。

## 目录结构

- `01-codemate-agentic-rag/`：CodeMate Agentic RAG 代码检索服务
- `02-milvus-knowhere-kernel/`：Milvus/Knowhere 高吞吐检索链路优化
- `03-opengauss-vector-engine/`：OpenGauss 内核级向量检索引擎

## 运行方式

### 项目一（Python）

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 examples/index_repo.py . --snapshot-dir .code_index
PYTHONPATH=. python3 examples/query_repo.py "rrf fusion" --snapshot-dir .code_index --module agentic_rag
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

## 一体化构建（C++项目）

```bash
cmake -S . -B build
cmake --build build -j
```
