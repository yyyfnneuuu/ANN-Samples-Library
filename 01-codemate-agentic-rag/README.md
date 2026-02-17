# 项目一：CodeMate Agentic RAG 代码检索服务

该目录实现一个面向代码仓库的 Agentic RAG 检索服务原型，覆盖简历中项目一的核心链路：

- `Module-File-Symbol-Chunk` 分层索引
- Dense / Sparse / Symbol / Graph 多路并行召回
- RRF 融合与重排
- 结构化过滤（模块、语言、符号）
- 蓝绿索引版本切换与增量更新
- EvalCase 评估接口（Hit@5 / MRR）

## 目录

- `agentic_rag/ingestion.py`：仓库扫描、符号抽取、Adaptive Chunking、关系边构建
- `agentic_rag/index.py`：倒排索引、哈希向量索引、图索引
- `agentic_rag/retrieval.py`：检索路由规划、并行执行、RRF、重排
- `agentic_rag/store.py`：版本化快照持久化、active 指针原子切换
- `agentic_rag/pipeline.py`：蓝绿发布与增量构建编排
- `examples/index_repo.py`：构建/发布新版本索引
- `examples/query_repo.py`：在线查询入口

## 快速开始

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 examples/index_repo.py . --snapshot-dir .code_index
PYTHONPATH=. python3 examples/query_repo.py "rrf fusion and rerank" --snapshot-dir .code_index --module agentic_rag
```

## 端到端演示

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 examples/run_demo.py
```

## 测试

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 -m pytest -q
```
