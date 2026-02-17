# 项目一：CodeMate Agentic RAG 代码搜索系统（样例）

该目录是对简历项目一的可运行化样例实现，聚焦以下能力：

- `Module-File-Symbol-Chunk` 四层结构化索引
- Agent 动态规划多路检索（Dense / Sparse / Symbol / Graph）
- RRF 融合 + 轻量重排（模拟 Cross-Encoder）
- 结构化过滤（模块、语言）
- 蓝绿索引版本切换（模拟 `<3s` 原子切换）
- EvalCase 自动评估（Hit@5 / MRR）

## 目录

- `agentic_rag/models.py`：核心数据模型
- `agentic_rag/index.py`：层级索引 + 四路检索
- `agentic_rag/retrieval.py`：检索规划、并行执行、RRF 融合、重排
- `agentic_rag/pipeline.py`：双版本索引管理与评估
- `examples/run_demo.py`：端到端演示
- `tests/test_retrieval.py`：最小单测

## 快速运行

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 examples/run_demo.py
```

## 运行测试

```bash
cd 01-codemate-agentic-rag
PYTHONPATH=. python3 -m pytest -q
```

## 与简历项目对应关系

- 多路异构检索 + RRF：`agentic_rag/retrieval.py`
- 图约束与过滤：`agentic_rag/index.py`
- 流批更新与原子切换：`agentic_rag/pipeline.py`
- 自动化评估：`agentic_rag/pipeline.py` + `examples/run_demo.py`
