"""Domain models for the CodeMate Agentic RAG showcase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CodeNode:
    """A chunk-level unit in the Module-File-Symbol-Chunk hierarchy."""

    node_id: str
    module: str
    file_path: str
    symbol: str
    language: str
    chunk_text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    neighbors: List[str] = field(default_factory=list)


@dataclass
class QueryContext:
    """Planning hints consumed by the retrieval agent."""

    text: str
    module_scope: str | None = None
    language: str | None = None
    symbol_hint: str | None = None
    graph_hops: int = 2


@dataclass
class ScoredNode:
    """Standardized search output from each retrieval route."""

    node_id: str
    route: str
    score: float
    rank: int
