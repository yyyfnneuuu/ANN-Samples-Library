"""Index lifecycle and evaluation pipeline for the CodeMate showcase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .index import HierarchicalCodeIndex
from .models import CodeNode, QueryContext
from .retrieval import MultiPathRetriever, RetrievalResponse


@dataclass
class EvalCase:
    name: str
    query: QueryContext
    expected_node_id: str


@dataclass
class EvalReport:
    total: int
    hit_at_5: float
    mrr: float


class BlueGreenIndexManager:
    """Supports dual-version indexing and atomic switch-over."""

    def __init__(self) -> None:
        self._versions: Dict[str, HierarchicalCodeIndex] = {}
        self._active_version: str | None = None

    @property
    def active_version(self) -> str | None:
        return self._active_version

    def load_version(self, version: str, nodes: Iterable[CodeNode]) -> None:
        self._versions[version] = HierarchicalCodeIndex(nodes)

    def activate(self, version: str) -> None:
        if version not in self._versions:
            raise KeyError(f"Index version not found: {version}")
        self._active_version = version

    def swap(self, version: str, nodes: Iterable[CodeNode]) -> None:
        """Build new index offline first, then atomically switch version."""

        self.load_version(version, nodes)
        self.activate(version)

    def retrieve(self, query: QueryContext, top_k: int = 8) -> RetrievalResponse:
        if not self._active_version:
            raise RuntimeError("No active index version")
        retriever = MultiPathRetriever(self._versions[self._active_version])
        return retriever.retrieve(query, top_k=top_k)

    def evaluate(self, cases: List[EvalCase]) -> EvalReport:
        if not cases:
            return EvalReport(total=0, hit_at_5=0.0, mrr=0.0)

        hit = 0
        reciprocal_ranks = 0.0
        for case in cases:
            response = self.retrieve(case.query, top_k=5)
            ranked_ids = [item.node_id for item in response.final_hits]
            if case.expected_node_id in ranked_ids:
                hit += 1
                rank = ranked_ids.index(case.expected_node_id) + 1
                reciprocal_ranks += 1.0 / rank

        return EvalReport(
            total=len(cases),
            hit_at_5=hit / len(cases),
            mrr=reciprocal_ranks / len(cases),
        )
