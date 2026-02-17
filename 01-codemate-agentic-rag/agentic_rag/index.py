"""In-memory hierarchical index for CodeMate-style retrieval demos."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from typing import Dict, Iterable, List, Set

from .models import CodeNode, QueryContext, ScoredNode

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

# A tiny semantic expansion table to mimic embedding recall behavior.
SEMANTIC_EXPANSION = {
    "latency": {"p95", "throughput", "qps", "delay"},
    "symbol": {"identifier", "function", "method", "api"},
    "graph": {"dependency", "callgraph", "edge", "neighbor"},
    "index": {"hnsw", "ivf", "diskann", "bitmap"},
    "consistency": {"transaction", "outbox", "atomic", "sync"},
}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HierarchicalCodeIndex:
    """Indexes code chunks while keeping module/file/symbol metadata."""

    def __init__(self, nodes: Iterable[CodeNode]):
        self.nodes: Dict[str, CodeNode] = {node.node_id: node for node in nodes}
        self.node_tokens: Dict[str, List[str]] = {}
        self.inverted: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_lookup: Dict[str, Set[str]] = defaultdict(set)
        self.graph: Dict[str, Set[str]] = defaultdict(set)

        for node in self.nodes.values():
            full_text = (
                f"{node.module} {node.file_path} {node.symbol} {node.chunk_text} "
                f"{' '.join(node.metadata.values())}"
            )
            tokens = tokenize(full_text)
            self.node_tokens[node.node_id] = tokens
            for token in set(tokens):
                self.inverted[token].add(node.node_id)
            self.symbol_lookup[node.symbol.lower()].add(node.node_id)
            for neighbor in node.neighbors:
                self.graph[node.node_id].add(neighbor)
                self.graph[neighbor].add(node.node_id)

        self._avg_doc_len = (
            sum(len(tokens) for tokens in self.node_tokens.values()) / len(self.node_tokens)
            if self.node_tokens
            else 1.0
        )

    def search_dense(self, query: QueryContext, top_k: int = 8) -> List[ScoredNode]:
        query_tokens = self._expand_tokens(tokenize(query.text))
        filtered_ids = self._filtered_node_ids(query)
        candidate_ids = set()
        for token in query_tokens:
            candidate_ids.update(self.inverted.get(token, set()))
        if not candidate_ids:
            candidate_ids = filtered_ids
        else:
            candidate_ids &= filtered_ids

        scored = {}
        for node_id in candidate_ids:
            node_tokens = set(self.node_tokens[node_id])
            overlap = len(node_tokens & set(query_tokens))
            score = overlap / max(1.0, len(set(query_tokens)))
            if query.symbol_hint and query.symbol_hint.lower() in self.nodes[node_id].symbol.lower():
                score += 0.25
            scored[node_id] = score

        return self._rank("dense", scored, top_k)

    def search_sparse(self, query: QueryContext, top_k: int = 8) -> List[ScoredNode]:
        query_tokens = tokenize(query.text)
        filtered_ids = self._filtered_node_ids(query)

        scores = {}
        k1 = 1.2
        b = 0.75
        for node_id in filtered_ids:
            tf = Counter(self.node_tokens[node_id])
            dl = len(self.node_tokens[node_id])
            bm25 = 0.0
            for token in query_tokens:
                if tf[token] == 0:
                    continue
                idf = self._idf(token)
                numerator = tf[token] * (k1 + 1)
                denominator = tf[token] + k1 * (1 - b + b * dl / self._avg_doc_len)
                bm25 += idf * numerator / denominator
            if bm25 > 0:
                scores[node_id] = bm25

        return self._rank("sparse", scores, top_k)

    def search_symbol(self, query: QueryContext, top_k: int = 8) -> List[ScoredNode]:
        hint = (query.symbol_hint or "").strip().lower()
        if not hint:
            for token in tokenize(query.text):
                if token.endswith("service") or token.endswith("index"):
                    hint = token
                    break
        if not hint:
            return []

        filtered_ids = self._filtered_node_ids(query)
        scores = {}
        for symbol, node_ids in self.symbol_lookup.items():
            score = 0.0
            if symbol == hint:
                score = 1.0
            elif symbol.startswith(hint):
                score = 0.85
            elif hint in symbol:
                score = 0.7
            if score == 0.0:
                continue
            for node_id in node_ids:
                if node_id in filtered_ids:
                    scores[node_id] = max(scores.get(node_id, 0.0), score)

        return self._rank("symbol", scores, top_k)

    def search_graph(self, query: QueryContext, top_k: int = 8) -> List[ScoredNode]:
        filtered_ids = self._filtered_node_ids(query)
        seed = self.search_symbol(query, top_k=3)
        if not seed:
            seed = self.search_dense(query, top_k=3)

        scores: Dict[str, float] = {}
        queue = deque((hit.node_id, 0) for hit in seed)
        visited = {hit.node_id for hit in seed}

        while queue:
            node_id, depth = queue.popleft()
            base_score = 1.0 / (depth + 1)
            if node_id in filtered_ids:
                scores[node_id] = max(scores.get(node_id, 0.0), base_score)
            if depth >= query.graph_hops:
                continue
            for neighbor in self.graph.get(node_id, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

        return self._rank("graph", scores, top_k)

    def _filtered_node_ids(self, query: QueryContext) -> Set[str]:
        selected = set()
        for node in self.nodes.values():
            if query.module_scope and query.module_scope.lower() not in node.module.lower():
                continue
            if query.language and query.language.lower() != node.language.lower():
                continue
            selected.add(node.node_id)
        return selected

    def _expand_tokens(self, query_tokens: List[str]) -> List[str]:
        expanded = list(query_tokens)
        for token in query_tokens:
            expanded.extend(SEMANTIC_EXPANSION.get(token, ()))
        return expanded

    def _idf(self, token: str) -> float:
        total = len(self.nodes)
        contain = len(self.inverted.get(token, ()))
        return math.log((total + 1) / (contain + 1)) + 1.0

    @staticmethod
    def _rank(route: str, score_map: Dict[str, float], top_k: int) -> List[ScoredNode]:
        ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            ScoredNode(node_id=node_id, route=route, score=score, rank=idx + 1)
            for idx, (node_id, score) in enumerate(ranked)
        ]
