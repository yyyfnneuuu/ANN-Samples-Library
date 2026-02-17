"""Agentic retrieval planner and fusion logic."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List

from .index import HierarchicalCodeIndex, tokenize
from .models import QueryContext, ScoredNode


@dataclass
class RetrievalResponse:
    routes: List[str]
    route_hits: Dict[str, List[ScoredNode]]
    fused_hits: List[ScoredNode]
    final_hits: List[ScoredNode]


class RetrievalPlanner:
    """Decides which retrieval routes should be activated per query."""

    def plan(self, query: QueryContext) -> List[str]:
        q = query.text.lower()
        routes = ["dense", "sparse"]

        if query.symbol_hint or "::" in q or "(" in q or "service" in q:
            routes.append("symbol")

        graph_keywords = {"graph", "dependency", "caller", "callee", "chain", "filter"}
        if any(keyword in q for keyword in graph_keywords):
            routes.append("graph")

        # Keep graph route as fallback for structure-aware retrieval in codebase.
        if "graph" not in routes:
            routes.append("graph")

        # Deduplicate while preserving order.
        return list(dict.fromkeys(routes))


class MultiPathRetriever:
    """Executes route-parallel retrieval + RRF + lightweight reranking."""

    def __init__(self, index: HierarchicalCodeIndex, planner: RetrievalPlanner | None = None):
        self.index = index
        self.planner = planner or RetrievalPlanner()

    def retrieve(self, query: QueryContext, top_k: int = 8) -> RetrievalResponse:
        routes = self.planner.plan(query)

        route_to_fn: Dict[str, Callable[[QueryContext, int], List[ScoredNode]]] = {
            "dense": self.index.search_dense,
            "sparse": self.index.search_sparse,
            "symbol": self.index.search_symbol,
            "graph": self.index.search_graph,
        }

        route_hits: Dict[str, List[ScoredNode]] = {}
        with ThreadPoolExecutor(max_workers=len(routes)) as executor:
            futures = {
                route: executor.submit(route_to_fn[route], query, top_k)
                for route in routes
                if route in route_to_fn
            }
            for route, future in futures.items():
                route_hits[route] = future.result()

        fused = reciprocal_rank_fusion(route_hits, top_k=top_k * 2)
        reranked = cross_encoder_rerank(self.index, query, fused, top_k=top_k)

        return RetrievalResponse(
            routes=routes,
            route_hits=route_hits,
            fused_hits=fused[:top_k],
            final_hits=reranked,
        )


def reciprocal_rank_fusion(
    route_hits: Dict[str, List[ScoredNode]],
    top_k: int,
    k_bias: int = 60,
) -> List[ScoredNode]:
    """RRF fusion to stabilize heterogeneous score spaces."""

    fused_scores: Dict[str, float] = defaultdict(float)
    for hits in route_hits.values():
        for hit in hits:
            fused_scores[hit.node_id] += 1.0 / (k_bias + hit.rank)

    ranked = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        ScoredNode(node_id=node_id, route="rrf", score=score, rank=idx + 1)
        for idx, (node_id, score) in enumerate(ranked)
    ]


def cross_encoder_rerank(
    index: HierarchicalCodeIndex,
    query: QueryContext,
    fused_hits: List[ScoredNode],
    top_k: int,
) -> List[ScoredNode]:
    """A deterministic proxy of cross-encoder reranking."""

    query_tokens = set(tokenize(query.text))
    rerank_scores = {}

    for hit in fused_hits:
        node = index.nodes[hit.node_id]
        node_tokens = set(index.node_tokens[hit.node_id])
        lexical_precision = len(query_tokens & node_tokens) / max(1.0, len(query_tokens))
        symbol_boost = 0.2 if query.symbol_hint and query.symbol_hint.lower() in node.symbol.lower() else 0.0
        rerank_scores[hit.node_id] = 0.65 * hit.score + 0.35 * lexical_precision + symbol_boost

    ranked = sorted(rerank_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        ScoredNode(node_id=node_id, route="rerank", score=score, rank=idx + 1)
        for idx, (node_id, score) in enumerate(ranked)
    ]
