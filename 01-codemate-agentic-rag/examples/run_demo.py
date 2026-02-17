"""Run a local demo for the CodeMate Agentic RAG sample."""

from __future__ import annotations

from agentic_rag import BlueGreenIndexManager, CodeNode, EvalCase, QueryContext


def build_nodes_v1() -> list[CodeNode]:
    return [
        CodeNode(
            node_id="n1",
            module="retrieval",
            file_path="retrieval/fusion.py",
            symbol="fuse_rrf",
            language="python",
            chunk_text="Fuse dense sparse and graph routes with reciprocal rank fusion.",
            neighbors=["n2", "n4"],
        ),
        CodeNode(
            node_id="n2",
            module="retrieval",
            file_path="retrieval/rerank.py",
            symbol="cross_encoder_rerank",
            language="python",
            chunk_text="Re-rank top candidates with cross encoder scores for symbol precision.",
            neighbors=["n1", "n3"],
        ),
        CodeNode(
            node_id="n3",
            module="agent",
            file_path="agent/planner.py",
            symbol="plan_routes",
            language="python",
            chunk_text="Plan dense sparse symbol and graph routes from query intent.",
            neighbors=["n2", "n5"],
        ),
        CodeNode(
            node_id="n4",
            module="graph",
            file_path="graph/filter.py",
            symbol="bitmap_filter",
            language="python",
            chunk_text="Apply bitmap constraints before expensive distance computation.",
            neighbors=["n1", "n6"],
        ),
        CodeNode(
            node_id="n5",
            module="indexing",
            file_path="indexing/hierarchical_index.py",
            symbol="build_module_file_symbol_chunk",
            language="python",
            chunk_text="Build module file symbol chunk hierarchy with adaptive chunking.",
            neighbors=["n3", "n7"],
        ),
        CodeNode(
            node_id="n6",
            module="sync",
            file_path="sync/outbox_pipeline.py",
            symbol="transactional_outbox_sync",
            language="python",
            chunk_text="Coordinate vector metadata and graph storage with outbox consistency.",
            neighbors=["n4", "n8"],
        ),
        CodeNode(
            node_id="n7",
            module="indexing",
            file_path="indexing/blue_green.py",
            symbol="atomic_version_switch",
            language="python",
            chunk_text="Support dual version index and less than three seconds lossless switch.",
            neighbors=["n5", "n8"],
        ),
        CodeNode(
            node_id="n8",
            module="evaluation",
            file_path="evaluation/ragas_runner.py",
            symbol="run_ragas_eval",
            language="python",
            chunk_text="Evaluate hit at five mrr and context recall over golden evalcases.",
            neighbors=["n6", "n7"],
        ),
    ]


def build_nodes_v2() -> list[CodeNode]:
    nodes = build_nodes_v1()
    nodes.append(
        CodeNode(
            node_id="n9",
            module="graph",
            file_path="graph/json_property_filter.py",
            symbol="json_graph_filter_pushdown",
            language="python",
            chunk_text="Push code property graph constraints into vector query via json filters.",
            neighbors=["n4", "n6"],
        )
    )
    return nodes


def print_results(title: str, response, manager: BlueGreenIndexManager) -> None:
    print(f"\n=== {title} (active={manager.active_version}) ===")
    print("Routes:", ", ".join(response.routes))
    for item in response.final_hits:
        print(f"#{item.rank:<2} {item.node_id:<3} score={item.score:.4f}")


def main() -> None:
    manager = BlueGreenIndexManager()
    manager.swap("v2025-01", build_nodes_v1())

    query_1 = QueryContext(
        text="How to reduce ranking jitter between dense and sparse search in codemate?",
        module_scope="retrieval",
        language="python",
    )
    response_1 = manager.retrieve(query_1, top_k=5)
    print_results("RRF Ranking Stability", response_1, manager)

    manager.swap("v2025-02", build_nodes_v2())

    query_2 = QueryContext(
        text="graph filter latency and json pushdown optimization",
        module_scope="graph",
        language="python",
        symbol_hint="json_graph_filter_pushdown",
    )
    response_2 = manager.retrieve(query_2, top_k=5)
    print_results("Graph Filter Pushdown", response_2, manager)

    report = manager.evaluate(
        [
            EvalCase("rrf", query_1, expected_node_id="n1"),
            EvalCase("graph", query_2, expected_node_id="n9"),
            EvalCase(
                "consistency",
                QueryContext(
                    text="vector metadata graph outbox consistency",
                    module_scope="sync",
                    language="python",
                    symbol_hint="transactional_outbox_sync",
                ),
                expected_node_id="n6",
            ),
        ]
    )

    print("\n=== Eval Report ===")
    print(f"Total Cases: {report.total}")
    print(f"Hit@5:      {report.hit_at_5:.2f}")
    print(f"MRR:        {report.mrr:.2f}")


if __name__ == "__main__":
    main()
