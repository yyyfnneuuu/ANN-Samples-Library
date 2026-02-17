from agentic_rag import BlueGreenIndexManager, CodeNode, QueryContext


def test_rrf_and_rerank_return_symbol_hit_first():
    manager = BlueGreenIndexManager()
    manager.swap(
        "v1",
        [
            CodeNode(
                node_id="a",
                module="retrieval",
                file_path="retrieval/fusion.py",
                symbol="fuse_rrf",
                language="python",
                chunk_text="Reciprocal rank fusion for dense and sparse retrieval.",
                neighbors=["b"],
            ),
            CodeNode(
                node_id="b",
                module="retrieval",
                file_path="retrieval/rerank.py",
                symbol="cross_encoder_rerank",
                language="python",
                chunk_text="Re-rank merged candidates for precision.",
                neighbors=["a"],
            ),
        ],
    )

    response = manager.retrieve(
        QueryContext(
            text="fuse dense sparse ranking",
            module_scope="retrieval",
            language="python",
            symbol_hint="fuse_rrf",
        ),
        top_k=2,
    )

    assert response.final_hits
    assert response.final_hits[0].node_id == "a"
