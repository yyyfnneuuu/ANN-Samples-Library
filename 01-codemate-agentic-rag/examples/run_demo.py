"""End-to-end run: build index from local repo and execute representative queries."""

from __future__ import annotations

import pathlib
import tempfile
from datetime import datetime

from agentic_rag import BlueGreenIndexManager, QueryContext


QUERIES = [
    QueryContext(text="How is reciprocal rank fusion implemented", module_scope="agentic_rag"),
    QueryContext(text="incremental index build and atomic version switch", module_scope="agentic_rag"),
    QueryContext(text="parse symbols from cpp files", module_scope="agentic_rag", symbol_hint="_extract_symbols"),
]


def print_response(query: QueryContext, manager: BlueGreenIndexManager) -> None:
    response = manager.retrieve(query, top_k=5)
    index = manager.active_index()

    print(f"\nquery: {query.text}")
    print(f"routes: {', '.join(response.routes)}")
    for hit in response.final_hits:
        node = index.nodes[hit.node_id]
        print(
            f"  #{hit.rank:<2} score={hit.score:.4f} "
            f"{node.file_path}:{node.start_line}-{node.end_line} symbol={node.symbol}"
        )


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="codemate-index-") as temp_dir:
        manager = BlueGreenIndexManager(snapshot_dir=pathlib.Path(temp_dir))
        version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
        changed_files, total_nodes = manager.build_from_repo(repo_root=repo_root, version=version)

        print("index build done")
        print(f"  version={version}")
        print(f"  changed_files={changed_files}")
        print(f"  total_nodes={total_nodes}")

        for query in QUERIES:
            print_response(query, manager)


if __name__ == "__main__":
    main()
