"""Query an active code index snapshot."""

from __future__ import annotations

import argparse
import pathlib

from agentic_rag import BlueGreenIndexManager, QueryContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query indexed codebase")
    parser.add_argument("query", type=str, help="Natural language or symbol query")
    parser.add_argument(
        "--snapshot-dir",
        type=pathlib.Path,
        default=pathlib.Path(".code_index"),
        help="Directory for index snapshots",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--module", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = BlueGreenIndexManager(snapshot_dir=args.snapshot_dir)
    if not manager.active_version:
        raise RuntimeError("No active index version found. Run index_repo.py first.")

    context = QueryContext(
        text=args.query,
        module_scope=args.module,
        language=args.language,
        symbol_hint=args.symbol,
    )
    response = manager.retrieve(context, top_k=args.top_k)
    active_index = manager.active_index()

    print(f"active_version={manager.active_version}")
    print(f"routes={','.join(response.routes)}")
    print("results:")
    for item in response.final_hits:
        node = active_index.nodes[item.node_id]
        print(
            f"  rank={item.rank} score={item.score:.4f} file={node.file_path}:{node.start_line}-{node.end_line} symbol={node.symbol}"
        )


if __name__ == "__main__":
    main()
