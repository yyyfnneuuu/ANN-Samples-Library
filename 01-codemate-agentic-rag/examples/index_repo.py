"""Index a repository and atomically activate the new version."""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime

from agentic_rag import BlueGreenIndexManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build code index for a repository")
    parser.add_argument("repo_root", type=pathlib.Path, help="Path to repository root")
    parser.add_argument(
        "--snapshot-dir",
        type=pathlib.Path,
        default=pathlib.Path(".code_index"),
        help="Directory for index snapshots",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=datetime.utcnow().strftime("v%Y%m%d%H%M%S"),
        help="Version label for this build",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Build full index instead of incremental update",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = BlueGreenIndexManager(snapshot_dir=args.snapshot_dir)
    changed_files, total_nodes = manager.build_from_repo(
        repo_root=args.repo_root,
        version=args.version,
        incremental=not args.full,
    )

    print(f"version={args.version}")
    print(f"changed_files={changed_files}")
    print(f"total_nodes={total_nodes}")
    print(f"active_version={manager.active_version}")


if __name__ == "__main__":
    main()
