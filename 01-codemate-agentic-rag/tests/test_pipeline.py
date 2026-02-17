from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agentic_rag import BlueGreenIndexManager, QueryContext


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_incremental_build_and_query(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(
        repo / "src" / "retrieval.py",
        """
def fuse_rrf(routes):
    return routes
""",
    )
    _write(
        repo / "src" / "pipeline.py",
        """
from src.retrieval import fuse_rrf

def build_index():
    return fuse_rrf([])
""",
    )

    manager = BlueGreenIndexManager(snapshot_dir=tmp_path / ".index")
    v1 = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    changed, total = manager.build_from_repo(repo_root=repo, version=v1, incremental=True)

    assert changed == 2
    assert total > 0

    response = manager.retrieve(QueryContext(text="fuse rrf retrieval", module_scope="src"), top_k=3)
    assert response.final_hits

    _write(
        repo / "src" / "retrieval.py",
        """
def fuse_rrf(routes):
    return sorted(routes)
""",
    )

    changed2, total2 = manager.build_from_repo(repo_root=repo, version=v1 + "a", incremental=True)
    assert changed2 == 1
    assert total2 > 0
