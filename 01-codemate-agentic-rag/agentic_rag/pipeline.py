"""Index lifecycle, blue/green switch, and evaluation pipeline."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .index import HierarchicalCodeIndex
from .ingestion import RepositoryIngestor
from .models import CodeNode, QueryContext
from .retrieval import MultiPathRetriever, RetrievalResponse
from .store import IndexSnapshot, SnapshotStore


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
    """Manages index versions, activation, and incremental rebuild."""

    def __init__(self, snapshot_dir: pathlib.Path | None = None) -> None:
        self._versions: Dict[str, HierarchicalCodeIndex] = {}
        self._active_version: str | None = None
        self.store = SnapshotStore(snapshot_dir) if snapshot_dir else None
        self.ingestor = RepositoryIngestor()

        if self.store:
            snapshot = self.store.load_active_snapshot()
            if snapshot:
                self._versions[snapshot.version] = HierarchicalCodeIndex(snapshot.nodes)
                self._active_version = snapshot.version

    @property
    def active_version(self) -> str | None:
        return self._active_version

    def load_version(self, version: str, nodes: Iterable[CodeNode]) -> None:
        self._versions[version] = HierarchicalCodeIndex(nodes)

    def activate(self, version: str) -> None:
        if version not in self._versions:
            raise KeyError(f"Index version not found: {version}")
        self._active_version = version
        if self.store:
            self.store.activate(version)

    def swap(self, version: str, nodes: Iterable[CodeNode], manifest: Dict[str, str] | None = None) -> None:
        """Build new index offline first, then atomically switch active version."""

        nodes_list = list(nodes)
        self.load_version(version, nodes_list)
        if self.store:
            snapshot = IndexSnapshot.create(version=version, manifest=manifest or {}, nodes=nodes_list)
            self.store.save_snapshot(snapshot)
        self.activate(version)

    def build_from_repo(self, repo_root: pathlib.Path, version: str, incremental: bool = True) -> Tuple[int, int]:
        """Builds index from repository, preserving unchanged chunks on incremental updates.

        Returns:
            (changed_file_count, total_node_count)
        """

        repo_root = repo_root.resolve()
        previous_snapshot = self.store.load_active_snapshot() if self.store and incremental else None

        if not previous_snapshot:
            ingest = self.ingestor.ingest(repo_root)
            self.swap(version, ingest.nodes, manifest=ingest.manifest)
            return len(ingest.manifest), len(ingest.nodes)

        new_manifest = self.ingestor.scan_manifest(repo_root)
        changed, deleted = _diff_manifest(previous_snapshot.manifest, new_manifest)
        if not changed and not deleted:
            # Persist a lightweight metadata-only version switch to keep release flow consistent.
            self.swap(version, previous_snapshot.nodes, manifest=new_manifest)
            return 0, len(previous_snapshot.nodes)

        changed_ingest = self.ingestor.ingest(repo_root, only_files=changed)

        retained_nodes = [
            node
            for node in previous_snapshot.nodes
            if node.file_path not in changed and node.file_path not in deleted
        ]
        merged_nodes = retained_nodes + changed_ingest.nodes

        self.swap(version, merged_nodes, manifest=new_manifest)
        return len(changed) + len(deleted), len(merged_nodes)

    def retrieve(self, query: QueryContext, top_k: int = 8) -> RetrievalResponse:
        if not self._active_version:
            raise RuntimeError("No active index version")
        retriever = MultiPathRetriever(self.active_index())
        return retriever.retrieve(query, top_k=top_k)

    def active_index(self) -> HierarchicalCodeIndex:
        if not self._active_version:
            raise RuntimeError("No active index version")
        return self._versions[self._active_version]

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


def _diff_manifest(old_manifest: Dict[str, str], new_manifest: Dict[str, str]) -> Tuple[set[str], set[str]]:
    changed = {
        path
        for path, digest in new_manifest.items()
        if path not in old_manifest or old_manifest[path] != digest
    }
    deleted = set(old_manifest.keys()) - set(new_manifest.keys())
    return changed, deleted
