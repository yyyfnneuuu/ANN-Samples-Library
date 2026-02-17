"""Versioned snapshot storage for index blue/green deployment."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List

from .models import CodeNode


@dataclass
class IndexSnapshot:
    version: str
    created_at: str
    manifest: Dict[str, str]
    nodes: List[CodeNode]

    @classmethod
    def create(cls, version: str, manifest: Dict[str, str], nodes: List[CodeNode]) -> "IndexSnapshot":
        return cls(
            version=version,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            manifest=manifest,
            nodes=nodes,
        )

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "version": self.version,
            "created_at": self.created_at,
            "manifest": self.manifest,
            "nodes": [asdict(node) for node in self.nodes],
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "IndexSnapshot":
        nodes = [CodeNode(**item) for item in payload.get("nodes", [])]
        return cls(
            version=str(payload["version"]),
            created_at=str(payload["created_at"]),
            manifest=dict(payload.get("manifest", {})),
            nodes=nodes,
        )


class SnapshotStore:
    """Stores immutable index snapshots and an atomically switched active pointer."""

    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = base_dir
        self.versions_dir = base_dir / "versions"
        self.pointer_path = base_dir / "active_version.txt"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, snapshot: IndexSnapshot) -> pathlib.Path:
        path = self.versions_dir / f"{snapshot.version}.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
        return path

    def load_snapshot(self, version: str) -> IndexSnapshot:
        path = self.versions_dir / f"{version}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return IndexSnapshot.from_dict(payload)

    def activate(self, version: str) -> None:
        temp = self.pointer_path.with_suffix(".tmp")
        temp.write_text(version, encoding="utf-8")
        os.replace(temp, self.pointer_path)

    def active_version(self) -> str | None:
        if not self.pointer_path.exists():
            return None
        value = self.pointer_path.read_text(encoding="utf-8").strip()
        return value or None

    def load_active_snapshot(self) -> IndexSnapshot | None:
        version = self.active_version()
        if not version:
            return None
        return self.load_snapshot(version)
