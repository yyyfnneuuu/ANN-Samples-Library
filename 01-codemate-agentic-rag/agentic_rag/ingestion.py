"""Repository ingestion pipeline for hierarchical code indexing."""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .models import CodeNode

LANG_BY_SUFFIX = {
    ".py": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
}

PY_SYMBOL = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
CPP_SYMBOL = re.compile(
    r"^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct|enum|namespace|"
    r"(?:[A-Za-z_:][A-Za-z0-9_:<>~]*\s+)+)([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|\{)",
    re.MULTILINE,
)
GENERIC_SYMBOL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
INCLUDE_RE = re.compile(r"^\s*#include\s+[\"<]([^\">]+)[\">]", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass
class IngestionResult:
    nodes: List[CodeNode]
    manifest: Dict[str, str]


class RepositoryIngestor:
    """Builds CodeNodes from a repository on disk."""

    def __init__(self, max_file_size: int = 2_000_000) -> None:
        self.max_file_size = max_file_size

    def ingest(
        self,
        repo_root: pathlib.Path,
        only_files: Set[str] | None = None,
    ) -> IngestionResult:
        repo_root = repo_root.resolve()
        files = list(self._iter_source_files(repo_root, only_files=only_files))
        manifest = self.scan_manifest(repo_root, only_files=only_files, _files_cache=files)

        nodes: List[CodeNode] = []
        per_file_nodes: Dict[str, List[str]] = defaultdict(list)
        symbol_to_nodes: Dict[str, List[str]] = defaultdict(list)

        for path in files:
            rel_path = str(path.relative_to(repo_root))
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")

            language = LANG_BY_SUFFIX[path.suffix.lower()]
            file_nodes = self._chunk_file(rel_path, text, language)
            nodes.extend(file_nodes)
            per_file_nodes[rel_path] = [node.node_id for node in file_nodes]
            for node in file_nodes:
                symbol_to_nodes[node.symbol.lower()].append(node.node_id)

        node_by_id = {node.node_id: node for node in nodes}
        self._link_file_locality(nodes, per_file_nodes)
        self._link_import_edges(node_by_id, per_file_nodes, repo_root, files)
        self._link_symbol_references(nodes, symbol_to_nodes)

        return IngestionResult(nodes=nodes, manifest=manifest)

    def scan_manifest(
        self,
        repo_root: pathlib.Path,
        only_files: Set[str] | None = None,
        _files_cache: Sequence[pathlib.Path] | None = None,
    ) -> Dict[str, str]:
        repo_root = repo_root.resolve()
        files = list(_files_cache) if _files_cache is not None else list(self._iter_source_files(repo_root, only_files))
        return {str(path.relative_to(repo_root)): self._sha1(path) for path in files}

    def _iter_source_files(
        self,
        repo_root: pathlib.Path,
        only_files: Set[str] | None,
    ) -> Iterable[pathlib.Path]:
        for path in repo_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in LANG_BY_SUFFIX:
                continue
            if path.stat().st_size > self.max_file_size:
                continue
            if any(part in {".git", "build", "dist", "node_modules", "__pycache__"} for part in path.parts):
                continue

            rel = str(path.relative_to(repo_root))
            if only_files is not None and rel not in only_files:
                continue
            yield path

    def _chunk_file(self, rel_path: str, text: str, language: str) -> List[CodeNode]:
        lines = text.splitlines()
        if not lines:
            return []

        chunk_size, overlap = self._chunk_policy(language, len(lines))
        symbols = self._extract_symbols(text, language)

        nodes: List[CodeNode] = []
        cursor = 0
        index = 0
        while cursor < len(lines):
            start = cursor
            end = min(len(lines), cursor + chunk_size)
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)
            chunk_symbol = self._best_symbol(symbols, start + 1, end)

            node_id = hashlib.sha1(f"{rel_path}:{start+1}:{end}:{chunk_text[:512]}".encode("utf-8")).hexdigest()[:16]
            module = rel_path.split("/", 1)[0] if "/" in rel_path else "root"

            nodes.append(
                CodeNode(
                    node_id=node_id,
                    module=module,
                    file_path=rel_path,
                    symbol=chunk_symbol,
                    language=language,
                    chunk_text=chunk_text,
                    start_line=start + 1,
                    end_line=end,
                    metadata={
                        "chunk_index": str(index),
                        "line_count": str(end - start),
                    },
                )
            )
            index += 1
            if end == len(lines):
                break
            cursor = end - overlap

        return nodes

    @staticmethod
    def _chunk_policy(language: str, line_count: int) -> Tuple[int, int]:
        if language == "markdown":
            base_size, base_overlap = 140, 30
        else:
            base_size, base_overlap = 100, 25

        if line_count <= 120:
            return line_count, 0
        if line_count > 1000:
            return base_size + 60, base_overlap + 10
        if line_count > 500:
            return base_size + 30, base_overlap + 5
        return base_size, base_overlap

    @staticmethod
    def _extract_symbols(text: str, language: str) -> List[Tuple[str, int]]:
        symbols: List[Tuple[str, int]] = []
        if language == "python":
            pattern = PY_SYMBOL
        elif language in {"cpp", "c", "java", "go", "rust"}:
            pattern = CPP_SYMBOL
        else:
            pattern = GENERIC_SYMBOL

        for match in pattern.finditer(text):
            name = match.group(1)
            line = text[: match.start()].count("\n") + 1
            symbols.append((name, line))
        return symbols

    @staticmethod
    def _best_symbol(symbols: Sequence[Tuple[str, int]], start_line: int, end_line: int) -> str:
        candidate = "global_scope"
        min_distance = 1_000_000
        for name, line in symbols:
            if start_line <= line <= end_line:
                return name
            distance = min(abs(line - start_line), abs(line - end_line))
            if distance < min_distance:
                min_distance = distance
                candidate = name
        return candidate

    @staticmethod
    def _sha1(path: pathlib.Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _link_file_locality(nodes: List[CodeNode], per_file_nodes: Dict[str, List[str]]) -> None:
        node_by_id = {node.node_id: node for node in nodes}
        for node_ids in per_file_nodes.values():
            for idx, node_id in enumerate(node_ids):
                if idx > 0:
                    node_by_id[node_id].neighbors.append(node_ids[idx - 1])
                if idx + 1 < len(node_ids):
                    node_by_id[node_id].neighbors.append(node_ids[idx + 1])

    @staticmethod
    def _link_import_edges(
        node_by_id: Dict[str, CodeNode],
        per_file_nodes: Dict[str, List[str]],
        repo_root: pathlib.Path,
        files: Sequence[pathlib.Path],
    ) -> None:
        rel_to_entry = {str(path.relative_to(repo_root)): per_file_nodes.get(str(path.relative_to(repo_root)), []) for path in files}
        basename_to_rel = defaultdict(list)
        for rel_path in rel_to_entry:
            basename_to_rel[pathlib.Path(rel_path).stem].append(rel_path)

        for node in node_by_id.values():
            includes = INCLUDE_RE.findall(node.chunk_text)
            imports = [left or right for left, right in IMPORT_RE.findall(node.chunk_text)]

            targets = set()
            for inc in includes:
                stem = pathlib.Path(inc).stem
                targets.update(basename_to_rel.get(stem, []))
            for imp in imports:
                stem = imp.split(".")[-1]
                targets.update(basename_to_rel.get(stem, []))

            for target in targets:
                node_ids = rel_to_entry.get(target, [])
                if node_ids:
                    node.neighbors.append(node_ids[0])

    @staticmethod
    def _link_symbol_references(nodes: List[CodeNode], symbol_to_nodes: Dict[str, List[str]]) -> None:
        for node in nodes:
            tokens = {token.lower() for token in TOKEN_RE.findall(node.chunk_text)}
            current_symbol = node.symbol.lower()
            for token in tokens:
                if token == current_symbol:
                    continue
                targets = symbol_to_nodes.get(token, [])
                if not targets:
                    continue
                node.neighbors.append(targets[0])

            node.neighbors = sorted(set(node.neighbors) - {node.node_id})
