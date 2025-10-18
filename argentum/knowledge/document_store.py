"""Lightweight document store for knowledge retrieval."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]{3,}")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        slice_tokens = tokens[start : start + chunk_size]
        if slice_tokens:
            chunks.append(" ".join(slice_tokens))
    return chunks


def _vectorize(tokens: Sequence[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {token: value / norm for token, value in counts.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    score = 0.0
    for token, weight in vec_a.items():
        score += weight * vec_b.get(token, 0.0)
    return score


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector: dict[str, float]


class DocumentStore:
    """File-backed store for document chunks with simple vector retrieval."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.documents_path = self.base_path / "documents.jsonl"
        self.chunks_path = self.base_path / "chunks.jsonl"

    # ------------------------------------------------------------------ ingestion
    def ingest_file(
        self,
        file_path: Path,
        *,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 200,
        overlap: int = 40,
    ) -> int:
        text = self._load_text(file_path)
        chunk_texts = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        chunk_count = len(chunk_texts)

        doc_record = {
            "doc_id": doc_id,
            "source_path": str(file_path),
            "metadata": metadata or {},
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_count": chunk_count,
            "ingested_at": _timestamp(),
        }
        self._append_json(self.documents_path, doc_record)

        for index, chunk in enumerate(chunk_texts, start=1):
            tokens = _tokenize(chunk)
            vector = _vectorize(tokens)
            chunk_record = DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::chunk-{index}",
                text=chunk,
                metadata={"position": index},
                vector=vector,
            )
            self._append_json(self.chunks_path, {
                "doc_id": chunk_record.doc_id,
                "chunk_id": chunk_record.chunk_id,
                "text": chunk_record.text,
                "metadata": chunk_record.metadata,
                "vector": chunk_record.vector,
            })
        return chunk_count

    def list_documents(self) -> list[dict[str, Any]]:
        return list(self._iter_jsonl(self.documents_path))

    def list_chunks(self) -> list[dict[str, Any]]:
        return list(self._iter_jsonl(self.chunks_path))

    # ------------------------------------------------------------------ retrieval
    def query(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        tokens = _tokenize(query)
        vector = _vectorize(tokens)
        if not vector:
            return []

        scored_chunks: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._iter_jsonl(self.chunks_path):
            similarity = _cosine_similarity(vector, chunk.get("vector", {}))
            if similarity > 0.0:
                scored_chunks.append((similarity, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        top_chunks = []
        for score, chunk in scored_chunks[:limit]:
            top_chunks.append(
                {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "metadata": chunk.get("metadata", {}),
                    "score": score,
                }
            )
        return top_chunks

    # ------------------------------------------------------------------ feedback
    def record_feedback(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        session_id: str,
        rating: str,
        notes: str | None = None,
    ) -> None:
        feedback_path = self.base_path / "feedback.jsonl"
        entry = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "session_id": session_id,
            "rating": rating,
            "notes": notes,
        }
        self._append_json(feedback_path, entry)

    # ------------------------------------------------------------------ utils
    def _load_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ""}:
            return file_path.read_text(encoding="utf-8")
        if suffix in {".md", ".markdown"}:
            return file_path.read_text(encoding="utf-8")
        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _append_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
