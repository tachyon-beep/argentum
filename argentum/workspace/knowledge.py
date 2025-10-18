"""Knowledge graph storage and highlights utilities for Argentum workspaces."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from argentum.knowledge.document_store import DocumentStore, _chunk_text, _tokenize, _vectorize
from argentum.models import OrchestrationResult
from argentum.workspace.manager import DEFAULT_RETRIEVAL_CONFIG
from argentum.workspace.warm_store import WarmCacheStore

if TYPE_CHECKING:  # pragma: no cover
    from argentum.workspace.manager import ProjectWorkspace


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


@dataclass(slots=True)
class KnowledgeNode:
    """Represents an entity in the knowledge graph."""

    node_id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeEdge:
    """Represents a relationship between nodes."""

    source: str
    target: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Simple file-backed knowledge graph."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.nodes_path = self.base_path / "nodes.jsonl"
        self.edges_path = self.base_path / "edges.jsonl"
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ---- Node helpers ------------------------------------------------------------------------
    def list_nodes(self) -> list[dict[str, Any]]:
        if not self.nodes_path.exists():
            return []
        with self.nodes_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _write_nodes(self, nodes: Iterable[dict[str, Any]]) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)
        with self.nodes_path.open("w", encoding="utf-8") as handle:
            for node in nodes:
                handle.write(json.dumps(node))
                handle.write("\n")

    def upsert_nodes(self, nodes: Iterable[KnowledgeNode]) -> None:
        current = {node["id"]: node for node in self.list_nodes()}
        for node in nodes:
            current[node.node_id] = {
                "id": node.node_id,
                "type": node.type,
                "attributes": node.attributes,
                "timestamp": _now(),
            }
        self._write_nodes(current.values())

    def query_nodes(
        self,
        *,
        type: str | None = None,
        attribute: str | None = None,
        value: Any | None = None,
    ) -> list[dict[str, Any]]:
        results = self.list_nodes()
        if type:
            results = [node for node in results if node.get("type") == type]
        if attribute is not None:
            results = [
                node
                for node in results
                if isinstance(node.get("attributes"), dict) and node["attributes"].get(attribute) == value
            ]
        return results

    # ---- Edge helpers ------------------------------------------------------------------------
    def list_edges(self) -> list[dict[str, Any]]:
        if not self.edges_path.exists():
            return []
        with self.edges_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _write_edges(self, edges: Iterable[dict[str, Any]]) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)
        with self.edges_path.open("w", encoding="utf-8") as handle:
            for edge in edges:
                handle.write(json.dumps(edge))
                handle.write("\n")

    @staticmethod
    def _edge_key(edge: KnowledgeEdge | dict[str, Any]) -> str:
        if isinstance(edge, KnowledgeEdge):
            return f"{edge.source}|{edge.type}|{edge.target}"
        return f"{edge['source']}|{edge['type']}|{edge['target']}"

    def upsert_edges(self, edges: Iterable[KnowledgeEdge]) -> None:
        current = {self._edge_key(edge): edge for edge in self.list_edges()}
        for edge in edges:
            current[self._edge_key(edge)] = {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "attributes": edge.attributes,
                "timestamp": _now(),
            }
        self._write_edges(current.values())


# ---- Highlights + Knowledge Integration -------------------------------------------------------

def build_session_highlights(
    session_id: str,
    metadata: dict[str, Any],
    result: OrchestrationResult,
    *,
    summary_fn: Callable[[OrchestrationResult, dict[str, Any], Sequence[dict[str, Any]]], str] | None = None,
    action_fn: Callable[[Sequence[dict[str, Any]], OrchestrationResult, dict[str, Any]], Sequence[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute lightweight highlights from an orchestration result."""
    retrieved_docs = metadata.get("retrieved_docs", []) or []
    retrieval_meta = metadata.get("retrieval")
    if isinstance(retrieval_meta, dict):
        retrieval_meta["hit_count"] = len(retrieved_docs)
    doc_lookup = _build_doc_lookup(retrieved_docs)

    quotes: list[dict[str, Any]] = []
    all_citations: list[dict[str, Any]] = []
    citation_seen: set[tuple[str | None, str | None, str | None]] = set()
    speaking_profiles: dict[str, dict[str, Any]] = {}

    for output in result.final_outputs:
        if not output.content:
            continue
        raw_metadata = output.metadata or {}
        raw_citations = []
        if isinstance(raw_metadata, dict):
            raw_citations = raw_metadata.get("citations") or []
        normalized_citations = _normalize_citations(raw_citations, doc_lookup)
        metadata_block = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        if normalized_citations:
            metadata_block["citations"] = normalized_citations
            for citation in normalized_citations:
                key = (
                    citation.get("doc_id"),
                    citation.get("chunk_id"),
                    citation.get("label"),
                )
                if key in citation_seen:
                    continue
                citation_seen.add(key)
                all_citations.append(citation)

        style_profile = {}
        for key in ("speaking_style", "speech_tags", "tts_voice"):
            value = metadata_block.get(key)
            if value:
                style_profile[key] = value
        if isinstance(style_profile.get("speech_tags"), str):
            style_profile["speech_tags"] = [style_profile["speech_tags"]]
        if style_profile:
            speaking_profiles[output.agent_name] = style_profile

        quotes.append(
            {
                "agent": output.agent_name,
                "content": output.content,
                "confidence": output.confidence,
                "metadata": metadata_block,
            }
        )
    agents = result.metadata.get("agent_names", [])
    topic = metadata.get("topic") or metadata.get("question") or metadata.get("memory_topic")

    summary = (
        summary_fn(result, metadata, quotes)
        if summary_fn
        else _default_summary(result, metadata, quotes)
    )
    if summary and all_citations:
        citation_labels = [citation.get("label") or citation.get("doc_id") for citation in all_citations]
        citation_labels = [label for label in citation_labels if label]
        if citation_labels:
            summary = f"{summary}\nSources: {', '.join(citation_labels)}"
    action_items = (
        list(action_fn(quotes, result, metadata))
        if action_fn
        else _derive_action_items(quotes, result, metadata)
    )

    session_style = metadata.get("session_style")
    if not session_style:
        session_style = next((info.get("speaking_style") for info in speaking_profiles.values() if info.get("speaking_style")), None)

    default_voice = metadata.get("default_tts_voice")
    if not default_voice:
        default_voice = next((info.get("tts_voice") for info in speaking_profiles.values() if info.get("tts_voice")), None)

    session_tags = metadata.get("session_speech_tags")
    if isinstance(session_tags, str):
        session_tags = [session_tags]

    items: list[dict[str, Any]] = []

    if result.consensus:
        items.append(
            {
                "session_id": session_id,
                "type": "consensus",
                "agent": None,
                "text": result.consensus,
                "metadata": {
                    "project_id": metadata.get("project_id"),
                    "topic": metadata.get("topic") or metadata.get("question"),
                },
            }
        )

    for quote in quotes:
        quote_metadata = quote.get("metadata") if isinstance(quote.get("metadata"), dict) else {}
        quote_citations = quote_metadata.get("citations") if isinstance(quote_metadata.get("citations"), list) else []
        statement_metadata = {
            "confidence": quote.get("confidence"),
            "project_id": metadata.get("project_id"),
            "topic": metadata.get("topic") or metadata.get("question"),
        }
        if quote_citations:
            statement_metadata["citations"] = quote_citations
        for key in ("speaking_style", "speech_tags", "tts_voice"):
            value = quote_metadata.get(key)
            if value:
                if key == "speech_tags" and isinstance(value, str):
                    statement_metadata[key] = [value]
                else:
                    statement_metadata[key] = value
        items.append(
            {
                "session_id": session_id,
                "type": "statement",
                "agent": quote.get("agent"),
                "text": quote.get("content"),
                "metadata": statement_metadata,
            }
        )

    if summary:
        items.insert(
            0,
            {
                "session_id": session_id,
                "type": "summary",
                "agent": None,
                "text": summary,
                "metadata": {
                    "project_id": metadata.get("project_id"),
                    "topic": topic,
                    "citations": all_citations,
                    **({"speaking_style": session_style} if session_style else {}),
                    **({"tts_voice": default_voice} if default_voice else {}),
                    **({"speech_tags": session_tags} if session_tags else {}),
                },
            },
        )

    for action in action_items:
        action_metadata = {
            "project_id": metadata.get("project_id"),
            "topic": topic,
            "source": action.get("source"),
        }
        if action.get("citations"):
            action_metadata["citations"] = action.get("citations")
        if session_style and "speaking_style" not in action_metadata:
            action_metadata["speaking_style"] = session_style
        if default_voice and "tts_voice" not in action_metadata:
            action_metadata["tts_voice"] = default_voice
        if session_tags and "speech_tags" not in action_metadata:
            action_metadata["speech_tags"] = session_tags
        items.append(
            {
                "session_id": session_id,
                "type": "action",
                "agent": action.get("agent"),
                "text": action.get("text"),
                "metadata": action_metadata,
            }
        )

    highlights = {
        "session_id": session_id,
        "command": metadata.get("command"),
        "consensus": result.consensus,
        "topic": topic,
        "project_id": metadata.get("project_id"),
        "agents": agents,
        "quotes": quotes,
        "timestamp": _now(),
        "summary": summary,
        "action_items": action_items,
        "retrieved_docs": metadata.get("retrieved_docs", []),
        "citations": all_citations,
        "speaking_styles": speaking_profiles,
        "retrieval_history": metadata.get("retrieval_history", []),
        "items": [item for item in items if item.get("text")],
    }
    if session_style:
        highlights["session_style"] = session_style
    if default_voice:
        highlights["default_tts_voice"] = default_voice
    if session_tags:
        highlights["session_speech_tags"] = session_tags

    for retrieved in metadata.get("retrieved_docs", []) or []:
        retrieved_metadata = retrieved.get("metadata") if isinstance(retrieved.get("metadata"), dict) else {}
        document_meta = retrieved_metadata.get("document") if isinstance(retrieved_metadata.get("document"), dict) else {}
        highlight_item = {
            "session_id": session_id,
            "type": "retrieval",
            "agent": None,
            "text": retrieved.get("text"),
            "metadata": {
                "doc_id": retrieved.get("doc_id"),
                "chunk_id": retrieved.get("chunk_id"),
                "score": retrieved.get("score"),
                "label": retrieved.get("label") or document_meta.get("label"),
                "title": document_meta.get("title") or retrieved_metadata.get("title"),
                "tags": document_meta.get("tags") or retrieved_metadata.get("tags"),
                "source_path": retrieved_metadata.get("source_path"),
                "ingested_at": retrieved_metadata.get("ingested_at"),
            },
        }
        highlights["items"].append(highlight_item)

    return highlights


def _normalize_text(text: str, *, preserve_newlines: bool = True) -> str:
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = re.sub(r"\[(Doc\s*\d+[^\]]*)\]", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if preserve_newlines and (not lines or lines[-1] != ""):
                lines.append("")
            continue
        if raw_line.lstrip().startswith("- "):
            line = raw_line.lstrip()[2:].strip()
        elif raw_line.lstrip().startswith("• "):
            line = raw_line.lstrip()[2:].strip()
        lines.append(line)

    if preserve_newlines:
        return "\n".join(lines).strip()
    return " ".join(lines).strip()


def build_tts_markdown(highlights: dict[str, Any]) -> str:
    """Render highlights into narration-friendly Markdown."""

    lines: list[str] = []
    project = highlights.get("project_id") or "project"
    topic = highlights.get("topic") or "session"
    summary = _normalize_text(highlights.get("summary") or "No summary available.")
    citations = highlights.get("citations") or []
    speaking_styles = highlights.get("speaking_styles") or {}
    session_style = highlights.get("session_style")
    default_voice = highlights.get("default_tts_voice")
    session_tags = highlights.get("session_speech_tags") or []
    if isinstance(session_tags, str):
        session_tags = [session_tags]

    lines.append(f"## {project.title()} — {topic}\n")
    style_bits: list[str] = []
    if session_style:
        style_bits.append(f"Tone: {session_style}")
    if default_voice:
        style_bits.append(f"Voice: {default_voice}")
    if session_tags:
        style_bits.append("Descriptors: " + ", ".join(session_tags))
    if speaking_styles:
        voice_desc: list[str] = []
        for agent, info in speaking_styles.items():
            parts: list[str] = []
            style_label = info.get("speaking_style")
            if style_label:
                parts.append(style_label)
            voice_label = info.get("tts_voice")
            if voice_label:
                parts.append(f"voice {voice_label}")
            tags = info.get("speech_tags")
            if isinstance(tags, str):
                tags = [tags]
            if tags:
                parts.append("tags " + ", ".join(tags))
            if parts:
                voice_desc.append(f"{agent} ({', '.join(parts)})")
            else:
                voice_desc.append(agent)
        if voice_desc:
            style_bits.append("Voices: " + ", ".join(voice_desc))
    if style_bits:
        lines.append(" | ".join(style_bits))
        lines.append("")
    lines.append(summary)
    lines.append("")

    for item in highlights.get("items", []):
        item_type = item.get("type")
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if item_type == "statement":
            speaker = item.get("agent") or "Unknown"
            lines.append(f"**{speaker}:** {_normalize_text(text)}")
            lines.append("")
        elif item_type == "action":
            lines.append(f"**Action:** {_normalize_text(text)}")
            lines.append("")


    if citations:
        lines.append("### Sources")
        for citation in citations:
            label = citation.get("label") or citation.get("doc_id")
            title = citation.get("title") or citation.get("doc_id")
            score = citation.get("score")
            lines.append(f"- {label}: {title} (score {score:.3f})" if score is not None else f"- {label}: {title}")

    return "\n".join(lines).strip() + "\n"


def build_tts_script(highlights: dict[str, Any]) -> dict[str, Any]:
    """Return a speech-friendly structure describing narrator/speaker turns."""

    script: dict[str, Any] = {
        "session_id": highlights.get("session_id"),
        "title": highlights.get("topic"),
        "project": highlights.get("project_id"),
        "summary": _normalize_text(highlights.get("summary", ""), preserve_newlines=False) if highlights.get("summary") else None,
        "segments": [],
        "sources": highlights.get("citations", []),
        "retrieval_history": highlights.get("retrieval_history", []),
        "voice_profiles": highlights.get("speaking_styles", {}),
        "session_style": highlights.get("session_style"),
        "default_voice": highlights.get("default_tts_voice"),
        "session_tags": highlights.get("session_speech_tags"),
    }

    session_tags = highlights.get("session_speech_tags")
    if isinstance(session_tags, str):
        session_tags = [session_tags]
    script["session_tags"] = session_tags

    # Narrator intro with summary
    if summary := highlights.get("summary"):
        intro_text = _normalize_text(summary, preserve_newlines=False)
        intro_metadata: dict[str, Any] = {}
        if script["session_style"]:
            intro_metadata["speaking_style"] = script["session_style"]
        if script["default_voice"]:
            intro_metadata["tts_voice"] = script["default_voice"]
        if session_tags:
            intro_metadata["speech_tags"] = session_tags
        script["segments"].append(
            {
                "speaker": "Narrator",
                "content": intro_text,
                "type": "intro",
                "metadata": intro_metadata,
            }
        )

    for item in highlights.get("items", []):
        if item.get("type") != "statement":
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        script["segments"].append(
            {
                "speaker": item.get("agent") or "Unknown",
                "content": _normalize_text(text, preserve_newlines=False),
                "type": "statement",
                "metadata": item.get("metadata") or {},
            }
        )

    consensus_item = next((item for item in highlights.get("items", []) if item.get("type") == "consensus"), None)
    if consensus_item and consensus_item.get("text"):
        if _normalize_text(consensus_item["text"], preserve_newlines=False) != _normalize_text(highlights.get("summary", ""), preserve_newlines=False):
            consensus_metadata = dict(consensus_item.get("metadata") or {})
            if script["session_style"] and "speaking_style" not in consensus_metadata:
                consensus_metadata["speaking_style"] = script["session_style"]
            if script["default_voice"] and "tts_voice" not in consensus_metadata:
                consensus_metadata["tts_voice"] = script["default_voice"]
            if session_tags and "speech_tags" not in consensus_metadata:
                consensus_metadata["speech_tags"] = session_tags
            script["segments"].append(
                {
                    "speaker": "Narrator",
                    "content": _normalize_text(consensus_item["text"], preserve_newlines=False),
                    "type": "consensus",
                    "metadata": consensus_metadata,
                }
            )

    for action_item in (item for item in highlights.get("items", []) if item.get("type") == "action"):
        text = (action_item.get("text") or "").strip()
        if not text:
            continue
        action_metadata = dict(action_item.get("metadata") or {})
        if script["session_style"] and "speaking_style" not in action_metadata:
            action_metadata["speaking_style"] = script["session_style"]
        if script["default_voice"] and "tts_voice" not in action_metadata:
            action_metadata["tts_voice"] = script["default_voice"]
        if session_tags and "speech_tags" not in action_metadata:
            action_metadata["speech_tags"] = session_tags
        script["segments"].append(
            {
                "speaker": "Narrator",
                "content": "Action item: " + _normalize_text(text, preserve_newlines=False),
                "type": "action",
                "metadata": action_metadata,
            }
        )

    return script


def save_session_highlights(workspace: "ProjectWorkspace", highlights: dict[str, Any]) -> Path:
    """Persist highlights into the session folder."""
    session_dir = workspace.root / "sessions" / highlights["session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)
    highlights_path = session_dir / "highlights.json"
    with highlights_path.open("w", encoding="utf-8") as handle:
        json.dump(highlights, handle, indent=2)
    return highlights_path


def _default_summary(result: OrchestrationResult, metadata: dict[str, Any], quotes: Sequence[dict[str, Any]]) -> str:
    if result.consensus:
        base = result.consensus.strip()
    elif quotes:
        base = (quotes[-1].get("content") or "").strip()
    else:
        return "No consensus captured."

    agent_count = len({quote.get("agent") for quote in quotes if quote.get("agent")})
    statement_count = len(quotes)
    extras: list[str] = []
    if agent_count:
        extras.append(f"{agent_count} agents contributed")
    if statement_count:
        extras.append(f"{statement_count} notable statements")
    if extras:
        base = f"{base} ({'; '.join(extras)})"
    return base


_ACTION_KEYWORDS = (
    "should",
    "must",
    "need to",
    "recommend",
    "plan",
    "propose",
    "increase",
    "decrease",
    "implement",
    "launch",
    "will",
)


def _derive_action_items(
    quotes: Sequence[dict[str, Any]],
    result: OrchestrationResult,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()

    def _should_capture(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in _ACTION_KEYWORDS)

    for quote in quotes:
        content = (quote.get("content") or "").strip()
        if not content or not _should_capture(content):
            continue
        signature = (quote.get("agent"), content)
        if signature in seen:
            continue
        items.append({"agent": quote.get("agent"), "text": content, "source": "statement"})
        seen.add(signature)

    consensus = (result.consensus or "").strip()
    if consensus and _should_capture(consensus) and (None, consensus) not in seen:
        items.append({"agent": None, "text": consensus, "source": "consensus"})
        seen.add((None, consensus))

    return items[:10]


def _doc_title(doc: dict[str, Any]) -> str | None:
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        return None
    document_meta = metadata.get("document")
    if isinstance(document_meta, dict):
        title = document_meta.get("title")
        if isinstance(title, str) and title:
            return title
    title = metadata.get("title")
    if isinstance(title, str) and title:
        return title
    return None


def _build_doc_lookup(retrieved_docs: Sequence[dict[str, Any]]) -> dict[tuple[str, str | None], dict[str, Any]]:
    lookup: dict[tuple[str, str | None], dict[str, Any]] = {}
    for doc in retrieved_docs:
        doc_id = str(doc.get("doc_id") or "")
        chunk_id = doc.get("chunk_id")
        chunk_key: str | None = str(chunk_id) if chunk_id else None
        if doc_id:
            lookup[(doc_id, chunk_key)] = doc
            lookup.setdefault((doc_id, None), doc)
    return lookup


def _normalize_citations(
    raw_citations: Sequence[dict[str, Any]] | None,
    doc_lookup: dict[tuple[str, str | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not raw_citations:
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for entry in raw_citations:
        if not isinstance(entry, dict):
            continue
        doc_id = str(entry.get("doc_id") or "")
        chunk_id_value = entry.get("chunk_id")
        chunk_id = str(chunk_id_value) if chunk_id_value else ""
        key = (doc_id or None, chunk_id or None, str(entry.get("label") or ""))
        if key in seen:
            continue
        seen.add(key)

        doc_info = doc_lookup.get((doc_id, chunk_id or None)) or doc_lookup.get((doc_id, None))
        doc = doc_info if isinstance(doc_info, dict) else {}
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        document_meta = metadata.get("document") if isinstance(metadata.get("document"), dict) else {}
        chunk_meta = metadata.get("chunk") if isinstance(metadata.get("chunk"), dict) else {}

        label = entry.get("label") or (doc and doc.get("label")) or document_meta.get("label") or (doc_id or None)
        title = entry.get("title") or _doc_title(doc or {}) or doc_id or None
        score = entry.get("score")
        if score is None and doc:
            score = doc.get("score")
        position = entry.get("position")
        if position is None:
            position = chunk_meta.get("position") if isinstance(chunk_meta, dict) else None
        tags = entry.get("tags")
        if tags is None:
            tags = document_meta.get("tags") if isinstance(document_meta, dict) else None
        if tags is not None and not isinstance(tags, list):
            if isinstance(tags, (set, tuple)):
                tags = list(tags)
            else:
                tags = [tags]
        if label is None:
            label = doc_id or chunk_id or None

        normalized.append(
            {
                "label": label,
                "doc_id": doc_id or None,
                "chunk_id": chunk_id or (doc.get("chunk_id") if doc else None),
                "title": title,
                "score": score,
                "position": position,
                "tags": tags,
                "source_path": metadata.get("source_path"),
                "ingested_at": metadata.get("ingested_at"),
            }
        )

    return normalized


def update_knowledge_graph(workspace: "ProjectWorkspace", highlights: dict[str, Any]) -> None:
    """Update the workspace knowledge graph with session highlights."""
    graph = KnowledgeGraph(workspace.root / "knowledge")
    project_slug = workspace.slug
    session_id = highlights["session_id"]
    session_node_id = f"session:{project_slug}:{session_id}"

    session_node = KnowledgeNode(
        node_id=session_node_id,
        type="session",
        attributes={
            "command": highlights.get("command"),
            "consensus": highlights.get("consensus"),
            "topic": highlights.get("topic"),
            "timestamp": highlights.get("timestamp"),
        },
    )

    nodes = [session_node]
    edges: list[KnowledgeEdge] = []

    topic = highlights.get("topic")
    if isinstance(topic, str) and topic:
        topic_node_id = f"topic:{project_slug}:{_slugify(topic)}"
        nodes.append(
            KnowledgeNode(
                node_id=topic_node_id,
                type="topic",
                attributes={"label": topic},
            )
        )
        edges.append(
            KnowledgeEdge(
                source=session_node_id,
                target=topic_node_id,
                type="DISCUSSED_TOPIC",
            )
        )

    for agent_name in highlights.get("agents", []):
        if not isinstance(agent_name, str):
            continue
        agent_node_id = f"agent:{project_slug}:{_slugify(agent_name)}"
        nodes.append(
            KnowledgeNode(
                node_id=agent_node_id,
                type="agent",
                attributes={"name": agent_name},
            )
        )
        edges.append(
            KnowledgeEdge(
                source=agent_node_id,
                target=session_node_id,
                type="PARTICIPATED_IN",
            )
        )

    quotes = highlights.get("quotes", [])
    for idx, quote in enumerate(quotes, start=1):
        agent_name = quote.get("agent")
        content = quote.get("content")
        if not agent_name or not content:
            continue
        quote_node_id = f"statement:{session_id}:{idx}"
        nodes.append(
            KnowledgeNode(
                node_id=quote_node_id,
                type="statement",
                attributes={
                    "agent": agent_name,
                    "content": content,
                    "confidence": quote.get("confidence"),
                },
            )
        )
        edges.append(
            KnowledgeEdge(
                source=quote_node_id,
                target=session_node_id,
                type="REFERENCES_SESSION",
            )
        )
        agent_node_id = f"agent:{project_slug}:{_slugify(agent_name)}"
        edges.append(
            KnowledgeEdge(
                source=agent_node_id,
                target=quote_node_id,
                type="STATED",
            )
        )

    graph.upsert_nodes(nodes)
    graph.upsert_edges(edges)


def index_highlights_in_warm_store(workspace: "ProjectWorkspace", highlights: dict[str, Any]) -> None:
    """Index highlight items in the warm cache store for retrieval."""
    items = highlights.get("items", [])
    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    store.replace_session_entries(highlights["session_id"], items)


def _extract_session_id(node_id: str) -> str:
    if node_id.startswith("session:"):
        parts = node_id.split(":", 2)
        if len(parts) >= 3:
            return parts[2]
    return node_id


def get_sessions_for_topic(graph: KnowledgeGraph, topic_label: str) -> list[str]:
    """Return session identifiers that discussed a given topic label."""
    label = topic_label.lower()
    nodes = graph.list_nodes()
    topic_ids = {
        node["id"]
        for node in nodes
        if node.get("type") == "topic"
        and str((node.get("attributes") or {}).get("label", "")).lower() == label
    }
    if not topic_ids:
        return []

    edges = graph.list_edges()
    session_node_ids = [
        edge.get("source")
        for edge in edges
        if edge.get("type") == "DISCUSSED_TOPIC" and edge.get("target") in topic_ids
    ]
    session_ids = {_extract_session_id(node_id) for node_id in session_node_ids if node_id}
    return sorted(session_ids)


def get_agent_activity(graph: KnowledgeGraph, agent_name: str) -> dict[str, list[dict[str, Any]] | list[str]]:
    """Return sessions and statements associated with a given agent."""
    name = agent_name.lower()
    nodes = graph.list_nodes()
    agent_nodes = {
        node["id"]: node
        for node in nodes
        if node.get("type") == "agent"
        and str((node.get("attributes") or {}).get("name", "")).lower() == name
    }
    if not agent_nodes:
        return {"sessions": [], "statements": []}

    edges = graph.list_edges()

    session_node_ids = {
        edge.get("target")
        for edge in edges
        if edge.get("type") == "PARTICIPATED_IN" and edge.get("source") in agent_nodes
    }
    session_ids = sorted({_extract_session_id(node_id) for node_id in session_node_ids if node_id})

    statement_node_ids = {
        edge.get("target")
        for edge in edges
        if edge.get("type") == "STATED" and edge.get("source") in agent_nodes
    }

    statement_nodes = {
        node["id"]: node
        for node in nodes
        if node.get("type") == "statement"
    }
    statement_sessions = {
        edge.get("source"): _extract_session_id(edge.get("target", ""))
        for edge in edges
        if edge.get("type") == "REFERENCES_SESSION"
    }

    statements: list[dict[str, Any]] = []
    for statement_id in statement_node_ids:
        node = statement_nodes.get(statement_id or "")
        if not node:
            continue
        content = (node.get("attributes") or {}).get("content")
        if not content:
            continue
        statements.append(
            {
                "session_id": statement_sessions.get(statement_id),
                "text": content,
            }
        )

    statements.sort(key=lambda item: (item.get("session_id") or "", item.get("text") or ""))
    return {"sessions": session_ids, "statements": statements}


def search_cold_transcripts(
    workspace: "ProjectWorkspace",
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search full transcripts for the query when warm cache misses."""

    sessions_dir = workspace.root / "sessions"
    if not sessions_dir.exists():
        return []

    query_lower = query.lower().strip()
    if not query_lower:
        return []

    results: list[dict[str, Any]] = []

    for session_path in sorted(sessions_dir.iterdir()):
        if not session_path.is_dir():
            continue

        session_id = session_path.name
        candidate_files = [
            session_path / "transcript.json",
            session_path / f"{session_id}.json",
        ]

        transcript_data: dict[str, Any] | None = None
        for file_path in candidate_files:
            if not file_path.exists():
                continue
            try:
                transcript_data = json.loads(file_path.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                continue

        if not transcript_data:
            continue

        for message in transcript_data.get("messages", []):
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if query_lower in content.lower():
                snippet = content
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                results.append(
                    {
                        "session_id": session_id,
                        "sender": message.get("sender") or "unknown",
                        "type": message.get("type") or "unknown",
                        "text": snippet,
                    }
                )
                if len(results) >= limit:
                    return results

    return results


def remove_session_from_knowledge(workspace: "ProjectWorkspace", session_id: str) -> None:
    """Remove knowledge graph data associated with a session."""
    graph = KnowledgeGraph(workspace.root / "knowledge")
    nodes = graph.list_nodes()
    edges = graph.list_edges()

    slug = workspace.slug
    session_node_id = f"session:{slug}:{session_id}"
    statement_prefix = f"statement:{session_id}:"

    filtered_nodes = [
        node
        for node in nodes
        if node.get("id") not in {session_node_id}
        and not str(node.get("id", "")).startswith(statement_prefix)
    ]

    def _references_session(edge: dict[str, Any]) -> bool:
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source == session_node_id or target == session_node_id:
            return True
        if str(source).startswith(statement_prefix) or str(target).startswith(statement_prefix):
            return True
        return False

    filtered_edges = [edge for edge in edges if not _references_session(edge)]

    graph._write_nodes(filtered_nodes)
    graph._write_edges(filtered_edges)


def remove_session_from_timeline(workspace: "ProjectWorkspace", session_id: str) -> None:
    timeline_path = workspace.root / "timeline.jsonl"
    if not timeline_path.exists():
        return

    lines = timeline_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if entry.get("session_id") == session_id:
            continue
        kept.append(json.dumps(entry))
    timeline_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def list_session_records(workspace: "ProjectWorkspace") -> list[dict[str, Any]]:
    """Return session metadata derived from highlights."""

    session_root = workspace.root / "sessions"
    if not session_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for highlights_path in session_root.glob("*/highlights.json"):
        session_id = highlights_path.parent.name
        try:
            data = json.loads(highlights_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        data = data.copy()
        data["session_id"] = session_id
        records.append(data)

    records.sort(key=lambda item: item.get("timestamp") or "")
    return records


# ---- Workspace document index ---------------------------------------------------------------

@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_size: int = 200
    overlap: int = 40
    chunk_count: int = 0
    ingested_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "doc_id": self.doc_id,
            "source_path": str(self.source_path),
            "metadata": self.metadata,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunk_count": self.chunk_count,
        }
        if self.ingested_at:
            payload["ingested_at"] = self.ingested_at
        return payload


@dataclass(slots=True)
class DocumentSearchResult:
    doc_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class WorkspaceDocumentIndex:
    """High-level helpers for managing workspace document embeddings."""

    def __init__(self, workspace: "ProjectWorkspace") -> None:
        self.workspace = workspace
        self.store = DocumentStore(self.workspace.root / "knowledge" / "docs")

    # ------------------------------------------------------------------ ingestion
    def ingest_files(
        self,
        files: Iterable[Path],
        *,
        title: str | None = None,
        tags: Sequence[str] | None = None,
        chunk_size: int = 200,
        overlap: int = 40,
        doc_id_factory: Callable[[Path, set[str]], str] | None = None,
    ) -> list[DocumentRecord]:
        tags_list = list(tags or [])
        existing_ids = {
            entry.get("doc_id", "")
            for entry in self.store.list_documents()
            if isinstance(entry, dict)
        }
        results: list[DocumentRecord] = []
        for raw_path in files:
            file_path = Path(raw_path)
            if not file_path.exists() or file_path.is_dir():
                continue
            doc_id = (
                doc_id_factory(file_path, existing_ids)
                if doc_id_factory
                else self._generate_doc_id(file_path, existing_ids)
            )
            existing_ids.add(doc_id)
            metadata = {
                "title": title or file_path.stem,
                "tags": list(tags_list),
            }
            chunk_count = self.store.ingest_file(
                file_path,
                doc_id=doc_id,
                metadata=metadata,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            record_data = self._fetch_document_data(doc_id)
            stored_metadata = dict(record_data.get("metadata") or metadata) if record_data else dict(metadata)
            stored_chunk_size = int(record_data.get("chunk_size") or chunk_size) if record_data else chunk_size
            stored_overlap = int(record_data.get("overlap") or overlap) if record_data else overlap
            stored_chunks = int(record_data.get("chunk_count") or chunk_count) if record_data else chunk_count
            record = DocumentRecord(
                doc_id=doc_id,
                source_path=file_path,
                metadata=stored_metadata,
                chunk_size=stored_chunk_size,
                overlap=stored_overlap,
                chunk_count=stored_chunks,
                ingested_at=record_data.get("ingested_at") if record_data else None,
            )
            results.append(record)
        return results

    # ------------------------------------------------------------------ inspection
    def list_documents(self) -> list[DocumentRecord]:
        documents = self.store.list_documents()
        chunk_counts: Counter[str] = Counter()
        for chunk in self.store.list_chunks():
            if not isinstance(chunk, dict):
                continue
            doc_id = chunk.get("doc_id")
            if doc_id:
                chunk_counts[str(doc_id)] += 1

        records: list[DocumentRecord] = []
        for entry in documents:
            if not isinstance(entry, dict):
                continue
            records.append(self._doc_from_data(entry, chunk_counts))
        return records

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        documents = {record.doc_id: record for record in self.list_documents()}
        hits: list[dict[str, Any]] = []

        for entry in self.store.query(query, limit=limit):
            if not isinstance(entry, dict):
                continue
            score = float(entry.get("score") or 0.0)
            if score < min_score:
                continue
            doc_id = str(entry.get("doc_id") or "")
            doc = documents.get(doc_id)
            metadata = {
                "chunk": entry.get("metadata") or {},
                "document": dict(doc.metadata) if doc else {},
                "source_path": str(doc.source_path) if doc else None,
                "title": doc.metadata.get("title") if doc else None,
                "tags": doc.metadata.get("tags", []) if doc else [],
                "ingested_at": doc.ingested_at if doc else None,
                "chunk_size": doc.chunk_size if doc else None,
                "overlap": doc.overlap if doc else None,
            }
            result = DocumentSearchResult(
                doc_id=doc_id,
                chunk_id=str(entry.get("chunk_id") or ""),
                text=str(entry.get("text") or ""),
                score=score,
                metadata=metadata,
            )
            hits.append(result.to_dict())
        return hits

    # ------------------------------------------------------------------ mutation
    def purge(self, doc_id: str) -> bool:
        target = doc_id.strip()
        if not target:
            return False

        documents = self.store.list_documents()
        filtered_docs = [doc for doc in documents if isinstance(doc, dict) and doc.get("doc_id") != target]
        if len(filtered_docs) == len(documents):
            return False

        self._write_jsonl(self.store.documents_path, filtered_docs)

        chunks = self.store.list_chunks()
        filtered_chunks = [chunk for chunk in chunks if isinstance(chunk, dict) and chunk.get("doc_id") != target]
        self._write_jsonl(self.store.chunks_path, filtered_chunks)

        feedback_path = self.store.base_path / "feedback.jsonl"
        if feedback_path.exists():
            kept_lines: list[str] = []
            for line in feedback_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    kept_lines.append(line)
                    continue
                if record.get("doc_id") == target:
                    continue
                kept_lines.append(json.dumps(record))
            feedback_path.write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else ""),
                encoding="utf-8",
            )

        return True

    def rebuild(self) -> dict[str, Any]:
        documents = self.store.list_documents()
        if not documents:
            self._write_jsonl(self.store.chunks_path, [])
            return {"documents": 0, "indexed": 0, "chunks": 0, "missing": []}

        chunk_records: list[dict[str, Any]] = []
        chunk_counts: Counter[str] = Counter()
        missing: list[str] = []

        for entry in documents:
            if not isinstance(entry, dict):
                continue
            doc_id = str(entry.get("doc_id") or "")
            source_path = Path(entry.get("source_path") or "")
            metadata = entry.get("metadata") or {}
            chunk_size = int(entry.get("chunk_size") or metadata.get("chunk_size") or 200)
            overlap = int(entry.get("overlap") or metadata.get("overlap") or 40)

            try:
                text = self.store._load_text(source_path)
            except (FileNotFoundError, ValueError):
                missing.append(doc_id)
                continue

            for index, chunk_text in enumerate(_chunk_text(text, chunk_size=chunk_size, overlap=overlap), start=1):
                tokens = _tokenize(chunk_text)
                vector = _vectorize(tokens)
                chunk_counts[doc_id] += 1
                chunk_records.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}::chunk-{index}",
                        "text": chunk_text,
                        "metadata": {"position": index},
                        "vector": vector,
                    }
                )

        self._write_jsonl(self.store.chunks_path, chunk_records)

        updated_docs: list[dict[str, Any]] = []
        for entry in documents:
            if not isinstance(entry, dict):
                continue
            payload = entry.copy()
            doc_id = str(payload.get("doc_id") or "")
            payload["chunk_count"] = int(chunk_counts.get(doc_id, payload.get("chunk_count") or 0))
            updated_docs.append(payload)
        self._write_jsonl(self.store.documents_path, updated_docs)

        return {
            "documents": len(documents),
            "indexed": len(documents) - len(missing),
            "chunks": int(sum(chunk_counts.values())),
            "missing": missing,
        }

    # ------------------------------------------------------------------ helpers
    def record_feedback(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        session_id: str,
        rating: str,
        notes: str | None = None,
    ) -> None:
        self.store.record_feedback(
            doc_id=doc_id,
            chunk_id=chunk_id,
            session_id=session_id,
            rating=rating,
            notes=notes,
        )

    def _fetch_document_data(self, doc_id: str) -> dict[str, Any] | None:
        for entry in self.store.list_documents():
            if isinstance(entry, dict) and entry.get("doc_id") == doc_id:
                return entry
        return None

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")

    @staticmethod
    def _generate_doc_id(file_path: Path, existing: set[str]) -> str:
        base = _slugify(file_path.stem) or "doc"
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        candidate = f"{base}-{timestamp}"
        counter = 1
        while candidate in existing:
            counter += 1
            candidate = f"{base}-{timestamp}-{counter}"
        return candidate

    def _doc_from_data(self, data: dict[str, Any], chunk_counts: Counter[str]) -> DocumentRecord:
        doc_id = str(data.get("doc_id") or "")
        metadata = dict(data.get("metadata") or {})
        source_path = Path(data.get("source_path") or "")
        chunk_size = int(data.get("chunk_size") or metadata.get("chunk_size") or 200)
        overlap = int(data.get("overlap") or metadata.get("overlap") or 40)
        chunk_count = int(chunk_counts.get(doc_id, data.get("chunk_count") or 0))
        return DocumentRecord(
            doc_id=doc_id,
            source_path=source_path,
            metadata=metadata,
            chunk_size=chunk_size,
            overlap=overlap,
            chunk_count=chunk_count,
            ingested_at=data.get("ingested_at"),
        )


def document_index_for_workspace(workspace: "ProjectWorkspace") -> WorkspaceDocumentIndex:
    """Return a document index helper for the given workspace."""
    return WorkspaceDocumentIndex(workspace)


def resolve_retrieval_config(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return retrieval configuration merged with defaults."""

    config = dict(DEFAULT_RETRIEVAL_CONFIG)
    if manifest:
        raw_config = manifest.get("retrieval")
        if isinstance(raw_config, Mapping):
            config.update({key: raw_config[key] for key in raw_config.keys()})

    config["enabled"] = bool(config.get("enabled", True))
    config["prefetch"] = bool(config.get("prefetch", True))

    limit_value = config.get("limit")
    try:
        limit_int = int(limit_value)
    except (TypeError, ValueError):
        limit_int = DEFAULT_RETRIEVAL_CONFIG["limit"]
    config["limit"] = max(1, limit_int)

    min_score_value = config.get("min_score")
    try:
        min_score_float = float(min_score_value)
    except (TypeError, ValueError):
        min_score_float = DEFAULT_RETRIEVAL_CONFIG["min_score"]
    config["min_score"] = max(0.0, min_score_float)

    return config


class SessionRetriever:
    """Manage per-session retrieval state, labels, and history."""

    def __init__(
        self,
        index: WorkspaceDocumentIndex,
        *,
        default_limit: int,
        default_min_score: float,
    ) -> None:
        self.index = index
        self.default_limit = max(1, default_limit)
        self.default_min_score = max(0.0, default_min_score)
        self._counter = 1
        self._seen_keys: set[tuple[str, str | None]] = set()
        self.history: list[dict[str, Any]] = []

    def register_documents(self, documents: Sequence[dict[str, Any]]) -> None:
        """Register existing documents so we avoid relabeling or duplicates."""
        for doc in documents:
            doc_id = str(doc.get("doc_id") or "")
            chunk_id = doc.get("chunk_id")
            chunk_key: str | None = str(chunk_id) if chunk_id else None
            if doc_id:
                self._seen_keys.add((doc_id, chunk_key))
                self._seen_keys.add((doc_id, None))

            label = doc.get("label")
            if not label:
                metadata = doc.get("metadata")
                if isinstance(metadata, dict):
                    document_meta = metadata.get("document")
                    if isinstance(document_meta, dict):
                        label = document_meta.get("label")
            label_num = self._extract_label_number(label)
            if label_num is not None:
                self._counter = max(self._counter, label_num + 1)

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a document search and return newly labeled documents."""
        effective_limit = max(1, limit or self.default_limit)
        effective_min_score = max(0.0, min_score if min_score is not None else self.default_min_score)

        raw_results = self.index.search(query, limit=effective_limit, min_score=effective_min_score)
        labeled_results: list[dict[str, Any]] = []

        for result in raw_results:
            doc_id = str(result.get("doc_id") or "")
            chunk_id_value = result.get("chunk_id")
            chunk_id: str | None = str(chunk_id_value) if chunk_id_value else None
            key = (doc_id, chunk_id)
            if key in self._seen_keys:
                continue

            doc = dict(result)
            doc_metadata = doc.setdefault("metadata", {})
            if not isinstance(doc_metadata, dict):
                doc_metadata = {}
                doc["metadata"] = doc_metadata
            document_meta = doc_metadata.setdefault("document", {})
            if not isinstance(document_meta, dict):
                document_meta = {}
                doc_metadata["document"] = document_meta

            label = doc.get("label") or document_meta.get("label")
            if not label:
                label = f"Doc {self._counter}"
                doc["label"] = label
                document_meta.setdefault("label", label)
                self._counter += 1
            else:
                label_num = self._extract_label_number(label)
                if label_num is not None:
                    self._counter = max(self._counter, label_num + 1)

            self._seen_keys.add(key)
            if doc_id:
                self._seen_keys.add((doc_id, None))

            labeled_results.append(doc)

        if labeled_results:
            self.history.append(
                {
                    "query": query,
                    "labels": [doc.get("label") for doc in labeled_results],
                    "count": len(labeled_results),
                }
            )

        return labeled_results

    @staticmethod
    def _extract_label_number(label: Any) -> int | None:
        if not isinstance(label, str):
            return None
        match = re.search(r"(\d+)$", label.strip())
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None
