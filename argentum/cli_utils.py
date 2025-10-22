"""Shared CLI utilities and session helpers for Argentum.

This module centralizes common helpers to keep the Click entrypoint thin and
modular. Tests import `_prepare_session_environment` and
`_persist_session_result` from `argentum.cli`, which re-exports these symbols
from here.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from argentum.agents.base import AgentConfig
from argentum.models import Role
from argentum.persistence import ConversationSession, JSONFileStore
from argentum.workspace import (
    ProjectWorkspace,
    WorkspaceManager,
    apply_speech_defaults,
    load_agent_profile,
    save_agent_profile,
)
from argentum.workspace.knowledge import (
    KnowledgeGraph,
    build_session_highlights,
    document_index_for_workspace,
    get_agent_activity,
    get_sessions_for_topic,
    index_highlights_in_warm_store,
    list_session_records,
    resolve_retrieval_config,
    SessionRetriever,
    remove_session_from_knowledge,
    remove_session_from_timeline,
    save_session_highlights,
    search_cold_transcripts,
    update_knowledge_graph,
)
from argentum.workspace.summarization import SummaryStrategy, get_summary_strategy
from argentum.workspace.warm_store import WarmCacheStore
from argentum.audio.factory import get_audio_controller
from argentum.audio.controller import AudioController


console = Console()
HEADER_STYLE = "bold blue"


def _workspace_manager() -> WorkspaceManager:
    return WorkspaceManager()


def _document_index(workspace: ProjectWorkspace | None):
    if workspace is None:
        return None
    return document_index_for_workspace(workspace)


def _default_agent_config(agent_key: str) -> AgentConfig:
    pretty_name = agent_key.replace("_", " ").replace("-", " ").strip() or agent_key
    pretty_name = pretty_name.title()
    return AgentConfig(
        name=pretty_name,
        role=Role.ADVISOR,
        persona=f"You are {pretty_name}. Provide thoughtful, domain-specific contributions.",
        model="gpt-4",
        temperature=0.7,
        max_tokens=600,
        metadata={"slug": agent_key},
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "session"


def _generate_session_id(prefix: str, seed: str | None = None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if seed:
        return f"{timestamp}-{_slugify(seed)[:40]}"
    return f"{timestamp}-{prefix}"


def _append_timeline(workspace: ProjectWorkspace, entry: dict[str, object]) -> None:
    timeline_path = workspace.root / "timeline.jsonl"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    with timeline_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")


def _resolve_summary_strategy(
    manifest: dict | None,
    mode_option: str | None,
    command_option: str | None,
):
    summary_config = {}
    if manifest:
        summary_section = manifest.get("summary")
        if isinstance(summary_section, dict):
            summary_config = summary_section

    mode = mode_option or summary_config.get("mode")
    command = command_option or summary_config.get("command")

    if isinstance(mode, str) and mode.lower() == "none":
        return None

    command_args = None
    if isinstance(command, str) and command.strip():
        command_args = shlex.split(command)
    elif isinstance(command, (list, tuple)):
        command_args = list(command)

    return get_summary_strategy(mode, command_args)


@dataclass(slots=True)
class SessionEnvironment:
    """Resolved configuration for executing and persisting a session."""

    session_id: str
    workspace: ProjectWorkspace | None
    manifest: dict[str, Any] | None
    session_dir: Path
    store: JSONFileStore
    memory_store: AgentMemoryStore | None
    summary_strategy: SummaryStrategy
    metadata: dict[str, Any]
    retriever: SessionRetriever | None
    audio_controller: AudioController


# Import here to avoid circular import at module top
from argentum.memory import AgentMemoryStore  # noqa: E402  (import-after-definition)


def _prepare_session_environment(
    *,
    command: str,
    project: str,
    session_id: str | None,
    seed: str | None,
    summary_mode: str | None,
    summary_command: str | None,
) -> SessionEnvironment:
    workspace = _workspace_manager().get_project(project)
    manifest = workspace.load_manifest()

    final_session_id = session_id or _generate_session_id(command, seed)
    session_dir = workspace.root / "sessions" / final_session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    store = JSONFileStore(base_path=session_dir)

    memory_root = workspace.root / "agents" / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    memory_store = AgentMemoryStore(base_path=memory_root)

    summary_strategy = _resolve_summary_strategy(manifest, summary_mode, summary_command)
    summary_strategy = summary_strategy or get_summary_strategy("heuristic")

    display_name = manifest.get("display_name") or workspace.slug.replace("-", " ").title()
    metadata: dict[str, Any] = {
        "command": command,
        "session_id": final_session_id,
        "project_id": workspace.slug,
        "project_display_name": display_name,
    }

    speech_cfg = manifest.get("speech", {}) if manifest else {}
    if isinstance(speech_cfg, dict):
        if speech_cfg.get("default_style"):
            metadata["session_style"] = speech_cfg.get("default_style")
        if speech_cfg.get("default_voice"):
            metadata["default_tts_voice"] = speech_cfg.get("default_voice")
        if speech_cfg.get("default_tags"):
            metadata["session_speech_tags"] = speech_cfg.get("default_tags")

    retrieval_config = resolve_retrieval_config(manifest)
    metadata["retrieval"] = {
        "enabled": retrieval_config["enabled"],
        "prefetch": retrieval_config["prefetch"],
        "limit": retrieval_config["limit"],
        "min_score": retrieval_config["min_score"],
        "query": None,
        "hit_count": 0,
    }

    retriever: SessionRetriever | None = None
    retrieved_docs: list[dict[str, Any]] = metadata.get("retrieved_docs") or []
    retrieval_query = (seed or "").strip()

    if workspace:
        doc_index = document_index_for_workspace(workspace)
        retriever = SessionRetriever(
            doc_index,
            default_limit=retrieval_config["limit"],
            default_min_score=retrieval_config["min_score"],
        )
        if retrieved_docs:
            retriever.register_documents(retrieved_docs)

        if retrieval_query:
            metadata["retrieval"]["query"] = retrieval_query

        if retrieval_query and retrieval_config["enabled"] and retrieval_config["prefetch"]:
            prefetched = retriever.search(
                retrieval_query,
                limit=retrieval_config["limit"],
                min_score=retrieval_config["min_score"],
            )
            if prefetched:
                retrieved_docs = prefetched

        metadata["retrieved_docs"] = retrieved_docs
        metadata["retrieval"]["hit_count"] = len(retrieved_docs)
        metadata["retrieval_history"] = retriever.history
    else:
        if retrieval_query:
            metadata["retrieval"]["query"] = retrieval_query
        metadata["retrieved_docs"] = retrieved_docs
        metadata["retrieval_history"] = []

    # Build audio controller from manifest (or defaults)
    audio_ctrl = get_audio_controller(manifest or {})

    return SessionEnvironment(
        session_id=final_session_id,
        workspace=workspace,
        manifest=manifest,
        session_dir=session_dir,
        store=store,
        memory_store=memory_store,
        summary_strategy=summary_strategy,
        metadata=metadata,
        retriever=retriever,
        audio_controller=audio_ctrl,
    )


async def _persist_session_result(
    env: SessionEnvironment,
    session: ConversationSession,
    metadata: dict[str, Any],
    result,
) -> Path:
    """Save conversation data, update workspace knowledge, and return transcript path."""
    await session.save(result)
    stored_path = env.store.base_path / f"{session.session_id}.json"
    transcript_path = stored_path

    if env.workspace:
        transcript_path = env.session_dir / "transcript.json"
        try:
            shutil.copy2(stored_path, transcript_path)
        except FileNotFoundError:
            transcript_path = stored_path

        summary_fn = lambda res, meta, quotes, strat=env.summary_strategy: strat.summarize(res, meta, quotes)
        highlights = build_session_highlights(
            session.session_id,
            metadata,
            result,
            summary_fn=summary_fn,
        )
        save_session_highlights(env.workspace, highlights)
        update_knowledge_graph(env.workspace, highlights)
        index_highlights_in_warm_store(env.workspace, highlights)

        timeline_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session.session_id,
            "command": metadata.get("command"),
            "project_id": metadata.get("project_id"),
            "topic": metadata.get("topic"),
            "question": metadata.get("question"),
            "agents": result.metadata.get("agent_names", []),
            "duration_seconds": result.duration_seconds,
            "consensus": result.consensus,
        }
        _append_timeline(env.workspace, timeline_entry)

    return transcript_path


# Rendering helpers used by project knowledge commands

def _render_warm_results(slug: str, results: list[dict[str, Any]], search: str | None) -> None:
    title = f"Warm cache search for {slug}: '{search}'" if search else f"Recent warm entries for {slug}"
    table = Table(title=title, show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Agent")
    table.add_column("Text", overflow="fold")
    for entry in results:
        table.add_row(entry.get("session_id", "-"), entry.get("agent", "-"), (entry.get("text") or "")[:280])
    console.print(table)



def _render_cold_results(slug: str, results: list[dict[str, Any]], search: str) -> None:
    # Print a plain title line to satisfy simple substring assertions.
    console.print(f"Cold transcript search for {slug}: '{search}'")
    table = Table(title=f"Cold transcript search for {slug}: '{search}'", show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Snippet", overflow="fold")
    for entry in results:
        session_id = entry.get("session_id", "-")
        snippet = (entry.get("text") or "")[:300]
        if snippet:
            console.print(f"Session {session_id}: {snippet}")
        table.add_row(session_id, snippet)
    console.print(table)


def _render_topic_results(workspace: ProjectWorkspace, topic: str, session_ids: list[str], limit: int) -> None:
    if not session_ids:
        console.print(Panel.fit(f"[dim]No sessions found for topic '{topic}'.[/dim]", title="Topic Results"))
        return

    table = Table(title=f"Sessions covering '{topic}'", show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Summary", overflow="fold")
    table.add_column("Timestamp")

    for session_id in session_ids[:limit]:
        highlight = _load_highlights(workspace, session_id)
        summary = highlight.get("summary") or highlight.get("consensus") or "-"
        timestamp = highlight.get("timestamp") or "-"
        table.add_row(session_id, summary[:300], str(timestamp))

    console.print(table)


def _render_agent_results(
    workspace: ProjectWorkspace,
    agent_name: str,
    activity: dict[str, list[dict[str, Any]] | list[str]],
    limit: int,
) -> None:
    sessions = activity.get("sessions", []) or []
    statements = activity.get("statements", []) or []

    if not sessions and not statements:
        console.print(Panel.fit(f"[dim]No activity found for agent '{agent_name}'.[/dim]", title="Agent Results"))
        return

    if sessions:
        table = Table(title=f"Sessions for {agent_name}", show_header=True)
        table.add_column("Session", style="cyan")
        table.add_column("Summary", overflow="fold")
        for session_id in sessions[:limit]:
            highlight = _load_highlights(workspace, session_id)
            summary = highlight.get("summary") or highlight.get("consensus") or "-"
            table.add_row(session_id, summary[:300])
        console.print(table)

    if statements:
        table = Table(title=f"Statements by {agent_name}", show_header=True)
        table.add_column("Session", style="cyan")
        table.add_column("Text", overflow="fold")
        for entry in statements[:limit]:
            session_id = entry.get("session_id") or "-"
            text = (entry.get("text") or "")[:300]
            table.add_row(session_id, text)
        console.print(table)


def _load_highlights(workspace: ProjectWorkspace, session_id: str) -> dict[str, Any]:
    path = workspace.root / "sessions" / session_id / "highlights.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


__all__ = [
    "console",
    "HEADER_STYLE",
    "SessionEnvironment",
    "_workspace_manager",
    "_document_index",
    "_default_agent_config",
    "_prepare_session_environment",
    "_persist_session_result",
    "_render_warm_results",
    "_render_cold_results",
    "_render_topic_results",
    "_render_agent_results",
]
