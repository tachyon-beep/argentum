from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
import re
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from argentum.cli_utils import (
    HEADER_STYLE,
    console,
    _workspace_manager,
    _document_index,
    _default_agent_config,
    _render_warm_results,
    _render_cold_results,
    _render_topic_results,
    _render_agent_results,
)
from argentum.models import Role
from argentum.workspace import ProjectWorkspace, apply_speech_defaults, load_agent_profile, save_agent_profile
from argentum.workspace.knowledge import (
    KnowledgeGraph,
    build_session_highlights,
    document_index_for_workspace,
    get_agent_activity,
    get_sessions_for_topic,
    list_session_records,
    remove_session_from_knowledge,
    remove_session_from_timeline,
    resolve_retrieval_config,
    search_cold_transcripts,
)
from argentum.workspace.warm_store import WarmCacheStore


@click.group()
def project() -> None:
    """Manage Argentum project workspaces."""


@project.command("list")
def project_list() -> None:
    manager = _workspace_manager()
    workspaces = manager.list_projects()
    if not workspaces:
        console.print("[yellow]No projects found. Use 'argentum project init <slug>' to create one.[/yellow]")
        return
    table = Table(title="Projects", show_header=True)
    table.add_column("Slug", style="cyan")
    table.add_column("Display Name")
    table.add_column("Path", overflow="fold")
    table.add_column("Created")
    for workspace in workspaces:
        try:
            manifest = workspace.load_manifest()
        except FileNotFoundError:
            manifest = {"display_name": workspace.slug, "created_at": "unknown"}
        table.add_row(workspace.slug, manifest.get("display_name", workspace.slug), str(workspace.root), manifest.get("created_at", "unknown"))
    console.print(table)


@project.command("init")
@click.argument("slug")
@click.option("--title", help="Human-friendly project name.")
@click.option("--description", help="Short description of the project.")
@click.option("--template", help="Template name or path to pre-populate the workspace.")
@click.option("--force", is_flag=True, help="Overwrite existing project manifest if it already exists.")
def project_init(slug: str, title: str | None, description: str | None, template: str | None, force: bool) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.create_project(slug, title=title, description=description, template=template, force=force)
    except (FileExistsError, FileNotFoundError, OSError) as error:
        console.print("[red]Failed to create project: " + str(error) + "[/red]")
        sys.exit(1)
    console.print(Panel.fit("[green]Project '" + slug + "' initialised at[/green]\n" + str(workspace.root)))
    manifest = workspace.load_manifest()
    console.print(json.dumps(manifest, indent=2))


@project.command("info")
@click.argument("slug")
def project_info(slug: str) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    info = workspace.info()
    console.print(Panel.fit("[bold blue]Project:[/bold blue] " + str(info.get("display_name", slug))))
    console.print(json.dumps(info, indent=2))


@project.command("knowledge")
@click.argument("slug")
@click.option("--show", type=click.Choice(["summary", "nodes", "edges", "warm", "all"], case_sensitive=False), default="summary", show_default=True)
@click.option("--limit", "-l", default=10, show_default=True)
@click.option("--search", help="Warm cache search query (FTS syntax). Overrides --show warm if provided.")
@click.option("--topic", help="Show sessions related to a specific topic label.")
@click.option("--agent", help="Show sessions and statements for a specific agent name.")
def project_knowledge(slug: str, show: str, limit: int, search: str | None, topic: str | None, agent: str | None) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)

    graph = KnowledgeGraph(workspace.root / "knowledge")
    warm_store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")

    topic = topic.strip() if topic else None
    agent = agent.strip() if agent else None

    if topic and agent:
        console.print("[red]Please specify either --topic or --agent, not both.[/red]")
        sys.exit(1)

    if topic:
        session_ids = get_sessions_for_topic(graph, topic)
        _render_topic_results(workspace, topic, session_ids, limit)
        return

    if agent:
        activity = get_agent_activity(graph, agent)
        _render_agent_results(workspace, agent, activity, limit)
        return

    nodes = graph.list_nodes()
    edges = graph.list_edges()

    show = show.lower()
    if search:
        results = warm_store.search(search, limit=limit)
        if results:
            _render_warm_results(slug, results, search=search)
        else:
            cold = search_cold_transcripts(workspace, search, limit=limit)
            _render_cold_results(slug, cold, search)
        return

    if show in ("summary", "all"):
        panel = Panel.fit(f"[bold]Knowledge graph for {slug}[/bold]\nNodes: {len(nodes)}\nEdges: {len(edges)}", title="Summary")
        console.print(panel)

    if show in ("nodes", "all"):
        table = Table(title=f"Nodes (limit {limit})", show_header=True)
        table.add_column("ID", style="cyan", overflow="fold")
        table.add_column("Type")
        table.add_column("Attributes", overflow="fold")
        for node in nodes[:limit]:
            attrs = json.dumps(node.get("attributes", {}), ensure_ascii=False)
            table.add_row(node.get("id", "?"), node.get("type", "?"), attrs)
        if nodes:
            console.print(table)
        else:
            console.print("[dim]No nodes recorded yet.[/dim]")

    if show in ("edges", "all"):
        table = Table(title=f"Edges (limit {limit})", show_header=True)
        table.add_column("Source", style="cyan", overflow="fold")
        table.add_column("Type")
        table.add_column("Target", style="cyan", overflow="fold")
        table.add_column("Attributes", overflow="fold")
        for edge in edges[:limit]:
            attrs = json.dumps(edge.get("attributes", {}), ensure_ascii=False)
            table.add_row(edge.get("source", "?"), edge.get("type", "?"), edge.get("target", "?"), attrs)
        if edges:
            console.print(table)
        else:
            console.print("[dim]No edges recorded yet.[/dim]")

    if show in ("warm", "all"):
        results = warm_store.list_recent(limit=limit)
        _render_warm_results(slug, results, search=None)


@project.group("agent")
def project_agent() -> None:
    """Manage agent profiles for a project."""


@project_agent.command("show")
@click.argument("project")
@click.argument("agent_key")
def project_agent_show(project: str, agent_key: str) -> None:
    try:
        workspace = _workspace_manager().get_project(project)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    manifest = workspace.load_manifest()
    fallback = apply_speech_defaults(_default_agent_config(agent_key), manifest, agent_key)
    config = load_agent_profile(workspace, agent_key, fallback)
    console.print(json.dumps(config.model_dump(mode="json"), indent=2))


@project_agent.command("update")
@click.argument("project")
@click.argument("agent_key")
@click.option("--name", help="Agent display name.")
@click.option("--role", type=click.Choice([role.value for role in Role], case_sensitive=False))
@click.option("--persona", help="Agent persona description.")
@click.option("--model", help="Default model identifier.")
@click.option("--temperature", type=float, help="Sampling temperature.")
@click.option("--max-tokens", type=int, help="Maximum tokens per response.")
@click.option("--tool", "tools", multiple=True, help="Replace tool list (may be repeated).")
@click.option("--speaking-style", help="Preferred speaking style (e.g., boardroom, podcast).")
@click.option("--speech-tag", "speech_tags", multiple=True, help="Additional style tags (repeatable).")
@click.option("--tts-voice", help="Suggested TTS voice identifier.")
def project_agent_update(
    project: str,
    agent_key: str,
    name: str | None,
    role: str | None,
    persona: str | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    tools: tuple[str, ...],
    speaking_style: str | None,
    speech_tags: tuple[str, ...],
    tts_voice: str | None,
) -> None:
    try:
        workspace = _workspace_manager().get_project(project)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    manifest = workspace.load_manifest()
    fallback = apply_speech_defaults(_default_agent_config(agent_key), manifest, agent_key)
    config = load_agent_profile(workspace, agent_key, fallback)

    updates: dict[str, Any] = {}
    if name:
        updates["name"] = name
    if role:
        updates["role"] = Role(role)
    if persona is not None:
        updates["persona"] = persona
    if model is not None:
        updates["model"] = model
    if temperature is not None:
        updates["temperature"] = temperature
    if max_tokens is not None:
        updates["max_tokens"] = max_tokens
    if tools:
        updates["tools"] = list(tools)
    if speaking_style is not None:
        updates["speaking_style"] = speaking_style or None
    if speech_tags:
        updates["speech_tags"] = list(speech_tags)
    if tts_voice is not None:
        updates["tts_voice"] = tts_voice or None

    if updates:
        config = config.model_copy(update=updates)
    config.metadata = dict(config.metadata)
    config.metadata.setdefault("slug", agent_key)

    try:
        save_agent_profile(workspace, agent_key, config)
    except ValueError as error:
        console.print("[red]Failed to save profile: " + str(error) + "[/red]")
        sys.exit(1)

    console.print(Panel.fit(f"[green]Updated profile for[/green] {agent_key}\n[dim]{workspace.root / 'agents' / agent_key}[/dim]", title="Agent Profile Updated", border_style="green"))


@project.group("docs")
def project_docs() -> None:
    """Manage workspace documents for RAG."""


@project_docs.command("ingest")
@click.argument("slug")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@click.option("--title", help="Override document title.")
@click.option("--tag", "tags", multiple=True, help="Attach tags to the document (repeatable).")
@click.option("--chunk-size", type=int, default=200, show_default=True)
@click.option("--overlap", type=int, default=40, show_default=True)
def project_docs_ingest(slug: str, files: tuple[Path, ...], title: str | None, tags: tuple[str, ...], chunk_size: int, overlap: int) -> None:
    if not files:
        console.print("[red]No files specified for ingestion.[/red]")
        sys.exit(1)
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)

    ingested: list[Any] = []
    for file_path in files:
        if not file_path.exists():
            console.print(f"[yellow]Skipping missing file: {file_path}[/yellow]")
            continue
        if file_path.is_dir():
            console.print(f"[yellow]Skipping directory: {file_path}[/yellow]")
            continue
        try:
            records = index.ingest_files((file_path,), title=title, tags=tags, chunk_size=chunk_size, overlap=overlap, doc_id_factory=lambda path, existing: _generate_doc_id(path, existing))
        except ValueError as error:
            console.print(f"[yellow]Skipping {file_path}: {error}[/yellow]")
            continue
        ingested.extend(records)

    records = ingested
    if not records:
        console.print("[yellow]No documents ingested.[/yellow]")
        return

    total_chunks = sum(record.chunk_count for record in records)
    lines = [f"[green]•[/green] {record.doc_id} ([cyan]{record.chunk_count} chunk(s)[/cyan])" for record in records]
    panel = Panel("\n".join(lines), title=f"Documents Ingested ({len(records)} total, {total_chunks} chunk(s))", border_style="green")
    console.print(panel)


def _generate_doc_id(file_path: Path, existing: set[str] | None = None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", file_path.stem.lower()).strip("-") or "doc"
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    candidate = f"{stem}-{timestamp}"
    counter = 1
    while existing and candidate in existing:
        counter += 1
        candidate = f"{stem}-{timestamp}-{counter}"
    return candidate


@project_docs.command("list")
@click.argument("slug")
def project_docs_list(slug: str) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)
    documents = index.list_documents()
    if not documents:
        console.print("[yellow]No documents ingested yet.[/yellow]")
        return
    table = Table(title=f"Documents in {slug}", show_header=True)
    table.add_column("Doc ID", style="cyan", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("Tags")
    table.add_column("Chunks", justify="right")
    table.add_column("Ingested")
    table.add_column("Source", overflow="fold")
    for record in documents:
        table.add_row(record.doc_id, record.metadata.get("title", "-"), ", ".join(record.metadata.get("tags", [])) or "-", str(record.chunk_count), record.ingested_at or "-", str(record.source_path))
    console.print(table)


@project_docs.command("search")
@click.argument("slug")
@click.option("--query", required=True)
@click.option("--limit", type=int, default=5, show_default=True)
@click.option("--min-score", type=float, default=0.0, show_default=True)
def project_docs_search(slug: str, query: str, limit: int, min_score: float) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)
    results = index.search(query, limit=limit, min_score=min_score)
    if not results:
        console.print("[yellow]No matching document chunks found.[/yellow]")
        return
    table = Table(title=f"Document search for {slug}: '{query}'", show_header=True)
    table.add_column("Doc ID", style="cyan")
    table.add_column("Title", overflow="fold")
    table.add_column("Tags")
    table.add_column("Chunk")
    table.add_column("Score")
    table.add_column("Snippet", overflow="fold")
    for entry in results:
        metadata = entry.get("metadata") or {}
        document_meta = metadata.get("document") or {}
        table.add_row(entry.get("doc_id", "?"), document_meta.get("title", "-"), ", ".join(document_meta.get("tags", [])) or "-", entry.get("chunk_id", "?"), f"{entry.get('score', 0):.3f}", (entry.get("text") or "")[:280])
    console.print(table)


@project_docs.command("feedback")
@click.argument("slug")
@click.option("--doc-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--session", "session_id", required=True)
@click.option("--rating", type=click.Choice(["useful", "partial", "irrelevant"], case_sensitive=False), required=True)
@click.option("--notes")
def project_docs_feedback(slug: str, doc_id: str, chunk_id: str, session_id: str, rating: str, notes: str | None) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)
    index.record_feedback(doc_id=doc_id, chunk_id=chunk_id, session_id=session_id, rating=rating, notes=notes)
    console.print(Panel.fit("Feedback recorded.", border_style="cyan", title="Document Feedback"))


@project_docs.command("purge")
@click.argument("slug")
@click.argument("doc_id")
def project_docs_purge(slug: str, doc_id: str) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)
    if index.purge(doc_id):
        console.print(Panel.fit(f"Removed document [cyan]{doc_id}[/cyan].", border_style="red", title="Document Purged"))
    else:
        console.print(f"[yellow]Document '{doc_id}' not found.[/yellow]")


@project_docs.command("rebuild")
@click.argument("slug")
def project_docs_rebuild(slug: str) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)
    summary = index.rebuild()
    missing = summary.get("missing") or []
    lines = [f"[green]Documents:[/green] {summary.get('indexed', 0)}/{summary.get('documents', 0)} reindexed", f"[green]Chunks:[/green] {summary.get('chunks', 0)}"]
    if missing:
        lines.append(f"[yellow]Missing sources:[/yellow] {', '.join(missing)}")
    console.print(Panel("\n".join(lines), title="Document Index Rebuilt", border_style="cyan"))


@project.command("compact")
@click.argument("slug")
@click.option("--days", type=int, help="Override retention window in days for warm cache entries.")
@click.option("--dry-run", is_flag=True, help="Preview sessions that would be compacted.")
def project_compact(slug: str, days: int | None, dry_run: bool) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)

    manifest = workspace.load_manifest()
    retention_days = days or int(manifest.get("retention", {}).get("warm_window_days", 0))
    if retention_days <= 0:
        console.print("[yellow]No warm cache retention window configured.[/yellow]")
        return

    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    candidates: list[str] = []
    for session in list_session_records(workspace):
        ts = session.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        if dt < cutoff:
            candidates.append(session["session_id"]) 

    if dry_run:
        if not candidates:
            console.print("[dim]No sessions exceed retention window.[/dim]")
            return
        console.print(Panel.fit("\n".join(candidates), title="Sessions to Compact", border_style="yellow"))
        return

    warm_store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    removed = 0
    for session_id in candidates:
        warm_store.replace_session_entries(session_id, [])
        # mark metadata in highlights file
        highlight_path = workspace.root / "sessions" / session_id / "highlights.json"
        if highlight_path.exists():
            try:
                payload = json.loads(highlight_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            meta = payload.get("metadata") or {}
            meta["warm_cache_compacted_at"] = datetime.now(UTC).isoformat()
            payload["metadata"] = meta
            highlight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        removed += 1
    console.print(Panel.fit(f"Removed warm cache entries for {removed} session(s).", title="Warm Cache Compacted", border_style="green"))


@project.command("purge")
@click.argument("slug")
@click.option("--session", "session_id", required=True)
@click.option("--force", is_flag=True)
def project_purge(slug: str, session_id: str, force: bool) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)

    if not force:
        console.print("[red]Refusing to purge without --force.[/red]")
        sys.exit(2)

    # Remove session directory
    session_dir = workspace.root / "sessions" / session_id
    if session_dir.exists():
        for p in session_dir.glob("**/*"):
            try:
                p.unlink()
            except IsADirectoryError:
                pass
        for p in sorted(session_dir.glob("**/*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            session_dir.rmdir()
        except OSError:
            pass

    # Remove from stores
    warm_store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    warm_store.replace_session_entries(session_id, [])
    remove_session_from_knowledge(workspace, session_id)
    remove_session_from_timeline(workspace, session_id)

    console.print(Panel.fit(f"Purged session [cyan]{session_id}[/cyan] from workspace.", title="Session Purged", border_style="red"))


@project.command("warm-rebuild")
@click.argument("slug")
def project_warm_rebuild(slug: str) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    store.clear()

    # Rebuild from highlights
    from argentum.cli_utils import _document_index
    records = list_session_records(workspace)
    for record in records:
        session_id = record.get("session_id")
        if not session_id:
            continue
        items = record.get("items") or []
        store.replace_session_entries(session_id, items)

    console.print(Panel.fit("Warm cache rebuilt from session highlights.", title="Warm Cache Rebuilt", border_style="green"))


@project.command("timeline")
@click.argument("slug")
@click.option("--limit", type=int, default=20, show_default=True)
def project_timeline(slug: str, limit: int) -> None:
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print("[red]" + str(error) + "[/red]")
        sys.exit(1)
    timeline_path = workspace.root / "timeline.jsonl"
    if not timeline_path.exists():
        console.print("[dim]No timeline entries.[/dim]")
        return
    lines = [l for l in timeline_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    table = Table(title=f"Timeline for {slug}", show_header=True)
    table.add_column("Timestamp")
    table.add_column("Session", style="cyan")
    table.add_column("Command")
    table.add_column("Topic/Question")
    table.add_column("Duration (s)")
    for raw in lines[::-1]:
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = e.get("timestamp", "-")
        table.add_row(ts, e.get("session_id", "-"), e.get("command", "-"), e.get("topic") or e.get("question") or "-", str(e.get("duration_seconds", "-")))
    console.print(table)
