"""Command-line interface for Argentum."""

import asyncio
import json
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from argentum.agents.base import AgentConfig
from argentum.memory import AgentMemoryStore
from argentum.models import Role
from argentum.persistence import ConversationSession, JSONFileStore
from argentum.scenarios.advisory import CTOAdvisoryPanel
from argentum.scenarios.debate import GovernmentDebate
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

console = Console()

# Style constants
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


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def _iterate_highlights(workspace: ProjectWorkspace) -> list[tuple[str, dict[str, Any], Path]]:
    session_root = workspace.root / "sessions"
    if not session_root.exists():
        return []

    records: list[tuple[str, dict[str, Any], Path]] = []
    for highlight_path in session_root.glob("*/highlights.json"):
        session_id = highlight_path.parent.name
        try:
            data = json.loads(highlight_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append((session_id, data, highlight_path))
    return records


def _reindex_warm_cache(workspace: ProjectWorkspace, store: WarmCacheStore) -> tuple[int, int]:
    records = list_session_records(workspace)
    store.clear()

    session_count = 0
    item_count = 0
    for record in records:
        session_id = record.get("session_id")
        if not session_id:
            continue
        items = record.get("items") or []
        store.replace_session_entries(session_id, items)
        session_count += 1
        item_count += len(items)
    return session_count, item_count


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


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Argentum - Multi-Agent AI Dialogue System."""


@cli.command()
@click.argument("topic")
@click.option(
    "--ministers",
    "-m",
    multiple=True,
    help="Ministers to include (e.g., finance, environment)",
)
@click.option("--rounds", "-r", default=3, help="Number of debate rounds")
@click.option(
    "--session-id",
    "-s",
    help="Existing session identifier to append transcript to.",
)
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project workspace slug to store transcripts, metadata, and defaults.",
)
@click.option(
    "--summary-mode",
    type=click.Choice(["heuristic", "frontier", "local", "none"], case_sensitive=False),
    help="Summarisation mode for session highlights.",
)
@click.option(
    "--summary-command",
    help="Command to run for local summarisation (e.g., 'ollama run llama3.1').",
)
@click.pass_context
def debate(
    ctx: click.Context,
    topic: str,
    ministers: tuple[str, ...],
    rounds: int,
    session_id: str | None,
    project: str,
    summary_mode: str | None,
    summary_command: str | None,
) -> None:
    """Run a government minister debate simulation.

    TOPIC: The policy topic to debate
    """
    console.print(Panel.fit("🎭 Argentum Government Debate", style=HEADER_STYLE))

    minister_list = list(ministers) if ministers else None

    env = _prepare_session_environment(
        command="debate",
        project=project,
        session_id=session_id,
        seed=topic,
        summary_mode=summary_mode,
        summary_command=summary_command,
    )

    session_id = env.session_id
    workspace = env.workspace
    manifest = env.manifest

    if workspace and not minister_list:
        defaults: list[str] | None = None
        if manifest:
            defaults = manifest.get("default_ministers") or manifest.get("default_agents")
        if isinstance(defaults, list):
            minister_list = [str(item) for item in defaults]

    debate_sim = GovernmentDebate(
        topic=topic,
        ministers=minister_list,
        rounds=rounds,
        memory_store=env.memory_store,
        workspace=env.workspace,
        context_documents=env.metadata.get("retrieved_docs"),
        retriever=env.retriever,
    )

    metadata = dict(env.metadata)
    metadata.update(
        {
            "topic": topic,
            "ministers": debate_sim.ministers,
            "rounds": rounds,
            "summary_mode": env.summary_strategy.name,
        }
    )

    session = ConversationSession(
        store=env.store,
        session_id=session_id,
        metadata=metadata,
    )

    async def run_debate() -> None:
        console.print(f"\n[bold]Topic:[/bold] {topic}")
        if debate_sim.ministers:
            console.print(f"[bold]Ministers:[/bold] {', '.join(debate_sim.ministers)}")
        console.print(f"[bold]Rounds:[/bold] {rounds}\n")

        with console.status("[bold green]Running debate..."):
            result = await debate_sim.run()

        console.print("\n" + "=" * 80)
        console.print("[bold]DEBATE TRANSCRIPT[/bold]", justify="center")
        console.print("=" * 80 + "\n")

        for msg in result.messages:
            if msg.sender != "orchestrator":
                console.print(f"\n[bold cyan][{msg.sender}][/bold cyan]")
                console.print(msg.content)

        console.print("\n" + "=" * 80)
        console.print("[bold]CONSENSUS[/bold]", justify="center")
        console.print("=" * 80)
        console.print(result.consensus)

        stats = result.metadata.get("statistics", {})
        console.print(f"\n[dim]Duration: {result.duration_seconds:.2f}s[/dim]")
        console.print(f"[dim]Total turns: {stats.get('total_turns', 0)}[/dim]")

        metadata["ministers"] = debate_sim.ministers

        try:
            transcript_path = await _persist_session_result(env, session, metadata, result)
            console.print(f"[dim]Transcript saved to: {transcript_path}[/dim]")
        except (OSError, ValueError) as error:
            console.print(f"[yellow]Warning: failed to persist transcript ({error}).[/yellow]")

    try:
        asyncio.run(run_debate())
    except KeyboardInterrupt:
        console.print("\n[yellow]Debate interrupted by user[/yellow]")
        sys.exit(0)
    except (RuntimeError, ValueError, OSError) as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option(
    "--advisors",
    "-a",
    multiple=True,
    help="Advisors to include (e.g., security, finance)",
)
@click.option("--rounds", "-r", default=2, help="Number of consultation rounds")
@click.option(
    "--session-id",
    "-s",
    help="Existing session identifier to append transcript to.",
)
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project workspace slug to store transcripts, metadata, and defaults.",
)
@click.option(
    "--summary-mode",
    type=click.Choice(["heuristic", "frontier", "local", "none"], case_sensitive=False),
    help="Summarisation mode for session highlights.",
)
@click.option(
    "--summary-command",
    help="Command to run for local summarisation (e.g., 'ollama run llama3.1').",
)
@click.pass_context
def advisory(
    ctx: click.Context,
    question: str,
    advisors: tuple[str, ...],
    rounds: int,
    session_id: str | None,
    project: str,
    summary_mode: str | None,
    summary_command: str | None,
) -> None:
    """Run a CTO advisory panel consultation.

    QUESTION: The technical question to discuss
    """
    console.print(Panel.fit("💼 Argentum CTO Advisory Panel", style=HEADER_STYLE))

    advisor_list = list(advisors) if advisors else None

    env = _prepare_session_environment(
        command="advisory",
        project=project,
        session_id=session_id,
        seed=question,
        summary_mode=summary_mode,
        summary_command=summary_command,
    )

    session_id = env.session_id
    workspace = env.workspace
    manifest = env.manifest

    if workspace and not advisor_list:
        defaults: list[str] | None = None
        if manifest:
            defaults = manifest.get("default_advisors") or manifest.get("default_agents")
        if isinstance(defaults, list):
            advisor_list = [str(item) for item in defaults]

    panel = CTOAdvisoryPanel(
        question=question,
        advisors=advisor_list,
        rounds=rounds,
        memory_store=env.memory_store,
        workspace=env.workspace,
        context_documents=env.metadata.get("retrieved_docs"),
        retriever=env.retriever,
    )

    metadata = dict(env.metadata)
    metadata.update(
        {
            "question": question,
            "advisors": panel.advisors,
            "rounds": rounds,
            "summary_mode": env.summary_strategy.name,
        }
    )

    session = ConversationSession(
        store=env.store,
        session_id=session_id,
        metadata=metadata,
    )

    async def run_advisory() -> None:
        console.print(f"\n[bold]Question:[/bold] {question}")
        if panel.advisors:
            console.print(f"[bold]Advisors:[/bold] {', '.join(panel.advisors)}")
        console.print(f"[bold]Rounds:[/bold] {rounds}\n")

        with console.status("[bold green]Running consultation..."):
            result = await panel.consult()

        console.print("\n" + "=" * 80)
        console.print("[bold]CONSULTATION TRANSCRIPT[/bold]", justify="center")
        console.print("=" * 80 + "\n")

        for msg in result.messages:
            if msg.sender != "orchestrator":
                console.print(f"\n[bold cyan][{msg.sender}][/bold cyan]")
                console.print(msg.content)

        console.print("\n" + "=" * 80)
        console.print("[bold]SUMMARY[/bold]", justify="center")
        console.print("=" * 80)
        console.print(result.consensus)

        stats = result.metadata.get("statistics", {})
        console.print(f"\n[dim]Duration: {result.duration_seconds:.2f}s[/dim]")
        console.print(f"[dim]Total turns: {stats.get('total_turns', 0)}[/dim]")
        metadata["advisors"] = panel.advisors

        try:
            transcript_path = await _persist_session_result(env, session, metadata, result)
            console.print(f"[dim]Transcript saved to: {transcript_path}[/dim]")
        except (OSError, ValueError) as error:
            console.print(f"[yellow]Warning: failed to persist transcript ({error}).[/yellow]")

    try:
        asyncio.run(run_advisory())
    except KeyboardInterrupt:
        console.print("\n[yellow]Consultation interrupted by user[/yellow]")
        sys.exit(0)
    except (RuntimeError, ValueError, OSError) as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def list_roles() -> None:
    """List available ministers and advisors."""
    console.print(Panel.fit("📋 Available Roles", style=HEADER_STYLE))

    # Ministers
    ministers_table = Table(title="\nGovernment Ministers", show_header=True)
    ministers_table.add_column("Role", style="cyan")
    ministers_table.add_column("Title")

    ministers = [
        ("finance", "Minister of Finance"),
        ("environment", "Minister of Environment"),
        ("defense", "Minister of Defense"),
        ("health", "Minister of Health"),
        ("education", "Minister of Education"),
        ("infrastructure", "Minister of Infrastructure"),
    ]

    for role, title in ministers:
        ministers_table.add_row(role, title)

    console.print(ministers_table)

    # Advisors
    advisors_table = Table(title="\nTechnical Advisors", show_header=True)
    advisors_table.add_column("Role", style="cyan")
    advisors_table.add_column("Title")

    advisors = [
        ("security", "Chief Security Officer"),
        ("finance", "Chief Financial Officer"),
        ("engineering", "VP of Engineering"),
        ("product", "Chief Product Officer"),
        ("operations", "VP of Operations"),
        ("data", "Chief Data Officer"),
    ]

    for role, title in advisors:
        advisors_table.add_row(role, title)

    console.print(advisors_table)


def main() -> None:
    """Main entry point."""
    cli()


@cli.group()
def project() -> None:
    """Manage Argentum project workspaces."""


@project.command("list")
def project_list() -> None:
    """List known project workspaces."""
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
        table.add_row(
            workspace.slug,
            manifest.get("display_name", workspace.slug),
            str(workspace.root),
            manifest.get("created_at", "unknown"),
        )

    console.print(table)


@project.command("init")
@click.argument("slug")
@click.option("--title", help="Human-friendly project name.")
@click.option("--description", help="Short description of the project.")
@click.option(
    "--template",
    help="Template name or path to pre-populate the workspace (overridable by user files).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing project manifest if it already exists.",
)
def project_init(slug: str, title: str | None, description: str | None, template: str | None, force: bool) -> None:
    """Create a new project workspace."""
    manager = _workspace_manager()
    try:
        workspace = manager.create_project(
            slug,
            title=title,
            description=description,
            template=template,
            force=force,
        )
    except (FileExistsError, FileNotFoundError, OSError) as error:
        console.print(f"[red]Failed to create project: {error}[/red]")
        sys.exit(1)

    console.print(Panel.fit(f"[green]Project '{slug}' initialised at[/green]\n{workspace.root}"))
    manifest = workspace.load_manifest()
    console.print(json.dumps(manifest, indent=2))


@project.command("info")
@click.argument("slug")
def project_info(slug: str) -> None:
    """Show project manifest and key paths."""
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    info = workspace.info()

    console.print(Panel.fit(f"[bold blue]Project:[/bold blue] {info.get('display_name', slug)}"))
    console.print(json.dumps(info, indent=2))


@project.command("knowledge")
@click.argument("slug")
@click.option(
    "--show",
    type=click.Choice(["summary", "nodes", "edges", "warm", "all"], case_sensitive=False),
    default="summary",
    show_default=True,
    help="Specify which information to display.",
)
@click.option(
    "--limit",
    "-l",
    default=10,
    show_default=True,
    help="Limit the number of records shown when listing results.",
)
@click.option(
    "--search",
    help="Warm cache search query (FTS syntax). Overrides --show warm if provided.",
)
@click.option(
    "--topic",
    help="Show sessions related to a specific topic label.",
)
@click.option(
    "--agent",
    help="Show sessions and statements for a specific agent name.",
)
def project_knowledge(
    slug: str,
    show: str,
    limit: int,
    search: str | None,
    topic: str | None,
    agent: str | None,
) -> None:
    """Inspect the knowledge graph for a project workspace."""
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
        panel = Panel.fit(
            f"[bold]Knowledge graph for {slug}[/bold]\nNodes: {len(nodes)}\nEdges: {len(edges)}",
            title="Summary",
        )
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
    """Display the stored profile for an agent."""
    try:
        workspace = _workspace_manager().get_project(project)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
    """Update profile attributes for an agent."""
    try:
        workspace = _workspace_manager().get_project(project)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
        console.print(f"[red]Failed to save profile: {error}[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[green]Updated profile for[/green] {agent_key}\n[dim]{workspace.root / 'agents' / agent_key}[/dim]",
            title="Agent Profile Updated",
        border_style="green",
    )
)


@project.group("docs")
def project_docs() -> None:
    """Manage workspace documents for RAG."""


def _generate_doc_id(file_path: Path, existing: set[str] | None = None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", file_path.stem.lower()).strip("-") or "doc"
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    candidate = f"{stem}-{timestamp}"
    counter = 1
    while existing and candidate in existing:
        counter += 1
        candidate = f"{stem}-{timestamp}-{counter}"
    return candidate


@project_docs.command("ingest")
@click.argument("slug")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@click.option("--title", help="Override document title.")
@click.option("--tag", "tags", multiple=True, help="Attach tags to the document (repeatable).")
@click.option("--chunk-size", type=int, default=200, show_default=True, help="Number of words per chunk.")
@click.option("--overlap", type=int, default=40, show_default=True, help="Word overlap between chunks.")
def project_docs_ingest(slug: str, files: tuple[Path, ...], title: str | None, tags: tuple[str, ...], chunk_size: int, overlap: int) -> None:
    if not files:
        console.print("[red]No files specified for ingestion.[/red]")
        sys.exit(1)

    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
            records = index.ingest_files(
                (file_path,),
                title=title,
                tags=tags,
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id_factory=lambda path, existing: _generate_doc_id(path, existing),
            )
        except ValueError as error:
            console.print(f"[yellow]Skipping {file_path}: {error}[/yellow]")
            continue
        ingested.extend(records)

    records = ingested

    if not records:
        console.print("[yellow]No documents ingested.[/yellow]")
        return

    total_chunks = sum(record.chunk_count for record in records)
    lines = [
        f"[green]•[/green] {record.doc_id} ([cyan]{record.chunk_count} chunk(s)[/cyan])"
        for record in records
    ]
    panel = Panel(
        "\n".join(lines),
        title=f"Documents Ingested ({len(records)} total, {total_chunks} chunk(s))",
        border_style="green",
    )
    console.print(panel)


@project_docs.command("list")
@click.argument("slug")
def project_docs_list(slug: str) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
        table.add_row(
            record.doc_id,
            record.metadata.get("title", "-"),
            ", ".join(record.metadata.get("tags", [])) or "-",
            str(record.chunk_count),
            record.ingested_at or "-",
            str(record.source_path),
        )

    console.print(table)


@project_docs.command("search")
@click.argument("slug")
@click.option("--query", required=True, help="Search query text.")
@click.option("--limit", type=int, default=5, show_default=True, help="Number of chunks to return.")
@click.option("--min-score", type=float, default=0.0, show_default=True, help="Minimum similarity score.")
def project_docs_search(slug: str, query: str, limit: int, min_score: float) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
        table.add_row(
            entry.get("doc_id", "?"),
            document_meta.get("title", "-"),
            ", ".join(document_meta.get("tags", [])) or "-",
            entry.get("chunk_id", "?"),
            f"{entry.get('score', 0):.3f}",
            (entry.get("text") or "")[:280],
        )

    console.print(table)


@project_docs.command("feedback")
@click.argument("slug")
@click.option("--doc-id", required=True, help="Document identifier.")
@click.option("--chunk-id", required=True, help="Chunk identifier.")
@click.option("--session", "session_id", required=True, help="Session referencing the chunk.")
@click.option("--rating", type=click.Choice(["useful", "partial", "irrelevant"], case_sensitive=False), required=True)
@click.option("--notes", help="Optional feedback notes.")
def project_docs_feedback(slug: str, doc_id: str, chunk_id: str, session_id: str, rating: str, notes: str | None) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)

    index.record_feedback(
        doc_id=doc_id,
        chunk_id=chunk_id,
        session_id=session_id,
        rating=rating,
        notes=notes,
    )

    console.print(Panel.fit("Feedback recorded.", border_style="cyan", title="Document Feedback"))


@project_docs.command("purge")
@click.argument("slug")
@click.argument("doc_id")
def project_docs_purge(slug: str, doc_id: str) -> None:
    try:
        workspace = _workspace_manager().get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
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
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    index = _document_index(workspace)
    if index is None:
        console.print("[red]Unable to access document index.[/red]")
        sys.exit(1)

    summary = index.rebuild()
    missing = summary.get("missing") or []
    lines = [
        f"[green]Documents:[/green] {summary.get('indexed', 0)}/{summary.get('documents', 0)} reindexed",
        f"[green]Chunks:[/green] {summary.get('chunks', 0)}",
    ]
    if missing:
        lines.append(f"[yellow]Missing sources:[/yellow] {', '.join(missing)}")

    console.print(
        Panel(
            "\n".join(lines),
            title="Document Index Rebuilt",
            border_style="cyan",
        )
    )
@project.command("compact")
@click.argument("slug")
@click.option("--days", type=int, help="Override retention window in days for warm cache entries.")
@click.option("--dry-run", is_flag=True, help="Preview sessions that would be compacted.")
def project_compact(slug: str, days: int | None, dry_run: bool) -> None:
    """Remove warm-cache entries that exceed the retention window."""
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    manifest = workspace.load_manifest()
    retention_days = days or int(manifest.get("retention", {}).get("warm_window_days", 0))
    if retention_days <= 0:
        console.print("[yellow]No warm cache retention window configured.[/yellow]")
        return

    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    candidates: list[str] = []
    highlights_to_update: list[tuple[str, dict[str, Any], Path]] = []

    for session_id, highlights, path in _iterate_highlights(workspace):
        timestamp = _parse_iso_timestamp(highlights.get("timestamp"))
        if timestamp and timestamp < cutoff:
            candidates.append(session_id)
            highlights_to_update.append((session_id, highlights, path))

    if not candidates:
        console.print("[green]Warm cache already within retention window.[/green]")
        return

    if dry_run:
        table = Table(title="Sessions eligible for compaction", show_header=True)
        table.add_column("Session", style="cyan")
        table.add_column("Timestamp")
        for session_id, highlights, _ in highlights_to_update:
            table.add_row(session_id, highlights.get("timestamp", "unknown"))
        console.print(table)
        return

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    before_count = store.count_entries()
    store.delete_sessions(candidates)
    after_count = store.count_entries()

    compacted_at = datetime.now(UTC).isoformat()
    for _, highlights, path in highlights_to_update:
        metadata = highlights.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["warm_cache_compacted_at"] = compacted_at
        highlights["metadata"] = metadata
        path.write_text(json.dumps(highlights, indent=2), encoding="utf-8")

    removed_count = before_count - after_count
    panel = Panel.fit(
        f"Removed {removed_count} indexed entries across {len(candidates)} session(s).\n"
        f"Warm cache size: {after_count} entries.",
        title="Warm Cache Compaction",
        border_style="blue",
    )
    console.print(panel)


@project.command("purge")
@click.argument("slug")
@click.option("--session", "session_id", required=True, help="Session identifier to remove.")
@click.option("--force", is_flag=True, help="Confirm deletion of session artifacts and indexes.")
def project_purge(slug: str, session_id: str, force: bool) -> None:
    """Permanently remove a session's artifacts and indexes."""
    if not force:
        console.print("[red]Refusing to purge without --force.[/red]")
        sys.exit(1)

    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    session_dir = workspace.root / "sessions" / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    warm_before = store.count_entries()
    store.delete_sessions([session_id])
    warm_after = store.count_entries()

    graph = KnowledgeGraph(workspace.root / "knowledge")
    nodes_before = len(graph.list_nodes())
    edges_before = len(graph.list_edges())
    remove_session_from_knowledge(workspace, session_id)
    nodes_after = len(graph.list_nodes())
    edges_after = len(graph.list_edges())

    remove_session_from_timeline(workspace, session_id)

    timeline_path = workspace.root / "timeline.jsonl"
    timeline_entries = 0
    if timeline_path.exists():
        timeline_entries = sum(1 for line in timeline_path.read_text(encoding="utf-8").splitlines() if line.strip())

    console.print(
        Panel.fit(
            f"Purged session '{session_id}' from workspace {slug}.\n"
            f"Warm cache entries: {warm_after} (removed {warm_before - warm_after}).\n"
            f"Knowledge graph nodes: {nodes_after} ({nodes_before - nodes_after} removed).\n"
            f"Knowledge graph edges: {edges_after} ({edges_before - edges_after} removed).\n"
            f"Timeline entries remaining: {timeline_entries}.",
            title="Session Purged",
            border_style="red",
        )
    )


@project.command("warm-rebuild")
@click.argument("slug")
def project_warm_rebuild(slug: str) -> None:
    """Rebuild the warm-cache index from stored highlights."""
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    session_count, item_count = _reindex_warm_cache(workspace, store)

    console.print(
        Panel.fit(
            f"Reindexed {item_count} highlight item(s) across {session_count} session(s).",
            title="Warm Cache Rebuilt",
            border_style="green",
        )
    )


@project.command("timeline")
@click.argument("slug")
@click.option("--limit", "limit", type=int, default=20, show_default=True, help="Number of entries to display.")
def project_timeline(slug: str, limit: int) -> None:
    """Display the project timeline entries."""
    manager = _workspace_manager()
    try:
        workspace = manager.get_project(slug)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    timeline_path = workspace.root / "timeline.jsonl"
    if not timeline_path.exists():
        console.print("[yellow]No timeline entries recorded yet.[/yellow]")
        return

    records: list[dict[str, Any]] = []
    for line in timeline_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            records.append(entry)

    if not records:
        console.print("[yellow]No timeline entries recorded yet.[/yellow]")
        return

    records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    table = Table(title=f"Timeline for {slug}", show_header=True)
    table.add_column("Timestamp", style="cyan")
    table.add_column("Session")
    table.add_column("Command")
    table.add_column("Topic/Question", overflow="fold")
    table.add_column("Duration (s)")

    for entry in records[:limit]:
        table.add_row(
            entry.get("timestamp", "unknown"),
            entry.get("session_id", "-"),
            entry.get("command", "-"),
            entry.get("topic") or entry.get("question") or "-",
            f"{entry.get('duration_seconds', '-')}",
        )

    console.print(table)


def _render_warm_results(slug: str, results: list[dict[str, Any]], *, search: str | None) -> None:
    if search:
        title = f"Warm cache search for {slug}: \"{search}\""
    else:
        title = f"Warm cache highlights for {slug}"

    if not results:
        console.print(Panel.fit("[dim]No warm cache entries found.[/dim]", title=title))
        return

    table = Table(title=title, show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Type")
    table.add_column("Agent")
    table.add_column("Text", overflow="fold")
    table.add_column("Metadata", overflow="fold")

    for entry in results:
        metadata = entry.get("metadata") or {}
        table.add_row(
            entry.get("session_id", "?"),
            entry.get("type", "?"),
            entry.get("agent") or "-",
            entry.get("text", "")[:500],
            json.dumps(metadata, ensure_ascii=False),
        )

    console.print(table)


def _render_cold_results(slug: str, results: list[dict[str, Any]], search: str) -> None:
    title = f"Cold transcript search for {slug}: \"{search}\""
    if not results:
        console.print(Panel.fit("[dim]No transcript entries matched.[/dim]", title=title))
        return

    table = Table(title=title, show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Sender")
    table.add_column("Type")
    table.add_column("Excerpt", overflow="fold")

    for entry in results:
        table.add_row(
            entry.get("session_id", "?"),
            entry.get("sender", "-"),
            entry.get("type", "-"),
            entry.get("text", ""),
        )

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


if __name__ == "__main__":
    main()
