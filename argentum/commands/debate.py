from __future__ import annotations

import asyncio
import sys
from typing import Any

import click
from rich.panel import Panel

from argentum.cli_utils import (
    HEADER_STYLE,
    console,
    _default_agent_config,
    _prepare_session_environment,
    _persist_session_result,
)
from argentum.scenarios.debate import GovernmentDebate
from argentum.persistence import ConversationSession


@click.command()
@click.argument("topic")
@click.option("--ministers", "-m", multiple=True, help="Ministers to include (e.g., finance, environment)")
@click.option("--rounds", "-r", default=2, help="Number of debate rounds")
@click.option("--session-id", "-s", help="Existing session identifier to append transcript to.")
@click.option("--project", "-p", required=True, help="Project workspace slug.")
@click.option("--summary-mode", type=click.Choice(["heuristic", "frontier", "local", "none"], case_sensitive=False), help="Summarisation mode.")
@click.option("--summary-command", help="Local summarisation command.")
def debate(topic: str, ministers: tuple[str, ...], rounds: int, session_id: str | None, project: str, summary_mode: str | None, summary_command: str | None) -> None:
    """Run a government debate simulation.

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

    metadata: dict[str, Any] = dict(env.metadata)
    metadata.update({"topic": topic, "ministers": debate_sim.ministers, "rounds": rounds, "summary_mode": env.summary_strategy.name})

    session = ConversationSession(store=env.store, session_id=env.session_id, metadata=metadata)

    async def run_debate() -> None:
        console.print("\n[bold]Topic:[/bold] " + topic)
        if debate_sim.ministers:
            console.print("[bold]Ministers:[/bold] " + ", ".join(debate_sim.ministers))
        console.print(f"[bold]Rounds:[/bold] {rounds}\n")
        with console.status("[bold green]Running debate..."):
            result = await debate_sim.run()
        console.print("\n" + "=" * 80)
        console.print("[bold]DEBATE TRANSCRIPT[/bold]", justify="center")
        console.print("=" * 80 + "\n")
        for msg in result.messages:
            if msg.sender != "orchestrator":
                console.print("\n[bold cyan][" + msg.sender + "][/bold cyan]")
                console.print(msg.content)
        console.print("\n" + "=" * 80)
        console.print("[bold]CONSENSUS[/bold]", justify="center")
        console.print("=" * 80)
        console.print(result.consensus)
        metadata["ministers"] = debate_sim.ministers
        try:
            transcript_path = await _persist_session_result(env, session, metadata, result)
            console.print("[dim]Transcript saved to: " + str(transcript_path) + "[/dim]")
        except (OSError, ValueError) as error:
            console.print("[yellow]Warning: failed to persist transcript (" + str(error) + ").[/yellow]")

    try:
        asyncio.run(run_debate())
    except KeyboardInterrupt:
        console.print("\n[yellow]Debate interrupted by user[/yellow]")
        sys.exit(0)
    except (RuntimeError, ValueError, OSError) as e:
        console.print("\n[red]Error: " + str(e) + "[/red]")
        sys.exit(1)
