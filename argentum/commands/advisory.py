from __future__ import annotations

import asyncio
import sys
from typing import Any

import click
from rich.panel import Panel

from argentum.cli_utils import (
    HEADER_STYLE,
    console,
    _prepare_session_environment,
    _persist_session_result,
)
from argentum.scenarios.advisory import CTOAdvisoryPanel
from argentum.persistence import ConversationSession


@click.command()
@click.argument("question")
@click.option("--advisors", "-a", multiple=True, help="Advisors to include (e.g., security, finance)")
@click.option("--rounds", "-r", default=2, help="Number of consultation rounds")
@click.option("--session-id", "-s", help="Existing session identifier to append transcript to.")
@click.option("--project", "-p", required=True, help="Project workspace slug.")
@click.option("--summary-mode", type=click.Choice(["heuristic", "frontier", "local", "none"], case_sensitive=False), help="Summarisation mode.")
@click.option("--summary-command", help="Local summarisation command.")
def advisory(
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

    metadata: dict[str, Any] = dict(env.metadata)
    metadata.update({"question": question, "advisors": panel.advisors, "rounds": rounds, "summary_mode": env.summary_strategy.name})

    session = ConversationSession(store=env.store, session_id=env.session_id, metadata=metadata)

    async def run_advisory() -> None:
        console.print("\n[bold]Question:[/bold] " + question)
        if panel.advisors:
            console.print("[bold]Advisors:[/bold] " + ", ".join(panel.advisors))
        console.print(f"[bold]Rounds:[/bold] {rounds}\n")
        with console.status("[bold green]Running consultation..."):
            result = await panel.consult()
        console.print("\n" + "=" * 80)
        console.print("[bold]CONSULTATION TRANSCRIPT[/bold]", justify="center")
        console.print("=" * 80 + "\n")
        for msg in result.messages:
            if msg.sender != "orchestrator":
                console.print("\n[bold cyan][" + msg.sender + "][/bold cyan]")
                console.print(msg.content)
        console.print("\n" + "=" * 80)
        console.print("[bold]SUMMARY[/bold]", justify="center")
        console.print("=" * 80)
        console.print(result.consensus)
        metadata["advisors"] = panel.advisors
        try:
            transcript_path = await _persist_session_result(env, session, metadata, result)
            console.print("[dim]Transcript saved to: " + str(transcript_path) + "[/dim]")
        except (OSError, ValueError) as error:
            console.print("[yellow]Warning: failed to persist transcript (" + str(error) + ").[/yellow]")

    try:
        asyncio.run(run_advisory())
    except KeyboardInterrupt:
        console.print("\n[yellow]Consultation interrupted by user[/yellow]")
        sys.exit(0)
    except (RuntimeError, ValueError, OSError) as e:
        console.print("\n[red]Error: " + str(e) + "[/red]")
        sys.exit(1)
