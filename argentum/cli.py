"""Command-line interface for Argentum."""

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from argentum.scenarios.advisory import CTOAdvisoryPanel
from argentum.scenarios.debate import GovernmentDebate

console = Console()

# Style constants
HEADER_STYLE = "bold blue"


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Argentum - Multi-Agent AI Dialogue System."""
    ...


@cli.command()
@click.argument("topic")
@click.option(
    "--ministers",
    "-m",
    multiple=True,
    help="Ministers to include (e.g., finance, environment)",
)
@click.option("--rounds", "-r", default=3, help="Number of debate rounds")
def debate(topic: str, ministers: tuple[str, ...], rounds: int) -> None:
    """Run a government minister debate simulation.

    TOPIC: The policy topic to debate
    """
    console.print(Panel.fit("🎭 Argentum Government Debate", style=HEADER_STYLE))

    minister_list = list(ministers) if ministers else None

    async def run_debate() -> None:
        debate_sim = GovernmentDebate(
            topic=topic,
            ministers=minister_list,
            rounds=rounds,
        )

        console.print(f"\n[bold]Topic:[/bold] {topic}")
        if minister_list:
            console.print(f"[bold]Ministers:[/bold] {', '.join(minister_list)}")
        console.print(f"[bold]Rounds:[/bold] {rounds}\n")

        with console.status("[bold green]Running debate..."):
            result = await debate_sim.run()

        # Display results
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

        # Show statistics
        stats = result.metadata.get("statistics", {})
        console.print(f"\n[dim]Duration: {result.duration_seconds:.2f}s[/dim]")
        console.print(f"[dim]Total turns: {stats.get('total_turns', 0)}[/dim]")

    try:
        asyncio.run(run_debate())
    except KeyboardInterrupt:
        console.print("\n[yellow]Debate interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
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
def advisory(question: str, advisors: tuple[str, ...], rounds: int) -> None:
    """Run a CTO advisory panel consultation.

    QUESTION: The technical question to discuss
    """
    console.print(Panel.fit("💼 Argentum CTO Advisory Panel", style=HEADER_STYLE))

    advisor_list = list(advisors) if advisors else None

    async def run_advisory() -> None:
        panel = CTOAdvisoryPanel(
            question=question,
            advisors=advisor_list,
            rounds=rounds,
        )

        console.print(f"\n[bold]Question:[/bold] {question}")
        if advisor_list:
            console.print(f"[bold]Advisors:[/bold] {', '.join(advisor_list)}")
        console.print(f"[bold]Rounds:[/bold] {rounds}\n")

        with console.status("[bold green]Running consultation..."):
            result = await panel.consult()

        # Display results
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

        # Show statistics
        stats = result.metadata.get("statistics", {})
        console.print(f"\n[dim]Duration: {result.duration_seconds:.2f}s[/dim]")
        console.print(f"[dim]Total turns: {stats.get('total_turns', 0)}[/dim]")

    try:
        asyncio.run(run_advisory())
    except KeyboardInterrupt:
        console.print("\n[yellow]Consultation interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
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


if __name__ == "__main__":
    main()
