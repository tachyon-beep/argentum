"""Command-line interface for Argentum (refactored aggregator)."""

from __future__ import annotations

import click

from argentum.cli_utils import (
    console,
    HEADER_STYLE,
    SessionEnvironment,
    _prepare_session_environment,
    _persist_session_result,
)
from argentum.commands.debate import debate
from argentum.commands.advisory import advisory
from argentum.commands.project import project
from argentum.commands.auction import auction
from rich.table import Table
from rich.panel import Panel


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Argentum CLI root group."""


# Simple top-level helper command retained from original CLI
@cli.command()
def list_roles() -> None:
    """List available ministers and advisors."""
    console.print(Panel.fit("📋 Available Roles", style=HEADER_STYLE))

    ministers_table = Table(title="\nGovernment Ministers", show_header=True)
    ministers_table.add_column("Role", style="cyan")
    ministers_table.add_column("Title")
    for role, title in [
        ("finance", "Minister of Finance"),
        ("environment", "Minister of Environment"),
        ("defense", "Minister of Defense"),
        ("health", "Minister of Health"),
        ("education", "Minister of Education"),
        ("infrastructure", "Minister of Infrastructure"),
    ]:
        ministers_table.add_row(role, title)
    console.print(ministers_table)

    advisors_table = Table(title="\nTechnical Advisors", show_header=True)
    advisors_table.add_column("Role", style="cyan")
    advisors_table.add_column("Title")
    for role, title in [
        ("security", "Chief Security Officer"),
        ("finance", "Chief Financial Officer"),
        ("engineering", "VP of Engineering"),
        ("product", "Chief Product Officer"),
        ("operations", "VP of Operations"),
        ("data", "Chief Data Officer"),
    ]:
        advisors_table.add_row(role, title)
    console.print(advisors_table)


# Register subcommands
cli.add_command(debate)
cli.add_command(advisory)
cli.add_command(project)
cli.add_command(auction)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

# Re-export test-accessed helpers for compatibility
from argentum.cli_utils import _prepare_session_environment as _prepare_session_environment  # noqa: E402,F401
from argentum.cli_utils import _persist_session_result as _persist_session_result  # noqa: E402,F401
