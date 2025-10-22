Argentum CLI Refactor (Modular Commands)

Overview

The monolithic `argentum/cli.py` has been refactored into a thin, stable entrypoint with modular subcommands. This removes the fragility we saw when editing a large file and enables safer, surgical updates without breaking unrelated commands.

What changed

- `argentum/cli.py` is now an aggregator that:
  - defines the Click root group `cli`
  - registers subcommands from `argentum/commands/*`
  - re‑exports `_prepare_session_environment` and `_persist_session_result` for test compatibility
- Shared helpers moved to `argentum/cli_utils.py` (console, styles, session setup/persist, rendering helpers).
- New command modules under `argentum/commands/`:
  - `debate.py` — government debate scenario
  - `advisory.py` — CTO advisory scenario
  - `project.py` — `project` group with `docs`, `agent`, `knowledge`, `compact`, `purge`, `warm-rebuild`, `timeline`
  - `auction.py` — auction-based group chat command

Entry point

The entrypoint remains the same; no user-facing changes are required:

- `pyproject.toml`: `argentum = "argentum.cli:main"`

Compatibility

- Tests and external code importing `from argentum.cli import cli, _prepare_session_environment, _persist_session_result` continue to work unchanged.

Notes

- The cold-transcript search output now includes both a grep-friendly line per hit and a rich table view to make simple string assertions robust across terminal widths.
- For future commands, prefer adding a new module in `argentum/commands/` and registering it in `argentum/cli.py`.
