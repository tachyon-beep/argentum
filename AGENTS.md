# Repository Guidelines

Argentum is a Python 3.11+ multi-agent dialogue system. Use this guide to keep contributions consistent and high-quality.

## Project Structure & Module Organization
- Core package lives in `argentum/` with agent definitions in `argentum/agents`, orchestration logic in `argentum/orchestration`, coordination utilities under `argentum/coordination`, and memory abstractions in `argentum/memory`.
- CLI entrypoint and shared wiring sit in `argentum/cli.py` and `argentum/models.py`; scenarios for demos live in `argentum/scenarios`.
- Tests mirror the package layout inside `tests/`, while runnable demos are in `examples/`; docs and proposals stay under `docs/`.

## Build, Test, and Development Commands
- Install in editable mode with tooling using `pip install -e ".[dev]"`.
- Run the CLI locally via `argentum --help` to inspect available orchestration commands.
- Execute the default test suite with `pytest`; include coverage via `pytest --cov=argentum --cov-report=term-missing`.
- Enforce linting and formatting with `ruff check argentum` and `black .`; ensure typing passes using `mypy argentum`.

## Coding Style & Naming Conventions
- Format Python code with Black (140-char lines) and keep imports ordered; `ruff` enforces PEP8 plus security and async rules.
- Prefer type-annotated functions and Pydantic models; disallow untyped defs per `mypy` config.
- Use snake_case for modules and functions, PascalCase for classes, and SCREAMING_SNAKE for constants; agent role strings should remain lowercase hyphenated (e.g., `policy-analyst`).

## Testing Guidelines
- Place new tests beside the related module using `test_<module>.py`; async behaviours should use `pytest.mark.asyncio`.
- Keep fixtures in `tests/conftest.py` when shared across modules; mindful of deterministic seeds for agent simulations.
- Aim to maintain coverage reported by `pytest --cov=argentum`; add targeted scenario tests when expanding orchestrators.

## Commit & Pull Request Guidelines
- Follow conventional prefixes from history (`fix:`, `refactor:`, `config:`) with concise, imperative descriptions.
- Each PR should summarize changes, note impacted agents/orchestrators, and reference linked issues.
- Include verification notes (e.g., `pytest --cov=argentum` outcome) and screenshots or transcripts for user-facing CLI updates.

## Configuration & Secrets
- Store provider credentials in `.env` (e.g., `OPENAI_API_KEY`, `ARGENTUM_DEFAULT_MODEL`) and avoid committing the file.
- Document any new settings in README or `docs/` and provide sensible defaults in `pydantic-settings` models.
