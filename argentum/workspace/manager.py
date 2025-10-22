"""Workspace manager for Argentum persistent projects."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

try:
    # Python 3.9+: importlib.resources.files
    from importlib.resources import files as pkg_files
except ImportError:  # pragma: no cover - fallback for older interpreters
    pkg_files = None  # type: ignore[assignment]


DEFAULT_PROJECT_DIRS = [
    "agents",
    "sessions",
    "knowledge",
    "cache/hot",
    "cache/warm",
    "cache/cold",
]

DEFAULT_RETRIEVAL_CONFIG: dict[str, Any] = {
    "enabled": True,
    "limit": 3,
    "min_score": 0.0,
    "prefetch": True,
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ProjectWorkspace:
    """Represents a single project workspace."""

    slug: str
    root: Path
    manifest_path: Path

    def ensure_structure(self) -> None:
        """Ensure standard directory layout is present."""
        for rel in DEFAULT_PROJECT_DIRS:
            (self.root / rel).mkdir(parents=True, exist_ok=True)
        (self.root / "timeline.jsonl").touch(exist_ok=True)

    def load_manifest(self) -> dict[str, Any]:
        """Load the project manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing at {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)

        retrieval_config = dict(DEFAULT_RETRIEVAL_CONFIG)
        retrieval_config.update(data.get("retrieval") or {})
        data["retrieval"] = retrieval_config
        return data

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def info(self) -> dict[str, Any]:
        """Return manifest merged with derived metadata."""
        manifest = self.load_manifest()
        manifest["_paths"] = {
            "root": str(self.root),
            "agents": str(self.root / "agents"),
            "sessions": str(self.root / "sessions"),
            "knowledge": str(self.root / "knowledge"),
            "cache": str(self.root / "cache"),
        }
        return manifest


class WorkspaceManager:
    """Manage creation and lookup of project workspaces."""

    def __init__(
        self,
        base_path: Path | None = None,
        *,
        user_templates_path: Path | None = None,
        system_templates_path: Path | None = None,
    ) -> None:
        self.base_path = (base_path or self._default_base_path()).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.user_templates_path = (user_templates_path or self._default_user_templates_path()).expanduser()
        self.user_templates_path.mkdir(parents=True, exist_ok=True)
        self.system_templates_path = system_templates_path or self._default_system_templates_path()

    @staticmethod
    def _default_base_path() -> Path:
        override = os.environ.get("ARGENTUM_WORKSPACES_DIR")
        if override:
            return Path(override)
        return Path(__file__).resolve().parents[2] / "workspace"

    @staticmethod
    def _default_user_templates_path() -> Path:
        override = os.environ.get("ARGENTUM_TEMPLATE_DIR")
        if override:
            return Path(override)
        return Path.home() / ".argentum" / "templates"

    @staticmethod
    def _default_system_templates_path() -> Path:
        if pkg_files:
            try:
                return Path(pkg_files("argentum.workspace").joinpath("templates"))
            except FileNotFoundError:
                pass
        # fall back to package-relative path
        return Path(__file__).resolve().parent / "templates"

    def list_projects(self) -> list[ProjectWorkspace]:
        """Return all known project workspaces."""
        workspaces: list[ProjectWorkspace] = []
        for entry in sorted(self.base_path.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "project.json"
            workspaces.append(ProjectWorkspace(slug=entry.name, root=entry, manifest_path=manifest_path))
        return workspaces

    def get_project(self, slug: str) -> ProjectWorkspace:
        """Return an existing workspace."""
        root = self.base_path / slug
        manifest_path = root / "project.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Project '{slug}' not found at {root}")
        workspace = ProjectWorkspace(slug=slug, root=root, manifest_path=manifest_path)
        workspace.ensure_structure()
        return workspace

    def create_project(
        self,
        slug: str,
        *,
        title: str | None = None,
        description: str | None = None,
        template: str | Path | None = None,
        force: bool = False,
    ) -> ProjectWorkspace:
        """Create a new workspace."""
        root = (self.base_path / slug).expanduser()
        manifest_path = root / "project.json"

        if manifest_path.exists() and not force:
            raise FileExistsError(f"Project '{slug}' already exists at {root}")

        root.mkdir(parents=True, exist_ok=True)

        template_path = self._resolve_template(template) if template else None
        if template_path and template_path.exists():
            shutil.copytree(template_path, root, dirs_exist_ok=True)

        workspace = ProjectWorkspace(slug=slug, root=root, manifest_path=manifest_path)
        workspace.ensure_structure()

        manifest = self._initial_manifest(slug, root, title=title, description=description)
        if manifest_path.exists():
            # Merge existing manifest (from template) with overrides
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            manifest = self._merge_manifest(existing, manifest, overrides={"display_name": title, "description": description})

        workspace.save_manifest(manifest)
        return workspace

    def _initial_manifest(
        self,
        slug: str,
        root: Path,
        *,
        title: str | None,
        description: str | None,
    ) -> dict[str, Any]:
        now = _timestamp()
        manifest: dict[str, Any] = {
            "project_id": slug,
            "display_name": title or slug,
            "description": description or "",
            "created_at": now,
            "updated_at": now,
            "default_agents": [],
            "conversation": {
                "auction": {
                    "emotion_control": "hybrid",
                    "interjection_min_importance": 0.5,
                    "tokens": {"max_bank": 8},
                    "interjections": {"cooldown_segments": 1},
                }
            },
            "summary": {
                "mode": "heuristic",
                "command": None,
            },
            "storage": {
                "base_path": str(root),
                "vector_store": None,
                "archive_format": "zip",
            },
            "retrieval": dict(DEFAULT_RETRIEVAL_CONFIG),
            "retention": {
                "hot_context_turns": 12,
                "warm_window_days": 120,
                "archive_after_days": 365,
            },
        }
        return manifest

    def _merge_manifest(
        self,
        template_manifest: dict[str, Any],
        defaults: dict[str, Any],
        *,
        overrides: dict[str, str | None],
    ) -> dict[str, Any]:
        merged = defaults | template_manifest  # template values win over defaults

        # Enforce required identifiers
        merged["project_id"] = defaults["project_id"]
        merged.setdefault("created_at", defaults["created_at"])
        merged["updated_at"] = _timestamp()
        storage = merged.get("storage", {})
        storage["base_path"] = defaults["storage"]["base_path"]
        merged["storage"] = storage

        # Merge retrieval settings with defaults to preserve new keys.
        retrieval_defaults = defaults.get("retrieval", {}) or {}
        template_retrieval = template_manifest.get("retrieval", {}) or {}
        merged["retrieval"] = {**retrieval_defaults, **template_retrieval}

        for key, value in overrides.items():
            if value:
                if key == "display_name":
                    merged["display_name"] = value
                elif key == "description":
                    merged["description"] = value
        return merged

    def _resolve_template(self, template: str | Path | None) -> Path | None:
        if template is None:
            return None

        if isinstance(template, Path):
            path = template.expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Template path '{path}' does not exist")
            return path

        candidate = Path(template).expanduser()
        if candidate.exists():
            return candidate

        # treat as template name
        for base in self._template_search_order():
            candidate_path = base / template
            if candidate_path.exists():
                return candidate_path

        raise FileNotFoundError(f"Template '{template}' not found in user or system template directories")

    def _template_search_order(self) -> Iterable[Path]:
        paths = [self.user_templates_path]
        if self.system_templates_path:
            paths.append(self.system_templates_path)
        return paths
