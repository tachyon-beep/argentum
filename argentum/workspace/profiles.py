"""Agent profile helpers for workspace-managed projects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from argentum.agents.base import AgentConfig
from argentum.workspace.manager import ProjectWorkspace

_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def _agent_directory(workspace: ProjectWorkspace, agent_key: str) -> Path:
    safe_key = _SAFE_PATTERN.sub("-", agent_key.strip().lower()).strip("-") or "agent"
    return workspace.root / "agents" / safe_key


def _profile_path(workspace: ProjectWorkspace, agent_key: str) -> Path:
    return _agent_directory(workspace, agent_key) / "profile.json"


def load_agent_profile(
    workspace: ProjectWorkspace | None,
    agent_key: str,
    fallback: AgentConfig,
) -> AgentConfig:
    """Load an AgentConfig from the workspace, or persist the fallback if none exists."""
    if workspace is None:
        return fallback

    path = _profile_path(workspace, agent_key)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fallback.model_dump(mode="json"), indent=2), encoding="utf-8")
        return fallback

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(raw, dict):
        return fallback

    # Merge fallback defaults with stored overrides, giving priority to the stored values.
    merged: dict[str, Any] = fallback.model_dump(mode="json")
    merged.update(raw)

    try:
        config = AgentConfig.model_validate(merged)
    except ValidationError:
        return fallback
    config.metadata.setdefault("slug", agent_key)
    return config


def save_agent_profile(workspace: ProjectWorkspace, agent_key: str, config: AgentConfig) -> None:
    """Persist an AgentConfig to the workspace."""
    path = _profile_path(workspace, agent_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(mode="json"), handle, indent=2)


def apply_speech_defaults(config: AgentConfig, manifest: dict[str, Any] | None, agent_key: str) -> AgentConfig:
    """Merge speech/tone defaults from the manifest into an agent config."""

    if not manifest:
        return config

    speech_cfg = manifest.get("speech") or {}
    if not speech_cfg:
        return config

    overrides = speech_cfg.get("overrides") or {}
    override = overrides.get(agent_key) if isinstance(overrides, dict) else None
    if not override and isinstance(config.metadata, dict):
        slug = config.metadata.get("slug")
        if slug and isinstance(overrides, dict):
            override = overrides.get(slug)

    def pick(key: str, default_key: str) -> Any:
        if isinstance(override, dict) and override.get(key) is not None:
            return override.get(key)
        return speech_cfg.get(default_key)

    style = pick("style", "default_style")
    tags = pick("tags", "default_tags")
    if isinstance(tags, str):
        tags = [tags]
    voice = pick("tts_voice", "default_voice")

    update: dict[str, Any] = {}
    if style and not config.speaking_style:
        update["speaking_style"] = style
    if tags and not config.speech_tags:
        update["speech_tags"] = tags
    if voice and not config.tts_voice:
        update["tts_voice"] = voice

    if not update:
        return config
    return config.model_copy(update=update)
