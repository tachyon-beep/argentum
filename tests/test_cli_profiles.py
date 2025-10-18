"""CLI tests for managing agent profiles."""

import json
from pathlib import Path

from click.testing import CliRunner

from argentum.cli import cli


def test_project_agent_profile_commands(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))

    runner = CliRunner()

    result = runner.invoke(cli, ["project", "init", "showtime", "--title", "Show Time"])
    assert result.exit_code == 0

    # Show command should create the default profile file
    result = runner.invoke(cli, ["project", "agent", "show", "showtime", "host"])
    assert result.exit_code == 0
    profile_path = tmp_path / "projects" / "showtime" / "agents" / "host" / "profile.json"
    assert profile_path.exists()

    # Update persona and temperature
    result = runner.invoke(
        cli,
        [
            "project",
            "agent",
            "update",
            "showtime",
            "host",
            "--persona",
            "Enthusiastic host",
            "--temperature",
            "0.5",
        ],
    )
    assert result.exit_code == 0

    contents = profile_path.read_text(encoding="utf-8")
    assert "Enthusiastic host" in contents
    assert "0.5" in contents

    result = runner.invoke(
        cli,
        [
            "project",
            "agent",
            "update",
            "showtime",
            "host",
            "--speaking-style",
            "podcast",
            "--speech-tag",
            "casual",
            "--speech-tag",
            "friendly",
            "--tts-voice",
            "voice_podcast_female",
        ],
    )
    assert result.exit_code == 0

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    assert data["speaking_style"] == "podcast"
    assert "casual" in data["speech_tags"]
    assert data["tts_voice"] == "voice_podcast_female"
