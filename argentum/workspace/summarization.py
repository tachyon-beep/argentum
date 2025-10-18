"""Summarization helpers for session highlights."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Sequence

from argentum.models import OrchestrationResult


def _summary_with_warning(message: str, result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
    print(f"[summarization warning] {message}")
    return HeuristicSummaryStrategy().summarize(result, metadata, quotes)

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


class SummaryStrategy:
    """Base summarization strategy."""

    name: str = "heuristic"

    def summarize(self, result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class HeuristicSummaryStrategy(SummaryStrategy):
    """Simple summariser using rule-based heuristics."""

    name: str = "heuristic"

    def summarize(self, result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
        consensus = (result.consensus or "").strip()
        if consensus:
            return consensus
        if quotes:
            return (quotes[-1].get("content") or "").strip() or "No consensus captured."
        return "No consensus captured."


@dataclass(slots=True)
class FrontierSummaryStrategy(SummaryStrategy):
    """Summariser that calls an OpenAI-compatible endpoint."""

    model: str = os.getenv("ARGENTUM_FRONTIER_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("ARGENTUM_FRONTIER_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("ARGENTUM_FRONTIER_MAX_TOKENS", "200"))
    name: str = "frontier"

    def summarize(self, result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
        if openai is None:
            return _summary_with_warning("OpenAI SDK not installed; falling back to heuristic.", result, metadata, quotes)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _summary_with_warning("OPENAI_API_KEY not set; falling back to heuristic.", result, metadata, quotes)

        kwargs = {"api_key": api_key}
        api_base = os.getenv("OPENAI_API_BASE")
        if api_base:
            kwargs["base_url"] = api_base

        client = openai.OpenAI(**kwargs)  # type: ignore[attr-defined]
        prompt_payload = _build_prompt_payload(result, metadata, quotes)

        try:  # pragma: no cover - network call
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meeting summarizer. Produce a concise bullet summary of the discussion, focusing on decisions and action items.",
                    },
                    {"role": "user", "content": prompt_payload},
                ],
            )
            content = response.choices[0].message.content if response.choices else None
            if content:
                return content.strip()
        except Exception as exc:  # pragma: no cover - network failures
            return _summary_with_warning(f"Frontier summariser failed: {exc}", result, metadata, quotes)

        return HeuristicSummaryStrategy().summarize(result, metadata, quotes)


@dataclass(slots=True)
class LocalSummaryStrategy(SummaryStrategy):
    """Summariser that shells out to a local command (e.g., Ollama)."""

    command: Sequence[str]
    timeout: int = 45
    name: str = "local"

    def summarize(self, result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
        if not self.command:
            return _summary_with_warning("Local summary command not provided.", result, metadata, quotes)

        prompt_payload = _build_prompt_payload(result, metadata, quotes)
        try:
            proc = subprocess.run(
                self.command,
                input=prompt_payload.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=True,
            )
            stdout = proc.stdout.decode("utf-8", errors="ignore").strip()
            if stdout:
                return stdout
        except Exception as exc:
            return _summary_with_warning(f"Local summariser failed: {exc}", result, metadata, quotes)
        return _summary_with_warning("Local summariser returned empty output.", result, metadata, quotes)

def _summary_with_warning(message: str, result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
    print(f"[summarization warning] {message}")
    return HeuristicSummaryStrategy().summarize(result, metadata, quotes)


def _build_prompt_payload(result: OrchestrationResult, metadata: dict, quotes: Sequence[dict]) -> str:
    payload = {
        "topic": metadata.get("topic") or metadata.get("question"),
        "consensus": result.consensus,
        "quotes": quotes,
    }
    return json.dumps(payload, ensure_ascii=False)


def get_summary_strategy(mode: str | None, command: Sequence[str] | None = None) -> SummaryStrategy:
    mode_normalized = (mode or "heuristic").lower()
    if mode_normalized in {"frontier", "openai"}:
        return FrontierSummaryStrategy()
    if mode_normalized in {"local", "command"}:
        return LocalSummaryStrategy(command=tuple(command or (os.getenv("ARGENTUM_LOCAL_SUMMARY_CMD") or "").split()) or ())
    return HeuristicSummaryStrategy()
