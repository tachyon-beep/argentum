# Engineering Status Report (Self)

Owner: Engineering (Audio/Orchestration/Agents)
Last Updated: now

## Overview

Argentum is running with a modular CLI, an auction-based orchestrator capable of interjections and hard interrupts, and an audio controller with simulated ElevenLabs behavior (offline-safe). We have full test coverage for current features and have introduced structured metadata for observability. Next up is production-grade ElevenLabs streaming integration (flagged), and completing manager surfaces (EmotionEngine, InterjectionPlanner, Reputation).

## Current State (Summary)

- CLI: Refactored into modules; thin entrypoint; auction command added; tests green.
- Audio: Sim + ElevenLabs controllers; measured beats/drift; ducking lease; crossfade; p95 guard helper; offline adapter emits chunk/mark events.
- Orchestrator (WP3):
  - Micro-turn loop; concurrency scaffolding; context_version invalidation.
  - Interjections: planner agent/type/importance; threshold gating; sanitize; duck-play.
  - Hard-interrupt cutoff: readiness timeout; guard-band commit; crossfade; token spend charge; interrupt policy record.
  - Provisional interrupts (flag: --emit-provisional-interrupt).
  - Reserve prefetch for likely next speaker; reuse on match; cancellable.
  - Talk-share pacing feedback via resolver.update_pacing(); sliding-window talk counts.
  - Rich metadata: timing (drift_ms, commit_guard_ms, p95_first_chunk_ms); auction (bids, pacing, talk_share); hard_interrupt timing.phase=confirmed.
- Managers (WP2): Simple + Enhanced resolvers; Simple + Windowed interrupt policy; Token manager; initial SimpleEmotionEngine + BasicInterjectionPlanner already present; reputation TODO.
- Docs: Updated status and plan to reflect WP3 features; CLI refactor note added.

## Work Packages (Status)

- WP1 (Audio/TTS):
  - DONE (MVP): Controller API, sim controller, ElevenLabs controller scaffold, drift/p95, guard band helper, factory wiring.
  - Next: Production ElevenLabs streaming client + env/manifest wiring; real chunk/mark events; robust error handling; timeouts/backoff.

- WP2 (Managers):
  - DONE (partial): Resolver (simple + enhanced), TokenManager, InterruptPolicy (simple + windowed).
  - Next: EmotionEngine (llm/hybrid/heuristic) caps; InterjectionPlanner (cooldowns/importance); Reputation EMA updates.

- WP3 (Orchestrator):
  - DONE: Concurrency; interjection planning; hard-interrupt cutoff; provisional events; reserve prefetch (reuse); pacing feedback; observability.
  - Next: Optional polish only (no blockers).

## Next WP (Detailed): WP1 – ElevenLabs Streaming (Production)

Goals
- Implement a real ElevenLabs streaming adapter:
  - Connect via API (env: ELEVENLABS_API_KEY); voice selection; latency mode.
  - Yield `chunk` events with audio bytes and `mark` events at boundaries.
  - Ensure first event timings feed the latency profiler (p95) correctly.
- Wire manifest/CLI:
  - Manifest: conversation.tts.provider=elevenlabs; voice; latency_mode.
  - Keep sim as default; guarded flag for networked mode.
- Preserve offline CI:
  - Default to sim adapter; production adapter behind an opt-in flag/env; add mocks/skips for network tests.

Tasks
- Adapter (argentum/tts/elevenlabs_adapter.py):
  - Implement `stream(text)` with real network calls (guarded); retries/backoff; timeouts; safe buffering; yield marks.
  - Optional: `synthesize` for non-streaming calls (low-priority).
- Controller (argentum/audio/elevenlabs_controller.py):
  - Validate handling of production `chunk`/`mark` events; no changes to API.
  - Consider exporting a small stats struct if needed.
- Factory (argentum/audio/factory.py):
  - Read manifest; conditionally build production adapter; else sim.
- CLI (argentum/commands/auction.py):
  - Optional flags: `--tts-provider`, `--tts-voice`, `--tts-latency` (only if helpful; defaults from manifest).

Acceptance Criteria
- Streaming adapter yields first chunk and subsequent marks for typical inputs; controller records drift/p95; guard band computed; no crashes on timeouts.
- Offline tests pass without network; production path behind env/flag.
- Jitter tolerance: interjections land ≤ ~250ms around target in sim; production measured via marks.

Risks & Mitigations
- Provider variability: compute dynamic guard_band (p95 + buffer); log drift_ms; fallback to finish path on miss.
- Network errors: timeouts; exponential backoff; safe abort with fallback.
- Rate limits/costs: low-rate test harness; opt-in network tests; sim default.
- Mark fidelity: if marks absent or laggy, synthesize with sentence splits; downgrade to beat-only timing.

## Instructions to Self (Files to Touch/Read Next)

- Adapter: `argentum/tts/elevenlabs_adapter.py`
- Controller: `argentum/audio/elevenlabs_controller.py`
- Factory: `argentum/audio/factory.py`
- CLI (optional flags): `argentum/commands/auction.py`
- Orchestrator (for any minor metadata alignment): `argentum/orchestration/auction_chat.py`
- Docs: `docs/ORGANIC_AUCTION_IMPLEMENTATION_PLAN.md`, `docs/ORGANIC_AUCTION_STATUS.md`, `docs/CLI_REFACTOR.md`
- Tests: `tests/test_audio_elevenlabs.py` (extend), new `tests/test_elevenlabs_adapter_offline.py`

## Test Plan (Incremental)

- Unit tests (offline): adapter streams first chunk + marks; controller receives and sets beats; p95 updated; guard computation stable.
- Integration (sim jitter): ±150/500/1000 ms delays; verify interrupt fallback; reserve path unaffected.
- CLI smoke: auction run with manifest tts=elevenlabs (adapter in sim mode); flag toggles for provisional interrupts.
- Network tests (optional, skipped by default): mock HTTP where possible; keep CI offline-safe.

## Lessons Learned / Reminders

- Use monotonic clocks only; record planned vs actual beat offsets; log drift_ms.
- Guard bands derived from p95 with a buffer and a floor; skip interrupts on miss.
- Maintain single ducking lease; crossfade cancels duck; serialize audio transitions.
- Always snapshot context_version and cancel stale tasks; never commit on outdated version.
- Keep metadata ergonomic: deterministic IDs; timing; auction snapshots; do not bloat messages unless flagged.
- Keep CLI modular; prefer flags/manifest over hard-coded defaults; preserve offline-friendly behavior.

## Open Questions / TODOs

- ElevenLabs marks semantics: per-phoneme vs per-sentence; do we map adapter marks to beats or synth fallback?
- Voice profile tuning: words/sec, break_ms tuning per voice; expose overrides in manifest.
- Optional: add `reserve_used: true` on segments when reserve consumed (observability only).
- WP2: finalize EmotionEngine modes and InterjectionPlanner importance; add Reputation EMA rule and hook to events.

## Quick Checklist (Next 1–2 PRs)

- [ ] ElevenLabs adapter: streaming `chunk` + `mark` (offline-safe mocks), timeouts/retries.
- [ ] Factory wiring from manifest; sim default.
- [ ] Extend tests: adapter offline behavior; controller guard-band p95 influence.
- [ ] Docs updates (implementation plan + status + quickstart snippet).
