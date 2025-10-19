# Organic Auction Conversation – Engineering Status Report

Owner: Engineering (Audio/Orchestration/Agents)
Last Updated: now

## Current State (TL;DR)

- WP1 (Audio/TTS) — Strong MVP: audio controller API, Sim + ElevenLabs controllers scaffolded, measured beats, drift/latency profiling, duck/crossfade controls, controller factory wired into CLI. Offline tests green.
- WP2 (Managers) — Partially implemented: simple + enhanced auction resolvers (pacing+softmax+evidence), simple + windowed interrupt policies, token manager. Emotion engine + interjection planner pending.
- WP3 (Orchestrator) — Started: micro‑turn loop in place with token accrual, bidding, segment playback via audio controller, and simple interjection. Concurrency + hard‑interrupt cutoff not yet wired.
- Docs — Design/Implementation plans up to date (chat plan, implementation plan, presets, governor, concurrency, lead time, interrupt modes).

## Work Packages — Status & Next

### WP1: Audio/TTS
Completed
- AudioController interface; SimAudioController with estimated beats
- ElevenLabsAudioController scaffold: mark‑driven or estimated beats; duck lease; crossfade cancels duck; planned vs actual beat timestamps and drift_ms reporting
- LatencyProfiler: collects first‑chunk latencies; p95 exposed; commit_guard helper (p95 + buffer)
- Audio factory: Sim by default; ElevenLabs via manifest/CLI (offline placeholder adapter)
- Tests: marks, finish, duck no‑overlap, crossfade cancellation, drift bounds, p95 guard band, factory selection

Remaining
- Real ElevenLabs streaming adapter (networked) + env/manifest/CLI wiring
- Export drift/latency/p95 in a small struct for orchestrator guard‑band tuning
- Additional race tests (cancel at beat boundary); mixer polish ordering (crossfade vs duck)

### WP2: Managers
Completed
- TokenManager (accrual/max bank)
- InterruptPolicy (simple + windowed sliding‑window budget)
- AuctionResolver (simple + enhanced: pacing EMA + softmax bid mapping + evidence tie‑break)

Remaining
- EmotionEngine (llm/hybrid/heuristic; EMA smoothing; deadband; cap influence)
- InterjectionPlanner (typed interjections + cooldowns/quotas; importance score)
- Reputation updates (EMA rule applied from events)

### WP3: Orchestrator
Completed
- AuctionGroupChatOrchestrator skeleton
- Micro‑turn loop MVP: token accrual → collect+resolve bids → speaker response → TTS playback → await beat → simple interjection → finish → append messages with metadata

Next
- Concurrency: spawn LLM tasks for interjections/bids/next‑segment prefetch with context_version tagging, timeouts, cancellation, and global/per‑agent concurrency limits
- Hard‑interrupt cutoff: guard‑band commit; prebuffer top‑1 kicker; fallback path on miss; record events
- Metadata: deterministic IDs (auction_id, segment_id, clip_id), bid details, pacing multipliers, drift

### WP12: Quality & Budget Governor
Planned (not started)
- Runtime scaling of concurrency, lead time, prefetch depth, interjections, interrupt mode, TTS expressivity; AIMD logic; budget and health caps

## Risks & Mitigations

- Provider timing drift → Use measured beats, drift_ms logging, and dynamic commit_guard_ms (p95 + buffer). AIMD lead_time controller when miss_rate is high.
- Audio race conditions (double‑duck/overlaps) → Single ducking lease + crossfade cancels duck; add last‑millisecond cancel tests.
- LLM latency spikes → Concurrency timeouts + fallbacks; pre‑resolve next speaker; reserve segment; disable interrupt on guard‑band miss.
- Economic oscillations (dominance/hoarding) → Pacing EMA; single accrual mode; decay caps; interrupt budgets and dynamic fees; evidence quality gates.

## Instructions to Self (Files to Read/Touch Next)

- Orchestrator: `argentum/orchestration/auction_chat.py` (wire concurrency; hard‑interrupt cutoff; metadata IDs)
- Managers: `argentum/coordination/auction_manager.py` (EmotionEngine, InterjectionPlanner; integrate reputation EMA)
- Audio: `argentum/audio/elevenlabs_controller.py` (export drift/p95 stats struct); `argentum/tts/elevenlabs_adapter.py` (real streaming impl)
- CLI wiring: `argentum/cli.py` (later: pass audio_controller into orchestrator scenarios)
- Docs: `docs/ORGANIC_AUCTION_CHAT_PLAN.md`, `docs/ORGANIC_AUCTION_IMPLEMENTATION_PLAN.md` (keep in sync)
- Tests: `tests/test_auction_orchestrator.py`, `tests/test_audio_elevenlabs.py` (expand for concurrency/hard‑interrupt)

## Next Steps (Prioritized Checklist)

1) WP3 — Concurrency
   - Add context_version snapshot; spawn async interjection/bid/prefetch calls with timeouts; cancellation on state change
   - Enforce `concurrency.global_limit` and per‑agent inflight=1; backoff on 429

2) WP3 — Hard‑Interrupt Cutoff
   - Implement guard‑band commit using `audio.compute_commit_guard_ms()`; prebuffer top‑1 kicker; fallback on miss
   - Append event metadata: `{event: 'hard_interrupt', costs, timing, ids}`

3) WP2 — Emotion & Interjections
   - EmotionEngine (llm/hybrid/heuristic) with EMA smoothing + deadband and influence cap; SSML prosody hints
   - InterjectionPlanner with types (support/clarify/challenge), cooldowns, and importance scoring

4) WP1 — Streaming Adapter (prod)
   - Real ElevenLabs stream + marks; honor env/manifest/CLI

## Lessons Learned / Engineering Notes

- Timing: Always use monotonic clocks in scheduling; drift_ms must be recorded and visible.
- Guard bands: Compute from p95 + buffer; apply a minimum floor; skip interrupt if winner not ready.
- Mixing: One ducking lease; crossfade cancels duck; serialize transitions to avoid artifacts.
- Concurrency: Snapshot context; cancel stale tasks; cap inflight per agent; drop low‑priority prefetch first.
- Token hygiene: Hard token caps at parser; truncate with ellipsis; sanitize interjections to strip SSML/quotes/fences.
- Defaults: Hard‑interrupt cutoff default; dual‑path segments optional for beat‑only; presets (debate_snappy, panel_civil) reduce config surface.

## Testing To Add

- Orchestrator concurrency: Ensure interjection/bid/prefetch tasks respect timeouts and concurrency caps.
- Hard‑interrupt cutoff: Commit only when winner audio ready; fallback otherwise; ensure no mid‑word artifacts (sim).
- Guard‑band tuning: Drift/miss metrics decrease after AIMD lead_time increases; commit_guard adapts.
- Economic stability: pacing convergence under synthetic agents; interrupt budgets enforced.

## Decision Log

- ElevenLabs selected as initial TTS; Sim controller for offline tests
- Hard‑interrupt cutoff set as default; dual‑path inline acks optional (beat‑only mode)
- Pacing EMA + softmax bid mapping; evidence‑backed tie‑breaks with quality gates
- AIMD adaptive lead_time and top_k prebuffer; respect discard budget

