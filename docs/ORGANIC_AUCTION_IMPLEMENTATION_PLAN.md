# Organic Auction Conversation – Implementation Plan (with Risk Mitigation)

Status: Planning (implementation-ready)
Target: v0.2 MVP behind flags; follow-ups in v0.2.x–v0.3
Owners: Orchestration, Agents, Workspace, Audio/TTS

## Scope

Implement the auction-driven, beat-aware multi-agent conversation mode with:
- ElevenLabs streaming TTS integration and “beat” scheduling
- New auction-based chat manager/orchestrator with dual-path segments
- Heuristic-first bidding, optional LLM mood/bids, typed interjections
- Token economy (single accrual mode per session), pacing, interrupt budgets
- Full observability via metadata and tests for timing/economic stability

## Milestones & Deliverables

- M1 (0.2-alpha, 1–2 sprints):
  - ElevenLabs adapter + audio controller with measured beats
  - AuctionChatManager (+ sub-managers) + AuctionGroupChatOrchestrator
  - Dual-path segments (s1, s2_no_ack, s2_ack_short), typed interjections
  - Heuristic bidding + pacing multiplier + interrupt budgets
  - Config/CLI flags; metadata IDs; core tests (unit + integration sim)

### WP1 Status (Audio/TTS)

Completed
- AudioController interface + SimAudioController with estimated beats
- ElevenLabsAudioController scaffold: mark-driven or estimated beats, ducking lease, crossfade cancellation
- Beat drift recording (planned vs actual) and first-chunk latency profiling (p95) + guard band computation helper
- Audio controller factory: selects Sim vs ElevenLabs via manifest/CLI config (offline-safe by default)
- Tests: marks/finish, ducking no-overlap, crossfade cancellation, estimated-beat drift bounds, p95 and guard-band exposure, factory selection

Remaining (production)
- Real ElevenLabs streaming adapter integration (networked) and env/manifest wiring for API key, voice, latency
- Drift and latency stats export to orchestrator for commit_guard_ms tuning
- Additional race tests (e.g., cancel interjection at exact beat boundary) and mixer polish

- M2 (0.2, 1 sprint):
  - EmotionEngine (LLM mood parsing, EMA smoothing, caps/hysteresis)
  - Evidence-priority lane (discount/tie-break) + reputation updates
  - One-segment reserve + guard band fallback handling (prefetch and reuse in orchestrator); more jitter tests
  - Docs + demo scenario (chaotic_debate)

- M3 (0.2.x–0.3):
  - Optional LLM bidding (JSON) with strict parsing + timeouts
  - Moderator guardrails (rate limits, anti-domination, topic discipline)
  - Telemetry/export for dashboards; tuning presets

## Work Packages (WPs) with Risk Reduction

### WP1: ElevenLabs TTS Adapter & Audio Controller

Tasks
- Implement `tts/elevenlabs_adapter.py` (streaming client, voice, latency mode)
- Audio controller API: `play(ssml|clips)`, `wait_for_beat(n)`, `duck_and_play(clip)`, `crossfade_to_silence(ms)`, `cancel()`
- Beat measurement: monotonic timestamps for marks/boundaries; compute drift
- Fallback when marks unavailable: pre-split S1/S2; schedule join as beat
- Client-side ducking/crossfade if provider lacks mixing primitives
- Voice profiling: record P95 first-chunk latency and set `commit_guard_ms = P95 + 50–75 ms` per voice/profile; store in `tts.voice_profile`.
 - Lead-time controller: AIMD-style adaptive lead time with sliding-window miss-rate; adjust `lead_time_ms` and `top_k_prebuffer` within configured caps; honor a discarded-audio budget.

Acceptance
- Start playback within latency budget; expose beat callbacks (or S1/S2 boundary)
- Interjection duck and resume; crossfade on hard interrupt
- Drift recorded in metadata; retry/timeout behavior defined
 - Under elevated miss-rate, lead time increases; when stable, it decreases gradually; discarded-audio stays under budget.

Risks & Mitigation
- Provider variability: aggressively profile first-chunk latency and cadence; calibrate guard bands
- No marks: use punctuation-based S1/S2 splitting; conservative beat windows
- Audio races/overlap: cancellation tokens; single active handle invariant; CI sim tests
 - Cost from prebuffering: enforce discarded-audio budget and `top_k_prebuffer_max`; back off lead time when exceeded.

### WP2: Auction Manager Decomposition

Tasks
- Implement sub-managers:
  - TokenManager (accrual mode, decay, anti-hoarding, stipend)
  - EmotionEngine (LLM mood parsing, EMA smoothing, caps, hysteresis)
  - AuctionResolver (heuristic desire→softmax bid; pacing multiplier EMA; tie-breaks)
  - InterruptPolicy (hard_interrupt cooldowns, budgets, dynamic fees)
  - InterjectionPlanner (typed interjections, quotas, cooldowns)
  - FairnessAdjuster (talk-time share window, least-recent fallback)
- `AuctionChatManager` composes interfaces; no “god class”
 - Reputation updates: use EMA rule `rep = (1 − α) * rep + α * event_value` with documented event values.

Acceptance
- Deterministic outcomes for same inputs; unit tests per sub-manager
- Pacing stabilizes talk-time share to target ±10% on synthetic agents

Risks & Mitigation
- Complexity growth: establish narrow interfaces and data contracts; test in isolation
- Oscillation/hoarding: EMA pacing + single accrual mode + decay; budgets for interrupts

### WP3: Auction Orchestrator

Tasks
- Micro-turn loop with measured beats and concurrency
- Interjection planning (typed, importance threshold) with safe duck-play
- Hard-interrupt cutoff with guard-band commit; provisional+confirmed metadata
- Reserve prefetch for likely next speaker; reuse next turn if selected; cancellable on context_version change
- Talk-share tracking and pacing update via resolver.update_pacing()
- Rich metadata on messages (IDs, timing drift, guard band/p95; auction bids/pacing/talk_share)

- `AuctionGroupChatOrchestrator` micro-turn loop (supports two interrupt modes):
  - Default: hard‑interrupt cutoff (discard current clip; crossfade 200 ms; prebuffer top‑1 kicker).
  - Optional: beat‑only inline acknowledgments via dual-path segments.
- Concurrency: spawn parallel LLM tasks (interjections, bids, next‑segment prefetch) with timeouts and cancellation; enforce global/per‑agent concurrency limits; guard band scheduling.
- Pre-resolve fallback next speaker; one-segment reserve caching
- Two-phase commit for interrupts; missed guard band → disable interrupt

Acceptance
- No deadlocks; progress probability ≥ 0.99 across seeds (liveness tests)
- Guard band respected; cutoff commits fired only when interrupter audio ready; fallback used otherwise

Risks & Mitigation
- Race conditions: cancellation tokens; explicit state machine for micro-turn
- Clicks/gaps: enforce crossfade durations; trim leading silence; prebuffer; tests for discontinuities
- Missed beats: reserve sentence; disable interrupt this turn; metrics on misses
 - Provider limits: global concurrency semaphore; backoff on 429; drop low‑priority prefetch tasks first.

### WP4: LLMAgent Extensions

Tasks
- `generate_segment()` returns JSON with `s1`, `s2_no_ack`, `s2_ack_short`; append optional `mood` block
- `generate_interjection()` returns one-liner per type; length/clamp/sanitize
- `propose_bid()` (Phase 3) returns tiny JSON or defers to heuristic
- Prompt templates; strict schema parsing; fallbacks
 - Parser token caps (segments ≤ 35 tokens; kickers ≤ 18 tokens) with ellipsis truncation; shared sanitize_text() for all TTS-bound content.

Acceptance
- Correct dual-path JSON parsed; total words within budget; SSML-safe text
- Mood values clamped [0,1]; EMA applied; hybrid influence capped (±1 token eq.)

Risks & Mitigation
- LLM drift/explanations: “JSON only” prompts; robust parsing; strip trailing prose
- Prompt injection: sanitize interjections (strip SSML, quotes, code fences)

### WP5: Config & CLI Wiring

Tasks
- Manifest keys: `conversation.mode=auction`, `emotion_control`, `tokens`, `pricing`, `anti_hoarding`, `cooldowns`, `evidence_priority`, `reputation`, `tts`
- CLI flags: `--mode auction`, `--emotion-control`, interjection quotas, interrupt cooldowns, TTS provider/voice/latency
- Feature-flag default: off; enabled per project/CLI
- Presets: add `debate_snappy`, `panel_civil` bundles (overridable), exposed via `--preset` and manifest `preset` key.
 - Concurrency block: `concurrency.global_limit`, `per_agent_limit`, `timeouts_ms`, and `prefetch` knobs.

Acceptance
- Defaults preserve existing behavior; new mode works when enabled
- Config precedence: CLI > manifest > defaults

Risks & Mitigation
- Backward compat: keep defaults identical; add validation + helpful errors
- Misconfig: schema and value validation with clear messages

### WP6: Persistence & Observability

Tasks
- Message.metadata: add IDs (`agent_id`, `speaker_id`, `auction_id`, `segment_id`, `clip_id`)
- Timing fields: `{ beat_planned_ms, beat_actual_ms, drift_ms }`
- Fairness fields: `{ pacing: float, talk_share_window: "N=60s" }`
- (Optional) quality fields `{ on_topic, toxicity }` for future reputation
- Session summary counters for KPIs
 - ID conventions: `auction_{sessionShort}_{turn:04d}`, `seg_{auctionId}_{agentIdx}_{nonce}`, `clip_{segId}`.

Acceptance
- All events traceable; IDs stable and unique; drift recorded; KPIs aggregated

Risks & Mitigation
- PII leakage (mood): keep session-scoped; opt-in persistence; anonymize/aggregate if exported
- Log volume: sampling for verbose traces; size caps

### WP7: Testing (Unit, Integration, Property, Stress)

Tasks
- Unit tests: all sub-managers, LLMAgent parsers, token invariants
- Integration sim: TTS jitter/delay (+250/+500/+1000 ms), race cancellations, missed beats
- Property tests: liveness, pacing stability, budgets enforcement, no negative balances, no overlaps
- Performance: N agents, target CPU/latency budgets

Acceptance
- Coverage on new modules ≥ 85%; core invariants hold; jitter tests pass

Risks & Mitigation
- Flaky timing tests: deterministic simulated clocks; seed control

### WP8: Docs & Demos

Tasks
- Expand `docs/ORGANIC_AUCTION_CHAT_PLAN.md` with API snippets + examples
- New `examples/chaotic_debate.py` demo using auction mode + ElevenLabs
- Update QUICKSTART for flags and manifest settings

Acceptance
- Reproducible demo; users can toggle modes and observe metrics

Risks & Mitigation
- Version skew: keep docs in PR; CI check that example imports

### WP9: Rollout & Feature Flags

Tasks
- Gated by manifest/CLI; default off
- Canary: dogfood internally with 2–3 agents; collect drift/miss metrics
- Gradual enablement in examples; document fallback and disable paths

Acceptance
- Safe enable/disable without data migration; clear rollback steps

Risks & Mitigation
- Unexpected costs: global rate limits; usage counters; alerting on thresholds

### WP10: Performance & Cost Controls

Tasks
- Per-minute LLM call budget; switch to heuristic bids when budget hit
- Interjection quotas; segment length adaptation when backlog grows
- Token accrual decay and caps

Acceptance
- Stay within CPU/network/LLM budgets under load scenarios

Risks & Mitigation
- Provider throttling: exponential backoff + fallback paths

### WP11: Safety & Moderation

Tasks
- Sanitize interjections; enforce length clamps
- Optional toxicity/on-topic scoring (local heuristic) to adjust reputation or mute
 - Shared sanitize_text() applied to all interjections/kickers before any TTS calls; strip SSML, quotes, fenced code, tricky Unicode.

Acceptance
- No control injection via interjections; safe content by default

Risks & Mitigation
- Overblocking: mild thresholds with appeal (no hard failure on low severity)

### WP12: Quality & Budget Governor

Tasks
- Implement a runtime governor that adjusts concurrency, prefetch depth, lead_time, interjection quotas, and TTS expressivity based on budget/health.
- Support modes: auto/baseline/enhanced/premium; load preset then apply overrides.
- Enforce LLM and discarded-audio budgets; collect KPIs (miss rate, drift, P95 first-chunk, overlaps).

Acceptance
- Under high budget/healthy conditions, governor raises features within caps; under stress, it steps down without audible artifacts; budgets respected.

Risks & Mitigation
- Oscillation: use AIMD; introduce hysteresis windows and minimum dwell before tier changes.
- User confusion: expose governor status and effective settings in logs/CLI; document presets and overrides.

## Cross-Cutting Engineering Practices

- Monotonic clocks everywhere; no `time.time()` in scheduling
- Cancellation tokens for all scheduled audio actions; single-active-handle invariant
- Deterministic IDs for events; traceability in logs/telemetry
- Strict JSON parsing with schema validation; fallbacks always defined

## Estimates (very rough, dev-days)

- WP1 TTS Adapter & Audio: 5–8
- WP2 Managers Decomposition: 6–9
- WP3 Orchestrator: 5–7
- WP4 LLMAgent Ext: 4–6
- WP5 Config/CLI: 2–3
- WP6 Persistence/Obs: 2–3
- WP7 Tests: 6–10 (in parallel with WPs)
- WP8 Docs/Demos: 2–3
- WP9 Rollout/Flags: 1–2
- WP10 Perf/Cost: 2–3
- WP11 Safety: 1–2

Total: ~36–56 dev-days (team-parallelized across domains)

## Acceptance Gates (per milestone)

- M1: Core flow works in simulation + basic TTS; unit/integration tests green; drift recorded; feature flag off by default
- M2: EmotionEngine active; evidence-priority lane; reserve/guard band fully exercised; demo published
- M3: Optional LLM bidding; moderator policies; telemetry/export stable; tuning presets documented

## Rollback Plan

- Feature-flagged: disable `conversation.mode=auction` to revert to existing orchestrators
- If audio issues: switch to sequenced interjections (no overlap), hard interrupts disabled
- If costs spike: force heuristic bids + shorter segments; cap interjections per minute

---

This plan decomposes the work to minimize risk, scopes acceptance criteria per package, and ties each risk to a specific mitigation and test.

## Parallelization Plan & Critical Path

Lanes (execute concurrently where possible):

- Lane A: Audio/TTS (WP1) – Critical Path
  - Deliver an early `AudioController` interface stub (play/cancel/crossfade/duck/wait_for_beat) in week 1 to unblock WP3.
  - Implement ElevenLabs adapter and measured-beat plumbing behind the interface.

- Lane B: Managers (WP2)
  - Define manager interfaces (TokenManager, EmotionEngine, AuctionResolver, InterruptPolicy, InterjectionPlanner, FairnessAdjuster) and data contracts in week 1.
  - Implement and unit-test each manager independently; provide a mock `AuctionChatManager` for WP3.

- Lane C: Orchestrator (WP3)
  - Start with mocks for AudioController and AuctionChatManager; implement hard‑interrupt cutoff flow, guard band, prebuffer logic.
  - Integrate with real WP1 and WP2 once interfaces are stable.

- Lane D: LLMAgent (WP4)
  - Implement dual-path (optional), kicker/follow-on prompts, parser token caps, and sanitize_text.

- Lane E: Config/CLI (WP5) & Observability (WP6)
  - Wire manifest schema, presets, CLI flags; add deterministic IDs and metadata fields concurrently.

- Lane F: Testing (WP7)
  - Build simulation/jitter harness; add unit/property tests for managers and orchestrator using mocks; later switch some to real AudioController.

- Lane G: Perf/Cost (WP10) & Safety (WP11)
  - Enforce quotas/decay, discarded-audio budget, and sanitization early; refine after end-to-end integration.

- Lane H: Docs/Demos (WP8) & Rollout (WP9)
  - Draft examples and QUICKSTART changes in parallel; keep feature flag off until M1 acceptance.

Critical Path
- AudioController (WP1) interface + Orchestrator (WP3) loop + Managers (WP2) API.
- Integration checkpoints:
  1) T+1 week: Interface freeze for AudioController and Manager APIs; orchestrator running with mocks.
  2) T+2 weeks: Orchestrator integrated with WP1/WP2; basic timing tests green.
  3) T+3 weeks: EmotionEngine + evidence lane wired; demo scenario dogfood.

Coordination Practices
- RFC/IDL for interfaces (pydantic models/TypedDicts) checked in before implementation.
- Daily 10‑min integration sync; CI runs sim/jitter tests on every PR.
- Feature flag remains off until M1 acceptance; canary runs for drift/miss metrics.
