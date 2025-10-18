# Organic Auction Conversation – Detailed Design Plan

Status: Draft (implementation-ready)
Owners: Core orchestration + workspace team
Target Version: 0.2 (incremental behind flags)

## 1) Goals & Non‑Goals

- Goals
  - Produce more organic multi-agent debates with interjections and occasional interruptions.
  - Keep latency and cost bounded while supporting TTS playback overlap with orchestration.
  - Provide tunable policy knobs (pricing, fairness, emotion) without breaking current flows.
  - Ensure observability: every event (bid, interjection, interrupt) appears in transcript metadata.

- Non‑Goals (for MVP)
  - True token-level barge-in of TTS (we target sentence/beat boundaries, not mid-word).
  - Full-blown streaming LLM content synthesis (we keep short segments instead).
  - Complex social dynamics (alliances, multithreaded side conversations) — future work.

## 2) Core Concepts

- Segment: a short unit of speech by the active speaker (2–3 short sentences or 1–2 long sentences), designed for 4–6 seconds of TTS playback. Segments include SSML breaks/marks to create barge points.
- Beat: a scheduled boundary within a segment where interjections or interrupts may occur (end-of-sentence SSML break or <mark/>).
- Micro-Turn: the lifecycle from starting playback of one segment to scheduling the next speaker decision (or interrupt at a beat).
- Token Economy: agents accrue “bid tokens” periodically; they spend tokens to win the next speaking slot, to interrupt with a “kicker”, or to inject a brief interjection.
- Emotion Model: each agent maintains simple emotional variables (e.g., frustration, engagement) that influence bidding and tone.

LLM‑controlled mood (option): The agent’s character can self‑determine mood via the LLM and report it explicitly each micro‑turn. Heuristics are retained as a fallback or for hybrid control.

## 3) Architecture Overview

New components (additive; no breaking changes):

- argentum/coordination/auction_manager.py (new)
  - Top-level coordinator; composes the following sub-managers to avoid a "god class":
    - TokenManager: accrual/decay, balances, anti-hoarding, catch-up stipend.
    - EmotionEngine: LLM mood parsing, smoothing (EMA), hybrid nudges, prosody hints.
    - AuctionResolver: collects bids (heuristic or LLM JSON), determines winners, applies pricing/fairness.
    - InterruptPolicy: decides soft/hard interrupts, cooldowns, dynamic fees; terminology in code: `hard_interrupt` (formerly "kicker").
    - InterjectionPlanner: selects and caps interjections, types and insertion beats, speaker acknowledgment hints.
    - FairnessAdjuster: talk-time share, progressive tax, tie-breakers (least-recent speaker).

- argentum/orchestration/auction_chat.py (new)
  - Orchestrates micro-turns: requests segments from the active speaker, kicks off TTS playback (via integration hooks), collects bids/interjections in parallel, and applies outcomes at beat marks.

- SSML/TTS Integration (provider-agnostic adapter) (optional new module or utilities)
  - Converts generated segments into SSML with <break time="…"/> and <mark name="beatN"/>.
  - Publishes “barge points” (beat timestamps or mark names) to the auction manager/orchestrator.
  - Supports ducking and crossfade cues for interjections/interrupts (if TTS stack permits), or simulates timing in CLI mode.
  - Adaptive beat placement (optional): consume TTS-provided speech marks or real-time callbacks to align barge points with prosodic units rather than fixed estimates.
  - Initial provider: ElevenLabs (streaming). We will:
    - Use the streaming API to begin playback as soon as first chunks arrive.
    - Approximate beat timing via sentence boundaries and measured wps if explicit marks are not available.
    - Handle ducking/crossfade client-side in the audio controller if the provider has no built-in mixing primitives.
    - Profile first-chunk latency and chunk cadence for dynamic scheduling.

Existing components extended (minimally):

- LLMAgent additions (non-breaking helpers):
  - generate_segment(messages, context, constraints): short segment (≤ 2–3 sentences) honoring persona, style, and current mood.
  - generate_interjection(messages, context): one-liner (≤ 12–15 words).
  - propose_bid(...): optional JSON-only tiny response with desired bid and kicker (enabled by flag; default is heuristic bidding).

LLM‑mood control interfaces (new options):

- LLMAgent may return both segment text and a compact MOOD block in machine-readable form. The orchestrator parses and updates the agent’s state. This is gated by a per-agent or per-session setting (see §11).
  - Output pattern example (robust to providers):
    - Either JSON envelope: `{ "segment": "…", "mood": { "frustration": 0.42, "engagement": 0.61, "confidence": 0.55 } }`
    - Or tagged block:
      ```
      <<<SEGMENT>>>
      … two short sentences …
      <<<MOOD>>>
      {"frustration":0.42,"engagement":0.61,"confidence":0.55}
      ```
  - No rationale is required/desired; we only consume the numeric summaries to avoid chain‑of‑thought leakage and keep cost low.

## 4) Data & Metadata Model

Reuse Message.metadata to annotate discrete events. Suggested envelope fields:

- message.metadata.event: "segment" | "interjection" | "bid" | "interrupt" | "auction" | "timing"
- Common subfields:
  - costs: { tokens_spent: int }
  - emotion: { frustration: float, engagement: float, confidence: float }
  - auction: { bid: int, kicker: bool, result: "win"|"lose"|"pass" }
  - timing: { beat: int, at_ms: int, duration_ms: int }
  - provenance: { policy: string, version: string }
  - citations: [] (if the segment uses RAG references; unchanged from existing pipeline)

Schema sketch:

```json
{
  "event": "interjection",
  "costs": { "tokens_spent": 2 },
  "emotion": { "frustration": 0.4, "engagement": 0.7 },
  "timing": { "beat": 1, "at_ms": 2600 },
  "auction": null,
  "provenance": { "policy": "auction/v1", "version": "0.1" }
}
```

Agent state (not persisted by default; session-scoped in AuctionChatManager):

```python
class AgentAuctionState(BaseModel):
    tokens: int = 0
    # LLM‑reported mood (authoritative if emotion_control == "llm")
    frustration: float = 0.0   # 0..1
    engagement: float = 0.5    # 0..1
    confidence: float = 0.5    # 0..1
    last_spoke_at: float | None = None
    last_bid: int = 0
    interrupts_made: int = 0
    interjections_made: int = 0
    cooldown_until: float | None = None
    reputation: float = 0.0

    # Control flags (resolved per session)
    emotion_control: Literal["llm","heuristic","hybrid"] = "heuristic"

Update policy:
- If `emotion_control == "llm"`, clamp and smooth LLM‑reported values (EMA) and treat them as authoritative.
- If `"hybrid"`, use LLM values as baseline and apply small heuristic nudges (e.g., ±0.05) from salient events.
- If `"heuristic"`, ignore LLM mood payloads and rely on event‑driven heuristics.
```

## 5) Timing Model

- Speaking rate: assume 150–180 wpm (≈ 2.5–3.0 words/sec).
- Segment length target: 4–6 seconds (≈ 12–18 words), min 3s, max 10s.
- Beats: define SSML boundaries between sentences with <break time="250ms"/> and/or <mark name="beatN"/>. Prefer TTS-provided speech marks when available.
  - ElevenLabs note: if speech marks are unavailable, use punctuation-based beats and provider-specific latency profiles to estimate barge points conservatively.
- Scheduling window per micro-turn:
  - T0: Start playback of segment S.
  - T0+100ms: open auction (concurrent heuristic or tiny JSON bids).
  - At first beat (≈1.5–3.0s): resolve interjection (if any) — duck −12dB for ~1.2s, play interjection clip, resume.
  - At next beat: evaluate interrupts — if a kicker won, crossfade 200–300ms to new speaker; else continue.
- Deadline: ensure auction result available ≥2.0–2.5s before segment end; if not, consume reserve sentence to buy time.
- Adaptive segments: shorten when bid backlog is high; lengthen when queue empty.

Robustness & profiling:
- Phase 1 feasibility check: verify ElevenLabs capabilities for streaming pause/stop, mid-stream cutover, and any timing callbacks. If marks/duck/crossfade are unavailable, degrade gracefully (no overlap; interjections queued to next beat or appended as short follow-ups) and perform mixing on the client.
- Aggressive profiling: measure end-to-end TTS latencies, first-chunk delay, average per-word duration. Calibrate break durations dynamically based on provider metrics.
- Latency handling: if beat callback arrives late (> X ms), skip interjection/interrupt for that beat and defer to next; optionally play reserve sentence.
- Dynamic guard band: compute commit_guard_ms per voice/provider as P95(first_chunk_latency) + 50–75 ms (from tts.voice_profile) and revisit after each session.

Adaptive lead time & prebuffer (A4):
- Maintain a sliding-window beat_miss_rate; if it exceeds a threshold (e.g., 20% over the last 20 segments), aggressively increase prefetch lead time and/or prebuffer depth; if it drops, reduce slowly.
- AIMD-style control: lead_time_ms += add_increase_ms on miss, lead_time_ms *= mult_decrease on sustained success; clamp to [baseline_ms, max_ms].
- Prebuffer strategy: increase `top_k_prebuffer` from 1→2 when miss_rate is high and budget allows; otherwise keep at 1 to limit cost.
- Cost guard: enforce a max discarded-audio budget (seconds/minute). If exceeded, cap lead_time_ms and reduce top_k_prebuffer.

Measured beats (A2):
- Use a monotonic clock (e.g., time.monotonic()) to timestamp actual beat/mark callbacks. Record planned_at_ms vs actual_at_ms and drift_ms in metadata.
- Resolve auctions relative to actual Beat 1/2 minus a guard band (150–250ms), not just based on scheduled offsets.
- If marks are unavailable, pre-split audio into S₁ and S₂ files and treat the join boundary as the beat.

Cancellation & race safety:
- All scheduled awaits for interjections/interrupts must accept cancellation tokens and ensure only one audio handle plays at any time.

## 6) Token Economy & Pricing (MVP)

- Accrual: +1 token per 30s wall clock or per completed segment (configurable). Max bank: 6–8.
- Costs:
  - Speak (next slot): bid ≥ 1 token (winner pays their bid).
  - Interrupt (kicker): bid ≥ current highest + kicker_delta (e.g., +2); winner pays (bid + kicker_fee).
  - Interjection: 1–2 tokens; capped at 1–2 per segment; per-agent cooldown (e.g., 2 segments).
- Heuristic bid (no-LLM, default):
  - desire = w_backlog*B + w_recency*R + w_emotion*(0.5 + frustration - 0.25*engagement)
  - base_bid = clamp(round(desire), 0, tokens)
  - fairness tax: if talk_time_share > target_share, increase effective cost by factor (>1).
  - kicker_policy: only allow after first sentence; require base_bid ≥ frontier and tokens ≥ (base_bid + kicker_fee).
- Dynamic pricing (optional): escalate interrupt fees with repeated hard interrupts or in heated sessions.

Separation from mood:
- Mood is not derived from token balances. Bidding aggressiveness may optionally reference mood (e.g., frustrated → higher bid), but tokens do not directly set mood. This preserves character agency.

### 6.1 Accrual Mode & Anti‑Hoarding

- Single accrual mode per session (choose one):
  - Per-segment: +1 on each completed segment.
  - Wall-clock: +1 per 30s when not speaking (gate accrual); apply periodic decay to prevent hoarding (e.g., tokens = floor(tokens*0.98) per minute).

- Anti‑hoarding:
  - Soft token decay on large balances (e.g., if tokens > 6, decay 10% per minute).
  - Diminishing returns: bidding discounts taper off when balance exceeds a threshold.
  - “Use it or lose it” reserve: a small portion of the bank (e.g., 1 token) expires after K segments if not spent.
- Catch‑up/anti‑poverty:
  - Baseline stipend: ensure min 1 token per M segments (or 30–60s) for silent agents.
  - Underspoke subsidy: agents below target talk‑time share receive temporary bid rebates.
  - Tie‑break priority for agents who have not spoken recently.

### 6.2 Cooldowns & Dynamic Interrupt Semantics

- Interrupt cooldown: after a successful hard_interrupt, an agent cannot interrupt again for the next N micro‑turns.
- Dynamic hard_interrupt_fee: scale by current speaker reputation/confidence and session heat (recent interrupts). Interrupting a high‑reputation, confident speaker costs more.
- Soft vs hard interrupts: default to hard only after at least one sentence; soft (defer to next beat) when the current segment is nearly complete.

Hard‑interrupt cutoff (discard‑and‑regenerate) mode (recommended default):
- Treat the current utterance as expendable; on interrupt commit, stop the current clip and switch to the interrupter’s short “kicker” line. The interrupted speaker will later regenerate their next segment from the new context (we never resume discarded text).
- Guard band + commit window: resolve interrupts ~150–250 ms before the cut. If the winner’s audio is not ready by the commit deadline, skip the interrupt and let the segment finish.
- Audio UX: on commit, apply ~150–250 ms exponential fade‑out on the current clip, then ~50–100 ms fade‑in on the interrupter; optionally duck −6 dB during crossfade. Trim leading breaths/silence from the kicker.
- Pre‑buffer: when an agent bids to interrupt, pre‑synth a very short kicker (8–12 words) for the top N bidders (N=1–2). If they lose, drop the buffer; if they win, the handoff is instant.
- Post‑interrupt: either grant the floor to the interrupter for a full micro‑turn or bounce back via the next auction. The original speaker always regenerates fresh text for their next turn.
- Budgeting/fairness: charge for the bid and (optionally) for pre‑synthesis usage; cap hard interrupts per window and escalate fees on repeated use.

### 6.3 Evidence Priority Lane (RAG)

- Bid flag: `evidence_backed: true` with at least one citation label from retrieved knowledge.
- Discounts/tie‑breaks: evidence‑backed bids receive a small discount or win ties.
- Rate limits: at most one evidence‑backed priority per agent every P segments to avoid gaming.
 - Quality gate: require provenance IDs and a minimum overlap/relevance score; otherwise ignore the evidence flag and apply a small, temporary reputation penalty.

### 6.4 Reputation Integration

- Reputation accrual:
  - +Δ after an uninterrupted, coherent segment.
  - +Δ for evidence‑backed contributions adopted into consensus or endorsed by others.
  - −Δ when successfully interrupted (especially early) or for low‑quality spam interjections.
- Reputation effects:
  - Small bidding discount (e.g., up to 10%).
  - Faster stipend accrual within caps.
  - Higher kicker_fee against high‑reputation speakers.
  - Decay over time to keep system responsive.
 - Update rule (explicit): `reputation = (1 − α) * reputation + α * event_value`, with α ≈ 0.05–0.10.
   - Example event_values: uninterrupted_segment +0.3; strong_evidence +0.2; successful_interrupt_against_you −0.2; low_quality_interjection −0.1.

### 6.5 Auction Pacing & Bid Selection (A3)

- Replace raw fairness tax with an EMA‑based pacing multiplier p_i ∈ (0,1] per agent, updated from talk‑time share vs target. Apply as b′_i = p_i * b_i.
- Compute heuristic desire as before, but map to base bid via softmax for smoother behavior:
  - b_i = min(tokens_i, round( softmax(desire_i * τ) * max_bid )), with τ≈0.2–0.4.
- Explicit interrupt budgets: limit hard_interrupts per sliding window (e.g., max 2 per 5 segments) and escalate hard_interrupt_fee with recent usage.
- Pre‑resolve fallback next speaker at segment start. If the proper auction misses the guard band, disable interrupt this turn and hand off to the pre‑resolved next speaker.
- Keep a one‑segment reserve pre‑generated for the selected next speaker to minimize handover latency; discard if auction outcome changes.

## 7) Interjections

- Types: support | clarify | challenge (simple classifier or specify in prompt).
- Constraints: ≤ 12–15 words; persona-aligned; no full arguments.
- Scheduling: at first available beat; if backlog exists, only 1 interjection allowed per segment globally.
- Routing: speaker receives interjection text and decides whether to acknowledge now (brief clause) or defer.

Acknowledgment guidance (prompt + heuristic):
- Each interjection includes a type (support/clarify/challenge) and a model‑estimated importance score 0..1 (tiny JSON, or heuristic based on keywords).
- Speaker prompt: “If importance ≥ τ or type == clarify that resolves ambiguity, acknowledge briefly (≤ 8 words) before continuing; otherwise ignore or defer.”
- Budget: at most one acknowledgment per segment to prevent derailment.

 Dual‑path segments for timely acknowledgments (A1) – optional:
 - Preferable for beat‑only sessions; optional when `interrupt.mode = cutoff` since acknowledgments can happen after the interrupt when the speaker regenerates.
 - If enabled: the speaking agent outputs `s1`, `s2_no_ack`, `s2_ack_short` (total ≤ ~18 words). We always play S₁; at Beat 1 select S₂ path based on whether an interjection occurred. If live splicing is unsupported, pre‑synthesise S₂a/S₂b and crossfade (~200–300 ms) at Beat 1.

Pass states:
- If all agents pass (bid=0 and no interjection intent), default behaviors:
  - If current speaker has not exceeded `max_contiguous_segments`, they continue for one more segment (shortened).
  - Else, select least-recent speaker (or fallback to round-robin) to maintain flow.

## 8) Emotion Model

- Variables: frustration, engagement, confidence (0..1, clamped).
- LLM‑controlled mode:
  - Each micro‑turn, the LLMAgent reports normalized mood values (0..1). We clamp to [0,1] and smooth via EMA (e.g., α=0.4) to avoid jitter.
  - Optional bounds and rate‑limit changes per turn to prevent extreme swings (e.g., Δ ≤ 0.3) and hysteresis: require a minimum dwell (e.g., 2 micro‑turns) before sign changes.
  - These values map to SSML prosody (rate/pitch/volume) and to bid heuristics only if configured.
- Heuristic mode:
  - Lose bid: frustration += 0.15; engagement += 0.05
  - Win bid: frustration -= 0.10; confidence += 0.05
  - Make interrupt: frustration += 0.05; engagement += 0.10
  - Receive interjection: confidence ± small (tone-dependent); frustration += 0.05 if adversarial
  - Speak (segment end): frustration -= 0.05; engagement -= 0.05 (cooling)
  - Decay each micro-turn: frustration *= 0.95; engagement *= 0.98
- Tone mapping: inject emotion into system prompt and SSML prosody (rate±5%, pitch±2%, volume subtle).

Notes on agency:
- LLM mood control is authoritative in `llm` mode. Heuristics never override mood; they may only suggest small deltas in `hybrid` mode.
- Influence cap: in `hybrid` (and by default), cap mood’s impact on bidding to ±1 token equivalent per micro‑turn to prevent gaming.
 - Deadband: apply a signed deadband of ±0.1 where mood has zero effect on bidding to reduce jitter and micro‑gaming.

## 9) Orchestrator Flow (Pseudocode)

```python
async def run_session(agents, task, ctx, policy):
    manager = AuctionChatManager(policy)
    active = None
    while not manager.should_terminate(ctx):
        manager.tick_tokens()
        if not active:
            active = manager.select_next_speaker(agents, ctx)  # auction win
        # 1) Get next segment pack (short)
        # Speaker produces dual-path segment
        segment = await active.generate_segment(messages=ctx.get_messages(), context=task.context)
        # Optionally parse LLM‑reported mood
        if policy.emotion_control_for(active) != "heuristic":
            mood = parse_optional_mood_payload(segment)
            if mood:
                state.update_mood_from_llm(mood)
        ssml, beats = to_ssml_with_beats(segment)
        # 2) Start playback (or simulate timing)
        playback = audio.play(ssml)
        # 3) In parallel, collect bids & interjections for next beats; pre-resolve fallback next speaker
        bids_task = manager.collect_bids_parallel(agents, ctx)  # heuristic or LLM JSON
        interjections = manager.maybe_interjections(agents, ctx)
        fallback_next = manager.preselect_next_speaker(agents, ctx)
        # 4) At first beat, insert interjection if any
        await playback.wait_for_beat(1)
        if interjections:
            audio.duck_and_play(interjections[0])
        # 5) Resolve auction decision with guard band + two-phase commit
        next_speaker, interrupt = await manager.resolve(bids_task)
        if interrupt and playback.can_interrupt():
            audio.crossfade_to_silence(300)
            active = next_speaker
            continue
        # If auction missed guard band, disable interrupt and use fallback
        if manager.missed_guard_band():
            next_speaker = fallback_next
        # 6) Finish segment; schedule next speaker (use reserve segment if available)
        await playback.finish()
        active = next_speaker
```

### 9.1 Concurrent LLM Generation & Speculation

- Context snapshots: when a micro‑turn starts, freeze a `context_version` (e.g., last committed message ID and turn counter). All concurrent LLM calls receive that snapshot and must echo it back in metadata.
- Parallel tasks (spawned with timeouts and cancellation):
  - Interjections: for all non‑speakers, generate ≤12‑word interjection based on the snapshot; priority high.
  - Bids: collect heuristic/LLM bids concurrently; priority high.
  - Prefetch next segment: for the most likely next speaker(s) (top‑K=1..2), generate a short segment ahead of time; priority medium; mark as `prefetch=true`.
- Cancellation & staleness: on interrupt/beat change, cancel outstanding tasks. On commit, only accept outputs whose `context_version` matches the current one; else discard/regenerate.
- Concurrency limits & budgets: enforce a global concurrency semaphore (e.g., 6–8) and per‑agent inflight=1. If over budget, drop lower‑priority prefetch tasks first. Respect provider rate limits and back off on 429.
- Latency goals: interjection ≤ 600 ms; bid JSON ≤ 500 ms; prefetch segment ≤ 1200 ms. Use timeouts and heuristics on miss.
- Observability: record `llm_latency_ms`, `queue_delay_ms`, `prefetch` flag, and `context_version` in metadata for all speculative outputs.

## 10) Prompts (LLM-Driven Options)

- Segment Prompt (system role merged with persona + mood):

  If `interrupt.mode = cutoff` (no inline acknowledgments during playback):

  "Produce 2 short sentences (≤18 words total) continuing your point. Maintain persona. If instructed to report mood, append a MOOD JSON block with normalized values 0..1 for frustration, engagement, confidence. No explanations."

  If `interrupt.mode = beat_only` (inline acknowledgement desired):

  "Output three fields: s1, s2_no_ack, s2_ack_short (total ≤18 words across all). Maintain persona. If you received interjections, s2_ack_short should briefly acknowledge (≤8 words) before continuing. If instructed to report mood, append a MOOD JSON block with normalized values 0..1 for frustration, engagement, confidence. No explanations."

- Interjection Prompt:

  "Produce one interjection (≤12 words) reacting to the last 1–2 sentences. Tone: support|clarify|challenge aligned with persona. Output only the sentence."

- Bid JSON Prompt (optional):

  "You have {tokens} tokens. Mood: frustration={f:.2f}, engagement={e:.2f}. Decide action for the next beat. Output JSON only: {\"action\": \"speak|interject|pass\", \"bid\": int, \"kicker\": bool}"

Strictness & validation:
- Use explicit “JSON only” instructions; strip trailing prose robustly; parse with strict schema and fallback to heuristics on error or timeout.
- For mood, clamp to [0,1] and apply EMA smoothing; ignore values if malformed.
 - Parser token caps & truncation: enforce hard caps at parse-time (segments ≤ 35 tokens; kickers ≤ 18 tokens). If exceeded, truncate and append ellipsis.
 - Sanitisation: all interjections/kickers pass through a shared sanitize_text() (strip SSML, quotes, fenced code, tricky Unicode) before any TTS call.

## 11) Configuration (Workspace Manifest)

Extend project.json (optional; defaults baked into manager):

```json
{
  "conversation": {
    "mode": "auction",               
    "segment_seconds": {"min": 3, "target": 5, "max": 10},
    "beats": 2,
    "interjections": {"max_per_segment": 1, "cost": 2, "cooldown_segments": 2},
    "tokens": {"accrual": "per_segment", "period_seconds": 30, "max_bank": 8},
    "pricing": {"kicker_delta": 2, "kicker_fee": 1},
    "fairness": {"target_share": 0.33, "tax_rate": 0.15},
    "emotion_control": "llm",
    "anti_hoarding": {"decay": 0.1, "threshold": 6, "reserve_expiry_segments": 6},
    "cooldowns": {"interrupt_microturns": 2},
    "evidence_priority": {"enabled": true, "discount": 0.1, "agent_window_segments": 3},
    "reputation": {"discount_cap": 0.1, "decay": 0.02},
    "tts": {"provider": "elevenlabs", "voice": "default", "latency_mode": "low"},
    "interrupt": {"mode": "cutoff", "crossfade_ms": 200, "commit_guard_ms": 200, "prebuffer_top": 1, "kicker_words": [8, 12]},
    "lead_time": {"baseline_ms": 1000, "max_ms": 10000, "add_increase_ms": 250, "mult_decrease": 0.8, "miss_rate_threshold": 0.2, "window_segments": 20, "top_k_prebuffer_max": 2, "discard_budget_s_per_min": 4}
    ,
    "concurrency": {
      "global_limit": 8,
      "per_agent_limit": 1,
      "timeouts_ms": {"segment": 1200, "interjection": 600, "bid": 500},
      "prefetch": {"segments_top_k": 1, "interjections": true, "bids": true}
    },
    "quality": {
      "mode": "auto",               
      "preset": "debate_snappy",     
      "budget_per_min": 0.00,         
      "llm_calls_per_min": 60,        
      "tiers": {
        "baseline": {
          "concurrency.global_limit": 6,
          "lead_time.baseline_ms": 800,
          "interrupt.prebuffer_top": 1,
          "interjections.max_per_segment": 1,
          "tts.enhanced_expressivity": false
        },
        "enhanced": {
          "concurrency.global_limit": 8,
          "lead_time.baseline_ms": 1200,
          "interrupt.prebuffer_top": 1,
          "interjections.max_per_segment": 1,
          "tts.enhanced_expressivity": true
        },
        "premium": {
          "concurrency.global_limit": 10,
          "lead_time.baseline_ms": 1500,
          "interrupt.prebuffer_top": 2,
          "interjections.max_per_segment": 2,
          "tts.enhanced_expressivity": true,
          "audio.spatial": true
        }
      }
    }
  }
}
```

CLI flags (examples):

- `argentum debate --mode auction --segments 8 --interjections 1 --tick-seconds 30`
 - `--emotion-control llm|heuristic|hybrid`
 - Additional knobs (examples): `--cooldown-interrupts 2 --evidence-priority --anti-hoarding on`
 - TTS provider flags (examples): `--tts-provider elevenlabs --tts-voice <voice_id> --tts-latency low|balanced|high` (alias to latency_mode)

Presets (high-level bundles, overridable):
- preset: `debate_snappy`
  - interrupt.mode=cutoff, crossfade_ms=200, commit_guard_ms=dynamic, prebuffer_top=1
  - tokens: accrual=per_segment, max_bank=8
  - interjections.max_per_segment=1; cooldowns.interrupt_microturns=2
  - evidence_priority.enabled=true; emotion_control=hybrid; mood cap=±1 token; deadband=±0.1
- preset: `panel_civil`
  - interrupt.mode=beat_only, beats=2–3, crossfade_ms=250, prebuffer_top=0
  - tokens: accrual=per_segment, max_bank=6
  - interjections.max_per_segment=0–1; cooldowns.interrupt_microturns=3
  - evidence_priority.enabled=true; emotion_control=llm; pacing α small

## 12) Observability & Metrics

- Counters: bids per agent (wins/losses), interrupts, interjections, tokens spent/earned, talk-time share, fairness index, on-topic ratio.
- Transcript: annotate events in `Message.metadata` and include timing marks.
- CLI: optional table showing auction results after each segment.

Metadata fields to add:
- IDs: `agent_id`, `speaker_id`, `auction_id`, `segment_id`, `clip_id`
- Timing: `{ beat_planned_ms, beat_actual_ms, drift_ms }`
- Fairness: `{ pacing: float, talk_share_window: "N=60s" }`
- Quality (future): `{ rater: "auto", on_topic: float, toxicity: float }`
 
ID conventions (deterministic):
- `auction_{sessionShort}_{turn:04d}`
- `seg_{auctionId}_{agentIdx}_{nonce}`
- `clip_{segId}`

Guard-band state machine:
- Use a single boolean `interrupt_armed` that flips to false at `T_cut − commit_guard_ms` if `winner_audio_ready == false`. This ensures we never half‑cut audio when the interrupter isn’t ready.

## 16) Quality & Budget Governor (Runtime Scaling)

- Purpose: dynamically scale realism and resource use based on live budget/health.
- Modes: `auto` (default), or fixed `baseline|enhanced|premium`. Preset can be applied then overridden per field.
- Inputs: budget_per_min, llm_calls_per_min, CPU/latency health, drift/miss rates, discarded-audio budget, audience signals.
- Controls (examples):
  - Concurrency (global/per-agent), prefetch depth (top_k), adaptive lead_time bounds
  - Segment target length; interjection quotas; interrupt mode (cutoff vs beat-only)
  - TTS expressivity toggles (prosody mapping strength); audio spatialization
  - Evidence lane strictness; mood influence cap; pacing α
- AIMD governance: widen when healthy/funded; tighten on overload/misses/rate limits.
- Safety: never exceed discarded-audio or LLM budget caps; degrade gracefully (e.g., drop prefetch first, then reduce concurrency) without audible artifacts.

Audience influence (optional):
- External signals (e.g., tips, claps) translate to temporary boosts (e.g., small reputation bump, evidence discount, or pacing relief) with caps and cooldowns to prevent dominance.

## 13) Persistence & Knowledge Integration

- No schema changes required; events are captured in message metadata and highlights.
- Warm cache: index interjections and segments as usual.
- Knowledge graph (optional enhancement): add edge types like INTERRUPTED to visualize dynamics.

## 14) Testing Strategy

- Unit tests (manager): token accrual, bid resolution, kicker policy, fairness tax, emotion updates.
- Orchestrator simulation: mock audio marks; verify interjection insertion and interrupt timing.
- Prompt contracts: bid JSON schema validation with fallback to heuristic.
- Mood contracts: mood JSON schema validation; clamping and EMA smoothing; hybrid mode applies small heuristic nudges only.
- Economic stability:
  - Verify anti‑hoarding works (balances don’t grow unbounded under passive agents).
  - Catch‑up: agents recover from extended losing streaks; ensure no starvation.
  - Cooldown: repeated interrupts are rate‑limited; costs escalate as configured.
  - Evidence lane: discounts apply only with valid citations and abide by rate limits.
 - Liveness/property tests: ensure progress ≥ 0.99 probability over any two micro‑turns across random seeds.
 - Drift budget: inject ±150ms jitter; interjection lands within ≤250ms of target in ≥95th percentile.
 - Token invariants: banks never <0 or >max_bank; interrupt budgets enforced per window.
 - Pacing stability: with synthetic quiet/fair/aggressive agents, talk‑time share converges to target ±10% over ~2 minutes.
 - Audio races: cancel interjection at last millisecond; assert no double-duck or overlapping clips.

Integration (timing) tests:
- Simulate TTS latency jitter and delayed beat callbacks (e.g., +250ms, +500ms, +1000ms) and verify:
  - Interjections are deferred cleanly to next beat if late.
  - Interrupts do not cut mid-word; reserve sentence pads when needed.
  - Deadlines met within configured bounds; no deadlocks when all pass.

## 16) Tuning & Adaptation

- Heuristic weights
  - Provide sensible defaults; expose in manifest; allow per‑agent overrides for persona styles.
  - Instrument key KPIs (talk‑time share, interrupts, interjections, queue depth) and log to metrics.
  - Auto‑tune via simple bandit/grid search offline; or adaptive small adjustments at runtime within safe bounds.
- Sensitivity audits
  - Run A/B with different weights; evaluate perceived naturalness and fairness via judge agent or human ratings.
- Property tests: no negative balances; no more than allowed interjections; progress always occurs.
- Performance tests: concurrency under N agents; deadlines met with reserve sentence.

## 15) Rollout Plan

Phase 1 (MVP)
- ElevenLabs integration as default TTS provider.
- Interrupts: adopt hard‑interrupt cutoff as default (200 ms crossfade; prebuffer top‑1 kicker). Keep beat‑only as a toggle for conservative sessions.
- Feasibility checks for ElevenLabs streaming and timing support. If unavailable, enable graceful degradation (sequenced insertion, no overlap).
- Heuristic bidding; segment packs; typed interjections (support/clarify/challenge); basic emotion tracking; CLI flags; tests.

Phase 2
- Emotion-driven prosody; dynamic pricing and fairness tax; reputation staking rebates; evidence-priority lane.

Phase 3 (Optional)
- LLM-driven bidding JSON; moderator policies; alliances; thread support; richer UI/telemetry.

### 15.1 Minimal 0.2 Deltas (from external review)

1) Hard‑interrupt cutoff default (crossfade + prebuffered kicker). Dual‑path segments optional for beat‑only mode.
2) Measured beats with monotonic timestamps and drift_ms recorded.
3) Paced first‑price with EMA pacing multiplier; softmax desire; explicit interrupt budgets.
4) One‑segment reserve for next speaker; fall back to no‑interrupt if auction misses guard band.
5) Deterministic IDs in metadata (segment/clip/auction).
6) Single accrual mode per session + token decay if wall‑clock accrual is enabled.
7) Hysteresis on mood + hard cap on its influence over bids (±1 token equiv).

Security & privacy:
- Keep mood vectors session‑scoped; exclude from analytics by default. If persisted, aggregate and salt; never log raw per‑turn mood without explicit enablement.
- Sanitize interjections: strip SSML, quotes, and code fences to prevent providers from treating injected text as control marks.

## 16) Risks & Mitigations

- Latency spikes: fallback to reserve sentence; defer interrupt to next beat; prefer heuristic bids.
- Cost creep: cap interjections per segment; cap LLM bid requests/min; heuristic default.
- Audio artifacts: enforce beat-only barge points; crossfade/ducking; adjust SSML breaks.
- Dominance: progressive tax; per-agent interrupt quotas; cool-downs.

## 17) Reference Defaults (Initial)

- segment_seconds: min=3, target=5, max=8
- beats_per_segment: 2
- token_accrual: +1 per segment or +1/30s (choose one)
- max_bank: 8
- interjection_cost: 2, max_per_segment: 1, cooldown: 2 segments
- kicker_delta: +2, kicker_fee: +1
- fairness_target_share: 1 / num_agents, tax_rate: 0.15
- emotion_decay: frustration*0.95, engagement*0.98 per micro-turn

---

This plan is intended to be implementation-ready with additive modules and clear integration points. See sections 9–11 for immediate scaffolding details and config.
Kicker Prompt (hard‑interrupt cutoff):

"Produce one sentence, 8–12 words, that clearly interrupts and reframes. No filler. Output only the sentence."

Follow‑on Prompt (next speaker after interrupt):

"Respond directly to the last interruption in 2 short sentences. Do not repeat what was already said."
