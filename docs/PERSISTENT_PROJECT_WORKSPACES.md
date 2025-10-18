# Persistent Project Workspaces & Memory Architecture

This document captures the conceptual design for making Argentum’s multi-agent ensembles feel like persistent casts—akin to a long-running podcast, advisory board, or VTuber persona. It builds on Phase 1 persistence work and paves the way for Phase 2/3 memory features by defining how we organise projects, store knowledge at different time horizons, and expose that history back to agents and humans.

## Goals & Representative Use Cases
- **AI Podcast / Variety Stream:** Maintain continuity between episodes so hosts remember recurring jokes, outstanding promises, and guest history. Comparable to Neuro-sama’s persistent persona where fans expect long-term memory continuity.
- **Executive Advisory Board:** Track strategic decisions, follow-ups, and rationales across quarters, enabling advisors to reference previous recommendations without restating context.
- **Knitting Circle / Community Panel:** Capture evolving culture, unresolved projects, and member-specific preferences (e.g., favoured patterns, ongoing collaborations).

Across all use cases we need:
1. **Project-level organisation** that groups sessions, cast members, and assets.
2. **Tiered memory** so agents access the right amount of history without dragging cold archives into their prompt.
3. **Human-friendly tooling** to browse transcripts, summaries, and logs per project.

## Project Workspace Model

Each long-lived show becomes a project living under `<repo-root>/workspace/<slug>/` (configurable). Keeping these directories beside the source package avoids polluting `argentum/` while giving contributors a predictable location inside the project tree. A workspace contains:

```
project.json                 # manifest (title, description, owners, defaults)
agents/
  <agent-id>/profile.json    # AgentConfig, persona, voice info
  <agent-id>/memory.json     # warm cache snapshot (if we want direct export)
sessions/
  <session-id>/
    transcript.json          # full ConversationSession payload
    summary.md               # human-readable recap
    highlights.json          # structured warm-cache entries
    artifacts/               # optional exports (audio, images)
timeline.jsonl               # append-only index of sessions & major events
cache/
  hot/                       # rolling prompt summaries, open action items
  warm/                      # vector index / SQLite FTS for episodic memories
  cold/                      # compressed archives, zipped transcripts
```

### Project Manifest (`project.json`)
```json
{
  "project_id": "knit-cast",
  "display_name": "The AI Knitting Circle",
  "description": "Weekly talk show following the adventures of four textile-obsessed agents.",
  "default_agents": ["patch", "loom", "hook", "spindle"],
  "storage": {
    "base_path": "./workspace/knit-cast",
    "vector_store": "chromadb",
    "archive_format": "zip"
  },
  "retention": {
    "hot_context_turns": 12,
    "warm_window_days": 120,
    "archive_after_days": 365
  },
  "summary": {
    "mode": "heuristic",
    "command": null
  },
  "speech": {
    "default_style": "boardroom",
    "default_tags": ["measured", "formal"],
    "default_voice": "boardroom_male",
    "overrides": {
      "environment": {
        "style": "podcast",
        "tags": ["warm", "optimistic"],
        "tts_voice": "podcast_female"
      }
    }
  }
}
```

The manifest provides defaults to CLI commands (project description, agent roster, storage connectors) and encodes retention rules that pruning jobs can follow.

### Agent Profiles (`agents/<id>/profile.json`)
- Persist `AgentConfig`, persona/backstory, speaking style, instrumentation flags, and memory limits.
- Link each profile with an `AgentMemoryStore` directory (Phase 2) so cast members have long-lived personal histories.

## Memory Temperature Layers

Industry guidance consistently recommends multi-layer memory to balance latency, cost, and completeness.[^pcg][^hypermode][^nimdinu]

| Layer | Content | Storage | Access Pattern | Expiry |
|-------|---------|---------|----------------|--------|
| **Hot** | Current conversation window (last N turns), rolling summary, open action items, short-term working notes per agent | In-memory context wrapper + lightweight JSON under `cache/hot/` | Synchronously injected into prompts (sub-10 ms) | Cleared at end of session; summary persisted downwards |
| **Warm** | Episodic snippets (highlights, decisions, quotes), agent-specific memories (stances, catchphrases), retrieval history, vector chunks for recent 120 days | File-backed document index under `knowledge/docs` (JSONL + embeddings) and SQLite FTS for highlights | Prefetched before sessions; agents can trigger additional retrievals mid-conversation | Summarise to meta entries after retention window, demote to cold |
| **Cold** | Full transcripts, raw tool traces, legacy sessions | Compressed JSONL / parquet archives in object storage or filesystem | Not used in prompt; retrieved via CLI search or explicit tool call | Never deleted without manual action; can be offloaded to cheaper storage |

### Retrieval Flow
1. **Session boot:** Load agent profiles, pull hot cache skeleton (recent summary + TODOs), and prefetch the top-k workspace documents that match the topic/seed. The results are labeled `[Doc n]`, cached in `retrieved_docs`, and stamped onto the retrieval preamble.
2. **During conversation:** Any agent can fetch more evidence by emitting `<<retrieve: keywords>>`. The shared `SessionRetriever` performs the search, appends new labeled chunks to the context, records the event in `retrieval_history`, and replays the turn so the agent can respond with citations.
3. **Session end:** Highlights capture `citations`, `retrieval_history`, and the labeled chunks; summaries automatically append `Sources: Doc 1, Doc 2` so humans can audit provenance. Warm cache, knowledge graph, and timeline entries are refreshed in the same pass.

### Speech & TTS Profiles
- Define global defaults under `speech` in `project.json` (style, descriptive tags, default TTS voice).
- Provide per-agent overrides in `speech.overrides` so voices can vary by role.
- `build_tts_markdown` and `build_tts_script` embed tone metadata and preferred voices alongside the conversation, making it easy to format for podcasts, boardroom briefings, or casual hangouts without reprocessing the raw transcript.

### Document Knowledge Workflow
- **Ingest:** `argentum project docs ingest <slug> <files...> [--tag --chunk-size --overlap]` writes full document metadata and chunk vectors to `knowledge/docs/documents.jsonl` & `chunks.jsonl`.
- **Explore:** `project docs list/search` exposes titles, tags, scores, and chunk positions so humans can validate what is indexed.
- **Maintenance:** `project docs purge` deletes a document (and trims feedback), while `project docs rebuild` re-chunks sources from disk—helpful after editing markdown or adjusting chunk parameters.
- **Feedback:** `project docs feedback` records post-session usefulness ratings; these land in `knowledge/docs/feedback.jsonl` for downstream tuning.
- **Transparency:** Each session’s `highlights.json` now includes the retrieval history, making it easy to audit what queries ran and which labels were delivered to the cast.

## Session Lifecycle & Timelines

Each session inherits defaults from the project and may accept overrides:
```bash
argentum session run \
  --project knit-cast \
  --topic "Episode 47 – The Lace Disaster" \
  --session-id 2025-10-16-lace
```
Steps:
1. CLI resolves project manifest and ensures directories exist.
2. Loads agent profiles + warm cache glimpses, constructs `ConversationSession` pointing at `sessions/<id>/transcript.json`.
3. Runs orchestrator scenario (debate/podcast) with agent-specific providers and memory injection.
4. After completion:
   - Writes transcript.
   - Calls summariser to generate `summary.md` and `highlights.json`.
   - Updates hot cache (store outstanding cliffhangers).
   - Records entry in `timeline.jsonl`:
     ```json
     {"session_id":"2025-10-16-lace","title":"Episode 47 – The Lace Disaster","agents":["patch","loom","hook","spindle"],"duration":1860,"topics":["lace","suppliers"],"consensus":"Delay launch by two weeks"}
     ```

## Tooling & CLI Extensions

### Project Commands
- `argentum project init --slug knit-cast --title "AI Knitting Circle"` → scaffolds workspace, prompts for default agents.
- `argentum project info knit-cast` → prints manifest, retention policies, last sessions, available agents.
- `argentum project compact knit-cast` → runs summarisation + pruning job (see Policies below).
- `argentum project knowledge knit-cast --topic "carbon tax"` → list sessions discussing a topic (uses knowledge graph).
- `argentum project knowledge knit-cast --agent "Agent 1" --limit 5` → show sessions/statements for an agent.
- `argentum project knowledge knit-cast --search "increase"` → query warm cache highlights via FTS.
- Add a `summary` section to `project.json` to control highlight summarisation:
  ```json
  {
    "summary": {
      "mode": "frontier",
      "command": null
    }
  }
  ```
  Supported modes: `heuristic` (default), `frontier` (OpenAI-compatible, requires `OPENAI_API_KEY`), `local` (executes `command` via shell), `none` (disable summarisation).

### Session Commands
- `argentum session run --project <slug> …` → wraps existing scenarios (debate, advisory, custom) with persistence.
- `argentum session recap --project <slug> --session <id>` → prints or exports summary.
- `argentum session search --project <slug> --query "carbon tax"` → runs warm (vector) search first, then cold (FTS) as fallback.

### Agent Utilities
- `argentum agent profile --project <slug> --agent <id>` → inspect persona, recent memories, sentiment markers.
- `argentum agent remember --project <slug> --agent <id> --content "..."` → manual memory entry for backstage adjustments.

## Pruning & Compaction Strategy

Goal: keep hot/warm stores lean while preserving long-term continuity.

1. **Hot → Warm:** After each session, roll the final conversation summary + TODO list into warm store entries and clear hot caches.
2. **Warm aging:** Nightly job checks entry age; after `warm_window_days` (default 120), merge related entries into a consolidated “season summary” and demote raw text to cold archive only.
3. **Cold maintenance:** Optionally compress transcripts older than `archive_after_days` into zipped batches per quarter; maintain FTS index to support `QueryArchive`.
4. **Manual overrides:** Provide `argentum project retain --session <id>` to pin episodes (no summarisation) and `argentum project purge --session <id>` for compliance removal (with audit log).

## Storage Technology Choices

| Layer | Recommended Tech | Notes |
|-------|------------------|-------|
| Hot | In-memory structures + JSON snapshot | Managed by orchestrator runtime; can sit in Redis if scaling out. |
| Warm (semantic) | ChromaDB / Milvus / Weaviate | Aligns with RAG plan; store embeddings of highlights and decisions. |
| Warm (structured) | SQLite (FTS5) or DuckDB | Index session metadata, quotes, TODOs for fast filter/search. |
| Agent Memory | Existing `AgentMemoryStore` (JSON) initially; optionally migrate to SQLite for concurrency. |
| Cold | Parquet/JSONL on disk or S3-compatible bucket | Keep zipped transcripts + tool logs; “cold” search uses FTS over aggregated indexes. |

The combination mirrors industry advice on tiered storage for agentic systems.[^monetizely]

## Status & Next Steps

### Completed
- Project workspace scaffolding, manifests, and CLI orchestration.
- Persistent session pipeline (transcript, highlights, timeline) with warm-cache indexing.
- Workspace document index + CLI (`project docs`) for ingestion, search, purge, rebuild, and feedback.
- Automatic prefetch and mid-session retrieval (`<<retrieve: …>>`) with labeled citations in transcripts and highlights.
- Retrieval history surfaced in session metadata so humans can audit provenance.

### Upcoming
1. Cold archive/query tooling (`project knowledge --archive`, background compaction jobs) for long-tail transcripts.
2. Higher-level retrieval APIs (role-scoped recall, feedback-aware reranking) and dashboard visualisations.
3. Optional remote/object storage backends for distributed deployments.
4. Operator guardrails (rate limits, moderation policies) once multi-tenant use cases appear.

## Open Design Questions
- **Summarisation Engine:** Should we rely on on-demand LLM calls or deterministic heuristics for highlight extraction? We’ll likely need a hybrid to manage cost and determinism.
- **Provider Mix:** How do we handle agents that use different LLM providers/models per session without duplicating warm cache embeddings? (Plan: store provider metadata per entry for filtering.)
- **Real-time updates:** For livestream-like experiences, do we persist mid-session (streaming) or wait until the session ends? Progressive writes would enable live dashboards but require idempotent append operations.
- **Access control:** Multi-project setups may require permissions; future `project.json` could include ACL lists or token requirements.

---

## References

[^pcg]: Madhura Raj, “Progressive Context Generation: A Multi-Layer Architecture for Persistent AI Agent Memory,” *Medium* (Oct 3, 2025). Highlights multi-layer memory (hot/warm/cold/semantic) and progressive assembly.
[^hypermode]: Hypermode Engineering, “How to optimize AI agent performance for real-time processing,” *Hypermode Blog* (May 1, 2025). Discusses hot vs cold context management and tiered storage for low latency.
[^monetizely]: “How to Design Databases for Agentic AI: Best Practices for Storing Knowledge and State,” *Monetizely* (Aug 30, 2025). Recommends multi-tier storage, event sourcing, and hybrid indexing for agent knowledge.
[^nimdinu]: Nimeth Nimdinu, “Memory for AI Agents: Designing Persistent, Adaptive Memory Systems,” *Medium* (Oct 5, 2025). Provides taxonomy for short-term, episodic, semantic, procedural, and meta memories.
