"""Auction-based group chat orchestrator with concurrency scaffolding.

This orchestrator coordinates micro-turns using an audio controller and the
auction chat manager. It supports concurrent planning tasks (e.g.,
interjections) and provides a guard-banded, hard-interrupt cutoff path that can
be enabled per session. The hard-interrupt path is conservative by default and
is only engaged when explicitly configured.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Sequence
from typing import Any, Callable

from argentum.agents.base import Agent
from argentum.audio import AudioController, NoOpAudioController
from argentum.coordination.auction_manager import AuctionChatManager
from argentum.memory.context import Context
from argentum.models import (
    OrchestrationPattern,
    OrchestrationResult,
    Task,
    TerminationReason,
    Message,
    MessageType,
)
from argentum.orchestration.base import Orchestrator


class AuctionGroupChatOrchestrator(Orchestrator):
    def __init__(
        self,
        manager: AuctionChatManager,
        audio: AudioController | None = None,
        *,
        micro_turns: int = 2,
        enable_interjections: bool = True,
        enable_interrupts: bool = False,
        global_concurrency_limit: int = 4,
        per_agent_inflight_limit: int = 1,
        interjection_min_importance: float = 0.5,
        emit_provisional_interrupt: bool = False,
    ) -> None:
        self.manager = manager
        self.audio = audio or NoOpAudioController()
        self.micro_turns = max(1, micro_turns)
        self.enable_interjections = enable_interjections
        self.enable_interrupts = enable_interrupts
        self._session_started_ns = time.monotonic_ns()
        self._global_sem = asyncio.Semaphore(max(1, global_concurrency_limit))
        self._per_agent_inflight: dict[str, int] = {}
        self._per_agent_inflight_limit = max(1, per_agent_inflight_limit)
        self._interj_min_importance = max(0.0, min(1.0, interjection_min_importance))
        self._emit_provisional_interrupt = bool(emit_provisional_interrupt)
        # talk-share tracking over a sliding window of recent segments
        self._talk_counts: dict[str, int] = {}
        self._talk_window: int = 20
        # reserve prefetch handle for next likely speaker
        self._reserve_task: asyncio.Task | None = None
        self._reserve_for: str | None = None
        self._reserve_response = None

    async def execute(
        self,
        agents: Sequence[Agent],
        task: Task | str,
        context: Context | None = None,
    ) -> OrchestrationResult:
        task_obj = self._prepare_task(task)
        ctx = self._prepare_context(context)
        # initial task message
        ctx.add_message(Message(type=MessageType.USER, sender="orchestrator", content=task_obj.description))

        # ensure state entries
        for a in agents:
            self.manager.get_or_create(a.name)
        # Keep resolver state in sync with manager state if supported
        try:
            if getattr(self.manager.resolver, "state", None) is not None:
                self.manager.resolver.state = self.manager.state  # type: ignore[attr-defined]
        except Exception:
            pass

        final_outputs = []
        context_version = 0

        # micro-turn loop
        for turn in range(self.micro_turns):
            context_version += 1
            # publish for invalidation checks in concurrent tasks
            self._current_context_version = context_version
            auction_id = f"auction_{self._short_session_id()}_{turn:04d}"
            # accrue tokens per segment
            self.manager.token_manager.accrue(self.manager.state)
            # advance interjection planner window if supported
            with contextlib.suppress(Exception):
                getattr(self.manager.interjections, "start_new_segment", lambda: None)()

            agent_names = [a.name for a in agents]
            # collect bids (speaker selection)
            bids = await self.manager.resolver.collect_bids(agent_names, context={})
            winner_name, _ = self.manager.resolver.resolve(bids)
            speaker = next((a for a in agents if a.name == winner_name), agents[0])

            # generate speaker response (reuse reserve if available for this speaker)
            if getattr(self, "_reserve_for", None) == speaker.name and getattr(self, "_reserve_response", None) is not None:
                response = self._reserve_response
                # clear reserve after use
                self._reserve_for = None
                self._reserve_response = None
                if getattr(self, "_reserve_task", None) is not None and not self._reserve_task.done():
                    self._reserve_task.cancel()
                    import contextlib as _ctx
                    with _ctx.suppress(asyncio.CancelledError):
                        await self._reserve_task
                self._reserve_task = None
            else:
                response = await speaker.generate_response(messages=ctx.get_messages(), context=task_obj.context)
            # optional mood parsing from content
            cleaned, mood = self._extract_mood_block(response.content or "")
            if mood:
                with contextlib.suppress(Exception):
                    self.manager.emotion_engine.update_from_llm(speaker.name, mood)
                response.content = cleaned

            # drive TTS playback (simulated) and interjection planning
            text = response.content or "One. Two."
            segment_id = f"seg_{auction_id}_{self._agent_index(agents, speaker.name)}"
            clip_id = f"clip_{segment_id}"
            handle = await self.audio.play(text, clip_id=clip_id)

            # Concurrency scaffolding: plan interjections and (optionally) interrupts
            interjection_task: asyncio.Task | None = None
            if self.enable_interjections:
                interjection_task = asyncio.create_task(
                    self._plan_and_maybe_duck_interjection(
                        ctx,
                        current_version=context_version,
                        speaker=speaker,
                        others=[a for a in agents if a.name != speaker.name],
                        handle=handle,
                        auction_id=auction_id,
                    )
                )

            interrupt_task: asyncio.Task | None = None
            if self.enable_interrupts:
                interrupt_task = asyncio.create_task(
                    self._maybe_hard_interrupt(
                        ctx,
                        current_version=context_version,
                        speaker=speaker,
                        agents=agents,
                        handle=handle,
                        auction_id=auction_id,
                    )
                )

            # Reserve prefetch for likely next speaker (second-highest bid)
            if self._reserve_task and not self._reserve_task.done():
                self._reserve_task.cancel()
                import contextlib as _ctx
                with _ctx.suppress(asyncio.CancelledError):
                    await self._reserve_task
            next_candidate = None
            try:
                ranked = sorted(((n, b.get("amount", 0)) for n, b in (bids or {}).items()), key=lambda t: (-t[1], t[0]))
                ranked = [n for n, _ in ranked if n != speaker.name]
                if ranked:
                    next_candidate = ranked[0]
            except Exception:
                next_candidate = None
            self._reserve_for = None
            self._reserve_response = None
            if next_candidate:
                cand_agent = next((a for a in agents if a.name == next_candidate), None)
                if cand_agent is not None:
                    async def _prefetch():
                        try:
                            async with self._agent_slot(cand_agent.name):
                                resp = await cand_agent.generate_response(messages=ctx.get_messages(), context=task_obj.context)
                                if getattr(self, "_current_context_version", context_version) == context_version:
                                    self._reserve_for = cand_agent.name
                                    self._reserve_response = resp
                        except Exception:
                            return
                    self._reserve_task = asyncio.create_task(_prefetch())

            # Wait for either an interrupt commit or normal finish
            finish_task = asyncio.create_task(handle.finish())
            winner_interrupt: dict[str, Any] | None = None
            if interrupt_task is not None:
                done, pending = await asyncio.wait({finish_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED)
                if interrupt_task in done:
                    try:
                        winner_interrupt = interrupt_task.result()
                    except Exception:
                        winner_interrupt = None
                # If no confirmed interrupt, ensure segment finishes normally
                if not (winner_interrupt and winner_interrupt.get("interrupted")):
                    if not finish_task.done():
                        try:
                            await finish_task
                        except asyncio.CancelledError:
                            pass
                else:
                    # Interrupt landed: cancel the finish waiter to cut current segment
                    if not finish_task.done():
                        finish_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await finish_task
            else:
                # No interrupts: finish playback normally
                try:
                    await finish_task
                except asyncio.CancelledError:
                    pass

            # Cancel any pending concurrent work for this turn
            for t in (interjection_task, interrupt_task):
                if t and not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

            # add messages based on path taken
            if winner_interrupt and winner_interrupt.get("interrupted"):
                # Record the interrupter's kicker as an event
                ctx.add_message(
                    Message(
                        type=MessageType.ASSISTANT,
                        sender=str(winner_interrupt.get("interrupter")),
                        content=str(winner_interrupt.get("text", "")),
                        metadata={
                            "event": "hard_interrupt",
                            "auction_id": auction_id,
                            "segment_id": winner_interrupt.get("segment_id"),
                            "clip_id": winner_interrupt.get("clip_id"),
                            "timing": winner_interrupt.get("timing"),
                            "costs": winner_interrupt.get("costs"),
                            "auction": winner_interrupt.get("auction"),
                            "policy": winner_interrupt.get("policy"),
                        },
                    )
                )
                # Emotion nudges
                import contextlib as _ctxlib
                with _ctxlib.suppress(Exception):
                    self.manager.emotion_engine.nudge_from_event(str(winner_interrupt.get("interrupter")), "won_interrupt")
                    self.manager.emotion_engine.nudge_from_event(speaker.name, "was_interrupted")
                # Do not append speaker's cut segment to outputs; proceed
            else:
                # Update talk-share window and pacing prior to logging the segment
                self._talk_counts[getattr(speaker, 'name', str(speaker))] = self._talk_counts.get(getattr(speaker, 'name', str(speaker)), 0) + 1
                if sum(self._talk_counts.values()) > getattr(self, '_talk_window', 20):
                    for _k in list(self._talk_counts.keys()):
                        self._talk_counts[_k] = max(0, self._talk_counts[_k] - 1)
                _talk_total = sum(self._talk_counts.values()) or 1
                _talk_share = {n: (self._talk_counts.get(n, 0) / _talk_total) for n in [a.name for a in agents]}
                _upd = getattr(self.manager.resolver, 'update_pacing', None)
                if callable(_upd):
                    try:
                        _upd(_talk_share)
                    except Exception:
                        pass
                ctx.add_message(
                    Message(
                        type=MessageType.ASSISTANT,
                        sender=speaker.name,
                        content=response.content,
                        metadata={
                            "event": "segment",
                            "auction_id": auction_id,
                            "segment_id": segment_id,
                            "clip_id": clip_id,
                            "timing": {
                                "drift_ms": getattr(handle, "drift_ms", lambda: [])(),
                                "commit_guard_ms": getattr(self.audio, "compute_commit_guard_ms", lambda: 0)(),
                                "p95_first_chunk_ms": getattr(self.audio, "first_chunk_p95_ms", lambda: 0)(),
                            },
                            "auction": {"bids": {k: int(v.get("amount", 0)) for k, v in (bids or {}).items()} if isinstance(bids, dict) else {}, "pacing": getattr(self.manager.resolver, "pacing", None), "talk_share": _talk_share},
                        },
                    )
                )
                final_outputs.append(response)
                # Emotion nudge for speaker having completed a segment
                import contextlib as _ctxlib
                with _ctxlib.suppress(Exception):
                    self.manager.emotion_engine.nudge_from_event(speaker.name, "spoke_segment")

        # simple consensus: last response
        consensus = final_outputs[-1].content if final_outputs else None
        return OrchestrationResult(
            pattern=OrchestrationPattern.GROUP_CHAT,
            messages=ctx.get_messages(),
            final_outputs=final_outputs,
            consensus=consensus,
            termination_reason=TerminationReason.MAX_TURNS_REACHED,
            metadata={"num_agents": len(agents), "agent_names": [a.name for a in agents]},
            duration_seconds=0.0,
        )

    # -------------------------- helpers ---------------------------------
    # concurrency helper
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _agent_slot(self, agent: str):
        await self._global_sem.acquire()
        try:
            cnt = self._per_agent_inflight.get(agent, 0)
            if cnt >= self._per_agent_inflight_limit:
                await asyncio.sleep(0)
            self._per_agent_inflight[agent] = cnt + 1
            yield
        finally:
            self._per_agent_inflight[agent] = max(0, self._per_agent_inflight.get(agent, 1) - 1)
            self._global_sem.release()

    def _short_session_id(self) -> str:
        ns = self._session_started_ns
        return f"{ns & 0xFFFF:X}"

    def _agent_index(self, agents: Sequence[Agent], name: str) -> int:
        for i, a in enumerate(agents):
            if a.name == name:
                return i
        return 0

    async def _plan_and_maybe_duck_interjection(
        self,
        ctx: Context,
        *,
        current_version: int,
        speaker: Agent,
        others: Sequence[Agent],
        handle: Any,
        auction_id: str,
    ) -> None:
        # Wait near first beat; tolerate missing planner
        try:
            await handle.wait_for_beat(1)
        except Exception:  # pragma: no cover - defensive
            pass
        if not others:
            return
        intr = others[0]
        # Fallback simple interjection text if no planner is provided
        plan_func: Callable[..., Any] | None = getattr(self.manager.interjections, "plan", None)  # type: ignore[attr-defined]
        text: str
        itype: str | None = None
        importance: float = 0.7
        if callable(plan_func):
            try:
                plans = await plan_func([a.name for a in others], context={})
                if plans:
                    plan = plans[0]
                    text = str(plan.get("text") or f"Interjection by {intr.name}")
                    itype = plan.get("type")
                    try:
                        importance = float(plan.get("importance", importance))
                    except Exception:
                        pass
                    pname = plan.get("agent")
                    if isinstance(pname, str):
                        for a in others:
                            if a.name == pname:
                                intr = a
                                break
                else:
                    text = f"Interjection by {intr.name}"
            except Exception:  # pragma: no cover - planner optional
                text = f"Interjection by {intr.name}"
        else:
            text = f"Interjection by {intr.name}"

        # Respect minimum importance threshold
        if importance < self._interj_min_importance:
            return

        # Ensure still current before committing
        if getattr(self, "_current_context_version", current_version) != current_version:
            return

        # Sanitize and duck/play a short clip
        safe = _sanitize_interjection(text)
        try:
            await self.audio.duck_and_play(safe, duration_ms=600)
        except Exception:  # pragma: no cover - defensive
            return

        ctx.add_message(
            Message(
                type=MessageType.ASSISTANT,
                sender=intr.name,
                content=safe,
                metadata={
                    "event": "interjection",
                    "auction_id": auction_id,
                    "interjection": {"type": itype, "importance": importance},
                },
            )
        )

    async def _maybe_hard_interrupt(
        self,
        ctx: Context,
        *,
        current_version: int,
        speaker: Agent,
        agents: Sequence[Agent],
        handle: Any,
        auction_id: str,
    ) -> None:
        # Determine guard band if available from audio, else default
        guard_ms = 150
        compute_guard = getattr(self.audio, "compute_commit_guard_ms", None)
        if callable(compute_guard):
            try:
                guard_ms = int(compute_guard())
            except Exception:  # pragma: no cover - fallback
                guard_ms = 150

        # Select a candidate interrupter (simple heuristic): first other agent with tokens
        others = [a for a in agents if a.name != speaker.name]
        if not others:
            return None
        # Ask resolver to score candidates for interrupt (reuse heuristic bids)
        names = [a.name for a in others]
        try:
            bids = await self.manager.resolver.collect_bids(names, context={"mode": "interrupt"})
        except Exception:
            bids = {}
        # Choose highest positive bid among eligible
        ranked = sorted(
            [(n, b.get("amount", 0)) for n, b in (bids or {}).items() if self.manager.interrupt_policy.can_interrupt(n)],
            key=lambda t: (-t[1], t[0]),
        )
        candidate = None
        for name, amt in ranked:
            st = self.manager.state.get(name)
            if st and st.tokens > 0 and amt > 0:
                candidate = next(a for a in others if a.name == name)
                break
        if candidate is None:
            # Fallback: pick first eligible by tokens/policy
            for a in others:
                st = self.manager.state.get(a.name)
                if st and st.tokens > 0 and self.manager.interrupt_policy.can_interrupt(a.name):
                    candidate = a
                    break
        if candidate is None:
            return None

        # Optionally emit a provisional (non-audio) interrupt event for observability
        if getattr(self, "_emit_provisional_interrupt", False):
            try:
                bids_meta = {k: int(v.get("amount", 0)) for k, v in (bids or {}).items()} if isinstance(bids, dict) else {}
            except Exception:
                bids_meta = {}
            ctx.add_message(
                Message(
                    type=MessageType.ASSISTANT,
                    sender=str(candidate.name),
                    content="",
                    metadata={
                        "event": "interrupt_provisional",
                        "auction_id": auction_id,
                        "interrupter": candidate.name,
                        "timing": {
                            "guard_ms": guard_ms,
                            "beat": 1,
                            "p95_first_chunk_ms": getattr(self.audio, "first_chunk_p95_ms", lambda: 0)(),
                            "phase": "provisional",
                        },
                        "auction": {"mode": "interrupt", "bids": bids_meta, "candidate": candidate.name},
                    },
                )
            )

        # Prefetch a short kicker line
        kicker_prompt = "Produce one short interruption sentence (8-12 words). No SSML."
        try:
            async with self._agent_slot(candidate.name):
                coro = candidate.generate_response(
                    messages=ctx.get_messages() + [
                        Message(type=MessageType.SYSTEM, sender="orchestrator", content=kicker_prompt)
                    ],
                    context=None,
                )
                # Readiness timeout bounded by guard band
                import asyncio as _asyncio
                timeout_s = max(0.05, guard_ms / 1000.0)
                kicker_resp = await _asyncio.wait_for(coro, timeout=timeout_s)
        except Exception:
            return None
        kicker_text = (kicker_resp.content or "Excuse me, I need to jump in here.").strip()

        # Wait for first beat (commit point) and check we can still interrupt
        try:
            await handle.wait_for_beat(1)
        except Exception:
            return None
        if not getattr(handle, "can_interrupt", lambda: True)():
            return None

        # Validate context still current before committing
        if getattr(self, "_current_context_version", current_version) != current_version:
            return None

        # Commit: fade current speaker, then play the kicker
        try:
            await self.audio.crossfade_to_silence(duration_ms=200)
        except Exception:
            return None
        intr_segment_id = f"seg_{auction_id}_intr"
        intr_clip_id = f"clip_{intr_segment_id}"
        async def _do_commit():
            try:
                await self.audio.crossfade_to_silence(duration_ms=200)
            except Exception:
                return
            try:
                await self.audio.play(kicker_text, clip_id=intr_clip_id)
            except Exception:
                return
        import asyncio as _asyncio
        _asyncio.create_task(_do_commit())

        # Record budget usage (charge tokens, record policy usage)
        self.manager.interrupt_policy.record_interrupt(candidate.name)

        stc = self.manager.state.get(candidate.name)
        amount_bid = 0
        if isinstance(bids, dict) and candidate.name in bids:
            amount_bid = int(bids[candidate.name].get("amount", 0))
        spend = min(stc.tokens if stc else 0, max(1, amount_bid))
        import contextlib as _ctx
        with _ctx.suppress(Exception):
            self.manager.token_manager.charge(candidate.name, spend)
        if stc:
            stc.tokens = max(0, stc.tokens - spend)

        return {
            "interrupted": True,
            "interrupter": candidate.name,
            "text": kicker_text,
            "segment_id": intr_segment_id,
            "clip_id": intr_clip_id,
            "timing": {"guard_ms": guard_ms, "beat": 1, "p95_first_chunk_ms": getattr(self.audio, "first_chunk_p95_ms", lambda: 0)(), "phase": "confirmed"},
            "costs": {"tokens_spent": spend},
            "auction": {"mode": "interrupt", "bid": amount_bid, "result": "win", "provisional": {"candidate": candidate.name}, "bids": ({k: int(v.get("amount", 0)) for k, v in (bids or {}).items()} if isinstance(bids, dict) else {})},
            "policy": type(self.manager.interrupt_policy).__name__,
        }


    def _extract_mood_block(self, text: str) -> tuple[str, dict[str, float] | None]:
        """Extract a trailing <<<MOOD>>> JSON block or inline JSON and strip it.

        Returns: (cleaned_text, mood_dict | None)
        """
        if not text:
            return text, None
        mood_pat = re.compile(r"<<<MOOD>>>\s*(\{.*?\})\s*$", re.DOTALL)
        m = mood_pat.search(text)
        blob = None
        cleaned = text
        if m:
            blob = m.group(1)
            cleaned = text[: m.start()].rstrip()
        else:
            # fallback: any trailing JSON object with frustration key
            m2 = re.search(r"(\{\s*\"frustration\".*?\})\s*$", text, re.DOTALL)
            if m2:
                blob = m2.group(1)
                cleaned = text[: m2.start()].rstrip()
        if not blob:
            return cleaned, None
        try:
            data = json.loads(blob)
        except Exception:
            return cleaned, None
        out = {}
        for k in ("frustration", "engagement", "confidence"):
            if k in data:
                try:
                    out[k] = float(data[k])
                except Exception:
                    pass
        return cleaned, (out or None)

def _sanitize_interjection(text: str) -> str:
    # Strip quotes, code fences, and SSML-like tags to keep audio safe
    s = text.strip()
    # Remove backticks and quotes
    s = s.replace("`", "").replace("\"", "").replace("'", "")
    # Remove any angle-bracketed tags
    out = []
    skip = 0
    for ch in s:
        if ch == "<":
            skip += 1
            continue
        if ch == ">" and skip > 0:
            skip -= 1
            continue
        if skip == 0:
            out.append(ch)
    s2 = "".join(out)
    return s2[:140]
