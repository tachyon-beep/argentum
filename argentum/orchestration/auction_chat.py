"""Auction-based group chat orchestrator (skeleton).

This orchestrator coordinates micro-turns with an audio controller and the
auction chat manager. Implementation will be built out iteratively.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    ) -> None:
        self.manager = manager
        self.audio = audio or NoOpAudioController()
        self.micro_turns = max(1, micro_turns)

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

        final_outputs = []

        # micro-turn loop
        for turn in range(self.micro_turns):
            # accrue tokens per segment
            self.manager.token_manager.accrue(self.manager.state)

            agent_names = [a.name for a in agents]
            bids = await self.manager.resolver.collect_bids(agent_names, context={})
            winner_name, _ = self.manager.resolver.resolve(bids)
            speaker = next((a for a in agents if a.name == winner_name), agents[0])

            # generate speaker response
            response = await speaker.generate_response(messages=ctx.get_messages(), context=task_obj.context)

            # drive TTS playback (simulated) and interjection planning
            text = response.content or "One. Two."
            handle = await self.audio.play(text, clip_id=f"seg_t{turn}")
            try:
                await handle.wait_for_beat(1)
            except Exception:  # pragma: no cover - defensive
                pass

            # simple interjection from the next agent, if any
            others = [a for a in agents if a.name != speaker.name]
            if others:
                intr = others[0]
                interj_text = f"Interjection by {intr.name}"
                ctx.add_message(
                    Message(
                        type=MessageType.ASSISTANT,
                        sender=intr.name,
                        content=interj_text,
                        metadata={"event": "interjection"},
                    )
                )

            try:
                await handle.finish()
            except Exception:  # pragma: no cover - defensive
                pass

            # add speaker's segment to context
            ctx.add_message(
                Message(
                    type=MessageType.ASSISTANT,
                    sender=speaker.name,
                    content=response.content,
                    metadata={"event": "segment"},
                )
            )
            final_outputs.append(response)

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
