"""The front desk: what happens to one patient message.

    1. emergency check (regex + Haiku). If it fires, a fixed reply: 108 or the nearest emergency room.
    2. router (Haiku): booking, billing, info or human; also the patient's language and name.
    3. the chosen agent runs, with its own history and the chat's shared state.

Code, not the model, decides the fields of the reply: `emergency` and `handoff` come from flags set by hooks and
tools (shared state), and a conversation that went wrong ends in a fixed handoff message, never in silence."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from agentcore.budget import Budget
from agentcore.session import InMemoryStore, SessionBusy, run_in_session
from agentcore.trace import Event, Tracer
from agentcore.types import RunContext

from .agents import build_agents, build_emergency_check, build_router_service
from .config import Settings
from .emergency import emergency_reply
from .hospital import HospitalClient
from .i18n import HANDOFF, TROUBLE, lang_of
from .schemas import ChatRequest, ChatResponse
from .state import flag_handoff, is_handoff
from .tools import Hospital

log = logging.getLogger("agent_service")

LOCK_TIMEOUT_S = 45.0
SPECIALISTS = ("booking", "billing", "info")
TRACE_KEYS = ("emergency", "router", "safety", "human", *SPECIALISTS)
DEFAULT_BUDGET = Budget(max_cost_usd=0.25, max_session_cost_usd=2.0, max_seconds=90.0)


@dataclass
class Conversation:
    """What the agents of one chat share. Each agent keeps its own message history (its own tools and
    prompt); this is the part they all see."""

    phone: str
    shared: dict = field(default_factory=dict)  # becomes ctx.state of every agent: phone, language, name, flags
    last_agent: str | None = None
    pending_agent: str | None = None  # the agent whose read-back is waiting for the patient's answer
    touched: str | None = None  # the agent that handled the current message, if any
    recent: deque = field(default_factory=lambda: deque(maxlen=6))  # (who, text), for the router and handovers

    @property
    def ctx(self) -> RunContext:
        """A context over the shared dict, so the state helpers (flag_handoff...) work on it."""
        return RunContext(key="", state=self.shared)


class SessionOwnedByAnotherPhone(Exception):
    """A session id was used with a different phone number than the one that started the chat."""


class FrontDesk:
    def __init__(
        self,
        client: Any,
        hospital: HospitalClient,
        settings: Settings | None = None,
        *,
        hospital_info: str | None = None,
        budget: Budget | None = DEFAULT_BUDGET,
        tracer: Tracer | None = None,
        max_conversations: int = 10_000,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.hospital_client = hospital
        self.tracer = tracer or Tracer()
        self.store = InMemoryStore()
        self.emergency = build_emergency_check(client, self.settings, tracer=self.tracer)
        self.router = build_router_service(client, self.settings, tracer=self.tracer)
        self.agents = build_agents(client, Hospital(hospital), self.settings, hospital_info=hospital_info,
                                   tracer=self.tracer, budget=budget)
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
        self._max_conversations = max_conversations

    async def aclose(self) -> None:
        await self.hospital_client.aclose()

    # ------------------------------------------------------------------ the public entry points

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Raises SessionOwnedByAnotherPhone if the session belongs to another phone number (the app answers
        403), SessionBusy if another message of the same chat is still being handled for too long."""
        conv = self._conversation(req)
        async with self.store.lock(req.session_id, LOCK_TIMEOUT_S):
            conv.shared["phone"] = req.phone  # always the verified number from the request
            conv.touched = None
            response = await self._respond(conv, req)
            await self._expire_unseen_confirmations(req.session_id, conv)
            conv.recent.append(("Patient", req.message[:300]))
            conv.recent.append(("Assistant", response.reply[:300]))
            return response

    def trace(self, session_id: str) -> list[dict]:
        """Every model and tool call of this chat, in time order."""
        traces = [t for name in TRACE_KEYS for t in self.tracer.traces(f"{session_id}::{name}")]
        traces += self.tracer.traces(session_id)  # the emergency check runs under the bare session id
        return [t.to_dict() for t in sorted(traces, key=lambda t: t.started_at)]

    # ------------------------------------------------------------------ one message

    def _conversation(self, req: ChatRequest) -> Conversation:
        conv = self._conversations.get(req.session_id)
        if conv is None:
            conv = self._conversations[req.session_id] = Conversation(phone=req.phone)
            while len(self._conversations) > self._max_conversations:
                self._conversations.popitem(last=False)
        elif conv.phone != req.phone:
            # A session id is not a secret: it must never open another patient's conversation.
            raise SessionOwnedByAnotherPhone("This session belongs to another phone number.")
        self._conversations.move_to_end(req.session_id)
        return conv

    async def _respond(self, conv: Conversation, req: ChatRequest) -> ChatResponse:
        shared, sid = conv.shared, req.session_id

        if shared.get("emergency"):  # once an emergency, always: nothing else is handled in this chat
            return self._safety(conv, sid, "sticky", None)
        verdict = await self.emergency.check(req.message, sid)
        if verdict.emergency:
            if verdict.language and not shared.get("language"):
                shared["language"] = verdict.language
            return self._safety(conv, sid, verdict.source or "regex", verdict.category, verdict.language)

        if conv.pending_agent:
            # A read-back is waiting for the patient's answer, so this message belongs to the agent that asked.
            # No router: it could send "yes" somewhere else. The emergency check above has already run.
            return await self._specialist(conv, req, conv.pending_agent)

        try:
            route = await self.router.route(req.message, sid, last_agent=conv.last_agent, recent=list(conv.recent))
        except Exception as e:
            return self._trouble(conv, sid, "router_failed", e)
        if route.language:
            shared["language"] = route.language
        if route.patient_name:
            shared["patient_name"] = route.patient_name
        if route.emergency:
            return self._safety(conv, sid, "router", None)

        if route.route == "human":
            flag_handoff(conv.ctx, "patient_requested")
            return self._fixed(conv, sid, "human", HANDOFF[lang_of(shared.get("language"))], "router")

        return await self._specialist(conv, req, route.route)

    async def _specialist(self, conv: Conversation, req: ChatRequest, name: str) -> ChatResponse:
        shared = conv.shared
        conv.touched = name
        if name != conv.last_agent and conv.recent:
            shared["recent"] = "\n".join(f"{who}: {said}" for who, said in conv.recent)
        else:
            shared.pop("recent", None)
        try:
            result = await run_in_session(self.agents[name], self.store, f"{req.session_id}::{name}", req.message,
                                          lock_timeout=LOCK_TIMEOUT_S, state=shared)
        except SessionBusy:
            raise
        except Exception as e:  # model outage after retries, or a bug: the patient still gets an honest answer
            return self._trouble(conv, req.session_id, f"{name}_failed", e)
        conv.last_agent = name
        session = await self.store.load(f"{req.session_id}::{name}")
        conv.pending_agent = name if session.ctx.pending else None
        if result.stop_reason not in ("end_turn", "paused") or not result.text.strip():
            return self._trouble(conv, req.session_id, f"{name}_{result.stop_reason}", result.reason or "")
        return ChatResponse(reply=result.text, agent=name, handoff=is_handoff(conv.ctx),
                            emergency=bool(shared.get("emergency")), trace_id=result.trace_id)

    async def _expire_unseen_confirmations(self, sid: str, conv: Conversation) -> None:
        """A read-back is only good for the patient's very next message, and only with the agent that asked.
        An agent's own turn counter does not move while others handle the chat, so without this a "yes" many
        messages later could still confirm an old question. Whoever did not handle this message forgets theirs."""
        for name in SPECIALISTS:
            key = f"{sid}::{name}"
            if name != conv.touched and key in self.store:
                session = await self.store.load(key)
                if session.ctx.pending:
                    session.ctx.pending.clear()
                    await self.store.save(session)
        if conv.pending_agent and conv.pending_agent != conv.touched:
            conv.pending_agent = None

    # ------------------------------------------------------------------ replies written by code

    def _own_trace(self, sid: str, agent: str, event: str, data: dict) -> str:
        trace = self.tracer.start(f"{sid}::{agent}", agent)
        trace.add(Event("check", event, time.time(), 0.0, data))
        trace.stop_reason = event
        return trace.trace_id

    def _safety(self, conv: Conversation, sid: str, source: str, category: str | None,
                language: str | None = None) -> ChatResponse:
        shared = conv.shared
        shared["emergency"] = True
        flag_handoff(conv.ctx, "emergency")
        reply = emergency_reply(language or shared.get("language") or "en")
        trace_id = self._own_trace(sid, "safety", "emergency", {"source": source, "category": category})
        return ChatResponse(reply=reply, agent="safety", handoff=True, emergency=True, trace_id=trace_id)

    def _fixed(self, conv: Conversation, sid: str, agent: str, reply: str, why: str) -> ChatResponse:
        trace_id = self._own_trace(sid, agent, why, {})
        return ChatResponse(reply=reply, agent=agent, handoff=is_handoff(conv.ctx),
                            emergency=bool(conv.shared.get("emergency")), trace_id=trace_id)

    def _trouble(self, conv: Conversation, sid: str, reason: str, error: object) -> ChatResponse:
        """Something failed (model unreachable, run stopped early). Hand over rather than guess or go silent."""
        log.warning("handing over %s: %s: %s", sid, reason, error)
        flag_handoff(conv.ctx, reason)
        trace_id = self._own_trace(sid, "human", reason, {"error": str(error)[:300]})
        return ChatResponse(reply=TROUBLE[lang_of(conv.shared.get("language"))], agent="human", handoff=True,
                            emergency=bool(conv.shared.get("emergency")), trace_id=trace_id)
