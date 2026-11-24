"""Conversation sessions: stored history, a lock per conversation, and safe trimming."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from .loop import Agent
from .types import RunContext, RunResult


class SessionBusy(Exception):
    """Another message in the same conversation is still being handled."""


@dataclass
class Session:
    """One conversation: the message history plus its run context (turn count, pending confirmations)."""

    key: str
    messages: list[dict] = field(default_factory=list)
    ctx: RunContext = field(default_factory=RunContext)

    def __post_init__(self) -> None:
        if not self.ctx.key:
            self.ctx.key = self.key


class SessionStore(Protocol):
    """Where sessions live. Swap InMemoryStore for Redis or a database by implementing this."""

    async def load(self, key: str) -> Session: ...

    async def save(self, session: Session) -> None: ...

    def lock(self, key: str, timeout: float | None = None): ...  # async context manager


class InMemoryStore:
    """A single-process store. Bounded: past `max_sessions` it forgets the oldest idle conversations."""

    def __init__(self, max_sessions: int = 10_000) -> None:
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._locks: dict[str, list] = {}  # key -> [asyncio.Lock, number of users]

    @property
    def lock_count(self) -> int:
        return len(self._locks)

    def __contains__(self, key: object) -> bool:
        return key in self._sessions

    async def load(self, key: str) -> Session:
        session = self._sessions.get(key)
        if session is None:
            return Session(key=key)
        self._sessions.move_to_end(key)
        return session

    async def save(self, session: Session) -> None:
        self._sessions[session.key] = session
        self._sessions.move_to_end(session.key)
        while len(self._sessions) > self.max_sessions:
            idle = next((k for k in self._sessions if k != session.key and k not in self._locks), None)
            if idle is None:
                break
            del self._sessions[idle]

    @asynccontextmanager
    async def lock(self, key: str, timeout: float | None = None) -> AsyncIterator[None]:
        """One at a time per conversation. Raises SessionBusy if not free within `timeout` seconds."""
        entry = self._locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            try:
                await asyncio.wait_for(entry[0].acquire(), timeout)
            except asyncio.TimeoutError:
                raise SessionBusy(f"conversation {key!r} is busy") from None
            try:
                yield
            finally:
                entry[0].release()
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._locks.pop(key, None)


# ------------------------------------------------------------------ history safety


def _blocks(message: dict) -> list[dict]:
    content = message.get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _is_plain_user(message: dict) -> bool:
    return message.get("role") == "user" and not any(b.get("type") == "tool_result" for b in _blocks(message))


def trim_history(messages: list[dict], max_messages: int | None) -> list[dict]:
    """Keep about the last `max_messages` messages without breaking the conversation.

    The result always starts with a real user message, never with a tool result, and a tool
    call is never separated from its result. If that means keeping more than asked, it keeps
    more. If there is no user message at all, it returns nothing."""
    if max_messages is None or len(messages) <= max_messages:
        return list(messages)
    start = len(messages) - max_messages
    while start < len(messages) and not _is_plain_user(messages[start]):
        start += 1
    if start < len(messages):
        return messages[start:]
    last_user = next((i for i in range(len(messages) - 1, -1, -1) if _is_plain_user(messages[i])), None)
    return [] if last_user is None else messages[last_user:]


def repair_history(messages: list[dict]) -> list[dict]:
    """Cut the history at the first tool call that never got all of its results.

    The API rejects such a history, so one crash mid-run must not poison the conversation."""
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        wanted = {b["id"] for b in _blocks(message) if b.get("type") == "tool_use"}
        if not wanted:
            continue
        follow = messages[i + 1] if i + 1 < len(messages) else {}
        answered = {b.get("tool_use_id") for b in _blocks(follow) if b.get("type") == "tool_result"}
        if not wanted <= answered:
            return list(messages[:i])
    return list(messages)


# ------------------------------------------------------------------ one turn


async def run_in_session(
    agent: Agent,
    store: SessionStore,
    key: str,
    user_message: str | list,
    *,
    max_history: int | None = 40,
    lock_timeout: float | None = None,
) -> RunResult:
    """Handle one user message in a conversation.

    Takes the conversation's lock, so two messages in the same chat never run at once (and
    two bookings cannot race), loads and tidies the history, runs the agent, and saves the
    result. If the run raises, nothing is saved and the conversation is left as it was."""
    async with store.lock(key, lock_timeout):
        session = await store.load(key)
        history = trim_history(repair_history(session.messages), max_history)
        messages = [*history, {"role": "user", "content": user_message}]
        result = await agent.run(messages, session.ctx)
        session.messages = trim_history([*messages, *result.messages], max_history)
        await store.save(session)
        return result
