"""Tracing and cost: what every model call and tool call did, how long it took, what it cost."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_PREVIEW_CHARS = 500


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens. You pass this in; agentcore hardcodes no prices.

    Cache reads default to 10% of the input price and cache writes to 125% of it
    (Anthropic's standard multipliers). Override if your model differs."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
        read = self.cache_read_per_mtok if self.cache_read_per_mtok is not None else self.input_per_mtok * 0.1
        write = self.cache_write_per_mtok if self.cache_write_per_mtok is not None else self.input_per_mtok * 1.25
        total = (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read * read
            + cache_write * write
        )
        return total / 1_000_000


def preview(value: Any, limit: int = _PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class Event:
    """One model call or one tool call. `kind` is "model" or "tool"."""

    kind: str
    name: str
    started_at: float  # unix time
    duration_s: float
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "started_at": self.started_at,
                "duration_s": round(self.duration_s, 4), **self.data}


@dataclass
class Trace:
    """Everything one agent run did."""

    trace_id: str
    key: str
    agent: str
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    stop_reason: str | None = None
    events: list[Event] = field(default_factory=list)

    def add(self, event: Event) -> None:
        self.events.append(event)

    def _sum(self, field_name: str) -> float:
        return sum(e.data.get(field_name, 0) or 0 for e in self.events if e.kind == "model")

    @property
    def input_tokens(self) -> int:
        return int(self._sum("input_tokens"))

    @property
    def output_tokens(self) -> int:
        return int(self._sum("output_tokens"))

    @property
    def cost_usd(self) -> float:
        return self._sum("cost_usd")

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id, "key": self.key, "agent": self.agent,
            "started_at": self.started_at, "duration_s": round(self.duration_s, 4),
            "stop_reason": self.stop_reason, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6),
            "events": [e.to_dict() for e in self.events],
        }


class Tracer:
    """Keeps recent traces in memory, grouped by key (for example one key per conversation).

    Bounded: it forgets the least recently used key past `max_keys`, and the oldest
    traces of a key past `max_traces_per_key`."""

    def __init__(self, max_keys: int = 1000, max_traces_per_key: int = 50) -> None:
        self.max_keys = max_keys
        self.max_traces_per_key = max_traces_per_key
        self._by_key: OrderedDict[str, list[Trace]] = OrderedDict()

    def start(self, key: str, agent: str) -> Trace:
        trace = new_trace(key, agent)
        traces = self._by_key.setdefault(key, [])
        self._by_key.move_to_end(key)
        traces.append(trace)
        del traces[: -self.max_traces_per_key]
        while len(self._by_key) > self.max_keys:
            self._by_key.popitem(last=False)
        return trace

    def traces(self, key: str) -> list[Trace]:
        return list(self._by_key.get(key, []))


def new_trace(key: str, agent: str) -> Trace:
    return Trace(trace_id=uuid.uuid4().hex, key=key, agent=agent)
