"""Core data types: tools, the tool registry, run context and run result."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Iterator

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

Handler = Callable[[dict, "RunContext"], Any | Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    """A function the model may call.

    `writes=True` marks a tool that changes the outside world. The loop can cap
    how many such tools run per turn. The handler may be sync or async.
    """

    name: str
    description: str
    input_schema: dict
    handler: Handler
    writes: bool = False

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"Invalid tool name: {self.name!r} (use 1-64 chars of a-z, A-Z, 0-9, _ or -)")

    def to_api(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class Registry:
    """An ordered set of tools. Order is stable so the tool list stays cacheable."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def subset(self, names: Iterable[str]) -> Registry:
        """A new registry with only these tools, in the order given."""
        return Registry(self._tools[n] for n in names)

    def api_schemas(self) -> list[dict]:
        return [t.to_api() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    content: Any
    is_error: bool = False


@dataclass(frozen=True)
class PendingConfirmation:
    """A tool call that was paused until the user confirms it."""

    tool: str
    input: dict
    prompt: str
    turn: int

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.tool, self.input)


def fingerprint(tool: str, tool_input: dict) -> str:
    return tool + ":" + json.dumps(tool_input, sort_keys=True, default=str)


@dataclass
class RunContext:
    """Per-conversation state that outlives a single run.

    `state` is free-form for hooks and tools. `turn` counts runs and is bumped by
    the loop. `pending` holds confirmations waiting for the user's next message.
    """

    state: dict = field(default_factory=dict)
    turn: int = 0
    pending: list[PendingConfirmation] = field(default_factory=list)


@dataclass
class ToolRecord:
    """What happened to one tool call. `outcome`: ok | error | denied | paused | unknown."""

    call: ToolCall
    result: ToolResult
    outcome: str


@dataclass
class RunResult:
    text: str
    stop_reason: str
    messages: list[dict] = field(default_factory=list)  # messages added by this run
    tool_records: list[ToolRecord] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    pending: list[PendingConfirmation] = field(default_factory=list)
    model_calls: int = 0
