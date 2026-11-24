"""Core data types: tools, the tool registry, run context and run result."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Iterator

from pydantic import BaseModel, ConfigDict, ValidationError

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

Handler = Callable[[dict, "RunContext"], Any | Awaitable[Any]]


class ToolInputError(ValueError):
    """The model sent input that does not fit the tool. The message goes back to the model."""


def _strip_titles(node: Any) -> Any:
    """Drop the auto-generated "title" keys pydantic adds. They only cost tokens.
    A property that is itself called "title" is kept."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "title" and isinstance(value, str):
                continue
            if key in ("properties", "$defs") and isinstance(value, dict):
                out[key] = {name: _strip_titles(sub) for name, sub in value.items()}
            else:
                out[key] = _strip_titles(value)
        return out
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


def _forbid_extra(model: type[BaseModel]) -> type[BaseModel]:
    """The same model, but rejecting unknown fields. The original class is left alone."""
    if model.model_config.get("extra") == "forbid":
        return model
    return type(model.__name__, (model,), {"model_config": ConfigDict(extra="forbid"), "__module__": model.__module__})


@dataclass(frozen=True)
class Tool:
    """A function the model may call.

    Build one from a plain function and a pydantic model with `Tool.from_function`
    (or `Registry.register`): the schema comes from the model, and input is validated
    with it before the handler runs. The handler then receives the model instance.

    Or build one by hand with a raw `input_schema`: the handler then receives the
    input dict, and only required fields are checked.

    `writes=True` marks a tool that changes the outside world. The loop can cap
    how many such tools run per turn. The handler may be sync or async.
    """

    name: str
    description: str
    input_schema: dict
    handler: Handler
    writes: bool = False
    input_model: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"Invalid tool name: {self.name!r} (use 1-64 chars of a-z, A-Z, 0-9, _ or -)")

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        input_model: type[BaseModel],
        *,
        name: str | None = None,
        description: str | None = None,
        writes: bool = False,
    ) -> Tool:
        """Make a tool from `fn(args: input_model, ctx)` (or `fn(args)`).

        Name defaults to the function name and description to its docstring.
        Unknown input fields are rejected."""
        description = description or inspect.getdoc(fn)
        if not description:
            raise ValueError(f"Tool {name or fn.__name__!r} needs a description or a docstring: the model reads it.")
        model = _forbid_extra(input_model)
        positional = [
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        handler = fn if len(positional) >= 2 else (lambda args, ctx: fn(args))
        return cls(
            name=name or fn.__name__,
            description=description,
            input_schema=_strip_titles(model.model_json_schema()),
            handler=handler,
            writes=writes,
            input_model=model,
        )

    def parse(self, raw: Any) -> Any:
        """Check the model's input. Returns what the handler will receive, or raises ToolInputError."""
        if not isinstance(raw, dict):
            raise ToolInputError("Invalid input: expected a JSON object.")
        if self.input_model is None:
            missing = [k for k in self.input_schema.get("required", []) if k not in raw]
            if missing:
                raise ToolInputError(f"Missing required field(s): {', '.join(missing)}.")
            return raw
        try:
            # JSON mode + strict: "3" is not an int, but "2026-12-04" is a valid date.
            return self.input_model.model_validate_json(json.dumps(raw, default=str), strict=True)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or 'input'}: {err['msg']}" for err in e.errors()
            )
            raise ToolInputError(f"Invalid input. {problems}") from None

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

    def register(
        self,
        input_model: type[BaseModel],
        *,
        name: str | None = None,
        description: str | None = None,
        writes: bool = False,
    ) -> Callable[[Callable], Callable]:
        """Decorator: turn a function into a tool. The function is returned unchanged.

            @registry.register(SearchArgs, writes=False)
            async def search(args: SearchArgs, ctx): ...
        """

        def decorator(fn: Callable) -> Callable:
            self.add(Tool.from_function(fn, input_model, name=name, description=description, writes=writes))
            return fn

        return decorator

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
