"""The agent loop: call the model, run the tools it asks for, repeat."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from .types import Registry, RunContext, RunResult, ToolCall, ToolRecord, ToolResult


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)


def _check_input(schema: dict, tool_input: Any) -> str | None:
    """Cheap guard so a malformed call never reaches a handler. Not a full JSON-schema check."""
    if not isinstance(tool_input, dict):
        return "Tool input must be a JSON object."
    missing = [k for k in schema.get("required", []) if k not in tool_input]
    return f"Missing required field(s): {', '.join(missing)}." if missing else None


class Agent:
    """One model + one system prompt + one set of tools.

    `client` is anything with `messages.create(...)`: `anthropic.AsyncAnthropic`,
    `anthropic.Anthropic`, or a test fake. The result may be awaitable or not.
    `max_writes_per_run` caps calls to tools marked `writes=True` across the whole run.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        system: str,
        tools: Registry | None = None,
        *,
        max_turns: int = 10,
        max_tokens: int = 1024,
        max_writes_per_run: int | None = None,
        request_params: dict | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.tools = tools if tools is not None else Registry()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_writes_per_run = max_writes_per_run
        self.request_params = dict(request_params or {})

    async def run(self, messages: list[dict], ctx: RunContext | None = None) -> RunResult:
        """Run until the model stops asking for tools. `messages` is not modified;
        the messages this run adds are in `result.messages`."""
        ctx = ctx if ctx is not None else RunContext()
        ctx.turn += 1
        history = list(messages)
        result = RunResult(text="", stop_reason="end_turn")
        writes = 0

        for step in range(1, self.max_turns + 1):
            response = await self._call_model(history)
            result.model_calls += 1
            result.usage["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
            result.usage["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0

            blocks = [b.model_dump(exclude_none=True) for b in response.content]
            result.text = "\n".join(b["text"] for b in blocks if b["type"] == "text").strip()
            calls = [ToolCall(b["id"], b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"]

            if response.stop_reason != "tool_use" or not calls:
                # A half-written tool call must never enter history without a result.
                kept = [b for b in blocks if b["type"] != "tool_use"]
                if kept:
                    self._add(history, result, {"role": "assistant", "content": kept})
                result.stop_reason = response.stop_reason or "end_turn"
                return result

            self._add(history, result, {"role": "assistant", "content": blocks})
            writes = await self._run_tools(calls, ctx, result, history, writes)
            if step == self.max_turns:
                result.stop_reason = "max_turns"
                return result
        return result  # unreachable when max_turns >= 1

    # ------------------------------------------------------------------ internals

    async def _call_model(self, history: list[dict]) -> Any:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": history,
            **self.request_params,
        }
        if len(self.tools):
            kwargs["tools"] = self.tools.api_schemas()
        return await _maybe_await(self.client.messages.create(**kwargs))

    @staticmethod
    def _add(history: list[dict], result: RunResult, message: dict) -> None:
        history.append(message)
        result.messages.append(message)

    async def _run_tools(
        self, calls: list[ToolCall], ctx: RunContext, result: RunResult, history: list[dict], writes: int
    ) -> int:
        """Run one response's tool calls and append a single user message with all results.

        Checks happen one by one, in order; the calls that pass then run concurrently.
        Returns the updated write count."""
        outcomes: dict[str, tuple[ToolResult, str]] = {}
        to_run: list[ToolCall] = []

        for call in calls:
            tool = self.tools.get(call.name)
            if tool is None:
                outcomes[call.id] = (ToolResult(f"Unknown tool: {call.name}", True), "unknown")
                continue
            problem = _check_input(tool.input_schema, call.input)
            if problem:
                outcomes[call.id] = (ToolResult(problem, True), "error")
                continue
            if tool.writes and self.max_writes_per_run is not None:
                if writes >= self.max_writes_per_run:
                    limit = "one change" if self.max_writes_per_run == 1 else f"{self.max_writes_per_run} changes"
                    outcomes[call.id] = (
                        ToolResult(f"Not executed: only {limit} can be made per message. "
                                   "Tell the user and continue in their next message.", True),
                        "denied",
                    )
                    continue
                writes += 1
            to_run.append(call)

        async def execute(call: ToolCall) -> tuple[ToolResult, str]:
            try:
                value = await _maybe_await(self.tools.get(call.name).handler(call.input, ctx))
                return ToolResult(value), "ok"
            except Exception as e:  # a failing tool must not kill the run
                return ToolResult(f"{type(e).__name__}: {e}", True), "error"

        for call, outcome in zip(to_run, await asyncio.gather(*(execute(c) for c in to_run))):
            outcomes[call.id] = outcome

        blocks = []
        for call in calls:
            res, kind = outcomes[call.id]
            result.tool_records.append(ToolRecord(call, res, kind))
            blocks.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": _content_text(res.content),
                "is_error": res.is_error,
            })
        self._add(history, result, {"role": "user", "content": blocks})
        return writes
