"""The agent loop: call the model, run the tools it asks for, repeat."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, Sequence

from ._util import maybe_await as _maybe_await
from .hooks import Confirm, Deny, Hook, Modify
from .types import (
    PendingConfirmation,
    Registry,
    RunContext,
    RunResult,
    ToolCall,
    ToolInputError,
    ToolRecord,
    ToolResult,
)


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)


class Agent:
    """One model + one system prompt + one set of tools.

    `client` is anything with `messages.create(...)`: `anthropic.AsyncAnthropic`,
    `anthropic.Anthropic`, or a test fake. The result may be awaitable or not.
    `max_writes_per_run` caps calls to tools marked `writes=True` across the whole run.
    `hooks` run around the model and its tools (see hooks.py).
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
        hooks: Sequence[Hook] = (),
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.tools = tools if tools is not None else Registry()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_writes_per_run = max_writes_per_run
        self.request_params = dict(request_params or {})
        self.hooks = list(hooks)

    async def run(self, messages: list[dict], ctx: RunContext | None = None) -> RunResult:
        """Run until the model stops asking for tools. `messages` is not modified;
        the messages this run adds are in `result.messages`."""
        ctx = ctx if ctx is not None else RunContext()
        ctx.turn += 1
        # A confirmation only stays valid for the user's very next message.
        ctx.pending[:] = [p for p in ctx.pending if p.turn == ctx.turn - 1]
        result = await self._run(messages, ctx)
        # Whatever was not asked in this run is no longer waiting for an answer.
        ctx.pending[:] = [p for p in ctx.pending if p.turn == ctx.turn]
        for hook in self.hooks:
            await _maybe_await(hook.on_finish(result, ctx))
        return result

    async def _run(self, messages: list[dict], ctx: RunContext) -> RunResult:
        history = list(messages)
        result = RunResult(text="", stop_reason="end_turn")
        writes = 0

        for step in range(1, self.max_turns + 1):
            response = await self._call_model(history, ctx)
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
            writes, asked = await self._run_tools(calls, ctx, result, history, writes)
            if asked:
                # Paused for the user's confirmation. The question is written by code.
                question = "\n\n".join(dict.fromkeys(p.prompt for p in asked))
                self._add(history, result, {"role": "assistant", "content": [{"type": "text", "text": question}]})
                result.text, result.stop_reason, result.pending = question, "paused", list(asked)
                return result
            if step == self.max_turns:
                result.stop_reason = "max_turns"
                return result
        return result  # unreachable when max_turns >= 1

    # ------------------------------------------------------------------ internals

    async def _call_model(self, history: list[dict], ctx: RunContext) -> Any:
        messages = history
        for hook in self.hooks:
            replaced = await _maybe_await(hook.before_model(messages, ctx))
            if replaced is not None:
                messages = replaced
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": messages,
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
    ) -> tuple[int, list[PendingConfirmation]]:
        """Run one response's tool calls and append a single user message with all results.

        Checks happen one by one, in order; the calls that pass then run concurrently.
        Returns the updated write count and any confirmations that were asked for."""
        outcomes: dict[str, tuple[ToolResult, str]] = {}
        executed: dict[str, ToolCall] = {}  # calls as run (hooks may have changed their input)
        parsed: dict[str, Any] = {}  # validated input per call, in the form the handler takes
        to_run: list[ToolCall] = []
        asked: list[PendingConfirmation] = []

        for call in calls:
            tool = self.tools.get(call.name)
            if tool is None:
                outcomes[call.id] = (ToolResult(f"Unknown tool: {call.name}", True), "unknown")
                continue
            try:
                parsed[call.id] = tool.parse(call.input)
            except ToolInputError as e:
                outcomes[call.id] = (ToolResult(str(e), True), "error")
                continue

            verdict = None
            for hook in self.hooks:
                outcome = await _maybe_await(hook.before_tool(call, ctx))
                if isinstance(outcome, Modify):
                    call = replace(call, input=outcome.input)
                    try:
                        parsed[call.id] = tool.parse(call.input)
                    except ToolInputError as e:
                        verdict = (ToolResult(str(e), True), "error")
                        break
                elif isinstance(outcome, (Deny, Confirm)):
                    verdict = outcome
                    break
            if isinstance(verdict, Deny):
                outcomes[call.id] = (ToolResult(f"Not executed: {verdict.reason}", True), "denied")
                continue
            if isinstance(verdict, Confirm):
                pending = PendingConfirmation(call.name, call.input, verdict.prompt, ctx.turn)
                ctx.pending[:] = [p for p in ctx.pending if p.fingerprint != pending.fingerprint] + [pending]
                asked.append(pending)
                outcomes[call.id] = (
                    ToolResult(
                        "Not executed yet. The user has been asked to confirm: "
                        f"{verdict.prompt}\nIf their next message clearly agrees, call this tool again "
                        "with exactly the same input. Otherwise do not.",
                        True,
                    ),
                    "paused",
                )
                continue
            if verdict is not None:  # a failed Modify
                outcomes[call.id] = verdict
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
                value = await _maybe_await(self.tools.get(call.name).handler(parsed[call.id], ctx))
                return ToolResult(value), "ok"
            except Exception as e:  # a failing tool must not kill the run
                return ToolResult(f"{type(e).__name__}: {e}", True), "error"

        for call, (res, kind) in zip(to_run, await asyncio.gather(*(execute(c) for c in to_run))):
            for hook in self.hooks:
                replaced = await _maybe_await(hook.after_tool(call, res, ctx))
                if replaced is not None:
                    res = replaced
            outcomes[call.id] = (res, kind)
            executed[call.id] = call

        blocks = []
        for call in calls:
            res, kind = outcomes[call.id]
            result.tool_records.append(ToolRecord(executed.get(call.id, call), res, kind))
            blocks.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": _content_text(res.content),
                "is_error": res.is_error,
            })
        self._add(history, result, {"role": "user", "content": blocks})
        return writes, asked
