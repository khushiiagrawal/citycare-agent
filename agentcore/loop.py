"""The agent loop: call the model, run the tools it asks for, repeat."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from ._util import maybe_await as _maybe_await
from ._util import reject_sync_client
from .budget import Budget
from .hooks import Confirm, Deny, Hook, Modify
from .resilience import ModelUnavailable, RetryPolicy, is_retryable
from .trace import Event, Price, Trace, Tracer, new_trace, preview, snapshot
from .types import (
    PendingConfirmation,
    Registry,
    RunContext,
    RunResult,
    ToolCall,
    ToolInputError,
    ToolRecord,
    ToolResult,
    fingerprint,
)


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0


class Agent:
    """One model + one system prompt + one set of tools.

    `client` is `anthropic.AsyncAnthropic`, or a test fake with `messages.create(...)`
    (the fake's result may be awaitable or not). Sync Anthropic clients are rejected.
    `max_writes_per_run` caps calls to tools marked `writes=True` across the whole run.
    `hooks` run around the model and its tools (see hooks.py).
    `price` turns token counts into dollars (`result.cost_usd`); without it cost is 0.
    `tracer` keeps a trace of every model and tool call, grouped by `ctx.key`.
    `name` labels the agent in traces.
    `retry` retries model calls that fail for temporary reasons (None turns it off). When it
    gives up it raises ModelUnavailable. `budget` stops a run that spends too much.
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
        name: str = "agent",
        price: Price | None = None,
        tracer: Tracer | None = None,
        retry: RetryPolicy | None = RetryPolicy(),
        budget: Budget | None = None,
    ) -> None:
        reject_sync_client(client)
        self.client = client
        self.model = model
        self.system = system
        self.tools = tools if tools is not None else Registry()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_writes_per_run = max_writes_per_run
        self.request_params = dict(request_params or {})
        self.hooks = list(hooks)
        self.name = name
        self.price = price
        self.tracer = tracer
        self.retry = retry
        self.budget = budget

    async def run(self, messages: list[dict], ctx: RunContext | None = None) -> RunResult:
        """Run until the model stops asking for tools. `messages` is not modified;
        the messages this run adds are in `result.messages`."""
        ctx = ctx if ctx is not None else RunContext()
        ctx.turn += 1
        # A confirmation only stays valid for the user's very next message.
        ctx.pending[:] = [p for p in ctx.pending if p.turn == ctx.turn - 1]
        trace = self.tracer.start(ctx.key, self.name) if self.tracer else new_trace(ctx.key, self.name)
        started = time.perf_counter()
        try:
            result = await self._run(messages, ctx, trace)
        except BaseException:
            trace.stop_reason, trace.duration_s = "error", time.perf_counter() - started
            raise
        trace.stop_reason, trace.duration_s = result.stop_reason, time.perf_counter() - started
        result.trace_id = trace.trace_id
        # Whatever was not asked in this run is no longer waiting for an answer.
        ctx.pending[:] = [p for p in ctx.pending if p.turn == ctx.turn]
        for hook in self.hooks:
            await _maybe_await(hook.on_finish(result, ctx))
        return result

    async def _run(self, messages: list[dict], ctx: RunContext, trace: Trace) -> RunResult:
        history = list(messages)
        result = RunResult(text="", stop_reason="end_turn")
        writes = 0
        started = time.perf_counter()

        for step in range(1, self.max_turns + 1):
            if self.budget:
                over = self.budget.exceeded(
                    spent_usd=result.cost_usd,
                    tokens=sum(result.usage.values()) + result.cache_read_tokens + result.cache_write_tokens,
                    seconds=time.perf_counter() - started,
                    session_usd=ctx.total_cost_usd,
                )
                if over:
                    result.reason, result.stop_reason = over, "budget_exceeded"
                    return result
            messages = await self._before_model(history, ctx)
            if isinstance(messages, Deny):
                result.reason, result.stop_reason = messages.reason, "blocked"
                return result
            response, used = await self._call_model(messages, trace)
            result.model_calls += 1
            # Cost is counted from what the model really returned, before any hook rewrites it.
            result.usage["input_tokens"] += used.input_tokens
            result.usage["output_tokens"] += used.output_tokens
            result.cache_read_tokens += used.cache_read
            result.cache_write_tokens += used.cache_write
            result.cost_usd += used.cost
            ctx.total_cost_usd += used.cost
            ctx.total_tokens += used.input_tokens + used.output_tokens + used.cache_read + used.cache_write
            for hook in self.hooks:
                replaced = await _maybe_await(hook.after_model(response, ctx))
                if replaced is not None:
                    response = replaced

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
            writes, asked = await self._run_tools(calls, ctx, result, history, writes, trace)
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

    async def _before_model(self, history: list[dict], ctx: RunContext) -> list[dict] | Deny:
        """Run before_model hooks. Returns the messages to send, or a Deny to stop the run."""
        messages = history
        for hook in self.hooks:
            outcome = await _maybe_await(hook.before_model(messages, ctx))
            if isinstance(outcome, Deny):
                return outcome
            if isinstance(outcome, list):
                messages = outcome
        return messages

    def _usage_of(self, response: Any) -> _Usage:
        usage = response.usage
        used = _Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        if self.price:
            used.cost = self.price.cost(used.input_tokens, used.output_tokens, used.cache_read, used.cache_write)
        return used

    async def _call_model(self, messages: list[dict], trace: Trace) -> tuple[Any, _Usage]:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": messages,
            **self.request_params,
        }
        if len(self.tools):
            kwargs["tools"] = self.tools.api_schemas()
        attempts = self.retry.max_attempts if self.retry else 1
        for attempt in range(1, attempts + 1):
            started_at, clock = time.time(), time.perf_counter()
            try:
                response = await self._create(kwargs)
            except Exception as e:
                error = preview(f"{type(e).__name__}: {e}")
                trace.add(Event("model", self.model, started_at, time.perf_counter() - clock,
                                {"error": error, "attempt": attempt}))
                if self.retry is None or not is_retryable(e):
                    raise
                if attempt == attempts or self.retry.too_long(e):
                    # Out of attempts, or the server asked for a wait longer than we will do:
                    # retrying early would just hit the same limit. Fail so the caller can hand over.
                    raise ModelUnavailable(attempt, e) from e
                await self.retry.sleep(self.retry.delay(attempt, e))
                continue
            used = self._usage_of(response)
            trace.add(Event("model", self.model, started_at, time.perf_counter() - clock, {
                "input_tokens": used.input_tokens, "output_tokens": used.output_tokens,
                "cache_read_tokens": used.cache_read, "cache_write_tokens": used.cache_write,
                "cost_usd": used.cost, "stop_reason": response.stop_reason, "attempt": attempt,
            }))
            return response, used
        raise AssertionError("unreachable")  # the loop above always returns or raises

    async def _create(self, kwargs: dict) -> Any:
        """One attempt at the model call, with the per-attempt timeout."""
        call = _maybe_await(self.client.messages.create(**kwargs))
        timeout = self.retry.attempt_timeout_s if self.retry else None
        if timeout is None:
            return await call
        try:
            return await asyncio.wait_for(call, timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"model call timed out after {timeout}s") from None

    @staticmethod
    def _add(history: list[dict], result: RunResult, message: dict) -> None:
        history.append(message)
        result.messages.append(message)

    async def _run_tools(
        self, calls: list[ToolCall], ctx: RunContext, result: RunResult, history: list[dict], writes: int,
        trace: Trace,
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

            if tool.writes and self.max_writes_per_run is not None and writes >= self.max_writes_per_run:
                # Checked before the hooks so a confirmation is not used up by a call that cannot run.
                # The user did say yes, so keep it alive for their next message.
                for i, p in enumerate(ctx.pending):
                    if p.fingerprint == fingerprint(call.name, call.input):
                        ctx.pending[i] = replace(p, turn=ctx.turn)
                limit = "one change" if self.max_writes_per_run == 1 else f"{self.max_writes_per_run} changes"
                outcomes[call.id] = (
                    ToolResult(f"Not executed: only {limit} can be made per message. "
                               "Tell the user and continue in their next message.", True),
                    "denied",
                )
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
            if tool.writes:
                writes += 1
            to_run.append(call)

        timing: dict[str, tuple[float, float]] = {}  # call id -> (unix start, seconds)

        async def execute(call: ToolCall) -> tuple[ToolResult, str]:
            started_at, clock = time.time(), time.perf_counter()
            try:
                value = await _maybe_await(self.tools.get(call.name).handler(parsed[call.id], ctx))
                return ToolResult(value), "ok"
            except Exception as e:  # a failing tool must not kill the run
                return ToolResult(f"{type(e).__name__}: {e}", True), "error"
            finally:
                timing[call.id] = (started_at, time.perf_counter() - clock)

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
            ran = executed.get(call.id, call)
            content = _content_text(res.content)
            result.tool_records.append(ToolRecord(ran, res, kind))
            blocks.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": res.is_error})
            started_at, seconds = timing.get(call.id, (time.time(), 0.0))
            trace.add(Event("tool", call.name, started_at, seconds, {
                "input": snapshot(ran.input), "outcome": kind, "is_error": res.is_error, "output": preview(content),
            }))
        self._add(history, result, {"role": "user", "content": blocks})
        return writes, asked
