"""Hooks: code that runs around the model and its tools, and can say no.

Five hook points, all optional, all sync or async:

  before_model(messages, ctx)      -> Deny(reason) to block the call, a list of messages to
                                      use for this call only, or None (= go ahead)
  after_model(response, ctx)       -> a replacement response, or None (see `with_text`)
  before_tool(call, ctx)           -> Allow | Deny | Modify | Confirm, or None (= Allow)
  after_tool(call, result, ctx)    -> a replacement ToolResult, or None
  on_finish(result, ctx)           -> nothing; called once when a run ends

A blocked model call ends the run with stop_reason "blocked" and the reason in
`result.text`; the model is not called and nothing is added to the history.
A replaced model response is treated exactly as if the model had said it: its text is
stored in the history, and any tool calls in it still go through the before_tool hooks.
The cost of the original response is still counted.

Before-tool outcomes:
  Allow            run the tool
  Deny(reason)     do not run it; the model sees `reason` as an error result
  Modify(input)    run it with this input instead
  Confirm(prompt)  do not run it; pause the run and send `prompt` to the user.
                   The run resumes with the user's next message.

Hooks run in the order given. The first Deny or Confirm stops the chain; Modify passes
its new input on to the next hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from anthropic.types import TextBlock

from ._util import maybe_await
from .types import RunContext, RunResult, ToolCall, ToolResult, fingerprint


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Deny:
    reason: str


@dataclass(frozen=True)
class Modify:
    input: dict


@dataclass(frozen=True)
class Confirm:
    prompt: str


Outcome = Allow | Deny | Modify | Confirm


class Hook:
    """Base class. Override only what you need."""

    def before_model(self, messages: list[dict], ctx: RunContext) -> Any:
        return None

    def after_model(self, response: Any, ctx: RunContext) -> Any:
        return None

    def before_tool(self, call: ToolCall, ctx: RunContext) -> Any:
        return None

    def after_tool(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> Any:
        return None

    def on_finish(self, result: RunResult, ctx: RunContext) -> Any:
        return None


def with_text(response: Any, text: str) -> Any:
    """A copy of a model response that says `text` instead, with no tool calls.
    Usage and ids are kept. Use it from `after_model` to swap an unsafe reply."""
    return response.model_copy(update={"content": [TextBlock(type="text", text=text)], "stop_reason": "end_turn"})


PromptBuilder = Callable[[ToolCall, RunContext], "str | Deny | Awaitable[str | Deny]"]


class ConfirmGate(Hook):
    """Make chosen tools wait for the user's yes.

    `prompts` maps a tool name to a function that builds the question from the real
    call (so the wording comes from code, not from the model). It may return a string,
    or a Deny to refuse the call outright, and may be async.

    How it works: the first call to a guarded tool pauses the run with the prompt.
    The call is allowed only if the model makes the same call again (same tool, same
    input) in the very next run. Anything else asks again. Each confirmation works
    once. The model decides whether the user's reply was a yes; code decides that the
    call cannot happen in the same turn it was proposed.

    Put this hook last, so earlier checks can deny before a confirmation is used up.
    """

    def __init__(self, prompts: Mapping[str, PromptBuilder]) -> None:
        self.prompts = dict(prompts)

    async def before_tool(self, call: ToolCall, ctx: RunContext) -> Outcome | None:
        builder = self.prompts.get(call.name)
        if builder is None:
            return None
        fp = fingerprint(call.name, call.input)
        for pending in ctx.pending:
            if pending.fingerprint == fp and pending.turn < ctx.turn:
                ctx.pending.remove(pending)
                return Allow()
        built = await maybe_await(builder(call, ctx))
        return Confirm(built) if isinstance(built, str) else built
