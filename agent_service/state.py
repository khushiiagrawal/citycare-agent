"""What agent_service keeps in `RunContext.state`. One place, so tools and hooks agree on the names."""

from __future__ import annotations

from agentcore.types import RunContext


def phone_of(ctx: RunContext) -> str:
    """The verified phone of the person in this conversation. Set by the service, never by the model."""
    phone = ctx.state.get("phone")
    if not phone:
        raise RuntimeError("No phone in the run context: the service must set ctx.state['phone'] before running an agent.")
    return phone


def flag_handoff(ctx: RunContext, reason: str) -> None:
    """Mark the chat for hospital staff. The service reads this to set `handoff: true`."""
    ctx.state["handoff"] = True
    reasons = ctx.state.setdefault("handoff_reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def is_handoff(ctx: RunContext) -> bool:
    return bool(ctx.state.get("handoff"))


def handoff_reasons(ctx: RunContext) -> list[str]:
    return list(ctx.state.get("handoff_reasons", []))


def failures(ctx: RunContext) -> int:
    """How many hospital calls in a row have failed (a success resets it)."""
    return int(ctx.state.get("failures", 0))


def record_failure(ctx: RunContext) -> None:
    ctx.state["failures"] = failures(ctx) + 1


def reset_failures(ctx: RunContext) -> None:
    ctx.state["failures"] = 0
