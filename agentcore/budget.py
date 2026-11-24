"""Spending limits for one run and for a whole conversation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """Ceilings checked before every model call. Any left as None is not enforced.

    A run that hits one stops with stop_reason "budget_exceeded" and the reason in
    `result.text`. History stays valid. Because the check is before a call, the call that
    crosses the line still happens; `max_tokens` on the agent bounds how far over it can go."""

    max_cost_usd: float | None = None  # this run
    max_tokens: int | None = None  # this run, all token types
    max_seconds: float | None = None  # this run, including slow tools and retry waits
    max_session_cost_usd: float | None = None  # the whole conversation

    def exceeded(self, *, spent_usd: float, tokens: int, seconds: float, session_usd: float) -> str | None:
        if self.max_cost_usd is not None and spent_usd >= self.max_cost_usd:
            return f"Cost limit reached (${spent_usd:.4f} of ${self.max_cost_usd:.4f})."
        if self.max_session_cost_usd is not None and session_usd >= self.max_session_cost_usd:
            return f"Session cost limit reached (${session_usd:.4f} of ${self.max_session_cost_usd:.4f})."
        if self.max_tokens is not None and tokens >= self.max_tokens:
            return f"Token limit reached ({tokens} of {self.max_tokens})."
        if self.max_seconds is not None and seconds >= self.max_seconds:
            return f"Time limit reached ({seconds:.2f} of {self.max_seconds} seconds)."
        return None
