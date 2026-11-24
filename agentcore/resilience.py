"""Retrying model calls that fail for temporary reasons."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anthropic

# 429s that mean "you are out of money or quota", not "slow down". Waiting does not help.
_NEVER_RETRY = tuple(
    cls
    for name in ("EnforcedSpendLimitReachedError", "ContractSpendLimitReachedError",
                 "LinkedOrganizationSpendLimitReachedError", "OrganizationOnHoldError")
    if (cls := getattr(anthropic, name, None)) is not None
)


class ModelUnavailable(Exception):
    """The model could not be reached after every retry. `__cause__` is the last error."""

    def __init__(self, attempts: int, cause: Exception) -> None:
        super().__init__(f"model unavailable after {attempts} attempt(s): {type(cause).__name__}: {cause}")
        self.attempts = attempts


def is_retryable(error: BaseException) -> bool:
    """True for errors that a retry can fix: timeouts, dropped connections, 408/409/429 and 5xx
    (including 529 overloaded). Bad requests, auth problems and spend limits are not retried."""
    if isinstance(error, _NEVER_RETRY):
        return False
    if isinstance(error, (anthropic.APIConnectionError, TimeoutError)):
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def _retry_after(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:  # an HTTP date: ignore it and use our own backoff
        return None


@dataclass(frozen=True)
class RetryPolicy:
    """How to retry a model call.

    `max_attempts` counts the first try. Waits double each time from `base_delay_s` up to
    `max_delay_s`, plus up to `jitter` (a fraction) extra so many clients do not retry in
    step, never beyond `max_delay_s`. A `retry-after` header is respected; if it asks for more
    than `max_delay_s` we give up at once instead of retrying too early.
    `attempt_timeout_s` limits one attempt; a hung call counts as a retryable failure.

    Build the client with `make_async_client()` so the SDK's own retries are off. Otherwise
    the two layers multiply."""

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.25
    attempt_timeout_s: float | None = 60.0
    sleep: Callable[[float], Awaitable[Any]] = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def too_long(self, error: BaseException) -> bool:
        """True if the server asked us to wait longer than `max_delay_s`."""
        hinted = _retry_after(error)
        return hinted is not None and hinted > self.max_delay_s

    def delay(self, attempt: int, error: BaseException) -> float:
        wait = self.base_delay_s * 2 ** (attempt - 1)
        hinted = _retry_after(error)
        if hinted is not None:
            wait = max(wait, hinted)
        return min(wait * (1 + random.random() * self.jitter), self.max_delay_s)
