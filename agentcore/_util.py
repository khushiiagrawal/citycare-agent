"""Small helpers shared inside agentcore."""

from __future__ import annotations

import inspect
from typing import Any

from anthropic._base_client import SyncAPIClient  # base of every sync client; fails loudly if the SDK moves it


async def maybe_await(value: Any) -> Any:
    """Hooks, handlers and clients may be sync or async. Await only if needed."""
    return await value if inspect.isawaitable(value) else value


def reject_sync_client(client: Any) -> None:
    """Raise if `client` is a synchronous Anthropic client (Anthropic, AnthropicBedrock, ...).

    A sync client blocks the event loop while it waits for the model, so one slow reply
    would stall every other conversation. Pass `anthropic.AsyncAnthropic` instead."""
    if isinstance(client, SyncAPIClient):
        raise TypeError(
            f"{type(client).__name__} is a synchronous client and would block the event loop. "
            "Use anthropic.AsyncAnthropic."
        )
