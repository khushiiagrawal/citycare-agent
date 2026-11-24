"""Small helpers shared inside agentcore."""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Hooks, handlers and clients may be sync or async. Await only if needed."""
    return await value if inspect.isawaitable(value) else value
