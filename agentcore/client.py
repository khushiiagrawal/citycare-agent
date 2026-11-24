"""Building the model client."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic


def make_async_client(**kwargs: Any) -> AsyncAnthropic:
    """An AsyncAnthropic client with the SDK's own retries turned off (`max_retries=0`).

    Retrying is done once, by `RetryPolicy`, so each failure is traced and bounded.
    Any keyword argument (api_key, timeout, base_url...) is passed through."""
    kwargs.setdefault("max_retries", 0)
    return AsyncAnthropic(**kwargs)
