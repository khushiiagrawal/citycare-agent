"""Test-wide safety net: tests never see an API key and can never reach the real API."""

import anthropic
import pytest


def _blocked(*_args, **_kwargs):
    raise RuntimeError("Tests must not create a real Anthropic client. Use tests/fake_claude.py.")


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", _blocked)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _blocked)
