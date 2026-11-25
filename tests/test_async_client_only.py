"""Service code uses the async Anthropic client only. A sync client blocks the event loop,
which would stall every other conversation while one waits for the model."""

import ast
from pathlib import Path

import pytest
from anthropic import Anthropic, AnthropicBedrock, AsyncAnthropic

from agentcore.loop import Agent
from agentcore.types import Registry
from tests.fake_claude import FakeClaude

ROOT = Path(__file__).resolve().parent.parent
LEGACY: set[str] = set()  # files exempt from the scan (none: the naive agent is gone)


def _agent(client):
    return Agent(client=client, model="m", system="s", tools=Registry())


@pytest.mark.parametrize("sync_cls", [Anthropic, AnthropicBedrock])
def test_agent_rejects_sync_anthropic_clients(sync_cls):
    client = object.__new__(sync_cls)  # no key or network needed
    with pytest.raises(TypeError, match="AsyncAnthropic"):
        _agent(client)


def test_agent_accepts_the_async_client_and_test_fakes():
    _agent(object.__new__(AsyncAnthropic))
    _agent(FakeClaude([]))


def _sync_client_uses(path: Path) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and node.attr in {"Anthropic", "AnthropicBedrock", "AnthropicVertex"}:
            lines.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module == "anthropic":
            lines += [node.lineno for a in node.names if a.name in {"Anthropic", "AnthropicBedrock", "AnthropicVertex"}]
    return lines


def test_no_sync_anthropic_client_in_service_code():
    offenders = {}
    for folder in ("agentcore", "agent_service"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.name in LEGACY:
                continue
            if lines := _sync_client_uses(path):
                offenders[f"{folder}/{path.name}"] = lines
    assert not offenders, f"sync Anthropic client used in: {offenders}"
