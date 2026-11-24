import pytest

from agentcore.types import PendingConfirmation, Registry, RunContext, RunResult, Tool, ToolCall, ToolResult


async def _noop(input, ctx):
    return "ok"


def _tool(name="echo", **kw):
    return Tool(name=name, description="Echo.", input_schema={"type": "object", "properties": {}}, handler=_noop, **kw)


def test_tool_to_api_has_only_api_fields():
    api = _tool().to_api()
    assert set(api) == {"name", "description", "input_schema"}
    assert api["name"] == "echo"


def test_tool_writes_defaults_to_false():
    assert _tool().writes is False
    assert _tool(writes=True).writes is True


@pytest.mark.parametrize("bad", ["", "has space", "x" * 65, "naïve", "book\n"])
def test_tool_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        _tool(name=bad)


def test_registry_add_get_contains_iter():
    reg = Registry([_tool("a"), _tool("b")])
    assert "a" in reg and "z" not in reg
    assert reg.get("b").name == "b"
    assert reg.get("z") is None
    assert [t.name for t in reg] == ["a", "b"]
    assert len(reg) == 2


def test_registry_rejects_duplicates():
    reg = Registry([_tool("a")])
    with pytest.raises(ValueError):
        reg.add(_tool("a"))


def test_registry_subset_keeps_order_and_rejects_unknown():
    reg = Registry([_tool("a"), _tool("b"), _tool("c")])
    assert [t.name for t in reg.subset(["c", "a"])] == ["c", "a"]
    with pytest.raises(KeyError):
        reg.subset(["nope"])


def test_registry_api_schemas_is_stable_list():
    reg = Registry([_tool("a"), _tool("b")])
    assert [s["name"] for s in reg.api_schemas()] == ["a", "b"]


def test_run_context_defaults_are_not_shared():
    a, b = RunContext(), RunContext()
    a.state["x"] = 1
    a.pending.append(PendingConfirmation("t", {}, "p", 1))
    assert b.state == {} and b.pending == []
    assert a.turn == 0


def test_pending_confirmation_fingerprint_ignores_key_order():
    p1 = PendingConfirmation("t", {"a": 1, "b": 2}, "p", 1)
    p2 = PendingConfirmation("t", {"b": 2, "a": 1}, "p", 1)
    assert p1.fingerprint == p2.fingerprint
    assert p1.fingerprint != PendingConfirmation("t", {"a": 1, "b": 3}, "p", 1).fingerprint


def test_toolcall_and_result_basics():
    c = ToolCall(id="toolu_1", name="echo", input={"x": 1})
    r = ToolResult(content={"ok": True})
    assert c.name == "echo" and r.is_error is False


def test_run_result_defaults():
    r = RunResult(text="hi", stop_reason="end_turn")
    assert r.messages == [] and r.tool_records == [] and r.pending == []
    assert r.usage == {"input_tokens": 0, "output_tokens": 0}
