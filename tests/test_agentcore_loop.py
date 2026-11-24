import asyncio
import json

import pytest

from anthropic.types import TextBlock, ToolUseBlock

from agentcore.loop import Agent
from agentcore.types import Registry, RunContext, Tool
from tests.fake_claude import FakeClaude, _message, text, tool_use, tool_uses

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def make_tool(name="echo", handler=None, writes=False):
    async def default(input, ctx):
        return {"echo": input["x"]}

    return Tool(name=name, description=name, input_schema=SCHEMA, handler=handler or default, writes=writes)


def make_agent(script, tools=(), **kw):
    fake = FakeClaude(script)
    agent = Agent(client=fake, model="m", system="sys", tools=Registry(tools), **kw)
    return agent, fake


def user(msg="hi"):
    return [{"role": "user", "content": msg}]


def run(agent, messages=None, ctx=None):
    return asyncio.run(agent.run(messages or user(), ctx))


def tool_result_block(request_messages):
    last = request_messages[-1]
    assert last["role"] == "user"
    return last["content"]


def test_plain_reply():
    agent, fake = make_agent([text("hello", input_tokens=10, output_tokens=3)])
    msgs = user()
    res = run(agent, msgs)
    assert res.text == "hello" and res.stop_reason == "end_turn"
    assert res.model_calls == 1
    assert res.usage == {"input_tokens": 10, "output_tokens": 3}
    assert [m["role"] for m in res.messages] == ["assistant"]
    assert msgs == user()  # caller's list is not mutated
    req = fake.messages.calls[0]
    assert req["model"] == "m" and req["system"] == "sys" and req["max_tokens"] == 1024
    assert "tools" not in req


def test_tools_are_sent_when_registered():
    agent, fake = make_agent([text("ok")], tools=[make_tool()])
    run(agent)
    assert fake.messages.calls[0]["tools"][0]["name"] == "echo"


def test_request_params_are_passed_through():
    agent, fake = make_agent([text("ok")], request_params={"temperature": 0})
    run(agent)
    assert fake.messages.calls[0]["temperature"] == 0


def test_tool_call_then_reply():
    seen = {}

    async def handler(input, ctx):
        seen["input"], seen["ctx"] = input, ctx
        return {"echo": input["x"]}

    agent, fake = make_agent([tool_use("echo", {"x": 7}), text("done")], tools=[make_tool(handler=handler)])
    ctx = RunContext()
    res = run(agent, ctx=ctx)
    assert res.text == "done" and res.model_calls == 2
    assert seen["input"] == {"x": 7} and seen["ctx"] is ctx
    block = tool_result_block(fake.messages.calls[1]["messages"])[0]
    assert block["type"] == "tool_result" and block["is_error"] is False
    assert json.loads(block["content"]) == {"echo": 7}
    assert [r.outcome for r in res.tool_records] == ["ok"]
    # assistant tool_use is stored as a plain dict
    assistant = fake.messages.calls[1]["messages"][1]
    assert assistant["role"] == "assistant" and assistant["content"][0]["type"] == "tool_use"
    assert isinstance(assistant["content"][0], dict)


def test_sync_handler_and_string_result():
    tool = make_tool(handler=lambda input, ctx: "plain text")
    agent, fake = make_agent([tool_use("echo", {"x": 1}), text("done")], tools=[tool])
    run(agent)
    assert tool_result_block(fake.messages.calls[1]["messages"])[0]["content"] == "plain text"


def test_parallel_tool_calls_run_concurrently_and_keep_order():
    started = {"a": asyncio.Event(), "b": asyncio.Event()}

    def waiting(me, other):
        async def handler(input, ctx):
            started[me].set()
            await asyncio.wait_for(started[other].wait(), timeout=1)  # deadlocks if sequential
            return me

        return handler

    tools = [make_tool("a", waiting("a", "b")), make_tool("b", waiting("b", "a"))]
    agent, fake = make_agent([tool_uses([("a", {"x": 1}), ("b", {"x": 1})]), text("done")], tools=tools)

    async def go():
        started["a"], started["b"] = asyncio.Event(), asyncio.Event()
        return await agent.run(user())

    asyncio.run(go())
    blocks = tool_result_block(fake.messages.calls[1]["messages"])
    assert [b["content"] for b in blocks] == ["a", "b"]
    assert blocks[0]["tool_use_id"] != blocks[1]["tool_use_id"]


def test_unknown_tool_is_an_error_result_not_a_crash():
    agent, fake = make_agent([tool_use("nope", {"x": 1}), text("sorry")], tools=[make_tool()])
    res = run(agent)
    block = tool_result_block(fake.messages.calls[1]["messages"])[0]
    assert block["is_error"] is True and "nope" in block["content"]
    assert res.tool_records[0].outcome == "unknown" and res.text == "sorry"


def test_handler_exception_becomes_error_result():
    async def boom(input, ctx):
        raise RuntimeError("kaput")

    agent, fake = make_agent([tool_use("echo", {"x": 1}), text("sorry")], tools=[make_tool(handler=boom)])
    res = run(agent)
    block = tool_result_block(fake.messages.calls[1]["messages"])[0]
    assert block["is_error"] is True and "kaput" in block["content"]
    assert res.tool_records[0].outcome == "error"


def test_missing_required_field_is_rejected_before_the_handler_runs():
    called = []
    tool = make_tool(handler=lambda input, ctx: called.append(1))
    agent, fake = make_agent([tool_use("echo", {}), text("ok")], tools=[tool])
    res = run(agent)
    block = tool_result_block(fake.messages.calls[1]["messages"])[0]
    assert block["is_error"] is True and "x" in block["content"]
    assert called == [] and res.tool_records[0].outcome == "error"


def test_max_turns_stops_with_valid_history():
    agent, fake = make_agent(
        [tool_use("echo", {"x": 1}), tool_use("echo", {"x": 2}), text("never")], tools=[make_tool()], max_turns=2
    )
    res = run(agent)
    assert res.stop_reason == "max_turns" and res.model_calls == 2
    assert len(fake.messages.script) == 1  # third reply never requested
    assert res.messages[-1]["role"] == "user"  # every tool_use has its tool_result


def test_max_tokens_drops_half_written_tool_calls_from_history():
    cut = _message(
        [TextBlock(type="text", text="Let me"), ToolUseBlock(type="tool_use", id="t1", name="echo", input={})],
        "max_tokens",
    )
    agent, _ = make_agent([cut], tools=[make_tool()])
    res = run(agent)
    assert res.stop_reason == "max_tokens" and res.text == "Let me"
    assert all(b["type"] != "tool_use" for m in res.messages for b in m["content"])


def test_max_tokens_with_only_a_tool_call_adds_no_message():
    cut = _message([ToolUseBlock(type="tool_use", id="t1", name="echo", input={})], "max_tokens")
    agent, _ = make_agent([cut], tools=[make_tool()])
    res = run(agent)
    assert res.stop_reason == "max_tokens" and res.text == "" and res.messages == []


def test_refusal_is_reported():
    agent, _ = make_agent([text("I can't help with that.", stop_reason="refusal")])
    res = run(agent)
    assert res.stop_reason == "refusal" and res.text == "I can't help with that."


def test_multiple_text_blocks_are_joined():
    two = _message([TextBlock(type="text", text="Hello."), TextBlock(type="text", text="How can I help?")], "end_turn")
    agent, _ = make_agent([two])
    assert run(agent).text == "Hello.\nHow can I help?"


def test_max_writes_per_run_denies_the_second_write():
    calls = []

    async def write(input, ctx):
        calls.append(input["x"])
        return "written"

    tools = [make_tool("w", write, writes=True), make_tool("r")]
    script = [tool_uses([("w", {"x": 1}), ("w", {"x": 2}), ("r", {"x": 3})]), tool_use("w", {"x": 4}), text("done")]
    agent, fake = make_agent(script, tools=tools, max_writes_per_run=1)
    res = run(agent)
    assert calls == [1]
    blocks = tool_result_block(fake.messages.calls[1]["messages"])
    assert [b["is_error"] for b in blocks] == [False, True, False]
    assert "one change" in blocks[1]["content"].lower()
    later = tool_result_block(fake.messages.calls[2]["messages"])[0]
    assert later["is_error"] is True  # the cap spans the whole run, not one response
    assert [r.outcome for r in res.tool_records] == ["ok", "denied", "ok", "denied"]


def test_writes_are_uncapped_by_default():
    calls = []
    tool = make_tool("w", lambda input, ctx: calls.append(1), writes=True)
    agent, _ = make_agent([tool_uses([("w", {"x": 1}), ("w", {"x": 2})]), text("ok")], tools=[tool])
    run(agent)
    assert len(calls) == 2


def test_async_client_is_supported():
    class AsyncMessages:
        def __init__(self, reply):
            self.reply = reply

        async def create(self, **kwargs):
            return self.reply

    class AsyncClient:
        def __init__(self, reply):
            self.messages = AsyncMessages(reply)

    agent = Agent(client=AsyncClient(text("async hi")), model="m", system="s", tools=Registry())
    assert run(agent).text == "async hi"


def test_client_errors_propagate():
    agent, _ = make_agent([ConnectionError("down")])
    with pytest.raises(ConnectionError):
        run(agent)


def test_turn_counter_goes_up_each_run():
    agent, _ = make_agent([text("a"), text("b")])
    ctx = RunContext()
    run(agent, ctx=ctx)
    run(agent, ctx=ctx)
    assert ctx.turn == 2
