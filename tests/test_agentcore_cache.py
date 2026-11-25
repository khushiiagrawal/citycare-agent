"""Prompt caching in the loop (system prompt and tool list), and state shared between sessions."""

import asyncio

from pydantic import BaseModel

from agentcore.loop import Agent
from agentcore.session import InMemoryStore, run_in_session
from agentcore.types import Registry, RunContext
from tests.fake_claude import FakeClaude, text, tool_use


class Args(BaseModel):
    x: int = 0


def registry():
    reg = Registry()

    @reg.register(Args, name="first", description="one")
    def first(args, ctx):
        return {"ok": 1}

    @reg.register(Args, name="second", description="two")
    def second(args, ctx):
        return {"ok": 2}

    return reg


def run(agent, message="hi", ctx=None):
    return asyncio.run(agent.run([{"role": "user", "content": message}], ctx or RunContext(key="k")))


def test_by_default_nothing_is_marked_for_caching():
    fake = FakeClaude([text("hi")])
    run(Agent(client=fake, model="m", system="SYS", tools=registry()))
    call = fake.messages.calls[0]
    assert call["system"] == "SYS"
    assert not any("cache_control" in t for t in call["tools"])


def test_with_caching_the_system_prompt_and_the_last_tool_carry_a_cache_breakpoint():
    fake = FakeClaude([text("hi")])
    run(Agent(client=fake, model="m", system="SYS", tools=registry(), cache=True))
    call = fake.messages.calls[0]
    assert call["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert [("cache_control" in t) for t in call["tools"]] == [False, True]
    assert call["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_caching_does_not_change_the_tool_definitions_between_calls():
    fake = FakeClaude([tool_use("first", {}), text("done")])
    agent = Agent(client=fake, model="m", system="SYS", tools=registry(), cache=True)
    run(agent)
    first, second = fake.messages.calls
    assert first["tools"] == second["tools"] and first["system"] == second["system"]
    assert "cache_control" not in agent.tools.api_schemas()[-1]  # the registry itself is untouched


def test_caching_with_no_tools_marks_only_the_system_prompt():
    fake = FakeClaude([text("hi")])
    run(Agent(client=fake, model="m", system="SYS", cache=True))
    call = fake.messages.calls[0]
    assert "tools" not in call and call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_tokens_from_the_response_are_priced_and_counted():
    from agentcore.trace import Price

    reply = text("hi")
    reply.usage.cache_read_input_tokens = 1000
    reply.usage.cache_creation_input_tokens = 500
    result = run(Agent(client=FakeClaude([reply]), model="m", system="S", cache=True, price=Price(1.0, 2.0)))
    assert (result.cache_read_tokens, result.cache_write_tokens) == (1000, 500)
    assert abs(result.cost_usd - (100 * 1.0 + 20 * 2.0 + 1000 * 0.1 + 500 * 1.25) / 1e6) < 1e-12


# ------------------------------------------------------------------ state shared by reference


def test_sessions_can_share_one_state_dict():
    shared = {"name": "Priya"}
    store = InMemoryStore()
    seen = []

    class Spy(Agent):
        async def run(self, messages, ctx=None):
            seen.append(ctx.state)
            ctx.state["touched_by"] = self.name
            return await super().run(messages, ctx)

    a = Spy(client=FakeClaude([text("a")]), model="m", system="s", name="a")
    b = Spy(client=FakeClaude([text("b")]), model="m", system="s", name="b")

    async def go():
        await run_in_session(a, store, "chat::a", "hi", state=shared)
        await run_in_session(b, store, "chat::b", "hi", state=shared)

    asyncio.run(go())
    assert seen[0] is shared and seen[1] is shared
    assert shared == {"name": "Priya", "touched_by": "b"}


def test_without_a_state_argument_each_session_keeps_its_own():
    store = InMemoryStore()
    agent = Agent(client=FakeClaude([text("a"), text("b")]), model="m", system="s")

    async def go():
        await run_in_session(agent, store, "one", "hi")
        await run_in_session(agent, store, "two", "hi")
        return (await store.load("one")).ctx.state is not (await store.load("two")).ctx.state

    assert asyncio.run(go())
