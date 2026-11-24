import asyncio

from agentcore.hooks import Allow, Confirm, ConfirmGate, Deny, Hook, Modify
from agentcore.loop import Agent
from agentcore.types import Registry, RunContext, Tool, ToolResult
from tests.fake_claude import FakeClaude, text, tool_use, tool_uses

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def make_tool(name="t", calls=None, writes=False):
    async def handler(input, ctx):
        if calls is not None:
            calls.append((name, input))
        return {"ran": name, "input": input}

    return Tool(name=name, description=name, input_schema=SCHEMA, handler=handler, writes=writes)


def make_agent(script, tools, hooks=(), **kw):
    fake = FakeClaude(script)
    return Agent(client=fake, model="m", system="s", tools=Registry(tools), hooks=list(hooks), **kw), fake


def run(agent, messages=None, ctx=None):
    return asyncio.run(agent.run(messages or [{"role": "user", "content": "hi"}], ctx))


def results_sent(fake, call_index):
    return fake.messages.calls[call_index]["messages"][-1]["content"]


# ------------------------------------------------------------------ before_tool


def test_deny_blocks_the_tool_and_tells_the_model_why():
    calls = []

    class No(Hook):
        def before_tool(self, call, ctx):
            return Deny("not allowed here")

    agent, fake = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool(calls=calls)], [No()])
    res = run(agent)
    block = results_sent(fake, 1)[0]
    assert calls == []
    assert block["is_error"] is True and "not allowed here" in block["content"]
    assert res.tool_records[0].outcome == "denied" and res.text == "ok"


def test_modify_changes_what_the_handler_receives():
    calls = []

    class Force(Hook):
        async def before_tool(self, call, ctx):
            return Modify({"x": 99})

    agent, _ = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool(calls=calls)], [Force()])
    res = run(agent)
    assert calls == [("t", {"x": 99})]
    assert res.tool_records[0].call.input == {"x": 99}


def test_modify_that_breaks_the_schema_is_rejected():
    calls = []

    class Break(Hook):
        def before_tool(self, call, ctx):
            return Modify({})

    agent, _ = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool(calls=calls)], [Break()])
    res = run(agent)
    assert calls == [] and res.tool_records[0].outcome == "error"


def test_hooks_run_in_order_modify_chains_and_deny_short_circuits():
    seen = []

    class A(Hook):
        def before_tool(self, call, ctx):
            seen.append(("A", call.input))
            return Modify({"x": 2})

    class B(Hook):
        def before_tool(self, call, ctx):
            seen.append(("B", call.input))
            return Deny("stop")

    class C(Hook):
        def before_tool(self, call, ctx):
            seen.append(("C", call.input))

    agent, _ = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool()], [A(), B(), C()])
    run(agent)
    assert seen == [("A", {"x": 1}), ("B", {"x": 2})]


def test_none_and_allow_both_mean_continue():
    calls = []

    class Quiet(Hook):
        def before_tool(self, call, ctx):
            return None

    class Yes(Hook):
        def before_tool(self, call, ctx):
            return Allow()

    agent, _ = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool(calls=calls)], [Quiet(), Yes()])
    run(agent)
    assert len(calls) == 1


def test_hook_sees_the_run_context():
    class Mark(Hook):
        def before_tool(self, call, ctx):
            ctx.state["seen"] = call.name

    ctx = RunContext()
    agent, _ = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool()], [Mark()])
    run(agent, ctx=ctx)
    assert ctx.state["seen"] == "t"


# ------------------------------------------------------------------ after_tool


def test_after_tool_can_rewrite_results_including_errors():
    async def boom(input, ctx):
        raise RuntimeError("kaput")

    class Redact(Hook):
        def after_tool(self, call, result, ctx):
            return ToolResult("redacted", result.is_error)

    bad = Tool(name="t", description="t", input_schema=SCHEMA, handler=boom)
    agent, fake = make_agent([tool_use("t", {"x": 1}), text("ok")], [bad], [Redact()])
    run(agent)
    block = results_sent(fake, 1)[0]
    assert block["content"] == "redacted" and block["is_error"] is True


def test_after_tool_is_not_called_for_denied_calls():
    called = []

    class No(Hook):
        def before_tool(self, call, ctx):
            return Deny("no")

        def after_tool(self, call, result, ctx):
            called.append(1)

    agent, _ = make_agent([tool_use("t", {"x": 1}), text("ok")], [make_tool()], [No()])
    run(agent)
    assert called == []


# ------------------------------------------------------------------ before_model / on_finish


def test_before_model_replaces_messages_for_that_call_only():
    class Trim(Hook):
        def before_model(self, messages, ctx):
            return messages[-1:]

    agent, fake = make_agent([text("ok")], [], [Trim()])
    msgs = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "new"}]
    res = run(agent, msgs)
    assert fake.messages.calls[0]["messages"] == [{"role": "user", "content": "new"}]
    assert len(msgs) == 3 and res.text == "ok"


def test_on_finish_runs_once_per_run_on_every_exit_path():
    finished = []

    class Watch(Hook):
        def on_finish(self, result, ctx):
            finished.append(result.stop_reason)

    agent, _ = make_agent([text("a")], [], [Watch()])
    run(agent)
    agent, _ = make_agent([tool_use("t", {"x": 1})], [make_tool()], [Watch()], max_turns=1)
    run(agent)
    gate = ConfirmGate({"t": lambda call, ctx: "sure?"})
    agent, _ = make_agent([tool_use("t", {"x": 1})], [make_tool()], [Watch(), gate])
    run(agent)
    assert finished == ["end_turn", "max_turns", "paused"]


# ------------------------------------------------------------------ confirm outcome


class AlwaysConfirm(Hook):
    def before_tool(self, call, ctx):
        return Confirm("Book Dr X at 5pm?")


def test_confirm_pauses_the_run_and_returns_the_code_written_prompt():
    calls = []
    agent, fake = make_agent([tool_use("t", {"x": 1})], [make_tool(calls=calls)], [AlwaysConfirm()])
    ctx = RunContext()
    res = run(agent, ctx=ctx)
    assert calls == []
    assert res.stop_reason == "paused" and res.text == "Book Dr X at 5pm?"
    assert res.model_calls == 1  # the model is not asked to word the question
    assert res.tool_records[0].outcome == "paused"
    assert [p.prompt for p in res.pending] == ["Book Dr X at 5pm?"] and len(ctx.pending) == 1
    roles = [m["role"] for m in res.messages]
    assert roles == ["assistant", "user", "assistant"]  # valid history for the next turn
    assert res.messages[1]["content"][0]["is_error"] is True
    assert res.messages[2]["content"] == [{"type": "text", "text": "Book Dr X at 5pm?"}]
    assert len(fake.messages.script) == 0


def test_other_tool_calls_in_the_same_response_still_run():
    calls = []

    class OnlyW(Hook):
        def before_tool(self, call, ctx):
            return Confirm("ok?") if call.name == "w" else None

    tools = [make_tool("w", calls), make_tool("r", calls)]
    agent, _ = make_agent([tool_uses([("w", {"x": 1}), ("r", {"x": 2})])], tools, [OnlyW()])
    res = run(agent)
    assert calls == [("r", {"x": 2})] and res.stop_reason == "paused"


def test_a_paused_write_does_not_use_up_the_write_cap():
    calls = []

    class OnlyW1(Hook):
        def before_tool(self, call, ctx):
            return Confirm("ok?") if call.name == "w1" else None

    tools = [make_tool("w1", calls, writes=True), make_tool("w2", calls, writes=True)]
    agent, _ = make_agent([tool_uses([("w1", {"x": 1}), ("w2", {"x": 2})])], tools, [OnlyW1()], max_writes_per_run=1)
    run(agent)
    assert calls == [("w2", {"x": 2})]


# ------------------------------------------------------------------ ConfirmGate


def gate_for(*names, prompt="Confirm?"):
    return ConfirmGate({n: (lambda call, ctx: prompt) for n in names})


def two_turns(script, tools, hooks):
    """Turn 1 pauses. Turn 2 the user says yes and the model calls the tool again."""
    agent, fake = make_agent(script, tools, hooks)
    ctx = RunContext()
    first = run(agent, [{"role": "user", "content": "book it"}], ctx)
    history = [{"role": "user", "content": "book it"}, *first.messages, {"role": "user", "content": "yes"}]
    second = run(agent, history, ctx)
    return first, second, ctx, fake


def test_gate_runs_the_tool_only_after_the_user_confirms_in_a_later_turn():
    calls = []
    script = [tool_use("w", {"x": 1}), tool_use("w", {"x": 1}), text("Booked.")]
    first, second, ctx, _ = two_turns(script, [make_tool("w", calls, writes=True)], [gate_for("w")])
    assert first.stop_reason == "paused" and calls == [("w", {"x": 1})]
    assert second.stop_reason == "end_turn" and second.text == "Booked." and ctx.pending == []


def test_gate_asks_again_if_the_input_changed():
    calls = []
    script = [tool_use("w", {"x": 1}), tool_use("w", {"x": 2})]
    _, second, ctx, _ = two_turns(script, [make_tool("w", calls)], [gate_for("w")])
    assert calls == [] and second.stop_reason == "paused"
    assert [p.input for p in ctx.pending] == [{"x": 2}]


def test_gate_confirmation_is_used_up_after_one_run():
    calls = []
    script = [tool_use("w", {"x": 1}), tool_use("w", {"x": 1}), text("done"), tool_use("w", {"x": 1})]
    agent, _ = make_agent(script, [make_tool("w", calls)], [gate_for("w")])
    ctx = RunContext()
    history = [{"role": "user", "content": "a"}]
    for msg in ["b", "c"]:
        res = run(agent, history, ctx)
        history = [*history, *res.messages, {"role": "user", "content": msg}]
    third = run(agent, history, ctx)
    assert len(calls) == 1 and third.stop_reason == "paused"


def test_gate_confirmation_expires_if_the_user_talks_about_something_else():
    calls = []
    script = [tool_use("w", {"x": 1}), text("sure"), tool_use("w", {"x": 1})]
    agent, _ = make_agent(script, [make_tool("w", calls)], [gate_for("w")])
    ctx = RunContext()
    h = [{"role": "user", "content": "a"}]
    r1 = run(agent, h, ctx)
    h = [*h, *r1.messages, {"role": "user", "content": "what are your timings?"}]
    r2 = run(agent, h, ctx)
    h = [*h, *r2.messages, {"role": "user", "content": "ok yes"}]
    r3 = run(agent, h, ctx)
    assert calls == [] and r3.stop_reason == "paused"


def test_gate_does_not_allow_in_the_same_turn_it_asked():
    calls = []
    agent, _ = make_agent([tool_uses([("w", {"x": 1}), ("w", {"x": 1})])], [make_tool("w", calls)], [gate_for("w")])
    ctx = RunContext()
    res = run(agent, ctx=ctx)
    assert calls == [] and len(ctx.pending) == 1  # duplicates collapse into one pending item
    assert res.text == "Confirm?"


def test_gate_ignores_tools_it_does_not_guard():
    calls = []
    agent, _ = make_agent([tool_use("r", {"x": 1}), text("ok")], [make_tool("r", calls)], [gate_for("w")])
    run(agent)
    assert len(calls) == 1


def test_gate_prompt_builder_can_deny_and_can_be_async():
    async def builder(call, ctx):
        return Deny("that is not yours")

    agent, fake = make_agent([tool_use("w", {"x": 1}), text("sorry")], [make_tool("w")], [ConfirmGate({"w": builder})])
    res = run(agent)
    assert res.tool_records[0].outcome == "denied" and res.pending == []
    assert "not yours" in results_sent(fake, 1)[0]["content"]


def test_gate_prompt_is_built_from_the_actual_call():
    gate = ConfirmGate({"w": lambda call, ctx: f"Book slot {call.input['x']}?"})
    agent, _ = make_agent([tool_use("w", {"x": 5})], [make_tool("w")], [gate])
    assert run(agent).text == "Book slot 5?"


def test_a_write_refused_by_the_cap_keeps_its_confirmation_for_the_next_message():
    from agentcore.session import InMemoryStore, run_in_session

    calls = []
    tools = [make_tool("w1", calls, writes=True), make_tool("w2", calls, writes=True)]
    both = tool_uses([("w1", {"x": 1}), ("w2", {"x": 2})])
    script = [both, both, text("first one done"), tool_use("w2", {"x": 2}), text("second one done")]
    gate = gate_for("w1", "w2")
    agent, _ = make_agent(script, tools, [gate], max_writes_per_run=1)

    async def go():
        store = InMemoryStore()
        asked = await run_in_session(agent, store, "k", "book both")
        yes = await run_in_session(agent, store, "k", "yes to both")
        later = await run_in_session(agent, store, "k", "ok go on")
        return asked, yes, later

    asked, yes, later = asyncio.run(go())
    assert asked.stop_reason == "paused"
    assert calls == [("w1", {"x": 1}), ("w2", {"x": 2})]  # w2 ran on the next message without asking again
    assert yes.tool_records[1].outcome == "denied" and later.text == "second one done"
