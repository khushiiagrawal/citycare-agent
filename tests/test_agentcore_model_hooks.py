import asyncio

from agentcore.hooks import Allow, Deny, Hook, with_text
from agentcore.loop import Agent
from agentcore.types import Registry, RunContext, Tool
from tests.fake_claude import FakeClaude, text, tool_use

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def tool(calls):
    return Tool(name="t", description="t", input_schema=SCHEMA, handler=lambda i, c: calls.append(i) or "ran")


def make(script, hooks, tools=(), **kw):
    fake = FakeClaude(script)
    return Agent(client=fake, model="m", system="s", tools=Registry(tools), hooks=list(hooks), **kw), fake


def run(agent, ctx=None):
    return asyncio.run(agent.run([{"role": "user", "content": "hi"}], ctx))


# ------------------------------------------------------------------ before_model can block


class Block(Hook):
    def before_model(self, messages, ctx):
        return Deny("budget used up")


def test_before_model_deny_stops_the_run_without_calling_the_model():
    agent, fake = make([text("never")], [Block()])
    res = run(agent)
    assert res.stop_reason == "blocked" and res.reason == "budget used up"
    assert res.text == ""  # internal reasons are never in the user-facing text
    assert res.model_calls == 0 and res.messages == []
    assert len(fake.messages.calls) == 0


def test_a_block_in_the_middle_of_a_run_leaves_valid_history():
    class BlockSecond(Hook):
        def before_model(self, messages, ctx):
            return Deny("stop") if len(messages) > 1 else None

    calls = []
    agent, _ = make([tool_use("t", {"x": 1}), text("never")], [BlockSecond()], [tool(calls)])
    res = run(agent)
    assert res.stop_reason == "blocked" and res.model_calls == 1 and calls == [{"x": 1}]
    assert res.messages[-1]["role"] == "user"  # the tool_use still has its tool_result


def test_deny_short_circuits_and_replacements_chain():
    seen = []

    class Replace(Hook):
        def before_model(self, messages, ctx):
            seen.append(("replace", len(messages)))
            return [{"role": "user", "content": "short"}]

    class Check(Hook):
        def before_model(self, messages, ctx):
            seen.append(("check", messages[0]["content"]))
            return Deny("no")

    class Never(Hook):
        def before_model(self, messages, ctx):
            seen.append("never")

    agent, _ = make([text("x")], [Replace(), Check(), Never()])
    run(agent)
    assert seen == [("replace", 1), ("check", "short")]


def test_allow_and_none_from_before_model_mean_continue():
    class Yes(Hook):
        async def before_model(self, messages, ctx):
            return Allow()

    agent, fake = make([text("ok")], [Yes()])
    assert run(agent).text == "ok" and len(fake.messages.calls) == 1


def test_on_finish_still_runs_when_blocked():
    done = []

    class Watch(Hook):
        def on_finish(self, result, ctx):
            done.append(result.stop_reason)

    agent, _ = make([text("x")], [Block(), Watch()])
    run(agent)
    assert done == ["blocked"]


# ------------------------------------------------------------------ after_model


def test_after_model_can_look_at_the_reply_and_leave_it_alone():
    seen = []

    class Look(Hook):
        def after_model(self, response, ctx):
            seen.append((response.stop_reason, response.content[0].text))

    agent, _ = make([text("hello")], [Look()])
    assert run(agent).text == "hello" and seen == [("end_turn", "hello")]


def test_after_model_can_replace_the_reply():
    class Fix(Hook):
        async def after_model(self, response, ctx):
            return with_text(response, "Please call 108.") if "paracetamol" in response.content[0].text else None

    agent, _ = make([text("take a paracetamol")], [Fix()])
    res = run(agent)
    assert res.text == "Please call 108."
    assert res.messages[0]["content"] == [{"type": "text", "text": "Please call 108."}]  # history holds the safe text


def test_replacing_a_reply_does_not_hide_its_cost():
    class Fix(Hook):
        def after_model(self, response, ctx):
            return with_text(response, "replaced")

    agent, _ = make([text("original", input_tokens=50, output_tokens=7)], [Fix()])
    assert run(agent).usage == {"input_tokens": 50, "output_tokens": 7}


def test_replacing_a_tool_call_with_text_means_the_tool_never_runs():
    class NoTools(Hook):
        def after_model(self, response, ctx):
            return with_text(response, "I can't do that.") if response.stop_reason == "tool_use" else None

    calls = []
    agent, fake = make([tool_use("t", {"x": 1})], [NoTools()], [tool(calls)])
    res = run(agent)
    assert calls == [] and res.text == "I can't do that." and res.stop_reason == "end_turn"
    assert len(fake.messages.calls) == 1


def test_a_tool_call_in_a_replacement_still_goes_through_before_tool_hooks():
    class Swap(Hook):
        def after_model(self, response, ctx):
            return tool_use("t", {"x": 9}) if response.stop_reason == "end_turn" and "go" in response.content[0].text else None

        def before_tool(self, call, ctx):
            return Deny("blocked by policy")

    calls = []
    agent, _ = make([text("go"), text("done")], [Swap()], [tool(calls)])
    res = run(agent)
    assert calls == [] and res.tool_records[0].outcome == "denied"


def test_after_model_runs_on_every_model_call_in_order_and_chains():
    order = []

    class A(Hook):
        def after_model(self, response, ctx):
            order.append("A")
            return with_text(response, response.content[-1].text + "+A") if response.stop_reason == "end_turn" else None

    class B(Hook):
        def after_model(self, response, ctx):
            order.append("B")
            return with_text(response, response.content[-1].text + "+B") if response.stop_reason == "end_turn" else None

    calls = []
    agent, _ = make([tool_use("t", {"x": 1}), text("hi")], [A(), B()], [tool(calls)])
    res = run(agent)
    assert res.text == "hi+A+B" and order == ["A", "B", "A", "B"]


def test_with_text_keeps_usage_and_resets_stop_reason():
    original = text("cut off", stop_reason="max_tokens", input_tokens=11, output_tokens=5)
    new = with_text(original, "safe")
    assert new.stop_reason == "end_turn" and new.usage.input_tokens == 11
    assert [b.text for b in new.content] == ["safe"]
    assert original.content[0].text == "cut off"  # the original is not changed


def test_hooks_get_the_run_context():
    seen = {}

    class H(Hook):
        def before_model(self, messages, ctx):
            seen["before"] = ctx

        def after_model(self, response, ctx):
            seen["after"] = ctx

    ctx = RunContext()
    agent, _ = make([text("x")], [H()])
    run(agent, ctx)
    assert seen["before"] is ctx and seen["after"] is ctx
