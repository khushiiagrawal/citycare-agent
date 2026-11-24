import asyncio

import pytest
from anthropic.types import Usage

from agentcore.hooks import Confirm, Deny, Hook
from agentcore.loop import Agent
from agentcore.trace import Price, Tracer
from agentcore.types import Registry, RunContext, Tool
from tests.fake_claude import FakeClaude, text, tool_use

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
PRICE = Price(input_per_mtok=2.0, output_per_mtok=10.0)


def tool(handler=None, name="t"):
    return Tool(name=name, description=name, input_schema=SCHEMA, handler=handler or (lambda i, c: {"ok": i["x"]}))


def make(script, tools=(), **kw):
    fake = FakeClaude(script)
    return Agent(client=fake, model="m", system="s", tools=Registry(tools), **kw), fake


def run(agent, ctx=None):
    return asyncio.run(agent.run([{"role": "user", "content": "hi"}], ctx))


# ------------------------------------------------------------------ price and cost


def test_price_math_with_default_cache_rates():
    p = Price(input_per_mtok=2.0, output_per_mtok=10.0)
    assert p.cost(1_000_000, 0) == pytest.approx(2.0)
    assert p.cost(0, 1_000_000) == pytest.approx(10.0)
    assert p.cost(0, 0, cache_read=1_000_000) == pytest.approx(0.2)  # 10% of input price
    assert p.cost(0, 0, cache_write=1_000_000) == pytest.approx(2.5)  # 125% of input price


def test_price_cache_rates_can_be_overridden():
    p = Price(2.0, 10.0, cache_read_per_mtok=0.5, cache_write_per_mtok=3.0)
    assert p.cost(0, 0, cache_read=1_000_000, cache_write=1_000_000) == pytest.approx(3.5)


def test_result_cost_is_computed_from_the_price_table_you_pass_in():
    agent, _ = make([text("a", input_tokens=1000, output_tokens=200)], price=PRICE)
    assert run(agent).cost_usd == pytest.approx(0.004)


def test_cost_adds_up_over_model_calls_and_includes_cache_tokens():
    cached = text("done").model_copy(update={"usage": Usage(
        input_tokens=100, output_tokens=10, cache_read_input_tokens=1000, cache_creation_input_tokens=500)})
    agent, _ = make([tool_use("t", {"x": 1}, input_tokens=1000, output_tokens=0), cached], [tool()], price=PRICE)
    res = run(agent)
    expected = 0.002 + (100 * 2 + 10 * 10 + 1000 * 0.2 + 500 * 2.5) / 1_000_000
    assert res.cost_usd == pytest.approx(expected)
    assert res.cache_read_tokens == 1000 and res.cache_write_tokens == 500


def test_no_price_means_zero_cost_not_a_crash():
    agent, _ = make([text("a")])
    assert run(agent).cost_usd == 0.0


def test_session_totals_accumulate_on_the_context():
    agent, _ = make([text("a", input_tokens=1000, output_tokens=0), text("b", input_tokens=1000, output_tokens=0)],
                    price=PRICE)
    ctx = RunContext(key="s1")
    run(agent, ctx)
    run(agent, ctx)
    assert ctx.total_cost_usd == pytest.approx(0.004) and ctx.total_tokens == 2000


# ------------------------------------------------------------------ traces


def test_trace_records_model_and_tool_events_in_order():
    tracer = Tracer()
    agent, _ = make([tool_use("t", {"x": 5}, input_tokens=1000, output_tokens=100), text("done")],
                    [tool()], name="booking", price=PRICE, tracer=tracer)
    res = run(agent, RunContext(key="s1"))
    [trace] = tracer.traces("s1")
    assert trace.trace_id == res.trace_id and trace.agent == "booking" and trace.stop_reason == "end_turn"
    assert [(e.kind, e.name) for e in trace.events] == [("model", "m"), ("tool", "t"), ("model", "m")]
    model, tool_event, _ = trace.events
    assert model.data["input_tokens"] == 1000 and model.data["output_tokens"] == 100
    assert model.data["cost_usd"] == pytest.approx(0.003) and model.data["stop_reason"] == "tool_use"
    assert tool_event.data["input"] == {"x": 5} and tool_event.data["outcome"] == "ok"
    assert tool_event.data["is_error"] is False and "ok" in tool_event.data["output"]
    assert trace.cost_usd == pytest.approx(res.cost_usd)
    assert trace.input_tokens == 1000 + 100 and trace.output_tokens == 100 + 20


def test_events_have_timing():
    async def slow(input, ctx):
        await asyncio.sleep(0.05)
        return "ok"

    tracer = Tracer()
    agent, _ = make([tool_use("t", {"x": 1}), text("done")], [tool(slow)], tracer=tracer)
    run(agent, RunContext(key="s1"))
    [trace] = tracer.traces("s1")
    tool_event = next(e for e in trace.events if e.kind == "tool")
    assert tool_event.duration_s >= 0.04
    assert trace.duration_s >= tool_event.duration_s and trace.started_at > 0
    assert all(e.started_at >= trace.started_at for e in trace.events)


def test_denied_errored_and_unknown_tool_calls_are_traced_with_their_outcome():
    class No(Hook):
        def before_tool(self, call, ctx):
            return Deny("nope") if call.input["x"] == 1 else None

    def boom(input, ctx):
        raise RuntimeError("kaput")

    from tests.fake_claude import tool_uses

    tracer = Tracer()
    agent, _ = make([tool_uses([("t", {"x": 1}), ("t", {"x": 2}), ("zzz", {"x": 3})]), text("ok")],
                    [tool(boom)], hooks=[No()], tracer=tracer)
    run(agent, RunContext(key="s1"))
    events = [e for e in tracer.traces("s1")[0].events if e.kind == "tool"]
    assert [e.data["outcome"] for e in events] == ["denied", "error", "unknown"]
    assert "kaput" in events[1].data["output"]


def test_long_tool_output_is_shortened_in_the_trace_only():
    big = "x" * 5000
    tracer = Tracer()
    agent, fake = make([tool_use("t", {"x": 1}), text("ok")], [tool(lambda i, c: big)], tracer=tracer)
    run(agent, RunContext(key="s1"))
    event = next(e for e in tracer.traces("s1")[0].events if e.kind == "tool")
    assert len(event.data["output"]) < 700 and event.data["output"].endswith("…")
    assert fake.messages.calls[1]["messages"][-1]["content"][0]["content"] == big  # the model still got it all


def test_paused_and_blocked_runs_record_their_stop_reason():
    class Ask(Hook):
        def before_tool(self, call, ctx):
            return Confirm("sure?")

    class Stop(Hook):
        def before_model(self, messages, ctx):
            return Deny("no")

    tracer = Tracer()
    agent, _ = make([tool_use("t", {"x": 1})], [tool()], hooks=[Ask()], tracer=tracer)
    run(agent, RunContext(key="a"))
    agent, _ = make([], hooks=[Stop()], tracer=tracer)
    run(agent, RunContext(key="b"))
    assert tracer.traces("a")[0].stop_reason == "paused" and tracer.traces("b")[0].stop_reason == "blocked"


def test_a_failing_model_call_is_traced_then_raised():
    tracer = Tracer()
    agent, _ = make([ConnectionError("down")], tracer=tracer)
    with pytest.raises(ConnectionError):
        run(agent, RunContext(key="s1"))
    [trace] = tracer.traces("s1")
    event = trace.events[0]
    assert event.kind == "model" and "down" in event.data["error"] and trace.stop_reason == "error"


def test_every_run_has_a_trace_id_even_without_a_tracer():
    agent, _ = make([text("a"), text("b")])
    ids = {run(agent).trace_id, run(agent).trace_id}
    assert len(ids) == 2 and all(ids)


def test_tracer_keeps_runs_per_key_and_forgets_old_ones():
    tracer = Tracer(max_keys=2, max_traces_per_key=2)
    agent, _ = make([text("1"), text("2"), text("3"), text("4"), text("5")], tracer=tracer)
    for key in ["a", "a", "a", "b", "c"]:
        run(agent, RunContext(key=key))
    assert tracer.traces("a") == []  # oldest key was forgotten when "c" arrived
    assert len(tracer.traces("b")) == 1 and len(tracer.traces("c")) == 1


def test_tracer_caps_traces_per_key():
    tracer = Tracer(max_traces_per_key=2)
    agent, _ = make([text("1"), text("2"), text("3")], tracer=tracer)
    ids = [run(agent, RunContext(key="a")).trace_id for _ in range(3)]
    assert [t.trace_id for t in tracer.traces("a")] == ids[1:]


def test_unknown_key_has_no_traces():
    assert Tracer().traces("nope") == []


def test_trace_to_dict_is_json_friendly():
    import json

    tracer = Tracer()
    agent, _ = make([tool_use("t", {"x": 1}), text("ok")], [tool()], price=PRICE, tracer=tracer)
    run(agent, RunContext(key="s1"))
    data = tracer.traces("s1")[0].to_dict()
    json.dumps(data)
    assert {"trace_id", "agent", "stop_reason", "duration_s", "input_tokens", "output_tokens", "cost_usd",
            "events"} <= set(data)
    assert data["events"][0]["kind"] == "model" and "duration_s" in data["events"][0]


def test_a_limit_of_zero_keeps_no_traces():
    tracer = Tracer(max_traces_per_key=0)
    agent, _ = make([text("1")], tracer=tracer)
    run(agent, RunContext(key="a"))
    assert tracer.traces("a") == []


def test_tool_input_in_the_trace_is_a_copy_and_is_bounded():
    original = {"x": 1, "pad": "y" * 5000}
    tracer = Tracer()
    agent, _ = make([tool_use("t", original), text("ok")], [tool()], tracer=tracer)
    run(agent, RunContext(key="a"))
    event = next(e for e in tracer.traces("a")[0].events if e.kind == "tool")
    assert isinstance(event.data["input"], str) and len(event.data["input"]) <= 2001  # too big: shortened
    small = {"x": 1}
    agent, _ = make([tool_use("t", small), text("ok")], [tool()], tracer=tracer)
    run(agent, RunContext(key="b"))
    stored = next(e for e in tracer.traces("b")[0].events if e.kind == "tool").data["input"]
    assert stored == {"x": 1} and stored is not small
