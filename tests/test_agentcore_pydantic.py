import asyncio
import datetime as dt

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agentcore.hooks import Hook, Modify
from agentcore.loop import Agent
from agentcore.types import Registry, RunContext, Tool
from tests.fake_claude import FakeClaude, text, tool_use


class Args(BaseModel):
    """Add things up."""

    count: int = Field(description="How many")
    label: str = "none"
    when: dt.date | None = None


def build(script, **kw):
    seen = []
    reg = Registry()

    @reg.register(Args, name="add", description="Add things.")
    async def add(args: Args, ctx: RunContext):
        seen.append(args)
        return {"total": args.count}

    fake = FakeClaude(script)
    return Agent(client=fake, model="m", system="s", tools=reg, **kw), fake, seen


def run(agent, ctx=None):
    return asyncio.run(agent.run([{"role": "user", "content": "hi"}], ctx))


def sent_block(fake, i=1):
    return fake.messages.calls[i]["messages"][-1]["content"][0]


# ------------------------------------------------------------------ building the tool


def test_schema_name_and_description_come_from_the_function_and_model():
    async def add_items(args: Args, ctx):
        """Add items to the list.

        Longer notes."""

    tool = Tool.from_function(add_items, Args)
    api = tool.to_api()
    assert api["name"] == "add_items"
    assert api["description"].startswith("Add items to the list.")
    schema = api["input_schema"]
    assert schema["type"] == "object" and schema["required"] == ["count"]
    assert schema["properties"]["count"]["description"] == "How many"
    assert schema["additionalProperties"] is False
    assert "title" not in schema and "title" not in schema["properties"]["count"]


def test_explicit_name_and_description_win():
    tool = Tool.from_function(lambda a, c: None, Args, name="x", description="Does x.")
    assert tool.name == "x" and tool.description == "Does x."


def test_missing_description_is_an_error_at_build_time():
    def nodoc(args, ctx):
        pass

    with pytest.raises(ValueError):
        Tool.from_function(nodoc, Args)


def test_a_property_called_title_survives_schema_cleanup():
    class T(BaseModel):
        title: str

    schema = Tool.from_function(lambda a, c: None, T, name="t", description="d").to_api()["input_schema"]
    assert "title" in schema["properties"] and schema["required"] == ["title"]


def test_register_returns_the_function_and_adds_the_tool():
    reg = Registry()

    @reg.register(Args, description="d")
    def thing(args, ctx):
        return 1

    assert thing(None, None) == 1 and "thing" in reg and reg.get("thing").writes is False


def test_register_passes_the_writes_flag():
    reg = Registry()
    reg.register(Args, name="w", description="d", writes=True)(lambda a, c: None)
    assert reg.get("w").writes is True


def test_users_model_class_is_not_changed():
    class Loose(BaseModel):
        a: int

    Tool.from_function(lambda a, c: None, Loose, name="t", description="d")
    assert Loose.model_config.get("extra") != "forbid"
    assert Loose(a=1, b=2).a == 1  # still tolerant outside the tool


# ------------------------------------------------------------------ running it


def test_handler_receives_a_validated_model_instance():
    agent, fake, seen = build([tool_use("add", {"count": 3, "when": "2026-12-04"}), text("ok")])
    run(agent)
    assert isinstance(seen[0], Args)
    assert seen[0].count == 3 and seen[0].label == "none" and seen[0].when == dt.date(2026, 12, 4)
    assert sent_block(fake)["is_error"] is False


def test_handler_may_take_only_the_args():
    reg = Registry()
    reg.register(Args, name="solo", description="d")(lambda args: {"n": args.count})
    fake = FakeClaude([tool_use("solo", {"count": 2}), text("ok")])
    run(Agent(client=fake, model="m", system="s", tools=reg))
    assert '"n": 2' in sent_block(fake)["content"]


@pytest.mark.parametrize(
    "bad_input, expect_in_message",
    [
        ({"count": "abc"}, "count"),  # wrong type
        ({"count": "3"}, "count"),  # strict: a string is not an int
        ({"count": 1.5}, "count"),
        ({"count": True}, "count"),
        ({}, "count"),  # missing field
        ({"count": 1, "bogus": 1}, "bogus"),  # extra field
        ({"count": 1, "when": "not-a-date"}, "when"),
        ({"count": 1, "label": 5}, "label"),
    ],
)
def test_bad_input_comes_back_as_an_error_result_and_the_handler_never_runs(bad_input, expect_in_message):
    agent, fake, seen = build([tool_use("add", bad_input), text("sorry")])
    res = run(agent)
    block = sent_block(fake)
    assert seen == []
    assert block["is_error"] is True
    assert block["content"].startswith("Invalid input")
    assert expect_in_message in block["content"]
    assert res.tool_records[0].outcome == "error" and res.text == "sorry"


def test_all_problems_are_reported_together_so_the_model_can_fix_them_in_one_go():
    agent, fake, _ = build([tool_use("add", {"count": "x", "bogus": 1}), text("ok")])
    run(agent)
    content = sent_block(fake)["content"]
    assert "count" in content and "bogus" in content


def test_nested_errors_name_the_path():
    class Item(BaseModel):
        qty: int

    class Order(BaseModel):
        items: list[Item]

    reg = Registry()
    reg.register(Order, name="order", description="d")(lambda a, c: "ok")
    fake = FakeClaude([tool_use("order", {"items": [{"qty": 1}, {"qty": "many"}]}), text("ok")])
    run(Agent(client=fake, model="m", system="s", tools=reg))
    assert "items.1.qty" in sent_block(fake)["content"]


def test_a_model_that_already_forbids_extras_works_too():
    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int

    reg = Registry()
    reg.register(Strict, name="s", description="d")(lambda a, c: "ok")
    fake = FakeClaude([tool_use("s", {"a": 1, "b": 2}), text("ok")])
    run(Agent(client=fake, model="m", system="s", tools=reg))
    assert sent_block(fake)["is_error"] is True


def test_hooks_see_the_raw_dict_and_modified_input_is_validated_again():
    seen_inputs = []

    class Spy(Hook):
        def before_tool(self, call, ctx):
            seen_inputs.append(call.input)
            return Modify({"count": "oops"})

    agent, fake, handler_calls = build([tool_use("add", {"count": 1}), text("ok")], hooks=[Spy()])
    run(agent)
    assert seen_inputs == [{"count": 1}]
    assert handler_calls == [] and sent_block(fake)["is_error"] is True


def test_a_valid_modify_reaches_the_handler_as_a_model():
    class Cap(Hook):
        def before_tool(self, call, ctx):
            return Modify({**call.input, "count": 1})

    agent, _, seen = build([tool_use("add", {"count": 50}), text("ok")], hooks=[Cap()])
    run(agent)
    assert seen[0].count == 1


def test_a_handler_error_is_still_an_error_result():
    reg = Registry()

    @reg.register(Args, name="boom", description="d")
    def boom(args, ctx):
        raise RuntimeError("kaput")

    fake = FakeClaude([tool_use("boom", {"count": 1}), text("ok")])
    run(Agent(client=fake, model="m", system="s", tools=reg))
    assert "kaput" in sent_block(fake)["content"]


def test_ctx_is_only_passed_when_the_function_asks_for_it():
    def with_option(args, top_k=5):
        return top_k

    def with_named_ctx(args, ctx=None):
        return ctx

    def with_required_second(args, context):
        return context

    def only_args(args):
        return "solo"

    def build(fn):
        return Tool.from_function(fn, Args, name="t", description="d").handler

    assert build(with_option)("A", "CTX") == 5  # an option, not the context
    assert build(with_named_ctx)("A", "CTX") == "CTX"
    assert build(with_required_second)("A", "CTX") == "CTX"
    assert build(only_args)("A", "CTX") == "solo"
