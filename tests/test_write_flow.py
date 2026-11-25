"""One confirmation only (the code read-back), a refused write that changed nothing keeps the write allowance,
a lost race is explained by code and the next time is offered in the same turn."""

import asyncio

import pytest

from agent_service.config import Settings
from agent_service.emergency import emergency_reply
from agent_service.guards import ClaimGuard
from agent_service.hospital import HospitalError, InvalidSlot, NotFound, OutcomeUnknown, SlotTaken, Unavailable
from agent_service.hooks import ReadBacks
from agent_service.prompts import billing_prompt, booking_prompt
from agent_service.service import FrontDesk
from agent_service.schemas import ChatRequest
from agent_service.tools import Hospital
from agentcore.loop import Agent
from agentcore.types import Registry, RunContext, Tool, ToolCall, ToolResult
from tests.fake_claude import FakeClaude, text, tool_use
from tests.helpers import FREE_SLOT, NEW_PHONE, with_mock

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


class Refused(Exception):
    changed_nothing = True


class Unsure(Exception):
    pass


def write_tool(effects):
    """A write tool that fails with the given exceptions in order, then succeeds."""
    queue = list(effects)

    async def handler(args, ctx):
        if queue:
            raise queue.pop(0)
        return {"done": args["x"]}

    return Tool(name="w", description="w", input_schema=SCHEMA, handler=handler, writes=True)


def run_with(effects, script, cap=1):
    fake = FakeClaude(script)
    agent = Agent(client=fake, model="m", system="s", tools=Registry([write_tool(effects)]), max_writes_per_run=cap)
    result = asyncio.run(agent.run([{"role": "user", "content": "go"}], RunContext(key="k")))
    return result, fake


# ------------------------------------------------------------------ the write allowance


def test_a_write_that_failed_and_changed_nothing_does_not_use_up_the_allowance():
    result, _ = run_with([Refused("taken")], [tool_use("w", {"x": 1}), tool_use("w", {"x": 2}), text("ok")])
    assert [r.outcome for r in result.tool_records] == ["error", "ok"]


def test_a_write_that_failed_in_an_unknown_way_still_counts():
    result, _ = run_with([Unsure("who knows")], [tool_use("w", {"x": 1}), tool_use("w", {"x": 2}), text("ok")])
    assert [r.outcome for r in result.tool_records] == ["error", "denied"]
    assert "one change" in str(result.tool_records[1].result.content)


def test_a_successful_write_still_uses_the_allowance():
    result, _ = run_with([], [tool_use("w", {"x": 1}), tool_use("w", {"x": 2}), text("ok")])
    assert [r.outcome for r in result.tool_records] == ["ok", "denied"]


def test_refunds_do_not_let_two_writes_through_in_one_response():
    from tests.fake_claude import tool_uses

    result, _ = run_with([Refused("a")], [tool_uses([("w", {"x": 1}), ("w", {"x": 2})]), text("ok")])
    assert [r.outcome for r in result.tool_records] == ["error", "denied"]  # both were checked before either ran


@pytest.mark.parametrize("cls,flag", [(SlotTaken, True), (InvalidSlot, True), (NotFound, True),
                                      (Unavailable, False), (OutcomeUnknown, False), (HospitalError, False)])
def test_only_refusals_from_the_hospital_count_as_changing_nothing(cls, flag):
    assert getattr(cls("x"), "changed_nothing", False) is flag


# ------------------------------------------------------------------ the notice, written by code


def ctx_for(language=None, turn=3):
    ctx = RunContext(key="c")
    ctx.turn = turn
    ctx.state["phone"] = "9876543210"
    if language:
        ctx.state["language"] = language
    return ctx


@with_mock
async def test_a_lost_slot_is_explained_in_front_of_the_next_read_back_and_only_once(mock):
    rb = ReadBacks(Hospital(mock.client))
    ctx = ctx_for()
    from agent_service.hooks import SlotTakenNotice

    SlotTakenNotice().after_tool(ToolCall("1", "book_appointment", {}), ToolResult("SlotTaken: taken", True), ctx)
    call = ToolCall("2", "book_appointment", {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "R"})
    first = await rb.book(call, ctx)
    assert first.startswith("Sorry, that time was just taken by someone else.\n\nPlease confirm: book Dr. Imran Khan")
    assert "just taken" not in await rb.book(call, ctx)  # shown once


@with_mock
async def test_a_notice_from_an_earlier_turn_is_not_shown(mock):
    from agent_service.hooks import SlotTakenNotice

    ctx = ctx_for(turn=3)
    SlotTakenNotice().after_tool(ToolCall("1", "book_appointment", {}), ToolResult("SlotTaken: taken", True), ctx)
    ctx.turn = 4
    call = ToolCall("2", "book_appointment", {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "R"})
    assert "just taken" not in await ReadBacks(Hospital(mock.client)).book(call, ctx)


def test_other_errors_and_other_tools_set_no_notice():
    from agent_service.hooks import SlotTakenNotice

    ctx = ctx_for()
    hook = SlotTakenNotice()
    hook.after_tool(ToolCall("1", "book_appointment", {}), ToolResult("NotFound: x", True), ctx)
    hook.after_tool(ToolCall("1", "cancel_appointment", {}), ToolResult("SlotTaken: x", True), ctx)
    hook.after_tool(ToolCall("1", "book_appointment", {}), ToolResult({"ok": 1}, False), ctx)
    assert "notice" not in ctx.state


@with_mock
async def test_the_notice_follows_the_patient_language(mock):
    from agent_service.hooks import SlotTakenNotice

    ctx = ctx_for("hinglish")
    SlotTakenNotice().after_tool(ToolCall("1", "book_appointment", {}), ToolResult("SlotTaken: x", True), ctx)
    call = ToolCall("2", "book_appointment", {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "R"})
    text_ = await ReadBacks(Hospital(mock.client)).book(call, ctx)
    assert "abhi kisi aur ne le liya" in text_ and "Please confirm karein" in text_


# ------------------------------------------------------------------ a whole lost race, like scenario 03


BOOK = {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "Neha Joshi"}
NEXT = {**BOOK, "start": "2026-12-02T10:30:00+05:30"}
OK = lambda: text("OK en")


def routed(route="booking"):
    return text('{"route": "%s", "language": "en", "patient_name": "Neha Joshi", "emergency": false}' % route)


async def chat(d, message, session="race"):
    return await d.chat(ChatRequest(session_id=session, phone=NEW_PHONE, message=message))


@with_mock
async def test_after_a_lost_race_the_next_time_is_offered_in_the_same_turn_and_one_yes_books_it(mock):
    await mock.reset(race_next=1)
    fake = FakeClaude([
        OK(), routed(), tool_use("book_appointment", BOOK),  # turn 1: read-back, nothing asked by the model
        OK(), tool_use("book_appointment", BOOK),  # turn 2 "yes": the slot was taken in between
        tool_use("free_slots", {"doctor_id": "d5", "date": "2026-12-02"}),
        tool_use("book_appointment", NEXT),  # offers the next time by calling the tool: the code asks
        OK(), tool_use("book_appointment", NEXT), text("Done, I've booked your appointment."),  # turn 3 "yes"
    ])
    d = FrontDesk(fake, mock.client, Settings.from_env({}), hospital_info="x")
    first = await chat(d, "I'm Neha Joshi. Book Dr. Khan on 2 Dec at 10am")
    assert "Please confirm: book Dr. Imran Khan" in first.reply
    second = await chat(d, "Yes, book it.")
    assert second.reply.startswith("Sorry, that time was just taken by someone else.")
    assert "10:30 AM" in second.reply and "Please confirm" in second.reply and not second.handoff
    assert not [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]
    third = await chat(d, "Yes, book that one.")
    assert "booked" in third.reply and not third.handoff
    mine = [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]
    assert len(mine) == 1 and mine[0]["start"].startswith("2026-12-02T10:30")


# ------------------------------------------------------------------ the model never asks to confirm itself


@pytest.mark.parametrize("prompt", [booking_prompt("H", "108"), billing_prompt("H", "108")])
def test_the_prompts_tell_the_model_to_call_the_tool_and_never_ask_for_the_confirmation_itself(prompt):
    low = " ".join(prompt.lower().split())
    assert "as soon as you have" in low and "right away" in low
    assert "never ask" in low and "shall i" in low  # names the exact phrase it must not use
    assert "the system asks" in low


def test_the_booking_prompt_covers_a_lost_slot_and_status_questions():
    low = " ".join(booking_prompt("H", "108").lower().split())
    assert "slot is taken" in low and "same message" in low and "call book_appointment" in low
    assert "my_appointments" in low and "is it booked" in low


# ------------------------------------------------------------------ claims after a real earlier booking


def test_describing_a_booking_made_earlier_in_the_chat_is_allowed_if_nothing_failed_this_turn():
    guard, ctx = ClaimGuard(), RunContext(key="k")
    ctx.turn = 1
    guard.after_tool(ToolCall("1", "book_appointment", {}), ToolResult({"ok": 1}), ctx)
    ctx.turn = 2  # "so is it booked?" and the model answers from what it knows
    assert guard.after_model(text("Yes, your appointment is booked."), ctx) is None


def test_an_earlier_success_does_not_excuse_a_claim_of_another_kind():
    guard, ctx = ClaimGuard(), RunContext(key="k")
    ctx.turn = 1
    guard.after_tool(ToolCall("1", "book_appointment", {}), ToolResult({"ok": 1}), ctx)
    ctx.turn = 2
    assert guard.after_model(text("I've cancelled your appointment."), ctx) is not None


def test_an_earlier_success_does_not_excuse_a_claim_after_a_failed_write_this_turn():
    guard, ctx = ClaimGuard(), RunContext(key="k")
    ctx.turn = 1
    guard.after_tool(ToolCall("1", "book_appointment", {}), ToolResult({"ok": 1}), ctx)
    ctx.turn = 2
    guard.after_tool(ToolCall("2", "book_appointment", {}), ToolResult("SlotTaken: x", True), ctx)
    assert guard.after_model(text("Great, I've booked your appointment."), ctx) is not None


# ------------------------------------------------------------------ facts from config


CUSTOM = {"HOSPITAL_NAME": "Sunrise Clinic, Delhi", "HOSPITAL_ADDRESS": "5 Ring Road, Delhi", "EMERGENCY_NUMBER": "112"}


def test_settings_carry_the_hospital_facts():
    s = Settings.from_env(CUSTOM)
    assert (s.hospital_name, s.hospital_address, s.emergency_number) == ("Sunrise Clinic, Delhi", "5 Ring Road, Delhi", "112")
    d = Settings.from_env({})
    assert (d.hospital_address, d.emergency_number) == ("12 MG Road, Camp, Pune", "108")
    assert d.short_name == "City Care Hospital" and s.short_name == "Sunrise Clinic"


@pytest.mark.parametrize("language", ["en", "hi", "hinglish"])
def test_the_emergency_reply_uses_the_configured_facts(language):
    reply = emergency_reply(language, Settings.from_env(CUSTOM))
    assert "112" in reply and "5 Ring Road, Delhi" in reply and "Sunrise Clinic" in reply
    assert "108" not in reply and "City Care" not in reply and "MG Road" not in reply
    assert "108" in emergency_reply(language) and "MG Road" in emergency_reply(language)  # defaults unchanged


@with_mock
async def test_the_service_answers_an_emergency_with_the_configured_number(mock):
    d = FrontDesk(FakeClaude([]), mock.client, Settings.from_env(CUSTOM), hospital_info="x")
    r = await chat(d, "he is unconscious", session="e1")
    assert "112" in r.reply and "108" not in r.reply


def test_the_prompts_use_the_configured_emergency_number():
    assert "call 112" in booking_prompt("Sunrise Clinic, Delhi", "112") and "108" not in booking_prompt("Sunrise Clinic, Delhi", "112")


# ------------------------------------------------------------------ /trace can be turned off


def http_app(**kw):
    from agent_service.app import create_app

    async def factory():
        mock_ = __import__("tests.helpers", fromlist=["Mock"]).Mock()
        return FrontDesk(FakeClaude([OK(), routed("info"), text("hi there")]), mock_.client, Settings.from_env({}),
                         hospital_info="x")

    return create_app(factory, **kw)


def test_trace_works_by_default_and_can_be_turned_off_by_argument_or_env(monkeypatch):
    from fastapi.testclient import TestClient

    body = {"session_id": "t1", "phone": NEW_PHONE, "message": "hello"}
    with TestClient(http_app()) as http:
        assert http.post("/chat", json=body).status_code == 200
        assert http.get("/trace/t1").status_code == 200
    with TestClient(http_app(trace_enabled=False)) as http:
        assert http.post("/chat", json=body).status_code == 200
        res = http.get("/trace/t1")
        assert res.status_code == 404 and "disabled" in res.json()["detail"].lower()
    for off in ("0", "false", "off", "no"):
        monkeypatch.setenv("TRACE_ENABLED", off)
        with TestClient(http_app()) as http:
            assert http.get("/trace/t1").status_code == 404
    monkeypatch.setenv("TRACE_ENABLED", "1")
    with TestClient(http_app()) as http:
        http.post("/chat", json=body)
        assert http.get("/trace/t1").status_code == 200
