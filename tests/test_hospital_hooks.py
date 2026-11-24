"""The safety rules, enforced by hooks around the hospital tools. FakeClaude plays a model that may be
confused or tricked; the hooks decide what actually happens. Runs against the real mock hospital, in process."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_service.hooks import (
    OwnershipHook,
    build_hooks,
    needs_staff,
    scrub_result,
)
from agent_service.hospital import IST
from agent_service.state import failures, handoff_reasons, is_handoff
from agent_service.tools import Hospital, build_tools
from agentcore.hooks import ConfirmGate, Deny
from agentcore.loop import Agent
from agentcore.session import InMemoryStore, Session, run_in_session
from agentcore.types import RunContext, Tool, ToolCall
from tests.fake_claude import FakeClaude, text, tool_use, tool_uses
from tests.helpers import FREE_SLOT, NEW_PHONE, down, ok, retry, scripted_client, with_mock

PRIYA, ARJUN = "9876543210", "9123456780"
KEY = "chat-1"


def agent_for(hospital, script, extra_tools=(), **kw):
    tools = build_tools(hospital)
    for t in extra_tools:
        tools.add(t)
    return Agent(client=FakeClaude(script), model="m", system="s", tools=tools, hooks=build_hooks(hospital),
                 max_writes_per_run=1, retry=None, **kw)


async def new_chat(phone):
    store = InMemoryStore()
    await store.save(Session(key=KEY, ctx=RunContext(key=KEY, state={"phone": phone})))
    return store


async def say(agent, store, message):
    return await run_in_session(agent, store, KEY, message)


async def ctx_of(store):
    return (await store.load(KEY)).ctx


async def appointment(mock, appt_id):
    return next(a for a in (await mock.state())["appointments"] if a["id"] == appt_id)


def result_block(agent, call_index=1):
    return agent.client.messages.calls[call_index]["messages"][-1]["content"]


async def todays_early_slot(mock):
    """A free slot on Tuesday 1 Dec, which is less than 4 hours from the hospital's 09:00 clock."""
    return (await mock.client.free_slots("d3", "2026-12-01"))[0]


# ------------------------------------------------------------------ own data only


@with_mock
async def test_someone_elses_things_are_refused_before_any_confirmation_is_asked(mock):
    h = Hospital(mock.client)
    script = [
        tool_use("cancel_appointment", {"appointment_id": "A1002"}), text("sorry"),
        tool_use("reschedule_appointment", {"appointment_id": "A1002", "new_start": FREE_SLOT}), text("sorry"),
        tool_use("send_payment_link", {"bill_id": "B3"}), text("sorry"),
    ]
    agent, store = agent_for(h, script), await new_chat(PRIYA)  # Priya, asking about Arjun's things
    for _ in range(3):
        res = await say(agent, store, "do it for the other number")
        assert res.stop_reason == "end_turn" and res.tool_records[0].outcome == "denied"
    assert (await ctx_of(store)).pending == []  # never even asked Priya to confirm
    state = await mock.state()
    assert (await appointment(mock, "A1002"))["status"] == "booked" and state["payment_links"] == []


@with_mock
async def test_someone_elses_id_looks_exactly_like_an_id_that_does_not_exist(mock):
    h = Hospital(mock.client)
    agent = agent_for(h, [tool_use("cancel_appointment", {"appointment_id": "A1002"}), text("x"),
                          tool_use("cancel_appointment", {"appointment_id": "A9999"}), text("x")])
    store = await new_chat(PRIYA)
    await say(agent, store, "a")
    other = result_block(agent, 1)[0]["content"]
    await say(agent, store, "b")
    missing = result_block(agent, 3)[0]["content"]
    assert other.replace("A1002", "ID") == missing.replace("A9999", "ID")
    assert "Arjun" not in other and ARJUN not in other


@with_mock
async def test_if_ownership_cannot_be_checked_nothing_is_done(mock):
    client, _ = scripted_client({("GET", "/appointments"): [down()]}, retry=retry(attempts=1))
    h = Hospital(client)
    agent, store = agent_for(h, [tool_use("cancel_appointment", {"appointment_id": "A1001"}), text("sorry")]), await new_chat(PRIYA)
    res = await say(agent, store, "cancel it")
    assert res.tool_records[0].outcome == "denied" and "nothing was changed" in result_block(agent)[0]["content"].lower()
    assert (await ctx_of(store)).pending == []


@with_mock
async def test_the_ownership_hook_itself_refuses_foreign_ids_and_lets_own_ones_through(mock):
    # The tools and the read-backs also check ownership (defence in depth), so test this layer on its own.
    hook, ctx = OwnershipHook(Hospital(mock.client)), RunContext(key=KEY, state={"phone": PRIYA})
    mine = await hook.before_tool(ToolCall("t1", "cancel_appointment", {"appointment_id": "A1001"}), ctx)
    theirs = await hook.before_tool(ToolCall("t2", "cancel_appointment", {"appointment_id": "A1002"}), ctx)
    their_bill = await hook.before_tool(ToolCall("t3", "send_payment_link", {"bill_id": "B3"}), ctx)
    other_tool = await hook.before_tool(ToolCall("t4", "find_doctors", {}), ctx)
    assert mine is None and other_tool is None
    assert isinstance(theirs, Deny) and isinstance(their_bill, Deny)


# ------------------------------------------------------------------ the 4-hour rule


def at(hours_from_now):
    now = datetime(2026, 12, 1, 9, 0, tzinfo=IST)
    return now + timedelta(hours=hours_from_now), now


def test_the_four_hour_boundary():
    assert needs_staff(*at(3)) and needs_staff(*at(0)) and needs_staff(*at(-1))  # inside 4h, started, or past
    start, now = at(4)
    assert not needs_staff(start, now)  # exactly 4 hours is still "up to 4 hours before"
    assert needs_staff(start - timedelta(seconds=1), now)
    assert not needs_staff(*at(30))
    utc_start = datetime(2026, 12, 1, 6, 0, tzinfo=timezone.utc)  # 11:30 IST: 2.5h away, other zone
    assert needs_staff(utc_start, datetime(2026, 12, 1, 9, 0, tzinfo=IST))


@with_mock
async def test_cancel_or_reschedule_inside_four_hours_goes_to_staff(mock):
    h = Hospital(mock.client)
    script = [
        tool_use("cancel_appointment", {"appointment_id": "A1002"}), text("Staff will help."),
        tool_use("reschedule_appointment", {"appointment_id": "A1002", "new_start": "2026-12-07T10:00:00+05:30"}),
        text("Staff will help."),
    ]
    agent, store = agent_for(h, script), await new_chat(ARJUN)  # A1002 is today 12:00, 3 hours away
    first = await say(agent, store, "cancel my appointment")
    assert first.tool_records[0].outcome == "denied"
    reason = result_block(agent, 1)[0]["content"].lower()
    assert "staff" in reason and "nothing was changed" in reason  # the model is told to say staff will help
    ctx = await ctx_of(store)
    assert is_handoff(ctx) and "inside_4_hours" in handoff_reasons(ctx) and ctx.pending == []
    assert (await appointment(mock, "A1002"))["status"] == "booked"
    # moving it to a far-off day does not get around the rule: the existing time is what counts
    second = await say(agent, store, "move it to next Monday instead")
    assert second.tool_records[0].outcome == "denied"
    assert (await appointment(mock, "A1002"))["start"].startswith("2026-12-01")


@with_mock
async def test_the_rule_looks_at_the_existing_appointment_not_the_new_slot(mock):
    h = Hospital(mock.client)
    early = await todays_early_slot(mock)  # a slot less than 4h from now: irrelevant, A1001 is a day away
    args = {"appointment_id": "A1001", "new_start": early}
    agent = agent_for(h, [tool_use("reschedule_appointment", args), tool_use("reschedule_appointment", args), text("Moved.")])
    store = await new_chat(PRIYA)
    asked = await say(agent, store, "move it to this morning")
    assert asked.stop_reason == "paused" and not is_handoff(await ctx_of(store))
    await say(agent, store, "yes")
    assert (await appointment(mock, "A1001"))["start"] == early


# ------------------------------------------------------------------ confirm before changing anything


@with_mock
async def test_cancel_asks_with_a_code_written_read_back_and_acts_only_after_yes(mock):
    h = Hospital(mock.client)
    call = {"appointment_id": "A1001"}
    agent = agent_for(h, [tool_use("cancel_appointment", call), tool_use("cancel_appointment", call), text("Cancelled.")])
    store = await new_chat(PRIYA)
    asked = await say(agent, store, "cancel my appointment")
    assert asked.stop_reason == "paused" and asked.model_calls == 1  # the model wrote none of this
    for part in ["cancel", "Dr. Sunita Iyer", "Wednesday, 2 Dec 2026", "10:30 AM", "YES"]:
        assert part in asked.text
    assert (await appointment(mock, "A1001"))["status"] == "booked"
    done = await say(agent, store, "yes")
    assert done.text == "Cancelled." and (await appointment(mock, "A1001"))["status"] == "cancelled"


@with_mock
async def test_a_model_that_asks_twice_in_one_turn_still_changes_nothing(mock):
    h = Hospital(mock.client)
    call = ("cancel_appointment", {"appointment_id": "A1001"})
    agent, store = agent_for(h, [tool_uses([call, call])]), await new_chat(PRIYA)
    await say(agent, store, "cancel it, cancel it")
    assert (await appointment(mock, "A1001"))["status"] == "booked"


@with_mock
async def test_a_different_change_after_the_yes_is_asked_about_again(mock):
    h = Hospital(mock.client)
    script = [tool_use("cancel_appointment", {"appointment_id": "A1001"}),
              tool_use("send_payment_link", {"bill_id": "B1"})]  # the model changes its mind after the yes
    agent, store = agent_for(h, script), await new_chat(PRIYA)
    await say(agent, store, "cancel it")
    again = await say(agent, store, "yes")
    assert again.stop_reason == "paused" and "B1" in again.text
    assert (await appointment(mock, "A1001"))["status"] == "booked" and (await mock.state())["payment_links"] == []


@with_mock
async def test_a_yes_that_comes_a_message_too_late_does_not_count(mock):
    h = Hospital(mock.client)
    call = {"appointment_id": "A1001"}
    agent = agent_for(h, [tool_use("cancel_appointment", call), text("Our timings are 10-1."),
                          tool_use("cancel_appointment", call)])
    store = await new_chat(PRIYA)
    await say(agent, store, "cancel it")
    await say(agent, store, "what are your timings?")
    late = await say(agent, store, "yes")
    assert late.stop_reason == "paused" and (await appointment(mock, "A1001"))["status"] == "booked"


@with_mock
async def test_booking_reads_back_doctor_day_date_time_and_name_then_books_once(mock):
    h = Hospital(mock.client)
    args = {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "Ravi Kumar"}
    agent = agent_for(h, [tool_use("book_appointment", args), tool_use("book_appointment", args), text("Booked!")])
    store = await new_chat(NEW_PHONE)
    asked = await say(agent, store, "book Dr Khan wednesday 10")
    for part in ["Dr. Imran Khan", "Wednesday, 2 Dec 2026", "10:00 AM", "Ravi Kumar"]:
        assert part in asked.text
    assert not [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]
    await say(agent, store, "yes")
    assert len([a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]) == 1


@with_mock
async def test_a_slot_that_is_not_free_is_refused_before_asking_anyone_to_confirm(mock):
    h = Hospital(mock.client)
    taken = {"doctor_id": "d1", "start": "2026-12-04T17:00:00+05:30", "patient_name": "R"}  # Dr. Mehta, full
    agent, store = agent_for(h, [tool_use("book_appointment", taken), text("Sorry, that one is taken.")]), await new_chat(NEW_PHONE)
    res = await say(agent, store, "book it")
    assert res.tool_records[0].outcome == "denied" and (await ctx_of(store)).pending == []
    assert "free slots" in result_block(agent)[0]["content"].lower()


@with_mock
async def test_payment_link_reads_back_bill_and_amount_and_a_paid_bill_is_refused(mock):
    h = Hospital(mock.client)
    agent = agent_for(h, [tool_use("send_payment_link", {"bill_id": "B1"}), tool_use("send_payment_link", {"bill_id": "B1"}),
                          text("Sent.")])
    store = await new_chat(PRIYA)
    asked = await say(agent, store, "send me the link")
    assert "B1" in asked.text and "Rs 500" in asked.text and (await mock.state())["payment_links"] == []
    await say(agent, store, "yes")
    assert [link["bill_id"] for link in (await mock.state())["payment_links"]] == ["B1"]
    paid = agent_for(h, [tool_use("send_payment_link", {"bill_id": "B2"}), text("It is already paid.")])
    res = await say(paid, await new_chat(ARJUN), "link for B2")
    assert res.tool_records[0].outcome == "denied"


@with_mock
async def test_only_one_change_per_message_and_the_second_keeps_its_yes(mock):
    h = Hospital(mock.client)
    cancel, pay = ("cancel_appointment", {"appointment_id": "A1001"}), ("send_payment_link", {"bill_id": "B1"})
    script = [tool_uses([cancel, pay]), tool_uses([cancel, pay]), text("Cancelled. Link next."),
              tool_use(*pay), text("Link sent.")]
    agent, store = agent_for(h, script), await new_chat(PRIYA)
    asked = await say(agent, store, "cancel and send the bill link")
    assert asked.stop_reason == "paused" and "cancel" in asked.text and "B1" in asked.text  # both read back
    await say(agent, store, "yes to both")
    state = await mock.state()
    assert (await appointment(mock, "A1001"))["status"] == "cancelled" and state["payment_links"] == []  # only one
    await say(agent, store, "ok go on")
    assert [link["bill_id"] for link in (await mock.state())["payment_links"]] == ["B1"]  # no second confirmation


# ------------------------------------------------------------------ nothing leaks


def test_scrub_result_withholds_other_numbers_and_strips_the_callers_own():
    other = {"appointments": [{"id": "A1002", "phone": ARJUN}]}
    assert scrub_result(other, PRIYA) is None
    assert scrub_result({"note": f"call {ARJUN} now"}, PRIYA) is None  # a number hiding in free text
    assert scrub_result([{"id": "A1", "phone": PRIYA, "x": 1}], PRIYA) == [{"id": "A1", "x": 1}]
    clean = {"free": ["2026-12-02T10:00:00+05:30"], "url": "https://pay.citycare.example/b/B1/a1b2c3d4"}
    assert scrub_result(clean, PRIYA) == clean
    assert scrub_result("fine", PRIYA) == "fine"


@with_mock
async def test_a_tool_that_returns_someone_elses_data_is_withheld(mock):
    h = Hospital(mock.client)
    leaky = Tool(name="leaky", description="d", input_schema={"type": "object", "properties": {}},
                 handler=lambda i, c: {"appointments": [{"id": "A1002", "phone": ARJUN, "patient_name": "Arjun Desai"}]})
    agent = agent_for(h, [tool_use("leaky", {}), text("nothing to show")], extra_tools=[leaky])
    res = await say(agent, await new_chat(PRIYA), "show everything")
    block = result_block(agent)[0]
    assert block["is_error"] is True and "withheld" in block["content"].lower()
    assert ARJUN not in json.dumps(agent.client.messages.calls) and "Arjun" not in block["content"]
    assert res.tool_records[0].result.is_error


# ------------------------------------------------------------------ repeated failures hand over to staff


def test_two_failures_in_a_row_hand_over_and_one_does_not():
    client, _ = scripted_client({("GET", "/bills"): [down(), down(), ok({"bills": []})]}, retry=retry(attempts=1))
    h = Hospital(client)
    agent = agent_for(h, [tool_use("my_bills", {}), tool_use("my_bills", {}), tool_use("my_bills", {}), text("ok")])
    async def go():
        store = await new_chat(NEW_PHONE)
        await say(agent, store, "my bills")
        return await ctx_of(store)

    ctx = asyncio.run(go())
    assert "repeated_failures" in handoff_reasons(ctx) and failures(ctx) == 0  # the third call worked, count reset


def test_one_failure_alone_is_not_a_handoff():
    client, _ = scripted_client({("GET", "/bills"): [down(), ok({"bills": []})]}, retry=retry(attempts=1))
    agent = agent_for(Hospital(client), [tool_use("my_bills", {}), tool_use("my_bills", {}), text("ok")])
    async def go():
        store = await new_chat(NEW_PHONE)
        await say(agent, store, "my bills")
        return await ctx_of(store)

    assert not is_handoff(asyncio.run(go()))


# ------------------------------------------------------------------ wiring


@with_mock
async def test_hooks_are_ordered_so_checks_run_before_the_confirmation_is_used_up(mock):
    hooks = build_hooks(Hospital(mock.client))
    names = [type(h).__name__ for h in hooks]
    assert names.index("OwnershipHook") < names.index("FourHourRule") < names.index("ConfirmGate")
    assert isinstance(hooks[names.index("ConfirmGate")], ConfirmGate)
    assert set(names) >= {"OwnershipHook", "FourHourRule", "ConfirmGate", "LeakGuard", "RepeatedFailureHandoff"}
