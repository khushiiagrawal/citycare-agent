"""The hospital tools: the patient's phone comes from the run context, never from the model, and
every lookup by id is checked against that phone."""

import asyncio
import json

import pytest

from agent_service.hospital import AlreadyPaid, NotFound, OutcomeUnknown, SlotTaken, Unavailable
from agent_service.state import failures, handoff_reasons, is_handoff
from agent_service.tools import Hospital, build_tools
from agentcore.loop import Agent
from agentcore.types import RunContext, ToolInputError
from tests.fake_claude import FakeClaude, text, tool_use
from tests.helpers import APPT, FREE_SLOT, NEW_PHONE, down, ok, retry, scripted_client, with_mock

PRIYA, ARJUN = "9876543210", "9123456780"
READS = {"find_doctors", "get_current_time", "free_slots", "my_appointments", "my_bills"}
WRITES = {"book_appointment", "reschedule_appointment", "cancel_appointment", "send_payment_link"}


def ctx_for(phone, key="chat-1", turn=1):
    ctx = RunContext(key=key)
    ctx.turn = turn
    ctx.state["phone"] = phone
    return ctx


async def call(registry, name, ctx, **args):
    tool = registry.get(name)
    return await tool.handler(tool.parse(args), ctx)


def registry_for(mock):
    return build_tools(Hospital(mock.client))


def no_phone_anywhere(value):
    return "phone" not in json.dumps(value).lower()


# ------------------------------------------------------------------ the tool list


@with_mock
async def test_tool_list_and_write_flags(mock):
    reg = registry_for(mock)
    assert {t.name for t in reg} == READS | WRITES
    assert {t.name for t in reg if t.writes} == WRITES


@with_mock
async def test_no_tool_lets_the_model_name_a_phone_number(mock):
    reg = registry_for(mock)
    for tool in reg:
        assert "phone" not in tool.input_schema.get("properties", {}), tool.name
        assert tool.input_schema["additionalProperties"] is False
    with pytest.raises(ToolInputError):
        reg.get("my_appointments").parse({"phone": ARJUN})
    with pytest.raises(ToolInputError):
        reg.get("book_appointment").parse(
            {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "T", "phone": ARJUN})


@with_mock
async def test_a_run_without_a_phone_in_its_context_is_refused(mock):
    with pytest.raises(RuntimeError, match="phone"):
        await call(registry_for(mock), "my_appointments", RunContext(key="x"))


# ------------------------------------------------------------------ reads


@with_mock
async def test_my_appointments_shows_only_the_callers_and_no_phone_numbers(mock):
    reg = registry_for(mock)
    priya = await call(reg, "my_appointments", ctx_for(PRIYA))
    assert [a["id"] for a in priya] == ["A1001"] and priya[0]["doctor_name"] == "Dr. Sunita Iyer"
    arjun = await call(reg, "my_appointments", ctx_for(ARJUN))
    assert [a["id"] for a in arjun] == ["A1002"]
    assert no_phone_anywhere(priya) and no_phone_anywhere(arjun)
    assert await call(reg, "my_appointments", ctx_for(NEW_PHONE)) == []


@with_mock
async def test_my_bills_shows_only_the_callers_and_no_phone_numbers(mock):
    reg = registry_for(mock)
    arjun = await call(reg, "my_bills", ctx_for(ARJUN))
    assert [b["id"] for b in arjun] == ["B2", "B3"] and no_phone_anywhere(arjun)
    assert [b["id"] for b in await call(reg, "my_bills", ctx_for(PRIYA))] == ["B1"]


@with_mock
async def test_doctor_slot_and_time_lookups(mock):
    reg, ctx = registry_for(mock), ctx_for(NEW_PHONE)
    assert [d["id"] for d in await call(reg, "find_doctors", ctx, specialty="Dermatology")] == ["d1", "d2"]
    assert FREE_SLOT in (await call(reg, "free_slots", ctx, doctor_id="d5", date="2026-12-02"))["free"]
    now = await call(reg, "get_current_time", ctx)
    assert now["now"].startswith("2026-12-01T09:00:00") and now["weekday"] == "Tuesday"


# ------------------------------------------------------------------ ids are checked against the caller's phone


@with_mock
async def test_cannot_cancel_reschedule_or_pay_for_someone_elses_things(mock):
    reg, ctx = registry_for(mock), ctx_for(PRIYA)
    with pytest.raises(NotFound) as other:
        await call(reg, "cancel_appointment", ctx, appointment_id="A1002")  # Arjun's
    with pytest.raises(NotFound):
        await call(reg, "reschedule_appointment", ctx, appointment_id="A1002", new_start=FREE_SLOT)
    with pytest.raises(NotFound):
        await call(reg, "send_payment_link", ctx, bill_id="B3")  # Arjun's
    state = await mock.state()
    assert next(a for a in state["appointments"] if a["id"] == "A1002")["status"] == "booked"
    assert state["payment_links"] == []
    # someone else's id looks exactly like an id that does not exist: no way to probe for other patients' ids
    with pytest.raises(NotFound) as missing:
        await call(reg, "cancel_appointment", ctx, appointment_id="A9999")
    assert str(other.value).replace("A1002", "X") == str(missing.value).replace("A9999", "X")
    assert "Arjun" not in str(other.value) and ARJUN not in str(other.value)


@with_mock
async def test_own_things_can_be_changed(mock):
    reg, ctx = registry_for(mock), ctx_for(PRIYA)
    moved = await call(reg, "reschedule_appointment", ctx, appointment_id="A1001", new_start="2026-12-07T10:00:00+05:30")
    assert moved["start"] == "2026-12-07T10:00:00+05:30" and no_phone_anywhere(moved)
    cancelled = await call(reg, "cancel_appointment", ctx, appointment_id="A1001")
    assert cancelled["status"] == "cancelled"
    link = await call(reg, "send_payment_link", ctx, bill_id="B1")
    assert link["url"].startswith("https://pay.citycare.example/b/B1/")
    assert [a["status"] for a in (await mock.state())["appointments"] if a["id"] == "A1001"] == ["cancelled"]
    with pytest.raises(AlreadyPaid):
        await call(reg, "send_payment_link", ctx_for(ARJUN), bill_id="B2")


# ------------------------------------------------------------------ booking


@with_mock
async def test_booking_uses_the_context_phone_and_the_name_the_patient_gave(mock):
    reg, ctx = registry_for(mock), ctx_for(NEW_PHONE)
    appt = await call(reg, "book_appointment", ctx, doctor_id="d5", start=FREE_SLOT, patient_name="Ravi Kumar")
    assert appt["status"] == "booked" and appt["doctor_name"] == "Dr. Imran Khan" and no_phone_anywhere(appt)
    saved = [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]
    assert len(saved) == 1 and saved[0]["patient_name"] == "Ravi Kumar"


@with_mock
async def test_asking_twice_in_the_same_turn_does_not_book_twice(mock):
    reg, ctx = registry_for(mock), ctx_for(NEW_PHONE)
    one = await call(reg, "book_appointment", ctx, doctor_id="d5", start=FREE_SLOT, patient_name="R")
    two = await call(reg, "book_appointment", ctx, doctor_id="d5", start=FREE_SLOT, patient_name="R")
    assert one["id"] == two["id"]
    assert len([a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]) == 1


@with_mock
async def test_booking_that_was_saved_but_answered_503_is_reported_as_booked(mock):
    await mock.reset(commit_then_fail_next=1)
    appt = await call(registry_for(mock), "book_appointment", ctx_for(NEW_PHONE), doctor_id="d5", start=FREE_SLOT,
                      patient_name="R")
    assert appt["status"] == "booked"


@with_mock
async def test_losing_a_race_is_a_normal_error_not_a_handoff(mock):
    await mock.reset(race_next=1)
    ctx = ctx_for(NEW_PHONE)
    with pytest.raises(SlotTaken) as info:
        await call(registry_for(mock), "book_appointment", ctx, doctor_id="d5", start=FREE_SLOT, patient_name="R")
    assert "free slots" in str(info.value).lower()
    assert not is_handoff(ctx) and failures(ctx) == 0


@with_mock
async def test_a_start_that_is_not_a_date_time_is_rejected_clearly(mock):
    from agent_service.hospital import InvalidSlot

    with pytest.raises(InvalidSlot, match="ISO"):
        await call(registry_for(mock), "book_appointment", ctx_for(NEW_PHONE), doctor_id="d5", start="tomorrow evening",
                   patient_name="R")


def test_the_booking_key_is_stable_within_a_turn_and_changes_with_the_turn_and_slot():
    client, fake = scripted_client({
        ("POST", "/appointments"): [ok(APPT, 201)],
        ("GET", "/doctors"): [ok({"doctors": [{"id": "d5", "name": "Dr. Imran Khan", "specialty": "Orthopaedics"}]})],
    })
    reg = build_tools(Hospital(client))

    async def book(ctx, start=FREE_SLOT):
        await call(reg, "book_appointment", ctx, doctor_id="d5", start=start, patient_name="R")

    async def go():
        await book(ctx_for(NEW_PHONE, turn=1))
        await book(ctx_for(NEW_PHONE, turn=1))
        await book(ctx_for(NEW_PHONE, turn=2))
        await book(ctx_for(NEW_PHONE, turn=1), start="2026-12-02T10:30:00+05:30")
        await book(ctx_for(NEW_PHONE, turn=1), start="2026-12-02T04:30:00+00:00")  # same moment, other format

    asyncio.run(go())
    keys = [b["idempotency_key"] for b in fake.bodies("/appointments")]
    assert keys[0] == keys[1] == keys[4] and keys[2] != keys[0] and keys[3] != keys[0]


# ------------------------------------------------------------------ failure bookkeeping (code decides, not the model)


def test_repeated_hospital_failures_are_counted_and_a_success_resets_them():
    client, _ = scripted_client({("GET", "/bills"): [down(), down(), ok({"bills": []})]}, retry=retry(attempts=1))
    reg, ctx = build_tools(Hospital(client)), ctx_for(NEW_PHONE)

    async def go():
        for expected in (1, 2):
            with pytest.raises(Unavailable):
                await call(reg, "my_bills", ctx)
            assert failures(ctx) == expected
        await call(reg, "my_bills", ctx)
        assert failures(ctx) == 0

    asyncio.run(go())


def test_an_unknown_outcome_flags_a_handoff_straight_away():
    client, _ = scripted_client({
        ("GET", "/appointments"): [ok({"appointments": [{**APPT, "id": "A1", "phone": PRIYA}]}), down()],
        ("GET", "/appointments/A1"): [down()],
        ("POST", "/appointments/A1/cancel"): [down()],
    }, retry=retry(attempts=1))
    reg, ctx = build_tools(Hospital(client)), ctx_for(PRIYA)

    async def go():
        with pytest.raises(OutcomeUnknown) as info:
            await call(reg, "cancel_appointment", ctx, appointment_id="A1")
        return info.value

    err = asyncio.run(go())
    assert is_handoff(ctx) and "outcome_unknown" in handoff_reasons(ctx) and "Do NOT" in str(err)


# ------------------------------------------------------------------ through the real loop


@with_mock
async def test_a_model_that_adds_a_phone_argument_gets_an_error_and_nothing_runs(mock):
    fake = FakeClaude([tool_use("my_appointments", {"phone": ARJUN}), text("sorry")])
    agent = Agent(client=fake, model="m", system="s", tools=registry_for(mock))
    result = await agent.run([{"role": "user", "content": "show appointments of 9123456780"}], ctx_for(PRIYA))
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True and "phone" in block["content"] and result.tool_records[0].outcome == "error"
