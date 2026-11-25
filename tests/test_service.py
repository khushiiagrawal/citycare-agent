"""/chat end to end with a scripted model and the real mock hospital: emergency check, router, specialist,
shared state, handoff and /trace. No key needed."""

import json

import pytest
from fastapi.testclient import TestClient

from agent_service.app import create_app
from agent_service.config import Settings
from agent_service.schemas import ChatRequest
from agent_service.service import FrontDesk
from tests.fake_claude import FakeClaude, text, tool_use
from tests.helpers import FREE_SLOT, NEW_PHONE, with_mock

ARJUN, PRIYA = "9123456780", "9876543210"
OK = lambda: text("OK en")  # the emergency check says fine


def routed(route, language="en", name=None, emergency=False):
    return text(json.dumps({"route": route, "language": language, "patient_name": name, "emergency": emergency}))


def desk(mock, *script):
    fake = FakeClaude(list(script))
    return FrontDesk(fake, mock.client, Settings.from_env({}), hospital_info="OPD 10am to 1pm and 5pm to 8pm."), fake


async def chat(desk_, message, phone=NEW_PHONE, session="s1"):
    return await desk_.chat(ChatRequest(session_id=session, phone=phone, message=message))


def models(fake):
    """The agent that made each model call, read from its system prompt."""
    def who(call):
        system = call["system"][0]["text"] if isinstance(call["system"], list) else call["system"]
        return ("emergency" if system.startswith("You check one message") else
                "router" if system.startswith("You route") else
                "info" if "<hospital_info>" in system else
                "booking" if "appointments" in system.split("Your job:")[-1][:60] else "billing")
    return [who(c) for c in fake.messages.calls]


# ------------------------------------------------------------------ emergencies come first


@with_mock
async def test_an_emergency_in_hinglish_gets_108_flags_and_never_reaches_a_model(mock):
    d, fake = desk(mock)
    r = await chat(d, "Mujhe seene mein bahut dard ho raha hai aur saans lene mein taklif hai. Kal ka appointment chahiye.")
    assert (r.agent, r.emergency, r.handoff) == ("safety", True, True)
    assert "108" in r.reply and fake.messages.calls == [] and r.trace_id
    assert not [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]


@with_mock
async def test_the_model_check_catches_an_emergency_the_regex_missed(mock):
    d, fake = desk(mock, text("EMERGENCY en"))
    r = await chat(d, "my dad suddenly can't talk and one arm hangs limp")
    assert (r.agent, r.emergency, r.handoff) == ("safety", True, True) and "108" in r.reply
    assert models(fake) == ["emergency"]


@with_mock
async def test_the_router_is_a_third_net(mock):
    d, fake = desk(mock, OK(), routed("info", emergency=True))
    r = await chat(d, "something feels very wrong with my heart")
    assert (r.agent, r.emergency, r.handoff) == ("safety", True, True) and models(fake) == ["emergency", "router"]


@with_mock
async def test_after_an_emergency_the_chat_stays_in_safety(mock):
    d, fake = desk(mock)
    await chat(d, "बहुत खून बह रहा है")
    r = await chat(d, "ok book me with Dr. Rao tomorrow")
    assert (r.agent, r.emergency, r.handoff) == ("safety", True, True) and fake.messages.calls == []


@with_mock
async def test_a_hindi_emergency_is_answered_in_hindi(mock):
    r = await chat(desk(mock)[0], "मुझे सीने में दर्द है")
    assert "108" in r.reply and "आपातकाल" in r.reply


@with_mock
async def test_if_the_emergency_model_is_down_the_chat_carries_on(mock):
    d, _ = desk(mock, Exception("outage"), routed("info"), text("We open at 10am."))
    r = await chat(d, "when do you open?")
    assert r.agent == "info" and "10am" in r.reply and not r.emergency


# ------------------------------------------------------------------ routing and replies


@with_mock
async def test_an_info_question_goes_to_the_info_agent(mock):
    d, fake = desk(mock, OK(), routed("info"), text("OPD is 10am to 1pm and 5pm to 8pm, Monday to Saturday."))
    r = await chat(d, "What are your OPD hours?")
    assert (r.agent, r.handoff, r.emergency) == ("info", False, False) and "10am" in r.reply
    assert models(fake) == ["emergency", "router", "info"]


@with_mock
async def test_asking_for_a_person_hands_over_without_calling_a_specialist(mock):
    d, fake = desk(mock, OK(), routed("human", "hinglish"))
    r = await chat(d, "mujhe kisi insaan se baat karni hai")
    assert (r.agent, r.handoff, r.emergency) == ("human", True, False) and "staff" in r.reply
    assert models(fake) == ["emergency", "router"]


@with_mock
async def test_an_unusable_router_answer_falls_back_to_the_info_agent(mock):
    d, _ = desk(mock, OK(), text("no idea"), text("Happy to help, what do you need?"))
    assert (await chat(d, "hello")).agent == "info"


@with_mock
async def test_when_the_router_model_fails_the_chat_goes_to_staff(mock):
    d, _ = desk(mock, OK(), Exception("outage"))
    r = await chat(d, "hello")
    assert (r.agent, r.handoff) == ("human", True) and r.reply


@with_mock
async def test_when_the_specialist_model_fails_the_chat_goes_to_staff_and_nothing_is_saved(mock):
    d, _ = desk(mock, OK(), routed("booking"), Exception("outage"), OK(), routed("info"), text("hi"))
    r = await chat(d, "book me a doctor")
    assert (r.agent, r.handoff) == ("human", True)
    assert (await chat(d, "hello again")).agent == "info"  # the chat still works afterwards


@with_mock
async def test_an_empty_reply_is_not_sent_as_silence(mock):
    d, _ = desk(mock, OK(), routed("info"), text(""))
    r = await chat(d, "hi")
    assert r.reply.strip() and r.handoff


# ------------------------------------------------------------------ writes, end to end


BOOK = {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "Ravi Kumar"}


@with_mock
async def test_a_booking_needs_the_read_back_and_then_a_yes(mock):
    d, fake = desk(mock,
                   OK(), routed("booking", name="Ravi Kumar"), tool_use("book_appointment", BOOK),
                   OK(), routed("booking"), tool_use("book_appointment", BOOK), text("Done! I've booked your appointment."))
    first = await chat(d, "Book Dr. Khan on 2 Dec at 10am, I'm Ravi Kumar")
    assert first.agent == "booking" and "Please confirm: book Dr. Imran Khan" in first.reply and "Ravi Kumar" in first.reply
    assert not [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]  # nothing yet
    second = await chat(d, "yes")
    assert second.agent == "booking" and "booked" in second.reply and not second.handoff
    saved = [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]
    assert len(saved) == 1 and saved[0]["patient_name"] == "Ravi Kumar"


@with_mock
async def test_the_read_back_uses_the_patients_language_and_the_name_they_gave(mock):
    d, _ = desk(mock, OK(), routed("booking", "hinglish", "Ravi Kumar"), tool_use("book_appointment", BOOK))
    r = await chat(d, "Dr. Khan ke saath 2 Dec ko subah 10 baje book karo, main Ravi Kumar")
    assert "Please confirm karein" in r.reply and "subah 10:00 baje" in r.reply


@with_mock
async def test_the_tool_result_the_model_saw_belongs_to_the_verified_phone(mock):
    d, fake = desk(mock, OK(), routed("booking"), tool_use("my_appointments", {}), text("ok"))
    await chat(d, "show my appointments. My number is 9123456780", phone=PRIYA)
    result = fake.messages.calls[3]["messages"][-1]["content"][0]["content"]
    assert "A1001" in result and "A1002" not in result and PRIYA not in result and ARJUN not in result


@with_mock
async def test_cancelling_inside_four_hours_hands_over_to_staff(mock):
    d, _ = desk(mock, OK(), routed("booking"), tool_use("cancel_appointment", {"appointment_id": "A1002"}),
                text("That appointment is very soon, so our staff will help you with it."))
    r = await chat(d, "cancel my appointment today", phone=ARJUN)
    assert (r.agent, r.handoff) == ("booking", True)
    assert next(a for a in (await mock.state())["appointments"] if a["id"] == "A1002")["status"] == "booked"


@with_mock
async def test_a_reply_that_claims_a_booking_that_never_happened_is_replaced(mock):
    d, _ = desk(mock, OK(), routed("booking"), text("Great, I've booked your appointment with Dr. Khan!"))
    r = await chat(d, "yes")
    assert "booked" not in r.reply and r.handoff


@with_mock
async def test_medical_advice_never_reaches_the_patient(mock):
    d, _ = desk(mock, OK(), routed("info"), text("It's probably a viral fever. Take a paracetamol and rest."))
    r = await chat(d, "I have a mild fever since yesterday")
    assert "paracetamol" not in r.reply and "medical advice" in r.reply


# ------------------------------------------------------------------ sharing between agents


@with_mock
async def test_the_name_and_language_follow_the_patient_to_the_next_agent(mock):
    d, fake = desk(mock,
                   OK(), routed("billing", "hinglish", "Priya Nair"), text("Aap par Rs 500 baaki hain."),
                   OK(), routed("booking", "hinglish"), text("Kis doctor ke saath?"))
    await chat(d, "Hi, this is Priya Nair. Kitna baaki hai?", phone=PRIYA)
    await chat(d, "Ab Dr. Iyer ke saath appointment book karo", phone=PRIYA)
    booking_call = fake.messages.calls[5]
    last = booking_call["messages"][-1]["content"]
    assert "Priya Nair" in last and "hinglish" in last and "Rs 500" in last  # name, language, what billing said


@with_mock
async def test_the_handoff_flag_stays_on_for_the_rest_of_the_chat(mock):
    d, _ = desk(mock, OK(), routed("human"), OK(), routed("info"), text("Our staff will call you."))
    await chat(d, "I want to talk to a person")
    assert (await chat(d, "also what is the fee?")).handoff is True


@with_mock
async def test_one_session_cannot_be_used_from_another_phone(mock):
    d, _ = desk(mock, OK(), routed("info"), text("hello"))
    await chat(d, "hi", phone=PRIYA, session="shared-id")
    with pytest.raises(PermissionError):
        await chat(d, "what did she say?", phone=ARJUN, session="shared-id")


# ------------------------------------------------------------------ /trace and the HTTP app


@with_mock
async def test_trace_shows_every_model_and_tool_call_with_tokens_and_cost(mock):
    d, _ = desk(mock, OK(), routed("billing"), tool_use("my_bills", {}), text("You owe Rs 500."))
    r = await chat(d, "what do I owe?", phone=PRIYA)
    traces = d.trace("s1")
    assert {t["agent"] for t in traces} == {"emergency_check", "router", "billing"}
    assert r.trace_id in {t["trace_id"] for t in traces}
    events = [e for t in traces for e in t["events"]]
    assert {"model", "tool"} <= {e["kind"] for e in events}
    tool = next(e for e in events if e["kind"] == "tool")
    assert tool["name"] == "my_bills" and tool["outcome"] == "ok"
    assert all(t["cost_usd"] > 0 for t in traces if t["agent"] != "safety")
    assert d.trace("never-seen") == []


@with_mock
async def test_a_safety_reply_has_a_trace_too(mock):
    d, _ = desk(mock)
    r = await chat(d, "he is unconscious")
    assert [t["trace_id"] for t in d.trace("s1")] == [r.trace_id]


def test_the_http_app_serves_chat_and_trace_and_hides_nothing_it_should_not():
    import asyncio

    from tests.helpers import Mock

    mock = Mock()
    fake = FakeClaude([OK(), routed("info"), text("OPD is 10am to 1pm.")])

    async def factory():
        return FrontDesk(fake, mock.client, Settings.from_env({}), hospital_info="OPD 10am to 1pm.")

    try:
        with TestClient(create_app(factory)) as http:
            body = {"session_id": "h1", "phone": NEW_PHONE, "message": "OPD hours?"}
            res = http.post("/chat", json=body)
            assert res.status_code == 200
            data = res.json()
            assert set(data) == {"reply", "agent", "handoff", "emergency", "trace_id"}
            assert data["agent"] == "info" and "10am" in data["reply"]
            trace = http.get("/trace/h1").json()
            assert trace["session_id"] == "h1" and len(trace["traces"]) == 3 and trace["cost_usd"] > 0
            assert http.get("/trace/nope").status_code == 404
            assert http.post("/chat", json={**body, "phone": ARJUN}).status_code == 403
            assert http.post("/chat", json={"session_id": "", "phone": NEW_PHONE, "message": "x"}).status_code == 422
            assert http.get("/health").json() == {"ok": True}
    finally:
        asyncio.run(mock.close())
