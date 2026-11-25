"""after_model guards (no false "it's done", no medical advice), the shared-facts hook and the handoff tool."""

import asyncio

import pytest

from agent_service.guards import ClaimGuard, KnownFacts, NoMedicalAdvice, claims, gives_medical_advice
from agent_service.state import handoff_reasons, is_handoff
from agent_service.tools import Hospital, build_handoff_tool, build_tools
from agentcore.loop import Agent
from agentcore.types import Registry, RunContext, ToolCall, ToolResult
from tests.fake_claude import FakeClaude, text, tool_use
from tests.helpers import FREE_SLOT, NEW_PHONE, with_mock

# ------------------------------------------------------------------ what counts as a claim


@pytest.mark.parametrize("reply,kind", [
    ("Done! I've booked your appointment with Dr. Khan.", "book"),
    ("Your appointment has been booked for Wednesday.", "book"),
    ("You're all set, the booking is confirmed.", "book"),
    ("I have cancelled your appointment.", "cancel"),
    ("Your appointment is cancelled.", "cancel"),
    ("I've rescheduled it to Monday at 10.", "reschedule"),
    ("Your appointment has been moved to Friday.", "reschedule"),
    ("I've sent you the payment link.", "payment"),
    ("Your payment link has been sent.", "payment"),
    ("आपका अपॉइंटमेंट बुक कर दिया गया है।", "book"),
    ("अपॉइंटमेंट रद्द कर दिया है।", "cancel"),
    ("पेमेंट लिंक भेज दिया है।", "payment"),
    ("Aapka appointment book ho gaya hai.", "book"),
    ("Maine appointment cancel kar diya.", "cancel"),
    ("Payment link bhej diya hai.", "payment"),
])
def test_claims_of_a_finished_change_are_found(reply, kind):
    assert kind in claims(reply)


@pytest.mark.parametrize("reply", [
    "Shall I book Dr. Khan for Wednesday at 10?",
    "I couldn't book that slot, it was just taken.",
    "Your appointment was not cancelled, please try again.",
    "I can book it once you confirm.",
    "Kya main appointment book kar dun?",
    "Appointment book nahi ho paya.",
    "Please tell me the doctor you want.",
])
def test_questions_offers_and_failures_are_not_claims(reply):
    assert not claims(reply)


@pytest.mark.parametrize("reply", [
    "You should take a paracetamol and rest.",
    "Try taking some ibuprofen for the pain.",
    "It's probably just a viral infection.",
    "You probably have an allergy.",
    "Don't worry, it's nothing serious.",
    "Take 500 mg twice a day.",
    "Apply some cream on it, it will go away on its own.",
    "पैरासिटामोल की गोली ले लीजिए।",
    "घबराने की कोई बात नहीं है।",
    "Paracetamol le lo, theek ho jaayega.",
    "Chinta mat karo, kuch serious nahi hai.",
])
def test_medical_advice_is_found(reply):
    assert gives_medical_advice(reply)


@pytest.mark.parametrize("reply", [
    "The consultation fee is Rs 500.",
    "Dr. Imran Khan sees bone and joint problems. Shall I book a visit?",
    "Our pharmacy is on the ground floor, open 8am to 10pm.",
    "For fasting tests, no food for 10 to 12 hours; water is fine.",
    "Don't worry, I can book that for you.",
    "I can't give medical advice. A doctor can help, shall I book one?",
])
def test_ordinary_replies_are_not_medical_advice(reply):
    assert not gives_medical_advice(reply)


# ------------------------------------------------------------------ in the loop


def agent_with(fake, hooks, tools=None):
    return Agent(client=fake, model="m", system="s", tools=tools, hooks=hooks)


def run(agent, message="hi", ctx=None):
    ctx = ctx or RunContext(key="k")
    ctx.state.setdefault("phone", NEW_PHONE)
    return asyncio.run(agent.run([{"role": "user", "content": message}], ctx)), ctx


def test_a_reply_that_says_booked_with_no_booking_made_is_replaced_and_flagged():
    fake = FakeClaude([text("All done, I've booked your appointment!")])
    result, ctx = run(agent_with(fake, [ClaimGuard()]), "yes")
    assert "booked" not in result.text.lower() and "staff" in result.text.lower()
    assert is_handoff(ctx) and "unverified_claim" in handoff_reasons(ctx)


def test_the_replacement_follows_the_patient_language():
    fake = FakeClaude([text("Aapka appointment book ho gaya hai.")])
    ctx = RunContext(key="k")
    ctx.state["language"] = "hinglish"
    result, _ = run(agent_with(fake, [ClaimGuard()]), "yes", ctx)
    assert "book ho gaya" not in result.text and "staff" in result.text.lower()


@with_mock
async def test_a_reply_that_says_booked_after_a_real_booking_is_left_alone(mock):
    fake = FakeClaude([
        tool_use("book_appointment", {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "R"}),
        text("Done, I've booked your appointment."),
    ])
    agent = agent_with(fake, [ClaimGuard()], build_tools(Hospital(mock.client)))
    ctx = RunContext(key="k")
    ctx.state["phone"] = NEW_PHONE
    result = await agent.run([{"role": "user", "content": "book it"}], ctx)
    assert result.text == "Done, I've booked your appointment." and not is_handoff(ctx)


@with_mock
async def test_a_failed_booking_does_not_count_and_the_false_claim_is_caught(mock):
    await mock.reset(race_next=1)
    fake = FakeClaude([
        tool_use("book_appointment", {"doctor_id": "d5", "start": FREE_SLOT, "patient_name": "R"}),
        text("Great, your appointment is booked!"),
    ])
    agent = agent_with(fake, [ClaimGuard()], build_tools(Hospital(mock.client)))
    ctx = RunContext(key="k")
    ctx.state["phone"] = NEW_PHONE
    result = await agent.run([{"role": "user", "content": "book it"}], ctx)
    assert "is booked" not in result.text and is_handoff(ctx)


def test_reading_an_existing_booking_allows_describing_it():
    guard, ctx = ClaimGuard(), RunContext(key="k")
    ctx.turn = 1
    guard.after_tool(ToolCall("1", "my_appointments", {}), ToolResult([{"id": "A1", "status": "booked"}]), ctx)
    assert guard.after_model(text("Your appointment is booked for Friday at 5."), ctx) is None


def test_a_reply_that_wants_a_tool_is_not_checked():
    guard, ctx = ClaimGuard(), RunContext(key="k")
    assert guard.after_model(tool_use("my_appointments", {}, preamble="I've booked nothing yet, checking."), ctx) is None


def test_medical_advice_is_replaced_with_a_safe_reply_that_offers_a_doctor():
    fake = FakeClaude([text("It's probably just a viral fever, take a paracetamol.")])
    result, ctx = run(agent_with(fake, [NoMedicalAdvice()]), "I have fever")
    assert "paracetamol" not in result.text and "medical advice" in result.text and "doctor" in result.text.lower()
    assert not is_handoff(ctx)


def test_medical_advice_replacement_in_hindi():
    fake = FakeClaude([text("पैरासिटामोल की गोली ले लीजिए।")])
    ctx = RunContext(key="k")
    ctx.state["language"] = "hi"
    result, _ = run(agent_with(fake, [NoMedicalAdvice()]), "बुखार है", ctx)
    assert "पैरासिटामोल" not in result.text and "डॉक्टर" in result.text


# ------------------------------------------------------------------ shared facts


def test_known_facts_are_added_to_the_last_message_for_one_call_only():
    hook, ctx = KnownFacts(), RunContext(key="k")
    ctx.state.update(patient_name="Priya Nair", language="hinglish")
    history = [{"role": "user", "content": "book me"}]
    sent = hook.before_model(history, ctx)
    assert history == [{"role": "user", "content": "book me"}]  # history untouched
    assert "Priya Nair" in str(sent[-1]["content"]) and "hinglish" in str(sent[-1]["content"])
    assert str(sent[-1]["content"]).startswith("book me")


def test_known_facts_go_after_tool_results_in_a_tool_loop():
    hook, ctx = KnownFacts(), RunContext(key="k")
    ctx.state["patient_name"] = "Priya"
    results = [{"type": "tool_result", "tool_use_id": "t", "content": "{}", "is_error": False}]
    sent = hook.before_model([{"role": "user", "content": "hi"}, {"role": "assistant", "content": []},
                              {"role": "user", "content": results}], ctx)
    blocks = sent[-1]["content"]
    assert blocks[0]["type"] == "tool_result" and blocks[-1]["type"] == "text" and "Priya" in blocks[-1]["text"]


def test_nothing_is_added_when_nothing_is_known():
    assert KnownFacts().before_model([{"role": "user", "content": "hi"}], RunContext(key="k")) is None


def test_recent_context_from_another_agent_is_included_once():
    hook, ctx = KnownFacts(), RunContext(key="k")
    ctx.state["recent"] = "Patient: how much do I owe?\nAssistant: You owe Rs 500."
    sent = hook.before_model([{"role": "user", "content": "also book Dr. Iyer"}], ctx)
    assert "You owe Rs 500" in str(sent[-1]["content"])


# ------------------------------------------------------------------ the handoff tool


def test_the_handoff_tool_flags_the_chat_for_staff():
    reg = Registry([build_handoff_tool()])
    tool, ctx = reg.get("handoff_to_staff"), RunContext(key="k")
    assert not tool.writes
    out = asyncio.run(tool.handler(tool.parse({"reason": "patient_asked_for_person"}), ctx))
    assert is_handoff(ctx) and handoff_reasons(ctx) == ["patient_asked_for_person"] and "staff" in str(out).lower()


def test_the_handoff_tool_only_takes_known_reasons():
    from agentcore.types import ToolInputError

    with pytest.raises(ToolInputError):
        build_handoff_tool().parse({"reason": "because"})
