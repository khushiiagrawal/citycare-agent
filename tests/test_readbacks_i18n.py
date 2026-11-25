"""Read-backs come in the patient's language (English, Hindi, Hinglish). The wording is fixed text in
code and every fact in it (doctor, day, date, time, amount) is filled in by code from real data."""

from agent_service.hooks import ReadBacks, fmt_slot
from agent_service.tools import Hospital
from agentcore.types import RunContext, ToolCall
from tests.helpers import FREE_SLOT, with_mock

PRIYA = "9876543210"


def ctx(language=None, phone=PRIYA):
    c = RunContext(key="c")
    c.state["phone"] = phone
    if language:
        c.state["language"] = language
    return c


def call(name, **args):
    return ToolCall("t1", name, args)


def test_dates_are_written_in_each_language():
    assert fmt_slot(FREE_SLOT) == "Wednesday, 2 Dec 2026 at 10:00 AM"
    assert fmt_slot(FREE_SLOT, "hi") == "बुधवार, 2 दिसंबर 2026, सुबह 10:00 बजे"
    assert fmt_slot(FREE_SLOT, "hinglish") == "Wednesday, 2 December 2026, subah 10:00 baje"
    assert fmt_slot("2026-12-04T17:00:00+05:30", "hi") == "शुक्रवार, 4 दिसंबर 2026, शाम 5:00 बजे"
    assert fmt_slot("2026-12-04T13:30:00+05:30", "hinglish") == "Friday, 4 December 2026, dopahar 1:30 baje"
    assert fmt_slot(FREE_SLOT, "fr") == fmt_slot(FREE_SLOT)  # unknown language -> English


@with_mock
async def test_booking_read_back_in_three_languages(mock):
    rb = ReadBacks(Hospital(mock.client))
    args = dict(doctor_id="d5", start=FREE_SLOT, patient_name="Ravi Kumar")
    en = await rb.book(call("book_appointment", **args), ctx())
    hi = await rb.book(call("book_appointment", **args), ctx("hi"))
    hl = await rb.book(call("book_appointment", **args), ctx("hinglish"))
    assert en == ("Please confirm: book Dr. Imran Khan on Wednesday, 2 Dec 2026 at 10:00 AM for Ravi Kumar? "
                  "Reply YES to confirm (haan / हाँ), or tell me what to change.")
    for text, slot in ((hi, fmt_slot(FREE_SLOT, "hi")), (hl, fmt_slot(FREE_SLOT, "hinglish"))):
        assert "Dr. Imran Khan" in text and "Ravi Kumar" in text and slot in text
    assert "कृपया पुष्टि करें" in hi and "बुक" in hi and "Please confirm" not in hi
    assert "Please confirm karein" in hl and "book karein" in hl and "haan" in hl


@with_mock
async def test_reschedule_cancel_and_payment_read_backs_keep_their_facts(mock):
    rb = ReadBacks(Hospital(mock.client))
    for language, marker in (("hi", "रद्द"), ("hinglish", "cancel karein")):
        cancel = await rb.cancel(call("cancel_appointment", appointment_id="A1001"), ctx(language))
        assert marker in cancel and "Dr. Sunita Iyer" in cancel and "A1001" not in cancel
        assert fmt_slot("2026-12-02T10:30:00+05:30", language) in cancel
        move = await rb.reschedule(
            call("reschedule_appointment", appointment_id="A1001", new_start="2026-12-07T10:30:00+05:30"), ctx(language))
        assert fmt_slot("2026-12-02T10:30:00+05:30", language) in move and fmt_slot("2026-12-07T10:30:00+05:30", language) in move
        pay = await rb.payment(call("send_payment_link", bill_id="B1"), ctx(language))
        assert "Rs 500" in pay and "B1" in pay


@with_mock
async def test_refusals_stay_in_english_for_the_model_whatever_the_patient_language(mock):
    from agentcore.hooks import Deny

    rb = ReadBacks(Hospital(mock.client))
    result = await rb.cancel(call("cancel_appointment", appointment_id="A1002"), ctx("hi"))  # Arjun's, not Priya's
    assert isinstance(result, Deny) and "Nothing was changed" in result.reason
