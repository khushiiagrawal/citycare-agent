"""The hospital's safety rules, as hooks. These run in code whatever the model is persuaded to do.

Order matters (see `build_hooks`): ownership, then the 4-hour rule, then the confirmation. The
confirmation comes last so a request that is not allowed never uses up (or even asks for) a yes.

  OwnershipHook          rule 1  only the caller's own appointments and bills can be touched
  FourHourRule           rule 3  inside 4 hours, no change by chat: hand over to staff
  ConfirmGate            rule 2  read the change back, wording written by code, and wait for a yes
  LeakGuard              rule 1  nothing with another person's phone number reaches the model
  RepeatedFailureHandoff rule 8  two failures in a row hand the chat to staff
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from agentcore.hooks import ConfirmGate, Deny, Hook
from agentcore.types import RunContext, ToolCall, ToolResult

from .i18n import SLOT_TAKEN, lang_of, readback, slot_text
from .hospital import IST, NotFound, Unavailable, parse_time, same_time
from .state import failures, flag_handoff, phone_of, record_failure
from .tools import Hospital

FOUR_HOURS = timedelta(hours=4)

_OWN_ID = {  # tool -> (argument holding an id, what the id refers to)
    "cancel_appointment": ("appointment_id", "appointment"),
    "reschedule_appointment": ("appointment_id", "appointment"),
    "send_payment_link": ("bill_id", "bill"),
}
_CHANGES_BY_CHAT = {"cancel_appointment", "reschedule_appointment"}
_NOT_ANSWERING = ("The hospital system is not answering, so this could not be checked. Nothing was changed. "
                  "Offer to try again; if it keeps failing, hand over to staff.")


def needs_staff(start: datetime, now: datetime) -> bool:
    """Rule 3. Free changes by chat are allowed up to 4 hours before the slot, so exactly 4 hours is
    still fine. Anything closer, already started or in the past, goes to staff."""
    return start - now < FOUR_HOURS


# ------------------------------------------------------------------ ownership


class OwnershipHook(Hook):
    """Refuse any appointment or bill id that is not the caller's. Someone else's id gets exactly the same
    answer as an id that does not exist."""

    def __init__(self, hospital: Hospital) -> None:
        self.hospital = hospital

    async def before_tool(self, call: ToolCall, ctx: RunContext) -> Any:
        spec = _OWN_ID.get(call.name)
        if spec is None:
            return None
        field, kind = spec
        phone, ident = phone_of(ctx), call.input.get(field)
        try:
            if kind == "appointment":
                await self.hospital.own_appointment(phone, ident)
            else:
                await self.hospital.own_bill(phone, ident)
        except NotFound as e:
            return Deny(f"{e} Nothing was changed.")
        except Unavailable:
            record_failure(ctx)
            return Deny(_NOT_ANSWERING)
        return None


# ------------------------------------------------------------------ the 4-hour rule


class FourHourRule(Hook):
    """Cancelling or rescheduling inside 4 hours of the EXISTING appointment goes to staff. The new slot
    is irrelevant: only the time you would be giving up counts."""

    def __init__(self, hospital: Hospital) -> None:
        self.hospital = hospital

    async def before_tool(self, call: ToolCall, ctx: RunContext) -> Any:
        if call.name not in _CHANGES_BY_CHAT:
            return None
        try:
            appt = await self.hospital.own_appointment(phone_of(ctx), call.input.get("appointment_id"))
            now = await self.hospital.client.now()
        except NotFound:
            return None  # OwnershipHook has already refused it
        except Unavailable:
            record_failure(ctx)
            return Deny(_NOT_ANSWERING)
        if needs_staff(parse_time(appt["start"]), now):
            flag_handoff(ctx, "inside_4_hours")
            return Deny("This appointment is less than 4 hours away (or has already started), so it cannot be "
                        "changed or cancelled by chat. Our staff will help the patient with it: tell them so, "
                        "kindly, and do not try again. Nothing was changed.")
        return None


# ------------------------------------------------------------------ confirmation read-backs


def fmt_slot(start: str, language: str = "en") -> str:
    return slot_text(parse_time(start).astimezone(IST), language)


def _lang(ctx: RunContext) -> str:
    return lang_of(ctx.state.get("language"))


class ReadBacks:
    """Builds each confirmation question from the real data, in the patient's language. The model never
    writes it: the wording is fixed text (i18n.py) and the facts come from the hospital system."""

    def __init__(self, hospital: Hospital) -> None:
        self.hospital = hospital

    def prompts(self) -> dict:
        return {
            "book_appointment": self.book,
            "reschedule_appointment": self.reschedule,
            "cancel_appointment": self.cancel,
            "send_payment_link": self.payment,
        }

    async def _free(self, doctor_id: str, start: str) -> Deny | None:
        """Do not ask the patient to confirm a slot that cannot be booked."""
        try:
            day = parse_time(start).astimezone(IST).date().isoformat()
        except ValueError:
            return Deny("The start is not an ISO 8601 date-time. Nothing was changed.")
        try:
            free = await self.hospital.client.free_slots(doctor_id, day)
        except NotFound as e:
            return Deny(f"{e} Nothing was changed.")
        except Unavailable:
            return None  # cannot check; ask anyway, the booking itself checks again
        if not any(same_time(slot, start) for slot in free):
            return Deny("That slot is not free. Fetch the free slots again and offer other times. Nothing was changed.")
        return None

    @staticmethod
    def _with_notice(ctx: RunContext, question: str) -> str:
        """If a booking was just lost to someone else, say so in front of the next question (once, this turn only)."""
        turn, kind = ctx.state.pop("notice", (None, None))
        if turn == ctx.turn and kind == "slot_taken":
            return f"{SLOT_TAKEN[_lang(ctx)]}\n\n{question}"
        return question

    async def book(self, call: ToolCall, ctx: RunContext) -> str | Deny:
        a = call.input
        if (problem := await self._free(a["doctor_id"], a["start"])) is not None:
            return problem
        doctor = await self.hospital.doctor_name(a["doctor_id"])
        return self._with_notice(ctx, readback("book", _lang(ctx), doctor=doctor, slot=fmt_slot(a["start"], _lang(ctx)),
                                               name=a["patient_name"]))

    async def reschedule(self, call: ToolCall, ctx: RunContext) -> str | Deny:
        a = call.input
        appt = await self._own_booked(ctx, a["appointment_id"])
        if isinstance(appt, Deny):
            return appt
        if (problem := await self._free(appt["doctor_id"], a["new_start"])) is not None:
            return problem
        doctor = await self.hospital.doctor_name(appt["doctor_id"])
        lang = _lang(ctx)
        return self._with_notice(ctx, readback("reschedule", lang, doctor=doctor, old=fmt_slot(appt["start"], lang),
                                               new=fmt_slot(a["new_start"], lang)))

    async def cancel(self, call: ToolCall, ctx: RunContext) -> str | Deny:
        appt = await self._own_booked(ctx, call.input["appointment_id"])
        if isinstance(appt, Deny):
            return appt
        doctor = await self.hospital.doctor_name(appt["doctor_id"])
        return readback("cancel", _lang(ctx), doctor=doctor, slot=fmt_slot(appt["start"], _lang(ctx)))

    async def payment(self, call: ToolCall, ctx: RunContext) -> str | Deny:
        try:
            bill = await self.hospital.own_bill(phone_of(ctx), call.input["bill_id"])
        except NotFound as e:
            return Deny(f"{e} Nothing was changed.")
        except Unavailable:
            record_failure(ctx)
            return Deny(_NOT_ANSWERING)
        if bill["status"] == "paid":
            return Deny("This bill is already paid, so there is nothing to pay. Nothing was changed.")
        return readback("payment", _lang(ctx), bill=bill["id"], amount=bill["amount_inr"], description=bill["description"])

    async def _own_booked(self, ctx: RunContext, appointment_id: str) -> dict | Deny:
        try:
            appt = await self.hospital.own_appointment(phone_of(ctx), appointment_id)
        except NotFound as e:
            return Deny(f"{e} Nothing was changed.")
        except Unavailable:
            record_failure(ctx)
            return Deny(_NOT_ANSWERING)
        if appt["status"] != "booked":
            return Deny("That appointment is not active (it is already cancelled). Nothing was changed.")
        return appt


# ------------------------------------------------------------------ leaks and repeated failures

_MOBILE = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")


def _without_phone_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_phone_keys(v) for k, v in value.items() if k != "phone"}
    if isinstance(value, list):
        return [_without_phone_keys(v) for v in value]
    return value


def scrub_result(value: Any, phone: str) -> Any | None:
    """What a tool result may look like to the model.

    Returns None (withhold everything) if any mobile number other than `phone` appears anywhere in it,
    even inside free text. Otherwise returns the result without any `phone` fields."""
    text = json.dumps(value, default=str, ensure_ascii=False)
    if any(number != phone for number in _MOBILE.findall(text)):
        return None
    return _without_phone_keys(value)


class LeakGuard(Hook):
    """Last line of defence for rule 1: even if a tool or the hospital system returns someone else's
    data, the model never sees it."""

    def after_tool(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> Any:
        phone = ctx.state.get("phone")
        if not phone:
            return None
        cleaned = scrub_result(result.content, phone)
        if cleaned is None:
            ctx.state["leaks_blocked"] = ctx.state.get("leaks_blocked", 0) + 1
            return ToolResult("Result withheld: it contained another person's data. Nothing was shown. "
                              "Tell the patient you could not do that.", True)
        return ToolResult(cleaned, result.is_error) if cleaned != result.content else None


class SlotTakenNotice(Hook):
    """When a booking or reschedule loses its slot, remember it for this turn so the next read-back (for the
    time the agent offers instead) starts by telling the patient, in words written by code."""

    def after_tool(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> Any:
        if (call.name in ("book_appointment", "reschedule_appointment") and result.is_error
                and str(result.content).startswith("SlotTaken")):
            ctx.state["notice"] = (ctx.turn, "slot_taken")
        return None


class RepeatedFailureHandoff(Hook):
    """Rule 8: when the agent fails twice in a row, hand over to staff instead of trying a third time."""

    def after_tool(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> Any:
        if failures(ctx) >= 2:
            flag_handoff(ctx, "repeated_failures")
        return None


def build_hooks(hospital: Hospital) -> list[Hook]:
    return [
        OwnershipHook(hospital),
        FourHourRule(hospital),
        ConfirmGate(ReadBacks(hospital).prompts()),  # last of the before_tool hooks
        LeakGuard(),
        SlotTakenNotice(),
        RepeatedFailureHandoff(),
    ]
