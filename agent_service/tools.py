"""The hospital tools the agents can use.

Rules built in here, not left to the prompt:
- The patient's phone comes from the run context. No tool has a phone parameter, so the model
  cannot ask about another number even if it is tricked into trying.
- Anything looked up by id (appointment, bill) is checked against the caller's own records first.
  Someone else's id gives exactly the same "not found" as an id that does not exist.
- Results never contain phone numbers.
- Hospital failures are counted in the context, and an unknown outcome flags a handoff.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any, Awaitable

from pydantic import BaseModel, Field

from agentcore.types import Registry, RunContext

from .hospital import IST, HospitalClient, InvalidSlot, NotFound, OutcomeUnknown, Unavailable, parse_time
from .state import flag_handoff, phone_of, record_failure, reset_failures


class Hospital:
    """The hospital client plus the small amount of lookup logic that tools and hooks share."""

    def __init__(self, client: HospitalClient) -> None:
        self.client = client
        self._names: dict[str, str] | None = None

    async def doctor_name(self, doctor_id: str) -> str:
        if self._names is None:
            try:
                self._names = {d["id"]: d["name"] for d in await self.client.doctors()}
            except Unavailable:
                return doctor_id  # a missing name must never block a booking
        return self._names.get(doctor_id, doctor_id)

    async def own_appointment(self, phone: str, appointment_id: str) -> dict:
        """One of this phone's appointments, or NotFound. The same NotFound whether the id belongs to
        someone else or does not exist, so ids cannot be probed."""
        for appt in await self.client.appointments(phone):
            if appt["id"] == appointment_id and appt["phone"] == phone:
                return appt
        raise NotFound(f"No appointment {appointment_id} was found for this phone number.", code="appointment_not_found")

    async def own_bill(self, phone: str, bill_id: str) -> dict:
        for bill in await self.client.bills(phone):
            if bill["id"] == bill_id and bill["phone"] == phone:
                return bill
        raise NotFound(f"No bill {bill_id} was found for this phone number.", code="bill_not_found")

    async def public_appointment(self, appt: dict) -> dict:
        """An appointment as the model sees it: names instead of codes, and no phone number."""
        return {
            "id": appt["id"], "doctor_id": appt["doctor_id"], "doctor_name": await self.doctor_name(appt["doctor_id"]),
            "start": appt["start"], "patient_name": appt["patient_name"], "status": appt["status"],
        }


async def _guard(ctx: RunContext, work: Awaitable[Any]) -> Any:
    """Run a hospital call and keep the failure bookkeeping in code."""
    try:
        result = await work
    except OutcomeUnknown:
        flag_handoff(ctx, "outcome_unknown")
        record_failure(ctx)
        raise
    except Unavailable:
        record_failure(ctx)
        raise
    reset_failures(ctx)
    return result


def _booking_key(ctx: RunContext, phone: str, doctor_id: str, start: str) -> str:
    """Same confirmed booking in the same turn -> same key (a retry cannot double book).
    A later turn or another slot -> a new key."""
    moment = parse_time(start).astimezone(IST).isoformat()
    raw = f"{ctx.key}|{ctx.turn}|{phone}|{doctor_id}|{moment}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ------------------------------------------------------------------ tool inputs


class NoArgs(BaseModel):
    pass


class FindDoctorsArgs(BaseModel):
    specialty: str | None = Field(None, description="e.g. Dermatology, General Medicine, Orthopaedics, Paediatrics")


class FreeSlotsArgs(BaseModel):
    doctor_id: str = Field(description="A doctor id from find_doctors, e.g. d1")
    date: dt.date = Field(description="YYYY-MM-DD")


class BookArgs(BaseModel):
    doctor_id: str
    start: str = Field(description="Slot start exactly as listed by free_slots, ISO 8601 with +05:30")
    patient_name: str = Field(min_length=1, description="The name the patient gave for the appointment")


class RescheduleArgs(BaseModel):
    appointment_id: str
    new_start: str = Field(description="New slot start exactly as listed by free_slots, ISO 8601 with +05:30")


class CancelArgs(BaseModel):
    appointment_id: str


class PaymentArgs(BaseModel):
    bill_id: str


def build_tools(hospital: Hospital) -> Registry:
    client = hospital.client
    reg = Registry()

    @reg.register(FindDoctorsArgs, name="find_doctors", description="List the hospital's doctors, optionally for one specialty.")
    async def find_doctors(args: FindDoctorsArgs, ctx: RunContext):
        return await _guard(ctx, client.doctors(args.specialty))

    @reg.register(NoArgs, name="get_current_time",
                  description="The hospital's current date, time and weekday. Use it for 'today', 'tomorrow', 'Friday'.")
    async def get_current_time(args: NoArgs, ctx: RunContext):
        now = await _guard(ctx, client.now())
        return {"now": now.isoformat(timespec="seconds"), "weekday": now.strftime("%A"), "timezone": "Asia/Kolkata"}

    @reg.register(FreeSlotsArgs, name="free_slots", description="Free 30-minute slots for one doctor on one date.")
    async def free_slots(args: FreeSlotsArgs, ctx: RunContext):
        free = await _guard(ctx, client.free_slots(args.doctor_id, args.date.isoformat()))
        return {"doctor_id": args.doctor_id, "date": args.date.isoformat(), "free": free}

    @reg.register(NoArgs, name="my_appointments",
                  description="The appointments of the person you are chatting with. Never ask them for their phone number.")
    async def my_appointments(args: NoArgs, ctx: RunContext):
        appts = await _guard(ctx, client.appointments(phone_of(ctx)))
        return [await hospital.public_appointment(a) for a in appts]

    @reg.register(NoArgs, name="my_bills", description="The bills of the person you are chatting with.")
    async def my_bills(args: NoArgs, ctx: RunContext):
        bills = await _guard(ctx, client.bills(phone_of(ctx)))
        return [{k: v for k, v in b.items() if k != "phone"} for b in bills]

    @reg.register(BookArgs, name="book_appointment", writes=True,
                  description="Book a free slot for the person you are chatting with. The system asks them to confirm "
                              "first: after they say yes, call this again with exactly the same input.")
    async def book_appointment(args: BookArgs, ctx: RunContext):
        phone = phone_of(ctx)
        try:
            key = _booking_key(ctx, phone, args.doctor_id, args.start)
        except ValueError:
            raise InvalidSlot("The start is not an ISO 8601 date-time.", code="invalid_slot") from None
        appt = await _guard(ctx, client.book(doctor_id=args.doctor_id, start=args.start, patient_name=args.patient_name,
                                             phone=phone, idempotency_key=key))
        return await hospital.public_appointment(appt)

    @reg.register(RescheduleArgs, name="reschedule_appointment", writes=True,
                  description="Move one of their appointments to another free slot of the same doctor. The system asks "
                              "them to confirm first: after they say yes, call this again with exactly the same input.")
    async def reschedule_appointment(args: RescheduleArgs, ctx: RunContext):
        await hospital.own_appointment(phone_of(ctx), args.appointment_id)
        appt = await _guard(ctx, client.reschedule(args.appointment_id, args.new_start))
        return await hospital.public_appointment(appt)

    @reg.register(CancelArgs, name="cancel_appointment", writes=True,
                  description="Cancel one of their appointments. The system asks them to confirm first: after they "
                              "say yes, call this again with exactly the same input.")
    async def cancel_appointment(args: CancelArgs, ctx: RunContext):
        await hospital.own_appointment(phone_of(ctx), args.appointment_id)
        appt = await _guard(ctx, client.cancel(args.appointment_id))
        return await hospital.public_appointment(appt)

    @reg.register(PaymentArgs, name="send_payment_link", writes=True,
                  description="Create a payment link for one of their unpaid bills. The system asks them to confirm "
                              "first: after they say yes, call this again with exactly the same input.")
    async def send_payment_link(args: PaymentArgs, ctx: RunContext):
        await hospital.own_bill(phone_of(ctx), args.bill_id)
        return await _guard(ctx, client.payment_link(args.bill_id))

    return reg
