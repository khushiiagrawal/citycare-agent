"""Async client for the hospital system.

What it does that the starter client did not:
- timeouts, and retries (with backoff) on reads that fail for temporary reasons
- typed errors, each with advice the model can act on
- every booking carries an idempotency key, so retrying it cannot book twice
- when a write answers 503 it does NOT guess: it reads the real state to see whether the
  change was saved, and only reports success, failure or "unknown" after looking
- it can only reach the normal endpoints, never /_admin
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import httpx

from agentcore.resilience import RetryPolicy

IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_RETRY = RetryPolicy(max_attempts=3, base_delay_s=0.2, max_delay_s=1.5, attempt_timeout_s=None)


# ------------------------------------------------------------------ errors


class HospitalError(Exception):
    """Something the hospital system refused or failed to do. `advice` tells the model what to do next."""

    advice = ""

    def __init__(self, message: str, code: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.message, self.code, self.status = message, code, status

    def __str__(self) -> str:
        return f"{self.message} {self.advice}".strip()


class SlotTaken(HospitalError):
    advice = "Fetch the free slots again and offer the patient other times."


class InvalidSlot(HospitalError):
    advice = "Check the date and time against the free slots."


class InvalidPhone(HospitalError):
    advice = "The phone number is not valid."


class NotFound(HospitalError):
    advice = "Check the id."


class NotActive(HospitalError):
    advice = "It is not active any more, so it cannot be changed."


class AlreadyPaid(HospitalError):
    advice = "There is nothing to pay."


class Unavailable(HospitalError):
    advice = ("The hospital system is not answering. Nothing was changed. "
              "Offer to try again; if it keeps failing, hand over to staff.")


class OutcomeUnknown(HospitalError):
    advice = ("Do NOT tell the patient it worked, and do NOT tell them it failed. "
              "Say staff will confirm it, and hand over to staff.")


_BY_CODE: dict[str, type[HospitalError]] = {
    "slot_taken": SlotTaken, "invalid_slot": InvalidSlot, "invalid_phone": InvalidPhone,
    "not_active": NotActive, "already_paid": AlreadyPaid, "appointment_not_found": NotFound,
    "bill_not_found": NotFound, "doctor_not_found": NotFound, "service_unavailable": Unavailable,
}


# ------------------------------------------------------------------ helpers


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=IST)  # no zone means hospital time


def same_time(a: str, b: str) -> bool:
    """True if two ISO times are the same moment, whatever their format or zone."""
    return parse_time(a) == parse_time(b)


def _find_booking(appointments: list[dict], doctor_id: str, start: str) -> dict | None:
    return next((a for a in appointments
                 if a["status"] == "booked" and a["doctor_id"] == doctor_id and same_time(a["start"], start)), None)


# ------------------------------------------------------------------ the client


class HospitalClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        timeout: float = 10.0,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url or os.getenv("HOSPITAL_URL", "http://localhost:8001"),
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )
        self._retry = retry

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---- one attempt

    async def _request(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
        try:
            response = await self._http.request(method, path, params=params, json=json)
        except httpx.TimeoutException as e:
            raise Unavailable(f"The hospital system timed out ({type(e).__name__}).") from e
        except httpx.TransportError as e:
            raise Unavailable(f"The hospital system could not be reached ({type(e).__name__}).") from e
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code < 400:
            return body
        if not isinstance(body, dict) or "error" not in body:
            if response.status_code >= 500:
                raise Unavailable(f"The hospital system returned HTTP {response.status_code}.", status=response.status_code)
            raise HospitalError(f"Unexpected HTTP {response.status_code} from the hospital system.", status=response.status_code)
        cls = _BY_CODE.get(body["error"], HospitalError)
        raise cls(body.get("message", body["error"]), code=body["error"], status=response.status_code)

    # ---- reads: retried

    async def _read(self, path: str, **params: str) -> Any:
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return await self._request("GET", path, params=params or None)
            except Unavailable as e:
                if attempt == self._retry.max_attempts:
                    raise
                await self._retry.sleep(self._retry.delay(attempt, e))
        raise AssertionError("unreachable")

    async def now(self) -> datetime:
        """The hospital's clock. The only clock agent code should use."""
        return parse_time((await self._read("/now"))["now"])

    async def doctors(self, specialty: str | None = None) -> list[dict]:
        return (await self._read("/doctors", **({"specialty": specialty} if specialty else {})))["doctors"]

    async def free_slots(self, doctor_id: str, date: str) -> list[str]:
        return (await self._read(f"/doctors/{doctor_id}/slots", date=date))["free"]

    async def appointments(self, phone: str) -> list[dict]:
        return (await self._read("/appointments", phone=phone))["appointments"]

    async def appointment(self, appointment_id: str) -> dict:
        return await self._read(f"/appointments/{appointment_id}")

    async def bills(self, phone: str) -> list[dict]:
        return (await self._read("/bills", phone=phone))["bills"]

    # ---- writes: never trust a 503

    async def _write(
        self,
        send: Callable[[], Awaitable[Any]],
        verify: Callable[[], Awaitable[dict | None]] | None,
        *,
        retry_when_unsure: bool,
    ) -> Any:
        """Send a write. If it answers 503, look at the real state before deciding anything.

        `verify` returns the saved record if the change happened, or None if it did not.
        If it cannot be read, the outcome is unsure: retry only when that is safe (the write is
        idempotent), otherwise stop with OutcomeUnknown. With no `verify` (nothing to read back)
        the write is retried, which is only acceptable for harmless writes such as payment links."""
        unsure = False
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return await send()
            except Unavailable as failure:
                if verify is not None:
                    try:
                        saved = await verify()
                    except Unavailable:
                        saved, unsure = None, True
                        if not retry_when_unsure:
                            raise OutcomeUnknown("The hospital system did not answer, and the result could not be checked.") from failure
                    if saved is not None:
                        return saved
                if attempt == self._retry.max_attempts:
                    if unsure:
                        raise OutcomeUnknown("The hospital system did not answer, and the result could not be checked.") from failure
                    raise
                await self._retry.sleep(self._retry.delay(attempt, failure))
        raise AssertionError("unreachable")

    async def book(self, *, doctor_id: str, start: str, patient_name: str, phone: str,
                   idempotency_key: str | None = None) -> dict:
        """Book a slot. Safe to retry: the same key always gives back the same booking."""
        body = {"doctor_id": doctor_id, "start": start, "patient_name": patient_name, "phone": phone,
                "idempotency_key": idempotency_key or uuid.uuid4().hex}

        async def verify() -> dict | None:
            return _find_booking(await self.appointments(phone), doctor_id, start)

        try:
            return await self._write(lambda: self._request("POST", "/appointments", json=body), verify,
                                     retry_when_unsure=True)
        except SlotTaken:
            # The slot may be taken by us: an earlier attempt could have been saved. Look before refusing.
            try:
                mine = _find_booking(await self.appointments(phone), doctor_id, start)
            except Unavailable as e:
                raise OutcomeUnknown("The slot is taken, but it could not be checked whether it is this patient's own booking.") from e
            if mine is not None:
                return mine
            raise

    async def reschedule(self, appointment_id: str, new_start: str) -> dict:
        async def verify() -> dict | None:
            appt = await self.appointment(appointment_id)
            return appt if appt["status"] == "booked" and same_time(appt["start"], new_start) else None

        return await self._write(
            lambda: self._request("POST", f"/appointments/{appointment_id}/reschedule", json={"new_start": new_start}),
            verify, retry_when_unsure=False)

    async def cancel(self, appointment_id: str) -> dict:
        async def verify() -> dict | None:
            appt = await self.appointment(appointment_id)
            return appt if appt["status"] == "cancelled" else None

        return await self._write(
            lambda: self._request("POST", f"/appointments/{appointment_id}/cancel"), verify, retry_when_unsure=False)

    async def payment_link(self, bill_id: str) -> dict:
        """The hospital cannot show which links exist, so a 503 is simply retried. At worst the
        patient gets a second link for the same bill, which is harmless."""
        return await self._write(lambda: self._request("POST", f"/bills/{bill_id}/payment-link"), None,
                                 retry_when_unsure=True)
