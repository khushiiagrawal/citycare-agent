"""The async hospital client: typed errors, retries, idempotency, and checking the real
state after a write that answered 503."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from agent_service.hospital import (
    AlreadyPaid,
    HospitalClient,
    HospitalError,
    InvalidPhone,
    InvalidSlot,
    NotActive,
    NotFound,
    OutcomeUnknown,
    SlotTaken,
    Unavailable,
    same_time,
)
from agentcore.resilience import RetryPolicy
from mock_hospital.app import app as mock_app

IST = timezone(timedelta(hours=5, minutes=30))
NEW_PHONE = "9000000001"
FREE_SLOT = "2026-12-02T10:00:00+05:30"  # Dr. Khan (d5), Wednesday: free all day


class Sleeps:
    def __init__(self):
        self.delays = []

    async def __call__(self, seconds):
        self.delays.append(seconds)


def retry(sleeps=None, attempts=3):
    return RetryPolicy(max_attempts=attempts, base_delay_s=0.1, max_delay_s=1.0, jitter=0.0,
                       sleep=sleeps or Sleeps(), attempt_timeout_s=None)


# ------------------------------------------------------------------ a scripted hospital


def ok(body, status=200):
    return httpx.Response(status, json=body)


def down():
    return httpx.Response(503, json={"error": "service_unavailable", "message": "try again"})


class Scripted:
    """Answers each (method, path) from a list, in order. The last answer repeats."""

    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        queue = self.routes[(request.method, request.url.path)]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def count(self, method, path):
        return sum(1 for r in self.requests if (r.method, r.url.path) == (method, path))

    def bodies(self, path):
        return [json.loads(r.content) for r in self.requests if r.method == "POST" and r.url.path == path]


def scripted_client(routes, **kw):
    fake = Scripted(routes)
    return HospitalClient(base_url="http://hospital", transport=httpx.MockTransport(fake), retry=kw.pop("retry", retry()),
                          **kw), fake


APPT = {"id": "A2000", "doctor_id": "d5", "start": FREE_SLOT, "patient_name": "Test", "phone": NEW_PHONE,
        "status": "booked", "created_at": "2026-12-01T09:00:00+05:30"}


# ------------------------------------------------------------------ the real mock, in process


class Mock:
    """The mock hospital app, driven in-process. Admin calls are for test setup only."""

    def __init__(self):
        transport = httpx.ASGITransport(app=mock_app)
        self.client = HospitalClient(base_url="http://hospital", transport=transport, retry=retry())
        self.admin = httpx.AsyncClient(transport=transport, base_url="http://hospital")

    async def reset(self, **chaos):
        await self.admin.post("/_admin/reset")
        await self.admin.post("/_admin/chaos", json=chaos)

    async def state(self):
        return (await self.admin.get("/_admin/state")).json()

    async def close(self):
        await self.client.aclose()
        await self.admin.aclose()


def with_mock(test):
    """Run an async test body against a freshly reset mock hospital."""

    def wrapper():
        async def go():
            mock = Mock()
            await mock.reset()
            try:
                await test(mock)
            finally:
                await mock.reset()
                await mock.close()

        asyncio.run(go())

    wrapper.__name__ = test.__name__
    return wrapper


# ------------------------------------------------------------------ reads and typed errors


@with_mock
async def test_reads(mock):
    c = mock.client
    assert [d["id"] for d in await c.doctors("Dermatology")] == ["d1", "d2"]
    assert len(await c.doctors()) == 6
    slots = await c.free_slots("d5", "2026-12-02")
    assert FREE_SLOT in slots
    assert [a["id"] for a in await c.appointments("9876543210")] == ["A1001"]
    assert [b["id"] for b in await c.bills("9123456780")] == ["B2", "B3"]
    assert await c.now() == datetime(2026, 12, 1, 9, 0, tzinfo=IST)


@with_mock
async def test_errors_are_typed_and_carry_advice(mock):
    c = mock.client
    with pytest.raises(InvalidSlot):
        await c.book(doctor_id="d5", start="2026-12-01T08:00:00+05:30", patient_name="T", phone=NEW_PHONE)
    with pytest.raises(InvalidPhone):
        await c.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone="123")
    with pytest.raises(NotFound):
        await c.cancel("A9999")
    with pytest.raises(NotFound):
        await c.free_slots("d99", "2026-12-02")
    await c.cancel("A1001")
    with pytest.raises(NotActive):
        await c.cancel("A1001")
    with pytest.raises(AlreadyPaid):
        await c.payment_link("B2")
    with pytest.raises(SlotTaken) as info:
        await c.reschedule("A1002", "2026-12-04T17:00:00+05:30")  # Dr. Mehta is full on Friday evening
    assert "free slots" in str(info.value).lower() and info.value.code == "slot_taken"
    assert isinstance(info.value, HospitalError)


def test_same_time_ignores_format_and_zone():
    assert same_time("2026-12-04T17:00:00+05:30", "2026-12-04T11:30:00+00:00")
    assert same_time("2026-12-04T17:00:00", "2026-12-04T17:00:00+05:30")  # no zone means hospital time
    assert not same_time("2026-12-04T17:00:00+05:30", "2026-12-04T17:30:00+05:30")


# ------------------------------------------------------------------ retries on reads


def test_a_read_is_retried_when_the_hospital_is_down():
    sleeps = Sleeps()
    client, fake = scripted_client({("GET", "/doctors"): [down(), down(), ok({"doctors": [{"id": "d1"}]})]},
                                   retry=retry(sleeps))
    assert asyncio.run(client.doctors()) == [{"id": "d1"}]
    assert fake.count("GET", "/doctors") == 3 and sleeps.delays == [0.1, 0.2]


def test_a_read_gives_up_with_unavailable():
    client, fake = scripted_client({("GET", "/bills"): [down()]})
    with pytest.raises(Unavailable):
        asyncio.run(client.bills(NEW_PHONE))
    assert fake.count("GET", "/bills") == 3


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])
def test_timeouts_and_connection_errors_count_as_unavailable_and_are_retried(error):
    client, fake = scripted_client({("GET", "/bills"): [error, ok({"bills": []})]})
    assert asyncio.run(client.bills(NEW_PHONE)) == []
    assert fake.count("GET", "/bills") == 2


def test_client_errors_are_not_retried():
    body = {"error": "appointment_not_found", "message": "No appointment A1"}
    client, fake = scripted_client({("GET", "/appointments/A1"): [ok(body, 404)]})
    with pytest.raises(NotFound):
        asyncio.run(client.appointment("A1"))
    assert fake.count("GET", "/appointments/A1") == 1


def test_a_non_json_error_page_is_treated_as_unavailable():
    client, _ = scripted_client({("GET", "/bills"): [httpx.Response(502, text="<html>bad gateway</html>")]})
    with pytest.raises(Unavailable):
        asyncio.run(client.bills(NEW_PHONE))


# ------------------------------------------------------------------ writes: idempotency


def test_every_booking_carries_an_idempotency_key_reused_across_retries():
    client, fake = scripted_client({
        ("POST", "/appointments"): [down(), ok(APPT, 201)],
        ("GET", "/appointments"): [ok({"appointments": []})],  # verified: not saved
    })
    asyncio.run(client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE))
    keys = [b["idempotency_key"] for b in fake.bodies("/appointments")]
    assert len(keys) == 2 and keys[0] == keys[1] and len(keys[0]) >= 16


def test_a_caller_supplied_key_is_used():
    client, fake = scripted_client({("POST", "/appointments"): [ok(APPT, 201)]})
    asyncio.run(client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE,
                            idempotency_key="my-key"))
    assert fake.bodies("/appointments")[0]["idempotency_key"] == "my-key"


# ------------------------------------------------------------------ writes: 503, then look at the real state


@with_mock
async def test_booking_saved_but_503_is_found_and_reported_as_booked(mock):
    await mock.reset(commit_then_fail_next=1)
    appt = await mock.client.book(doctor_id="d5", start=FREE_SLOT, patient_name="Test", phone=NEW_PHONE)
    assert appt["status"] == "booked" and appt["phone"] == NEW_PHONE
    mine = [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]
    assert len(mine) == 1  # exactly one booking, no duplicate from a retry


@with_mock
async def test_reschedule_saved_but_503_is_reported_as_moved(mock):
    await mock.reset(commit_then_fail_next=1)
    appt = await mock.client.reschedule("A1001", "2026-12-07T10:00:00+05:30")
    assert appt["start"] == "2026-12-07T10:00:00+05:30"


@with_mock
async def test_cancel_saved_but_503_is_reported_as_cancelled(mock):
    await mock.reset(commit_then_fail_next=1)
    appt = await mock.client.cancel("A1001")
    assert appt["status"] == "cancelled"


def test_a_503_before_anything_was_saved_is_retried():
    client, fake = scripted_client({
        ("POST", "/appointments/A1/reschedule"): [down(), ok({**APPT, "id": "A1", "start": FREE_SLOT})],
        ("GET", "/appointments/A1"): [ok({**APPT, "id": "A1", "start": "2026-12-03T10:00:00+05:30"})],  # not moved
    })
    appt = asyncio.run(client.reschedule("A1", FREE_SLOT))
    assert appt["start"] == FREE_SLOT and fake.count("POST", "/appointments/A1/reschedule") == 2


def test_if_the_state_cannot_be_checked_a_reschedule_is_not_blindly_retried():
    client, fake = scripted_client({
        ("POST", "/appointments/A1/reschedule"): [down()],
        ("GET", "/appointments/A1"): [down()],
    })
    with pytest.raises(OutcomeUnknown) as info:
        asyncio.run(client.reschedule("A1", FREE_SLOT))
    assert fake.count("POST", "/appointments/A1/reschedule") == 1  # a blind retry could double-apply
    assert "do not tell" in str(info.value).lower()


def test_if_the_state_cannot_be_checked_a_cancel_is_not_blindly_retried():
    client, fake = scripted_client({("POST", "/appointments/A1/cancel"): [down()], ("GET", "/appointments/A1"): [down()]})
    with pytest.raises(OutcomeUnknown):
        asyncio.run(client.cancel("A1"))
    assert fake.count("POST", "/appointments/A1/cancel") == 1


def test_a_booking_that_cannot_be_checked_is_retried_with_its_key_then_reported_unknown():
    client, fake = scripted_client({("POST", "/appointments"): [down()], ("GET", "/appointments"): [down()]})
    with pytest.raises(OutcomeUnknown):
        asyncio.run(client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE))
    keys = {b["idempotency_key"] for b in fake.bodies("/appointments")}
    assert fake.count("POST", "/appointments") == 3 and len(keys) == 1  # safe to retry: same key


def test_a_booking_retry_that_succeeds_after_an_unverifiable_503_is_reported_as_booked():
    client, fake = scripted_client({
        ("POST", "/appointments"): [down(), ok(APPT, 200)],  # 200: the hospital returned the same booking for the key
        ("GET", "/appointments"): [down()],
    })
    appt = asyncio.run(client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE))
    assert appt["id"] == "A2000"


def test_a_booking_that_is_known_not_saved_after_every_try_is_unavailable_not_unknown():
    client, _ = scripted_client({("POST", "/appointments"): [down()], ("GET", "/appointments"): [ok({"appointments": []})]})
    with pytest.raises(Unavailable) as info:
        asyncio.run(client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE))
    assert not isinstance(info.value, OutcomeUnknown)
    assert "nothing was changed" in str(info.value).lower()


def test_payment_links_are_retried_on_503():
    link = {"bill_id": "B1", "url": "https://pay.example/b/B1/x"}
    client, fake = scripted_client({("POST", "/bills/B1/payment-link"): [down(), ok(link)]})
    assert asyncio.run(client.payment_link("B1")) == link
    assert fake.count("POST", "/bills/B1/payment-link") == 2


# ------------------------------------------------------------------ races and double booking


@with_mock
async def test_losing_a_race_is_not_retried_and_nothing_is_booked(mock):
    await mock.reset(race_next=1)
    with pytest.raises(SlotTaken):
        await mock.client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE)
    assert not [a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]


@with_mock
async def test_booking_a_slot_the_same_patient_already_holds_returns_it_instead_of_failing(mock):
    first = await mock.client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE)
    again = await mock.client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE)
    assert again["id"] == first["id"]
    assert len([a for a in (await mock.state())["appointments"] if a["phone"] == NEW_PHONE]) == 1


@with_mock
async def test_a_slot_held_by_someone_else_is_still_slot_taken(mock):
    await mock.client.book(doctor_id="d5", start=FREE_SLOT, patient_name="A", phone=NEW_PHONE)
    with pytest.raises(SlotTaken):
        await mock.client.book(doctor_id="d5", start=FREE_SLOT, patient_name="B", phone="9000000002")


def test_if_the_slot_check_cannot_read_after_a_409_the_outcome_is_unknown():
    taken = {"error": "slot_taken", "message": "taken"}
    client, _ = scripted_client({("POST", "/appointments"): [ok(taken, 409)], ("GET", "/appointments"): [down()]})
    with pytest.raises(OutcomeUnknown):
        asyncio.run(client.book(doctor_id="d5", start=FREE_SLOT, patient_name="T", phone=NEW_PHONE))


def test_the_client_has_no_way_to_reach_admin_endpoints():
    assert not [n for n in dir(HospitalClient) if "admin" in n.lower() or "chaos" in n.lower()]
