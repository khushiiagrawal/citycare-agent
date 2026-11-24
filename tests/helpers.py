"""Shared test helpers for the hospital client, tools and hooks: a scripted hospital transport and
the real mock hospital driven in process."""

import asyncio
import json
from datetime import timedelta, timezone

import httpx

from agent_service.hospital import HospitalClient
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


