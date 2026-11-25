"""The agent service. Run: uvicorn agent_service.app:app --port 8000"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402

from agentcore.client import make_async_client  # noqa: E402
from agentcore.session import SessionBusy  # noqa: E402

from .hospital import HospitalClient  # noqa: E402
from .schemas import ChatRequest, ChatResponse  # noqa: E402
from .service import FrontDesk, SessionOwnedByAnotherPhone  # noqa: E402


async def default_service() -> FrontDesk:
    """The real thing: the async Anthropic client (SDK retries off, ours are traced) and the hospital system."""
    return FrontDesk(make_async_client(timeout=float(os.getenv("MODEL_TIMEOUT_S", "60"))), HospitalClient())


def trace_enabled_from_env() -> bool:
    return os.getenv("TRACE_ENABLED", "1").strip().lower() not in ("0", "false", "off", "no")


def create_app(factory: Callable[[], Awaitable[FrontDesk]] = default_service, *, trace_enabled: bool | None = None) -> FastAPI:
    """`factory` builds the service on first use (so importing this module needs no key). Tests pass their own.

    `GET /trace/{session_id}` returns tool inputs and outputs, which include a patient's appointments and bills,
    to anyone who knows the session id. Set TRACE_ENABLED=0 (or pass trace_enabled=False) to turn it off: the URL
    stays the same and answers 404."""
    trace_enabled = trace_enabled_from_env() if trace_enabled is None else trace_enabled
    holder: dict[str, FrontDesk] = {}

    async def service() -> FrontDesk:
        if "desk" not in holder:
            holder["desk"] = await factory()
        return holder["desk"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if "desk" in holder:
            await holder["desk"].aclose()

    app = FastAPI(title="City Care front desk agent", version="0.2", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        desk = await service()
        try:
            return await desk.chat(req)
        except SessionOwnedByAnotherPhone:
            raise HTTPException(status_code=403, detail="This session belongs to another phone number.")
        except SessionBusy:
            raise HTTPException(status_code=409, detail="Another message in this chat is still being handled.")

    @app.get("/trace/{session_id}")
    async def trace(session_id: str):
        if not trace_enabled:
            raise HTTPException(status_code=404, detail="Tracing is disabled on this server")
        traces = (await service()).trace(session_id)
        if not traces:
            raise HTTPException(status_code=404, detail="No trace for this session")
        return {
            "session_id": session_id,
            "cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
            "traces": traces,
        }

    return app


app = create_app()
