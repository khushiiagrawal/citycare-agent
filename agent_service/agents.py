"""Builds the router, the emergency check and the three specialists from `Settings`.

Every specialist has the same safety hooks (ownership, 4-hour rule, confirmation, leak guard, repeated-failure
handoff, no false "done", no medical advice, shared facts), one write per message, cached prompts, and the
handoff tool. They differ only in prompt and tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentcore.budget import Budget
from agentcore.loop import Agent
from agentcore.trace import Tracer
from agentcore.types import Registry

from .config import ModelSetting, Settings, price_for, request_params
from .emergency import CLASSIFIER_SYSTEM, EmergencyCheck
from .guards import ClaimGuard, KnownFacts, NoMedicalAdvice
from .hooks import build_hooks
from .prompts import billing_prompt, booking_prompt, info_prompt
from .router import Router, router_system
from .tools import Hospital, build_handoff_tool, build_tools

HOSPITAL_INFO = Path(__file__).resolve().parent.parent / "hospital_info.md"

BOOKING_TOOLS = ["find_doctors", "get_current_time", "free_slots", "my_appointments",
                 "book_appointment", "reschedule_appointment", "cancel_appointment"]
BILLING_TOOLS = ["my_bills", "send_payment_link"]


def _classifier(client: Any, setting: ModelSetting, system: str, name: str, tracer: Tracer | None, max_tokens: int) -> Agent:
    return Agent(client, setting.model, system, name=name, max_tokens=max_tokens, cache=True, tracer=tracer,
                 price=price_for(setting.model), request_params=request_params(setting, "classifier"))


def build_router(client: Any, settings: Settings, *, tracer: Tracer | None = None) -> Agent:
    return _classifier(client, settings.router, router_system(settings.hospital_name), "router", tracer, 200)


def build_emergency_check(client: Any, settings: Settings, *, tracer: Tracer | None = None) -> EmergencyCheck:
    return EmergencyCheck(_classifier(client, settings.emergency, CLASSIFIER_SYSTEM, "emergency_check", tracer, 20))


def build_agents(
    client: Any,
    hospital: Hospital,
    settings: Settings,
    *,
    hospital_info: str | None = None,
    tracer: Tracer | None = None,
    budget: Budget | None = None,
) -> dict[str, Agent]:
    info = hospital_info if hospital_info is not None else HOSPITAL_INFO.read_text(encoding="utf-8")
    tools = build_tools(hospital)
    handoff = build_handoff_tool()

    def make(name: str, setting: ModelSetting, system: str, names: list[str]) -> Agent:
        registry = tools.subset(names)
        registry.add(handoff)
        return Agent(
            client, setting.model, system, registry, name=name, max_tokens=2048, max_turns=8,
            max_writes_per_run=1, cache=True, tracer=tracer, budget=budget, price=price_for(setting.model),
            request_params=request_params(setting, "specialist"),
            hooks=[*build_hooks(hospital), ClaimGuard(), NoMedicalAdvice(), KnownFacts()],
        )

    return {
        "booking": make("booking", settings.booking, booking_prompt(settings.hospital_name, settings.emergency_number), BOOKING_TOOLS),
        "billing": make("billing", settings.billing, billing_prompt(settings.hospital_name, settings.emergency_number), BILLING_TOOLS),
        "info": make("info", settings.info, info_prompt(info, settings.hospital_name, settings.emergency_number), []),
    }


def build_router_service(client: Any, settings: Settings, *, tracer: Tracer | None = None) -> Router:
    return Router(build_router(client, settings, tracer=tracer))
