"""Which model each agent uses, and what it costs. Every agent has its own setting, so one can move to
Sonnet while the rest stay on Haiku:

    BOOKING_MODEL=claude-sonnet-5-5 BOOKING_EFFORT=medium uvicorn agent_service.app:app

Variables: ROUTER_, EMERGENCY_, BOOKING_, BILLING_, INFO_ + MODEL or EFFORT (low|medium|high|xhigh|max).
HOSPITAL_NAME, HOSPITAL_ADDRESS and EMERGENCY_NUMBER are the facts used in the prompts and in the fixed
emergency reply (the info agent still gets everything else from hospital_info.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from agentcore.trace import Price

HAIKU = "claude-haiku-5-5"
DEFAULT_HOSPITAL_NAME = "City Care Hospital, Pune"
DEFAULT_HOSPITAL_ADDRESS = "12 MG Road, Camp, Pune"
DEFAULT_EMERGENCY_NUMBER = "108"
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Dollars per million tokens (context.md section 6). Cache reads and writes follow Price's defaults.
PRICES = {
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-opus-5-5": Price(4.0, 20.0),
}


def price_for(model: str) -> Price:
    """An unknown model is priced like the dearest one we know, so a budget never undercounts."""
    return PRICES.get(model) or max(PRICES.values(), key=lambda p: p.output_per_mtok)


@dataclass(frozen=True)
class ModelSetting:
    model: str
    effort: str | None = None  # None = the model's own default


@dataclass(frozen=True)
class Settings:
    router: ModelSetting
    emergency: ModelSetting
    booking: ModelSetting
    billing: ModelSetting
    info: ModelSetting
    hospital_name: str = DEFAULT_HOSPITAL_NAME
    hospital_address: str = DEFAULT_HOSPITAL_ADDRESS
    emergency_number: str = DEFAULT_EMERGENCY_NUMBER

    @property
    def short_name(self) -> str:
        """The name without the city: "City Care Hospital, Pune" -> "City Care Hospital"."""
        return self.hospital_name.split(",")[0].strip() or self.hospital_name

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        def one(name: str, effort: str | None = None) -> ModelSetting:
            chosen = env.get(f"{name}_EFFORT") or effort
            if chosen is not None and chosen not in EFFORTS:
                raise ValueError(f"{name}_EFFORT must be one of {', '.join(EFFORTS)}, not {chosen!r}")
            return ModelSetting(env.get(f"{name}_MODEL") or HAIKU, chosen)

        return cls(router=one("ROUTER"), emergency=one("EMERGENCY"), booking=one("BOOKING"),
                   billing=one("BILLING"), info=one("INFO"),
                   hospital_name=env.get("HOSPITAL_NAME") or DEFAULT_HOSPITAL_NAME,
                   hospital_address=env.get("HOSPITAL_ADDRESS") or DEFAULT_HOSPITAL_ADDRESS,
                   emergency_number=env.get("EMERGENCY_NUMBER") or DEFAULT_EMERGENCY_NUMBER)


def request_params(setting: ModelSetting, role: str) -> dict:
    """Extra request fields for a model.

    Nothing here forces a tool (`tool_choice` any/tool is rejected by the Sonnet and Opus 5.5 models; the
    classifiers answer in plain text instead). A classifier on Haiku also turns thinking off: it is allowed
    on Haiku 5.5 at low effort, it is cheaper, and a one-line label does not need it. Sonnet and Opus 5.5
    reject `thinking: disabled`, so they are left on their default."""
    params: dict = {}
    if role == "classifier":
        if setting.model.startswith("claude-haiku-5"):
            params["thinking"] = {"type": "disabled"}
        params["output_config"] = {"effort": setting.effort or "low"}
    elif setting.effort:
        params["output_config"] = {"effort": setting.effort}
    return params
