"""Settings per agent, the router (a Haiku classifier) and the three specialist agents."""

import asyncio

import pytest

from agent_service.agents import build_agents, build_router
from agent_service.config import ModelSetting, Settings, price_for, request_params
from agent_service.router import Route, parse_route
from agent_service.tools import Hospital
from agentcore.hooks import ConfirmGate
from agentcore.types import RunContext
from tests.fake_claude import FakeClaude, text
from tests.helpers import with_mock

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"

# ------------------------------------------------------------------ settings


def test_every_agent_defaults_to_haiku():
    s = Settings.from_env({})
    assert {m.model for m in (s.router, s.emergency, s.booking, s.billing, s.info)} == {HAIKU}


def test_each_agents_model_and_effort_can_be_set_separately():
    s = Settings.from_env({"BOOKING_MODEL": SONNET, "BOOKING_EFFORT": "high", "INFO_MODEL": "claude-opus-5-5"})
    assert s.booking == ModelSetting(SONNET, "high") and s.info.model == "claude-opus-5-5"
    assert s.billing.model == HAIKU and s.router.model == HAIKU


def test_a_bad_effort_is_rejected_at_start_up():
    with pytest.raises(ValueError, match="BOOKING_EFFORT"):
        Settings.from_env({"BOOKING_EFFORT": "turbo"})


def test_prices_follow_the_model_and_an_unknown_model_is_priced_high_not_free():
    assert price_for(HAIKU).input_per_mtok == 0.10 and price_for(SONNET).output_per_mtok == 10
    assert price_for("claude-new-thing").input_per_mtok >= price_for(SONNET).input_per_mtok


def test_request_params_never_force_a_tool_and_only_disable_thinking_where_it_is_allowed():
    for model in (HAIKU, SONNET, "claude-opus-5-5"):
        for role in ("classifier", "specialist"):
            params = request_params(ModelSetting(model, "low"), role)
            assert "tool_choice" not in params  # forced tool use is rejected by the Sonnet and Opus 5.5 models
            if params.get("thinking", {}).get("type") == "disabled":
                assert model == HAIKU and role == "classifier"
    assert request_params(ModelSetting(HAIKU, None), "classifier")["thinking"] == {"type": "disabled"}
    assert "thinking" not in request_params(ModelSetting(SONNET, "low"), "classifier")
    assert request_params(ModelSetting(SONNET, "high"), "specialist") == {"output_config": {"effort": "high"}}
    assert request_params(ModelSetting(HAIKU, None), "specialist") == {}


# ------------------------------------------------------------------ reading the router's answer


@pytest.mark.parametrize("raw", [
    '{"route": "billing", "language": "hinglish", "patient_name": "Priya Nair", "emergency": false}',
    'Sure!\n```json\n{"route": "billing", "language": "hinglish", "patient_name": "Priya Nair", "emergency": false}\n```',
    '{"route": "BILLING", "language": "Hinglish", "patient_name": "Priya Nair", "emergency": "false"}',
])
def test_parse_route_reads_json_however_it_is_wrapped(raw):
    assert parse_route(raw, last_agent=None) == Route("billing", "hinglish", "Priya Nair", False, False)


def test_parse_route_treats_empty_names_as_unknown_and_reads_the_emergency_flag():
    r = parse_route('{"route": "info", "language": "en", "patient_name": "", "emergency": true}', None)
    assert r.patient_name is None and r.emergency is True


@pytest.mark.parametrize("raw", ["", "I think booking", "{not json", '{"route": "pharmacy"}', '{"route": 7}'])
def test_an_unusable_answer_keeps_the_current_agent_or_falls_back_to_info(raw):
    assert parse_route(raw, last_agent="booking") == Route("booking", None, None, False, True)
    assert parse_route(raw, last_agent=None).route == "info"


def test_an_unknown_language_is_dropped_not_guessed():
    assert parse_route('{"route": "info", "language": "klingon"}', None).language is None


# ------------------------------------------------------------------ the router agent


def route_with(*script, settings=None):
    fake = FakeClaude(list(script))
    return build_router(fake, settings or Settings.from_env({})), fake


def test_the_router_sends_the_recent_chat_and_the_current_agent():
    from agent_service.router import Router

    agent, fake = route_with(text('{"route": "booking", "language": "en", "patient_name": null, "emergency": false}'))
    result = asyncio.run(Router(agent).route(
        "yes please", "s1", last_agent="booking", recent=[("Patient", "book Dr. Iyer"), ("Assistant", "Shall I book?")]))
    assert result.route == "booking"
    sent = fake.messages.calls[0]["messages"][0]["content"]
    assert "Last agent: booking" in sent and "book Dr. Iyer" in sent and "<message>\nyes please\n</message>" in sent
    call = fake.messages.calls[0]
    assert call["model"] == HAIKU and call["thinking"] == {"type": "disabled"} and "tools" not in call
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


# ------------------------------------------------------------------ the specialists


def names(agent):
    return {t.name for t in agent.tools}


@with_mock
async def test_each_specialist_gets_only_its_own_tools_plus_handoff(mock):
    agents = build_agents(FakeClaude([]), Hospital(mock.client), Settings.from_env({}), hospital_info="INFO")
    assert names(agents["booking"]) == {"find_doctors", "get_current_time", "free_slots", "my_appointments",
                                        "book_appointment", "reschedule_appointment", "cancel_appointment",
                                        "handoff_to_staff"}
    assert names(agents["billing"]) == {"my_bills", "send_payment_link", "handoff_to_staff"}
    assert names(agents["info"]) == {"handoff_to_staff"}


@with_mock
async def test_the_info_agent_has_the_hospital_facts_and_no_hospital_tools(mock):
    agents = build_agents(FakeClaude([]), Hospital(mock.client), Settings.from_env({}), hospital_info="OPD 10am to 1pm")
    info = agents["info"]
    assert "OPD 10am to 1pm" in info.system and "do not know" in info.system.lower()
    assert not any(t.name in {"my_bills", "my_appointments", "find_doctors"} for t in info.tools)


@with_mock
async def test_the_real_hospital_info_file_is_used_by_default(mock):
    info = build_agents(FakeClaude([]), Hospital(mock.client), Settings.from_env({}))["info"]
    assert "12 MG Road" in info.system and "020-4000-1234" in info.system


@with_mock
async def test_specialists_cache_their_prompts_allow_one_write_and_use_their_own_model(mock):
    s = Settings.from_env({"BILLING_MODEL": SONNET})
    agents = build_agents(FakeClaude([]), Hospital(mock.client), s, hospital_info="x")
    assert all(a.cache for a in agents.values()) and all(a.max_writes_per_run == 1 for a in agents.values())
    assert agents["billing"].model == SONNET and agents["booking"].model == HAIKU
    assert agents["billing"].price.output_per_mtok == 10 and agents["booking"].price.output_per_mtok == 0.5


@with_mock
async def test_safety_hooks_are_on_every_specialist_with_the_confirm_gate_last_of_the_tool_hooks(mock):
    from agent_service.guards import ClaimGuard, KnownFacts, NoMedicalAdvice
    from agent_service.hooks import FourHourRule, OwnershipHook

    for agent in build_agents(FakeClaude([]), Hospital(mock.client), Settings.from_env({}), hospital_info="x").values():
        kinds = [type(h) for h in agent.hooks]
        for needed in (OwnershipHook, FourHourRule, ConfirmGate, ClaimGuard, NoMedicalAdvice, KnownFacts):
            assert needed in kinds
        tool_hooks = [k for k in kinds if k in (OwnershipHook, FourHourRule, ConfirmGate)]
        assert tool_hooks[-1] is ConfirmGate


@with_mock
async def test_the_prompts_state_the_rules_the_code_also_enforces(mock):
    agents = build_agents(FakeClaude([]), Hospital(mock.client), Settings.from_env({}), hospital_info="x")
    for agent in agents.values():
        low = agent.system.lower()
        for must in ("medical advice", "hindi", "hinglish", "handoff_to_staff"):
            assert must in low, (agent.name, must)
    assert "phone" in agents["booking"].system.lower()  # never ask for it
