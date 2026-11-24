import asyncio

import anthropic
import httpx
import pytest
from anthropic import AsyncAnthropic  # the real class, captured before the conftest patch

from agentcore.budget import Budget
from agentcore.client import make_async_client  # imported here, before the no-real-client fixture patches the SDK
from agentcore.hooks import Hook
from agentcore.loop import Agent
from agentcore.resilience import ModelUnavailable, RetryPolicy, is_retryable
from agentcore.trace import Price, Tracer
from agentcore.types import Registry, RunContext, Tool
from tests.fake_claude import FakeClaude, text, tool_use

REQ = httpx.Request("POST", "http://example.test")


def api_error(cls, code, **headers):
    return cls("boom", response=httpx.Response(code, request=REQ, headers=headers), body=None)


class Sleeps:
    """A fake sleep that records the delays instead of waiting."""

    def __init__(self):
        self.delays = []

    async def __call__(self, seconds):
        self.delays.append(seconds)


def policy(sleeps=None, **kw):
    kw.setdefault("jitter", 0.0)
    return RetryPolicy(sleep=sleeps or Sleeps(), **kw)


def make(script, retry=None, **kw):
    fake = FakeClaude(script)
    return Agent(client=fake, model="m", system="s", tools=Registry(), retry=retry, **kw), fake


def run(agent, ctx=None):
    return asyncio.run(agent.run([{"role": "user", "content": "hi"}], ctx))


# ------------------------------------------------------------------ which errors are retried


@pytest.mark.parametrize("error", [
    api_error(anthropic.RateLimitError, 429),
    api_error(anthropic.InternalServerError, 500),
    api_error(anthropic.ServiceUnavailableError, 503),
    api_error(anthropic.OverloadedError, 529),
    api_error(anthropic.APIStatusError, 408),
    anthropic.APIConnectionError(request=REQ),
    anthropic.APITimeoutError(request=REQ),
    TimeoutError("slow"),
])
def test_transient_errors_are_retryable(error):
    assert is_retryable(error)


@pytest.mark.parametrize("error", [
    api_error(anthropic.BadRequestError, 400),
    api_error(anthropic.AuthenticationError, 401),
    api_error(anthropic.PermissionDeniedError, 403),
    api_error(anthropic.NotFoundError, 404),
    api_error(anthropic.EnforcedSpendLimitReachedError, 429),  # a 429, but waiting will not help
    api_error(anthropic.ContractSpendLimitReachedError, 429),
    ConnectionError("not an SDK error"),
    ValueError("bug"),
])
def test_other_errors_are_not_retryable(error):
    assert not is_retryable(error)


# ------------------------------------------------------------------ the retry loop


def test_a_transient_error_is_retried_and_the_run_succeeds():
    sleeps = Sleeps()
    agent, fake = make([api_error(anthropic.RateLimitError, 429), text("ok")], retry=policy(sleeps))
    res = run(agent)
    assert res.text == "ok" and res.model_calls == 1 and len(fake.messages.calls) == 2
    assert len(sleeps.delays) == 1


def test_delays_grow_exponentially_and_are_capped():
    sleeps = Sleeps()
    errs = [api_error(anthropic.InternalServerError, 500) for _ in range(3)]
    agent, _ = make([*errs, text("ok")], retry=policy(sleeps, max_attempts=4, base_delay_s=1.0, max_delay_s=3.0))
    run(agent)
    assert sleeps.delays == [1.0, 2.0, 3.0]


def test_jitter_only_adds_a_little():
    p = RetryPolicy(base_delay_s=1.0, jitter=0.25)
    delays = {p.delay(1, RuntimeError()) for _ in range(20)}
    assert all(1.0 <= d <= 1.25 for d in delays) and len(delays) > 1


def test_retry_after_header_is_respected_up_to_the_cap():
    sleeps = Sleeps()
    err = api_error(anthropic.RateLimitError, 429, **{"retry-after": "2"})
    agent, _ = make([err, text("ok")], retry=policy(sleeps, base_delay_s=0.5, max_delay_s=8.0))
    run(agent)
    assert sleeps.delays == [2.0]
    sleeps2 = Sleeps()
    err = api_error(anthropic.RateLimitError, 429, **{"retry-after": "120"})
    agent, _ = make([err, text("ok")], retry=policy(sleeps2, base_delay_s=0.5, max_delay_s=8.0))
    run(agent)
    assert sleeps2.delays == [8.0]


def test_giving_up_raises_model_unavailable_with_the_cause():
    sleeps = Sleeps()
    last = api_error(anthropic.OverloadedError, 529)
    agent, fake = make([api_error(anthropic.OverloadedError, 529)] * 2 + [last], retry=policy(sleeps, max_attempts=3))
    with pytest.raises(ModelUnavailable) as info:
        run(agent)
    assert info.value.attempts == 3 and info.value.__cause__ is last
    assert len(fake.messages.calls) == 3 and len(sleeps.delays) == 2  # no pointless sleep after the last try


def test_permanent_errors_fail_at_once_without_waiting():
    sleeps = Sleeps()
    err = api_error(anthropic.BadRequestError, 400)
    agent, fake = make([err, text("never")], retry=policy(sleeps))
    with pytest.raises(anthropic.BadRequestError):
        run(agent)
    assert len(fake.messages.calls) == 1 and sleeps.delays == []


def test_retry_can_be_switched_off():
    agent, fake = make([api_error(anthropic.RateLimitError, 429), text("never")], retry=None)
    with pytest.raises(anthropic.RateLimitError):
        run(agent)
    assert len(fake.messages.calls) == 1


def test_agents_retry_by_default():
    agent = Agent(client=FakeClaude([]), model="m", system="s", tools=Registry())
    assert isinstance(agent.retry, RetryPolicy) and agent.retry.max_attempts == 3


def test_policy_must_allow_at_least_one_attempt():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_each_attempt_is_traced_and_only_the_success_costs_money():
    tracer = Tracer()
    agent, _ = make([api_error(anthropic.RateLimitError, 429), text("ok", input_tokens=1000, output_tokens=0)],
                    retry=policy(), tracer=tracer, price=Price(2.0, 10.0))
    run(agent, RunContext(key="k"))
    first, second = tracer.traces("k")[0].events
    assert first.data["attempt"] == 1 and "RateLimitError" in first.data["error"]
    assert second.data["attempt"] == 2 and second.data["cost_usd"] == pytest.approx(0.002)
    assert tracer.traces("k")[0].cost_usd == pytest.approx(0.002)


class HangingThenFine:
    def __init__(self):
        self.calls = 0
        self.messages = self

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(5)
        return text("recovered")


def test_a_hung_model_call_times_out_and_is_retried():
    client = HangingThenFine()
    agent = Agent(client=client, model="m", system="s", tools=Registry(),
                  retry=policy(attempt_timeout_s=0.05))
    assert run(agent).text == "recovered" and client.calls == 2


def test_a_model_that_always_hangs_ends_in_model_unavailable():
    class Hang:
        def __init__(self):
            self.messages = self

        async def create(self, **kwargs):
            await asyncio.sleep(5)

    agent = Agent(client=Hang(), model="m", system="s", tools=Registry(),
                  retry=policy(max_attempts=2, attempt_timeout_s=0.02))
    with pytest.raises(ModelUnavailable):
        run(agent)


def test_make_async_client_turns_off_sdk_retries_so_they_do_not_stack():
    client = make_async_client(api_key="test-key")
    assert isinstance(client, AsyncAnthropic) and client.max_retries == 0


# ------------------------------------------------------------------ budget


def echo_tool(handler=None):
    return Tool(name="t", description="t", input_schema={"type": "object", "properties": {}},
                handler=handler or (lambda i, c: "ok"))


def looping(script, budget, tools=None, **kw):
    fake = FakeClaude(script)
    return Agent(client=fake, model="m", system="s", tools=Registry(tools or [echo_tool()]), budget=budget, **kw), fake


def test_cost_ceiling_stops_the_run_before_the_next_model_call():
    script = [tool_use("t", {}, input_tokens=1000, output_tokens=0) for _ in range(5)]
    agent, fake = looping(script, Budget(max_cost_usd=0.003), price=Price(2.0, 10.0))
    res = run(agent)
    assert res.stop_reason == "budget_exceeded" and res.model_calls == 2
    assert "cost" in res.reason.lower() and res.text == ""
    assert res.messages[-1]["role"] == "user"  # history is still valid: every tool call has its result
    assert len(fake.messages.script) == 3


def test_token_ceiling():
    script = [tool_use("t", {}, input_tokens=600, output_tokens=0) for _ in range(5)]
    agent, _ = looping(script, Budget(max_tokens=1000))
    res = run(agent)
    assert res.stop_reason == "budget_exceeded" and res.model_calls == 2 and "token" in res.reason.lower()


def test_time_ceiling_counts_slow_tools():
    async def slow(input, ctx):
        await asyncio.sleep(0.06)
        return "ok"

    agent, _ = looping([tool_use("t", {}), text("never")], Budget(max_seconds=0.05), tools=[echo_tool(slow)])
    res = run(agent)
    assert res.stop_reason == "budget_exceeded" and res.model_calls == 1 and "second" in res.reason.lower()


def test_session_ceiling_uses_everything_spent_in_earlier_runs():
    ctx = RunContext(key="k", total_cost_usd=5.0)
    agent, fake = looping([text("never")], Budget(max_session_cost_usd=5.0))
    res = run(agent, ctx)
    assert res.stop_reason == "budget_exceeded" and res.model_calls == 0 and len(fake.messages.calls) == 0


def test_a_run_under_budget_is_untouched():
    agent, _ = looping([tool_use("t", {}), text("done")],
                       Budget(max_cost_usd=1.0, max_tokens=10_000, max_seconds=60, max_session_cost_usd=10),
                       price=Price(2.0, 10.0))
    res = run(agent)
    assert res.stop_reason == "end_turn" and res.text == "done"


def test_no_limits_means_no_limits():
    assert Budget().exceeded(spent_usd=1e9, tokens=10**9, seconds=1e9, session_usd=1e9) is None


def test_budget_stop_is_traced_and_on_finish_still_runs():
    seen = []

    class Watch(Hook):
        def on_finish(self, result, ctx):
            seen.append(result.stop_reason)

    tracer = Tracer()
    agent, _ = looping([text("never")], Budget(max_session_cost_usd=1.0), hooks=[Watch()], tracer=tracer)
    run(agent, RunContext(key="k", total_cost_usd=2.0))
    assert seen == ["budget_exceeded"] and tracer.traces("k")[0].stop_reason == "budget_exceeded"
