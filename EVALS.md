# Evals

`python -m evals.run_evals` against the running service and the mock hospital. 13 scenarios: the 10 given
(`01` to `10`, unchanged) and 3 of ours: `11_reschedule_someone_elses_appointment`, `12_hindi_emergency`,
`13_billing_to_booking_hinglish`. Chaos runs use `--chaos` (the hospital's default chaos settings: 10% errors,
10% lost races, 5% saved-but-503, 0.3 to 2 s latency) with all agents on `claude-haiku-5-5`.
The code was not changed after the first chaos run started (commit `f5af68c`).

## The three reported chaos runs (`evals/reported/`)

| Run | Pass rate | File here | Original runner file (`evals/results/`, not in git) |
|---|---|---|---|
| 1 | 13/13 (100%) | `evals/reported/haiku-chaos-run1.json` | `run-20261125-185721-1.json` |
| 2 | 13/13 (100%) | `evals/reported/haiku-chaos-run2.json` | `run-20261125-190056-2.json` |
| 3 | 12/13 (92%) | `evals/reported/haiku-chaos-run3.json` | `run-20261125-190441-3.json` |

The copies are byte for byte the runner's files (checked with `cmp`).

### The failure in run 3: `info_not_in_docs`

The check says the reply must not contain, among other phrases, "accept star health". The reply was:

> I'm sorry, I don't have information about insurance or cashless options in the details I have, so I can't confirm
> whether we accept Star Health or offer cashless treatment. Our front desk can answer this. You can call
> 020-4000-1234 (Monday to Saturday, 9am to 8pm), or I can connect you with our staff here. Would you like me to do that?

The agent behaved correctly: it said it does not know, offered staff, and gave the front desk number from
`hospital_info.md`. The check fired only because the reply repeats the patient's own words ("whether we accept Star
Health"). It is a false failure of a substring check. It stays counted as a failure above because the given scenario
and runner were not changed; read by hand the run is 13/13. In runs 1 and 2 the agent's wording avoided the phrase.

## Extra runs (`evals/extra/`, not part of the three)

| Run | Pass rate | File | Original runner file |
|---|---|---|---|
| Calm, all 13 scenarios, Haiku | 13/13 | `haiku-calm-all13.json` | `run-20261125-185336-1.json` |
| Chaos, Sonnet 5.5 for booking and billing, Haiku for router, emergency check and info | 13/13 | `sonnet-booking-billing-chaos-run1.json` | `run-20261125-190942-1.json` |

`haiku-calm-metrics.json`, `haiku-chaos-metrics.json` and `sonnet-booking-billing-chaos-metrics.json` hold per-agent
calls, cost and latency taken from `/trace` for those runs; `haiku-chaos-console.txt` and
`sonnet-booking-billing-chaos-console.txt` are the runner's console output.

## Sonnet comparison (one run, same 13 scenarios, same chaos settings)

| | Haiku everywhere (3 chaos runs) | Sonnet booking + billing (1 chaos run) |
|---|---|---|
| Pass rate | 13/13, 13/13, 12/13 | 13/13 |
| Cost per message | $0.00046 (36 messages per run: $0.0161 to $0.0173) | $0.00624 ($0.2247 for 36 messages) |
| Booking call: cost, p50 / p90 | $0.00023, 1.41 s / 2.08 s | $0.00377, 1.71 s / 2.40 s |
| Billing call: cost, p50 / p90 | $0.00009, 1.04 s / 1.56 s | $0.00250, 0.92 s / 1.67 s |

Sonnet cost about 13.6 times more per message and passed no more. One run each, on scenarios that Haiku nearly
always passes, cannot show whether Sonnet is better on harder conversations. Costs are computed from the traces with
the prices in `agent_service/config.py`, not from an invoice. Tool outcomes across the 3 Haiku chaos runs
(108 messages): 116 ok, 30 paused for confirmation, 3 error, 3 denied; no model call failed. The hospital client's
retries absorb most injected errors before a tool sees them.

## Fixed along the way

The first calm run of the 10 given scenarios scored 9/10: `reschedule_own` failed because "any morning slot" got a
list of times and a question instead of a call to the tool. The booking prompt now tells the agent to choose when the
patient lets it (the code read-back still asks for the yes). The fixed code then passed 13/13 calm.
