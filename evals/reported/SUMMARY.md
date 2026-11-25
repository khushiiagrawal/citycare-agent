# Reported eval runs

Agent: this repo at commit `f5af68c` plus nothing else (code was not changed after the first chaos run started).
Scenarios: the 10 given plus 3 of ours (`11_*`, `12_*`, `13_*`). Hospital: `python -m evals.run_evals --chaos` defaults
(10% errors, 10% races, 5% commit-then-fail, 0.3 to 2 s latency). Costs come from `/trace` with the prices in `agent_service/config.py`.

| Run | Models | Chaos | Pass rate |
|---|---|---|---|
| haiku-calm-all13 | Haiku everywhere | no | 13/13 |
| haiku-chaos-run1 | Haiku everywhere | yes | 13/13 |
| haiku-chaos-run2 | Haiku everywhere | yes | 13/13 |
| haiku-chaos-run3 | Haiku everywhere | yes | 12/13 |
| sonnet-booking-billing-chaos-run1 | Sonnet 5.5 booking and billing, Haiku elsewhere | yes | 13/13 |

## The one failure: `info_not_in_docs`, haiku-chaos-run3

Check: the reply must not contain "accept star health" (among other phrases). The reply was:

> I'm sorry, I don't have information about insurance or cashless options in the details I have, so I can't confirm whether we accept Star Health or offer cashless treatment. Our front desk can answer this. You can call 020-4000-1234 ... or I can connect you with our staff here.

The agent did the right thing (said it does not know, offered staff, gave the front desk number from `hospital_info.md`).
The phrase matched only because the reply repeats the patient's words ("whether we accept Star Health"). It is a false failure of a substring check, counted as a failure here because the given scenario and runner were not changed. Real pass rate for that run is 13/13 by reading the replies; reported rate is 12/13.

## Earlier fix (before these runs)

The first calm run of the 10 given scenarios scored 9/10: `reschedule_own` failed because "any morning slot" got a list of times and a question instead of a call to the tool. The booking prompt was changed (commit "Booking prompt: when the patient lets the agent choose, choose"); the fixed code passed 13/13 calm.

## Files

`*-run*.json` are the runner's own results (transcripts and failures). `*-metrics.json` are per-agent calls, cost and latency from `/trace`.
Tool outcomes in `haiku-chaos-metrics.json`: 116 ok, 30 paused for confirmation, 3 error, 3 denied across 108 messages; no model call failed.
