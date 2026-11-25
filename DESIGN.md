# Design

City Care front desk on WhatsApp: `agentcore/` is a small generic agent runtime, `agent_service/` is the hospital on top of it. Everything below is what the code does today.

## agentcore: what it is and why

A plain async loop around `anthropic.AsyncAnthropic` (no framework, no SDK tool runner; a sync client is rejected with a `TypeError`). It knows nothing about hospitals: `tests/test_agentcore_boundary.py` fails if it imports `agent_service` or uses domain words.

- **Loop** (`loop.py`): call the model, run the tools it asked for concurrently, repeat up to `max_turns`. Results come back as one `RunResult` (text, stop reason, tool records, tokens, cost). A failing tool becomes an error result for the model; it never kills the run.
- **Tools** (`types.py`): a function plus a pydantic input model. The schema is built from the model and every call is validated strictly (wrong type, missing or extra field) before the handler runs. The error goes back to the model.
- **Hooks**: `before_model`, `after_model`, `before_tool`, `after_tool`, `on_finish`. `before_tool` returns `Allow`, `Deny(reason)`, `Modify(input)` or `Confirm(prompt)`. `Confirm` pauses the run and returns code-written text to the user. `ConfirmGate` allows a guarded call only if the model repeats the identical call (same tool, same input) in the next run; each confirmation works once.
- **Writes**: tools marked `writes=True` are capped per run (we use 1). The cap is checked before the hooks, so a paused call does not count. A write that raises an exception with `changed_nothing = True` is given back, so a lost race does not block offering another time.
- **Sessions** (`session.py`): one lock per conversation, bounded in-memory store, history trimmed only at user-message boundaries, and a history that ends in a `tool_use` without a result is repaired. If a run raises, nothing is saved and the turn counter is restored.
- **Resilience**: retries for 408/409/429/5xx, timeouts and dropped connections, honouring `retry-after`; spend-limit 429s are never retried; per-attempt timeout; `ModelUnavailable` when it gives up. `Budget` caps cost, tokens and seconds per run and cost per session.
- **Tracing**: every model call and tool call with timing, tokens, cache tokens and cost (prices are passed in). Optional prompt-cache breakpoints on the system prompt and the end of the tool list (`cache=True`).

## What the model decides, what code decides

| Model (Haiku 5.5 by default) | Code |
|---|---|
| Understanding the patient, in English, Hindi or Hinglish | Whose data (phone comes from the request; no tool has a phone argument; ids checked against the caller's own records; results scrubbed of other numbers) |
| Which tool to call and with what | Whether the call may run: ownership, the 4-hour rule (against the existing appointment, with the hospital's clock `/now`), one write per message |
| Wording of ordinary replies | The confirmation: the read-back is built by code from real data in the patient's language, and the model never asks "shall I book it?" itself. The write runs only if the model repeats the same call after the patient's next message |
| Routing and the patient's language and name | Emergency reply (108 or nearest emergency room, no advice), handoff messages, the `emergency` and `handoff` fields, the "that time was just taken" notice |
| Choosing a nearby time after a lost slot | Replacing a reply that claims "booked/cancelled/moved/link sent" without a tool having done it (`ClaimGuard`), or that gives medical advice (`NoMedicalAdvice`). Both are regex nets in three languages, not proofs |

Message path: emergency check (regex, then a Haiku check) → router (Haiku: booking, billing, info or human; also language and name) → one specialist with only its own tools plus `handoff_to_staff`. While a read-back is waiting, the next message skips the router and goes to the agent that asked, and any agent that did not handle a message forgets its pending confirmation. Agents share the patient's name, language and flags through one state dict; a new agent also sees the last few messages once.

## Failures

| Failure | Handling |
|---|---|
| Hospital 503 on a write | The client reads the real state before answering: saved → return it; not saved → retry; unreadable → bookings retry with the same idempotency key, reschedule/cancel raise `OutcomeUnknown` (never "done", never "failed"), which flags a handoff |
| Saved but 503 (`commit_then_fail`) | Same check; one booking, reported as booked |
| Slot lost to someone else | `SlotTaken`; code tells the patient, the agent offers the nearest free time in the same message and one yes books it. The failed write does not use the cap |
| Hospital reads failing | 3 attempts with backoff, then an error with advice; two failures in a row flag a handoff |
| Model unreachable or run stops early | Retries, then a fixed apology and `handoff: true`; the chat keeps working afterwards. If only the emergency check fails, the regex result stands and the router is a third net |
| Runaway loop or cost | `max_turns` 8 and a budget of $0.25, 90 s per run and $2 per agent session → handoff |
| Same session id, other phone | 403 before anything runs |
| Two messages at once | Per-session lock (409 after 45 s) |
| Prompt injection / other people's data | Tool results never contain phones; a result holding another mobile number is withheld; ownership is checked in the hook, the tool and the read-back |

## Models, cost and speed

Each agent has its own `*_MODEL` and `*_EFFORT` setting (`config.py`). All five default to `claude-haiku-5-5`. The router and emergency check run with thinking off and effort low; specialists use the model's default. Nothing forces `tool_choice` (Sonnet and Opus 5.5 reject it), so the classifiers answer in plain text.

Numbers below come from `/trace` of the 13 scenarios on a calm hospital, using the prices in `config.py` (not an invoice). Haiku everywhere: 36 patient messages cost $0.0162, about **$0.00045 per message**.

| Call | Count | Cost per call | p50 / p90 latency |
|---|---|---|---|
| emergency check | 33 | $0.00004 | 0.71 s / 0.82 s |
| router | 24 | $0.00009 | 0.83 s / 1.23 s |
| booking | 52 | $0.00023 | 1.35 s / 1.99 s |
| billing | 10 | $0.00010 | 1.02 s / 1.42 s |

A non-emergency message took 2.7 to 5.5 s end to end (scenario time divided by messages: emergency check, router, one to three specialist calls, hospital calls). Specialist prompts are cached (booking read about 139k tokens from cache against 58k uncached input); the router and emergency prompts are below the minimum cacheable size.

**Sonnet 5.5 for booking and billing** (router, emergency check and info stay on Haiku), one chaos run of the same 13 scenarios, same traces:

| | Haiku everywhere (3 chaos runs) | Sonnet booking + billing (1 chaos run) |
|---|---|---|
| Pass rate | 13/13, 13/13, 12/13 | 13/13 |
| Cost per message | $0.00046 | $0.00624 (13.6 times more) |
| Booking call: cost, p50 / p90 | $0.00023, 1.41 s / 2.08 s | $0.00377, 1.71 s / 2.40 s |
| Billing call: cost, p50 / p90 | $0.00009, 1.04 s / 1.56 s | $0.00250, 0.92 s / 1.67 s |

On these scenarios Sonnet did not pass more, and cost about 14 times as much; each call also costs about 16 to 27 times more than on Haiku. One run is not enough to say Sonnet is no better: both models passed nearly everything, so the scenarios cannot separate them. Raw files are in `evals/reported/`.

## Phone calls: what would change

The loop, hooks, tools, read-back and guards stay. What does not carry over:
- **Latency**: 3 to 5 s per turn is too slow for speech. The emergency check and the router run one after the other today; they would run in parallel, and replies would need to stream.
- **Identity**: the phone is trusted because WhatsApp verified it. Caller ID is not verified, so identity needs a second factor before any own-data tool.
- **Confirmations**: a spoken yes through speech-to-text is noisier than a typed one; read-backs need shorter wording and a stricter yes.
- **Emergency and language**: the regex runs on transcripts with recognition errors and no script cue; language comes from the speech, not the text.
- **Handoff**: today it is a flag in the response. A call needs a warm transfer to a staff line.

## Missing before real patients

- **State is in memory**: sessions, pending confirmations, conversations and traces vanish on restart, and locks work in one process only. It needs a persistent store behind `SessionStore` and shared locks.
- **No auth on the service**: `/chat` trusts the caller's phone. `/trace/{session_id}` returns tool inputs and outputs to anyone with the session id; `TRACE_ENABLED=0` turns it off. Traces hold patient data with no retention rule.
- **Handoff goes nowhere yet**: the flag is returned and stays on for the chat, but there is no staff queue or screen.
- **The guards are regexes**: no measured false-positive or false-negative rate for the claim, advice and emergency patterns, and the emergency list has had no clinical review.
- **Thin evidence**: 13 scenarios, a handful of runs, Haiku variance. See `EVALS.md`.
- **Smaller gaps**: the budget's session limit applies per agent session, not per chat; doctor names are cached for the life of the process; `hospital_info.md` is read at start-up; the info agent trusts that file completely.
