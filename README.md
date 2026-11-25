# City Care agent: starter repo

Starter code for the **agentcore** assignment. Read [`context.md`](context.md) first; the full brief is on the assignment page we sent you.

## What's here

| Path | What it is |
|---|---|
| `context.md` | The product, the hospital rules, the hospital system, models and cost, evals. Read first |
| `hospital_info.md` | The only facts the info agent may use |
| `mock_hospital/` | A fake hospital system (appointments and billing) with a chaos mode. Read it, don't change how it behaves |
| `agent_service/` | The FastAPI app (`POST /chat`, `GET /trace/{session_id}`): emergency check, router, booking / billing / info agents |
| `evals/` | 10 scenarios and a runner that checks what really happened |
| `tests/` | Example tests and `fake_claude.py`, a scripted Claude client so tests run without a key |
| `tools/pack_sessions.py` | Packs your Claude Code transcripts for hand-in (see [Hand-in](#hand-in)) |

You'll add `agentcore/` (the runtime) and your agents.

## Your own copy

Clone or fork this repo into your own GitHub repo, and work there. To clone it into a new repo, create an empty repo on GitHub, then:

```bash
git clone <this repo's URL> citycare-agent
cd citycare-agent
git remote set-url origin <your new repo's URL>
git push -u origin main
```

Your copy can be **public**, or **private** with the **Fire-Dolphin** account invited as a collaborator (Settings → Collaborators → Add people). Either way, you send us its link when you reply to our email. Commit and push as you go.

## Setup

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cp .env.example .env   # then put your API key in .env
```

Or with pip:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip install -r requirements.txt
```

(With pip, drop the `uv run` prefix from the commands below and activate the venv instead.)

Use the API key we sent you for both Claude Code and your agents. In your terminal, before starting Claude Code:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Run it

Terminal 1, the hospital system (calm):

```bash
uv run uvicorn mock_hospital.app:app --port 8001
```

The same in chaos mode (slow, flaky, races):

```bash
CHAOS=1 uv run uvicorn mock_hospital.app:app --port 8001      # PowerShell: $env:CHAOS="1"; uv run uvicorn ...
```

Terminal 2, the agent:

```bash
uv run uvicorn agent_service.app:app --port 8000
```

Try it:

```bash
curl -s localhost:8000/chat -H "content-type: application/json" -d '{"session_id":"t1","phone":"9000000001","message":"I need a skin doctor on Friday evening"}'
```

## Evals

With both services running:

```bash
uv run python -m evals.run_evals                         # every scenario, once
uv run python -m evals.run_evals --runs 3 --chaos        # what we ask you to report
uv run python -m evals.run_evals --only emergency_hinglish
```

The runner resets the hospital before each scenario and turns chaos on or off itself, so you don't need to restart the hospital between runs.

## Tests

```bash
uv run pytest
```

They must pass without an API key.

## Claude Code sessions

We read your sessions as part of the review, so:

- **Always start Claude Code from this folder** (the repo root). Claude Code files transcripts by folder.
- To pick up where you left off the next day: `claude --continue`.
- **Before you `/clear` or close a session, export it** into `claude-sessions/`, numbered in order:

```
/export claude-sessions/01-planning.txt
/export claude-sessions/02-agentcore-loop.txt
```

- Don't edit exports or transcripts, except to remove an API key from an export.

## Hand-in

Before the deadline, run:

```bash
uv run python tools/pack_sessions.py
```

It zips the full transcripts Claude Code keeps on your machine into `claude-sessions/transcripts.zip`, including the parts of long sessions that Claude Code compacted. It also stops if it finds an API key anywhere in the repo. Then commit and push.

Your repo must contain:

| Path | What |
|---|---|
| `agentcore/` and your agents | The code |
| `README.md` | Updated so we can run your version |
| `CLAUDE.md` | Plus any other files you wrote to guide Claude Code |
| `DESIGN.md` | See the assignment page |
| `EVALS.md` | The pass rate of each of your 3 chaos runs, and why each failing case fails |
| `evals/reported/` | The 3 result files (from `evals/results/`) that `EVALS.md` reports |
| `evals/scenarios/` | The 10 given scenarios, unchanged, plus at least 3 of yours (`11_...yaml` and up) |
| `claude-sessions/` | One `/export` `.txt` per session, plus `transcripts.zip` |
