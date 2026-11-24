# CLAUDE.md

Take-home: `agentcore` (generic agent runtime) + router/booking/billing/info agents for City Care Hospital.
Read `context.md` first. Hospital rules are in context.md section 5. Facts for the info agent come from `hospital_info.md` only.

## Hard rules
- `agentcore/` never imports hospital code (no `agent_service`, `mock_hospital`, doctors, appointments, bills). It stays generic. A test checks its imports.
- Do not modify `mock_hospital/` or `evals/scenarios/01`-`10`. Do not change how existing checks in `run_evals.py` behave. New scenarios are `11_*.yaml` and up.
- Agents never call `/_admin/*`.
- Tests must pass without an API key (`uv run pytest`). Use `tests/fake_claude.py`. Tests never call the real API.
- Time comes from hospital `GET /now`, never the machine clock. No `datetime.now()` in agent code.
- Patient identity = `phone` from the `/chat` request. Tools take it from run context, never from model arguments or message text.
- Safety rules are enforced in code (hooks/tools). Prompts only support them.
- No agent frameworks, no SDK tool runner. Only the `anthropic` SDK plus our own loop.
- Never commit `.env`, API keys or secrets. Check `git diff --staged` before every commit.

## Design decisions
- Hooks: `before_tool` returns allow / deny(reason) / modify(input) / confirm(prompt). `confirm` pauses the run and returns the prompt to the patient; the run resumes on the next patient turn.
- Confirm read-back text is built by code from real hospital data, never by the model.
- At most one write per turn, so a retry or a confused model cannot double book.
- Ownership is checked on reads and writes. A tool result never contains another phone's data.
- 4-hour rule uses the existing appointment's start time (not the new slot). Denied requests still tell the patient staff will help, and set `handoff`.
- One lock per session so two messages in the same chat never run at once.
- Emergency check = regex (English, Hindi, Hinglish) plus a Haiku check. Either one triggers it.

## Workflow
- Small commits, one logical change each. The message says what changed and why.
- Write the test first when practical. Run `uv run pytest` before each commit.
- Check current Anthropic docs for model params (forced `tool_choice`, thinking defaults) before relying on memory.
- Budget is $30 total. Haiku for router and cheap paths. Real-API evals sparingly. Check `/cost` regularly.
- Export each Claude Code session to `claude-sessions/NN-name.txt` before `/clear`.
