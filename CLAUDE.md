# CLAUDE.md - armsmith

## Project Overview
armsmith is a reproducible Graviton LLM-optimization lab: a deterministic autotuner explores inference configs over a fixed 5-lever action space, an LLM analyst reads Arm Performix counters to prioritize the search and narrate each keep/revert, and the winning config replays from a saved recipe. Built for the Arm AI Optimization Challenge 2026 (Cloud AI track). See `docs/PRD.md` and `docs/ARCHITECTURE.md`.

## Architecture
Control plane (x86 PC) decides; Graviton target measures; pluggable LLM brain chooses from a registry. Full detail in `docs/ARCHITECTURE.md`. Read it before touching `agent.py`, `actions.py`, or `executor.py`.

## Non-Negotiable Rules
1. **NEVER report a benchmark number measured anywhere but a real Arm target.** The dev PC is x86; x86 numbers are meaningless here and reporting them is fabrication.
2. **The deterministic tuner is the optimizer; the LLM only analyzes and never emits free-form shell.** The LLM returns `{priority:[action_id...], rationale}` (and at most a suggested `{action_id, params}`), all validated against the fixed registry; the deterministic tuner decides what runs and the Executor only runs registry `ActionSpec.apply` templates. A validator rejects off-registry ids before execution and a unit test asserts it. A remote LLM running arbitrary code on a billed instance is unacceptable, and an LLM control loop over 6 levers would be theater (it must earn its place as analyst/narrator).
3. **Every reported speedup must be CI-significant** (non-overlapping 95% CIs, N>=5 repeats). One-shot numbers are noise; credibility is the entire value proposition.
4. **Never claim a result you cannot reproduce** via `armsmith repro` replaying the saved recipe (no LLM in the loop) on a fresh instance. If it is not replayable, it is not a result.
5. **Do NOT build the agent loop until Spike 0 passes.** Performix MCP scriptability is unverified; building on it before the GO wastes the 46 days if it fails.
6. **The naive baseline must be honest, and a speed win must not cost quality.** The baseline is a reasonable default build (not crippled); the expert config is pre-registered before discovery. Every config is measured for quality (perplexity/KL vs baseline) and any config that regresses quality past the threshold is rejected, never reported as a win. A fake-slow baseline or a quietly-degraded model makes the submission dishonest.
7. **License stays Apache-2.0 and visible.** The challenge requires a detectable OSS license; without it the entry is ineligible.
8. **Tear down Graviton instances after each session.** Cost control; the only paid item in the project.

## Coding Conventions
- Python 3.11+, full type hints, `ruff` + `black`, `typer` for CLI.
- `@dataclass(frozen=True)` for the record types in `models.py`; `Protocol` for `Brain`/`Profiler`/`Engine`.
- `logging`, never `print`, in library code. Errors handled explicitly at the SSH/Performix boundaries.
- Secrets via env only (`ANTHROPIC_API_KEY`, AWS creds, SSH key path). Nothing secret committed.
- Tests: `pytest`, real parsing fixtures (no mocked Performix output where a real capture exists).

## Critical Files
- `src/armsmith/actions.py` — the optimization action space. The substance of the Tech score.
- `src/armsmith/profiler.py` — Performix client (MCP + CLI fallback).
- `src/armsmith/bench.py` — the statistics. Get the CI math right or every claim is suspect.
- `src/armsmith/agent.py` — the loop and the keep/revert decision.

## Safety Rules
- Brain output is constrained to an action id + params, validated against the registry before the Executor runs anything (rule 2).
- SSH commands are templated from `ActionSpec`, never interpolated from LLM free text.
- No API keys, AWS creds, or SSH keys in the repo or in committed logs/trajectories.

## Common Mistakes to Avoid
- Reporting x86 numbers, or a crippled baseline (rules 1, 6).
- Claiming "done" before a fresh-instance repro (rule 4).
- Forgetting `-mcpu=native` is the *first* real lever, not the baseline — define what "naive" means and pin it.
- CRLF churn: `.gitattributes` forces LF. Use targeted edits, not bulk rewrites, on existing files.
- Leaving a Graviton instance running overnight (rule 8).
