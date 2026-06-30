# armsmith - Architecture

## System Overview
Three parts. The control plane runs the loop, the target measures, and the brain (LLM) analyzes Performix evidence while a **deterministic tuner** drives the actual search. The LLM prioritizes and explains; it does not control the optimizer.

```
┌──────────────────────────────┐       SSH (commands only)      ┌───────────────────────────────┐
│ CONTROL PLANE (x86 Windows PC)│ ─────────────────────────────►│ TARGET: AWS Graviton4 r8g      │
│                              │                                 │ (Neoverse V2: SVE2/BF16/i8mm)  │
│  armsmith (Python CLI)       │                                 │                               │
│   ├─ agent loop (orchestrator)│◄──── Performix counters ───────│  • llama.cpp server (GGUF)     │
│   ├─ Brain (LLM, pluggable)   │      (MCP server / CLI)         │  • Arm Performix (profiler)    │
│   ├─ Executor (SSH actions)   │                                 │  • benchmark workload          │
│   ├─ Profiler client          │                                 │                               │
│   ├─ Bench harness (CI stats)  │                                 └───────────────────────────────┘
│   └─ Report renderer (HTML)   │                                 (secondary target: c7g Graviton3)
└──────────────────────────────┘     (optional) final GGUF → iPhone 17 Pro via App Store runner
```

**Optimizer spine (deterministic) + LLM analyst.** A deterministic search (grid/Bayesian over the registry) is the source of truth for which configs get tried - this is what makes runs reproducible and defensible. The LLM analyst sits on top: it reads the Performix snapshot, **reorders/prunes the search** (try the lever the counters implicate first), and writes the human explanation + recipe + report. If the LLM is removed, the deterministic tuner still finds the answer (just slower and without narration). That is the honesty test.

**Key invariant:** the LLM NEVER emits free-form shell to the target. Its output is `{priority: [action_id...], rationale}` (and at most a *suggested* `{action_id, params}`), all validated against the fixed registry. The Executor only ever runs `ActionSpec.apply` command templates chosen by the deterministic tuner. LLM free text never reaches a shell; an off-registry id is a hard reject with a logged event (a unit test asserts it). This enforces the PRD's "fixed action space" and keeps a remote LLM from running arbitrary code on a billed instance.

## Tech Stack
| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Agent runtime | Python 3.11+ | SSH/subprocess/stats ecosystem; Keith's stack |
| Search/tuner (core) | deterministic grid + optional Optuna (Bayesian) | reproducible optimizer spine; the defensible part |
| Agent brain (analyst) | Pluggable LLM client; default Claude (Anthropic API) | reads Performix, prioritizes search, narrates; model-agnostic = reusability (20 pts) + run with gpt-5.5 if wanted |
| CLI | `typer` | clean DX, auto `--help` |
| Inference engine (target) | llama.cpp / ggml | canonical Arm CPU LLM engine; KleidiAI integrated |
| Profiler (target) | Arm Performix (MCP server; CLI fallback) | the Arm-native tool the challenge promotes |
| Remote control | SSH via `paramiko`/`fabric` | drive Graviton from the PC |
| Provisioning | Terraform (boto3/shell fallback) | reproducible target, one command |
| Benchmark | `llama-bench` + custom timing + stats | rigorous median + 95% CI |
| Report | self-contained HTML (Jinja2 + inlined CSS/JS) | HTML-first, no build step, portable artifact |
| Config | YAML | simple workload/target specs |
| Packaging | `pyproject.toml`, pip-installable | DX (15 pts) |

## Repository Structure
```
armsmith/
  README.md                  # Devpost sections + quickstart
  LICENSE                    # Apache-2.0 (visible, detectable)
  CLAUDE.md                  # non-negotiable rules for AI sessions
  pyproject.toml             # package + CLI entrypoint
  .gitattributes             # force LF (avoid CRLF churn)
  docs/                      # PRD, ARCHITECTURE, plans, spike results
    plans/                   # one dated plan file per phase
  src/armsmith/
    cli.py                   # typer commands: provision/baseline/optimize/report/repro
    agent.py                 # the loop: diagnose -> select -> apply -> measure -> keep/revert
    brain.py                 # LLM client (Protocol); default Claude; selects ActionSpec
    actions.py               # the ActionSpec registry (the optimization levers)
    executor.py              # applies actions on the target over SSH (audited commands only)
    profiler.py              # Performix client (MCP + CLI fallback) -> PerformixSnapshot
    bench.py                 # benchmark harness -> BenchmarkResult (median + CI)
    engine.py                # Engine adapter Protocol; llama.cpp impl first
    models.py                # frozen dataclasses: RunManifest, BenchmarkResult, etc.
    report.py                # render self-contained HTML from a run
    provision/               # Terraform + SSH helpers to stand up Graviton
  examples/
    bench.yaml               # sample workload spec
  trajectories/              # JSONL decision logs (one per run)
  tests/                     # pytest; action apply/revert, stats, parsing
```

**MVP module note:** the tree above is the *target* structure. Start the MVP with ~5 modules - `cli`, `agent`, `target` (ssh + build + run, folding in executor/engine), `profiler`, `report`, with the record types inline - and only split out `executor`/`engine`/`models` and extract the Protocols when a second engine or implementation actually appears. Explicit over speculative abstraction.

## Data Model (on-disk artifacts, not a DB)
Frozen dataclasses, serialized to JSON/JSONL under `trajectories/<run_id>/`.

- **RunManifest** — `run_id` (PK), `target` (instance type, core, region), `model` (name, quant, sha), `workload_ref`, `baseline_ref`, `created_at`, `armsmith_version`.
- **BenchmarkResult** — `config_id`, `decode_tok_s` {median, ci_low, ci_high}, `prefill_ttft_ms` {median, ci_low, ci_high}, `quality` {perplexity, kl_vs_baseline}, `peak_mem_mb`, `model_size_mb`, `n_repeats` (>=5), `stage` (screen | confirm), `raw_samples[]`. *Constraint: n_repeats>=5 + CI required on `confirm`; a config with quality regression past threshold is invalid.*
- **PerformixSnapshot** — `config_id`, `cache_miss_rate`, `mem_bandwidth_gbps`, `ipc`, `hotspots[]` {symbol, pct}. Source tag: `mcp` | `cli`.
- **ActionSpec** — `id` (PK), `name`, `params`, `apply` (registered cmd template), `revert`, `preconditions` (e.g. requires i8mm). *Constraint: registered only; no free-form shell.*
- **TrajectoryStep** — `step_idx`, `diagnosis` (bottleneck class + evidence), `action_id`, `rationale`, `before_config_id`, `after_config_id`, `kept` (bool), `delta` {metric, pct, ci_significant}.

## API Design (CLI + internal interfaces)
No web API. The surface is the CLI + three Python `Protocol`s for extensibility:

CLI commands:
- `armsmith provision [--instance r8g.4xlarge] [--destroy]` — stand up / tear down a target.
- `armsmith baseline --target HOST --model M --workload W` — build stock engine, capture honest naive baseline.
- `armsmith optimize --target HOST --model M --workload W [--budget N] [--brain claude]` — run the loop.
- `armsmith report RUN_ID` — render the self-contained HTML.
- `armsmith repro RUN_ID` — reproduce a run on a fresh instance.

Interfaces (Protocols): `Brain.choose(diagnosis, history) -> {action_id, params}` (validated against the registry); `Profiler.snapshot(target, config) -> PerformixSnapshot`; `Engine.build(target, flags) / serve / bench`.

Auth model: no users. Secrets via env only — `ANTHROPIC_API_KEY` (brain), AWS creds (provisioner), SSH key path (executor). Nothing secret committed.

## Service Boundaries
- **Control plane vs target** — the SSH line. All reasoning on the PC; the target only builds, serves, profiles, benchmarks.
- **Brain / Executor / Profiler / Bench** — Brain *chooses*, Executor *applies* (audited cmds), Profiler *measures counters*, Bench *scores with stats*. No component does two of these.
- **Engine adapter** — llama.cpp first, behind `Engine` so vLLM-arm / ONNX-RT can be added without touching the loop.

## Data Flow (primary use case)
1. `provision` (or connect) target → 2. pin the **pre-registered expert config** + build/run the **honest baseline**, capture `BenchmarkResult` (incl. quality) → 3. `profiler.snapshot` (Performix) → 4. LLM analyst **diagnoses** the bottleneck and **reorders the deterministic tuner's candidate queue** (counters implicate which lever first) → 5. the **deterministic tuner** pops the next candidate config → 6. Executor **applies** it via a registry `ActionSpec.apply` template (rebuild/reconfig) → 7. **two-stage bench**: cheap screen first; only promising candidates get the rigorous N>=5 + 95% CI confirm → 8. **keep** if CI-significant speed gain AND no quality regression past threshold, else **revert** → 9. LLM writes a `TrajectoryStep` rationale → 10. loop until the tuner's space is exhausted or budget hit → 11. save the winning **recipe**; `report` renders the visual trajectory; the JSONL is the audit trail.

## Discovery vs Replay (how reproducibility survives a nondeterministic agent)
`optimize` is the **discovery** run: the LLM brain explores, so it is nondeterministic by nature. Its output is a saved **recipe** - the winning config (build flags, quant format, thread/affinity, KV-cache) plus the model + baseline refs - written to the run manifest. `repro` is **replay**: it applies the saved recipe deterministically with NO brain in the loop, so a stranger reproduces the headline number fast and for free. The reproducibility success metric applies to replay, never to re-running the search. This resolves the tension between "autonomous LLM agent" and "reproducible in <30 min."

```
optimize (discovery)            repro (replay)
  brain explores  ──► recipe ──►  apply recipe, benchmark
  (nondeterministic)   (saved)     (deterministic, no LLM)
```

## The "honest baseline" rule
The naive baseline must be *what a reasonable developer ships by default* (stock build, default threads), NOT an artificially crippled build. A fake-slow baseline makes the win fake. The baseline AND a **pre-registered expert config** (hand-tuned, pinned before discovery so the >=90%-gap target cannot be reverse-engineered) are documented and version-pinned in the run manifest.

## Measurement integrity (or the report is theater)
Decode is memory-bandwidth-bound and EC2 is noisy, so within-run CIs are necessary but not sufficient. Required: a non-burstable, socket-filling instance (`r8g`, never a `t`-class burstable); fixed thread/core mapping; warm page cache policy; fixed prompt/batch/context sizes; baseline/candidate **A/B/A/B interleaving** so systematic drift cancels; and the headline number re-confirmed on a **fresh instance** (not the tuning instance). Instance type, kernel, and CPU governor are captured in the manifest.
