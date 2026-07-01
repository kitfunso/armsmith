# armsmith - Product Requirements Document

## One-Line Description
A reproducible Graviton LLM-optimization lab: a deterministic autotuner explores inference configs, an LLM analyst reads Arm Performix counters to prioritize the search and narrate why each change won, and the result replays from a saved recipe on a fresh instance.

## Problem Statement
Optimizing LLM inference for Arm is expert work: you must know which build flags, quant formats, microkernels (KleidiAI / i8mm / SME2), and runtime knobs matter for a *specific* core, then profile and iterate by hand. Most developers ship a naive build and leave 2x+ on the table; the few who optimize do it slowly and rarely document *why* a change helped. Arm just shipped Performix to expose the performance data, but reading the counters and deciding what to change next is still entirely manual.

## Target Users
- **Arm-cloud LLM developers** (mid-level): deploying models on Graviton/Cobalt/Axion who want speed and cost wins without becoming Arm-perf specialists.
- **Performance engineers** (advanced): want a reproducible, evidence-logged optimization baseline they can audit and extend.
- **(Evaluators)** Arm developer-evangelist judges who reward genuine use of the Arm stack (Performix, KleidiAI), agentic workflows, and reusable artifacts.

## Core Features (MVP)
1. **Performix-driven profiling** - connect to an Arm target, run the workload, capture performance counters via Performix (MCP server, CLI fallback). *(solves: reading perf data is manual)*
2. **Deterministic tuner + LLM analyst loop** - a deterministic search (grid/Bayesian) over the action space drives config exploration (reproducible, defensible); the LLM analyst reads the Performix bottleneck (memory-bound vs compute-bound), **prioritizes/prunes the search order**, and explains each keep/revert. The optimizer core is deterministic; the LLM earns its place as the diagnostician and narrator, not the control loop. *(solves: deciding what to try first + explaining why)*
3. **Optimization action space (5 levers)** - `-mcpu=native`, KleidiAI microkernels, quant format (Q4_0/Q8_0/Q4_K_M), threading/affinity, KV-cache type. (A plain "generic build" is the baseline/reset state, not an action.) *(solves: the levers most devs don't know exist)*
3b. **Quality guard** - every config is measured for quality (perplexity / KL vs baseline), not just speed. A config that degrades quality past a set threshold is **rejected, not counted as a win** - so the agent can never "win" by quantizing the model into mush. *(solves: speed-only metrics reward silent quality loss)*
4. **Rigorous benchmark harness** - warmup discarded, N>=5 repeats, median + 95% CI, decode vs prefill reported separately. *(solves: untrustworthy one-shot numbers)*
5. **Reproducible report + trajectory** - self-contained HTML (before/after metrics, Performix counters, the decision log, size x speed Pareto) + a JSONL trajectory of every decision and its evidence. *(solves: optimizations go undocumented)*
6. **One-command reproduce** - a provisioner + CLI so a fresh Arm instance reproduces the result end to end. *(solves: not reusable by others)*

## What This Product IS NOT
1. **NOT a training or fine-tuning tool.** It optimizes *inference*, not the model's learned quality.
2. **NOT a GPU optimizer.** Arm CPU inference only. No CUDA/Metal paths.
3. **NOT a new inference engine.** It orchestrates and tunes *existing* engines (llama.cpp first); it does not reimplement kernels.
4. **NOT a hosted service or SaaS.** A local CLI run against your own Arm target. No backend, no accounts, no pricing (free, Apache-2.0).
5. **NOT a general autonomous coding agent.** Its action space is a fixed, audited set of performance optimizations, not arbitrary code edits.
6. **NOT an iOS / on-device app (in MVP).** The iPhone 17 Pro cross-target run is an optional post-MVP demo only.
7. **NOT a model zoo.** Bring your own GGUF; armsmith does not host or distribute models.
8. **NOT an LLM-driven control loop or an RL-trained policy.** The optimizer core is a deterministic search; the LLM is the Performix analyst + narrator on top. No trained controller (keeps scope sane and the result reproducible in 46 days).

## Success Metrics
- **Headline (honest gap, not a fixed multiple):** armsmith autonomously closes **>=90% of the naive->expert-tuned speedup gap** on Graviton4 and reports the absolute decode tokens/sec and prefill TTFT it reached. The win is recovering the gap autonomously, CI-significant (non-overlapping 95% CIs), with the reasoning shown - whatever the absolute multiple turns out to be. (A pre-committed "2x" would incentivize baseline-gaming, which the judges punish.)
- **Honest baseline (defined, not gameable):** the baseline is a *real-world default* a developer actually ships - the stock `llama-cpp-python` wheel, or a vanilla cmake build WITHOUT `-mcpu=native` - pinned and documented in the run manifest. Never an artificially crippled build.
- **Reproducible (replay, not re-search):** the winning config is saved as a deterministic recipe; `armsmith repro` replays it on a fresh instance with **no LLM in the loop**, within +-10% of reported numbers, in **<30 min**.
- **Credible:** the auto-found config **matches or beats a documented expert hand-tuned config** on the same model/instance. This expert config is **pre-registered and pinned BEFORE the discovery run** (so >=90%-of-gap cannot be gamed) and defines the "100%".
- **No quality regression:** the winning config's perplexity stays within a set threshold of the baseline; speed gains that cost quality past the threshold do not count.
- **Generalizes (stretch, not core):** ideally runs on a 2nd Arm core (Graviton3 `c7g`) and adapts, but the primary submission proves one core (Graviton4 `r8g`) well rather than diluting across two.
- **Reusable:** a second person clones and runs (`pipx install` + provision) on their own Graviton in **<30 min** following the README.
- **Submission outcome (target):** win a **Cloud AI category**; stretch: Overall. Complete repo (Apache-2.0 visible), README with all 3 required sections, `<3 min` video.

## Constraints
- **Timeline:** deadline **14 Aug 2026, 4:00pm PDT** (Devpost-verified 2026-07-01; internal cutoff 13 Aug). Solo (Keith + Claude).
- **Hardware:** control plane is x86 Windows and cannot produce valid Arm numbers; all Arm measurement happens on Graviton (and optionally iPhone 17 Pro).
- **Budget:** Graviton EC2 spot hours only (~tens of dollars). No other paid dependencies.
- **Dependency risk:** Arm Performix is ~3 months old. MCP headless scriptability was desk-verified 2026-07-01 (public `armlimited/arm-mcp` Docker stdio server, client-agnostic; `apx` CLI also installs on Windows x64 hosts). **Spike 0 gate remains** for the target-side half: structured output + counters that discriminate between levers.
- **License:** Apache-2.0, detectable and visible at repo top.
- **Agent brain:** model-agnostic (bring-your-own key/endpoint), default Claude via the Anthropic API.
