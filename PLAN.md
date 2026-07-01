# armsmith — Plan

**Arm Create: AI Optimization Challenge 2026** · Track: **Cloud AI** · Deadline: **14 Aug 2026, 4:00pm PDT** (Devpost-verified 2026-07-01; = 15 Aug 00:00 UK — treat 13 Aug as the internal cutoff)
License: **Apache-2.0** · Repo: public, open-source

> An autonomous agent that optimizes LLM inference on AWS Graviton by driving **Arm Performix**:
> it profiles the workload, reads the bottleneck from real Arm performance counters, applies an
> optimization from a defined action space, re-benchmarks, and loops until it converges — then emits
> a reproducible before/after report showing *what it changed and why*.

One line: **an AI agent that optimizes AI for Arm, using Arm's own brand-new tooling.**

> **Canonical docs:** `docs/PRD.md`, `docs/ARCHITECTURE.md`, `CLAUDE.md`, `docs/plans/2026-06-30-phase-1.md` are the source of truth. This file is the original overview; where it disagrees with those, they win. Post-review pivot (2026-06-30): the optimizer core is a **deterministic tuner**, the LLM is the **Performix analyst/narrator** on top (not the control loop); 5-lever action space; a quality guard blocks speed wins that degrade the model.

---

## 1. Why this wins (rubric mapping)

| Rubric axis | Pts | How armsmith scores |
|---|---|---|
| Technological Implementation | 40 | Real, measured speedups on Graviton via a rich evidence-driven action space (build flags, KleidiAI, quant format, threading, KV-cache). Uses Performix + KleidiAI — Arm's own stack. |
| "WOW" factor | 25 | Self-referential and dead-on Arm's flagship narrative: an *agent* that *autonomously* optimizes for Arm using *Performix's MCP server*. Shows its reasoning at each step. |
| Potential Impact | 20 | Reusable `pip`-installable tool + Graviton provisioner + a trajectory dataset + an HTML "learning-ready" report. Anyone can point it at their own workload. |
| Developer Experience | 15 | One command; clean README with the 3 required sections; one-script reproduce-on-fresh-instance. |

Why it fits **Keith specifically**: it is `dev-framework-rl` (action space + critics + trajectory logging) aimed at Arm perf; MCP is 2chain home turf; the rigorous benchmark methodology is the quant muscle that most hackathon entries lack. It needs **zero special hardware** (PC is control plane, Graviton is target).

The Settled lesson ("useful != wow"): here the wow is *structural* (autonomous agent + on-narrative + visible reasoning), not cosmetic.

---

## 2. Architecture

```
┌─────────────────────────────┐         SSH          ┌──────────────────────────────┐
│  Control plane (your PC)    │ ───────────────────► │  Target: AWS Graviton4 (r8g)  │
│  Windows 11 · Ryzen 9900X   │                       │  Neoverse V2 · SVE2/BF16/i8mm │
│                             │ ◄─── Performix MCP ──► │                              │
│  armsmith agent (Python)    │   counters / hotspots │  • llama.cpp server (GGUF)    │
│   • Claude (Anthropic API)  │                       │  • Arm Performix (profiler)   │
│     = the "brain"           │                       │  • benchmark workload         │
│   • action executor (SSH)   │                       │                              │
│   • trajectory log (JSONL)  │                       │                              │
│   • HTML report renderer    │                       │                              │
└─────────────────────────────┘                       └──────────────────────────────┘
        (optional bonus) final optimized GGUF → iPhone 17 Pro via App Store runner (SME2)
```

- **Control plane = your PC.** x86, cannot produce valid Arm numbers — it orchestrates only. Runs the agent loop, SSHes to the target, parses Performix output, renders the report.
- **Target = Graviton4 `r8g`** (Neoverse V2: SVE2, BF16, i8mm/MMLA). `c7g` (Graviton3) kept as a cheaper secondary target to show the agent generalizes across cores. (Graviton5 `m9g`/`m9gd` went GA Jun 2026 but only in us-east-1/2, us-west-2, eu-central-1 as of Jul 2026 — no London; `r8g` stays primary, `m9g` in Frankfurt is an optional "newest core" garnish.)
- **Optimizer = deterministic tuner** (grid/Bayesian over the registry); **Brain = LLM analyst** (default Claude, model-agnostic) that reads Performix evidence, reorders/prunes the tuner's search, and narrates each keep/revert. Remove the LLM and the tuner still finds the answer (the honesty test). Not an LLM control loop, not RL.

---

## 3. The optimization action space (the substance behind the 40 pts)

The agent does NOT just "turn on a flag." It has a real toolkit; each action is independently measured and kept/reverted on evidence:

1. **Baseline** — generic build, no `-mcpu=native`. The naive starting point everyone ships.
2. **`-mcpu=native`** — unlocks Neon / SVE / MMLA matmul kernels (AWS-documented path).
3. **KleidiAI microkernels** — DotProd / I8MM / SME2 repacking of Q4_0/Q8_0 weights (verify exact CMake toggle, e.g. `GGML_CPU_KLEIDIAI`).
4. **Quant format** — Q4_0 (Arm-repack-friendly) vs Q4_K_M vs Q8_0 on this core.
5. **Threading / affinity** — threads = physical cores, NUMA pinning, `--cpu-mask`, batch / ubatch sizing.
6. **Runtime knobs** — KV-cache type (f16 vs q8), flash-attention, context size, mmap, huge pages.
7. **(stretch) targeted hot-loop fix** — if Performix flags one specific hotspot.

Each step records: decode tokens/sec, prefill TTFT, **quality (perplexity/KL)**, peak memory, model size, **and** Performix counters (cache-miss rate, memory bandwidth, IPC). The **deterministic tuner** sweeps the space; the **LLM analyst** uses the counters to prioritize *which* lever to try first (memory-bound → quant/KV-cache; compute-bound → kernels/threads) and to explain each keep/revert. That Performix-evidenced narration over a reproducible sweep is the part judges can't get from a plain benchmark script. (Items 1 and 7 are the baseline/reset and a stretch; the registry proper is the 5 levers, items 2-6.)

---

## 4. Benchmark methodology (the differentiator)

Most entries will report one cherry-picked number. armsmith ships honest measurement:
- Warmup runs discarded; N>=5 repeats; report median + 95% CI.
- Fixed prompt set + fixed token budget; decode and prefill reported separately.
- Same instance, same model, same seed across before/after.
- Every number regenerable by one command on a fresh instance.
- Report distinguishes "config win" from measurement noise (CI overlap test).

This is the quant-grade rigor that makes the speedup *credible*, which is itself part of the wow for ML-engineer judges.

---

## 5. Deliverables / artifacts

- **`armsmith` CLI** (pip): `armsmith optimize --target <graviton-host> --model <gguf> --workload bench.yaml`
- **HTML report** (self-contained): before/after tokens-sec + TTFT + memory + size, Performix counters, the **decision trajectory** ("why each move"), and a size×speed×quality Pareto chart. This is the wow artifact and the "learning-ready content" Impact bullet.
- **Trajectory log** (JSONL): every decision + evidence + result. Reusable template/dataset.
- **Graviton provisioner**: one Terraform/CloudFormation or shell script so anyone reproduces from scratch.
- **README** with the 3 required Devpost sections (Overview / Functionality / Setup) + one-command quickstart.
- **Optional**: `<3 min` demo video; iPhone 17 Pro cross-target garnish.

---

## 6. WOW set-piece (for the video + Devpost)

A **visual reasoning trajectory** (not log dumps): baseline -> Performix diagnosis -> action -> measured delta -> keep/revert -> final recipe, rendered as the demo's centerpiece. "I built a Graviton LLM-optimization lab. The deterministic tuner closed >=90% of the naive-to-expert gap, and the LLM analyst read Arm's own Performix counters to prioritize the search and explain every win - reproducibly, replayable from a saved recipe." Optional garnish: replay the winning GGUF on an **iPhone 17 Pro** (newest Apple Arm core, SME2) via an App Store runner to show "same optimization, cloud to phone" (no Mac needed).

---

## 7. Spike 0 — GO / NO-GO gate (front-loaded de-risk, days 1–3)

**Desk-verified 2026-07-01 (details in `docs/spike0-result.md`):** the scriptability half of the gate is largely resolved. The MCP server is public — [github.com/arm/mcp](https://github.com/arm/mcp) (Apache-2.0), ships as a Docker **stdio** server (`armlimited/arm-mcp`), works with any MCP client (Claude Code, Codex CLI, Gemini CLI are documented), and exposes an Arm Performix tool that "runs APX recipe workflows against a target device over SSH". Separately, Performix host installers cover **Windows x64**, so the control-plane PC can also drive the `apx` CLI directly (the fallback path needs no IDE either). Targets supported: Amazon Linux 2023, Ubuntu 22.04/24.04.

Spike 0 checklist (remaining — needs a real target):
- [ ] Stand up a Graviton4 `r8g` instance (your AWS account; interactive login needed).
- [ ] Install Arm Performix on the target; confirm it runs and profiles a workload.
- [ ] Call the Performix MCP tool programmatically against the target; confirm the returned perf data is **structured/parsable** (not prose-only).
- [ ] Confirm the counters **discriminate between levers** (generic vs `-mcpu=native` build).
- [ ] Build baseline llama.cpp + capture a baseline benchmark.

**GO** → build the full agent loop.
**NO-GO** (MCP gated/IDE-only) → fallback: agent parses **Performix CLI output** directly (still on-narrative, slightly less slick), or worst case `perf`/Arm Streamline counters. Decide at the gate, do not build on the unverified path.

---

## 8. Milestones (deadline 14 Aug 4pm PDT; internal cutoff 13 Aug)

| # | Days | Goal | Verify |
|---|---|---|---|
| Spike 0 | 1–3 | Graviton up; Performix + MCP confirmed scriptable; baseline captured | GO/NO-GO on the MCP assumption |
| M1 | 4–12 | Rigorous benchmark harness + action space implemented as callable steps | Each lever independently moves tokens/sec on Graviton |
| M2 | 13–25 | Agent loop: Claude brain + Performix evidence + select/apply/revert + trajectory log | Agent autonomously finds a real, CI-significant speedup |
| M3 | 26–34 | HTML report + CLI polish + provisioner + README | One-command run reproduces on a fresh instance |
| M4 | 35–40 | iPhone 17 Pro bonus demo (optional) + `<3min` video + Devpost writeup | Video < 3 min; writeup has all 3 required sections |
| Buffer | 41–46 | Outside-voice review, polish, slop pass, **submit early** | Submitted by 13 Aug (hard deadline 14 Aug 4pm PDT) |

---

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Performix MCP gated / not scriptable | Spike 0 GO/NO-GO; CLI-parse fallback identified |
| "Optimizations" look thin | 6-lever action space; report each lever's measured contribution |
| Speedups marginal | Pick a workload where Arm features bite (Q4_0/Q8_0 matmul); Graviton4 + KleidiAI has documented >2x decode headroom vs naive |
| Scope creep (iPhone, fancy RL) | iPhone explicitly optional; agent is hill-climb, not trained RL |
| Local dev can't validate Arm | All real numbers on Graviton; PC orchestrates only |

---

## 10. Cost

Only paid item: Graviton EC2 hours. `r8g.4xlarge` on spot for short bursts over 46 days = tens of dollars. Flag before spinning up; tear down between sessions.

---

## 11. To-verify (don't fabricate — confirm in Spike 0 / early build)

- ~~Exact KleidiAI CMake toggle~~ **RESOLVED 2026-07-01:** `-DGGML_CPU_KLEIDIAI=ON` (llama.cpp `docs/build.md`); runtime confirms with `load_tensors: CPU_KLEIDIAI model buffer size = ...`; `GGML_KLEIDIAI_SME` env var controls SME kernel use.
- ~~Performix MCP headless-scriptable~~ **RESOLVED 2026-07-01:** yes — `armlimited/arm-mcp` Docker stdio server, MCP-client-agnostic; and the `apx` CLI installs on Windows x64 hosts. Remaining (Spike 0): structured output + counter discrimination on a real target.
- ~~Graviton5 availability~~ **RESOLVED 2026-07-01:** M9g/M9gd GA (Jun 2026) in us-east-1/2, us-west-2, eu-central-1 only; C9g/R9g later in 2026. Not in London — `r8g` stays primary.
- iPhone 17 Pro chip ISA (SME2?) + a no-Mac GGUF-runner deploy path — only if we add the on-device garnish.
- llama-bench facts (verified 2026-07-01): `-o json|jsonl|csv|md|sql`; `-r` repetitions default 5; most params accept multiple values in one invocation (native sweep support). `llama-perplexity --kl-divergence-base <f.kld> --kl-divergence` gives KL vs an FP16 base, but the base logits file is 11–37 GiB on Wikitext-2 — pin a **small eval text** for the quality guard.

---

## 12. Repo layout (firms up after Spike 0)

```
armsmith/
  README.md            # Devpost-required sections + quickstart
  LICENSE              # Apache-2.0 (visible, detectable)
  PLAN.md              # this file
  pyproject.toml
  src/armsmith/
    agent.py           # the loop: brain + select/apply/revert
    actions.py         # the optimization action space
    performix.py       # Performix MCP / CLI client
    bench.py           # rigorous benchmark harness
    report.py          # self-contained HTML report
    provision/         # Terraform/shell to stand up Graviton
  examples/bench.yaml
  trajectories/        # JSONL decision logs
```

---

## Sources (verified 2026-06-30, extended 2026-07-01)
- Arm Performix: https://newsroom.arm.com/news/announcing-arm-performix · https://developer.arm.com/servers-and-cloud-computing/arm-performix · install guide: https://learn.arm.com/install-guides/performix/
- Arm MCP server (Performix over SSH, Docker stdio): https://github.com/arm/mcp · https://hub.docker.com/mcp/server/arm-mcp/overview
- KleidiAI + llama.cpp: https://pytorch.org/blog/unleashing-ai-mobile/ · https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction · build flag: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Graviton llama.cpp: https://github.com/aws/aws-graviton-getting-started/blob/main/machinelearning/llama.cpp.md
- Graviton5 GA (M9g regions): https://aws.amazon.com/blogs/aws/now-available-amazon-ec2-m9g-and-m9gd-instances-powered-by-new-aws-graviton5-processors/
- Challenge page (deadline, rules, judging): https://arm-ai-optimization-challenge.devpost.com/
