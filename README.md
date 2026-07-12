# armsmith

**An autonomous agent that optimizes LLM inference on AWS Graviton — using Arm's own Performix tooling.**

armsmith profiles a workload on an Arm (Graviton) target with **Arm Performix**, reads the bottleneck
from real Arm performance counters, applies an optimization from a defined action space (build flags,
KleidiAI microkernels, quant format, threading, KV-cache), re-benchmarks, and loops until it converges —
then emits a reproducible before/after report showing *what it changed and why*.

> Arm Create: AI Optimization Challenge 2026 · **Cloud AI** track · License: Apache-2.0

---

## Project Overview
<!-- Devpost-required. A brief description of the project and its purpose; what makes it interesting and why it should win. -->
armsmith is an AI that optimizes AI for Arm. Point it at a Graviton instance and a GGUF
model; a **deterministic autotuner** sweeps a 5-lever action space (build flags, KleidiAI
microkernels, quant format, threading/affinity, KV-cache type) while a **Claude analyst**
reads Arm Performix counter evidence, prioritizes the search, and narrates every
keep/revert. On Graviton4 (r8g, Qwen2.5-7B Q4_0) it took decode from **21.85 to
45.46 tok/s (2.1x)** and prefill TTFT from 8.7s to 1.6s — **beating the expert config
pre-registered before the run** (41.35 tok/s, 121% of the naive→expert gap closed),
with a quality guard (KL vs baseline 0.002) proving the model was not degraded.
The LLM never emits shell and never decides: it proposes, a registry-validated gate
disposes, and removing it entirely still finds the same winner.

## Functionality / Output
<!-- Devpost-required. What the project does and what the final output is. -->
Each run produces, under `trajectories/<run-id>/`:
- **`recipe.json`** — the winning config plus baseline/winner results with medians and
  95% CIs; `armsmith repro` replays it on a fresh instance with **no LLM in the loop**.
- **`report.html`** — self-contained visual trajectory: before/after metrics, per-lever
  deltas, quality numbers, and the analyst's rationale for every decision.
- **`trajectory.jsonl`** — every candidate: action, params, screen/confirm stats,
  CI-significance, kept/reverted, diagnosis and rationale. A reusable dataset.
- **`manifest.json` / `baseline.json` / `expert.json`** — the pinned target, model
  (with SHA-256), honest baseline, and pre-registered expert config.

All numbers are measured on real Arm silicon: warmup discarded, N>=5 repeats,
median + 95% CI, decode and prefill reported separately. One-shot numbers are noise.

## Setup Instructions
<!-- Devpost-required. Step-by-step build/run/validate on an Arm64 environment. -->
```bash
# 0. Control plane: any machine with Python 3.11+ and SSH. Targets: Ubuntu 22.04/24.04 arm64.
git clone https://github.com/kitfunso/armsmith && cd armsmith
pip install -e .

# 1. Stand up a Graviton target (prints the aws cli checklist; also runnable as
#    scripts/provision_graviton.sh). r8g = Graviton4.
armsmith provision --instance r8g.4xlarge

# 2. Capture the honest baseline + pre-registered expert config (mints the run id).
armsmith baseline --target <graviton-ip> --model examples/model-qwen25-7b.json \
  --workload examples/bench-full.yaml --ssh-key ~/.ssh/armsmith.pem

# 3. Discovery: deterministic tuner + Claude analyst (needs ANTHROPIC_API_KEY;
#    --brain null runs the same search with no LLM).
armsmith optimize --target <graviton-ip> --model examples/model-qwen25-7b.json \
  --workload examples/bench-full.yaml --brain claude --ssh-key ~/.ssh/armsmith.pem

# 4. Render the visual trajectory report.
armsmith report <run-id>   # prints the path to report.html

# 5. Validate: replay the recipe on a FRESH instance, no LLM, +-10% tolerance.
armsmith repro <run-id> --target <fresh-ip> --ssh-key ~/.ssh/armsmith.pem
```

---

## Status
M1–M3 complete: 194 tests green, full narrated discovery runs on real r8g hardware,
recipe + HTML report + trajectory artifacts committed under `trajectories/`.
Spike 0a evidence: [docs/spike0-result.md](./docs/spike0-result.md). History: [PLAN.md](./PLAN.md).
