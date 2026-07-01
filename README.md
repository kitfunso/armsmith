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
_TODO (fill at M4): the "AI that optimizes AI for Arm" pitch + the headline speedup number._

## Functionality / Output
<!-- Devpost-required. What the project does and what the final output is. -->
_TODO: the optimized model + the HTML report (before/after metrics, Performix counters, decision trajectory, Pareto chart) + the JSONL trajectory log._

## Setup Instructions
<!-- Devpost-required. Step-by-step build/run/validate on an Arm64 environment. -->
```bash
# 1. Stand up a Graviton4 target (see provision/)
# 2. pip install armsmith
# 3. armsmith optimize --target <graviton-host> --model <model.gguf> --workload examples/bench.yaml
# 4. open report.html
```
_TODO: flesh out after Spike 0 confirms the architecture._

---

## Status
Pre-build. See [PLAN.md](./PLAN.md). Spike 0a (Performix MCP headless scriptability) **desk-verified GO** 2026-07-01 — `armlimited/arm-mcp` v3.4.2 drives Performix's `apx_recipe_run` from any MCP client ([evidence](./docs/spike0-result.md)). Remaining gate: target-side structured output + counter discrimination on a real Graviton.
