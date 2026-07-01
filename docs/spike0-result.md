# Spike 0 Result Log

Gate defined in `docs/plans/2026-06-30-phase-1.md` (steps 3-5). This file records evidence as it lands.

## Desk verification — 2026-07-01 (no AWS spend)

### Verdict so far: scriptability (Spike 0a "headless MCP") = **GO**, proven hands-on. Target-side checks remain.

**Handshake evidence (run on the x86 control-plane PC):**

```
docker pull armlimited/arm-mcp:latest
python scripts/spike0_mcp_handshake.py
```

Output (2026-07-01):

- `SERVER: {'name': 'arm-mcp', 'version': '3.4.2'} protocol=2024-11-05`
- `TOOL_COUNT: 7` — `knowledge_base_search`, `check_image`, `sysreport_instructions`, `migrate_ease_scan`, **`apx_recipe_run`**, `skopeo`, `mca`
- **`apx_recipe_run`** — "Run a sample workload on the given target using a Performix recipe, and interpret the results." Params: `cmd` (required), `remote_ip_addr` (required), `remote_usr` (required), `recipe` (optional; docs suggest `code_hotspots` as the default), `invocation_reason`. The tool runs `apx` *inside the container* and reaches the target over SSH (keys mounted at `/run/keys/`).
- Bonus analyst tools for later: `mca` (LLVM-MCA assembly perf analysis — pairs with hotspot disassembly), `knowledge_base_search` (Arm docs semantic search — citable evidence for the narrator).

**Supporting facts (sources fetched 2026-07-01):**

| Fact | Detail | Source |
|---|---|---|
| MCP server is public + client-agnostic | `github.com/arm/mcp`, Apache-2.0, Docker stdio (`armlimited/arm-mcp`); Claude Code / Codex CLI / Gemini CLI documented | github.com/arm/mcp |
| Performix hosts include Windows x64 | GUI + `apx` CLI install on Windows 10+/macOS/Debian-based, Arm64 or x64 — the control-plane PC can drive `apx` directly (CLI fallback needs no IDE either) | learn.arm.com/install-guides/performix |
| Supported targets | Amazon Linux 2023, Ubuntu 22.04, Ubuntu 24.04; `apx target add ubuntu@<ip>` registers a remote over SSH | learn.arm.com/install-guides/performix |
| KleidiAI toggle | `cmake -B build -DGGML_CPU_KLEIDIAI=ON`; runtime confirms via `load_tensors: CPU_KLEIDIAI model buffer size = ...`; `GGML_KLEIDIAI_SME` env var | llama.cpp docs/build.md |
| llama-bench | `-o json|jsonl|csv|md|sql`; `-r` default 5; most params accept multiple values (native sweeps) | llama.cpp tools/llama-bench/README.md |
| Quality guard tooling | `llama-perplexity --kl-divergence-base <f.kld> --kl-divergence` vs an FP16 base; base logits are 11-37 GiB on Wikitext-2, so pin a small eval text | llama.cpp tools/perplexity/README.md |
| Graviton5 | M9g/M9gd GA Jun 2026, only us-east-1/2, us-west-2, eu-central-1; C9g/R9g later in 2026 — `r8g` (Graviton4) stays primary | aws.amazon.com/blogs/aws (M9g GA post) |
| Challenge deadline | **14 Aug 2026, 4:00pm PDT** (= 15 Aug 00:00 UK); video optional, max 3 min; MIT/Apache-2.0 license must be detectable | arm-ai-optimization-challenge.devpost.com |

## Target-side checks — PENDING (needs a Graviton instance)

- [ ] Graviton4 `r8g` up (interactive AWS login; set auto-shutdown backstop).
- [ ] `apx_recipe_run` against the real target: does it return **structured/parsable** perf data (counters, hotspots), or prose only? Record a raw capture here.
- [ ] Enumerate available APX recipes beyond `code_hotspots`.
- [ ] **Counters discriminate between levers:** profile generic build vs `-mcpu=native`; confirm Performix clearly shows the difference (memory-bound vs compute-bound, cache/bandwidth deltas).
- [ ] Baseline llama.cpp build + `llama-bench -o json` capture.

## Verdict

- **Spike 0a (scriptability): GO** — desk-proven 2026-07-01.
- **Spike 0b (structured output + discrimination): OPEN** — decides the loop framing (diagnosis-driven vs try-measure-keep-best), not project viability.
