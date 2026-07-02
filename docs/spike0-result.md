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

## Target-side checks — 2026-07-02

Setup (recorded per phase-1 step 3):
- IAM: created `armsmith-provisioner` user + `AmazonEC2FullAccess` + access key via CloudShell (h0-app key was Bedrock-only). Keys in `~/.aws` (`[default]`), h0 keys moved to `[h0]` profile.
- **Spot quota = 0 in the account (`MaxSpotInstanceCountExceeded`); launched on-demand instead.** Request a spot quota increase for cheaper M1/M2 tuning runs.
- Target: `i-07ae005fc2dd04d83`, **r8g.4xlarge on-demand** (16 vCPU Neoverse V2 / CPU part 0xd4f, 123 GB), eu-west-2c, AMI `ami-05dcc391311f872c0` (Ubuntu 24.04 arm64 20260626), 64 GB gp3. SG `sg-07a39d205c7da2e37` (SSH from home IP only), key pair `armsmith` (`~/.ssh/armsmith.pem`). User-data backstop: `shutdown -h +180`.
- r8g availability in London confirmed via `describe-instance-type-offerings` (r8g.2xlarge/4xlarge, c7g/c8g.4xlarge all offered).

Checks:
- [x] Graviton4 `r8g` up with auto-shutdown backstop.
- [x] **`apx_recipe_run` returns fully structured data — CONFIRMED 2026-07-02.** `code_hotspots` on an `openssl speed sha256` workload over SSH returned `structuredContent` JSON: `status/recipe/stage`, `columns`, `rows` (FUNCTION_NAME, PERIODIC_SAMPLES_SELF, PERIODIC_SAMPLES_SELF_PERCENT — libcrypto 65.5%, SHA256_Final 9.6%, ...), plus the SQL it ran (captures live in a queryable DB). `profiler.py` can consume `structuredContent.rows` directly. Errors are structured too (`status:error`, `details`, error `Code`).
- [x] Recipes enumerated (`apx recipe list` in the container): **`instruction_mix`, `memory_access`, `asct`, `code_hotspots`, `cpu_microarchitecture`** — `memory_access` + `cpu_microarchitecture` are the counter recipes for memory-bound vs compute-bound diagnosis.
- [x] Generic (no `-mcpu=native`) llama.cpp build OK on target; model = ungated `bartowski/Qwen2.5-7B-Instruct-GGUF` Q4_0 (no HF auth needed - keeps judge repro friction zero).
- [x] Baseline `llama-bench -o json` captured (Qwen2.5-7B-Instruct Q4_0, `-p 512 -n 128 -r 3 -t 16`):

  | build | pp512 (prefill) | tg128 (decode) |
  |---|---|---|
  | **portable** (`GGML_NATIVE=OFF` — what pip wheels/docker ship) | 29.17 ± 0.00 | 21.81 ± 0.00 |
  | generic (vanilla cmake) | 134.68 ± 0.09 | 44.49 ± 1.31 |
  | native (`-mcpu=native` flags) | 134.97 ± 0.14 | 44.96 ± 0.01 |

  **Finding 1 (baseline definition):** current llama.cpp defaults `GGML_NATIVE:BOOL=ON` (verified in `build-generic/CMakeCache.txt`), so a vanilla cmake build ≈ `-mcpu=native` — the "add `-mcpu=native`" lever as originally specced is a no-op. The honest naive baseline is the **portable build** (`GGML_NATIVE=OFF`, which is what pip wheels and generic docker images ship), exactly as the PRD's baseline definition anticipated. The real lever-1 gap: **4.6x prefill / 2.1x decode**, before KleidiAI (`GGML_CPU_KLEIDIAI` confirmed OFF by default), quant format, threading, and KV-cache levers stack.

- [x] **Counter discrimination — resolved NO on virtualized targets (2026-07-02):**
  - `cpu_microarchitecture`: FAILS — "Insufficient counters... Minimum of 3 counters required, found 2 for cpu 0" (virtualized r8g exposes only 2 PMU counters).
  - `memory_access`: FAILS — requires SPE; "The Arm SPE driver was not detected on the target" (SPE = metal instances only).
  - `instruction_mix`: FAILS on the same ≥3-counter requirement (after fixing its `python3-venv` prereq on the target).
  - `code_hotspots`: WORKS (sampling-based). Rich on a hot-loop workload (18k samples on `openssl speed`), but thin on short `llama-cli` runs (single-digit samples, ggml kernels unresolved as "Unknown symbol") — needs longer windows / `llama-bench` as the profiled workload / symbol resolution in M1 before hotspot-level lever discrimination is credible.

## Verdict

- **Spike 0a (scriptability): GO** — desk-proven 2026-07-01, re-proven end-to-end against a live Graviton target 2026-07-02.
- **Spike 0b(a) (structured output): GO** — structured JSON with hotspot rows + SQL-queryable captures.
- **Spike 0b(b) (counter discrimination): NO-GO on virtualized Graviton** (2 PMU counters, no SPE — recipes hard-fail; hotspots-on-llama unproven pending M1 sampling fixes). Per the pre-registered decision branch: **loop framing = "try-measure-keep-best"** driven by the deterministic tuner + rigorous timing; Performix `code_hotspots` serves as analyst evidence/narration where samples are rich, not as the control signal. Optional M2 garnish: one profiling pass on a metal instance (full PMU + SPE) for the report.
- **Project: GO.** The measured 4.6x/2.1x naive-to-native gap plus untouched KleidiAI/quant/thread/KV levers confirm the headroom the submission needs.
