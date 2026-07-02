# armsmith — Module Contracts

**Status:** authoritative for Phase 2 build. Every builder codes against this file.
**Source of truth chain:** `docs/PRD.md` → `docs/ARCHITECTURE.md` → `CLAUDE.md` → `docs/spike0-result.md` → **this file** (pins the exact types/signatures) → `docs/plans/2026-07-02-phase-2.md` (the build order).
If code and this file disagree, this file wins until amended by a commit that edits it.

**Ground rules baked in (from spike0 + CLAUDE.md, do not relitigate in code):**
- The honest **baseline is the portable build** (`GGML_NATIVE=OFF`, what pip wheels / generic docker ship). **Lever 1 flips `GGML_NATIVE` OFF→ON** — vanilla cmake already defaults it ON, so "add `-mcpu=native`" as originally worded is a no-op; the real lever is turning the *portable* build native. Measured headroom: 4.6x prefill / 2.1x decode before the other four levers stack.
- On a **virtualized Graviton (`r8g`)** only the **`code_hotspots`** Performix recipe works (2 PMU counters, no SPE → `cpu_microarchitecture`/`memory_access`/`instruction_mix` hard-fail). So `PerformixSnapshot.cache_miss_rate / mem_bandwidth_gbps / ipc` are **`None`** on the primary target; only `hotspots[]` is populated. The loop framing is **try-measure-keep-best** driven by the deterministic tuner; Performix is analyst evidence/narration, not the control signal.
- The **LLM never emits shell**. `Brain` returns `{priority:[action_id...], rationale, suggestion?}`; every id passes `validate_suggestion` before anything runs; `target.py` is the only module that opens SSH and it builds commands from a fixed skeleton + schema-validated `BenchConfig` fields. A unit test asserts an off-registry id is rejected.

---

## 1. Module ownership map

One concern, one owner (ARCHITECTURE "Service Boundaries"). No module does two of {choose, apply, measure-counters, score}.

| Module | Owns | May import | MUST NOT |
|---|---|---|---|
| `models.py` | All frozen dataclasses + `config_id()`/`build_key()` + JSON (de)serialize | stdlib, `numpy` (types only) | do I/O, SSH, spawn processes, call an LLM |
| `bench.py` | CI statistics, `llama-bench` JSON parse, two-stage screen/confirm policy | `models` | open SSH (uses an injected `BenchRunner`) |
| `actions.py` | The 5-lever `REGISTRY`, `validate_suggestion` (the safety gate), `apply_to_config`, capability check | `models` | do I/O; accept a value not in a param schema |
| `profiler.py` | Performix client: docker-stdio MCP + CLI fallback → `PerformixSnapshot` | `models` | build/bench; open the target SSH directly |
| `target.py` | **The only SSH module.** build / bench / quality / describe on the target | `models` (for `build_key`/`config_id`) | choose actions; call the LLM; parse counters |
| `brain.py` | **The only LLM module.** `Brain` protocol + `AnthropicBrain` + `NullBrain` | `models`, `actions` (`validate_suggestion`) | run anything on the target; return an unvalidated id |
| `agent.py` | The loop, keep/revert, trajectory JSONL, recipe, `replay` | all of the above | interpolate LLM text into a command |
| `report.py` | Self-contained HTML render (jinja2) | `models` | mutate the run dir; hit the network |
| `cli.py` | Typer wiring only | all | hold business logic |

## 2. On-disk artifact ownership map

Everything lives under `trajectories/<run_id>/`. JSON via `models.to_json`/`from_json` (dataclass ⇄ dict, ISO-8601 UTC timestamps).

| Path | Type | Written by | Read by |
|---|---|---|---|
| `manifest.json` | `RunManifest` | `cli baseline` | agent, report, repro |
| `baseline.json` | `BenchmarkResult` | `cli baseline` | agent, report |
| `expert.json` | `BenchmarkResult` | `cli baseline` (pre-registered expert config) | agent (gap%), report |
| `configs/<config_id>.json` | `BenchConfig` | agent (each candidate) | audit only (the winning/baseline configs the report + repro need are embedded in `recipe.json`) |
| `snapshots/<config_id>.json` | `PerformixSnapshot` | agent (each step, from profiler) | audit only (the report reads each step's `Diagnosis` from `trajectory.jsonl`) |
| `trajectory.jsonl` | `TrajectoryStep` per line | agent (append per step) | report |
| `recipe.json` | `Recipe` | `cli optimize` (end) | repro, report |
| `report.html` | HTML | `cli report` | humans |
| `examples/bench.yaml` | `WorkloadSpec` (repo, not run dir) | hand-authored | cli, agent |

---

## 3. `models.py` — record types (all `@dataclass(frozen=True)`)

Type aliases (module top):

```python
from typing import Literal, Mapping, Sequence
Stage  = Literal["screen", "confirm"]
Source = Literal["mcp", "cli"]
Bottleneck = Literal["memory-bound", "compute-bound", "unknown"]
ActionKind = Literal["build", "runtime"]   # build ⇒ needs rebuild/model-swap; runtime ⇒ llama-bench args only
```

### 3.1 Value objects

```python
@dataclass(frozen=True)
class MetricStat:
    """A point estimate + 95% CI. Used for both decode and prefill."""
    median:  float
    ci_low:  float      # 2.5th percentile of the bootstrap distribution
    ci_high: float      # 97.5th percentile

@dataclass(frozen=True)
class QualityScore:
    perplexity:    float | None       # llama-perplexity on the pinned small eval text
    kl_vs_baseline: float | None      # mean KL vs the FP16/baseline logits; None on the baseline itself

@dataclass(frozen=True)
class RawSamples:
    """Per-repeat measurements, kept so median+CI are reconstructable off-disk."""
    decode_tok_s:    tuple[float, ...]   # from samples_ts of the n_gen row
    prefill_ttft_ms: tuple[float, ...]   # from samples_ns/1e6 of the n_prompt row
```

### 3.2 `RunManifest` (+ `TargetSpec`, `ModelSpec`)

```python
@dataclass(frozen=True)
class TargetSpec:
    host: str                 # SSH host/IP
    user: str                 # "ubuntu"
    instance_type: str        # "r8g.4xlarge"
    core: str                 # "Neoverse V2 (Graviton4)"
    region: str               # "eu-west-2"
    kernel: str               # uname -r
    cpu_governor: str         # e.g. "performance"
    n_physical_cores: int     # for the threads lever ceiling
    capabilities: tuple[str, ...]   # ("sve2","bf16","i8mm",...) parsed from lscpu → action preconditions

@dataclass(frozen=True)
class ModelSpec:
    name:           str                            # "Qwen2.5-7B-Instruct"
    variants:       Mapping[str, tuple[str, str]]  # quant -> (remote GGUF path, sha256); ONE entry per
                                                   # quant the run may bench, e.g. "Q4_0" -> ("~/m/qwen-q4_0.gguf","<sha>")
    baseline_quant: str                            # the quant the honest baseline uses, e.g. "Q4_0"

    def resolve(self, quant: str) -> tuple[str, str]:
        """(remote path, sha256) for `quant`; KeyError if that variant isn't pinned.
           The quant_format lever (§5 #3) may only select a quant present in `variants`."""

@dataclass(frozen=True)
class RunManifest:
    run_id:           str     # PK, e.g. "2026-07-05T14-03-11Z-r8g"
    target:           TargetSpec
    model:            ModelSpec
    workload_ref:     str     # path to the bench.yaml used
    baseline_ref:     str     # config_id of the honest baseline
    expert_ref:       str     # config_id of the pre-registered expert config (pins the "100%")
    created_at:       str     # ISO-8601 UTC
    armsmith_version: str     # armsmith.__version__
```

### 3.3 `BenchConfig` — one candidate, fully

`config_id` is the PK and is **derived**, never hand-set. `WorkloadSpec` fields are copied in so a config is self-describing and hashable.

```python
@dataclass(frozen=True)
class BenchConfig:
    config_id: str                     # = config_id(self) ; set via BenchConfig.create(...)
    # --- build-time (changing any of these ⇒ target.build re-runs) ---
    cmake_flags: tuple[str, ...]       # e.g. ("-DGGML_NATIVE=ON","-DGGML_CPU_KLEIDIAI=ON")
    quant: str                         # "Q4_0" | "Q8_0" | "Q4_K_M" (selects the GGUF variant)
    # --- runtime knobs (no rebuild) ---
    n_threads: int
    cpu_mask: str | None               # affinity mask e.g. "0xFFFF"; None = llama-bench default
    type_k: str                        # KV key cache: "f16" | "q8_0"
    type_v: str                        # KV value cache: "f16" | "q8_0"
    flash_attn: bool | None            # None = engine default
    env: tuple[tuple[str, str], ...]   # extra env, e.g. (("GGML_KLEIDIAI_SME","1"),)
    # --- workload (fixed within a run, copied from WorkloadSpec) ---
    n_prompt: int                      # prefill token count (llama-bench -p)
    n_gen:    int                      # decode token count   (llama-bench -n)
    n_batch:  int
    n_ubatch: int
    # NOTE: n_repeats is deliberately NOT a field here. It is a RUN parameter (screen_repeats vs
    # confirm_repeats), not config identity — passed to the runner (§4 BenchRunner) so the SAME
    # candidate keeps ONE config_id across screen and confirm. It is recorded on BenchmarkResult only.

    @classmethod
    def create(cls, **fields) -> "BenchConfig":
        """Fill config_id = config_id(fields) then construct. Only way to build one."""
```

Helpers (module-level, pure, deterministic):

```python
def config_id(fields: Mapping) -> str:
    """12-hex-char sha256 over canonical JSON of every field except config_id itself.
       Same knobs ⇒ same id ⇒ result/build caching keys off this."""

def build_key(cfg: BenchConfig) -> str:
    """sha256 over ONLY the rebuild-determining subset: (cmake_flags sorted, quant).
       target.py caches build dirs by this so a revert to a prior build is free."""
```

### 3.4 `BenchmarkResult`

```python
@dataclass(frozen=True)
class BenchmarkResult:
    config_id:      str
    decode_tok_s:   MetricStat         # {median, ci_low, ci_high} — higher is better
    prefill_ttft_ms: MetricStat        # {median, ci_low, ci_high} — lower is better
    quality:        QualityScore
    peak_mem_mb:    float | None       # from the /usr/bin/time -v stderr RSS the BenchRunner returns (§4); None if unmeasured
    model_size_mb:  float              # model_size bytes / 1048576 (MiB)
    n_repeats:      int                # >=5 required when stage=="confirm"
    stage:          Stage              # "screen" | "confirm"
    raw_samples:    RawSamples
```

**Invariants (enforced in `__post_init__` or a `validate()` called by bench.py):** `stage=="confirm" ⇒ n_repeats>=5 and len(raw_samples.decode_tok_s)==n_repeats`. A result whose quality regresses past threshold is *not* a valid win — that check lives in `agent.py`, not here.

### 3.5 `PerformixSnapshot` (+ `HotspotRow`)

```python
@dataclass(frozen=True)
class HotspotRow:
    symbol:       str      # FUNCTION_NAME (may be "Unknown symbol @ 0x..." on unresolved ggml kernels)
    self_samples: int      # PERIODIC_SAMPLES_SELF
    self_pct:     float    # PERIODIC_SAMPLES_SELF_PERCENT
    node_type:    str      # NODE_TYPE ("function")

@dataclass(frozen=True)
class PerformixSnapshot:
    config_id: str
    recipe:    str                       # "code_hotspots" on virtualized Graviton
    source:    Source                    # "mcp" | "cli"
    status:    str                       # "success" | "error"
    hotspots:  tuple[HotspotRow, ...]    # from structuredContent.rows; sparse on short runs
    cache_miss_rate:    float | None = None   # None on virtualized r8g (only 2 PMU counters)
    mem_bandwidth_gbps: float | None = None   # None on virtualized r8g (no SPE)
    ipc:                float | None = None   # None on virtualized r8g
    raw_columns: tuple[str, ...] = ()    # structuredContent.columns (provenance)
    warnings:    tuple[str, ...] = ()     # structuredContent.warnings + stderr summary
```

### 3.6 `ActionSpec` (+ `ParamSpec`)

```python
@dataclass(frozen=True)
class ParamSpec:
    type:    Literal["enum", "int"]
    choices: tuple[str, ...] = ()        # for enum
    min:     int | None = None           # for int
    max:     int | None = None           # for int: the STATIC safety ceiling validate_suggestion ALWAYS
                                         # enforces target-free (e.g. 1024 for n_threads). The per-target
                                         # enumeration bound (n_physical_cores) is applied separately in
                                         # enumerate_candidates — schema max = safety bound, core count = enum bound.

@dataclass(frozen=True)
class ActionSpec:
    id:            str                       # PK; one of the 5 lever ids (see §5)
    name:          str                       # human label
    kind:          ActionKind                # "build" | "runtime"
    params_schema: Mapping[str, ParamSpec]   # param name → allowed values; validation = membership/range
    sets:          tuple[str, ...]           # BenchConfig fields this lever writes (mostly 1:1 from validated
                                             # params; cpu_mask's symbolic value is resolved to a hex mask
                                             # against the target in enumerate_candidates first — see §5)
    apply:         str                       # registered command/flag-fragment TEMPLATE; {slots} filled ONLY from validated params
    revert:        str                       # fragment restoring the default/baseline
    preconditions: tuple[str, ...]           # capability tags required (checked vs TargetSpec.capabilities)
```

`apply`/`revert` are fragments, never full shell lines: for `kind=="build"` a cmake fragment (`-DGGML_NATIVE={state}`), for `kind=="runtime"` a llama-bench arg fragment (`-ctk {type_k} -ctv {type_v}`) or an env pair. `target.py` splices the fragment into a fixed command skeleton. No `{slot}` is ever filled from LLM text — only from `params` that passed `validate_suggestion`.

### 3.7 `TrajectoryStep` (+ `Diagnosis`, `Delta`)

```python
@dataclass(frozen=True)
class Diagnosis:
    bottleneck: Bottleneck               # from hotspots; "unknown" when samples too thin (common on short runs)
    evidence:   str                      # one-line summary of the snapshot the brain read
    source:     Source

@dataclass(frozen=True)
class Delta:
    metric:         str                  # "decode_tok_s"
    pct:            float                 # signed % vs the incumbent
    ci_significant: bool                  # non-overlapping 95% CIs (bench.significant)

@dataclass(frozen=True)
class TrajectoryStep:
    step_idx:         int
    diagnosis:        Diagnosis
    action_id:        str
    params:           Mapping[str, object]
    rationale:        str                 # brain narration (NullBrain: deterministic string)
    before_config_id: str
    after_config_id:  str
    screen:           BenchmarkResult | None   # screen-stage result
    confirm:          BenchmarkResult | None   # confirm-stage result; None if screen didn't promote
    kept:             bool
    delta:            Delta
    quality_ok:       bool                # quality within threshold on confirm
```

### 3.8 `Recipe` — the replayable winner

```python
@dataclass(frozen=True)
class Recipe:
    run_id:          str
    armsmith_version: str
    target_class:    str                  # instance family it was tuned for, e.g. "r8g"
    model:           ModelSpec
    winning_config:  BenchConfig
    baseline_config: BenchConfig
    expert_config:   BenchConfig | None
    baseline_result: BenchmarkResult
    winning_result:  BenchmarkResult
    gap_closed_pct:  float | None         # 100 * (winning-baseline)/(expert-baseline) on decode median;
                                          # None when (expert-baseline) <= eps (degenerate/weak expert) — see gap() §9
    created_at:      str
```

### 3.9 `WorkloadSpec` (from `examples/bench.yaml`)

```python
@dataclass(frozen=True)
class WorkloadSpec:
    n_prompt: int
    n_gen:    int
    n_batch:  int
    n_ubatch: int
    screen_repeats:  int          # cheap stage, e.g. 3
    confirm_repeats: int          # rigorous stage, >=5 (e.g. 7)
    eval_text_path:  str          # small pinned text for llama-perplexity quality guard
    prompt: str | None = None     # fixed prompt; None ⇒ llama-bench synthetic tokens

def load_workload(path: str) -> WorkloadSpec: ...   # YAML → WorkloadSpec
```

### 3.10 Serialization

```python
def to_json(obj) -> str          # any frozen dataclass above → canonical JSON (sorted keys, UTC ISO times)
def from_json(cls, s: str)       # inverse; validates required fields, raises ModelDecodeError on shape mismatch
def append_jsonl(path: Path, step: TrajectoryStep) -> None
def read_jsonl(path: Path) -> list[TrajectoryStep]
```

---

## 4. `bench.py` — statistics, parsing, two-stage policy

`bench.py` owns the CI math and never opens SSH; the target run is injected as a callable.

```python
BenchRunner = Callable[[BenchConfig, int], tuple[str, float | None]]
    # (cfg, n_repeats) -> (raw `llama-bench -o json` stdout, peak_mem_mb).
    # n_repeats is a RUN parameter (screen vs confirm), NOT part of config_id, so one candidate keeps one id.
    # peak_mem_mb is parsed from the `/usr/bin/time -v` stderr the runner wraps (None if unmeasured).
QualityFn   = Callable[[BenchConfig], QualityScore]  # runs llama-perplexity (target.run_quality)

def parse_llama_bench_output(
    raw: str,                     # a JSON ARRAY of llama-bench rows
    *,
    config_id: str,
    stage: Stage,
    quality: QualityScore,
    peak_mem_mb: float | None = None,
) -> BenchmarkResult:
    """Find the prefill row (n_gen==0 and n_prompt>0) and the decode row (n_prompt==0 and n_gen>0).
       decode_tok_s samples = row['samples_ts']; prefill_ttft_ms samples = [ns/1e6 for ns in row['samples_ns']].
       median+CI via median_ci(). model_size_mb = row['model_size']/1048576. n_repeats = len(samples).
       Raises BenchParseError if either row is absent or samples arrays are empty/mismatched."""

def median_ci(samples: Sequence[float], *, confidence: float = 0.95, seed: int = 0, n_boot: int = 10_000) -> MetricStat:
    """Median point estimate + percentile bootstrap CI (seeded ⇒ deterministic).
       ci_low/ci_high = the 2.5/97.5 percentiles of n_boot resampled medians.
       For N<3, ci_low=ci_high=median (flagged, not significant)."""

def significant(candidate: MetricStat, incumbent: MetricStat, *, higher_is_better: bool = True) -> bool:
    """CI-significance = the two 95% CIs are DISJOINT and candidate is on the better side.
       higher_is_better: candidate.ci_low > incumbent.ci_high.
       else:             candidate.ci_high < incumbent.ci_low."""

def ci_disjoint(a: MetricStat, b: MetricStat) -> bool:
    """Symmetric non-overlap test; used by report deltas."""

def promote_to_confirm(screen: BenchmarkResult, incumbent: BenchmarkResult, *, gate_pct: float) -> bool:
    """Screen→confirm gate: promote iff decode median improved by >= gate_pct vs the incumbent.
       Cheap screen (no CI claim) filters candidates before the expensive confirm."""

class Benchmarker:
    """Composes the injected runner + quality fn into the two-stage policy. No SSH here."""
    def __init__(self, run: BenchRunner, quality: QualityFn, workload: WorkloadSpec, *, seed: int = 0): ...
    def screen(self, cfg: BenchConfig) -> BenchmarkResult:
        """Cheap gate. Runs the SAME pp+tg command as confirm (both llama-bench rows present, so the
           parser is satisfied) with n_repeats=workload.screen_repeats; quality=QualityScore(None,None);
           stage='screen'. GATES on the decode metric only — no CI claim, NOT a 'decode-only' command."""
    def confirm(self, cfg: BenchConfig) -> BenchmarkResult:
        """SINGLE-config rigorous capture (baseline/expert absolute numbers): n_repeats=confirm_repeats
           (>=5), computes quality, stage='confirm'. NOT for candidate-vs-incumbent keep/revert — that
           MUST use confirm_ab so the comparison is drift-cancelled."""
    def confirm_ab(self, incumbent_cfg: BenchConfig, candidate_cfg: BenchConfig
                   ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """A/B/A/B INTERLEAVED confirm. Alternates incumbent/candidate repeats within ONE measurement
           window (A,B,A,B,...) up to confirm_repeats each, so systematic thermal/neighbour drift
           cancels, then returns (incumbent_result, candidate_result) — two FRESH confirm-stage results
           measured back-to-back. The agent feeds BOTH to significant(). Sequential confirm(inc) then
           confirm(cand) (A/A/A/B/B/B) does NOT satisfy interleaving (ARCHITECTURE 'Measurement
           integrity', CLAUDE.md rule 3) and must never drive a keep/revert decision."""
```

---

## 5. `actions.py` — the 5-lever registry + the safety gate

`REGISTRY: tuple[ActionSpec, ...]` has **exactly 5** entries. A plain generic/portable build is the **baseline/reset state, not an action**.

| # | `id` | kind | params (schema) | `sets` | apply fragment | precond |
|---|---|---|---|---|---|---|
| 1 | `ggml_native` | build | `state: enum(ON,OFF)` | `cmake_flags` | `-DGGML_NATIVE={state}` | — |
| 2 | `kleidiai` | build | `state: enum(ON,OFF)`, `sme: enum(0,1)` | `cmake_flags,env` | `-DGGML_CPU_KLEIDIAI={state}` (+ env `GGML_KLEIDIAI_SME={sme}`) | `("i8mm",)`; **`sme=1` additionally needs `sme2`** |
| 3 | `quant_format` | build | `quant: enum(Q4_0,Q8_0,Q4_K_M)` | `quant` | selects the pinned GGUF variant `{quant}` (must be a key of `ModelSpec.variants`) | — |
| 4 | `threads` | runtime | `n_threads: int(1..1024)`, `cpu_mask: enum(default,physical,…)` | `n_threads,cpu_mask` | `-t {n_threads} --cpu-mask {cpu_mask}` | — |
| 5 | `kv_cache_type` | runtime | `type_k: enum(f16,q8_0)`, `type_v: enum(f16,q8_0)`, `flash_attn: enum(on,off)` | `type_k,type_v,flash_attn` | `-ctk {type_k} -ctv {type_v} -fa {flash_attn}` | — |

Lever 1 note (spike-pinned): the honest baseline is `state=OFF` (portable); the lever's job is `OFF→ON`.

Schema, coercion, and enumeration notes (so string params land in typed `BenchConfig` fields):
- **enum → field coercion** happens in `apply_to_config`: build-lever `state` values `ON/OFF` splice verbatim into the cmake fragment (`-DGGML_NATIVE=ON`); `flash_attn` `on/off` coerces to `bool` before it fills `BenchConfig.flash_attn`; KV `type_k/type_v` strings pass through. Each enum value maps to its field's real type — never a raw `"on"` into a `bool|None` field.
- **threads ceiling:** the schema `max` on `n_threads` (1024) is the STATIC safety bound `validate_suggestion` always enforces target-free (so Step-3's `n_threads=9999 → ParamValidationError` fires with no target loaded). The real per-target bound (`n_physical_cores`) is applied in `enumerate_candidates`, which additionally SAMPLES a small set of thread counts `{cores, cores//2, cores//4}` rather than sweeping every integer `1..cores` — keeping the grid small (see §9 budget-vs-grid).
- **cpu_mask resolution:** `cpu_mask` is a symbolic enum (`default`,`physical`,…); `enumerate_candidates` (which holds the `TargetSpec`) resolves it to a concrete hex mask (`physical → 0x…` from `n_physical_cores`) before building the `ValidatedAction`, so `apply_to_config` only ever writes a concrete `BenchConfig.cpu_mask`. (This is why `sets` is not strictly 1:1 for this lever.)
- **kleidiai `sme`:** `sme=1` enables SME kernels only where the hardware has them; `enumerate_candidates` emits `sme=1` ONLY when `sme2 ∈ target.capabilities`. Neoverse-V2 (Graviton4 `r8g`) has SVE2/BF16/i8mm but NOT SME, so kleidiai collapses to 2 candidates (state ON/OFF, `sme=0`) there.
- **quant variants:** the `quant_format` lever may only select a quant present in `ModelSpec.variants`; `fetch_model` pre-verifies each variant's sha256 and `run_bench` resolves the GGUF by `cfg.quant` (§7).

```python
REGISTRY: tuple[ActionSpec, ...]           # len == 5; ids exactly the set above
ACTIONS: dict[str, ActionSpec]             # id → spec

@dataclass(frozen=True)
class ValidatedAction:
    action_id: str
    params: Mapping[str, object]           # coerced, schema-checked values only

def validate_suggestion(action_id: str, params: Mapping[str, object]) -> ValidatedAction:
    """THE SAFETY GATE (CLAUDE.md rule 2). 
       - action_id not in ACTIONS            -> raise OffRegistryError
       - any param not in schema             -> raise ParamValidationError
       - enum value not in choices / int out of [min,max] -> raise ParamValidationError
       Returns a ValidatedAction the tuner/target can trust. Unit-tested: an off-registry id
       and an out-of-schema param each raise and NEVER reach the executor."""

def apply_to_config(action: ValidatedAction, cfg: BenchConfig) -> BenchConfig:
    """Deterministic transform: return a NEW BenchConfig with the lever's `sets` fields updated
       from validated params (config_id recomputed). COERCES each enum value to its target field's
       type first (flash_attn on/off → bool; cmake state ON/OFF → fragment string; cpu_mask already
       resolved to a hex mask by enumerate_candidates). This is how the tuner produces a candidate."""

def capabilities_ok(action: ActionSpec, target: TargetSpec) -> bool:
    """All action.preconditions present in target.capabilities. Illegal actions are filtered
       out of the candidate queue before the loop sees them."""

def baseline_config(workload: WorkloadSpec, model: ModelSpec) -> BenchConfig:
    """The honest naive baseline: GGML_NATIVE=OFF, KleidiAI OFF, quant=model.baseline_quant (e.g. Q4_0),
       default threads, f16 KV."""

def expert_config(workload: WorkloadSpec, model: ModelSpec, target: TargetSpec) -> BenchConfig:
    """The pre-registered hand-tuned config, pinned BEFORE discovery so >=90%-of-gap can't be gamed."""
```

---

## 6. `profiler.py` — Performix client → `PerformixSnapshot`

Implements the `Profiler` protocol. `code_hotspots` is the only recipe that returns usable data on a virtualized Graviton (spike0). Counter fields stay `None`.

```python
class Profiler(Protocol):
    def snapshot(self, target: TargetSpec, cfg: BenchConfig, *, workload_cmd: str,
                 recipe: str = "code_hotspots") -> PerformixSnapshot: ...
    # `workload_cmd` is NOT built here (profiler must not import target, §1). The agent passes
    # `shlex.join(target.bench_command(cfg))` — the EXACT argv run_bench uses for cfg — so the
    # profiled command is provably the benched command (same build-<build_key>, same knobs).
    # `target.build(cfg)` must have run before snapshot so the binary being profiled exists.

class MCPProfiler:
    """Spawns the Arm MCP server as a docker stdio subprocess and calls apx_recipe_run over it."""
    DOCKER_CMD = ["docker", "run", "--rm", "-i",
                  "-v", "{keys_dir}:/run/keys/:ro",       # mount the key's PARENT DIR onto the dir path;
                  "armlimited/arm-mcp:latest"]            # the container's apx picks the key inside /run/keys/
    def __init__(self, ssh_key_path: str, timeout_s: int = 300):
        """{keys_dir} = _docker_host_path(dirname(ssh_key_path)) — mount the DIRECTORY containing the
           key, NEVER the file onto a dir path (Docker would create a directory). On the x86 Windows
           control plane the host path is normalized to Docker-Desktop form
           (C:\\Users\\me\\.ssh → /c/Users/me/.ssh) and the drive must be enabled in Docker Desktop
           file-sharing; a raw Windows path fails the bind mount."""

    @staticmethod
    def _docker_host_path(p: str) -> str:
        """Normalize a host path for a Docker Desktop bind mount:
           C:\\Users\\me\\.ssh → /c/Users/me/.ssh (POSIX paths pass through). Unit-tested for the
           Windows arg construction so the -v mount is correct on the actual control plane."""
    def snapshot(self, target, cfg, *, workload_cmd, recipe="code_hotspots") -> PerformixSnapshot:
        """1. Popen(DOCKER_CMD) with text stdio.
           2. initialize (protocolVersion 2024-11-05) -> serverInfo; then notifications/initialized.
           3. tools/call name='apx_recipe_run' arguments={cmd: workload_cmd, remote_ip_addr: target.host,
              remote_usr: target.user, recipe: recipe, invocation_reason: '<config_id>'}.
           4. parse_structured_content(result['structuredContent'], config_id=cfg.config_id, source='mcp').
           Transport/handshake failure -> raise MCPError. A workload-level status:'error' is NOT raised:
           it returns a PerformixSnapshot(status='error', warnings=[...])."""
    def _rpc(self, method: str, params: dict, *, msg_id: int) -> dict: ...   # write JSON line, read matching id

def parse_structured_content(sc: Mapping, *, config_id: str, source: Source) -> PerformixSnapshot:
    """PURE + unit-tested vs tests/fixtures/hotspots_*.json.
       rows -> HotspotRow(symbol=r['FUNCTION_NAME'], self_samples=r['PERIODIC_SAMPLES_SELF'],
                          self_pct=r['PERIODIC_SAMPLES_SELF_PERCENT'], node_type=r.get('NODE_TYPE','function')).
       recipe=sc['recipe'], status=sc['status'], raw_columns=tuple(sc.get('columns',[])),
       warnings=tuple(sc.get('warnings',[])) + ([sc['stderr']] if sc.get('stderr') else []).
       Counter fields left None (recipe returns hotspots only). Missing/extra keys tolerated."""

class NullProfiler:
    """No Performix. Returns PerformixSnapshot(status='success', source='cli', hotspots=()).
       Lets the loop run 'try-measure-keep-best' where Performix is unavailable (spike0 NO-GO(b) path)."""
```

CLI-fallback path (optional, same protocol): `CliProfiler` shells the host `apx` directly and parses the same `structuredContent`; `source='cli'`. Not required for MVP GO.

---

## 7. `target.py` — the only SSH module (paramiko)

Builds a llama.cpp variant for a `BenchConfig`, runs `llama-bench`/`llama-perplexity`, fetches JSON. Commands are `argv` lists spliced from a fixed skeleton + validated config fields — never a shell string interpolated from untrusted input.

```python
@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str

class Target:
    def __init__(self, host: str, user: str, key_path: str, model: ModelSpec, *,
                 remote_root: str = "~/armsmith-work", connect_timeout: int = 30): ...
        # holds `model` so run_bench/bench_command resolve the GGUF by cfg.quant (§3.2 variants).
    def connect(self) -> None                      # paramiko SSHClient, key auth only
    def close(self) -> None
    def __enter__/__exit__                          # context-managed connection

    def describe(self) -> TargetSpec:
        """uname -r, lscpu (core count + capabilities sve2/bf16/i8mm), CPU governor, instance metadata."""
    def fetch_model(self, quants: Sequence[str]) -> None:
        """Ensure EVERY needed GGUF variant is present and integrity-checked. For each quant in
           `quants` resolve (path, sha256) via self.model.resolve(quant); download if absent; verify
           sha256 and raise on mismatch. The quant_format lever needs every swept variant staged;
           `baseline` stages just {model.baseline_quant, expert quant}, `optimize` stages the grid's."""
    def build(self, cfg: BenchConfig) -> str:
        """cmake -B build-<build_key> <cfg.cmake_flags...> ; cmake --build. Cached by build_key(cfg):
           a config that only changes runtime knobs reuses the cached build dir (free revert).
           For kleidiai=on, assert the runtime 'load_tensors: CPU_KLEIDIAI ...' line appears; else BuildError.
           Returns the remote build dir. Raises BuildError on nonzero exit (stderr captured)."""
    def bench_command(self, cfg: BenchConfig) -> tuple[str, ...]:
        """PURE, SSH-free: the exact llama-bench argv for cfg — build-<build_key(cfg)>/bin/llama-bench,
           -m <self.model.resolve(cfg.quant)[0]>, -p/-n/-b/-ub, -t {n_threads} [--cpu-mask] [-ctk/-ctv/-fa].
           Used by run_bench AND by the agent to build the profiler's workload_cmd, so the profiled and
           benched commands are provably identical. Does NOT include -r n_repeats or the time -v wrapper."""
    def run_bench(self, cfg: BenchConfig, n_repeats: int) -> tuple[str, float | None]:
        """`/usr/bin/time -v <bench_command(cfg)> -r {n_repeats}` with cfg.env exported. Returns
           (raw stdout = a JSON array, peak_mem_mb parsed from the time -v 'Maximum resident set size'
           on STDERR). Raises BenchError on nonzero exit. This is the injected BenchRunner (§4)."""
    def run_quality(self, cfg: BenchConfig, eval_text_remote: str) -> QualityScore:
        """llama-perplexity --kl-divergence-base <base.kld> --kl-divergence over the pinned small eval
           text -> QualityScore(perplexity, kl_vs_baseline). Base logits captured once for the baseline."""
    def _run(self, argv: Sequence[str], *, timeout: int | None = None) -> CommandResult:
        """exec_command wrapper; joins argv with shlex.quote; nonzero -> TargetError subclass with stderr."""
```

`run_bench` is the `BenchRunner` injected into `Benchmarker`; `run_quality` is the `QualityFn`.

---

## 8. `brain.py` — the only LLM module (Brain Protocol)

The brain reads evidence and returns a **priority ordering** over registry action ids (+ optional single suggestion). It never runs anything; every id is re-validated. If the LLM is removed the tuner still converges to a winner within CI of the brain-guided run (the honesty test — see §9 budget-vs-grid) via `NullBrain`.

```python
@dataclass(frozen=True)
class Evidence:
    snapshot:   PerformixSnapshot
    history:    tuple[TrajectoryStep, ...]
    candidates: tuple[str, ...]          # remaining legal action ids (post capability filter)
    baseline:   BenchmarkResult
    current:    BenchmarkResult

@dataclass(frozen=True)
class BrainVerdict:
    priority:   tuple[str, ...]          # reordered subset of candidates; validated ids only
    rationale:  str
    suggestion: ValidatedAction | None   # at most one concrete {action_id, params}; validated or None

class Brain(Protocol):
    def analyze(self, evidence: Evidence) -> BrainVerdict: ...

class AnthropicBrain:
    """Default brain. Sends Evidence as JSON, asks for {priority, rationale, suggestion} JSON back,
       parses it, runs EVERY id through validate_suggestion, DROPS invalid ids (logs a warning),
       keeps only ids present in evidence.candidates. Model id + key from env
       (ANTHROPIC_API_KEY, ARMSMITH_BRAIN_MODEL); low temperature. On API/parse error ->
       delegates to NullBrain (never crashes the loop). Uses the `anthropic` SDK."""
    def __init__(self, model: str | None = None, *, client=None): ...
    def analyze(self, evidence: Evidence) -> BrainVerdict: ...

class NullBrain:
    """Deterministic fallback / honesty test. priority = evidence.candidates in fixed registry order,
       rationale='no LLM: deterministic registry order', suggestion=None. Used when --brain=null,
       when no key is set, and as AnthropicBrain's error fallback."""
    def analyze(self, evidence: Evidence) -> BrainVerdict: ...
```

Wiring rule: `brain_from_name(name: str) -> Brain` maps `"claude"→AnthropicBrain`, `"null"→NullBrain`; unknown → `NullBrain` + warning. `repro` uses **no brain at all** (the recipe is replayed directly).

---

## 9. `agent.py` — the loop, keep/revert, trajectory, recipe, replay

Deterministic tuner is the spine; the brain only reorders the candidate queue.

```python
def enumerate_candidates(registry: Sequence[ActionSpec], target: TargetSpec,
                         base: BenchConfig) -> list[tuple[ValidatedAction, BenchConfig]]:
    """The tuner's deterministic, BOUNDED grid off `base`: for each capability-legal action, for each
       schema-legal param combo, produce (ValidatedAction, apply_to_config(action, base)). Stable order.
       Bounding (keeps |grid| ≈ 15-20 so a budget-20 sweep is ~exhaustive — see §9 budget-vs-grid):
         - n_threads is SAMPLED at {cores, cores//2, cores//4}, not every integer 1..cores;
         - cpu_mask symbolic values are resolved to concrete hex masks here (target is in scope);
         - kleidiai `sme=1` is emitted only when `sme2 ∈ target.capabilities`.
       Candidates compose onto `base`, so passing the CURRENT incumbent makes kept levers STACK (§9)."""

def optimize(
    target: Target,
    profiler: Profiler,
    brain: Brain,
    benchmarker: Benchmarker,
    *,
    manifest: RunManifest,
    baseline: BenchmarkResult,
    baseline_cfg: BenchConfig,
    expert: BenchmarkResult | None,
    registry: Sequence[ActionSpec] = REGISTRY,
    budget: int = 20,
    screen_gate_pct: float = 2.0,          # min screen improvement to spend a confirm
    quality_threshold_pct: float = 1.0,    # max tolerated perplexity rise
    kl_max: float = 0.10,                  # max tolerated KL vs baseline
    trajectory_dir: Path,
) -> Recipe:
    """
    # Greedy COORDINATE ASCENT: candidates compose onto the CURRENT incumbent so kept levers STACK
    # (native + kleidiai + quant + threads + kv-cache), not just one lever off the frozen baseline.
    incumbent_cfg, incumbent_res, best = baseline_cfg, baseline, baseline
    applied: set[str] = set()                                  # action_ids already kept (each lever once)
    def fresh_queue(cfg):                                      # recompose remaining levers onto `cfg`
        return [(va, c) for (va, c) in enumerate_candidates(registry, manifest.target, cfg)
                if va.action_id not in applied]
    queue = fresh_queue(incumbent_cfg)
    for step_idx in range(budget):
        if not queue: break
        workload_cmd = shlex.join(target.bench_command(incumbent_cfg))   # EXACT benched argv (§6/§7)
        snap = profiler.snapshot(manifest.target, incumbent_cfg, workload_cmd=workload_cmd)  # may be Null
        verdict = brain.analyze(Evidence(snap, tuple(history), tuple(ids(queue)), baseline, incumbent_res))
        queue = reorder_by_priority(queue, verdict.priority)   # brain reorders; never adds off-registry
        if verdict.suggestion is not None:                     # honor the analyst's concrete pick, if any
            queue = hoist_suggested(queue, verdict.suggestion) # move matching tuner candidate to the front
        action, cand_cfg = queue.pop(0)
        write(trajectory_dir/f'configs/{cand_cfg.config_id}.json', cand_cfg)        # audit
        write(trajectory_dir/f'snapshots/{incumbent_cfg.config_id}.json', snap)     # audit (evidence read)
        screen = benchmarker.screen(cand_cfg)
        if not promote_to_confirm(screen, incumbent_res, gate_pct=screen_gate_pct):
            record(step, kept=False, confirm=None); continue
        inc_fresh, cand_fresh = benchmarker.confirm_ab(incumbent_cfg, cand_cfg)     # A/B/A/B, BOTH fresh
        faster = significant(cand_fresh.decode_tok_s, inc_fresh.decode_tok_s, higher_is_better=True)
        q_ok   = quality_ok(cand_fresh.quality, baseline.quality, quality_threshold_pct, kl_max,
                            changed_quant=('quant' in ACTIONS[action.action_id].sets))
        kept   = faster and q_ok
        if kept:
            incumbent_cfg, incumbent_res, best = cand_cfg, cand_fresh, cand_fresh
            applied.add(action.action_id)
            queue = fresh_queue(incumbent_cfg)                 # ascent: recompose onto the NEW incumbent
        append_jsonl(trajectory_dir/'trajectory.jsonl',
                     TrajectoryStep(... confirm=cand_fresh, kept=kept, delta=delta,
                                    quality_ok=q_ok, rationale=verdict.rationale ...))
    recipe = Recipe(winning_config=incumbent_cfg, winning_result=best, baseline_*, expert_*,
                    gap_closed_pct=gap(best, baseline, expert), ...)
    write(trajectory_dir/'recipe.json', recipe); return recipe
    """

def quality_ok(cand: QualityScore, base: QualityScore, threshold_pct: float, kl_max: float,
               *, changed_quant: bool) -> bool:
    """Reject a speed win that costs quality: if cand.kl_vs_baseline is not None -> kl <= kl_max;
       else perplexity rise <= threshold_pct. Missing quality on BOTH -> NOT ok when `changed_quant`
       (a build lever touched quant, so unmeasured quality is unsafe), ok on pure-runtime levers.
       `changed_quant` is resolved by the agent caller as `'quant' in ACTIONS[action.action_id].sets`
       — quality_ok itself stays purely numeric + the one boolean."""

def gap(winning: BenchmarkResult, baseline: BenchmarkResult, expert: BenchmarkResult | None) -> float | None:
    """gap_closed_pct on decode median = 100*(winning-baseline)/(expert-baseline). Returns None when
       `expert` is None or (expert-baseline) <= eps (e.g. 1e-6): a degenerate/weak expert whose config
       decodes at ~baseline would otherwise divide by zero / emit inf (§3.8)."""

def replay(target: Target, benchmarker: Benchmarker, recipe: Recipe, *, tol_pct: float = 10.0) -> BenchmarkResult:
    """repro path: NO brain, NO profiler. build(recipe.winning_config) + confirm bench on a FRESH instance.
       Assert decode within tol_pct of recipe.winning_result.decode_tok_s.median else raise ReproToleranceError.
       Returns the fresh result (the reproducibility metric)."""
```

`reorder_by_priority` is stable: ids named in `verdict.priority` move to the front in that order; everything else keeps deterministic order. `hoist_suggested` then moves the single tuner-generated candidate whose `(action_id, params)` equals `verdict.suggestion` to the very front (a no-op if the suggestion matches no queued candidate). Both only reorder the tuner's OWN queue — the brain can never inject a config the tuner didn't generate, so the analyst's concrete pick is honored *and* still validated/safe.

**Budget vs grid (honesty invariant).** `enumerate_candidates` is bounded to ≈15-20 candidates and the default `budget=20` is chosen to cover it, so a single coordinate-ascent pass is ~exhaustive. For a full-credibility run `budget` MUST be `>= |grid|`. The honesty test is therefore: removing the LLM (`NullBrain`) reaches a winner **within CI of** the brain-guided run (the brain changes search SPEED and narration, not the reachable optimum) — NOT a bit-identical config, because with coordinate ascent the visiting order can decide which CI-equivalent optimum is reached first. "No worse, within CI" is the guarantee; `repro`'s ±10% then holds on a fresh instance.

---

## 10. `report.py` — self-contained HTML (jinja2)

```python
@dataclass(frozen=True)
class ReportModel:
    manifest: RunManifest
    baseline: BenchmarkResult
    expert:   BenchmarkResult | None
    recipe:   Recipe
    steps:    tuple[TrajectoryStep, ...]
    per_lever: tuple[tuple[str, Delta], ...]   # kept-lever contributions, in apply order

def build_report_model(run_dir: Path) -> ReportModel:
    """Load manifest.json, baseline.json, expert.json, trajectory.jsonl, recipe.json."""

def render_report(run_dir: Path, *, out: Path | None = None) -> Path:
    """Render ONE self-contained HTML (inline CSS/JS, no external assets — portable, CSP-safe).
       Sections: (1) before/after table — decode tok/s, prefill TTFT ms, quality, mem, size, with
       CI bars + red/green deltas + gap-closed%; (2) trajectory timeline — per step: diagnosis,
       action+params, screen vs confirm deltas, keep/revert badge, brain rationale; (3) per-lever
       delta bars; (4) size x speed x quality view. Default out = run_dir/'report.html'. Returns the path."""
```

Templates are either INLINE in the module (default — no packaging risk) or in `report_templates/*.html.j2` loaded via jinja2 `PackageLoader`. If the `.j2` route is taken, they are DATA not modules: `pyproject.toml` MUST ship them via `[tool.setuptools.package-data]` (`armsmith = ["report_templates/*.html.j2"]`) + `include-package-data = true` (or a `MANIFEST.in`), and a test must install the wheel (`pipx`/`pip install`) and render a report so the `<30-min reuse` DX metric can't be broken by a template missing from the wheel. Numbers are formatted from the dataclasses only — the report never recomputes stats.

---

## 11. `cli.py` — wiring (extends the existing stubs; command shape is frozen)

| Command | Builds | Calls | Writes |
|---|---|---|---|
| `provision` | `Provisioner` (Terraform/boto3 — deps: `boto3` + the `terraform` binary) | up / `--destroy` | — |
| `baseline` | `Target`, `Benchmarker` | `describe`, `build`+`confirm` on `baseline_config` and `expert_config` | mints+prints `run_id`, `trajectories/latest`, `manifest.json`, `baseline.json`, `expert.json` |
| `optimize` | `Target`, `Profiler` (MCP else Null), `Brain` (`brain_from_name`), `Benchmarker` | load `manifest`/`baseline`/`expert` from `--run-id` (or `latest`); `agent.optimize(...)` | `trajectory.jsonl`, `recipe.json`, `configs/`, `snapshots/`; prints recipe summary |
| `report` | — | `report.render_report(run_dir)` | `report.html` |
| `repro` | `Target`, `Benchmarker` | load `recipe.json` → `agent.replay(...)` | prints PASS/FAIL vs `tol_pct` |

`run_id` linkage (baseline → optimize → report/repro must share ONE run dir):
- `baseline` is the command that MINTS the `run_id` (`<ISO>-<instance_family>`), creates `trajectories/<run_id>/`, writes `manifest.json`/`baseline.json`/`expert.json`, PRINTS the `run_id`, and updates a `trajectories/latest` pointer.
- `optimize` does NOT mint a fresh id; it takes `--run-id <id>` (default: resolve `trajectories/latest`) so it reads the manifest/baseline/expert `baseline` wrote and appends to the SAME dir.
- `report`/`repro` take the `run_id` as their positional argument; run dir = `trajectories/<run_id>/`.

This AMENDS the earlier "existing signatures stay as-is" note: `optimize` gains `--run-id/--from-baseline` (the frozen `--target/--model/--workload/--budget/--brain` set is otherwise unchanged). The frozen signatures as originally written made the documented artifact ownership impossible — two commands would mint two ids and split the run dir.

---

## 12. Error-handling conventions

Single hierarchy rooted at `ArmsmithError` (in `models.py` or an `errors.py`). Handle at the boundary, log with context, never silently swallow (CLAUDE.md coding rules).

```
ArmsmithError
├─ ModelDecodeError          # bad JSON/shape in from_json
├─ BenchParseError           # llama-bench rows missing / samples empty
├─ ValidationError
│   ├─ OffRegistryError      # unknown action_id  (SAFETY — unit-tested)
│   └─ ParamValidationError  # param off-schema   (SAFETY — unit-tested)
├─ TargetError
│   ├─ BuildError            # cmake/make nonzero, or KleidiAI expected-but-absent
│   └─ BenchError            # llama-bench/perplexity nonzero
├─ ProfilerError
│   ├─ MCPError              # handshake/transport failure
│   └─ ProfilerUnavailable   # docker/apx not present
├─ BrainError                # unrecoverable brain failure (before NullBrain fallback engages)
└─ ReproToleranceError       # replay outside tol_pct
```

Rules:
1. **Safety first.** `validate_suggestion` *raises* on any off-registry id or off-schema param — it never returns a partial. Brain output is advisory: `AnthropicBrain` filters+drops invalid ids and logs; an invalid id can never reach `target.py`. A unit test asserts an off-registry id is rejected and no command runs.
2. **Degrade, don't crash, on the optional layers.** LLM error → `NullBrain`. Performix error / absent → `NullProfiler` and the loop continues "try-measure-keep-best". These are logged at WARNING, not fatal.
3. **Fail fast at the target boundary.** Any nonzero SSH command → a `TargetError` subclass carrying `argv` + captured stderr. Never proceed with a half-built binary.
4. **CLI is the only place errors become exits.** `cli.py` catches `ArmsmithError` → clean message + `raise typer.Exit(1)`; unexpected exceptions propagate (with a stack trace) so bugs are visible.
5. **`logging` everywhere in library code; `print` never.** `cli.py` uses `typer.echo` for user output only. No secrets (keys, creds) in logs or committed trajectories.
6. **Determinism where it's claimed.** All bootstrap CIs seeded (`seed=0`). `repro` asserts within `tol_pct` and raises `ReproToleranceError` otherwise — an unreplayable result is not a result (CLAUDE.md rule 4).
