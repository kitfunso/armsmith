"""armsmith record types.

Owns every frozen dataclass in the project plus deterministic id helpers
(`config_id`, `build_key`) and JSON (de)serialization (`to_json`/`from_json`,
`append_jsonl`/`read_jsonl`). See docs/CONTRACTS.md §3 for the authoritative
shape; this module is the implementation of that section.

Also hosts the `ArmsmithError` hierarchy (docs/CONTRACTS.md §12) since every
other module imports `models` and needs a common error root.
"""

from __future__ import annotations

import hashlib
import json
import types
from collections.abc import Mapping as ABCMapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, get_args, get_origin, get_type_hints

# --------------------------------------------------------------------------
# Type aliases
# --------------------------------------------------------------------------

Stage = Literal["screen", "confirm"]
Source = Literal["mcp", "cli"]
Bottleneck = Literal["memory-bound", "compute-bound", "unknown"]
ActionKind = Literal[
    "build", "runtime"
]  # build => rebuild/model-swap; runtime => llama-bench args only


# --------------------------------------------------------------------------
# Error hierarchy (docs/CONTRACTS.md §12)
# --------------------------------------------------------------------------


class ArmsmithError(Exception):
    """Root of every armsmith exception."""


class ModelDecodeError(ArmsmithError):
    """Bad JSON / shape mismatch in `from_json`."""


class BenchParseError(ArmsmithError):
    """`llama-bench` rows missing or samples arrays empty/mismatched."""


class ValidationError(ArmsmithError):
    """Base for the safety-gate errors raised by `actions.validate_suggestion`."""


class OffRegistryError(ValidationError):
    """An action id not present in `ACTIONS`."""


class ParamValidationError(ValidationError):
    """A param not in the action's schema, or off-schema value."""


class TargetError(ArmsmithError):
    """Nonzero exit from a command run on the SSH target."""


class BuildError(TargetError):
    """`cmake`/`make` nonzero, or KleidiAI expected-but-absent at runtime."""


class BenchError(TargetError):
    """`llama-bench`/`llama-perplexity` nonzero."""


class ProfilerError(ArmsmithError):
    """Base for Performix client failures."""


class MCPError(ProfilerError):
    """MCP handshake/transport failure."""


class ProfilerUnavailable(ProfilerError):
    """docker/apx not present."""


class BrainError(ArmsmithError):
    """Unrecoverable brain failure (before the `NullBrain` fallback engages)."""


class ReproToleranceError(ArmsmithError):
    """`agent.replay` result fell outside the reproducibility tolerance."""


# --------------------------------------------------------------------------
# §3.1 Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricStat:
    """A point estimate + 95% CI. Used for both decode and prefill."""

    median: float
    ci_low: float  # 2.5th percentile of the bootstrap distribution
    ci_high: float  # 97.5th percentile


@dataclass(frozen=True)
class QualityScore:
    perplexity: float | None  # llama-perplexity on the pinned small eval text
    kl_vs_baseline: (
        float | None
    )  # mean KL vs the FP16/baseline logits; None on the baseline itself


@dataclass(frozen=True)
class RawSamples:
    """Per-repeat measurements, kept so median+CI are reconstructable off-disk."""

    decode_tok_s: tuple[float, ...]  # from samples_ts of the n_gen row
    prefill_ttft_ms: tuple[float, ...]  # from samples_ns/1e6 of the n_prompt row


# --------------------------------------------------------------------------
# §3.2 RunManifest (+ TargetSpec, ModelSpec)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    host: str  # SSH host/IP
    user: str  # "ubuntu"
    instance_type: str  # "r8g.4xlarge"
    core: str  # "Neoverse V2 (Graviton4)"
    region: str  # "eu-west-2"
    kernel: str  # uname -r
    cpu_governor: str  # e.g. "performance"
    n_physical_cores: int  # for the threads lever ceiling
    capabilities: tuple[str, ...]  # ("sve2","bf16","i8mm",...) parsed from lscpu


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "Qwen2.5-7B-Instruct"
    variants: Mapping[str, tuple[str, str]]  # quant -> (remote GGUF path, sha256)
    baseline_quant: str  # the quant the honest baseline uses, e.g. "Q4_0"

    def resolve(self, quant: str) -> tuple[str, str]:
        """(remote path, sha256) for `quant`; KeyError if that variant isn't pinned."""
        return self.variants[quant]


@dataclass(frozen=True)
class RunManifest:
    run_id: str  # PK, e.g. "2026-07-05T14-03-11Z-r8g"
    target: TargetSpec
    model: ModelSpec
    workload_ref: str  # path to the bench.yaml used
    baseline_ref: str  # config_id of the honest baseline
    expert_ref: str  # config_id of the pre-registered expert config
    created_at: str  # ISO-8601 UTC
    armsmith_version: str  # armsmith.__version__


# --------------------------------------------------------------------------
# §3.3 BenchConfig
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    """One candidate, fully. `config_id` is the PK and is derived, never hand-set."""

    config_id: str  # = config_id(self); set via BenchConfig.create(...)
    # --- build-time (changing any of these => target.build re-runs) ---
    cmake_flags: tuple[str, ...]  # e.g. ("-DGGML_NATIVE=ON","-DGGML_CPU_KLEIDIAI=ON")
    quant: str  # "Q4_0" | "Q8_0" | "Q4_K_M" (selects the GGUF variant)
    # --- runtime knobs (no rebuild) ---
    n_threads: int
    cpu_mask: str | None  # affinity mask e.g. "0xFFFF"; None = llama-bench default
    type_k: str  # KV key cache: "f16" | "q8_0"
    type_v: str  # KV value cache: "f16" | "q8_0"
    flash_attn: bool | None  # None = engine default
    env: tuple[tuple[str, str], ...]  # extra env, e.g. (("GGML_KLEIDIAI_SME","1"),)
    # --- workload (fixed within a run, copied from WorkloadSpec) ---
    n_prompt: int  # prefill token count (llama-bench -p)
    n_gen: int  # decode token count   (llama-bench -n)
    n_batch: int
    n_ubatch: int
    # NOTE: n_repeats is deliberately NOT a field here (see docs/CONTRACTS.md §3.3).

    @classmethod
    def create(cls, **fields: Any) -> "BenchConfig":
        """Fill config_id = config_id(fields) then construct. Only way to build one."""
        cid = config_id(fields)
        return cls(config_id=cid, **fields)


def config_id(fields: Mapping[str, Any]) -> str:
    """12-hex-char sha256 over canonical JSON of every field except config_id itself."""
    payload = {k: v for k, v in fields.items() if k != "config_id"}
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def build_key(cfg: BenchConfig) -> str:
    """sha256 over ONLY the rebuild-determining subset: (cmake_flags sorted, quant)."""
    payload = {"cmake_flags": sorted(cfg.cmake_flags), "quant": cfg.quant}
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# §3.4 BenchmarkResult
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    config_id: str
    decode_tok_s: MetricStat  # higher is better
    prefill_ttft_ms: MetricStat  # lower is better
    quality: QualityScore
    peak_mem_mb: float | None  # None if unmeasured
    model_size_mb: float  # model_size bytes / 1048576 (MiB)
    n_repeats: int  # >=5 required when stage=="confirm"
    stage: Stage
    raw_samples: RawSamples

    def __post_init__(self) -> None:
        if self.stage == "confirm":
            if self.n_repeats < 5:
                raise ValueError(
                    f"BenchmarkResult stage='confirm' requires n_repeats>=5, got {self.n_repeats}"
                )
            if len(self.raw_samples.decode_tok_s) != self.n_repeats:
                raise ValueError(
                    "BenchmarkResult stage='confirm' requires "
                    f"len(raw_samples.decode_tok_s)=={self.n_repeats}, "
                    f"got {len(self.raw_samples.decode_tok_s)}"
                )


# --------------------------------------------------------------------------
# §3.5 PerformixSnapshot (+ HotspotRow)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HotspotRow:
    symbol: str  # FUNCTION_NAME (may be "Unknown symbol @ 0x..." on unresolved ggml kernels)
    self_samples: int  # PERIODIC_SAMPLES_SELF
    self_pct: float  # PERIODIC_SAMPLES_SELF_PERCENT
    node_type: str  # NODE_TYPE ("function")


@dataclass(frozen=True)
class PerformixSnapshot:
    config_id: str
    recipe: str  # "code_hotspots" on virtualized Graviton
    source: Source
    status: str  # "success" | "error"
    hotspots: tuple[
        HotspotRow, ...
    ]  # from structuredContent.rows; sparse on short runs
    cache_miss_rate: float | None = (
        None  # None on virtualized r8g (only 2 PMU counters)
    )
    mem_bandwidth_gbps: float | None = None  # None on virtualized r8g (no SPE)
    ipc: float | None = None  # None on virtualized r8g
    raw_columns: tuple[str, ...] = ()  # structuredContent.columns (provenance)
    warnings: tuple[str, ...] = ()  # structuredContent.warnings + stderr summary


# --------------------------------------------------------------------------
# §3.6 ActionSpec (+ ParamSpec)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    type: Literal["enum", "int"]
    choices: tuple[str, ...] = ()  # for enum
    min: int | None = None  # for int
    max: int | None = None  # for int: the STATIC safety ceiling


@dataclass(frozen=True)
class ActionSpec:
    id: str  # PK; one of the 5 lever ids
    name: str  # human label
    kind: ActionKind  # "build" | "runtime"
    params_schema: Mapping[str, ParamSpec]  # param name -> allowed values
    sets: tuple[str, ...]  # BenchConfig fields this lever writes
    apply: str  # registered command/flag-fragment TEMPLATE
    revert: str  # fragment restoring the default/baseline
    preconditions: tuple[str, ...]  # capability tags required


# --------------------------------------------------------------------------
# §3.7 TrajectoryStep (+ Diagnosis, Delta)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnosis:
    bottleneck: Bottleneck  # from hotspots; "unknown" when samples too thin
    evidence: str  # one-line summary of the snapshot the brain read
    source: Source


@dataclass(frozen=True)
class Delta:
    metric: str  # "decode_tok_s"
    pct: float  # signed % vs the incumbent
    ci_significant: bool  # non-overlapping 95% CIs (bench.significant)


@dataclass(frozen=True)
class TrajectoryStep:
    step_idx: int
    diagnosis: Diagnosis
    action_id: str
    params: Mapping[str, Any]
    rationale: str  # brain narration (NullBrain: deterministic string)
    before_config_id: str
    after_config_id: str
    screen: BenchmarkResult | None  # screen-stage result
    confirm: (
        BenchmarkResult | None
    )  # confirm-stage result; None if screen didn't promote
    kept: bool
    delta: Delta
    quality_ok: bool  # quality within threshold on confirm


# --------------------------------------------------------------------------
# §3.8 Recipe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    run_id: str
    armsmith_version: str
    target_class: str  # instance family it was tuned for, e.g. "r8g"
    model: ModelSpec
    winning_config: BenchConfig
    baseline_config: BenchConfig
    expert_config: BenchConfig | None
    baseline_result: BenchmarkResult
    winning_result: BenchmarkResult
    gap_closed_pct: float | None  # None when (expert-baseline) <= eps
    created_at: str


# --------------------------------------------------------------------------
# §3.9 WorkloadSpec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadSpec:
    n_prompt: int
    n_gen: int
    n_batch: int
    n_ubatch: int
    screen_repeats: int  # cheap stage, e.g. 3
    confirm_repeats: int  # rigorous stage, >=5 (e.g. 7)
    eval_text_path: str  # small pinned text for llama-perplexity quality guard
    prompt: str | None = None  # fixed prompt; None => llama-bench synthetic tokens


def load_workload(path: str) -> WorkloadSpec:
    """YAML -> WorkloadSpec."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return WorkloadSpec(
        n_prompt=data["n_prompt"],
        n_gen=data["n_gen"],
        n_batch=data["n_batch"],
        n_ubatch=data["n_ubatch"],
        screen_repeats=data["screen_repeats"],
        confirm_repeats=data["confirm_repeats"],
        eval_text_path=data["eval_text_path"],
        prompt=data.get("prompt"),
    )


# --------------------------------------------------------------------------
# §3.10 Serialization
# --------------------------------------------------------------------------


def _optional_inner(tp: Any) -> Any | None:
    """If `tp` is `X | None` (or `Optional[X]`), return X; else None."""
    origin = get_origin(tp)
    if origin is not None and (
        origin is types.UnionType or str(origin).endswith("Union")
    ):
        args = get_args(tp)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
    return None


def _encode_value(value: Any, tp: Any) -> Any:
    if value is None:
        return None
    inner = _optional_inner(tp)
    if inner is not None:
        return _encode_value(value, inner)
    origin = get_origin(tp)
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return [_encode_value(v, elem_t) for v in value]
        return [_encode_value(v, t) for v, t in zip(value, args)]
    if origin is dict or origin is ABCMapping:
        args = get_args(tp)
        val_t = args[1] if len(args) == 2 else Any
        return {k: _encode_value(v, val_t) for k, v in value.items()}
    if is_dataclass(tp) and isinstance(tp, type):
        return _dataclass_to_dict(value)
    return value


def _decode_value(value: Any, tp: Any) -> Any:
    if value is None:
        return None
    inner = _optional_inner(tp)
    if inner is not None:
        return _decode_value(value, inner)
    origin = get_origin(tp)
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return tuple(_decode_value(v, elem_t) for v in value)
        return tuple(_decode_value(v, t) for v, t in zip(value, args))
    if origin is dict or origin is ABCMapping:
        args = get_args(tp)
        val_t = args[1] if len(args) == 2 else Any
        return {k: _decode_value(v, val_t) for k, v in value.items()}
    if is_dataclass(tp) and isinstance(tp, type):
        return _dict_to_dataclass(tp, value)
    return value


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    hints = get_type_hints(type(obj))
    result: dict[str, Any] = {}
    for f in fields(obj):
        result[f.name] = _encode_value(getattr(obj, f.name), hints[f.name])
    return result


def _dict_to_dataclass(cls: type, data: Any) -> Any:
    if not isinstance(data, ABCMapping):
        raise ModelDecodeError(
            f"expected a JSON object for {cls.__name__}, got {type(data).__name__}"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            if f.default is not MISSING or f.default_factory is not MISSING:  # type: ignore[misc]
                continue
            raise ModelDecodeError(f"{cls.__name__} missing required field {f.name!r}")
        kwargs[f.name] = _decode_value(data[f.name], hints[f.name])
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ModelDecodeError(f"failed to construct {cls.__name__}: {exc}") from exc


def to_json(obj: Any) -> str:
    """Any frozen dataclass above -> canonical JSON (sorted keys)."""
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"to_json expects a dataclass instance, got {type(obj)!r}")
    return json.dumps(_dataclass_to_dict(obj), sort_keys=True)


def from_json(cls: type, s: str) -> Any:
    """Inverse of to_json; validates required fields, raises ModelDecodeError on shape mismatch."""
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ModelDecodeError(f"invalid JSON for {cls.__name__}: {exc}") from exc
    return _dict_to_dataclass(cls, data)


def append_jsonl(path: Path, step: TrajectoryStep) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(to_json(step))
        fh.write("\n")


def read_jsonl(path: Path) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            steps.append(from_json(TrajectoryStep, line))
    return steps
