"""armsmith optimization action space: the 5-lever registry + the safety gate.

CONTRACTS.md §5. The LLM analyst never emits shell (CLAUDE.md rule 2): it returns
action ids + params, and every id/param MUST pass :func:`validate_suggestion`
before anything is rendered into a command or run. ``validate_suggestion`` hard-
rejects off-registry ids (``OffRegistryError``) and out-of-schema params
(``ParamValidationError``), so an unknown id or a bad value can never reach the
executor. ``apply/revert`` templates carry named ``{slots}`` filled ONLY from
validated params — never free-text interpolation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from armsmith.models import (
    ActionSpec,
    BenchConfig,
    ModelSpec,
    OffRegistryError,
    ParamSpec,
    ParamValidationError,
    TargetSpec,
    WorkloadSpec,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# The registry: EXACTLY 5 levers (CONTRACTS §5). A plain portable build is the  #
# baseline/reset state, NOT an action.                                          #
# --------------------------------------------------------------------------- #
REGISTRY: tuple[ActionSpec, ...] = (
    ActionSpec(
        id="ggml_native",
        name="GGML native build (portable OFF -> ON)",
        kind="build",
        params_schema={"state": ParamSpec(type="enum", choices=("ON", "OFF"))},
        sets=("cmake_flags",),
        apply="-DGGML_NATIVE={state}",
        revert="-DGGML_NATIVE=OFF",
        preconditions=(),
    ),
    ActionSpec(
        id="kleidiai",
        name="KleidiAI CPU microkernels",
        kind="build",
        params_schema={
            "state": ParamSpec(type="enum", choices=("ON", "OFF")),
            "sme": ParamSpec(type="enum", choices=("0", "1")),
        },
        sets=("cmake_flags", "env"),
        apply="-DGGML_CPU_KLEIDIAI={state}",
        revert="-DGGML_CPU_KLEIDIAI=OFF",
        preconditions=("i8mm",),
    ),
    ActionSpec(
        id="quant_format",
        name="Quantization format swap",
        kind="build",
        params_schema={
            "quant": ParamSpec(type="enum", choices=("Q4_0", "Q8_0", "Q4_K_M"))
        },
        sets=("quant",),
        apply="{quant}",
        revert="Q4_0",
        preconditions=(),
    ),
    ActionSpec(
        id="threads",
        name="Threads / CPU affinity",
        kind="runtime",
        params_schema={
            "n_threads": ParamSpec(type="int", min=1, max=1024),
            "cpu_mask": ParamSpec(type="enum", choices=("default", "physical")),
        },
        sets=("n_threads", "cpu_mask"),
        apply="-t {n_threads} --cpu-mask {cpu_mask}",
        revert="-t 0",
        preconditions=(),
    ),
    ActionSpec(
        id="kv_cache_type",
        name="KV-cache type / flash-attention",
        kind="runtime",
        params_schema={
            "type_k": ParamSpec(type="enum", choices=("f16", "q8_0")),
            "type_v": ParamSpec(type="enum", choices=("f16", "q8_0")),
            "flash_attn": ParamSpec(type="enum", choices=("on", "off")),
        },
        sets=("type_k", "type_v", "flash_attn"),
        apply="-ctk {type_k} -ctv {type_v} -fa {flash_attn}",
        revert="-ctk f16 -ctv f16",
        preconditions=(),
    ),
)

ACTIONS: dict[str, ActionSpec] = {spec.id: spec for spec in REGISTRY}


@dataclass(frozen=True)
class ValidatedAction:
    """An action id + coerced, schema-checked params. The ONLY thing the tuner /
    target act on — produced solely by :func:`validate_suggestion`."""

    action_id: str
    params: Mapping[str, object]


# --------------------------------------------------------------------------- #
# The safety gate (CLAUDE.md rule 2 / CONTRACTS §5, §12).                        #
# --------------------------------------------------------------------------- #
def validate_suggestion(
    action_id: str, params: Mapping[str, object]
) -> ValidatedAction:
    """THE SAFETY GATE. Raise on anything off-registry or off-schema; never
    return a partial or an unchecked value.

    - ``action_id`` not in :data:`ACTIONS`            -> ``OffRegistryError``
    - a param key not in the spec's schema            -> ``ParamValidationError``
    - an enum value not in ``choices`` (case-folded)  -> ``ParamValidationError``
    - an int out of ``[min, max]`` (the STATIC ceiling) -> ``ParamValidationError``

    Enum values are canonicalized to their declared spelling (e.g. ``"on" ->
    "ON"``) so downstream rendering is deterministic. Because this raises before
    any caller renders a command from ``ActionSpec.apply``, an invalid suggestion
    can never reach the executor.
    """
    if action_id not in ACTIONS:
        raise OffRegistryError(
            f"unknown action id {action_id!r}; not one of {sorted(ACTIONS)}"
        )
    spec = ACTIONS[action_id]
    coerced: dict[str, object] = {}
    for key, value in params.items():
        if key not in spec.params_schema:
            raise ParamValidationError(
                f"{action_id!r}: unknown param {key!r}; "
                f"allowed {sorted(spec.params_schema)}"
            )
        coerced[key] = _coerce_param(action_id, key, spec.params_schema[key], value)
    _check_cross_param_rules(action_id, coerced)
    logger.debug("validated %s params=%s", action_id, coerced)
    return ValidatedAction(action_id=action_id, params=coerced)


def _check_cross_param_rules(action_id: str, params: Mapping[str, object]) -> None:
    """Engine invariants that span params. llama.cpp refuses to create a
    context with a QUANTIZED V-cache unless flash attention is enabled
    (observed live on r8g 2026-07-02: `-ctv q8_0 -fa 0` -> "failed to create
    context with model"). The gate is the single source of combo legality:
    the tuner's enumerator and any brain suggestion both flow through here."""
    if action_id == "kv_cache_type":
        type_v = params.get("type_v")
        flash_attn = params.get("flash_attn")
        if type_v is not None and type_v != "f16" and flash_attn != "on":
            raise ParamValidationError(
                "kv_cache_type: a quantized V-cache (type_v="
                f"{type_v!r}) requires flash_attn='on' (llama.cpp cannot "
                "create the context otherwise)"
            )


def _coerce_param(action_id: str, name: str, pspec: ParamSpec, value: object) -> object:
    """Membership/range check + canonicalization for one param. Raises
    ``ParamValidationError`` on any violation."""
    if pspec.type == "enum":
        text = str(value)
        for choice in pspec.choices:
            if text.casefold() == choice.casefold():
                return choice  # canonical declared spelling
        raise ParamValidationError(
            f"{action_id!r}: {name}={value!r} not in choices {pspec.choices}"
        )
    if pspec.type == "int":
        # bool is an int subclass; reject it — a flag is not a thread count.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ParamValidationError(
                f"{action_id!r}: {name} must be an int, got {type(value).__name__}"
            )
        if pspec.min is not None and value < pspec.min:
            raise ParamValidationError(
                f"{action_id!r}: {name}={value} below min {pspec.min}"
            )
        if pspec.max is not None and value > pspec.max:
            raise ParamValidationError(
                f"{action_id!r}: {name}={value} above max {pspec.max}"
            )
        return value
    raise ParamValidationError(  # pragma: no cover - guards a malformed schema
        f"{action_id!r}: {name} has unknown param type {pspec.type!r}"
    )


# --------------------------------------------------------------------------- #
# Applying a validated action to a config (the tuner's candidate generator).     #
# --------------------------------------------------------------------------- #
def apply_to_config(action: ValidatedAction, cfg: BenchConfig) -> BenchConfig:
    """Deterministic transform: return a NEW :class:`BenchConfig` with the lever's
    ``sets`` fields updated from validated params (``config_id`` recomputed).

    Each enum value is coerced to its target field's real type: build ``state``
    ``ON/OFF`` splices into a cmake fragment; ``flash_attn`` ``on/off`` becomes a
    ``bool``; ``cpu_mask`` ``"default"`` becomes ``None`` (any other value is the
    concrete hex mask ``enumerate_candidates`` already resolved). The input ``cfg``
    is never mutated (frozen).
    """
    params = action.params
    fields = _config_fields(cfg)
    aid = action.action_id

    if aid == "ggml_native":
        if "state" in params:
            fields["cmake_flags"] = _set_cmake_flag(
                cfg.cmake_flags, "GGML_NATIVE", str(params["state"])
            )
    elif aid == "kleidiai":
        if "state" in params:
            fields["cmake_flags"] = _set_cmake_flag(
                cfg.cmake_flags, "GGML_CPU_KLEIDIAI", str(params["state"])
            )
        if "sme" in params:
            fields["env"] = _set_env(cfg.env, "GGML_KLEIDIAI_SME", str(params["sme"]))
    elif aid == "quant_format":
        if "quant" in params:
            fields["quant"] = str(params["quant"])
    elif aid == "threads":
        if "n_threads" in params:
            fields["n_threads"] = int(params["n_threads"])  # type: ignore[arg-type]
        if "cpu_mask" in params:
            mask = params["cpu_mask"]
            fields["cpu_mask"] = None if mask == "default" else str(mask)
    elif aid == "kv_cache_type":
        if "type_k" in params:
            fields["type_k"] = str(params["type_k"])
        if "type_v" in params:
            fields["type_v"] = str(params["type_v"])
        if "flash_attn" in params:
            fields["flash_attn"] = params["flash_attn"] == "on"

    return BenchConfig.create(**fields)


def capabilities_ok(action: ActionSpec, target: TargetSpec) -> bool:
    """True iff every ``action.preconditions`` tag is present in
    ``target.capabilities``. Illegal actions are filtered before the loop."""
    return all(cap in target.capabilities for cap in action.preconditions)


# --------------------------------------------------------------------------- #
# Reference configs: the honest baseline and the pre-registered expert.          #
# --------------------------------------------------------------------------- #
def baseline_config(workload: WorkloadSpec, model: ModelSpec) -> BenchConfig:
    """The honest naive baseline (spike0): the portable build ``GGML_NATIVE=OFF``,
    KleidiAI off, ``quant=model.baseline_quant``, engine-default threads
    (``n_threads=0`` — llama-bench then picks all physical cores), f16 KV."""
    return BenchConfig.create(
        cmake_flags=("-DGGML_NATIVE=OFF",),
        quant=model.baseline_quant,
        n_threads=0,
        cpu_mask=None,
        type_k="f16",
        type_v="f16",
        flash_attn=None,
        env=(),
        n_prompt=workload.n_prompt,
        n_gen=workload.n_gen,
        n_batch=workload.n_batch,
        n_ubatch=workload.n_ubatch,
    )


def expert_config(
    workload: WorkloadSpec, model: ModelSpec, target: TargetSpec
) -> BenchConfig:
    """The pre-registered hand-tuned config, pinned BEFORE discovery so the
    gap-closed metric can't be gamed. Native build + (where the core has i8mm)
    KleidiAI, the Arm-repack-friendly baseline quant, threads = physical cores,
    quality-safe f16 KV. SME kernels are enabled only on cores that expose
    ``sme2`` (Neoverse-V2 / Graviton4 does not)."""
    cmake = ["-DGGML_NATIVE=ON"]
    env: list[tuple[str, str]] = []
    if "i8mm" in target.capabilities:
        cmake.append("-DGGML_CPU_KLEIDIAI=ON")
        if "sme2" in target.capabilities:
            env.append(("GGML_KLEIDIAI_SME", "1"))
    return BenchConfig.create(
        cmake_flags=tuple(sorted(cmake)),
        quant=model.baseline_quant,
        n_threads=target.n_physical_cores,
        cpu_mask=None,
        type_k="f16",
        type_v="f16",
        flash_attn=None,
        env=tuple(sorted(env)),
        n_prompt=workload.n_prompt,
        n_gen=workload.n_gen,
        n_batch=workload.n_batch,
        n_ubatch=workload.n_ubatch,
    )


# --------------------------------------------------------------------------- #
# Internal helpers (pure, deterministic).                                        #
# --------------------------------------------------------------------------- #
def _config_fields(cfg: BenchConfig) -> dict[str, object]:
    """Every ``BenchConfig`` field except the derived ``config_id`` — the kwargs
    for ``BenchConfig.create`` after a lever updates its own subset."""
    return {
        "cmake_flags": tuple(cfg.cmake_flags),
        "quant": cfg.quant,
        "n_threads": cfg.n_threads,
        "cpu_mask": cfg.cpu_mask,
        "type_k": cfg.type_k,
        "type_v": cfg.type_v,
        "flash_attn": cfg.flash_attn,
        "env": tuple(cfg.env),
        "n_prompt": cfg.n_prompt,
        "n_gen": cfg.n_gen,
        "n_batch": cfg.n_batch,
        "n_ubatch": cfg.n_ubatch,
    }


def _set_cmake_flag(flags: tuple[str, ...], key: str, value: str) -> tuple[str, ...]:
    """Return ``flags`` with ``-D{key}=...`` replaced by ``-D{key}={value}``
    (appended if absent). Sorted so semantically equal flag sets share one id."""
    prefix = f"-D{key}="
    kept = [flag for flag in flags if not flag.startswith(prefix)]
    kept.append(f"{prefix}{value}")
    return tuple(sorted(kept))


def _set_env(
    env: tuple[tuple[str, str], ...], key: str, value: str
) -> tuple[tuple[str, str], ...]:
    """Return ``env`` with ``key`` set to ``value``, sorted by key for canonical
    ordering."""
    kept = [(k, v) for (k, v) in env if k != key]
    kept.append((key, value))
    return tuple(sorted(kept))
