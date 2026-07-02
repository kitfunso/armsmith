"""agent.py -- the discovery loop, keep/revert, trajectory, recipe, replay.

CONTRACTS.md section 9. The deterministic tuner is the spine; the LLM ``Brain``
only reorders / hoists the tuner's own candidate queue -- it never injects a
config the tuner did not generate, and every id it names is already validated
against the registry (CLAUDE.md rule 2).

The loop is greedy **coordinate ascent**: candidates are recomposed onto the
CURRENT incumbent after every keep, so accepted levers STACK (native + kleidiai
+ quant + threads + kv-cache) rather than each being measured only off the frozen
baseline. A candidate is kept iff its decode gain is CI-significant (non-
overlapping 95% CIs, via ``bench.significant`` on an interleaved A/B/A/B confirm)
AND its quality is within threshold (``quality_ok``); otherwise it is reverted.
Every evaluated candidate appends one ``TrajectoryStep`` to
``trajectories/<run_id>/trajectory.jsonl``; the winning config is written to
``recipe.json`` for deterministic, LLM-free replay via :func:`replay`.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from armsmith.actions import (
    ACTIONS,
    REGISTRY,
    ValidatedAction,
    apply_to_config,
    capabilities_ok,
    validate_suggestion,
)
from armsmith.bench import promote_to_confirm, significant
from armsmith.brain import Evidence
from armsmith.models import (
    ActionSpec,
    BenchConfig,
    BenchmarkResult,
    Delta,
    Diagnosis,
    PerformixSnapshot,
    ProfilerError,
    QualityScore,
    Recipe,
    ReproToleranceError,
    RunManifest,
    TargetSpec,
    TrajectoryStep,
    append_jsonl,
    to_json,
)

if (
    TYPE_CHECKING
):  # import-cycle-free type hints; these are Protocols/duck-typed at runtime
    from armsmith.bench import Benchmarker
    from armsmith.brain import Brain
    from armsmith.profiler import Profiler
    from armsmith.target import Target

logger = logging.getLogger(__name__)

# Coarse bottleneck heuristic, only reachable when Performix returns PMU counters
# (bare-metal Graviton); on the virtualized r8g the counter fields are None and the
# diagnosis stays "unknown" (spike0-result.md).
_HIGH_MISS_RATE = 0.20
_GAP_EPS = (
    1e-6  # a weak/degenerate expert (decodes ~= baseline) makes gap-closed undefined
)


# --------------------------------------------------------------------------------------
# Candidate enumeration -- the tuner's deterministic, bounded grid
# --------------------------------------------------------------------------------------
def enumerate_candidates(
    registry: Sequence[ActionSpec],
    target: TargetSpec,
    base: BenchConfig,
) -> list[tuple[ValidatedAction, BenchConfig]]:
    """The tuner's deterministic, BOUNDED grid off ``base``.

    For each capability-legal action, for each schema-legal param combo, produce
    ``(ValidatedAction, apply_to_config(action, base))`` in a stable order.
    Bounding (keeps ``|grid|`` ~= 15-20 so a budget-20 sweep is ~exhaustive):

    * ``threads`` samples ``n_threads`` at ``{cores, cores//2, cores//4}`` rather
      than every integer ``1..cores``;
    * ``cpu_mask`` symbolic values are resolved to concrete hex masks here (the
      ``TargetSpec`` is in scope), so ``apply_to_config`` only ever writes a
      concrete mask;
    * ``kleidiai`` emits ``sme=1`` only when ``sme2`` is a target capability.

    Candidates compose onto ``base``, so passing the CURRENT incumbent makes kept
    levers STACK. A combo whose resulting ``config_id`` equals ``base`` (a pure
    no-op, e.g. re-selecting the baseline quant) is dropped.
    """
    candidates: list[tuple[ValidatedAction, BenchConfig]] = []
    for action in registry:
        if not capabilities_ok(action, target):
            continue
        for raw_params in _param_combos(action, target):
            validated = validate_suggestion(action.id, raw_params)
            validated = _resolve_target_params(validated, target)
            cand_cfg = apply_to_config(validated, base)
            if cand_cfg.config_id == base.config_id:
                continue  # no-op relative to the incumbent
            candidates.append((validated, cand_cfg))
    return candidates


def _param_combos(action: ActionSpec, target: TargetSpec) -> list[dict[str, object]]:
    """Stable, bounded list of raw param dicts for ``action`` (values pre-validation)."""
    names = list(action.params_schema.keys())
    value_lists = [_param_values(action, name, target) for name in names]
    combos: list[dict[str, object]] = []
    for values in _product(value_lists):
        combos.append(dict(zip(names, values)))
    return combos


def _param_values(action: ActionSpec, name: str, target: TargetSpec) -> list[object]:
    """The bounded set of candidate values for one param (schema order preserved)."""
    pspec = action.params_schema[name]
    if pspec.type == "int":
        if action.id == "threads" and name == "n_threads":
            cores = target.n_physical_cores
            sampled = {c for c in (cores, cores // 2, cores // 4) if c >= 1}
            return sorted(sampled, reverse=True)  # type: ignore[return-value]
        bounds = [b for b in (pspec.min, pspec.max) if b is not None]
        return list(bounds)  # type: ignore[return-value]
    # enum
    choices = list(pspec.choices)
    if action.id == "kleidiai" and name == "sme":
        return [c for c in choices if c == "0" or "sme2" in target.capabilities]
    return choices  # type: ignore[return-value]


def _product(value_lists: Sequence[Sequence[object]]) -> list[list[object]]:
    """Deterministic cartesian product (avoids importing itertools for one call)."""
    result: list[list[object]] = [[]]
    for values in value_lists:
        result = [prefix + [v] for prefix in result for v in values]
    return result


def _resolve_target_params(
    action: ValidatedAction, target: TargetSpec
) -> ValidatedAction:
    """Resolve target-derived symbolic params (post-validation, deterministic).

    ``cpu_mask="physical"`` becomes the concrete hex mask covering every physical
    core, so the queued candidate carries a concrete mask ``apply_to_config`` can
    write verbatim. ``"default"`` is left symbolic (``apply_to_config`` maps it to
    ``None``). This runs only on values that already passed the safety gate.
    """
    if action.action_id != "threads":
        return action
    mask = action.params.get("cpu_mask")
    if mask != "physical":
        return action
    resolved = dict(action.params)
    resolved["cpu_mask"] = _hex_mask(target.n_physical_cores)
    return ValidatedAction(action_id=action.action_id, params=resolved)


def _hex_mask(n_cores: int) -> str:
    """Affinity mask with the low ``n_cores`` bits set, e.g. 16 -> ``0xffff``."""
    return hex((1 << max(n_cores, 0)) - 1)


# --------------------------------------------------------------------------------------
# Brain-queue shaping (reorder / hoist) -- only ever permutes the tuner's OWN queue
# --------------------------------------------------------------------------------------
def reorder_by_priority(
    queue: Sequence[tuple[ValidatedAction, BenchConfig]],
    priority: Sequence[str],
) -> list[tuple[ValidatedAction, BenchConfig]]:
    """Stable reorder: ids named in ``priority`` move to the front in that order;
    every other candidate keeps its deterministic relative order. The brain can
    only permute the queue -- it can never add an off-registry candidate."""
    rank = {action_id: i for i, action_id in enumerate(priority)}
    default = len(priority)
    return sorted(queue, key=lambda item: rank.get(item[0].action_id, default))


def hoist_suggested(
    queue: Sequence[tuple[ValidatedAction, BenchConfig]],
    suggestion: ValidatedAction | None,
) -> list[tuple[ValidatedAction, BenchConfig]]:
    """Move the single queued candidate whose ``(action_id, params)`` equals
    ``suggestion`` to the very front (a no-op if nothing matches). Honors the
    analyst's concrete pick while keeping it validated and tuner-generated."""
    items = list(queue)
    if suggestion is None:
        return items
    target_params = dict(suggestion.params)
    for i, (validated, _cfg) in enumerate(items):
        if (
            validated.action_id == suggestion.action_id
            and dict(validated.params) == target_params
        ):
            return [items[i]] + items[:i] + items[i + 1 :]
    return items


# --------------------------------------------------------------------------------------
# Keep/revert predicates
# --------------------------------------------------------------------------------------
def quality_ok(
    cand: QualityScore,
    base: QualityScore | None,
    threshold_pct: float,
    kl_max: float,
    *,
    changed_quant: bool,
) -> bool:
    """Reject a speed win that costs quality.

    * If ``cand.kl_vs_baseline`` is measured -> keep iff ``kl <= kl_max``.
    * Else, if a perplexity pair is available -> keep iff the perplexity rise vs
      ``base`` is ``<= threshold_pct``.
    * If quality is unmeasured on BOTH -> unsafe (``False``) when a build lever
      changed the quant (``changed_quant``), safe (``True``) on pure-runtime levers.

    Purely numeric plus the one boolean; the agent resolves ``changed_quant`` as
    ``'quant' in ACTIONS[action_id].sets``.
    """
    if cand.kl_vs_baseline is not None:
        return cand.kl_vs_baseline <= kl_max
    if cand.perplexity is not None and base is not None and base.perplexity is not None:
        rise_pct = (cand.perplexity - base.perplexity) / base.perplexity * 100.0
        return rise_pct <= threshold_pct
    return not changed_quant


def gap(
    winning: BenchmarkResult,
    baseline: BenchmarkResult,
    expert: BenchmarkResult | None,
) -> float | None:
    """``gap_closed_pct`` on decode median = ``100*(winning-baseline)/(expert-baseline)``.

    Returns ``None`` when ``expert`` is ``None`` or ``(expert-baseline) <= eps`` (a
    degenerate / weak expert decoding at ~baseline would otherwise divide by zero).
    """
    if expert is None:
        return None
    denom = expert.decode_tok_s.median - baseline.decode_tok_s.median
    if denom <= _GAP_EPS:
        return None
    return 100.0 * (winning.decode_tok_s.median - baseline.decode_tok_s.median) / denom


# --------------------------------------------------------------------------------------
# The discovery loop
# --------------------------------------------------------------------------------------
def optimize(
    target: "Target",
    profiler: "Profiler",
    brain: "Brain",
    benchmarker: "Benchmarker",
    *,
    manifest: RunManifest,
    baseline: BenchmarkResult,
    baseline_cfg: BenchConfig,
    expert: BenchmarkResult | None,
    trajectory_dir: Path,
    registry: Sequence[ActionSpec] = REGISTRY,
    budget: int = 20,
    screen_gate_pct: float = 2.0,
    quality_threshold_pct: float = 1.0,
    kl_max: float = 0.10,
    expert_cfg: BenchConfig | None = None,
) -> Recipe:
    """Greedy coordinate-ascent tuner (CONTRACTS.md section 9).

    Each iteration: read a Performix snapshot on the incumbent, let the brain
    reorder / hoist the tuner's queue, pop the front candidate, cheap-screen it,
    and -- only if it clears the screen gate -- run an interleaved A/B/A/B confirm
    against the incumbent. Keep iff the decode gain is CI-significant AND quality
    is within threshold, then recompose the queue onto the NEW incumbent so kept
    levers stack. One ``TrajectoryStep`` is appended per evaluated candidate; the
    winner is written to ``recipe.json``.

    ``expert_cfg`` supplies ``Recipe.expert_config`` (the pre-registered expert
    ``BenchConfig``); it is separate from ``expert`` (that config's measured
    ``BenchmarkResult``, used only for ``gap``). It is keyword-optional so callers
    that never pinned an expert still work -- ``Recipe.expert_config`` is then
    ``None``, matching CONTRACTS.md section 3.8.
    """
    trajectory_dir = Path(trajectory_dir)
    (trajectory_dir / "configs").mkdir(parents=True, exist_ok=True)
    (trajectory_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    traj_path = trajectory_dir / "trajectory.jsonl"

    incumbent_cfg = baseline_cfg
    incumbent_res = baseline
    best = baseline
    applied: set[str] = set()
    history: list[TrajectoryStep] = []

    def fresh_queue(cfg: BenchConfig) -> list[tuple[ValidatedAction, BenchConfig]]:
        return [
            (validated, cand)
            for (validated, cand) in enumerate_candidates(
                registry, manifest.target, cfg
            )
            if validated.action_id not in applied
        ]

    queue = fresh_queue(incumbent_cfg)
    for step_idx in range(budget):
        if not queue:
            break

        before_cfg = incumbent_cfg
        # EXACT benched argv, so the profiled command is provably the benched command.
        workload_cmd = shlex.join(target.bench_command(before_cfg))
        try:
            snapshot = profiler.snapshot(
                manifest.target, before_cfg, workload_cmd=workload_cmd
            )
        except ProfilerError as exc:
            # CONTRACTS §12 rule 2: Performix is a degradable optional layer. A
            # transport/parse failure must NOT crash the run -- narrate honestly
            # and continue try-measure-keep-best with no counter evidence this step.
            logger.warning(
                "profiler snapshot failed (%s); degrading to an empty snapshot", exc
            )
            snapshot = PerformixSnapshot(
                config_id=before_cfg.config_id,
                recipe="code_hotspots",
                source="cli",
                status="error",
                hotspots=(),
            )
        diagnosis = _diagnose(snapshot)

        candidate_ids = tuple(
            dict.fromkeys(validated.action_id for (validated, _c) in queue)
        )
        verdict = brain.analyze(
            Evidence(
                snapshot=snapshot,
                history=tuple(history),
                candidates=candidate_ids,
                baseline=baseline,
                current=incumbent_res,
            )
        )
        queue = reorder_by_priority(queue, verdict.priority)
        queue = hoist_suggested(queue, verdict.suggestion)

        action, cand_cfg = queue.pop(0)
        _write_json(trajectory_dir / "configs" / f"{cand_cfg.config_id}.json", cand_cfg)
        _write_json(
            trajectory_dir / "snapshots" / f"{before_cfg.config_id}.json", snapshot
        )

        # Build the candidate BEFORE benching it (CONTRACTS §6): a build-lever
        # candidate (native / kleidiai / quant) lives in a build-<build_key> dir
        # that does not exist yet; without this, screen -> run_bench executes a
        # missing binary and aborts the run. Idempotent + cached by build_key, so
        # a runtime-only candidate reuses the incumbent's build dir for free.
        target.build(cand_cfg)
        screen = benchmarker.screen(cand_cfg)
        if not promote_to_confirm(screen, incumbent_res, gate_pct=screen_gate_pct):
            step = _make_step(
                step_idx=step_idx,
                diagnosis=diagnosis,
                action=action,
                rationale=verdict.rationale,
                before_config_id=before_cfg.config_id,
                after_config_id=cand_cfg.config_id,
                screen=screen,
                confirm=None,
                kept=False,
                quality_ok=False,
                delta=_delta(screen.decode_tok_s, incumbent_res.decode_tok_s, False),
            )
            history.append(step)
            append_jsonl(traj_path, step)
            logger.info(
                "step %d: %s screened below gate (%.1f%%), reverted",
                step_idx,
                action.action_id,
                step.delta.pct,
            )
            continue

        inc_fresh, cand_fresh = benchmarker.confirm_ab(before_cfg, cand_cfg)
        faster = significant(
            cand_fresh.decode_tok_s, inc_fresh.decode_tok_s, higher_is_better=True
        )
        changed_quant = "quant" in ACTIONS[action.action_id].sets
        q_ok = quality_ok(
            cand_fresh.quality,
            baseline.quality,
            quality_threshold_pct,
            kl_max,
            changed_quant=changed_quant,
        )
        kept = faster and q_ok
        delta = _delta(cand_fresh.decode_tok_s, inc_fresh.decode_tok_s, faster)

        if kept:
            incumbent_cfg = cand_cfg
            incumbent_res = cand_fresh
            best = cand_fresh
            applied.add(action.action_id)
            queue = fresh_queue(
                incumbent_cfg
            )  # ascent: recompose onto the new incumbent

        step = _make_step(
            step_idx=step_idx,
            diagnosis=diagnosis,
            action=action,
            rationale=verdict.rationale,
            before_config_id=before_cfg.config_id,
            after_config_id=cand_cfg.config_id,
            screen=screen,
            confirm=cand_fresh,
            kept=kept,
            quality_ok=q_ok,
            delta=delta,
        )
        history.append(step)
        append_jsonl(traj_path, step)
        logger.info(
            "step %d: %s %s (decode %+.1f%%, ci_significant=%s, quality_ok=%s)",
            step_idx,
            action.action_id,
            "KEPT" if kept else "reverted",
            delta.pct,
            faster,
            q_ok,
        )

    recipe = Recipe(
        run_id=manifest.run_id,
        armsmith_version=manifest.armsmith_version,
        target_class=_target_class(manifest.target),
        model=manifest.model,
        winning_config=incumbent_cfg,
        baseline_config=baseline_cfg,
        expert_config=expert_cfg,
        baseline_result=baseline,
        winning_result=best,
        gap_closed_pct=gap(best, baseline, expert),
        created_at=_now_iso(),
    )
    _write_json(trajectory_dir / "recipe.json", recipe)
    logger.info(
        "optimize complete: winner=%s decode=%.2f tok/s (baseline %.2f), gap_closed=%s",
        recipe.winning_config.config_id,
        best.decode_tok_s.median,
        baseline.decode_tok_s.median,
        recipe.gap_closed_pct,
    )
    return recipe


# --------------------------------------------------------------------------------------
# Deterministic replay (repro path -- NO brain, NO profiler)
# --------------------------------------------------------------------------------------
def replay(
    target: "Target",
    benchmarker: "Benchmarker",
    recipe: Recipe,
    *,
    tol_pct: float = 10.0,
) -> BenchmarkResult:
    """Rebuild ``recipe.winning_config`` and confirm-bench it on a FRESH instance,
    with no LLM in the loop. Assert the fresh decode median is within ``tol_pct``
    of ``recipe.winning_result``; raise ``ReproToleranceError`` otherwise (an
    unreplayable result is not a result -- CLAUDE.md rule 4). Returns the fresh
    result (the reproducibility metric)."""
    target.build(recipe.winning_config)
    fresh = benchmarker.confirm(recipe.winning_config)
    expected = recipe.winning_result.decode_tok_s.median
    got = fresh.decode_tok_s.median
    if expected <= 0.0:
        raise ReproToleranceError(
            f"recipe winning decode median is non-positive ({expected}); cannot replay"
        )
    drift_pct = abs(got - expected) / expected * 100.0
    if drift_pct > tol_pct:
        raise ReproToleranceError(
            f"replay decode {got:.2f} tok/s is {drift_pct:.1f}% from the recorded "
            f"{expected:.2f} tok/s (tolerance {tol_pct:.1f}%)"
        )
    logger.info(
        "replay OK: %.2f tok/s vs recorded %.2f (%.1f%% drift, tol %.1f%%)",
        got,
        expected,
        drift_pct,
        tol_pct,
    )
    return fresh


# --------------------------------------------------------------------------------------
# Internal helpers (pure, deterministic)
# --------------------------------------------------------------------------------------
def _diagnose(snapshot: PerformixSnapshot) -> Diagnosis:
    """Summarize a Performix snapshot for the trajectory.

    On the virtualized r8g the PMU counter fields are ``None`` and ggml kernels
    resolve as "Unknown symbol @ 0x...", so ``bottleneck`` is almost always
    ``"unknown"`` and the evidence narrates hotspot sampling honestly, making NO
    counter claim. Where a bare-metal capture provides counters, a coarse
    cache-miss-rate heuristic classifies memory- vs compute-bound.
    """
    if snapshot.cache_miss_rate is not None:
        bottleneck = (
            "memory-bound"
            if snapshot.cache_miss_rate >= _HIGH_MISS_RATE
            else "compute-bound"
        )
        evidence = f"cache_miss_rate={snapshot.cache_miss_rate:.3f}" + (
            f", ipc={snapshot.ipc:.2f}" if snapshot.ipc is not None else ""
        )
    elif snapshot.hotspots:
        top = snapshot.hotspots[0]
        bottleneck = "unknown"
        evidence = (
            f"top hotspot {top.symbol} at {top.self_pct:.1f}% "
            f"({len(snapshot.hotspots)} sampled; PMU counters unavailable)"
        )
    else:
        bottleneck = "unknown"
        evidence = (
            f"no hotspot samples (status={snapshot.status}); PMU counters unavailable"
        )
    return Diagnosis(bottleneck=bottleneck, evidence=evidence, source=snapshot.source)


def _delta(candidate, incumbent, ci_significant: bool) -> Delta:
    """Signed decode delta (%) of ``candidate`` vs ``incumbent`` medians."""
    base = incumbent.median
    pct = 0.0 if base == 0.0 else (candidate.median - base) / base * 100.0
    return Delta(metric="decode_tok_s", pct=pct, ci_significant=ci_significant)


def _make_step(
    *,
    step_idx: int,
    diagnosis: Diagnosis,
    action: ValidatedAction,
    rationale: str,
    before_config_id: str,
    after_config_id: str,
    screen: BenchmarkResult | None,
    confirm: BenchmarkResult | None,
    kept: bool,
    quality_ok: bool,
    delta: Delta,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_idx=step_idx,
        diagnosis=diagnosis,
        action_id=action.action_id,
        params=dict(action.params),
        rationale=rationale,
        before_config_id=before_config_id,
        after_config_id=after_config_id,
        screen=screen,
        confirm=confirm,
        kept=kept,
        delta=delta,
        quality_ok=quality_ok,
    )


def _target_class(target: TargetSpec) -> str:
    """Instance family the run was tuned for, e.g. ``"r8g.4xlarge"`` -> ``"r8g"``."""
    return target.instance_type.split(".", 1)[0]


def _now_iso() -> str:
    """Current time as an ISO-8601 UTC timestamp (``...Z``)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_json(path: Path, obj: object) -> None:
    """Write a frozen dataclass to ``path`` as canonical JSON (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(obj), encoding="utf-8")
