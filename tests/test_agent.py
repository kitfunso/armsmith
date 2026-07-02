"""Unit tests for armsmith.agent -- the discovery loop, keep/revert, trajectory, recipe, replay.

The loop is driven entirely by in-memory fakes of ``Target`` / ``Profiler`` / ``Brain`` plus a
scripted ``Benchmarker`` double, so every test runs offline with zero AWS spend (CLAUDE.md rule 1:
no non-Arm numbers are ever reported -- the scripted decode values here are transparent test
doubles, not measured results). Where a test needs a real llama-bench capture or a real Performix
hotspots payload it uses the committed ``tests/fixtures/`` JSONs (``bench-*.json`` medians ground
the screen gate; ``hotspots_portable.json`` grounds the diagnosis), never invented fixture data.
Paths are resolved with ``pathlib`` relative to this file so the suite runs identically on the
Windows host and Linux CI.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from armsmith.actions import (
    REGISTRY,
    apply_to_config,
    baseline_config,
    expert_config,
    validate_suggestion,
)
from armsmith.agent import (
    enumerate_candidates,
    gap,
    hoist_suggested,
    optimize,
    quality_ok,
    reorder_by_priority,
    replay,
)
from armsmith.bench import parse_llama_bench_output, promote_to_confirm, significant
from armsmith.brain import BrainVerdict
from armsmith.models import (
    BenchmarkResult,
    MCPError,
    MetricStat,
    ModelSpec,
    QualityScore,
    RawSamples,
    Recipe,
    ReproToleranceError,
    RunManifest,
    TargetSpec,
    WorkloadSpec,
    from_json,
    read_jsonl,
)
from armsmith.profiler import NullProfiler, parse_structured_content

FIXTURES = Path(__file__).parent / "fixtures"
_NO_QUALITY = QualityScore(perplexity=None, kl_vs_baseline=None)


# --------------------------------------------------------------------------------------
# Real-fixture ground truth (medians the tuner's screen gate keys off)
# --------------------------------------------------------------------------------------
def _parse_screen(name: str) -> BenchmarkResult:
    raw = (FIXTURES / name).read_text(encoding="utf-8")
    return parse_llama_bench_output(
        raw, config_id=name.replace(".json", ""), stage="screen", quality=_NO_QUALITY
    )


DECODE_BASELINE = _parse_screen("bench-portable.json").decode_tok_s.median  # ~21.81
DECODE_NATIVE = _parse_screen("bench-native.json").decode_tok_s.median  # ~44.96
_STACK_BONUS = (
    10.0  # synthetic 2nd-tier increment atop the REAL native median (Q8_0 lever)
)


# --------------------------------------------------------------------------------------
# Shared builders
# --------------------------------------------------------------------------------------
def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
        screen_repeats=3,
        confirm_repeats=5,
        eval_text_path="eval.txt",
    )


def _model() -> ModelSpec:
    return ModelSpec(
        name="Qwen2.5-7B-Instruct",
        variants={
            "Q4_0": ("~/m/qwen-q4_0.gguf", "sha-q4_0"),
            "Q8_0": ("~/m/qwen-q8_0.gguf", "sha-q8_0"),
            "Q4_K_M": ("~/m/qwen-q4km.gguf", "sha-q4km"),
        },
        baseline_quant="Q4_0",
    )


def _target(*, capabilities=("sve2", "bf16", "i8mm"), cores=16) -> TargetSpec:
    return TargetSpec(
        host="1.2.3.4",
        user="ubuntu",
        instance_type="r8g.4xlarge",
        core="Neoverse V2 (Graviton4)",
        region="eu-west-2",
        kernel="6.8.0-1015-aws",
        cpu_governor="performance",
        n_physical_cores=cores,
        capabilities=capabilities,
    )


def _manifest(target: TargetSpec, model: ModelSpec, baseline_ref: str) -> RunManifest:
    return RunManifest(
        run_id="2026-07-05T14-03-11Z-r8g",
        target=target,
        model=model,
        workload_ref="examples/bench.yaml",
        baseline_ref=baseline_ref,
        expert_ref="expert",
        created_at="2026-07-05T14:03:11Z",
        armsmith_version="0.0.1",
    )


def _result(
    config_id: str,
    decode_median: float,
    *,
    quality: QualityScore = _NO_QUALITY,
    stage: str = "confirm",
    n_repeats: int = 5,
    ci_frac: float = 0.005,
    prefill_median: float = 17552.0,
) -> BenchmarkResult:
    """A BenchmarkResult with a controlled decode median + tight symmetric CI. Tight CIs make
    well-separated medians CI-significant and near-equal medians (native vs generic) overlap,
    matching the real fixtures' significance structure."""
    d_lo, d_hi = decode_median * (1 - ci_frac), decode_median * (1 + ci_frac)
    p_lo, p_hi = prefill_median * (1 - ci_frac), prefill_median * (1 + ci_frac)
    return BenchmarkResult(
        config_id=config_id,
        decode_tok_s=MetricStat(median=decode_median, ci_low=d_lo, ci_high=d_hi),
        prefill_ttft_ms=MetricStat(median=prefill_median, ci_low=p_lo, ci_high=p_hi),
        quality=quality,
        peak_mem_mb=1000.0,
        model_size_mb=4232.5,
        n_repeats=n_repeats,
        stage=stage,
        raw_samples=RawSamples(
            decode_tok_s=tuple([decode_median] * n_repeats),
            prefill_ttft_ms=tuple([prefill_median] * n_repeats),
        ),
    )


# --------------------------------------------------------------------------------------
# In-memory fakes of Target / Profiler / Brain / Benchmarker
# --------------------------------------------------------------------------------------
class FakeTarget:
    """Only the surface agent.optimize / agent.replay touch: bench_command (pure argv) + build."""

    def __init__(self) -> None:
        self.built: list[str] = []

    def bench_command(self, cfg) -> tuple[str, ...]:
        return ("llama-bench", "-m", cfg.quant, "-t", str(cfg.n_threads))

    def build(self, cfg) -> str:
        self.built.append(cfg.config_id)
        return f"build-{cfg.config_id}"


class FakeProfiler:
    """Returns a fixed real snapshot (config_id rebound to the incumbent) for diagnosis tests."""

    def __init__(self, snapshot) -> None:
        self._snap = snapshot
        self.calls: list[tuple[str, str]] = []

    def snapshot(self, target, cfg, *, workload_cmd, recipe="code_hotspots"):
        self.calls.append((cfg.config_id, workload_cmd))
        return replace(self._snap, config_id=cfg.config_id)


class RaisingProfiler:
    """Simulates a Performix transport/parse failure on every snapshot -- the loop
    must degrade (CONTRACTS §12 rule 2), not crash."""

    def snapshot(self, target, cfg, *, workload_cmd, recipe="code_hotspots"):
        raise MCPError("simulated transport failure")


class ScriptedBenchmarker:
    """A Benchmarker double scripted by two pure callables over a BenchConfig: ``decode_for`` (the
    decode tok/s a config would measure) and ``quality_for`` (its QualityScore). Records calls so
    tests can assert the screen gate skipped confirm_ab (cost control)."""

    def __init__(self, decode_for, quality_for, *, n_repeats: int = 5) -> None:
        self._decode_for = decode_for
        self._quality_for = quality_for
        self._n = n_repeats
        self.screen_calls: list[str] = []
        self.confirm_ab_calls: list[tuple[str, str]] = []
        self.confirm_calls: list[str] = []

    def screen(self, cfg) -> BenchmarkResult:
        self.screen_calls.append(cfg.config_id)
        return _result(
            cfg.config_id, self._decode_for(cfg), stage="screen", n_repeats=3
        )

    def confirm_ab(self, incumbent_cfg, candidate_cfg):
        self.confirm_ab_calls.append((incumbent_cfg.config_id, candidate_cfg.config_id))
        inc = _result(
            incumbent_cfg.config_id,
            self._decode_for(incumbent_cfg),
            quality=self._quality_for(incumbent_cfg),
            n_repeats=self._n,
        )
        cand = _result(
            candidate_cfg.config_id,
            self._decode_for(candidate_cfg),
            quality=self._quality_for(candidate_cfg),
            n_repeats=self._n,
        )
        return inc, cand

    def confirm(self, cfg) -> BenchmarkResult:
        self.confirm_calls.append(cfg.config_id)
        return _result(
            cfg.config_id,
            self._decode_for(cfg),
            quality=self._quality_for(cfg),
            n_repeats=self._n,
        )


class PassthroughBrain:
    """No reordering (empty priority) and no suggestion -- the tuner's own deterministic order."""

    def analyze(self, evidence) -> BrainVerdict:
        return BrainVerdict(priority=(), rationale="passthrough", suggestion=None)


class ReversedBrain:
    """Reverses the candidate action-id order every step (a different search path than the tuner)."""

    def analyze(self, evidence) -> BrainVerdict:
        return BrainVerdict(
            priority=tuple(reversed(evidence.candidates)),
            rationale="reversed",
            suggestion=None,
        )


class SuggestQuantBrain:
    """Hoists a concrete validated suggestion (quant_format -> Q8_0) to the front of the queue."""

    def analyze(self, evidence) -> BrainVerdict:
        return BrainVerdict(
            priority=(),
            rationale="try q8_0 first",
            suggestion=validate_suggestion("quant_format", {"quant": "Q8_0"}),
        )


# --------------------------------------------------------------------------------------
# decode_for / quality_for scripts (grounded in the REAL fixture medians)
# --------------------------------------------------------------------------------------
def _native_only_decode(cfg) -> float:
    return DECODE_NATIVE if _is_native(cfg) else DECODE_BASELINE


def _stacking_decode(cfg) -> float:
    """Two independent levers that STACK: GGML_NATIVE=ON and the Q8_0 quant swap."""
    d = DECODE_NATIVE if _is_native(cfg) else DECODE_BASELINE
    if cfg.quant == "Q8_0":
        d += _STACK_BONUS
    return d


def _is_native(cfg) -> bool:
    return any("GGML_NATIVE=ON" in flag for flag in cfg.cmake_flags)


def _quality_good(cfg) -> QualityScore:
    """Quant swaps report a measured, in-threshold KL; runtime/native levers leave it unmeasured."""
    if cfg.quant == "Q8_0":
        return QualityScore(perplexity=8.0, kl_vs_baseline=0.05)
    return QualityScore(perplexity=None, kl_vs_baseline=None)


def _baseline_pieces():
    target, model, workload = _target(), _model(), _workload()
    base_cfg = baseline_config(workload, model)
    baseline = _result(
        base_cfg.config_id,
        DECODE_BASELINE,
        quality=QualityScore(perplexity=8.0, kl_vs_baseline=None),
    )
    manifest = _manifest(target, model, base_cfg.config_id)
    return target, model, workload, base_cfg, baseline, manifest


# ======================================================================================
# enumerate_candidates -- the bounded grid
# ======================================================================================
def test_enumerate_bounded_grid_and_target_resolution():
    target, model, workload = _target(), _model(), _workload()
    base = baseline_config(workload, model)
    cands = enumerate_candidates(REGISTRY, target, base)

    registry_ids = {spec.id for spec in REGISTRY}
    assert all(va.action_id in registry_ids for va, _ in cands)
    # Bounded to ~15-20 so a budget-20 sweep is ~exhaustive (CONTRACTS section 9).
    assert 15 <= len(cands) <= 20
    # No candidate is a no-op relative to the incumbent.
    assert all(cfg.config_id != base.config_id for _, cfg in cands)

    # threads is SAMPLED at {cores, cores//2, cores//4}, not every integer.
    thread_counts = {
        va.params["n_threads"] for va, _ in cands if va.action_id == "threads"
    }
    assert thread_counts == {16, 8, 4}
    # cpu_mask "physical" is resolved to a concrete hex mask here (0xffff for 16 cores).
    masks = {va.params["cpu_mask"] for va, _ in cands if va.action_id == "threads"}
    assert "physical" not in masks
    assert "0xffff" in masks
    assert "default" in masks

    # sme=1 is emitted ONLY when sme2 is a capability; Graviton4 (no sme2) never offers it.
    assert all(
        va.params.get("sme") != "1" for va, _ in cands if va.action_id == "kleidiai"
    )


def test_enumerate_kleidiai_filtered_without_i8mm():
    target = _target(
        capabilities=("sve2", "bf16")
    )  # no i8mm -> kleidiai precondition fails
    base = baseline_config(_workload(), _model())
    cands = enumerate_candidates(REGISTRY, target, base)
    assert all(va.action_id != "kleidiai" for va, _ in cands)


def test_enumerate_filters_quants_to_available_variants():
    """The model registry is the source of truth for which GGUF variants exist:
    a quant candidate the ModelSpec does not pin must never be enumerated
    (observed live: ModelSpec.resolve raised KeyError('Q8_0') mid-optimize when
    the registry pinned only Q4_0)."""
    target, model, workload = _target(), _model(), _workload()
    base = baseline_config(workload, model)

    cands = enumerate_candidates(REGISTRY, target, base, available_quants=[base.quant])

    # Q8_0/Q4_K_M are filtered out; the pinned quant equals the baseline's, so
    # its candidate is a no-op and is dropped too -> no quant_format candidates.
    assert all(va.action_id != "quant_format" for va, _ in cands)
    # Other levers are untouched by the filter.
    assert any(va.action_id == "ggml_native" for va, _ in cands)


def test_enumerate_never_emits_engine_illegal_kv_combos():
    """Every enumerated kv_cache_type candidate with a quantized V-cache must
    carry flash_attn=True; the validation gate rejects the illegal combo and
    the enumerator skips it (observed live: `-ctv q8_0 -fa 0` crashed the
    optimize run with "failed to create context")."""
    target, model, workload = _target(), _model(), _workload()
    base = baseline_config(workload, model)

    cands = enumerate_candidates(REGISTRY, target, base)

    kv_cands = [cfg for va, cfg in cands if va.action_id == "kv_cache_type"]
    assert kv_cands  # the lever still contributes legal candidates
    for cfg in kv_cands:
        if cfg.type_v != "f16":
            assert cfg.flash_attn is True


def test_enumerate_composes_onto_incumbent_so_levers_stack():
    target, model, workload = _target(), _model(), _workload()
    base = baseline_config(workload, model)
    native = apply_to_config(validate_suggestion("ggml_native", {"state": "ON"}), base)
    cands = enumerate_candidates(REGISTRY, target, native)
    # Every non-ggml_native candidate carries the incumbent's GGML_NATIVE=ON (coordinate ascent).
    for va, cfg in cands:
        if va.action_id != "ggml_native":
            assert any("GGML_NATIVE=ON" in flag for flag in cfg.cmake_flags)


# ======================================================================================
# reorder_by_priority / hoist_suggested -- brain only permutes the tuner's OWN queue
# ======================================================================================
def test_reorder_by_priority_is_stable_and_fronts_named_ids():
    target, base = _target(), baseline_config(_workload(), _model())
    queue = enumerate_candidates(REGISTRY, target, base)
    reordered = reorder_by_priority(queue, ("kv_cache_type", "threads"))
    ids = [va.action_id for va, _ in reordered]
    # kv_cache_type block first, then threads, then everyone else in original order.
    assert ids[0] == "kv_cache_type"
    assert set(ids) == {va.action_id for va, _ in queue}  # nothing added/dropped
    first_other = next(
        i for i, a in enumerate(ids) if a not in ("kv_cache_type", "threads")
    )
    assert all(a in ("kv_cache_type", "threads") for a in ids[:first_other])


def test_reorder_empty_priority_is_identity():
    target, base = _target(), baseline_config(_workload(), _model())
    queue = enumerate_candidates(REGISTRY, target, base)
    assert reorder_by_priority(queue, ()) == list(queue)


def test_hoist_suggested_matches_and_is_noop_when_absent():
    target, base = _target(), baseline_config(_workload(), _model())
    queue = enumerate_candidates(REGISTRY, target, base)
    suggestion = validate_suggestion("quant_format", {"quant": "Q8_0"})
    hoisted = hoist_suggested(queue, suggestion)
    front_va, _ = hoisted[0]
    assert front_va.action_id == "quant_format"
    assert dict(front_va.params) == {"quant": "Q8_0"}
    # A permutation: same elements, only reordered (ValidatedAction holds a dict, so no sets).
    assert len(hoisted) == len(queue) and all(item in queue for item in hoisted)

    # A suggestion matching no queued candidate is a no-op.
    no_match = validate_suggestion(
        "ggml_native", {"state": "OFF"}
    )  # OFF == baseline, not queued
    assert hoist_suggested(queue, no_match) == list(queue)
    assert hoist_suggested(queue, None) == list(queue)


# ======================================================================================
# quality_ok / gap -- keep/revert predicates
# ======================================================================================
def test_quality_ok_rules():
    base = QualityScore(perplexity=8.0, kl_vs_baseline=None)
    # KL measured: within / over kl_max.
    assert (
        quality_ok(QualityScore(None, 0.05), base, 1.0, 0.10, changed_quant=True)
        is True
    )
    assert (
        quality_ok(QualityScore(None, 0.50), base, 1.0, 0.10, changed_quant=True)
        is False
    )
    # Perplexity rise within / over threshold.
    assert (
        quality_ok(QualityScore(8.05, None), base, 1.0, 0.10, changed_quant=False)
        is True
    )
    assert (
        quality_ok(QualityScore(9.00, None), base, 1.0, 0.10, changed_quant=False)
        is False
    )
    # Both unmeasured: unsafe iff a build lever changed the quant.
    assert quality_ok(_NO_QUALITY, base, 1.0, 0.10, changed_quant=True) is False
    assert quality_ok(_NO_QUALITY, base, 1.0, 0.10, changed_quant=False) is True


def test_gap_closed_pct():
    baseline = _result("b", 20.0)
    winning = _result("w", 40.0)
    expert = _result("e", 45.0)
    assert gap(winning, baseline, expert) == pytest.approx(100 * (40 - 20) / (45 - 20))
    assert gap(winning, baseline, None) is None  # no expert pinned
    # Degenerate/weak expert (decodes ~= baseline) -> undefined, not a divide-by-zero.
    assert gap(winning, baseline, _result("e", 20.0)) is None


# ======================================================================================
# optimize -- keep / revert / stacking / cost-control
# ======================================================================================
def test_keep_on_ci_significant_speedup_with_good_quality(tmp_path):
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(_native_only_decode, _quality_good)
    expert = _result("expert", 50.0)
    expert_cfg = expert_config(workload, model, target)

    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=expert,
        trajectory_dir=tmp_path,
        expert_cfg=expert_cfg,
        budget=20,
    )

    # The GGML_NATIVE=ON lever is CI-significantly faster with safe quality -> kept and stacked in.
    assert any("GGML_NATIVE=ON" in f for f in recipe.winning_config.cmake_flags)
    assert recipe.winning_result.decode_tok_s.median == pytest.approx(DECODE_NATIVE)
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    kept = [s for s in steps if s.kept]
    assert [s.action_id for s in kept] == ["ggml_native"]
    assert kept[0].quality_ok is True
    assert kept[0].delta.ci_significant is True
    assert kept[0].confirm is not None and kept[0].confirm.stage == "confirm"


def test_two_levers_stack_via_coordinate_ascent(tmp_path):
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(_stacking_decode, _quality_good)

    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )

    win = recipe.winning_config
    assert any("GGML_NATIVE=ON" in f for f in win.cmake_flags)  # lever 1
    assert win.quant == "Q8_0"  # lever 2 stacked on top
    assert recipe.winning_result.decode_tok_s.median == pytest.approx(
        DECODE_NATIVE + _STACK_BONUS
    )
    kept = [s.action_id for s in read_jsonl(tmp_path / "trajectory.jsonl") if s.kept]
    assert sorted(kept) == ["ggml_native", "quant_format"]  # multiple levers, not one


def test_quality_regressing_speedup_is_rejected(tmp_path):
    """A config that is CI-significantly FASTER but regresses quality past threshold is reverted."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()

    def bad_quality(cfg) -> QualityScore:
        return QualityScore(perplexity=None, kl_vs_baseline=0.50)  # 0.50 > kl_max 0.10

    bench = ScriptedBenchmarker(_native_only_decode, bad_quality)
    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )

    # Winner stays the baseline; the fast-but-degraded native config was NOT kept.
    assert recipe.winning_config.config_id == base_cfg.config_id
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    native_steps = [s for s in steps if s.action_id == "ggml_native"]
    assert native_steps and all(not s.kept for s in native_steps)
    native = native_steps[0]
    assert native.delta.ci_significant is True  # it WAS faster
    assert native.quality_ok is False  # but the quality guard bit
    assert not any(s.kept for s in steps)


def test_quant_change_with_unmeasured_quality_is_rejected(tmp_path):
    """A quant-changing build lever with quality unmeasured on both metrics is unsafe (changed_quant)."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    # Q8_0 is faster but reports NO quality at all -> must be rejected because it changed the quant.
    bench = ScriptedBenchmarker(_stacking_decode, lambda cfg: _NO_QUALITY)
    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    quant_steps = [s for s in steps if s.action_id == "quant_format"]
    assert quant_steps and all(not s.kept for s in quant_steps)
    assert all(s.quality_ok is False for s in quant_steps if s.confirm is not None)
    assert recipe.winning_config.quant != "Q8_0"


def test_optimize_builds_each_candidate_before_screening(tmp_path):
    """Finding 4 regression: a build-lever candidate lives in a build-<build_key>
    dir that must be compiled before it is benched. optimize must call
    target.build on every candidate it screens, else run_bench executes a missing
    binary and aborts the run."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(_native_only_decode, _quality_good)
    tgt = FakeTarget()

    optimize(
        tgt,
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )

    assert bench.screen_calls  # candidates WERE screened
    # Every screened candidate was built first (build is idempotent, cached by build_key).
    assert set(bench.screen_calls) <= set(tgt.built)


def test_optimize_degrades_when_profiler_raises(tmp_path):
    """Finding 7 regression: a Performix transport/parse failure must NOT crash the
    run (CONTRACTS §12 rule 2). The loop degrades to an empty snapshot and keeps
    going -- a recipe is still written and levers are still kept."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(_native_only_decode, _quality_good)

    recipe = optimize(
        FakeTarget(),
        RaisingProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )

    assert (tmp_path / "recipe.json").exists()  # completed, did not crash
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    assert steps and all(s.diagnosis.bottleneck == "unknown" for s in steps)
    assert any("GGML_NATIVE=ON" in f for f in recipe.winning_config.cmake_flags)


def test_screen_gate_failure_skips_confirm_ab(tmp_path):
    """Cost control: a candidate that fails the cheap screen gate never triggers an A/B confirm."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    # Nothing beats the baseline decode -> every candidate is gated out before confirm.
    bench = ScriptedBenchmarker(lambda cfg: DECODE_BASELINE, _quality_good)
    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )
    assert bench.screen_calls  # candidates WERE screened
    assert bench.confirm_ab_calls == []  # but never confirmed
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    assert steps and all(not s.kept and s.confirm is None for s in steps)
    assert recipe.winning_config.config_id == base_cfg.config_id


def test_screen_gate_grounded_in_real_fixture_medians():
    """The gate the loop relies on, on REAL captures: portable->native clears +2%, native->generic
    does not (the spike's 'GGML_NATIVE already ON in vanilla cmake' finding)."""
    portable = _parse_screen("bench-portable.json")
    native = _parse_screen("bench-native.json")
    generic = _parse_screen("bench-generic.json")
    assert promote_to_confirm(native, portable, gate_pct=2.0) is True
    assert promote_to_confirm(generic, native, gate_pct=2.0) is False


def test_suggestion_is_hoisted_and_evaluated_first(tmp_path):
    """The analyst's concrete suggestion is honored: the Q8_0 candidate is confirmed before native."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(_stacking_decode, _quality_good)
    optimize(
        FakeTarget(),
        NullProfiler(),
        SuggestQuantBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )
    # First candidate to reach the (expensive) confirm stage is the hoisted quant swap.
    first_confirmed_cand = bench.confirm_ab_calls[0][1]
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    first_step = steps[0]
    assert first_step.action_id == "quant_format"
    assert first_step.after_config_id == first_confirmed_cand


# ======================================================================================
# Honesty invariant -- NullBrain-order and a reordered brain reach the same optimum
# ======================================================================================
def test_brain_reorder_does_not_change_reachable_optimum(tmp_path):
    """Removing/altering the brain's ordering must not change the reachable optimum (CONTRACTS
    section 9 budget-vs-grid: the brain changes search SPEED, not the CI-equivalent winner).
    """
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()

    def run(brain, sub):
        bench = ScriptedBenchmarker(_stacking_decode, _quality_good)
        return optimize(
            FakeTarget(),
            NullProfiler(),
            brain,
            bench,
            manifest=manifest,
            baseline=baseline,
            baseline_cfg=base_cfg,
            expert=None,
            trajectory_dir=tmp_path / sub,
            budget=20,
        )

    tuner_order = run(PassthroughBrain(), "tuner")
    reordered = run(ReversedBrain(), "reordered")

    assert tuner_order.winning_config.config_id == reordered.winning_config.config_id
    a, b = (
        tuner_order.winning_result.decode_tok_s,
        reordered.winning_result.decode_tok_s,
    )
    assert not significant(a, b) and not significant(b, a)  # within CI of each other


# ======================================================================================
# Trajectory + recipe artifacts on disk
# ======================================================================================
def test_optimize_writes_trajectory_recipe_and_audit_artifacts(tmp_path):
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(_stacking_decode, _quality_good)
    expert = _result("expert", 60.0)

    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=expert,
        trajectory_dir=tmp_path,
        expert_cfg=expert_config(workload, model, target),
        budget=20,
    )

    # trajectory.jsonl: one parseable TrajectoryStep per screened candidate, indices in order.
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    assert len(steps) == len(bench.screen_calls)
    assert [s.step_idx for s in steps] == list(range(len(steps)))

    # recipe.json round-trips and carries the computed gap-closed %.
    recipe_disk = from_json(
        Recipe, (tmp_path / "recipe.json").read_text(encoding="utf-8")
    )
    assert recipe_disk.winning_config.config_id == recipe.winning_config.config_id
    assert recipe_disk.run_id == manifest.run_id
    assert recipe_disk.target_class == "r8g"
    assert recipe.gap_closed_pct == pytest.approx(
        100
        * (recipe.winning_result.decode_tok_s.median - DECODE_BASELINE)
        / (60.0 - DECODE_BASELINE)
    )

    # Per-candidate audit configs + per-step snapshots were written.
    assert list((tmp_path / "configs").glob("*.json"))
    assert list((tmp_path / "snapshots").glob("*.json"))


def test_optimize_rotates_stale_trajectory_from_prior_invocation(tmp_path):
    """A crashed or repeated optimize run must not leave its steps interleaved
    with this run's trajectory (observed live: a KeyError-aborted run left
    steps 0-2 duplicated). Prior data is rotated aside, never deleted."""
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    stale = tmp_path / "trajectory.jsonl"
    stale.write_text('{"stale": "partial step from crashed run"}\n', encoding="utf-8")

    bench = ScriptedBenchmarker(_stacking_decode, _quality_good)
    optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )

    rotated = tmp_path / "trajectory.jsonl.1"
    assert rotated.exists()
    assert "stale" in rotated.read_text(encoding="utf-8")
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    assert steps  # fresh file holds only this invocation's parseable steps
    assert [s.step_idx for s in steps] == list(range(len(steps)))


def test_budget_caps_iterations(tmp_path):
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(lambda cfg: DECODE_BASELINE, _quality_good)
    optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=3,
    )
    steps = read_jsonl(tmp_path / "trajectory.jsonl")
    assert len(steps) == 3  # never exceeds the budget


# ======================================================================================
# Diagnosis from a real Performix hotspots capture
# ======================================================================================
def test_diagnosis_reads_real_hotspots_snapshot(tmp_path):
    sc = json.loads((FIXTURES / "hotspots_portable.json").read_text(encoding="utf-8"))
    snap = parse_structured_content(sc, config_id="seed", source="mcp")
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    bench = ScriptedBenchmarker(lambda cfg: DECODE_BASELINE, _quality_good)

    optimize(
        FakeTarget(),
        FakeProfiler(snap),
        PassthroughBrain(),
        bench,
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=1,
    )
    step = read_jsonl(tmp_path / "trajectory.jsonl")[0]
    diag = step.diagnosis
    # r8g exposes only 2 PMU counters -> no counter claim; bottleneck stays "unknown".
    assert diag.bottleneck == "unknown"
    assert diag.source == "mcp"
    assert "Unknown symbol @ 0x00081dd0" in diag.evidence
    assert "37.5%" in diag.evidence


# ======================================================================================
# replay -- deterministic repro (no brain, no profiler)
# ======================================================================================
def test_replay_within_tolerance_passes_and_rebuilds(tmp_path):
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    recipe = optimize(
        FakeTarget(),
        NullProfiler(),
        PassthroughBrain(),
        ScriptedBenchmarker(_native_only_decode, _quality_good),
        manifest=manifest,
        baseline=baseline,
        baseline_cfg=base_cfg,
        expert=None,
        trajectory_dir=tmp_path,
        budget=20,
    )
    win_decode = recipe.winning_result.decode_tok_s.median

    tgt = FakeTarget()
    # Fresh instance lands within +8% (inside the ±10% tolerance).
    fresh_bench = ScriptedBenchmarker(lambda cfg: win_decode * 1.08, _quality_good)
    fresh = replay(tgt, fresh_bench, recipe, tol_pct=10.0)
    assert fresh.decode_tok_s.median == pytest.approx(win_decode * 1.08)
    assert tgt.built == [recipe.winning_config.config_id]  # rebuilt from the recipe
    assert fresh_bench.confirm_calls == [recipe.winning_config.config_id]


def test_replay_outside_tolerance_raises():
    target, model, workload, base_cfg, baseline, manifest = _baseline_pieces()
    winning = _result("win", 45.0)
    recipe = Recipe(
        run_id=manifest.run_id,
        armsmith_version="0.0.1",
        target_class="r8g",
        model=model,
        winning_config=apply_to_config(
            validate_suggestion("ggml_native", {"state": "ON"}), base_cfg
        ),
        baseline_config=base_cfg,
        expert_config=None,
        baseline_result=baseline,
        winning_result=winning,
        gap_closed_pct=None,
        created_at="2026-07-05T14:03:11Z",
    )
    # Fresh decode is 30% low -> outside ±10% -> ReproToleranceError (rule 4: not replayable = not a result).
    bad_bench = ScriptedBenchmarker(lambda cfg: 45.0 * 0.70, _quality_good)
    with pytest.raises(ReproToleranceError):
        replay(FakeTarget(), bad_bench, recipe, tol_pct=10.0)
