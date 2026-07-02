"""Tests for armsmith.report: self-contained HTML render (CONTRACTS.md §10).

Builds a synthetic run dir with valid `RunManifest`/`BenchmarkResult`/
`TrajectoryStep`/`Recipe` records (constructed via the real frozen
dataclasses in `armsmith.models`, not hand-rolled dicts) and asserts the
rendered `report.html` carries the required content and stays a portable,
standalone file (no http(s):// resource references, no external assets).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from armsmith.models import (
    BenchConfig,
    BenchmarkResult,
    Delta,
    Diagnosis,
    MetricStat,
    ModelSpec,
    QualityScore,
    RawSamples,
    Recipe,
    RunManifest,
    TargetSpec,
    TrajectoryStep,
    append_jsonl,
    to_json,
)
from armsmith.report import ReportModel, build_report_model, render_report

# --------------------------------------------------------------------------
# builders (mirrors tests/test_models.py fixture style)
# --------------------------------------------------------------------------


def _target_spec() -> TargetSpec:
    return TargetSpec(
        host="1.2.3.4",
        user="ubuntu",
        instance_type="r8g.4xlarge",
        core="Neoverse V2 (Graviton4)",
        region="eu-west-2",
        kernel="6.8.0-1015-aws",
        cpu_governor="performance",
        n_physical_cores=16,
        capabilities=("sve2", "bf16", "i8mm"),
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        name="Qwen2.5-7B-Instruct",
        variants={
            "Q4_0": ("~/m/qwen-q4_0.gguf", "abc123"),
            "Q8_0": ("~/m/qwen-q8_0.gguf", "def456"),
        },
        baseline_quant="Q4_0",
    )


def _bench_config(**overrides) -> BenchConfig:
    fields = dict(
        cmake_flags=("-DGGML_NATIVE=OFF",),
        quant="Q4_0",
        n_threads=16,
        cpu_mask=None,
        type_k="f16",
        type_v="f16",
        flash_attn=None,
        env=(),
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
    )
    fields.update(overrides)
    return BenchConfig.create(**fields)


def _metric_stat(median: float) -> MetricStat:
    return MetricStat(median=median, ci_low=median - 1, ci_high=median + 1)


def _raw_samples(n: int = 7, base: float = 21.0) -> RawSamples:
    return RawSamples(
        decode_tok_s=tuple(base + i * 0.1 for i in range(n)),
        prefill_ttft_ms=tuple(45.0 + i for i in range(n)),
    )


def _bench_result(
    cfg: BenchConfig,
    *,
    decode: float,
    prefill: float = 45.0,
    stage: str = "confirm",
    n_repeats: int = 7,
    quality: QualityScore | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        config_id=cfg.config_id,
        decode_tok_s=_metric_stat(decode),
        prefill_ttft_ms=_metric_stat(prefill),
        quality=quality or QualityScore(perplexity=6.12, kl_vs_baseline=None),
        peak_mem_mb=4300.5,
        model_size_mb=4233.1,
        n_repeats=n_repeats,
        stage=stage,
        raw_samples=_raw_samples(n_repeats, base=decode),
    )


def _manifest(baseline_cfg: BenchConfig, expert_cfg: BenchConfig) -> RunManifest:
    return RunManifest(
        run_id="2026-07-05T14-03-11Z-r8g",
        target=_target_spec(),
        model=_model_spec(),
        workload_ref="examples/bench.yaml",
        baseline_ref=baseline_cfg.config_id,
        expert_ref=expert_cfg.config_id,
        created_at="2026-07-05T14:03:11Z",
        armsmith_version="0.1.0",
    )


def _trajectory_step(
    step_idx: int,
    *,
    action_id: str,
    before_cfg: BenchConfig,
    after_cfg: BenchConfig,
    confirm: BenchmarkResult | None,
    kept: bool,
    pct: float,
    ci_significant: bool,
    quality_ok: bool = True,
    rationale: str = "kleidiai raised i8mm utilization on the hot matmul kernel",
    bottleneck: str = "compute-bound",
) -> TrajectoryStep:
    return TrajectoryStep(
        step_idx=step_idx,
        diagnosis=Diagnosis(
            bottleneck=bottleneck,
            evidence="hotspots dominated by ggml_compute_forward_mul_mat (62.3pct self samples)",
            source="mcp",
        ),
        action_id=action_id,
        params={"state": "ON"},
        rationale=rationale,
        before_config_id=before_cfg.config_id,
        after_config_id=after_cfg.config_id,
        screen=confirm,
        confirm=confirm,
        kept=kept,
        delta=Delta(metric="decode_tok_s", pct=pct, ci_significant=ci_significant),
        quality_ok=quality_ok,
    )


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A synthetic but structurally valid run dir: manifest, baseline, expert,
    trajectory.jsonl (2 steps: one kept, one reverted), recipe.json."""
    baseline_cfg = _bench_config(cmake_flags=("-DGGML_NATIVE=OFF",))
    native_cfg = _bench_config(cmake_flags=("-DGGML_NATIVE=ON",))
    kleidiai_cfg = _bench_config(
        cmake_flags=("-DGGML_NATIVE=ON", "-DGGML_CPU_KLEIDIAI=ON")
    )
    expert_cfg = _bench_config(
        cmake_flags=("-DGGML_NATIVE=ON", "-DGGML_CPU_KLEIDIAI=ON"), n_threads=16
    )

    baseline_result = _bench_result(baseline_cfg, decode=21.81, prefill=17552.0)
    expert_result = _bench_result(expert_cfg, decode=44.96, prefill=3800.0)
    native_result = _bench_result(native_cfg, decode=44.49, prefill=3820.0)
    reverted_cand_result = _bench_result(kleidiai_cfg, decode=44.60, prefill=3810.0)

    manifest = _manifest(baseline_cfg, expert_cfg)

    step1 = _trajectory_step(
        0,
        action_id="ggml_native",
        before_cfg=baseline_cfg,
        after_cfg=native_cfg,
        confirm=native_result,
        kept=True,
        pct=104.0,
        ci_significant=True,
        rationale="native build closes the prefill/decode gap the spike measured",
        bottleneck="compute-bound",
    )
    step2 = _trajectory_step(
        1,
        action_id="kleidiai",
        before_cfg=native_cfg,
        after_cfg=kleidiai_cfg,
        confirm=reverted_cand_result,
        kept=False,
        pct=0.2,
        ci_significant=False,
        quality_ok=True,
        rationale="kleidiai improvement was within CI noise, reverted",
        bottleneck="unknown",
    )

    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(to_json(manifest), encoding="utf-8")
    (run / "baseline.json").write_text(to_json(baseline_result), encoding="utf-8")
    (run / "expert.json").write_text(to_json(expert_result), encoding="utf-8")
    append_jsonl(run / "trajectory.jsonl", step1)
    append_jsonl(run / "trajectory.jsonl", step2)

    recipe = Recipe(
        run_id=manifest.run_id,
        armsmith_version="0.1.0",
        target_class="r8g",
        model=manifest.model,
        winning_config=native_cfg,
        baseline_config=baseline_cfg,
        expert_config=expert_cfg,
        baseline_result=baseline_result,
        winning_result=native_result,
        gap_closed_pct=98.3,
        created_at="2026-07-05T15:00:00Z",
    )
    (run / "recipe.json").write_text(to_json(recipe), encoding="utf-8")
    return run


# --------------------------------------------------------------------------
# build_report_model
# --------------------------------------------------------------------------


def test_build_report_model_loads_all_artifacts(run_dir: Path):
    model = build_report_model(run_dir)
    assert isinstance(model, ReportModel)
    assert model.manifest.run_id == "2026-07-05T14-03-11Z-r8g"
    assert model.baseline.decode_tok_s.median == pytest.approx(21.81)
    assert model.expert is not None
    assert model.expert.decode_tok_s.median == pytest.approx(44.96)
    assert len(model.steps) == 2
    assert model.recipe.gap_closed_pct == pytest.approx(98.3)


def test_build_report_model_per_lever_only_kept_steps(run_dir: Path):
    model = build_report_model(run_dir)
    # Only step1 (ggml_native) was kept; step2 (kleidiai) was reverted.
    lever_ids = [action_id for action_id, _ in model.per_lever]
    assert lever_ids == ["ggml_native"]
    assert model.per_lever[0][1].pct == pytest.approx(104.0)


def test_build_report_model_missing_expert_file_is_none(run_dir: Path):
    (run_dir / "expert.json").unlink()
    model = build_report_model(run_dir)
    assert model.expert is None


def test_build_report_model_missing_trajectory_is_empty(run_dir: Path):
    (run_dir / "trajectory.jsonl").unlink()
    model = build_report_model(run_dir)
    assert model.steps == ()
    assert model.per_lever == ()


# --------------------------------------------------------------------------
# render_report
# --------------------------------------------------------------------------


def test_render_report_writes_default_path(run_dir: Path):
    out = render_report(run_dir)
    assert out == run_dir / "report.html"
    assert out.exists()


def test_render_report_writes_custom_out(run_dir: Path, tmp_path: Path):
    custom = tmp_path / "elsewhere" / "custom.html"
    custom.parent.mkdir()
    out = render_report(run_dir, out=custom)
    assert out == custom
    assert out.exists()


def test_render_report_contains_key_content(run_dir: Path):
    html = render_report(run_dir).read_text(encoding="utf-8")

    # run identity
    assert "2026-07-05T14-03-11Z-r8g" in html
    assert "r8g.4xlarge" in html
    assert "Qwen2.5-7B-Instruct" in html

    # baseline vs winning numbers (from the dataclasses, not recomputed)
    assert "21.81" in html
    assert "44.49" in html

    # gap-closed
    assert "98.3" in html

    # trajectory: action ids, kept/reverted badges, rationale
    assert "ggml_native" in html
    assert "kleidiai" in html
    assert "kept" in html
    assert "reverted" in html
    assert "native build closes the prefill/decode gap the spike measured" in html
    assert "kleidiai improvement was within CI noise, reverted" in html

    # per-lever section only lists the kept lever
    assert html.count("ggml_native") >= 2  # timeline + per-lever section


def test_render_report_no_em_dashes(run_dir: Path):
    html = render_report(run_dir).read_text(encoding="utf-8")
    assert "—" not in html  # em dash banned in UI strings (CLAUDE.md)


def test_render_report_is_standalone_no_external_resources(run_dir: Path):
    html = render_report(run_dir).read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src=" not in html
    assert "<link " not in html  # no external stylesheet/font links
    assert "cdn." not in html.lower()


def test_render_report_handles_missing_expert(run_dir: Path):
    (run_dir / "expert.json").unlink()
    html = render_report(run_dir).read_text(encoding="utf-8")
    assert html  # renders without raising
    assert "2026-07-05T14-03-11Z-r8g" in html


def test_render_report_handles_empty_trajectory(run_dir: Path):
    (run_dir / "trajectory.jsonl").unlink()
    html = render_report(run_dir).read_text(encoding="utf-8")
    assert "No trajectory steps recorded" in html
    assert "No levers were kept" in html
