"""Tests for armsmith.models: round-trip serialization + invariant checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from armsmith.models import (
    ActionSpec,
    BenchConfig,
    BenchmarkResult,
    Delta,
    Diagnosis,
    HotspotRow,
    MetricStat,
    ModelDecodeError,
    ModelSpec,
    ParamSpec,
    PerformixSnapshot,
    QualityScore,
    RawSamples,
    Recipe,
    RunManifest,
    TargetSpec,
    TrajectoryStep,
    WorkloadSpec,
    append_jsonl,
    build_key,
    config_id,
    from_json,
    load_workload,
    read_jsonl,
    to_json,
)

# --------------------------------------------------------------------------
# fixtures / builders
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


def _metric_stat(median: float = 21.8) -> MetricStat:
    return MetricStat(median=median, ci_low=median - 1, ci_high=median + 1)


def _raw_samples(n: int = 7) -> RawSamples:
    return RawSamples(
        decode_tok_s=tuple(21.0 + i * 0.1 for i in range(n)),
        prefill_ttft_ms=tuple(45.0 + i for i in range(n)),
    )


def _bench_result(
    cfg: BenchConfig, *, stage: str = "confirm", n_repeats: int = 7
) -> BenchmarkResult:
    return BenchmarkResult(
        config_id=cfg.config_id,
        decode_tok_s=_metric_stat(21.8),
        prefill_ttft_ms=_metric_stat(45.0),
        quality=QualityScore(perplexity=6.12, kl_vs_baseline=None),
        peak_mem_mb=4300.5,
        model_size_mb=4233.1,
        n_repeats=n_repeats,
        stage=stage,
        raw_samples=_raw_samples(n_repeats),
    )


# --------------------------------------------------------------------------
# config_id / build_key
# --------------------------------------------------------------------------


def test_config_id_deterministic_and_excludes_config_id_key():
    fields = dict(
        cmake_flags=("-DGGML_NATIVE=ON",),
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
    id1 = config_id(fields)
    id2 = config_id({**fields, "config_id": "should-be-ignored"})
    assert id1 == id2
    assert len(id1) == 12
    assert all(c in "0123456789abcdef" for c in id1)


def test_config_id_changes_with_any_field():
    base = dict(
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
    changed = {**base, "n_threads": 8}
    assert config_id(base) != config_id(changed)


def test_bench_config_create_sets_derived_config_id():
    cfg = _bench_config()
    fields = {k: v for k, v in cfg.__dict__.items() if k != "config_id"}
    assert cfg.config_id == config_id(fields)


def test_build_key_ignores_runtime_knobs_but_not_build_flags():
    cfg_a = _bench_config(n_threads=16)
    cfg_b = _bench_config(n_threads=8)  # runtime-only diff
    cfg_c = _bench_config(cmake_flags=("-DGGML_NATIVE=ON",))  # build diff

    assert build_key(cfg_a) == build_key(cfg_b)
    assert build_key(cfg_a) != build_key(cfg_c)


def test_build_key_insensitive_to_cmake_flag_order():
    cfg_a = _bench_config(cmake_flags=("-DGGML_NATIVE=ON", "-DGGML_CPU_KLEIDIAI=ON"))
    cfg_b = _bench_config(cmake_flags=("-DGGML_CPU_KLEIDIAI=ON", "-DGGML_NATIVE=ON"))
    assert build_key(cfg_a) == build_key(cfg_b)


# --------------------------------------------------------------------------
# BenchmarkResult invariants
# --------------------------------------------------------------------------


def test_confirm_stage_requires_n_repeats_at_least_5():
    cfg = _bench_config()
    with pytest.raises(ValueError, match="n_repeats>=5"):
        _bench_result(cfg, stage="confirm", n_repeats=3)


def test_confirm_stage_allows_exactly_5():
    cfg = _bench_config()
    result = _bench_result(cfg, stage="confirm", n_repeats=5)
    assert result.n_repeats == 5


def test_confirm_stage_requires_sample_count_match_n_repeats():
    cfg = _bench_config()
    bad_samples = RawSamples(
        decode_tok_s=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),  # 6 samples
        prefill_ttft_ms=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    with pytest.raises(ValueError, match="len\\(raw_samples.decode_tok_s\\)"):
        BenchmarkResult(
            config_id=cfg.config_id,
            decode_tok_s=_metric_stat(),
            prefill_ttft_ms=_metric_stat(),
            quality=QualityScore(None, None),
            peak_mem_mb=None,
            model_size_mb=4233.1,
            n_repeats=7,  # claims 7 but only 6 samples present
            stage="confirm",
            raw_samples=bad_samples,
        )


def test_screen_stage_does_not_enforce_n_repeats_floor():
    cfg = _bench_config()
    result = _bench_result(cfg, stage="screen", n_repeats=3)
    assert result.n_repeats == 3
    assert result.stage == "screen"


# --------------------------------------------------------------------------
# Round-trip serialization
# --------------------------------------------------------------------------


def test_round_trip_metric_stat():
    obj = _metric_stat()
    assert from_json(MetricStat, to_json(obj)) == obj


def test_round_trip_quality_score_with_nones():
    obj = QualityScore(perplexity=None, kl_vs_baseline=0.03)
    assert from_json(QualityScore, to_json(obj)) == obj


def test_round_trip_target_spec():
    obj = _target_spec()
    assert from_json(TargetSpec, to_json(obj)) == obj


def test_round_trip_model_spec_resolve():
    obj = _model_spec()
    restored = from_json(ModelSpec, to_json(obj))
    assert restored == obj
    assert restored.resolve("Q4_0") == ("~/m/qwen-q4_0.gguf", "abc123")
    with pytest.raises(KeyError):
        restored.resolve("Q4_K_M")


def test_round_trip_run_manifest():
    obj = RunManifest(
        run_id="2026-07-05T14-03-11Z-r8g",
        target=_target_spec(),
        model=_model_spec(),
        workload_ref="examples/bench.yaml",
        baseline_ref="abc123456789",
        expert_ref="def987654321",
        created_at="2026-07-05T14:03:11Z",
        armsmith_version="0.0.1",
    )
    assert from_json(RunManifest, to_json(obj)) == obj


def test_round_trip_bench_config_with_nested_tuples():
    cfg = _bench_config(
        cpu_mask="0xFFFF", flash_attn=True, env=(("GGML_KLEIDIAI_SME", "1"),)
    )
    restored = from_json(BenchConfig, to_json(cfg))
    assert restored == cfg
    assert restored.env == (("GGML_KLEIDIAI_SME", "1"),)


def test_round_trip_bench_result():
    cfg = _bench_config()
    result = _bench_result(cfg)
    restored = from_json(BenchmarkResult, to_json(result))
    assert restored == result


def test_round_trip_performix_snapshot_from_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "hotspots_portable.json"
    sc = json.loads(fixture_path.read_text(encoding="utf-8"))
    snapshot = PerformixSnapshot(
        config_id="abc123456789",
        recipe=sc["recipe"],
        source="mcp",
        status=sc["status"],
        hotspots=tuple(
            HotspotRow(
                symbol=r["FUNCTION_NAME"],
                self_samples=r["PERIODIC_SAMPLES_SELF"],
                self_pct=r["PERIODIC_SAMPLES_SELF_PERCENT"],
                node_type=r.get("NODE_TYPE", "function"),
            )
            for r in sc["rows"]
        ),
        raw_columns=tuple(sc.get("columns", [])),
        warnings=tuple(sc.get("warnings", [])),
    )
    restored = from_json(PerformixSnapshot, to_json(snapshot))
    assert restored == snapshot
    assert restored.cache_miss_rate is None
    assert restored.mem_bandwidth_gbps is None
    assert restored.ipc is None


def test_round_trip_action_spec_with_nested_param_specs():
    spec = ActionSpec(
        id="kv_cache_type",
        name="KV cache type",
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
    )
    restored = from_json(ActionSpec, to_json(spec))
    assert restored == spec
    assert restored.params_schema["type_k"].choices == ("f16", "q8_0")


def test_round_trip_trajectory_step_with_optional_confirm():
    cfg_before = _bench_config()
    cfg_after = _bench_config(n_threads=8)
    step = TrajectoryStep(
        step_idx=0,
        diagnosis=Diagnosis(
            bottleneck="unknown", evidence="thin samples", source="mcp"
        ),
        action_id="threads",
        params={"n_threads": 8, "cpu_mask": "default"},
        rationale="no LLM: deterministic registry order",
        before_config_id=cfg_before.config_id,
        after_config_id=cfg_after.config_id,
        screen=_bench_result(cfg_after, stage="screen", n_repeats=3),
        confirm=None,
        kept=False,
        delta=Delta(metric="decode_tok_s", pct=1.2, ci_significant=False),
        quality_ok=True,
    )
    restored = from_json(TrajectoryStep, to_json(step))
    assert restored == step
    assert restored.confirm is None


def test_round_trip_recipe_with_none_expert():
    cfg = _bench_config()
    result = _bench_result(cfg)
    recipe = Recipe(
        run_id="2026-07-05T14-03-11Z-r8g",
        armsmith_version="0.0.1",
        target_class="r8g",
        model=_model_spec(),
        winning_config=cfg,
        baseline_config=cfg,
        expert_config=None,
        baseline_result=result,
        winning_result=result,
        gap_closed_pct=None,
        created_at="2026-07-05T14:03:11Z",
    )
    restored = from_json(Recipe, to_json(recipe))
    assert restored == recipe
    assert restored.expert_config is None
    assert restored.gap_closed_pct is None


def test_round_trip_workload_spec_with_default_prompt():
    obj = WorkloadSpec(
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
        screen_repeats=3,
        confirm_repeats=7,
        eval_text_path="examples/eval.txt",
    )
    assert from_json(WorkloadSpec, to_json(obj)) == obj
    assert obj.prompt is None


# --------------------------------------------------------------------------
# from_json error handling
# --------------------------------------------------------------------------


def test_from_json_raises_on_invalid_json():
    with pytest.raises(ModelDecodeError):
        from_json(MetricStat, "{not valid json")


def test_from_json_raises_on_missing_required_field():
    payload = json.dumps({"median": 1.0, "ci_low": 0.5})  # missing ci_high
    with pytest.raises(ModelDecodeError):
        from_json(MetricStat, payload)


def test_from_json_raises_on_non_object_payload():
    with pytest.raises(ModelDecodeError):
        from_json(MetricStat, "[1, 2, 3]")


# --------------------------------------------------------------------------
# JSONL append/read
# --------------------------------------------------------------------------


def test_append_and_read_jsonl_round_trips_multiple_steps(tmp_path: Path):
    cfg = _bench_config()
    steps = [
        TrajectoryStep(
            step_idx=i,
            diagnosis=Diagnosis(bottleneck="unknown", evidence="e", source="cli"),
            action_id="ggml_native",
            params={"state": "ON"},
            rationale="r",
            before_config_id=cfg.config_id,
            after_config_id=cfg.config_id,
            screen=_bench_result(cfg, stage="screen", n_repeats=3),
            confirm=_bench_result(cfg, stage="confirm", n_repeats=5),
            kept=i % 2 == 0,
            delta=Delta(metric="decode_tok_s", pct=5.0 * i, ci_significant=True),
            quality_ok=True,
        )
        for i in range(3)
    ]

    path = tmp_path / "trajectory.jsonl"
    for step in steps:
        append_jsonl(path, step)

    restored = read_jsonl(path)
    assert restored == steps


def test_append_jsonl_creates_parent_dirs(tmp_path: Path):
    cfg = _bench_config()
    step = TrajectoryStep(
        step_idx=0,
        diagnosis=Diagnosis(bottleneck="compute-bound", evidence="e", source="cli"),
        action_id="ggml_native",
        params={},
        rationale="r",
        before_config_id=cfg.config_id,
        after_config_id=cfg.config_id,
        screen=None,
        confirm=None,
        kept=False,
        delta=Delta(metric="decode_tok_s", pct=0.0, ci_significant=False),
        quality_ok=True,
    )
    path = tmp_path / "nested" / "trajectory.jsonl"
    append_jsonl(path, step)
    assert path.exists()
    assert read_jsonl(path) == [step]


# --------------------------------------------------------------------------
# load_workload (YAML)
# --------------------------------------------------------------------------


def test_load_workload_from_yaml(tmp_path: Path):
    yaml_text = """
n_prompt: 512
n_gen: 128
n_batch: 2048
n_ubatch: 512
screen_repeats: 3
confirm_repeats: 7
eval_text_path: examples/eval.txt
"""
    path = tmp_path / "bench.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    workload = load_workload(str(path))
    assert workload == WorkloadSpec(
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
        screen_repeats=3,
        confirm_repeats=7,
        eval_text_path="examples/eval.txt",
        prompt=None,
    )


def test_load_workload_with_explicit_prompt(tmp_path: Path):
    yaml_text = """
n_prompt: 256
n_gen: 64
n_batch: 1024
n_ubatch: 256
screen_repeats: 3
confirm_repeats: 5
eval_text_path: examples/eval.txt
prompt: "Explain the theory of relativity."
"""
    path = tmp_path / "bench.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    workload = load_workload(str(path))
    assert workload.prompt == "Explain the theory of relativity."
