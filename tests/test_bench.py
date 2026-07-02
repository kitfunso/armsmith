"""Unit tests for armsmith.bench -- CI statistics, llama-bench parsing, two-stage policy.

All measurement-math tests run against the REAL committed captures in ``tests/fixtures/``
(``bench-{portable,native,generic}.json`` = ``llama-bench -o json`` output, one pp512 prefill
row + one tg128 decode row each), so the numbers asserted here are the numbers the tool saw on
the Graviton target -- no invented data (CLAUDE.md rule 1). Fixture paths are resolved relative
to this file via ``pathlib`` so the suite runs identically on the Windows host and Linux CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from armsmith.bench import (
    Benchmarker,
    ci_disjoint,
    median_ci,
    parse_llama_bench_output,
    promote_to_confirm,
    significant,
)
from armsmith.models import BenchParseError
from armsmith.models import BenchConfig, QualityScore, WorkloadSpec

FIXTURES = Path(__file__).parent / "fixtures"

# Screen-stage: the real fixtures carry 3 repeats, so they are parsed at stage="screen"
# (the confirm invariant requires N>=5). significant()/ci_disjoint() work on any MetricStat.
_NO_QUALITY = QualityScore(perplexity=None, kl_vs_baseline=None)


def _parse_fixture(name: str):
    raw = (FIXTURES / name).read_text(encoding="utf-8")
    return parse_llama_bench_output(
        raw, config_id=name.replace(".json", ""), stage="screen", quality=_NO_QUALITY
    )


# --------------------------------------------------------------------------------------
# Parsing the real fixtures
# --------------------------------------------------------------------------------------
def test_parse_portable_fixture_values():
    result = _parse_fixture("bench-portable.json")
    # decode = MEDIAN of samples_ts [21.8057, 21.8128, 21.8117] -> 21.8117
    assert result.decode_tok_s.median == pytest.approx(21.81, abs=0.01)
    # prefill TTFT = MEDIAN of samples_ns/1e6 [17554.27, 17551.76, 17550.07] -> 17551.76
    assert result.prefill_ttft_ms.median == pytest.approx(17552, abs=1.0)
    # model_size 4438167552 bytes / 1048576
    assert result.model_size_mb == pytest.approx(4232.5, abs=0.5)
    assert result.n_repeats == 3
    assert result.stage == "screen"
    assert result.raw_samples.decode_tok_s == (21.8057, 21.8128, 21.8117)


def test_parse_generic_fixture_uses_median_not_mean():
    """bench-generic.json is the fixture where median != mean: samples_ts
    [42.9782, 45.277, 45.2007] have median 45.2007 but avg_ts (mean) 44.49. The contract
    computes the MEDIAN, so this must land at ~45.20, never 44.49."""
    result = _parse_fixture("bench-generic.json")
    assert result.decode_tok_s.median == pytest.approx(45.20, abs=0.01)
    assert result.decode_tok_s.median != pytest.approx(44.49, abs=0.05)


def test_parse_native_fixture_values():
    result = _parse_fixture("bench-native.json")
    # median of [44.9741, 44.9627, 44.9558] -> 44.9627
    assert result.decode_tok_s.median == pytest.approx(44.96, abs=0.01)


# --------------------------------------------------------------------------------------
# Significance against the real portable/native/generic captures
# --------------------------------------------------------------------------------------
def test_native_vs_portable_is_significant():
    """The headline lever (GGML_NATIVE OFF->ON): native decode (~45 tok/s) is CI-significantly
    faster than the portable baseline (~21.8 tok/s). Non-overlapping CIs -> True."""
    native = _parse_fixture("bench-native.json")
    portable = _parse_fixture("bench-portable.json")
    assert (
        significant(native.decode_tok_s, portable.decode_tok_s, higher_is_better=True)
        is True
    )


def test_native_vs_generic_is_not_significant():
    """Spike-0 finding: vanilla cmake already defaults GGML_NATIVE=ON, so 'native' and
    'generic' are the same build within noise -- native (median 44.96) is NOT CI-significantly
    faster than generic (median 45.20). Must report not-significant."""
    native = _parse_fixture("bench-native.json")
    generic = _parse_fixture("bench-generic.json")
    assert (
        significant(native.decode_tok_s, generic.decode_tok_s, higher_is_better=True)
        is False
    )


def test_ci_disjoint_symmetric():
    native = _parse_fixture("bench-native.json")
    portable = _parse_fixture("bench-portable.json")
    generic = _parse_fixture("bench-generic.json")
    assert ci_disjoint(native.decode_tok_s, portable.decode_tok_s) is True
    assert ci_disjoint(portable.decode_tok_s, native.decode_tok_s) is True  # symmetry
    assert ci_disjoint(native.decode_tok_s, generic.decode_tok_s) is False


# --------------------------------------------------------------------------------------
# median_ci: determinism, bounds, small-N degeneracy
# --------------------------------------------------------------------------------------
def test_median_ci_is_deterministic_for_a_seed():
    samples = [44.9741, 44.9627, 44.9558, 45.01, 44.90]
    a = median_ci(samples, seed=0)
    b = median_ci(samples, seed=0)
    assert a == b
    assert a.ci_low <= a.median <= a.ci_high


def test_median_ci_bounds_are_within_sample_range():
    samples = [42.9782, 45.277, 45.2007]
    stat = median_ci(samples, seed=0)
    assert min(samples) <= stat.ci_low <= stat.ci_high <= max(samples)
    assert stat.median == pytest.approx(45.2007, abs=1e-6)


def test_median_ci_small_n_collapses_to_point():
    one = median_ci([21.5], seed=0)
    assert one.ci_low == one.ci_high == one.median == 21.5
    two = median_ci([21.5, 22.5], seed=0)
    assert two.ci_low == two.ci_high == two.median == 22.0


def test_median_ci_empty_raises():
    with pytest.raises(BenchParseError):
        median_ci([], seed=0)


# --------------------------------------------------------------------------------------
# promote_to_confirm gate
# --------------------------------------------------------------------------------------
def test_promote_to_confirm_gate():
    native = _parse_fixture("bench-native.json")  # decode ~44.96
    portable = _parse_fixture("bench-portable.json")  # decode ~21.81
    # native is ~106% faster than portable -> well past a 2% gate
    assert promote_to_confirm(native, portable, gate_pct=2.0) is True
    # portable is slower than native -> negative improvement -> gate rejects
    assert promote_to_confirm(portable, native, gate_pct=2.0) is False
    # native vs generic (~45.20): improvement is negative -> rejected even at gate 0
    generic = _parse_fixture("bench-generic.json")
    assert promote_to_confirm(native, generic, gate_pct=0.0) is False


# --------------------------------------------------------------------------------------
# Parse error paths
# --------------------------------------------------------------------------------------
def test_parse_missing_decode_row_raises():
    # only a prefill row (n_gen==0) -- no decode row
    raw = json.dumps(
        [
            {
                "n_prompt": 512,
                "n_gen": 0,
                "model_size": 1,
                "samples_ns": [1.0],
                "samples_ts": [1.0],
            }
        ]
    )
    with pytest.raises(BenchParseError):
        parse_llama_bench_output(
            raw, config_id="x", stage="screen", quality=_NO_QUALITY
        )


def test_parse_empty_samples_raises():
    raw = json.dumps(
        [
            {
                "n_prompt": 512,
                "n_gen": 0,
                "model_size": 1,
                "samples_ns": [],
                "samples_ts": [],
            },
            {
                "n_prompt": 0,
                "n_gen": 128,
                "model_size": 1,
                "samples_ns": [1.0],
                "samples_ts": [1.0],
            },
        ]
    )
    with pytest.raises(BenchParseError):
        parse_llama_bench_output(
            raw, config_id="x", stage="screen", quality=_NO_QUALITY
        )


def test_parse_non_array_raises():
    with pytest.raises(BenchParseError):
        parse_llama_bench_output(
            "{}", config_id="x", stage="screen", quality=_NO_QUALITY
        )


def test_parse_sample_length_mismatch_raises():
    raw = json.dumps(
        [
            {
                "n_prompt": 512,
                "n_gen": 0,
                "model_size": 1,
                "samples_ns": [1.0, 2.0],
                "samples_ts": [1.0],
            },
            {
                "n_prompt": 0,
                "n_gen": 128,
                "model_size": 1,
                "samples_ns": [1.0],
                "samples_ts": [1.0],
            },
        ]
    )
    with pytest.raises(BenchParseError):
        parse_llama_bench_output(
            raw, config_id="x", stage="screen", quality=_NO_QUALITY
        )


# --------------------------------------------------------------------------------------
# Benchmarker two-stage policy with an injected fake runner (no SSH)
# --------------------------------------------------------------------------------------
def _cfg(n_threads: int) -> BenchConfig:
    return BenchConfig.create(
        cmake_flags=("-DGGML_NATIVE=OFF",),
        quant="Q4_0",
        n_threads=n_threads,
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


def _bench_raw(decode_ts: list[float], model_size: int = 4438167552) -> str:
    """Build a minimal llama-bench raw (one prefill + one decode row) with ``len(decode_ts)``
    repeats. Prefill values are fixed; only the decode tok/s samples are parameterised.
    """
    n = len(decode_ts)
    prefill = {
        "n_prompt": 512,
        "n_gen": 0,
        "model_size": model_size,
        "samples_ns": [1.75e10] * n,
        "samples_ts": [29.0] * n,
    }
    decode = {
        "n_prompt": 0,
        "n_gen": 128,
        "model_size": model_size,
        "samples_ns": [1_000_000_000] * n,
        "samples_ts": decode_ts,
    }
    return json.dumps([prefill, decode])


class _FakeRunner:
    """Injected BenchRunner. Emits a config-specific decode speed with a tiny deterministic
    per-repeat spread (so the bootstrap is exercised on a genuine interval), and records the
    (config_id, n_repeats) call order so the A/B/A/B interleaving can be asserted."""

    def __init__(self, speeds: dict[str, float]) -> None:
        self._speeds = speeds
        self.calls: list[tuple[str, int]] = []

    def __call__(self, cfg: BenchConfig, n_repeats: int) -> tuple[str, float | None]:
        self.calls.append((cfg.config_id, n_repeats))
        base = self._speeds[cfg.config_id]
        decode_ts = [base + 0.01 * i for i in range(n_repeats)]
        return _bench_raw(decode_ts), 512.0


def _workload(screen_repeats: int = 3, confirm_repeats: int = 5) -> WorkloadSpec:
    return WorkloadSpec(
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
        screen_repeats=screen_repeats,
        confirm_repeats=confirm_repeats,
        eval_text_path="examples/eval.txt",
    )


def test_benchmarker_screen_is_screen_stage_without_quality():
    inc = _cfg(16)
    runner = _FakeRunner({inc.config_id: 21.8})
    bench = Benchmarker(
        runner, lambda cfg: QualityScore(5.0, 0.0), _workload(screen_repeats=3), seed=0
    )
    result = bench.screen(inc)
    assert result.stage == "screen"
    assert result.n_repeats == 3
    assert result.quality.perplexity is None  # screen never measures quality
    assert runner.calls == [(inc.config_id, 3)]


def test_benchmarker_confirm_is_confirm_stage_with_quality():
    inc = _cfg(16)
    runner = _FakeRunner({inc.config_id: 45.0})
    bench = Benchmarker(
        runner,
        lambda cfg: QualityScore(5.0, 0.01),
        _workload(confirm_repeats=7),
        seed=0,
    )
    result = bench.confirm(inc)
    assert result.stage == "confirm"
    assert result.n_repeats == 7
    assert result.quality.perplexity == 5.0
    assert len(result.raw_samples.decode_tok_s) == 7


def test_confirm_ab_interleaves_and_finds_significant_gain():
    incumbent = _cfg(8)
    candidate = _cfg(16)
    runner = _FakeRunner({incumbent.config_id: 21.8, candidate.config_id: 44.9})
    bench = Benchmarker(
        runner, lambda cfg: QualityScore(5.0, 0.0), _workload(confirm_repeats=5), seed=0
    )

    inc_result, cand_result = bench.confirm_ab(incumbent, candidate)

    # Both are FRESH confirm-stage results with confirm_repeats samples.
    assert inc_result.stage == "confirm" and cand_result.stage == "confirm"
    assert inc_result.n_repeats == 5 and cand_result.n_repeats == 5
    assert len(cand_result.raw_samples.decode_tok_s) == 5

    # A/B/A/B interleaving: every call is a single repeat, alternating inc, cand, inc, cand...
    # (a sequential A/A/A/B/B/B would be [(inc,1)*5, (cand,1)*5] -- explicitly forbidden).
    assert len(runner.calls) == 10
    assert all(n == 1 for _, n in runner.calls)
    expected = [incumbent.config_id, candidate.config_id] * 5
    assert [cid for cid, _ in runner.calls] == expected

    # The candidate is CI-significantly faster on decode.
    assert significant(
        cand_result.decode_tok_s, inc_result.decode_tok_s, higher_is_better=True
    )


def test_confirm_ab_equal_speed_is_not_significant():
    incumbent = _cfg(8)
    candidate = _cfg(16)
    # same speed -> overlapping CIs -> not significant (guards against a false keep)
    runner = _FakeRunner({incumbent.config_id: 30.0, candidate.config_id: 30.0})
    bench = Benchmarker(
        runner, lambda cfg: QualityScore(5.0, 0.0), _workload(confirm_repeats=5), seed=0
    )
    inc_result, cand_result = bench.confirm_ab(incumbent, candidate)
    assert (
        significant(
            cand_result.decode_tok_s, inc_result.decode_tok_s, higher_is_better=True
        )
        is False
    )
