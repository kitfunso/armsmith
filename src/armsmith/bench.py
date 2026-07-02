"""bench.py -- CI statistics, llama-bench JSON parse, two-stage screen/confirm policy.

Owns the measurement math for armsmith. No SSH lives here: the target run is injected as
a ``BenchRunner`` callable (CONTRACTS section 4), so every function in this module is
unit-testable against the committed ``tests/fixtures/bench-*.json`` captures with zero AWS
spend.

Statistics rationale (CLAUDE.md rule 3 -- get the CI math right or every claim is suspect):

The reported point estimate is the **median** of the per-repeat llama-bench samples, not the
mean. llama-bench's own ``avg_ts`` is a mean and is dragged around by the occasional slow
repeat (bench-generic.json is the worked example: mean ``avg_ts`` 44.49 vs median 45.20 over
samples ``[42.98, 45.28, 45.20]``); the median is the robust summary of a small, noisy,
not-necessarily-normal timing sample.

The interval is a **seeded percentile bootstrap**, not a Student-t interval. A t-interval
assumes a normal sampling distribution of the *mean*; it is the wrong tool for (a) the median
and (b) N in [3, 7] repeats where normality is unwarranted. The bootstrap instead estimates
the sampling distribution of the median directly: resample the repeats with replacement
``n_boot`` times, take the median of each resample, and read the 2.5 / 97.5 percentiles of
that bootstrap distribution as the 95% CI. It assumes nothing about the shape of the timing
distribution and is fully deterministic because the numpy ``Generator`` is seeded (``seed=0``)
-- the same samples always yield the same CI, which is what lets a reported speedup replay.

Significance is the conservative **non-overlapping-CI** test (CLAUDE.md rule 3): a speedup
counts only when the candidate's 95% CI is entirely on the better side of the incumbent's.
Overlapping CIs are reported not-significant -- one-shot numbers are noise.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from armsmith.models import BenchParseError
from armsmith.models import (
    BenchConfig,
    BenchmarkResult,
    MetricStat,
    QualityScore,
    RawSamples,
    Stage,
    WorkloadSpec,
)

logger = logging.getLogger(__name__)

# Injected by target.py at run time; kept as a type here so bench.py never opens SSH.
#   (cfg, n_repeats) -> (raw `llama-bench -o json` stdout, peak_mem_mb | None)
BenchRunner = Callable[[BenchConfig, int], tuple[str, float | None]]
#   cfg -> QualityScore (target.run_quality; llama-perplexity)
QualityFn = Callable[[BenchConfig], QualityScore]

_MIB = 1_048_576.0  # bytes per MiB (llama-bench model_size is bytes)


# --------------------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------------------
def median_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    seed: int = 0,
    n_boot: int = 10_000,
) -> MetricStat:
    """Median point estimate + seeded percentile-bootstrap 95% CI.

    ``ci_low`` / ``ci_high`` are the 2.5 / 97.5 percentiles of ``n_boot`` resampled medians.
    Deterministic for a fixed ``seed`` (CLAUDE.md rule 4: an unreplayable result is not a
    result). For ``N < 3`` the bootstrap of a median is degenerate, so the CI collapses to the
    point estimate; callers treat such a result as non-significant by policy (``screen`` never
    claims a CI, and ``confirm`` requires ``N >= 5``).
    """
    arr = np.asarray(samples, dtype=float)
    n = int(arr.size)
    if n == 0:
        raise BenchParseError("median_ci received an empty sample array")

    median = float(np.median(arr))
    if n < 3:
        logger.warning(
            "median_ci: N=%d < 3 -- collapsing CI to the point estimate (not significant)",
            n,
        )
        return MetricStat(median=median, ci_low=median, ci_high=median)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(arr[idx], axis=1)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_medians, 100.0 * alpha))
    ci_high = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha)))
    return MetricStat(median=median, ci_low=ci_low, ci_high=ci_high)


def significant(
    candidate: MetricStat,
    incumbent: MetricStat,
    *,
    higher_is_better: bool = True,
) -> bool:
    """CI-significance: the two 95% CIs are disjoint AND ``candidate`` is on the better side.

    ``higher_is_better`` (decode tok/s): ``candidate.ci_low > incumbent.ci_high``.
    Otherwise (prefill TTFT ms, lower is better): ``candidate.ci_high < incumbent.ci_low``.
    """
    if higher_is_better:
        return candidate.ci_low > incumbent.ci_high
    return candidate.ci_high < incumbent.ci_low


def ci_disjoint(a: MetricStat, b: MetricStat) -> bool:
    """Symmetric non-overlap test (direction-agnostic); used by report deltas."""
    return a.ci_high < b.ci_low or b.ci_high < a.ci_low


def promote_to_confirm(
    screen: BenchmarkResult,
    incumbent: BenchmarkResult,
    *,
    gate_pct: float,
) -> bool:
    """Screen -> confirm gate: promote iff the cheap screen decode median beat the incumbent's
    by at least ``gate_pct`` percent. No CI claim here -- the gate just filters candidates so a
    confirm (the expensive, CI-bearing stage) is not spent on a non-improver.
    """
    base = incumbent.decode_tok_s.median
    if base <= 0.0:
        return False
    improvement_pct = (screen.decode_tok_s.median - base) / base * 100.0
    return improvement_pct >= gate_pct


# --------------------------------------------------------------------------------------
# llama-bench JSON parsing
# --------------------------------------------------------------------------------------
def _rows(raw: str) -> list[Mapping[str, object]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchParseError(f"llama-bench output is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise BenchParseError(
            "llama-bench output must be a non-empty JSON array of rows"
        )
    return parsed


def _find_bench_rows(raw: str) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Return ``(prefill_row, decode_row)`` from one llama-bench raw.

    prefill row = ``n_gen == 0 and n_prompt > 0``; decode row = ``n_prompt == 0 and n_gen > 0``.
    Raises ``BenchParseError`` if either row is absent.
    """
    prefill: Mapping[str, object] | None = None
    decode: Mapping[str, object] | None = None
    for row in _rows(raw):
        if not isinstance(row, dict):
            continue
        n_prompt = row.get("n_prompt")
        n_gen = row.get("n_gen")
        if n_gen == 0 and isinstance(n_prompt, int) and n_prompt > 0:
            prefill = row
        elif n_prompt == 0 and isinstance(n_gen, int) and n_gen > 0:
            decode = row
    if prefill is None:
        raise BenchParseError(
            "no prefill row (n_gen==0, n_prompt>0) in llama-bench output"
        )
    if decode is None:
        raise BenchParseError(
            "no decode row (n_prompt==0, n_gen>0) in llama-bench output"
        )
    return prefill, decode


def _samples(row: Mapping[str, object], key: str) -> list[float]:
    vals = row.get(key)
    if not isinstance(vals, list) or len(vals) == 0:
        raise BenchParseError(
            f"llama-bench row missing non-empty '{key}' samples array"
        )
    return [float(v) for v in vals]


def _extract(raw: str) -> tuple[list[float], list[float], float]:
    """Extract ``(decode samples_ts, prefill_ttft_ms, model_size_mb)`` from one raw.

    decode samples = the decode row's ``samples_ts``; prefill TTFT ms = the prefill row's
    ``samples_ns`` divided by 1e6. Raises ``BenchParseError`` on empty or length-mismatched
    sample arrays, or a missing ``model_size``.
    """
    prefill, decode = _find_bench_rows(raw)
    decode_ts = _samples(decode, "samples_ts")
    prefill_ns = _samples(prefill, "samples_ns")

    # Integrity: within a row llama-bench always emits samples_ns and samples_ts pairwise;
    # a length mismatch means a corrupt/truncated capture -- refuse it rather than average junk.
    for row in (prefill, decode):
        ns = row.get("samples_ns")
        ts = row.get("samples_ts")
        if isinstance(ns, list) and isinstance(ts, list) and len(ns) != len(ts):
            raise BenchParseError(
                f"llama-bench row samples_ns/samples_ts length mismatch: {len(ns)} vs {len(ts)}"
            )

    size = decode.get("model_size", prefill.get("model_size"))
    if not isinstance(size, (int, float)):
        raise BenchParseError("llama-bench row missing numeric 'model_size'")

    prefill_ttft_ms = [ns / 1e6 for ns in prefill_ns]
    return decode_ts, prefill_ttft_ms, float(size) / _MIB


def _result_from_samples(
    *,
    config_id: str,
    decode_samples: Sequence[float],
    prefill_ttft_samples: Sequence[float],
    model_size_mb: float,
    quality: QualityScore,
    peak_mem_mb: float | None,
    stage: Stage,
    seed: int = 0,
) -> BenchmarkResult:
    """Assemble a ``BenchmarkResult`` from raw per-repeat samples (single point of stats truth
    shared by ``parse_llama_bench_output`` and ``Benchmarker.confirm_ab``)."""
    return BenchmarkResult(
        config_id=config_id,
        decode_tok_s=median_ci(decode_samples, seed=seed),
        prefill_ttft_ms=median_ci(prefill_ttft_samples, seed=seed),
        quality=quality,
        peak_mem_mb=peak_mem_mb,
        model_size_mb=model_size_mb,
        n_repeats=len(decode_samples),
        stage=stage,
        raw_samples=RawSamples(
            decode_tok_s=tuple(float(x) for x in decode_samples),
            prefill_ttft_ms=tuple(float(x) for x in prefill_ttft_samples),
        ),
    )


def parse_llama_bench_output(
    raw: str,
    *,
    config_id: str,
    stage: Stage,
    quality: QualityScore,
    peak_mem_mb: float | None = None,
) -> BenchmarkResult:
    """Parse a single ``llama-bench -o json`` array (one prefill + one decode row) into a
    ``BenchmarkResult``.

    decode_tok_s uses the decode row's ``samples_ts``; prefill_ttft_ms uses the prefill row's
    ``samples_ns`` / 1e6; both get a median + bootstrap CI via ``median_ci``. ``model_size_mb``
    = ``model_size`` bytes / 1048576. ``n_repeats`` = number of decode samples. Raises
    ``BenchParseError`` if either row is absent or the sample arrays are empty / mismatched.
    """
    decode_ts, prefill_ttft_ms, model_size_mb = _extract(raw)
    return _result_from_samples(
        config_id=config_id,
        decode_samples=decode_ts,
        prefill_ttft_samples=prefill_ttft_ms,
        model_size_mb=model_size_mb,
        quality=quality,
        peak_mem_mb=peak_mem_mb,
        stage=stage,
    )


def _merge_peak(peaks: Sequence[float | None]) -> float | None:
    present = [p for p in peaks if p is not None]
    return max(present) if present else None


# --------------------------------------------------------------------------------------
# Two-stage screen / confirm policy
# --------------------------------------------------------------------------------------
class Benchmarker:
    """Composes the injected ``BenchRunner`` + ``QualityFn`` into the two-stage policy.

    No SSH here: the runner is the only thing that touches the target. ``screen`` is the cheap
    filter; ``confirm`` is a single-config rigorous capture (baseline / expert absolutes);
    ``confirm_ab`` is the interleaved A/B/A/B comparison that must drive every keep/revert
    decision so systematic thermal / neighbour drift cancels (CLAUDE.md rule 3).
    """

    def __init__(
        self,
        run: BenchRunner,
        quality: QualityFn,
        workload: WorkloadSpec,
        *,
        seed: int = 0,
    ) -> None:
        self._run = run
        self._quality = quality
        self._workload = workload
        self._seed = seed

    def screen(self, cfg: BenchConfig) -> BenchmarkResult:
        """Cheap gate. Runs the SAME pp+tg command as confirm (so the parser sees both rows)
        with ``screen_repeats`` repeats; quality is left unmeasured and stage is ``'screen'``.
        Gating is on the decode median only (see ``promote_to_confirm``) -- no CI claim.
        """
        raw, peak = self._run(cfg, self._workload.screen_repeats)
        return parse_llama_bench_output(
            raw,
            config_id=cfg.config_id,
            stage="screen",
            quality=QualityScore(perplexity=None, kl_vs_baseline=None),
            peak_mem_mb=peak,
        )

    def confirm(self, cfg: BenchConfig) -> BenchmarkResult:
        """Single-config rigorous capture (baseline / expert absolute numbers): ``confirm_repeats``
        (>= 5) repeats, measured quality, stage ``'confirm'``. NOT for candidate-vs-incumbent
        keep/revert -- that MUST use ``confirm_ab`` so the comparison is drift-cancelled.
        """
        raw, peak = self._run(cfg, self._workload.confirm_repeats)
        return parse_llama_bench_output(
            raw,
            config_id=cfg.config_id,
            stage="confirm",
            quality=self._quality(cfg),
            peak_mem_mb=peak,
        )

    def confirm_ab(
        self,
        incumbent_cfg: BenchConfig,
        candidate_cfg: BenchConfig,
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """A/B/A/B **interleaved** confirm.

        Alternates one incumbent repeat then one candidate repeat within a single measurement
        window, ``confirm_repeats`` pairs total, so systematic drift over the window hits both
        configs equally and cancels. Returns ``(incumbent_result, candidate_result)`` -- two
        FRESH confirm-stage results the agent feeds to ``significant()``. A sequential
        ``confirm(inc)`` then ``confirm(cand)`` (A/A/A/B/B/B) does NOT interleave and must never
        drive a keep/revert decision (ARCHITECTURE 'Measurement integrity', CLAUDE.md rule 3).
        """
        n = self._workload.confirm_repeats
        inc_decode: list[float] = []
        inc_prefill: list[float] = []
        cand_decode: list[float] = []
        cand_prefill: list[float] = []
        inc_peaks: list[float | None] = []
        cand_peaks: list[float | None] = []
        inc_size: float | None = None
        cand_size: float | None = None

        for _ in range(n):
            raw_inc, peak_inc = self._run(incumbent_cfg, 1)  # A
            raw_cand, peak_cand = self._run(candidate_cfg, 1)  # B
            d_inc, p_inc, size_inc = _extract(raw_inc)
            d_cand, p_cand, size_cand = _extract(raw_cand)
            inc_decode.extend(d_inc)
            inc_prefill.extend(p_inc)
            cand_decode.extend(d_cand)
            cand_prefill.extend(p_cand)
            inc_peaks.append(peak_inc)
            cand_peaks.append(peak_cand)
            if inc_size is None:
                inc_size = size_inc
            if cand_size is None:
                cand_size = size_cand

        assert (
            inc_size is not None and cand_size is not None
        )  # n >= 1 guaranteed by confirm_repeats

        incumbent_result = _result_from_samples(
            config_id=incumbent_cfg.config_id,
            decode_samples=inc_decode,
            prefill_ttft_samples=inc_prefill,
            model_size_mb=inc_size,
            quality=self._quality(incumbent_cfg),
            peak_mem_mb=_merge_peak(inc_peaks),
            stage="confirm",
            seed=self._seed,
        )
        candidate_result = _result_from_samples(
            config_id=candidate_cfg.config_id,
            decode_samples=cand_decode,
            prefill_ttft_samples=cand_prefill,
            model_size_mb=cand_size,
            quality=self._quality(candidate_cfg),
            peak_mem_mb=_merge_peak(cand_peaks),
            stage="confirm",
            seed=self._seed,
        )
        return incumbent_result, candidate_result
