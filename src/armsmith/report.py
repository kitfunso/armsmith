"""Self-contained HTML report render (CONTRACTS.md §10).

Loads the on-disk run artifacts (manifest, baseline, expert, trajectory,
recipe) into a `ReportModel` and renders ONE self-contained `report.html`:
inline CSS/JS only, no external assets, no network calls, no CDN links -
portable and CSP-safe. Every number shown comes from the dataclasses already
written by `bench.py`/`agent.py`; this module formats, it never recomputes a
statistic (median, CI, significance).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment

from armsmith.models import (
    BenchmarkResult,
    Delta,
    MetricStat,
    Recipe,
    RunManifest,
    TrajectoryStep,
    from_json,
    read_jsonl,
)

# --------------------------------------------------------------------------
# §10 ReportModel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportModel:
    manifest: RunManifest
    baseline: BenchmarkResult
    expert: BenchmarkResult | None
    recipe: Recipe
    steps: tuple[TrajectoryStep, ...]
    per_lever: tuple[tuple[str, Delta], ...]  # kept-lever contributions, in apply order


def build_report_model(run_dir: Path) -> ReportModel:
    """Load manifest.json, baseline.json, expert.json, trajectory.jsonl, recipe.json."""
    run_dir = Path(run_dir)
    manifest = from_json(
        RunManifest, (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    baseline = from_json(
        BenchmarkResult, (run_dir / "baseline.json").read_text(encoding="utf-8")
    )
    expert_path = run_dir / "expert.json"
    expert = (
        from_json(BenchmarkResult, expert_path.read_text(encoding="utf-8"))
        if expert_path.exists()
        else None
    )
    trajectory_path = run_dir / "trajectory.jsonl"
    steps = tuple(read_jsonl(trajectory_path)) if trajectory_path.exists() else ()
    recipe = from_json(Recipe, (run_dir / "recipe.json").read_text(encoding="utf-8"))
    per_lever = tuple((step.action_id, step.delta) for step in steps if step.kept)
    return ReportModel(
        manifest=manifest,
        baseline=baseline,
        expert=expert,
        recipe=recipe,
        steps=steps,
        per_lever=per_lever,
    )


# --------------------------------------------------------------------------
# Formatting helpers (display only - never recompute a statistic)
# --------------------------------------------------------------------------


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def _fmt_ci(stat: MetricStat, digits: int = 2) -> str:
    return (
        f"{stat.median:.{digits}f} (95% CI {stat.ci_low:.{digits}f} "
        f"to {stat.ci_high:.{digits}f})"
    )


def _fmt_quality(result: BenchmarkResult) -> str:
    q = result.quality
    parts = []
    if q.perplexity is not None:
        parts.append(f"perplexity {q.perplexity:.3f}")
    if q.kl_vs_baseline is not None:
        parts.append(f"KL vs baseline {q.kl_vs_baseline:.4f}")
    return ", ".join(parts) if parts else "not measured"


def _bar_bounds(stats: list[MetricStat]) -> tuple[float, float]:
    """Local min/max across ci_low/ci_high of the given stats, for bar scaling."""
    lows = [s.ci_low for s in stats]
    highs = [s.ci_high for s in stats]
    lo, hi = min(lows), max(highs)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _bar_style(stat: MetricStat, lo: float, hi: float) -> str:
    span = hi - lo
    left_pct = 100.0 * (stat.ci_low - lo) / span
    width_pct = 100.0 * (stat.ci_high - stat.ci_low) / span
    marker_pct = 100.0 * (stat.median - lo) / span
    return (
        f"left:{left_pct:.2f}%;width:{max(width_pct, 0.5):.2f}%;"
        f"--marker:{marker_pct:.2f}%;"
    )


def _delta_class(delta: Delta) -> str:
    if not delta.ci_significant:
        return "neutral"
    lower_is_better = "prefill" in delta.metric or "ttft" in delta.metric
    improved = delta.pct < 0 if lower_is_better else delta.pct > 0
    return "good" if improved else "bad"


def _kept_class(kept: bool) -> str:
    return "kept" if kept else "reverted"


def _kept_label(kept: bool) -> str:
    return "kept" if kept else "reverted"


# --------------------------------------------------------------------------
# Template (inline - no packaging risk, no external assets)
# --------------------------------------------------------------------------

_TEMPLATE_SRC = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>armsmith report: {{ manifest.run_id }}</title>
<style>
  :root {
    --bg: #0f1216; --panel: #171b21; --border: #2a303a; --text: #e6e9ee;
    --muted: #9aa4b2; --good: #3ecf8e; --bad: #ef6461; --neutral: #6f7a8a;
    --accent: #5b9dff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem; background: var(--bg); color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
  .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
  .metric-cell { min-width: 220px; }
  .bar-track { position: relative; height: 10px; background: #232935; border-radius: 5px; margin-top: 0.3rem; }
  .bar-fill { position: absolute; top: 0; height: 100%; background: var(--accent); border-radius: 5px; }
  .bar-fill::after {
    content: ""; position: absolute; left: var(--marker); top: -3px; width: 2px; height: 16px;
    background: var(--text);
  }
  .delta { font-weight: 600; }
  .delta.good { color: var(--good); }
  .delta.bad { color: var(--bad); }
  .delta.neutral { color: var(--neutral); }
  .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
  .badge.kept { background: rgba(62,207,142,0.15); color: var(--good); }
  .badge.reverted { background: rgba(239,100,97,0.15); color: var(--bad); }
  .step { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
  .step-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
  .step-title { font-weight: 600; }
  .step-meta { color: var(--muted); font-size: 0.85rem; }
  .rationale { margin-top: 0.6rem; font-style: italic; color: var(--muted); }
  .evidence { margin-top: 0.4rem; font-size: 0.85rem; color: var(--muted); }
  .params { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.85rem; color: var(--text); }
  .lever-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.5rem 0; }
  .lever-name { width: 160px; flex: none; font-size: 0.9rem; }
  .lever-bar-track { flex: 1; height: 12px; background: #232935; border-radius: 6px; position: relative; overflow: hidden; }
  .lever-bar-fill { position: absolute; top: 0; bottom: 0; left: 50%; background: var(--good); }
  .lever-bar-fill.neg { background: var(--bad); }
  .lever-pct { width: 70px; text-align: right; font-variant-numeric: tabular-nums; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
  .summary-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
  .summary-card .label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
  .summary-card .value { font-size: 1.4rem; font-weight: 700; margin-top: 0.25rem; }
  .gap { font-size: 1.1rem; margin-top: 1rem; }
  .gap .value { color: var(--accent); font-weight: 700; }
  footer { margin-top: 3rem; color: var(--muted); font-size: 0.8rem; }
</style>
</head>
<body>

<h1>armsmith optimization report</h1>
<div class="sub">
  run: {{ manifest.run_id }} - target: {{ manifest.target.instance_type }}
  ({{ manifest.target.core }}) - model: {{ manifest.model.name }} - generated by armsmith {{ manifest.armsmith_version }}
</div>

<h2>1. Before / after</h2>
<table>
  <thead>
    <tr><th>Metric</th><th>Baseline</th><th>Winning</th>{% if expert %}<th>Expert</th>{% endif %}<th>Delta (baseline to winning)</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Decode (tok/s, higher is better)</td>
      <td class="metric-cell">
        {{ fmt_ci(baseline.decode_tok_s) }}
        <div class="bar-track"><div class="bar-fill" style="{{ bar_style(baseline.decode_tok_s, decode_lo, decode_hi) }}"></div></div>
      </td>
      <td class="metric-cell">
        {{ fmt_ci(recipe.winning_result.decode_tok_s) }}
        <div class="bar-track"><div class="bar-fill" style="{{ bar_style(recipe.winning_result.decode_tok_s, decode_lo, decode_hi) }}"></div></div>
      </td>
      {% if expert %}
      <td class="metric-cell">
        {{ fmt_ci(expert.decode_tok_s) }}
        <div class="bar-track"><div class="bar-fill" style="{{ bar_style(expert.decode_tok_s, decode_lo, decode_hi) }}"></div></div>
      </td>
      {% endif %}
      <td class="delta {{ decode_delta_class }}">{{ decode_delta_pct }}</td>
    </tr>
    <tr>
      <td>Prefill TTFT (ms, lower is better)</td>
      <td class="metric-cell">
        {{ fmt_ci(baseline.prefill_ttft_ms) }}
        <div class="bar-track"><div class="bar-fill" style="{{ bar_style(baseline.prefill_ttft_ms, ttft_lo, ttft_hi) }}"></div></div>
      </td>
      <td class="metric-cell">
        {{ fmt_ci(recipe.winning_result.prefill_ttft_ms) }}
        <div class="bar-track"><div class="bar-fill" style="{{ bar_style(recipe.winning_result.prefill_ttft_ms, ttft_lo, ttft_hi) }}"></div></div>
      </td>
      {% if expert %}
      <td class="metric-cell">
        {{ fmt_ci(expert.prefill_ttft_ms) }}
        <div class="bar-track"><div class="bar-fill" style="{{ bar_style(expert.prefill_ttft_ms, ttft_lo, ttft_hi) }}"></div></div>
      </td>
      {% endif %}
      <td class="delta {{ ttft_delta_class }}">{{ ttft_delta_pct }}</td>
    </tr>
    <tr>
      <td>Quality</td>
      <td>{{ fmt_quality(baseline) }}</td>
      <td>{{ fmt_quality(recipe.winning_result) }}</td>
      {% if expert %}<td>{{ fmt_quality(expert) }}</td>{% endif %}
      <td>-</td>
    </tr>
    <tr>
      <td>Peak memory (MiB)</td>
      <td>{{ fmt_num(baseline.peak_mem_mb) }}</td>
      <td>{{ fmt_num(recipe.winning_result.peak_mem_mb) }}</td>
      {% if expert %}<td>{{ fmt_num(expert.peak_mem_mb) }}</td>{% endif %}
      <td>-</td>
    </tr>
    <tr>
      <td>Model size (MiB)</td>
      <td>{{ fmt_num(baseline.model_size_mb, 1) }}</td>
      <td>{{ fmt_num(recipe.winning_result.model_size_mb, 1) }}</td>
      {% if expert %}<td>{{ fmt_num(expert.model_size_mb, 1) }}</td>{% endif %}
      <td>-</td>
    </tr>
  </tbody>
</table>
<div class="gap">
  Gap closed vs the pre-registered expert config:
  <span class="value">{{ gap_closed_pct }}</span>
</div>

<h2>2. Trajectory timeline</h2>
{% if steps %}
  {% for row in step_rows %}
  <div class="step">
    <div class="step-head">
      <div class="step-title">Step {{ row.step_idx }}: {{ row.action_id }}</div>
      <span class="badge {{ row.kept_class }}">{{ row.kept_label }}</span>
    </div>
    <div class="step-meta">params: <span class="params">{{ row.params }}</span></div>
    <div class="evidence">
      diagnosis: {{ row.bottleneck }} ({{ row.diagnosis_source }}) - {{ row.evidence }}
    </div>
    <table>
      <thead><tr><th></th><th>Decode delta</th><th>CI significant</th><th>Quality ok</th></tr></thead>
      <tbody>
        <tr>
          <td>screen to confirm</td>
          <td class="delta {{ row.delta_class }}">{{ row.delta_pct }}</td>
          <td>{{ row.ci_significant }}</td>
          <td>{{ row.quality_ok }}</td>
        </tr>
      </tbody>
    </table>
    <div class="rationale">rationale: {{ row.rationale }}</div>
  </div>
  {% endfor %}
{% else %}
  <p class="sub">No trajectory steps recorded for this run.</p>
{% endif %}

<h2>3. Per-lever contribution</h2>
{% if per_lever_rows %}
  {% for lever in per_lever_rows %}
  <div class="lever-row">
    <div class="lever-name">{{ lever.action_id }}</div>
    <div class="lever-bar-track">
      <div class="lever-bar-fill {{ lever.bar_class }}" style="{{ lever.bar_style }}"></div>
    </div>
    <div class="lever-pct delta {{ lever.delta_class }}">{{ lever.delta_pct }}</div>
  </div>
  {% endfor %}
{% else %}
  <p class="sub">No levers were kept in this run.</p>
{% endif %}

<h2>4. Size x speed x quality</h2>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Decode speedup</div>
    <div class="value">{{ decode_delta_pct }}</div>
  </div>
  <div class="summary-card">
    <div class="label">Prefill TTFT change</div>
    <div class="value">{{ ttft_delta_pct }}</div>
  </div>
  <div class="summary-card">
    <div class="label">Model size change</div>
    <div class="value">{{ size_delta_pct }}</div>
  </div>
  <div class="summary-card">
    <div class="label">Quality</div>
    <div class="value">{{ fmt_quality(recipe.winning_result) }}</div>
  </div>
</div>

<footer>
  Generated by armsmith {{ manifest.armsmith_version }} - run {{ manifest.run_id }} - created {{ manifest.created_at }}.
  This file is self-contained: no external stylesheets, scripts, or network calls.
</footer>

</body>
</html>
"""


def _step_view(step: TrajectoryStep) -> dict:
    result = step.confirm if step.confirm is not None else step.screen
    if result is not None:
        delta_pct = _fmt_pct(step.delta.pct)
        ci_sig = "yes" if step.delta.ci_significant else "no"
    else:
        delta_pct = "n/a"
        ci_sig = "no"
    return {
        "step_idx": step.step_idx,
        "action_id": step.action_id,
        "params": ", ".join(f"{k}={v}" for k, v in step.params.items()) or "none",
        "kept_class": _kept_class(step.kept),
        "kept_label": _kept_label(step.kept),
        "bottleneck": step.diagnosis.bottleneck,
        "diagnosis_source": step.diagnosis.source,
        "evidence": step.diagnosis.evidence,
        "delta_class": _delta_class(step.delta),
        "delta_pct": delta_pct,
        "ci_significant": ci_sig,
        "quality_ok": "yes" if step.quality_ok else "no",
        "rationale": step.rationale,
    }


def _lever_view(action_id: str, delta: Delta) -> dict:
    magnitude = min(abs(delta.pct), 100.0)
    return {
        "action_id": action_id,
        "delta_class": _delta_class(delta),
        "delta_pct": _fmt_pct(delta.pct),
        "bar_class": "neg" if delta.pct < 0 else "",
        "bar_style": f"width:{magnitude / 2:.2f}%;"
        + (
            "left:calc(50% - " + f"{magnitude / 2:.2f}%);"
            if delta.pct < 0
            else "left:50%;"
        ),
    }


def render_report(run_dir: Path, *, out: Path | None = None) -> Path:
    """Render ONE self-contained HTML report for the run at `run_dir`.

    Inline CSS/JS, no external assets - portable and CSP-safe. Default
    `out` = `run_dir/'report.html'`. Returns the path written.
    """
    run_dir = Path(run_dir)
    model = build_report_model(run_dir)

    baseline_decode = model.baseline.decode_tok_s
    winning_decode = model.recipe.winning_result.decode_tok_s
    baseline_ttft = model.baseline.prefill_ttft_ms
    winning_ttft = model.recipe.winning_result.prefill_ttft_ms

    decode_stats = [baseline_decode, winning_decode]
    ttft_stats = [baseline_ttft, winning_ttft]
    if model.expert is not None:
        decode_stats.append(model.expert.decode_tok_s)
        ttft_stats.append(model.expert.prefill_ttft_ms)
    decode_lo, decode_hi = _bar_bounds(decode_stats)
    ttft_lo, ttft_hi = _bar_bounds(ttft_stats)

    decode_delta_pct_val = (
        100.0
        * (winning_decode.median - baseline_decode.median)
        / baseline_decode.median
        if baseline_decode.median
        else None
    )
    ttft_delta_pct_val = (
        100.0 * (winning_ttft.median - baseline_ttft.median) / baseline_ttft.median
        if baseline_ttft.median
        else None
    )
    size_delta_pct_val = (
        100.0
        * (model.recipe.winning_result.model_size_mb - model.baseline.model_size_mb)
        / model.baseline.model_size_mb
        if model.baseline.model_size_mb
        else None
    )

    decode_delta = Delta(
        metric="decode_tok_s",
        pct=decode_delta_pct_val or 0.0,
        ci_significant=not (
            baseline_decode.ci_high >= winning_decode.ci_low
            and winning_decode.ci_high >= baseline_decode.ci_low
        ),
    )
    ttft_delta = Delta(
        metric="prefill_ttft_ms",
        pct=ttft_delta_pct_val or 0.0,
        ci_significant=not (
            baseline_ttft.ci_high >= winning_ttft.ci_low
            and winning_ttft.ci_high >= baseline_ttft.ci_low
        ),
    )

    env = Environment(autoescape=True)
    template = env.from_string(_TEMPLATE_SRC)
    html = template.render(
        manifest=model.manifest,
        baseline=model.baseline,
        expert=model.expert,
        recipe=model.recipe,
        steps=model.steps,
        step_rows=[_step_view(s) for s in model.steps],
        per_lever_rows=[_lever_view(a, d) for a, d in model.per_lever],
        fmt_num=_fmt_num,
        fmt_pct=_fmt_pct,
        fmt_ci=_fmt_ci,
        fmt_quality=_fmt_quality,
        bar_style=_bar_style,
        decode_lo=decode_lo,
        decode_hi=decode_hi,
        ttft_lo=ttft_lo,
        ttft_hi=ttft_hi,
        decode_delta_pct=_fmt_pct(decode_delta_pct_val),
        decode_delta_class=_delta_class(decode_delta),
        ttft_delta_pct=_fmt_pct(ttft_delta_pct_val),
        ttft_delta_class=_delta_class(ttft_delta),
        size_delta_pct=_fmt_pct(size_delta_pct_val),
        gap_closed_pct=_fmt_pct(model.recipe.gap_closed_pct),
    )

    out_path = out if out is not None else run_dir / "report.html"
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path
