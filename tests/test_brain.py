"""Tests for the LLM analyst (CONTRACTS §8).

The load-bearing tests here mirror CLAUDE.md rule 2: any off-registry or
off-schema id the (faked) Anthropic client returns must be filtered out by
`AnthropicBrain` before it reaches a `BrainVerdict`, and `NullBrain` must be a
faithful passthrough of the candidate list so the loop is honest with no LLM
at all. The real `anthropic` client is never invoked - a fake stands in via
the `client=` injection point, matching only the `.messages.create(...)`
shape `brain.py` calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from armsmith.actions import ValidatedAction
from armsmith.brain import (
    AnthropicBrain,
    Evidence,
    NullBrain,
    brain_from_name,
)
from armsmith.models import (
    BenchmarkResult,
    Delta,
    Diagnosis,
    HotspotRow,
    MetricStat,
    PerformixSnapshot,
    QualityScore,
    RawSamples,
    TrajectoryStep,
)

# --------------------------------------------------------------------------- #
# Fixtures: real record instances (full field sets; models.py has no defaults). #
# --------------------------------------------------------------------------- #


def _metric(median: float) -> MetricStat:
    return MetricStat(median=median, ci_low=median * 0.95, ci_high=median * 1.05)


def _result(config_id: str, decode: float) -> BenchmarkResult:
    samples = (decode,) * 5
    return BenchmarkResult(
        config_id=config_id,
        decode_tok_s=_metric(decode),
        prefill_ttft_ms=_metric(120.0),
        quality=QualityScore(perplexity=6.0, kl_vs_baseline=None),
        peak_mem_mb=4096.0,
        model_size_mb=4300.0,
        n_repeats=5,
        stage="confirm",
        raw_samples=RawSamples(decode_tok_s=samples, prefill_ttft_ms=(120.0,) * 5),
    )


def _snapshot() -> PerformixSnapshot:
    return PerformixSnapshot(
        config_id="cfg-base",
        recipe="code_hotspots",
        source="mcp",
        status="success",
        hotspots=(
            HotspotRow(
                symbol="ggml_compute_forward_mul_mat",
                self_samples=900,
                self_pct=61.2,
                node_type="function",
            ),
        ),
    )


def _history() -> tuple[TrajectoryStep, ...]:
    step = TrajectoryStep(
        step_idx=0,
        diagnosis=Diagnosis(
            bottleneck="compute-bound", evidence="mul_mat dominates", source="mcp"
        ),
        action_id="ggml_native",
        params={"state": "ON"},
        rationale="native build wins on compute-bound hotspot",
        before_config_id="cfg-base",
        after_config_id="cfg-native",
        screen=_result("cfg-native", 40.0),
        confirm=_result("cfg-native", 42.0),
        kept=True,
        delta=Delta(metric="decode_tok_s", pct=12.0, ci_significant=True),
        quality_ok=True,
    )
    return (step,)


def _evidence(candidates: tuple[str, ...]) -> Evidence:
    return Evidence(
        snapshot=_snapshot(),
        history=_history(),
        candidates=candidates,
        baseline=_result("cfg-base", 30.0),
        current=_result("cfg-native", 42.0),
    )


# --------------------------------------------------------------------------- #
# Fake Anthropic client: mirrors only `.messages.create(...)` -> object with    #
# `.content` = list of blocks exposing `.type` / `.text`.                       #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list[Any]


class _FakeMessages:
    def __init__(
        self, reply_text: str | None = None, *, raises: Exception | None = None
    ) -> None:
        self._reply_text = reply_text
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        assert self._reply_text is not None
        return _FakeResponse(content=[_FakeTextBlock(text=self._reply_text)])


class _FakeClient:
    def __init__(
        self, reply_text: str | None = None, *, raises: Exception | None = None
    ) -> None:
        self.messages = _FakeMessages(reply_text, raises=raises)


def _brain_with_reply(payload: dict[str, Any]) -> tuple[AnthropicBrain, _FakeClient]:
    client = _FakeClient(json.dumps(payload))
    return AnthropicBrain(model="claude-sonnet-4-6", client=client), client


# --------------------------------------------------------------------------- #
# NullBrain: the honest no-LLM passthrough.                                     #
# --------------------------------------------------------------------------- #


def test_null_brain_passthrough_preserves_candidate_order() -> None:
    candidates = ("kleidiai", "threads", "kv_cache_type")
    verdict = NullBrain().analyze(_evidence(candidates))

    assert verdict.priority == candidates
    assert verdict.rationale == "no LLM: deterministic registry order"
    assert verdict.suggestion is None


def test_null_brain_passthrough_empty_candidates() -> None:
    verdict = NullBrain().analyze(_evidence(()))
    assert verdict.priority == ()


# --------------------------------------------------------------------------- #
# AnthropicBrain: happy path + the safety-filtering (CLAUDE.md rule 2).         #
# --------------------------------------------------------------------------- #


def test_anthropic_brain_happy_path_returns_priority_and_rationale() -> None:
    candidates = ("kleidiai", "threads", "kv_cache_type")
    brain, client = _brain_with_reply(
        {
            "priority": ["threads", "kleidiai", "kv_cache_type"],
            "rationale": "threads first since the hotspot is compute-bound",
            "suggestion": None,
        }
    )

    verdict = brain.analyze(_evidence(candidates))

    assert verdict.priority == ("threads", "kleidiai", "kv_cache_type")
    assert verdict.rationale == "threads first since the hotspot is compute-bound"
    assert verdict.suggestion is None
    # exactly one call, with the model/temperature the constructor was given
    assert len(client.messages.calls) == 1
    assert client.messages.calls[0]["model"] == "claude-sonnet-4-6"
    assert client.messages.calls[0]["temperature"] == pytest.approx(0.2)


def test_anthropic_brain_drops_off_registry_priority_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = ("kleidiai", "threads")
    brain, _ = _brain_with_reply(
        {
            "priority": ["threads", "delete_the_disk", "kleidiai"],
            "rationale": "ok",
            "suggestion": None,
        }
    )

    with caplog.at_level(logging.WARNING):
        verdict = brain.analyze(_evidence(candidates))

    assert verdict.priority == ("threads", "kleidiai")
    assert "delete_the_disk" not in verdict.priority
    assert any("off-registry" in rec.message for rec in caplog.records)


def test_anthropic_brain_drops_priority_id_not_in_remaining_candidates() -> None:
    # 'quant_format' is a real registry id, but not offered as a candidate
    # for this step (already applied / capability-filtered) -> must be dropped.
    candidates = ("kleidiai", "threads")
    brain, _ = _brain_with_reply(
        {
            "priority": ["quant_format", "threads"],
            "rationale": "ok",
            "suggestion": None,
        }
    )

    verdict = brain.analyze(_evidence(candidates))

    assert verdict.priority == ("threads",)


def test_anthropic_brain_valid_suggestion_is_validated_action() -> None:
    candidates = ("threads",)
    brain, _ = _brain_with_reply(
        {
            "priority": ["threads"],
            "rationale": "bump threads",
            "suggestion": {
                "action_id": "threads",
                "params": {"n_threads": 16, "cpu_mask": "physical"},
            },
        }
    )

    verdict = brain.analyze(_evidence(candidates))

    assert verdict.suggestion == ValidatedAction(
        action_id="threads", params={"n_threads": 16, "cpu_mask": "physical"}
    )


def test_anthropic_brain_drops_off_registry_suggestion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = ("threads",)
    brain, _ = _brain_with_reply(
        {
            "priority": ["threads"],
            "rationale": "ok",
            "suggestion": {"action_id": "rm_rf_root", "params": {}},
        }
    )

    with caplog.at_level(logging.WARNING):
        verdict = brain.analyze(_evidence(candidates))

    assert verdict.suggestion is None
    assert any("invalid suggestion" in rec.message for rec in caplog.records)


def test_anthropic_brain_drops_out_of_schema_suggestion_param() -> None:
    candidates = ("threads",)
    brain, _ = _brain_with_reply(
        {
            "priority": ["threads"],
            "rationale": "ok",
            "suggestion": {"action_id": "threads", "params": {"n_threads": 999999}},
        }
    )

    verdict = brain.analyze(_evidence(candidates))

    # priority is still honored even though the suggestion was rejected
    assert verdict.priority == ("threads",)
    assert verdict.suggestion is None


def test_anthropic_brain_drops_suggestion_not_in_remaining_candidates() -> None:
    candidates = ("threads",)  # kleidiai not offered this step
    brain, _ = _brain_with_reply(
        {
            "priority": ["threads"],
            "rationale": "ok",
            "suggestion": {
                "action_id": "kleidiai",
                "params": {"state": "ON", "sme": "0"},
            },
        }
    )

    verdict = brain.analyze(_evidence(candidates))
    assert verdict.suggestion is None


def test_anthropic_brain_falls_back_to_null_brain_on_client_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = ("threads", "kleidiai")
    client = _FakeClient(raises=RuntimeError("network exploded"))
    brain = AnthropicBrain(model="claude-sonnet-4-6", client=client)

    with caplog.at_level(logging.WARNING):
        verdict = brain.analyze(_evidence(candidates))

    assert verdict.priority == candidates
    assert verdict.rationale == "no LLM: deterministic registry order"
    assert verdict.suggestion is None
    assert any("falling back to NullBrain" in rec.message for rec in caplog.records)


def test_anthropic_brain_falls_back_on_unparsable_json() -> None:
    candidates = ("threads",)
    client = _FakeClient("not json at all { totally broken")
    brain = AnthropicBrain(model="claude-sonnet-4-6", client=client)

    verdict = brain.analyze(_evidence(candidates))

    assert verdict.priority == candidates
    assert verdict.rationale == "no LLM: deterministic registry order"


def test_anthropic_brain_tolerates_markdown_fenced_json() -> None:
    candidates = ("threads",)
    fenced = (
        "```json\n" + json.dumps({"priority": ["threads"], "rationale": "ok"}) + "\n```"
    )
    client = _FakeClient(fenced)
    brain = AnthropicBrain(model="claude-sonnet-4-6", client=client)

    verdict = brain.analyze(_evidence(candidates))

    assert verdict.priority == ("threads",)
    assert verdict.rationale == "ok"


def test_anthropic_brain_uses_env_model_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARMSMITH_BRAIN_MODEL", "claude-haiku-4-5")
    client = _FakeClient(json.dumps({"priority": [], "rationale": "ok"}))
    brain = AnthropicBrain(client=client)

    brain.analyze(_evidence(()))

    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# brain_from_name wiring rule (CONTRACTS §8).                                   #
# --------------------------------------------------------------------------- #


def test_brain_from_name_maps_known_names() -> None:
    assert isinstance(brain_from_name("null"), NullBrain)
    assert isinstance(brain_from_name("claude"), AnthropicBrain)


def test_brain_from_name_falls_back_to_null_on_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        brain = brain_from_name("something-made-up")

    assert isinstance(brain, NullBrain)
    assert any("unknown brain" in rec.message for rec in caplog.records)
