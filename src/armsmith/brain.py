"""armsmith LLM analyst: the only module that calls an LLM (CLAUDE.md rule 2).

CONTRACTS.md §8. The brain reads :class:`Evidence` (a Performix snapshot, the
trajectory history so far, and the remaining legal action ids) and returns a
:class:`BrainVerdict`: a reordering of those candidate ids plus at most one
concrete ``{action_id, params}`` suggestion. It never runs anything and never
emits shell (CLAUDE.md rule 2) — every id/param in a model's response is
re-validated through :func:`armsmith.actions.validate_suggestion` before it
can reach the tuner; anything off-registry or off-schema is dropped and
logged, never smuggled downstream. If the LLM is unavailable, unreachable, or
returns something unparsable, :class:`NullBrain` is the honest deterministic
fallback — the tuner still converges without it (CONTRACTS §9 honesty test).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from armsmith.actions import ACTIONS, ValidatedAction, validate_suggestion
from armsmith.models import (
    BenchmarkResult,
    BrainError,
    OffRegistryError,
    ParamValidationError,
    PerformixSnapshot,
    TrajectoryStep,
    to_json,
)

logger = logging.getLogger(__name__)

# Sonnet 4.6, not the skill's Opus-4.8 default: this brain asks for a low,
# near-deterministic temperature (CONTRACTS §8), and `temperature` 400s on
# Opus 4.8/4.7/Sonnet 5/Fable 5 (adaptive-thinking-only models). Override via
# ARMSMITH_BRAIN_MODEL for a different tier.
DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.2

_SYSTEM_PROMPT = """You are the optimization analyst for armsmith, a Graviton \
LLM-inference autotuner. You never run anything and never write shell or code \
- you only read evidence and return a JSON verdict that reorders a FIXED list \
of candidate action ids by how likely each is to help next, based on the \
Performix hotspot snapshot and the trajectory of steps taken so far.

Respond with a single JSON object and nothing else (no markdown fences, no \
commentary before or after) with exactly these keys:
  "priority": an array containing a permutation of (a subset of) the given \
    "candidates" list, most-promising first. Only use ids that appear in \
    "candidates" - never invent an id.
  "rationale": one or two sentences explaining the ordering, grounded in the \
    evidence (hotspot symbols, bottleneck shape, prior kept/reverted steps).
  "suggestion": either null, or a single object {"action_id": <one of \
    "candidates">, "params": {...}} naming the ONE concrete next action you \
    would take and its parameter values, if you have a specific pick beyond \
    just reordering. Omit params you are not sure about.
"""


@dataclass(frozen=True)
class Evidence:
    """Everything the brain sees for one step. Read-only; the brain returns a
    verdict, it never mutates or acts on this."""

    snapshot: PerformixSnapshot
    history: tuple[TrajectoryStep, ...]
    candidates: tuple[str, ...]  # remaining legal action ids (post capability filter)
    baseline: BenchmarkResult
    current: BenchmarkResult


@dataclass(frozen=True)
class BrainVerdict:
    """The brain's advice. `priority` and `suggestion` are ALWAYS validated
    before construction here - nothing off-registry or off-schema can be
    represented by this type."""

    priority: tuple[str, ...]  # reordered subset of candidates; validated ids only
    rationale: str
    suggestion: ValidatedAction | None  # at most one concrete pick; validated or None


class Brain(Protocol):
    def analyze(self, evidence: Evidence) -> BrainVerdict: ...


class NullBrain:
    """Deterministic fallback / honesty test (CONTRACTS §8, §9). Leaves the
    candidate order exactly as given - fixed registry order, since
    `enumerate_candidates` produces `evidence.candidates` in that order.
    Used when `--brain=null`, when no API key is configured, and as
    `AnthropicBrain`'s error fallback."""

    def analyze(self, evidence: Evidence) -> BrainVerdict:
        return BrainVerdict(
            priority=tuple(evidence.candidates),
            rationale="no LLM: deterministic registry order",
            suggestion=None,
        )


class AnthropicBrain:
    """Default brain. Sends `Evidence` as JSON, asks the model for a
    `{priority, rationale, suggestion}` JSON verdict, parses it, and runs
    EVERY id it names through `validate_suggestion` / registry membership -
    dropping anything invalid with a logged warning - before returning a
    `BrainVerdict`. On any API or parse error it delegates to `NullBrain`
    rather than crashing the loop (CONTRACTS §12 rule 2: degrade, don't
    crash, on the optional layers).

    Model id comes from the `model` constructor arg, else `ARMSMITH_BRAIN_MODEL`,
    else `DEFAULT_MODEL`. The API key is resolved by the `anthropic` SDK from
    `ANTHROPIC_API_KEY` (or an `ant auth login` profile) - never read directly
    here. `client` is an injection point for tests (any object exposing
    `.messages.create(...)` with the same shape as `anthropic.Anthropic()`).
    """

    def __init__(self, model: str | None = None, *, client: Any = None) -> None:
        self._model = model or os.environ.get("ARMSMITH_BRAIN_MODEL", DEFAULT_MODEL)
        self._client = client
        self._fallback = NullBrain()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic  # imported lazily so NullBrain-only usage needs no SDK/key

        self._client = anthropic.Anthropic()
        return self._client

    def analyze(self, evidence: Evidence) -> BrainVerdict:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=_DEFAULT_MAX_TOKENS,
                temperature=_DEFAULT_TEMPERATURE,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _render_evidence(evidence)}],
            )
            parsed = _parse_verdict_json(_extract_text(response))
        except (
            Exception
        ):  # noqa: BLE001 - any failure degrades to NullBrain, never crashes the loop
            logger.warning(
                "AnthropicBrain.analyze failed; falling back to NullBrain",
                exc_info=True,
            )
            return self._fallback.analyze(evidence)

        return _validate_verdict(parsed, evidence)


def brain_from_name(name: str) -> Brain:
    """`"claude"` -> `AnthropicBrain()`; `"null"` -> `NullBrain()`; any other
    name -> `NullBrain()` with a logged warning (CONTRACTS §8 wiring rule)."""
    if name == "claude":
        return AnthropicBrain()
    if name == "null":
        return NullBrain()
    logger.warning("brain_from_name: unknown brain %r; falling back to NullBrain", name)
    return NullBrain()


# --------------------------------------------------------------------------- #
# Internal helpers (pure where possible; no shell, no unvalidated execution).    #
# --------------------------------------------------------------------------- #
def _render_evidence(evidence: Evidence) -> str:
    """Serialize `Evidence` to the JSON payload sent as the user turn. Uses
    `models.to_json` so every field renders exactly as the rest of the
    project persists it - no separate ad hoc encoding to drift out of sync."""
    payload = {
        "snapshot": json.loads(to_json(evidence.snapshot)),
        "history": [json.loads(to_json(step)) for step in evidence.history],
        "candidates": list(evidence.candidates),
        "baseline": json.loads(to_json(evidence.baseline)),
        "current": json.loads(to_json(evidence.current)),
    }
    return "Evidence for the next optimization step:\n" + json.dumps(
        payload, indent=2, sort_keys=True
    )


def _extract_text(response: Any) -> str:
    """First `text`-type content block in an Anthropic Messages response."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return text
    raise BrainError("AnthropicBrain: response contained no text block")


def _parse_verdict_json(text: str) -> dict[str, Any]:
    """Parse the model's JSON verdict, tolerating an accidental ```json fence
    around it. Raises `BrainError` (caught by `analyze`) on anything else."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        newline = stripped.find("\n")
        if newline != -1 and stripped[:newline].strip().lower() in ("json", ""):
            stripped = stripped[newline + 1 :]
        stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise BrainError(
            f"AnthropicBrain: could not parse JSON verdict: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise BrainError("AnthropicBrain: verdict JSON is not an object")
    return parsed


def _validate_verdict(parsed: Mapping[str, Any], evidence: Evidence) -> BrainVerdict:
    """THE SAFETY FILTER for LLM output (CLAUDE.md rule 2 / CONTRACTS §8):
    every priority id must be a real registry id AND still a legal candidate
    for this step; the (at most one) suggestion must pass
    `actions.validate_suggestion` in full. Anything that fails is dropped
    with a logged warning - never raised, never smuggled through."""
    candidate_set = set(evidence.candidates)

    raw_priority = parsed.get("priority", [])
    valid_ids: list[str] = []
    if isinstance(raw_priority, list):
        for item in raw_priority:
            if not isinstance(item, str):
                logger.warning(
                    "AnthropicBrain: dropping non-string priority entry %r", item
                )
                continue
            if item not in ACTIONS:
                logger.warning(
                    "AnthropicBrain: dropping off-registry action id %r", item
                )
                continue
            if item not in candidate_set:
                logger.warning(
                    "AnthropicBrain: dropping action id %r not in remaining candidates",
                    item,
                )
                continue
            if item not in valid_ids:
                valid_ids.append(item)
    else:
        logger.warning("AnthropicBrain: 'priority' was not a list; ignoring")

    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale:
        rationale = "AnthropicBrain: no rationale provided"

    suggestion: ValidatedAction | None = None
    raw_suggestion = parsed.get("suggestion")
    if isinstance(raw_suggestion, dict):
        action_id = raw_suggestion.get("action_id")
        params = raw_suggestion.get("params", {})
        if isinstance(action_id, str) and isinstance(params, dict):
            try:
                candidate_suggestion = validate_suggestion(action_id, params)
            except (OffRegistryError, ParamValidationError) as exc:
                logger.warning("AnthropicBrain: dropping invalid suggestion: %s", exc)
            else:
                if candidate_suggestion.action_id in candidate_set:
                    suggestion = candidate_suggestion
                else:
                    logger.warning(
                        "AnthropicBrain: dropping suggestion %r not in remaining candidates",
                        candidate_suggestion.action_id,
                    )
        else:
            logger.warning(
                "AnthropicBrain: malformed suggestion shape %r", raw_suggestion
            )
    elif raw_suggestion is not None:
        logger.warning(
            "AnthropicBrain: 'suggestion' was neither null nor an object; ignoring"
        )

    return BrainVerdict(
        priority=tuple(valid_ids), rationale=rationale, suggestion=suggestion
    )
