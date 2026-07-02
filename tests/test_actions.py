"""Tests for the 5-lever action registry and the safety gate (CONTRACTS §5).

The load-bearing tests here are the safety ones (CLAUDE.md rule 2): an off-registry
action id and a schema-violating param must each be REJECTED by
``validate_suggestion`` *before* any command is ever rendered from an
``ActionSpec.apply`` template. A spy stands in for the executor and asserts it was
never touched on the rejection path.
"""

from __future__ import annotations

import re

import pytest

from armsmith.actions import (
    ACTIONS,
    REGISTRY,
    ValidatedAction,
    apply_to_config,
    baseline_config,
    capabilities_ok,
    expert_config,
    validate_suggestion,
)
from armsmith.models import (
    BenchConfig,
    ModelSpec,
    OffRegistryError,
    ParamValidationError,
    TargetSpec,
    WorkloadSpec,
    build_key,
)

EXPECTED_IDS = {"ggml_native", "kleidiai", "quant_format", "threads", "kv_cache_type"}
_SLOT = re.compile(r"\{(\w+)\}")


# --------------------------------------------------------------------------- #
# Fixtures: real model / target / workload records (full contract field sets,   #
# so the tests pass against the real models.py, which has no defaults).         #
# --------------------------------------------------------------------------- #
def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
        screen_repeats=3,
        confirm_repeats=7,
        eval_text_path="examples/eval.txt",
        prompt=None,
    )


def _model() -> ModelSpec:
    return ModelSpec(
        name="Qwen2.5-7B-Instruct",
        variants={
            "Q4_0": ("~/m/qwen-q4_0.gguf", "a" * 64),
            "Q8_0": ("~/m/qwen-q8_0.gguf", "b" * 64),
            "Q4_K_M": ("~/m/qwen-q4_k_m.gguf", "c" * 64),
        },
        baseline_quant="Q4_0",
    )


def _target(capabilities: tuple[str, ...] = ("sve2", "bf16", "i8mm")) -> TargetSpec:
    return TargetSpec(
        host="10.0.0.1",
        user="ubuntu",
        instance_type="r8g.4xlarge",
        core="Neoverse V2 (Graviton4)",
        region="eu-west-2",
        kernel="6.8.0-1000-aws",
        cpu_governor="performance",
        n_physical_cores=16,
        capabilities=capabilities,
    )


class _Executor:
    """Spy standing in for target.py's command renderer. Records every call so a
    test can prove NO command was rendered on the rejection path."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def render(self, template: str, params: object) -> str:
        self.calls.append((template, params))
        return template


def _tuner_apply(executor: _Executor, action_id: str, params: dict) -> str:
    """Mimic the tuner/target contract: VALIDATE first, and only then render a
    command from the (now trusted) action. If validation raises, the executor is
    never reached."""
    validated = validate_suggestion(action_id, params)
    return executor.render(ACTIONS[validated.action_id].apply, validated.params)


# --------------------------------------------------------------------------- #
# Registry shape.                                                               #
# --------------------------------------------------------------------------- #
def test_registry_has_exactly_five_levers() -> None:
    assert len(REGISTRY) == 5
    assert {spec.id for spec in REGISTRY} == EXPECTED_IDS


def test_actions_index_matches_registry() -> None:
    assert set(ACTIONS) == EXPECTED_IDS
    assert len(ACTIONS) == 5
    for spec in REGISTRY:
        assert ACTIONS[spec.id] is spec


def test_every_spec_is_well_formed() -> None:
    for spec in REGISTRY:
        assert spec.apply, spec.id
        assert spec.revert, spec.id
        assert spec.params_schema, spec.id
        assert isinstance(spec.preconditions, tuple), spec.id
        assert isinstance(spec.sets, tuple) and spec.sets, spec.id
        assert spec.kind in ("build", "runtime"), spec.id


def test_kinds_match_the_contract_table() -> None:
    assert ACTIONS["ggml_native"].kind == "build"
    assert ACTIONS["kleidiai"].kind == "build"
    assert ACTIONS["quant_format"].kind == "build"
    assert ACTIONS["threads"].kind == "runtime"
    assert ACTIONS["kv_cache_type"].kind == "runtime"


def test_kleidiai_precondition_is_i8mm() -> None:
    assert ACTIONS["kleidiai"].preconditions == ("i8mm",)


def test_templates_use_named_placeholders_only() -> None:
    """No free-text interpolation: every ``{slot}`` in an apply/revert template
    must be a declared schema param, so a slot can only ever be filled from a
    validated param (CLAUDE.md rule 2 / CONTRACTS §3.6)."""
    for spec in REGISTRY:
        for template in (spec.apply, spec.revert):
            for slot in _SLOT.findall(template):
                assert slot in spec.params_schema, (spec.id, template, slot)


# --------------------------------------------------------------------------- #
# THE safety gate (load-bearing — CLAUDE.md rule 2).                            #
# --------------------------------------------------------------------------- #
def test_off_registry_id_rejected_before_any_command_runs() -> None:
    executor = _Executor()
    with pytest.raises(OffRegistryError):
        _tuner_apply(executor, "rm_rf", {})
    assert executor.calls == []  # nothing was ever rendered


def test_schema_violating_param_rejected_before_any_command_runs() -> None:
    executor = _Executor()
    with pytest.raises(ParamValidationError):
        _tuner_apply(executor, "threads", {"n_threads": 9999})
    assert executor.calls == []  # nothing was ever rendered


@pytest.mark.parametrize("bad_id", ["rm_rf", "sudo", "", "GGML_NATIVE", "native"])
def test_off_registry_ids_raise(bad_id: str) -> None:
    with pytest.raises(OffRegistryError):
        validate_suggestion(bad_id, {})


def test_unknown_param_key_raises() -> None:
    with pytest.raises(ParamValidationError):
        validate_suggestion("ggml_native", {"state": "ON", "shell": "rm -rf /"})


def test_enum_value_off_schema_raises() -> None:
    with pytest.raises(ParamValidationError):
        validate_suggestion("kv_cache_type", {"type_k": "int4"})
    with pytest.raises(ParamValidationError):
        validate_suggestion("ggml_native", {"state": "maybe"})
    with pytest.raises(ParamValidationError):
        validate_suggestion("quant_format", {"quant": "Q2_K"})


def test_int_out_of_range_raises() -> None:
    with pytest.raises(ParamValidationError):
        validate_suggestion("threads", {"n_threads": 0})  # below min
    with pytest.raises(ParamValidationError):
        validate_suggestion("threads", {"n_threads": 1025})  # above max


def test_bool_is_not_accepted_as_int() -> None:
    with pytest.raises(ParamValidationError):
        validate_suggestion("threads", {"n_threads": True})


# --------------------------------------------------------------------------- #
# Valid suggestions pass and canonicalize.                                      #
# --------------------------------------------------------------------------- #
def test_valid_suggestion_returns_validated_action() -> None:
    validated = validate_suggestion("ggml_native", {"state": "on"})
    assert isinstance(validated, ValidatedAction)
    assert validated.action_id == "ggml_native"
    assert validated.params["state"] == "ON"  # canonicalized to declared spelling


def test_enum_canonicalization_is_case_folded() -> None:
    assert (
        validate_suggestion("quant_format", {"quant": "q4_0"}).params["quant"] == "Q4_0"
    )
    kv = validate_suggestion(
        "kv_cache_type", {"type_k": "F16", "type_v": "Q8_0", "flash_attn": "ON"}
    )
    assert kv.params == {"type_k": "f16", "type_v": "q8_0", "flash_attn": "on"}


def test_int_and_int_like_sme_accepted() -> None:
    threads = validate_suggestion("threads", {"n_threads": 16, "cpu_mask": "physical"})
    assert threads.params == {"n_threads": 16, "cpu_mask": "physical"}
    # sme is an enum over "0"/"1"; an int is coerced to its string choice.
    kleidiai = validate_suggestion("kleidiai", {"state": "ON", "sme": 1})
    assert kleidiai.params == {"state": "ON", "sme": "1"}


# --------------------------------------------------------------------------- #
# apply_to_config: deterministic candidate generation.                          #
# --------------------------------------------------------------------------- #
def test_apply_ggml_native_changes_flags_and_ids() -> None:
    base = baseline_config(_workload(), _model())
    candidate = apply_to_config(
        validate_suggestion("ggml_native", {"state": "on"}), base
    )
    assert "-DGGML_NATIVE=ON" in candidate.cmake_flags
    assert "-DGGML_NATIVE=OFF" not in candidate.cmake_flags
    assert candidate.config_id != base.config_id
    assert build_key(candidate) != build_key(base)


def test_apply_does_not_mutate_the_input_config() -> None:
    base = baseline_config(_workload(), _model())
    before = base.config_id
    apply_to_config(validate_suggestion("ggml_native", {"state": "ON"}), base)
    assert base.config_id == before
    assert "-DGGML_NATIVE=OFF" in base.cmake_flags


def test_apply_kv_cache_coerces_flash_attn_to_bool() -> None:
    base = baseline_config(_workload(), _model())
    candidate = apply_to_config(
        validate_suggestion(
            "kv_cache_type", {"type_k": "q8_0", "type_v": "q8_0", "flash_attn": "on"}
        ),
        base,
    )
    assert candidate.type_k == "q8_0"
    assert candidate.type_v == "q8_0"
    assert candidate.flash_attn is True

    off = apply_to_config(
        validate_suggestion("kv_cache_type", {"flash_attn": "off"}), base
    )
    assert off.flash_attn is False


def test_apply_threads_resolves_default_mask_to_none() -> None:
    base = baseline_config(_workload(), _model())
    candidate = apply_to_config(
        validate_suggestion("threads", {"n_threads": 8, "cpu_mask": "default"}), base
    )
    assert candidate.n_threads == 8
    assert candidate.cpu_mask is None


def test_apply_quant_swaps_the_variant() -> None:
    base = baseline_config(_workload(), _model())
    candidate = apply_to_config(
        validate_suggestion("quant_format", {"quant": "Q8_0"}), base
    )
    assert candidate.quant == "Q8_0"
    assert build_key(candidate) != build_key(base)  # quant is a rebuild key


def test_apply_kleidiai_sets_cmake_and_env() -> None:
    base = baseline_config(_workload(), _model())
    candidate = apply_to_config(
        validate_suggestion("kleidiai", {"state": "ON", "sme": "0"}), base
    )
    assert "-DGGML_CPU_KLEIDIAI=ON" in candidate.cmake_flags
    assert ("GGML_KLEIDIAI_SME", "0") in candidate.env


def test_levers_stack_onto_the_incumbent() -> None:
    """Coordinate ascent: applying a second lever onto a config that already kept
    one preserves both (CONTRACTS §9)."""
    base = baseline_config(_workload(), _model())
    step1 = apply_to_config(validate_suggestion("ggml_native", {"state": "ON"}), base)
    step2 = apply_to_config(
        validate_suggestion("threads", {"n_threads": 16, "cpu_mask": "physical"}), step1
    )
    assert "-DGGML_NATIVE=ON" in step2.cmake_flags
    assert step2.n_threads == 16


# --------------------------------------------------------------------------- #
# capabilities_ok.                                                              #
# --------------------------------------------------------------------------- #
def test_capabilities_ok_gates_on_preconditions() -> None:
    assert (
        capabilities_ok(ACTIONS["kleidiai"], _target(("sve2", "bf16", "i8mm"))) is True
    )
    assert capabilities_ok(ACTIONS["kleidiai"], _target(("sve2", "bf16"))) is False


def test_capabilities_ok_is_true_when_no_preconditions() -> None:
    for action_id in ("ggml_native", "quant_format", "threads", "kv_cache_type"):
        assert capabilities_ok(ACTIONS[action_id], _target(())) is True


# --------------------------------------------------------------------------- #
# Reference configs.                                                            #
# --------------------------------------------------------------------------- #
def test_baseline_config_is_the_honest_portable_build() -> None:
    base = baseline_config(_workload(), _model())
    assert isinstance(base, BenchConfig)
    assert "-DGGML_NATIVE=OFF" in base.cmake_flags
    assert base.quant == "Q4_0"
    assert base.type_k == "f16"
    assert base.type_v == "f16"
    assert base.n_prompt == 512 and base.n_gen == 128


def test_expert_config_is_native_plus_kleidiai_on_v2() -> None:
    exp = expert_config(_workload(), _model(), _target(("sve2", "bf16", "i8mm")))
    assert "-DGGML_NATIVE=ON" in exp.cmake_flags
    assert "-DGGML_CPU_KLEIDIAI=ON" in exp.cmake_flags
    assert exp.n_threads == 16
    assert exp.env == ()  # r8g has no sme2, so no SME env
    base = baseline_config(_workload(), _model())
    assert exp.config_id != base.config_id


def test_expert_config_skips_kleidiai_without_i8mm() -> None:
    exp = expert_config(_workload(), _model(), _target(("sve2", "bf16")))
    assert "-DGGML_NATIVE=ON" in exp.cmake_flags
    assert "-DGGML_CPU_KLEIDIAI=ON" not in exp.cmake_flags


def test_expert_config_enables_sme_only_with_sme2() -> None:
    exp = expert_config(
        _workload(), _model(), _target(("sve2", "bf16", "i8mm", "sme2"))
    )
    assert ("GGML_KLEIDIAI_SME", "1") in exp.env
