"""armsmith CLI command surface.

Typer wiring only (CONTRACTS.md §11) -- every command builds its collaborators
(`Target`, `Benchmarker`, `Profiler`, `Brain`) and calls into the owning
module; no business logic lives here. Downstream modules (`agent.py`,
`brain.py`, `report.py`) are imported lazily inside each command so `--help`
and `--version` keep working while those modules are still being built
elsewhere in the tree (CONTRACTS.md §1 ownership map).

`run_id` linkage (CONTRACTS.md §11): `baseline` MINTS the run id and writes
`manifest.json`/`baseline.json`/`expert.json`; `optimize` resolves an existing
id (default: `trajectories/latest`) and appends to the same run dir;
`report`/`repro` take the run id positionally.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer

from armsmith.models import ArmsmithError

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="armsmith",
    help=(
        "Reproducible Graviton LLM-optimization lab: a deterministic autotuner "
        "plus an LLM Arm Performix analyst."
    ),
    no_args_is_help=True,
    add_completion=False,
)

TRAJECTORIES_DIR = Path("trajectories")
_LATEST_POINTER = TRAJECTORIES_DIR / "latest"
_PROVISION_SCRIPT = Path("scripts") / "provision_graviton.sh"


def _version_callback(value: bool) -> None:
    if value:
        from armsmith import __version__

        typer.echo(f"armsmith {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the armsmith version and exit.",
    ),
) -> None:
    """Reproducible Graviton LLM optimization: deterministic tuner + LLM Performix analyst."""


# --------------------------------------------------------------------------- #
# Shared helpers (CONTRACTS §12 rule 4: CLI is the only place errors become   #
# exits; ArmsmithError -> a clean message, anything else propagates).        #
# --------------------------------------------------------------------------- #
def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def _run_dir(run_id: str) -> Path:
    return TRAJECTORIES_DIR / run_id


def _resolve_run_id(run_id: str) -> str:
    """`run_id` as given, or `trajectories/latest`'s contents when `run_id=='latest'`."""
    if run_id != "latest":
        return run_id
    if not _LATEST_POINTER.exists():
        _fail("no run found: pass --run-id explicitly or run `armsmith baseline` first")
    return _LATEST_POINTER.read_text(encoding="utf-8").strip()


def _mint_run_id(instance_family: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{timestamp}-{instance_family}"


def _load_model_spec(path: str):
    """Load a `ModelSpec` from a small JSON model-registry file:
    ``{"name": ..., "baseline_quant": ..., "variants": {"Q4_0": [remote_path, sha256]}}``.
    Pure I/O (mirrors `models.load_workload`'s YAML loader) -- no business logic."""
    from armsmith.models import ModelDecodeError, ModelSpec

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        variants = {
            quant: (remote_path, sha256)
            for quant, (remote_path, sha256) in data["variants"].items()
        }
        return ModelSpec(
            name=data["name"],
            variants=variants,
            baseline_quant=data["baseline_quant"],
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ModelDecodeError(f"failed to load model spec {path!r}: {exc}") from exc


# --------------------------------------------------------------------------- #
# provision -- prints the documented aws cli checklist (spike0-result.md);   #
# boto3/Terraform is deferred, scripts/provision_graviton.sh is the runnable #
# form of the same sequence.                                                  #
# --------------------------------------------------------------------------- #
_PROVISION_CHECKLIST = (
    "1. IAM: create an `armsmith-provisioner` user with `AmazonEC2FullAccess` "
    "+ an access key (CloudShell); put the keys in `~/.aws/credentials`.",
    "2. Key pair: "
    "`aws ec2 create-key-pair --key-name armsmith --query 'KeyMaterial' "
    "--output text > ~/.ssh/armsmith.pem && chmod 400 ~/.ssh/armsmith.pem`",
    "3. Security group: "
    "`aws ec2 create-security-group --group-name armsmith-sg "
    "--description 'armsmith Graviton target'` then "
    "`aws ec2 authorize-security-group-ingress --port 22 --cidr <home-ip>/32`",
    "4. Launch: `aws ec2 run-instances --image-id <ubuntu-24.04-arm64-ami> "
    "--instance-type INSTANCE_PLACEHOLDER --key-name armsmith --security-group-ids <sg-id> "
    '--block-device-mappings \'[{"DeviceName":"/dev/sda1","Ebs":'
    '{"VolumeSize":64,"VolumeType":"gp3"}}]\' '
    "--user-data '#!/bin/bash\\nshutdown -h +180' "  # auto-shutdown backstop
    "--tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=armsmith}]'`",
    "5. Wait: `aws ec2 wait instance-running --instance-ids <id>`",
    "6. IP: `aws ec2 describe-instances --instance-ids <id> "
    "--query 'Reservations[0].Instances[0].PublicIpAddress' --output text`",
    "7. Verify: `ssh -i ~/.ssh/armsmith.pem ubuntu@<ip>`",
)


@app.command()
def provision(
    instance: str = typer.Option("r8g.4xlarge", help="Graviton instance type."),
    destroy: bool = typer.Option(
        False, help="Tear the target down instead of creating it."
    ),
) -> None:
    """Print the aws cli checklist to stand up (or tear down) a Graviton target.

    Full Terraform/boto3 automation is deferred (docs/plans/2026-07-02-phase-2.md
    Step 10); the exact command sequence -- key pair, security group, run-instances
    with the `shutdown -h +180` auto-shutdown backstop (CLAUDE.md rule 8), wait,
    fetch the IP -- is also runnable directly as `{script}`.
    """
    if destroy:
        typer.echo(
            "Tear down (CLAUDE.md rule 8 -- do this after every session):\n"
            "  aws ec2 describe-instances --filters Name=tag:Name,Values=armsmith "
            "--query 'Reservations[].Instances[].InstanceId' --output text\n"
            "  aws ec2 terminate-instances --instance-ids <id>"
        )
        return
    typer.echo(
        f"Provision checklist for {instance} (or run `bash {_PROVISION_SCRIPT}`):"
    )
    for step in _PROVISION_CHECKLIST:
        typer.echo(f"  {step.replace('INSTANCE_PLACEHOLDER', instance)}")


# --------------------------------------------------------------------------- #
# baseline -- mints the run_id; captures the honest baseline + pinned expert. #
# --------------------------------------------------------------------------- #
@app.command()
def baseline(
    target: str = typer.Option(..., help="SSH host of the Arm target."),
    model: str = typer.Option(..., help="Path or id of the GGUF model."),
    workload: str = typer.Option("examples/bench.yaml", help="Workload spec."),
    user: str = typer.Option("ubuntu", help="SSH user on the target."),
    ssh_key: str = typer.Option(
        ...,
        "--ssh-key",
        envvar="ARMSMITH_SSH_KEY",
        help="Path to the SSH private key for the target.",
    ),
) -> None:
    """Capture the honest naive baseline (portable, GGML_NATIVE=OFF) and the
    pre-registered expert config; mint a `run_id` for `optimize`/`report`/`repro`."""
    from armsmith import __version__
    from armsmith.actions import baseline_config, expert_config
    from armsmith.bench import Benchmarker
    from armsmith.models import RunManifest, load_workload, to_json
    from armsmith.target import Target

    try:
        workload_spec = load_workload(workload)
        model_spec = _load_model_spec(model)

        with Target(target, user, ssh_key, model_spec) as tgt:
            target_spec = tgt.describe()
            base_cfg = baseline_config(workload_spec, model_spec)
            exp_cfg = expert_config(workload_spec, model_spec, target_spec)
            tgt.fetch_model(sorted({base_cfg.quant, exp_cfg.quant}))
            tgt.build(base_cfg)
            tgt.build(exp_cfg)
            # Stage the eval text on the target and capture the base logits ONCE
            # from the honest baseline so the KL quality guard (CLAUDE.md rule 6)
            # is operative: run_quality reads base.kld and needs a remote eval path.
            eval_remote = tgt.upload_eval_text(workload_spec.eval_text_path)
            tgt.capture_base_logits(base_cfg, eval_remote)
            benchmarker = Benchmarker(
                tgt.run_bench,
                lambda cfg: tgt.run_quality(cfg, eval_remote),
                workload_spec,
            )
            baseline_result = benchmarker.confirm(base_cfg)
            expert_result = benchmarker.confirm(exp_cfg)

        instance_family = target_spec.instance_type.split(".")[0] or "target"
        run_id = _mint_run_id(instance_family)
        run_dir = _run_dir(run_id)
        (run_dir / "configs").mkdir(parents=True, exist_ok=True)
        (run_dir / "configs" / f"{base_cfg.config_id}.json").write_text(
            to_json(base_cfg), encoding="utf-8"
        )
        (run_dir / "configs" / f"{exp_cfg.config_id}.json").write_text(
            to_json(exp_cfg), encoding="utf-8"
        )

        manifest = RunManifest(
            run_id=run_id,
            target=target_spec,
            model=model_spec,
            workload_ref=workload,
            baseline_ref=base_cfg.config_id,
            expert_ref=exp_cfg.config_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            armsmith_version=__version__,
        )
        (run_dir / "manifest.json").write_text(to_json(manifest), encoding="utf-8")
        (run_dir / "baseline.json").write_text(
            to_json(baseline_result), encoding="utf-8"
        )
        (run_dir / "expert.json").write_text(to_json(expert_result), encoding="utf-8")
        _LATEST_POINTER.write_text(run_id, encoding="utf-8")

        typer.echo(run_id)
    except ArmsmithError as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------- #
# optimize -- discovery run against an EXISTING run_id.                       #
# --------------------------------------------------------------------------- #
@app.command()
def optimize(
    target: str = typer.Option(..., help="SSH host of the Arm target."),
    model: str = typer.Option(..., help="Path or id of the GGUF model."),
    workload: str = typer.Option("examples/bench.yaml", help="Workload spec."),
    budget: int = typer.Option(20, help="Max tuner iterations."),
    brain: str = typer.Option("claude", help="LLM analyst backend (model-agnostic)."),
    run_id: str = typer.Option(
        "latest",
        "--run-id",
        help="Run id to append to (default: resolve trajectories/latest).",
    ),
    user: str = typer.Option("ubuntu", help="SSH user on the target."),
    ssh_key: str = typer.Option(
        ...,
        "--ssh-key",
        envvar="ARMSMITH_SSH_KEY",
        help="Path to the SSH private key for the target.",
    ),
) -> None:
    """Discovery run: deterministic tuner + LLM Performix analyst -> saved recipe.

    Reads the `manifest`/`baseline`/`expert` `baseline` wrote for `--run-id`
    (CONTRACTS §11) and appends `trajectory.jsonl`/`recipe.json` to that run dir.
    `--model`/`--workload` are unused overrides kept for CLI-shape parity with
    `baseline` -- the manifest's pinned `model`/`workload_ref` are load-bearing.
    """
    del model, workload  # pinned by the manifest; kept for CLI-shape parity

    # Resolve the run dir BEFORE importing the still-in-progress agent/brain/
    # profiler modules (CONTRACTS §1), so a missing --run-id fails cleanly
    # without depending on build order.
    resolved_id = _resolve_run_id(run_id)
    run_dir = _run_dir(resolved_id)
    if not run_dir.exists():
        _fail(f"run {resolved_id!r} not found under {TRAJECTORIES_DIR}/")

    from armsmith import agent, brain as brain_module
    from armsmith.bench import Benchmarker
    from armsmith.models import BenchConfig, BenchmarkResult, RunManifest, from_json
    from armsmith.models import load_workload
    from armsmith.profiler import MCPProfiler, NullProfiler
    from armsmith.target import Target

    try:
        manifest = from_json(
            RunManifest, (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        baseline_result = from_json(
            BenchmarkResult, (run_dir / "baseline.json").read_text(encoding="utf-8")
        )
        expert_result = from_json(
            BenchmarkResult, (run_dir / "expert.json").read_text(encoding="utf-8")
        )
        baseline_cfg = from_json(
            BenchConfig,
            (run_dir / "configs" / f"{manifest.baseline_ref}.json").read_text(
                encoding="utf-8"
            ),
        )
        workload_spec = load_workload(manifest.workload_ref)

        with Target(target, user, ssh_key, manifest.model) as tgt:
            # Stage EVERY GGUF variant the tuner may select (CONTRACTS §5/§7:
            # optimize stages the grid's variants) so a quant_format candidate
            # does not bench against a model file that was never downloaded; and
            # re-stage the eval text so each candidate's quality run can open it.
            tgt.fetch_model(sorted(manifest.model.variants))
            eval_remote = tgt.upload_eval_text(workload_spec.eval_text_path)
            try:
                profiler = MCPProfiler(ssh_key)
            except ArmsmithError:
                logger.warning("MCP profiler unavailable; falling back to NullProfiler")
                profiler = NullProfiler()
            benchmarker = Benchmarker(
                tgt.run_bench,
                lambda cfg: tgt.run_quality(cfg, eval_remote),
                workload_spec,
            )
            recipe = agent.optimize(
                tgt,
                profiler,
                brain_module.brain_from_name(brain),
                benchmarker,
                manifest=manifest,
                baseline=baseline_result,
                baseline_cfg=baseline_cfg,
                expert=expert_result,
                budget=budget,
                trajectory_dir=run_dir,
            )
        typer.echo(
            f"recipe: {recipe.winning_config.config_id} "
            f"decode_tok_s={recipe.winning_result.decode_tok_s.median:.2f} "
            f"gap_closed_pct={recipe.gap_closed_pct}"
        )
    except ArmsmithError as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------- #
# report -- render the self-contained HTML trajectory report.                 #
# --------------------------------------------------------------------------- #
@app.command()
def report(run_id: str = typer.Argument(..., help="Run id to render.")) -> None:
    """Render the visual-trajectory HTML report for a run."""
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        _fail(
            f"run {run_id!r} not found under {TRAJECTORIES_DIR}/ (looked in {run_dir})"
        )

    from armsmith.report import render_report

    try:
        out_path = render_report(run_dir)
    except ArmsmithError as exc:
        _fail(str(exc))
    typer.echo(str(out_path))


# --------------------------------------------------------------------------- #
# repro -- replay a saved recipe deterministically (no brain, no profiler).   #
# --------------------------------------------------------------------------- #
@app.command()
def repro(
    run_id: str = typer.Argument(..., help="Run id whose recipe to replay."),
    target: str = typer.Option(..., help="SSH host of the FRESH Arm target."),
    user: str = typer.Option("ubuntu", help="SSH user on the target."),
    ssh_key: str = typer.Option(
        ...,
        "--ssh-key",
        envvar="ARMSMITH_SSH_KEY",
        help="Path to the SSH private key for the target.",
    ),
    tol_pct: float = typer.Option(
        10.0, help="Allowed decode-tok/s tolerance vs the recorded result."
    ),
) -> None:
    """Replay a saved recipe deterministically (no LLM) on a fresh instance."""
    run_dir = _run_dir(run_id)
    recipe_path = run_dir / "recipe.json"
    if not recipe_path.exists():
        _fail(f"no recipe.json for run {run_id!r} (looked in {recipe_path})")

    from armsmith import agent
    from armsmith.bench import Benchmarker
    from armsmith.models import Recipe, RunManifest, from_json, load_workload
    from armsmith.target import Target

    try:
        recipe = from_json(Recipe, recipe_path.read_text(encoding="utf-8"))
        manifest = from_json(
            RunManifest, (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        workload_spec = load_workload(manifest.workload_ref)

        with Target(target, user, ssh_key, recipe.model) as tgt:
            benchmarker = Benchmarker(
                tgt.run_bench,
                lambda cfg: tgt.run_quality(cfg, workload_spec.eval_text_path),
                workload_spec,
            )
            result = agent.replay(tgt, benchmarker, recipe, tol_pct=tol_pct)
        typer.echo(
            f"PASS: decode_tok_s.median={result.decode_tok_s.median:.2f} "
            f"(recipe={recipe.winning_result.decode_tok_s.median:.2f}, "
            f"tol_pct={tol_pct})"
        )
    except ArmsmithError as exc:
        _fail(f"FAIL: {exc}")


if __name__ == "__main__":
    app()
