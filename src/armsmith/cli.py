"""armsmith CLI command surface.

Commands are stubs at the scaffold stage; each is implemented in its phase
(see docs/plans/). The command shape here is the contract the rest of the
package builds against.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="armsmith",
    help=(
        "Reproducible Graviton LLM-optimization lab: a deterministic autotuner "
        "plus an LLM Arm Performix analyst."
    ),
    no_args_is_help=True,
    add_completion=False,
)

_STUB = "not implemented yet (scaffold stage; see docs/plans/)"


@app.command()
def provision(
    instance: str = typer.Option("r8g.4xlarge", help="Graviton instance type."),
    destroy: bool = typer.Option(False, help="Tear the target down instead of creating it."),
) -> None:
    """Stand up (or tear down) a Graviton target."""
    typer.echo(f"provision (instance={instance}, destroy={destroy}): {_STUB}")


@app.command()
def baseline(
    target: str = typer.Option(..., help="SSH host of the Arm target."),
    model: str = typer.Option(..., help="Path or id of the GGUF model."),
    workload: str = typer.Option("examples/bench.yaml", help="Workload spec."),
) -> None:
    """Capture the honest naive baseline and pin the pre-registered expert config."""
    typer.echo(f"baseline (target={target}, model={model}, workload={workload}): {_STUB}")


@app.command()
def optimize(
    target: str = typer.Option(..., help="SSH host of the Arm target."),
    model: str = typer.Option(..., help="Path or id of the GGUF model."),
    workload: str = typer.Option("examples/bench.yaml", help="Workload spec."),
    budget: int = typer.Option(20, help="Max tuner iterations."),
    brain: str = typer.Option("claude", help="LLM analyst backend (model-agnostic)."),
) -> None:
    """Discovery run: deterministic tuner + LLM Performix analyst -> saved recipe."""
    typer.echo(f"optimize (target={target}, brain={brain}, budget={budget}): {_STUB}")


@app.command()
def report(run_id: str = typer.Argument(..., help="Run id to render.")) -> None:
    """Render the visual-trajectory HTML report for a run."""
    typer.echo(f"report (run_id={run_id}): {_STUB}")


@app.command()
def repro(run_id: str = typer.Argument(..., help="Run id whose recipe to replay.")) -> None:
    """Replay a saved recipe deterministically (no LLM) on a fresh instance."""
    typer.echo(f"repro (run_id={run_id}): {_STUB}")


if __name__ == "__main__":
    app()
