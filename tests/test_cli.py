"""Tests for armsmith.cli -- the typer wiring surface (CONTRACTS.md §11).

Scope kept to what cli.py itself owns and can assert without a live Graviton
target or the still-in-progress `agent.py`/`brain.py`/`report.py` modules
(CONTRACTS §1 ownership map): the command surface exists, `--version` works,
and `report` fails cleanly (no traceback, exit 1) on a missing run dir --
that check happens in cli.py before `report.py` is ever imported, so it holds
regardless of build order.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from armsmith import __version__
from armsmith.cli import app

runner = CliRunner()


def test_help_lists_all_five_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("provision", "baseline", "optimize", "report", "repro"):
        assert command in result.output


def test_version_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    # `no_args_is_help=True` prints help; click's exit code for that path is 2
    # (missing-command), not 0 -- assert on the help content, not the code.
    result = runner.invoke(app, [])
    assert "provision" in result.output
    assert "baseline" in result.output


def test_report_errors_cleanly_on_missing_run_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["report", "no-such-run-id"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "no-such-run-id" in result.output
    assert "not found" in result.output


def test_repro_errors_cleanly_on_missing_recipe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "repro",
            "no-such-run-id",
            "--target",
            "1.2.3.4",
            "--ssh-key",
            "/dev/null",
        ],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "no-such-run-id" in result.output


def test_optimize_errors_cleanly_on_missing_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "--target",
            "1.2.3.4",
            "--model",
            "unused",
            "--run-id",
            "no-such-run-id",
            "--ssh-key",
            "/dev/null",
        ],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "no-such-run-id" in result.output


def test_provision_prints_checklist() -> None:
    result = runner.invoke(app, ["provision"])
    assert result.exit_code == 0
    assert "aws ec2" in result.output
    assert "r8g.4xlarge" in result.output


def test_provision_destroy_prints_teardown() -> None:
    result = runner.invoke(app, ["provision", "--destroy"])
    assert result.exit_code == 0
    assert "terminate-instances" in result.output
