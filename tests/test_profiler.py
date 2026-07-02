"""Tests for armsmith.profiler: structuredContent parsing, MCP stdio client, NullProfiler.

MCPProfiler is exercised against a fake stdio subprocess (JSON-RPC id matching);
no docker is spawned in these tests. Parsing tests use the real captured
fixtures in tests/fixtures/ (no invented fixture data).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from armsmith.models import MCPError
from armsmith.models import BenchConfig, TargetSpec
from armsmith.profiler import (
    DEFAULT_RECIPE,
    MCPProfiler,
    NullProfiler,
    parse_structured_content,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def target_spec() -> TargetSpec:
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


@pytest.fixture
def bench_config() -> BenchConfig:
    return BenchConfig.create(
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


# --- parse_structured_content ------------------------------------------------


def test_parse_structured_content_portable_fixture() -> None:
    sc = _load_fixture("hotspots_portable.json")
    snap = parse_structured_content(sc, config_id="cfg123", source="mcp")

    assert snap.config_id == "cfg123"
    assert snap.recipe == "code_hotspots"
    assert snap.status == "success"
    assert snap.source == "mcp"
    assert len(snap.hotspots) == 10
    assert snap.hotspots[0].symbol == "Unknown symbol @ 0x00081dd0"
    assert snap.hotspots[0].self_pct == 37.5
    assert snap.cache_miss_rate is None
    assert snap.mem_bandwidth_gbps is None
    assert snap.ipc is None
    assert snap.raw_columns == tuple(sc["columns"])


def test_parse_structured_content_native_fixture() -> None:
    sc = _load_fixture("hotspots_native.json")
    snap = parse_structured_content(sc, config_id="cfg456", source="mcp")

    assert len(snap.hotspots) == 10
    assert snap.status == "success"
    assert snap.hotspots[0].self_samples == 3


def test_parse_structured_content_error_fixture_does_not_raise() -> None:
    sc = _load_fixture("hotspots_error.json")
    snap = parse_structured_content(sc, config_id="cfg789", source="mcp")

    assert snap.status == "error"
    assert snap.hotspots == ()
    assert snap.warnings  # stderr folded into warnings
    assert any("Insufficient counters" in w for w in snap.warnings)


def test_parse_structured_content_tolerates_missing_node_type() -> None:
    sc = {
        "status": "success",
        "recipe": "code_hotspots",
        "columns": ["FUNCTION_NAME"],
        "rows": [
            {
                "FUNCTION_NAME": "foo",
                "PERIODIC_SAMPLES_SELF": 1,
                "PERIODIC_SAMPLES_SELF_PERCENT": 100.0,
            }
        ],
        "warnings": [],
        "stderr": "",
    }
    snap = parse_structured_content(sc, config_id="cfg", source="cli")
    assert snap.hotspots[0].node_type == "function"
    assert snap.warnings == ()  # empty stderr not folded in


def test_parse_structured_content_tolerates_missing_required_keys() -> None:
    """A sparse/malformed payload (row without FUNCTION_NAME/PERCENT, no top-level
    recipe/status) must NOT raise KeyError -- the profiler is a degradable layer
    (CONTRACTS §12 rule 2); it yields a status='error' snapshot the loop degrades
    on instead of crashing the whole optimize run."""
    sc = {"rows": [{"PERIODIC_SAMPLES_SELF": 3}]}
    snap = parse_structured_content(sc, config_id="cfg", source="mcp")

    assert snap.status == "error"
    assert snap.recipe == DEFAULT_RECIPE
    assert len(snap.hotspots) == 1
    assert snap.hotspots[0].symbol == "Unknown symbol"
    assert snap.hotspots[0].self_samples == 3
    assert snap.hotspots[0].self_pct == 0.0


# --- NullProfiler --------------------------------------------------------------


def test_null_profiler_returns_empty_success_snapshot(
    target_spec, bench_config
) -> None:
    snap = NullProfiler().snapshot(
        target_spec, bench_config, workload_cmd="anything -p 512"
    )

    assert snap.status == "success"
    assert snap.source == "cli"
    assert snap.hotspots == ()
    assert snap.config_id == bench_config.config_id
    assert snap.recipe == DEFAULT_RECIPE


# --- MCPProfiler._docker_host_path ---------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("C:\\Users\\me\\.ssh", "/c/Users/me/.ssh"),
        ("C:/Users/me/.ssh", "/c/Users/me/.ssh"),
        ("/home/me/.ssh", "/home/me/.ssh"),
        ("d:\\keys", "/d/keys"),
    ],
)
def test_docker_host_path_normalizes_windows_drive(raw: str, expected: str) -> None:
    assert MCPProfiler._docker_host_path(raw) == expected


def test_mcpprofiler_builds_docker_cmd_with_keys_dir_mount() -> None:
    profiler = MCPProfiler(ssh_key_path="C:\\Users\\me\\.ssh\\armsmith.pem")
    assert "/c/Users/me/.ssh:/run/keys/:ro" in profiler._docker_cmd
    assert profiler._docker_cmd[-1] == "armlimited/arm-mcp:latest"


# --- MCPProfiler.snapshot over a fake stdio subprocess -------------------------


class _FakeMCPProcess:
    """Stand-in for subprocess.Popen satisfying MCPProfiler's stdio protocol.

    Queues a canned JSON-RPC response line per request id; readline() drains
    the queue and returns "" (EOF) once exhausted / for unmatched ids.
    """

    def __init__(self, responses: dict[int, dict[str, Any]]) -> None:
        self._responses = responses
        self._out_queue: list[str] = []
        self.stdin = self
        self.stdout = self
        self.stderr = _FakeStderr("")
        self.killed = False

    def write(self, data: str) -> None:
        for line in data.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            msg_id = msg.get("id")
            if msg_id in self._responses:
                self._out_queue.append(json.dumps(self._responses[msg_id]) + "\n")

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        if self._out_queue:
            return self._out_queue.pop(0)
        return ""

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


class _FakeStderr:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


def _install_fake_proc(
    monkeypatch: pytest.MonkeyPatch, fake_proc: _FakeMCPProcess
) -> None:
    import armsmith.profiler as profiler_module

    monkeypatch.setattr(profiler_module.subprocess, "Popen", lambda *a, **k: fake_proc)


def test_mcpprofiler_snapshot_parses_structured_content(
    monkeypatch: pytest.MonkeyPatch, target_spec, bench_config
) -> None:
    portable = _load_fixture("hotspots_portable.json")
    fake_proc = _FakeMCPProcess(
        {
            1: {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "arm-mcp"},
                },
            },
            2: {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"structuredContent": portable},
            },
        }
    )
    _install_fake_proc(monkeypatch, fake_proc)

    result = MCPProfiler(ssh_key_path="/fake/.ssh/id_rsa").snapshot(
        target_spec, bench_config, workload_cmd="llama-bench -p 512 -n 128"
    )

    expected = parse_structured_content(
        portable, config_id=bench_config.config_id, source="mcp"
    )
    assert result == expected
    assert fake_proc.killed  # process torn down after the call


def test_mcpprofiler_snapshot_returns_error_status_without_raising(
    monkeypatch: pytest.MonkeyPatch, target_spec, bench_config
) -> None:
    error_sc = _load_fixture("hotspots_error.json")
    fake_proc = _FakeMCPProcess(
        {
            1: {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
            2: {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": error_sc}},
        }
    )
    _install_fake_proc(monkeypatch, fake_proc)

    result = MCPProfiler(ssh_key_path="/fake/.ssh/id_rsa").snapshot(
        target_spec, bench_config, workload_cmd="llama-bench -p 512 -n 128"
    )

    assert result.status == "error"
    assert result.hotspots == ()


def test_mcpprofiler_snapshot_raises_mcperror_on_stdout_close(
    monkeypatch: pytest.MonkeyPatch, target_spec, bench_config
) -> None:
    # No response queued for id=1 -> readline() returns "" immediately -> transport failure.
    fake_proc = _FakeMCPProcess({})
    _install_fake_proc(monkeypatch, fake_proc)

    with pytest.raises(MCPError):
        MCPProfiler(ssh_key_path="/fake/.ssh/id_rsa").snapshot(
            target_spec, bench_config, workload_cmd="llama-bench -p 512 -n 128"
        )


def test_mcpprofiler_snapshot_raises_mcperror_on_jsonrpc_error(
    monkeypatch: pytest.MonkeyPatch, target_spec, bench_config
) -> None:
    fake_proc = _FakeMCPProcess(
        {
            1: {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32000, "message": "handshake rejected"},
            },
        }
    )
    _install_fake_proc(monkeypatch, fake_proc)

    with pytest.raises(MCPError):
        MCPProfiler(ssh_key_path="/fake/.ssh/id_rsa").snapshot(
            target_spec, bench_config, workload_cmd="llama-bench -p 512 -n 128"
        )


def test_mcpprofiler_snapshot_raises_mcperror_when_no_structured_content(
    monkeypatch: pytest.MonkeyPatch, target_spec, bench_config
) -> None:
    fake_proc = _FakeMCPProcess(
        {
            1: {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
            2: {"jsonrpc": "2.0", "id": 2, "result": {}},
        }
    )
    _install_fake_proc(monkeypatch, fake_proc)

    with pytest.raises(MCPError):
        MCPProfiler(ssh_key_path="/fake/.ssh/id_rsa").snapshot(
            target_spec, bench_config, workload_cmd="llama-bench -p 512 -n 128"
        )
