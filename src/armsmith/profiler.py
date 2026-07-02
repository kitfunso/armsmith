"""Performix client: MCP stdio client (docker) -> `PerformixSnapshot`.

`code_hotspots` is the only recipe that returns usable data on a virtualized
Graviton target (spike0); counter fields (`cache_miss_rate`, `mem_bandwidth_gbps`,
`ipc`) stay `None`. This module must not import `target.py` (CONTRACTS.md §1) —
callers build `workload_cmd` themselves and pass it in.

Reference handshake pattern: scripts/spike0_mcp_handshake.py.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from typing import Mapping, Protocol

from armsmith.models import MCPError, ProfilerUnavailable
from armsmith.models import (
    BenchConfig,
    HotspotRow,
    PerformixSnapshot,
    Source,
    TargetSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_RECIPE = "code_hotspots"
_DOCKER_IMAGE = "armlimited/arm-mcp:latest"
_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "armsmith", "version": "0.0.1"}


class Profiler(Protocol):
    def snapshot(
        self,
        target: TargetSpec,
        cfg: BenchConfig,
        *,
        workload_cmd: str,
        recipe: str = DEFAULT_RECIPE,
    ) -> PerformixSnapshot: ...


def parse_structured_content(
    sc: Mapping[str, object], *, config_id: str, source: Source
) -> PerformixSnapshot:
    """Pure parse of an `apx_recipe_run` `structuredContent` payload.

    Unit-tested against tests/fixtures/hotspots_*.json. Counter fields are
    left `None` (the `code_hotspots` recipe returns hotspots only).

    MISSING keys are tolerated at every level (CONTRACTS §3.5: hotspot rows are
    "sparse on short runs"; §12 rule 2: the profiler must not crash the loop).
    Every row/top-level access uses `.get` with a safe default, so a sparse or
    malformed payload yields a `status='error'` snapshot the loop degrades on
    rather than a raw `KeyError` that escapes the (unwrapped) call site.
    """
    rows = sc.get("rows") or ()
    hotspots = tuple(
        HotspotRow(
            symbol=row.get("FUNCTION_NAME", "Unknown symbol"),
            self_samples=row.get("PERIODIC_SAMPLES_SELF", 0),
            self_pct=row.get("PERIODIC_SAMPLES_SELF_PERCENT", 0.0),
            node_type=row.get("NODE_TYPE", "function"),
        )
        for row in rows
    )
    warnings = tuple(sc.get("warnings") or ())
    stderr = sc.get("stderr")
    if stderr:
        warnings = warnings + (stderr,)
    return PerformixSnapshot(
        config_id=config_id,
        recipe=sc.get("recipe", DEFAULT_RECIPE),
        source=source,
        status=sc.get("status", "error"),
        hotspots=hotspots,
        raw_columns=tuple(sc.get("columns") or ()),
        warnings=warnings,
    )


class MCPProfiler:
    """Spawns the Arm MCP server as a docker stdio subprocess and calls `apx_recipe_run`."""

    DOCKER_CMD = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        "{keys_dir}:/run/keys/:ro",
        _DOCKER_IMAGE,
    ]

    def __init__(self, ssh_key_path: str, timeout_s: int = 300) -> None:
        self._ssh_key_path = ssh_key_path
        self._timeout_s = timeout_s
        keys_dir = _posix_dirname(ssh_key_path)
        host_keys_dir = self._docker_host_path(keys_dir)
        self._docker_cmd = [
            part.format(keys_dir=host_keys_dir) for part in self.DOCKER_CMD
        ]
        self._proc: subprocess.Popen[str] | None = None

    @staticmethod
    def _docker_host_path(p: str) -> str:
        """Normalize a host path for a Docker Desktop bind mount.

        `C:\\Users\\me\\.ssh` -> `/c/Users/me/.ssh` (POSIX paths pass through).
        """
        normalized = p.replace("\\", "/")
        if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
            drive, rest = normalized[0], normalized[2:]
            return f"/{drive.lower()}{rest}"
        return normalized

    def snapshot(
        self,
        target: TargetSpec,
        cfg: BenchConfig,
        *,
        workload_cmd: str,
        recipe: str = DEFAULT_RECIPE,
    ) -> PerformixSnapshot:
        self._proc = self._spawn()
        timer = threading.Timer(self._timeout_s, self._proc.kill)
        timer.start()
        try:
            init = self._rpc(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
                msg_id=1,
            )
            if "error" in init:
                raise MCPError(f"MCP initialize failed: {init['error']}")
            self._notify("notifications/initialized", {})
            call = self._rpc(
                "tools/call",
                {
                    "name": "apx_recipe_run",
                    "arguments": {
                        "cmd": workload_cmd,
                        "remote_ip_addr": target.host,
                        "remote_usr": target.user,
                        "recipe": recipe,
                        "invocation_reason": cfg.config_id,
                    },
                },
                msg_id=2,
            )
            if "error" in call:
                raise MCPError(f"apx_recipe_run failed: {call['error']}")
            result = call.get("result", {})
        except OSError as exc:
            raise MCPError(f"MCP transport failure: {exc}") from exc
        finally:
            timer.cancel()
            self._close(self._proc)
            self._proc = None

        try:
            structured_content = result["structuredContent"]
        except (KeyError, TypeError) as exc:
            raise MCPError(
                f"apx_recipe_run returned no structuredContent: {result!r}"
            ) from exc
        return parse_structured_content(
            structured_content, config_id=cfg.config_id, source="mcp"
        )

    def _rpc(self, method: str, params: dict, *, msg_id: int) -> dict:
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        return self._read(expect_id=msg_id)

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _read(self, *, expect_id: int) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise MCPError(
                    f"MCP stdout closed before response id={expect_id}; stderr={stderr[:2000]}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("non-JSON MCP stdout line: %s", line[:200])
                continue
            if data.get("id") == expect_id:
                return data

    def _spawn(self) -> subprocess.Popen[str]:
        try:
            return subprocess.Popen(
                self._docker_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            raise ProfilerUnavailable(
                f"failed to start {self._docker_cmd[0]!r}: {exc}"
            ) from exc

    @staticmethod
    def _close(proc: subprocess.Popen[str]) -> None:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass


def _posix_dirname(p: str) -> str:
    """`dirname` that works on Windows- or POSIX-style path strings regardless of host OS."""
    normalized = p.replace("\\", "/").rstrip("/")
    head, _, _tail = normalized.rpartition("/")
    return head or "."


class NullProfiler:
    """No Performix. Lets the loop run 'try-measure-keep-best' where Performix is unavailable."""

    def snapshot(
        self,
        target: TargetSpec,
        cfg: BenchConfig,
        *,
        workload_cmd: str,
        recipe: str = DEFAULT_RECIPE,
    ) -> PerformixSnapshot:
        return PerformixSnapshot(
            config_id=cfg.config_id,
            recipe=recipe,
            source="cli",
            status="success",
            hotspots=(),
        )
