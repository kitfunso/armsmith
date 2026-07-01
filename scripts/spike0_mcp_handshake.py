"""Spike 0a evidence: handshake the armlimited/arm-mcp container over stdio.

Sends MCP initialize + tools/list and prints the tool names, descriptions,
and input schemas. No SSH keys mounted: this only enumerates the tool surface,
proving the server is headless-scriptable from any MCP client (no IDE needed).

Reproduce:
    docker pull armlimited/arm-mcp:latest
    python scripts/spike0_mcp_handshake.py
"""

import json
import subprocess
import sys
import threading

TIMEOUT_S = 90


def main() -> int:
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "-i", "armlimited/arm-mcp:latest"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    timer = threading.Timer(TIMEOUT_S, proc.kill)
    timer.start()

    def send(msg: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def read_response(expect_id: int) -> dict:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("stdout closed before response")
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[non-json stdout] {line[:200]}", file=sys.stderr)
                continue
            if data.get("id") == expect_id:
                return data

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "armsmith-spike0", "version": "0.0.1"},
                },
            }
        )
        init = read_response(1)
        server_info = init.get("result", {}).get("serverInfo", {})
        proto = init.get("result", {}).get("protocolVersion")
        print(f"SERVER: {server_info} protocol={proto}")

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools_resp = read_response(2)
        tools = tools_resp.get("result", {}).get("tools", [])
        print(f"TOOL_COUNT: {len(tools)}")
        for t in tools:
            print("=" * 70)
            print(f"NAME: {t['name']}")
            desc = (t.get("description") or "").strip()
            print(f"DESC: {desc[:600]}")
            schema = t.get("inputSchema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            print(f"PARAMS: {list(props.keys())} required={required}")
        return 0
    finally:
        timer.cancel()
        try:
            proc.kill()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
