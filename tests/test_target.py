"""Tests for armsmith.target.Target — the paramiko SSH executor.

All tests inject a fake paramiko-shaped transport at the boundary (`Target(...,
client=...)`); no real network/SSH is ever opened. Commands are asserted
exactly against what the fixed templates + a known `BenchConfig` should render.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from armsmith.models import BenchError, BuildError, TargetError
from armsmith.models import BenchConfig, ModelSpec, build_key
from armsmith.target import (
    CommandResult,
    Target,
    _expand_home,
    _parse_lscpu,
    _parse_peak_mem_mb,
    _remote_home,
)

# ---------------------------------------------------------------------------
# Fake paramiko transport double
# ---------------------------------------------------------------------------


class _FakeChannel:
    def __init__(self, exit_code: int) -> None:
        self._exit_code = exit_code

    def recv_exit_status(self) -> int:
        return self._exit_code


class _FakeChannelFile:
    def __init__(self, text: str, exit_code: int) -> None:
        self._data = text.encode("utf-8")
        self.channel = _FakeChannel(exit_code)

    def read(self) -> bytes:
        return self._data


class _FakeSFTP:
    def __init__(self) -> None:
        self.put_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.closed = False

    def put(self, local_path: str, remote_path: str) -> None:
        self.put_calls.append((local_path, remote_path))

    def get(self, remote_path: str, local_path: str) -> None:
        self.get_calls.append((remote_path, local_path))

    def close(self) -> None:
        self.closed = True


class FakeSSHClient:
    """Scripted stand-in for paramiko.SSHClient. Maps an exact rendered
    command string to (exit_code, stdout, stderr); records every call."""

    def __init__(
        self,
        responses: dict[str, tuple[int, str, str]] | None = None,
        default: tuple[int, str, str] = (0, "", ""),
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []
        self.closed = False
        self._sftp = _FakeSFTP()

    def exec_command(self, command: str, timeout: int | None = None):
        self.calls.append(command)
        exit_code, stdout_text, stderr_text = self.responses.get(command, self.default)
        stdout = _FakeChannelFile(stdout_text, exit_code)
        stderr = _FakeChannelFile(stderr_text, exit_code)
        return None, stdout, stderr

    def open_sftp(self) -> _FakeSFTP:
        return self._sftp

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The tilde form is what callers pass; Target expands a leading `~` to an
# absolute home in __init__ (shlex.join quotes `~`, so a tilde left in a remote
# path is never expanded by the login shell). Commands are asserted against the
# EXPANDED paths for user "ubuntu".
REMOTE_ROOT_ARG = "~/armsmith-work"
REMOTE_ROOT = "/home/ubuntu/armsmith-work"
REPO_DIR = f"{REMOTE_ROOT}/llama.cpp"
MODEL_REMOTE = "/home/ubuntu/models/qwen-q4_0.gguf"


@pytest.fixture
def model_spec() -> ModelSpec:
    return ModelSpec(
        name="Qwen2.5-7B-Instruct",
        variants={"Q4_0": ("~/models/qwen-q4_0.gguf", "a" * 64)},
        baseline_quant="Q4_0",
    )


@pytest.fixture
def cfg() -> BenchConfig:
    return BenchConfig.create(
        cmake_flags=("-DGGML_NATIVE=ON", "-DGGML_CPU_KLEIDIAI=OFF"),
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


def make_target(model_spec: ModelSpec, client: FakeSSHClient) -> Target:
    return Target(
        "1.2.3.4",
        "ubuntu",
        "/tmp/armsmith.pem",
        model_spec,
        remote_root=REMOTE_ROOT_ARG,
        client=client,
    )


# ---------------------------------------------------------------------------
# _run: timeout plumbing + explicit error on nonzero exit
# ---------------------------------------------------------------------------


def test_run_returns_command_result_on_success(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient({"uname -r": (0, "6.8.0-generic\n", "")})
    target = make_target(model_spec, fake)

    result = target._run(["uname", "-r"])

    assert result == CommandResult(
        argv=("uname", "-r"), exit_code=0, stdout="6.8.0-generic\n", stderr=""
    )
    assert fake.calls == ["uname -r"]


def test_run_passes_timeout_through_to_exec_command(model_spec: ModelSpec) -> None:
    seen_timeouts: list[int | None] = []
    fake = FakeSSHClient({"uname -r": (0, "ok", "")})
    orig_exec = fake.exec_command

    def spy_exec(command: str, timeout: int | None = None):
        seen_timeouts.append(timeout)
        return orig_exec(command, timeout=timeout)

    fake.exec_command = spy_exec  # type: ignore[method-assign]
    target = make_target(model_spec, fake)

    target._run(["uname", "-r"], timeout=45)

    assert seen_timeouts == [45]


def test_run_raises_target_error_on_nonzero_exit(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient(default=(1, "", "boom: no such file"))
    target = make_target(model_spec, fake)

    with pytest.raises(TargetError, match="boom: no such file"):
        target._run(["false"])


def test_run_raises_custom_error_cls_on_nonzero_exit(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient(default=(2, "", "compile failed"))
    target = make_target(model_spec, fake)

    with pytest.raises(BuildError):
        target._run(["cmake", "--build", "x"], error_cls=BuildError)


def test_run_raises_target_error_when_not_connected(model_spec: ModelSpec) -> None:
    target = Target("1.2.3.4", "ubuntu", "/tmp/key.pem", model_spec)

    with pytest.raises(TargetError):
        target._run(["uname", "-r"])


# ---------------------------------------------------------------------------
# bench_command / run_bench: exact argv rendering from validated fields only
# ---------------------------------------------------------------------------


def test_bench_command_renders_exact_argv(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    argv = target.bench_command(cfg)

    expected_bin = f"{REPO_DIR}/build-{build_key(cfg)}/bin/llama-bench"
    assert argv == (
        expected_bin,
        "-m",
        MODEL_REMOTE,
        "-p",
        "512",
        "-n",
        "128",
        "-b",
        "2048",
        "-ub",
        "512",
        "-t",
        "16",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-o",
        "json",
    )


def test_bench_command_omits_thread_flag_for_engine_default(
    model_spec: ModelSpec,
) -> None:
    """n_threads == 0 means engine default: `-t` must be OMITTED. A literal
    `-t 0` aborts ggml with GGML_ASSERT(cplan->n_threads > 0) - observed on a
    real r8g, 2026-07-02."""
    cfg_default_threads = BenchConfig.create(
        cmake_flags=("-DGGML_NATIVE=OFF",),
        quant="Q4_0",
        n_threads=0,
        cpu_mask=None,
        type_k="f16",
        type_v="f16",
        flash_attn=None,
        env=(),
        n_prompt=256,
        n_gen=64,
        n_batch=2048,
        n_ubatch=512,
    )
    target = make_target(model_spec, FakeSSHClient())

    argv = target.bench_command(cfg_default_threads)

    assert "-t" not in argv


def test_bench_command_includes_cpu_mask_and_flash_attn_when_set(
    model_spec: ModelSpec,
) -> None:
    cfg_with_extras = BenchConfig.create(
        cmake_flags=(),
        quant="Q4_0",
        n_threads=8,
        cpu_mask="0xFFFF",
        type_k="q8_0",
        type_v="q8_0",
        flash_attn=True,
        env=(),
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
    )
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    argv = target.bench_command(cfg_with_extras)

    assert "--cpu-mask" in argv
    assert argv[argv.index("--cpu-mask") + 1] == "0xFFFF"
    assert "-fa" in argv
    assert argv[argv.index("-fa") + 1] == "1"


def test_run_bench_wraps_bench_command_with_time_and_env_and_repeats(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)
    expected_argv = (
        "env",
        "/usr/bin/time",
        "-v",
        *target.bench_command(cfg),
        "-r",
        "3",
    )
    expected_command = shlex.join(expected_argv)
    fake.responses[expected_command] = (
        0,
        '[{"n_prompt":512}]',
        "\tMaximum resident set size (kbytes): 204800",
    )

    raw_stdout, peak_mem_mb = target.run_bench(cfg, 3)

    assert fake.calls == [expected_command]
    assert raw_stdout == '[{"n_prompt":512}]'
    assert peak_mem_mb == pytest.approx(200.0)


def test_run_bench_exports_env_vars(model_spec: ModelSpec) -> None:
    cfg_with_env = BenchConfig.create(
        cmake_flags=(),
        quant="Q4_0",
        n_threads=8,
        cpu_mask=None,
        type_k="f16",
        type_v="f16",
        flash_attn=None,
        env=(("GGML_KLEIDIAI_SME", "1"),),
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
    )
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    expected_argv = (
        "env",
        "GGML_KLEIDIAI_SME=1",
        "/usr/bin/time",
        "-v",
        *target.bench_command(cfg_with_env),
        "-r",
        "5",
    )
    expected_command = shlex.join(expected_argv)
    fake.responses[expected_command] = (0, "[]", "")

    target.run_bench(cfg_with_env, 5)

    assert fake.calls == [expected_command]


def test_run_bench_raises_bench_error_on_nonzero_exit(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    fake = FakeSSHClient(default=(1, "", "llama-bench crashed"))
    target = make_target(model_spec, fake)

    with pytest.raises(BenchError):
        target.run_bench(cfg, 3)


# ---------------------------------------------------------------------------
# build: cmake -S/-B skeleton, distinct build dir per config hash, caching
# ---------------------------------------------------------------------------


def test_build_renders_configure_and_build_commands(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    build_dir = f"{REPO_DIR}/build-{build_key(cfg)}"
    configure_cmd = shlex.join(
        [
            "cmake",
            "-S",
            REPO_DIR,
            "-B",
            build_dir,
            "-DGGML_NATIVE=ON",
            "-DGGML_CPU_KLEIDIAI=OFF",
        ]
    )
    build_cmd = shlex.join(
        [
            "cmake",
            "--build",
            build_dir,
            "-j",
            "--target",
            "llama-bench",
            "llama-perplexity",
        ]
    )
    fake = FakeSSHClient(
        {configure_cmd: (0, "configured", ""), build_cmd: (0, "built", "")}
    )
    target = make_target(model_spec, fake)

    result = target.build(cfg)

    assert result == build_dir
    assert fake.calls == [configure_cmd, build_cmd]


def test_build_dir_differs_for_different_config_hash(cfg: BenchConfig) -> None:
    other_cfg = BenchConfig.create(
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
    assert f"build-{build_key(cfg)}" != f"build-{build_key(other_cfg)}"


def test_build_raises_build_error_on_nonzero_exit(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    fake = FakeSSHClient(default=(1, "", "cmake configure failed"))
    target = make_target(model_spec, fake)

    with pytest.raises(BuildError):
        target.build(cfg)


def test_build_asserts_kleidiai_marker_when_requested(model_spec: ModelSpec) -> None:
    kleidiai_cfg = BenchConfig.create(
        cmake_flags=("-DGGML_CPU_KLEIDIAI=ON",),
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
    build_dir = f"{REPO_DIR}/build-{build_key(kleidiai_cfg)}"
    configure_cmd = shlex.join(
        ["cmake", "-S", REPO_DIR, "-B", build_dir, "-DGGML_CPU_KLEIDIAI=ON"]
    )
    build_cmd = shlex.join(
        [
            "cmake",
            "--build",
            build_dir,
            "-j",
            "--target",
            "llama-bench",
            "llama-perplexity",
        ]
    )
    # -v is load-bearing: llama-bench hides the load_tensors marker without it
    # (observed on a real r8g, 2026-07-02).
    probe_cmd = shlex.join(
        [
            f"{build_dir}/bin/llama-bench",
            "-m",
            MODEL_REMOTE,
            "-p",
            "1",
            "-n",
            "1",
            "-r",
            "1",
            "-v",
        ]
    )
    fake = FakeSSHClient(
        {
            configure_cmd: (0, "", ""),
            build_cmd: (0, "", ""),
            probe_cmd: (0, "load_tensors: CPU_KLEIDIAI model buffer size = 1", ""),
        }
    )
    target = make_target(model_spec, fake)

    target.build(kleidiai_cfg)

    assert probe_cmd in fake.calls


def test_build_raises_when_kleidiai_marker_absent(model_spec: ModelSpec) -> None:
    kleidiai_cfg = BenchConfig.create(
        cmake_flags=("-DGGML_CPU_KLEIDIAI=ON",),
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
    fake = FakeSSHClient(default=(0, "load_tensors: CPU buffer size = 1", ""))
    target = make_target(model_spec, fake)

    with pytest.raises(BuildError, match="CPU_KLEIDIAI"):
        target.build(kleidiai_cfg)


# ---------------------------------------------------------------------------
# upload / download helpers
# ---------------------------------------------------------------------------


def test_upload_puts_file_over_sftp(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    target.upload("local/model.gguf", "~/armsmith-work/model.gguf")

    assert fake._sftp.put_calls == [("local/model.gguf", "~/armsmith-work/model.gguf")]


def test_download_gets_file_over_sftp(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    target.download("~/armsmith-work/results.json", "local/results.json")

    assert fake._sftp.get_calls == [
        ("~/armsmith-work/results.json", "local/results.json")
    ]


def test_upload_raises_when_not_connected(model_spec: ModelSpec) -> None:
    target = Target("1.2.3.4", "ubuntu", "/tmp/key.pem", model_spec)

    with pytest.raises(TargetError):
        target.upload("local", "remote")


# ---------------------------------------------------------------------------
# fetch_model
# ---------------------------------------------------------------------------


def test_fetch_model_skips_upload_when_sha256_matches(model_spec: ModelSpec) -> None:
    sha = "a" * 64
    sha_cmd = shlex.join(["sha256sum", MODEL_REMOTE])
    fake = FakeSSHClient({sha_cmd: (0, f"{sha}  path\n", "")})
    target = make_target(model_spec, fake)

    target.fetch_model(["Q4_0"])

    assert fake._sftp.put_calls == []


def test_fetch_model_uploads_when_sha256_missing_then_verifies(
    model_spec: ModelSpec,
) -> None:
    sha = "a" * 64
    cmd = shlex.join(["sha256sum", MODEL_REMOTE])
    calls = {"n": 0}
    fake = FakeSSHClient()

    def exec_command(command: str, timeout: int | None = None):
        if command == cmd:
            calls["n"] += 1
            text = "" if calls["n"] == 1 else f"{sha}  path\n"
            return None, _FakeChannelFile(text, 0), _FakeChannelFile("", 0)
        fake.calls.append(command)
        return None, _FakeChannelFile("", 0), _FakeChannelFile("", 0)

    fake.exec_command = exec_command  # type: ignore[method-assign]
    target = make_target(model_spec, fake)

    target.fetch_model(["Q4_0"])

    expected_local = str(Path("models") / f"{model_spec.name}-Q4_0.gguf")
    assert fake._sftp.put_calls == [(expected_local, MODEL_REMOTE)]


def test_fetch_model_raises_on_persistent_sha256_mismatch(
    model_spec: ModelSpec,
) -> None:
    fake = FakeSSHClient(default=(0, "deadbeef  path\n", ""))
    target = make_target(model_spec, fake)

    with pytest.raises(TargetError, match="sha256 mismatch"):
        target.fetch_model(["Q4_0"])


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

_LSCPU_OUTPUT = """Architecture:            aarch64
CPU(s):                  16
Vendor ID:                ARM
Flags:                    fp asimd sve2 bf16 i8mm
"""


_IMDS_TOKEN_CMD = shlex.join(
    [
        "curl",
        "-sX",
        "PUT",
        "-m",
        "2",
        "http://169.254.169.254/latest/api/token",
        "-H",
        "X-aws-ec2-metadata-token-ttl-seconds: 60",
    ]
)


def _imds_get_cmd(path: str, token: str) -> str:
    return shlex.join(
        [
            "curl",
            "-s",
            "-m",
            "2",
            "-H",
            f"X-aws-ec2-metadata-token: {token}",
            f"http://169.254.169.254/latest/meta-data/{path}",
        ]
    )


def test_describe_parses_target_spec(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient(
        {
            "uname -r": (0, "6.8.0-1015-aws\n", ""),
            "lscpu": (0, _LSCPU_OUTPUT, ""),
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor": (
                0,
                "performance\n",
                "",
            ),
            _IMDS_TOKEN_CMD: (0, "TOK123", ""),
            _imds_get_cmd("instance-type", "TOK123"): (0, "r8g.4xlarge", ""),
            _imds_get_cmd("placement/region", "TOK123"): (0, "eu-west-2", ""),
        }
    )
    target = make_target(model_spec, fake)

    spec = target.describe()

    assert spec.kernel == "6.8.0-1015-aws"
    assert spec.n_physical_cores == 16
    assert spec.capabilities == ("sve2", "bf16", "i8mm")
    assert spec.cpu_governor == "performance"
    assert spec.instance_type == "r8g.4xlarge"
    assert spec.region == "eu-west-2"


def test_describe_degrades_without_cpufreq_or_imds_token(
    model_spec: ModelSpec,
) -> None:
    """Virtualized Graviton: no cpufreq sysfs node; IMDSv2 returns empty without
    a token. Both are environment metadata - recorded as unavailable/unknown,
    never fatal (observed on a real r8g, 2026-07-02)."""
    fake = FakeSSHClient(
        {
            "uname -r": (0, "6.8.0-1015-aws\n", ""),
            "lscpu": (0, _LSCPU_OUTPUT, ""),
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor": (
                1,
                "",
                "cat: /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: "
                "No such file or directory",
            ),
            _IMDS_TOKEN_CMD: (0, "", ""),
        }
    )
    target = make_target(model_spec, fake)

    spec = target.describe()

    assert spec.cpu_governor == "unavailable"
    assert spec.instance_type == "unknown"
    assert spec.region == "unknown"


def test_parse_lscpu_helper_is_pure() -> None:
    n_cores, caps = _parse_lscpu(_LSCPU_OUTPUT)

    assert n_cores == 16
    assert caps == ("sve2", "bf16", "i8mm")


# ---------------------------------------------------------------------------
# _parse_peak_mem_mb helper
# ---------------------------------------------------------------------------


def test_parse_peak_mem_mb_extracts_kbytes_to_mb() -> None:
    stderr = "\tElapsed (wall clock) time: 0:12.34\n\tMaximum resident set size (kbytes): 4096000\n"

    assert _parse_peak_mem_mb(stderr) == pytest.approx(4000.0)


def test_parse_peak_mem_mb_returns_none_when_absent() -> None:
    assert _parse_peak_mem_mb("no time -v output here") is None


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_client(model_spec: ModelSpec) -> None:
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    with target as t:
        assert t is target

    assert fake.closed is True


# ---------------------------------------------------------------------------
# tilde expansion (blocker: shlex.join quotes `~`, so the login shell never
# expands it) -- every remote path Target emits must be absolute
# ---------------------------------------------------------------------------


def test_expand_home_and_remote_home_helpers() -> None:
    assert (
        _expand_home("~/armsmith-work", "/home/ubuntu") == "/home/ubuntu/armsmith-work"
    )
    assert _expand_home("~", "/home/ubuntu") == "/home/ubuntu"
    assert _expand_home("/abs/path", "/home/ubuntu") == "/abs/path"
    assert _remote_home("ubuntu") == "/home/ubuntu"
    assert _remote_home("root") == "/root"


def test_bench_command_paths_are_tilde_free(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    target = make_target(model_spec, FakeSSHClient())
    rendered = shlex.join(target.bench_command(cfg))
    assert "~" not in rendered  # a quoted tilde would never expand on the target
    assert rendered.startswith(f"{REPO_DIR}/build-")
    assert MODEL_REMOTE in target.bench_command(cfg)


# ---------------------------------------------------------------------------
# quality guard: run_quality carries the KV-cache flags + reads base.kld;
# capture_base_logits writes base.kld (WRITE mode); eval text is uploaded
# ---------------------------------------------------------------------------


def test_run_quality_includes_kv_cache_flags_and_kld_base(
    model_spec: ModelSpec,
) -> None:
    """A q8_0 KV-cache candidate is measured for QUALITY with its own -ctk/-ctv/-fa
    (finding 1) against base.kld (findings 2/5), not the default f16 cache."""
    cfg = BenchConfig.create(
        cmake_flags=("-DGGML_NATIVE=ON",),
        quant="Q4_0",
        n_threads=16,
        cpu_mask=None,
        type_k="q8_0",
        type_v="q8_0",
        flash_attn=True,
        env=(),
        n_prompt=512,
        n_gen=128,
        n_batch=2048,
        n_ubatch=512,
    )
    build_dir = f"{REPO_DIR}/build-{build_key(cfg)}"
    eval_remote = f"{REMOTE_ROOT}/eval.txt"
    expected_cmd = shlex.join(
        [
            f"{build_dir}/bin/llama-perplexity",
            "-m",
            MODEL_REMOTE,
            "-f",
            eval_remote,
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-fa",
            "1",
            "--kl-divergence-base",
            f"{REMOTE_ROOT}/base.kld",
            "--kl-divergence",
        ]
    )
    stdout = "Mean PPL(Q)                   :   8.123400 +/-   0.5\nMean    KLD:   0.042100 +/-   0.0001\n"
    fake = FakeSSHClient({expected_cmd: (0, stdout, "")})
    target = make_target(model_spec, fake)

    score = target.run_quality(cfg, eval_remote)

    assert fake.calls == [expected_cmd]
    assert score.perplexity == pytest.approx(8.1234)
    assert score.kl_vs_baseline == pytest.approx(0.0421)


def test_parse_quality_output_against_real_kld_capture() -> None:
    """Parses the VERBATIM stdout of llama-perplexity --kl-divergence captured
    on a real r8g (2026-07-02). The old regexes expected 'Final estimate: PPL'
    and 'Mean KL divergence', matched nothing, and silently returned
    QualityScore(None, None) through a full live baseline run."""
    from armsmith.target import _parse_quality_output

    fixture = Path(__file__).parent / "fixtures" / "llama_perplexity_kld.txt"
    text = fixture.read_text(encoding="utf-8")

    score = _parse_quality_output(text)

    assert score.perplexity == pytest.approx(15.972201)
    assert score.kl_vs_baseline == pytest.approx(0.002036)


def test_parse_quality_output_plain_perplexity_fallback() -> None:
    from armsmith.target import _parse_quality_output

    score = _parse_quality_output("Final estimate: PPL = 8.1234 +/- 0.5\n")

    assert score.perplexity == pytest.approx(8.1234)
    assert score.kl_vs_baseline is None


def test_capture_base_logits_writes_kld_in_write_mode(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    build_dir = f"{REPO_DIR}/build-{build_key(cfg)}"
    eval_remote = f"{REMOTE_ROOT}/eval.txt"
    expected_cmd = shlex.join(
        [
            f"{build_dir}/bin/llama-perplexity",
            "-m",
            MODEL_REMOTE,
            "-f",
            eval_remote,
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "--kl-divergence-base",
            f"{REMOTE_ROOT}/base.kld",
        ]
    )
    stat_cmd = shlex.join(["stat", "-c", "%s", f"{REMOTE_ROOT}/base.kld"])
    fake = FakeSSHClient({expected_cmd: (0, "", ""), stat_cmd: (0, "412345678\n", "")})
    target = make_target(model_spec, fake)

    kld = target.capture_base_logits(cfg, eval_remote)

    assert kld == f"{REMOTE_ROOT}/base.kld"
    assert fake.calls == [expected_cmd, stat_cmd]
    # WRITE mode (produce base logits) must NOT pass the read/compare flag.
    assert "--kl-divergence" not in fake.calls[0].split()


def test_capture_base_logits_raises_bench_error_on_nonzero_exit(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    fake = FakeSSHClient(default=(1, "", "perplexity crashed"))
    target = make_target(model_spec, fake)

    with pytest.raises(BenchError):
        target.capture_base_logits(cfg, f"{REMOTE_ROOT}/eval.txt")


def test_capture_base_logits_rejects_stub_file(
    model_spec: ModelSpec, cfg: BenchConfig
) -> None:
    """llama-perplexity exits 0 even when it evaluates nothing: an eval text
    under 1024 tokens produced a 12-byte base.kld stub on a real r8g
    (2026-07-02), and every later KL read failed. The capture must fail loudly
    instead."""
    stat_cmd = shlex.join(["stat", "-c", "%s", f"{REMOTE_ROOT}/base.kld"])
    fake = FakeSSHClient(
        {stat_cmd: (0, "12\n", "")},
        default=(0, "", "perplexity: you need at least 1024 tokens"),
    )
    target = make_target(model_spec, fake)

    with pytest.raises(BenchError, match="invalid stub"):
        target.capture_base_logits(cfg, f"{REMOTE_ROOT}/eval.txt")


def test_upload_eval_text_uploads_and_returns_remote_path(
    model_spec: ModelSpec,
) -> None:
    local = str(Path("examples") / "eval.txt")
    fake = FakeSSHClient()
    target = make_target(model_spec, fake)

    remote = target.upload_eval_text(local)

    assert remote == f"{REMOTE_ROOT}/eval.txt"
    assert fake._sftp.put_calls == [(local, f"{REMOTE_ROOT}/eval.txt")]
