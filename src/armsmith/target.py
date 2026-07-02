"""armsmith target.py — the only SSH module (paramiko).

Builds a llama.cpp variant for a `BenchConfig` on a remote Graviton target,
runs `llama-bench` / `llama-perplexity`, and fetches results. Every remote
command is an `argv` list spliced from a fixed skeleton plus already-validated
`BenchConfig` fields — never a shell string interpolated from untrusted
(e.g. LLM) text (CLAUDE.md rule 2, CONTRACTS.md §7).
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import paramiko

from armsmith.models import BenchError, BuildError, TargetError
from armsmith.models import BenchConfig, ModelSpec, QualityScore, TargetSpec, build_key

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800
"""Default per-command timeout (30 min) for build/bench/quality commands."""

_CAPABILITY_FLAGS = ("sve2", "bf16", "i8mm", "sme", "sme2")
"""lscpu `Flags:` tokens armsmith cares about, checked in this fixed order."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


class Target:
    """The only SSH module. build / bench / quality / describe on the target."""

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str,
        model: ModelSpec,
        *,
        remote_root: str = "~/armsmith-work",
        connect_timeout: int = 30,
        client: paramiko.SSHClient | None = None,
    ) -> None:
        self.host = host
        self.user = user
        self.key_path = key_path
        self.model = model
        # Expand a leading `~` to an absolute home NOW: shlex.join quotes `~`
        # (verified: shlex.join(["cmake","-S","~/x"]) -> "cmake -S '~/x'"), so a
        # tilde left in a remote path is never expanded by the login shell.
        self._home = _remote_home(user)
        self.remote_root = _expand_home(remote_root, self._home)
        self.connect_timeout = connect_timeout
        self._repo_dir = f"{self.remote_root}/llama.cpp"
        self._client = client

    # -- connection lifecycle -------------------------------------------------

    def connect(self) -> None:
        """paramiko SSHClient, key auth only. No-op if a client is already set
        (e.g. one was injected for tests)."""
        if self._client is not None:
            return
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            username=self.user,
            key_filename=self.key_path,
            timeout=self.connect_timeout,
        )
        self._client = client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "Target":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- low-level executor ----------------------------------------------------

    def _run(
        self,
        argv: Sequence[str],
        *,
        timeout: int | None = None,
        error_cls: type[TargetError] = TargetError,
    ) -> CommandResult:
        """exec_command wrapper; joins argv with shlex.quote (via shlex.join).
        Nonzero exit -> `error_cls` (a TargetError subclass) carrying argv + stderr."""
        if self._client is None:
            raise TargetError("target not connected; call connect() first")
        command = shlex.join(argv)
        logger.debug("target run: %s", command)
        _stdin, stdout, stderr = self._client.exec_command(
            command, timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_S
        )
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        result = CommandResult(
            argv=tuple(argv),
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
        )
        if exit_code != 0:
            logger.warning("target command failed (exit %d): %s", exit_code, command)
            raise error_cls(
                f"command failed (exit {exit_code}): {command}\nstderr:\n{stderr_text}"
            )
        return result

    # -- file transfer helpers --------------------------------------------------

    def upload(self, local_path: str, remote_path: str) -> None:
        """Copy a local file to `remote_path` over SFTP."""
        if self._client is None:
            raise TargetError("target not connected; call connect() first")
        sftp = self._client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()

    def download(self, remote_path: str, local_path: str) -> None:
        """Copy a remote file to `local_path` over SFTP."""
        if self._client is None:
            raise TargetError("target not connected; call connect() first")
        sftp = self._client.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

    # -- describe -----------------------------------------------------------

    def describe(self) -> TargetSpec:
        """uname -r, lscpu (core count + capabilities), CPU governor, instance metadata."""
        kernel = self._run(["uname", "-r"]).stdout.strip()
        lscpu_out = self._run(["lscpu"]).stdout
        n_physical_cores, capabilities = _parse_lscpu(lscpu_out)
        # Virtualized Graviton guests expose no cpufreq driver; the governor is
        # environment metadata, so its absence is recorded, never fatal.
        try:
            governor = self._run(
                ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]
            ).stdout.strip()
        except TargetError:
            governor = "unavailable"
        instance_type = self._imds("instance-type")
        region = self._imds("placement/region")
        return TargetSpec(
            host=self.host,
            user=self.user,
            instance_type=instance_type,
            core="",
            region=region,
            kernel=kernel,
            cpu_governor=governor,
            n_physical_cores=n_physical_cores,
            capabilities=capabilities,
        )

    def _imds(self, path: str) -> str:
        """EC2 instance metadata via IMDSv2 (token required on current instances;
        tokenless requests return empty). Environment metadata only - degrades to
        "unknown" rather than failing the run."""
        try:
            token = self._run(
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
            ).stdout.strip()
            if not token:
                return "unknown"
            value = self._run(
                [
                    "curl",
                    "-s",
                    "-m",
                    "2",
                    "-H",
                    f"X-aws-ec2-metadata-token: {token}",
                    f"http://169.254.169.254/latest/meta-data/{path}",
                ]
            ).stdout.strip()
            return value or "unknown"
        except TargetError:
            return "unknown"

    # -- model staging --------------------------------------------------------

    def fetch_model(self, quants: Sequence[str]) -> None:
        """Ensure every needed GGUF variant is present and integrity-checked."""
        for quant in quants:
            remote_path, expected_sha256 = self.model.resolve(quant)
            remote_path = _expand_home(remote_path, self._home)
            if self._remote_sha256(remote_path) == expected_sha256:
                continue
            local_path = str(Path("models") / f"{self.model.name}-{quant}.gguf")
            self.upload(local_path, remote_path)
            actual_sha256 = self._remote_sha256(remote_path)
            if actual_sha256 != expected_sha256:
                raise TargetError(
                    f"sha256 mismatch for quant {quant!r} after upload: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

    def _remote_sha256(self, remote_path: str) -> str | None:
        try:
            result = self._run(["sha256sum", remote_path])
        except TargetError:
            return None
        stdout = result.stdout.strip()
        return stdout.split()[0] if stdout else None

    def _model_path(self, quant: str) -> str:
        """Absolute remote GGUF path for `quant`, with a leading `~` expanded so
        shlex.join can quote it without breaking remote tilde expansion."""
        remote_path, _sha256 = self.model.resolve(quant)
        return _expand_home(remote_path, self._home)

    @staticmethod
    def _kv_cache_args(cfg: BenchConfig) -> list[str]:
        """`-ctk/-ctv [-fa]` fragment shared by bench_command, run_quality and
        capture_base_logits so the KV-cache lever is applied identically when a
        config is measured for speed AND for quality (CLAUDE.md rule 6)."""
        args = ["-ctk", cfg.type_k, "-ctv", cfg.type_v]
        if cfg.flash_attn is not None:
            args += ["-fa", "1" if cfg.flash_attn else "0"]
        return args

    # -- build ----------------------------------------------------------------

    def build(self, cfg: BenchConfig) -> str:
        """cmake -S/-B build-<build_key> <cfg.cmake_flags...> ; cmake --build.
        Cached by build_key(cfg): a config that only changes runtime knobs
        reuses the same build dir (free revert)."""
        build_dir = f"{self._repo_dir}/build-{build_key(cfg)}"
        configure_argv = [
            "cmake",
            "-S",
            self._repo_dir,
            "-B",
            build_dir,
            *cfg.cmake_flags,
        ]
        self._run(configure_argv, timeout=DEFAULT_TIMEOUT_S, error_cls=BuildError)
        build_argv = ["cmake", "--build", build_dir, "-j"]
        self._run(build_argv, timeout=DEFAULT_TIMEOUT_S, error_cls=BuildError)
        if any("GGML_CPU_KLEIDIAI=ON" in flag for flag in cfg.cmake_flags):
            self._assert_kleidiai_active(cfg, build_dir)
        return build_dir

    def _assert_kleidiai_active(self, cfg: BenchConfig, build_dir: str) -> None:
        model_path = self._model_path(cfg.quant)
        # -v is required: llama-bench suppresses model-load logs (including the
        # `load_tensors: CPU_KLEIDIAI` marker) at default verbosity - verified
        # on a real r8g 2026-07-02.
        probe_argv = [
            f"{build_dir}/bin/llama-bench",
            "-m",
            model_path,
            "-p",
            "1",
            "-n",
            "1",
            "-r",
            "1",
            "-v",
        ]
        result = self._run(probe_argv, timeout=DEFAULT_TIMEOUT_S, error_cls=BuildError)
        if "CPU_KLEIDIAI" not in (result.stdout + result.stderr):
            raise BuildError(
                f"KleidiAI requested (config {cfg.config_id}) but "
                "'load_tensors: CPU_KLEIDIAI' was not observed"
            )

    # -- bench ------------------------------------------------------------

    def bench_command(self, cfg: BenchConfig) -> tuple[str, ...]:
        """PURE, SSH-free: the exact llama-bench argv for `cfg`."""
        build_dir = f"{self._repo_dir}/build-{build_key(cfg)}"
        model_path = self._model_path(cfg.quant)
        argv: list[str] = [
            f"{build_dir}/bin/llama-bench",
            "-m",
            model_path,
            "-p",
            str(cfg.n_prompt),
            "-n",
            str(cfg.n_gen),
            "-b",
            str(cfg.n_batch),
            "-ub",
            str(cfg.n_ubatch),
        ]
        # n_threads == 0 means "engine default": OMIT -t entirely (the honest
        # naive baseline is a user who never passes -t). Passing a literal
        # `-t 0` aborts ggml with GGML_ASSERT(cplan->n_threads > 0) - observed
        # on a real r8g, 2026-07-02.
        if cfg.n_threads > 0:
            argv += ["-t", str(cfg.n_threads)]
        if cfg.cpu_mask is not None:
            argv += ["--cpu-mask", cfg.cpu_mask]
        argv += self._kv_cache_args(cfg)
        argv += ["-o", "json"]
        return tuple(argv)

    def run_bench(self, cfg: BenchConfig, n_repeats: int) -> tuple[str, float | None]:
        """`/usr/bin/time -v <bench_command(cfg)> -r {n_repeats}` with cfg.env exported.
        Returns (raw stdout = a JSON array, peak_mem_mb parsed from stderr)."""
        env_prefix = [f"{key}={value}" for key, value in cfg.env]
        argv = [
            "env",
            *env_prefix,
            "/usr/bin/time",
            "-v",
            *self.bench_command(cfg),
            "-r",
            str(n_repeats),
        ]
        result = self._run(argv, timeout=DEFAULT_TIMEOUT_S, error_cls=BenchError)
        return result.stdout, _parse_peak_mem_mb(result.stderr)

    def upload_eval_text(self, local_path: str) -> str:
        """Upload the pinned eval text (a CONTROL-PLANE path) to the target and
        return its remote path (``{remote_root}/{basename}``). Every quality run
        passes `-f <remote path>`; without this upload llama-perplexity is handed a
        local path it cannot open. Idempotent — `baseline` calls it once before
        capturing base logits, `optimize` calls it so the same file is present for
        each candidate's quality run."""
        remote_path = f"{self.remote_root}/{Path(local_path).name}"
        self.upload(local_path, remote_path)
        return remote_path

    def capture_base_logits(self, cfg: BenchConfig, eval_text_remote: str) -> str:
        """Save the base logits to ``base.kld`` ONCE from the baseline reference
        `cfg`. This is llama-perplexity's WRITE mode: ``--kl-divergence-base
        <file>`` WITHOUT ``--kl-divergence``. Every later :meth:`run_quality` call
        reads this file to compute KL vs the baseline. Nothing else writes it, so
        without this step the quality guard (CLAUDE.md rule 6) hard-fails. Returns
        the remote ``base.kld`` path."""
        kld_path = f"{self.remote_root}/base.kld"
        argv = [
            f"{self._repo_dir}/build-{build_key(cfg)}/bin/llama-perplexity",
            "-m",
            self._model_path(cfg.quant),
            "-f",
            eval_text_remote,
            *self._kv_cache_args(cfg),
            "--kl-divergence-base",
            kld_path,
        ]
        result = self._run(argv, timeout=DEFAULT_TIMEOUT_S, error_cls=BenchError)
        # llama-perplexity exits 0 even when it evaluates nothing (observed on a
        # real r8g: an eval text under 1024 tokens produced a 12-byte header
        # stub, and every later KL read failed with "failed reading n_vocab").
        # Real base logits are n_vocab floats per token - megabytes at minimum.
        size_out = self._run(["stat", "-c", "%s", kld_path]).stdout.strip()
        if not size_out.isdigit() or int(size_out) < 1024:
            raise BenchError(
                f"base logits capture produced an invalid stub at {kld_path} "
                f"({size_out or 'unknown'} bytes). The eval text must tokenize "
                "to at least 1024 tokens (2x the 512 context). stderr tail:\n"
                + result.stderr[-800:]
            )
        return kld_path

    def run_quality(self, cfg: BenchConfig, eval_text_remote: str) -> QualityScore:
        """llama-perplexity ``--kl-divergence-base <base.kld> --kl-divergence`` over
        the pinned small eval text -> QualityScore(perplexity, kl_vs_baseline).

        Passes the candidate's KV-cache knobs (``-ctk/-ctv/-fa``) so a KV-cache
        quantization lever is measured for QUALITY, not just speed — otherwise a
        q8_0 KV config runs perplexity with the DEFAULT f16 cache and reports the
        incumbent's unchanged quality, letting a quality-regressing config be kept
        (CLAUDE.md rule 6). ``base.kld`` is produced once by
        :meth:`capture_base_logits`; the eval text must already be on the target
        (:meth:`upload_eval_text`)."""
        kld_path = f"{self.remote_root}/base.kld"
        argv = [
            f"{self._repo_dir}/build-{build_key(cfg)}/bin/llama-perplexity",
            "-m",
            self._model_path(cfg.quant),
            "-f",
            eval_text_remote,
            *self._kv_cache_args(cfg),
            "--kl-divergence-base",
            kld_path,
            "--kl-divergence",
        ]
        result = self._run(argv, timeout=DEFAULT_TIMEOUT_S, error_cls=BenchError)
        # llama.cpp tools log results via common_log -> STDERR; parse both
        # streams (parsing stdout alone yielded QualityScore(None, None) on a
        # real r8g run, 2026-07-02).
        return _parse_quality_output(result.stdout + "\n" + result.stderr)


def _remote_home(user: str) -> str:
    """Absolute home directory for `user` on a standard Ubuntu/AL2023 target
    (``/root`` for root, ``/home/<user>`` otherwise). Used to expand a leading
    ``~`` in remote paths so shlex.join can quote them without the login shell
    losing tilde expansion."""
    return "/root" if user == "root" else f"/home/{user}"


def _expand_home(path: str, home: str) -> str:
    """Replace a leading ``~`` in `path` with the absolute `home`. Absolute or
    already-expanded paths pass through unchanged."""
    if path == "~":
        return home
    if path.startswith("~/"):
        return f"{home}{path[1:]}"
    return path


def _parse_peak_mem_mb(stderr: str) -> float | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if match is None:
        return None
    return int(match.group(1)) / 1024


def _parse_quality_output(stdout: str) -> QualityScore:
    """Extract (perplexity, mean KL) from llama-perplexity output.

    Formats verified against a REAL r8g capture 2026-07-02 (fixture
    ``tests/fixtures/llama_perplexity_kld.txt``): KL mode prints
    ``Mean PPL(Q)                   :  15.972201 +/- ...`` and
    ``Mean    KLD:   0.002036 +/- ...``. Plain perplexity mode (no KL base)
    prints ``Final estimate: PPL = ...`` - kept as the fallback."""
    perplexity: float | None = None
    kl_vs_baseline: float | None = None
    ppl_match = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", stdout)
    if ppl_match is None:
        ppl_match = re.search(r"Final estimate: PPL\s*=\s*([\d.]+)", stdout)
    if ppl_match is not None:
        perplexity = float(ppl_match.group(1))
    kld_match = re.search(r"Mean\s+KLD:\s*(-?[\d.eE+]+)", stdout)
    if kld_match is not None:
        kl_vs_baseline = float(kld_match.group(1))
    return QualityScore(perplexity=perplexity, kl_vs_baseline=kl_vs_baseline)


def _parse_lscpu(lscpu_out: str) -> tuple[int, tuple[str, ...]]:
    n_physical_cores = 0
    core_match = re.search(r"^CPU\(s\):\s*(\d+)", lscpu_out, re.MULTILINE)
    if core_match is not None:
        n_physical_cores = int(core_match.group(1))
    flags: tuple[str, ...] = ()
    flags_match = re.search(r"^Flags:\s*(.+)$", lscpu_out, re.MULTILINE)
    if flags_match is not None:
        present = set(flags_match.group(1).split())
        flags = tuple(flag for flag in _CAPABILITY_FLAGS if flag in present)
    return n_physical_cores, flags
