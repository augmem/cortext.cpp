#!/usr/bin/env python3
"""Run Julie release ablations and emit an explicit ablation provenance plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORTEXT_RELEASE_ENV_ALLOWLIST = {
    "CORTEXT_JUDGE_BASE_URL",
    "CORTEXT_OLLAMA_BASE_URL",
}
NO_GRAPH_EXPANSION_ENV = {
    "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION": "1",
    "CORTEXT_DISABLE_DURABLE_SOURCE_SET_RETRIEVAL": "1",
    "CORTEXT_DISABLE_PRECONSOLIDATED_LABEL_GRAPH": "1",
}
CASES = [
    ("no_daily_consolidation", {}, False),
    (
        "no_graph_expansion",
        NO_GRAPH_EXPANSION_ENV,
        True,
    ),
    ("no_media_source_blobs", {"CORTEXT_DISABLE_SOURCE_BLOBS": "1"}, True),
    (
        "no_stm_ltm_graph_label_handoff",
        {"CORTEXT_DISABLE_STM_LABEL_HANDOFF": "1"},
        True,
    ),
    (
        "no_temporal_retrieval",
        {"CORTEXT_DISABLE_TEMPORAL_RETRIEVAL": "1"},
        True,
    ),
    ("no_fact_boosts", {"CORTEXT_DISABLE_FACTS": "1"}, True),
    (
        "no_temporal_fact_boosts",
        {"CORTEXT_DISABLE_TEMPORAL_RETRIEVAL": "1", "CORTEXT_DISABLE_FACTS": "1"},
        True,
    ),
]

REQUIRED_BENCH_FLAGS = [
    "--input-dir",
    "--max-messages",
    "--media-limit",
    "--probe-stride",
    "--warmup-events",
    "--rag-top-k",
    "--active-history-token-budget",
    "--focus",
    "--sensitivity",
    "--stability",
]
HOSTED_PROVIDER_ENV_MARKERS = (
    "OPENAI",
    "ANTHROPIC",
    "GEMINI",
    "GOOGLE_API",
    "GOOGLE_GENAI",
    "VERTEX",
    "AZURE_OPENAI",
    "TOGETHER",
    "MISTRAL",
    "COHERE",
    "GROQ",
)


def is_hosted_provider_env_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in HOSTED_PROVIDER_ENV_MARKERS)


def redacted_env_value(key: str, value: str) -> str:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    if any(part in key.upper() for part in blocked):
        return "<redacted>"
    return value


def is_cortext_behavior_env_key(key: str) -> bool:
    return key.startswith("CORTEXT_") and key not in CORTEXT_RELEASE_ENV_ALLOWLIST


def cortext_behavior_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: redacted_env_value(key, value)
        for key, value in sorted(env.items())
        if is_cortext_behavior_env_key(key)
    }


def cortext_behavior_env_guard(env: dict[str, str]) -> dict:
    leaked = cortext_behavior_env(env)
    return {
        "mode": "fail_closed",
        "policy": (
            "Julie release ablations reject ambient CORTEXT_* variables except "
            "local judge endpoint metadata before applying named ablation "
            "overrides."
        ),
        "allowed_cortext_env_keys": sorted(CORTEXT_RELEASE_ENV_ALLOWLIST),
        "leakage_detected": bool(leaked),
        "leaked_cortext_behavior_env": leaked,
    }


def ensure_no_ambient_cortext_behavior_env(env: dict[str, str]) -> None:
    guard = cortext_behavior_env_guard(env)
    if guard["leakage_detected"]:
        raise RuntimeError(
            "refusing to launch Julie release ablations with ambient CORTEXT_* "
            "behavior variables: "
            + json.dumps(guard, sort_keys=True)
        )


def sanitized_subprocess_env() -> tuple[dict[str, str], list[str]]:
    ensure_no_ambient_cortext_behavior_env(dict(os.environ))
    env = dict(os.environ)
    stripped = sorted(key for key in env if is_hosted_provider_env_key(key))
    for key in stripped:
        env.pop(key, None)
    return env, stripped


def is_json(path: pathlib.Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def wait_for_json(path: pathlib.Path) -> None:
    fswatch = shutil.which("fswatch")
    while not is_json(path):
        if fswatch:
            subprocess.run(
                [fswatch, "-1", str(path.parent)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            time.sleep(5)


def command_flag_value(parts: list[str], flag: str) -> str:
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts):
            return parts[i + 1]
    raise RuntimeError(f"benchmark command is missing required flag {flag}")


def command_has_flag(parts: list[str], flag: str) -> bool:
    return flag in parts


def remove_existing_outputs(paths: list[pathlib.Path]) -> None:
    for path in paths:
        for candidate in [
            path,
            pathlib.Path(str(path) + "-wal"),
            pathlib.Path(str(path) + "-shm"),
        ]:
            if candidate.exists():
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()


def write_environment_snapshot(
    path: pathlib.Path,
    name: str,
    env_overrides: dict[str, str],
    stripped_hosted_provider_env_keys: list[str],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "cortext_ablation_environment_snapshot_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "case": name,
                "reuse_existing": False,
                "reuse_policy": "disabled_for_release_provenance",
                "env_overrides": env_overrides,
                "actual_env_overrides": {
                    key: env_overrides[key]
                    for key in sorted(env_overrides)
                },
                "ambient_cortext_behavior_env_guard": cortext_behavior_env_guard(
                    dict(os.environ)
                ),
                "ambient_cortext_behavior_env_policy": (
                    "ambient parent CORTEXT_* behavior variables are rejected; "
                    "only explicit named ablation env_overrides are applied "
                    "after sanitizing the parent environment"
                ),
                "hosted_provider_env_policy": (
                    "hosted-provider variables are stripped before launching "
                    "ablation benchmarks, judges, and reports"
                ),
                "stripped_hosted_provider_env_keys": stripped_hosted_provider_env_keys,
            },
            indent=2,
        )
        + "\n"
    )


def command_sha256(command: str) -> str:
    return hashlib.sha256(" ".join(shlex.split(command)).encode("utf-8")).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_contains(path: pathlib.Path, needle: bytes) -> bool:
    overlap = max(0, len(needle) - 1)
    previous = b""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            haystack = previous + chunk
            if needle in haystack:
                return True
            previous = haystack[-overlap:] if overlap else b""
    return False


def command_executable_artifact(command: str) -> dict:
    parts = shlex.split(command)
    raw = parts[0] if parts else ""
    resolved = ""
    if raw:
        candidate = pathlib.Path(raw)
        if candidate.exists() or "/" in raw:
            resolved = str(candidate.resolve())
        else:
            resolved = shutil.which(raw) or ""
    path = pathlib.Path(resolved) if resolved else None
    exists = bool(path and path.exists() and path.is_file())
    return {
        "command": raw,
        "path": str(path) if path else "",
        "exists": exists,
        "sha256": file_sha256(path) if path and exists else "",
        "bytes": path.stat().st_size if path and exists else 0,
        "mtime_ns": path.stat().st_mtime_ns if path and exists else 0,
    }


def executable_supports_probe_stream(artifact: dict) -> bool:
    path = pathlib.Path(str(artifact.get("path", "")))
    return bool(
        path.exists()
        and path.is_file()
        and file_contains(path, b".probes.jsonl")
        and file_contains(path, b"native probe rows appended")
    )


def strict_early_judge_required(
    early_judge_mode: str,
    require_final_report_pass: bool,
    early_judge_enabled: bool,
) -> bool:
    return early_judge_mode == "on" or (
        require_final_report_pass and early_judge_enabled
    )


def ensure_initial_report_freezable(report: dict) -> None:
    checks = report.get("release_gate", {}).get("checks", [])
    if not isinstance(checks, list):
        raise RuntimeError("initial report does not contain release_gate.checks")
    unexpected_failures = []
    for item in checks:
        if not isinstance(item, dict) or item.get("status") != "fail":
            continue
        name = str(item.get("name", ""))
        if name.startswith("claim_"):
            continue
        unexpected_failures.append(
            {
                "name": name,
                "detail": item.get("detail", ""),
            }
        )
    if unexpected_failures:
        raise RuntimeError(
            "refusing to freeze initial report with protocol/provenance failures: "
            + json.dumps(unexpected_failures, sort_keys=True)
        )


def write_or_validate_release_freeze(
    path: pathlib.Path,
    report_path: pathlib.Path,
    benchmark_command: str,
) -> None:
    report = json.loads(report_path.read_text())
    ensure_initial_report_freezable(report)
    source_fingerprint = report.get("source_run", {}).get("source_input_fingerprint", {})
    schedule = report.get("frozen_probe_schedule", {})
    git = report.get("git", {})
    benchmark_executable = report.get("artifacts", {}).get("benchmark_executable", {})
    freeze = {
        "schema": "cortext_julie_release_protocol_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "main_release_report",
        "source_report_path": str(report_path),
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_input_manifest_sha256": source_fingerprint.get("manifest_sha256", ""),
        "source_input_transcript_sha256": source_fingerprint.get("transcript_sha256", ""),
        "frozen_probe_schedule_sha256": schedule.get("schedule_sha256", ""),
        "frozen_probe_count": schedule.get("probe_count"),
        "benchmark_command_sha256": command_sha256(benchmark_command),
        "benchmark_executable_sha256": benchmark_executable.get("sha256", ""),
        "git_commit": git.get("commit", ""),
        "git_dirty": git.get("dirty"),
        "git_status_sha256": git.get("status_sha256", ""),
        "git_worktree_manifest_sha256": git.get("worktree_manifest_sha256", ""),
        "privacy": "hashes only; no message text or media paths",
    }
    if not freeze["source_input_manifest_sha256"] or not freeze["frozen_probe_schedule_sha256"]:
        raise RuntimeError("main report does not contain source/probe hashes for release freeze")
    if not freeze["benchmark_executable_sha256"]:
        raise RuntimeError("main report does not contain benchmark executable hash for release freeze")
    if not freeze["git_commit"] or not freeze["git_worktree_manifest_sha256"]:
        raise RuntimeError("main report does not contain git worktree hashes for release freeze")
    if path.exists():
        existing = json.loads(path.read_text())
        for key in [
            "source_input_manifest_sha256",
            "frozen_probe_schedule_sha256",
            "benchmark_command_sha256",
            "benchmark_executable_sha256",
            "git_commit",
            "git_worktree_manifest_sha256",
        ]:
            if existing.get(key) != freeze.get(key):
                raise RuntimeError(
                    f"existing release freeze mismatch for {key}: "
                    f"{existing.get(key)} != {freeze.get(key)}"
                )
        return
    path.write_text(json.dumps(freeze, indent=2) + "\n")


def frozen_benchmark_common(benchmark_command: str) -> list[str]:
    parts = shlex.split(benchmark_command)
    if not parts:
        raise RuntimeError("--benchmark-command must not be empty")

    common = [parts[0]]
    for flag in REQUIRED_BENCH_FLAGS:
        common += [flag, command_flag_value(parts, flag)]
    if command_has_flag(parts, "--skip-messages"):
        common += ["--skip-messages", command_flag_value(parts, "--skip-messages")]
    if command_has_flag(parts, "--deep"):
        common.append("--deep")
    return common


def probe_stream_path(summary: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(summary) + ".probes.jsonl")


def build_early_judge_command(
    args: argparse.Namespace,
    bench_cmd: list[str],
    summary: pathlib.Path,
    db: pathlib.Path,
    daily: bool,
) -> list[str]:
    cmd = [
        "python3",
        "tools/watch_julie_probe_stream_judge.py",
        "--probe-stream",
        str(probe_stream_path(summary)),
        "--out-dir",
        str(summary.parent / "early_judge"),
        "--input-dir",
        command_flag_value(bench_cmd, "--input-dir"),
        "--db",
        str(db),
        "--timeline-skip-messages",
        command_flag_value(bench_cmd, "--skip-messages")
        if command_has_flag(bench_cmd, "--skip-messages")
        else "0",
        "--timeline-max-messages",
        command_flag_value(bench_cmd, "--max-messages"),
        "--timeline-media-limit",
        command_flag_value(bench_cmd, "--media-limit"),
        "--milestones",
        args.early_judge_milestones,
        "--periodic-stride",
        str(args.early_judge_periodic_stride),
        "--completion-summary",
        str(summary),
        "--poll-seconds",
        str(args.early_judge_poll_seconds),
        "--warmup-events",
        command_flag_value(bench_cmd, "--warmup-events"),
        "--probe-stride",
        command_flag_value(bench_cmd, "--probe-stride"),
        "--rag-top-k",
        command_flag_value(bench_cmd, "--rag-top-k"),
        "--active-history-token-budget",
        command_flag_value(bench_cmd, "--active-history-token-budget"),
        "--focus",
        command_flag_value(bench_cmd, "--focus"),
        "--sensitivity",
        command_flag_value(bench_cmd, "--sensitivity"),
        "--stability",
        command_flag_value(bench_cmd, "--stability"),
        "--judge-provider",
        "ollama",
        "--model",
        "gemma4:12b-it-qat",
        "--ollama-base-url",
        "http://127.0.0.1:11434",
        "--judge-repetitions",
        str(args.early_judge_repetitions),
        "--confirm-fail-repetitions",
        str(args.early_confirm_fail_repetitions),
        "--judge-seed",
        "42",
        "--bootstrap-samples",
        str(args.early_judge_bootstrap_samples),
        "--judge-timeout-s",
        str(args.early_judge_timeout_s),
        "--judge-context-window-tokens",
        str(args.early_judge_context_window_tokens),
        "--context-limit",
        str(args.early_judge_packet_item_limit),
        "--quality-gate-min-milestone",
        str(args.early_quality_gate_min_milestone),
        "--quality-trend-gate-min-milestone",
        str(args.early_quality_trend_gate_min_milestone),
        "--quality-trend-window",
        str(args.early_quality_trend_window),
        "--quality-gate-min-history-budget-ratio",
        str(args.early_quality_gate_min_history_budget_ratio),
        "--max-media-per-system",
        "-1",
        "--blind-packets",
        "--min-mean-cortext-context-tokens",
        str(args.early_min_mean_cortext_context_tokens),
        "--min-cortext-token-savings-vs-rag",
        str(args.early_min_cortext_token_savings_vs_rag),
        "--min-cortext-quality-delta-vs-rag",
        str(args.early_min_cortext_quality_delta_vs_rag),
        "--require-fairness-checks",
    ]
    if args.early_judge_periodic_stride == 0:
        cmd.append("--stop-after-last")
    if args.early_min_cortext_win_rate is not None:
        cmd += ["--min-cortext-win-rate", str(args.early_min_cortext_win_rate)]
    if not args.early_quality_gate_requires_rag_pressure:
        cmd.append("--no-quality-gate-requires-rag-pressure")
    if daily:
        cmd.append("--daily-consolidation")
    if command_has_flag(bench_cmd, "--deep"):
        cmd.append("--deep")
    return cmd


def finish_early_judge_process(
    process: subprocess.Popen | None,
    log_file,
    *,
    strict: bool,
    timeout_s: int = 900,
) -> None:
    if process is None:
        return
    try:
        rc = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print("[ablation-pipeline] early judge still waiting; terminating", flush=True)
        process.terminate()
        try:
            rc = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            rc = process.wait()
    finally:
        if log_file is not None:
            log_file.close()
    if rc != 0 and strict:
        raise RuntimeError(f"early judge exited with rc={rc}")


def terminate_process(process: subprocess.Popen, name: str, timeout_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    print(f"[ablation-pipeline] terminating {name} pid={process.pid}", flush=True)
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[ablation-pipeline] killing {name} pid={process.pid}", flush=True)
        process.kill()
        process.wait()


def run_ablation_benchmark_with_early_judge(
    *,
    name: str,
    bench_cmd: list[str],
    early_cmd_text: str,
    env: dict[str, str],
    summary: pathlib.Path,
    strict_early_judge: bool,
) -> None:
    run_log = summary.parent / "run.log"
    early_log = summary.parent / "early_judge_pipeline.log"
    early_process: subprocess.Popen | None = None
    early_log_file = None
    benchmark_process: subprocess.Popen | None = None

    with run_log.open("w") as run_log_file:
        benchmark_process = subprocess.Popen(
            bench_cmd,
            cwd=REPO_ROOT,
            stdout=run_log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        if early_cmd_text:
            early_log.parent.mkdir(parents=True, exist_ok=True)
            early_log_file = early_log.open("w")
            early_launch_cmd = [
                *shlex.split(early_cmd_text),
                "--pause-pid-during-judge",
                str(benchmark_process.pid),
            ]
            (summary.parent / "early_judge_command_launched.txt").write_text(
                shlex.join(early_launch_cmd) + "\n"
            )
            print(
                "[ablation-pipeline] early judge "
                f"{name}: {shlex.join(early_launch_cmd)}",
                flush=True,
            )
            early_process = subprocess.Popen(
                early_launch_cmd,
                cwd=REPO_ROOT,
                stdout=early_log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

        try:
            while True:
                benchmark_code = benchmark_process.poll()
                early_code = early_process.poll() if early_process else None
                if early_process and early_code is not None and early_code != 0:
                    if strict_early_judge:
                        terminate_process(benchmark_process, f"ablation benchmark {name}")
                        if early_log_file is not None:
                            early_log_file.close()
                            early_log_file = None
                        raise RuntimeError(
                            f"early judge for ablation {name} exited with rc={early_code}; "
                            f"see {early_log}"
                        )
                if benchmark_code is None:
                    time.sleep(5)
                    continue
                if benchmark_code != 0:
                    if early_process is not None and early_process.poll() is None:
                        terminate_process(early_process, f"early judge {name}")
                    if early_log_file is not None:
                        early_log_file.close()
                        early_log_file = None
                    raise RuntimeError(
                        f"ablation benchmark {name} exited with rc={benchmark_code}; "
                        f"see {run_log}"
                    )
                break
        finally:
            if benchmark_process is not None and benchmark_process.poll() is None:
                terminate_process(benchmark_process, f"ablation benchmark {name}")

    finish_early_judge_process(
        early_process,
        early_log_file,
        strict=strict_early_judge,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--main-report", type=pathlib.Path, required=True)
    parser.add_argument("--main-summary", type=pathlib.Path, required=True)
    parser.add_argument("--main-judge", type=pathlib.Path, required=True)
    parser.add_argument("--final-report", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-command", required=True)
    parser.add_argument("--judge-command", required=True)
    parser.add_argument("--ablation-plan", type=pathlib.Path, required=True)
    parser.add_argument("--judge-media-smoke", type=pathlib.Path)
    parser.add_argument("--human-labels", type=pathlib.Path)
    parser.add_argument("--human-label-eval", type=pathlib.Path)
    parser.add_argument("--target-freeze", type=pathlib.Path)
    parser.add_argument(
        "--require-final-report-pass",
        action="store_true",
        help=(
            "Exit non-zero unless the final release report passes every gate, "
            "including human labels, frozen targets, media smoke, CIs, costs, "
            "and ablations."
        ),
    )
    parser.add_argument("--early-judge", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--early-judge-milestones",
        default="1,2,3,4,5,6,7,8,9,10,11,12,16",
    )
    parser.add_argument("--early-judge-periodic-stride", type=int, default=1)
    parser.add_argument("--early-judge-poll-seconds", type=float, default=5.0)
    parser.add_argument("--early-judge-repetitions", type=int, default=1)
    parser.add_argument("--early-confirm-fail-repetitions", type=int, default=3)
    parser.add_argument("--early-judge-bootstrap-samples", type=int, default=200)
    parser.add_argument("--early-judge-timeout-s", type=int, default=180)
    parser.add_argument("--early-judge-context-window-tokens", type=int, default=32768)
    parser.add_argument("--early-judge-packet-item-limit", type=int, default=256)
    parser.add_argument("--judge-packet-item-limit", type=int, default=-1)
    parser.add_argument("--early-quality-gate-min-milestone", type=int, default=8)
    parser.add_argument("--early-quality-trend-gate-min-milestone", type=int, default=4)
    parser.add_argument("--early-quality-trend-window", type=int, default=2)
    parser.add_argument(
        "--early-quality-gate-requires-rag-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--early-quality-gate-min-history-budget-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument("--early-min-mean-cortext-context-tokens", type=float, default=1.0)
    parser.add_argument("--early-min-cortext-token-savings-vs-rag", type=float, default=0.50)
    parser.add_argument("--early-min-cortext-quality-delta-vs-rag", type=float, default=-0.50)
    parser.add_argument("--early-min-cortext-win-rate", type=float)
    args = parser.parse_args()

    if args.early_judge_poll_seconds <= 0:
        raise RuntimeError("--early-judge-poll-seconds must be positive")
    if args.early_judge_repetitions <= 0:
        raise RuntimeError("--early-judge-repetitions must be positive")
    if args.early_confirm_fail_repetitions < 0:
        raise RuntimeError("--early-confirm-fail-repetitions must be >= 0")
    if args.early_judge_periodic_stride < 0:
        raise RuntimeError("--early-judge-periodic-stride must be >= 0")
    if args.early_judge_timeout_s < 1:
        raise RuntimeError("--early-judge-timeout-s must be >= 1")
    if args.early_judge_context_window_tokens < 1:
        raise RuntimeError("--early-judge-context-window-tokens must be >= 1")
    if (
        args.early_judge_packet_item_limit == 0
        or args.early_judge_packet_item_limit < -1
    ):
        raise RuntimeError(
            "--early-judge-packet-item-limit must be -1 or a positive item limit"
        )
    if args.judge_packet_item_limit == 0 or args.judge_packet_item_limit < -1:
        raise RuntimeError("--judge-packet-item-limit must be -1 or a positive item limit")
    if args.require_final_report_pass and args.judge_packet_item_limit != -1:
        raise RuntimeError(
            "--require-final-report-pass requires --judge-packet-item-limit -1"
        )
    if args.early_quality_gate_min_milestone < 1:
        raise RuntimeError("--early-quality-gate-min-milestone must be >= 1")
    if args.early_quality_trend_gate_min_milestone < 1:
        raise RuntimeError("--early-quality-trend-gate-min-milestone must be >= 1")
    if args.early_quality_trend_window < 0:
        raise RuntimeError("--early-quality-trend-window must be >= 0")
    if not 0.0 <= args.early_quality_gate_min_history_budget_ratio <= 1.0:
        raise RuntimeError(
            "--early-quality-gate-min-history-budget-ratio must be in [0, 1]"
        )
    ensure_no_ambient_cortext_behavior_env(dict(os.environ))

    wait_for_json(args.main_report)
    print(f"[ablation-pipeline] main report ready {datetime.now().isoformat()}", flush=True)
    judge_media_smoke = args.judge_media_smoke or (args.base / "judge_media_smoke_ollama.json")

    release_freeze = args.base / "release_protocol_freeze.json"
    write_or_validate_release_freeze(
        release_freeze,
        args.main_report,
        args.benchmark_command,
    )
    print(f"[ablation-pipeline] release freeze ready {release_freeze}", flush=True)

    common = frozen_benchmark_common(args.benchmark_command)
    expected_benchmark_executable_sha256 = json.loads(release_freeze.read_text()).get(
        "benchmark_executable_sha256", ""
    )

    plan_cases = []
    for name, env_overrides, daily in CASES:
        out_dir = args.base / f"ablation_{name}"
        summary = out_dir / "summary.json"
        judge = out_dir / "gemma4_12b_ollama_blind_judge_reps3.json"
        db = out_dir / "live.sqlite"
        bench_cmd = [*common]
        if daily:
            bench_cmd.append("--daily-consolidation")
        bench_cmd += ["--db", str(db), "--out", str(summary)]
        bench_command_text = shlex.join(bench_cmd)
        benchmark_executable = command_executable_artifact(bench_command_text)
        if (
            expected_benchmark_executable_sha256
            and benchmark_executable.get("sha256") != expected_benchmark_executable_sha256
        ):
            raise RuntimeError(
                "ablation benchmark executable hash mismatch before launch: "
                f"{benchmark_executable.get('sha256')} != "
                f"{expected_benchmark_executable_sha256}"
            )
        supports_probe_stream = executable_supports_probe_stream(benchmark_executable)
        if args.early_judge == "on" and not supports_probe_stream:
            raise RuntimeError(
                "early judge requested, but benchmark executable does not emit "
                "probe streams"
            )
        early_judge_enabled = args.early_judge != "off" and supports_probe_stream
        if args.require_final_report_pass and not early_judge_enabled:
            raise RuntimeError(
                "strict final release requires ablation early judging, but it "
                f"is unavailable for {name}: early_judge={args.early_judge} "
                f"supports_probe_stream={supports_probe_stream}"
            )
        strict_early_judge = strict_early_judge_required(
            args.early_judge,
            args.require_final_report_pass,
            early_judge_enabled,
        )
        early_judge_cmd = (
            build_early_judge_command(args, bench_cmd, summary, db, daily)
            if early_judge_enabled
            else []
        )
        judge_cmd = [
            "python3",
            "tools/judge_julie_live_run.py",
            "--judge-provider",
            "ollama",
            "--model",
            "gemma4:12b-it-qat",
            "--summary",
            str(summary),
            "--db",
            str(db),
            "--out",
            str(judge),
            "--judge-repetitions",
            "3",
            "--judge-seed",
            "42",
            "--bootstrap-samples",
            "2000",
            "--judge-timeout-s",
            str(args.early_judge_timeout_s),
            "--judge-context-window-tokens",
            str(args.early_judge_context_window_tokens),
            "--context-limit",
            str(args.judge_packet_item_limit),
            "--ollama-base-url",
            "http://127.0.0.1:11434",
            "--ollama-keep-alive",
            "0s",
            "--blind-packets",
            "--max-media-per-system",
            "-1",
        ]
        plan_cases.append(
            {
                "name": name,
                "daily_consolidation": daily,
                "env_overrides": env_overrides,
                "reuse_existing": False,
                "reuse_policy": "disabled_for_release_provenance",
                "summary_path": str(summary),
                "judge_path": str(judge),
                "db_path": str(db),
                "environment_snapshot_path": str(
                    out_dir / "benchmark_environment_snapshot.json"
                ),
                "benchmark_command": bench_command_text,
                "benchmark_executable": benchmark_executable,
                "early_judge_enabled": early_judge_enabled,
                "early_judge_skip_reason": ""
                if early_judge_enabled
                else (
                    "disabled_by_flag"
                    if args.early_judge == "off"
                    else "benchmark_executable_has_no_probe_stream"
                ),
                "early_judge_command": shlex.join(early_judge_cmd)
                if early_judge_cmd
                else "",
                "early_judge_pauses_benchmark_during_local_judge": early_judge_enabled,
                "strict_early_judge": strict_early_judge,
                "early_quality_gate_requires_rag_pressure": (
                    args.early_quality_gate_requires_rag_pressure
                ),
                "early_quality_gate_min_history_budget_ratio": (
                    args.early_quality_gate_min_history_budget_ratio
                ),
                "early_judge_manifest_path": str(
                    summary.parent / "early_judge" / "early_judge_manifest.json"
                ),
                "judge_command": shlex.join(judge_cmd),
            }
        )

    args.ablation_plan.parent.mkdir(parents=True, exist_ok=True)
    args.ablation_plan.write_text(
        json.dumps(
            {
                "schema": "cortext_julie_ablation_plan_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "main_summary_path": str(args.main_summary),
                "main_judge_path": str(args.main_judge),
                "cases": plan_cases,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[ablation-pipeline] wrote plan {args.ablation_plan}", flush=True)

    ablation_args: list[str] = []
    for case, plan_case in zip(CASES, plan_cases):
        name, env_overrides, _ = case
        summary = pathlib.Path(plan_case["summary_path"])
        judge = pathlib.Path(plan_case["judge_path"])
        db = pathlib.Path(plan_case["db_path"])
        summary.parent.mkdir(parents=True, exist_ok=True)
        remove_existing_outputs(
            [
                summary,
                judge,
                pathlib.Path(str(judge) + ".rows.jsonl"),
                db,
                probe_stream_path(summary),
                summary.parent / "early_judge",
                summary.parent / "early_judge_command_launched.txt",
                summary.parent / "julie_live_media_tmp",
            ]
        )
        env, stripped_hosted_provider_env_keys = sanitized_subprocess_env()
        env.update(env_overrides)
        write_environment_snapshot(
            pathlib.Path(plan_case["environment_snapshot_path"]),
            name,
            env_overrides,
            stripped_hosted_provider_env_keys,
        )
        bench_cmd = shlex.split(plan_case["benchmark_command"])
        print(
            "[ablation-pipeline] run "
            f"{name}: {plan_case['benchmark_command']} "
            f"env_overrides={env_overrides}",
            flush=True,
        )
        run_ablation_benchmark_with_early_judge(
            name=name,
            bench_cmd=bench_cmd,
            early_cmd_text=str(plan_case.get("early_judge_command", "") or ""),
            env=env,
            summary=summary,
            strict_early_judge=bool(plan_case.get("strict_early_judge")),
        )

        judge_cmd = shlex.split(plan_case["judge_command"])
        print(
            f"[ablation-pipeline] judge {name}: {plan_case['judge_command']}",
            flush=True,
        )
        judge_env, _ = sanitized_subprocess_env()
        subprocess.run(judge_cmd, check=True, env=judge_env)

        ablation_args += ["--ablation", f"{name}:{summary}:{judge}"]

    report_cmd = [
        "python3",
        "tools/julie_release_protocol_report.py",
        "--summary",
        str(args.main_summary),
        "--judge",
        str(args.main_judge),
        "--out",
        str(args.final_report),
        "--benchmark-command",
        args.benchmark_command,
        "--judge-command",
        args.judge_command,
        "--judge-media-smoke",
        str(judge_media_smoke),
        "--freeze-file",
        str(release_freeze),
        "--ablation-plan",
        str(args.ablation_plan),
        *ablation_args,
    ]
    if args.human_labels is not None:
        report_cmd += ["--human-labels", str(args.human_labels)]
    if args.human_label_eval is not None:
        report_cmd += ["--human-label-eval", str(args.human_label_eval)]
    if args.target_freeze is not None:
        report_cmd += ["--target-freeze", str(args.target_freeze)]
    if args.require_final_report_pass:
        report_cmd.append("--require-pass")
    print(f"[ablation-pipeline] final report: {shlex.join(report_cmd)}", flush=True)
    report_env, _ = sanitized_subprocess_env()
    subprocess.run(report_cmd, check=True, env=report_env)
    print("[ablation-pipeline] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
