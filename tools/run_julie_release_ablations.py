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


CASES = [
    ("no_daily_consolidation", {}, False),
    (
        "no_graph_expansion",
        {"CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION": "1"},
        True,
    ),
    ("no_media_source_blobs", {"CORTEXT_DISABLE_SOURCE_BLOBS": "1"}, True),
    ("no_stm_label_handoff", {"CORTEXT_DISABLE_STM_LABEL_HANDOFF": "1"}, True),
    (
        "no_temporal_retrieval",
        {"CORTEXT_DISABLE_TEMPORAL_RETRIEVAL": "1"},
        True,
    ),
    ("no_fact_boosts", {"CORTEXT_DISABLE_FACTS": "1"}, True),
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


def write_environment_snapshot(path: pathlib.Path, name: str, env_overrides: dict[str, str]) -> None:
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
        "--timeline-max-messages",
        command_flag_value(bench_cmd, "--max-messages"),
        "--timeline-media-limit",
        command_flag_value(bench_cmd, "--media-limit"),
        "--milestones",
        args.early_judge_milestones,
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
        "--judge-seed",
        "42",
        "--bootstrap-samples",
        str(args.early_judge_bootstrap_samples),
        "--max-media-per-system",
        "-1",
        "--blind-packets",
        "--stop-after-last",
    ]
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
) -> None:
    if process is None:
        return
    try:
        rc = process.wait(timeout=60)
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
    parser.add_argument("--early-judge", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--early-judge-milestones", default="4,8")
    parser.add_argument("--early-judge-poll-seconds", type=float, default=30.0)
    parser.add_argument("--early-judge-repetitions", type=int, default=1)
    parser.add_argument("--early-judge-bootstrap-samples", type=int, default=200)
    args = parser.parse_args()

    wait_for_json(args.main_report)
    print(f"[ablation-pipeline] main report ready {datetime.now().isoformat()}", flush=True)

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
            "180",
            "--ollama-base-url",
            "http://127.0.0.1:11434",
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
                summary.parent / "julie_live_media_tmp",
            ]
        )
        write_environment_snapshot(
            pathlib.Path(plan_case["environment_snapshot_path"]),
            name,
            env_overrides,
        )

        env = os.environ.copy()
        env.update(env_overrides)
        bench_cmd = shlex.split(plan_case["benchmark_command"])
        print(
            "[ablation-pipeline] run "
            f"{name}: {plan_case['benchmark_command']} "
            f"env_overrides={env_overrides}",
            flush=True,
        )
        early_process = None
        early_log_file = None
        if plan_case.get("early_judge_command"):
            early_log = summary.parent / "early_judge_pipeline.log"
            early_log.parent.mkdir(parents=True, exist_ok=True)
            early_log_file = early_log.open("w")
            print(
                f"[ablation-pipeline] early judge {name}: "
                f"{plan_case['early_judge_command']}",
                flush=True,
            )
            early_process = subprocess.Popen(
                shlex.split(plan_case["early_judge_command"]),
                cwd=pathlib.Path.cwd(),
                stdout=early_log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
        try:
            subprocess.run(bench_cmd, check=True, env=env)
        finally:
            finish_early_judge_process(
                early_process,
                early_log_file,
                strict=args.early_judge == "on",
            )

        judge_cmd = shlex.split(plan_case["judge_command"])
        print(
            f"[ablation-pipeline] judge {name}: {plan_case['judge_command']}",
            flush=True,
        )
        subprocess.run(judge_cmd, check=True)

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
        str(args.base / "judge_media_smoke_ollama.json"),
        "--freeze-file",
        str(release_freeze),
        "--ablation-plan",
        str(args.ablation_plan),
        *ablation_args,
    ]
    print(f"[ablation-pipeline] final report: {shlex.join(report_cmd)}", flush=True)
    subprocess.run(report_cmd, check=True)
    print("[ablation-pipeline] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
