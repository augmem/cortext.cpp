#!/usr/bin/env python3
"""Finalize the chat-replay release protocol after judge, ablations, and labels finish."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ABLATION_NAMES = [
    "no_daily_consolidation",
    "no_graph_expansion",
    "no_media_source_blobs",
    "no_stm_ltm_graph_label_handoff",
    "no_temporal_retrieval",
    "no_fact_boosts",
]
DEFAULT_ABLATION_JUDGE_FILENAME = "gemma4_12b_ollama_blind_judge_reps3.json"
DEFAULT_JUDGE_MODEL = "gemma4:12b-it-qat"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
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


def sanitized_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in sorted(key for key in env if is_hosted_provider_env_key(key)):
        env.pop(key, None)
    return env


def run_checked(cmd: list[str], log_path: pathlib.Path | None = None) -> None:
    print(f"[finalizer] run: {shlex.join(cmd)}", flush=True)
    env = sanitized_subprocess_env()
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def command_flag_value(command: str, flag: str, default: str = "") -> str:
    parts = shlex.split(command)
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts):
            return parts[i + 1]
    return default


def command_int_flag_value(command: str, flag: str, default: int) -> int:
    raw = command_flag_value(command, flag, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{flag} must be an integer in --judge-command") from exc


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


def sample_complete(path: pathlib.Path) -> bool:
    if not is_json(path):
        return False
    sample = json.loads(path.read_text())
    tasks = sample.get("tasks", [])
    if not tasks:
        return False
    candidate_count = 0
    for task in tasks:
        candidates = task.get("candidates", [])
        labels = task.get("labels", {})
        candidate_count += len(candidates)
        for cand in candidates:
            if cand.get("candidate_id") not in labels:
                return False
    return candidate_count > 0


def wait_for_completed_sample(path: pathlib.Path) -> None:
    fswatch = shutil.which("fswatch")
    while not sample_complete(path):
        if fswatch:
            subprocess.run(
                [fswatch, "-1", str(path.parent)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            time.sleep(5)


def resolve_plan_path(raw: object, fallback: pathlib.Path) -> pathlib.Path:
    if raw is None or str(raw).strip() == "":
        return fallback
    path = pathlib.Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def fallback_ablation_artifacts(
    base: pathlib.Path,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    artifacts = []
    for name in ABLATION_NAMES:
        out_dir = base / f"ablation_{name}"
        artifacts.append(
            (
                name,
                out_dir / "summary.json",
                out_dir / DEFAULT_ABLATION_JUDGE_FILENAME,
            )
        )
    return artifacts


def required_ablation_artifacts(
    base: pathlib.Path,
    ablation_plan: pathlib.Path,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    plan = json.loads(ablation_plan.read_text())
    if not isinstance(plan, dict) or "cases" not in plan:
        return fallback_ablation_artifacts(base)

    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("ablation_plan cases must be a non-empty list")

    artifacts = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise RuntimeError(f"ablation_plan case {index} must be an object")
        name = str(case.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"ablation_plan case {index} is missing name")
        if name in seen:
            raise RuntimeError(f"duplicate ablation_plan case name: {name}")
        seen.add(name)

        out_dir = base / f"ablation_{name}"
        artifacts.append(
            (
                name,
                resolve_plan_path(case.get("summary_path"), out_dir / "summary.json"),
                resolve_plan_path(
                    case.get("judge_path"),
                    out_dir / DEFAULT_ABLATION_JUDGE_FILENAME,
                ),
            )
        )
    return artifacts


def judge_media_smoke_needs_generation(
    judge_media_smoke: pathlib.Path,
    judge_model: str,
) -> bool:
    if not judge_media_smoke.exists():
        return True
    try:
        smoke = json.loads(judge_media_smoke.read_text())
    except Exception:
        return True
    if smoke.get("schema") != "cortext_local_ollama_judge_media_smoke_v1":
        return True
    if smoke.get("private_data_used") is not False:
        return True
    if smoke.get("selected_release_judge_model") != judge_model:
        return True
    selected = (smoke.get("results") or {}).get(judge_model, {})
    if not isinstance(selected, dict):
        return True
    image = selected.get("image", {}) if isinstance(selected, dict) else {}
    audio = selected.get("audio", {}) if isinstance(selected, dict) else {}
    image_parsed = image.get("parsed", {}) if isinstance(image, dict) else {}
    audio_parsed = audio.get("parsed", {}) if isinstance(audio, dict) else {}
    return not (
        isinstance(image_parsed, dict)
        and isinstance(audio_parsed, dict)
        and image_parsed.get("image_seen") is True
        and audio_parsed.get("audio_seen") is True
    )


def ensure_judge_media_smoke(
    args: argparse.Namespace,
    judge_media_smoke: pathlib.Path,
) -> None:
    judge_model = command_flag_value(
        args.judge_command,
        "--model",
        DEFAULT_JUDGE_MODEL,
    )
    ollama_base_url = command_flag_value(
        args.judge_command,
        "--ollama-base-url",
        args.ollama_base_url,
    )
    timeout_s = command_int_flag_value(args.judge_command, "--judge-timeout-s", 420)
    num_ctx = command_int_flag_value(
        args.judge_command,
        "--judge-context-window-tokens",
        32768,
    )
    if not judge_media_smoke_needs_generation(judge_media_smoke, judge_model):
        print(f"[finalizer] ready judge media smoke {judge_media_smoke}", flush=True)
        return
    cmd = [
        args.python,
        "tools/smoke_ollama_judge_media.py",
        "--out",
        str(judge_media_smoke),
        "--model",
        judge_model,
        "--ollama-base-url",
        ollama_base_url,
        "--timeout-s",
        str(timeout_s),
        "--num-ctx",
        str(num_ctx),
    ]
    judge_media_smoke.parent.mkdir(parents=True, exist_ok=True)
    (judge_media_smoke.parent / "judge_media_smoke_command.txt").write_text(
        shlex.join(cmd) + "\n"
    )
    run_checked(cmd, judge_media_smoke.parent / "judge_media_smoke_pipeline.log")
    if judge_media_smoke_needs_generation(judge_media_smoke, judge_model):
        raise RuntimeError(
            "judge media smoke was not created or did not prove image/audio support: "
            f"{judge_media_smoke}"
        )
    print(f"[finalizer] generated judge media smoke {judge_media_smoke}", flush=True)


def ensure_target_freeze(args: argparse.Namespace, target_freeze: pathlib.Path) -> None:
    if is_json(target_freeze):
        print(f"[finalizer] ready target freeze {target_freeze}", flush=True)
        return
    cmd = [
        args.python,
        "tools/frozen_chat_replay_retrieval_eval.py",
        "judge-freeze",
        "--summary",
        str(args.summary),
        "--db",
        str(args.db),
        "--out",
        str(target_freeze),
        "--judge-provider",
        "ollama",
        "--model",
        args.target_freeze_model,
        "--ollama-base-url",
        args.ollama_base_url,
        "--max-targets",
        str(args.target_freeze_max_targets),
        "--min-overlap",
        str(args.target_freeze_min_overlap),
        "--max-candidates",
        str(args.target_freeze_max_candidates),
        "--keep-unanswerable",
    ]
    target_freeze.parent.mkdir(parents=True, exist_ok=True)
    (target_freeze.parent / "target_freeze_command.txt").write_text(
        shlex.join(cmd) + "\n"
    )
    run_checked(cmd, target_freeze.parent / "target_freeze_pipeline.log")
    if not is_json(target_freeze):
        raise RuntimeError(f"target freeze was not created: {target_freeze}")
    print(f"[finalizer] generated target freeze {target_freeze}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--db", type=pathlib.Path, required=True)
    parser.add_argument("--main-judge", type=pathlib.Path, required=True)
    parser.add_argument("--target-freeze", type=pathlib.Path)
    parser.add_argument("--sample", type=pathlib.Path, required=True)
    parser.add_argument("--ablation-plan", type=pathlib.Path, required=True)
    parser.add_argument("--final-report", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-command", required=True)
    parser.add_argument("--judge-command", required=True)
    parser.add_argument("--judge-media-smoke", type=pathlib.Path)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--target-freeze-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--target-freeze-max-targets", type=int, default=3)
    parser.add_argument("--target-freeze-min-overlap", type=float, default=0.20)
    parser.add_argument("--target-freeze-max-candidates", type=int, default=12)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    args.base = args.base.resolve()
    args.summary = args.summary.resolve()
    args.db = args.db.resolve()
    args.main_judge = args.main_judge.resolve()
    args.sample = args.sample.resolve()
    args.ablation_plan = args.ablation_plan.resolve()
    args.final_report = args.final_report.resolve()
    target_freeze = (
        args.target_freeze.resolve()
        if args.target_freeze is not None
        else args.base / "judge_frozen_targets.json"
    )
    judge_media_smoke = (
        args.judge_media_smoke.resolve()
        if args.judge_media_smoke is not None
        else args.base / "judge_media_smoke_ollama.json"
    )
    if args.target_freeze_max_targets < 1:
        raise RuntimeError("--target-freeze-max-targets must be >= 1")
    if args.target_freeze_max_candidates < 1:
        raise RuntimeError("--target-freeze-max-candidates must be >= 1")
    if args.target_freeze_min_overlap < 0.0:
        raise RuntimeError("--target-freeze-min-overlap must be non-negative")

    print(f"[finalizer] waiting for artifacts {datetime.now().isoformat()}", flush=True)
    for path in [args.summary, args.main_judge, args.ablation_plan]:
        wait_for_json(path)
        print(f"[finalizer] ready {path}", flush=True)

    release_freeze = args.base / "release_protocol_freeze.json"
    wait_for_json(release_freeze)
    print(f"[finalizer] ready release freeze {release_freeze}", flush=True)
    ensure_judge_media_smoke(args, judge_media_smoke)
    ensure_target_freeze(args, target_freeze)

    wait_for_completed_sample(args.sample)
    print(f"[finalizer] completed human sample {args.sample}", flush=True)

    ablation_args: list[str] = []
    for name, summary, judge in required_ablation_artifacts(
        args.base,
        args.ablation_plan,
    ):
        wait_for_json(summary)
        wait_for_json(judge)
        print(f"[finalizer] ready ablation {name}", flush=True)
        ablation_args += ["--ablation", f"{name}:{summary}:{judge}"]

    human_frozen = args.base / "human_frozen_targets.json"
    human_score = args.base / "human_label_score.json"
    human_eval = args.base / "human_label_eval.json"
    score_cmd = [
        args.python,
        "tools/chat_replay_human_label_harness.py",
        "score",
        "--sample",
        str(args.sample),
        "--summary",
        str(args.summary),
        "--db",
        str(args.db),
        "--judge-frozen",
        str(target_freeze),
        "--out-frozen",
        str(human_frozen),
        "--out-report",
        str(human_score),
        "--eval-out",
        str(human_eval),
        "--keep-unanswerable",
    ]
    run_checked(score_cmd, args.base / "human_label_score_pipeline.log")

    report_cmd = [
        args.python,
        "tools/chat_replay_release_protocol_report.py",
        "--summary",
        str(args.summary),
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
        "--human-labels",
        str(human_score),
        "--human-label-eval",
        str(human_eval),
        "--target-freeze",
        str(target_freeze),
        "--ablation-plan",
        str(args.ablation_plan),
        "--require-pass",
        *ablation_args,
    ]
    run_checked(report_cmd, args.base / "final_report_pipeline.log")
    print(f"[finalizer] complete {args.final_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
