#!/usr/bin/env python3
"""Prepare and run normalized memory evals with cortext_memory_eval_runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def safe_label(value: str) -> str:
    chars: list[str] = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        elif ch in (" ", ".", "/"):
            chars.append("_")
    return "".join(chars).strip("_")


def run(cmd: list[str], log_path: Path | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.call(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def ensure_binary(binary: Path, build_dir: Path) -> None:
    if binary.exists():
        return
    if run(["cmake", "-S", ".", "-B", str(build_dir), "-DCORTEXT_BUILD_TOOLS=ON"]) != 0:
        raise SystemExit("CMake configure failed")
    if run(["cmake", "--build", str(build_dir), "--target", "cortext_memory_eval_runner", "-j"]) != 0:
        raise SystemExit("cortext_memory_eval_runner build failed")
    if not binary.exists():
        raise SystemExit(f"Expected runner at {binary}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Cortext memory eval adapters.")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--raw-root", default="data/raw/memory_evals")
    parser.add_argument("--prepared-root", default="data/memory_evals")
    parser.add_argument("--out", default="")
    parser.add_argument("--binary", default="build/tools/memory_eval_runner/cortext_memory_eval_runner")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--focus", type=float, default=0.5)
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--prepare-limit-episodes", type=int, default=-1)
    parser.add_argument("--prepare-limit-queries-per-episode", type=int, default=-1)
    parser.add_argument("--prepare-limit-turns-per-episode", type=int, default=-1)
    parser.add_argument("--prepare-limit-context-chars", type=int, default=-1)
    parser.add_argument("--prepare-limit-lme-v2-trajectories", type=int, default=-1)
    parser.add_argument("--prepare-limit-lme-v2-states", type=int, default=-1)
    parser.add_argument(
        "--answer-mode",
        choices=["none", "packet", "openai", "command", "harness"],
        default="packet",
        help="Generate and score answers after retrieval runs.",
    )
    parser.add_argument("--answer-model", default="")
    parser.add_argument(
        "--judge-command",
        default="",
        help=(
            "Shell command for --answer-mode command or harness. The judge prompt is sent on stdin; "
            "{prompt_file} and {input_json} are expanded by the scorer."
        ),
    )
    parser.add_argument(
        "--judge-label",
        default="",
        help="Optional label for command-judge output directories, such as codex or grok.",
    )
    args = parser.parse_args()

    binary = Path(args.binary)
    ensure_binary(binary, Path(args.build_dir))

    prepared_root = Path(args.prepared_root)
    if not args.no_prepare:
        prepare_cmd = [
            sys.executable,
            "scripts/prepare_memory_evals.py",
            "--benchmarks",
            args.benchmarks,
            "--profile",
            args.profile,
            "--raw-root",
            args.raw_root,
            "--out-root",
            str(prepared_root),
        ]
        if args.include_large:
            prepare_cmd.append("--include-large")
        prepare_overrides = {
            "--limit-episodes": args.prepare_limit_episodes,
            "--limit-queries-per-episode": args.prepare_limit_queries_per_episode,
            "--limit-turns-per-episode": args.prepare_limit_turns_per_episode,
            "--limit-context-chars": args.prepare_limit_context_chars,
            "--limit-lme-v2-trajectories": args.prepare_limit_lme_v2_trajectories,
            "--limit-lme-v2-states": args.prepare_limit_lme_v2_states,
        }
        for flag, value in prepare_overrides.items():
            if value >= 0:
                prepare_cmd.extend([flag, str(value)])
        rc = run(prepare_cmd)
        if rc != 0:
            return rc

    manifest_path = prepared_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(args.out) if args.out else Path("logs") / "memory_evals" / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for entry in manifest.get("benchmarks", []):
        name = entry["name"]
        result_path = out_root / name / "query_rows.jsonl"
        cmd = [
            str(binary),
            f"--episodes={entry['episodes']}",
            f"--answer-key={entry['answer_key']}",
            f"--out={result_path}",
            f"--benchmark={name}",
            f"--top-k={args.top_k}",
            f"--focus={args.focus}",
            f"--sensitivity={args.sensitivity}",
            f"--stability={args.stability}",
        ]
        if args.max_episodes > 0:
            cmd.append(f"--max-episodes={args.max_episodes}")
        if args.max_queries > 0:
            cmd.append(f"--max-queries={args.max_queries}")
        rc = run(cmd, out_root / name / "run.log")
        summary_path = result_path.with_suffix(".summary.json")
        summary = {
            "name": name,
            "returncode": rc,
            "summary": str(summary_path),
            "query_rows": str(result_path),
            "run_log": str(out_root / name / "run.log"),
        }
        if summary_path.exists():
            summary["metrics"] = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)

    aggregate_path = out_root / "summary.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "prepared_manifest": str(manifest_path),
                "runs": summaries,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {aggregate_path}")
    if args.answer_mode != "none" and all(row["returncode"] == 0 for row in summaries):
        answer_cmd = [
            sys.executable,
            "scripts/score_memory_eval_answers.py",
            "--run",
            str(out_root),
            "--mode",
            args.answer_mode,
        ]
        if args.answer_model:
            answer_cmd.extend(["--model", args.answer_model])
        if args.judge_command:
            answer_cmd.extend(["--judge-command", args.judge_command])
        if args.judge_label:
            answer_cmd.extend(["--judge-label", args.judge_label])
        answer_log_tag = args.answer_mode
        if args.answer_mode in ("command", "harness") and args.judge_label:
            label = safe_label(args.judge_label)
            if label:
                answer_log_tag = f"{args.answer_mode}_{label}"
        rc = run(answer_cmd, out_root / f"answers_{answer_log_tag}.log")
        if rc != 0:
            return rc
    return 0 if all(row["returncode"] == 0 for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
