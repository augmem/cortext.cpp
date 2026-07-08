#!/usr/bin/env python3
"""Run one memory-eval retrieval pass and score it with available judge CLIs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = ROOT / "scripts" / "judges"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run(cmd: list[str], log_path: Path) -> int:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def command_available(name: str, env_bin: str) -> bool:
    return bool(shutil.which(os.environ.get(env_bin, name)))


def agy_available() -> bool:
    return bool(
        os.environ.get("CORTEXT_AGY_COMMAND")
        or shutil.which(os.environ.get("CORTEXT_AGY_BIN", "agy"))
        or shutil.which("antigravity")
    )


def available_judges() -> dict[str, bool]:
    return {
        "packet": True,
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "codex": command_available("codex", "CORTEXT_CODEX_BIN"),
        "grok": command_available("grok", "CORTEXT_GROK_BIN"),
        "agy": agy_available(),
    }


def judge_command(label: str) -> list[str]:
    wrapper = JUDGE_DIR / f"{label}_judge.py"
    return [
        "--mode",
        "harness",
        "--judge-label",
        label,
        "--judge-command",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))} {{prompt_file}}",
    ]


def resolve_judges(requested: str, include_unavailable: bool) -> tuple[list[str], dict[str, bool]]:
    availability = available_judges()
    if requested == "all":
        wanted = ["packet", "openai", "codex", "grok", "agy"]
    else:
        wanted = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in availability]
    if unknown:
        raise SystemExit(f"Unknown judge(s): {', '.join(unknown)}")
    if include_unavailable:
        return wanted, availability
    return [item for item in wanted if availability[item]], availability


def ensure_run(args: argparse.Namespace) -> Path:
    if args.run:
        return Path(args.run)
    out = Path(args.out) if args.out else Path("logs") / "memory_evals" / (
        "judges_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    cmd = [
        sys.executable,
        "scripts/run_memory_evals.py",
        "--profile",
        args.profile,
        "--benchmarks",
        args.benchmarks,
        "--out",
        str(out),
        "--answer-mode",
        "none",
        "--max-episodes",
        str(args.max_episodes),
        "--max-queries",
        str(args.max_queries),
    ]
    if args.no_prepare:
        cmd.append("--no-prepare")
    rc = run(cmd, out / "retrieval.log")
    if rc != 0:
        raise SystemExit(f"retrieval run failed with exit {rc}; see {out / 'retrieval.log'}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score memory evals with packet/OpenAI/CLI judge adapters."
    )
    parser.add_argument("--run", default="", help="Existing memory eval run directory.")
    parser.add_argument("--out", default="", help="Run directory when --run is omitted.")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=1)
    parser.add_argument(
        "--judges",
        default="all",
        help="Comma list from packet,openai,codex,grok,agy or all.",
    )
    parser.add_argument("--include-unavailable", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-packet-chars", type=int, default=12000)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    judges, availability = resolve_judges(args.judges, args.include_unavailable)
    run_dir = ensure_run(args)
    os.environ["CORTEXT_JUDGE_TIMEOUT_S"] = str(args.timeout_s)

    results: list[dict[str, Any]] = []
    for label in judges:
        if not availability[label]:
            results.append({"judge": label, "available": False, "returncode": 127})
            continue
        cmd = [
            sys.executable,
            "scripts/score_memory_eval_answers.py",
            "--run",
            str(run_dir),
            "--timeout-s",
            str(args.timeout_s),
            "--max-packet-chars",
            str(args.max_packet_chars),
        ]
        if label == "packet":
            cmd.extend(["--mode", "packet"])
            summary_path = run_dir / "answer_summary_packet.json"
            log_path = run_dir / "judge_packet.log"
        elif label == "openai":
            cmd.extend(["--mode", "openai"])
            summary_path = run_dir / "answer_summary_openai.json"
            log_path = run_dir / "judge_openai.log"
        else:
            cmd.extend(judge_command(label))
            summary_path = run_dir / f"answer_summary_harness_{label}.json"
            log_path = run_dir / f"judge_{label}.log"
        rc = run(cmd, log_path)
        result: dict[str, Any] = {
            "judge": label,
            "available": True,
            "returncode": rc,
            "log": str(log_path),
            "summary": str(summary_path),
        }
        if summary_path.exists():
            result["metrics"] = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append(result)

    output = {
        "run": str(run_dir),
        "generated_at": time.time(),
        "availability": availability,
        "results": results,
    }
    out_path = run_dir / "judge_runs.summary.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return 0 if all(item["returncode"] == 0 for item in results if item["available"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
