#!/usr/bin/env python3
"""Finalize the Julie release protocol after judge, ablations, and labels finish."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime


ABLATION_NAMES = [
    "no_daily_consolidation",
    "no_graph_expansion",
    "no_media_source_blobs",
    "no_stm_label_handoff",
    "no_temporal_retrieval",
    "no_fact_boosts",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--db", type=pathlib.Path, required=True)
    parser.add_argument("--main-judge", type=pathlib.Path, required=True)
    parser.add_argument("--target-freeze", type=pathlib.Path, required=True)
    parser.add_argument("--sample", type=pathlib.Path, required=True)
    parser.add_argument("--ablation-plan", type=pathlib.Path, required=True)
    parser.add_argument("--final-report", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-command", required=True)
    parser.add_argument("--judge-command", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    print(f"[finalizer] waiting for artifacts {datetime.now().isoformat()}", flush=True)
    for path in [args.summary, args.main_judge, args.target_freeze, args.ablation_plan]:
        wait_for_json(path)
        print(f"[finalizer] ready {path}", flush=True)

    wait_for_completed_sample(args.sample)
    print(f"[finalizer] completed human sample {args.sample}", flush=True)

    ablation_args: list[str] = []
    for name in ABLATION_NAMES:
        summary = args.base / f"ablation_{name}" / "summary.json"
        judge = args.base / f"ablation_{name}" / "gemma4_12b_ollama_blind_judge_reps3.json"
        wait_for_json(summary)
        wait_for_json(judge)
        print(f"[finalizer] ready ablation {name}", flush=True)
        ablation_args += ["--ablation", f"{name}:{summary}:{judge}"]

    human_frozen = args.base / "human_frozen_targets.json"
    human_score = args.base / "human_label_score.json"
    human_eval = args.base / "human_label_eval.json"
    score_cmd = [
        args.python,
        "tools/julie_human_label_harness.py",
        "score",
        "--sample",
        str(args.sample),
        "--summary",
        str(args.summary),
        "--db",
        str(args.db),
        "--judge-frozen",
        str(args.target_freeze),
        "--out-frozen",
        str(human_frozen),
        "--out-report",
        str(human_score),
        "--eval-out",
        str(human_eval),
        "--keep-unanswerable",
    ]
    print(f"[finalizer] score human labels: {' '.join(score_cmd)}", flush=True)
    subprocess.run(score_cmd, check=True)

    report_cmd = [
        "python3",
        "tools/julie_release_protocol_report.py",
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
        "--human-labels",
        str(human_score),
        "--ablation-plan",
        str(args.ablation_plan),
        *ablation_args,
    ]
    print(f"[finalizer] final report: {' '.join(report_cmd)}", flush=True)
    subprocess.run(report_cmd, check=True)
    print(f"[finalizer] complete {args.final_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
