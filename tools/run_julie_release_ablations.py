#!/usr/bin/env python3
"""Run Julie release ablations and emit an explicit ablation provenance plan."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    wait_for_json(args.main_report)
    print(f"[ablation-pipeline] main report ready {datetime.now().isoformat()}", flush=True)

    common = [
        "./build/examples/benchmark/cortext_julie_live_run",
        "--input-dir",
        "build/julie_mixed_media_native_input",
        "--max-messages",
        "1200",
        "--media-limit",
        "16",
        "--probe-stride",
        "100",
        "--warmup-events",
        "200",
        "--deep",
        "--rag-top-k",
        "5",
        "--active-history-token-budget",
        "8000",
        "--focus",
        "0.5",
        "--sensitivity",
        "0.5",
        "--stability",
        "0.5",
    ]

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
            "--bootstrap-samples",
            "1000",
            "--judge-timeout-s",
            "180",
            "--blind-packets",
        ]
        plan_cases.append(
            {
                "name": name,
                "daily_consolidation": daily,
                "env_overrides": env_overrides,
                "summary_path": str(summary),
                "judge_path": str(judge),
                "db_path": str(db),
                "benchmark_command": shlex.join(bench_cmd),
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
        summary.parent.mkdir(parents=True, exist_ok=True)

        if not is_json(summary):
            env = os.environ.copy()
            env.update(env_overrides)
            bench_cmd = shlex.split(plan_case["benchmark_command"])
            print(
                "[ablation-pipeline] run "
                f"{name}: {plan_case['benchmark_command']} "
                f"env_overrides={env_overrides}",
                flush=True,
            )
            subprocess.run(bench_cmd, check=True, env=env)
        else:
            print(f"[ablation-pipeline] reuse summary {summary}", flush=True)

        if not is_json(judge):
            judge_cmd = shlex.split(plan_case["judge_command"])
            print(
                f"[ablation-pipeline] judge {name}: {plan_case['judge_command']}",
                flush=True,
            )
            subprocess.run(judge_cmd, check=True)
        else:
            print(f"[ablation-pipeline] reuse judge {judge}", flush=True)

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
