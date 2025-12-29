#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def parse_metrics(log_path: Path) -> dict:
    metrics = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("===") or line.startswith("[LOG]"):
                continue
            parts = line.split()
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    metrics[key] = value
    return metrics


def parse_float_list(raw: str) -> list[float]:
    return [float(v) for v in raw.split(",") if v.strip()]


def parse_cases(raw: str) -> list[tuple[float, float, float]]:
    cases: list[tuple[float, float, float]] = []
    entries = raw.strip()
    if entries.startswith("@"):
        path = Path(entries[1:]).expanduser()
        entries = path.read_text(encoding="utf-8")
    for chunk in entries.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid case '{chunk}'. Expected f,s,t.")
        f, s, t = (float(parts[0]), float(parts[1]), float(parts[2]))
        cases.append((f, s, t))
    return cases


def write_git_state(out_dir: Path) -> None:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        (out_dir / "git_rev.txt").write_text(rev + "\n", encoding="utf-8")
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "-sb"], text=True)
        (out_dir / "git_status.txt").write_text(status, encoding="utf-8")
    except Exception:
        pass
    try:
        diff = subprocess.check_output(["git", "diff"], text=True)
        (out_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    except Exception:
        pass


def write_status(run_dir: Path, state: str, payload: dict | None = None) -> None:
    data = {
        "state": state,
        "timestamp": time.time(),
    }
    if payload:
        data.update(payload)
    (run_dir / "status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_case(case: dict, args: argparse.Namespace, out_root: Path, summary_rows: list):
    print(f"[RUN] {case['name']}")
    run_dir = out_root / case["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "cortext.db"
    log_path = run_dir / "run.log"

    cmd = [
        args.binary,
        f"--data={args.data}",
        f"--models={args.models}",
        f"--db={db_path}",
        f"--focus={case['focus']}",
        f"--sensitivity={case['sensitivity']}",
        f"--stability={case['stability']}",
        f"--max-conversations={case['max_conversations']}",
        f"--max-turns={case['max_turns']}",
        f"--max-total={case['max_total']}",
        "--no-cadence",
        "--semantic",
    ]
    if case.get("reuse", True):
        cmd.append("--reuse")
    if case.get("interleave", 1) > 1:
        cmd.append(f"--interleave={case['interleave']}")
    if args.consolidate_cycles > 0:
        cmd.append("--consolidate")
        cmd.append(f"--consolidate-cycles={args.consolidate_cycles}")

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    write_status(run_dir, "starting", {"cmd": cmd})
    start_time = time.time()
    returncode = 1
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            write_status(run_dir, "running", {"pid": proc.pid})
            last_heartbeat = time.time()
            while True:
                ret = proc.poll()
                now = time.time()
                if now - last_heartbeat >= 5.0:
                    write_status(run_dir, "running",
                                 {"pid": proc.pid, "elapsed_s": now - start_time})
                    last_heartbeat = now
                if ret is not None:
                    returncode = ret
                    break
                time.sleep(0.5)
    except Exception as exc:
        (run_dir / "exception.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
        write_status(run_dir, "exception", {"error": str(exc)})
        raise
    duration_s = time.time() - start_time
    (run_dir / "returncode.txt").write_text(f"{returncode}\n", encoding="utf-8")
    write_status(run_dir, "finished",
                 {"returncode": returncode, "duration_s": duration_s})
    if returncode != 0:
        print(f"[WARN] Run {case['name']} failed with code {returncode}")

    metrics = parse_metrics(log_path)
    row = {
        "name": case["name"],
        "focus": case["focus"],
        "sensitivity": case["sensitivity"],
        "stability": case["stability"],
        "turns": metrics.get("turns", "0"),
        "writes": metrics.get("writes", "0"),
        "consolidation_runs": metrics.get("consolidation_runs", "0"),
        "consolidation_failures": metrics.get("consolidation_failures", "0"),
        "consolidation_every_turns": metrics.get("consolidation_every_turns", "0"),
        "consolidation_association_created": metrics.get("consolidation_association_created", "0"),
        "consolidation_label_created": metrics.get("consolidation_label_created", "0"),
        "consolidation_summary_count": metrics.get("consolidation_summary_count", "0"),
        "consolidation_summaries_with_model": metrics.get("consolidation_summaries_with_model", "0"),
        "consolidation_summaries_fallback": metrics.get("consolidation_summaries_fallback", "0"),
        "consolidation_extraction_jobs": metrics.get("consolidation_extraction_jobs", "0"),
        "consolidation_extraction_results": metrics.get("consolidation_extraction_results", "0"),
        "consolidation_labels_seen": metrics.get("consolidation_labels_seen", "0"),
        "consolidation_relations_seen": metrics.get("consolidation_relations_seen", "0"),
        "returncode": str(returncode),
        "duration_s": f"{duration_s:.2f}",
        "retrieval_turn_rate": metrics.get("retrieval_turn_rate", "0"),
        "retrieval_avg_candidates": metrics.get("retrieval_avg_candidates", "0"),
        "retrieval_overlap_mean": metrics.get("retrieval_overlap_mean", "0"),
        "retrieval_context_overlap_mean": metrics.get("retrieval_context_overlap_mean", "0"),
        "retrieval_semantic_overlap_mean": metrics.get("retrieval_semantic_overlap_mean", "0"),
        "retrieval_context_semantic_overlap_mean": metrics.get("retrieval_context_semantic_overlap_mean", "0"),
        "retrieval_association_candidate_rate": metrics.get("retrieval_association_candidate_rate", "0"),
        "retrieval_label_candidate_rate": metrics.get("retrieval_label_candidate_rate", "0"),
        "retrieval_association_turn_rate": metrics.get("retrieval_association_turn_rate", "0"),
        "retrieval_label_turn_rate": metrics.get("retrieval_label_turn_rate", "0"),
        "retrieval_summary_hit_rate": metrics.get("retrieval_summary_hit_rate", "0"),
        "retrieval_summary_only_turn_rate": metrics.get("retrieval_summary_only_turn_rate", "0"),
        "summary_hit_overlap_mean": metrics.get("summary_hit_overlap_mean", "0"),
        "interrupt_turn_rate": metrics.get("interrupt_turn_rate", "0"),
        "interrupt_precision": metrics.get("interrupt_precision", "0"),
        "interrupt_recall": metrics.get("interrupt_recall", "0"),
        "interrupt_false_positive_rate": metrics.get("interrupt_false_positive_rate", "0"),
        "interrupt_false_negative_rate": metrics.get("interrupt_false_negative_rate", "0"),
        "interrupt_true_positive": metrics.get("interrupt_true_positive", "0"),
        "interrupt_false_positive": metrics.get("interrupt_false_positive", "0"),
        "interrupt_false_negative": metrics.get("interrupt_false_negative", "0"),
        "interrupt_semantic_overlap_mean": metrics.get("interrupt_semantic_overlap_mean", "0"),
        "interrupt_context_semantic_overlap_mean": metrics.get("interrupt_context_semantic_overlap_mean", "0"),
        "non_interrupt_semantic_overlap_mean": metrics.get("non_interrupt_semantic_overlap_mean", "0"),
        "non_interrupt_context_semantic_overlap_mean": metrics.get("non_interrupt_context_semantic_overlap_mean", "0"),
        "interrupt_semantic_delta": metrics.get("interrupt_semantic_delta", "0"),
        "interrupt_context_semantic_delta": metrics.get("interrupt_context_semantic_delta", "0"),
        "interrupt_association_candidate_rate": metrics.get("interrupt_association_candidate_rate", "0"),
        "interrupt_label_candidate_rate": metrics.get("interrupt_label_candidate_rate", "0"),
        "interrupt_association_turn_rate": metrics.get("interrupt_association_turn_rate", "0"),
        "interrupt_label_turn_rate": metrics.get("interrupt_label_turn_rate", "0"),
    }
    summary_rows.append(row)


def write_summary(summary_path: Path, summary_rows: list) -> None:
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cortext memory harness")
    parser.add_argument("--binary", default="build/examples/topical_chat_analysis/cortext_topical_chat_analysis")
    parser.add_argument("--data", default="data/topical_chat/valid_freq.jsonl")
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-conversations", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-total", type=int, default=400)
    parser.add_argument("--multi-interleave", type=int, default=4)
    parser.add_argument("--multi-total", type=int, default=800)
    parser.add_argument("--consolidate-cycles", type=int, default=2)
    parser.add_argument("--sweep", default="0.4,0.5,0.6")
    parser.add_argument("--focus-list", default="")
    parser.add_argument("--sensitivity-list", default="")
    parser.add_argument("--stability-list", default="")
    parser.add_argument("--cases", default="")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-multi", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else Path("logs") / "memory_harness" / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    write_git_state(out_root)

    summary_rows = []
    summary_path = out_root / "summary.csv"
    base_case = {
        "name": "baseline_050",
        "focus": 0.5,
        "sensitivity": 0.5,
        "stability": 0.5,
        "max_conversations": args.max_conversations,
        "max_turns": args.max_turns,
        "max_total": args.max_total,
        "interleave": 1,
        "reuse": True,
    }
    if not args.no_baseline:
        run_case(base_case, args, out_root, summary_rows)
        write_summary(summary_path, summary_rows)

    if not args.no_multi:
        multi_case = dict(base_case)
        multi_case.update({
            "name": "multi_participant",
            "max_conversations": args.max_conversations,
            "max_turns": args.max_turns,
            "max_total": args.multi_total,
            "interleave": max(1, args.multi_interleave),
        })
        run_case(multi_case, args, out_root, summary_rows)
        write_summary(summary_path, summary_rows)

    if not args.no_sweep:
        cases = []
        if args.cases:
            cases = parse_cases(args.cases)
        if cases:
            for f, s, t in cases:
                case = dict(base_case)
                case.update({
                    "name": f"case_F{f:.2f}_S{s:.2f}_T{t:.2f}",
                    "focus": f,
                    "sensitivity": s,
                    "stability": t,
                    "max_conversations": 1,
                    "max_turns": min(120, args.max_turns),
                    "max_total": min(120, args.max_total),
                    "interleave": 1,
                })
                run_case(case, args, out_root, summary_rows)
                write_summary(summary_path, summary_rows)
        else:
            vals = parse_float_list(args.sweep) if args.sweep else []
            focus_vals = parse_float_list(args.focus_list) if args.focus_list else vals
            sens_vals = parse_float_list(args.sensitivity_list) if args.sensitivity_list else vals
            stab_vals = parse_float_list(args.stability_list) if args.stability_list else vals
            for f in focus_vals:
                for s in sens_vals:
                    for t in stab_vals:
                        case = dict(base_case)
                        case.update({
                            "name": f"sweep_F{f:.2f}_S{s:.2f}_T{t:.2f}",
                            "focus": f,
                            "sensitivity": s,
                            "stability": t,
                            "max_conversations": 1,
                            "max_turns": min(120, args.max_turns),
                            "max_total": min(120, args.max_total),
                            "interleave": 1,
                        })
                        run_case(case, args, out_root, summary_rows)
                        write_summary(summary_path, summary_rows)

    write_summary(summary_path, summary_rows)
    print(f"Wrote harness output to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
