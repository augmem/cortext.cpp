#!/usr/bin/env python3
"""Run local early-warning judges from streamed Julie probe rows.

This helper is intentionally not a release gate. It waits for native probe
rows emitted by cortext_julie_live_run, materializes a partial summary, and
runs the same local judge adapter used by the final protocol. The outputs are
private fail-fast signals only; final claims still require the complete frozen
summary and release report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone


QUALITY_COMPOSITE_FIELDS = {
    "relevance": 1.0,
    "sufficiency": 1.0,
    "noise": -1.0,
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from judge_julie_live_run import build_timeline  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_json(path: pathlib.Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_milestones(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise RuntimeError("--milestones values must be positive")
        out.append(value)
    if not out:
        raise RuntimeError("--milestones must contain at least one value")
    return sorted(set(out))


def probe_rows_available(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError:
                continue
            count += 1
    return count


def load_probe_rows(path: pathlib.Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def partial_counts(args: argparse.Namespace, milestone: int) -> dict[str, int]:
    rows = load_probe_rows(args.probe_stream, milestone)
    max_event_index = max((int(row.get("event_index", -1)) for row in rows), default=-1)
    timeline = build_timeline(
        args.input_dir,
        args.timeline_max_messages,
        args.timeline_media_limit,
    )
    processed = [doc for doc in timeline if 0 <= doc.index <= max_event_index]
    text = sum(1 for doc in processed if doc.modality == "text")
    audio = sum(1 for doc in processed if doc.modality == "audio")
    video = sum(
        1
        for doc in processed
        if doc.modality == "image" and doc.text.startswith("[video source blob:")
    )
    image = sum(
        1
        for doc in processed
        if doc.modality == "image" and not doc.text.startswith("[video source blob:")
    )
    media = audio + image + video
    return {
        "processed_text_messages": text,
        "media_attempted": media,
        "media_processed": media,
        "audio_processed": audio,
        "image_processed": image,
        "video_processed": video,
        "media_failures": 0,
    }


def run_checked(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def quality_composite(system_quality: dict) -> float:
    total = 0.0
    for field, weight in QUALITY_COMPOSITE_FIELDS.items():
        try:
            total += float(system_quality.get(f"mean_{field}", 0.0) or 0.0) * weight
        except (TypeError, ValueError):
            pass
    return total


def judge_metrics(judge_path: pathlib.Path) -> dict:
    judge = json.loads(judge_path.read_text())
    quality = judge.get("quality", {})
    cortext = quality.get("cortext_native", {})
    rag = quality.get("traditional_chat_rag", {})
    full = quality.get("full_history_upper_bound", {})
    cortext_judged = int(cortext.get("judged", judge.get("judged", 0)) or 0)
    cortext_wins = int(cortext.get("wins", 0) or 0)
    rag_wins = int(rag.get("wins", 0) or 0)
    full_wins = int(full.get("wins", 0) or 0)
    cortext_quality = quality_composite(cortext)
    rag_quality = quality_composite(rag)
    full_quality = quality_composite(full)
    tokens = judge.get("tokens", {})
    return {
        "judged_rows": int(judge.get("judged", cortext_judged) or 0),
        "probe_count": int(judge.get("probe_count", 0) or 0),
        "cortext_wins": cortext_wins,
        "traditional_chat_rag_wins": rag_wins,
        "full_history_upper_bound_wins": full_wins,
        "cortext_win_rate": (
            cortext_wins / cortext_judged if cortext_judged > 0 else 0.0
        ),
        "cortext_quality_composite": cortext_quality,
        "traditional_chat_rag_quality_composite": rag_quality,
        "full_history_upper_bound_quality_composite": full_quality,
        "cortext_quality_delta_vs_traditional_chat_rag": (
            cortext_quality - rag_quality
        ),
        "cortext_quality_delta_vs_full_history_upper_bound": (
            cortext_quality - full_quality
        ),
        "cortext_token_savings_vs_traditional_chat_rag": float(
            tokens.get("mean_cortext_token_savings_vs_traditional_chat_rag", 0.0)
            or 0.0
        ),
        "mean_cortext_context_tokens": float(
            tokens.get("mean_cortext_context_tokens", 0.0) or 0.0
        ),
        "mean_traditional_chat_rag_tokens": float(
            tokens.get("mean_traditional_chat_rag_tokens", 0.0) or 0.0
        ),
        "fairness_checks": judge.get("fairness_checks", {}),
        "judge_validation": judge.get("judge_validation", {}),
    }


def check_floor(name: str, value: float, floor: float | None) -> dict | None:
    if floor is None:
        return None
    return {
        "name": name,
        "value": value,
        "floor": floor,
        "status": "pass" if value >= floor else "fail",
    }


def fail_fast_checks(args: argparse.Namespace, metrics: dict) -> list[dict]:
    checks: list[dict] = []
    for check in [
        check_floor(
            "cortext_win_rate",
            float(metrics["cortext_win_rate"]),
            args.min_cortext_win_rate,
        ),
        check_floor(
            "cortext_quality_delta_vs_traditional_chat_rag",
            float(metrics["cortext_quality_delta_vs_traditional_chat_rag"]),
            args.min_cortext_quality_delta_vs_rag,
        ),
        check_floor(
            "cortext_token_savings_vs_traditional_chat_rag",
            float(metrics["cortext_token_savings_vs_traditional_chat_rag"]),
            args.min_cortext_token_savings_vs_rag,
        ),
        check_floor(
            "mean_cortext_context_tokens",
            float(metrics["mean_cortext_context_tokens"]),
            args.min_mean_cortext_context_tokens,
        ),
    ]:
        if check is not None:
            checks.append(check)

    if args.require_fairness_checks:
        fairness = metrics.get("fairness_checks", {})
        for name, value in sorted(fairness.items()):
            if isinstance(value, bool):
                checks.append(
                    {
                        "name": f"fairness.{name}",
                        "value": value,
                        "expected": True,
                        "status": "pass" if value else "fail",
                    }
                )
    return checks


def materialize_partial(args: argparse.Namespace, milestone: int, summary_path: pathlib.Path) -> None:
    counts = partial_counts(args, milestone)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/materialize_julie_probe_stream_summary.py"),
        "--probe-stream",
        str(args.probe_stream),
        "--out",
        str(summary_path),
        "--input-dir",
        str(args.input_dir),
        "--db",
        str(args.db),
        "--processed-text-messages",
        str(counts["processed_text_messages"]),
        "--media-attempted",
        str(counts["media_attempted"]),
        "--media-processed",
        str(counts["media_processed"]),
        "--audio-processed",
        str(counts["audio_processed"]),
        "--image-processed",
        str(counts["image_processed"]),
        "--video-processed",
        str(counts["video_processed"]),
        "--media-failures",
        str(counts["media_failures"]),
        "--timeline-max-messages",
        str(args.timeline_max_messages),
        "--timeline-media-limit",
        str(args.timeline_media_limit),
        "--probe-limit",
        str(milestone),
        "--warmup-events",
        str(args.warmup_events),
        "--probe-stride",
        str(args.probe_stride),
        "--rag-top-k",
        str(args.rag_top_k),
        "--active-history-token-budget",
        str(args.active_history_token_budget),
        "--focus",
        str(args.focus),
        "--sensitivity",
        str(args.sensitivity),
        "--stability",
        str(args.stability),
    ]
    if args.daily_consolidation:
        cmd.append("--daily-consolidation")
    if args.deep:
        cmd.append("--deep")
    run_checked(cmd)


def run_judge(args: argparse.Namespace, milestone: int, summary_path: pathlib.Path, judge_path: pathlib.Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/judge_julie_live_run.py"),
        "--summary",
        str(summary_path),
        "--db",
        str(args.db),
        "--out",
        str(judge_path),
        "--judge-provider",
        args.judge_provider,
        "--model",
        args.model,
        "--judge-repetitions",
        str(args.judge_repetitions),
        "--judge-seed",
        str(args.judge_seed),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--judge-limit",
        str(milestone),
        "--max-media-per-system",
        str(args.max_media_per_system),
    ]
    if args.ollama_base_url:
        cmd.extend(["--ollama-base-url", args.ollama_base_url])
    if args.blind_packets:
        cmd.append("--blind-packets")
    run_checked(cmd)


def write_manifest(args: argparse.Namespace, completed: list[dict]) -> None:
    manifest = {
        "schema": "julie_probe_stream_early_judge_manifest_v1",
        "created_at_utc": utc_now(),
        "release_gate_use": "none_non_release_early_warning_only",
        "privacy": "private local artifact; may point to private summaries and judge rows",
        "probe_stream": str(args.probe_stream),
        "probe_stream_sha256": file_sha256(args.probe_stream)
        if args.probe_stream.exists()
        else "",
        "input_dir": str(args.input_dir),
        "db": str(args.db),
        "timeline_max_messages": args.timeline_max_messages,
        "timeline_media_limit": args.timeline_media_limit,
        "judge_provider": args.judge_provider,
        "judge_model": args.model,
        "judge_repetitions": args.judge_repetitions,
        "blind_packets": args.blind_packets,
        "completed": completed,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


def run_milestone(args: argparse.Namespace, milestone: int) -> dict:
    summary_path = args.out_dir / f"early_probe_{milestone:03d}_summary.json"
    judge_path = args.out_dir / f"early_probe_{milestone:03d}_judge.json"
    materialize_partial(args, milestone, summary_path)
    run_judge(args, milestone, summary_path, judge_path)
    metrics = judge_metrics(judge_path)
    checks = fail_fast_checks(args, metrics)
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "milestone": milestone,
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "judge": str(judge_path),
        "judge_sha256": file_sha256(judge_path),
        "metrics": metrics,
        "fail_fast_checks": checks,
        "fail_fast_status": status,
        "completed_at_utc": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-stream", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--db", type=pathlib.Path, required=True)
    parser.add_argument("--timeline-max-messages", type=int, required=True)
    parser.add_argument("--timeline-media-limit", type=int, required=True)
    parser.add_argument("--milestones", default="8,16,32")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--stop-after-last", action="store_true")
    parser.add_argument("--warmup-events", type=int, default=0)
    parser.add_argument("--probe-stride", type=int, default=0)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--active-history-token-budget", type=int, default=8000)
    parser.add_argument("--focus", type=float, default=0.5)
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--daily-consolidation", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--judge-provider", default="ollama", choices=["ollama", "nemotron"])
    parser.add_argument("--model", default="gemma4:12b-it-qat")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--judge-repetitions", type=int, default=1)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--max-media-per-system", type=int, default=-1)
    parser.add_argument(
        "--min-cortext-win-rate",
        type=float,
        help="Fail if an early judge milestone has lower Cortext win rate.",
    )
    parser.add_argument(
        "--min-cortext-quality-delta-vs-rag",
        type=float,
        help=(
            "Fail if Cortext's early quality composite minus traditional "
            "chat+RAG is below this value."
        ),
    )
    parser.add_argument(
        "--min-cortext-token-savings-vs-rag",
        type=float,
        help="Fail if early mean token savings versus traditional chat+RAG is below this value.",
    )
    parser.add_argument(
        "--min-mean-cortext-context-tokens",
        type=float,
        help="Fail if Cortext returns less mean context than this at a milestone.",
    )
    parser.add_argument(
        "--require-fairness-checks",
        action="store_true",
        help="Fail if any boolean judge fairness check is false.",
    )
    parser.add_argument(
        "--blind-packets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        raise RuntimeError("--poll-seconds must be positive")
    if args.judge_repetitions <= 0:
        raise RuntimeError("--judge-repetitions must be positive")

    args.probe_stream = args.probe_stream.resolve()
    args.out_dir = args.out_dir.resolve()
    args.input_dir = args.input_dir.resolve()
    args.db = args.db.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest is None:
        args.manifest = args.out_dir / "early_judge_manifest.json"

    milestones = parse_milestones(args.milestones)
    completed: list[dict] = []
    completed_milestones: set[int] = set()
    while True:
        available = probe_rows_available(args.probe_stream)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"probe_rows={available} completed={sorted(completed_milestones)}",
            flush=True,
        )
        for milestone in milestones:
            if milestone in completed_milestones or available < milestone:
                continue
            record = run_milestone(args, milestone)
            completed.append(record)
            completed_milestones.add(milestone)
            write_manifest(args, completed)
            if record["fail_fast_status"] == "fail":
                print(
                    "[early-judge] fail-fast threshold failed "
                    + json.dumps(record["fail_fast_checks"], sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
                return 2
        if completed_milestones == set(milestones) and args.stop_after_last:
            break
        time.sleep(args.poll_seconds)
    write_manifest(args, completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
