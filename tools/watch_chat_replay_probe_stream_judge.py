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
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone


QUALITY_COMPOSITE_FIELDS = {
    "relevance": 1.0,
    "sufficiency": 1.0,
    "noise": -0.25,
}
FAIRNESS_EXPECTED_VALUES = {
    "cortext_audio_image_transcript_shortcuts": False,
    "traditional_chat_rag_text_only": True,
    "full_history_text_only": True,
    "no_future_context_violations": True,
    "no_current_turn_context_inclusions": True,
    "blind_prompt_hidden_labels_absent": True,
    "traditional_chat_rag_contentful_compaction": True,
    "judge_prompt_fits_context_window": True,
    "full_history_prompt_fits_judge_context": True,
    "judge_supports_attached_audio": True,
    "judge_supports_attached_images": True,
    "attached_audio_judged_when_present": True,
    "attached_images_judged_when_present": True,
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from judge_julie_live_run import FIELDS, SYSTEMS, build_timeline, confidence_intervals  # noqa: E402


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
        args.timeline_skip_messages,
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


def signal_process(pid: int, sig: signal.Signals, label: str) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, sig)
        print(f"[early-judge] sent {sig.name} to {label} pid={pid}", flush=True)
        return True
    except ProcessLookupError:
        print(f"[early-judge] {label} pid={pid} is no longer running", flush=True)
    except PermissionError as exc:
        print(
            f"[early-judge] cannot signal {label} pid={pid}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return False


def quality_composite(system_quality: dict) -> float:
    total = 0.0
    for field, weight in QUALITY_COMPOSITE_FIELDS.items():
        try:
            total += float(system_quality.get(f"mean_{field}", 0.0) or 0.0) * weight
        except (TypeError, ValueError):
            pass
    return total


def row_quality_composite(row: dict, system: str) -> float:
    scores = row.get("systems", {}).get(system, {})
    if not isinstance(scores, dict):
        return 0.0
    total = 0.0
    for field, weight in QUALITY_COMPOSITE_FIELDS.items():
        total += float(scores.get(field, 0.0) or 0.0) * weight
    return total


def prior_quality_delta_stats(segment_judges: list[pathlib.Path]) -> tuple[float, int]:
    seen: set[tuple[int, int]] = set()
    total = 0.0
    count = 0
    for segment in segment_judges:
        row_path = segment.with_name(segment.name + ".rows.jsonl")
        for row in load_jsonl(row_path):
            key = (int(row.get("event_index", -1)), int(row.get("repetition", 0)))
            if key in seen:
                continue
            seen.add(key)
            total += row_quality_composite(row, "cortext_native") - row_quality_composite(
                row, "traditional_chat_rag"
            )
            count += 1
    return total, count


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_json_if_valid(path: pathlib.Path) -> dict:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_resume_state(args: argparse.Namespace) -> tuple[list[dict], set[int], list[pathlib.Path]]:
    manifest = load_json_if_valid(args.manifest)
    if not manifest:
        return [], set(), []
    try:
        args.benchmark_pause_ms_total = float(
            manifest.get("benchmark_pause_ms_total", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        args.benchmark_pause_ms_total = 0.0
    try:
        args.benchmark_pause_count = int(
            manifest.get("benchmark_pause_count", 0) or 0
        )
    except (TypeError, ValueError):
        args.benchmark_pause_count = 0

    manifest_probe_stream = manifest.get("probe_stream", "")
    if manifest_probe_stream:
        previous_probe_stream = pathlib.Path(str(manifest_probe_stream)).resolve()
        if previous_probe_stream != args.probe_stream:
            raise RuntimeError(
                "refusing to resume early judge from a manifest for a different "
                f"probe stream: {previous_probe_stream} != {args.probe_stream}"
            )

    for name, current in (
        ("judge_provider", args.judge_provider),
        ("judge_model", args.model),
        ("judge_repetitions", args.judge_repetitions),
    ):
        previous = manifest.get(name)
        if previous not in (None, current):
            raise RuntimeError(
                "refusing to resume early judge with changed "
                f"{name}: {previous!r} != {current!r}"
            )

    completed: list[dict] = []
    completed_milestones: set[int] = set()
    segment_judges: list[pathlib.Path] = []
    for item in manifest.get("completed", []):
        if not isinstance(item, dict):
            continue
        milestone = int(item.get("milestone", 0) or 0)
        if milestone <= 0 or milestone in completed_milestones:
            continue
        if item.get("fail_fast_status") == "fail":
            raise RuntimeError(
                "refusing to resume early judge after a confirmed failing "
                f"checkpoint: milestone={milestone}"
            )
        completed.append(item)
        completed_milestones.add(milestone)

        if item.get("confirm_fail_triggered") and item.get("confirm_judge"):
            segment_judges = [pathlib.Path(str(item["confirm_judge"]))]
        elif item.get("delta_judge"):
            segment_judges.append(pathlib.Path(str(item["delta_judge"])))

    missing_segments = [path for path in segment_judges if not is_json(path)]
    if missing_segments:
        raise RuntimeError(
            "refusing to resume early judge because prior judge segments are "
            "missing or invalid: "
            + ", ".join(str(path) for path in missing_segments[:5])
        )

    if completed:
        print(
            "[early-judge] resumed completed milestones="
            f"{sorted(completed_milestones)} from {args.manifest}",
            flush=True,
        )
    return completed, completed_milestones, segment_judges


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def aggregate_judge_rows(
    *,
    summary_path: pathlib.Path,
    judge_path: pathlib.Path,
    rows_path: pathlib.Path,
    segment_judges: list[pathlib.Path],
    bootstrap_samples: int,
    judge_seed: int,
) -> None:
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    fairness_expected: dict[str, bool] = {}
    fairness_counters = Counter()
    judge_validation = Counter()
    failure_reasons = Counter()
    segment_paths: list[str] = []
    segment_row_paths: list[str] = []
    early_stops: list[dict] = []
    judge_model = ""
    judge_provider = ""

    for segment in segment_judges:
        if not is_json(segment):
            raise RuntimeError(f"missing incremental judge artifact: {segment}")
        segment_payload = json.loads(segment.read_text())
        segment_paths.append(str(segment))
        if not judge_model:
            judge_model = str(segment_payload.get("judge_model", ""))
        if not judge_provider:
            judge_provider = str(segment_payload.get("judge_provider", ""))
        fairness = segment_payload.get("fairness_checks", {})
        if isinstance(fairness, dict):
            for name, value in fairness.items():
                if name == "counters" and isinstance(value, dict):
                    fairness_counters.update(value)
                elif isinstance(value, bool):
                    if FAIRNESS_EXPECTED_VALUES.get(name, True) is False:
                        fairness_expected[name] = fairness_expected.get(name, False) or value
                    else:
                        fairness_expected[name] = fairness_expected.get(name, True) and value
        judge_validation.update(segment_payload.get("judge_validation", {}) or {})
        failure_reasons.update(segment_payload.get("failure_reasons", {}) or {})
        early_stop = segment_payload.get("early_stop")
        if isinstance(early_stop, dict) and early_stop:
            early_stops.append({"judge": str(segment), **early_stop})
        row_path = segment.with_name(segment.name + ".rows.jsonl")
        segment_row_paths.append(str(row_path))
        for row in load_jsonl(row_path):
            key = (int(row.get("event_index", -1)), int(row.get("repetition", 0)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    rows.sort(key=lambda row: (int(row.get("event_index", 0)), int(row.get("repetition", 0))))
    if not rows:
        raise RuntimeError("no incremental judge rows available for cumulative aggregate")

    quality: dict[str, dict] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if system in row.get("systems", {})]
        judged = len(system_rows)
        system_quality = {
            "judged": judged,
            "wins": sum(1 for row in rows if row.get("winner") == system),
        }
        for field in FIELDS:
            system_quality[f"mean_{field}"] = mean(
                [
                    float(row.get("systems", {}).get(system, {}).get(field, 0.0) or 0.0)
                    for row in system_rows
                ]
            )
        quality[system] = system_quality

    cortext_tokens = mean([float(row.get("cortext_context_tokens", 0.0) or 0.0) for row in rows])
    rag_tokens = mean([float(row.get("traditional_chat_rag_tokens", 0.0) or 0.0) for row in rows])
    token_savings = 1.0 - (cortext_tokens / rag_tokens) if rag_tokens > 0 else 0.0

    payload = {
        "schema": "julie_probe_stream_cumulative_early_judge_v1",
        "summary_path": str(summary_path),
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "incremental_judges": segment_paths,
        "incremental_rows": segment_row_paths,
        "judged": len(rows),
        "probe_count": len({int(row.get("event_index", 0)) for row in rows}),
        "quality": quality,
        "tokens": {
            "mean_cortext_context_tokens": cortext_tokens,
            "mean_traditional_chat_rag_tokens": rag_tokens,
            "mean_cortext_token_savings_vs_traditional_chat_rag": token_savings,
        },
        "confidence_intervals": confidence_intervals(
            rows,
            SYSTEMS,
            FIELDS,
            bootstrap_samples,
            judge_seed,
        ),
        "failure_reasons": dict(failure_reasons),
        "judge_validation": dict(judge_validation),
        "early_stop": early_stops[-1] if early_stops else None,
        "early_stop_segments": early_stops,
        "fairness_checks": {
            **fairness_expected,
            "counters": dict(fairness_counters),
        },
        "judgments": rows,
    }
    judge_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_jsonl(rows_path, rows)


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


def rag_phase_for_probe(probe: dict) -> str:
    rolling_tokens = int(probe.get("rolling_history_tokens", 0) or 0)
    rag_tokens = int(probe.get("traditional_chat_rag_tokens", 0) or 0)
    compaction_events = int(probe.get("normal_rag_compaction_events", 0) or 0)
    compacted_items = int(probe.get("normal_rag_compacted_history_items", 0) or 0)
    rag_additional = probe.get("rag_top_k_additional", [])
    rag_additional_count = len(rag_additional) if isinstance(rag_additional, list) else 0
    if rolling_tokens <= 0:
        return "unknown"
    if compaction_events > 0 or compacted_items > 0:
        return "post_compaction"
    if rag_tokens > rolling_tokens or rag_additional_count > 0:
        return "pre_compaction_vector_augmented"
    return "pre_compaction_raw_history"


def quality_gate_phase(args: argparse.Namespace, summary_path: pathlib.Path) -> dict:
    summary = load_json_if_valid(summary_path)
    probes = summary.get("probes", [])
    if not isinstance(probes, list):
        probes = []

    budget = int(summary.get("active_history_token_budget", 0) or 0)
    if budget <= 0:
        budget = int(args.active_history_token_budget or 0)

    phase_counts = Counter()
    max_rolling_tokens = 0
    compaction_probe_count = 0
    vector_augmented_probe_count = 0
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        phase = rag_phase_for_probe(probe)
        phase_counts[phase] += 1
        rolling_tokens = int(probe.get("rolling_history_tokens", 0) or 0)
        max_rolling_tokens = max(max_rolling_tokens, rolling_tokens)
        if phase == "post_compaction":
            compaction_probe_count += 1
        elif phase == "pre_compaction_vector_augmented":
            vector_augmented_probe_count += 1

    budget_ratio = (
        max_rolling_tokens / float(budget)
        if budget > 0 and max_rolling_tokens > 0
        else 0.0
    )
    pressure_ready = (
        compaction_probe_count > 0
        or vector_augmented_probe_count > 0
        or budget_ratio >= args.quality_gate_min_history_budget_ratio
    )
    phase_ready = True
    reason = "disabled"
    if args.quality_gate_requires_rag_pressure:
        phase_ready = pressure_ready
        reason = (
            "rag_compaction_seen"
            if compaction_probe_count > 0
            else (
                "rag_vector_augmented"
                if vector_augmented_probe_count > 0
                else (
                    "rolling_history_at_budget"
                    if budget_ratio >= args.quality_gate_min_history_budget_ratio
                    else "pre_compaction_raw_history"
                )
            )
        )

    return {
        "quality_gate_phase_ready": phase_ready,
        "quality_gate_phase_reason": reason,
        "quality_gate_requires_rag_pressure": args.quality_gate_requires_rag_pressure,
        "quality_gate_min_history_budget_ratio": args.quality_gate_min_history_budget_ratio,
        "active_history_token_budget": budget,
        "max_rolling_history_tokens": max_rolling_tokens,
        "max_rolling_history_budget_ratio": budget_ratio,
        "compaction_probe_count": compaction_probe_count,
        "vector_augmented_probe_count": vector_augmented_probe_count,
        "rag_phase_counts": dict(phase_counts),
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


def fail_fast_checks(
    args: argparse.Namespace,
    metrics: dict,
    milestone: int,
    phase: dict,
) -> list[dict]:
    checks: list[dict] = []
    phase_ready = bool(phase.get("quality_gate_phase_ready", True))
    quality_gate_active = (
        milestone >= args.quality_gate_min_milestone and phase_ready
    )
    checks_to_apply = [
        check_floor(
            "mean_cortext_context_tokens",
            float(metrics["mean_cortext_context_tokens"]),
            args.min_mean_cortext_context_tokens,
        ),
    ]
    if phase_ready:
        # The token-savings floor is only meaningful once normal RAG is under
        # real history-budget pressure: against a raw history smaller than a
        # single Cortext packet (e.g. 60 tokens at the first probe of a short
        # window), a 50% savings floor is unsatisfiable for any system.
        checks_to_apply.insert(
            0,
            check_floor(
                "cortext_token_savings_vs_traditional_chat_rag",
                float(metrics["cortext_token_savings_vs_traditional_chat_rag"]),
                args.min_cortext_token_savings_vs_rag,
            ),
        )
    else:
        checks.append(
            {
                "name": "token_savings_gate_deferred",
                "value": float(
                    metrics["cortext_token_savings_vs_traditional_chat_rag"]
                ),
                "floor": args.min_cortext_token_savings_vs_rag,
                "phase_ready": phase_ready,
                "phase_reason": phase.get("quality_gate_phase_reason", ""),
                "max_rolling_history_budget_ratio": phase.get(
                    "max_rolling_history_budget_ratio", 0.0
                ),
                "min_history_budget_ratio": phase.get(
                    "quality_gate_min_history_budget_ratio", 0.0
                ),
                "status": "pass",
            }
        )
    if quality_gate_active:
        if args.min_cortext_win_rate is not None:
            checks_to_apply.append(
                check_floor(
                    "cortext_win_rate",
                    float(metrics["cortext_win_rate"]),
                    args.min_cortext_win_rate,
                )
            )
        if args.quality_trend_window <= 1:
            checks_to_apply.append(
                check_floor(
                    "cortext_quality_delta_vs_traditional_chat_rag",
                    float(metrics["cortext_quality_delta_vs_traditional_chat_rag"]),
                    args.min_cortext_quality_delta_vs_rag,
                )
            )
        else:
            checks.append(
                {
                    "name": "cortext_quality_delta_vs_traditional_chat_rag.observed",
                    "value": float(
                        metrics["cortext_quality_delta_vs_traditional_chat_rag"]
                    ),
                    "floor": args.min_cortext_quality_delta_vs_rag,
                    "quality_trend_window": args.quality_trend_window,
                    "status": "pass",
                }
            )
    else:
        checks.append(
            {
                "name": "quality_gate_deferred",
                "value": milestone,
                "floor": args.quality_gate_min_milestone,
                "phase_ready": phase.get("quality_gate_phase_ready", True),
                "phase_reason": phase.get("quality_gate_phase_reason", ""),
                "max_rolling_history_tokens": phase.get(
                    "max_rolling_history_tokens", 0
                ),
                "active_history_token_budget": phase.get(
                    "active_history_token_budget", 0
                ),
                "max_rolling_history_budget_ratio": phase.get(
                    "max_rolling_history_budget_ratio", 0.0
                ),
                "min_history_budget_ratio": phase.get(
                    "quality_gate_min_history_budget_ratio", 0.0
                ),
                "status": "pass",
            }
        )

    for check in checks_to_apply:
        if check is not None:
            checks.append(check)

    if args.require_fairness_checks:
        fairness = metrics.get("fairness_checks", {})
        for name, expected in sorted(FAIRNESS_EXPECTED_VALUES.items()):
            value = fairness.get(name) if isinstance(fairness, dict) else None
            checks.append(
                {
                    "name": f"fairness.{name}",
                    "value": value,
                    "expected": expected,
                    "status": "pass" if value is expected else "fail",
                }
            )
    return checks


def trend_check_floor(
    *,
    name: str,
    records: list[dict],
    metric: str,
    floor: float | None,
    min_milestone: int,
    window: int,
) -> dict | None:
    if floor is None or window <= 0:
        return None
    current_milestone = int(records[-1].get("milestone", 0) or 0) if records else 0
    if current_milestone < min_milestone:
        return {
            "name": f"quality_trend.{name}.deferred",
            "value": current_milestone,
            "floor": min_milestone,
            "status": "pass",
        }
    latest_phase_ready = (
        bool(records[-1].get("quality_gate_phase_ready", True)) if records else True
    )
    if not latest_phase_ready:
        return {
            "name": f"quality_trend.{name}.deferred_phase",
            "value": records[-1].get("quality_gate_phase_reason", "")
            if records
            else "",
            "max_rolling_history_tokens": records[-1].get(
                "max_rolling_history_tokens", 0
            )
            if records
            else 0,
            "active_history_token_budget": records[-1].get(
                "active_history_token_budget", 0
            )
            if records
            else 0,
            "max_rolling_history_budget_ratio": records[-1].get(
                "max_rolling_history_budget_ratio", 0.0
            )
            if records
            else 0.0,
            "floor": records[-1].get("quality_gate_min_history_budget_ratio", 0.0)
            if records
            else 0.0,
            "status": "pass",
        }
    records = [record for record in records if record.get("quality_gate_phase_ready", True)]
    if len(records) < window:
        return {
            "name": f"quality_trend.{name}.deferred_window",
            "value": len(records),
            "floor": window,
            "status": "pass",
        }
    window_records = records[-window:]
    values = [
        float(record.get("metrics", {}).get(metric, 0.0) or 0.0)
        for record in window_records
    ]
    return {
        "name": f"quality_trend.{name}",
        "values": values,
        "milestones": [record.get("milestone") for record in window_records],
        "floor": floor,
        "window": window,
        "status": "fail" if all(value < floor for value in values) else "pass",
    }


def trend_fail_fast_checks(args: argparse.Namespace, records: list[dict]) -> list[dict]:
    checks: list[dict] = []
    if not records or args.quality_trend_window <= 0:
        return checks
    quality_check = trend_check_floor(
        name="cortext_quality_delta_vs_traditional_chat_rag",
        records=records,
        metric="cortext_quality_delta_vs_traditional_chat_rag",
        floor=args.min_cortext_quality_delta_vs_rag,
        min_milestone=args.quality_trend_gate_min_milestone,
        window=args.quality_trend_window,
    )
    if quality_check is not None:
        checks.append(quality_check)
    win_rate_check = trend_check_floor(
        name="cortext_win_rate",
        records=records,
        metric="cortext_win_rate",
        floor=args.min_cortext_win_rate,
        min_milestone=args.quality_trend_gate_min_milestone,
        window=args.quality_trend_window,
    )
    if win_rate_check is not None:
        checks.append(win_rate_check)
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
        "--timeline-skip-messages",
        str(args.timeline_skip_messages),
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


def run_judge(
    args: argparse.Namespace,
    milestone: int,
    start_index: int,
    summary_path: pathlib.Path,
    judge_path: pathlib.Path,
    judge_repetitions: int | None = None,
    early_stop_quality_delta_floor: float | None = None,
    early_stop_prior_quality_delta_sum: float = 0.0,
    early_stop_prior_judgment_count: int = 0,
) -> dict:
    rows_path = judge_path.with_name(judge_path.name + ".rows.jsonl")
    for path in [judge_path, rows_path]:
        if path.exists():
            path.unlink()
    repetitions = (
        args.judge_repetitions
        if judge_repetitions is None
        else judge_repetitions
    )
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
        str(repetitions),
        "--judge-seed",
        str(args.judge_seed),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--judge-limit",
        str(milestone),
        "--judge-start-index",
        str(start_index),
        "--judge-timeout-s",
        str(args.judge_timeout_s),
        "--judge-context-window-tokens",
        str(args.judge_context_window_tokens),
        "--judge-max-output-tokens",
        str(args.judge_max_output_tokens),
        "--ollama-keep-alive",
        args.ollama_keep_alive,
        "--context-limit",
        str(args.context_limit),
        "--checkpoint-rows",
        str(rows_path),
        "--max-media-per-system",
        str(args.max_media_per_system),
    ]
    if args.ollama_base_url:
        cmd.extend(["--ollama-base-url", args.ollama_base_url])
    if args.blind_packets:
        cmd.append("--blind-packets")
    if early_stop_quality_delta_floor is not None:
        cmd.extend(
            [
                "--early-stop-min-quality-delta-vs-rag",
                str(early_stop_quality_delta_floor),
                "--early-stop-prior-quality-delta-sum",
                str(early_stop_prior_quality_delta_sum),
                "--early-stop-prior-judgment-count",
                str(early_stop_prior_judgment_count),
            ]
        )
    pause_started = None
    pause_ms = 0.0
    paused = signal_process(
        args.pause_pid_during_judge, signal.SIGSTOP, "benchmark"
    )
    if paused:
        pause_started = time.monotonic()
    try:
        run_checked(cmd)
    finally:
        if paused:
            signal_process(
                args.pause_pid_during_judge, signal.SIGCONT, "benchmark"
            )
            if pause_started is not None:
                pause_ms = max(0.0, (time.monotonic() - pause_started) * 1000.0)
                args.benchmark_pause_ms_total += pause_ms
                args.benchmark_pause_count += 1
    return {
        "benchmark_pause_ms": round(pause_ms, 3),
        "benchmark_pause_count": 1 if paused else 0,
        "benchmark_pause_pid": args.pause_pid_during_judge if paused else 0,
        "benchmark_pause_ms_total": round(args.benchmark_pause_ms_total, 3),
        "benchmark_pause_count_total": args.benchmark_pause_count,
    }


def write_loss_audit(
    judge_path: pathlib.Path,
    summary_path: pathlib.Path,
    audit_path: pathlib.Path,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/audit_julie_judge_losses.py"),
        "--judge",
        str(judge_path),
        "--summary",
        str(summary_path),
        "--out",
        str(audit_path),
    ]
    run_checked(cmd)


def write_manifest(args: argparse.Namespace, completed: list[dict]) -> None:
    latest = completed[-1] if completed else None
    manifest = {
        "schema": "julie_probe_stream_early_judge_manifest_v1",
        "created_at_utc": utc_now(),
        "release_gate_use": "fail_fast_screen_only_not_release_claim",
        "privacy": "private local artifact; may point to private summaries and judge rows",
        "probe_stream": str(args.probe_stream),
        "probe_stream_sha256": file_sha256(args.probe_stream)
        if args.probe_stream.exists()
        else "",
        "input_dir": str(args.input_dir),
        "db": str(args.db),
        "timeline_skip_messages": args.timeline_skip_messages,
        "timeline_max_messages": args.timeline_max_messages,
        "timeline_media_limit": args.timeline_media_limit,
        "fixed_milestones": parse_milestones(args.milestones),
        "periodic_stride": args.periodic_stride,
        "completion_summary": str(args.completion_summary)
        if args.completion_summary
        else "",
        "judge_provider": args.judge_provider,
        "judge_model": args.model,
        "ollama_keep_alive": args.ollama_keep_alive,
        "judge_repetitions": args.judge_repetitions,
        "confirm_fail_repetitions": args.confirm_fail_repetitions,
        "judge_packet_item_limit": args.context_limit,
        "benchmark_pause_pid": args.pause_pid_during_judge,
        "benchmark_pause_ms_total": round(args.benchmark_pause_ms_total, 3),
        "benchmark_pause_count": args.benchmark_pause_count,
        "quality_gate_min_milestone": args.quality_gate_min_milestone,
        "quality_trend_gate_min_milestone": args.quality_trend_gate_min_milestone,
        "quality_trend_window": args.quality_trend_window,
        "quality_gate_requires_rag_pressure": args.quality_gate_requires_rag_pressure,
        "quality_gate_min_history_budget_ratio": (
            args.quality_gate_min_history_budget_ratio
        ),
        "blind_packets": args.blind_packets,
        "latest": latest,
        "completed": completed,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if latest is not None:
        latest_path = args.manifest.with_name("early_judge_latest.json")
        latest_payload = {
            "schema": "julie_probe_stream_early_judge_latest_v1",
            "created_at_utc": utc_now(),
            "release_gate_use": "fail_fast_screen_only_not_release_claim",
            "privacy": "private local artifact; contains aggregate checkpoint metrics only",
            "manifest": str(args.manifest),
            "probe_stream": str(args.probe_stream),
            "probe_stream_rows": probe_rows_available(args.probe_stream),
            "fixed_milestones": parse_milestones(args.milestones),
            "periodic_stride": args.periodic_stride,
            "judge_provider": args.judge_provider,
            "judge_model": args.model,
            "ollama_keep_alive": args.ollama_keep_alive,
            "judge_repetitions": args.judge_repetitions,
            "confirm_fail_repetitions": args.confirm_fail_repetitions,
            "benchmark_pause_pid": args.pause_pid_during_judge,
            "benchmark_pause_ms_total": round(args.benchmark_pause_ms_total, 3),
            "benchmark_pause_count": args.benchmark_pause_count,
            "quality_gate_min_milestone": args.quality_gate_min_milestone,
            "quality_trend_gate_min_milestone": args.quality_trend_gate_min_milestone,
            "quality_trend_window": args.quality_trend_window,
            "quality_gate_requires_rag_pressure": (
                args.quality_gate_requires_rag_pressure
            ),
            "quality_gate_min_history_budget_ratio": (
                args.quality_gate_min_history_budget_ratio
            ),
            "latest": latest,
        }
        latest_path.write_text(json.dumps(latest_payload, indent=2) + "\n")


def run_milestone(
    args: argparse.Namespace,
    milestone: int,
    start_index: int,
    segment_judges: list[pathlib.Path],
) -> dict:
    summary_path = args.out_dir / f"early_probe_{milestone:03d}_summary.json"
    judge_path = args.out_dir / f"early_probe_{milestone:03d}_judge.json"
    judge_rows_path = judge_path.with_name(judge_path.name + ".rows.jsonl")
    delta_judge_path = args.out_dir / f"early_probe_{milestone:03d}_delta_judge.json"
    loss_audit_path = args.out_dir / f"early_probe_{milestone:03d}_loss_audit.json"
    materialize_partial(args, milestone, summary_path)
    pause_stats = run_judge(
        args, milestone, start_index, summary_path, delta_judge_path
    )
    next_segment_judges = [*segment_judges, delta_judge_path]
    aggregate_judge_rows(
        summary_path=summary_path,
        judge_path=judge_path,
        rows_path=judge_rows_path,
        segment_judges=next_segment_judges,
        bootstrap_samples=args.bootstrap_samples,
        judge_seed=args.judge_seed,
    )
    write_loss_audit(judge_path, summary_path, loss_audit_path)
    metrics = judge_metrics(judge_path)
    phase = quality_gate_phase(args, summary_path)
    checks = fail_fast_checks(args, metrics, milestone, phase)
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    quality_gate_active = (
        milestone >= args.quality_gate_min_milestone
        and bool(phase.get("quality_gate_phase_ready", True))
    )
    return {
        "milestone": milestone,
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "judge": str(judge_path),
        "judge_sha256": file_sha256(judge_path),
        "loss_audit": str(loss_audit_path),
        "loss_audit_sha256": file_sha256(loss_audit_path),
        "delta_judge": str(delta_judge_path),
        "delta_judge_sha256": file_sha256(delta_judge_path),
        "judge_start_index": start_index,
        "judge_limit": milestone,
        **pause_stats,
        "metrics": metrics,
        **phase,
        "quality_gate_active": quality_gate_active,
        "fail_fast_checks": checks,
        "fail_fast_status": status,
        "completed_at_utc": utc_now(),
    }


def apply_trend_checks(
    args: argparse.Namespace,
    record: dict,
    previous_records: list[dict],
) -> None:
    trend_checks = trend_fail_fast_checks(args, [*previous_records, record])
    if trend_checks:
        milestone = int(record.get("milestone", 0) or 0)
        record["quality_trend_gate_active"] = (
            milestone >= args.quality_trend_gate_min_milestone
            and args.quality_trend_window > 0
            and bool(record.get("quality_gate_phase_ready", True))
        )
        record["fail_fast_checks"].extend(trend_checks)
        record["fail_fast_status"] = (
            "pass"
            if all(
                check["status"] == "pass"
                for check in record["fail_fast_checks"]
            )
            else "fail"
        )


def confirm_failed_record(
    args: argparse.Namespace,
    record: dict,
    previous_records: list[dict],
    segment_judges: list[pathlib.Path],
) -> tuple[dict, pathlib.Path | None]:
    if record.get("fail_fast_status") != "fail":
        return record, None
    if args.confirm_fail_repetitions <= args.judge_repetitions:
        return record, None

    milestone = int(record.get("milestone", 0) or 0)
    start_index = int(record.get("judge_start_index", 0) or 0)
    summary_path = pathlib.Path(str(record["summary"]))
    confirm_judge_path = args.out_dir / f"early_probe_{milestone:03d}_confirm_judge.json"
    confirm_rows_path = confirm_judge_path.with_name(
        confirm_judge_path.name + ".rows.jsonl"
    )
    confirm_delta_judge_path = (
        args.out_dir / f"early_probe_{milestone:03d}_confirm_delta_judge.json"
    )
    confirm_loss_audit_path = (
        args.out_dir / f"early_probe_{milestone:03d}_confirm_loss_audit.json"
    )
    print(
        "[early-judge] fail-fast threshold failed; confirming with "
        f"{args.confirm_fail_repetitions} judge repetitions",
        flush=True,
    )
    prior_delta_sum, prior_judgment_count = prior_quality_delta_stats(segment_judges)
    confirm_pause_stats = run_judge(
        args,
        milestone,
        start_index,
        summary_path,
        confirm_delta_judge_path,
        judge_repetitions=args.confirm_fail_repetitions,
        early_stop_quality_delta_floor=args.min_cortext_quality_delta_vs_rag,
        early_stop_prior_quality_delta_sum=prior_delta_sum,
        early_stop_prior_judgment_count=prior_judgment_count,
    )
    aggregate_judge_rows(
        summary_path=summary_path,
        judge_path=confirm_judge_path,
        rows_path=confirm_rows_path,
        segment_judges=[*segment_judges, confirm_delta_judge_path],
        bootstrap_samples=args.bootstrap_samples,
        judge_seed=args.judge_seed,
    )
    write_loss_audit(confirm_judge_path, summary_path, confirm_loss_audit_path)

    phase = quality_gate_phase(args, summary_path)
    metrics = judge_metrics(confirm_judge_path)
    confirm_payload = load_json_if_valid(confirm_judge_path)
    checks = fail_fast_checks(args, metrics, milestone, phase)
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    quality_gate_active = (
        milestone >= args.quality_gate_min_milestone
        and bool(phase.get("quality_gate_phase_ready", True))
    )
    preliminary = {
        "judge": record.get("judge"),
        "judge_sha256": record.get("judge_sha256"),
        "loss_audit": record.get("loss_audit"),
        "loss_audit_sha256": record.get("loss_audit_sha256"),
        "metrics": record.get("metrics"),
        "fail_fast_checks": record.get("fail_fast_checks"),
        "fail_fast_status": record.get("fail_fast_status"),
    }
    record.update(
        {
            "confirm_fail_triggered": True,
            "confirm_fail_repetitions": args.confirm_fail_repetitions,
            "pre_confirm": preliminary,
            "judge": str(confirm_judge_path),
            "judge_sha256": file_sha256(confirm_judge_path),
            "loss_audit": str(confirm_loss_audit_path),
            "loss_audit_sha256": file_sha256(confirm_loss_audit_path),
            "confirm_judge": str(confirm_judge_path),
            "confirm_judge_sha256": file_sha256(confirm_judge_path),
            "confirm_delta_judge": str(confirm_delta_judge_path),
            "confirm_delta_judge_sha256": file_sha256(confirm_delta_judge_path),
            "confirm_loss_audit": str(confirm_loss_audit_path),
            "confirm_loss_audit_sha256": file_sha256(confirm_loss_audit_path),
            "confirm_benchmark_pause_ms": confirm_pause_stats.get(
                "benchmark_pause_ms"
            ),
            "confirm_benchmark_pause_count": confirm_pause_stats.get(
                "benchmark_pause_count"
            ),
            "benchmark_pause_ms_total": confirm_pause_stats.get(
                "benchmark_pause_ms_total"
            ),
            "benchmark_pause_count_total": confirm_pause_stats.get(
                "benchmark_pause_count_total"
            ),
            "early_stop": confirm_payload.get("early_stop"),
            "metrics": metrics,
            **phase,
            "quality_gate_active": quality_gate_active,
            "fail_fast_checks": checks,
            "fail_fast_status": status,
            "confirmed_at_utc": utc_now(),
        }
    )
    apply_trend_checks(args, record, previous_records)
    if record.get("fail_fast_status") == "pass":
        print(
            "[early-judge] confirmation passed; resuming benchmark",
            flush=True,
        )
        return record, confirm_judge_path
    print(
        "[early-judge] confirmation failed; stopping benchmark",
        flush=True,
    )
    return record, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-stream", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--db", type=pathlib.Path, required=True)
    parser.add_argument("--timeline-skip-messages", type=int, default=0)
    parser.add_argument("--timeline-max-messages", type=int, required=True)
    parser.add_argument("--timeline-media-limit", type=int, required=True)
    parser.add_argument(
        "--milestones",
        default=(
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,"
            "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"
        ),
    )
    parser.add_argument(
        "--periodic-stride",
        type=int,
        default=0,
        help=(
            "After the largest fixed milestone, keep judging every N probe rows. "
            "Use 0 to only run the fixed milestones."
        ),
    )
    parser.add_argument(
        "--completion-summary",
        type=pathlib.Path,
        help=(
            "Optional final summary JSON. When periodic judging is enabled, "
            "the watcher exits once this summary is valid JSON and all due "
            "probe rows through the current stream have been judged."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
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
    parser.add_argument(
        "--ollama-keep-alive",
        default="0s",
        help=(
            "Ollama model residency after each streamed judge call. The default "
            "unloads the local judge model before resuming Cortext replay."
        ),
    )
    parser.add_argument("--judge-repetitions", type=int, default=3)
    parser.add_argument(
        "--confirm-fail-repetitions",
        type=int,
        default=3,
        help=(
            "If a fail-fast gate fails with fewer repetitions than this, rerun "
            "the same checkpoint from scratch with this many repetitions before "
            "terminating the benchmark. Set <= --judge-repetitions to disable."
        ),
    )
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--judge-timeout-s", type=int, default=180)
    parser.add_argument("--judge-context-window-tokens", type=int, default=32768)
    parser.add_argument("--judge-max-output-tokens", type=int, default=1300)
    parser.add_argument(
        "--context-limit",
        type=int,
        default=-1,
        help="Maximum items per packet passed to the local judge.",
    )
    parser.add_argument("--max-media-per-system", type=int, default=-1)
    parser.add_argument(
        "--pause-pid-during-judge",
        type=int,
        default=0,
        help=(
            "Optional benchmark PID to SIGSTOP while each local judge call runs "
            "and SIGCONT afterward. This prevents early checkpoints from "
            "racing replay work on one local machine."
        ),
    )
    parser.add_argument(
        "--quality-gate-min-milestone",
        type=int,
        default=8,
        help=(
            "Do not fail fast on quality/win-rate floors before this probe "
            "milestone; early milestones still enforce token and fairness gates."
        ),
    )
    parser.add_argument(
        "--quality-trend-gate-min-milestone",
        type=int,
        default=4,
        help=(
            "Before the hard quality gate, fail fast once this probe milestone "
            "is reached if the rolling quality trend is consistently below "
            "the configured quality floors."
        ),
    )
    parser.add_argument(
        "--quality-trend-window",
        type=int,
        default=2,
        help=(
            "Number of consecutive early checkpoints that must miss a quality "
            "floor before the trend gate fails. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--quality-gate-requires-rag-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, quality/win-rate gates are deferred until the "
            "traditional chat+RAG baseline has compacted prior chat, added "
            "vector RAG outside raw rolling history, or reached the full "
            "rolling-history budget. "
            "Fairness, privacy, packet, and token checks still run at every "
            "milestone."
        ),
    )
    parser.add_argument(
        "--quality-gate-min-history-budget-ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum max rolling-history-token / active-history-token-budget "
            "ratio required before pre-compaction quality gates apply."
        ),
    )
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
    if args.periodic_stride < 0:
        raise RuntimeError("--periodic-stride must be >= 0")
    if args.timeline_skip_messages < 0:
        raise RuntimeError("--timeline-skip-messages must be >= 0")
    if args.periodic_stride > 0 and args.completion_summary is None:
        raise RuntimeError("--completion-summary is required when --periodic-stride > 0")
    if args.judge_repetitions <= 0:
        raise RuntimeError("--judge-repetitions must be positive")
    if args.confirm_fail_repetitions < 0:
        raise RuntimeError("--confirm-fail-repetitions must be >= 0")
    if args.judge_timeout_s < 1:
        raise RuntimeError("--judge-timeout-s must be >= 1")
    if args.judge_context_window_tokens < 1:
        raise RuntimeError("--judge-context-window-tokens must be >= 1")
    if args.judge_max_output_tokens < 1:
        raise RuntimeError("--judge-max-output-tokens must be >= 1")
    if args.context_limit == 0 or args.context_limit < -1:
        raise RuntimeError("--context-limit must be -1 or a positive item limit")
    if args.pause_pid_during_judge < 0:
        raise RuntimeError("--pause-pid-during-judge must be >= 0")
    if args.quality_gate_min_milestone < 1:
        raise RuntimeError("--quality-gate-min-milestone must be >= 1")
    if args.quality_trend_gate_min_milestone < 1:
        raise RuntimeError("--quality-trend-gate-min-milestone must be >= 1")
    if args.quality_trend_window < 0:
        raise RuntimeError("--quality-trend-window must be >= 0")
    if not 0.0 <= args.quality_gate_min_history_budget_ratio <= 1.0:
        raise RuntimeError("--quality-gate-min-history-budget-ratio must be in [0, 1]")

    args.probe_stream = args.probe_stream.resolve()
    args.out_dir = args.out_dir.resolve()
    args.input_dir = args.input_dir.resolve()
    args.db = args.db.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()
    if args.completion_summary is not None:
        args.completion_summary = args.completion_summary.resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest is None:
        args.manifest = args.out_dir / "early_judge_manifest.json"
    args.benchmark_pause_ms_total = 0.0
    args.benchmark_pause_count = 0

    fixed_milestones = parse_milestones(args.milestones)
    completed, completed_milestones, segment_judges = load_resume_state(args)
    while True:
        available = probe_rows_available(args.probe_stream)
        completion_summary_ready = bool(
            args.completion_summary is not None and is_json(args.completion_summary)
        )
        due_milestones = set(fixed_milestones)
        if args.periodic_stride > 0:
            milestone = max(fixed_milestones) + args.periodic_stride
            while milestone <= available:
                due_milestones.add(milestone)
                milestone += args.periodic_stride
            if completion_summary_ready and available > 0:
                due_milestones.add(available)
        milestones = sorted(due_milestones)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"probe_rows={available} completed={sorted(completed_milestones)}",
            flush=True,
        )
        for milestone in milestones:
            if milestone in completed_milestones or available < milestone:
                continue
            last_completed = max(completed_milestones) if completed_milestones else 0
            if milestone <= last_completed:
                continue
            start_index = last_completed
            record = run_milestone(args, milestone, start_index, segment_judges)
            apply_trend_checks(args, record, completed)
            confirmation_segment = None
            if record["fail_fast_status"] == "fail":
                record, confirmation_segment = confirm_failed_record(
                    args, record, completed, segment_judges
                )
            completed.append(record)
            completed_milestones.add(milestone)
            if confirmation_segment is not None:
                segment_judges = [confirmation_segment]
            else:
                segment_judges.append(pathlib.Path(record["delta_judge"]))
            write_manifest(args, completed)
            if record["fail_fast_status"] == "fail":
                print(
                    "[early-judge] fail-fast threshold failed "
                    + json.dumps(record["fail_fast_checks"], sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
                return 2
        if (
            args.periodic_stride == 0
            and completed_milestones == set(fixed_milestones)
            and args.stop_after_last
        ):
            break
        if (
            args.periodic_stride > 0
            and completion_summary_ready
            and set(milestones).issubset(completed_milestones)
        ):
            break
        time.sleep(args.poll_seconds)
    write_manifest(args, completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
