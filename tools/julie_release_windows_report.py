#!/usr/bin/env python3
"""Build a public-safe aggregate report for Julie release windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import subprocess
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_SPEC = REPO_ROOT / "tools/julie_release_protocol_spec.json"
SYSTEMS = ["cortext_native", "traditional_chat_rag", "full_history_upper_bound"]
FIELDS = ["relevance", "sufficiency", "noise"]
QUALITY_WEIGHTS = {"relevance": 1.0, "sufficiency": 1.0, "noise": -0.25}
QUALITY_DEFINITION = "relevance + sufficiency - 0.25*noise"
MIN_RELEASE_PROBES = 30
MIN_JUDGE_REPETITIONS = 3
MIN_HUMAN_PROBES = 30
MIN_TOKEN_SAVINGS_CI95_LOWER = 0.5
REQUIRED_ABLATIONS = {
    "no_daily_consolidation",
    "no_graph_expansion",
    "no_media_source_blobs",
    "no_stm_ltm_graph_label_handoff",
    "no_temporal_retrieval",
    "no_fact_boosts",
    "no_temporal_fact_boosts",
}


def ablation_effect_evidence(row: dict[str, Any]) -> dict[str, Any]:
    effect = row.get("effect_vs_full", {})
    if not isinstance(effect, dict):
        effect = {}
    quality_delta = effect.get("quality_delta_full_minus_ablation", {})
    if not isinstance(quality_delta, dict):
        quality_delta = {}
    ci95 = quality_delta.get("ci95", [0.0, 0.0])
    lower = float(ci95[0]) if isinstance(ci95, list) and ci95 else 0.0
    mean_delta = float_or_default(quality_delta.get("mean"), 0.0)
    shared = int_or_default(effect.get("shared_quality_probe_count"), 0)
    return {
        "name": row.get("name"),
        "shared_quality_probe_count": shared,
        "quality_delta_mean": mean_delta,
        "quality_delta_ci95": ci95,
        "quality_delta_ci95_lower": lower,
        "has_effect_ci": bool(isinstance(ci95, list) and len(ci95) == 2 and shared > 0),
        "ci_supports_full": bool(
            isinstance(ci95, list) and len(ci95) == 2 and shared > 0 and lower > 0.0
        ),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_json_if_valid(path: pathlib.Path) -> dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_protocol_spec(path: pathlib.Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != "cortext_julie_release_protocol_spec_v1":
        raise RuntimeError(f"unexpected protocol spec schema: {payload.get('schema')}")
    return payload


def normalize_window(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(window.get("name", "")),
        "skip_messages": int_or_default(window.get("skip_messages"), -1),
        "max_messages": int_or_default(window.get("max_messages"), -1),
        "media_limit": int_or_default(window.get("media_limit"), -1),
        "probe_stride": int_or_default(window.get("probe_stride"), -1),
        "warmup_events": int_or_default(window.get("warmup_events"), -1),
        "min_probe_rows_after_benchmark": int_or_default(
            window.get("min_probe_rows_after_benchmark"), -1
        ),
        "required_media_modalities": sorted(
            str(item) for item in window.get("required_media_modalities", [])
        ),
        "purpose": str(window.get("purpose", "")),
    }


def normalize_windows(windows: list[Any]) -> list[dict[str, Any]]:
    return [normalize_window(window) for window in windows if isinstance(window, dict)]


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_ref(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_provenance() -> dict[str, Any]:
    status = git_output(["status", "--porcelain=v1"])
    return {
        "commit": git_output(["rev-parse", "HEAD"]) or "unknown",
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_path_count": len([line for line in status.splitlines() if line.strip()]),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    out = {"n": len(values), "mean": mean(values), "ci95": [0.0, 0.0]}
    if not values:
        return out
    if len(values) == 1 or samples <= 0:
        out["ci95"] = [out["mean"], out["mean"]]
        return out
    rng = random.Random(seed)
    boot = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    out["ci95"] = [percentile(boot, 0.025), percentile(boot, 0.975)]
    return out


def system_quality(scores: dict[str, Any]) -> float:
    total = 0.0
    for field, weight in QUALITY_WEIGHTS.items():
        try:
            total += float(scores.get(field, 0.0) or 0.0) * weight
        except Exception:
            pass
    return total


def check(name: str, ok: bool, detail: str = "", severity: str = "required") -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "severity": severity,
        "detail": detail,
    }


def pending(name: str, detail: str, severity: str = "required") -> dict[str, Any]:
    return {"name": name, "status": "pending", "severity": severity, "detail": detail}


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def number_close(value: Any, expected: float, eps: float = 1e-9) -> bool:
    try:
        return abs(float(value) - expected) <= eps
    except Exception:
        return False


def float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def numeric_probe_values(summary: dict[str, Any], key: str) -> list[float]:
    values: list[float] = []
    probes = summary.get("probes", [])
    if not isinstance(probes, list):
        return values
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        try:
            values.append(float(probe.get(key, 0.0) or 0.0))
        except Exception:
            pass
    return values


def path_size_bytes(path: pathlib.Path) -> int:
    total = 0
    for suffix in ["", "-wal", "-shm"]:
        candidate = pathlib.Path(str(path) + suffix)
        if candidate.exists() and candidate.is_file():
            total += candidate.stat().st_size
    return total


def cost_summary(summary: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    wall_ms = float_or_default(summary.get("wall_ms"))
    consolidation_ms = float_or_default(summary.get("consolidation_ms_total"))
    wall_ms_excluding = float_or_default(
        summary.get("wall_ms_excluding_consolidation"), wall_ms - consolidation_ms
    )
    processed = int_or_default(summary.get("processed_text_messages"), 0) + int_or_default(
        summary.get("media_processed"), 0
    )

    consolidation_events = summary.get("consolidation_events")
    consolidation_ms_by_day = None
    if isinstance(consolidation_events, list):
        by_day: dict[str, float] = {}
        for event in consolidation_events:
            if not isinstance(event, dict):
                continue
            day = str(event.get("local_day_bucket", "unknown"))
            elapsed = float_or_default(event.get("elapsed_ms"))
            by_day[day] = by_day.get(day, 0.0) + elapsed
        consolidation_ms_by_day = by_day

    cortext_probe_latencies = numeric_probe_values(summary, "cortext_latency_ms")
    rag_total_latencies = numeric_probe_values(summary, "normal_rag_total_latency_ms")
    rag_retrieval_latencies = numeric_probe_values(
        summary, "normal_rag_retrieval_latency_ms"
    )
    cortext_tokens = numeric_probe_values(summary, "cortext_context_tokens")
    rag_tokens = numeric_probe_values(summary, "normal_rag_context_tokens")
    if not rag_tokens:
        rag_tokens = numeric_probe_values(summary, "normal_rag_active_history_tokens")
    full_history_tokens = numeric_probe_values(summary, "full_history_tokens")
    db_path = pathlib.Path(str(summary.get("db_path", "")))
    db_bytes = path_size_bytes(db_path) if str(db_path) else 0

    judge_latency = judge.get("latency", {}) if isinstance(judge, dict) else {}
    if not isinstance(judge_latency, dict):
        judge_latency = {}

    return {
        "processed_events_estimate": processed,
        "wall_ms": wall_ms,
        "wall_ms_excluding_consolidation": wall_ms_excluding,
        "consolidation_ms_total": consolidation_ms,
        "consolidation_runs": summary.get("consolidation_runs"),
        "consolidation_event_count": (
            len(consolidation_events) if isinstance(consolidation_events, list) else None
        ),
        "consolidation_ms_by_day": consolidation_ms_by_day,
        "mean_consolidation_ms_per_run": (
            consolidation_ms / float(summary.get("consolidation_runs") or 1)
            if consolidation_ms > 0
            else 0.0
        ),
        "mean_ingest_total_ms": summary.get("mean_total_ms"),
        "mean_encode_ms": summary.get("mean_encode_ms"),
        "mean_process_ms": summary.get("mean_process_ms"),
        "mean_hydrate_ms": summary.get("mean_hydrate_ms"),
        "mean_cortext_probe_latency_ms": (
            mean(cortext_probe_latencies)
            if cortext_probe_latencies
            else judge_latency.get("mean_cortext_probe_latency_ms")
        ),
        "p50_cortext_probe_latency_ms": percentile(cortext_probe_latencies, 0.50),
        "p95_cortext_probe_latency_ms": percentile(cortext_probe_latencies, 0.95),
        "p50_rag_total_latency_ms": percentile(rag_total_latencies, 0.50),
        "p95_rag_total_latency_ms": percentile(rag_total_latencies, 0.95),
        "p50_rag_retrieval_latency_ms": percentile(rag_retrieval_latencies, 0.50),
        "p95_rag_retrieval_latency_ms": percentile(rag_retrieval_latencies, 0.95),
        "mean_cortext_prompt_tokens": mean(cortext_tokens) if cortext_tokens else None,
        "mean_rag_prompt_tokens": mean(rag_tokens) if rag_tokens else None,
        "mean_full_history_prompt_tokens": (
            mean(full_history_tokens) if full_history_tokens else None
        ),
        "db_disk_bytes": db_bytes if db_bytes > 0 else None,
        "db_disk_bytes_per_event": db_bytes / processed
        if db_bytes > 0 and processed > 0
        else None,
        "peak_rss_mb": summary.get("peak_rss_mb"),
        "events_per_second_excluding_consolidation": processed
        / (wall_ms_excluding / 1000.0)
        if wall_ms_excluding > 0
        else 0.0,
        "events_per_second_including_consolidation": processed / (wall_ms / 1000.0)
        if wall_ms > 0
        else 0.0,
    }


def media_attachment_count(judge: dict[str, Any], system: str) -> int:
    media = judge.get("media_attachments", {})
    if not isinstance(media, dict):
        return 0
    body = media.get(system, {})
    if not isinstance(body, dict):
        return 0
    total = 0
    for key, value in body.items():
        if key == "total" and isinstance(value, (int, float)):
            total += int(value)
        elif isinstance(value, (int, float)):
            total += int(value)
    return total


def packet_randomization_summary(judge: dict[str, Any]) -> dict[str, Any]:
    labels_by_system: dict[str, set[str]] = {system: set() for system in SYSTEMS}
    unique_mappings = set()
    mapped_rows = 0
    missing = 0
    for row in judge.get("judgments", []) or []:
        if not isinstance(row, dict):
            continue
        blinding = row.get("packet_blinding", {})
        real_to_label = blinding.get("real_to_label", {}) if isinstance(blinding, dict) else {}
        if not isinstance(real_to_label, dict):
            missing += 1
            continue
        mapping: list[tuple[str, str]] = []
        for system in SYSTEMS:
            label = real_to_label.get(system)
            if not isinstance(label, str) or not label:
                continue
            labels_by_system[system].add(label)
            mapping.append((system, label))
        if len(mapping) == len(SYSTEMS):
            mapped_rows += 1
            unique_mappings.add(tuple(mapping))
        else:
            missing += 1
    return {
        "judgment_rows": len(judge.get("judgments", []) or []),
        "mapped_rows": mapped_rows,
        "rows_missing_mapping": missing,
        "unique_mapping_count": len(unique_mappings),
        "labels_seen_by_system": {
            system: sorted(labels) for system, labels in labels_by_system.items()
        },
        "min_labels_per_system": min((len(labels) for labels in labels_by_system.values()), default=0),
    }


def window_paths(base_dir: pathlib.Path, window: dict[str, Any]) -> dict[str, pathlib.Path]:
    out_dir = base_dir / str(window["name"])
    return {
        "out_dir": out_dir,
        "preflight": out_dir / "preflight_report.json",
        "summary": out_dir / "summary.json",
        "summary_gate": out_dir / "summary_gate.json",
        "judge": out_dir / "judge_gemma4_12b_local.json",
        "initial_report": out_dir / "release_protocol_report_initial.json",
        "final_report": out_dir / "release_protocol_report_final.json",
        "ablation_plan": out_dir / "ablation_plan.json",
    }


def collect_window(
    base_dir: pathlib.Path,
    window: dict[str, Any],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = window_paths(base_dir, window)
    preflight = load_json_if_valid(paths["preflight"])
    summary = load_json_if_valid(paths["summary"])
    summary_gate = load_json_if_valid(paths["summary_gate"])
    judge = load_json_if_valid(paths["judge"])
    initial_report = load_json_if_valid(paths["initial_report"])
    final_report = load_json_if_valid(paths["final_report"])
    ablation_plan = load_json_if_valid(paths["ablation_plan"])
    final_public_summary = (
        final_report.get("public_aggregate_summary", {})
        if isinstance(final_report.get("public_aggregate_summary"), dict)
        else {}
    )
    checks: list[dict[str, Any]] = []

    name = str(window["name"])
    checks.append(
        check(
            f"{name}_preflight_passed",
            preflight.get("status") == "pass",
            f"status={preflight.get('status')}",
        )
        if preflight
        else pending(f"{name}_preflight_present", f"missing {paths['preflight']}")
    )

    required_modalities = window.get("required_media_modalities", [])
    selected_counts = ((preflight.get("input") or {}).get("selected_media_kind_counts") or {})
    for modality in required_modalities:
        checks.append(
            check(
                f"{name}_required_{modality}_covered",
                int_or_default(selected_counts.get(modality), 0) > 0,
                f"selected_media_kind_counts={selected_counts}",
            )
            if preflight
            else pending(
                f"{name}_required_{modality}_covered",
                "preflight artifact missing",
            )
        )

    if not summary:
        checks.append(pending(f"{name}_summary_present", f"missing {paths['summary']}"))
    else:
        knobs = summary.get("knobs", {})
        if not isinstance(knobs, dict):
            knobs = {}
        checks.extend(
            [
                check(
                    f"{name}_summary_gate_passed",
                    summary_gate.get("overall_status") == "pass",
                    f"overall_status={summary_gate.get('overall_status')}",
                )
                if summary_gate
                else pending(f"{name}_summary_gate_present", f"missing {paths['summary_gate']}"),
                check(
                    f"{name}_default_knobs",
                    all(number_close(knobs.get(key), 0.5) for key in ["focus", "sensitivity", "stability"]),
                    f"knobs={knobs}",
                ),
                check(
                    f"{name}_daily_deep_consolidation",
                    bool(summary.get("daily_consolidation")) and bool(summary.get("deep_consolidation")),
                    f"daily={summary.get('daily_consolidation')} deep={summary.get('deep_consolidation')}",
                ),
                check(
                    f"{name}_text_rag_baseline",
                    summary.get("normal_rag_retrieval") == "raw_chat_vector"
                    and summary.get("normal_rag_baseline_modality") == "text_only",
                    (
                        f"normal_rag_retrieval={summary.get('normal_rag_retrieval')} "
                        f"normal_rag_baseline_modality={summary.get('normal_rag_baseline_modality')}"
                    ),
                ),
            ]
        )

    if not judge:
        checks.append(pending(f"{name}_judge_present", f"missing {paths['judge']}"))
    else:
        protocol = judge.get("protocol", {})
        fairness = judge.get("fairness_checks", {})
        randomization = packet_randomization_summary(judge)
        repetitions = int_or_default(judge.get("judge_repetitions") or protocol.get("judge_repetitions"), 0)
        systems = protocol.get("systems", [])
        score_fields = protocol.get("score_fields", [])
        checks.extend(
            [
                check(
                    f"{name}_judge_local_gemma4",
                    judge.get("judge_provider") == "local_ollama"
                    and judge.get("judge_model") == "gemma4:12b-it-qat"
                    and judge.get("remote_provider_allowed") is False,
                    (
                        f"provider={judge.get('judge_provider')} "
                        f"model={judge.get('judge_model')} "
                        f"remote_provider_allowed={judge.get('remote_provider_allowed')}"
                    ),
                ),
                check(
                    f"{name}_judge_repeated",
                    MIN_JUDGE_REPETITIONS <= repetitions <= 5,
                    f"judge_repetitions={repetitions}",
                ),
                check(
                    f"{name}_judgment_complete",
                    judge.get("judgment_complete") is True
                    and int_or_default(judge.get("judged"), 0)
                    == int_or_default(judge.get("expected_judgments"), 0)
                    and int_or_default(judge.get("missing_judgments"), 0) == 0,
                    (
                        f"judgment_complete={judge.get('judgment_complete')} "
                        f"judged={judge.get('judged')} "
                        f"expected_judgments={judge.get('expected_judgments')} "
                        f"missing_judgments={judge.get('missing_judgments')}"
                    ),
                ),
                check(
                    f"{name}_packets_blinded_randomized",
                    bool(protocol.get("packet_blinding"))
                    and randomization["mapped_rows"] == randomization["judgment_rows"]
                    and randomization["unique_mapping_count"] > 1
                    and randomization["min_labels_per_system"] >= 2,
                    f"packet_randomization={randomization}",
                ),
                check(
                    f"{name}_required_systems_and_fields_present",
                    all(system in systems for system in SYSTEMS)
                    and all(field in score_fields for field in FIELDS),
                    f"systems={systems} score_fields={score_fields}",
                ),
                check(
                    f"{name}_confidence_intervals_present",
                    bool(judge.get("confidence_intervals")),
                    "judge artifact includes per-window CIs",
                ),
                check(
                    f"{name}_rag_text_only",
                    fairness.get("traditional_chat_rag_text_only") is True
                    and fairness.get("full_history_text_only") is True
                    and media_attachment_count(judge, "traditional_chat_rag") == 0
                    and media_attachment_count(judge, "full_history_upper_bound") == 0,
                    (
                        f"fairness={fairness} "
                        f"rag_media={media_attachment_count(judge, 'traditional_chat_rag')} "
                        f"full_media={media_attachment_count(judge, 'full_history_upper_bound')}"
                    ),
                ),
                check(
                    f"{name}_no_future_or_current_context",
                    fairness.get("no_future_context_violations") is True
                    and fairness.get("no_current_turn_context_inclusions") is True,
                    f"fairness={fairness}",
                ),
                check(
                    f"{name}_hidden_labels_absent",
                    fairness.get("blind_prompt_hidden_labels_absent") is True,
                    f"fairness={fairness}",
                ),
                check(
                    f"{name}_media_judged_when_present",
                    fairness.get("attached_audio_judged_when_present") is True
                    and fairness.get("attached_images_judged_when_present") is True,
                    f"fairness={fairness}",
                ),
            ]
        )

    if not initial_report:
        checks.append(pending(f"{name}_initial_report_present", f"missing {paths['initial_report']}"))
    else:
        protocol_freeze = initial_report.get("protocol_freeze", {})
        frozen_manifest = initial_report.get("frozen_probe_manifest", {})
        frozen_schedule = initial_report.get("frozen_probe_schedule", {})
        checks.append(
            check(
                f"{name}_frozen_probe_protocol_recorded",
                isinstance(protocol_freeze, dict)
                and bool(protocol_freeze.get("sha256"))
                and isinstance(frozen_manifest, dict)
                and bool(frozen_manifest.get("manifest_sha256"))
                and isinstance(frozen_schedule, dict)
                and bool(frozen_schedule.get("schedule_sha256")),
                (
                    f"protocol_freeze_sha256={protocol_freeze.get('sha256') if isinstance(protocol_freeze, dict) else None} "
                    f"manifest_sha256={frozen_manifest.get('manifest_sha256') if isinstance(frozen_manifest, dict) else None} "
                    f"schedule_sha256={frozen_schedule.get('schedule_sha256') if isinstance(frozen_schedule, dict) else None}"
                ),
            )
        )

    planned_ablations: list[dict[str, Any]] = []
    if ablation_plan:
        planned = ablation_plan.get("cases", []) or ablation_plan.get("ablations", [])
        planned_ablations = [item for item in planned if isinstance(item, dict)]
    planned_names = {
        str(item.get("name")) for item in planned_ablations if item.get("name")
    }
    missing_planned_ablations = sorted(REQUIRED_ABLATIONS - planned_names)
    checks.append(
        check(
            f"{name}_required_ablation_plan_recorded",
            not missing_planned_ablations,
            f"missing={missing_planned_ablations} present={sorted(planned_names)}",
        )
        if planned_ablations
        else pending(f"{name}_required_ablation_plan_recorded", "ablation plan missing")
    )

    ablations = final_public_summary.get("ablations", [])
    if not ablations:
        ablations = final_report.get("ablations", [])
    judged_ablations = [item for item in ablations if isinstance(item, dict)]
    judged_names = {str(item.get("name")) for item in judged_ablations if item.get("name")}
    missing_judged_ablations = sorted(REQUIRED_ABLATIONS - judged_names)
    checks.append(
        check(
            f"{name}_required_judged_ablations_recorded",
            not missing_judged_ablations,
            f"missing={missing_judged_ablations} present={sorted(judged_names)}",
        )
        if judged_ablations
        else pending(
            f"{name}_required_judged_ablations_recorded",
            "judged ablation artifacts missing",
        )
    )
    effect_evidence = [
        ablation_effect_evidence(item)
        for item in judged_ablations
        if str(item.get("name")) in REQUIRED_ABLATIONS
    ]
    unsupported_required = sorted(
        str(item.get("name"))
        for item in effect_evidence
        if not item.get("ci_supports_full")
    )
    checks.append(
        check(
            f"{name}_required_ablation_effects_ci_supported",
            not unsupported_required
            and REQUIRED_ABLATIONS.issubset(
                {str(item.get("name")) for item in effect_evidence}
            ),
            f"unsupported={unsupported_required} evidence={effect_evidence}",
        )
        if effect_evidence
        else pending(
            f"{name}_required_ablation_effects_ci_supported",
            "no judged ablation effect CIs available",
        )
    )

    rows: list[dict[str, Any]] = []
    for row in judge.get("judgments", []) if judge else []:
        if not isinstance(row, dict):
            continue
        rows.append({"window": name, **row})

    summary_out = {
        "name": name,
        "configured": window,
        "selected_media_kind_counts": selected_counts,
        "probe_count": judge.get("probe_count") if judge else summary.get("probe_count"),
        "judge_repetitions": judge.get("judge_repetitions") if judge else None,
        "processed": judge.get("processed") if judge else {},
        "tokens": judge.get("tokens") if judge else {},
        "latency": judge.get("latency") if judge else {},
        "db_metrics": judge.get("db_metrics") if judge else {},
        "costs": cost_summary(summary, judge) if summary else {},
        "human_labels": final_public_summary.get("human_labels")
        or final_report.get("human_labels"),
        "human_label_eval": final_public_summary.get("human_label_eval")
        or final_report.get("human_label_eval"),
        "ablation_plan_names": sorted(planned_names),
        "ablations": judged_ablations,
        "ablation_effect_evidence": effect_evidence,
        "release_gate_status": (
            (final_report.get("release_gate") or {}).get("overall_status")
            if isinstance(final_report.get("release_gate"), dict)
            else (initial_report.get("release_gate") or {}).get("overall_status")
            if isinstance(initial_report.get("release_gate"), dict)
            else ""
        ),
        "artifacts": {key: artifact_ref(path) for key, path in paths.items() if key != "out_dir"},
    }
    return summary_out, checks, rows


def aggregate_confidence_intervals(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            event_index = int(row["event_index"])
        except Exception:
            continue
        grouped.setdefault((str(row["window"]), event_index), []).append(row)

    out: dict[str, Any] = {
        "unit": "window_probe",
        "bootstrap_samples": bootstrap_samples,
        "probe_count": len(grouped),
        "systems": {},
        "tokens": {},
        "quality_delta": {},
    }
    for system in SYSTEMS:
        win_values: list[float] = []
        field_values: dict[str, list[float]] = {field: [] for field in FIELDS}
        quality_values: list[float] = []
        for group_rows in grouped.values():
            win_values.append(
                sum(1.0 for row in group_rows if row.get("winner") == system)
                / len(group_rows)
            )
            scores = []
            for row in group_rows:
                system_scores = (row.get("systems") or {}).get(system, {})
                if isinstance(system_scores, dict):
                    scores.append(system_scores)
            if not scores:
                continue
            for field in FIELDS:
                field_values[field].append(
                    mean([float(score.get(field, 0.0) or 0.0) for score in scores])
                )
            quality_values.append(mean([system_quality(score) for score in scores]))
        system_out = {
            "win_rate": bootstrap_mean_ci(
                win_values,
                bootstrap_samples,
                stable_seed(seed, "aggregate", system, "win_rate"),
            ),
            "quality_composite": bootstrap_mean_ci(
                quality_values,
                bootstrap_samples,
                stable_seed(seed, "aggregate", system, "quality"),
            ),
        }
        for field, values in field_values.items():
            system_out[field] = bootstrap_mean_ci(
                values,
                bootstrap_samples,
                stable_seed(seed, "aggregate", system, field),
            )
        out["systems"][system] = system_out

    token_savings: list[float] = []
    quality_delta_rag: list[float] = []
    quality_delta_full: list[float] = []
    for group_rows in grouped.values():
        cortext_tokens = mean([float(row.get("cortext_context_tokens", 0.0) or 0.0) for row in group_rows])
        rag_tokens = mean([float(row.get("traditional_chat_rag_tokens", 0.0) or 0.0) for row in group_rows])
        if rag_tokens > 0:
            token_savings.append(1.0 - cortext_tokens / rag_tokens)

        per_system_quality: dict[str, float] = {}
        for system in SYSTEMS:
            values = []
            for row in group_rows:
                system_scores = (row.get("systems") or {}).get(system, {})
                if isinstance(system_scores, dict):
                    values.append(system_quality(system_scores))
            if values:
                per_system_quality[system] = mean(values)
        if "cortext_native" in per_system_quality and "traditional_chat_rag" in per_system_quality:
            quality_delta_rag.append(
                per_system_quality["cortext_native"]
                - per_system_quality["traditional_chat_rag"]
            )
        if "cortext_native" in per_system_quality and "full_history_upper_bound" in per_system_quality:
            quality_delta_full.append(
                per_system_quality["cortext_native"]
                - per_system_quality["full_history_upper_bound"]
            )

    out["tokens"]["cortext_savings_vs_traditional_chat_rag"] = bootstrap_mean_ci(
        token_savings,
        bootstrap_samples,
        stable_seed(seed, "aggregate", "tokens", "savings"),
    )
    out["quality_delta"]["cortext_minus_traditional_chat_rag"] = bootstrap_mean_ci(
        quality_delta_rag,
        bootstrap_samples,
        stable_seed(seed, "aggregate", "quality", "rag"),
    )
    out["quality_delta"]["cortext_minus_full_history_upper_bound"] = bootstrap_mean_ci(
        quality_delta_full,
        bootstrap_samples,
        stable_seed(seed, "aggregate", "quality", "full"),
    )
    return out


def aggregate_costs(windows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: list[str]) -> list[float]:
        out = []
        for window in windows:
            value: Any = window
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    cost_fields = [
        "processed_events_estimate",
        "wall_ms",
        "wall_ms_excluding_consolidation",
        "consolidation_ms_total",
        "mean_consolidation_ms_per_run",
        "mean_ingest_total_ms",
        "mean_encode_ms",
        "mean_process_ms",
        "mean_hydrate_ms",
        "mean_cortext_probe_latency_ms",
        "p50_cortext_probe_latency_ms",
        "p95_cortext_probe_latency_ms",
        "p50_rag_total_latency_ms",
        "p95_rag_total_latency_ms",
        "p50_rag_retrieval_latency_ms",
        "p95_rag_retrieval_latency_ms",
        "mean_cortext_prompt_tokens",
        "mean_rag_prompt_tokens",
        "mean_full_history_prompt_tokens",
        "db_disk_bytes",
        "db_disk_bytes_per_event",
        "peak_rss_mb",
        "events_per_second_excluding_consolidation",
        "events_per_second_including_consolidation",
    ]

    costs: dict[str, Any] = {
        field: bootstrap_mean_ci(values(["costs", field]), 0, 0)
        for field in cost_fields
    }

    consolidation_by_window_day: dict[str, dict[str, float]] = {}
    for window in windows:
        name = str(window.get("name", ""))
        by_day = ((window.get("costs") or {}).get("consolidation_ms_by_day") or {})
        if isinstance(by_day, dict) and by_day:
            consolidation_by_window_day[name] = {
                str(day): float_or_default(ms) for day, ms in by_day.items()
            }

    # Backward-compatible aliases for older readers.
    costs.update(
        {
            "mean_cortext_context_tokens": costs["mean_cortext_prompt_tokens"],
            "mean_traditional_chat_rag_tokens": costs["mean_rag_prompt_tokens"],
            "mean_ingest_total_ms": costs["mean_ingest_total_ms"],
            "mean_cortext_probe_latency_ms": costs[
                "mean_cortext_probe_latency_ms"
            ],
            "db_file_bytes": costs["db_disk_bytes"],
            "consolidation_ms_by_window_day": consolidation_by_window_day,
        }
    )
    return costs


def aggregate_human_labels(windows: list[dict[str, Any]]) -> dict[str, Any]:
    window_rows: list[dict[str, Any]] = []
    total_probe_count = 0
    total_shared_probe_events = 0
    total_candidate_pairs = 0
    weighted_kappa_sum = 0.0
    kappas: list[float] = []
    eval_present = 0

    for window in windows:
        name = str(window.get("name", ""))
        human = window.get("human_labels")
        if not isinstance(human, dict):
            continue
        agreement = human.get("agreement", {})
        if not isinstance(agreement, dict):
            agreement = {}
        probe_count = int_or_default(human.get("probe_count"), 0)
        shared = int_or_default(agreement.get("shared_probe_events"), 0)
        candidate_pairs = int_or_default(agreement.get("candidate_pair_count"), 0)
        kappa = float_or_default(
            agreement.get("cohen_kappa_binary_target_membership"), 0.0
        )
        human_eval = window.get("human_label_eval")
        has_eval = isinstance(human_eval, dict) and human_eval.get("schema")

        total_probe_count += probe_count
        total_shared_probe_events += shared
        total_candidate_pairs += candidate_pairs
        weighted_kappa_sum += kappa * max(1, candidate_pairs)
        kappas.append(kappa)
        if has_eval:
            eval_present += 1
        window_rows.append(
            {
                "name": name,
                "schema": human.get("schema"),
                "probe_count": probe_count,
                "shared_probe_events": shared,
                "candidate_pair_count": candidate_pairs,
                "cohen_kappa_binary_target_membership": kappa,
                "probe_event_jaccard": agreement.get("probe_event_jaccard"),
                "human_label_eval_present": bool(has_eval),
                "human_freeze_sha256": human.get("human_freeze_sha256"),
            }
        )

    return {
        "window_count": len(window_rows),
        "total_probe_count": total_probe_count,
        "total_shared_probe_events": total_shared_probe_events,
        "total_candidate_pair_count": total_candidate_pairs,
        "weighted_cohen_kappa_binary_target_membership": (
            weighted_kappa_sum / max(1, total_candidate_pairs)
            if total_candidate_pairs
            else 0.0
        ),
        "min_cohen_kappa_binary_target_membership": min(kappas) if kappas else None,
        "human_label_eval_window_count": eval_present,
        "windows": window_rows,
    }


def aggregate_ablation_evidence(windows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "planned_window_count": 0,
            "judged_window_count": 0,
            "ci_supported_window_count": 0,
            "evidence": [],
        }
        for name in sorted(REQUIRED_ABLATIONS)
    }
    for window in windows:
        window_name = str(window.get("name", ""))
        planned_names = {
            str(name)
            for name in window.get("ablation_plan_names", [])
            if str(name) in REQUIRED_ABLATIONS
        }
        judged_evidence = window.get("ablation_effect_evidence", [])
        if not isinstance(judged_evidence, list):
            judged_evidence = []
        for name in planned_names:
            rows[name]["planned_window_count"] += 1
        for evidence in judged_evidence:
            if not isinstance(evidence, dict):
                continue
            name = str(evidence.get("name"))
            if name not in rows:
                continue
            rows[name]["judged_window_count"] += 1
            if evidence.get("ci_supports_full") is True:
                rows[name]["ci_supported_window_count"] += 1
            rows[name]["evidence"].append({"window": window_name, **evidence})

    missing_planned = [
        name for name, row in rows.items() if row["planned_window_count"] == 0
    ]
    missing_judged = [
        name for name, row in rows.items() if row["judged_window_count"] == 0
    ]
    unsupported = [
        name for name, row in rows.items() if row["ci_supported_window_count"] == 0
    ]
    return {
        "required_ablations": sorted(REQUIRED_ABLATIONS),
        "missing_planned": missing_planned,
        "missing_judged": missing_judged,
        "unsupported_by_ci": unsupported,
        "rows": [rows[name] for name in sorted(rows)],
    }


def aggregate_ablation_checks(ablations: dict[str, Any]) -> list[dict[str, Any]]:
    missing_planned = list(ablations.get("missing_planned", []))
    missing_judged = list(ablations.get("missing_judged", []))
    unsupported = list(ablations.get("unsupported_by_ci", []))
    return [
        check(
            "aggregate_required_ablation_plan_recorded",
            not missing_planned,
            f"missing_planned={missing_planned}",
        )
        if not missing_planned
        else pending(
            "aggregate_required_ablation_plan_recorded",
            f"missing_planned={missing_planned}",
        ),
        check(
            "aggregate_required_judged_ablations_recorded",
            not missing_judged,
            f"missing_judged={missing_judged}",
        )
        if not missing_judged
        else pending(
            "aggregate_required_judged_ablations_recorded",
            f"missing_judged={missing_judged}",
        ),
        check(
            "aggregate_required_ablation_effects_ci_supported",
            not unsupported,
            f"unsupported_by_ci={unsupported} evidence={ablations.get('rows')}",
        )
        if not missing_judged
        else pending(
            "aggregate_required_ablation_effects_ci_supported",
            f"missing_judged={missing_judged}",
        ),
    ]


def aggregate_cost_checks(costs: dict[str, Any]) -> list[dict[str, Any]]:
    def ci_mean(name: str) -> float:
        value = costs.get(name, {})
        return float_or_default(value.get("mean") if isinstance(value, dict) else None)

    def ci_n(name: str) -> int:
        value = costs.get(name, {})
        return int_or_default(value.get("n") if isinstance(value, dict) else None, 0)

    required = [
        "events_per_second_excluding_consolidation",
        "events_per_second_including_consolidation",
        "p50_cortext_probe_latency_ms",
        "p95_cortext_probe_latency_ms",
        "p50_rag_retrieval_latency_ms",
        "p95_rag_retrieval_latency_ms",
        "mean_cortext_prompt_tokens",
        "mean_rag_prompt_tokens",
        "db_disk_bytes",
        "db_disk_bytes_per_event",
        "peak_rss_mb",
    ]
    missing = [name for name in required if ci_n(name) == 0]
    checks = [
        check(
            "aggregate_production_cost_metrics_present",
            not missing,
            f"missing={missing}",
        )
        if not missing
        else pending(
            "aggregate_production_cost_metrics_present",
            f"missing={missing}",
        )
    ]

    checks.extend(
        [
            check(
                "aggregate_ingest_throughput_recorded",
                ci_mean("events_per_second_excluding_consolidation") > 0.0
                and ci_mean("events_per_second_including_consolidation") > 0.0,
                (
                    "excluding="
                    f"{costs.get('events_per_second_excluding_consolidation')} "
                    "including="
                    f"{costs.get('events_per_second_including_consolidation')}"
                ),
            )
            if ci_n("events_per_second_excluding_consolidation")
            and ci_n("events_per_second_including_consolidation")
            else pending(
                "aggregate_ingest_throughput_recorded",
                "completed summaries not available yet",
            ),
            check(
                "aggregate_retrieval_latency_percentiles_recorded",
                ci_mean("p50_cortext_probe_latency_ms") > 0.0
                and ci_mean("p95_cortext_probe_latency_ms")
                >= ci_mean("p50_cortext_probe_latency_ms")
                and ci_mean("p50_rag_retrieval_latency_ms") > 0.0
                and ci_mean("p95_rag_retrieval_latency_ms")
                >= ci_mean("p50_rag_retrieval_latency_ms"),
                (
                    f"cortext_p50={costs.get('p50_cortext_probe_latency_ms')} "
                    f"cortext_p95={costs.get('p95_cortext_probe_latency_ms')} "
                    f"rag_p50={costs.get('p50_rag_retrieval_latency_ms')} "
                    f"rag_p95={costs.get('p95_rag_retrieval_latency_ms')}"
                ),
            )
            if ci_n("p50_cortext_probe_latency_ms")
            and ci_n("p50_rag_retrieval_latency_ms")
            else pending(
                "aggregate_retrieval_latency_percentiles_recorded",
                "completed probe summaries not available yet",
            ),
            check(
                "aggregate_consolidation_wall_time_by_day_recorded",
                bool(costs.get("consolidation_ms_by_window_day")),
                f"consolidation_ms_by_window_day={costs.get('consolidation_ms_by_window_day')}",
            )
            if ci_n("consolidation_ms_total")
            else pending(
                "aggregate_consolidation_wall_time_by_day_recorded",
                "completed daily-consolidation summaries not available yet",
            ),
            check(
                "aggregate_disk_growth_recorded",
                ci_mean("db_disk_bytes") > 0.0
                and ci_mean("db_disk_bytes_per_event") > 0.0,
                (
                    f"db_disk_bytes={costs.get('db_disk_bytes')} "
                    f"db_disk_bytes_per_event={costs.get('db_disk_bytes_per_event')}"
                ),
            )
            if ci_n("db_disk_bytes")
            else pending(
                "aggregate_disk_growth_recorded",
                "completed DB metrics not available yet",
            ),
            check(
                "aggregate_memory_use_recorded",
                ci_mean("peak_rss_mb") > 0.0,
                f"peak_rss_mb={costs.get('peak_rss_mb')}",
            )
            if ci_n("peak_rss_mb")
            else pending(
                "aggregate_memory_use_recorded",
                "completed RSS metrics not available yet",
            ),
        ]
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--protocol-spec", type=pathlib.Path)
    parser.add_argument("--status", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_dir = args.base_dir.resolve()
    manifest_path = (args.manifest or args.base_dir / "release_windows_manifest.json").resolve()
    status_path = (args.status or args.base_dir / "release_windows_status.json").resolve()
    out_path = (args.out or args.base_dir / "release_windows_report.json").resolve()

    manifest = load_json(manifest_path)
    manifest_spec_ref = manifest.get("protocol_spec", {})
    manifest_spec_path = (
        pathlib.Path(str(manifest_spec_ref.get("path")))
        if isinstance(manifest_spec_ref, dict) and manifest_spec_ref.get("path")
        else DEFAULT_PROTOCOL_SPEC
    )
    protocol_spec_path = (args.protocol_spec or manifest_spec_path).resolve()
    protocol_spec = load_protocol_spec(protocol_spec_path)
    status = load_json_if_valid(status_path)
    fixed = manifest.get("fixed_protocol", {})
    windows_config = fixed.get("windows", [])
    if not isinstance(windows_config, list):
        windows_config = []
    spec_windows = protocol_spec.get("windows", [])
    if not isinstance(spec_windows, list):
        spec_windows = []
    protocol_compliance = manifest.get("protocol_spec_compliance", {})
    if not isinstance(protocol_compliance, dict):
        protocol_compliance = {}
    spec_ablation_names = {
        str(item.get("name"))
        for item in protocol_spec.get("ablations", [])
        if isinstance(item, dict) and item.get("name")
    }
    fixed_required_ablations = set(str(item) for item in fixed.get("required_ablations", []))

    checks: list[dict[str, Any]] = []
    checks.extend(
        [
            check(
                "manifest_present",
                manifest.get("schema") == "cortext_julie_release_windows_manifest_v1",
                f"schema={manifest.get('schema')}",
            ),
            check(
                "protocol_spec_present",
                protocol_spec.get("schema") == "cortext_julie_release_protocol_spec_v1"
                and protocol_spec_path.exists(),
                f"path={protocol_spec_path} schema={protocol_spec.get('schema')}",
            ),
            check(
                "protocol_spec_hash_matches_manifest",
                isinstance(manifest_spec_ref, dict)
                and manifest_spec_ref.get("sha256") == file_sha256(protocol_spec_path)
                and manifest.get("protocol_spec_payload_sha256")
                == canonical_json_sha256(protocol_spec),
                (
                    f"manifest_file_sha={manifest_spec_ref.get('sha256') if isinstance(manifest_spec_ref, dict) else None} "
                    f"actual_file_sha={file_sha256(protocol_spec_path) if protocol_spec_path.exists() else ''} "
                    f"manifest_payload_sha={manifest.get('protocol_spec_payload_sha256')} "
                    f"actual_payload_sha={canonical_json_sha256(protocol_spec)}"
                ),
            ),
            check(
                "protocol_windows_match_manifest",
                normalize_windows(windows_config) == normalize_windows(spec_windows)
                and protocol_compliance.get("windows_match_spec") is True,
                (
                    f"manifest_windows={normalize_windows(windows_config)} "
                    f"spec_windows={normalize_windows(spec_windows)} "
                    f"compliance={protocol_compliance.get('windows_match_spec')}"
                ),
            ),
            check(
                "default_knobs_only",
                fixed.get("knobs") == (protocol_spec.get("cortext", {}) or {}).get("knobs")
                == {"focus": 0.5, "sensitivity": 0.5, "stability": 0.5},
                f"knobs={fixed.get('knobs')} spec={(protocol_spec.get('cortext', {}) or {}).get('knobs')}",
            ),
            check(
                "daily_deep_consolidation_required",
                fixed.get("daily_consolidation") is True and fixed.get("deep_consolidation") is True,
                f"daily={fixed.get('daily_consolidation')} deep={fixed.get('deep_consolidation')}",
            ),
            check(
                "text_rag_baseline_recorded",
                fixed.get("normal_rag") == "rolling chat history until compaction plus text vector RAG",
                f"normal_rag={fixed.get('normal_rag')}",
            ),
            check(
                "rag_baseline_matches_protocol_spec",
                fixed.get("normal_rag_retrieval")
                == ((protocol_spec.get("baselines", {}) or {}).get("traditional_chat_rag", {}) or {}).get("normal_rag_retrieval")
                and fixed.get("normal_rag_baseline_modality")
                == ((protocol_spec.get("baselines", {}) or {}).get("traditional_chat_rag", {}) or {}).get("modality")
                and int_or_default(fixed.get("rag_top_k"), -1)
                == int_or_default(
                    ((protocol_spec.get("baselines", {}) or {}).get("traditional_chat_rag", {}) or {}).get("rag_top_k"),
                    -2,
                )
                and int_or_default(fixed.get("active_history_token_budget"), -1)
                == int_or_default(
                    ((protocol_spec.get("baselines", {}) or {}).get("traditional_chat_rag", {}) or {}).get("active_history_token_budget"),
                    -2,
                ),
                (
                    f"rag_fixed={fixed.get('normal_rag_retrieval')}/"
                    f"{fixed.get('normal_rag_baseline_modality')}/"
                    f"{fixed.get('rag_top_k')}/"
                    f"{fixed.get('active_history_token_budget')}"
                ),
            ),
            check(
                "local_repeated_blind_judge_configured",
                fixed.get("judge_provider") == "local_ollama"
                and fixed.get("judge_model") == "gemma4:12b-it-qat"
                and int_or_default(fixed.get("judge_repetitions"), 0) >= MIN_JUDGE_REPETITIONS
                and fixed.get("blind_packets") is True,
                (
                    f"judge_provider={fixed.get('judge_provider')} "
                    f"judge_model={fixed.get('judge_model')} "
                    f"judge_repetitions={fixed.get('judge_repetitions')} "
                    f"blind_packets={fixed.get('blind_packets')}"
                ),
            ),
            check(
                "judge_packet_randomization_recorded",
                fixed.get("randomize_packet_order") is True
                and fixed.get("packet_aliases") == ["A", "B", "C"],
                (
                    f"randomize_packet_order={fixed.get('randomize_packet_order')} "
                    f"packet_aliases={fixed.get('packet_aliases')}"
                ),
            ),
            check(
                "required_ablations_match_protocol_spec",
                spec_ablation_names == fixed_required_ablations
                and REQUIRED_ABLATIONS.issubset(spec_ablation_names),
                (
                    f"spec={sorted(spec_ablation_names)} "
                    f"manifest={sorted(fixed_required_ablations)} "
                    f"required={sorted(REQUIRED_ABLATIONS)}"
                ),
            ),
            check(
                "window_plan_present",
                len(windows_config) > 0,
                f"window_count={len(windows_config)}",
            ),
        ]
    )

    shared_smoke = status.get("shared_judge_media_smoke", {})
    if status:
        checks.append(
            check(
                "shared_judge_media_smoke_present",
                isinstance(shared_smoke, dict) and shared_smoke.get("exists") is True,
                f"shared_judge_media_smoke={shared_smoke}",
            )
        )
    else:
        checks.append(pending("status_present", f"missing {status_path}"))

    window_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for window in windows_config:
        if not isinstance(window, dict):
            continue
        summary, window_checks, rows = collect_window(
            args.base_dir,
            window,
            args.bootstrap_samples,
        )
        window_summaries.append(summary)
        checks.extend(window_checks)
        all_rows.extend(rows)

    ci = aggregate_confidence_intervals(all_rows, args.bootstrap_samples, args.seed)
    costs = aggregate_costs(window_summaries)
    human_labels = aggregate_human_labels(window_summaries)
    ablation_evidence = aggregate_ablation_evidence(window_summaries)
    checks.extend(aggregate_cost_checks(costs))
    checks.extend(aggregate_ablation_checks(ablation_evidence))
    aggregate_probe_count = int(ci.get("probe_count", 0) or 0)
    token_savings = ((ci.get("tokens") or {}).get("cortext_savings_vs_traditional_chat_rag") or {})
    quality_delta = ((ci.get("quality_delta") or {}).get("cortext_minus_traditional_chat_rag") or {})
    token_ci = token_savings.get("ci95", [0.0, 0.0])
    quality_ci = quality_delta.get("ci95", [0.0, 0.0])
    token_lower = float(token_ci[0]) if isinstance(token_ci, list) and token_ci else 0.0
    quality_lower = float(quality_ci[0]) if isinstance(quality_ci, list) and quality_ci else 0.0

    checks.extend(
        [
            check(
                "aggregate_probe_count_floor",
                aggregate_probe_count >= MIN_RELEASE_PROBES,
                f"aggregate_probe_count={aggregate_probe_count} min_required={MIN_RELEASE_PROBES}",
            )
            if aggregate_probe_count
            else pending(
                "aggregate_probe_count_floor",
                "no completed judge rows available yet",
            ),
            check(
                "aggregate_token_savings_ci_supported",
                token_lower >= MIN_TOKEN_SAVINGS_CI95_LOWER,
                (
                    f"mean={token_savings.get('mean')} ci95={token_ci} "
                    f"required_lower={MIN_TOKEN_SAVINGS_CI95_LOWER}"
                ),
            )
            if token_savings.get("n", 0)
            else pending("aggregate_token_savings_ci_supported", "no completed token CI yet"),
            check(
                "aggregate_quality_matches_or_beats_rag_ci_supported",
                quality_lower >= 0.0 and int_or_default(quality_delta.get("n"), 0) >= MIN_RELEASE_PROBES,
                f"mean={quality_delta.get('mean')} ci95={quality_ci}",
            )
            if quality_delta.get("n", 0)
            else pending("aggregate_quality_matches_or_beats_rag_ci_supported", "no completed quality CI yet"),
            check(
                "human_label_agreement_present",
                human_labels["total_probe_count"] >= MIN_HUMAN_PROBES
                and human_labels["total_shared_probe_events"] >= MIN_HUMAN_PROBES
                and human_labels["human_label_eval_window_count"]
                == human_labels["window_count"]
                and human_labels["window_count"] > 0,
                (
                    "human_labels="
                    f"{human_labels} min_required_shared_probes={MIN_HUMAN_PROBES}"
                ),
            )
            if human_labels["window_count"] > 0
            else pending(
                "human_label_agreement_present",
                (
                    "multi-window aggregate human-label agreement is not attached; "
                    f"requires >= {MIN_HUMAN_PROBES} shared probes"
                ),
            ),
        ]
    )

    status_counts: dict[str, int] = {"pass": 0, "fail": 0, "pending": 0}
    for item in checks:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    overall_status = (
        "pass"
        if status_counts.get("fail", 0) == 0 and status_counts.get("pending", 0) == 0
        else "not_ready"
    )

    output = {
        "schema": "cortext_julie_release_windows_report_v1",
        "created_at_utc": utc_now(),
        "privacy": (
            "public-safe aggregate report: no message text, media bytes, "
            "per-candidate labels, or judge reason strings"
        ),
        "git": git_provenance(),
        "artifacts": {
            "manifest": artifact_ref(manifest_path),
            "protocol_spec": artifact_ref(protocol_spec_path),
            "status": artifact_ref(status_path),
            "shared_judge_media_smoke": shared_smoke,
        },
        "protocol_spec_schema": protocol_spec.get("schema"),
        "protocol_spec_version": protocol_spec.get("version"),
        "fixed_protocol": fixed,
        "windows": window_summaries,
        "aggregate": {
            "quality_composite_definition": QUALITY_DEFINITION,
            "quality_composite_weights": QUALITY_WEIGHTS,
            "confidence_intervals": ci,
            "costs": costs,
            "human_labels": human_labels,
            "claim_support": {
                "token_savings_threshold": {
                    "required_ci95_lower_bound": MIN_TOKEN_SAVINGS_CI95_LOWER,
                    "observed": token_savings,
                },
                "quality_vs_traditional_chat_rag": quality_delta,
                "quality_vs_full_history_upper_bound": (
                    (ci.get("quality_delta") or {}).get(
                        "cortext_minus_full_history_upper_bound"
                    )
                    or {}
                ),
                "ablation_evidence": ablation_evidence,
            },
        },
        "release_gate": {
            "overall_status": overall_status,
            "status_counts": status_counts,
            "checks": checks,
            "narrow_claim_template": (
                "On a frozen private mixed-media conversation replay, production "
                "Cortext at default knobs reduced prompt tokens substantially versus "
                "traditional chat+RAG while matching or exceeding judged context "
                "usefulness on the evaluated probes. Daily consolidation and "
                "graph-backed retrieval contributed measurable value under ablation."
            ),
        },
    }
    write_json(out_path, output)
    print(out_path)
    print(f"overall_status={overall_status} checks={status_counts}")
    if args.require_pass and overall_status != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
