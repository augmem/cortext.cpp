#!/usr/bin/env python3
"""Build a public-safe Julie release-eval protocol report.

The report intentionally excludes private message text and judge reason strings.
It is a reproducibility and release-gating wrapper around an existing Cortext
live-run summary plus one or more local judge artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


REQUIRED_SYSTEMS = [
    "cortext_native",
    "traditional_chat_rag",
    "full_history_upper_bound",
]
REQUIRED_FIELDS = ["relevance", "sufficiency", "noise"]
DEFAULT_KNOBS = {"focus": 0.5, "sensitivity": 0.5, "stability": 0.5}
MIN_RELEASE_PROBES = 10
MIN_JUDGE_REPETITIONS = 3
MIN_HUMAN_PROBES = 10
MIN_HUMAN_SHARED_PROBES = 10
MIN_HUMAN_KAPPA = 0.4
ABLATION_BOOTSTRAP_SAMPLES = 2000
MIN_TOKEN_SAVINGS_CI95_LOWER = 0.5
QUALITY_COMPOSITE_FIELDS = {
    "relevance": 1.0,
    "sufficiency": 1.0,
    "noise": -1.0,
}
REQUIRED_ABLATION_CATEGORIES = {
    "no_daily_consolidation": {
        "no_daily",
        "no-daily",
        "no_daily_consolidation",
        "final_batch_consolidation",
    },
    "no_graph_expansion": {
        "no_graph",
        "no-graph",
        "no_graph_expansion",
        "no-source-seed-graph",
        "no_source_seed_graph",
        "source_seed_graph_disabled",
    },
    "no_media_source_blobs": {
        "no_media",
        "no-media",
        "no_media_source_blobs",
        "text_only",
    },
    "no_stm_ltm_graph_label_handoff": {
        "no_stm_label_handoff",
        "no-stm-label-handoff",
        "no_stm_ltm_label_handoff",
        "no_stm_ltm_graph_label_handoff",
        "stm_label_handoff_disabled",
    },
    "no_temporal_retrieval": {
        "no_temporal",
        "no-temporal",
        "no_temporal_retrieval",
        "temporal_retrieval_disabled",
    },
    "no_fact_boosts": {
        "no_facts",
        "no-facts",
        "no_fact_boosts",
        "facts_disabled",
        "fact_boosts_disabled",
    },
}
REQUIRED_ABLATION_ENV_BY_CATEGORY = {
    "no_graph_expansion": {"CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION": "1"},
    "no_media_source_blobs": {"CORTEXT_DISABLE_SOURCE_BLOBS": "1"},
    "no_stm_ltm_graph_label_handoff": {"CORTEXT_DISABLE_STM_LABEL_HANDOFF": "1"},
    "no_temporal_retrieval": {"CORTEXT_DISABLE_TEMPORAL_RETRIEVAL": "1"},
    "no_fact_boosts": {"CORTEXT_DISABLE_FACTS": "1"},
}


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def git_info() -> dict[str, Any]:
    status = git_value(["status", "--porcelain"], "")
    return {
        "commit": git_value(["rev-parse", "HEAD"], "unknown"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def check(name: str, passed: bool, detail: str = "", severity: str = "required") -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "severity": severity,
        "detail": detail,
    }


def pending(name: str, detail: str, severity: str = "required") -> dict[str, Any]:
    return {
        "name": name,
        "status": "pending",
        "severity": severity,
        "detail": detail,
    }


def number_close(a: Any, b: float, eps: float = 1e-9) -> bool:
    try:
        return abs(float(a) - b) <= eps
    except Exception:
        return False


def int_or_default(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(values), "mean": mean(values), "ci95": [0.0, 0.0]}
    if not values:
        return summary
    if len(values) == 1 or samples <= 0:
        summary["ci95"] = [values[0], values[0]]
        return summary
    rng = random.Random(seed)
    boot = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    lo = percentile(boot, 0.025)
    hi = percentile(boot, 0.975)
    summary["ci95"] = [lo if lo is not None else 0.0, hi if hi is not None else 0.0]
    return summary


def system_quality_composite(system_scores: dict[str, Any]) -> float:
    total = 0.0
    for field, weight in QUALITY_COMPOSITE_FIELDS.items():
        try:
            total += float(system_scores.get(field, 0.0) or 0.0) * weight
        except Exception:
            pass
    return total


def probe_quality_by_event(judge: dict[str, Any], system: str) -> dict[int, float]:
    by_event: dict[int, list[float]] = {}
    for row in judge.get("judgments", []):
        if not isinstance(row, dict):
            continue
        try:
            event_index = int(row["event_index"])
        except Exception:
            continue
        systems = row.get("systems", {})
        if not isinstance(systems, dict) or not isinstance(systems.get(system), dict):
            continue
        by_event.setdefault(event_index, []).append(
            system_quality_composite(systems[system])
        )
    return {event_index: mean(values) for event_index, values in by_event.items()}


def probe_wins_by_event(judge: dict[str, Any], system: str) -> dict[int, float]:
    by_event: dict[int, list[float]] = {}
    for row in judge.get("judgments", []):
        if not isinstance(row, dict):
            continue
        try:
            event_index = int(row["event_index"])
        except Exception:
            continue
        by_event.setdefault(event_index, []).append(
            1.0 if row.get("winner") == system else 0.0
        )
    return {event_index: mean(values) for event_index, values in by_event.items()}


def paired_delta_summary(
    main_judge: dict[str, Any],
    ablation_judge: dict[str, Any],
    system: str,
    name: str,
) -> dict[str, Any]:
    main_quality = probe_quality_by_event(main_judge, system)
    ablation_quality = probe_quality_by_event(ablation_judge, system)
    shared_quality_events = sorted(set(main_quality) & set(ablation_quality))
    quality_deltas = [
        main_quality[event_index] - ablation_quality[event_index]
        for event_index in shared_quality_events
    ]

    main_wins = probe_wins_by_event(main_judge, system)
    ablation_wins = probe_wins_by_event(ablation_judge, system)
    shared_win_events = sorted(set(main_wins) & set(ablation_wins))
    win_rate_deltas = [
        main_wins[event_index] - ablation_wins[event_index]
        for event_index in shared_win_events
    ]

    return {
        "system": system,
        "composite_definition": "relevance + sufficiency - noise",
        "shared_quality_probe_count": len(shared_quality_events),
        "shared_win_probe_count": len(shared_win_events),
        "quality_delta_full_minus_ablation": bootstrap_mean_ci(
            quality_deltas,
            ABLATION_BOOTSTRAP_SAMPLES,
            stable_seed("ablation-quality", name),
        ),
        "win_rate_delta_full_minus_ablation": bootstrap_mean_ci(
            win_rate_deltas,
            ABLATION_BOOTSTRAP_SAMPLES,
            stable_seed("ablation-win-rate", name),
        ),
    }


def system_quality_delta_summary(
    judge: dict[str, Any],
    system: str,
    baseline: str,
    name: str,
) -> dict[str, Any]:
    system_quality = probe_quality_by_event(judge, system)
    baseline_quality = probe_quality_by_event(judge, baseline)
    shared_quality_events = sorted(set(system_quality) & set(baseline_quality))
    quality_deltas = [
        system_quality[event_index] - baseline_quality[event_index]
        for event_index in shared_quality_events
    ]

    system_wins = probe_wins_by_event(judge, system)
    baseline_wins = probe_wins_by_event(judge, baseline)
    shared_win_events = sorted(set(system_wins) & set(baseline_wins))
    win_rate_deltas = [
        system_wins[event_index] - baseline_wins[event_index]
        for event_index in shared_win_events
    ]

    return {
        "system": system,
        "baseline": baseline,
        "composite_definition": "relevance + sufficiency - noise",
        "shared_quality_probe_count": len(shared_quality_events),
        "shared_win_probe_count": len(shared_win_events),
        "quality_delta_system_minus_baseline": bootstrap_mean_ci(
            quality_deltas,
            ABLATION_BOOTSTRAP_SAMPLES,
            stable_seed("system-quality", name, system, baseline),
        ),
        "win_rate_delta_system_minus_baseline": bootstrap_mean_ci(
            win_rate_deltas,
            ABLATION_BOOTSTRAP_SAMPLES,
            stable_seed("system-win-rate", name, system, baseline),
        ),
    }


def nested_value(body: dict[str, Any], path: list[str]) -> Any:
    value: Any = body
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def claim_support_summary(judge: dict[str, Any]) -> dict[str, Any]:
    token_savings = nested_value(
        judge,
        [
            "confidence_intervals",
            "tokens",
            "cortext_savings_vs_traditional_chat_rag",
        ],
    )
    if not isinstance(token_savings, dict):
        token_savings = {}
    return {
        "token_savings_threshold": {
            "metric": "cortext_savings_vs_traditional_chat_rag",
            "required_ci95_lower_bound": MIN_TOKEN_SAVINGS_CI95_LOWER,
            "observed": token_savings,
        },
        "quality_vs_traditional_chat_rag": system_quality_delta_summary(
            judge,
            "cortext_native",
            "traditional_chat_rag",
            "cortext_vs_traditional_chat_rag",
        ),
        "quality_vs_full_history_upper_bound": system_quality_delta_summary(
            judge,
            "cortext_native",
            "full_history_upper_bound",
            "cortext_vs_full_history_upper_bound",
        ),
    }


def claim_support_checks(claim_support: dict[str, Any]) -> list[dict[str, Any]]:
    token_observed = claim_support.get("token_savings_threshold", {}).get(
        "observed", {}
    )
    token_ci95 = token_observed.get("ci95", [0.0, 0.0])
    token_lower = float(token_ci95[0]) if isinstance(token_ci95, list) and token_ci95 else 0.0
    token_mean = float(token_observed.get("mean", 0.0) or 0.0)

    quality = claim_support.get("quality_vs_traditional_chat_rag", {})
    quality_delta = quality.get("quality_delta_system_minus_baseline", {})
    quality_ci95 = quality_delta.get("ci95", [0.0, 0.0])
    quality_lower = (
        float(quality_ci95[0]) if isinstance(quality_ci95, list) and quality_ci95 else 0.0
    )
    quality_mean = float(quality_delta.get("mean", 0.0) or 0.0)
    shared_quality_probe_count = int(quality.get("shared_quality_probe_count", 0) or 0)

    return [
        check(
            "claim_token_savings_substantial_ci_supported",
            token_lower >= MIN_TOKEN_SAVINGS_CI95_LOWER,
            (
                f"mean={token_mean} ci95={token_ci95} "
                f"required_lower_bound={MIN_TOKEN_SAVINGS_CI95_LOWER}"
            ),
        ),
        check(
            "claim_quality_matches_or_beats_rag_ci_supported",
            shared_quality_probe_count >= MIN_RELEASE_PROBES
            and quality_lower >= 0.0,
            (
                f"shared_quality_probe_count={shared_quality_probe_count} "
                f"mean_delta={quality_mean} ci95={quality_ci95}"
            ),
        ),
    ]


def ci_metric_is_present(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    ci95 = value.get("ci95")
    return (
        isinstance(value.get("n"), int)
        and isinstance(value.get("mean"), (int, float))
        and isinstance(ci95, list)
        and len(ci95) == 2
        and all(isinstance(item, (int, float)) for item in ci95)
    )


def missing_required_ci_metrics(judge: dict[str, Any]) -> list[str]:
    ci = judge.get("confidence_intervals", {})
    missing = []
    for system in REQUIRED_SYSTEMS:
        system_ci = nested_value(ci, ["systems", system])
        if not isinstance(system_ci, dict):
            missing.append(f"systems.{system}")
            continue
        for field in ["win_rate", *REQUIRED_FIELDS]:
            if not ci_metric_is_present(system_ci.get(field)):
                missing.append(f"systems.{system}.{field}")
    token_ci = nested_value(
        ci,
        ["tokens", "cortext_savings_vs_traditional_chat_rag"],
    )
    if not ci_metric_is_present(token_ci):
        missing.append("tokens.cortext_savings_vs_traditional_chat_rag")
    return missing


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def command_flag_value(parts: list[str], flag: str) -> str | None:
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts):
            return parts[i + 1]
        prefix = flag + "="
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def command_has_flag(parts: list[str], flag: str) -> bool:
    return flag in parts or any(part.startswith(flag + "=") for part in parts)


def command_float_value(parts: list[str], flag: str) -> float | None:
    value = command_flag_value(parts, flag)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def command_int_value(parts: list[str], flag: str) -> int | None:
    value = command_flag_value(parts, flag)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def command_protocol_checks(
    benchmark_command: str,
    judge_command: str,
) -> list[dict[str, Any]]:
    bench = split_command(benchmark_command)
    judge = split_command(judge_command)
    judge_repetitions = command_int_value(judge, "--judge-repetitions")
    return [
        check(
            "benchmark_command_recorded",
            bool(bench),
            f"benchmark_command={benchmark_command!r}",
        ),
        check(
            "benchmark_command_freezes_slice",
            command_flag_value(bench, "--input-dir") is not None
            and command_int_value(bench, "--max-messages") is not None
            and command_int_value(bench, "--media-limit") is not None,
            (
                f"input_dir={command_flag_value(bench, '--input-dir')} "
                f"max_messages={command_flag_value(bench, '--max-messages')} "
                f"media_limit={command_flag_value(bench, '--media-limit')}"
            ),
        ),
        check(
            "benchmark_command_freezes_probe_schedule",
            command_int_value(bench, "--probe-stride") is not None
            and command_int_value(bench, "--warmup-events") is not None,
            (
                f"probe_stride={command_flag_value(bench, '--probe-stride')} "
                f"warmup_events={command_flag_value(bench, '--warmup-events')}"
            ),
        ),
        check(
            "benchmark_command_default_knobs",
            all(
                number_close(command_float_value(bench, f"--{key}"), value)
                for key, value in DEFAULT_KNOBS.items()
            ),
            (
                f"focus={command_flag_value(bench, '--focus')} "
                f"sensitivity={command_flag_value(bench, '--sensitivity')} "
                f"stability={command_flag_value(bench, '--stability')}"
            ),
        ),
        check(
            "benchmark_command_daily_deep_consolidation",
            command_has_flag(bench, "--daily-consolidation")
            and command_has_flag(bench, "--deep"),
            (
                f"daily_consolidation={command_has_flag(bench, '--daily-consolidation')} "
                f"deep={command_has_flag(bench, '--deep')}"
            ),
        ),
        check(
            "benchmark_command_fixed_rag_baseline",
            command_int_value(bench, "--rag-top-k") is not None
            and command_int_value(bench, "--active-history-token-budget") is not None,
            (
                f"rag_top_k={command_flag_value(bench, '--rag-top-k')} "
                "active_history_token_budget="
                f"{command_flag_value(bench, '--active-history-token-budget')}"
            ),
        ),
        check(
            "judge_command_recorded",
            bool(judge),
            f"judge_command={judge_command!r}",
        ),
        check(
            "judge_command_local_gemma4_blind_repeated",
            command_flag_value(judge, "--judge-provider") == "ollama"
            and command_flag_value(judge, "--model") == "gemma4:12b-it-qat"
            and command_has_flag(judge, "--blind-packets")
            and judge_repetitions is not None
            and MIN_JUDGE_REPETITIONS <= judge_repetitions <= 5,
            (
                f"judge_provider={command_flag_value(judge, '--judge-provider')} "
                f"model={command_flag_value(judge, '--model')} "
                f"blind={command_has_flag(judge, '--blind-packets')} "
                f"judge_repetitions={judge_repetitions}"
            ),
        ),
    ]


def numeric_probe_values(summary: dict[str, Any], key: str) -> list[float]:
    values = []
    for probe in summary.get("probes", []):
        value = probe.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def non_text_probe_rows(summary: dict[str, Any], field: str) -> list[dict[str, Any]]:
    rows = []
    for probe in summary.get("probes", []):
        bad = [
            row
            for row in probe.get(field, [])
            if isinstance(row, dict) and row.get("modality") != "text"
        ]
        if bad:
            rows.append(
                {
                    "event_index": probe.get("event_index"),
                    "field": field,
                    "count": len(bad),
                }
            )
    return rows


def probes_missing_numeric(summary: dict[str, Any], field: str) -> list[Any]:
    missing = []
    for probe in summary.get("probes", []):
        if not isinstance(probe.get(field), (int, float)):
            missing.append(probe.get("event_index"))
    return missing


def rag_top_k_size_mismatches(summary: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    try:
        top_k = int(summary.get("rag_top_k", 0) or 0)
    except Exception:
        top_k = 0
    if top_k <= 0:
        return mismatches
    for probe in summary.get("probes", []):
        rag_rows = probe.get("rag_top_k", [])
        if not isinstance(rag_rows, list):
            rag_rows = []
        prior_rows = probe.get("normal_rag_vector_prior_chat_rows", 0)
        try:
            prior_count = int(prior_rows or 0)
        except Exception:
            prior_count = 0
        if prior_count >= top_k and len(rag_rows) != top_k:
            mismatches.append(
                {
                    "event_index": probe.get("event_index"),
                    "rag_rows": len(rag_rows),
                    "expected": top_k,
                    "prior_rows": prior_count,
                }
            )
    return mismatches


def path_size_bytes(path: pathlib.Path) -> int:
    total = 0
    candidates = [path, pathlib.Path(str(path) + "-wal"), pathlib.Path(str(path) + "-shm")]
    for candidate in candidates:
        try:
            if candidate.exists():
                total += candidate.stat().st_size
        except OSError:
            pass
    return total


def canonical_ablation_category(name: str) -> str | None:
    normalized = name.strip().lower().replace(" ", "_")
    normalized_dash = normalized.replace("_", "-")
    for category, aliases in REQUIRED_ABLATION_CATEGORIES.items():
        if normalized == category or normalized in aliases or normalized_dash in aliases:
            return category
    return None


def summary_protocol_checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out = [
        check(
            "daily_consolidation_enabled",
            bool(summary.get("daily_consolidation")),
            f"daily_consolidation={summary.get('daily_consolidation')!r}",
        ),
        check(
            "deep_consolidation_enabled",
            bool(summary.get("deep_consolidation")),
            f"deep_consolidation={summary.get('deep_consolidation')!r}",
        ),
        check(
            "fixed_probes_present",
            int(summary.get("probe_count", 0) or 0) == len(summary.get("probes", []))
            and len(summary.get("probes", [])) > 0,
            f"probe_count={summary.get('probe_count')} probes={len(summary.get('probes', []))}",
        ),
        check(
            "rag_baseline_fixed",
            int(summary.get("rag_top_k", -1) or -1) > 0
            and int(summary.get("active_history_token_budget", -1) or -1) > 0,
            (
                f"rag_top_k={summary.get('rag_top_k')} "
                f"active_history_token_budget={summary.get('active_history_token_budget')}"
            ),
        ),
        check(
            "rag_baseline_vector_text",
            summary.get("normal_rag_retrieval") == "raw_chat_vector"
            and summary.get("normal_rag_baseline_modality") == "text_only",
            (
                f"normal_rag_retrieval={summary.get('normal_rag_retrieval')!r} "
                f"normal_rag_baseline_modality={summary.get('normal_rag_baseline_modality')!r}"
            ),
        ),
        check(
            "rag_vector_candidate_k_matches_top_k",
            int(summary.get("normal_rag_vector_candidate_k", -1) or -1)
            == int(summary.get("rag_top_k", -2) or -2),
            (
                f"normal_rag_vector_candidate_k={summary.get('normal_rag_vector_candidate_k')} "
                f"rag_top_k={summary.get('rag_top_k')}"
            ),
        ),
        check(
            "rag_context_tokens_recorded",
            not probes_missing_numeric(summary, "normal_rag_context_tokens"),
            (
                "missing_probe_events="
                f"{probes_missing_numeric(summary, 'normal_rag_context_tokens')}"
            ),
        ),
        check(
            "rag_top_k_packet_rows_present",
            not rag_top_k_size_mismatches(summary),
            f"mismatches={rag_top_k_size_mismatches(summary)}",
        ),
        check(
            "rag_text_only_packet_rows",
            not non_text_probe_rows(summary, "rolling_history")
            and not non_text_probe_rows(summary, "rag_top_k"),
            (
                f"rolling_non_text={non_text_probe_rows(summary, 'rolling_history')} "
                f"rag_non_text={non_text_probe_rows(summary, 'rag_top_k')}"
            ),
        ),
        check(
            "rag_compaction_exercised",
            int(
                summary.get("normal_rag_compaction", {}).get(
                    "compaction_events", 0
                )
                or 0
            )
            > 0,
            (
                "normal_rag_compaction_events="
                f"{summary.get('normal_rag_compaction', {}).get('compaction_events')}"
            ),
        ),
        check(
            "release_probe_count_floor",
            int(summary.get("probe_count", 0) or 0) >= MIN_RELEASE_PROBES,
            f"probe_count={summary.get('probe_count')} min_required={MIN_RELEASE_PROBES}",
        ),
        check(
            "mixed_media_slice_present",
            int(summary.get("processed_text_messages", 0) or 0) > 0
            and int(summary.get("audio_processed", 0) or 0) > 0
            and int(summary.get("image_processed", 0) or 0) > 0,
            (
                f"text={summary.get('processed_text_messages')} "
                f"audio={summary.get('audio_processed')} "
                f"image={summary.get('image_processed')} "
                f"video={summary.get('video_processed')}"
            ),
        ),
        check(
            "source_ids_speaker_scoped_not_modality_scoped",
            "chat/user" in str(summary.get("source_id_policy", ""))
            and "chat/assistant" in str(summary.get("source_id_policy", ""))
            and "media is not encoded into source_id"
            in str(summary.get("source_id_policy", "")),
            f"source_id_policy={summary.get('source_id_policy')!r}",
        ),
        check(
            "timestamped_media_ingress_recorded",
            "media uses internal replay timestamped ingress"
            in str(summary.get("timeline_policy", ""))
            and "signal timestamps match source event timestamps"
            in str(summary.get("media_timestamp_policy", "")),
            (
                f"timeline_policy={summary.get('timeline_policy')!r} "
                f"media_timestamp_policy={summary.get('media_timestamp_policy')!r}"
            ),
        ),
        check(
            "native_cortext_only_behavior_recorded",
            "Live Cortext-only run" in str(summary.get("behavior_note", ""))
            and "no custom retrieval or scoring"
            in str(summary.get("behavior_note", "")),
            f"behavior_note={summary.get('behavior_note')!r}",
        ),
    ]
    knobs = summary.get("knobs")
    if isinstance(knobs, dict):
        out.append(
            check(
                "default_knobs_0_5",
                all(number_close(knobs.get(k), v) for k, v in DEFAULT_KNOBS.items()),
                f"knobs={knobs}",
            )
        )
    else:
        out.append(
            check(
                "default_knobs_0_5",
                False,
                "summary does not record knobs; rerun with current harness before release",
            )
        )

    if summary.get("daily_consolidation"):
        out.append(
            check(
                "daily_final_window_consolidated",
                bool(summary.get("daily_final_window_consolidated")),
                (
                    "daily_final_window_consolidated="
                    f"{summary.get('daily_final_window_consolidated')}"
                ),
            )
        )
    return out


def judge_protocol_checks(judge: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = judge.get("protocol", {})
    fairness = judge.get("fairness_checks", {})
    tokens = judge.get("tokens", {})
    systems = set(protocol.get("systems", []))
    fields = set(protocol.get("score_fields", []))
    repetitions = int(
        judge.get("judge_repetitions") or protocol.get("judge_repetitions") or 0
    )
    return [
        check(
            "judge_local_only",
            judge.get("remote_provider_allowed") is False
            and str(judge.get("judge_base_url", "")).startswith(
                ("http://127.0.0.1", "http://localhost", "http://0.0.0.0")
            ),
            f"provider={judge.get('judge_provider')} base_url={judge.get('judge_base_url')}",
        ),
        check(
            "packets_blinded",
            bool(protocol.get("packet_blinding")),
            f"packet_blinding={protocol.get('packet_blinding')}",
        ),
        check(
            "judge_packets_uncropped",
            int(tokens.get("context_limit_memories", 0) or 0) == -1,
            f"context_limit_memories={tokens.get('context_limit_memories')}",
        ),
        check(
            "judge_repetitions_at_least_3",
            repetitions >= MIN_JUDGE_REPETITIONS,
            f"judge_repetitions={repetitions} min_required={MIN_JUDGE_REPETITIONS}",
        ),
        check(
            "judge_repetitions_within_release_range",
            MIN_JUDGE_REPETITIONS <= repetitions <= 5,
            f"judge_repetitions={repetitions} required_range=3..5",
        ),
        check(
            "multimodal_judge_enabled",
            judge.get("multimodal_judge") is True,
            f"multimodal_judge={judge.get('multimodal_judge')}",
        ),
        check(
            "required_systems_present",
            all(system in systems for system in REQUIRED_SYSTEMS),
            f"systems={sorted(systems)}",
        ),
        check(
            "required_score_fields_present",
            all(field in fields for field in REQUIRED_FIELDS),
            f"score_fields={sorted(fields)}",
        ),
        check(
            "confidence_intervals_present",
            bool(judge.get("confidence_intervals")),
            "judge artifact includes probe-level confidence intervals",
        ),
        check(
            "required_confidence_interval_metrics_present",
            not missing_required_ci_metrics(judge),
            f"missing={missing_required_ci_metrics(judge)}",
        ),
        check(
            "rag_text_only",
            fairness.get("traditional_chat_rag_text_only") is True,
            f"traditional_chat_rag_text_only={fairness.get('traditional_chat_rag_text_only')}",
        ),
        check(
            "full_history_text_only",
            fairness.get("full_history_text_only") is True,
            f"full_history_text_only={fairness.get('full_history_text_only')}",
        ),
        check(
            "no_future_context",
            fairness.get("no_future_context_violations") is True,
            f"fairness_counters={fairness.get('counters', {})}",
        ),
        check(
            "no_current_turn_context_in_candidate_packets",
            fairness.get("no_current_turn_context_inclusions") is True,
            f"fairness_counters={fairness.get('counters', {})}",
        ),
        check(
            "blind_prompt_hidden_labels_absent",
            fairness.get("blind_prompt_hidden_labels_absent") is True,
            f"fairness_counters={fairness.get('counters', {})}",
        ),
        check(
            "no_cortext_media_transcript_shortcut",
            judge.get("cortext_audio_image_transcript_shortcuts") is False,
            (
                "cortext_audio_image_transcript_shortcuts="
                f"{judge.get('cortext_audio_image_transcript_shortcuts')}"
            ),
        ),
    ]


def cross_artifact_checks(
    summary: dict[str, Any], judge: dict[str, Any], summary_path: pathlib.Path
) -> list[dict[str, Any]]:
    judged_summary = pathlib.Path(str(judge.get("summary_path", "")))
    processed = judge.get("processed", {})
    return [
        check(
            "judge_summary_path_matches",
            judged_summary == summary_path,
            f"judge_summary_path={judged_summary} expected={summary_path}",
        ),
        check(
            "judge_probe_count_matches_summary",
            int_or_default(judge.get("probe_count"), -1)
            == int_or_default(summary.get("probe_count"), -2),
            f"judge_probe_count={judge.get('probe_count')} summary_probe_count={summary.get('probe_count')}",
        ),
        check(
            "judge_processed_counts_match_summary",
            int_or_default(processed.get("text"), -1)
            == int_or_default(summary.get("processed_text_messages"), -2)
            and int_or_default(processed.get("audio"), -1)
            == int_or_default(summary.get("audio_processed"), -2)
            and int_or_default(processed.get("image"), -1)
            == int_or_default(summary.get("image_processed"), -2)
            and int_or_default(processed.get("video"), -1)
            == int_or_default(summary.get("video_processed"), -2),
            (
                f"judge_processed={processed} summary_text={summary.get('processed_text_messages')} "
                f"summary_audio={summary.get('audio_processed')} "
                f"summary_image={summary.get('image_processed')} "
                f"summary_video={summary.get('video_processed')}"
            ),
        ),
        check(
            "judge_daily_consolidation_matches_summary",
            judge.get("daily_consolidation") == summary.get("daily_consolidation"),
            (
                f"judge_daily_consolidation={judge.get('daily_consolidation')} "
                f"summary_daily_consolidation={summary.get('daily_consolidation')}"
            ),
        ),
    ]


def human_label_checks(
    human: dict[str, Any] | None, human_path: pathlib.Path | None, summary_path: pathlib.Path
) -> list[dict[str, Any]]:
    if human is None or human_path is None:
        return [
            pending(
                "human_labels_present",
                "no --human-labels artifact supplied; release claim needs human agreement",
            )
        ]

    checks = [
        check(
            "human_labels_present",
            bool(human.get("tasks") or human.get("labels") or human.get("judgments") or human.get("agreement")),
            f"path={human_path}",
        )
    ]

    if human.get("schema") == "cortext_human_label_score_v1":
        agreement = human.get("agreement", {})
        sample_path = pathlib.Path(str(human.get("sample", "")))
        sample_matches = False
        if sample_path.exists():
            try:
                sample = load_json(sample_path)
                sample_matches = pathlib.Path(str(sample.get("source_summary", ""))) == summary_path
                sample_composition = sample.get("sample_composition", {})
            except Exception:
                sample_matches = False
                sample_composition = {}
        else:
            sample_composition = {}
        checks.extend(
            [
                check(
                    "human_labels_same_summary",
                    sample_matches,
                    f"sample={sample_path} summary={summary_path}",
                ),
                check(
                    "human_probe_count_floor",
                    int(human.get("probe_count", 0) or 0) >= MIN_HUMAN_PROBES,
                    f"probe_count={human.get('probe_count')} min_required={MIN_HUMAN_PROBES}",
                ),
                check(
                    "human_judge_shared_probe_floor",
                    int(agreement.get("shared_probe_events", 0) or 0)
                    >= MIN_HUMAN_SHARED_PROBES,
                    (
                        f"shared_probe_events={agreement.get('shared_probe_events')} "
                        f"min_required={MIN_HUMAN_SHARED_PROBES}"
                    ),
                ),
                check(
                    "human_judge_agreement_floor",
                    float(agreement.get("cohen_kappa_binary_target_membership", -1.0) or -1.0)
                    >= MIN_HUMAN_KAPPA,
                    (
                        "cohen_kappa_binary_target_membership="
                        f"{agreement.get('cohen_kappa_binary_target_membership')} "
                        f"min_required={MIN_HUMAN_KAPPA}"
                    ),
                ),
                check(
                    "human_sample_includes_cortext_active_packet_candidates",
                    sample_composition.get("includes_cortext_active_packet_candidates")
                    is True,
                    f"sample_composition={sample_composition}",
                ),
                check(
                    "human_sample_includes_media_candidates",
                    sample_composition.get("includes_media_candidates") is True,
                    f"sample_composition={sample_composition}",
                ),
            ]
        )
        return checks

    tasks = human.get("tasks", [])
    labeled_probe_count = 0
    if isinstance(tasks, list):
        for task in tasks:
            if task.get("labels"):
                labeled_probe_count += 1
    checks.append(
        check(
            "human_probe_count_floor",
            labeled_probe_count >= MIN_HUMAN_PROBES,
            f"labeled_probe_count={labeled_probe_count} min_required={MIN_HUMAN_PROBES}",
        )
    )
    checks.append(
        pending(
            "human_judge_agreement_floor",
            "raw human labels supplied, but no agreement report was supplied",
        )
    )
    return checks


def ablation_checks(ablation_names: list[str]) -> list[dict[str, Any]]:
    if not ablation_names:
        return [
            pending(
                "architecture_ablations_supplied",
                (
                    "no ablation artifacts supplied; release claim cannot say daily "
                    "consolidation/graph/media/temporal components earned their keep"
                ),
            )
        ]

    present = {
        category
        for name in ablation_names
        for category in [canonical_ablation_category(name)]
        if category is not None
    }
    missing = sorted(set(REQUIRED_ABLATION_CATEGORIES) - present)
    return [
        check(
            "architecture_ablations_supplied",
            not missing,
            (
                f"present={sorted(present)} missing={missing} "
                f"raw_names={ablation_names}"
            ),
        )
    ]


def ablation_plan_checks(
    ablation_names: list[str],
    ablation_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not ablation_names:
        return []
    if not ablation_plan:
        return [
            check(
                "architecture_ablation_plan_recorded",
                False,
                "ablation artifacts supplied without --ablation-plan provenance",
            )
        ]

    cases = ablation_plan.get("cases", [])
    if not isinstance(cases, list):
        cases = []
    by_name = {
        str(case.get("name")): case
        for case in cases
        if isinstance(case, dict) and case.get("name")
    }
    missing_names = sorted(set(ablation_names) - set(by_name))
    extra_names = sorted(set(by_name) - set(ablation_names))

    missing_env = []
    bad_daily = []
    missing_commands = []
    for name in ablation_names:
        case = by_name.get(name, {})
        category = canonical_ablation_category(name)
        env = case.get("env_overrides", {})
        if not isinstance(env, dict):
            env = {}
        if category == "no_daily_consolidation":
            if case.get("daily_consolidation") is not False:
                bad_daily.append(
                    {
                        "name": name,
                        "daily_consolidation": case.get("daily_consolidation"),
                    }
                )
        elif category is not None:
            if case.get("daily_consolidation") is not True:
                bad_daily.append(
                    {
                        "name": name,
                        "daily_consolidation": case.get("daily_consolidation"),
                    }
                )
            for env_name, env_value in REQUIRED_ABLATION_ENV_BY_CATEGORY.get(
                category, {}
            ).items():
                if str(env.get(env_name, "")) != env_value:
                    missing_env.append(
                        {
                            "name": name,
                            "category": category,
                            "expected": {env_name: env_value},
                            "observed_env_overrides": env,
                        }
                    )
        if not case.get("benchmark_command") or not case.get("judge_command"):
            missing_commands.append(name)

    return [
        check(
            "architecture_ablation_plan_recorded",
            True,
            f"path={ablation_plan.get('path', '')}",
        ),
        check(
            "architecture_ablation_plan_names_match_artifacts",
            not missing_names and not extra_names,
            f"missing_names={missing_names} extra_names={extra_names}",
        ),
        check(
            "architecture_ablation_plan_env_overrides_match_categories",
            not missing_env,
            f"missing_or_wrong_env={missing_env}",
        ),
        check(
            "architecture_ablation_plan_daily_modes_match_categories",
            not bad_daily,
            f"bad_daily_modes={bad_daily}",
        ),
        check(
            "architecture_ablation_commands_recorded",
            not missing_commands,
            f"missing_commands={missing_commands}",
        ),
    ]


def ablation_artifact_checks(
    ablations: list[dict[str, Any]],
    main_schedule_sha256: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not ablations:
        return []

    manifest_mismatches = [
        row["name"]
        for row in ablations
        if row.get("frozen_probe_schedule_sha256") != main_schedule_sha256
    ]
    non_default_knobs = [
        row["name"]
        for row in ablations
        if not isinstance(row.get("knobs"), dict)
        or not all(number_close(row["knobs"].get(k), v) for k, v in DEFAULT_KNOBS.items())
    ]
    weak_judges = []
    for row in ablations:
        judge = row.get("judge_protocol", {})
        repetitions = int(judge.get("judge_repetitions") or 0)
        if (
            row.get("judge_provider") != "local_ollama"
            or row.get("remote_provider_allowed") is not False
            or not judge.get("packet_blinding")
            or repetitions < MIN_JUDGE_REPETITIONS
        ):
            weak_judges.append(row["name"])
    main_processed = {
        "text": summary.get("processed_text_messages"),
        "media_attempted": summary.get("media_attempted"),
        "media": summary.get("media_processed"),
        "audio": summary.get("audio_processed"),
        "image": summary.get("image_processed"),
        "video": summary.get("video_processed"),
    }
    processed_mismatches = [
        row["name"]
        for row in ablations
        if row.get("processed") != main_processed
    ]
    too_few_shared_probes = []
    ablation_beats_full = []
    unsupported_positive_deltas = []
    for row in ablations:
        effect = row.get("effect_vs_full", {})
        quality_delta = effect.get("quality_delta_full_minus_ablation", {})
        shared_quality_probe_count = int(effect.get("shared_quality_probe_count", 0) or 0)
        ci95 = quality_delta.get("ci95", [0.0, 0.0])
        mean_delta = float(quality_delta.get("mean", 0.0) or 0.0)
        lower = float(ci95[0]) if isinstance(ci95, list) and ci95 else 0.0
        if shared_quality_probe_count < MIN_RELEASE_PROBES:
            too_few_shared_probes.append(
                {
                    "name": row["name"],
                    "shared_quality_probe_count": shared_quality_probe_count,
                }
            )
        if mean_delta < 0.0:
            ablation_beats_full.append(
                {
                    "name": row["name"],
                    "mean_quality_delta_full_minus_ablation": mean_delta,
                    "ci95": ci95,
                }
            )
        if lower <= 0.0:
            unsupported_positive_deltas.append(
                {
                    "name": row["name"],
                    "mean_quality_delta_full_minus_ablation": mean_delta,
                    "ci95": ci95,
                }
            )

    return [
        check(
            "architecture_ablations_share_frozen_probe_schedule",
            not manifest_mismatches,
            f"mismatched={manifest_mismatches}",
        ),
        check(
            "architecture_ablations_default_knobs",
            not non_default_knobs,
            f"non_default_or_missing={non_default_knobs}",
        ),
        check(
            "architecture_ablation_judges_release_protocol",
            not weak_judges,
            f"weak_or_mismatched_judges={weak_judges}",
        ),
        check(
            "architecture_ablations_share_processed_event_counts",
            not processed_mismatches,
            f"main={main_processed} mismatched={processed_mismatches}",
        ),
        check(
            "architecture_ablations_share_judged_probe_floor",
            not too_few_shared_probes,
            f"too_few_shared_probes={too_few_shared_probes}",
        ),
        check(
            "full_cortext_not_worse_than_ablations",
            not ablation_beats_full,
            f"ablation_beats_full={ablation_beats_full}",
        ),
        check(
            "full_cortext_ablation_value_supported_by_ci",
            not unsupported_positive_deltas,
            (
                "requires paired bootstrap ci95 lower bound > 0 for "
                f"quality composite; unsupported={unsupported_positive_deltas}"
            ),
        ),
    ]


def probe_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    probes = []
    for i, probe in enumerate(summary.get("probes", [])):
        query = probe.get("query", {})
        probes.append(
            {
                "probe_ordinal": i,
                "event_index": probe.get("event_index"),
                "query": {
                    "timestamp": query.get("timestamp"),
                    "source_id": query.get("source_id"),
                    "modality": query.get("modality"),
                    "tokens": query.get("tokens"),
                },
                "rag_top_k_indices": probe.get("rag_top_k_indices", []),
                "rolling_history_indices": [
                    row.get("index") for row in probe.get("rolling_history", [])
                ],
                "cortext_working_memory_ids": probe.get("cortext_working_memory_ids", []),
                "cortext_retrieved_memory_ids": probe.get("cortext_retrieved_memory_ids", []),
                "full_history_items": probe.get("full_history_items"),
                "full_history_tokens": probe.get("full_history_tokens"),
                "rolling_history_tokens": probe.get(
                    "normal_rag_active_history_tokens",
                    probe.get("rolling_history_tokens"),
                ),
                "normal_rag_context_tokens": probe.get("normal_rag_context_tokens"),
            }
        )
    body = {
        "schema": "cortext_julie_frozen_probe_manifest_v1",
        "privacy": "no message text or source blob bytes",
        "probe_count": len(probes),
        "probes": probes,
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body


def probe_schedule_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    probes = []
    for i, probe in enumerate(summary.get("probes", [])):
        query = probe.get("query", {})
        probes.append(
            {
                "probe_ordinal": i,
                "event_index": probe.get("event_index"),
                "query": {
                    "timestamp": query.get("timestamp"),
                    "source_id": query.get("source_id"),
                    "modality": query.get("modality"),
                    "tokens": query.get("tokens"),
                },
                "rag_top_k_indices": probe.get("rag_top_k_indices", []),
                "rolling_history_indices": [
                    row.get("index") for row in probe.get("rolling_history", [])
                ],
                "full_history_items": probe.get("full_history_items"),
                "full_history_tokens": probe.get("full_history_tokens"),
                "rolling_history_tokens": probe.get(
                    "normal_rag_active_history_tokens",
                    probe.get("rolling_history_tokens"),
                ),
                "normal_rag_context_tokens": probe.get("normal_rag_context_tokens"),
            }
        )
    body = {
        "schema": "cortext_julie_frozen_probe_schedule_v1",
        "privacy": "no message text, source blob bytes, or Cortext result IDs",
        "probe_count": len(probes),
        "warmup_events": summary.get("warmup_events"),
        "probe_stride": summary.get("probe_stride"),
        "rag_top_k": summary.get("rag_top_k"),
        "active_history_token_budget": summary.get("active_history_token_budget"),
        "probes": probes,
    }
    body["schedule_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body


def public_judge_summary(judge: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": judge.get("schema"),
        "judge_provider": judge.get("judge_provider"),
        "judge_model": judge.get("judge_model"),
        "judge_repetitions": judge.get("judge_repetitions"),
        "probe_count": judge.get("probe_count"),
        "judged": judge.get("judged"),
        "protocol": judge.get("protocol"),
        "quality": judge.get("quality"),
        "confidence_intervals": judge.get("confidence_intervals"),
        "failure_reasons": judge.get("failure_reasons"),
        "judge_validation": judge.get("judge_validation"),
        "fairness_checks": judge.get("fairness_checks"),
        "tokens": judge.get("tokens"),
        "latency": judge.get("latency"),
        "media_attachments": judge.get("media_attachments"),
    }


def cost_summary(summary: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    wall_ms = float(summary.get("wall_ms", 0.0) or 0.0)
    consolidation_ms = float(summary.get("consolidation_ms_total", 0.0) or 0.0)
    wall_ms_excluding = float(summary.get("wall_ms_excluding_consolidation", 0.0) or 0.0)
    processed = int(summary.get("processed_text_messages", 0) or 0) + int(
        summary.get("media_processed", 0) or 0
    )
    consolidation_events = summary.get("consolidation_events")
    consolidation_ms_by_day = None
    if isinstance(consolidation_events, list):
        by_day: dict[str, float] = {}
        for event in consolidation_events:
            if not isinstance(event, dict):
                continue
            day = str(event.get("local_day_bucket", "unknown"))
            try:
                elapsed = float(event.get("elapsed_ms", 0.0) or 0.0)
            except Exception:
                elapsed = 0.0
            by_day[day] = by_day.get(day, 0.0) + elapsed
        consolidation_ms_by_day = by_day
    cortext_probe_latencies = numeric_probe_values(summary, "cortext_latency_ms")
    rag_total_latencies = numeric_probe_values(summary, "normal_rag_total_latency_ms")
    rag_retrieval_latencies = numeric_probe_values(summary, "normal_rag_retrieval_latency_ms")
    cortext_tokens = numeric_probe_values(summary, "cortext_context_tokens")
    rag_tokens = numeric_probe_values(summary, "normal_rag_context_tokens")
    if not rag_tokens:
        rag_tokens = numeric_probe_values(summary, "normal_rag_active_history_tokens")
    full_history_tokens = numeric_probe_values(summary, "full_history_tokens")
    db_path = pathlib.Path(str(summary.get("db_path", "")))
    return {
        "processed_events_estimate": processed,
        "wall_ms": wall_ms,
        "wall_ms_excluding_consolidation": summary.get("wall_ms_excluding_consolidation"),
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
        "mean_cortext_probe_latency_ms": judge.get("latency", {}).get(
            "mean_cortext_probe_latency_ms"
        ),
        "p50_cortext_probe_latency_ms": percentile(cortext_probe_latencies, 0.50),
        "p95_cortext_probe_latency_ms": percentile(cortext_probe_latencies, 0.95),
        "p50_rag_total_latency_ms": percentile(rag_total_latencies, 0.50),
        "p95_rag_total_latency_ms": percentile(rag_total_latencies, 0.95),
        "p50_rag_retrieval_latency_ms": percentile(rag_retrieval_latencies, 0.50),
        "p95_rag_retrieval_latency_ms": percentile(rag_retrieval_latencies, 0.95),
        "mean_cortext_prompt_tokens": (
            sum(cortext_tokens) / len(cortext_tokens) if cortext_tokens else None
        ),
        "mean_rag_prompt_tokens": sum(rag_tokens) / len(rag_tokens) if rag_tokens else None,
        "mean_full_history_prompt_tokens": (
            sum(full_history_tokens) / len(full_history_tokens)
            if full_history_tokens
            else None
        ),
        "db_disk_bytes": path_size_bytes(db_path) if str(db_path) else None,
        "db_disk_bytes_per_event": (
            path_size_bytes(db_path) / processed
            if str(db_path) and processed > 0
            else None
        ),
        "peak_rss_mb": summary.get("peak_rss_mb"),
        "events_per_second_excluding_consolidation": (
            processed / (wall_ms_excluding / 1000.0)
            if wall_ms_excluding > 0
            else 0.0
        ),
        "events_per_second_including_consolidation": (
            processed / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
        ),
    }


def cost_checks(costs: dict[str, Any]) -> list[dict[str, Any]]:
    required_present = [
        "wall_ms_excluding_consolidation",
        "consolidation_ms_total",
        "consolidation_ms_by_day",
        "mean_consolidation_ms_per_run",
        "p50_cortext_probe_latency_ms",
        "p95_cortext_probe_latency_ms",
        "mean_cortext_prompt_tokens",
        "mean_rag_prompt_tokens",
        "db_disk_bytes",
        "db_disk_bytes_per_event",
        "events_per_second_excluding_consolidation",
    ]
    missing = [key for key in required_present if costs.get(key) is None]
    checks = [
        check(
            "production_cost_metrics_present",
            not missing,
            f"missing={missing}",
        )
    ]
    checks.append(
        check(
            "consolidation_wall_time_by_day_present",
            bool(costs.get("consolidation_ms_by_day"))
            and int(costs.get("consolidation_event_count") or 0) > 0,
            (
                f"consolidation_event_count={costs.get('consolidation_event_count')} "
                f"consolidation_ms_by_day={costs.get('consolidation_ms_by_day')}"
            ),
        )
    )
    checks.append(
        check(
            "ingest_throughput_excluding_consolidation_present",
            float(costs.get("events_per_second_excluding_consolidation") or 0.0)
            > 0.0,
            (
                "events_per_second_excluding_consolidation="
                f"{costs.get('events_per_second_excluding_consolidation')}"
            ),
        )
    )
    if costs.get("peak_rss_mb") is None:
        checks.append(
            pending(
                "memory_use_recorded",
                "summary does not record peak_rss_mb; release cost story needs memory use",
            )
        )
    else:
        checks.append(
            check(
                "memory_use_recorded",
                True,
                f"peak_rss_mb={costs.get('peak_rss_mb')}",
            )
        )
    return checks


def parse_ablation(value: str) -> tuple[str, pathlib.Path, pathlib.Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--ablation must use name:summary_path:judge_path"
        )
    return parts[0], pathlib.Path(parts[1]), pathlib.Path(parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--judge", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-command", default="")
    parser.add_argument("--judge-command", default="")
    parser.add_argument("--human-labels", type=pathlib.Path)
    parser.add_argument("--ablation", type=parse_ablation, action="append", default=[])
    parser.add_argument("--ablation-plan", type=pathlib.Path)
    args = parser.parse_args()

    summary = load_json(args.summary)
    judge = load_json(args.judge)
    ablation_plan = load_json(args.ablation_plan) if args.ablation_plan else None
    if ablation_plan is not None:
        ablation_plan["path"] = str(args.ablation_plan)
    manifest = probe_manifest(summary)
    schedule_manifest = probe_schedule_manifest(summary)
    costs = cost_summary(summary, judge)
    claim_support = claim_support_summary(judge)
    ablation_names = [name for name, _, _ in args.ablation]

    checks: list[dict[str, Any]] = []
    checks.extend(command_protocol_checks(args.benchmark_command, args.judge_command))
    checks.extend(summary_protocol_checks(summary))
    checks.extend(judge_protocol_checks(judge))
    checks.extend(claim_support_checks(claim_support))
    checks.extend(cross_artifact_checks(summary, judge, args.summary))
    human = load_json(args.human_labels) if args.human_labels else None
    checks.extend(human_label_checks(human, args.human_labels, args.summary))
    checks.extend(ablation_checks(ablation_names))
    checks.extend(ablation_plan_checks(ablation_names, ablation_plan))
    checks.extend(cost_checks(costs))

    ablations = []
    for name, summary_path, judge_path in args.ablation:
        ablation_summary = load_json(summary_path)
        ablation_judge = load_json(judge_path)
        ablation_manifest = probe_manifest(ablation_summary)
        ablation_schedule_manifest = probe_schedule_manifest(ablation_summary)
        ablations.append(
            {
                "name": name,
                "summary_path": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
                "judge_path": str(judge_path),
                "judge_sha256": file_sha256(judge_path),
                "frozen_probe_manifest_sha256": ablation_manifest["manifest_sha256"],
                "frozen_probe_schedule_sha256": ablation_schedule_manifest[
                    "schedule_sha256"
                ],
                "knobs": ablation_summary.get("knobs"),
                "judge_provider": ablation_judge.get("judge_provider"),
                "remote_provider_allowed": ablation_judge.get("remote_provider_allowed"),
                "judge_protocol": ablation_judge.get("protocol", {}),
                "quality": ablation_judge.get("quality"),
                "tokens": ablation_judge.get("tokens"),
                "fairness_checks": ablation_judge.get("fairness_checks"),
                "effect_vs_full": paired_delta_summary(
                    judge,
                    ablation_judge,
                    "cortext_native",
                    name,
                ),
                "processed": {
                    "text": ablation_summary.get("processed_text_messages"),
                    "media_attempted": ablation_summary.get("media_attempted"),
                    "media": ablation_summary.get("media_processed"),
                    "audio": ablation_summary.get("audio_processed"),
                    "image": ablation_summary.get("image_processed"),
                    "video": ablation_summary.get("video_processed"),
                },
            }
        )

    checks.extend(
        ablation_artifact_checks(ablations, schedule_manifest["schedule_sha256"], summary)
    )

    status_counts: dict[str, int] = {}
    for item in checks:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    output = {
        "schema": "cortext_julie_release_protocol_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy": {
            "public_safe": True,
            "excludes_private_text": True,
            "excludes_judge_reason_strings": True,
            "private_artifacts_remain_local": True,
        },
        "git": git_info(),
        "commands": {
            "report": " ".join(sys.argv),
            "benchmark": args.benchmark_command,
            "judge": args.judge_command,
        },
        "artifacts": {
            "summary_path": str(args.summary),
            "summary_sha256": file_sha256(args.summary),
            "judge_path": str(args.judge),
            "judge_sha256": file_sha256(args.judge),
            "human_labels_path": str(args.human_labels) if args.human_labels else "",
            "ablation_plan_path": str(args.ablation_plan) if args.ablation_plan else "",
        },
        "source_run": {
            "input_dir_recorded": summary.get("input_dir"),
            "db_path_recorded": summary.get("db_path"),
            "processed_text_messages": summary.get("processed_text_messages"),
            "media_attempted": summary.get("media_attempted"),
            "media_processed": summary.get("media_processed"),
            "audio_processed": summary.get("audio_processed"),
            "image_processed": summary.get("image_processed"),
            "video_processed": summary.get("video_processed"),
            "probe_stride": summary.get("probe_stride"),
            "warmup_events": summary.get("warmup_events"),
            "rag_top_k": summary.get("rag_top_k"),
            "normal_rag_retrieval": summary.get("normal_rag_retrieval"),
            "normal_rag_baseline_modality": summary.get(
                "normal_rag_baseline_modality"
            ),
            "normal_rag_vector_query_encoder": summary.get(
                "normal_rag_vector_query_encoder"
            ),
            "normal_rag_vector_candidate_k": summary.get(
                "normal_rag_vector_candidate_k"
            ),
            "normal_rag_compaction": summary.get("normal_rag_compaction"),
            "active_history_token_budget": summary.get("active_history_token_budget"),
            "knobs": summary.get("knobs"),
            "daily_consolidation": summary.get("daily_consolidation"),
            "final_window_consolidated": summary.get("final_window_consolidated"),
            "daily_final_window_consolidated": summary.get(
                "daily_final_window_consolidated"
            ),
            "deep_consolidation": summary.get("deep_consolidation"),
        },
        "frozen_probe_manifest": manifest,
        "frozen_probe_schedule": schedule_manifest,
        "judge": public_judge_summary(judge),
        "claim_support": claim_support,
        "costs": costs,
        "ablation_plan": ablation_plan,
        "ablations": ablations,
        "release_gate": {
            "overall_status": (
                "pass"
                if status_counts.get("fail", 0) == 0 and status_counts.get("pending", 0) == 0
                else "not_ready"
            ),
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(args.out)
    print(
        "overall_status="
        f"{output['release_gate']['overall_status']} "
        f"checks={status_counts} manifest_sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
