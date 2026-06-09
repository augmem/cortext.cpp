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
import shutil
import shlex
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


REQUIRED_SYSTEMS = [
    "cortext_native",
    "traditional_chat_rag",
    "full_history_upper_bound",
]
REQUIRED_FIELDS = ["relevance", "sufficiency", "noise"]
DEFAULT_KNOBS = {"focus": 0.5, "sensitivity": 0.5, "stability": 0.5}
MIN_RELEASE_PROBES = 30
MIN_JUDGE_REPETITIONS = 3
MIN_HUMAN_PROBES = 30
MIN_HUMAN_SHARED_PROBES = 30
MIN_HUMAN_KAPPA = 0.4
ABLATION_BOOTSTRAP_SAMPLES = 2000
MIN_TOKEN_SAVINGS_CI95_LOWER = 0.5
BENCHMARK_ENV_SNAPSHOT_NAME = "benchmark_environment_snapshot.json"
MIN_JUDGE_MAJORITY_EVENT_RATE = 0.9
MIN_JUDGE_MEAN_MAJORITY_FRACTION = 2.0 / 3.0
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_value(args: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""


def git_hash_output(args: list[str]) -> dict[str, Any]:
    text = git_output(args)
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text.encode("utf-8")),
        "line_count": len(text.splitlines()),
    }


def git_info() -> dict[str, Any]:
    status = git_output(["status", "--porcelain=v1"])
    status_entries = sorted(
        line[3:] if len(line) > 3 else line for line in status.splitlines()
    )
    untracked = sorted(
        line for line in git_output(["ls-files", "--others", "--exclude-standard"]).splitlines()
        if line
    )
    staged = git_hash_output(["diff", "--cached", "--binary", "--no-ext-diff"])
    unstaged = git_hash_output(["diff", "--binary", "--no-ext-diff"])
    submodule = git_hash_output(["diff", "--submodule=short", "--no-ext-diff"])
    fingerprint_payload = {
        "commit": git_value(["rev-parse", "HEAD"], "unknown"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_entries": status_entries,
        "staged_diff_sha256": staged["sha256"],
        "unstaged_diff_sha256": unstaged["sha256"],
        "submodule_diff_sha256": submodule["sha256"],
        "untracked_paths_sha256": hashlib.sha256(
            "\n".join(untracked).encode("utf-8")
        ).hexdigest(),
    }
    return {
        "commit": git_value(["rev-parse", "HEAD"], "unknown"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_entry_count": len(status_entries),
        "status_paths": status_entries,
        "staged_diff_sha256": staged["sha256"],
        "staged_diff_bytes": staged["bytes"],
        "unstaged_diff_sha256": unstaged["sha256"],
        "unstaged_diff_bytes": unstaged["bytes"],
        "submodule_diff_sha256": submodule["sha256"],
        "submodule_diff_bytes": submodule["bytes"],
        "untracked_path_count": len(untracked),
        "untracked_paths": untracked,
        "untracked_paths_sha256": fingerprint_payload["untracked_paths_sha256"],
        "worktree_manifest_sha256": hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dirty_reproducibility_policy": (
            "dirty releases are allowed only when status paths plus staged, "
            "unstaged, submodule, and untracked-path hashes are recorded"
        ),
    }


def git_provenance_checks(info: dict[str, Any]) -> list[dict[str, Any]]:
    commit = str(info.get("commit", ""))
    return [
        check(
            "git_commit_recorded",
            len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit),
            f"commit={commit!r}",
        ),
        check(
            "git_dirty_flag_recorded",
            isinstance(info.get("dirty"), bool)
            and bool(info.get("status_sha256")),
            (
                f"dirty={info.get('dirty')!r} "
                f"status_sha256={info.get('status_sha256')!r}"
            ),
        ),
        check(
            "git_dirty_worktree_reproducibility_recorded",
            (
                info.get("dirty") is False
                or (
                    bool(info.get("worktree_manifest_sha256"))
                    and bool(info.get("status_sha256"))
                    and isinstance(info.get("status_paths"), list)
                    and info.get("staged_diff_sha256")
                    and info.get("unstaged_diff_sha256")
                    and info.get("submodule_diff_sha256")
                    and info.get("untracked_paths_sha256")
                )
            ),
            (
                f"dirty={info.get('dirty')!r} "
                f"status_entry_count={info.get('status_entry_count')} "
                f"staged_diff_bytes={info.get('staged_diff_bytes')} "
                f"unstaged_diff_bytes={info.get('unstaged_diff_bytes')} "
                f"submodule_diff_bytes={info.get('submodule_diff_bytes')} "
                f"untracked_path_count={info.get('untracked_path_count')} "
                f"worktree_manifest_sha256={info.get('worktree_manifest_sha256')!r}"
            ),
        ),
    ]


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


def float_or_default(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def is_loopback_url(value: Any) -> bool:
    try:
        parsed = urlparse(str(value))
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "0.0.0.0",
        "::1",
    }


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
        and value.get("n") >= MIN_RELEASE_PROBES
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


def packet_randomization_summary(judge: dict[str, Any]) -> dict[str, Any]:
    systems = list(judge.get("protocol", {}).get("systems", REQUIRED_SYSTEMS))
    mappings: list[dict[str, str]] = []
    labels_by_system: dict[str, set[str]] = {system: set() for system in systems}
    rows_missing_mapping = 0
    for row in judge.get("judgments", []):
        if not isinstance(row, dict):
            continue
        packet_blinding = row.get("packet_blinding", {})
        if not isinstance(packet_blinding, dict):
            rows_missing_mapping += 1
            continue
        real_to_label = packet_blinding.get("real_to_label", {})
        if not isinstance(real_to_label, dict):
            rows_missing_mapping += 1
            continue
        mapping: dict[str, str] = {}
        for system in systems:
            label = real_to_label.get(system)
            if not isinstance(label, str) or not label:
                continue
            mapping[system] = label
            labels_by_system.setdefault(system, set()).add(label)
        if len(mapping) != len(systems):
            rows_missing_mapping += 1
            continue
        mappings.append(mapping)

    unique_mappings = {
        tuple((system, mapping.get(system, "")) for system in systems)
        for mapping in mappings
    }
    return {
        "judgment_rows": len(judge.get("judgments", [])),
        "mapped_rows": len(mappings),
        "rows_missing_mapping": rows_missing_mapping,
        "unique_mapping_count": len(unique_mappings),
        "labels_seen_by_system": {
            system: sorted(labels) for system, labels in labels_by_system.items()
        },
        "min_labels_per_system": (
            min((len(labels) for labels in labels_by_system.values()), default=0)
        ),
    }


def judge_repetition_consistency_summary(judge: dict[str, Any]) -> dict[str, Any]:
    protocol = judge.get("protocol", {})
    expected_repetitions = int(
        judge.get("judge_repetitions") or protocol.get("judge_repetitions") or 0
    )
    by_event: dict[int, list[str]] = {}
    for row in judge.get("judgments", []):
        if not isinstance(row, dict):
            continue
        try:
            event_index = int(row["event_index"])
        except Exception:
            continue
        winner = str(row.get("winner", "tie_or_unclear"))
        by_event.setdefault(event_index, []).append(winner)

    majority_fractions: list[float] = []
    majority_counts: list[int] = []
    unanimous_event_count = 0
    split_disagreement_event_count = 0
    events_with_expected_repetitions = 0
    majority_winners: dict[str, int] = {}
    per_event: list[dict[str, Any]] = []
    for event_index in sorted(by_event):
        winners = by_event[event_index]
        counts: dict[str, int] = {}
        for winner in winners:
            counts[winner] = counts.get(winner, 0) + 1
        majority_winner, majority_count = max(
            sorted(counts.items()), key=lambda item: item[1]
        )
        majority_fraction = majority_count / float(len(winners)) if winners else 0.0
        majority_counts.append(majority_count)
        majority_fractions.append(majority_fraction)
        majority_winners[majority_winner] = majority_winners.get(majority_winner, 0) + 1
        if majority_count == len(winners):
            unanimous_event_count += 1
        if expected_repetitions > 0 and len(winners) == expected_repetitions:
            events_with_expected_repetitions += 1
        if expected_repetitions > 0 and majority_count < (expected_repetitions // 2 + 1):
            split_disagreement_event_count += 1
        per_event.append(
            {
                "event_index": event_index,
                "repetitions": len(winners),
                "winner_counts": counts,
                "majority_winner": majority_winner,
                "majority_count": majority_count,
                "majority_fraction": majority_fraction,
            }
        )

    return {
        "expected_repetitions": expected_repetitions,
        "probe_count": int(judge.get("probe_count", 0) or 0),
        "events_with_rows": len(by_event),
        "events_with_expected_repetitions": events_with_expected_repetitions,
        "majority_event_count": len(by_event) - split_disagreement_event_count,
        "majority_event_rate": (
            (len(by_event) - split_disagreement_event_count) / float(len(by_event))
            if by_event
            else 0.0
        ),
        "min_majority_count": min(majority_counts, default=0),
        "mean_majority_fraction": mean(majority_fractions),
        "min_majority_fraction": min(majority_fractions, default=0.0),
        "unanimous_event_count": unanimous_event_count,
        "split_disagreement_event_count": split_disagreement_event_count,
        "majority_winners": majority_winners,
        "per_event": per_event,
    }


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
            and command_int_value(bench, "--media-limit") is not None
            and (
                command_int_value(bench, "--skip-messages") is None
                or command_int_value(bench, "--skip-messages") == 0
            ),
            (
                f"input_dir={command_flag_value(bench, '--input-dir')} "
                f"max_messages={command_flag_value(bench, '--max-messages')} "
                f"media_limit={command_flag_value(bench, '--media-limit')} "
                f"skip_messages={command_flag_value(bench, '--skip-messages')}"
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


def paths_equivalent(left: Any, right: Any) -> bool:
    left_text = str(left or "")
    right_text = str(right or "")
    if not left_text or not right_text:
        return False
    try:
        return pathlib.Path(left_text).resolve() == pathlib.Path(right_text).resolve()
    except OSError:
        return left_text == right_text


def command_summary_consistency_checks(
    benchmark_command: str,
    summary: dict[str, Any],
    summary_path: pathlib.Path,
) -> list[dict[str, Any]]:
    bench = split_command(benchmark_command)
    knobs = summary.get("knobs", {})
    if not isinstance(knobs, dict):
        knobs = {}
    return [
        check(
            "benchmark_command_matches_summary_input_dir",
            paths_equivalent(command_flag_value(bench, "--input-dir"), summary.get("input_dir")),
            (
                f"command_input_dir={command_flag_value(bench, '--input-dir')} "
                f"summary_input_dir={summary.get('input_dir')}"
            ),
        ),
        check(
            "benchmark_command_matches_summary_output_paths",
            paths_equivalent(command_flag_value(bench, "--db"), summary.get("db_path"))
            and paths_equivalent(command_flag_value(bench, "--out"), summary_path),
            (
                f"command_db={command_flag_value(bench, '--db')} "
                f"summary_db={summary.get('db_path')} "
                f"command_out={command_flag_value(bench, '--out')} "
                f"summary_path={summary_path}"
            ),
        ),
        check(
            "benchmark_command_matches_summary_slice",
            command_int_value(bench, "--max-messages")
            == int_or_default(summary.get("processed_text_messages"), -1)
            and command_int_value(bench, "--media-limit")
            == int_or_default(summary.get("media_attempted"), -2)
            and (
                command_int_value(bench, "--skip-messages") is None
                or command_int_value(bench, "--skip-messages")
                == int_or_default(summary.get("skipped_transcript_messages"), -3)
            ),
            (
                f"command_max_messages={command_flag_value(bench, '--max-messages')} "
                f"processed_text_messages={summary.get('processed_text_messages')} "
                f"command_media_limit={command_flag_value(bench, '--media-limit')} "
                f"media_attempted={summary.get('media_attempted')} "
                f"command_skip={command_flag_value(bench, '--skip-messages')} "
                f"summary_skipped={summary.get('skipped_transcript_messages')}"
            ),
        ),
        check(
            "benchmark_command_matches_summary_probe_schedule",
            command_int_value(bench, "--probe-stride")
            == int_or_default(summary.get("probe_stride"), -1)
            and command_int_value(bench, "--warmup-events")
            == int_or_default(summary.get("warmup_events"), -2),
            (
                f"command_probe_stride={command_flag_value(bench, '--probe-stride')} "
                f"summary_probe_stride={summary.get('probe_stride')} "
                f"command_warmup_events={command_flag_value(bench, '--warmup-events')} "
                f"summary_warmup_events={summary.get('warmup_events')}"
            ),
        ),
        check(
            "benchmark_command_matches_summary_rag_baseline",
            command_int_value(bench, "--rag-top-k")
            == int_or_default(summary.get("rag_top_k"), -1)
            and command_int_value(bench, "--active-history-token-budget")
            == int_or_default(summary.get("active_history_token_budget"), -2),
            (
                f"command_rag_top_k={command_flag_value(bench, '--rag-top-k')} "
                f"summary_rag_top_k={summary.get('rag_top_k')} "
                "command_active_history_token_budget="
                f"{command_flag_value(bench, '--active-history-token-budget')} "
                f"summary_active_history_token_budget={summary.get('active_history_token_budget')}"
            ),
        ),
        check(
            "benchmark_command_matches_summary_knobs",
            all(
                number_close(command_float_value(bench, f"--{key}"), value)
                and number_close(knobs.get(key), value)
                for key, value in DEFAULT_KNOBS.items()
            ),
            (
                f"command_focus={command_flag_value(bench, '--focus')} "
                f"command_sensitivity={command_flag_value(bench, '--sensitivity')} "
                f"command_stability={command_flag_value(bench, '--stability')} "
                f"summary_knobs={knobs}"
            ),
        ),
        check(
            "benchmark_command_matches_summary_consolidation_mode",
            command_has_flag(bench, "--daily-consolidation")
            == bool(summary.get("daily_consolidation"))
            and command_has_flag(bench, "--deep") == bool(summary.get("deep_consolidation")),
            (
                f"command_daily={command_has_flag(bench, '--daily-consolidation')} "
                f"summary_daily={summary.get('daily_consolidation')} "
                f"command_deep={command_has_flag(bench, '--deep')} "
                f"summary_deep={summary.get('deep_consolidation')}"
            ),
        ),
    ]


def command_judge_consistency_checks(
    judge_command: str,
    judge: dict[str, Any],
    summary_path: pathlib.Path,
    judge_path: pathlib.Path,
) -> list[dict[str, Any]]:
    command = split_command(judge_command)
    protocol = judge.get("protocol", {})
    tokens = judge.get("tokens", {})
    media = judge.get("media_attachments", {})
    expected_provider = (
        "local_ollama"
        if command_flag_value(command, "--judge-provider") == "ollama"
        else command_flag_value(command, "--judge-provider")
    )
    return [
        check(
            "judge_command_matches_artifact_paths",
            paths_equivalent(command_flag_value(command, "--summary"), summary_path)
            and paths_equivalent(command_flag_value(command, "--db"), judge.get("db_path"))
            and paths_equivalent(command_flag_value(command, "--out"), judge_path),
            (
                f"command_summary={command_flag_value(command, '--summary')} "
                f"summary_path={summary_path} "
                f"command_db={command_flag_value(command, '--db')} "
                f"judge_db={judge.get('db_path')} "
                f"command_out={command_flag_value(command, '--out')} "
                f"judge_path={judge_path}"
            ),
        ),
        check(
            "judge_command_matches_artifact_provider",
            expected_provider == judge.get("judge_provider")
            and command_flag_value(command, "--model") == judge.get("judge_model")
            and paths_equivalent(
                command_flag_value(command, "--ollama-base-url"),
                judge.get("judge_base_url"),
            ),
            (
                f"command_provider={command_flag_value(command, '--judge-provider')} "
                f"artifact_provider={judge.get('judge_provider')} "
                f"command_model={command_flag_value(command, '--model')} "
                f"artifact_model={judge.get('judge_model')} "
                f"command_base_url={command_flag_value(command, '--ollama-base-url')} "
                f"artifact_base_url={judge.get('judge_base_url')}"
            ),
        ),
        check(
            "judge_command_matches_artifact_protocol",
            command_has_flag(command, "--blind-packets") == bool(
                protocol.get("packet_blinding")
            )
            and command_int_value(command, "--judge-repetitions")
            == int_or_default(protocol.get("judge_repetitions"), -1)
            and command_int_value(command, "--judge-seed")
            == int_or_default(protocol.get("judge_seed"), -2)
            and command_int_value(command, "--bootstrap-samples")
            == int_or_default(protocol.get("bootstrap_samples"), -3),
            (
                f"command_blind={command_has_flag(command, '--blind-packets')} "
                f"artifact_blind={protocol.get('packet_blinding')} "
                f"command_repetitions={command_flag_value(command, '--judge-repetitions')} "
                f"artifact_repetitions={protocol.get('judge_repetitions')} "
                f"command_seed={command_flag_value(command, '--judge-seed')} "
                f"artifact_seed={protocol.get('judge_seed')} "
                f"command_bootstrap={command_flag_value(command, '--bootstrap-samples')} "
                f"artifact_bootstrap={protocol.get('bootstrap_samples')}"
            ),
        ),
        check(
            "judge_command_matches_artifact_context_and_media",
            (
                command_int_value(command, "--context-limit")
                if command_int_value(command, "--context-limit") is not None
                else -1
            )
            == int_or_default(tokens.get("context_limit_memories"), -2)
            and (
                command_int_value(command, "--judge-context-window-tokens")
                if command_int_value(command, "--judge-context-window-tokens") is not None
                else 262144
            )
            == int_or_default(tokens.get("judge_context_window_tokens"), -4)
            and command_int_value(command, "--max-media-per-system")
            == int_or_default(media.get("max_media_per_system"), -3),
            (
                f"command_context_limit={command_flag_value(command, '--context-limit')} "
                f"artifact_context_limit={tokens.get('context_limit_memories')} "
                "command_judge_context_window_tokens="
                f"{command_flag_value(command, '--judge-context-window-tokens')} "
                "artifact_judge_context_window_tokens="
                f"{tokens.get('judge_context_window_tokens')} "
                "command_max_media_per_system="
                f"{command_flag_value(command, '--max-media-per-system')} "
                f"artifact_max_media_per_system={media.get('max_media_per_system')}"
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


def probes_missing_positive_numeric(summary: dict[str, Any], field: str) -> list[Any]:
    missing = []
    for probe in summary.get("probes", []):
        value = probe.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
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


def marker_only_compaction_probes(summary: dict[str, Any]) -> list[Any]:
    out = []
    for probe in summary.get("probes", []):
        try:
            compacted_items = int(
                probe.get("normal_rag_compacted_history_items", 0) or 0
            )
        except Exception:
            compacted_items = 0
        if compacted_items <= 0:
            continue
        text = str(probe.get("normal_rag_compacted_summary", "")).strip()
        if not text or text.startswith("[compacted_history "):
            out.append(probe.get("event_index"))
    return out


def probes_missing_cortext_frozen_packets(summary: dict[str, Any]) -> list[Any]:
    out = []
    for probe in summary.get("probes", []):
        if probe.get("cortext_frozen_packet_policy") != "probe_time_hydrated_context_snapshot":
            out.append(probe.get("event_index"))
            continue
        if not isinstance(probe.get("cortext_frozen_working_memory"), list):
            out.append(probe.get("event_index"))
            continue
        if not isinstance(probe.get("cortext_frozen_retrieved_memory"), list):
            out.append(probe.get("event_index"))
    return out


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


def source_input_fingerprint(summary: dict[str, Any]) -> dict[str, Any]:
    input_dir = pathlib.Path(str(summary.get("input_dir", "")))
    body: dict[str, Any] = {
        "schema": "cortext_julie_source_input_fingerprint_v1",
        "privacy": (
            "private local provenance: records content hashes and aggregate "
            "file metadata only, never message text or media bytes"
        ),
        "input_dir": str(input_dir),
        "exists": input_dir.exists(),
        "recursive": True,
        "file_count": 0,
        "readable_file_count": 0,
        "symlink_count": 0,
        "total_bytes": 0,
        "extension_counts": {},
        "transcript_present": False,
        "transcript_sha256": "",
        "manifest_sha256": "",
        "unreadable_files": 0,
    }
    if not input_dir.exists() or not input_dir.is_dir():
        return body

    private_entries: list[dict[str, Any]] = []
    extension_counts: dict[str, int] = {}
    for path in sorted(input_dir.rglob("*"), key=lambda item: str(item.relative_to(input_dir))):
        if path.is_dir():
            continue
        try:
            relative_path = path.relative_to(input_dir)
        except ValueError:
            relative_path = pathlib.Path(path.name)
        suffix = path.suffix.lower()
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        entry: dict[str, Any] = {
            "relative_path_sha256": sha256_text(str(relative_path)),
            "suffix": suffix,
            "is_symlink": path.is_symlink(),
            "readable": False,
            "size_bytes": None,
            "sha256": "",
        }
        body["file_count"] += 1
        if path.is_symlink():
            body["symlink_count"] += 1
        try:
            stat = path.stat()
            entry["size_bytes"] = stat.st_size
            entry["sha256"] = file_sha256(path)
            entry["readable"] = True
            body["readable_file_count"] += 1
            body["total_bytes"] += stat.st_size
            if str(relative_path) == "Messages - Julie Willen.txt":
                body["transcript_present"] = True
                body["transcript_sha256"] = entry["sha256"]
        except OSError as exc:
            body["unreadable_files"] += 1
            entry["error"] = exc.__class__.__name__
        private_entries.append(entry)

    body["extension_counts"] = extension_counts
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": body["schema"],
                "files": private_entries,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return body


def source_input_checks(fingerprint: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check(
            "source_input_fingerprint_recorded",
            bool(fingerprint.get("exists"))
            and int(fingerprint.get("file_count", 0) or 0) > 0
            and bool(fingerprint.get("manifest_sha256")),
            (
                f"exists={fingerprint.get('exists')} "
                f"file_count={fingerprint.get('file_count')} "
                f"manifest_sha256={fingerprint.get('manifest_sha256')}"
            ),
        ),
        check(
            "source_input_files_readable",
            int(fingerprint.get("file_count", 0) or 0)
            == int(fingerprint.get("readable_file_count", -1) or -1)
            and int(fingerprint.get("unreadable_files", 0) or 0) == 0,
            (
                f"file_count={fingerprint.get('file_count')} "
                f"readable_file_count={fingerprint.get('readable_file_count')} "
                f"unreadable_files={fingerprint.get('unreadable_files')}"
            ),
        ),
        check(
            "source_transcript_hash_recorded",
            bool(fingerprint.get("transcript_present"))
            and bool(fingerprint.get("transcript_sha256")),
            (
                f"transcript_present={fingerprint.get('transcript_present')} "
                f"transcript_sha256={fingerprint.get('transcript_sha256')}"
            ),
        ),
        check(
            "source_input_fingerprint_public_shape",
            "files" not in fingerprint
            and "private_entries" not in fingerprint
            and "filenames" not in fingerprint,
            (
                "forbidden_keys_present="
                f"{[key for key in ['files', 'private_entries', 'filenames'] if key in fingerprint]}"
            ),
        ),
    ]


def source_id_audit(summary: dict[str, Any]) -> dict[str, Any]:
    db_path = pathlib.Path(str(summary.get("db_path", "")))
    audit: dict[str, Any] = {
        "schema": "cortext_source_id_audit_v1",
        "db_path_recorded": str(db_path),
        "db_exists": db_path.exists(),
        "signal_rows": 0,
        "source_counts": {},
        "modality_source_counts": {},
        "unexpected_source_id_sha256": [],
        "media_source_mismatch_count": 0,
        "media_encoded_source_id_count": 0,
        "load_error": "",
    }
    if not db_path.exists():
        return audit

    allowed_sources = {"chat/user", "chat/assistant", "cortext/consolidate"}
    media_modalities = {"audio", "image", "video"}
    media_suffixes = (".wav", ".mp3", ".m4a", ".aac", ".jpg", ".jpeg", ".png", ".mov", ".mp4")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                select modality, source_id, count(*)
                from signals
                group by modality, source_id
                """
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        audit["load_error"] = exc.__class__.__name__
        return audit

    unexpected: set[str] = set()
    for modality, source_id, count in rows:
        modality_text = str(modality or "")
        source_text = str(source_id or "")
        count_int = int(count or 0)
        audit["signal_rows"] += count_int
        if source_text in allowed_sources:
            audit["source_counts"][source_text] = (
                audit["source_counts"].get(source_text, 0) + count_int
            )
            key = f"{modality_text}|{source_text}"
            audit["modality_source_counts"][key] = count_int
        else:
            unexpected.add(source_text)
        source_lower = source_text.lower()
        if source_lower.endswith(media_suffixes):
            audit["media_encoded_source_id_count"] += count_int
        if modality_text in media_modalities and source_text not in {
            "chat/user",
            "chat/assistant",
        }:
            audit["media_source_mismatch_count"] += count_int

    audit["unexpected_source_id_sha256"] = [
        sha256_text(value) for value in sorted(unexpected)
    ]
    return audit


def source_id_audit_checks(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check(
            "source_id_audit_available",
            bool(audit.get("db_exists"))
            and not audit.get("load_error")
            and int_or_default(audit.get("signal_rows"), 0) > 0,
            (
                f"db_exists={audit.get('db_exists')} "
                f"signal_rows={audit.get('signal_rows')} "
                f"load_error={audit.get('load_error')!r}"
            ),
        ),
        check(
            "source_ids_verified_from_signals",
            not audit.get("unexpected_source_id_sha256")
            and int_or_default(audit.get("media_source_mismatch_count"), 0) == 0
            and int_or_default(audit.get("media_encoded_source_id_count"), 0) == 0,
            (
                "unexpected_source_id_sha256="
                f"{audit.get('unexpected_source_id_sha256')} "
                "media_source_mismatch_count="
                f"{audit.get('media_source_mismatch_count')} "
                "media_encoded_source_id_count="
                f"{audit.get('media_encoded_source_id_count')}"
            ),
        ),
        check(
            "media_modalities_share_speaker_source_ids",
            any(
                key.startswith("audio|chat/")
                for key in audit.get("modality_source_counts", {})
            )
            and any(
                key.startswith("image|chat/")
                for key in audit.get("modality_source_counts", {})
            ),
            f"modality_source_counts={audit.get('modality_source_counts')}",
        ),
    ]


def load_benchmark_environment_snapshot(summary_path: pathlib.Path) -> dict[str, Any] | None:
    snapshot_path = summary_path.parent / BENCHMARK_ENV_SNAPSHOT_NAME
    try:
        if snapshot_path.exists():
            body = load_json(snapshot_path)
            body["path"] = str(snapshot_path)
            return body
    except Exception:
        return {
            "path": str(snapshot_path),
            "schema": "invalid",
            "load_error": True,
        }
    return None


def benchmark_environment_checks(
    snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if snapshot is None:
        return [
            check(
                "benchmark_environment_snapshot_present",
                False,
                (
                    "missing "
                    f"{BENCHMARK_ENV_SNAPSHOT_NAME}; release report needs a "
                    "runtime environment snapshot for the main benchmark process"
                ),
            )
        ]
    behavior_env = snapshot.get("cortext_behavior_env", {})
    if not isinstance(behavior_env, dict):
        behavior_env = {}
    hosted_provider_env = snapshot.get("hosted_provider_behavior_env", {})
    if not isinstance(hosted_provider_env, dict):
        hosted_provider_env = {}
    return [
        check(
            "benchmark_environment_snapshot_present",
            snapshot.get("schema") == "cortext_benchmark_environment_snapshot_v1"
            and not snapshot.get("load_error"),
            f"path={snapshot.get('path')} schema={snapshot.get('schema')}",
        ),
        check(
            "benchmark_environment_snapshot_identifies_process",
            int_or_default(snapshot.get("pid"), 0) > 0
            and bool(snapshot.get("process_name")),
            (
                f"pid={snapshot.get('pid')} "
                f"process_name={snapshot.get('process_name')!r}"
            ),
        ),
        check(
            "benchmark_runtime_cortext_behavior_env_clean",
            not behavior_env,
            f"cortext_behavior_env={behavior_env}",
        ),
        check(
            "benchmark_runtime_no_hosted_provider_dependency",
            not hosted_provider_env,
            f"hosted_provider_behavior_env={hosted_provider_env}",
        ),
    ]


def load_optional_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            body = load_json(path)
            body["path"] = str(path)
            body["sha256"] = file_sha256(path)
            return body
    except Exception as exc:
        return {
            "path": str(path),
            "schema": "invalid",
            "load_error": exc.__class__.__name__,
        }
    return None


def selected_early_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    selected = {}
    for key in [
        "judged_rows",
        "probe_count",
        "cortext_wins",
        "traditional_chat_rag_wins",
        "full_history_upper_bound_wins",
        "cortext_win_rate",
        "cortext_quality_delta_vs_traditional_chat_rag",
        "cortext_quality_delta_vs_full_history_upper_bound",
        "cortext_token_savings_vs_traditional_chat_rag",
        "mean_cortext_context_tokens",
        "mean_traditional_chat_rag_tokens",
    ]:
        if key in metrics:
            selected[key] = metrics.get(key)
    return selected


def early_judge_summary(
    benchmark_status: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    status_body = benchmark_status if isinstance(benchmark_status, dict) else {}
    manifest_body = manifest if isinstance(manifest, dict) else {}
    latest_payload = status_body.get("early_judge_latest", {})
    if not isinstance(latest_payload, dict):
        latest_payload = {}
    latest_record = latest_payload.get("latest", {})
    if not isinstance(latest_record, dict):
        latest_record = manifest_body.get("latest", {})
    if not isinstance(latest_record, dict):
        latest_record = {}
    completed = manifest_body.get("completed", [])
    if not isinstance(completed, list):
        completed = []

    status_probe_stream = status_body.get("probe_stream", {})
    if not isinstance(status_probe_stream, dict):
        status_probe_stream = {}

    completed_status_counts: dict[str, int] = {}
    completed_early_stop_count = 0
    completed_milestones = []
    for record in completed:
        if not isinstance(record, dict):
            continue
        try:
            completed_milestones.append(int(record.get("milestone")))
        except Exception:
            pass
        record_status = str(record.get("fail_fast_status", "unknown"))
        completed_status_counts[record_status] = (
            completed_status_counts.get(record_status, 0) + 1
        )
        early_stop = record.get("early_stop")
        if isinstance(early_stop, dict) and early_stop:
            completed_early_stop_count += 1

    fixed_milestones = latest_payload.get(
        "fixed_milestones", manifest_body.get("fixed_milestones", [])
    )
    if not isinstance(fixed_milestones, list):
        fixed_milestones = []

    return {
        "schema": "cortext_julie_release_early_judge_summary_v1",
        "benchmark_status_present": benchmark_status is not None
        and not status_body.get("load_error"),
        "benchmark_status_schema": status_body.get("schema"),
        "benchmark_status": status_body.get("status"),
        "benchmark_exit_code": status_body.get("benchmark_exit_code"),
        "early_judge_exit_code": status_body.get("early_judge_exit_code"),
        "elapsed_s": status_body.get("elapsed_s"),
        "probe_stream_rows": status_probe_stream.get("rows"),
        "required_rows_after_benchmark": status_probe_stream.get(
            "required_rows_after_benchmark"
        ),
        "latest_payload_schema": latest_payload.get("schema"),
        "release_gate_use": latest_payload.get("release_gate_use"),
        "judge_provider": latest_payload.get(
            "judge_provider", manifest_body.get("judge_provider")
        ),
        "judge_model": latest_payload.get(
            "judge_model", manifest_body.get("judge_model")
        ),
        "judge_repetitions": latest_payload.get(
            "judge_repetitions", manifest_body.get("judge_repetitions")
        ),
        "confirm_fail_repetitions": latest_payload.get(
            "confirm_fail_repetitions",
            manifest_body.get("confirm_fail_repetitions"),
        ),
        "blind_packets": manifest_body.get("blind_packets"),
        "judge_packet_item_limit": manifest_body.get("judge_packet_item_limit"),
        "fixed_milestones": fixed_milestones,
        "periodic_stride": latest_payload.get(
            "periodic_stride", manifest_body.get("periodic_stride")
        ),
        "quality_gate_min_milestone": latest_payload.get(
            "quality_gate_min_milestone",
            manifest_body.get("quality_gate_min_milestone"),
        ),
        "quality_trend_gate_min_milestone": latest_payload.get(
            "quality_trend_gate_min_milestone",
            manifest_body.get("quality_trend_gate_min_milestone"),
        ),
        "quality_trend_window": latest_payload.get(
            "quality_trend_window", manifest_body.get("quality_trend_window")
        ),
        "quality_gate_requires_rag_pressure": latest_payload.get(
            "quality_gate_requires_rag_pressure",
            manifest_body.get("quality_gate_requires_rag_pressure"),
        ),
        "quality_gate_min_history_budget_ratio": latest_payload.get(
            "quality_gate_min_history_budget_ratio",
            manifest_body.get("quality_gate_min_history_budget_ratio"),
        ),
        "latest_milestone": latest_record.get("milestone"),
        "latest_fail_fast_status": latest_record.get("fail_fast_status"),
        "latest_quality_gate_active": latest_record.get("quality_gate_active"),
        "latest_quality_trend_gate_active": latest_record.get(
            "quality_trend_gate_active"
        ),
        "latest_quality_gate_phase_ready": latest_record.get(
            "quality_gate_phase_ready"
        ),
        "latest_quality_gate_phase_reason": latest_record.get(
            "quality_gate_phase_reason"
        ),
        "latest_early_stop": latest_record.get("early_stop"),
        "latest_max_rolling_history_budget_ratio": latest_record.get(
            "max_rolling_history_budget_ratio"
        ),
        "latest_metrics": selected_early_metrics(latest_record),
        "completed_milestones": sorted(completed_milestones),
        "completed_status_counts": completed_status_counts,
        "completed_early_stop_count": completed_early_stop_count,
        "manifest_present": manifest is not None and not manifest_body.get("load_error"),
        "manifest_schema": manifest_body.get("schema"),
        "manifest_sha256": manifest_body.get("sha256", ""),
    }


def early_judge_checks(early: dict[str, Any]) -> list[dict[str, Any]]:
    fixed_milestones = early.get("fixed_milestones", [])
    if not isinstance(fixed_milestones, list):
        fixed_milestones = []
    completed_milestones = early.get("completed_milestones", [])
    if not isinstance(completed_milestones, list):
        completed_milestones = []
    latest_milestone = int_or_default(early.get("latest_milestone"), 0)
    quality_gate_min = int_or_default(early.get("quality_gate_min_milestone"), 0)
    trend_gate_min = int_or_default(
        early.get("quality_trend_gate_min_milestone"), 0
    )
    early_code = early.get("early_judge_exit_code")
    early_code_ok = early_code == 0
    if early_code is None and early.get("benchmark_status") not in {
        "early_judge_failed",
        None,
    }:
        early_code_ok = False

    return [
        check(
            "early_judge_status_recorded",
            early.get("benchmark_status_present") is True
            and early.get("benchmark_status_schema")
            == "cortext_julie_release_benchmark_status_v1",
            (
                f"benchmark_status_present={early.get('benchmark_status_present')} "
                f"benchmark_status_schema={early.get('benchmark_status_schema')}"
            ),
        ),
        check(
            "early_judge_manifest_recorded",
            early.get("manifest_present") is True
            and early.get("manifest_schema")
            == "julie_probe_stream_early_judge_manifest_v1"
            and bool(early.get("manifest_sha256")),
            (
                f"manifest_present={early.get('manifest_present')} "
                f"manifest_schema={early.get('manifest_schema')} "
                f"manifest_sha256={early.get('manifest_sha256')}"
            ),
        ),
        check(
            "early_judge_local_gemma4_configured",
            early.get("latest_payload_schema")
            == "julie_probe_stream_early_judge_latest_v1"
            and early.get("judge_provider") == "ollama"
            and early.get("judge_model") == "gemma4:12b-it-qat"
            and int_or_default(early.get("judge_repetitions"), 0) >= 1,
            (
                f"latest_payload_schema={early.get('latest_payload_schema')} "
                f"judge_provider={early.get('judge_provider')} "
                f"judge_model={early.get('judge_model')} "
                f"judge_repetitions={early.get('judge_repetitions')}"
            ),
        ),
        check(
            "early_judge_failures_confirmed",
            int_or_default(early.get("confirm_fail_repetitions"), 0)
            >= MIN_JUDGE_REPETITIONS,
            (
                "confirm_fail_repetitions="
                f"{early.get('confirm_fail_repetitions')} "
                f"min_required={MIN_JUDGE_REPETITIONS}"
            ),
        ),
        check(
            "early_judge_blind_packet_screening",
            early.get("blind_packets") is True
            and int_or_default(early.get("judge_packet_item_limit"), 0) != 0,
            (
                f"blind_packets={early.get('blind_packets')} "
                f"judge_packet_item_limit={early.get('judge_packet_item_limit')}"
            ),
        ),
        check(
            "early_judge_quality_gates_configured",
            16 in [int_or_default(item, -1) for item in fixed_milestones]
            and 0 < quality_gate_min <= 16
            and 0 < trend_gate_min <= 8
            and int_or_default(early.get("quality_trend_window"), -1) >= 0,
            (
                f"fixed_milestones={fixed_milestones} "
                f"quality_gate_min_milestone={early.get('quality_gate_min_milestone')} "
                "quality_trend_gate_min_milestone="
                f"{early.get('quality_trend_gate_min_milestone')} "
                f"quality_trend_window={early.get('quality_trend_window')} "
                "quality_gate_requires_rag_pressure="
                f"{early.get('quality_gate_requires_rag_pressure')} "
                "quality_gate_min_history_budget_ratio="
                f"{early.get('quality_gate_min_history_budget_ratio')}"
            ),
        ),
        check(
            "early_judge_stream_progress_recorded",
            int_or_default(early.get("probe_stream_rows"), 0) >= latest_milestone
            and latest_milestone > 0,
            (
                f"probe_stream_rows={early.get('probe_stream_rows')} "
                f"latest_milestone={early.get('latest_milestone')} "
                f"completed_milestones={completed_milestones}"
            ),
        ),
        check(
            "early_judge_gate_passed_for_final_release",
            early.get("benchmark_status") != "early_judge_failed"
            and early_code_ok
            and early.get("latest_fail_fast_status") == "pass",
            (
                f"benchmark_status={early.get('benchmark_status')} "
                f"early_judge_exit_code={early.get('early_judge_exit_code')} "
                f"latest_fail_fast_status={early.get('latest_fail_fast_status')} "
                f"latest_milestone={early.get('latest_milestone')} "
                f"latest_early_stop={early.get('latest_early_stop')}"
            ),
        ),
    ]


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
            "fixed_slice_starts_at_export_beginning",
            int(summary.get("skipped_transcript_messages", 0) or 0) == 0,
            (
                "skipped_transcript_messages="
                f"{summary.get('skipped_transcript_messages')}"
            ),
        ),
        check(
            "cortext_packets_frozen_at_probe_time",
            not probes_missing_cortext_frozen_packets(summary),
            (
                "missing_or_legacy_probe_events="
                f"{probes_missing_cortext_frozen_packets(summary)}"
            ),
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
            "full_history_upper_bound_recorded",
            not probes_missing_numeric(summary, "full_history_tokens")
            and not probes_missing_positive_numeric(summary, "full_history_items"),
            (
                "missing_token_events="
                f"{probes_missing_numeric(summary, 'full_history_tokens')} "
                "missing_item_events="
                f"{probes_missing_positive_numeric(summary, 'full_history_items')}"
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
            "rag_compaction_has_content_summary",
            summary.get("normal_rag_compaction_summary_policy")
            == "deterministic_extractive_prior_chat"
            and not marker_only_compaction_probes(summary),
            (
                "normal_rag_compaction_summary_policy="
                f"{summary.get('normal_rag_compaction_summary_policy')!r} "
                f"marker_only_or_missing_events={marker_only_compaction_probes(summary)}"
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
    media = judge.get("media_attachments", {})
    judge_validation = judge.get("judge_validation", {})
    processed = judge.get("processed", {})
    media_capabilities = judge.get("judge_media_capabilities", {})
    if not isinstance(media_capabilities, dict):
        media_capabilities = {}
    systems = set(protocol.get("systems", []))
    fields = set(protocol.get("score_fields", []))
    cortext_packet_source = protocol.get("cortext_packet_source", {})
    if not isinstance(cortext_packet_source, dict):
        cortext_packet_source = {}
    repetitions = int(
        judge.get("judge_repetitions") or protocol.get("judge_repetitions") or 0
    )
    cortext_media_count = media_attachment_count(media, "cortext_native")
    rag_media_count = media_attachment_count(media, "traditional_chat_rag")
    full_history_media_count = media_attachment_count(media, "full_history_upper_bound")
    randomization = packet_randomization_summary(judge)
    repetition_consistency = judge_repetition_consistency_summary(judge)
    expected_repetitions = int(repetition_consistency.get("expected_repetitions", 0) or 0)
    required_majority_count = expected_repetitions // 2 + 1 if expected_repetitions > 0 else 0
    return [
        check(
            "judge_local_only",
            judge.get("remote_provider_allowed") is False
            and is_loopback_url(judge.get("judge_base_url", "")),
            f"provider={judge.get('judge_provider')} base_url={judge.get('judge_base_url')}",
        ),
        check(
            "packets_blinded",
            bool(protocol.get("packet_blinding")),
            f"packet_blinding={protocol.get('packet_blinding')}",
        ),
        check(
            "judge_packet_surface_structurally_normalized",
            protocol.get("packet_surface")
            == "structurally_normalized_event_evidence_v1",
            f"packet_surface={protocol.get('packet_surface')!r}",
        ),
        check(
            "packet_labels_randomized_across_judgments",
            bool(protocol.get("packet_blinding"))
            and int(randomization.get("mapped_rows", 0) or 0)
            == int(randomization.get("judgment_rows", -1) or -1)
            and int(randomization.get("unique_mapping_count", 0) or 0) > 1
            and int(randomization.get("min_labels_per_system", 0) or 0) >= 2,
            f"packet_randomization={randomization}",
        ),
        check(
            "judge_used_probe_time_cortext_snapshots",
            int(cortext_packet_source.get("probe_time_summary_snapshot", 0) or 0)
            == int(judge.get("probe_count", 0) or 0)
            and int(cortext_packet_source.get("final_db_rehydration_fallback", 0) or 0)
            == 0,
            (
                f"cortext_packet_source={cortext_packet_source} "
                f"probe_count={judge.get('probe_count')}"
            ),
        ),
        check(
            "judge_packets_uncropped",
            int(tokens.get("context_limit_memories", 0) or 0) == -1,
            f"context_limit_memories={tokens.get('context_limit_memories')}",
        ),
        check(
            "judge_context_window_recorded",
            int_or_default(tokens.get("judge_context_window_tokens"), 0) > 0
            and int_or_default(tokens.get("max_judge_prompt_tokens_estimate"), 0) > 0,
            (
                f"judge_context_window_tokens={tokens.get('judge_context_window_tokens')} "
                "max_judge_prompt_tokens_estimate="
                f"{tokens.get('max_judge_prompt_tokens_estimate')}"
            ),
        ),
        check(
            "full_history_prompt_fits_judge_context",
            fairness.get("full_history_prompt_fits_judge_context") is True
            and int_or_default(tokens.get("max_judge_prompt_tokens_estimate"), 0)
            <= int_or_default(tokens.get("judge_context_window_tokens"), 0),
            (
                "full_history_prompt_fits_judge_context="
                f"{fairness.get('full_history_prompt_fits_judge_context')} "
                "max_judge_prompt_tokens_estimate="
                f"{tokens.get('max_judge_prompt_tokens_estimate')} "
                f"judge_context_window_tokens={tokens.get('judge_context_window_tokens')}"
            ),
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
            "judge_repetition_rows_complete",
            expected_repetitions >= MIN_JUDGE_REPETITIONS
            and int(repetition_consistency.get("events_with_expected_repetitions", 0) or 0)
            == int(judge.get("probe_count", 0) or 0),
            f"judge_repetition_consistency={repetition_consistency}",
        ),
        check(
            "judge_repetition_majority_agreement",
            required_majority_count > 0
            and float(repetition_consistency.get("majority_event_rate", 0.0) or 0.0)
            >= MIN_JUDGE_MAJORITY_EVENT_RATE
            and float(repetition_consistency.get("mean_majority_fraction", 0.0) or 0.0)
            >= MIN_JUDGE_MEAN_MAJORITY_FRACTION,
            (
                f"required_majority_count={required_majority_count} "
                f"required_majority_event_rate={MIN_JUDGE_MAJORITY_EVENT_RATE} "
                "required_mean_majority_fraction="
                f"{MIN_JUDGE_MEAN_MAJORITY_FRACTION} "
                f"judge_repetition_consistency={repetition_consistency}"
            ),
        ),
        check(
            "multimodal_judge_enabled",
            judge.get("multimodal_judge") is True,
            f"multimodal_judge={judge.get('multimodal_judge')}",
        ),
        check(
            "judge_image_capability_available_for_image_slice",
            int_or_default(processed.get("image"), 0) == 0
            or (
                media_capabilities.get("image") is True
                and fairness.get("attached_images_judged_when_present") is True
            ),
            (
                f"processed_image={processed.get('image')} "
                f"judge_media_capabilities={media_capabilities} "
                "attached_images_judged_when_present="
                f"{fairness.get('attached_images_judged_when_present')}"
            ),
        ),
        check(
            "judge_audio_capability_available_for_audio_slice",
            int_or_default(processed.get("audio"), 0) == 0
            or (
                media_capabilities.get("audio") is True
                and fairness.get("attached_audio_judged_when_present") is True
            ),
            (
                f"processed_audio={processed.get('audio')} "
                f"judge_media_capabilities={media_capabilities} "
                "attached_audio_judged_when_present="
                f"{fairness.get('attached_audio_judged_when_present')}"
            ),
        ),
        check(
            "judge_media_attachments_uncapped",
            media.get("enabled") is True
            and int(media.get("max_media_per_system", 0) or 0) == -1,
            (
                f"enabled={media.get('enabled')} "
                f"max_media_per_system={media.get('max_media_per_system')}"
            ),
        ),
        check(
            "cortext_media_evidence_judged",
            cortext_media_count > 0,
            (
                f"cortext_media_attachments={cortext_media_count} "
                f"cortext_media={media.get('cortext_native')}"
            ),
        ),
        check(
            "text_only_baselines_received_no_media_attachments",
            rag_media_count == 0 and full_history_media_count == 0,
            (
                f"traditional_chat_rag_media_attachments={rag_media_count} "
                f"full_history_media_attachments={full_history_media_count}"
            ),
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
            "rag_compaction_contentful_in_judge_packets",
            fairness.get("traditional_chat_rag_contentful_compaction") is True,
            f"fairness_counters={fairness.get('counters', {})}",
        ),
        check(
            "judge_structural_reasons_present",
            int(judge_validation.get("missing_system_reason", 0) or 0) == 0,
            f"judge_validation={judge_validation}",
        ),
        check(
            "judge_failure_reason_validation_clean",
            int(judge_validation.get("invalid_failure_reason", 0) or 0) == 0
            and int(judge_validation.get("winner_failure_mismatch", 0) or 0) == 0,
            f"judge_validation={judge_validation}",
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


def judge_media_smoke_checks(
    smoke: dict[str, Any] | None,
    judge_command: str,
) -> list[dict[str, Any]]:
    if smoke is None:
        return [
            check(
                "judge_media_smoke_recorded",
                False,
                "missing --judge-media-smoke artifact",
            )
        ]

    command = split_command(judge_command)
    command_model = command_flag_value(command, "--model")
    selected_model = str(smoke.get("selected_release_judge_model", ""))
    results = smoke.get("results", {})
    selected = results.get(selected_model, {}) if isinstance(results, dict) else {}
    image = selected.get("image", {}) if isinstance(selected, dict) else {}
    audio = selected.get("audio", {}) if isinstance(selected, dict) else {}
    image_parsed = image.get("parsed", {}) if isinstance(image, dict) else {}
    audio_parsed = audio.get("parsed", {}) if isinstance(audio, dict) else {}
    if not isinstance(image_parsed, dict):
        image_parsed = {}
    if not isinstance(audio_parsed, dict):
        audio_parsed = {}

    return [
        check(
            "judge_media_smoke_recorded",
            smoke.get("schema") == "cortext_local_ollama_judge_media_smoke_v1"
            and smoke.get("private_data_used") is False,
            (
                f"schema={smoke.get('schema')} "
                f"private_data_used={smoke.get('private_data_used')}"
            ),
        ),
        check(
            "judge_media_smoke_model_matches_judge",
            bool(selected_model) and selected_model == command_model,
            f"selected_model={selected_model} command_model={command_model}",
        ),
        check(
            "judge_media_smoke_image_supported",
            image_parsed.get("image_seen") is True,
            f"selected_model={selected_model} image={image_parsed}",
        ),
        check(
            "judge_media_smoke_audio_supported",
            audio_parsed.get("audio_seen") is True,
            f"selected_model={selected_model} audio={audio_parsed}",
        ),
    ]


def media_attachment_count(media: dict[str, Any], system: str) -> int:
    system_media = media.get(system, {})
    if not isinstance(system_media, dict):
        return 0
    return int(system_media.get("image", 0) or 0) + int(
        system_media.get("audio", 0) or 0
    )


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
    human: dict[str, Any] | None,
    human_path: pathlib.Path | None,
    summary_path: pathlib.Path,
    summary: dict[str, Any],
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
        frozen_path = pathlib.Path(str(human.get("human_frozen", "")))
        judge_frozen_path = pathlib.Path(str(agreement.get("judge_frozen", "")))
        sample: dict[str, Any] = {}
        frozen: dict[str, Any] = {}
        sample_matches = False
        sample_hash_matches = False
        frozen_matches = False
        judge_frozen_labeling: dict[str, Any] = {}
        sample_event_indices: set[int] = set()
        summary_probe_event_indices = {
            int(probe.get("event_index", -1))
            for probe in summary.get("probes", [])
            if isinstance(probe, dict)
        }
        if sample_path.exists():
            try:
                sample = load_json(sample_path)
                sample_matches = paths_equivalent(sample.get("source_summary", ""), summary_path)
                sample_hash_matches = (
                    sample.get("source_summary_sha256") == file_sha256(summary_path)
                    and human.get("source_summary_sha256") == file_sha256(summary_path)
                )
                sample_composition = sample.get("sample_composition", {})
                sample_event_indices = {
                    int(task.get("event_index", -1))
                    for task in sample.get("tasks", [])
                    if isinstance(task, dict)
                }
            except Exception:
                sample_matches = False
                sample_hash_matches = False
                sample_composition = {}
        else:
            sample_composition = {}
        human_blinding_policy = sample.get("human_blinding_policy", {})
        if not isinstance(human_blinding_policy, dict):
            human_blinding_policy = {}
        non_random_candidate_tasks = [
            task.get("probe_id", task.get("event_index"))
            for task in sample.get("tasks", [])
            if isinstance(task, dict)
            and task.get("candidate_order_policy") != "randomized_blind_order_seeded"
        ]
        future_or_non_prior_candidates = []
        for task in sample.get("tasks", []):
            if not isinstance(task, dict):
                continue
            try:
                event_index = int(task.get("event_index", -1))
            except Exception:
                event_index = -1
            for cand in task.get("candidates", []):
                if not isinstance(cand, dict):
                    continue
                try:
                    candidate_event = int(cand.get("event_index", -1))
                except Exception:
                    candidate_event = -1
                if candidate_event >= event_index:
                    future_or_non_prior_candidates.append(
                        {
                            "probe_event_index": event_index,
                            "candidate_event_index": candidate_event,
                        }
                    )
        if frozen_path.exists():
            try:
                frozen = load_json(frozen_path)
                frozen_matches = (
                    frozen.get("freeze_sha256") == human.get("human_freeze_sha256")
                )
            except Exception:
                frozen_matches = False
        if judge_frozen_path.exists():
            try:
                judge_frozen = load_json(judge_frozen_path)
                judge_frozen_labeling = judge_frozen.get("labeling", {})
                if not isinstance(judge_frozen_labeling, dict):
                    judge_frozen_labeling = {}
            except Exception:
                judge_frozen_labeling = {}
        frozen_candidate_sources = [
            str(item).lower()
            for item in judge_frozen_labeling.get("candidate_sources", [])
        ]
        checks.extend(
            [
                check(
                    "human_labels_same_summary",
                    sample_matches,
                    f"sample={sample_path} summary={summary_path}",
                ),
                check(
                    "human_labels_summary_hash_matches",
                    sample_hash_matches,
                    (
                        f"sample_source_summary_sha256={sample.get('source_summary_sha256')} "
                        f"score_source_summary_sha256={human.get('source_summary_sha256')} "
                        f"summary_sha256={file_sha256(summary_path)}"
                    ),
                ),
                check(
                    "human_sample_probe_events_subset_of_frozen_summary",
                    sample_event_indices.issubset(summary_probe_event_indices)
                    and len(sample_event_indices) >= MIN_HUMAN_PROBES,
                    (
                        f"sample_probe_count={len(sample_event_indices)} "
                        f"summary_probe_count={len(summary_probe_event_indices)} "
                        f"extra={sorted(sample_event_indices - summary_probe_event_indices)}"
                    ),
                ),
                check(
                    "human_frozen_targets_hash_matches",
                    frozen_matches,
                    (
                        f"human_frozen={frozen_path} "
                        f"human_freeze_sha256={human.get('human_freeze_sha256')}"
                    ),
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
                check(
                    "human_candidate_order_randomized_blind",
                    not non_random_candidate_tasks
                    and human_blinding_policy.get("candidate_order")
                    == "randomized_with_seed"
                    and human_blinding_policy.get("candidate_provenance_hidden_in_ui")
                    is True
                    and "candidate_sources"
                    in human_blinding_policy.get("hidden_candidate_fields", []),
                    (
                        f"human_blinding_policy={human_blinding_policy} "
                        f"non_random_candidate_tasks={non_random_candidate_tasks}"
                    ),
                ),
                check(
                    "human_sample_candidates_prior_only",
                    not future_or_non_prior_candidates,
                    f"future_or_non_prior_candidates={future_or_non_prior_candidates[:10]}",
                ),
                check(
                    "human_judge_frozen_targets_include_active_packet_candidates",
                    any("active packet" in item for item in frozen_candidate_sources),
                    (
                        f"judge_frozen={judge_frozen_path} "
                        "candidate_sources="
                        f"{judge_frozen_labeling.get('candidate_sources')}"
                    ),
                ),
                check(
                    "human_judge_frozen_targets_include_media_candidates",
                    any(
                        "audio" in item or "image" in item or "video" in item
                        or "media" in item
                        for item in frozen_candidate_sources
                    ),
                    (
                        f"judge_frozen={judge_frozen_path} "
                        "candidate_sources="
                        f"{judge_frozen_labeling.get('candidate_sources')}"
                    ),
                ),
                check(
                    "human_judge_frozen_targets_local_only",
                    judge_frozen_labeling.get("remote_provider_allowed") is False
                    and is_loopback_url(judge_frozen_labeling.get("judge_base_url", "")),
                    (
                        f"judge_frozen={judge_frozen_path} "
                        f"judge_provider={judge_frozen_labeling.get('judge_provider')} "
                        f"judge_base_url={judge_frozen_labeling.get('judge_base_url')} "
                        "remote_provider_allowed="
                        f"{judge_frozen_labeling.get('remote_provider_allowed')}"
                    ),
                ),
                check(
                    "human_label_context_prior_only",
                    sample.get("labeling_context_policy", {}).get(
                        "future_turns_visible"
                    )
                    is False
                    and sample.get("labeling_context_policy", {}).get(
                        "query_context"
                    )
                    == "current_turn_plus_prior_context_only",
                    (
                        "labeling_context_policy="
                        f"{sample.get('labeling_context_policy', {})}"
                    ),
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


def public_human_label_summary(human: dict[str, Any] | None) -> dict[str, Any] | None:
    if human is None:
        return None
    summary = {
        "schema": human.get("schema"),
        "probe_count": human.get("probe_count"),
        "agreement": human.get("agreement", {}),
    }
    if human.get("schema") == "cortext_human_label_score_v1":
        summary["human_freeze_sha256"] = human.get("human_freeze_sha256")
    return summary


def frozen_target_artifact_checks(
    target_freeze: dict[str, Any] | None,
    target_freeze_path: pathlib.Path | None,
    human_eval: dict[str, Any] | None,
    human_eval_path: pathlib.Path | None,
    summary: dict[str, Any],
    summary_path: pathlib.Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if target_freeze is None or target_freeze_path is None:
        checks.append(
            pending(
                "judge_frozen_targets_present",
                "no --target-freeze artifact supplied; release protocol needs a frozen judged target manifest",
            )
        )
    else:
        checks.extend(
            [
                check(
                    "judge_frozen_targets_present",
                    target_freeze.get("schema") == "cortext_frozen_retrieval_probe_set_v1"
                    and bool(target_freeze.get("freeze_sha256")),
                    (
                        f"path={target_freeze_path} "
                        f"schema={target_freeze.get('schema')} "
                        f"freeze_sha256={target_freeze.get('freeze_sha256')}"
                    ),
                ),
                check(
                    "judge_frozen_targets_same_summary",
                    pathlib.Path(str(target_freeze.get("source_summary", ""))) == summary_path,
                    (
                        f"target_source_summary={target_freeze.get('source_summary')} "
                        f"summary_path={summary_path}"
                    ),
                ),
                check(
                    "judge_frozen_targets_keep_full_probe_schedule",
                    int_or_default(target_freeze.get("probe_count"), -1)
                    == int_or_default(summary.get("probe_count"), -2)
                    and len(target_freeze.get("probes", []))
                    == int_or_default(summary.get("probe_count"), -3),
                    (
                        f"target_probe_count={target_freeze.get('probe_count')} "
                        f"target_probes={len(target_freeze.get('probes', []))} "
                        f"summary_probe_count={summary.get('probe_count')}"
                    ),
                ),
            ]
        )

    if human_eval is None or human_eval_path is None:
        checks.append(
            pending(
                "human_label_retrieval_eval_present",
                "no --human-label-eval artifact supplied; final release report should include retrieval scoring against human labels",
            )
        )
    else:
        checks.append(
            check(
                "human_label_retrieval_eval_present",
                human_eval.get("schema") == "cortext_frozen_retrieval_eval_v1"
                and pathlib.Path(str(human_eval.get("summary_path", ""))) == summary_path,
                (
                    f"path={human_eval_path} schema={human_eval.get('schema')} "
                    f"summary_path={human_eval.get('summary_path')} "
                    f"n={human_eval.get('n')}"
                ),
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
    missing_executable_hashes = []
    reuse_enabled = []
    bad_environment_snapshots = []
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
        executable = case.get("benchmark_executable", {})
        if not isinstance(executable, dict) or not executable.get("sha256"):
            missing_executable_hashes.append(name)
        if case.get("reuse_existing") is not False:
            reuse_enabled.append(
                {
                    "name": name,
                    "reuse_existing": case.get("reuse_existing"),
                    "reuse_policy": case.get("reuse_policy"),
                }
            )
        snapshot_path = case.get("environment_snapshot_path")
        snapshot = None
        if snapshot_path:
            try:
                snapshot = load_json(pathlib.Path(str(snapshot_path)))
            except Exception as exc:
                bad_environment_snapshots.append(
                    {
                        "name": name,
                        "environment_snapshot_path": snapshot_path,
                        "load_error": exc.__class__.__name__,
                    }
                )
        else:
            bad_environment_snapshots.append(
                {
                    "name": name,
                    "environment_snapshot_path": "",
                    "load_error": "missing_path",
                }
            )
        if isinstance(snapshot, dict):
            snapshot_env = snapshot.get("actual_env_overrides", {})
            if not isinstance(snapshot_env, dict):
                snapshot_env = {}
            stripped_keys = snapshot.get("stripped_hosted_provider_env_keys", [])
            if (
                snapshot.get("schema") != "cortext_ablation_environment_snapshot_v1"
                or "stripped" not in str(snapshot.get("hosted_provider_env_policy", ""))
                or not isinstance(stripped_keys, list)
                or snapshot_env != {
                    key: env[key]
                    for key in sorted(env)
                }
            ):
                bad_environment_snapshots.append(
                    {
                        "name": name,
                        "environment_snapshot_path": snapshot_path,
                        "schema": snapshot.get("schema"),
                        "hosted_provider_env_policy": snapshot.get(
                            "hosted_provider_env_policy"
                        ),
                        "stripped_hosted_provider_env_keys_type": type(
                            stripped_keys
                        ).__name__,
                        "actual_env_overrides": snapshot_env,
                        "expected_env_overrides": {
                            key: env[key]
                            for key in sorted(env)
                        },
                    }
                )

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
        check(
            "architecture_ablation_benchmark_executable_hashes_recorded",
            not missing_executable_hashes,
            f"missing_executable_hashes={missing_executable_hashes}",
        ),
        check(
            "architecture_ablation_reuse_disabled_for_release",
            not reuse_enabled,
            f"reuse_enabled_or_unrecorded={reuse_enabled}",
        ),
        check(
            "architecture_ablation_environment_snapshots_sanitize_hosted_providers",
            not bad_environment_snapshots,
            f"bad_environment_snapshots={bad_environment_snapshots}",
        ),
    ]


def ablation_artifact_checks(
    ablations: list[dict[str, Any]],
    main_schedule_sha256: str,
    main_source_input_manifest_sha256: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not ablations:
        return []

    manifest_mismatches = [
        row["name"]
        for row in ablations
        if row.get("frozen_probe_schedule_sha256") != main_schedule_sha256
    ]
    source_input_mismatches = [
        {
            "name": row["name"],
            "source_input_manifest_sha256": row.get("source_input_manifest_sha256"),
        }
        for row in ablations
        if row.get("source_input_manifest_sha256")
        != main_source_input_manifest_sha256
    ]
    non_default_knobs = [
        row["name"]
        for row in ablations
        if not isinstance(row.get("knobs"), dict)
        or not all(number_close(row["knobs"].get(k), v) for k, v in DEFAULT_KNOBS.items())
    ]
    weak_judges = []
    bad_judge_randomization = []
    bad_judge_repetition_consistency = []
    bad_packet_surfaces = []
    bad_context_fit = []
    bad_media_capabilities = []
    for row in ablations:
        judge = row.get("judge_protocol", {})
        media = row.get("media_attachments", {})
        tokens = row.get("tokens", {})
        fairness = row.get("fairness_checks", {})
        media_capabilities = row.get("judge_media_capabilities", {})
        processed = row.get("processed", {})
        if not isinstance(media_capabilities, dict):
            media_capabilities = {}
        repetitions = int(
            row.get("judge_repetitions") or judge.get("judge_repetitions") or 0
        )
        if (
            row.get("judge_provider") != "local_ollama"
            or row.get("remote_provider_allowed") is not False
            or not is_loopback_url(row.get("judge_base_url", ""))
            or not judge.get("packet_blinding")
            or repetitions < MIN_JUDGE_REPETITIONS
            or media.get("enabled") is not True
            or int(media.get("max_media_per_system", 0) or 0) != -1
        ):
            weak_judges.append(row["name"])
        if judge.get("packet_surface") != "structurally_normalized_event_evidence_v1":
            bad_packet_surfaces.append(
                {
                    "name": row["name"],
                    "packet_surface": judge.get("packet_surface"),
                }
            )
        if (
            fairness.get("full_history_prompt_fits_judge_context") is not True
            or int_or_default(tokens.get("max_judge_prompt_tokens_estimate"), 0)
            > int_or_default(tokens.get("judge_context_window_tokens"), 0)
        ):
            bad_context_fit.append(
                {
                    "name": row["name"],
                    "full_history_prompt_fits_judge_context": fairness.get(
                        "full_history_prompt_fits_judge_context"
                    ),
                    "max_judge_prompt_tokens_estimate": tokens.get(
                        "max_judge_prompt_tokens_estimate"
                    ),
                    "judge_context_window_tokens": tokens.get(
                        "judge_context_window_tokens"
                    ),
                }
            )
        if (
            int_or_default(processed.get("audio"), 0) > 0
            and (
                media_capabilities.get("audio") is not True
                or fairness.get("attached_audio_judged_when_present") is not True
            )
        ) or (
            int_or_default(processed.get("image"), 0) > 0
            and (
                media_capabilities.get("image") is not True
                or fairness.get("attached_images_judged_when_present") is not True
            )
        ):
            bad_media_capabilities.append(
                {
                    "name": row["name"],
                    "processed": processed,
                    "judge_media_capabilities": media_capabilities,
                    "fairness_checks": {
                        "attached_audio_judged_when_present": fairness.get(
                            "attached_audio_judged_when_present"
                        ),
                        "attached_images_judged_when_present": fairness.get(
                            "attached_images_judged_when_present"
                        ),
                    },
                }
            )
        randomization = row.get("packet_randomization", {})
        if (
            int_or_default(randomization.get("mapped_rows"), 0)
            != int_or_default(randomization.get("judgment_rows"), -1)
            or int_or_default(randomization.get("unique_mapping_count"), 0) <= 1
            or int_or_default(randomization.get("min_labels_per_system"), 0) < 2
        ):
            bad_judge_randomization.append(
                {
                    "name": row["name"],
                    "packet_randomization": randomization,
                }
            )
        repetition_consistency = row.get("judge_repetition_consistency", {})
        expected_repetitions = int_or_default(
            repetition_consistency.get("expected_repetitions"), 0
        )
        if (
            expected_repetitions < MIN_JUDGE_REPETITIONS
            or int_or_default(
                repetition_consistency.get("events_with_expected_repetitions"), 0
            )
            != int_or_default(row.get("probe_count"), -1)
            or float_or_default(
                repetition_consistency.get("majority_event_rate"), 0.0
            )
            < MIN_JUDGE_MAJORITY_EVENT_RATE
            or float_or_default(
                repetition_consistency.get("mean_majority_fraction"), 0.0
            )
            < MIN_JUDGE_MEAN_MAJORITY_FRACTION
        ):
            bad_judge_repetition_consistency.append(
                {
                    "name": row["name"],
                    "probe_count": row.get("probe_count"),
                    "judge_repetition_consistency": repetition_consistency,
                }
            )
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
    command_mismatches = [
        {
            "name": row["name"],
            "failures": row.get("command_check_failures", []),
        }
        for row in ablations
        if row.get("command_check_failures")
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
            "architecture_ablations_share_source_input_manifest",
            not source_input_mismatches,
            (
                f"main_source_input_manifest_sha256="
                f"{main_source_input_manifest_sha256} "
                f"mismatched={source_input_mismatches}"
            ),
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
            "architecture_ablation_judge_packet_surfaces_normalized",
            not bad_packet_surfaces,
            f"bad_packet_surfaces={bad_packet_surfaces}",
        ),
        check(
            "architecture_ablation_judge_context_windows_fit",
            not bad_context_fit,
            f"bad_context_fit={bad_context_fit}",
        ),
        check(
            "architecture_ablation_judge_media_capabilities_match_slice",
            not bad_media_capabilities,
            f"bad_media_capabilities={bad_media_capabilities}",
        ),
        check(
            "architecture_ablation_judge_packets_randomized",
            not bad_judge_randomization,
            f"bad_randomization={bad_judge_randomization}",
        ),
        check(
            "architecture_ablation_judge_repetition_consistency",
            not bad_judge_repetition_consistency,
            (
                "requires complete repeated local judging and majority agreement; "
                f"bad_repetition_consistency={bad_judge_repetition_consistency}"
            ),
        ),
        check(
            "architecture_ablations_share_processed_event_counts",
            not processed_mismatches,
            f"main={main_processed} mismatched={processed_mismatches}",
        ),
        check(
            "architecture_ablation_commands_match_artifacts",
            not command_mismatches,
            f"command_mismatches={command_mismatches}",
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
        "judge_media_capabilities": judge.get("judge_media_capabilities"),
        "packet_randomization": packet_randomization_summary(judge),
        "judge_repetition_consistency": judge_repetition_consistency_summary(judge),
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
        "p50_rag_total_latency_ms",
        "p95_rag_total_latency_ms",
        "p50_rag_retrieval_latency_ms",
        "p95_rag_retrieval_latency_ms",
        "mean_cortext_prompt_tokens",
        "mean_rag_prompt_tokens",
        "db_disk_bytes",
        "db_disk_bytes_per_event",
        "events_per_second_excluding_consolidation",
        "events_per_second_including_consolidation",
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
    checks.append(
        check(
            "ingest_throughput_including_consolidation_present",
            float(costs.get("events_per_second_including_consolidation") or 0.0)
            > 0.0,
            (
                "events_per_second_including_consolidation="
                f"{costs.get('events_per_second_including_consolidation')}"
            ),
        )
    )
    checks.append(
        check(
            "retrieval_latency_percentiles_recorded",
            float_or_default(costs.get("p50_rag_retrieval_latency_ms"), 0.0) > 0.0
            and float_or_default(costs.get("p95_rag_retrieval_latency_ms"), 0.0)
            >= float_or_default(costs.get("p50_rag_retrieval_latency_ms"), 0.0),
            (
                "p50_rag_retrieval_latency_ms="
                f"{costs.get('p50_rag_retrieval_latency_ms')} "
                "p95_rag_retrieval_latency_ms="
                f"{costs.get('p95_rag_retrieval_latency_ms')}"
            ),
        )
    )
    checks.append(
        check(
            "cortext_probe_latency_percentiles_recorded",
            float_or_default(costs.get("p50_cortext_probe_latency_ms"), 0.0) > 0.0
            and float_or_default(costs.get("p95_cortext_probe_latency_ms"), 0.0)
            >= float_or_default(costs.get("p50_cortext_probe_latency_ms"), 0.0),
            (
                "p50_cortext_probe_latency_ms="
                f"{costs.get('p50_cortext_probe_latency_ms')} "
                "p95_cortext_probe_latency_ms="
                f"{costs.get('p95_cortext_probe_latency_ms')}"
            ),
        )
    )
    checks.append(
        check(
            "disk_growth_recorded",
            int_or_default(costs.get("db_disk_bytes"), 0) > 0
            and float_or_default(costs.get("db_disk_bytes_per_event"), 0.0) > 0.0,
            (
                f"db_disk_bytes={costs.get('db_disk_bytes')} "
                f"db_disk_bytes_per_event={costs.get('db_disk_bytes_per_event')}"
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


def command_sha256(command: str) -> str:
    return sha256_text(" ".join(shlex.split(command))) if command else ""


def command_executable_artifact(command: str) -> dict[str, Any]:
    parts = split_command(command)
    raw = parts[0] if parts else ""
    resolved = ""
    if raw:
        candidate = pathlib.Path(raw)
        if candidate.exists() or "/" in raw:
            resolved = str(candidate.resolve())
        else:
            resolved = shutil.which(raw) or ""
    path = pathlib.Path(resolved) if resolved else None
    exists = bool(path and path.exists() and path.is_file())
    return {
        "command": raw,
        "path": str(path) if path else "",
        "exists": exists,
        "sha256": file_sha256(path) if path and exists else "",
        "bytes": path.stat().st_size if path and exists else 0,
        "mtime_ns": path.stat().st_mtime_ns if path and exists else 0,
    }


def load_protocol_freeze(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        body = load_json(path)
        body["path"] = str(path)
        body["sha256"] = file_sha256(path)
        return body
    except Exception as exc:
        return {
            "path": str(path),
            "schema": "invalid",
            "load_error": exc.__class__.__name__,
        }


def protocol_freeze_checks(
    freeze: dict[str, Any] | None,
    input_fingerprint: dict[str, Any],
    schedule_manifest: dict[str, Any],
    benchmark_command: str,
    benchmark_executable: dict[str, Any],
    git: dict[str, Any],
) -> list[dict[str, Any]]:
    if freeze is None:
        return [
            pending(
                "protocol_freeze_file_present",
                "no --freeze-file supplied; final release needs explicit pinned source/probe hashes",
            )
        ]

    expected_source_manifest = freeze.get("source_input_manifest_sha256")
    expected_probe_schedule = freeze.get("frozen_probe_schedule_sha256")
    expected_benchmark_command = freeze.get("benchmark_command_sha256")
    expected_benchmark_executable = freeze.get("benchmark_executable_sha256")
    expected_git_commit = freeze.get("git_commit")
    expected_git_worktree = freeze.get("git_worktree_manifest_sha256")
    return [
        check(
            "protocol_freeze_file_present",
            freeze.get("schema") == "cortext_julie_release_protocol_freeze_v1"
            and not freeze.get("load_error"),
            (
                f"path={freeze.get('path')} schema={freeze.get('schema')} "
                f"load_error={freeze.get('load_error')}"
            ),
        ),
        check(
            "source_input_manifest_matches_freeze",
            bool(expected_source_manifest)
            and expected_source_manifest == input_fingerprint.get("manifest_sha256"),
            (
                f"expected={expected_source_manifest} "
                f"actual={input_fingerprint.get('manifest_sha256')}"
            ),
        ),
        check(
            "frozen_probe_schedule_matches_freeze",
            bool(expected_probe_schedule)
            and expected_probe_schedule == schedule_manifest.get("schedule_sha256"),
            (
                f"expected={expected_probe_schedule} "
                f"actual={schedule_manifest.get('schedule_sha256')}"
            ),
        ),
        check(
            "benchmark_command_matches_freeze",
            not expected_benchmark_command
            or expected_benchmark_command == command_sha256(benchmark_command),
            (
                f"expected={expected_benchmark_command} "
                f"actual={command_sha256(benchmark_command)}"
            ),
        ),
        check(
            "benchmark_executable_hash_recorded",
            bool(benchmark_executable.get("exists"))
            and bool(benchmark_executable.get("sha256")),
            (
                f"path={benchmark_executable.get('path')} "
                f"exists={benchmark_executable.get('exists')} "
                f"sha256={benchmark_executable.get('sha256')}"
            ),
        ),
        check(
            "benchmark_executable_matches_freeze",
            not expected_benchmark_executable
            or expected_benchmark_executable == benchmark_executable.get("sha256"),
            (
                f"expected={expected_benchmark_executable} "
                f"actual={benchmark_executable.get('sha256')}"
            ),
        ),
        check(
            "git_commit_matches_freeze",
            not expected_git_commit or expected_git_commit == git.get("commit"),
            f"expected={expected_git_commit} actual={git.get('commit')}",
        ),
        check(
            "git_worktree_manifest_matches_freeze",
            not expected_git_worktree
            or expected_git_worktree == git.get("worktree_manifest_sha256"),
            (
                f"expected={expected_git_worktree} "
                f"actual={git.get('worktree_manifest_sha256')}"
            ),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--judge", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-command", default="")
    parser.add_argument("--judge-command", default="")
    parser.add_argument("--human-labels", type=pathlib.Path)
    parser.add_argument("--human-label-eval", type=pathlib.Path)
    parser.add_argument("--target-freeze", type=pathlib.Path)
    parser.add_argument("--ablation", type=parse_ablation, action="append", default=[])
    parser.add_argument("--ablation-plan", type=pathlib.Path)
    parser.add_argument("--judge-media-smoke", type=pathlib.Path)
    parser.add_argument(
        "--freeze-file",
        type=pathlib.Path,
        help="Pinned source/probe freeze JSON for final release verification.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero unless the release gate status is pass.",
    )
    args = parser.parse_args()

    summary = load_json(args.summary)
    judge = load_json(args.judge)
    judge_media_smoke = (
        load_json(args.judge_media_smoke) if args.judge_media_smoke else None
    )
    ablation_plan = load_json(args.ablation_plan) if args.ablation_plan else None
    if ablation_plan is not None:
        ablation_plan["path"] = str(args.ablation_plan)
    manifest = probe_manifest(summary)
    schedule_manifest = probe_schedule_manifest(summary)
    input_fingerprint = source_input_fingerprint(summary)
    protocol_freeze = load_protocol_freeze(args.freeze_file)
    source_ids = source_id_audit(summary)
    environment_snapshot = load_benchmark_environment_snapshot(args.summary)
    benchmark_status = load_optional_json(args.summary.parent / "benchmark_status.json")
    early_manifest = load_optional_json(
        args.summary.parent / "early_judge" / "early_judge_manifest.json"
    )
    early = early_judge_summary(benchmark_status, early_manifest)
    costs = cost_summary(summary, judge)
    claim_support = claim_support_summary(judge)
    ablation_names = [name for name, _, _ in args.ablation]
    git = git_info()
    benchmark_executable = command_executable_artifact(args.benchmark_command)

    checks: list[dict[str, Any]] = []
    checks.extend(git_provenance_checks(git))
    checks.extend(command_protocol_checks(args.benchmark_command, args.judge_command))
    checks.extend(
        command_summary_consistency_checks(
            args.benchmark_command, summary, args.summary
        )
    )
    checks.extend(
        command_judge_consistency_checks(
            args.judge_command, judge, args.summary, args.judge
        )
    )
    checks.extend(source_input_checks(input_fingerprint))
    checks.extend(
        protocol_freeze_checks(
            protocol_freeze,
            input_fingerprint,
            schedule_manifest,
            args.benchmark_command,
            benchmark_executable,
            git,
        )
    )
    checks.extend(source_id_audit_checks(source_ids))
    checks.extend(benchmark_environment_checks(environment_snapshot))
    checks.extend(early_judge_checks(early))
    checks.extend(summary_protocol_checks(summary))
    checks.extend(judge_protocol_checks(judge))
    checks.extend(judge_media_smoke_checks(judge_media_smoke, args.judge_command))
    checks.extend(claim_support_checks(claim_support))
    checks.extend(cross_artifact_checks(summary, judge, args.summary))
    human = load_json(args.human_labels) if args.human_labels else None
    human_eval = load_json(args.human_label_eval) if args.human_label_eval else None
    target_freeze = load_json(args.target_freeze) if args.target_freeze else None
    checks.extend(human_label_checks(human, args.human_labels, args.summary, summary))
    checks.extend(
        frozen_target_artifact_checks(
            target_freeze,
            args.target_freeze,
            human_eval,
            args.human_label_eval,
            summary,
            args.summary,
        )
    )
    checks.extend(ablation_checks(ablation_names))
    checks.extend(ablation_plan_checks(ablation_names, ablation_plan))
    checks.extend(cost_checks(costs))

    ablations = []
    ablation_plan_cases_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(ablation_plan, dict):
        cases = ablation_plan.get("cases", [])
        if isinstance(cases, list):
            ablation_plan_cases_by_name = {
                str(case.get("name")): case
                for case in cases
                if isinstance(case, dict) and case.get("name")
            }
    for name, summary_path, judge_path in args.ablation:
        ablation_summary = load_json(summary_path)
        ablation_judge = load_json(judge_path)
        ablation_manifest = probe_manifest(ablation_summary)
        ablation_schedule_manifest = probe_schedule_manifest(ablation_summary)
        ablation_source_input_fingerprint = source_input_fingerprint(ablation_summary)
        plan_case = ablation_plan_cases_by_name.get(name, {})
        ablation_benchmark_command = str(plan_case.get("benchmark_command", "") or "")
        ablation_judge_command = str(plan_case.get("judge_command", "") or "")
        command_checks: list[dict[str, Any]] = []
        if ablation_benchmark_command:
            command_checks.extend(
                command_summary_consistency_checks(
                    ablation_benchmark_command, ablation_summary, summary_path
                )
            )
            plan_executable = plan_case.get("benchmark_executable", {})
            if not isinstance(plan_executable, dict):
                plan_executable = {}
            actual_executable = command_executable_artifact(ablation_benchmark_command)
            command_checks.append(
                check(
                    "ablation_benchmark_executable_matches_plan",
                    bool(plan_executable.get("sha256"))
                    and plan_executable.get("sha256")
                    == actual_executable.get("sha256"),
                    (
                        f"planned={plan_executable.get('sha256')} "
                        f"actual={actual_executable.get('sha256')} "
                        f"path={actual_executable.get('path')}"
                    ),
                )
            )
        if ablation_judge_command:
            command_checks.extend(
                command_judge_consistency_checks(
                    ablation_judge_command, ablation_judge, summary_path, judge_path
                )
            )
        command_check_failures = [
            item for item in command_checks if item.get("status") != "pass"
        ]
        ablations.append(
            {
                "name": name,
                "summary_path": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
                "judge_path": str(judge_path),
                "judge_sha256": file_sha256(judge_path),
                "benchmark_command": ablation_benchmark_command,
                "judge_command": ablation_judge_command,
                "early_judge": {
                    "enabled": bool(plan_case.get("early_judge_enabled")),
                    "skip_reason": str(
                        plan_case.get("early_judge_skip_reason", "") or ""
                    ),
                    "manifest_path": str(
                        plan_case.get("early_judge_manifest_path", "") or ""
                    ),
                    "command_recorded": bool(plan_case.get("early_judge_command")),
                },
                "command_check_failures": command_check_failures,
                "frozen_probe_manifest_sha256": ablation_manifest["manifest_sha256"],
                "frozen_probe_schedule_sha256": ablation_schedule_manifest[
                    "schedule_sha256"
                ],
                "source_input_manifest_sha256": ablation_source_input_fingerprint.get(
                    "manifest_sha256"
                ),
                "knobs": ablation_summary.get("knobs"),
                "judge_provider": ablation_judge.get("judge_provider"),
                "judge_base_url": ablation_judge.get("judge_base_url"),
                "judge_repetitions": ablation_judge.get("judge_repetitions"),
                "remote_provider_allowed": ablation_judge.get("remote_provider_allowed"),
                "probe_count": ablation_judge.get("probe_count"),
                "judge_protocol": ablation_judge.get("protocol", {}),
                "judge_media_capabilities": ablation_judge.get(
                    "judge_media_capabilities", {}
                ),
                "quality": ablation_judge.get("quality"),
                "tokens": ablation_judge.get("tokens"),
                "fairness_checks": ablation_judge.get("fairness_checks"),
                "media_attachments": ablation_judge.get("media_attachments"),
                "packet_randomization": packet_randomization_summary(ablation_judge),
                "judge_repetition_consistency": judge_repetition_consistency_summary(
                    ablation_judge
                ),
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
        ablation_artifact_checks(
            ablations,
            schedule_manifest["schedule_sha256"],
            input_fingerprint.get("manifest_sha256", ""),
            summary,
        )
    )

    status_counts: dict[str, int] = {}
    for item in checks:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    output = {
        "schema": "cortext_julie_release_protocol_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy": {
            "public_safe": False,
            "public_aggregate_summary_safe": True,
            "excludes_private_text": True,
            "excludes_judge_reason_strings": True,
            "private_artifacts_remain_local": True,
            "why_private": "contains local paths and exact local commands for reproducibility",
        },
        "git": git,
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
            "benchmark_status_path": str(args.summary.parent / "benchmark_status.json"),
            "benchmark_status_sha256": (
                benchmark_status.get("sha256", "") if benchmark_status else ""
            ),
            "early_judge_manifest_path": str(
                args.summary.parent / "early_judge" / "early_judge_manifest.json"
            ),
            "early_judge_manifest_sha256": (
                early_manifest.get("sha256", "") if early_manifest else ""
            ),
            "human_labels_path": str(args.human_labels) if args.human_labels else "",
            "human_label_eval_path": (
                str(args.human_label_eval) if args.human_label_eval else ""
            ),
            "target_freeze_path": str(args.target_freeze) if args.target_freeze else "",
            "ablation_plan_path": str(args.ablation_plan) if args.ablation_plan else "",
            "judge_media_smoke_path": (
                str(args.judge_media_smoke) if args.judge_media_smoke else ""
            ),
            "judge_media_smoke_sha256": (
                file_sha256(args.judge_media_smoke) if args.judge_media_smoke else ""
            ),
            "release_freeze_path": str(args.freeze_file) if args.freeze_file else "",
            "release_freeze_sha256": (
                file_sha256(args.freeze_file)
                if args.freeze_file and args.freeze_file.exists()
                else ""
            ),
            "benchmark_executable": benchmark_executable,
        },
        "source_run": {
            "input_dir_recorded": summary.get("input_dir"),
            "source_input_fingerprint": input_fingerprint,
            "source_id_audit": source_ids,
            "benchmark_environment_snapshot": environment_snapshot,
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
            "normal_rag_compaction_summary_policy": summary.get(
                "normal_rag_compaction_summary_policy"
            ),
            "active_history_token_budget": summary.get("active_history_token_budget"),
            "knobs": summary.get("knobs"),
            "daily_consolidation": summary.get("daily_consolidation"),
            "final_window_consolidated": summary.get("final_window_consolidated"),
            "daily_final_window_consolidated": summary.get(
                "daily_final_window_consolidated"
            ),
            "deep_consolidation": summary.get("deep_consolidation"),
        },
        "protocol_freeze": protocol_freeze,
        "frozen_probe_manifest": manifest,
        "frozen_probe_schedule": schedule_manifest,
        "early_judge": early,
        "judge": public_judge_summary(judge),
        "judge_media_smoke": {
            "schema": judge_media_smoke.get("schema"),
            "private_data_used": judge_media_smoke.get("private_data_used"),
            "selected_release_judge_model": judge_media_smoke.get(
                "selected_release_judge_model"
            ),
            "selection_reason": judge_media_smoke.get("selection_reason"),
        }
        if judge_media_smoke
        else None,
        "human_labels": public_human_label_summary(human),
        "human_label_eval": {
            "schema": human_eval.get("schema"),
            "n": human_eval.get("n"),
            "freeze_sha256": human_eval.get("freeze_sha256"),
            "token_vs_quality": human_eval.get("token_vs_quality"),
        }
        if human_eval
        else None,
        "judge_frozen_targets": {
            "schema": target_freeze.get("schema"),
            "probe_count": target_freeze.get("probe_count"),
            "freeze_sha256": target_freeze.get("freeze_sha256"),
            "labeling": target_freeze.get("labeling", {}),
        }
        if target_freeze
        else None,
        "claim_support": claim_support,
        "costs": costs,
        "ablation_plan": ablation_plan,
        "ablations": ablations,
        "public_aggregate_summary": {
            "schema": "cortext_julie_release_public_aggregate_v1",
            "public_safe": True,
            "aggregate_only": True,
            "processed": {
                "text": summary.get("processed_text_messages"),
                "media_attempted": summary.get("media_attempted"),
                "media_processed": summary.get("media_processed"),
                "audio_processed": summary.get("audio_processed"),
                "image_processed": summary.get("image_processed"),
                "video_processed": summary.get("video_processed"),
                "probe_count": summary.get("probe_count"),
            },
            "judge": public_judge_summary(judge),
            "judge_media_smoke": {
                "selected_release_judge_model": judge_media_smoke.get(
                    "selected_release_judge_model"
                ),
                "private_data_used": judge_media_smoke.get("private_data_used"),
            }
            if judge_media_smoke
            else None,
            "protocol_freeze": {
                "schema": protocol_freeze.get("schema"),
                "source_input_manifest_sha256": protocol_freeze.get(
                    "source_input_manifest_sha256"
                ),
                "frozen_probe_schedule_sha256": protocol_freeze.get(
                    "frozen_probe_schedule_sha256"
                ),
                "sha256": protocol_freeze.get("sha256"),
            }
            if protocol_freeze
            else None,
            "early_judge": early,
            "human_labels": public_human_label_summary(human),
            "claim_support": claim_support,
            "costs": costs,
            "ablations": [
                {
                    "name": row.get("name"),
                    "quality": row.get("quality"),
                    "tokens": row.get("tokens"),
                    "effect_vs_full": row.get("effect_vs_full"),
                    "processed": row.get("processed"),
                }
                for row in ablations
            ],
        },
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
    if args.require_pass and output["release_gate"]["overall_status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
