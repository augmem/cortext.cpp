#!/usr/bin/env python3
"""Build a range-bound write-gate incidence attribution artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


WRITE_GATE_KEYS = (
    "WriteGate.flush_trigger",
    "WriteGate.spike_bypass",
    "WriteGate.accumulator_available",
    "WriteGate.n_signals",
    "WriteGate.coverage",
    "WriteGate.window_score",
    "WriteGate.threshold_dynamic",
    "WriteGate.refractory_multiplier",
    "WriteGate.write_scale",
    "WriteGate.effective_threshold",
    "WriteGate.score_margin",
    "WriteGate.force_write",
    "WriteGate.write_accumulator",
    "WriteGate.reason_code",
)
PROCESS_RATIO_RANGE = (0.90, 1.10)
THROUGHPUT_RATIO_RANGE = (0.90, 1.10)
RELATIVE_SLOPE_MAX = 0.02
RELATIVE_UPPER_MAX = 0.05


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lround_positive(value: float) -> int:
    return math.floor(value + 0.5)


def derived_parameters(focus: float, sensitivity: float, stability: float) -> dict[str, int]:
    for label, value in (
        ("focus", focus),
        ("sensitivity", sensitivity),
        ("stability", stability),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{label} must be finite and within [0, 1]")
    capacity = lround_positive(
        256 + 256 * focus + 128 * sensitivity + 128 * stability
    )
    backfill = lround_positive(
        64 + 64 * focus + 32 * sensitivity + 32 * stability
    )
    return {
        "C": capacity,
        "B": backfill,
        "A": 2 * capacity + 2 * backfill,
        "public_query_node_budget": 5 * capacity,
        "construction_node_budget": capacity + backfill,
        "construction_queue_effort": 2 * backfill,
        "logical_B_plus_one_only": backfill + 1,
    }


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def operation_value(row: dict[str, Any], key: str) -> float:
    operations = row.get("operation_ms")
    if not isinstance(operations, dict) or key not in operations:
        raise ValueError(f"event {row.get('event_index')} lacks {key}")
    return finite_number(operations[key], key)


def integer_sum(rows: Sequence[dict[str, Any]], key: str) -> int:
    value = sum(operation_value(row, key) for row in rows)
    rounded = round(value)
    if value < 0.0 or not math.isclose(value, rounded, abs_tol=1e-9):
        raise ValueError(f"{key} aggregate must be a nonnegative integer")
    return int(rounded)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("attribution range is empty")
    reason_counts = {reason: 0 for reason in range(1, 7)}
    scored_rows = []
    for row in rows:
        reason_value = operation_value(row, "WriteGate.reason_code")
        reason = int(reason_value)
        if reason_value != reason or reason not in reason_counts:
            raise ValueError("WriteGate.reason_code must be an integer in [1, 6]")
        reason_counts[reason] += 1
        if reason in (5, 6):
            scored_rows.append(row)
    if sum(reason_counts.values()) != len(rows):
        raise ValueError("write-gate reason coverage is incomplete")

    boundaries = integer_sum(rows, "WriteGate.flush_trigger")
    writes = integer_sum(rows, "WriteGate.write_accumulator")
    forced = integer_sum(rows, "WriteGate.force_write")
    if forced != reason_counts[4]:
        raise ValueError("forced-write count differs from reason code")
    if writes != reason_counts[4] + reason_counts[5]:
        raise ValueError("write count differs from forced plus scored accepts")
    if boundaries != reason_counts[3] + reason_counts[4] + reason_counts[5] + reason_counts[6]:
        raise ValueError("boundary count differs from write-gate exit classes")
    if writes <= 0 or not scored_rows:
        raise ValueError("attribution range needs writes and scored decisions")

    def summed(key: str) -> float:
        return sum(operation_value(row, key) for row in rows)

    def scored_mean(key: str) -> float:
        return statistics.mean(operation_value(row, key) for row in scored_rows)

    return {
        "start_event_index": int(rows[0]["event_index"]),
        "end_event_index_inclusive": int(rows[-1]["event_index"]),
        "events": len(rows),
        "boundaries": boundaries,
        "writes": writes,
        "forced_writes": forced,
        "score_accepted": reason_counts[5],
        "score_rejected": reason_counts[6],
        "mean_scored_window_score": scored_mean("WriteGate.window_score"),
        "mean_scored_effective_threshold": scored_mean(
            "WriteGate.effective_threshold"
        ),
        "mean_scored_margin": scored_mean("WriteGate.score_margin"),
        "memory_storage_ms_per_write": summed(
            "cortext::operations::MemoryStorage"
        ) / writes,
        "supersession_ms_per_write": summed(
            "MemoryStorage.supersession_edges"
        ) / writes,
        "current_candidates_per_write": summed(
            "MemoryStorage.supersession_current_candidate_count"
        ) / writes,
        "current_rows_visited_per_write": summed(
            "MemoryStorage.supersession_current_rows_visited"
        ) / writes,
        "sparse_route_node_rows_per_write": summed(
            "MemoryStorage.supersession_sparse_route_node_rows"
        ) / writes,
    }


def mature_suffix_shape(audit: dict[str, Any]) -> dict[str, Any]:
    plateau = audit.get("plateau")
    candidates = plateau.get("candidate_failures") if isinstance(plateau, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("plateau audit lacks candidate failures")
    eligible = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if (
            candidate.get("operation_failure_count") == 0
            and candidate.get("counter_failure_count") == 0
            and candidate.get("height_passed") is True
            and candidate.get("store_growth_passed") is True
            and PROCESS_RATIO_RANGE[0]
            <= finite_number(candidate.get("process_ratio"), "process_ratio")
            <= PROCESS_RATIO_RANGE[1]
            and THROUGHPUT_RATIO_RANGE[0]
            <= finite_number(candidate.get("throughput_ratio"), "throughput_ratio")
            <= THROUGHPUT_RATIO_RANGE[1]
            and finite_number(candidate.get("relative_slope"), "relative_slope")
            <= RELATIVE_SLOPE_MAX
            and finite_number(candidate.get("relative_upper"), "relative_upper")
            <= RELATIVE_UPPER_MAX
        ):
            eligible.append(candidate)
    if not eligible:
        raise ValueError("plateau audit has no mature suffix passing shape gates")
    selected = min(eligible, key=lambda item: int(item["plateau_start_event"]))
    return {
        "plateau_start_event": int(selected["plateau_start_event"]),
        "plateau_window_count": int(selected["plateau_window_count"]),
        "process_ratio": float(selected["process_ratio"]),
        "throughput_ratio": float(selected["throughput_ratio"]),
        "relative_slope": float(selected["relative_slope"]),
        "relative_upper": float(selected["relative_upper"]),
        "operation_failure_count": 0,
        "counter_failure_count": 0,
        "height_passed": True,
        "store_growth_passed": True,
        "raw_consolidation_mode": selected.get("consolidation_mode"),
        "raw_consolidation_epoch_passed": selected.get(
            "consolidation_epoch_passed"
        ),
    }


def build(
    profile: dict[str, Any], profile_sha256: str,
    plateau_audit: dict[str, Any], plateau_audit_sha256: str,
    window_size: int = 500, window_count: int = 5,
) -> dict[str, Any]:
    rows = profile.get("working_set_curve")
    if not isinstance(rows, list) or not rows:
        raise ValueError("profile has no working_set_curve")
    if profile.get("processed_events") != len(rows):
        raise ValueError("processed_events differs from row count")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("event_index") != index:
            raise ValueError("event indices must be contiguous and zero-based")
        for key in WRITE_GATE_KEYS:
            operation_value(row, key)
    if window_size <= 0 or window_count <= 0:
        raise ValueError("window size and count must be positive")
    span = window_size * window_count
    if len(rows) < 2 * span:
        raise ValueError("profile is too short for disjoint first/final ranges")

    focus = finite_number(profile.get("focus"), "focus")
    sensitivity = finite_number(profile.get("sensitivity"), "sensitivity")
    stability = finite_number(profile.get("stability"), "stability")
    parameters = derived_parameters(focus, sensitivity, stability)
    route = profile.get("sparse_route_parameters")
    expected_route = {
        "route_capacity": parameters["C"],
        "backfill_batch_size": parameters["B"],
        "activation_identity_target": parameters["A"],
        "search_node_budget": parameters["public_query_node_budget"],
        "backfill_search_node_budget": parameters["construction_node_budget"],
        "backfill_search_effort": parameters["construction_queue_effort"],
    }
    if not isinstance(route, dict) or any(
        route.get(key) != value for key, value in expected_route.items()
    ):
        raise ValueError("profile sparse-route parameters differ from F/S/T formulas")

    first = aggregate(rows[:span])
    final = aggregate(rows[-span:])
    full = aggregate(rows)
    boundary_ratio = final["boundaries"] / first["boundaries"]
    write_ratio = final["writes"] / first["writes"]
    incidence_is_flat = (
        0.90 <= boundary_ratio <= 1.10
        and 0.90 <= write_ratio <= 1.10
    )
    neutral = derived_parameters(0.5, 0.5, 0.5)
    suffix = mature_suffix_shape(plateau_audit)
    plateau = plateau_audit.get("plateau")
    whole_plateau_passed = bool(
        isinstance(plateau, dict) and plateau.get("passed") is True
    )
    raw_reset_passed = suffix["raw_consolidation_epoch_passed"] is True
    status = (
        "attribution-only-write-gate-rejected-as-growth-source"
        if incidence_is_flat
        else "attribution-only-write-gate-growth-source-unresolved"
    )
    return {
        "schema": "cortext_write_gate_incidence_attribution_v2",
        "status": status,
        "profile_sha256": profile_sha256,
        "plateau_audit_sha256": plateau_audit_sha256,
        "processed_events": len(rows),
        "consolidation_runs": int(profile.get("consolidation_runs", 0)),
        "focus": focus,
        "sensitivity": sensitivity,
        "stability": stability,
        "window_contract": {
            "window_size": window_size,
            "window_count": window_count,
            "first_range": [0, span - 1],
            "final_range": [len(rows) - span, len(rows) - 1],
            "final_range_is_end_anchored": True,
            "ranges_are_disjoint": True,
        },
        "derived_parameters": {
            **parameters,
            "neutral_C": neutral["C"],
            "neutral_B": neutral["B"],
            "neutral_A": neutral["A"],
            "neutral_logical_B_plus_one_only": neutral["logical_B_plus_one_only"],
        },
        "first_range_aggregate": first,
        "final_range_aggregate": final,
        "full_run": full,
        "incidence_ratios": {
            "final_over_first_boundaries": boundary_ratio,
            "final_over_first_writes": write_ratio,
            "flat_within_ten_percent": incidence_is_flat,
        },
        "mature_suffix_shape": suffix,
        "decision": {
            "write_incidence_rejected_as_growth_source": incidence_is_flat,
            "write_incidence_growth_source": (
                False if incidence_is_flat else None
            ),
            "observed_growth_surface": (
                "MemoryStorage sparse-route visited-node warm-up to the fixed 5C envelope"
                if incidence_is_flat else "unresolved"
            ),
            "mature_suffix_process_plateau_shape_passed": True,
            "raw_reset_gate_passed": raw_reset_passed,
            "raw_reset_gate_waived": False,
            "whole_plateau_passed": whole_plateau_passed,
            "runtime_repair_authorized_from_this_artifact": False,
        },
        "required_follow_on": {
            "structural_knob_points": 27,
            "production_shaped_knob_points": 9,
            "modality_and_source_id_agnostic": True,
            "activated_identity_overlap": "pending",
        },
        "nonclaims": [
            "whole-engine plateau",
            "raw reset waiver",
            "retrieval full-cycle activated-identity overlap",
            "bounded restart",
            "PR readiness",
            "merge",
            "release",
            "deployment",
            "publication",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--plateau-audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = build(
        load_json(args.profile), file_sha256(args.profile),
        load_json(args.plateau_audit), file_sha256(args.plateau_audit),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
