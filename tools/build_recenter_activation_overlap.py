#!/usr/bin/env python3
"""Build a fail-closed consolidation recenter overlap artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


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


def exact_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number != round(number):
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(number)


def build(
    profile: dict[str, Any],
    profile_sha256: str,
    database_sha256: str,
    benchmark_binary_sha256: str,
    corpus_sha256: str,
) -> dict[str, Any]:
    focus = float(profile["focus"])
    sensitivity = float(profile["sensitivity"])
    stability = float(profile["stability"])
    parameters = derived_parameters(focus, sensitivity, stability)
    recorded = profile.get("sparse_route_parameters")
    if not isinstance(recorded, dict):
        raise ValueError("profile lacks sparse_route_parameters")
    expected_recorded = {
        "route_capacity": parameters["C"],
        "backfill_batch_size": parameters["B"],
        "activation_identity_target": parameters["A"],
        "search_node_budget": parameters["public_query_node_budget"],
        "backfill_search_node_budget": parameters["construction_node_budget"],
        "backfill_search_effort": parameters["construction_queue_effort"],
    }
    for key, expected in expected_recorded.items():
        if exact_nonnegative_integer(recorded.get(key), key) != expected:
            raise ValueError(f"{key} does not match F/S/T-derived value")

    events = profile.get("consolidation_events")
    if not isinstance(events, list):
        raise ValueError("profile lacks consolidation_events")
    profiled = [
        event for event in events
        if isinstance(event, dict)
        and event.get("sqlite_sparse_route_recenter_overlap_profiled") is True
    ]
    if not profiled:
        raise ValueError("profile has no recenter overlap observations")

    observations = []
    for event in profiled:
        index = exact_nonnegative_integer(event.get("event_index"), "event_index")
        if event.get("sqlite_sparse_route_recenter_overlap_pair_valid") is not True:
            raise ValueError(f"event {index} has an invalid recenter overlap pair")
        failure = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_overlap_failure_code"),
            "overlap_failure_code",
        )
        if failure != 0:
            raise ValueError(f"event {index} has overlap failure code {failure}")
        if event.get("sqlite_sparse_route_recenter_succeeded") is not True:
            raise ValueError(f"event {index} did not recenter successfully")
        if event.get("sqlite_sparse_route_recenter_derived_centroid") is not True:
            raise ValueError(
                f"event {index} did not use a consolidation-derived centroid"
            )
        centroid_source = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_centroid_source"),
            "centroid_source",
        )
        if centroid_source not in (1, 2):
            raise ValueError(f"event {index} has an invalid centroid source")
        cluster_count = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_centroid_cluster_count"),
            "centroid_cluster_count",
        )
        member_count = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_centroid_member_count"),
            "centroid_member_count",
        )
        if member_count == 0 or (centroid_source == 1 and cluster_count == 0):
            raise ValueError(
                f"event {index} has an empty consolidation-derived centroid"
            )
        if centroid_source == 2 and cluster_count != 0:
            raise ValueError(
                f"event {index} mixes cluster and active-ring centroid sources"
            )
        pre = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_pre_activated_count"),
            "pre_activated_count",
        )
        post = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_post_activated_count"),
            "post_activated_count",
        )
        overlap = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_overlap_count"),
            "overlap_count",
        )
        pre_nodes = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_pre_node_count"),
            "pre_node_count",
        )
        post_nodes = exact_nonnegative_integer(
            event.get("sqlite_sparse_route_recenter_post_node_count"),
            "post_node_count",
        )
        if pre == 0 or post == 0:
            raise ValueError(f"event {index} has an empty valid pair")
        if pre > parameters["A"] or post > parameters["A"]:
            raise ValueError(f"event {index} exceeds F/S/T-derived A")
        if pre_nodes > parameters["public_query_node_budget"] or post_nodes > parameters["public_query_node_budget"]:
            raise ValueError(f"event {index} exceeds F/S/T-derived 5C")
        if overlap > min(pre, post):
            raise ValueError(f"event {index} overlap exceeds pair cardinality")
        unchanged = pre == post == overlap
        observations.append({
            "event_index": index,
            "pre": pre,
            "post": post,
            "overlap": overlap,
            "pre_nodes": pre_nodes,
            "post_nodes": post_nodes,
            "centroid_clusters": cluster_count,
            "centroid_members": member_count,
            "centroid_source": centroid_source,
            "unchanged": unchanged,
            "overlap_fraction": overlap / max(pre, post),
        })

    neutral = derived_parameters(0.5, 0.5, 0.5)
    unchanged_count = sum(int(row["unchanged"]) for row in observations)
    full_target_count = sum(
        int(row["pre"] == parameters["A"] and row["post"] == parameters["A"])
        for row in observations
    )
    return {
        "schema": "cortext_recenter_activation_overlap_v3",
        "status": "measured-recenter-does-not-change-activated-identity-set"
        if unchanged_count == len(observations)
        else "measured-recenter-changes-activated-identity-set",
        "decision": "reject-current-recenter-as-retrieval-centroid-reset"
        if unchanged_count == len(observations)
        else "recenter-materiality-requires-quality-and-cycle-evaluation",
        "processed_events": exact_nonnegative_integer(
            profile.get("processed_events"), "processed_events"
        ),
        "consolidation_runs": exact_nonnegative_integer(
            profile.get("consolidation_runs"), "consolidation_runs"
        ),
        "profiled_successful_recenters": len(observations),
        "invalid_profiled_pairs": 0,
        "first_profiled_recenter_event_index": observations[0]["event_index"],
        "last_profiled_recenter_event_index": observations[-1]["event_index"],
        "full_activation_target_recenters": full_target_count,
        "changed_activated_set_count": len(observations) - unchanged_count,
        "unchanged_activated_set_count": unchanged_count,
        "minimum_pre_post_overlap_fraction": min(
            row["overlap_fraction"] for row in observations
        ),
        "maximum_pre_activated_identities": max(row["pre"] for row in observations),
        "maximum_post_activated_identities": max(row["post"] for row in observations),
        "maximum_pre_node_rows": max(row["pre_nodes"] for row in observations),
        "maximum_post_node_rows": max(row["post_nodes"] for row in observations),
        "minimum_centroid_cluster_count": min(
            row["centroid_clusters"] for row in observations
        ),
        "minimum_centroid_member_count": min(
            row["centroid_members"] for row in observations
        ),
        "cluster_centroid_recenter_count": sum(
            row["centroid_source"] == 1 for row in observations
        ),
        "active_signal_ring_centroid_recenter_count": sum(
            row["centroid_source"] == 2 for row in observations
        ),
        "focus": focus,
        "sensitivity": sensitivity,
        "stability": stability,
        "derived_parameters": {
            **parameters,
            "neutral_C": neutral["C"],
            "neutral_B": neutral["B"],
            "neutral_A": neutral["A"],
            "neutral_logical_B_plus_one_only": neutral["logical_B_plus_one_only"],
        },
        "corpus_sha256": corpus_sha256,
        "raw_profile_sha256": profile_sha256,
        "database_sha256": database_sha256,
        "benchmark_binary_sha256": benchmark_binary_sha256,
        "required_follow_on": {
            "algorithm_target": "consolidation-driven-sparse-activation-centroid-reset",
            "fixed_activation_target": "A=2C+2B",
            "fixed_query_work": "5C",
            "structural_knob_points": 27,
            "production_shaped_knob_points": 9,
            "modality_and_source_id_agnostic": True,
            "raw_process_reset_gate_remains_independent": True,
        },
        "nonclaims": [
            "accepted retrieval cycle",
            "raw consolidation reset pass or waiver",
            "bounded restart",
            "Natural or Durable plateau",
            "whole-goal completion",
            "PR readiness",
            "merge",
            "release",
            "deployment",
            "publication",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--benchmark-binary", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        load_json(args.profile),
        file_sha256(args.profile),
        file_sha256(args.database),
        file_sha256(args.benchmark_binary),
        file_sha256(args.corpus),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
