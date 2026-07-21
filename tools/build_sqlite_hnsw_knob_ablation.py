#!/usr/bin/env python3
"""Build the content-addressed SQLite HNSW knob-ablation aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import struct
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CORPUS_KNOB_POINTS = {
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.5, 0.5, 0.5),
    (0.0, 0.5, 0.5),
    (1.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 1.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.5, 0.5, 1.0),
}
REQUIRED_PROCESSED_EVENTS_PER_RUN = 4000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable_identity(path: Path, target: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing executable for {target}: {path}")
    return {
        "target": target,
        "filename": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=reject_duplicates)


def clamped_knob(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return min(1.0, max(0.0, result))


def expected_parameters(focus: float, sensitivity: float, stability: float) -> dict:
    f = clamped_knob(focus, "focus")
    s = clamped_knob(sensitivity, "sensitivity")
    t = clamped_knob(stability, "stability")
    # C++ std::lround rounds halfway cases away from zero. These expressions
    # are positive, so floor(x + 0.5) is the exact evaluator-side equivalent.
    capacity = math.floor(256 + 256 * f + 128 * s + 128 * t + 0.5)
    backfill = math.floor(64 + 64 * f + 32 * s + 32 * t + 0.5)
    activation_node_min = 8 * capacity
    activation_step = max(2, backfill // 16)
    activation_target = 2 * capacity + 2 * backfill
    return {
        "backfill_batch_size": backfill,
        "fallback_hydration_signal_limit": backfill,
        "backfill_search_effort": 2 * backfill,
        "backfill_search_node_budget": capacity + backfill,
        "bootstrap_limit": 2 * capacity,
        "graph_level_zero_links": max(16, capacity // 4),
        "graph_neighbor_count": max(8, backfill // 2),
        "family_exact_comparison_limit": 2 * capacity,
        "hnsw_construction_effort": max(
            32, math.floor(capacity * 25 / 64 + 0.5)
        ),
        "hnsw_query_effort": max(
            capacity, math.floor(capacity * 5 / 2 + 0.5)
        ),
        "maximum_level": math.ceil(math.log2(capacity)),
        "reciprocal_update_count": max(2, backfill // 16),
        "route_capacity": capacity,
        "activation_identity_target": activation_target,
        "activation_snapshot_capacity": activation_target,
        "total_query_row_budget": 9 * capacity + activation_target,
        "activation_search_effort_min": 8 * capacity,
        "activation_search_effort_step": activation_step,
        "activation_search_node_budget_min": activation_node_min,
        "activation_search_node_budget_step": activation_step,
        "search_effort": 9 * capacity,
        "search_expansion_batch": max(8, backfill // 4),
        "search_node_budget": 9 * capacity,
        "shadow_cache_capacity": 24 * capacity,
    }


def expected_operational_limits(parameters: dict) -> dict:
    """Name the independently bounded live-dirty and historical surfaces."""
    capacity = parameters["route_capacity"]
    backfill = parameters["backfill_batch_size"]
    return {
        **parameters,
        "historical_backfill_drain_limit": backfill,
        "historical_backfill_overflow_probe_limit": backfill + 1,
        "dirty_identity_drain_limit": capacity,
        "dirty_identity_overflow_probe_limit": capacity + 1,
    }


def expected_active_epoch_limits(
    focus: float, sensitivity: float, stability: float
) -> dict:
    f = clamped_knob(focus, "focus")
    s = clamped_knob(sensitivity, "sensitivity")
    t = clamped_knob(stability, "stability")
    capacity = math.floor(256 + 256 * f + 128 * s + 128 * t + 0.5)
    return {
        "event_count": capacity,
        "mutation_count": 64 * capacity,
        "allocated_bytes": 128 * 1024 * capacity,
        "row_batch_size": math.floor(
            64 + 64 * f + 32 * s + 32 * t + 0.5
        ),
    }


def metric_values(profile: dict, name: str) -> list[float]:
    return [
        float(row.get("operation_ms", {}).get(name, 0.0))
        for row in profile["working_set_curve"]
    ]


def candidate_quality_passed(audit: dict) -> bool:
    return (
        audit.get("passed") is True
        and audit.get("candidate_complete") is True
        and audit.get("candidate_quality_passed") is True
        and audit.get("deterministic_tie_order") is True
        and float(audit.get("exact_identity_recall_at_k", 0.0)) >= 0.998
        and int(audit.get("exact_top1_miss_count", 2)) <= 1
        and float(audit.get("semantic_coverage", 0.0)) >= 0.95
    )


def quality_classification(
    *,
    sparse_traversed: bool,
    evaluated_recenter_count: int,
    node_rows: int,
    published_active_count: int | None,
    route_capacity: int,
) -> str:
    if sparse_traversed:
        return (
            "route-lifecycle-evaluated-system-quality-passed"
            if evaluated_recenter_count > 0
            else "system-quality-passed-sparse-route-traversed-recenter-unproven"
        )
    if (
        published_active_count is not None
        and published_active_count <= route_capacity
    ):
        return "system-quality-passed-published-below-threshold-sparse-route-unevaluated"
    if node_rows == 0:
        return "system-quality-passed-below-threshold-sparse-route-unevaluated"
    return "system-quality-passed-dirty-or-unpublished-sparse-route-unevaluated"


def top1_miss_cluster_audit(experiments: list[dict]) -> dict:
    misses = []
    for experiment in experiments:
        fingerprints = experiment["observed"].get("top1_miss_fingerprints")
        if not isinstance(fingerprints, list):
            raise ValueError(
                f"top-1 miss fingerprints are missing for {experiment['id']}"
            )
        if len(fingerprints) != experiment["observed"]["exact_top1_miss_count"]:
            raise ValueError(
                f"top-1 miss fingerprint count differs for {experiment['id']}"
            )
        for fingerprint in fingerprints:
            if not isinstance(fingerprint, dict):
                raise ValueError("top-1 miss fingerprint is not an object")
            age_events = fingerprint.get("control_top1_age_events")
            if not isinstance(age_events, int) or age_events < 0:
                raise ValueError("top-1 miss lacks a nonnegative memory age")
            misses.append(
                {
                    **fingerprint,
                    "run_id": experiment["id"],
                    "knobs": experiment["knobs"],
                    "memory_age_quartile": min(
                        3,
                        age_events * 4
                        // max(1, experiment["observed"]["processed_events"]),
                    ),
                }
            )

    dimensions = {
        "modality": lambda miss: miss["modality"],
        "source_id": lambda miss: miss["source_id_blake3"],
        "memory_age_quartile": lambda miss: miss["memory_age_quartile"],
        "knob_point": lambda miss: tuple(miss["knobs"]),
    }
    maximum_bucket_counts = {}
    clustered_dimensions = []
    for name, key in dimensions.items():
        counts: dict[object, int] = {}
        for miss in misses:
            bucket = key(miss)
            counts[bucket] = counts.get(bucket, 0) + 1
        maximum_bucket_counts[name] = max(counts.values(), default=0)
        if maximum_bucket_counts[name] > 1:
            clustered_dimensions.append(name)
    return {
        "passed": not clustered_dimensions,
        "status": "no-misses" if not misses else "misses-unclustered",
        "miss_count": len(misses),
        "cluster_definition": (
            "two or more exact top-1 misses sharing a modality, opaque source, "
            "memory-age quartile, or exact F/S/T point"
        ),
        "maximum_bucket_counts": maximum_bucket_counts,
        "clustered_dimensions": clustered_dimensions,
        "misses": misses,
    }


def decode_link_levels(blob: bytes, expected_level: int) -> list[list[int]]:
    offset = 0

    def read(fmt: str) -> int:
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(blob):
            raise ValueError("truncated sparse-route link blob")
        value = struct.unpack_from(fmt, blob, offset)[0]
        offset += size
        return int(value)

    level_count = read("<I")
    if level_count != expected_level + 1:
        raise ValueError("sparse-route link level count does not match node level")
    levels = []
    for _ in range(level_count):
        count = read("<I")
        levels.append([read("<q") for _ in range(count)])
    if offset != len(blob):
        raise ValueError("trailing sparse-route link bytes")
    return levels


def topology(db_path: Path) -> dict:
    with sqlite3.connect(db_path) as connection:
        max_level, node_rows, upper_rows = connection.execute(
            "SELECT MAX(level), COUNT(*), "
            "SUM(CASE WHEN level > 0 THEN 1 ELSE 0 END) "
            "FROM cortext_sparse_route_nodes"
        ).fetchone()
        dirty_rows = connection.execute(
            "SELECT COUNT(*) FROM cortext_sparse_route_dirty"
        ).fetchone()[0]
        meta = connection.execute(
            "SELECT max_level, active_count, generation "
            "FROM cortext_sparse_route_meta WHERE singleton = 1"
        ).fetchone()
        encoded_links = connection.execute(
            "SELECT level, links FROM cortext_sparse_route_nodes "
            "WHERE active = 1 AND level > 0 ORDER BY memory_id"
        ).fetchall()
    decoded_links = [decode_link_levels(blob, level) for level, blob in encoded_links]
    upper_degrees = [
        sum(len(level_links) for level_links in levels[1:])
        for levels in decoded_links
    ]
    return {
        "observed_max_level": max_level,
        "node_rows": node_rows,
        "upper_level_node_rows": upper_rows or 0,
        "remaining_dirty_rows": dirty_rows,
        "published_max_level": meta[0] if meta else None,
        "published_active_count": meta[1] if meta else None,
        "published_generation": meta[2] if meta else None,
        "upper_level_nonempty_adjacency_rows": sum(
            degree > 0 for degree in upper_degrees
        ),
        "maximum_upper_level_adjacency_degree": max(upper_degrees, default=0),
    }


def junit(path: Path, test_binary: dict) -> dict:
    suite = ET.parse(path).getroot().find("testsuite")
    if suite is None:
        raise ValueError(f"no testsuite in {path}")
    return {
        "result_artifact": f"sanitized/{path.name}",
        "result_sha256": sha256(path),
        "assertions": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "test_binary": test_binary,
    }


def parse_run(spec: str) -> tuple[str, Path, Path, Path]:
    pieces = spec.split(",", 3)
    if len(pieces) != 4:
        raise ValueError("--run must be label,profile,audit,db")
    return pieces[0], Path(pieces[1]), Path(pieces[2]), Path(pieces[3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--structural-junit", type=Path, required=True)
    parser.add_argument("--backfill-junit", type=Path, required=True)
    parser.add_argument("--restart-junit", type=Path, required=True)
    parser.add_argument("--core-knob-junit", type=Path, required=True)
    parser.add_argument("--invariance-junit", type=Path, required=True)
    parser.add_argument("--benchmark-binary", type=Path, required=True)
    parser.add_argument("--test-binary", type=Path, required=True)
    args = parser.parse_args()

    benchmark_binary = executable_identity(
        args.benchmark_binary, "cortext_chat_replay_live_run"
    )
    test_binary = executable_identity(args.test_binary, "cortext_tests")

    source_paths = [
        "src/operations/sparse_retrieval_knobs_internal.hpp",
        "src/operations/sparse_retrieval_route_internal.hpp",
        "src/operations/sparse_retrieval_route_internal.cpp",
        "src/operations/sparse_retrieval_route_sqlite_internal.hpp",
        "src/operations/sparse_retrieval_route_sqlite_internal.cpp",
        "src/operations/rif_active_epoch_cache_internal.hpp",
        "src/operations/historical_surface_search_cache_internal.hpp",
        "src/operations/graph_retrieval.cpp",
        "src/operations/memory_storage.cpp",
        "src/cortext.cpp",
        "src/signal_processor.cpp",
        "tests/operations_graph_retrieval.test.cpp",
        "tests/operations_memory_storage.test.cpp",
        "tests/cortext.test.cpp",
        "tests/signal_processor.test.cpp",
        "examples/benchmark/chat_replay_live_run.cpp",
        "tools/audit_public_retrieval_control.py",
        "tools/audit_storage_cost_profile.py",
        "tools/build_sqlite_hnsw_knob_ablation.py",
    ]
    experiments = []
    for spec in args.run:
        label, profile_path, audit_path, db_path = parse_run(spec)
        profile = load_json(profile_path)
        audit = load_json(audit_path)
        if "experimental_sparse_node_envelope_formula" in profile:
            raise ValueError(
                f"production-default run unexpectedly used an experiment selector for {label}"
            )
        if profile.get("processed_events") != REQUIRED_PROCESSED_EVENTS_PER_RUN:
            raise ValueError(
                f"{label} must contain exactly "
                f"{REQUIRED_PROCESSED_EVENTS_PER_RUN} processed events"
            )
        expected = expected_parameters(
            profile["focus"], profile["sensitivity"], profile["stability"]
        )
        if profile["sparse_route_parameters"] != expected:
            raise ValueError(f"resolved parameters do not match formulas for {label}")
        expected_epoch_limits = expected_active_epoch_limits(
            profile["focus"], profile["sensitivity"], profile["stability"]
        )
        if profile.get("active_epoch_limits") != expected_epoch_limits:
            raise ValueError(
                f"resolved active-epoch limits do not match formulas for {label}"
            )
        if not candidate_quality_passed(audit):
            raise ValueError(f"candidate quality audit did not pass for {label}")
        visits = metric_values(profile, "GraphRetrieve.sqlite_sparse_route_node_rows")
        snapshot_visits = metric_values(
            profile,
            "GraphRetrieve.sqlite_sparse_route_activation_snapshot_rows",
        )
        activated = metric_values(
            profile, "GraphRetrieve.sqlite_sparse_route_activated_identities"
        )
        node_budgets = metric_values(
            profile, "GraphRetrieve.sqlite_sparse_route_search_node_budget"
        )
        if any(
            visited > node_budget
            for visited, node_budget in zip(visits, node_budgets)
            if visited > 0.0
        ):
            raise ValueError(f"dynamic node ceiling exceeded for {label}")
        if any(
            visited > expected["activation_snapshot_capacity"]
            for visited in snapshot_visits
        ):
            raise ValueError(f"activation snapshot ceiling exceeded for {label}")
        if any(
            canonical + snapshot > expected["total_query_row_budget"]
            for canonical, snapshot in zip(visits, snapshot_visits)
        ):
            raise ValueError(f"total query-row ceiling exceeded for {label}")
        if any(
            identity_count > expected["activation_identity_target"]
            for identity_count in activated
        ):
            raise ValueError(f"activation identity target exceeded for {label}")
        search_failures = metric_values(
            profile, "GraphRetrieve.sqlite_sparse_route_search_failure_code"
        )
        route_metadata_open_rows = metric_values(
            profile, "GraphRetrieve.sqlite_sparse_route_restart_rows"
        )
        backfill_rows = [
            int(row.get("sqlite_sparse_route_backfill_rows", 0))
            for row in profile["consolidation_events"]
        ]
        backfill_failures = [
            int(row.get("sqlite_sparse_route_backfill_failure_count", 0))
            for row in profile["consolidation_events"]
        ]
        successful_resets = [
            event
            for event in profile["consolidation_events"]
            if event.get("sqlite_sparse_route_recenter_succeeded") is True
            and float(event.get("sqlite_sparse_route_activation_search_effort", 0.0))
            == expected["activation_search_effort_min"]
            and float(event.get("sqlite_sparse_route_activation_node_budget", 0.0))
            == expected["activation_search_node_budget_min"]
        ]
        valid_overlap_events = []
        changed_overlap_events = []
        overlap_ratios = []
        for event in successful_resets:
            centroid_source = event.get(
                "sqlite_sparse_route_recenter_centroid_source"
            )
            centroid_cluster_count = event.get(
                "sqlite_sparse_route_recenter_centroid_cluster_count"
            )
            centroid_member_count = event.get(
                "sqlite_sparse_route_recenter_centroid_member_count"
            )
            if (
                event.get("sqlite_sparse_route_recenter_derived_centroid")
                is not True
                or centroid_source not in (1.0, 2.0)
                or not isinstance(centroid_cluster_count, (int, float))
                or not isinstance(centroid_member_count, (int, float))
                or not math.isfinite(float(centroid_cluster_count))
                or not math.isfinite(float(centroid_member_count))
                or float(centroid_cluster_count) < 0.0
                or float(centroid_member_count) <= 0.0
                or (
                    centroid_source == 1.0
                    and float(centroid_cluster_count) <= 0.0
                )
                or (
                    centroid_source == 2.0
                    and float(centroid_cluster_count) != 0.0
                )
            ):
                raise ValueError(
                    f"successful recenter lacks a valid derived centroid for {label}"
                )
            overlap_profiled = (
                event.get("sqlite_sparse_route_recenter_overlap_profiled") is True
            )
            overlap_pair_valid = (
                event.get("sqlite_sparse_route_recenter_overlap_pair_valid") is True
            )
            if overlap_profiled != overlap_pair_valid:
                raise ValueError(
                    f"recenter overlap profile is internally inconsistent for {label}"
                )
            if not overlap_profiled:
                continue
            if float(event.get(
                "sqlite_sparse_route_recenter_overlap_failure_code", -1.0
            )) != 0.0:
                raise ValueError(f"recenter overlap failed for {label}")
            pre_count = float(event.get(
                "sqlite_sparse_route_recenter_pre_activated_count", -1.0
            ))
            post_count = float(event.get(
                "sqlite_sparse_route_recenter_post_activated_count", -1.0
            ))
            overlap_count = float(event.get(
                "sqlite_sparse_route_recenter_overlap_count", -1.0
            ))
            if (
                not all(math.isfinite(value) for value in (
                    pre_count, post_count, overlap_count
                ))
                or pre_count <= 0.0
                or post_count <= 0.0
                or pre_count > expected["activation_identity_target"]
                or post_count > expected["activation_identity_target"]
                or overlap_count < 0.0
                or overlap_count > min(pre_count, post_count)
            ):
                raise ValueError(f"invalid recenter overlap counts for {label}")
            valid_overlap_events.append(event)
            overlap_ratios.append(overlap_count / post_count)
            if overlap_count < post_count:
                changed_overlap_events.append(event)
        hydration_limits = metric_values(
            profile, "Cortext.fallback_hydration_signal_limit"
        )
        hydration_rows = metric_values(
            profile, "Cortext.fallback_hydration_signal_rows"
        )
        if any(
            value != expected["fallback_hydration_signal_limit"]
            for value in hydration_limits
        ):
            raise ValueError(f"fallback hydration limit mismatch for {label}")
        topo = topology(db_path)
        sparse_traversed = sum(value > 0 for value in visits) > 0
        route_lifecycle_evaluated = sparse_traversed and bool(successful_resets)
        if sparse_traversed and (
            topo["observed_max_level"] is None
            or topo["observed_max_level"] <= 0
            or topo["upper_level_node_rows"] <= 0
            or topo["upper_level_nonempty_adjacency_rows"] <= 0
        ):
            raise ValueError(f"flat or missing runtime hierarchy for {label}")
        quality_class = quality_classification(
            sparse_traversed=sparse_traversed,
            evaluated_recenter_count=len(valid_overlap_events),
            node_rows=topo["node_rows"],
            published_active_count=topo["published_active_count"],
            route_capacity=expected["route_capacity"],
        )
        experiments.append(
            {
                "id": label,
                "knobs": [
                    profile["focus"],
                    profile["sensitivity"],
                    profile["stability"],
                ],
                "resolved": expected_operational_limits(expected),
                "active_epoch_limits": expected_epoch_limits,
                "observed": {
                    "sqlite_sparse_route_event_count": sum(value > 0 for value in visits),
                    "maximum_activated_identities": max(activated),
                    "activation_identity_target": expected[
                        "activation_identity_target"
                    ],
                    "maximum_search_node_rows": max(visits),
                    "maximum_activation_snapshot_rows": max(snapshot_visits),
                    "maximum_total_query_rows": max(
                        canonical + snapshot
                        for canonical, snapshot in zip(visits, snapshot_visits)
                    ),
                    "maximum_dynamic_search_node_budget": max(node_budgets),
                    "successful_activation_reset_count": len(successful_resets),
                    "cluster_centroid_reset_count": sum(
                        event.get("sqlite_sparse_route_recenter_centroid_source")
                        == 1.0
                        for event in successful_resets
                    ),
                    "active_signal_ring_centroid_reset_count": sum(
                        event.get("sqlite_sparse_route_recenter_centroid_source")
                        == 2.0
                        for event in successful_resets
                    ),
                    "valid_activation_overlap_count": len(valid_overlap_events),
                    "changed_activation_set_count": len(changed_overlap_events),
                    "minimum_activation_overlap_ratio": min(
                        overlap_ratios, default=None
                    ),
                    "maximum_activation_overlap_ratio": max(
                        overlap_ratios, default=None
                    ),
                    "maximum_route_metadata_open_rows": max(
                        route_metadata_open_rows
                    ),
                    "maximum_backfill_rows": max(backfill_rows, default=0),
                    "total_backfill_rows": sum(backfill_rows),
                    "maximum_fallback_hydration_signal_rows": max(hydration_rows),
                    "fallback_hydration_signal_limit": max(hydration_limits),
                    "backfill_failures": max(backfill_failures, default=0),
                    "search_failure_events": sum(value > 0 for value in search_failures),
                    "consolidation_runs": profile["consolidation_runs"],
                    "processed_events": profile["processed_events"],
                    "wall_ms": profile["wall_ms"],
                    "mean_process_ms": profile["mean_process_ms"],
                    "mean_total_ms": profile["mean_total_ms"],
                    "peak_rss_mb": profile["peak_rss_mb"],
            "exact_identity_recall_at_16": audit["exact_identity_recall_at_k"],
            "exact_top1": audit["exact_top1"],
            "exact_top1_hits": audit["exact_top1_hits"],
                    "exact_top1_miss_count": audit["exact_top1_miss_count"],
                    "top1_miss_fingerprints": audit["top1_miss_fingerprints"],
                    "semantic_coverage": audit["semantic_coverage"],
                    **topo,
                },
                "quality_class": quality_class,
                "profile": f"private/raw/{profile_path.name}",
                "profile_sha256": sha256(profile_path),
                "audit": f"sanitized/{audit_path.name}",
                "audit_sha256": sha256(audit_path),
                "database": f"private/raw/{db_path.name}",
                "db_sha256": sha256(db_path),
                "benchmark_binary": benchmark_binary,
            }
        )

    if len(experiments) != 9:
        raise ValueError("the contract requires exactly nine corpus runs")
    if len({row["id"] for row in experiments}) != len(experiments):
        raise ValueError("corpus run labels must be unique")
    observed_knob_points = {tuple(row["knobs"]) for row in experiments}
    if observed_knob_points != REQUIRED_CORPUS_KNOB_POINTS:
        raise ValueError("corpus runs do not match the required nine knob points")
    aggregate_top1_hits = sum(
        row["observed"]["exact_top1_hits"] for row in experiments
    )
    aggregate_top1_queries = sum(
        load_json(parse_run(spec)[2])["query_count"] for spec in args.run
    )
    aggregate_top1 = aggregate_top1_hits / aggregate_top1_queries
    if aggregate_top1 < 0.999:
        raise ValueError("aggregate exact top-1 is below 0.999")
    miss_cluster_audit = top1_miss_cluster_audit(experiments)
    if not miss_cluster_audit["passed"]:
        raise ValueError("exact top-1 misses cluster across a protected dimension")
    route_lifecycle_evaluated = [
        row
        for row in experiments
        if row["quality_class"]
        == "route-lifecycle-evaluated-system-quality-passed"
    ]
    traversed_recenter_unproven = [
        row
        for row in experiments
        if row["quality_class"]
        == "system-quality-passed-sparse-route-traversed-recenter-unproven"
    ]
    structural = junit(args.structural_junit, test_binary)
    aggregate = {
        "schema": "cortext_sqlite_hnsw_knob_ablation_v7",
        "status": "passed-production-default-nine-quality-points-with-explicit-route-maturity-classification",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {name: sha256(ROOT / name) for name in source_paths},
        "execution_identity": {
            "benchmark_binary": benchmark_binary,
            "test_binary": test_binary,
            "identity_contract": "Every profile and retained SQLite topology database was generated by the benchmark binary named here; every JUnit artifact was generated by the test binary named here. The aggregate content-addresses all binaries, profiles, audits, databases, and XML results.",
        },
        "corpus": {
            "kind": "owner-authorized-private-claude-jsonl-derived-sentence-packets",
            "processed_events_per_run": REQUIRED_PROCESSED_EVENTS_PER_RUN,
            "sha256": sha256(args.corpus),
        },
        "formulas": {
            "route_capacity": "C = lround(256 + 256*clamp(F) + 128*clamp(S) + 128*clamp(T))",
            "backfill_batch_size": "B = lround(64 + 64*clamp(F) + 32*clamp(S) + 32*clamp(T))",
            "fallback_hydration_signal_limit": "B recent indexed signal rows per hydrated memory",
            "graph_neighbor_count": "max(8, B / 2)",
            "graph_level_zero_links": "max(16, C / 4)",
            "maximum_level": "ceil(log2(C))",
            "reciprocal_update_count_per_level": "max(2, B / 16)",
            "centroid_identity_quota": "max(2, B / 16), shared with the reciprocal-update budget",
            "hnsw_construction_effort": "max(32, lround(C * 25 / 64))",
            "hnsw_query_effort": "max(C, lround(C * 5 / 2))",
            "bootstrap_limit": "2 * C",
            "activation_identity_target": "2 * C + 2 * B retained activated identities per public search",
            "activation_snapshot_capacity": "A identities persisted by consolidation and exactly compared without displacing canonical HNSW candidates",
            "total_query_row_budget": "9 * C + A, with canonical HNSW and consolidation-snapshot rows recorded separately",
            "search_node_budget": "9 * C hard ceiling per public activation",
            "activation_search_node_budget_min": "8 * C at route open and after successful consolidation recenter",
            "activation_search_node_budget_step": "max(2, B / 16) per public activation until the 9 * C ceiling",
            "search_expansion_batch": "max(8, B / 4)",
            "search_effort": "9 * C hard ceiling per public activation",
            "activation_search_effort_min": "8 * C at route open and after successful consolidation recenter",
            "activation_search_effort_step": "max(2, B / 16) per public activation until the 9 * C ceiling",
            "retrieval_cycle_edge_samples": "max(2, B / 32) retrieval-active queries per phase edge",
            "shadow_cache_capacity": "24 * C",
            "backfill_search_node_budget": "C + B",
            "backfill_search_effort": "2 * B",
            "historical_backfill_drain_limit": "B ordered rows per successful construction edge",
            "historical_backfill_overflow_probe_limit": "logical B + 1 boundary: process B ordered entries, then test whether the in-memory iterator has another entry; no B + 1 row is read or processed",
            "dirty_identity_drain_limit": "C per successful construction edge",
            "dirty_identity_overflow_probe_limit": "C + 1 identities read only to detect overflow",
            "route_metadata_open_row_limit": 1,
        },
        "structural_invariants": {
            "persisted_hnsw_hierarchy_count": 1,
            "route_metadata_open_row_count": 1,
            "runtime_hierarchy_required_for_sparse_route_traversed": True,
            "valid_pre-post_activation_overlap_required_for_route_lifecycle_evaluated": True,
            "whole_engine_restart_bounded": False,
            "classification": "fixed SQLite representation with a consolidation-reset activation envelope and all operational work limits knob-derived",
        },
        "structural_grid": {
            "focus_values": [0.0, 0.5, 1.0],
            "sensitivity_values": [0.0, 0.5, 1.0],
            "stability_values": [0.0, 0.5, 1.0],
            "configuration_count": 27,
            "passed": structural["failures"] == 0 and structural["errors"] == 0,
            **structural,
        },
        "direct_regressions": {
            "bounded_dirty_backfill_and_live_journal": junit(
                args.backfill_junit, test_binary
            ),
            "route_metadata_open": junit(args.restart_junit, test_binary),
            "core_knob_endpoints": junit(args.core_knob_junit, test_binary),
            "active_route_source_modality_label_invariance": junit(
                args.invariance_junit, test_binary
            ),
        },
        "system_quality_contract": {
            "queries_per_run": 512,
            "result_k": 16,
            "exact_identity_recall_at_k_min": 0.998,
            "exact_top1_per_run_min": 511 / 512,
            "exact_top1_per_run_max_misses": 1,
            "exact_top1_nine_run_aggregate_min": 0.999,
            "semantic_coverage_min": 0.95,
            "deterministic_tie_order_required": True,
            "minimum_opaque_source_ids": 4,
            "modalities": ["text"],
            "modality_agnostic_proof": "active_route_source_modality_label_invariance",
        },
        "experiments": experiments,
        "aggregate": {
            "run_count": len(experiments),
            "system_quality_query_count": 512 * len(experiments),
            "system_quality_pass_count": len(experiments),
            "route_lifecycle_evaluated_run_count": len(
                route_lifecycle_evaluated
            ),
            "route_lifecycle_unevaluated_run_count": len(experiments)
            - len(route_lifecycle_evaluated),
            "valid_activation_overlap_count": sum(
                row["observed"]["valid_activation_overlap_count"]
                for row in experiments
            ),
            "changed_activation_set_count": sum(
                row["observed"]["changed_activation_set_count"]
                for row in experiments
            ),
            "changed_activation_set_run_count": sum(
                row["observed"]["changed_activation_set_count"] > 0
                for row in experiments
            ),
            "sparse_route_quality_separately_materialized": False,
            "sparse_route_quality_pass_count": None,
            "sparse_route_traversed_recenter_unproven_run_count": sum(
                row["quality_class"]
                == "system-quality-passed-sparse-route-traversed-recenter-unproven"
                for row in experiments
            ),
            "published_below_threshold_route_unevaluated_run_count": sum(
                row["quality_class"]
                == "system-quality-passed-published-below-threshold-sparse-route-unevaluated"
                for row in experiments
            ),
            "exact_top1_min": min(row["observed"]["exact_top1"] for row in experiments),
            "exact_top1_aggregate": aggregate_top1,
            "exact_top1_aggregate_hits": aggregate_top1_hits,
            "exact_top1_aggregate_queries": aggregate_top1_queries,
            "exact_top1_aggregate_misses": (
                aggregate_top1_queries - aggregate_top1_hits
            ),
            "exact_top1_miss_cluster_audit": miss_cluster_audit,
            "exact_identity_recall_at_16_min": min(
                row["observed"]["exact_identity_recall_at_16"] for row in experiments
            ),
            "semantic_coverage_min": min(
                row["observed"]["semantic_coverage"] for row in experiments
            ),
            "maximum_route_metadata_open_rows": max(
                row["observed"]["maximum_route_metadata_open_rows"]
                for row in experiments
            ),
            "maximum_backfill_failure_count": max(
                row["observed"]["backfill_failures"] for row in experiments
            ),
            "maximum_search_failure_event_count": max(
                row["observed"]["search_failure_events"] for row in experiments
            ),
            "all_activation_identity_targets_respected": all(
                row["observed"]["maximum_activated_identities"]
                <= row["observed"]["activation_identity_target"]
                for row in experiments
            ),
            "all_route_lifecycle_evaluated_runs_have_runtime_upper_levels": all(
                row["observed"]["observed_max_level"] > 0
                and row["observed"]["upper_level_node_rows"] > 0
                and row["observed"]["upper_level_nonempty_adjacency_rows"] > 0
                for row in route_lifecycle_evaluated
            ),
            "all_resolved_parameters_embedded_in_raw_profiles": True,
            "all_system_quality_contracts_passed": True,
        },
        "limits": [
            "Whole-engine latency is not an HNSW-only measure because F, S, and T also change admission, consolidation frequency, and surface evolution.",
            "A route lifecycle is evaluated only after both nonzero SQLite node visits and a successful consolidation recenter. The retained quality audit is whole-system quality and does not separately materialize a route-active query subset.",
            "The route metadata open reads one row, but SignalProcessor startup still hydrates the complete current memory surface and therefore remains O(history). The route restart regression is not whole-engine bounded-restart proof.",
            "The corpus matrix uses text ingress with four opaque source identifiers. Modality agnosticism is proved separately by the active SQLite-route label-invariance regression over text, audio, and image labels, so no fabricated media ingress is used in the corpus audit.",
            "This 4,000-event matrix proves production-default knob derivation, local work ceilings, bounded durable dirty drainage, explicit route-maturity classification at every point, accepted whole-system quality, and—through the separate active-route regression—source/modality label invariance. It does not claim a mature 8C-to-9C recenter cycle when a run lacks a valid pre/post activation pair; that requires the separate long-horizon sawtooth experiment. It also does not prove route-only quality, bounded whole-engine restart, the full 15,695-event plateau, or production-wide boundedness.",
        ],
        "production_cutover": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
