#!/usr/bin/env python3
"""Audit replay flatness and emit content-free behavior/database digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Sequence


STORAGE_OPERATIONS = (
    "cortext::operations::MemoryStorage",
    "MemoryStorage.supersession_edges",
)
ATTRIBUTION_PREFIXES = (
    "cortext::operations::",
    "SignalProcessor.",
)
DETAILED_PREFIXES = (
    "GraphRetrieve.",
    "Competition.",
    "MemoryStorage.",
    "EmotionalCascade.",
)
LOGICAL_TABLES = {
    "memories": "memory_id",
    "associations": "source_memory_id, target_memory_id, edge_type",
    "signals": "signal_id",
}
REQUIRED_WORK_COUNTERS = (
    "retrieval_suppression_id_count",
    "predictive_id_count",
    "rollback_full_cache_copy_count",
    "rollback_cache_entry_copy_count",
    "graph_candidate_count",
    "graph_exact_comparison_count",
    "graph_cache_rebuild_count",
    "graph_rows_visited",
    "competition_candidate_count",
    "competition_rows_visited",
    "competition_rows_touched",
    "supersession_current_candidate_count",
    "supersession_historical_candidate_count",
    "supersession_rows_visited",
    "emotional_source_count",
    "emotional_neighbor_count",
    "emotional_update_count",
)
# Historical profiles from the rejected emotional-priority experiment contain
# these extra counters. They remain readable evidence, but current profiles do
# not emit them and they never count as timing measurements.
LEGACY_REJECTED_EMOTIONAL_WORK_COUNTERS = {
    "emotional_source_candidate_count",
    "emotional_source_priority_comparison_count",
    "emotional_source_priority_move_count",
    "emotional_source_memory_emission_count",
    "emotional_source_embedding_emission_count",
    "emotional_edge_visit_count",
    "emotional_frontier_entry_count",
    "emotional_topology_footprint_entry_count",
    "emotional_activated_identity_count",
    "emotional_member_overflow_count",
    "emotional_member_budget_skip_count",
    "emotional_member_read_count",
    "emotional_sql_affected_row_count",
    "emotional_cache_row_mutation_count",
    "emotional_source_index_comparison_count",
    "emotional_source_index_move_count",
}
REQUIRED_CONSOLIDATION_EPOCH_COUNTERS = (
    "accumulator_signal_count",
    "working_memory_pending_signal_count",
    "consolidation_dirty_memory_count",
    "consolidation_dirty_association_count",
    "consolidation_dirty_index_count",
)
RESETTABLE_CONSOLIDATION_EPOCH_COUNTERS = (
    "working_memory_pending_signal_count",
    "consolidation_dirty_memory_count",
    "consolidation_dirty_association_count",
    "consolidation_dirty_index_count",
)


def active_epoch_limits(
    focus: float, sensitivity: float, stability: float
) -> dict[str, int]:
    def clamped(value: float, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return min(1.0, max(0.0, result))

    f = clamped(focus, "profile focus")
    s = clamped(sensitivity, "profile sensitivity")
    t = clamped(stability, "profile stability")
    capacity = math.floor(256.0 + 256.0 * f + 128.0 * s + 128.0 * t + 0.5)
    row_batch_size = math.floor(64.0 + 64.0 * f + 32.0 * s + 32.0 * t + 0.5)
    return {
        "event_count": capacity,
        "mutation_count": capacity * 64,
        "allocated_bytes": capacity * 128 * 1024,
        "row_batch_size": row_batch_size,
    }


COUNTER_ACTIVITY_OPERATIONS = {
    "retrieval_suppression_id_count": (
        "SignalProcessor.retrieval_suppression_id_activity",
    ),
    "predictive_id_count": ("SignalProcessor.predictive_id_activity",),
    "rollback_full_cache_copy_count": (
        "SignalProcessor.snapshot_full_cache_copy_activity",
    ),
    "rollback_cache_entry_copy_count": (
        "SignalProcessor.snapshot_cache_entry_copy_activity",
    ),
    "graph_candidate_count": ("GraphRetrieve.candidate_activity",),
    "graph_exact_comparison_count": (
        "GraphRetrieve.family_exact_comparison_activity",
    ),
    "graph_cache_rebuild_count": ("GraphRetrieve.cache_rebuild_activity",),
    "graph_rows_visited": ("GraphRetrieve.rows_visited_activity",),
    "competition_candidate_count": ("Competition.candidate_activity",),
    "competition_rows_visited": ("Competition.rows_visited_activity",),
    "competition_rows_touched": ("Competition.rows_touched_activity",),
    "supersession_current_candidate_count": (
        "MemoryStorage.supersession_current_candidate_activity",
    ),
    "supersession_historical_candidate_count": (
        "MemoryStorage.supersession_historical_candidate_activity",
    ),
    "supersession_rows_visited": (
        "MemoryStorage.supersession_rows_visited_activity",
    ),
    "emotional_source_count": ("EmotionalCascade.source_activity",),
    "emotional_neighbor_count": ("EmotionalCascade.neighbor_activity",),
    "emotional_update_count": ("EmotionalCascade.update_activity",),
}
COUNTER_SOURCE_OPERATIONS = {
    "retrieval_suppression_id_count": (
        "SignalProcessor.retrieval_suppression_id_count",
    ),
    "predictive_id_count": ("SignalProcessor.predictive_id_count",),
    "rollback_full_cache_copy_count": (
        "SignalProcessor.snapshot_full_cache_copy_count",
    ),
    "rollback_cache_entry_copy_count": (
        "SignalProcessor.snapshot_cache_entry_copy_count",
    ),
    "graph_candidate_count": ("GraphRetrieve.candidate_count",),
    "graph_exact_comparison_count": (
        "GraphRetrieve.family_exact_comparison_count",
    ),
    "graph_cache_rebuild_count": ("GraphRetrieve.cache_rebuild_count",),
    "graph_rows_visited": (
        "GraphRetrieve.rows_visited",
    ),
    "competition_candidate_count": ("Competition.candidate_count",),
    "competition_rows_visited": ("Competition.rows_visited",),
    "competition_rows_touched": ("Competition.rows_touched",),
    "supersession_current_candidate_count": (
        "MemoryStorage.supersession_current_candidate_count",
    ),
    "supersession_historical_candidate_count": (
        "MemoryStorage.supersession_historical_candidate_count",
    ),
    "supersession_rows_visited": (
        "MemoryStorage.supersession_current_rows_visited",
        "MemoryStorage.supersession_historical_rows_visited",
        "MemoryStorage.supersession_sparse_route_node_rows",
    ),
    "emotional_source_count": ("EmotionalCascade.source_count",),
    "emotional_neighbor_count": ("EmotionalCascade.neighbor_count",),
    "emotional_update_count": ("EmotionalCascade.update_count",),
}
OTHER_DIAGNOSTIC_OPERATION_KEYS = {
    "GraphRetrieve.sqlite_sparse_route_activated_identities",
    "GraphRetrieve.sqlite_sparse_route_node_rows",
    "GraphRetrieve.sqlite_sparse_route_activation_snapshot_rows",
    "GraphRetrieve.sqlite_sparse_route_activation_snapshot_cache_miss_rows",
    "GraphRetrieve.sqlite_sparse_route_distance_evaluations",
    "GraphRetrieve.sqlite_sparse_route_restart_rows",
    "GraphRetrieve.sqlite_sparse_route_dirty_rows",
    "GraphRetrieve.seed_cache_distance_rows",
    "GraphRetrieve.seed_cache_eligibility_rows",
    "GraphRetrieve.seed_cache_ranked_rows",
    "GraphRetrieve.seed_cache_selected_rows",
    "GraphRetrieve.seed_cache_rows",
    "SignalProcessor.rif_active_epoch_event_count",
    "SignalProcessor.rif_active_epoch_mutation_count",
    "SignalProcessor.rif_active_epoch_allocated_bytes",
    "SignalProcessor.rif_active_epoch_row_batch_high_water",
    "EmotionalCascade.update_limit_reached",
    "SignalProcessor.rif_active_epoch_required",
    "SignalProcessor.rif_epoch_publication_rebuild_count",
    "SignalProcessor.rif_epoch_publication_recovery_count",
    "SignalProcessor.sqlite_wal_checkpoint_failure_count",
    "MemoryStorage.supersession_current_rows_visited",
    "MemoryStorage.supersession_historical_rows_visited",
    "MemoryStorage.supersession_sparse_route_node_rows",
    "MemoryStorage.supersession_sparse_route_distance_evaluations",
    "MemoryStorage.supersession_sparse_route_dirty_rows",
    "MemoryStorage.supersession_sql_fallback_count",
    "MemoryStorage.supersession_population_mismatch_count",
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
    "Cortext.fallback_hydration_signal_rows",
    "EmotionalCascade.bounded_mode",
    "EmotionalCascade.source_candidate_count",
    "EmotionalCascade.source_priority_comparison_count",
    "EmotionalCascade.source_priority_move_count",
    "EmotionalCascade.source_memory_emission_count",
    "EmotionalCascade.source_embedding_emission_count",
    "EmotionalCascade.edge_visit_count",
    "EmotionalCascade.frontier_entry_count",
    "EmotionalCascade.topology_footprint_entry_count",
    "EmotionalCascade.activated_identity_count",
    "EmotionalCascade.member_overflow_count",
    "EmotionalCascade.member_budget_skip_count",
    "EmotionalCascade.member_read_count",
    "EmotionalCascade.sql_affected_row_count",
    "EmotionalCascade.cache_row_mutation_count",
    "EmotionalCascade.source_index_comparison_count",
    "EmotionalCascade.source_index_move_count",
    "EmotionalCascade.source_candidate_limit",
    "EmotionalCascade.source_execution_limit",
    "EmotionalCascade.edge_visit_limit",
    "EmotionalCascade.frontier_entry_limit",
    "EmotionalCascade.activated_identity_limit",
    "EmotionalCascade.update_statement_limit",
    "EmotionalCascade.shared_member_limit",
    "EmotionalCascade.logical_member_overflow_boundary",
    "EmotionalCascade.member_read_limit",
    "EmotionalCascade.sql_affected_row_limit",
    "EmotionalCascade.cache_row_mutation_limit",
    "EmotionalCascade.topology_footprint_entry_limit",
    "EmotionalCascade.source_index_maintenance_limit",
    "EmotionalCascade.source_priority_comparison_limit",
}
DIAGNOSTIC_OPERATION_SUFFIXES = (
    "_activity",
    "_active",
    "_ambiguous",
    "_budget",
    "_count",
    "_effort",
    "_equivalent",
    "_failure_code",
    "_limit",
    "_proven",
    "_published",
    "_reached",
    "_skipped",
    "_succeeded",
    "_target",
)


def diagnostic_operation_keys() -> set[str]:
    return (
        {name for names in COUNTER_SOURCE_OPERATIONS.values() for name in names}
        | {
            name
            for names in COUNTER_ACTIVITY_OPERATIONS.values()
            for name in names
            if name.endswith("_count") or name.endswith("_activity")
        }
        | OTHER_DIAGNOSTIC_OPERATION_KEYS
    )


def is_duration_operation(name: str) -> bool:
    return (
        name not in diagnostic_operation_keys()
        and not name.startswith("ConsolidationEpoch.")
        and not name.startswith("SignalProcessor.commit_table_row_count.")
        and not name.endswith(DIAGNOSTIC_OPERATION_SUFFIXES)
    )
MINIMUM_POST_WARMUP_WINDOWS = 10
PLATEAU_MINIMUM_WINDOWS = 6
PROCESS_HALF_RATIO_RANGE = (0.90, 1.10)
THROUGHPUT_HALF_RATIO_RANGE = (0.90, 1.10)
RELATIVE_THEIL_SEN_MAX = 0.02
RELATIVE_BOOTSTRAP_UPPER_MAX = 0.05
OPERATION_HALF_RATIO_MAX = 1.05
OPERATION_MINIMUM_MEAN_MS = 0.05
WORK_COUNTER_HALF_RATIO_MAX = 1.10
AUTHORITATIVE_ROW_GROWTH_RATIO_MIN = 1.05
BOOTSTRAP_REPETITIONS = 2000
PLATEAU_HEIGHT_REFERENCES = {
    "natural": {
        "tail_process_mean_ms": 10.0757183124,
        "ceiling_multiplier": 1.10,
        "source_profile_sha256": (
            "55ed3de1d59c307610de1fd8a7e30821f46a87b3f0a2bddbbe06e7eb12836f09"
        ),
        "source_audit_sha256": (
            "528c8dc167b11fab7f8a30a9b55946e58fa89f96c5c001eb21c6d7f39bea464d"
        ),
    },
    "durable": {
        "tail_process_mean_ms": 24.317203052,
        "ceiling_multiplier": 1.10,
        "source_profile_sha256": (
            "28f8410ad21384ac642567627d2b31bbef16976426ddc783d2dd47f3c5ed0768"
        ),
        "source_audit_sha256": (
            "fb015cd5f4b3194c318f3266c32fa180c63ac7efede44df6f6c7ebb3d4404793"
        ),
    },
}
CONSOLIDATION_EPOCH_PRE_POST_EVENTS = 50
CONSOLIDATION_EPOCH_MINIMUM_MATERIAL_RESETS = 4
RETRIEVAL_CYCLE_MINIMUM_MATERIAL_CYCLES = 10
RETRIEVAL_CYCLE_MATERIAL_RESET_FRACTION_MIN = 0.80
RETRIEVAL_CYCLE_RESAMPLE_BINS = 50
RETRIEVAL_CYCLE_SHAPE_P95_NORMALIZED_MAE_MAX = 0.20
RETRIEVAL_CYCLE_LATE_TEMPLATE_MAE_MAX = 0.10
RETRIEVAL_CYCLE_NEGATIVE_VARIATION_OVER_RAMP_MAX = 0.25
RETRIEVAL_CYCLE_MAX_OVER_TRAILING_MEAN_MAX = 1.25
RETRIEVAL_QUALITY_QUERY_COUNT = 512
RETRIEVAL_QUALITY_K = 16
RETRIEVAL_EXACT_ID_RECALL_MIN = 0.998
RETRIEVAL_EXACT_TOP1_MIN = 511 / 512
RETRIEVAL_EXACT_TOP1_MAX_MISSES = 1
RETRIEVAL_SEMANTIC_COVERAGE_MIN = 0.95
CONSOLIDATION_EPOCH_MATERIAL_RAMP_RATIO_MIN = 1.10
CONSOLIDATION_EPOCH_POST_PRE_PROCESS_RATIO_MAX = 0.90
CONSOLIDATION_EPOCH_RESET_COUNTER_RATIO_MAX = 0.90
CONSOLIDATION_EPOCH_PEAK_HALF_RATIO_MAX = 1.10
CONSOLIDATION_EPOCH_TROUGH_HALF_RATIO_MAX = 1.10
CONSOLIDATION_EPOCH_NORMALIZED_COST_HALF_RATIO_MAX = 1.10
CONSOLIDATION_EPOCH_FREQUENCY_HALF_RATIO_MAX = 1.10
CONSOLIDATION_EPOCH_RELATIONAL_CONTRACT_SHA256 = (
    "13806d20db179c8cf2dd0c7990f6355b6513c255bdf53df3b2c1082c636f0e4b"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def normalized_sql_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"blob_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("logical database contains a non-finite float")
        return value
    return value


def logical_database_digest(db_path: Path) -> tuple[str, dict[str, int]]:
    # The benchmark leaves SQLite in WAL mode.  A read-only connection cannot
    # initialize missing -shm state for a freshly closed artifact, so use an
    # ordinary writable handle for the read-only queries below.  This may
    # create WAL sidecars but does not execute mutating SQL.
    connection = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    try:
        tables: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for table, ordering in LOGICAL_TABLES.items():
            cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {ordering}")
            columns = [column[0] for column in cursor.description]
            rows = [
                [normalized_sql_value(value) for value in row]
                for row in cursor.fetchall()
            ]
            counts[table] = len(rows)
            tables[table] = {"columns": columns, "rows": rows}
        return digest(tables), counts
    finally:
        connection.close()


def behavior_digest(
    rows: Sequence[dict[str, Any]], content_address: str | None = None
) -> str:
    if content_address is not None:
        if len(content_address) != 64 or any(
            char not in "0123456789abcdef" for char in content_address
        ):
            raise ValueError("profile behavior_sha256 is invalid")
        if any("behavior_sha256" not in row for row in rows):
            raise ValueError("compact profile lacks per-row behavior digests")
        return content_address
    behavior = []
    for row in rows:
        if "behavior" not in row:
            raise ValueError("profile lacks behavior-oracle rows")
        public_behavior = json.loads(json.dumps(row["behavior"]))
        output = public_behavior.get("output")
        if isinstance(output, dict):
            # These public diagnostics measure execution duration and are not
            # behavioral outputs. Older profiles included them, so normalize
            # both old and new artifacts before content-addressing behavior.
            output.pop("soft_anchor_last_update_us", None)
            output.pop("soft_anchor_mean_update_us", None)
        behavior.append(
            {
                "event_index": row["event_index"],
                "modality": row["modality"],
                "behavior": public_behavior,
            }
        )
    return digest(behavior)


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> float:
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    if not slopes:
        raise ValueError("at least two distinct windows are required")
    return statistics.median(slopes)


def bootstrap_upper_slope(
    xs: Sequence[float], ys: Sequence[float], repetitions: int = 2000
) -> float:
    fitted_slope = theil_sen(xs, ys)
    intercept = statistics.median(
        y - fitted_slope * x for x, y in zip(xs, ys)
    )
    residuals = [
        y - (intercept + fitted_slope * x)
        for x, y in zip(xs, ys)
    ]
    mean_x = statistics.mean(xs)
    centered_xs = [x - mean_x for x in xs]
    denominator = sum(value * value for value in centered_xs)
    if denominator == 0.0:
        raise ValueError("at least two distinct windows are required")
    rng = random.Random(0)
    slopes = []
    for _ in range(repetitions):
        # Keep the robust Theil-Sen line as the bootstrap center, then fit the
        # resampled residual perturbation in O(n). Re-running the O(n^2)
        # pairwise estimator for every repetition makes a valid long epoch
        # series require billions of comparisons without changing the named
        # deterministic residual-bootstrap contract.
        residual_slope = sum(
            centered_x * rng.choice(residuals)
            for centered_x in centered_xs
        ) / denominator
        slopes.append(fitted_slope + residual_slope)
    slopes.sort()
    return slopes[math.ceil(0.95 * len(slopes)) - 1]


def windows(values: Sequence[float], start: int, size: int) -> list[float]:
    return [
        statistics.mean(values[offset : offset + size])
        for offset in range(start, len(values) - size + 1, size)
    ]


def operation_values(
    rows: Sequence[dict[str, Any]], operation: str
) -> list[float]:
    values = []
    for row in rows:
        operation_ms = row.get("operation_ms")
        if not isinstance(operation_ms, dict):
            raise ValueError("profile lacks full operation timing")
        values.append(float(operation_ms.get(operation, 0.0)))
    return values


def comparison(means: Sequence[float]) -> dict[str, float]:
    early = statistics.mean(means[:5])
    late = statistics.mean(means[-5:])
    return {
        "first_five_mean_ms": early,
        "final_five_mean_ms": late,
        "delta_ms": late - early,
        "final_five_over_first_five": late / early if early > 0.0 else math.inf,
    }


def counter_comparison(means: Sequence[float]) -> dict[str, float]:
    early = statistics.mean(means[:5])
    late = statistics.mean(means[-5:])
    return {
        "first_five_mean": early,
        "final_five_mean": late,
        "delta": late - early,
        "final_five_over_first_five": late / early if early > 0.0 else math.inf,
    }


def finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def finite_fraction(value: Any, field: str) -> float:
    result = finite_nonnegative(value, field)
    if result > 1.0:
        raise ValueError(f"{field} must not exceed one")
    return result


def plateau_height_reference(retention: str) -> dict[str, Any]:
    reference = PLATEAU_HEIGHT_REFERENCES.get(retention)
    if not isinstance(reference, dict):
        raise ValueError("missing digest-bound plateau height reference")
    tail = finite_nonnegative(
        reference.get("tail_process_mean_ms"),
        "plateau height reference tail_process_mean_ms",
    )
    multiplier = finite_nonnegative(
        reference.get("ceiling_multiplier"),
        "plateau height reference ceiling_multiplier",
    )
    if tail <= 0.0 or multiplier <= 0.0:
        raise ValueError("plateau height reference values must be positive")
    for field in ("source_profile_sha256", "source_audit_sha256"):
        value = reference.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError(f"plateau height reference {field} is invalid")
    return {
        **reference,
        "plateau_process_ceiling_ms": tail * multiplier,
    }


def nonnegative_integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def consolidation_epoch_counters(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(
        REQUIRED_CONSOLIDATION_EPOCH_COUNTERS
    ):
        raise ValueError(f"{field} schema mismatch")
    return {
        name: nonnegative_integer(value[name], f"{field}.{name}")
        for name in REQUIRED_CONSOLIDATION_EPOCH_COUNTERS
    }


def validate_consolidation_epoch_relations(
    profile: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if profile.get("schema") != "cortext_packet_storage_profile_v2":
        raise ValueError("profile lacks consolidation epoch schema v2")
    if (
        profile.get("consolidation_epoch_relational_contract_sha256")
        != CONSOLIDATION_EPOCH_RELATIONAL_CONTRACT_SHA256
    ):
        raise ValueError("profile consolidation epoch relational contract mismatch")
    events = profile.get("consolidation_events")
    if not isinstance(events, list):
        raise ValueError("profile lacks consolidation_events")
    event_by_index: dict[int, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("consolidation event must be an object")
        event_index = nonnegative_integer(
            event.get("event_index"), "consolidation event_index"
        )
        if event_index in event_by_index:
            raise ValueError("duplicate consolidation close record")
        closing_event_index = nonnegative_integer(
            event.get("closing_event_index"),
            "consolidation closing_event_index",
        )
        if closing_event_index != event_index:
            raise ValueError("consolidation closing event identity mismatch")
        if event.get("sealed_mutation_identity_verified") is not True:
            raise ValueError("sealed mutation identity was not verified")
        nonnegative_integer(
            event.get("consolidation_epoch_id"),
            "consolidation event epoch id",
        )
        nonnegative_integer(
            event.get("sealed_epoch_event_count"),
            "sealed_epoch_event_count",
            positive=True,
        )
        nonnegative_integer(
            event.get("sealed_epoch_mutation_count"),
            "sealed_epoch_mutation_count",
        )
        identity_count = nonnegative_integer(
            event.get("sealed_mutation_identity_count"),
            "sealed_mutation_identity_count",
        )
        if identity_count != event["sealed_epoch_mutation_count"]:
            raise ValueError("sealed epoch mutation identity count mismatch")
        finite_nonnegative(event.get("duration_ms"), "consolidation duration_ms")
        consolidation_epoch_counters(
            event.get("pre_reset_counters"), "pre_reset_counters"
        )
        consolidation_epoch_counters(
            event.get("post_reset_counters"), "post_reset_counters"
        )
        event_by_index[event_index] = event

    expected_epoch = 0
    expected_since = 1
    observed_close_indices: list[int] = []
    for index, row in enumerate(rows):
        if row.get("event_index") != index:
            raise ValueError("profile event_index values must be unique and contiguous")
        epoch_id = nonnegative_integer(
            row.get("consolidation_epoch_id"),
            "consolidation_epoch_id",
        )
        events_since = nonnegative_integer(
            row.get("events_since_epoch_start"),
            "events_since_epoch_start",
            positive=True,
        )
        if epoch_id != expected_epoch or events_since != expected_since:
            raise ValueError("consolidation epoch sequence mismatch")
        counters = consolidation_epoch_counters(
            row.get("consolidation_epoch_counters"),
            "consolidation_epoch_counters",
        )
        closes = row.get("consolidation_after_event")
        if not isinstance(closes, bool):
            raise ValueError("consolidation_after_event must be boolean")
        event = event_by_index.get(index)
        if closes != (event is not None):
            raise ValueError("orphan consolidation row or close record")
        if closes:
            assert event is not None
            if event["consolidation_epoch_id"] != epoch_id:
                raise ValueError("consolidation close epoch mismatch")
            if event["sealed_epoch_event_count"] != events_since:
                raise ValueError("sealed epoch event count mismatch")
            if event["pre_reset_counters"] != counters:
                raise ValueError("pre-reset counters are not adjacent to close row")
            if not math.isclose(
                float(row.get("consolidation_ms", -1.0)),
                float(event["duration_ms"]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError("consolidation row duration mismatch")
            observed_close_indices.append(index)
            expected_epoch += 1
            expected_since = 1
        else:
            if finite_nonnegative(
                row.get("consolidation_ms"), "consolidation_ms"
            ) != 0.0:
                raise ValueError("consolidation duration lacks close record")
            expected_since += 1
    if observed_close_indices != sorted(event_by_index):
        raise ValueError("consolidation close ordering mismatch")
    return [event_by_index[index] for index in observed_close_indices]


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def activity_normalized_reset_diagnostic(
    evaluation_inputs: Sequence[dict[str, Any]],
    raw_gate_failures: Sequence[str],
    *,
    knob_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify activity incidence without changing the raw sawtooth verdict."""
    if not isinstance(raw_gate_failures, Sequence) or isinstance(
        raw_gate_failures, (str, bytes)
    ):
        raise ValueError("raw gate failures must be a sequence")
    preserved_failures = []
    for failure in raw_gate_failures:
        if not isinstance(failure, str):
            raise ValueError("raw gate failure must be a string")
        preserved_failures.append(failure)

    resolved_knobs = None
    if knob_context is not None:
        if not isinstance(knob_context, dict):
            raise ValueError("knob context must be an object")
        limits = active_epoch_limits(
            knob_context.get("focus"),
            knob_context.get("sensitivity"),
            knob_context.get("stability"),
        )
        processed_rows = nonnegative_integer(
            knob_context.get("processed_backfill_rows", 0),
            "processed backfill rows",
        )
        backfill_batch = limits["row_batch_size"]
        if processed_rows > backfill_batch:
            raise ValueError("processed rows exceed F/S/T-derived B")
        resolved_knobs = {
            "backfill_batch": backfill_batch,
            "logical_backfill_boundary_only": backfill_batch + 1,
            "processed_backfill_rows": processed_rows,
        }

    evaluations = []
    epoch_windows: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    seen_epoch_operations: set[tuple[int, str]] = set()
    for item in evaluation_inputs:
        if not isinstance(item, dict):
            raise ValueError("activity-normalized evaluation must be an object")
        operation = item.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError("activity-normalized operation id must be a string")
        epoch = nonnegative_integer(item.get("epoch"), "activity epoch")
        sequence_names = (
            "trailing_event_indices",
            "post_event_indices",
            "trailing_ms",
            "post_ms",
            "trailing_activity",
            "post_activity",
        )
        sequences = {}
        for name in sequence_names:
            value = item.get(name)
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            sequences[name] = value
        lengths = {len(value) for value in sequences.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("activity normalization requires matching event windows")
        trailing_indices = [
            nonnegative_integer(value, "trailing event index")
            for value in sequences["trailing_event_indices"]
        ]
        post_indices = [
            nonnegative_integer(value, "post event index")
            for value in sequences["post_event_indices"]
        ]
        if len(set(trailing_indices + post_indices)) != len(
            trailing_indices + post_indices
        ):
            raise ValueError("activity normalization requires matching event windows")
        if (
            trailing_indices != list(range(
                trailing_indices[0], trailing_indices[0] + len(trailing_indices)
            ))
            or post_indices != list(range(
                post_indices[0], post_indices[0] + len(post_indices)
            ))
            or post_indices[0] != trailing_indices[-1] + 1
        ):
            raise ValueError(
                "activity normalization requires contiguous adjacent event windows"
            )
        window_identity = (tuple(trailing_indices), tuple(post_indices))
        prior_window = epoch_windows.setdefault(epoch, window_identity)
        if prior_window != window_identity:
            raise ValueError(
                "activity-normalized operations in the same epoch must share event windows"
            )
        epoch_operation = (epoch, operation)
        if epoch_operation in seen_epoch_operations:
            raise ValueError("duplicate activity-normalized epoch operation")
        seen_epoch_operations.add(epoch_operation)
        numeric = {
            name: [
                finite_nonnegative(value, f"activity-normalized {name}")
                for value in sequences[name]
            ]
            for name in (
                "trailing_ms",
                "post_ms",
                "trailing_activity",
                "post_activity",
            )
        }
        trailing_ms = finite_nonnegative(
            sum(numeric["trailing_ms"]), "aggregate trailing milliseconds"
        )
        post_ms = finite_nonnegative(
            sum(numeric["post_ms"]), "aggregate post milliseconds"
        )
        trailing_activity = finite_nonnegative(
            sum(numeric["trailing_activity"]), "aggregate trailing activity"
        )
        post_activity = finite_nonnegative(
            sum(numeric["post_activity"]), "aggregate post activity"
        )
        operation_ms_delta = post_ms - trailing_ms
        if not math.isfinite(operation_ms_delta):
            raise ValueError("aggregate operation delta must be finite")
        result = {
            "epoch": epoch,
            "operation": operation,
            "status": "valid",
            "trailing_ms": trailing_ms,
            "post_ms": post_ms,
            "trailing_activity": trailing_activity,
            "post_activity": post_activity,
            "activity_ratio": None,
            "trailing_unit_cost": None,
            "post_unit_cost": None,
            "unit_cost_ratio": None,
            "operation_ms_delta": operation_ms_delta,
            "classification": None,
        }
        if trailing_activity == 0.0 and post_activity == 0.0:
            result["status"] = "unevaluated"
            result["classification"] = "not-explanatory"
        elif trailing_activity == 0.0 or post_activity == 0.0:
            result["status"] = "incomparable"
            result["classification"] = "not-explanatory"
        else:
            trailing_unit_cost = finite_nonnegative(
                trailing_ms / trailing_activity,
                "activity-normalized trailing unit cost",
            )
            post_unit_cost = finite_nonnegative(
                post_ms / post_activity,
                "activity-normalized post unit cost",
            )
            unit_cost_ratio = finite_nonnegative(
                ratio(post_unit_cost, trailing_unit_cost),
                "activity-normalized unit-cost ratio",
            )
            activity_ratio = finite_nonnegative(
                post_activity / trailing_activity,
                "activity-normalized activity ratio",
            )
            result.update({
                "activity_ratio": activity_ratio,
                "trailing_unit_cost": trailing_unit_cost,
                "post_unit_cost": post_unit_cost,
                "unit_cost_ratio": unit_cost_ratio,
            })
            if unit_cost_ratio > OPERATION_HALF_RATIO_MAX:
                result["classification"] = "per-unit-regression"
            elif activity_ratio > 1.0 and operation_ms_delta > 0.0:
                result["classification"] = "activity-incidence"
            else:
                result["classification"] = "not-explanatory"
        evaluations.append(result)

    per_unit = [
        item for item in evaluations
        if item["classification"] == "per-unit-regression"
    ]
    incidence = [
        item for item in evaluations
        if item["classification"] == "activity-incidence"
    ]
    selected = None
    diagnostic_result = "diagnostic-inconclusive"
    continuation = "deepen-profile"
    if per_unit:
        selected = sorted(
            per_unit,
            key=lambda item: (
                -float(item["unit_cost_ratio"]),
                -max(0.0, float(item["operation_ms_delta"])),
                str(item["operation"]),
                int(item["epoch"]),
            ),
        )[0]
        diagnostic_result = "per-unit-regression"
        continuation = "implement-per-unit-cost-hotspot"
    elif incidence:
        selected = sorted(
            incidence,
            key=lambda item: (
                -float(item["operation_ms_delta"]),
                -float(item["activity_ratio"]),
                str(item["operation"]),
                int(item["epoch"]),
            ),
        )[0]
        diagnostic_result = "activity-incidence"
        continuation = "attribute-activity-generation"
    selected_hotspot = None if selected is None else {
        "operation": selected["operation"],
        "epoch": selected["epoch"],
        "classification": selected["classification"],
    }
    return {
        "diagnostic_result": diagnostic_result,
        "selected_hotspot": selected_hotspot,
        "continuation": continuation,
        "evaluations": evaluations,
        "raw_gate_passed": not preserved_failures,
        "raw_gate_failures": preserved_failures,
        "raw_gate_waived": False,
        "label_arithmetic": "ignored",
        "knob_context": resolved_knobs,
    }


def half_comparison(values: Sequence[float]) -> dict[str, float]:
    half = len(values) // 2
    if half == 0:
        raise ValueError("at least two windows are required for half comparison")
    first = statistics.mean(values[:half])
    second = statistics.mean(values[-half:])
    return {
        "first_half_mean": first,
        "second_half_mean": second,
        "second_over_first": ratio(second, first),
        "omitted_center_windows": len(values) - 2 * half,
    }


def end_anchored_ranges(event_count: int, window_size: int) -> list[tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window size must be positive")
    complete = event_count // window_size
    leading_remainder = event_count - complete * window_size
    return [
        (leading_remainder + index * window_size,
         leading_remainder + (index + 1) * window_size)
        for index in range(complete)
    ]


def validate_profile_rows(
    profile: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> None:
    processed_events = nonnegative_integer(
        profile.get("processed_events"), "profile processed_events", positive=True
    )
    if processed_events != len(rows):
        raise ValueError("profile processed_events does not match row count")
    retention = profile.get("retention")
    if retention not in {"natural", "durable"}:
        raise ValueError("profile retention must be natural or durable")
    expected_active_epoch_limits = active_epoch_limits(
        profile.get("focus"),
        profile.get("sensitivity"),
        profile.get("stability"),
    )
    if profile.get("active_epoch_limits") != expected_active_epoch_limits:
        raise ValueError("profile active-epoch limits do not match the contract")
    honors_required = profile.get("honor_required_consolidation")
    if not isinstance(honors_required, bool):
        raise ValueError("profile honor_required_consolidation must be boolean")
    counter_activity = {name: 0.0 for name in REQUIRED_WORK_COUNTERS}
    operation_activity = {name: 0.0 for name in COUNTER_ACTIVITY_OPERATIONS}
    row_consolidation_total = 0.0
    row_consolidation_events: list[int] = []
    warmup_events = math.ceil(0.20 * len(rows))
    sparse_parameters = profile.get("sparse_route_parameters")
    fallback_hydration_limit = None
    if (
        isinstance(sparse_parameters, dict)
        and "fallback_hydration_signal_limit" in sparse_parameters
    ):
        focus = finite_fraction(profile.get("focus"), "focus")
        sensitivity = finite_fraction(profile.get("sensitivity"), "sensitivity")
        stability = finite_fraction(profile.get("stability"), "stability")
        derived_limit = math.floor(
            64.0 + 64.0 * focus + 32.0 * sensitivity + 32.0 * stability + 0.5
        )
        fallback_hydration_limit = nonnegative_integer(
            sparse_parameters.get("fallback_hydration_signal_limit"),
            "sparse_route_parameters.fallback_hydration_signal_limit",
        )
        if fallback_hydration_limit != derived_limit:
            raise ValueError("fallback hydration limit does not match F/S/T")
    if isinstance(sparse_parameters, dict) and "search_node_budget" in sparse_parameters:
        focus = finite_fraction(profile.get("focus"), "focus")
        sensitivity = finite_fraction(profile.get("sensitivity"), "sensitivity")
        stability = finite_fraction(profile.get("stability"), "stability")
        rounded = lambda value: math.floor(value + 0.5)
        route_capacity = rounded(
            256.0 + 256.0 * focus + 128.0 * sensitivity + 128.0 * stability
        )
        backfill_batch_size = rounded(
            64.0 + 64.0 * focus + 32.0 * sensitivity + 32.0 * stability
        )
        expected_sparse_parameters = {
            "route_capacity": route_capacity,
            "activation_identity_target": route_capacity * 2 + backfill_batch_size * 2,
            "activation_snapshot_capacity": route_capacity * 2 + backfill_batch_size * 2,
            "total_query_row_budget": (
                route_capacity * 11 + backfill_batch_size * 2
            ),
            "bootstrap_limit": route_capacity * 2,
            "search_node_budget": route_capacity * 9,
            "activation_search_node_budget_min": route_capacity * 8,
            "activation_search_node_budget_step": max(
                2, backfill_batch_size // 16
            ),
            "search_expansion_batch": max(8, backfill_batch_size // 4),
            "search_effort": route_capacity * 9,
            "activation_search_effort_min": route_capacity * 8,
            "activation_search_effort_step": max(
                2, backfill_batch_size // 16
            ),
            "shadow_cache_capacity": route_capacity * 24,
            "backfill_batch_size": backfill_batch_size,
            "backfill_search_node_budget": route_capacity + backfill_batch_size,
            "backfill_search_effort": backfill_batch_size * 2,
            "graph_neighbor_count": max(8, backfill_batch_size // 2),
            "graph_level_zero_links": max(16, route_capacity // 4),
            "family_exact_comparison_limit": route_capacity * 2,
            "maximum_level": max(1, max(1, route_capacity - 1).bit_length()),
            "reciprocal_update_count": max(2, backfill_batch_size // 16),
            "hnsw_construction_effort": max(
                32, rounded(route_capacity * 25.0 / 64.0)
            ),
            "hnsw_query_effort": max(
                route_capacity, rounded(route_capacity * 5.0 / 2.0)
            ),
            "fallback_hydration_signal_limit": backfill_batch_size,
        }
        experimental_node_formula = profile.get(
            "experimental_sparse_node_envelope_formula"
        )
        experimental_node_budgets = {
            "C": route_capacity,
            "A": route_capacity * 2 + backfill_batch_size * 2,
            "C+B": route_capacity + backfill_batch_size,
            "2C": route_capacity * 2,
            "4C": route_capacity * 4,
            "4C+B/16": route_capacity * 4
            + max(2, backfill_batch_size // 16),
            "fixed-4C+B/16": route_capacity * 4
            + max(2, backfill_batch_size // 16),
            "4C+B/16-to-5C": route_capacity * 5,
            "5C-to-6C-by-B/16": route_capacity * 6,
            "6C-to-7C-by-B/16": route_capacity * 7,
            "7C-to-8C-by-B/16": route_capacity * 8,
            "8C-to-9C-by-B/16": route_capacity * 9,
            "5C": route_capacity * 5,
            "6C": route_capacity * 6,
            "10C": route_capacity * 10,
            "12C": route_capacity * 12,
            "16C": route_capacity * 16,
        }
        if experimental_node_formula is not None:
            expected_sparse_parameters["search_node_budget"] = route_capacity * 5
            expected_sparse_parameters[
                "activation_search_node_budget_min"
            ] = route_capacity * 5
            expected_sparse_parameters["activation_search_node_budget_step"] = 0
            expected_sparse_parameters["search_effort"] = route_capacity * 5
            expected_sparse_parameters[
                "activation_search_effort_min"
            ] = route_capacity * 5
            expected_sparse_parameters["activation_search_effort_step"] = 0
            expected_sparse_parameters["total_query_row_budget"] = (
                route_capacity * 7 + backfill_batch_size * 2
            )
            if experimental_node_formula not in experimental_node_budgets:
                raise ValueError(
                    "experimental sparse node envelope formula is invalid"
                )
            selected_node_budget = experimental_node_budgets[
                experimental_node_formula
            ]
            expected_sparse_parameters["search_node_budget"] = selected_node_budget
            expected_sparse_parameters[
                "activation_search_node_budget_min"
            ] = selected_node_budget
            if experimental_node_formula == "fixed-4C+B/16":
                expected_sparse_parameters["search_effort"] = selected_node_budget
                expected_sparse_parameters[
                    "activation_search_effort_min"
                ] = selected_node_budget
                expected_sparse_parameters["total_query_row_budget"] = (
                    selected_node_budget
                    + expected_sparse_parameters["activation_identity_target"]
                )
            if experimental_node_formula == "4C+B/16-to-5C":
                reciprocal_update_count = expected_sparse_parameters[
                    "reciprocal_update_count"
                ]
                minimum = route_capacity * 4 + reciprocal_update_count
                expected_sparse_parameters[
                    "activation_search_effort_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                expected_sparse_parameters[
                    "activation_search_node_budget_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
            if experimental_node_formula == "5C-to-6C-by-B/16":
                reciprocal_update_count = expected_sparse_parameters[
                    "reciprocal_update_count"
                ]
                minimum = route_capacity * 5
                expected_sparse_parameters["search_effort"] = selected_node_budget
                expected_sparse_parameters[
                    "activation_search_effort_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                expected_sparse_parameters[
                    "activation_search_node_budget_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                expected_sparse_parameters["total_query_row_budget"] = (
                    selected_node_budget
                    + expected_sparse_parameters["activation_identity_target"]
                )
            if experimental_node_formula == "6C-to-7C-by-B/16":
                reciprocal_update_count = expected_sparse_parameters[
                    "reciprocal_update_count"
                ]
                minimum = route_capacity * 6
                expected_sparse_parameters["search_effort"] = selected_node_budget
                expected_sparse_parameters[
                    "activation_search_effort_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                expected_sparse_parameters[
                    "activation_search_node_budget_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                expected_sparse_parameters["total_query_row_budget"] = (
                    selected_node_budget
                    + expected_sparse_parameters["activation_identity_target"]
                )
            if experimental_node_formula == "7C-to-8C-by-B/16":
                reciprocal_update_count = expected_sparse_parameters[
                    "reciprocal_update_count"
                ]
                minimum = route_capacity * 7
                expected_sparse_parameters["search_effort"] = selected_node_budget
                expected_sparse_parameters[
                    "activation_search_effort_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                expected_sparse_parameters[
                    "activation_search_node_budget_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                expected_sparse_parameters["total_query_row_budget"] = (
                    selected_node_budget
                    + expected_sparse_parameters["activation_identity_target"]
                )
            if experimental_node_formula == "8C-to-9C-by-B/16":
                reciprocal_update_count = expected_sparse_parameters[
                    "reciprocal_update_count"
                ]
                minimum = route_capacity * 8
                expected_sparse_parameters["search_effort"] = selected_node_budget
                expected_sparse_parameters[
                    "activation_search_effort_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                expected_sparse_parameters[
                    "activation_search_node_budget_min"
                ] = minimum
                expected_sparse_parameters[
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                expected_sparse_parameters["total_query_row_budget"] = (
                    selected_node_budget
                    + expected_sparse_parameters["activation_identity_target"]
                )
        if set(sparse_parameters) != set(expected_sparse_parameters):
            raise ValueError("sparse-route parameter schema does not match F/S/T contract")
        for name, expected in expected_sparse_parameters.items():
            actual = nonnegative_integer(
                sparse_parameters.get(name), f"sparse_route_parameters.{name}"
            )
            if actual != expected:
                raise ValueError(f"sparse-route parameter {name} does not match F/S/T")
    for expected_index, row in enumerate(rows):
        if row.get("event_index") != expected_index:
            raise ValueError("profile event_index values must be unique and contiguous")
        if row.get("retention") != retention:
            raise ValueError("profile row retention mismatch")
        process_ms = finite_nonnegative(row.get("process_ms"), "process_ms")
        end_to_end_ms = finite_nonnegative(
            row.get("end_to_end_ms"), "end_to_end_ms"
        )
        consolidation_ms = finite_nonnegative(
            row.get("consolidation_ms"), "consolidation_ms"
        )
        durable_barrier_ms = finite_nonnegative(
            row.get("durable_barrier_ms"), "durable_barrier_ms"
        )
        if end_to_end_ms + 1e-9 < process_ms + consolidation_ms:
            raise ValueError("end_to_end_ms excludes required event work")
        closes = row.get("consolidation_after_event")
        if not isinstance(closes, bool):
            raise ValueError("consolidation_after_event must be boolean")
        if closes:
            row_consolidation_events.append(expected_index)
            row_consolidation_total += consolidation_ms
        active_epoch = row.get("active_epoch")
        if not isinstance(active_epoch, dict):
            raise ValueError("profile row lacks active_epoch counters")
        expected_active_epoch_keys = {
            "event_count",
            "mutation_count",
            "allocated_bytes",
            "row_batch_high_water",
            "required",
        }
        if set(active_epoch) != expected_active_epoch_keys:
            raise ValueError("profile active_epoch counter schema mismatch")
        epoch_values = {
            name: nonnegative_integer(
                active_epoch.get(name), f"active_epoch.{name}"
            )
            for name in ("event_count", "mutation_count", "allocated_bytes")
        }
        row_batch_high_water = nonnegative_integer(
            active_epoch.get("row_batch_high_water"),
            "active_epoch.row_batch_high_water",
        )
        active_required = active_epoch.get("required")
        if not isinstance(active_required, bool):
            raise ValueError("active_epoch.required must be boolean")
        at_or_above_limit = any(
            epoch_values[name] >= expected_active_epoch_limits[name]
            for name in epoch_values
        )
        if at_or_above_limit and not active_required:
            raise ValueError("active-epoch boundary was not preserved")
        if honors_required and any(
            epoch_values[name] > expected_active_epoch_limits[name]
            for name in epoch_values
        ):
            raise ValueError("honored active epoch crossed a safety ceiling")
        if row_batch_high_water > expected_active_epoch_limits["row_batch_size"]:
            raise ValueError("active-epoch row batch crossed its F/S/T ceiling")
        counters = row.get("work_counters")
        if not isinstance(counters, dict):
            raise ValueError("profile row lacks work_counters")
        counter_names = set(counters)
        required_counter_names = set(REQUIRED_WORK_COUNTERS)
        missing = sorted(required_counter_names - counter_names)
        extra = sorted(counter_names - required_counter_names)
        if missing or not set(extra).issubset(
            LEGACY_REJECTED_EMOTIONAL_WORK_COUNTERS
        ):
            raise ValueError(
                f"work counter schema mismatch; missing={missing}, extra={extra}"
            )
        operations = row.get("operation_ms")
        if not isinstance(operations, dict):
            raise ValueError("profile lacks full operation timing")
        if (
            isinstance(sparse_parameters, dict)
            and "activation_identity_target" in sparse_parameters
        ):
            activation_target = nonnegative_integer(
                sparse_parameters.get("activation_identity_target"),
                "sparse_route_parameters.activation_identity_target",
                positive=True,
            )
            activated_identities = finite_nonnegative(
                operations.get(
                    "GraphRetrieve.sqlite_sparse_route_activated_identities"
                ),
                "operation_ms.GraphRetrieve.sqlite_sparse_route_activated_identities",
            )
            if activated_identities > activation_target:
                raise ValueError("sparse-route activation exceeded its F/S/T target")
            sparse_route_active = finite_nonnegative(
                operations.get("GraphRetrieve.seed_sparse_route_active", 0.0),
                "operation_ms.GraphRetrieve.seed_sparse_route_active",
            )
            if sparse_route_active > 0.0 and activated_identities <= 0.0:
                raise ValueError("active sparse route reported no activated identities")
        if fallback_hydration_limit is not None:
            observed_limit = finite_nonnegative(
                operations.get("Cortext.fallback_hydration_signal_limit"),
                "operation_ms.Cortext.fallback_hydration_signal_limit",
            )
            finite_nonnegative(
                operations.get("Cortext.fallback_hydration_signal_rows"),
                "operation_ms.Cortext.fallback_hydration_signal_rows",
            )
            if observed_limit != float(fallback_hydration_limit):
                raise ValueError("event fallback hydration limit differs from F/S/T")
        if (
            retention == "durable"
            and "SignalProcessor.sqlite_wal_checkpoint" not in operations
        ):
            raise ValueError("durable row lacks post-commit barrier attribution")
        if (
            retention == "durable"
            and "SignalProcessor.sqlite_wal_checkpoint_failure_count"
            not in operations
        ):
            raise ValueError("durable row lacks barrier failure diagnostic")
        checkpoint_failures = finite_nonnegative(
            operations.get(
                "SignalProcessor.sqlite_wal_checkpoint_failure_count", 0.0
            ),
            "operation_ms.SignalProcessor.sqlite_wal_checkpoint_failure_count",
        )
        if retention == "durable" and checkpoint_failures != 0.0:
            raise ValueError("durable row reports a failed post-commit barrier")
        if "MemoryStorage.supersession_sql_fallback_count" not in operations:
            raise ValueError("profile row lacks supersession fallback diagnostic")
        supersession_fallbacks = finite_nonnegative(
            operations["MemoryStorage.supersession_sql_fallback_count"],
            "operation_ms.MemoryStorage.supersession_sql_fallback_count",
        )
        if supersession_fallbacks != 0.0 and expected_index >= warmup_events:
            raise ValueError(
                "post-warmup supersession SQL fallback prevents bounded plateau accounting"
            )
        operation_barrier_ms = finite_nonnegative(
            operations.get("SignalProcessor.sqlite_wal_checkpoint", 0.0),
            "operation_ms.SignalProcessor.sqlite_wal_checkpoint",
        )
        if not math.isclose(
            durable_barrier_ms, operation_barrier_ms, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("durable barrier attribution mismatch")
        for name in REQUIRED_WORK_COUNTERS:
            counter_activity[name] += finite_nonnegative(
                counters[name], f"work_counters.{name}"
            )
            source_names = COUNTER_SOURCE_OPERATIONS[name]
            missing_sources = [
                source for source in source_names if source not in operations
            ]
            if missing_sources:
                raise ValueError(
                    f"work counter {name} lacks producer counters: {missing_sources}"
                )
            source_value = sum(
                finite_nonnegative(
                    operations[source], f"operation_ms.{source}"
                )
                for source in source_names
            )
            if not math.isclose(
                float(counters[name]), source_value, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    f"work counter {name} does not match producer counters"
                )
        for counter, activity_names in COUNTER_ACTIVITY_OPERATIONS.items():
            missing_activity = [
                name for name in activity_names if name not in operations
            ]
            if missing_activity:
                raise ValueError(
                    f"work counter {counter} lacks independent activity markers: "
                    f"{missing_activity}"
                )
            operation_activity[counter] += sum(
                finite_nonnegative(
                    operations[name], f"operation_ms.{name}"
                )
                for name in activity_names
            )
    for counter, operation_total in operation_activity.items():
        if operation_total > 0.0 and counter_activity[counter] == 0.0:
            raise ValueError(
                f"work counter {counter} is all zero beside positive operation work"
            )
    consolidation_runs = profile.get("consolidation_runs")
    if isinstance(consolidation_runs, bool) or not isinstance(
        consolidation_runs, int
    ) or consolidation_runs < 0:
        raise ValueError("profile consolidation_runs must be nonnegative integer")
    consolidation_total = finite_nonnegative(
        profile.get("consolidation_ms"), "profile.consolidation_ms"
    )
    if consolidation_runs != len(row_consolidation_events) or not math.isclose(
        consolidation_total, row_consolidation_total, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError("consolidation row attribution mismatch")
    consolidation_events = profile.get("consolidation_events")
    if not isinstance(consolidation_events, list):
        raise ValueError("profile lacks consolidation_events")
    event_indices = []
    for event in consolidation_events:
        if not isinstance(event, dict):
            raise ValueError("consolidation event must be an object")
        event_index = event.get("event_index")
        if isinstance(event_index, bool) or not isinstance(event_index, int):
            raise ValueError("consolidation event index must be an integer")
        finite_nonnegative(event.get("duration_ms"), "consolidation duration_ms")
        post_epoch = event.get("post_reset_active_epoch")
        if not isinstance(post_epoch, dict):
            raise ValueError("consolidation event lacks post-reset active epoch")
        if set(post_epoch) != {
            "event_count",
            "mutation_count",
            "allocated_bytes",
            "row_batch_high_water",
            "required",
        }:
            raise ValueError("post-reset active epoch schema mismatch")
        if post_epoch.get("required") is not False:
            raise ValueError("successful consolidation did not clear epoch boundary")
        for name in ("event_count", "mutation_count", "allocated_bytes"):
            limit = expected_active_epoch_limits[name]
            value = nonnegative_integer(
                post_epoch.get(name), f"post_reset_active_epoch.{name}"
            )
            if value >= limit:
                raise ValueError("successful consolidation did not reset active epoch")
        post_row_batch = nonnegative_integer(
            post_epoch.get("row_batch_high_water"),
            "post_reset_active_epoch.row_batch_high_water",
        )
        if post_row_batch > expected_active_epoch_limits["row_batch_size"]:
            raise ValueError("post-reset row batch crossed its F/S/T ceiling")
        event_indices.append(event_index)
    if event_indices != row_consolidation_events:
        raise ValueError("consolidation event indices do not match rows")
    validate_consolidation_epoch_relations(profile, rows)


def sequence_trend(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise ValueError("at least two values are required for sequence trend")
    mean_value = statistics.mean(values)
    xs = [float(index) for index in range(len(values))]
    slope = theil_sen(xs, values)
    upper = bootstrap_upper_slope(xs, values, BOOTSTRAP_REPETITIONS)
    if mean_value == 0.0:
        if any(value != 0.0 for value in values):
            raise ValueError("nonzero sequence has zero mean")
        relative_slope = 0.0
        relative_upper = 0.0
    else:
        relative_slope = slope / mean_value
        relative_upper = upper / mean_value
    return {
        "half": half_comparison(values),
        "relative_theil_sen_per_epoch": relative_slope,
        "relative_bootstrap_95_upper_per_epoch": relative_upper,
    }


def retrieval_work(row: dict[str, Any]) -> float:
    counters = row.get("work_counters")
    if not isinstance(counters, dict):
        raise ValueError("row lacks work_counters")
    return finite_nonnegative(
        counters.get("graph_candidate_count"),
        "work_counters.graph_candidate_count",
    ) + finite_nonnegative(
        counters.get("graph_exact_comparison_count"),
        "work_counters.graph_exact_comparison_count",
    ) + finite_nonnegative(
        counters.get("graph_rows_visited"),
        "work_counters.graph_rows_visited",
    ) + finite_nonnegative(
        counters.get("supersession_rows_visited"),
        "work_counters.supersession_rows_visited",
    )


def sparse_route_value(row: dict[str, Any], name: str) -> float:
    operation_ms = row.get("operation_ms")
    if not isinstance(operation_ms, dict):
        return 0.0
    return finite_nonnegative(operation_ms.get(name, 0.0), f"operation_ms.{name}")


def sparse_route_retrieval_work(row: dict[str, Any]) -> float:
    return sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
    )


def sparse_route_retrieval_active(row: dict[str, Any]) -> bool:
    return any(
        sparse_route_value(row, name) > 0.0
        for name in (
            "GraphRetrieve.seed_sparse_route_active",
            "GraphRetrieve.sqlite_sparse_route_search_effort",
            "GraphRetrieve.sqlite_sparse_route_search_node_budget",
            "GraphRetrieve.sqlite_sparse_route_node_rows",
            "GraphRetrieve.sqlite_sparse_route_activation_snapshot_rows",
            "GraphRetrieve.sqlite_sparse_route_dirty_rows",
            "GraphRetrieve.sqlite_sparse_route_distance_evaluations",
            "GraphRetrieve.sqlite_sparse_route_activated_identities",
        )
    )


def sparse_route_retrieval_queue_effort(row: dict[str, Any]) -> float:
    return sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_search_effort"
    )


def sparse_route_retrieval_effort(row: dict[str, Any]) -> float:
    return sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_search_node_budget"
    )


def sparse_route_retrieval_actual_visits(row: dict[str, Any]) -> float:
    return sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_node_rows"
    ) + sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_dirty_rows"
    )


def sparse_route_activation_snapshot_visits(row: dict[str, Any]) -> float:
    return sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_activation_snapshot_rows"
    )


def sparse_route_retrieval_activated_identities(
    row: dict[str, Any],
) -> float:
    return sparse_route_value(
        row, "GraphRetrieve.sqlite_sparse_route_activated_identities"
    )


def resample_equal_progress(values: Sequence[float], bins: int) -> list[float]:
    if bins <= 0 or not values:
        raise ValueError("retrieval cycle cannot be resampled to required bins")
    if len(values) < bins:
        if len(values) == 1:
            return [float(values[0])] * bins
        result = []
        for index in range(bins):
            position = index * (len(values) - 1) / (bins - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            fraction = position - lower
            result.append(
                float(values[lower]) * (1.0 - fraction)
                + float(values[upper]) * fraction
            )
        return result
    result = []
    for index in range(bins):
        begin = index * len(values) // bins
        end = (index + 1) * len(values) // bins
        if end <= begin:
            raise ValueError("retrieval cycle resample produced an empty bin")
        result.append(statistics.mean(values[begin:end]))
    return result


def normalized_cycle(
    values: Sequence[float], trough: float, peak: float,
) -> list[float]:
    span = peak - trough
    if span == 0.0:
        return [0.0 for _ in values]
    return [(value - trough) / span for value in values]


def mean_profile(profiles: Sequence[Sequence[float]]) -> list[float]:
    if not profiles:
        raise ValueError("retrieval cycle profile set is empty")
    width = len(profiles[0])
    if width == 0 or any(len(profile) != width for profile in profiles):
        raise ValueError("retrieval cycle profile widths differ")
    return [
        statistics.mean(profile[index] for profile in profiles)
        for index in range(width)
    ]


def profile_mae(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("retrieval cycle profile comparison is invalid")
    return statistics.mean(abs(a - b) for a, b in zip(left, right))


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid nearest-rank percentile input")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def knob_fixed_embedding_slots(artifact: dict[str, Any]) -> int:
    focus = finite_fraction(artifact.get("focus"), "focus")
    sensitivity = finite_fraction(artifact.get("sensitivity"), "sensitivity")
    stability = finite_fraction(artifact.get("stability"), "stability")
    rounded = lambda value: math.floor(value + 0.5)
    roots = 8 + rounded(24 * sensitivity)
    leaves = 128 + rounded(512 * focus + 512 * stability)
    representatives = 8 + rounded(32 * focus)
    return roots + leaves + leaves * representatives


def knob_normal_comparison_bound(artifact: dict[str, Any]) -> int:
    focus = finite_fraction(artifact.get("focus"), "focus")
    sensitivity = finite_fraction(artifact.get("sensitivity"), "sensitivity")
    stability = finite_fraction(artifact.get("stability"), "stability")
    rounded = lambda value: math.floor(value + 0.5)
    roots = 8 + rounded(24 * sensitivity)
    leaves = 128 + rounded(512 * focus + 512 * stability)
    representatives = 8 + rounded(32 * focus)
    children_per_root = math.ceil(leaves / roots)
    root_beam = 1 + rounded(3 * sensitivity)
    return (
        roots
        + root_beam * children_per_root
        + 2 * representatives
        + children_per_root
    )


def knob_sparse_retrieval_work_bound(artifact: dict[str, Any]) -> int:
    parameters = artifact.get("sparse_route_parameters")
    if not isinstance(parameters, dict):
        return knob_normal_comparison_bound(artifact)
    node_budget = nonnegative_integer(
        parameters.get("search_node_budget"),
        "sparse_route_parameters.search_node_budget",
    )
    if node_budget == 0:
        raise ValueError("sparse-route work bounds must be positive")
    return node_budget


def knob_sparse_total_query_row_bound(artifact: dict[str, Any]) -> int:
    parameters = artifact.get("sparse_route_parameters")
    if not isinstance(parameters, dict):
        return knob_normal_comparison_bound(artifact)
    return nonnegative_integer(
        parameters.get("total_query_row_budget"),
        "sparse_route_parameters.total_query_row_budget",
        positive=True,
    )


def knob_sparse_cycle_edge_samples(artifact: dict[str, Any]) -> int:
    parameters = artifact.get("sparse_route_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("sparse-route parameters are required for cycle edges")
    backfill_batch = nonnegative_integer(
        parameters.get("backfill_batch_size"),
        "sparse_route_parameters.backfill_batch_size",
    )
    if backfill_batch == 0:
        raise ValueError("sparse-route backfill batch must be positive")
    return max(2, backfill_batch // 32)


def knob_work_counter_bound(
    artifact: dict[str, Any], counter: str,
) -> int | None:
    if counter not in {
        "graph_exact_comparison_count",
        "graph_rows_visited",
        "supersession_rows_visited",
        "emotional_source_count",
        "emotional_neighbor_count",
        "emotional_update_count",
    }:
        return None
    parameters = artifact.get("sparse_route_parameters")
    if not isinstance(parameters, dict):
        return None
    if counter == "emotional_update_count":
        return nonnegative_integer(
            parameters.get("activation_identity_target"),
            "sparse_route_parameters.activation_identity_target",
            positive=True,
        )
    if counter == "graph_rows_visited":
        return nonnegative_integer(
            parameters.get("activation_identity_target"),
            "sparse_route_parameters.activation_identity_target",
            positive=True,
        )
    if counter == "supersession_rows_visited":
        return nonnegative_integer(
            parameters.get("total_query_row_budget"),
            "sparse_route_parameters.total_query_row_budget",
            positive=True,
        )
    if counter == "emotional_source_count":
        return nonnegative_integer(
            parameters.get("backfill_batch_size"),
            "sparse_route_parameters.backfill_batch_size",
            positive=True,
        )
    if counter == "emotional_neighbor_count":
        route_capacity = nonnegative_integer(
            parameters.get("route_capacity"),
            "sparse_route_parameters.route_capacity",
            positive=True,
        )
        return route_capacity * 5
    return nonnegative_integer(
        parameters.get("family_exact_comparison_limit"),
        "sparse_route_parameters.family_exact_comparison_limit",
        positive=True,
    )


def ranked_query_evidence_result(
    artifact: dict[str, Any], result_k: int,
) -> dict[str, Any]:
    evidence = artifact.get("query_evidence")
    if not isinstance(evidence, list) or len(evidence) != RETRIEVAL_QUALITY_QUERY_COUNT:
        raise ValueError("query evidence must contain exactly 512 rows")
    embedding_count = nonnegative_integer(
        artifact.get("embedding_count"), "embedding_count"
    )
    if embedding_count <= RETRIEVAL_QUALITY_QUERY_COUNT:
        raise ValueError("embedding corpus is too small for held-out queries")
    exact_recall_total = 0.0
    exact_top1_total = 0.0
    semantic_total = 0.0
    source_ids: set[str] = set()
    modalities: set[str] = set()
    history_segments: set[str] = set()
    by_number: dict[int, dict[str, Any]] = {}
    query_embedding_ids: set[int] = set()
    exact_rank_match = True
    for expected_number, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError("query evidence row must be an object")
        query_number = nonnegative_integer(
            item.get("query_number"), "query_evidence.query_number"
        )
        if query_number != expected_number or query_number in by_number:
            raise ValueError("query evidence numbers must be unique and contiguous")
        query_index = nonnegative_integer(
            item.get("query_index"), "query_evidence.query_index"
        )
        expected_index = math.floor(
            query_number * (embedding_count - 1)
            / (RETRIEVAL_QUALITY_QUERY_COUNT - 1)
        )
        query_embedding_id = item.get("query_embedding_id")
        if (
            query_index != expected_index
            or isinstance(query_embedding_id, bool)
            or not isinstance(query_embedding_id, int)
            or query_embedding_id in query_embedding_ids
        ):
            raise ValueError("query identity is not bound to deterministic sampling")
        query_embedding_ids.add(query_embedding_id)
        exact_ids = item.get("exact_ranked_ids")
        candidate_ids = item.get("candidate_ranked_ids")
        if (
            not isinstance(exact_ids, list)
            or not isinstance(candidate_ids, list)
            or len(exact_ids) != result_k
            or len(candidate_ids) != result_k
            or any(isinstance(value, bool) or not isinstance(value, int) for value in exact_ids)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in candidate_ids)
            or len(set(exact_ids)) != result_k
            or len(set(candidate_ids)) != result_k
        ):
            raise ValueError("query ranked identities are invalid")
        hits = len(set(exact_ids).intersection(candidate_ids))
        exact_recall_total += hits / result_k
        exact_top1_total += candidate_ids[0] == exact_ids[0]
        exact_rank_match = exact_rank_match and candidate_ids == exact_ids
        semantic_total += finite_fraction(
            item.get("exact_neighbor_semantic_coverage"),
            "query_evidence.exact_neighbor_semantic_coverage",
        )
        item_sources = item.get("source_ids")
        item_modalities = item.get("modalities")
        history_segment = item.get("history_segment")
        history_ordinal = nonnegative_integer(
            item.get("history_ordinal"), "query_evidence.history_ordinal"
        )
        history_count = nonnegative_integer(
            item.get("history_count"), "query_evidence.history_count"
        )
        if history_count == 0 or history_ordinal >= history_count:
            raise ValueError("query SQLite history position is invalid")
        expected_history_segment = ("early", "middle", "late")[
            min(2, history_ordinal * 3 // history_count)
        ]
        if (
            not isinstance(item_sources, list)
            or not item_sources
            or any(not isinstance(value, str) or not value for value in item_sources)
            or not isinstance(item_modalities, list)
            or not item_modalities
            or any(not isinstance(value, str) or not value for value in item_modalities)
            or history_segment != expected_history_segment
        ):
            raise ValueError("query provenance evidence is invalid")
        source_ids.update(item_sources)
        modalities.update(item_modalities)
        history_segments.add(history_segment)
        by_number[query_number] = item

    probes = artifact.get("fixed_identity_rank_probes")
    if not isinstance(probes, list) or len(probes) != 7:
        raise ValueError("exactly seven fixed identity-rank probes are required")
    fixed_probe_numbers = [
        math.floor(probe * (RETRIEVAL_QUALITY_QUERY_COUNT - 1) / 6)
        for probe in range(7)
    ]
    probe_payload = []
    probe_numbers: set[int] = set()
    fixed_probes_match = True
    for probe in probes:
        if not isinstance(probe, dict):
            raise ValueError("fixed identity-rank probe must be an object")
        query_number = nonnegative_integer(
            probe.get("query_number"), "fixed probe query_number"
        )
        if (
            query_number in probe_numbers
            or query_number not in by_number
            or query_number != fixed_probe_numbers[len(probe_numbers)]
        ):
            raise ValueError("fixed identity-rank probe query is invalid")
        exact_ids = probe.get("exact_ranked_ids")
        candidate_ids = probe.get("candidate_ranked_ids")
        if (
            exact_ids != by_number[query_number]["exact_ranked_ids"]
            or candidate_ids != by_number[query_number]["candidate_ranked_ids"]
        ):
            raise ValueError("fixed identity-rank probe is not bound to query evidence")
        fixed_probes_match = fixed_probes_match and exact_ids == candidate_ids
        probe_numbers.add(query_number)
        probe_payload.append({"query_number": query_number, "ranked_ids": exact_ids})

    count = len(evidence)
    candidate_rank_payload = [
        {
            "query_number": query_number,
            "candidate_ranked_ids": by_number[query_number][
                "candidate_ranked_ids"
            ],
        }
        for query_number in range(count)
    ]
    candidate_identity_rank_sha256 = hashlib.sha256(
        canonical_json(candidate_rank_payload)
    ).hexdigest()
    return {
        "count": count,
        "mean_exact_id_recall_at_k": exact_recall_total / count,
        "mean_top1_exact_id": exact_top1_total / count,
        "exact_top1_hit_count": int(exact_top1_total),
        "mean_exact_neighbor_semantic_coverage": semantic_total / count,
        "exact_rank_match": exact_rank_match,
        "candidate_identity_rank_sha256": candidate_identity_rank_sha256,
        "source_count": len(source_ids),
        "modalities": sorted(modalities),
        "history_segments": sorted(history_segments),
        "by_number": by_number,
        "fixed_probes_match": fixed_probes_match,
        "fixed_probe_identity_rank_sha256": hashlib.sha256(
            canonical_json(probe_payload)
        ).hexdigest(),
    }


def retrieval_cycle_symmetry_result(
    material_epochs: Sequence[dict[str, Any]], retrieval_work_bound: int,
    accepted_suffix_retrieval_work: Sequence[float],
    accepted_suffix_retrieval_effort: Sequence[float],
    *,
    accepted_suffix_queue_effort: Sequence[float] | None = None,
    queue_effort_bound: int | None = None,
    accepted_suffix_total_work: Sequence[float] | None = None,
    total_query_row_bound: int | None = None,
    accepted_suffix_snapshot_visits: Sequence[float] | None = None,
    activation_identity_target: int | None = None,
    accepted_suffix_activated_identities: Sequence[float] | None = None,
) -> dict[str, Any]:
    failures = []
    profiles = [epoch["_retrieval_cycle_profile"] for epoch in material_epochs]
    inverse_ramp_count = sum(
        epoch["trailing_peak_mean_retrieval_work"]
        < epoch["leading_mean_retrieval_work"]
        for epoch in material_epochs
    )
    if inverse_ramp_count:
        failures.append("retrieval work falls before consolidation")
    nonmaterial_ramp_count = sum(
        epoch["retrieval_trailing_over_leading"]
        < CONSOLIDATION_EPOCH_MATERIAL_RAMP_RATIO_MIN
        for epoch in material_epochs
    )
    if nonmaterial_ramp_count:
        failures.append("retrieval cycle lacks material ramp")
    excessive_negative_variation_count = sum(
        epoch["retrieval_negative_variation_over_ramp"]
        > RETRIEVAL_CYCLE_NEGATIVE_VARIATION_OVER_RAMP_MAX
        for epoch in material_epochs
    )
    if excessive_negative_variation_count:
        failures.append("retrieval cycle contains excessive negative variation")
    excessive_spike_count = sum(
        epoch["retrieval_max_over_trailing_mean"]
        > RETRIEVAL_CYCLE_MAX_OVER_TRAILING_MEAN_MAX
        for epoch in material_epochs
    )
    if excessive_spike_count:
        failures.append("retrieval cycle contains excessive spike")
    work_bound_exceeded_count = sum(
        work > retrieval_work_bound
        for work in accepted_suffix_retrieval_work
    )
    if work_bound_exceeded_count:
        failures.append("retrieval work exceeds F/S/T bound")
    if len(accepted_suffix_retrieval_work) != len(
        accepted_suffix_retrieval_effort
    ):
        raise ValueError("retrieval work and dynamic node ceilings differ in length")
    dynamic_bound_exceeded_count = sum(
        work > effort
        for work, effort in zip(
            accepted_suffix_retrieval_work,
            accepted_suffix_retrieval_effort,
        )
    )
    if dynamic_bound_exceeded_count:
        failures.append("retrieval work exceeds current dynamic node ceiling")
    dynamic_queue_bound_exceeded_count = 0
    total_work_bound_exceeded_count = 0
    snapshot_bound_exceeded_count = 0
    combined_visit_bound_exceeded_count = 0
    distance_coverage_mismatch_count = 0
    activated_identity_exceeded_count = 0
    optional_samples = (
        accepted_suffix_queue_effort,
        accepted_suffix_total_work,
        accepted_suffix_snapshot_visits,
        accepted_suffix_activated_identities,
    )
    if any(values is not None for values in optional_samples):
        if any(values is None for values in optional_samples):
            raise ValueError("dynamic sparse retrieval metrics are incomplete")
        if (
            queue_effort_bound is None
            or total_query_row_bound is None
            or activation_identity_target is None
        ):
            raise ValueError("dynamic sparse retrieval bounds are incomplete")
        sample_count = len(accepted_suffix_retrieval_work)
        if any(len(values) != sample_count for values in optional_samples):
            raise ValueError("dynamic sparse retrieval metric samples differ")
        queue_effort = accepted_suffix_queue_effort or ()
        total_work = accepted_suffix_total_work or ()
        snapshot_visits = accepted_suffix_snapshot_visits or ()
        activated_identities = accepted_suffix_activated_identities or ()
        dynamic_queue_bound_exceeded_count = sum(
            queue > queue_effort_bound for queue in queue_effort
        )
        total_work_bound_exceeded_count = sum(
            work > total_query_row_bound for work in total_work
        )
        snapshot_bound_exceeded_count = sum(
            visits > activation_identity_target
            for visits in snapshot_visits
        )
        combined_visit_bound_exceeded_count = sum(
            canonical + snapshot > total_query_row_bound
            for canonical, snapshot in zip(
                accepted_suffix_retrieval_work, snapshot_visits
            )
        )
        distance_coverage_mismatch_count = sum(
            work > canonical + snapshot
            for work, canonical, snapshot in zip(
                total_work,
                accepted_suffix_retrieval_work,
                snapshot_visits,
            )
        )
        activated_identity_exceeded_count = sum(
            count > activation_identity_target
            for count in activated_identities
        )
        if dynamic_queue_bound_exceeded_count:
            failures.append("dynamic queue exceeds F/S/T bound")
        if total_work_bound_exceeded_count:
            failures.append("retrieval work exceeds total F/S/T row bound")
        if snapshot_bound_exceeded_count:
            failures.append("retrieval snapshot visits exceed F/S/T bound")
        if combined_visit_bound_exceeded_count:
            failures.append("retrieval total visits exceed F/S/T bound")
        if distance_coverage_mismatch_count:
            failures.append("retrieval distance work exceeds fetched-row coverage")
        if activated_identity_exceeded_count:
            failures.append("retrieval activation exceeds F/S/T target")
    reset_count = sum(
        epoch["retrieval_post_over_trailing"]
        <= CONSOLIDATION_EPOCH_POST_PRE_PROCESS_RATIO_MAX
        for epoch in material_epochs
    )
    reset_fraction = ratio(float(reset_count), float(len(material_epochs)))
    template = None
    errors: list[float] = []
    p95 = None
    late_template_error = None
    retrieval_peak_trend = None
    retrieval_trough_trend = None
    if len(material_epochs) < RETRIEVAL_CYCLE_MINIMUM_MATERIAL_CYCLES:
        failures.append("fewer than ten material retrieval cycles")
    else:
        template = [
            statistics.median(profile[index] for profile in profiles)
            for index in range(RETRIEVAL_CYCLE_RESAMPLE_BINS)
        ]
        errors = [profile_mae(profile, template) for profile in profiles]
        p95 = nearest_rank_percentile(errors, 0.95)
        prior_template = mean_profile(profiles[-10:-5])
        final_template = mean_profile(profiles[-5:])
        late_template_error = profile_mae(prior_template, final_template)
        if p95 > RETRIEVAL_CYCLE_SHAPE_P95_NORMALIZED_MAE_MAX:
            failures.append("retrieval cycle shape p95 exceeds limit")
        if late_template_error > RETRIEVAL_CYCLE_LATE_TEMPLATE_MAE_MAX:
            failures.append("late retrieval cycle template drifts")
        retrieval_peak_trend = sequence_trend(
            [
                epoch["trailing_peak_mean_retrieval_work"]
                for epoch in material_epochs
            ]
        )
        retrieval_trough_trend = sequence_trend(
            [
                epoch["following_trough_mean_retrieval_work"]
                for epoch in material_epochs
            ]
        )
        for trend, name, half_ratio_limit in (
            (
                retrieval_peak_trend,
                "retrieval peak",
                CONSOLIDATION_EPOCH_PEAK_HALF_RATIO_MAX,
            ),
            (
                retrieval_trough_trend,
                "retrieval trough",
                CONSOLIDATION_EPOCH_TROUGH_HALF_RATIO_MAX,
            ),
        ):
            if trend["half"]["second_over_first"] > half_ratio_limit:
                failures.append(f"{name} rises")
            if trend["relative_theil_sen_per_epoch"] > RELATIVE_THEIL_SEN_MAX:
                failures.append(f"{name} relative slope rises")
            if trend["relative_bootstrap_95_upper_per_epoch"] > RELATIVE_BOOTSTRAP_UPPER_MAX:
                failures.append(f"{name} bootstrap upper rises")
    if reset_fraction < RETRIEVAL_CYCLE_MATERIAL_RESET_FRACTION_MIN:
        failures.append("too few material retrieval resets")
    return {
        "passed": not failures,
        "material_cycle_count": len(material_epochs),
        "inverse_ramp_count": inverse_ramp_count,
        "nonmaterial_ramp_count": nonmaterial_ramp_count,
        "excessive_negative_variation_count": (
            excessive_negative_variation_count
        ),
        "excessive_spike_count": excessive_spike_count,
        "work_bound_exceeded_count": work_bound_exceeded_count,
        "dynamic_bound_exceeded_count": dynamic_bound_exceeded_count,
        "dynamic_queue_bound_exceeded_count": (
            dynamic_queue_bound_exceeded_count
        ),
        "total_work_bound_exceeded_count": total_work_bound_exceeded_count,
        "snapshot_bound_exceeded_count": snapshot_bound_exceeded_count,
        "combined_visit_bound_exceeded_count": (
            combined_visit_bound_exceeded_count
        ),
        "distance_coverage_mismatch_count": distance_coverage_mismatch_count,
        "activated_identity_exceeded_count": (
            activated_identity_exceeded_count
        ),
        "retrieval_work_bound": retrieval_work_bound,
        "accepted_suffix_maximum_retrieval_work": max(
            accepted_suffix_retrieval_work, default=0.0
        ),
        "material_reset_count": reset_count,
        "material_reset_fraction": reset_fraction,
        "resample_bins": RETRIEVAL_CYCLE_RESAMPLE_BINS,
        "shape_p95_normalized_mae": p95,
        "late_template_normalized_mae": late_template_error,
        "retrieval_peak_trend": retrieval_peak_trend,
        "retrieval_trough_trend": retrieval_trough_trend,
        "cycle_error_min": min(errors) if errors else None,
        "cycle_error_max": max(errors) if errors else None,
        "failures": sorted(set(failures)),
    }


def fixed_retrieval_envelope_result(
    complete_epochs: Sequence[dict[str, Any]],
    recenter_events: Sequence[dict[str, Any]], canonical_node_bound: int,
    total_query_row_bound: int,
    accepted_suffix_retrieval_work: Sequence[float],
    accepted_suffix_queue_effort: Sequence[float],
    accepted_suffix_retrieval_effort: Sequence[float],
    accepted_suffix_actual_visits: Sequence[float],
    accepted_suffix_snapshot_visits: Sequence[float],
    accepted_suffix_activated_identities: Sequence[float],
    activation_identity_target: int,
) -> dict[str, Any]:
    failures = []
    sample_count = len(accepted_suffix_retrieval_work)
    if any(
        len(values) != sample_count
        for values in (
            accepted_suffix_queue_effort,
            accepted_suffix_retrieval_effort,
            accepted_suffix_actual_visits,
            accepted_suffix_snapshot_visits,
            accepted_suffix_activated_identities,
        )
    ):
        raise ValueError("fixed retrieval metric samples differ in length")
    unevaluated_epoch_count = sum(
        not epoch["retrieval_cycle_evaluated"] for epoch in complete_epochs
    )
    zero_work_count = sum(work <= 0.0 for work in accepted_suffix_retrieval_work)
    work_bound_exceeded_count = sum(
        work > total_query_row_bound for work in accepted_suffix_retrieval_work
    )
    queue_effort_mismatch_count = sum(
        effort != canonical_node_bound
        for effort in accepted_suffix_queue_effort
    )
    node_ceiling_mismatch_count = sum(
        effort != canonical_node_bound
        for effort in accepted_suffix_retrieval_effort
    )
    actual_visit_exceeded_count = sum(
        visits > canonical_node_bound
        for visits in accepted_suffix_actual_visits
    )
    snapshot_visit_exceeded_count = sum(
        visits > activation_identity_target
        for visits in accepted_suffix_snapshot_visits
    )
    total_visit_exceeded_count = sum(
        canonical + snapshot > total_query_row_bound
        for canonical, snapshot in zip(
            accepted_suffix_actual_visits,
            accepted_suffix_snapshot_visits,
        )
    )
    distance_coverage_mismatch_count = sum(
        work > canonical + snapshot
        for work, canonical, snapshot in zip(
            accepted_suffix_retrieval_work,
            accepted_suffix_actual_visits,
            accepted_suffix_snapshot_visits,
        )
    )
    activated_identity_exceeded_count = sum(
        count > activation_identity_target
        for count in accepted_suffix_activated_identities
    )
    epoch_edge_mismatch_count = sum(
        any(
            value <= 0.0 or value > total_query_row_bound
            for value in (
                epoch["leading_mean_retrieval_work"],
                epoch["trailing_peak_mean_retrieval_work"],
                epoch["following_trough_mean_retrieval_work"],
                epoch["maximum_retrieval_work"],
            )
        )
        for epoch in complete_epochs
    )
    recentered_events = [
        event
        for event in recenter_events
        if event.get("sqlite_sparse_route_recenter_succeeded") is True
    ]
    recenter_envelope_mismatch_count = sum(
        finite_nonnegative(
            event.get("sqlite_sparse_route_activation_search_effort", 0.0),
            "consolidation sqlite sparse route activation effort",
        )
        != canonical_node_bound
        or finite_nonnegative(
            event.get("sqlite_sparse_route_activation_node_budget", 0.0),
            "consolidation sqlite sparse route activation node budget",
        )
        != canonical_node_bound
        for event in recentered_events
    )
    overlap_missing_count = 0
    overlap_invalid_count = 0
    overlap_changed_count = 0
    overlap_ratios = []
    for event in recentered_events:
        if event.get("sqlite_sparse_route_recenter_overlap_profiled") is not True:
            overlap_missing_count += 1
            continue
        if (
            event.get("sqlite_sparse_route_recenter_overlap_pair_valid") is not True
            or finite_nonnegative(
                event.get("sqlite_sparse_route_recenter_overlap_failure_code", 0.0),
                "consolidation sqlite sparse route overlap failure code",
            )
            != 0.0
        ):
            overlap_invalid_count += 1
            continue
        pre_count = finite_nonnegative(
            event.get("sqlite_sparse_route_recenter_pre_activated_count"),
            "consolidation sqlite sparse route pre activation count",
        )
        post_count = finite_nonnegative(
            event.get("sqlite_sparse_route_recenter_post_activated_count"),
            "consolidation sqlite sparse route post activation count",
        )
        overlap_count = finite_nonnegative(
            event.get("sqlite_sparse_route_recenter_overlap_count"),
            "consolidation sqlite sparse route overlap count",
        )
        if (
            pre_count <= 0.0
            or post_count <= 0.0
            or pre_count > activation_identity_target
            or post_count > activation_identity_target
            or overlap_count > min(pre_count, post_count)
        ):
            overlap_invalid_count += 1
            continue
        overlap_ratios.append(ratio(overlap_count, post_count))
        if overlap_count < post_count:
            overlap_changed_count += 1
    if not accepted_suffix_retrieval_work:
        failures.append("fixed retrieval envelope has no activated work")
    if unevaluated_epoch_count:
        failures.append("fixed retrieval envelope lacks complete epoch coverage")
    if zero_work_count:
        failures.append("fixed retrieval work is absent on an active route")
    if work_bound_exceeded_count:
        failures.append("fixed retrieval work exceeds total F/S/T row bound")
    if queue_effort_mismatch_count:
        failures.append("fixed queue effort differs from F/S/T bound")
    if node_ceiling_mismatch_count:
        failures.append("fixed dynamic node ceiling differs from F/S/T bound")
    if actual_visit_exceeded_count:
        failures.append("fixed retrieval visits exceed F/S/T bound")
    if snapshot_visit_exceeded_count:
        failures.append("fixed retrieval snapshot visits exceed F/S/T bound")
    if total_visit_exceeded_count:
        failures.append("fixed retrieval total visits exceed F/S/T bound")
    if distance_coverage_mismatch_count:
        failures.append("fixed retrieval distance work exceeds fetched-row coverage")
    if activated_identity_exceeded_count:
        failures.append("fixed retrieval activation exceeds F/S/T target")
    if epoch_edge_mismatch_count:
        failures.append("fixed retrieval envelope changes across consolidation")
    if len(recentered_events) < RETRIEVAL_CYCLE_MINIMUM_MATERIAL_CYCLES:
        failures.append("fewer than ten mature fixed-envelope recenters")
    if recenter_envelope_mismatch_count:
        failures.append("fixed retrieval recenter changes F/S/T envelope")
    classifier_passed = not failures
    overlap_recorded = (
        bool(recentered_events)
        and overlap_missing_count == 0
        and overlap_invalid_count == 0
        and len(overlap_ratios) == len(recentered_events)
    )
    full_cycle_passed = (
        classifier_passed
        and overlap_recorded
        and overlap_changed_count > 0
    )
    full_cycle_unproven_reason = None
    if not overlap_recorded:
        full_cycle_unproven_reason = (
            "profile lacks valid pre/post activated-identity overlap across "
            "every recenter"
        )
    elif overlap_changed_count == 0:
        full_cycle_unproven_reason = (
            "recenter overlap is valid but activated identity membership never changes"
        )
    full_failures = list(failures)
    if full_cycle_unproven_reason is not None:
        full_failures.append(full_cycle_unproven_reason)
    return {
        "passed": full_cycle_passed,
        "fixed_envelope_classifier_passed": classifier_passed,
        "fixed_work_envelope_subcontract_passed": classifier_passed,
        "activated_identity_overlap_recorded": overlap_recorded,
        "full_cycle_passed": full_cycle_passed,
        "full_cycle_unproven_reason": full_cycle_unproven_reason,
        "overlap_profile_missing_count": overlap_missing_count,
        "overlap_profile_invalid_count": overlap_invalid_count,
        "changed_activation_set_count": overlap_changed_count,
        "minimum_activation_overlap_ratio": (
            min(overlap_ratios) if overlap_ratios else None
        ),
        "maximum_activation_overlap_ratio": (
            max(overlap_ratios) if overlap_ratios else None
        ),
        "mode": "fixed-envelope",
        "material_cycle_count": 0,
        "complete_cycle_count": len(complete_epochs),
        "unevaluated_epoch_count": unevaluated_epoch_count,
        "work_mismatch_count": zero_work_count + work_bound_exceeded_count,
        "zero_work_count": zero_work_count,
        "work_bound_exceeded_count": work_bound_exceeded_count,
        "effort_mismatch_count": queue_effort_mismatch_count,
        "queue_effort_mismatch_count": queue_effort_mismatch_count,
        "node_ceiling_mismatch_count": node_ceiling_mismatch_count,
        "actual_visit_exceeded_count": actual_visit_exceeded_count,
        "snapshot_visit_exceeded_count": snapshot_visit_exceeded_count,
        "total_visit_exceeded_count": total_visit_exceeded_count,
        "distance_coverage_mismatch_count": distance_coverage_mismatch_count,
        "activated_identity_exceeded_count": (
            activated_identity_exceeded_count
        ),
        "activation_identity_target": activation_identity_target,
        "epoch_edge_mismatch_count": epoch_edge_mismatch_count,
        "mature_recenter_count": len(recentered_events),
        "recenter_envelope_mismatch_count": (
            recenter_envelope_mismatch_count
        ),
        "retrieval_work_bound": total_query_row_bound,
        "canonical_retrieval_node_bound": canonical_node_bound,
        "activation_snapshot_row_bound": activation_identity_target,
        "accepted_suffix_maximum_retrieval_work": max(
            accepted_suffix_retrieval_work, default=0.0
        ),
        "failures": sorted(set(full_failures)),
    }


def bounded_activation_quality_result(
    artifact: dict[str, Any], control_artifact: dict[str, Any],
    approved_control_sha256: str,
) -> dict[str, Any]:
    if artifact.get("schema") != "cortext_bounded_activation_shadow_quality_v1":
        raise ValueError("bounded activation quality schema mismatch")
    if control_artifact.get("schema") != artifact.get("schema"):
        raise ValueError("bounded activation quality control schema mismatch")
    if artifact.get("quality_role") != "candidate":
        raise ValueError("bounded activation candidate role is missing")
    if control_artifact.get("quality_role") != "approved-control":
        raise ValueError("approved bounded activation control is missing")
    if control_artifact.get("control_kind") != "current-public-retrieval":
        raise ValueError("approved control is not the current public retrieval path")
    existing_probe_digest = artifact.get("existing_fixed_probe_identity_rank_sha256")
    control_existing_probe_digest = control_artifact.get(
        "existing_fixed_probe_identity_rank_sha256"
    )
    if (
        not isinstance(existing_probe_digest, str)
        or len(existing_probe_digest) != 64
        or any(char not in "0123456789abcdef" for char in existing_probe_digest)
        or existing_probe_digest != control_existing_probe_digest
    ):
        raise ValueError("existing seven-probe digest is not control-bound")
    control_sha256 = hashlib.sha256(canonical_json(control_artifact)).hexdigest()
    if approved_control_sha256 != control_sha256:
        raise ValueError("bounded activation control digest is not approved")
    query_count = nonnegative_integer(artifact.get("query_count"), "query_count")
    result_k = nonnegative_integer(artifact.get("result_k"), "result_k")
    quality = artifact.get("quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("all"), dict):
        raise ValueError("bounded activation quality summary is missing")
    all_quality = quality["all"]
    evidence = ranked_query_evidence_result(artifact, result_k)
    control_result_k = nonnegative_integer(
        control_artifact.get("result_k"), "control result_k"
    )
    control_evidence = ranked_query_evidence_result(
        control_artifact, control_result_k
    )
    baseline_binding_passed = (
        control_result_k == result_k
        and control_artifact.get("query_count") == artifact.get("query_count")
        and control_artifact.get("embedding_count") == artifact.get("embedding_count")
    )
    for query_number in range(RETRIEVAL_QUALITY_QUERY_COUNT):
        candidate_row = evidence["by_number"][query_number]
        control_row = control_evidence["by_number"][query_number]
        baseline_binding_passed = baseline_binding_passed and all(
            candidate_row.get(field) == control_row.get(field)
            for field in (
                "query_index",
                "query_embedding_id",
                "exact_ranked_ids",
                "source_ids",
                "modalities",
                "history_ordinal",
                "history_count",
                "history_segment",
            )
        )
    summary_count = nonnegative_integer(
        all_quality.get("count"), "quality.all.count"
    )
    exact_recall = finite_fraction(
        all_quality.get("mean_exact_id_recall_at_k"),
        "quality.all.mean_exact_id_recall_at_k",
    )
    exact_top1 = finite_fraction(
        all_quality.get("mean_top1_exact_id"),
        "quality.all.mean_top1_exact_id",
    )
    semantic_coverage = finite_fraction(
        all_quality.get("mean_exact_neighbor_semantic_coverage"),
        "quality.all.mean_exact_neighbor_semantic_coverage",
    )
    for name, reported, measured in (
        ("mean_exact_id_recall_at_k", exact_recall,
         evidence["mean_exact_id_recall_at_k"]),
        ("mean_top1_exact_id", exact_top1, evidence["mean_top1_exact_id"]),
        ("mean_exact_neighbor_semantic_coverage", semantic_coverage,
         evidence["mean_exact_neighbor_semantic_coverage"]),
    ):
        if not math.isclose(reported, measured, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"quality.all.{name} is not bound to query evidence")
    coverage_passed = (
        evidence["source_count"] >= 4
        and {"text", "audio", "image"}.issubset(evidence["modalities"])
        and {"early", "middle", "late"}.issubset(
            evidence["history_segments"]
        )
    )
    reported_candidate_rank_digest = artifact.get(
        "candidate_identity_rank_sha256"
    )
    repeated_candidate_rank_digest = artifact.get(
        "repeat_candidate_identity_rank_sha256"
    )
    deterministic_candidate_rank_passed = (
        isinstance(reported_candidate_rank_digest, str)
        and len(reported_candidate_rank_digest) == 64
        and reported_candidate_rank_digest
        == evidence["candidate_identity_rank_sha256"]
        and repeated_candidate_rank_digest == reported_candidate_rank_digest
    )
    restart = artifact.get("sqlite_restart_measurements")
    if not isinstance(restart, list) or len(restart) != 3:
        raise ValueError("exactly three bounded activation restart measurements are required")
    derived_slot_bound = knob_fixed_embedding_slots(artifact)
    reported_slot_bound = nonnegative_integer(
        artifact.get("fixed_embedding_slots"), "fixed_embedding_slots"
    )
    if reported_slot_bound != derived_slot_bound:
        raise ValueError("fixed embedding slot bound does not match F/S/T")
    restart_rows = []
    restart_probe_digest = evidence["fixed_probe_identity_rank_sha256"]
    for item in restart:
        if not isinstance(item, dict):
            raise ValueError("restart measurement must be an object")
        restart_rows.append(
            {
                "fraction": finite_fraction(item.get("fraction"), "restart fraction"),
                "retained_rows": nonnegative_integer(
                    item.get("retained_rows"), "restart retained_rows"
                ),
                "rows_visited": nonnegative_integer(
                    item.get("rows_visited"), "restart rows_visited"
                ),
                "measurement_authority": item.get("measurement_authority"),
                "restored_candidate_count": nonnegative_integer(
                    item.get("restored_candidate_count"),
                    "restart restored_candidate_count",
                ),
                "pre_restart_probe_identity_rank_sha256": item.get(
                    "pre_restart_probe_identity_rank_sha256"
                ),
                "post_restart_probe_identity_rank_sha256": item.get(
                    "post_restart_probe_identity_rank_sha256"
                ),
                "claimed_linear_history": item.get("linear_history"),
                "claimed_production_gate": item.get("restart_production_gate"),
            }
        )
    expected_fractions = (0.25, 0.5, 1.0)
    for item, expected_fraction in zip(restart_rows, expected_fractions):
        expected_retained = max(
            1, math.floor(artifact["embedding_count"] * expected_fraction)
        )
        pre_digest = item["pre_restart_probe_identity_rank_sha256"]
        post_digest = item["post_restart_probe_identity_rank_sha256"]
        if (
            item["fraction"] != expected_fraction
            or item["retained_rows"] != expected_retained
            or item["rows_visited"] == 0
            or item["restored_candidate_count"] < result_k
            or not isinstance(pre_digest, str)
            or len(pre_digest) != 64
            or any(char not in "0123456789abcdef" for char in pre_digest)
            or pre_digest != post_digest
            or (
                expected_fraction == 1.0
                and pre_digest != restart_probe_digest
            )
        ):
            raise ValueError(
                "restart measurement is not bound to restored corpus state"
            )
    maximum_retained = restart_rows[-1]["retained_rows"]
    maximum_rows_visited = max(item["rows_visited"] for item in restart_rows)
    measured_restart_bounded = (
        maximum_retained > derived_slot_bound
        and maximum_rows_visited <= derived_slot_bound
    )
    claims_consistent = all(
        item["measurement_authority"]
        == "production-shaped-persistent-restart"
        and item["claimed_linear_history"] is False
        and item["claimed_production_gate"] is True
        for item in restart_rows
    )
    restart_passed = measured_restart_bounded and claims_consistent
    failures = []
    if query_count != RETRIEVAL_QUALITY_QUERY_COUNT:
        failures.append("quality query count mismatch")
    if summary_count != query_count or evidence["count"] != query_count:
        failures.append("quality query evidence count mismatch")
    if result_k != RETRIEVAL_QUALITY_K:
        failures.append("quality result k mismatch")
    if exact_recall < RETRIEVAL_EXACT_ID_RECALL_MIN:
        failures.append("exact identity recall below invariant")
    if exact_top1 < RETRIEVAL_EXACT_TOP1_MIN:
        failures.append("exact top-1 below invariant")
    exact_top1_miss_count = query_count - evidence["exact_top1_hit_count"]
    if exact_top1_miss_count > RETRIEVAL_EXACT_TOP1_MAX_MISSES:
        failures.append("exact top-1 miss count exceeds invariant")
    if not deterministic_candidate_rank_passed:
        failures.append("candidate rank order is not repeat-deterministic")
    if semantic_coverage < RETRIEVAL_SEMANTIC_COVERAGE_MIN:
        failures.append("semantic coverage below threshold")
    if not coverage_passed:
        failures.append("mixed source modality and history coverage missing")
    if not evidence["fixed_probes_match"]:
        failures.append("fixed identity-rank probe differs")
    if (
        not baseline_binding_passed
        or evidence["fixed_probe_identity_rank_sha256"]
        != control_evidence["fixed_probe_identity_rank_sha256"]
    ):
        failures.append("candidate evidence differs from approved control corpus")
    if not restart_passed:
        failures.append("restart remains proportional to history")
    return {
        "schema": "cortext_bounded_activation_quality_audit_v1",
        "passed": not failures,
        "query_count": query_count,
        "result_k": result_k,
        "mean_exact_id_recall_at_k": exact_recall,
        "mean_top1_exact_id": exact_top1,
        "exact_top1_miss_count": exact_top1_miss_count,
        "mean_exact_neighbor_semantic_coverage": semantic_coverage,
        "deterministic_candidate_rank_order_passed": (
            deterministic_candidate_rank_passed
        ),
        "candidate_identity_rank_sha256": reported_candidate_rank_digest,
        "query_input_coverage_passed": coverage_passed,
        "query_source_count": evidence["source_count"],
        "query_modalities": evidence["modalities"],
        "query_history_segments": evidence["history_segments"],
        "fixed_identity_rank_probes_passed": evidence["fixed_probes_match"],
        "fixed_probe_identity_rank_sha256": evidence[
            "fixed_probe_identity_rank_sha256"
        ],
        "existing_fixed_probe_identity_rank_sha256": existing_probe_digest,
        "approved_control_artifact_sha256": control_sha256,
        "approved_control_binding_passed": baseline_binding_passed,
        "bounded_restart_passed": restart_passed,
        "derived_restart_row_visit_bound": derived_slot_bound,
        "maximum_restart_retained_rows": maximum_retained,
        "maximum_restart_rows_visited": maximum_rows_visited,
        "failures": sorted(set(failures)),
    }


def consolidation_epoch_result(
    profile: dict[str, Any], rows: Sequence[dict[str, Any]],
    suffix_start: int, suffix_end: int,
) -> dict[str, Any]:
    events = validate_consolidation_epoch_relations(profile, rows)
    events = [
        event for event in events
        if suffix_start <= event["event_index"] < suffix_end
    ]
    complete_epochs = []
    sparse_cycle_contract = isinstance(
        profile.get("sparse_route_parameters"), dict
    )
    dynamic_sparse_cycle = bool(
        sparse_cycle_contract
        and nonnegative_integer(
            profile["sparse_route_parameters"].get(
                "activation_search_node_budget_step"
            ),
            "sparse_route_parameters.activation_search_node_budget_step",
        )
        > 0
    )
    k = CONSOLIDATION_EPOCH_PRE_POST_EVENTS
    for event_index, event in enumerate(events):
        close = event["event_index"]
        sealed_count = event["sealed_epoch_event_count"]
        start = close - sealed_count + 1
        next_close = (
            events[event_index + 1]["event_index"]
            if event_index + 1 < len(events)
            else suffix_end
        )
        if (
            start < suffix_start
            or sealed_count < 2 * k
            or close + k >= suffix_end
            or next_close - close < k
        ):
            continue
        epoch_rows = rows[start : close + 1]
        post_rows = rows[close + 1 : close + 1 + k]
        leading = statistics.mean(
            float(row["process_ms"]) for row in epoch_rows[:k]
        )
        trailing = statistics.mean(
            float(row["process_ms"]) for row in epoch_rows[-k:]
        )
        post = statistics.mean(float(row["process_ms"]) for row in post_rows)
        if sparse_cycle_contract:
            retrieval_pairs = [
                (
                    (
                        sparse_route_retrieval_actual_visits(row)
                        if dynamic_sparse_cycle
                        else sparse_route_retrieval_work(row)
                    ),
                    sparse_route_retrieval_effort(row),
                )
                for row in epoch_rows
                if sparse_route_retrieval_active(row)
            ]
            post_pairs = [
                (
                    (
                        sparse_route_retrieval_actual_visits(row)
                        if dynamic_sparse_cycle
                        else sparse_route_retrieval_work(row)
                    ),
                    sparse_route_retrieval_effort(row),
                )
                for row in post_rows
                if sparse_route_retrieval_active(row)
            ]
            edge_samples = knob_sparse_cycle_edge_samples(profile)
            retrieval_cycle_evaluated = (
                len(retrieval_pairs) >= 2 * edge_samples
                and len(post_pairs) >= edge_samples
                and all(effort > 0.0 for _, effort in retrieval_pairs)
            )
            retrieval_values = [work for work, _ in retrieval_pairs]
            retrieval_efforts = [effort for _, effort in retrieval_pairs]
            if retrieval_cycle_evaluated:
                retrieval_leading = statistics.mean(
                    retrieval_values[:edge_samples]
                )
                retrieval_trailing = statistics.mean(
                    retrieval_values[-edge_samples:]
                )
                retrieval_post = statistics.mean(
                    work for work, _ in post_pairs[:edge_samples]
                )
                effort_leading = statistics.mean(
                    retrieval_efforts[:edge_samples]
                )
                effort_trailing = statistics.mean(
                    retrieval_efforts[-edge_samples:]
                )
            else:
                retrieval_leading = retrieval_trailing = retrieval_post = 0.0
                effort_leading = effort_trailing = 0.0
        else:
            retrieval_values = [retrieval_work(row) for row in epoch_rows]
            retrieval_efforts = retrieval_values
            retrieval_leading = statistics.mean(retrieval_values[:k])
            retrieval_trailing = statistics.mean(retrieval_values[-k:])
            retrieval_post = statistics.mean(
                retrieval_work(row) for row in post_rows
            )
            effort_leading = retrieval_leading
            effort_trailing = retrieval_trailing
            retrieval_cycle_evaluated = True
        retrieval_ramp = retrieval_trailing - retrieval_leading
        retrieval_effort_ramp = effort_trailing - effort_leading
        retrieval_negative_variation = sum(
            max(0.0, previous - current)
            for previous, current in zip(
                retrieval_efforts, retrieval_efforts[1:]
            )
        )
        ramp_ratio = ratio(trailing, leading)
        post_pre_ratio = ratio(post, trailing)
        reset_ratios = {
            name: ratio(
                float(event["post_reset_counters"][name]),
                float(event["pre_reset_counters"][name]),
            )
            for name in REQUIRED_CONSOLIDATION_EPOCH_COUNTERS
        }
        material = ramp_ratio >= CONSOLIDATION_EPOCH_MATERIAL_RAMP_RATIO_MIN
        retrieval_material = (
            retrieval_cycle_evaluated
            and ratio(retrieval_trailing, retrieval_leading)
            >= CONSOLIDATION_EPOCH_MATERIAL_RAMP_RATIO_MIN
        )
        reset_failures = {
            name: value for name, value in reset_ratios.items()
            if name in RESETTABLE_CONSOLIDATION_EPOCH_COUNTERS
            if value > CONSOLIDATION_EPOCH_RESET_COUNTER_RATIO_MAX
            and not (
                float(event["pre_reset_counters"][name]) == 0.0
                and float(event["post_reset_counters"][name]) == 0.0
            )
        }
        complete_epochs.append({
            "consolidation_epoch_id": event["consolidation_epoch_id"],
            "start_event": start,
            "closing_event": close,
            "sealed_epoch_event_count": sealed_count,
            "sealed_epoch_mutation_count": event["sealed_epoch_mutation_count"],
            "leading_mean_process_ms": leading,
            "trailing_peak_mean_process_ms": trailing,
            "following_trough_mean_process_ms": post,
            "trailing_over_leading": ramp_ratio,
            "post_over_trailing": post_pre_ratio,
            "leading_mean_retrieval_work": retrieval_leading,
            "trailing_peak_mean_retrieval_work": retrieval_trailing,
            "following_trough_mean_retrieval_work": retrieval_post,
            "retrieval_post_over_trailing": ratio(
                retrieval_post, retrieval_trailing
            ),
            "retrieval_trailing_over_leading": ratio(
                retrieval_trailing, retrieval_leading
            ),
            "maximum_retrieval_work": max(retrieval_values, default=0.0),
            "retrieval_negative_variation_over_ramp": ratio(
                retrieval_negative_variation,
                max(0.0, retrieval_effort_ramp),
            ),
            "retrieval_max_over_trailing_mean": ratio(
                max(retrieval_efforts, default=0.0), effort_trailing
            ),
            "_retrieval_cycle_profile": normalized_cycle(
                resample_equal_progress(
                    retrieval_efforts, RETRIEVAL_CYCLE_RESAMPLE_BINS
                ),
                effort_leading,
                effort_trailing,
            ) if retrieval_cycle_evaluated else [],
            "material": material,
            "retrieval_cycle_evaluated": retrieval_cycle_evaluated,
            "retrieval_material": retrieval_material,
            "reset_counter_ratios": reset_ratios,
            "reset_counter_failures": reset_failures,
            "sqlite_sparse_route_recenter_succeeded": (
                event.get("sqlite_sparse_route_recenter_succeeded") is True
            ),
            "sqlite_sparse_route_activation_search_effort": (
                finite_nonnegative(
                    event.get(
                        "sqlite_sparse_route_activation_search_effort", 0.0
                    ),
                    "consolidation sqlite sparse route activation effort",
                )
            ),
            "sqlite_sparse_route_activation_node_budget": (
                finite_nonnegative(
                    event.get(
                        "sqlite_sparse_route_activation_node_budget", 0.0
                    ),
                    "consolidation sqlite sparse route activation node budget",
                )
            ),
            "consolidation_ms_per_event": ratio(
                float(event["duration_ms"]), float(max(1, sealed_count))
            ),
            "consolidation_ms_per_mutation": ratio(
                float(event["duration_ms"]),
                float(max(1, event["sealed_epoch_mutation_count"])),
            ),
        })

    material_epochs = [epoch for epoch in complete_epochs if epoch["material"]]
    retrieval_material_epochs = [
        epoch for epoch in complete_epochs if epoch["retrieval_material"]
    ]
    failures = []
    mode = "invalid"
    peak_trend = None
    trough_trend = None
    normalized_event_cost = None
    normalized_mutation_cost = None
    frequency_trend = None
    frequency_epochs = [
        event for event in events
        if event["event_index"] - event["sealed_epoch_event_count"] + 1
        >= suffix_start
    ]
    if any(epoch["reset_counter_failures"] for epoch in complete_epochs):
        failures.append("consolidation reset counter does not fall")
    if len(material_epochs) >= CONSOLIDATION_EPOCH_MINIMUM_MATERIAL_RESETS:
        mode = "sawtooth"
        for epoch in material_epochs:
            if epoch["post_over_trailing"] > CONSOLIDATION_EPOCH_POST_PRE_PROCESS_RATIO_MAX:
                failures.append("material epoch does not lower following process time")
        peak_trend = sequence_trend(
            [epoch["trailing_peak_mean_process_ms"] for epoch in material_epochs]
        )
        trough_trend = sequence_trend(
            [epoch["following_trough_mean_process_ms"] for epoch in material_epochs]
        )
        if peak_trend["half"]["second_over_first"] > CONSOLIDATION_EPOCH_PEAK_HALF_RATIO_MAX:
            failures.append("sawtooth peaks rise")
        if trough_trend["half"]["second_over_first"] > CONSOLIDATION_EPOCH_TROUGH_HALF_RATIO_MAX:
            failures.append("sawtooth troughs rise")
        for trend, name in ((peak_trend, "peak"), (trough_trend, "trough")):
            if trend["relative_theil_sen_per_epoch"] > RELATIVE_THEIL_SEN_MAX:
                failures.append(f"{name} relative slope rises")
            if trend["relative_bootstrap_95_upper_per_epoch"] > RELATIVE_BOOTSTRAP_UPPER_MAX:
                failures.append(f"{name} bootstrap upper rises")
    elif len(material_epochs) > 0:
        failures.append("one to three material epochs are insufficient")
    elif not events:
        mode = "flat-envelope"
    elif len(complete_epochs) >= CONSOLIDATION_EPOCH_MINIMUM_MATERIAL_RESETS:
        mode = "flat-envelope"
    else:
        failures.append("successful consolidations lack four complete epochs")

    if len(frequency_epochs) >= CONSOLIDATION_EPOCH_MINIMUM_MATERIAL_RESETS:
        frequency_trend = sequence_trend([
            ratio(1.0, float(event["sealed_epoch_event_count"]))
            for event in frequency_epochs
        ])
        frequency_trend["observed_epoch_count"] = len(frequency_epochs)
        frequency_trend["left_censored_close_count"] = (
            len(events) - len(frequency_epochs)
        )
        if (
            frequency_trend["half"]["second_over_first"]
            > CONSOLIDATION_EPOCH_FREQUENCY_HALF_RATIO_MAX
        ):
            failures.append("consolidation frequency rises")
    if len(complete_epochs) >= CONSOLIDATION_EPOCH_MINIMUM_MATERIAL_RESETS:
        normalized_event_cost = sequence_trend(
            [epoch["consolidation_ms_per_event"] for epoch in complete_epochs]
        )
        normalized_mutation_cost = sequence_trend(
            [epoch["consolidation_ms_per_mutation"] for epoch in complete_epochs]
        )
        if normalized_event_cost["half"]["second_over_first"] > CONSOLIDATION_EPOCH_NORMALIZED_COST_HALF_RATIO_MAX:
            failures.append("consolidation cost per sealed event rises")
        if normalized_mutation_cost["half"]["second_over_first"] > CONSOLIDATION_EPOCH_NORMALIZED_COST_HALF_RATIO_MAX:
            failures.append("consolidation cost per sealed mutation rises")

    retrieval_cycle_symmetry = None
    if sparse_cycle_contract and profile.get("retention") == "natural":
        accepted_sparse_rows = [
            row
            for row in rows
            if suffix_start
            <= nonnegative_integer(row.get("event_index"), "event_index")
            < suffix_end
            and sparse_route_retrieval_active(row)
        ]
        accepted_sparse_work = [
            sparse_route_retrieval_work(row) for row in accepted_sparse_rows
        ]
        accepted_sparse_effort = [
            sparse_route_retrieval_effort(row) for row in accepted_sparse_rows
        ]
        accepted_sparse_queue_effort = [
            sparse_route_retrieval_queue_effort(row)
            for row in accepted_sparse_rows
        ]
        accepted_sparse_actual_visits = [
            sparse_route_retrieval_actual_visits(row)
            for row in accepted_sparse_rows
        ]
        accepted_sparse_snapshot_visits = [
            sparse_route_activation_snapshot_visits(row)
            for row in accepted_sparse_rows
        ]
        accepted_sparse_activated_identities = [
            sparse_route_retrieval_activated_identities(row)
            for row in accepted_sparse_rows
        ]
        retrieval_work_bound = knob_sparse_retrieval_work_bound(profile)
        total_query_row_bound = knob_sparse_total_query_row_bound(profile)
        fixed_envelope_selected = (
            nonnegative_integer(
                profile["sparse_route_parameters"].get(
                    "activation_search_node_budget_min"
                ),
                "sparse_route_parameters.activation_search_node_budget_min",
            )
            == retrieval_work_bound
            and nonnegative_integer(
                profile["sparse_route_parameters"].get(
                    "activation_search_node_budget_step"
                ),
                "sparse_route_parameters.activation_search_node_budget_step",
            )
            == 0
            and nonnegative_integer(
                profile["sparse_route_parameters"].get(
                    "activation_search_effort_min"
                ),
                "sparse_route_parameters.activation_search_effort_min",
            )
            == retrieval_work_bound
            and nonnegative_integer(
                profile["sparse_route_parameters"].get(
                    "activation_search_effort_step"
                ),
                "sparse_route_parameters.activation_search_effort_step",
            )
            == 0
        )
        if fixed_envelope_selected:
            retrieval_cycle_symmetry = fixed_retrieval_envelope_result(
                complete_epochs,
                events,
                retrieval_work_bound,
                total_query_row_bound,
                accepted_sparse_work,
                accepted_sparse_queue_effort,
                accepted_sparse_effort,
                accepted_sparse_actual_visits,
                accepted_sparse_snapshot_visits,
                accepted_sparse_activated_identities,
                nonnegative_integer(
                    profile["sparse_route_parameters"].get(
                        "activation_identity_target"
                    ),
                    "sparse_route_parameters.activation_identity_target",
                ),
            )
        else:
            retrieval_cycle_symmetry = retrieval_cycle_symmetry_result(
                retrieval_material_epochs,
                retrieval_work_bound,
                accepted_sparse_actual_visits,
                accepted_sparse_effort,
                accepted_suffix_queue_effort=accepted_sparse_queue_effort,
                queue_effort_bound=nonnegative_integer(
                    profile["sparse_route_parameters"].get("search_effort"),
                    "sparse_route_parameters.search_effort",
                    positive=True,
                ),
                accepted_suffix_total_work=accepted_sparse_work,
                total_query_row_bound=total_query_row_bound,
                accepted_suffix_snapshot_visits=(
                    accepted_sparse_snapshot_visits
                ),
                activation_identity_target=nonnegative_integer(
                    profile["sparse_route_parameters"].get(
                        "activation_identity_target"
                    ),
                    "sparse_route_parameters.activation_identity_target",
                ),
                accepted_suffix_activated_identities=(
                    accepted_sparse_activated_identities
                ),
            )
        failures.extend(retrieval_cycle_symmetry["failures"])
    elif mode == "sawtooth":
        retrieval_cycle_symmetry = retrieval_cycle_symmetry_result(
            material_epochs,
            knob_normal_comparison_bound(profile),
            [
                retrieval_work(row)
                for row in rows
                if suffix_start
                <= nonnegative_integer(row.get("event_index"), "event_index")
                < suffix_end
            ],
            [
                retrieval_work(row)
                for row in rows
                if suffix_start
                <= nonnegative_integer(row.get("event_index"), "event_index")
                < suffix_end
            ],
        )
        failures.extend(retrieval_cycle_symmetry["failures"])
    elif mode == "flat-envelope" and profile.get("retention") == "natural":
        failures.append("natural cutover requires retrieval cycle symmetry")

    activity_normalized_reset = None
    reset_miss_epochs = [
        epoch for epoch in material_epochs
        if epoch["post_over_trailing"]
        > CONSOLIDATION_EPOCH_POST_PRE_PROCESS_RATIO_MAX
    ]
    if reset_miss_epochs:
        evaluation_inputs = []
        operation_activity_pairs = (
            (
                "MemoryStorage",
                "cortext::operations::MemoryStorage",
                "supersession_current_candidate_count",
            ),
            (
                "PropagateEmotionalCascade",
                "cortext::operations::PropagateEmotionalCascade",
                "emotional_neighbor_count",
            ),
        )
        for epoch in reset_miss_epochs:
            close = int(epoch["closing_event"])
            trailing_rows = rows[close - k + 1 : close + 1]
            post_rows = rows[close + 1 : close + 1 + k]
            for operation, operation_key, activity_key in operation_activity_pairs:
                evaluation_inputs.append({
                    "epoch": epoch["consolidation_epoch_id"],
                    "operation": operation,
                    "trailing_event_indices": [
                        row["event_index"] for row in trailing_rows
                    ],
                    "post_event_indices": [
                        row["event_index"] for row in post_rows
                    ],
                    "trailing_ms": [
                        row["operation_ms"][operation_key]
                        for row in trailing_rows
                    ],
                    "post_ms": [
                        row["operation_ms"][operation_key]
                        for row in post_rows
                    ],
                    "trailing_activity": [
                        row["work_counters"][activity_key]
                        for row in trailing_rows
                    ],
                    "post_activity": [
                        row["work_counters"][activity_key]
                        for row in post_rows
                    ],
                })
        raw_gate_failure_names = {
            "material epoch does not lower following process time",
            "sawtooth peaks rise",
            "sawtooth troughs rise",
            "peak relative slope rises",
            "trough relative slope rises",
            "peak bootstrap upper rises",
            "trough bootstrap upper rises",
            "consolidation reset counter does not fall",
        }
        activity_normalized_reset = activity_normalized_reset_diagnostic(
            evaluation_inputs,
            sorted({
                failure for failure in failures
                if failure in raw_gate_failure_names
            }),
        )
    for epoch in complete_epochs:
        epoch.pop("_retrieval_cycle_profile", None)

    return {
        "passed": mode != "invalid" and not failures,
        "mode": mode,
        "successful_consolidation_count": len(events),
        "complete_epoch_count": len(complete_epochs),
        "material_epoch_count": len(material_epochs),
        "complete_epochs": complete_epochs,
        "peak_trend": peak_trend,
        "trough_trend": trough_trend,
        "normalized_event_cost_trend": normalized_event_cost,
        "normalized_mutation_cost_trend": normalized_mutation_cost,
        "frequency_trend": frequency_trend,
        "retrieval_cycle_symmetry": retrieval_cycle_symmetry,
        "activity_normalized_reset_diagnostic": activity_normalized_reset,
        "failures": sorted(set(failures)),
    }


def checkpoint_map(
    profile: dict[str, Any], required_boundaries: set[int]
) -> dict[int, dict[str, float]]:
    checkpoints = profile.get("store_checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("profile lacks store_checkpoints")
    mapped: dict[int, dict[str, float]] = {}
    required_counts = {
        "memories", "signals", "associations", "embeddings",
        "current_memory_embeddings", "memory_reconstructions",
        "rif_recovery_clock", "rif_generation_resets", "rif_active_state",
    }
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("store checkpoint must be an object")
        event_end = checkpoint.get("event_end")
        if isinstance(event_end, bool) or not isinstance(event_end, int):
            raise ValueError("store checkpoint event_end must be an integer")
        if event_end in mapped:
            raise ValueError("duplicate store checkpoint boundary")
        counts = checkpoint.get("counts")
        if not isinstance(counts, dict) or set(counts) != required_counts:
            raise ValueError("store checkpoint count schema mismatch")
        mapped[event_end] = {
            name: finite_nonnegative(value, f"store_checkpoints.{name}")
            for name, value in counts.items()
        }
    missing = sorted(required_boundaries - set(mapped))
    if missing:
        raise ValueError(f"store checkpoints missing window boundaries: {missing}")
    return mapped


def total_authoritative_rows(counts: dict[str, float]) -> float:
    return sum(counts.values())


def suffix_plateau_result(
    profile: dict[str, Any], rows: Sequence[dict[str, Any]], window_size: int
) -> dict[str, Any]:
    validate_profile_rows(profile, rows)
    event_count = len(rows)
    all_ranges = end_anchored_ranges(event_count, window_size)
    warmup_events = math.ceil(0.20 * event_count)
    ranges = [item for item in all_ranges if item[0] >= warmup_events]
    boundaries = {boundary for item in all_ranges for boundary in item}
    checkpoints = checkpoint_map(profile, boundaries)
    if len(ranges) < MINIMUM_POST_WARMUP_WINDOWS:
        return {
            "passed": False,
            "reason": "at least ten post-warmup end-anchored windows are required",
            "window_count": len(ranges),
            "warmup_events": warmup_events,
        }
    operation_names = sorted(
        name
        for name in set().union(*(row["operation_ms"] for row in rows))
        if is_duration_operation(name)
    )
    process_windows: list[float] = []
    throughput_windows: list[float] = []
    operation_windows = {name: [] for name in operation_names}
    counter_windows = {name: [] for name in REQUIRED_WORK_COUNTERS}
    window_table = []
    for start, end in ranges:
        window_rows = rows[start:end]
        process_mean = statistics.mean(float(row["process_ms"]) for row in window_rows)
        elapsed = sum(float(row["end_to_end_ms"]) for row in window_rows)
        throughput = 1000.0 * len(window_rows) / elapsed if elapsed > 0.0 else math.inf
        process_windows.append(process_mean)
        throughput_windows.append(throughput)
        for name in operation_names:
            operation_windows[name].append(
                statistics.mean(float(row["operation_ms"].get(name, 0.0)) for row in window_rows)
            )
        for name in REQUIRED_WORK_COUNTERS:
            counter_windows[name].append(
                statistics.mean(float(row["work_counters"][name]) for row in window_rows)
            )
        window_table.append({
            "start_event": start,
            "end_event": end,
            "mean_process_ms": process_mean,
            "messages_per_second": throughput,
            "end_to_end_ms": elapsed,
            "authoritative_rows_end": total_authoritative_rows(checkpoints[end]),
        })

    retention = profile.get("retention")
    height_reference = plateau_height_reference(retention)
    height_ceiling = height_reference["plateau_process_ceiling_ms"]

    candidate_failures = []
    accepted = None
    for suffix_start in range(0, len(ranges) - PLATEAU_MINIMUM_WINDOWS + 1):
        suffix_ranges = ranges[suffix_start:]
        suffix_process = process_windows[suffix_start:]
        suffix_throughput = throughput_windows[suffix_start:]
        process_half = half_comparison(suffix_process)
        throughput_half = half_comparison(suffix_throughput)
        suffix_mean = statistics.mean(suffix_process)
        xs = [float(index) for index in range(len(suffix_process))]
        slope = theil_sen(xs, suffix_process)
        upper = bootstrap_upper_slope(xs, suffix_process, BOOTSTRAP_REPETITIONS)
        relative_slope = slope / suffix_mean if suffix_mean > 0.0 else math.inf
        relative_upper = upper / suffix_mean if suffix_mean > 0.0 else math.inf
        operation_failures = {}
        operation_results = {}
        for name, values in operation_windows.items():
            suffix_values = values[suffix_start:]
            mean_value = statistics.mean(suffix_values)
            if mean_value < OPERATION_MINIMUM_MEAN_MS:
                continue
            result = half_comparison(suffix_values)
            operation_results[name] = {"mean_ms": mean_value, **result}
            if result["second_over_first"] > OPERATION_HALF_RATIO_MAX:
                operation_failures[name] = result["second_over_first"]
        start_event = suffix_ranges[0][0]
        end_event = suffix_ranges[-1][1]
        counter_results = {}
        counter_failures = {}
        for name, values in counter_windows.items():
            result = half_comparison(values[suffix_start:])
            bound = knob_work_counter_bound(profile, name)
            maximum = max(
                float(row["work_counters"][name])
                for row in rows[start_event:end_event]
            )
            absolute_bound_passed = bound is not None and maximum <= bound
            counter_results[name] = {
                **result,
                "maximum": maximum,
                "knob_derived_absolute_bound": bound,
                "absolute_bound_passed": absolute_bound_passed,
            }
            if (
                result["second_over_first"] > WORK_COUNTER_HALF_RATIO_MAX
                and not absolute_bound_passed
            ) or (bound is not None and maximum > bound):
                counter_failures[name] = result["second_over_first"]
        half = len(suffix_ranges) // 2
        middle_event = suffix_ranges[half][0]
        start_rows = total_authoritative_rows(checkpoints[start_event])
        middle_rows = total_authoritative_rows(checkpoints[middle_event])
        end_rows = total_authoritative_rows(checkpoints[end_event])
        store_growth_ratio = ratio(end_rows, start_rows)
        store_growth_passed = (
            store_growth_ratio >= AUTHORITATIVE_ROW_GROWTH_RATIO_MIN
            and middle_rows > start_rows
            and end_rows > middle_rows
        )
        epoch_result = consolidation_epoch_result(
            profile, rows, start_event, end_event
        )
        passed = (
            PROCESS_HALF_RATIO_RANGE[0] <= process_half["second_over_first"] <= PROCESS_HALF_RATIO_RANGE[1]
            and THROUGHPUT_HALF_RATIO_RANGE[0] <= throughput_half["second_over_first"] <= THROUGHPUT_HALF_RATIO_RANGE[1]
            and relative_slope <= RELATIVE_THEIL_SEN_MAX
            and relative_upper <= RELATIVE_BOOTSTRAP_UPPER_MAX
            and not operation_failures
            and not counter_failures
            and store_growth_passed
            and suffix_mean <= height_ceiling
            and end_event == event_count
            and epoch_result["passed"]
        )
        result = {
            "passed": passed,
            "plateau_start_event": start_event,
            "plateau_end_event": end_event,
            "plateau_window_count": len(suffix_ranges),
            "ramp_duration_events": start_event,
            "ramp_complete_window_count": len(all_ranges) - len(suffix_ranges),
            "plateau_mean_process_ms": suffix_mean,
            "height_ceiling_ms": height_ceiling,
            "height_reference": height_reference,
            "process_half": process_half,
            "throughput_half": throughput_half,
            "relative_theil_sen_per_window": relative_slope,
            "relative_bootstrap_95_upper_per_window": relative_upper,
            "operation_results": operation_results,
            "operation_failures": operation_failures,
            "work_counter_results": counter_results,
            "work_counter_failures": counter_failures,
            "authoritative_rows_start": start_rows,
            "authoritative_rows_middle": middle_rows,
            "authoritative_rows_end": end_rows,
            "authoritative_row_growth_ratio": store_growth_ratio,
            "authoritative_store_growth_passed": store_growth_passed,
            "consolidation_epoch": epoch_result,
        }
        if passed:
            accepted = result
            break
        candidate_failures.append({
            "plateau_start_event": start_event,
            "plateau_window_count": len(suffix_ranges),
            "process_ratio": process_half["second_over_first"],
            "throughput_ratio": throughput_half["second_over_first"],
            "relative_slope": relative_slope,
            "relative_upper": relative_upper,
            "operation_failure_count": len(operation_failures),
            "operation_failures": sorted(operation_failures),
            "counter_failure_count": len(counter_failures),
            "counter_failures": sorted(counter_failures),
            "store_growth_passed": store_growth_passed,
            "height_passed": suffix_mean <= height_ceiling,
            "consolidation_epoch_passed": epoch_result["passed"],
            "consolidation_mode": epoch_result["mode"],
            "consolidation_epoch_failures": epoch_result["failures"],
        })
    return {
        "passed": accepted is not None,
        "warmup_events": warmup_events,
        "window_size": window_size,
        "height_reference": height_reference,
        "eligible_window_count": len(ranges),
        "leading_remainder_events": event_count % window_size,
        "final_event_covered": bool(ranges and ranges[-1][1] == event_count),
        "accepted_suffix": accepted,
        "candidate_failures": candidate_failures,
        "window_table": window_table,
    }


def attribution_result(
    rows: Sequence[dict[str, Any]], window_size: int
) -> dict[str, Any]:
    warmup = math.ceil(0.20 * len(rows))
    process_means = windows(
        [float(row["process_ms"]) for row in rows], warmup, window_size
    )
    latency_means = windows(
        [float(row["latency_ms"]) for row in rows], warmup, window_size
    )
    if len(process_means) < 10:
        return {
            "passed": False,
            "reason": "at least ten post-warmup windows are required",
            "window_count": len(process_means),
        }

    all_operation_names = set().union(
        *(row.get("operation_ms", {}) for row in rows)
    )
    operation_names = sorted(
        name for name in all_operation_names
        if name.startswith(ATTRIBUTION_PREFIXES) and is_duration_operation(name)
    )
    detailed_names = sorted(
        name for name in all_operation_names
        if name.startswith(DETAILED_PREFIXES) and is_duration_operation(name)
    )
    counter_names = sorted(
        name for name in all_operation_names
        if not is_duration_operation(name)
    )
    operation_windows = {
        name: windows(operation_values(rows, name), warmup, window_size)
        for name in operation_names
    }
    detailed_windows = {
        name: windows(operation_values(rows, name), warmup, window_size)
        for name in detailed_names
    }
    operation_comparisons = {
        name: comparison(means) for name, means in operation_windows.items()
    }
    detailed_comparisons = {
        name: comparison(means) for name, means in detailed_windows.items()
    }
    counter_comparisons = {
        name: counter_comparison(
            windows(operation_values(rows, name), warmup, window_size)
        )
        for name in counter_names
    }
    process_comparison = comparison(process_means)
    positive_process_delta = max(0.0, process_comparison["delta_ms"])
    positive_named_delta = sum(
        max(0.0, values["delta_ms"])
        for values in operation_comparisons.values()
    )
    signed_named_delta = sum(
        values["delta_ms"] for values in operation_comparisons.values()
    )
    ranked = sorted(
        (
            {
                "operation": name,
                **values,
                "fraction_of_process_delta": (
                    max(0.0, values["delta_ms"]) / positive_process_delta
                    if positive_process_delta > 0.0
                    else 0.0
                ),
            }
            for name, values in operation_comparisons.items()
            if values["delta_ms"] > 0.0
        ),
        key=lambda item: item["delta_ms"],
        reverse=True,
    )
    window_table = []
    for index, process_mean in enumerate(process_means):
        start = warmup + index * window_size
        window_table.append(
            {
                "start_event": start,
                "end_event_exclusive": start + window_size,
                "mean_process_ms": process_mean,
                "mean_latency_ms": latency_means[index],
                "operation_ms": {
                    name: means[index]
                    for name, means in operation_windows.items()
                },
            }
        )
    raw_explained_fraction = (
        signed_named_delta / positive_process_delta
        if positive_process_delta > 0.0
        else 1.0
    )
    explained_fraction = min(1.0, max(0.0, raw_explained_fraction))
    return {
        "passed": explained_fraction >= 0.80,
        "warmup_events": warmup,
        "window_size": window_size,
        "window_count": len(process_means),
        "process": process_comparison,
        "named_positive_delta_ms": positive_named_delta,
        "named_signed_delta_ms": signed_named_delta,
        "residual_delta_ms": process_comparison["delta_ms"] - signed_named_delta,
        "raw_explained_fraction": raw_explained_fraction,
        "explained_fraction": explained_fraction,
        "ranked_positive_contributors": ranked,
        "detailed_subsections": detailed_comparisons,
        "diagnostic_counters": counter_comparisons,
        "window_table": window_table,
    }


def flatness_result(
    rows: Sequence[dict[str, Any]], operation: str, window_size: int
) -> dict[str, Any]:
    warmup = math.ceil(0.20 * len(rows))
    means = windows(operation_values(rows, operation), warmup, window_size)
    if len(means) < 10:
        return {
            "operation": operation,
            "passed": False,
            "reason": "at least ten post-warmup windows are required",
            "window_count": len(means),
        }
    xs = [float(warmup + window_size * index + window_size / 2) for index in range(len(means))]
    slope = theil_sen(xs, means)
    upper = bootstrap_upper_slope(xs, means)
    early = statistics.mean(means[:5])
    late = statistics.mean(means[-5:])
    ratio = late / early if early > 0.0 else math.inf
    return {
        "operation": operation,
        "passed": slope <= 0.01 and upper <= 0.02 and ratio <= 1.05,
        "window_count": len(means),
        "theil_sen_ms_per_message": slope,
        "bootstrap_95_upper_ms_per_message": upper,
        "final_five_over_first_five": ratio,
        "first_five_mean_ms": early,
        "final_five_mean_ms": late,
    }


def throughput_result(
    rows: Sequence[dict[str, Any]], window_size: int
) -> dict[str, Any]:
    warmup = math.ceil(0.20 * len(rows))
    process_means = windows(
        [float(row["process_ms"]) for row in rows], warmup, window_size
    )
    latency_means = windows(
        [float(row["latency_ms"]) for row in rows], warmup, window_size
    )
    if len(process_means) < 10 or len(latency_means) < 10:
        return {
            "passed": False,
            "reason": "at least ten post-warmup windows are required",
            "window_count": min(len(process_means), len(latency_means)),
        }
    messages_per_second = [
        1000.0 / latency_ms if latency_ms > 0.0 else math.inf
        for latency_ms in latency_means
    ]
    early_process = statistics.mean(process_means[:5])
    late_process = statistics.mean(process_means[-5:])
    early_rate = statistics.mean(messages_per_second[:5])
    late_rate = statistics.mean(messages_per_second[-5:])
    process_ratio = late_process / early_process if early_process > 0.0 else math.inf
    rate_ratio = late_rate / early_rate if early_rate > 0.0 else 0.0
    return {
        "passed": 0.90 <= process_ratio <= 1.10 and 0.90 <= rate_ratio <= 1.10,
        "window_count": len(process_means),
        "first_five_mean_process_ms": early_process,
        "final_five_mean_process_ms": late_process,
        "final_five_over_first_five_process": process_ratio,
        "first_five_mean_messages_per_second": early_rate,
        "final_five_mean_messages_per_second": late_rate,
        "final_five_over_first_five_messages_per_second": rate_ratio,
    }


def profile_summary(
    profile_path: Path, db_path: Path, window_size: int,
    include_plateau: bool = True,
) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = profile.get("working_set_curve")
    if not isinstance(rows, list) or not rows:
        raise ValueError("profile has no working-set curve")
    db_hash, counts = logical_database_digest(db_path)
    flatness = [flatness_result(rows, name, window_size) for name in STORAGE_OPERATIONS]
    throughput = throughput_result(rows, window_size)
    attribution = attribution_result(rows, window_size)
    plateau = suffix_plateau_result(profile, rows, window_size) if include_plateau else None
    return {
        "schema": "cortext_storage_cost_profile_audit_v3",
        "processed_events": len(rows),
        "behavior_sha256": behavior_digest(rows, profile.get("behavior_sha256")),
        "logical_database_sha256": db_hash,
        "logical_table_counts": counts,
        "mean_process_ms": float(profile["mean_process_ms"]),
        "mean_total_ms": float(profile["mean_total_ms"]),
        "wall_ms": int(profile["wall_ms"]),
        "flatness": flatness,
        "flatness_passed": all(result["passed"] for result in flatness),
        "legacy_latency_throughput": throughput,
        "legacy_latency_throughput_passed": throughput["passed"],
        "attribution": attribution,
        "attribution_passed": attribution["passed"],
        "plateau": plateau,
        "plateau_passed": bool(plateau and plateau["passed"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", type=Path)
    source.add_argument("--shadow-quality", type=Path)
    parser.add_argument("--quality-control", type=Path)
    parser.add_argument("--approved-control-sha256")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--require-attribution", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shadow_quality:
        if args.quality_control is None:
            raise ValueError("--quality-control is required with --shadow-quality")
        artifact = json.loads(args.shadow_quality.read_text(encoding="utf-8"))
        if args.approved_control_sha256 is None:
            raise ValueError(
                "--approved-control-sha256 is required with --shadow-quality"
            )
        control_artifact = json.loads(args.quality_control.read_text(encoding="utf-8"))
        result = bounded_activation_quality_result(
            artifact, control_artifact, args.approved_control_sha256
        )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if args.report_only or result["passed"] else 1
    if args.db is None:
        raise ValueError("--db is required with --profile")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    result = profile_summary(
        args.profile, args.db, args.window_size,
        include_plateau=not args.require_attribution,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_attribution:
        passed = result["attribution_passed"]
    else:
        passed = result["plateau_passed"]
    return 0 if args.report_only or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
