#!/usr/bin/env python3
"""Focused contract tests for storage-cost profile attribution and CLI gates."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import copy
from pathlib import Path

try:
    import audit_storage_cost_profile as audit
except ModuleNotFoundError:
    from tools import audit_storage_cost_profile as audit


def fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(130):
        process_ms = 1.0 + 0.01 * index
        operation_ms = {
            "cortext::operations::GraphAugmentedRetrieveCandidates": process_ms,
            "cortext::operations::MemoryStorage": 0.1,
            "MemoryStorage.supersession_edges": 0.01,
        }
        if index >= 65:
            operation_ms["cortext::operations::LateMaintenance"] = 0.2 + 0.005 * (
                index - 65
            )
        operation_ms["EmotionalCascade.source_count"] = float(index)
        operation_ms["SignalProcessor.snapshot_cache_entry_copy_count"] = float(
            index * 100
        )
        rows.append(
            {
                "event_index": index,
                "modality": "fixture",
                "process_ms": process_ms,
                "latency_ms": process_ms,
                "operation_ms": operation_ms,
                "active_epoch": {
                    "event_count": index + 1,
                    "mutation_count": index + 1,
                    "allocated_bytes": 16384,
                    "row_batch_high_water": 0,
                    "required": False,
                },
                "behavior": {},
            }
        )
    return rows


def fixture_operation_ms(counters: dict[str, float]) -> dict[str, float]:
    operations = {
        "cortext::operations::GraphAugmentedRetrieveCandidates": 0.2,
        "GraphRetrieve.total": 0.2,
        "GraphRetrieve.seed_knn_cache": 0.1,
        "GraphRetrieve.seed_knn_cache_rebuild": 0.1,
        "GraphRetrieve.seed_cache_family_compare": 0.1,
        "GraphRetrieve.seed_cache_distance": 0.1,
        "GraphRetrieve.sqlite_sparse_route_distance_evaluations": 0.0,
        "cortext::operations::ApplyRetrievalCompetition": 0.1,
        "Competition.rif_recovery_active_sql": 0.1,
        "Competition.score_candidates": 0.1,
        "Predictive.decay_active_sql": 0.1,
        "cortext::operations::MemoryStorage": 0.1,
        "MemoryStorage.supersession_edges": 0.1,
        "MemoryStorage.supersession_candidate_load": 0.1,
        "MemoryStorage.supersession_current_candidate_execution_count": 0.1,
        "MemoryStorage.supersession_historical_candidate_execution_count": 0.1,
        "MemoryStorage.supersession_current_rows_visited": 0.1,
        "MemoryStorage.supersession_historical_rows_visited": 0.1,
        "MemoryStorage.supersession_sparse_route_node_rows": 0.0,
        "MemoryStorage.supersession_sparse_route_dirty_rows": 0.0,
        "MemoryStorage.supersession_sql_fallback_count": 0.0,
        "cortext::operations::PropagateEmotionalCascade": 0.1,
        "EmotionalCascade.source_query": 0.1,
        "EmotionalCascade.source_execution_count": 0.1,
        "EmotionalCascade.neighbor_execution_count": 0.1,
        "EmotionalCascade.update_execution_count": 0.1,
        "SignalProcessor.snapshot_full_cache_copy": 0.1,
        "SignalProcessor.snapshot_cache_entry_copy": 0.1,
    }
    for counter, sources in audit.COUNTER_SOURCE_OPERATIONS.items():
        if len(sources) == 1:
            operations[sources[0]] = float(counters[counter])
    operations["GraphRetrieve.rows_visited"] = float(
        counters["graph_rows_visited"]
    )
    operations["GraphRetrieve.sqlite_sparse_route_node_rows"] = 0.0
    operations[
        "GraphRetrieve.sqlite_sparse_route_activation_snapshot_rows"
    ] = 0.0
    operations["GraphRetrieve.sqlite_sparse_route_activated_identities"] = 0.0
    operations["MemoryStorage.supersession_current_rows_visited"] = float(
        counters["supersession_current_candidate_count"]
    )
    operations["MemoryStorage.supersession_historical_rows_visited"] = float(
        counters["supersession_historical_candidate_count"]
    )
    for counter, markers in audit.COUNTER_ACTIVITY_OPERATIONS.items():
        for marker in markers:
            operations[marker] = 1.0 if counters[counter] > 0.0 else 0.0
    return operations


def plateau_fixture(
    *,
    process_windows: list[float] | None = None,
    elapsed_windows: list[float] | None = None,
    stop_store_growth_at: int | None = None,
    growing_counter: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    event_count = 126
    window_size = 10
    ranges = audit.end_anchored_ranges(event_count, window_size)
    if process_windows is None:
        process_windows = [2.0, 4.0, 6.0, 8.0] + [10.0] * 8
    if elapsed_windows is None:
        elapsed_windows = process_windows
    rows: list[dict[str, object]] = []
    for event_index in range(event_count):
        window_index = max(
            0,
            min(
                len(ranges) - 1,
                next(
                    (
                        index
                        for index, (start, end) in enumerate(ranges)
                        if start <= event_index < end
                    ),
                    0,
                ),
            ),
        )
        counters = {name: 10.0 for name in audit.REQUIRED_WORK_COUNTERS}
        counters["rollback_full_cache_copy_count"] = 0.0
        counters["rollback_cache_entry_copy_count"] = 0.0
        counters["graph_cache_rebuild_count"] = 1.0
        if growing_counter:
            counters[growing_counter] = float(2**window_index)
        counters["supersession_rows_visited"] = (
            counters["supersession_current_candidate_count"]
            + counters["supersession_historical_candidate_count"]
        )
        rows.append(
            {
                "event_index": event_index,
                "modality": "fixture",
                "retention": "natural",
                "process_ms": process_windows[window_index],
                "latency_ms": process_windows[window_index],
                "end_to_end_ms": elapsed_windows[window_index],
                "consolidation_ms": 0.0,
                "consolidation_epoch_id": 0,
                "events_since_epoch_start": event_index + 1,
                "consolidation_after_event": False,
                "consolidation_epoch_counters": {
                    name: 0 for name in audit.REQUIRED_CONSOLIDATION_EPOCH_COUNTERS
                },
                "active_epoch": {
                    "event_count": event_index + 1,
                    "mutation_count": event_index + 1,
                    "allocated_bytes": 16384,
                    "row_batch_high_water": 0,
                    "required": False,
                },
                "durable_barrier_ms": 0.0,
                "work_counters": counters,
                "operation_ms": fixture_operation_ms(counters),
                "behavior": {},
            }
        )
    checkpoints = []
    for event_end in sorted({boundary for item in ranges for boundary in item}):
        effective = (
            min(event_end, stop_store_growth_at)
            if stop_store_growth_at is not None
            else event_end
        )
        checkpoints.append(
            {
                "event_end": event_end,
                "counts": {
                    "memories": 1000 + effective * 5,
                    "signals": 1000 + effective * 5,
                    "associations": 1000 + effective * 5,
                    "embeddings": 1000 + effective * 5,
                    "current_memory_embeddings": 1000 + effective * 5,
                    "memory_reconstructions": 1000 + effective * 5,
                    "rif_recovery_clock": 1,
                    "rif_generation_resets": 0,
                    "rif_active_state": effective,
                },
            }
        )
    profile: dict[str, object] = {
        "schema": "cortext_packet_storage_profile_v2",
        "processed_events": event_count,
        "focus": 0.5,
        "sensitivity": 0.5,
        "stability": 0.5,
        "consolidation_epoch_relational_contract_sha256": (
            audit.CONSOLIDATION_EPOCH_RELATIONAL_CONTRACT_SHA256
        ),
        "retention": "natural",
        "consolidation_runs": 0,
        "consolidation_ms": 0.0,
        "consolidation_events": [],
        "honor_required_consolidation": True,
        "active_epoch_limits": audit.active_epoch_limits(0.5, 0.5, 0.5),
        "store_checkpoints": checkpoints,
    }
    return profile, rows


def epoch_fixture(
    *,
    epoch_count: int = 16,
    epoch_length: int = 120,
    material: bool = True,
    peak_growth: float = 1.0,
    trough_growth: float = 1.0,
    post_drop: bool = True,
    reset_counters: bool = True,
    consolidation_cost_growth: float = 1.0,
    epoch_lengths: list[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lengths = epoch_lengths or [epoch_length] * epoch_count
    event_count = sum(lengths)
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    consolidation_total = 0.0
    event_index = 0
    for epoch_id, current_epoch_length in enumerate(lengths):
        for events_since in range(1, current_epoch_length + 1):
            trough = (1.0 if post_drop else 1.2) * trough_growth**epoch_id
            peak = (1.2 if material else trough) * peak_growth**epoch_id
            process_ms = trough if events_since <= current_epoch_length - 50 else peak
            closes = events_since == current_epoch_length
            duration = consolidation_cost_growth**epoch_id if closes else 0.0
            counters = {
                name: events_since
                for name in audit.REQUIRED_CONSOLIDATION_EPOCH_COUNTERS
            }
            work_counters = {
                name: (0.0 if name.startswith("rollback_") else 10.0)
                for name in audit.REQUIRED_WORK_COUNTERS
            }
            work_counters["graph_candidate_count"] = process_ms * 5.0
            work_counters["graph_exact_comparison_count"] = process_ms * 5.0
            work_counters["graph_rows_visited"] = process_ms * 30.0
            work_counters["supersession_rows_visited"] = (
                work_counters["supersession_current_candidate_count"]
                + work_counters["supersession_historical_candidate_count"]
            )
            operation_ms = fixture_operation_ms(work_counters)
            operation_ms.update(
                {
                    "ConsolidationEpoch.epoch_id": float(epoch_id),
                    "ConsolidationEpoch.events_since_epoch_start": float(events_since),
                    **{
                        f"ConsolidationEpoch.{name}": float(value)
                        for name, value in counters.items()
                    },
                }
            )
            rows.append(
                {
                    "event_index": event_index,
                    "modality": "fixture",
                    "retention": "natural",
                    "process_ms": process_ms,
                    "latency_ms": process_ms,
                    "end_to_end_ms": process_ms + duration,
                    "consolidation_ms": duration,
                    "durable_barrier_ms": 0.0,
                    "consolidation_epoch_id": epoch_id,
                    "events_since_epoch_start": events_since,
                    "consolidation_after_event": closes,
                    "consolidation_epoch_counters": counters,
                    "active_epoch": {
                        "event_count": events_since,
                        "mutation_count": events_since * 3,
                        "allocated_bytes": 16384,
                        "row_batch_high_water": 0,
                        "required": False,
                    },
                    "work_counters": work_counters,
                    "operation_ms": operation_ms,
                    "behavior": {},
                }
            )
            if closes:
                post_counters = {
                    name: (0 if reset_counters else events_since)
                    for name in audit.REQUIRED_CONSOLIDATION_EPOCH_COUNTERS
                }
                event = {
                    "event_index": event_index,
                    "closing_event_index": event_index,
                    "consolidation_epoch_id": epoch_id,
                    "sealed_epoch_event_count": events_since,
                    "sealed_epoch_mutation_count": events_since * 3,
                    "sealed_mutation_identity_verified": True,
                    "sealed_mutation_identity_count": events_since * 3,
                    "duration_ms": duration,
                    "pre_reset_counters": counters,
                    "post_reset_counters": post_counters,
                    "post_reset_active_epoch": {
                        "event_count": 0,
                        "mutation_count": 0,
                        "allocated_bytes": 16384,
                        "row_batch_high_water": 0,
                        "required": False,
                    },
                    "state_after": "none",
                }
                events.append(event)
                consolidation_total += duration
            event_index += 1
    ranges = audit.end_anchored_ranges(event_count, 50)
    checkpoints = []
    for event_end in sorted({boundary for item in ranges for boundary in item}):
        checkpoints.append(
            {
                "event_end": event_end,
                "counts": {
                    "memories": 1000 + event_end * 5,
                    "signals": 1000 + event_end * 5,
                    "associations": 1000 + event_end * 5,
                    "embeddings": 1000 + event_end * 5,
                    "current_memory_embeddings": 1000 + event_end * 5,
                    "memory_reconstructions": 1000 + event_end * 5,
                    "rif_recovery_clock": 1,
                    "rif_generation_resets": 0,
                    "rif_active_state": event_end,
                },
            }
        )
    return {
        "schema": "cortext_packet_storage_profile_v2",
        "processed_events": event_count,
        "focus": 0.5,
        "sensitivity": 0.5,
        "stability": 0.5,
        "consolidation_epoch_relational_contract_sha256": (
            audit.CONSOLIDATION_EPOCH_RELATIONAL_CONTRACT_SHA256
        ),
        "retention": "natural",
        "consolidation_runs": len(events),
        "consolidation_ms": consolidation_total,
        "consolidation_events": events,
        "honor_required_consolidation": True,
        "active_epoch_limits": audit.active_epoch_limits(0.5, 0.5, 0.5),
        "store_checkpoints": checkpoints,
    }, rows


def set_fixture_retention(
    profile: dict[str, object], rows: list[dict[str, object]], retention: str,
) -> None:
    profile["retention"] = retention
    for row in rows:
        row["retention"] = retention
        if retention == "durable":
            row["operation_ms"]["SignalProcessor.sqlite_wal_checkpoint"] = 0.0
            row["operation_ms"][
                "SignalProcessor.sqlite_wal_checkpoint_failure_count"
            ] = 0.0


def fixture_sparse_parameters(
    focus: float = 0.5, sensitivity: float = 0.5, stability: float = 0.5,
) -> dict[str, int]:
    rounded = lambda value: math.floor(value + 0.5)
    route_capacity = rounded(
        256.0 + 256.0 * focus + 128.0 * sensitivity + 128.0 * stability
    )
    backfill_batch_size = rounded(
        64.0 + 64.0 * focus + 32.0 * sensitivity + 32.0 * stability
    )
    return {
        "route_capacity": route_capacity,
        "activation_identity_target": route_capacity * 2 + backfill_batch_size * 2,
        "activation_snapshot_capacity": route_capacity * 2 + backfill_batch_size * 2,
        "total_query_row_budget": (
            route_capacity * 11 + backfill_batch_size * 2
        ),
        "bootstrap_limit": route_capacity * 2,
        "search_node_budget": route_capacity * 9,
        "activation_search_node_budget_min": route_capacity * 8,
        "activation_search_node_budget_step": max(2, backfill_batch_size // 16),
        "search_expansion_batch": max(8, backfill_batch_size // 4),
        "search_effort": route_capacity * 9,
        "activation_search_effort_min": route_capacity * 8,
        "activation_search_effort_step": max(2, backfill_batch_size // 16),
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


def enable_sparse_retrieval_cycles(
    profile: dict[str, object], rows: list[dict[str, object]],
) -> None:
    profile["sparse_route_parameters"] = fixture_sparse_parameters()
    profile["sparse_route_parameters"].update(
        {
            "total_query_row_budget": 3840,
            "search_node_budget": 2560,
            "activation_search_node_budget_min": 1280,
            "activation_search_node_budget_step": 10,
            "search_effort": 2560,
            "activation_search_effort_min": 1024,
            "activation_search_effort_step": 12,
        }
    )
    for row in rows:
        epoch_id = int(row["consolidation_epoch_id"])
        phase = int(row["events_since_epoch_start"])
        operation_ms = row["operation_ms"]
        if phase % 2 == 0:
            if epoch_id == 0:
                effort = 2560.0
                node_budget = 2560.0
                visited = 2400.0
            else:
                progress = (phase - 2) / 116.0
                effort = min(2560.0, 1024.0 + 1536.0 * progress)
                node_budget = min(2560.0, 1280.0 + 1280.0 * progress)
                visited = 1200.0 + 800.0 * progress
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_search_effort"
            ] = effort
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_search_node_budget"
            ] = node_budget
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_node_rows"
            ] = visited
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
            ] = visited
        else:
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_search_effort"
            ] = 0.0
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_search_node_budget"
            ] = 0.0
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_node_rows"
            ] = 0.0
            operation_ms[
                "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
            ] = 0.0


def enable_fixed_sparse_retrieval_envelope(
    profile: dict[str, object], rows: list[dict[str, object]],
) -> None:
    profile["sparse_route_parameters"] = fixture_sparse_parameters()
    profile["experimental_sparse_node_envelope_formula"] = "5C"
    profile["sparse_route_parameters"].update(
        {
            "total_query_row_budget": 3840,
            "search_node_budget": 2560,
            "activation_search_node_budget_min": 2560,
            "activation_search_node_budget_step": 0,
            "search_effort": 2560,
            "activation_search_effort_min": 2560,
            "activation_search_effort_step": 0,
        }
    )
    bound = float(profile["sparse_route_parameters"]["search_node_budget"])
    for row in rows:
        operation_ms = row["operation_ms"]
        operation_ms["GraphRetrieve.sqlite_sparse_route_search_effort"] = bound
        operation_ms[
            "GraphRetrieve.sqlite_sparse_route_search_node_budget"
        ] = bound
        operation_ms["GraphRetrieve.sqlite_sparse_route_node_rows"] = bound
        operation_ms[
            "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
        ] = bound
    for event in profile["consolidation_events"]:
        event["sqlite_sparse_route_recenter_succeeded"] = True
        event["sqlite_sparse_route_activation_search_effort"] = bound
        event["sqlite_sparse_route_activation_node_budget"] = bound
        event["sqlite_sparse_route_recenter_overlap_profiled"] = True
        event["sqlite_sparse_route_recenter_overlap_pair_valid"] = True
        event["sqlite_sparse_route_recenter_overlap_failure_code"] = 0.0
        event["sqlite_sparse_route_recenter_pre_activated_count"] = 1280.0
        event["sqlite_sparse_route_recenter_post_activated_count"] = 1280.0
        event["sqlite_sparse_route_recenter_overlap_count"] = 640.0


class AttributionContractTest(unittest.TestCase):
    def test_active_epoch_limits_follow_low_mid_high_and_axis_knobs(self) -> None:
        self.assertEqual(
            audit.active_epoch_limits(0.0, 0.0, 0.0),
            {"event_count": 256, "mutation_count": 16384,
             "allocated_bytes": 33554432, "row_batch_size": 64},
        )
        self.assertEqual(
            audit.active_epoch_limits(0.5, 0.5, 0.5),
            {"event_count": 512, "mutation_count": 32768,
             "allocated_bytes": 67108864, "row_batch_size": 128},
        )
        self.assertEqual(
            audit.active_epoch_limits(1.0, 1.0, 1.0),
            {"event_count": 768, "mutation_count": 49152,
             "allocated_bytes": 100663296, "row_batch_size": 192},
        )
        points = [
            (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
            (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
            (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
            (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
        ]
        self.assertGreater(
            len({audit.active_epoch_limits(*point)["event_count"]
                 for point in points}),
            3,
        )

    def test_active_epoch_limits_reject_nonfinite_knobs(self) -> None:
        for point in (
            (math.nan, 0.5, 0.5),
            (0.5, math.inf, 0.5),
            (0.5, 0.5, -math.inf),
        ):
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    audit.active_epoch_limits(*point)

    def test_active_epoch_row_batch_high_water_is_independently_bounded(self) -> None:
        profile, rows = plateau_fixture()
        rows[0]["active_epoch"]["row_batch_high_water"] = 129
        with self.assertRaisesRegex(ValueError, "row batch crossed"):
            audit.validate_profile_rows(profile, rows)

    def test_late_operation_is_attributed_with_zero_for_absent_rows(self) -> None:
        result = audit.attribution_result(fixture_rows(), 10)
        contributors = {
            item["operation"] for item in result["ranked_positive_contributors"]
        }
        self.assertIn("cortext::operations::LateMaintenance", contributors)
        self.assertGreaterEqual(result["explained_fraction"], 0.80)
        self.assertIn("named_signed_delta_ms", result)
        self.assertIn("residual_delta_ms", result)
        self.assertNotIn(
            "EmotionalCascade.source_count", result["detailed_subsections"]
        )
        self.assertIn("EmotionalCascade.source_count", result["diagnostic_counters"])
        self.assertNotIn(
            "SignalProcessor.snapshot_cache_entry_copy_count", contributors
        )
        self.assertIn(
            "SignalProcessor.snapshot_cache_entry_copy_count",
            result["diagnostic_counters"],
        )

    def test_attribution_cli_does_not_require_prefix_flatness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            database = root / "profile.sqlite"
            output = root / "audit.json"
            rows = fixture_rows()
            profile.write_text(
                json.dumps(
                    {
                        "working_set_curve": rows,
                        "mean_process_ms": 1.0,
                        "mean_total_ms": 1.0,
                        "wall_ms": 1,
                    }
                ),
                encoding="utf-8",
            )
            connection = sqlite3.connect(database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE memories(memory_id INTEGER PRIMARY KEY);
                    CREATE TABLE associations(
                      source_memory_id INTEGER,
                      target_memory_id INTEGER,
                      edge_type TEXT
                    );
                    CREATE TABLE signals(signal_id INTEGER PRIMARY KEY);
                    """
                )
            finally:
                connection.close()

            command = [
                sys.executable,
                str(Path(__file__).with_name("audit_storage_cost_profile.py")),
                "--profile",
                str(profile),
                "--db",
                str(database),
                "--window-size",
                "10",
            ]
            attribution = subprocess.run(
                [*command, "--out", str(output), "--require-attribution"],
                check=False,
                capture_output=True,
                text=True,
            )
            full_gate = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(attribution.returncode, 0, attribution.stderr)
            self.assertEqual(full_gate.returncode, 1, full_gate.stderr)
            self.assertTrue(json.loads(output.read_text())["attribution_passed"])


class PlateauContractTest(unittest.TestCase):
    def test_sparse_route_parameter_vector_is_bound_to_all_three_knobs(
        self,
    ) -> None:
        for focus in (0.0, 0.5, 1.0):
            for sensitivity in (0.0, 0.5, 1.0):
                for stability in (0.0, 0.5, 1.0):
                    profile, rows = plateau_fixture()
                    profile.update(
                        {
                            "focus": focus,
                            "sensitivity": sensitivity,
                            "stability": stability,
                            "active_epoch_limits": audit.active_epoch_limits(
                                focus, sensitivity, stability
                            ),
                            "sparse_route_parameters": fixture_sparse_parameters(
                                focus, sensitivity, stability
                            ),
                        }
                    )
                    for row in rows:
                        row["operation_ms"][
                            "Cortext.fallback_hydration_signal_limit"
                        ] = float(
                            profile["sparse_route_parameters"][
                                "fallback_hydration_signal_limit"
                            ]
                        )
                        row["operation_ms"][
                            "Cortext.fallback_hydration_signal_rows"
                        ] = 0.0
                    audit.validate_profile_rows(profile, rows)

        midpoint = fixture_sparse_parameters()
        self.assertEqual(midpoint["backfill_batch_size"], 128)
        self.assertEqual(midpoint["fallback_hydration_signal_limit"], 128)
        self.assertEqual(midpoint["family_exact_comparison_limit"], 1024)
        self.assertEqual(midpoint["activation_identity_target"], 1280)
        self.assertEqual(midpoint["search_node_budget"], 4608)
        self.assertEqual(midpoint["activation_search_node_budget_min"], 4096)
        self.assertEqual(midpoint["activation_search_node_budget_step"], 8)

    def test_rejects_one_non_knob_sparse_route_parameter(self) -> None:
        profile, rows = plateau_fixture()
        profile.update(
            {
                "focus": 0.5,
                "sensitivity": 0.5,
                "stability": 0.5,
                "sparse_route_parameters": fixture_sparse_parameters(),
            }
        )
        profile["sparse_route_parameters"]["search_node_budget"] += 1
        with self.assertRaisesRegex(
            ValueError, "search_node_budget does not match F/S/T"
        ):
            audit.validate_profile_rows(profile, rows)

    def test_accepts_only_named_knob_derived_experimental_node_envelopes(
        self,
    ) -> None:
        expected = {
            "C": 512,
            "A": 1280,
            "C+B": 640,
            "2C": 1024,
            "4C": 2048,
            "4C+B/16": 2056,
            "fixed-4C+B/16": 2056,
            "4C+B/16-to-5C": 2560,
            "5C-to-6C-by-B/16": 3072,
            "6C-to-7C-by-B/16": 3584,
            "7C-to-8C-by-B/16": 4096,
            "8C-to-9C-by-B/16": 4608,
            "5C": 2560,
            "6C": 3072,
            "10C": 5120,
            "12C": 6144,
            "16C": 8192,
        }
        for formula, node_budget in expected.items():
            profile, rows = plateau_fixture()
            profile.update(
                {
                    "focus": 0.5,
                    "sensitivity": 0.5,
                    "stability": 0.5,
                    "experimental_sparse_node_envelope_formula": formula,
                    "sparse_route_parameters": fixture_sparse_parameters(),
                }
            )
            profile["sparse_route_parameters"].update(
                {
                    "total_query_row_budget": 3840,
                    "search_node_budget": 2560,
                    "activation_search_node_budget_min": 2560,
                    "activation_search_node_budget_step": 0,
                    "search_effort": 2560,
                    "activation_search_effort_min": 2560,
                    "activation_search_effort_step": 0,
                }
            )
            profile["sparse_route_parameters"]["search_node_budget"] = node_budget
            profile["sparse_route_parameters"][
                "activation_search_node_budget_min"
            ] = node_budget
            if formula == "fixed-4C+B/16":
                profile["sparse_route_parameters"]["search_effort"] = node_budget
                profile["sparse_route_parameters"][
                    "activation_search_effort_min"
                ] = node_budget
                profile["sparse_route_parameters"]["total_query_row_budget"] = (
                    node_budget
                    + profile["sparse_route_parameters"][
                        "activation_identity_target"
                    ]
                )
            if formula == "4C+B/16-to-5C":
                reciprocal_update_count = profile["sparse_route_parameters"][
                    "reciprocal_update_count"
                ]
                minimum = 2048 + reciprocal_update_count
                profile["sparse_route_parameters"][
                    "activation_search_effort_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
            if formula == "5C-to-6C-by-B/16":
                reciprocal_update_count = profile["sparse_route_parameters"][
                    "reciprocal_update_count"
                ]
                minimum = 2560
                profile["sparse_route_parameters"]["search_effort"] = node_budget
                profile["sparse_route_parameters"][
                    "activation_search_effort_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"]["total_query_row_budget"] = (
                    node_budget
                    + profile["sparse_route_parameters"][
                        "activation_identity_target"
                    ]
                )
            if formula == "6C-to-7C-by-B/16":
                reciprocal_update_count = profile["sparse_route_parameters"][
                    "reciprocal_update_count"
                ]
                minimum = 3072
                profile["sparse_route_parameters"]["search_effort"] = node_budget
                profile["sparse_route_parameters"][
                    "activation_search_effort_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"]["total_query_row_budget"] = (
                    node_budget
                    + profile["sparse_route_parameters"][
                        "activation_identity_target"
                    ]
                )
            if formula == "7C-to-8C-by-B/16":
                reciprocal_update_count = profile["sparse_route_parameters"][
                    "reciprocal_update_count"
                ]
                minimum = 3584
                profile["sparse_route_parameters"]["search_effort"] = node_budget
                profile["sparse_route_parameters"][
                    "activation_search_effort_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"]["total_query_row_budget"] = (
                    node_budget
                    + profile["sparse_route_parameters"][
                        "activation_identity_target"
                    ]
                )
            if formula == "8C-to-9C-by-B/16":
                reciprocal_update_count = profile["sparse_route_parameters"][
                    "reciprocal_update_count"
                ]
                minimum = 4096
                profile["sparse_route_parameters"]["search_effort"] = node_budget
                profile["sparse_route_parameters"][
                    "activation_search_effort_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_effort_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_min"
                ] = minimum
                profile["sparse_route_parameters"][
                    "activation_search_node_budget_step"
                ] = reciprocal_update_count
                profile["sparse_route_parameters"]["total_query_row_budget"] = (
                    node_budget
                    + profile["sparse_route_parameters"][
                        "activation_identity_target"
                    ]
                )
            for row in rows:
                row["operation_ms"][
                    "Cortext.fallback_hydration_signal_limit"
                ] = 128.0
                row["operation_ms"][
                    "Cortext.fallback_hydration_signal_rows"
                ] = 0.0
            audit.validate_profile_rows(profile, rows)

        profile, rows = plateau_fixture()
        profile.update(
            {
                "focus": 0.5,
                "sensitivity": 0.5,
                "stability": 0.5,
                "experimental_sparse_node_envelope_formula": "7C",
                "sparse_route_parameters": fixture_sparse_parameters(),
            }
        )
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 0.0
        with self.assertRaisesRegex(
            ValueError, "experimental sparse node envelope formula is invalid"
        ):
            audit.validate_profile_rows(profile, rows)

    def test_fallback_hydration_limit_is_bound_to_knobs_and_every_event(
        self,
    ) -> None:
        profile, rows = plateau_fixture()
        profile.update({"focus": 0.5, "sensitivity": 0.5, "stability": 0.5})
        profile["sparse_route_parameters"] = {
            "fallback_hydration_signal_limit": 128,
        }
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 64.0
        audit.validate_profile_rows(profile, rows)

        rows[-1]["operation_ms"][
            "Cortext.fallback_hydration_signal_limit"
        ] = 129.0
        with self.assertRaisesRegex(ValueError, "event fallback hydration"):
            audit.validate_profile_rows(profile, rows)

    def test_residual_bootstrap_is_exact_for_a_linear_series(self) -> None:
        xs = [float(index) for index in range(2048)]
        ys = [3.0 + 2.0 * value for value in xs]
        self.assertEqual(audit.bootstrap_upper_slope(xs, ys), 2.0)

    def test_residual_bootstrap_is_deterministic_for_long_epoch_series(self) -> None:
        xs = [float(index) for index in range(2048)]
        ys = [10.0 + 0.001 * value + (index % 7) * 0.01
              for index, value in enumerate(xs)]
        first = audit.bootstrap_upper_slope(xs, ys)
        second = audit.bootstrap_upper_slope(xs, ys)
        self.assertEqual(first, second)

    def test_accepts_bounded_consolidation_sawtooth(self) -> None:
        profile, rows = epoch_fixture()
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "sawtooth")
        self.assertGreaterEqual(result["material_epoch_count"], 4)
        self.assertTrue(result["retrieval_cycle_symmetry"]["passed"])
        full_result = audit.suffix_plateau_result(profile, rows, 50)
        self.assertTrue(full_result["passed"], full_result["candidate_failures"])

    def test_accepts_flat_process_with_knob_bounded_sparse_retrieval_cycles(
        self,
    ) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_sparse_retrieval_cycles(profile, rows)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "flat-envelope")
        self.assertEqual(result["material_epoch_count"], 0)
        self.assertTrue(result["retrieval_cycle_symmetry"]["passed"])
        self.assertGreaterEqual(
            result["retrieval_cycle_symmetry"]["material_cycle_count"], 10
        )

    def test_accepts_fixed_knob_bounded_sparse_retrieval_envelope(self) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "flat-envelope")
        self.assertEqual(
            result["retrieval_cycle_symmetry"]["mode"], "fixed-envelope"
        )
        self.assertEqual(
            result["retrieval_cycle_symmetry"]["retrieval_work_bound"], 3840
        )
        self.assertTrue(
            result["retrieval_cycle_symmetry"][
                "fixed_envelope_classifier_passed"
            ]
        )
        self.assertTrue(
            result["retrieval_cycle_symmetry"][
                "fixed_work_envelope_subcontract_passed"
            ]
        )
        self.assertTrue(
            result["retrieval_cycle_symmetry"][
                "activated_identity_overlap_recorded"
            ]
        )
        self.assertTrue(
            result["retrieval_cycle_symmetry"]["full_cycle_passed"]
        )

    def test_fixed_sparse_retrieval_requires_activation_overlap(self) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        for event in profile["consolidation_events"]:
            del event["sqlite_sparse_route_recenter_overlap_profiled"]
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["retrieval_cycle_symmetry"]["full_cycle_passed"]
        )
        self.assertEqual(
            result["retrieval_cycle_symmetry"][
                "overlap_profile_missing_count"
            ],
            len(profile["consolidation_events"]),
        )

    def test_fixed_sparse_retrieval_envelope_is_independent_of_process_mode(
        self,
    ) -> None:
        profile, rows = epoch_fixture(material=True)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "sawtooth")
        self.assertEqual(
            result["retrieval_cycle_symmetry"]["mode"], "fixed-envelope"
        )
        self.assertEqual(
            result["retrieval_cycle_symmetry"]["retrieval_work_bound"], 3840
        )

    def test_fixed_sparse_retrieval_requires_mature_recenter_evidence(
        self,
    ) -> None:
        profile, rows = epoch_fixture(material=True)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        for event in profile["consolidation_events"]:
            event["sqlite_sparse_route_recenter_succeeded"] = False
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "fewer than ten mature fixed-envelope recenters",
            result["failures"],
        )

    def test_fixed_sparse_retrieval_recenter_keeps_knob_envelope(self) -> None:
        profile, rows = epoch_fixture(material=True)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        profile["consolidation_events"][-2][
            "sqlite_sparse_route_activation_node_budget"
        ] -= 1.0
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "fixed retrieval recenter changes F/S/T envelope",
            result["failures"],
        )

    def test_fixed_envelope_counts_persisted_and_delta_distance_work(self) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        bound = float(profile["sparse_route_parameters"]["search_node_budget"])
        for row in rows:
            operation_ms = row["operation_ms"]
            if operation_ms[
                "GraphRetrieve.sqlite_sparse_route_search_node_budget"
            ] == 0.0:
                continue
            operation_ms["GraphRetrieve.sqlite_sparse_route_node_rows"] = (
                bound - 17.0
            )
            operation_ms["GraphRetrieve.sqlite_sparse_route_dirty_rows"] = 17.0
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(
            result["retrieval_cycle_symmetry"]["work_mismatch_count"], 0
        )

    def test_accepts_fixed_sparse_retrieval_below_ceiling(self) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        rows[1000]["operation_ms"][
            "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
        ] -= 1.0
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])

    def test_route_active_zero_or_missing_work_fails_closed(self) -> None:
        for remove_metric in (False, True):
            with self.subTest(remove_metric=remove_metric):
                profile, rows = epoch_fixture(material=False)
                enable_fixed_sparse_retrieval_envelope(profile, rows)
                operation_ms = rows[1000]["operation_ms"]
                operation_ms["GraphRetrieve.seed_sparse_route_active"] = 1.0
                if remove_metric:
                    operation_ms.pop(
                        "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
                    )
                else:
                    operation_ms[
                        "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
                    ] = 0.0
                result = audit.consolidation_epoch_result(
                    profile, rows, 0, len(rows)
                )
                self.assertFalse(result["passed"])
                self.assertIn(
                    "fixed retrieval work is absent on an active route",
                    result["failures"],
                )

    def test_fixed_sparse_retrieval_metrics_are_independently_bounded(
        self,
    ) -> None:
        cases = (
            (
                "GraphRetrieve.sqlite_sparse_route_search_effort",
                "fixed queue effort differs from F/S/T bound",
            ),
            (
                "GraphRetrieve.sqlite_sparse_route_search_node_budget",
                "fixed dynamic node ceiling differs from F/S/T bound",
            ),
            (
                "GraphRetrieve.sqlite_sparse_route_node_rows",
                "fixed retrieval visits exceed F/S/T bound",
            ),
            (
                "GraphRetrieve.sqlite_sparse_route_activated_identities",
                "fixed retrieval activation exceeds F/S/T target",
            ),
        )
        for metric, expected_failure in cases:
            with self.subTest(metric=metric):
                profile, rows = epoch_fixture(material=False)
                enable_fixed_sparse_retrieval_envelope(profile, rows)
                bound = float(
                    profile["sparse_route_parameters"]["search_node_budget"]
                )
                value = (
                    float(
                        profile["sparse_route_parameters"][
                            "activation_identity_target"
                        ]
                    )
                    if metric.endswith("activated_identities")
                    else bound
                )
                rows[1000]["operation_ms"][metric] = value + 1.0
                result = audit.consolidation_epoch_result(
                    profile, rows, 0, len(rows)
                )
                self.assertFalse(result["passed"])
                self.assertIn(expected_failure, result["failures"])

    def test_rejects_fixed_sparse_retrieval_over_total_row_bound(self) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_fixed_sparse_retrieval_envelope(profile, rows)
        rows[1000]["operation_ms"][
            "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
        ] = float(
            profile["sparse_route_parameters"]["total_query_row_budget"] + 1
        )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "fixed retrieval work exceeds total F/S/T row bound",
            result["failures"],
        )

    def test_rejects_sparse_retrieval_that_exceeds_its_current_node_ceiling(
        self,
    ) -> None:
        profile, rows = epoch_fixture(material=False)
        enable_sparse_retrieval_cycles(profile, rows)
        row = next(
            row
            for row in rows
            if row["operation_ms"].get(
                "GraphRetrieve.sqlite_sparse_route_search_node_budget", 0.0
            )
            > 0.0
        )
        row["operation_ms"][
            "GraphRetrieve.sqlite_sparse_route_node_rows"
        ] = (
            row["operation_ms"][
                "GraphRetrieve.sqlite_sparse_route_search_node_budget"
            ]
            + 1.0
        )
        row["operation_ms"][
            "GraphRetrieve.sqlite_sparse_route_distance_evaluations"
        ] = row["operation_ms"][
            "GraphRetrieve.sqlite_sparse_route_node_rows"
        ]
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "retrieval work exceeds current dynamic node ceiling",
            result["failures"],
        )
        self.assertEqual(
            result["retrieval_cycle_symmetry"][
                "dynamic_bound_exceeded_count"
            ],
            1,
        )

    def test_accumulator_state_is_observed_but_not_a_consolidation_reset(
        self,
    ) -> None:
        profile, rows = epoch_fixture()
        for event in profile["consolidation_events"]:
            event["post_reset_counters"]["accumulator_signal_count"] = event[
                "pre_reset_counters"
            ]["accumulator_signal_count"]
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertTrue(
            all(
                epoch["reset_counter_ratios"]["accumulator_signal_count"]
                == 1.0
                for epoch in result["complete_epochs"]
            )
        )

    def test_accepts_durable_flat_envelope_without_consolidation(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "flat-envelope")

    def test_accepts_durable_flat_envelope_with_complete_nonmaterial_epochs(self) -> None:
        profile, rows = epoch_fixture(material=False)
        set_fixture_retention(profile, rows, "durable")
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "flat-envelope")
        self.assertGreaterEqual(result["complete_epoch_count"], 4)
        self.assertEqual(result["material_epoch_count"], 0)

    def test_rejects_natural_flat_envelope_without_retrieval_cycles(self) -> None:
        profile, rows = epoch_fixture(material=False)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "natural cutover requires retrieval cycle symmetry",
            result["failures"],
        )

    def test_rejects_successful_consolidations_without_complete_epochs(self) -> None:
        profile, rows = epoch_fixture(epoch_length=60, material=False)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "successful consolidations lack four complete epochs",
            result["failures"],
        )

    def test_rejects_one_to_three_material_resets(self) -> None:
        profile, rows = epoch_fixture(epoch_count=4)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "one to three material epochs are insufficient",
            result["failures"],
        )

    def test_rejects_rising_sawtooth_peaks(self) -> None:
        profile, rows = epoch_fixture(peak_growth=1.2)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("sawtooth peaks rise", result["failures"])

    def test_rejects_rising_sawtooth_troughs(self) -> None:
        profile, rows = epoch_fixture(peak_growth=1.2, trough_growth=1.2)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("sawtooth troughs rise", result["failures"])

    def test_rejects_material_reset_without_process_drop(self) -> None:
        profile, rows = epoch_fixture(peak_growth=1.2, trough_growth=1.2)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "material epoch does not lower following process time",
            result["failures"],
        )

    def test_rejects_material_cycles_without_retrieval_work_reset(self) -> None:
        profile, rows = epoch_fixture()
        for row in rows:
            row["work_counters"]["graph_candidate_count"] = 10.0
            row["work_counters"]["graph_exact_comparison_count"] = 10.0
            row["operation_ms"] = fixture_operation_ms(row["work_counters"])
            row["operation_ms"].update(
                {
                    "ConsolidationEpoch.epoch_id": float(
                        row["consolidation_epoch_id"]
                    ),
                    "ConsolidationEpoch.events_since_epoch_start": float(
                        row["events_since_epoch_start"]
                    ),
                    **{
                        f"ConsolidationEpoch.{name}": float(value)
                        for name, value in row[
                            "consolidation_epoch_counters"
                        ].items()
                    },
                }
            )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("too few material retrieval resets", result["failures"])

    def test_rejects_one_dissimilar_retrieval_cycle(self) -> None:
        profile, rows = epoch_fixture()
        target_epoch = 10
        target = [
            row for row in rows
            if row["consolidation_epoch_id"] == target_epoch
        ]
        for index, row in enumerate(target):
            value = 100.0 if index % 2 == 0 else 0.0
            row["work_counters"]["graph_candidate_count"] = value
            row["work_counters"]["graph_exact_comparison_count"] = value
            row["operation_ms"] = fixture_operation_ms(row["work_counters"])
            row["operation_ms"].update(
                {
                    "ConsolidationEpoch.epoch_id": float(target_epoch),
                    "ConsolidationEpoch.events_since_epoch_start": float(
                        row["events_since_epoch_start"]
                    ),
                    **{
                        f"ConsolidationEpoch.{name}": float(value)
                        for name, value in row[
                            "consolidation_epoch_counters"
                        ].items()
                    },
                }
            )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "retrieval cycle shape p95 exceeds limit", result["failures"]
        )

    def test_rejects_late_retrieval_cycle_template_drift(self) -> None:
        profile, rows = epoch_fixture()
        baseline = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        late_epoch_ids = {
            epoch["consolidation_epoch_id"]
            for epoch in baseline["complete_epochs"][-5:]
        }
        for epoch_id in late_epoch_ids:
            target = [
                row for row in rows
                if row["consolidation_epoch_id"] == epoch_id
            ]
            for index, row in enumerate(target):
                progress = index / (len(target) - 1)
                value = (10.0 + 90.0 * progress * progress) / 2.0
                row["work_counters"]["graph_candidate_count"] = value
                row["work_counters"]["graph_exact_comparison_count"] = value
                row["operation_ms"] = fixture_operation_ms(row["work_counters"])
                row["operation_ms"].update(
                    {
                        "ConsolidationEpoch.epoch_id": float(epoch_id),
                        "ConsolidationEpoch.events_since_epoch_start": float(
                            row["events_since_epoch_start"]
                        ),
                        **{
                            f"ConsolidationEpoch.{name}": float(counter)
                            for name, counter in row[
                                "consolidation_epoch_counters"
                            ].items()
                        },
                    }
                )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "late retrieval cycle template drifts", result["failures"]
        )

    def test_rejects_inverse_retrieval_sawtooth(self) -> None:
        profile, rows = epoch_fixture()
        for row in rows:
            progress = (row["events_since_epoch_start"] - 1) / 119.0
            value = (100.0 - 90.0 * progress) / 2.0
            row["work_counters"]["graph_candidate_count"] = value
            row["work_counters"]["graph_exact_comparison_count"] = value
            row["operation_ms"] = fixture_operation_ms(row["work_counters"])
            row["operation_ms"].update(
                {
                    "ConsolidationEpoch.epoch_id": float(
                        row["consolidation_epoch_id"]
                    ),
                    "ConsolidationEpoch.events_since_epoch_start": float(
                        row["events_since_epoch_start"]
                    ),
                    **{
                        f"ConsolidationEpoch.{name}": float(counter)
                        for name, counter in row[
                            "consolidation_epoch_counters"
                        ].items()
                    },
                }
            )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "retrieval work falls before consolidation", result["failures"]
        )

    def test_rejects_flat_retrieval_cycles_with_geometric_epoch_decay(self) -> None:
        profile, rows = epoch_fixture()
        for row in rows:
            epoch_id = row["consolidation_epoch_id"]
            value = 100.0 * 0.9**epoch_id / 2.0
            row["work_counters"]["graph_candidate_count"] = value
            row["work_counters"]["graph_exact_comparison_count"] = value
            row["operation_ms"] = fixture_operation_ms(row["work_counters"])
            row["operation_ms"].update(
                {
                    "ConsolidationEpoch.epoch_id": float(epoch_id),
                    "ConsolidationEpoch.events_since_epoch_start": float(
                        row["events_since_epoch_start"]
                    ),
                    **{
                        f"ConsolidationEpoch.{name}": float(counter)
                        for name, counter in row[
                            "consolidation_epoch_counters"
                        ].items()
                    },
                }
            )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "retrieval cycle lacks material ramp", result["failures"]
        )

    def test_rejects_repeatable_spiky_retrieval_cycles(self) -> None:
        profile, rows = epoch_fixture()
        for row in rows:
            events_since = row["events_since_epoch_start"]
            if events_since <= 50:
                retrieval_work = 10.0
            elif events_since <= 70:
                retrieval_work = 1000.0 if events_since % 2 == 0 else 0.0
            else:
                retrieval_work = 200.0 if events_since % 2 == 0 else 0.0
            value = retrieval_work / 2.0
            row["work_counters"]["graph_candidate_count"] = value
            row["work_counters"]["graph_exact_comparison_count"] = value
            row["operation_ms"] = fixture_operation_ms(row["work_counters"])
            row["operation_ms"].update(
                {
                    "ConsolidationEpoch.epoch_id": float(
                        row["consolidation_epoch_id"]
                    ),
                    "ConsolidationEpoch.events_since_epoch_start": float(
                        events_since
                    ),
                    **{
                        f"ConsolidationEpoch.{name}": float(counter)
                        for name, counter in row[
                            "consolidation_epoch_counters"
                        ].items()
                    },
                }
            )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn(
            "retrieval cycle contains excessive negative variation",
            result["failures"],
        )
        self.assertIn("retrieval work exceeds F/S/T bound", result["failures"])

    def test_rejects_work_bound_violation_in_final_partial_cycle(self) -> None:
        profile, rows = epoch_fixture()
        row = rows[-1]
        row["work_counters"]["graph_candidate_count"] = 100.0
        row["work_counters"]["graph_exact_comparison_count"] = 100.0
        row["operation_ms"] = fixture_operation_ms(row["work_counters"])
        row["operation_ms"].update(
            {
                "ConsolidationEpoch.epoch_id": float(
                    row["consolidation_epoch_id"]
                ),
                "ConsolidationEpoch.events_since_epoch_start": float(
                    row["events_since_epoch_start"]
                ),
                **{
                    f"ConsolidationEpoch.{name}": float(counter)
                    for name, counter in row[
                        "consolidation_epoch_counters"
                    ].items()
                },
            }
        )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("retrieval work exceeds F/S/T bound", result["failures"])
        self.assertEqual(
            result["retrieval_cycle_symmetry"][
                "accepted_suffix_maximum_retrieval_work"
            ],
            256.0,
        )

    def test_rejects_consolidation_reset_counter_that_does_not_fall(self) -> None:
        profile, rows = epoch_fixture(reset_counters=False)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("consolidation reset counter does not fall", result["failures"])

    def test_zero_reset_counter_must_remain_zero(self) -> None:
        profile, rows = epoch_fixture()
        counter = "working_memory_pending_signal_count"
        for event in profile["consolidation_events"]:
            close = event["event_index"]
            rows[close]["consolidation_epoch_counters"][counter] = 0
            rows[close]["operation_ms"][f"ConsolidationEpoch.{counter}"] = 0.0
            event["pre_reset_counters"][counter] = 0
            event["post_reset_counters"][counter] = 0
        accepted = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(accepted["passed"], accepted["failures"])
        self.assertTrue(
            all(
                epoch["reset_counter_ratios"][counter] == 1.0
                for epoch in accepted["complete_epochs"]
            )
        )

        profile["consolidation_events"][-2]["post_reset_counters"][counter] = 1
        rejected = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(rejected["passed"])
        self.assertIn("consolidation reset counter does not fall", rejected["failures"])
        self.assertEqual(
            rejected["complete_epochs"][-1]["reset_counter_ratios"][counter],
            float("inf"),
        )

        suffix = audit.suffix_plateau_result(profile, rows, 50)
        self.assertFalse(suffix["passed"])

    def test_nonmaterial_zero_reset_counter_must_remain_zero(self) -> None:
        profile, rows = epoch_fixture(material=False)
        set_fixture_retention(profile, rows, "durable")
        counter = "working_memory_pending_signal_count"
        for event in profile["consolidation_events"]:
            close = event["event_index"]
            rows[close]["consolidation_epoch_counters"][counter] = 0
            rows[close]["operation_ms"][f"ConsolidationEpoch.{counter}"] = 0.0
            event["pre_reset_counters"][counter] = 0
            event["post_reset_counters"][counter] = 0
        accepted = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(accepted["passed"], accepted["failures"])
        self.assertEqual(accepted["mode"], "flat-envelope")

        profile["consolidation_events"][-2]["post_reset_counters"][counter] = 1
        rejected = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(rejected["passed"])
        self.assertIn("consolidation reset counter does not fall", rejected["failures"])
        self.assertEqual(
            rejected["complete_epochs"][-1]["reset_counter_ratios"][counter],
            float("inf"),
        )
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 50)["passed"])

    def test_rejects_growing_normalized_consolidation_cost(self) -> None:
        profile, rows = epoch_fixture(consolidation_cost_growth=1.3)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("consolidation cost per sealed event rises", result["failures"])
        self.assertIn(
            "consolidation cost per sealed mutation rises", result["failures"]
        )

    def test_rejects_increasing_consolidation_frequency(self) -> None:
        profile, rows = epoch_fixture(
            material=False,
            epoch_lengths=[160, 150, 140, 130, 120, 110, 100],
        )
        total = 0.0
        for event in profile["consolidation_events"]:
            row = rows[event["event_index"]]
            duration = event["sealed_epoch_event_count"] / 120.0
            row["end_to_end_ms"] += duration - row["consolidation_ms"]
            row["consolidation_ms"] = duration
            event["duration_ms"] = duration
            total += duration
        profile["consolidation_ms"] = total
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("consolidation frequency rises", result["failures"])

    def test_rejects_frequency_hidden_in_short_epochs(self) -> None:
        profile, rows = epoch_fixture(
            material=False,
            epoch_lengths=[120] * 5 + [20] * 20,
        )
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("consolidation frequency rises", result["failures"])

    def test_rejects_corrupt_consolidation_epoch_relations(self) -> None:
        profile, rows = epoch_fixture()

        corruptions = []

        duplicate_event_index = copy.deepcopy(rows)
        duplicate_event_index[1]["event_index"] = 0
        corruptions.append(("unique and contiguous", profile, duplicate_event_index))

        skipped_event_index = copy.deepcopy(rows)
        skipped_event_index[1]["event_index"] = 2
        corruptions.append(("unique and contiguous", profile, skipped_event_index))

        nonmonotonic_epoch = copy.deepcopy(rows)
        nonmonotonic_epoch[1]["consolidation_epoch_id"] = 2
        corruptions.append(("epoch sequence", profile, nonmonotonic_epoch))

        bad_events_since = copy.deepcopy(rows)
        bad_events_since[1]["events_since_epoch_start"] = 4
        corruptions.append(("epoch sequence", profile, bad_events_since))

        orphan_profile = copy.deepcopy(profile)
        del orphan_profile["consolidation_events"][0]
        corruptions.append(("event indices do not match", orphan_profile, rows))

        duplicate_close_profile = copy.deepcopy(profile)
        duplicate_close_profile["consolidation_events"].append(
            copy.deepcopy(duplicate_close_profile["consolidation_events"][0])
        )
        corruptions.append(
            ("event indices do not match", duplicate_close_profile, rows)
        )

        bad_sealed_event_profile = copy.deepcopy(profile)
        bad_sealed_event_profile["consolidation_events"][0][
            "sealed_epoch_event_count"
        ] -= 1
        corruptions.append(("sealed epoch event count", bad_sealed_event_profile, rows))

        bad_sealed_mutation_profile = copy.deepcopy(profile)
        bad_sealed_mutation_profile["consolidation_events"][0][
            "sealed_mutation_identity_count"
        ] -= 1
        corruptions.append(
            ("mutation identity count", bad_sealed_mutation_profile, rows)
        )

        nonadjacent_profile = copy.deepcopy(profile)
        nonadjacent_profile["consolidation_events"][0]["pre_reset_counters"][
            "accumulator_signal_count"
        ] -= 1
        corruptions.append(("not adjacent", nonadjacent_profile, rows))

        reset_without_close = copy.deepcopy(rows)
        reset_without_close[1]["consolidation_epoch_id"] = 1
        reset_without_close[1]["events_since_epoch_start"] = 1
        corruptions.append(("epoch sequence", profile, reset_without_close))

        for expected, corrupt_profile, corrupt_rows in corruptions:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    audit.validate_profile_rows(corrupt_profile, corrupt_rows)

    def test_zero_duration_consolidation_keeps_its_close_identity(self) -> None:
        profile, rows = epoch_fixture()
        event = profile["consolidation_events"][0]
        closing_row = rows[event["event_index"]]
        profile["consolidation_ms"] -= event["duration_ms"]
        closing_row["end_to_end_ms"] -= closing_row["consolidation_ms"]
        closing_row["consolidation_ms"] = 0.0
        event["duration_ms"] = 0.0
        audit.validate_profile_rows(profile, rows)

    def test_rejects_end_to_end_time_that_excludes_consolidation(self) -> None:
        profile, rows = epoch_fixture()
        closing_row = rows[profile["consolidation_events"][0]["event_index"]]
        closing_row["end_to_end_ms"] = max(
            closing_row["process_ms"], closing_row["consolidation_ms"]
        )
        with self.assertRaisesRegex(ValueError, "excludes required event work"):
            audit.validate_profile_rows(profile, rows)

    def test_rejects_processed_event_count_mismatch(self) -> None:
        profile, rows = plateau_fixture()
        profile["processed_events"] += 100
        with self.assertRaisesRegex(ValueError, "does not match row count"):
            audit.suffix_plateau_result(profile, rows, 10)

    def test_accepts_end_anchored_ramp_then_plateau(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        result = audit.suffix_plateau_result(profile, rows, 10)
        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_suffix"]["plateau_end_event"], 126)
        self.assertGreaterEqual(result["accepted_suffix"]["plateau_window_count"], 6)
        self.assertEqual(
            result["accepted_suffix"]["ramp_duration_events"],
            result["accepted_suffix"]["plateau_start_event"],
        )
        self.assertEqual(len(result["height_reference"]["source_profile_sha256"]), 64)

    def test_rejects_missing_digest_bound_height_reference(self) -> None:
        profile, rows = plateau_fixture()
        reference = audit.PLATEAU_HEIGHT_REFERENCES.pop("natural")
        try:
            with self.assertRaisesRegex(ValueError, "digest-bound"):
                audit.suffix_plateau_result(profile, rows, 10)
        finally:
            audit.PLATEAU_HEIGHT_REFERENCES["natural"] = reference

    def test_rejects_never_ending_ramp(self) -> None:
        profile, rows = plateau_fixture(
            process_windows=[2.0 + index for index in range(12)]
        )
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])

    def test_rejects_acceleration_only_in_final_remainder(self) -> None:
        for retention in ("natural", "durable"):
            with self.subTest(retention=retention):
                profile, rows = plateau_fixture()
                profile["retention"] = retention
                for row in rows:
                    row["retention"] = retention
                    if retention == "durable":
                        row["durable_barrier_ms"] = 0.25
                        row["operation_ms"][
                            "SignalProcessor.sqlite_wal_checkpoint"
                        ] = 0.25
                        row["operation_ms"][
                            "SignalProcessor.sqlite_wal_checkpoint_failure_count"
                        ] = 0.0
                for row in rows[-6:]:
                    row["process_ms"] = 30.0
                    row["end_to_end_ms"] = 30.0
                result = audit.suffix_plateau_result(profile, rows, 10)
                self.assertFalse(result["passed"])
                self.assertTrue(result["final_event_covered"])

    def test_durable_barrier_requires_attribution_key_not_positive_clock_tick(
        self,
    ) -> None:
        profile, rows = plateau_fixture()
        profile["retention"] = "durable"
        for row in rows:
            row["retention"] = "durable"
            row["durable_barrier_ms"] = 0.0
            row["operation_ms"]["SignalProcessor.sqlite_wal_checkpoint"] = 0.0
            row["operation_ms"][
                "SignalProcessor.sqlite_wal_checkpoint_failure_count"
            ] = 0.0
        audit.validate_profile_rows(profile, rows)
        del rows[0]["operation_ms"]["SignalProcessor.sqlite_wal_checkpoint"]
        with self.assertRaisesRegex(ValueError, "barrier attribution"):
            audit.validate_profile_rows(profile, rows)

    def test_rejects_failed_durable_barrier(self) -> None:
        profile, rows = plateau_fixture()
        profile["retention"] = "durable"
        for row in rows:
            row["retention"] = "durable"
            row["durable_barrier_ms"] = 0.0
            row["operation_ms"]["SignalProcessor.sqlite_wal_checkpoint"] = 0.0
            row["operation_ms"][
                "SignalProcessor.sqlite_wal_checkpoint_failure_count"
            ] = 0.0
        rows[-1]["operation_ms"][
            "SignalProcessor.sqlite_wal_checkpoint_failure_count"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "failed post-commit barrier"):
            audit.validate_profile_rows(profile, rows)

    def test_allows_finite_warmup_supersession_fallback(self) -> None:
        profile, rows = plateau_fixture()
        rows[0]["operation_ms"][
            "MemoryStorage.supersession_sql_fallback_count"
        ] = 1.0
        audit.validate_profile_rows(profile, rows)

    def test_rejects_post_warmup_supersession_fallback(self) -> None:
        profile, rows = plateau_fixture()
        first_post_warmup = math.ceil(0.20 * len(rows))
        rows[first_post_warmup]["operation_ms"][
            "MemoryStorage.supersession_sql_fallback_count"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "post-warmup.*fallback"):
            audit.validate_profile_rows(profile, rows)

    def test_rejects_flat_plateau_above_height_ceiling(self) -> None:
        profile, rows = plateau_fixture(
            process_windows=[2.0, 4.0, 6.0, 8.0] + [12.0] * 8
        )
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])

    def test_rejects_flat_work_when_store_stops_growing(self) -> None:
        profile, rows = plateau_fixture(stop_store_growth_at=66)
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])

    def test_rejects_missing_and_vacuous_work_counters(self) -> None:
        profile, rows = plateau_fixture()
        missing = copy.deepcopy(rows)
        del missing[0]["work_counters"]["graph_rows_visited"]
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            audit.suffix_plateau_result(profile, missing, 10)
        vacuous = copy.deepcopy(rows)
        for row in vacuous:
            row["work_counters"]["graph_exact_comparison_count"] = 0.0
            row["operation_ms"]["GraphRetrieve.family_exact_comparison_count"] = 0.0
        with self.assertRaisesRegex(ValueError, "all zero"):
            audit.suffix_plateau_result(profile, vacuous, 10)

    def test_accepts_only_known_rejected_candidate_counter_extensions(self) -> None:
        profile, rows = plateau_fixture()
        for row in rows:
            for name in audit.LEGACY_REJECTED_EMOTIONAL_WORK_COUNTERS:
                row["work_counters"][name] = 0.0
        audit.validate_profile_rows(profile, rows)
        self.assertFalse(
            audit.is_duration_operation("EmotionalCascade.edge_visit_count")
        )
        self.assertFalse(
            audit.is_duration_operation("EmotionalCascade.edge_visit_limit")
        )
        self.assertFalse(
            audit.is_duration_operation(
                "MemoryStorage.supersession_current_candidate_execution_count"
            )
        )
        self.assertFalse(
            audit.is_duration_operation(
                "GraphRetrieve.sqlite_sparse_route_search_effort"
            )
        )
        for name in (
            "GraphRetrieve.sqlite_sparse_route_activation_snapshot_cache_miss_rows",
            "GraphRetrieve.sqlite_sparse_route_restart_rows",
            "GraphRetrieve.sqlite_sparse_route_dirty_rows",
        ):
            self.assertFalse(audit.is_duration_operation(name), name)

        rows[0]["work_counters"]["unknown_candidate_counter"] = 0.0
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            audit.validate_profile_rows(profile, rows)

    def test_write_gate_profile_values_are_diagnostics_not_durations(self) -> None:
        for name in (
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
        ):
            self.assertFalse(audit.is_duration_operation(name), name)

    def test_zero_suppression_population_is_not_vacuous_recovery_work(self) -> None:
        profile, rows = plateau_fixture()
        for row in rows:
            row["work_counters"]["retrieval_suppression_id_count"] = 0.0
            row["operation_ms"]["SignalProcessor.retrieval_suppression_id_count"] = 0.0
            row["operation_ms"]["SignalProcessor.retrieval_suppression_id_activity"] = 0.0
            self.assertGreater(
                row["operation_ms"]["Competition.rif_recovery_active_sql"], 0.0
            )
        audit.validate_profile_rows(profile, rows)

    def test_rejects_emotional_neighbor_producer_mismatch(self) -> None:
        profile, rows = plateau_fixture()
        for row in rows:
            row["work_counters"]["emotional_neighbor_count"] = 0.0
            row["operation_ms"]["EmotionalCascade.neighbor_count"] = 1.0
        with self.assertRaisesRegex(ValueError, "does not match producer"):
            audit.suffix_plateau_result(profile, rows, 10)

    def test_all_work_counters_reject_correlated_zero_placeholders(self) -> None:
        profile, original_rows = plateau_fixture()
        for target in audit.REQUIRED_WORK_COUNTERS:
            with self.subTest(counter=target):
                rows = copy.deepcopy(original_rows)
                for row in rows:
                    counters = row["work_counters"]
                    operations = row["operation_ms"]
                    zeroed = {target}
                    if target == "supersession_rows_visited":
                        zeroed.update(
                            {
                                "supersession_current_candidate_count",
                                "supersession_historical_candidate_count",
                            }
                        )
                    for counter in zeroed:
                        counters[counter] = 0.0
                        for source in audit.COUNTER_SOURCE_OPERATIONS[counter]:
                            operations[source] = 0.0
                        for marker in audit.COUNTER_ACTIVITY_OPERATIONS[counter]:
                            operations[marker] = 0.1
                with self.assertRaisesRegex(ValueError, "all zero"):
                    audit.suffix_plateau_result(profile, rows, 10)

    def test_diagnostic_row_counts_are_not_timing_operations(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        ranges = audit.end_anchored_ranges(len(rows), 10)
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 0.0
            window_index = next(
                (
                    index
                    for index, (start, end) in enumerate(ranges)
                    if start <= row["event_index"] < end
                ),
                0,
            )
            row["operation_ms"]["GraphRetrieve.seed_cache_distance_rows"] = (
                1.08**window_index
            )
            row["operation_ms"]["GraphRetrieve.seed_cache_rows"] = (
                1.09**window_index
            )
            row["operation_ms"][
                "MemoryStorage.supersession_population_mismatch_count"
            ] = 1.10**window_index
            row["operation_ms"][
                "SignalProcessor.commit_table_row_count.embeddings_chunks"
            ] = 1.11**window_index
        result = audit.suffix_plateau_result(profile, rows, 10)
        self.assertTrue(result["passed"], result.get("candidate_failures"))
        self.assertNotIn(
            "GraphRetrieve.seed_cache_distance_rows",
            result["accepted_suffix"]["operation_results"],
        )
        self.assertNotIn(
            "GraphRetrieve.seed_cache_rows",
            result["accepted_suffix"]["operation_results"],
        )
        self.assertNotIn(
            "MemoryStorage.supersession_population_mismatch_count",
            result["accepted_suffix"]["operation_results"],
        )
        self.assertNotIn(
            "SignalProcessor.commit_table_row_count.embeddings_chunks",
            result["accepted_suffix"]["operation_results"],
        )

    def test_rejects_growing_signal_row_insert_duration(self) -> None:
        profile, rows = plateau_fixture()
        ranges = audit.end_anchored_ranges(len(rows), 10)
        for row in rows:
            window_index = next(
                (
                    index
                    for index, (start, end) in enumerate(ranges)
                    if start <= row["event_index"] < end
                ),
                0,
            )
            row["operation_ms"]["MemoryStorage.insert_signal_rows"] = (
                0.1 * (1.08**window_index)
            )
        result = audit.suffix_plateau_result(profile, rows, 10)
        self.assertFalse(result["passed"])
        self.assertTrue(audit.is_duration_operation("MemoryStorage.insert_signal_rows"))

    def test_rejects_one_growing_active_counter(self) -> None:
        profile, rows = plateau_fixture(growing_counter="graph_rows_visited")
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])

    def test_emotional_work_bounds_are_knob_derived(self) -> None:
        profile = {"sparse_route_parameters": fixture_sparse_parameters()}
        self.assertEqual(
            audit.knob_work_counter_bound(profile, "graph_rows_visited"),
            1280,
        )
        self.assertEqual(
            audit.knob_work_counter_bound(
                profile, "supersession_rows_visited"
            ),
            5888,
        )
        self.assertEqual(
            audit.knob_work_counter_bound(profile, "emotional_source_count"),
            128,
        )
        self.assertEqual(
            audit.knob_work_counter_bound(profile, "emotional_neighbor_count"),
            2560,
        )
        self.assertEqual(
            audit.knob_work_counter_bound(profile, "emotional_update_count"),
            1280,
        )
        self.assertFalse(
            audit.is_duration_operation(
                "EmotionalCascade.edge_visit_limit_reached"
            )
        )
        self.assertFalse(
            audit.is_duration_operation(
                "GraphRetrieve.sqlite_sparse_route_node_rows"
            )
        )
        self.assertFalse(
            audit.is_duration_operation("Cortext.fallback_hydration_signal_rows")
        )
        self.assertEqual(
            audit.COUNTER_SOURCE_OPERATIONS["graph_rows_visited"],
            ("GraphRetrieve.rows_visited",),
        )

    def test_accepts_variable_exact_comparisons_under_knob_bound(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        profile.update(
            {
                "focus": 0.5,
                "sensitivity": 0.5,
                "stability": 0.5,
                "sparse_route_parameters": fixture_sparse_parameters(),
            }
        )
        ranges = audit.end_anchored_ranges(len(rows), 10)
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 0.0
            window_index = next(
                (
                    index
                    for index, (start, end) in enumerate(ranges)
                    if start <= row["event_index"] < end
                ),
                0,
            )
            value = min(1024.0, 10.0 * 1.2**window_index)
            row["work_counters"]["graph_exact_comparison_count"] = value
            row["operation_ms"][
                "GraphRetrieve.family_exact_comparison_count"
            ] = value
        result = audit.suffix_plateau_result(profile, rows, 10)
        self.assertTrue(result["passed"], result.get("candidate_failures"))
        counter = result["accepted_suffix"]["work_counter_results"][
            "graph_exact_comparison_count"
        ]
        self.assertGreater(counter["second_over_first"], 1.10)
        self.assertTrue(counter["absolute_bound_passed"])

    def test_accepts_sparse_route_rows_under_knob_bounds(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        profile.update(
            {
                "focus": 0.5,
                "sensitivity": 0.5,
                "stability": 0.5,
                "sparse_route_parameters": fixture_sparse_parameters(),
            }
        )
        ranges = audit.end_anchored_ranges(len(rows), 10)
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 0.0
            window_index = next(
                (
                    index
                    for index, (start, end) in enumerate(ranges)
                    if start <= row["event_index"] < end
                ),
                0,
            )
            graph_rows = min(1280.0, 10.0 * 1.2**window_index)
            supersession_rows = min(5888.0, 20.0 * 1.2**window_index)
            row["work_counters"]["graph_rows_visited"] = graph_rows
            row["operation_ms"]["GraphRetrieve.rows_visited"] = graph_rows
            row["work_counters"][
                "supersession_rows_visited"
            ] = supersession_rows
            row["operation_ms"][
                "MemoryStorage.supersession_sparse_route_node_rows"
            ] = supersession_rows - 20.0
        result = audit.suffix_plateau_result(profile, rows, 10)
        self.assertTrue(result["passed"], result.get("candidate_failures"))
        for counter_name, expected_bound in (
            ("graph_rows_visited", 1280),
            ("supersession_rows_visited", 5888),
        ):
            counter = result["accepted_suffix"]["work_counter_results"][
                counter_name
            ]
            self.assertEqual(
                counter["knob_derived_absolute_bound"], expected_bound
            )
            self.assertTrue(counter["absolute_bound_passed"])

    def test_accepts_variable_emotional_updates_under_knob_bound(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        profile.update(
            {
                "focus": 0.5,
                "sensitivity": 0.5,
                "stability": 0.5,
                "sparse_route_parameters": fixture_sparse_parameters(),
            }
        )
        ranges = audit.end_anchored_ranges(len(rows), 10)
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 0.0
            window_index = next(
                (
                    index
                    for index, (start, end) in enumerate(ranges)
                    if start <= row["event_index"] < end
                ),
                0,
            )
            value = min(1280.0, 4.0 * 1.2**window_index)
            row["work_counters"]["emotional_update_count"] = value
            row["operation_ms"]["EmotionalCascade.update_count"] = value
        result = audit.suffix_plateau_result(profile, rows, 10)
        self.assertTrue(result["passed"], result.get("candidate_failures"))
        counter = result["accepted_suffix"]["work_counter_results"][
            "emotional_update_count"
        ]
        self.assertEqual(counter["knob_derived_absolute_bound"], 1280)
        self.assertTrue(counter["absolute_bound_passed"])

    def test_rejects_exact_comparison_above_knob_bound(self) -> None:
        profile, rows = plateau_fixture()
        set_fixture_retention(profile, rows, "durable")
        profile.update(
            {
                "focus": 0.5,
                "sensitivity": 0.5,
                "stability": 0.5,
                "sparse_route_parameters": fixture_sparse_parameters(),
            }
        )
        for row in rows:
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_limit"
            ] = 128.0
            row["operation_ms"][
                "Cortext.fallback_hydration_signal_rows"
            ] = 0.0
        rows[-1]["work_counters"]["graph_exact_comparison_count"] = 1025.0
        rows[-1]["operation_ms"][
            "GraphRetrieve.family_exact_comparison_count"
        ] = 1025.0
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])

    def test_rejects_growing_consolidation_cost_with_flat_process(self) -> None:
        profile, rows = plateau_fixture(
            elapsed_windows=[2.0, 4.0, 6.0, 8.0]
            + [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
        )
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])

    def test_rejects_plateau_shorter_than_six_windows(self) -> None:
        profile, rows = plateau_fixture(
            process_windows=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0] + [10.0] * 5
        )
        self.assertFalse(audit.suffix_plateau_result(profile, rows, 10)["passed"])


class ActivityNormalizedResetDiagnosticTest(unittest.TestCase):
    @staticmethod
    def evaluation(
        *,
        operation: object = "MemoryStorage",
        epoch: int = 28,
        trailing_ms: list[float] | None = None,
        post_ms: list[float] | None = None,
        trailing_activity: list[float] | None = None,
        post_activity: list[float] | None = None,
        modality: str = "text",
        source_id: str = "opaque-a",
    ) -> dict[str, object]:
        return {
            "epoch": epoch,
            "operation": operation,
            "trailing_event_indices": [100, 101],
            "post_event_indices": [102, 103],
            "trailing_ms": trailing_ms if trailing_ms is not None else [1.0, 1.0],
            "post_ms": post_ms if post_ms is not None else [2.0, 2.0],
            "trailing_activity": (
                trailing_activity if trailing_activity is not None else [2.0, 2.0]
            ),
            "post_activity": (
                post_activity if post_activity is not None else [8.0, 8.0]
            ),
            "modality": modality,
            "source_id": source_id,
        }

    def test_rising_raw_peak_cannot_be_waived(self) -> None:
        result = audit.activity_normalized_reset_diagnostic(
            [self.evaluation()], ["sawtooth peaks rise"]
        )
        self.assertFalse(result["raw_gate_passed"])
        self.assertEqual(result["raw_gate_failures"], ["sawtooth peaks rise"])

    def test_rising_raw_trough_cannot_be_waived(self) -> None:
        result = audit.activity_normalized_reset_diagnostic(
            [self.evaluation()], ["sawtooth troughs rise"]
        )
        self.assertFalse(result["raw_gate_passed"])
        self.assertEqual(result["raw_gate_failures"], ["sawtooth troughs rise"])

    def test_raw_reset_miss_cannot_be_waived(self) -> None:
        result = audit.activity_normalized_reset_diagnostic(
            [self.evaluation()], ["material epoch does not lower following process time"]
        )
        self.assertFalse(result["raw_gate_passed"])
        self.assertEqual(result["diagnostic_result"], "activity-incidence")

    def test_positive_positive_uses_ratio_of_sums(self) -> None:
        result = audit.activity_normalized_reset_diagnostic([
            self.evaluation(
                trailing_ms=[1.0, 9.0],
                trailing_activity=[1.0, 9.0],
                post_ms=[2.0, 18.0],
                post_activity=[1.0, 99.0],
            )
        ], [])
        row = result["evaluations"][0]
        self.assertEqual(row["trailing_unit_cost"], 1.0)
        self.assertEqual(row["post_unit_cost"], 0.2)
        self.assertEqual(row["unit_cost_ratio"], 0.2)

    def test_zero_zero_is_unevaluated_and_non_rescuing(self) -> None:
        result = audit.activity_normalized_reset_diagnostic([
            self.evaluation(
                trailing_ms=[0.0, 0.0],
                post_ms=[0.0, 0.0],
                trailing_activity=[0.0, 0.0],
                post_activity=[0.0, 0.0],
            )
        ], ["raw miss"])
        self.assertEqual(result["evaluations"][0]["status"], "unevaluated")
        self.assertIsNone(result["selected_hotspot"])
        self.assertFalse(result["raw_gate_passed"])

    def test_one_sided_zero_is_incomparable(self) -> None:
        result = audit.activity_normalized_reset_diagnostic([
            self.evaluation(
                trailing_ms=[0.0, 0.0],
                trailing_activity=[0.0, 0.0],
                post_ms=[1.0, 1.0],
                post_activity=[2.0, 2.0],
            )
        ], [])
        self.assertEqual(result["evaluations"][0]["status"], "incomparable")
        self.assertEqual(result["diagnostic_result"], "diagnostic-inconclusive")

    def test_negative_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            audit.activity_normalized_reset_diagnostic([
                self.evaluation(trailing_ms=[-1.0, 1.0])
            ], [])

    def test_nonfinite_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            audit.activity_normalized_reset_diagnostic([
                self.evaluation(trailing_ms=[math.nan, 1.0])
            ], [])

    def test_unequal_windows_are_rejected(self) -> None:
        item = self.evaluation()
        item["post_event_indices"] = [102]
        with self.assertRaisesRegex(ValueError, "matching event windows"):
            audit.activity_normalized_reset_diagnostic([item], [])

    def test_equal_length_interleaved_windows_are_rejected(self) -> None:
        item = self.evaluation()
        item["trailing_event_indices"] = [100, 102]
        item["post_event_indices"] = [101, 103]
        with self.assertRaisesRegex(ValueError, "contiguous adjacent event windows"):
            audit.activity_normalized_reset_diagnostic([item], [])

    def test_operations_in_one_epoch_require_identical_windows(self) -> None:
        first = self.evaluation(operation="MemoryStorage", epoch=28)
        second = self.evaluation(operation="PropagateEmotionalCascade", epoch=28)
        second["trailing_event_indices"] = [200, 201]
        second["post_event_indices"] = [202, 203]
        with self.assertRaisesRegex(ValueError, "same epoch must share event windows"):
            audit.activity_normalized_reset_diagnostic([first, second], [])

    def test_finite_samples_that_overflow_aggregate_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "aggregate.*finite"):
            audit.activity_normalized_reset_diagnostic([
                self.evaluation(post_activity=[1e308, 1e308])
            ], [])

    def test_pooled_operations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "operation id"):
            audit.activity_normalized_reset_diagnostic([
                self.evaluation(operation=["MemoryStorage", "PropagateEmotionalCascade"])
            ], [])

    def test_modality_and_source_labels_do_not_change_arithmetic(self) -> None:
        baseline = audit.activity_normalized_reset_diagnostic([
            self.evaluation(modality="text", source_id="shared")
        ], [])
        for modality, source_id in (("audio", "opaque-a"), ("image", "opaque-b")):
            with self.subTest(modality=modality, source_id=source_id):
                varied = audit.activity_normalized_reset_diagnostic([
                    self.evaluation(modality=modality, source_id=source_id)
                ], [])
                self.assertEqual(varied, baseline)

    def test_non_neutral_knobs_reject_hidden_midpoint_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "processed rows exceed F/S/T-derived B"):
            audit.activity_normalized_reset_diagnostic(
                [self.evaluation()], [],
                knob_context={"focus": 0.0, "sensitivity": 0.0, "stability": 0.0,
                              "processed_backfill_rows": 128},
            )

    def test_neutral_b_plus_one_is_logical_not_processed(self) -> None:
        with self.assertRaisesRegex(ValueError, "processed rows exceed F/S/T-derived B"):
            audit.activity_normalized_reset_diagnostic(
                [self.evaluation()], [],
                knob_context={"focus": 0.5, "sensitivity": 0.5, "stability": 0.5,
                              "processed_backfill_rows": 129},
            )

    def test_no_evaluable_operation_returns_deepen_profile(self) -> None:
        result = audit.activity_normalized_reset_diagnostic([
            self.evaluation(
                trailing_ms=[0.0, 0.0], post_ms=[0.0, 0.0],
                trailing_activity=[0.0, 0.0], post_activity=[0.0, 0.0],
            )
        ], [])
        self.assertEqual(result["diagnostic_result"], "diagnostic-inconclusive")
        self.assertEqual(result["continuation"], "deepen-profile")
        self.assertIsNone(result["selected_hotspot"])

    def test_deterministic_tie_uses_operation_then_epoch(self) -> None:
        result = audit.activity_normalized_reset_diagnostic([
            self.evaluation(operation="PropagateEmotionalCascade", epoch=28),
            self.evaluation(operation="MemoryStorage", epoch=35),
        ], [])
        self.assertEqual(result["selected_hotspot"]["operation"], "MemoryStorage")
        self.assertEqual(result["selected_hotspot"]["epoch"], 35)

    def test_consolidation_result_reports_diagnostic_without_waiving_raw_miss(
        self,
    ) -> None:
        profile, rows = epoch_fixture()
        close = profile["consolidation_events"][0]["event_index"]
        for row in rows[close + 1 : close + 51]:
            row["process_ms"] = 1.14
            row["end_to_end_ms"] = 1.14
            row["operation_ms"]["cortext::operations::MemoryStorage"] = 0.2
            row["work_counters"]["supersession_current_candidate_count"] = 20.0
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        diagnostic = result["activity_normalized_reset_diagnostic"]
        self.assertFalse(result["passed"])
        self.assertIn(
            "material epoch does not lower following process time",
            result["failures"],
        )
        self.assertFalse(diagnostic["raw_gate_passed"])
        self.assertFalse(diagnostic["raw_gate_waived"])
        self.assertEqual(diagnostic["diagnostic_result"], "activity-incidence")
        self.assertEqual(
            diagnostic["selected_hotspot"]["operation"], "MemoryStorage"
        )


class BoundedActivationQualityContractTest(unittest.TestCase):
    def quality_fixture(self, role: str = "candidate") -> dict[str, object]:
        evidence = []
        modalities = ["text", "audio", "image"]
        embedding_count = 40000
        for query_number in range(512):
            query_index = query_number * (embedding_count - 1) // 511
            ranked_ids = [query_number * 100 + rank for rank in range(16)]
            evidence.append(
                {
                    "query_number": query_number,
                    "query_index": query_index,
                    "query_embedding_id": 1000000 + query_index,
                    "exact_ranked_ids": ranked_ids,
                    "candidate_ranked_ids": ranked_ids.copy(),
                    "exact_neighbor_semantic_coverage": 1.0,
                    "source_ids": [f"opaque-{query_number % 4}"],
                    "modalities": [modalities[query_number % 3]],
                    "history_ordinal": query_index,
                    "history_count": embedding_count,
                    "history_segment": ("early", "middle", "late")[
                        min(2, query_index * 3 // embedding_count)
                    ],
                }
            )
        fixed_probe_numbers = [0, 85, 170, 255, 340, 425, 511]
        artifact = {
            "schema": "cortext_bounded_activation_shadow_quality_v1",
            "quality_role": role,
            "control_kind": (
                "current-public-retrieval"
                if role == "approved-control"
                else "candidate-router"
            ),
            "existing_fixed_probe_identity_rank_sha256": "1" * 64,
            "embedding_count": embedding_count,
            "query_count": 512,
            "result_k": 16,
            "focus": 0.5,
            "sensitivity": 0.5,
            "stability": 0.5,
            "fixed_embedding_slots": 16020,
            "quality": {
                "all": {
                    "count": 512,
                    "mean_exact_id_recall_at_k": 1.0,
                    "mean_top1_exact_id": 1.0,
                    "mean_exact_neighbor_semantic_coverage": 1.0,
                }
            },
            "query_evidence": evidence,
            "fixed_identity_rank_probes": [
                {
                    "query_number": query_number,
                    "exact_ranked_ids": evidence[query_number][
                        "exact_ranked_ids"
                    ],
                    "candidate_ranked_ids": evidence[query_number][
                        "candidate_ranked_ids"
                    ],
                }
                for query_number in fixed_probe_numbers
            ],
            "sqlite_restart_measurements": [
                {
                    "fraction": 0.25,
                    "retained_rows": 10000,
                    "rows_visited": 8000,
                    "measurement_authority": "production-shaped-persistent-restart",
                    "restored_candidate_count": 16,
                    "pre_restart_probe_identity_rank_sha256": "0" * 64,
                    "post_restart_probe_identity_rank_sha256": "0" * 64,
                    "restart_production_gate": True,
                    "linear_history": False,
                },
                {
                    "fraction": 0.5,
                    "retained_rows": 20000,
                    "rows_visited": 12000,
                    "measurement_authority": "production-shaped-persistent-restart",
                    "restored_candidate_count": 16,
                    "pre_restart_probe_identity_rank_sha256": "0" * 64,
                    "post_restart_probe_identity_rank_sha256": "0" * 64,
                    "restart_production_gate": True,
                    "linear_history": False,
                },
                {
                    "fraction": 1.0,
                    "retained_rows": 40000,
                    "rows_visited": 16020,
                    "measurement_authority": "production-shaped-persistent-restart",
                    "restored_candidate_count": 16,
                    "pre_restart_probe_identity_rank_sha256": "0" * 64,
                    "post_restart_probe_identity_rank_sha256": "0" * 64,
                    "restart_production_gate": True,
                    "linear_history": False,
                }
            ],
        }
        candidate_rank_digest = audit.ranked_query_evidence_result(
            artifact, 16
        )["candidate_identity_rank_sha256"]
        artifact["candidate_identity_rank_sha256"] = candidate_rank_digest
        artifact["repeat_candidate_identity_rank_sha256"] = (
            candidate_rank_digest
        )
        restart_probe_digest = audit.ranked_query_evidence_result(
            artifact, 16
        )["fixed_probe_identity_rank_sha256"]
        for item in artifact["sqlite_restart_measurements"]:
            item["pre_restart_probe_identity_rank_sha256"] = restart_probe_digest
            item["post_restart_probe_identity_rank_sha256"] = restart_probe_digest
        return artifact

    def control_fixture(self) -> dict[str, object]:
        return self.quality_fixture("approved-control")

    def audit_result(
        self, candidate: dict[str, object],
        control: dict[str, object] | None = None,
    ) -> dict[str, object]:
        control = control or self.control_fixture()
        control_sha256 = audit.hashlib.sha256(
            audit.canonical_json(control)
        ).hexdigest()
        return audit.bounded_activation_quality_result(
            candidate, control, control_sha256
        )

    def test_accepts_exact_quality_and_bounded_restart(self) -> None:
        result = self.audit_result(self.quality_fixture())
        self.assertTrue(result["passed"], result["failures"])

    def test_accepts_approved_lower_rank_loss_above_floor(self) -> None:
        fixture = self.quality_fixture()
        fixture["query_evidence"][37]["candidate_ranked_ids"][-1] = 99999999
        fixture["quality"]["all"]["mean_exact_id_recall_at_k"] = (
            8191 / 8192
        )
        digest = audit.ranked_query_evidence_result(
            fixture, 16
        )["candidate_identity_rank_sha256"]
        fixture["candidate_identity_rank_sha256"] = digest
        fixture["repeat_candidate_identity_rank_sha256"] = digest
        result = self.audit_result(fixture)
        self.assertTrue(result["passed"], result["failures"])

    def test_rejects_unrepeated_candidate_rank_digest(self) -> None:
        fixture = self.quality_fixture()
        fixture["repeat_candidate_identity_rank_sha256"] = "f" * 64
        result = self.audit_result(fixture)
        self.assertFalse(result["passed"])
        self.assertIn(
            "candidate rank order is not repeat-deterministic",
            result["failures"],
        )

    def test_rejects_unapproved_control_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "control digest is not approved"):
            audit.bounded_activation_quality_result(
                self.quality_fixture(), self.control_fixture(), "0" * 64
            )

    def test_rejects_nonpublic_control_kind(self) -> None:
        control = self.control_fixture()
        control["control_kind"] = "exhaustive-embedding-neighbor-candidate"
        with self.assertRaisesRegex(ValueError, "not the current public retrieval"):
            self.audit_result(self.quality_fixture(), control)

    def test_rejects_unbound_existing_public_probe_digest(self) -> None:
        fixture = self.quality_fixture()
        fixture["existing_fixed_probe_identity_rank_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "seven-probe digest"):
            self.audit_result(fixture)

    def test_rejects_candidate_that_differs_from_approved_control_rank(self) -> None:
        control = self.control_fixture()
        control["query_evidence"][37]["exact_ranked_ids"][1:3] = reversed(
            control["query_evidence"][37]["exact_ranked_ids"][1:3]
        )
        result = self.audit_result(self.quality_fixture(), control)
        self.assertFalse(result["passed"])
        self.assertIn(
            "candidate evidence differs from approved control corpus",
            result["failures"],
        )

    def test_rejects_missing_deterministic_query_identity(self) -> None:
        fixture = self.quality_fixture()
        fixture["query_evidence"][0].pop("query_embedding_id")
        with self.assertRaisesRegex(
            ValueError, "query identity is not bound to deterministic sampling"
        ):
            self.audit_result(fixture)

    def test_rejects_arbitrary_fixed_probe_positions(self) -> None:
        fixture = self.quality_fixture()
        fixture["fixed_identity_rank_probes"] = [
            {
                "query_number": query_number,
                "exact_ranked_ids": fixture["query_evidence"][query_number][
                    "exact_ranked_ids"
                ],
                "candidate_ranked_ids": fixture["query_evidence"][query_number][
                    "candidate_ranked_ids"
                ],
            }
            for query_number in range(7)
        ]
        with self.assertRaisesRegex(
            ValueError, "fixed identity-rank probe query is invalid"
        ):
            self.audit_result(fixture)

    def test_rejects_current_observation_residuals(self) -> None:
        fixture = self.quality_fixture()
        control = self.control_fixture()
        for item in fixture["query_evidence"]:
            item["candidate_ranked_ids"] = [
                value + 100000000 for value in item["exact_ranked_ids"]
            ]
        for probe in fixture["fixed_identity_rank_probes"]:
            query_number = probe["query_number"]
            probe["candidate_ranked_ids"] = fixture["query_evidence"][
                query_number
            ]["candidate_ranked_ids"]
        digest = audit.ranked_query_evidence_result(
            fixture, 16
        )["candidate_identity_rank_sha256"]
        fixture["candidate_identity_rank_sha256"] = digest
        fixture["repeat_candidate_identity_rank_sha256"] = digest
        fixture["quality"]["all"]["mean_exact_id_recall_at_k"] = 0.0
        fixture["quality"]["all"]["mean_top1_exact_id"] = 0.0
        for item in fixture["sqlite_restart_measurements"]:
            item["rows_visited"] = item["retained_rows"]
            item["restart_production_gate"] = False
            item["linear_history"] = True
        result = self.audit_result(fixture, control)
        self.assertFalse(result["passed"])
        self.assertIn("exact identity recall below invariant", result["failures"])
        self.assertIn("exact top-1 below invariant", result["failures"])
        self.assertIn("restart remains proportional to history", result["failures"])

    def test_accepts_one_nonclusterable_top1_miss_in_512_queries(self) -> None:
        fixture = self.quality_fixture()
        fixture["query_evidence"][37]["candidate_ranked_ids"][0:2] = reversed(
            fixture["query_evidence"][37]["candidate_ranked_ids"][0:2]
        )
        digest = audit.ranked_query_evidence_result(
            fixture, 16
        )["candidate_identity_rank_sha256"]
        fixture["candidate_identity_rank_sha256"] = digest
        fixture["repeat_candidate_identity_rank_sha256"] = digest
        fixture["quality"]["all"]["mean_top1_exact_id"] = 511 / 512
        result = self.audit_result(fixture)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["exact_top1_miss_count"], 1)

    def test_rejects_missing_mixed_input_coverage(self) -> None:
        fixture = self.quality_fixture()
        control = self.control_fixture()
        for item in fixture["query_evidence"]:
            item["source_ids"] = ["one-source"]
            item["modalities"] = ["text"]
            item["history_ordinal"] = 0
            item["history_segment"] = "early"
        result = self.audit_result(fixture, control)
        self.assertFalse(result["passed"])
        self.assertIn(
            "mixed source modality and history coverage missing",
            result["failures"],
        )

    def test_rejects_out_of_range_quality_fraction(self) -> None:
        fixture = self.quality_fixture()
        fixture["quality"]["all"]["mean_top1_exact_id"] = 1.01
        with self.assertRaisesRegex(ValueError, "must not exceed one"):
            self.audit_result(fixture)

    def test_rejects_reordered_exact_neighbors(self) -> None:
        fixture = self.quality_fixture()
        control = self.control_fixture()
        for item in fixture["query_evidence"]:
            item["candidate_ranked_ids"][1:3] = reversed(
                item["candidate_ranked_ids"][1:3]
            )
        for probe in fixture["fixed_identity_rank_probes"]:
            query_number = probe["query_number"]
            probe["candidate_ranked_ids"] = fixture["query_evidence"][
                query_number
            ]["candidate_ranked_ids"]
        result = self.audit_result(fixture, control)
        self.assertFalse(result["passed"])
        self.assertIn("fixed identity-rank probe differs", result["failures"])

    def test_rejects_unbound_query_count(self) -> None:
        fixture = self.quality_fixture()
        fixture["quality"]["all"]["count"] = 0
        result = self.audit_result(fixture)
        self.assertFalse(result["passed"])
        self.assertIn("quality query evidence count mismatch", result["failures"])

    def test_rejects_linear_restart_despite_passing_labels(self) -> None:
        fixture = self.quality_fixture()
        for item in fixture["sqlite_restart_measurements"]:
            item["rows_visited"] = item["retained_rows"]
        result = self.audit_result(fixture)
        self.assertFalse(result["passed"])
        self.assertIn(
            "restart remains proportional to history", result["failures"]
        )

    def test_accepts_fraction_local_restart_probe_digests(self) -> None:
        fixture = self.quality_fixture()
        for index, digest in enumerate(("a" * 64, "b" * 64)):
            fixture["sqlite_restart_measurements"][index][
                "pre_restart_probe_identity_rank_sha256"
            ] = digest
            fixture["sqlite_restart_measurements"][index][
                "post_restart_probe_identity_rank_sha256"
            ] = digest
        result = self.audit_result(fixture)
        self.assertTrue(result["passed"], result["failures"])

    def test_rejects_vacuous_restart_series(self) -> None:
        fixture = self.quality_fixture()
        for index, item in enumerate(fixture["sqlite_restart_measurements"]):
            item["retained_rows"] = 16021 + index
            item["rows_visited"] = 0
            item["restored_candidate_count"] = 0
        with self.assertRaisesRegex(
            ValueError, "not bound to restored corpus state"
        ):
            self.audit_result(fixture)

    def test_rejects_restart_probe_digest_mismatch(self) -> None:
        fixture = self.quality_fixture()
        fixture["sqlite_restart_measurements"][1][
            "post_restart_probe_identity_rank_sha256"
        ] = "6" * 64
        with self.assertRaisesRegex(
            ValueError, "not bound to restored corpus state"
        ):
            self.audit_result(fixture)

    def test_rejects_restart_probe_digest_that_is_consistent_but_wrong(self) -> None:
        fixture = self.quality_fixture()
        for item in fixture["sqlite_restart_measurements"]:
            item["pre_restart_probe_identity_rank_sha256"] = "6" * 64
            item["post_restart_probe_identity_rank_sha256"] = "6" * 64
        with self.assertRaisesRegex(
            ValueError, "not bound to restored corpus state"
        ):
            self.audit_result(fixture)

    def test_quality_cli_enforces_contract_and_report_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "quality.json"
            control = root / "control.json"
            output = root / "audit.json"
            fixture = self.quality_fixture()
            fixture["query_evidence"][1]["candidate_ranked_ids"][1:3] = reversed(
                fixture["query_evidence"][1]["candidate_ranked_ids"][1:3]
            )
            artifact.write_text(json.dumps(fixture), encoding="utf-8")
            control.write_text(
                json.dumps(self.control_fixture()),
                encoding="utf-8",
            )
            approved_control_sha256 = audit.hashlib.sha256(
                audit.canonical_json(self.control_fixture())
            ).hexdigest()
            command = [
                sys.executable,
                str(Path(__file__).with_name("audit_storage_cost_profile.py")),
                "--shadow-quality",
                str(artifact),
                "--quality-control",
                str(control),
                "--approved-control-sha256",
                approved_control_sha256,
                "--out",
                str(output),
            ]
            enforced = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            report_only = subprocess.run(
                [*command, "--report-only"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(enforced.returncode, 1, enforced.stderr)
            self.assertEqual(report_only.returncode, 0, report_only.stderr)
            self.assertFalse(json.loads(output.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
