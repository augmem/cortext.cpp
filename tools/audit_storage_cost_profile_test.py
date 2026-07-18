#!/usr/bin/env python3
"""Focused contract tests for storage-cost profile attribution and CLI gates."""

from __future__ import annotations

import json
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
        "consolidation_epoch_relational_contract_sha256": (
            audit.CONSOLIDATION_EPOCH_RELATIONAL_CONTRACT_SHA256
        ),
        "retention": "natural",
        "consolidation_runs": 0,
        "consolidation_ms": 0.0,
        "consolidation_events": [],
        "honor_required_consolidation": True,
        "active_epoch_limits": audit.ACTIVE_EPOCH_LIMITS,
        "store_checkpoints": checkpoints,
    }
    return profile, rows


def epoch_fixture(
    *,
    epoch_count: int = 9,
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
        "consolidation_epoch_relational_contract_sha256": (
            audit.CONSOLIDATION_EPOCH_RELATIONAL_CONTRACT_SHA256
        ),
        "retention": "natural",
        "consolidation_runs": len(events),
        "consolidation_ms": consolidation_total,
        "consolidation_events": events,
        "honor_required_consolidation": True,
        "active_epoch_limits": audit.ACTIVE_EPOCH_LIMITS,
        "store_checkpoints": checkpoints,
    }, rows


class AttributionContractTest(unittest.TestCase):
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
    def test_accepts_bounded_consolidation_sawtooth(self) -> None:
        profile, rows = epoch_fixture()
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "sawtooth")
        self.assertGreaterEqual(result["material_epoch_count"], 4)
        full_result = audit.suffix_plateau_result(profile, rows, 50)
        self.assertTrue(full_result["passed"], full_result["candidate_failures"])

    def test_accepts_flat_envelope_without_consolidation(self) -> None:
        profile, rows = plateau_fixture()
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "flat-envelope")

    def test_accepts_flat_envelope_with_complete_nonmaterial_epochs(self) -> None:
        profile, rows = epoch_fixture(material=False)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["mode"], "flat-envelope")
        self.assertGreaterEqual(result["complete_epoch_count"], 4)
        self.assertEqual(result["material_epoch_count"], 0)

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

    def test_rejects_consolidation_reset_counter_that_does_not_fall(self) -> None:
        profile, rows = epoch_fixture(reset_counters=False)
        result = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(result["passed"])
        self.assertIn("consolidation reset counter does not fall", result["failures"])

    def test_zero_reset_counter_must_remain_zero(self) -> None:
        profile, rows = epoch_fixture()
        counter = "accumulator_signal_count"
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

        profile["consolidation_events"][4]["post_reset_counters"][counter] = 1
        rejected = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(rejected["passed"])
        self.assertIn("consolidation reset counter does not fall", rejected["failures"])
        self.assertEqual(
            rejected["complete_epochs"][4]["reset_counter_ratios"][counter],
            float("inf"),
        )

        suffix = audit.suffix_plateau_result(profile, rows, 50)
        self.assertFalse(suffix["passed"])

    def test_nonmaterial_zero_reset_counter_must_remain_zero(self) -> None:
        profile, rows = epoch_fixture(material=False)
        counter = "accumulator_signal_count"
        for event in profile["consolidation_events"]:
            close = event["event_index"]
            rows[close]["consolidation_epoch_counters"][counter] = 0
            rows[close]["operation_ms"][f"ConsolidationEpoch.{counter}"] = 0.0
            event["pre_reset_counters"][counter] = 0
            event["post_reset_counters"][counter] = 0
        accepted = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertTrue(accepted["passed"], accepted["failures"])
        self.assertEqual(accepted["mode"], "flat-envelope")

        profile["consolidation_events"][4]["post_reset_counters"][counter] = 1
        rejected = audit.consolidation_epoch_result(profile, rows, 0, len(rows))
        self.assertFalse(rejected["passed"])
        self.assertIn("consolidation reset counter does not fall", rejected["failures"])
        self.assertEqual(
            rejected["complete_epochs"][4]["reset_counter_ratios"][counter],
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

    def test_rejects_supersession_fallback_without_exact_visit_count(self) -> None:
        profile, rows = plateau_fixture()
        rows[0]["operation_ms"][
            "MemoryStorage.supersession_sql_fallback_count"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "fallback"):
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
            row["operation_ms"]["GraphRetrieve.seed_cache_distance_rows"] = (
                1.08**window_index
            )
            row["operation_ms"]["GraphRetrieve.seed_cache_rows"] = (
                1.09**window_index
            )
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

    def test_rejects_one_growing_active_counter(self) -> None:
        profile, rows = plateau_fixture(growing_counter="graph_rows_visited")
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


if __name__ == "__main__":
    unittest.main()
