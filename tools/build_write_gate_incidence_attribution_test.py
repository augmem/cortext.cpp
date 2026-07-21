#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock


PATH = Path(__file__).with_name("build_write_gate_incidence_attribution.py")
SPEC = importlib.util.spec_from_file_location("write_gate_builder", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WriteGateAttributionBuilderTest(unittest.TestCase):
    @staticmethod
    def row(index: int, reason: int) -> dict:
        forced = reason == 4
        accepted = reason in (4, 5)
        boundary = reason in (3, 4, 5, 6)
        return {
            "event_index": index,
            "operation_ms": {
                "WriteGate.flush_trigger": float(boundary),
                "WriteGate.spike_bypass": float(forced),
                "WriteGate.accumulator_available": float(boundary),
                "WriteGate.n_signals": 3.0 if boundary else 0.0,
                "WriteGate.coverage": 1.0 if boundary else 0.0,
                "WriteGate.window_score": 0.6 if reason in (5, 6) else 0.0,
                "WriteGate.threshold_dynamic": 0.3,
                "WriteGate.refractory_multiplier": 1.0,
                "WriteGate.write_scale": 1.0,
                "WriteGate.effective_threshold": 0.5 if reason in (5, 6) else 0.0,
                "WriteGate.score_margin": 0.1 if reason in (5, 6) else 0.0,
                "WriteGate.force_write": float(forced),
                "WriteGate.write_accumulator": float(accepted),
                "WriteGate.reason_code": float(reason),
                "cortext::operations::MemoryStorage": 2.0 if accepted else 0.0,
                "MemoryStorage.supersession_edges": 1.0 if accepted else 0.0,
                "MemoryStorage.supersession_current_candidate_count": 4.0 if accepted else 0.0,
                "MemoryStorage.supersession_current_rows_visited": 8.0 if accepted else 0.0,
                "MemoryStorage.supersession_sparse_route_node_rows": 10.0 if accepted else 0.0,
            },
        }

    @classmethod
    def profile(cls) -> dict:
        reasons = [4, 5, 6, 2, 6, 2, 4, 5]
        params = MODULE.derived_parameters(0.5, 0.5, 0.5)
        return {
            "processed_events": len(reasons),
            "consolidation_runs": 2,
            "focus": 0.5,
            "sensitivity": 0.5,
            "stability": 0.5,
            "sparse_route_parameters": {
                "route_capacity": params["C"],
                "backfill_batch_size": params["B"],
                "activation_identity_target": params["A"],
                "search_node_budget": params["public_query_node_budget"],
                "backfill_search_node_budget": params["construction_node_budget"],
                "backfill_search_effort": params["construction_queue_effort"],
            },
            "working_set_curve": [
                cls.row(index, reason) for index, reason in enumerate(reasons)
            ],
        }

    @staticmethod
    def audit() -> dict:
        return {
            "plateau": {
                "candidate_failures": [{
                    "plateau_start_event": 2,
                    "plateau_window_count": 6,
                    "process_ratio": 1.0,
                    "throughput_ratio": 1.0,
                    "relative_slope": 0.0,
                    "relative_upper": 0.0,
                    "operation_failure_count": 0,
                    "counter_failure_count": 0,
                    "height_passed": True,
                    "store_growth_passed": True,
                    "consolidation_mode": "invalid",
                    "consolidation_epoch_passed": False,
                }]
            }
        }

    def test_build_binds_exact_first_and_final_ranges(self) -> None:
        result = MODULE.build(
            self.profile(), "1" * 64, self.audit(), "2" * 64,
            window_size=2, window_count=1,
        )
        self.assertEqual(result["window_contract"]["first_range"], [0, 1])
        self.assertEqual(result["window_contract"]["final_range"], [6, 7])
        self.assertEqual(
            result["final_range_aggregate"]["writes"], 2
        )
        self.assertEqual(result["derived_parameters"]["neutral_B"], 128)
        self.assertEqual(
            result["derived_parameters"]["neutral_logical_B_plus_one_only"],
            129,
        )
        self.assertFalse(result["decision"]["raw_reset_gate_waived"])
        self.assertTrue(
            result["decision"]["write_incidence_rejected_as_growth_source"]
        )

    def test_build_does_not_reject_nonflat_incidence(self) -> None:
        profile = self.profile()
        profile["working_set_curve"][-2] = self.row(6, 5)
        profile["working_set_curve"][-1] = self.row(7, 2)
        result = MODULE.build(
            profile, "1" * 64, self.audit(), "2" * 64,
            window_size=2, window_count=1,
        )
        self.assertFalse(result["incidence_ratios"]["flat_within_ten_percent"])
        self.assertFalse(
            result["decision"]["write_incidence_rejected_as_growth_source"]
        )
        self.assertIsNone(result["decision"]["write_incidence_growth_source"])
        self.assertEqual(
            result["status"],
            "attribution-only-write-gate-growth-source-unresolved",
        )

    def test_build_derives_audit_gate_state(self) -> None:
        audit = self.audit()
        audit["plateau"]["passed"] = True
        candidate = audit["plateau"]["candidate_failures"][0]
        candidate["consolidation_mode"] = "sawtooth"
        candidate["consolidation_epoch_passed"] = True
        result = MODULE.build(
            self.profile(), "1" * 64, audit, "2" * 64,
            window_size=2, window_count=1,
        )
        self.assertTrue(result["decision"]["raw_reset_gate_passed"])
        self.assertTrue(result["decision"]["whole_plateau_passed"])

    def test_cli_rejects_noncanonical_window_override(self) -> None:
        argv = [
            str(PATH),
            "--profile", "/does/not/exist-profile.json",
            "--plateau-audit", "/does/not/exist-audit.json",
            "--out", "/does/not/exist-output.json",
            "--window-size", "1",
        ]
        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    MODULE.main()

    def test_build_rejects_noncontiguous_event_identity(self) -> None:
        profile = self.profile()
        profile["working_set_curve"][-1]["event_index"] = 99
        with self.assertRaisesRegex(ValueError, "contiguous"):
            MODULE.build(
                profile, "1" * 64, self.audit(), "2" * 64,
                window_size=2, window_count=1,
            )

    def test_build_rejects_formula_drift(self) -> None:
        profile = self.profile()
        profile["sparse_route_parameters"]["backfill_batch_size"] = 129
        with self.assertRaisesRegex(ValueError, "F/S/T"):
            MODULE.build(
                profile, "1" * 64, self.audit(), "2" * 64,
                window_size=2, window_count=1,
            )


if __name__ == "__main__":
    unittest.main()
