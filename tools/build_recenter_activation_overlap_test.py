#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("build_recenter_activation_overlap.py")
SPEC = importlib.util.spec_from_file_location("recenter_overlap_builder", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecenterActivationOverlapBuilderTest(unittest.TestCase):
    @staticmethod
    def profile() -> dict:
        parameters = MODULE.derived_parameters(0.5, 0.5, 0.5)
        return {
            "processed_events": 1000,
            "consolidation_runs": 2,
            "focus": 0.5,
            "sensitivity": 0.5,
            "stability": 0.5,
            "sparse_route_parameters": {
                "route_capacity": parameters["C"],
                "backfill_batch_size": parameters["B"],
                "activation_identity_target": parameters["A"],
                "search_node_budget": parameters["public_query_node_budget"],
                "backfill_search_node_budget": parameters["construction_node_budget"],
                "backfill_search_effort": parameters["construction_queue_effort"],
            },
            "consolidation_events": [{
                "event_index": 999,
                "sqlite_sparse_route_recenter_succeeded": True,
                "sqlite_sparse_route_recenter_derived_centroid": True,
                "sqlite_sparse_route_recenter_centroid_source": 1.0,
                "sqlite_sparse_route_recenter_centroid_cluster_count": 2.0,
                "sqlite_sparse_route_recenter_centroid_member_count": 12.0,
                "sqlite_sparse_route_recenter_overlap_profiled": True,
                "sqlite_sparse_route_recenter_overlap_pair_valid": True,
                "sqlite_sparse_route_recenter_overlap_failure_code": 0.0,
                "sqlite_sparse_route_recenter_pre_activated_count": 1280.0,
                "sqlite_sparse_route_recenter_post_activated_count": 1280.0,
                "sqlite_sparse_route_recenter_overlap_count": 1280.0,
                "sqlite_sparse_route_recenter_pre_node_count": 2560.0,
                "sqlite_sparse_route_recenter_post_node_count": 2560.0,
            }],
        }

    def test_build_binds_knob_derived_cardinalities(self) -> None:
        result = MODULE.build(
            self.profile(), "1" * 64, "2" * 64, "3" * 64, "4" * 64
        )
        self.assertEqual(result["derived_parameters"]["neutral_B"], 128)
        self.assertEqual(
            result["derived_parameters"]["neutral_logical_B_plus_one_only"],
            129,
        )
        self.assertEqual(result["invalid_profiled_pairs"], 0)
        self.assertEqual(result["unchanged_activated_set_count"], 1)
        self.assertEqual(result["minimum_centroid_cluster_count"], 2)
        self.assertEqual(result["minimum_centroid_member_count"], 12)
        self.assertEqual(result["cluster_centroid_recenter_count"], 1)
        self.assertEqual(result["active_signal_ring_centroid_recenter_count"], 0)
        self.assertEqual(result["required_follow_on"]["structural_knob_points"], 27)
        self.assertEqual(
            result["required_follow_on"]["production_shaped_knob_points"], 9
        )

    def test_build_rejects_invalid_profiled_pair(self) -> None:
        profile = self.profile()
        profile["consolidation_events"][0][
            "sqlite_sparse_route_recenter_overlap_pair_valid"
        ] = False
        profile["consolidation_events"][0][
            "sqlite_sparse_route_recenter_overlap_failure_code"
        ] = 2.0
        with self.assertRaisesRegex(ValueError, "invalid recenter overlap pair"):
            MODULE.build(profile, "1" * 64, "2" * 64, "3" * 64, "4" * 64)

    def test_build_rejects_hidden_midpoint_constant(self) -> None:
        profile = self.profile()
        profile["focus"] = 0.0
        with self.assertRaisesRegex(ValueError, "route_capacity does not match"):
            MODULE.build(profile, "1" * 64, "2" * 64, "3" * 64, "4" * 64)

    def test_build_rejects_signal_embedding_recenter(self) -> None:
        profile = self.profile()
        profile["consolidation_events"][0][
            "sqlite_sparse_route_recenter_derived_centroid"
        ] = False
        with self.assertRaisesRegex(
            ValueError, "did not use a consolidation-derived centroid"
        ):
            MODULE.build(profile, "1" * 64, "2" * 64, "3" * 64, "4" * 64)

    def test_build_accepts_bounded_active_signal_ring_centroid(self) -> None:
        profile = self.profile()
        event = profile["consolidation_events"][0]
        event["sqlite_sparse_route_recenter_centroid_source"] = 2.0
        event["sqlite_sparse_route_recenter_centroid_cluster_count"] = 0.0
        result = MODULE.build(
            profile, "1" * 64, "2" * 64, "3" * 64, "4" * 64
        )
        self.assertEqual(result["cluster_centroid_recenter_count"], 0)
        self.assertEqual(result["active_signal_ring_centroid_recenter_count"], 1)


if __name__ == "__main__":
    unittest.main()
