import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_sqlite_hnsw_knob_ablation.py")
SPEC = importlib.util.spec_from_file_location("knob_ablation_builder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class KnobAblationBuilderTest(unittest.TestCase):
    def test_corpus_ablation_uses_the_quality_matrix_length(self):
        self.assertEqual(MODULE.REQUIRED_PROCESSED_EVENTS_PER_RUN, 4000)

    def test_control_only_audit_cannot_pass_as_candidate_quality(self):
        control_only = {
            "passed": True,
            "candidate_complete": False,
            "candidate_quality_passed": False,
        }
        self.assertFalse(MODULE.candidate_quality_passed(control_only))

        candidate = {
            "passed": True,
            "candidate_complete": True,
            "candidate_quality_passed": True,
            "deterministic_tie_order": True,
            "exact_identity_recall_at_k": 0.998,
            "exact_top1": 1.0,
            "exact_top1_miss_count": 0,
            "semantic_coverage": 0.95,
        }
        self.assertTrue(MODULE.candidate_quality_passed(candidate))

    def test_route_quality_classification_preserves_below_threshold_state(self):
        self.assertEqual(
            MODULE.quality_classification(
                sparse_traversed=False,
                evaluated_recenter_count=0,
                node_rows=395,
                published_active_count=395,
                route_capacity=448,
            ),
            "system-quality-passed-published-below-threshold-sparse-route-unevaluated",
        )
        self.assertEqual(
            MODULE.quality_classification(
                sparse_traversed=True,
                evaluated_recenter_count=0,
                node_rows=500,
                published_active_count=500,
                route_capacity=448,
            ),
            "system-quality-passed-sparse-route-traversed-recenter-unproven",
        )
        self.assertEqual(
            MODULE.quality_classification(
                sparse_traversed=True,
                evaluated_recenter_count=1,
                node_rows=500,
                published_active_count=500,
                route_capacity=448,
            ),
            "route-lifecycle-evaluated-system-quality-passed",
        )

    def test_top1_miss_cluster_audit_accepts_zero_and_rejects_a_repeat(self):
        base = {
            "id": "midpoint",
            "knobs": [0.5, 0.5, 0.5],
            "observed": {
                "processed_events": 4000,
                "exact_top1_miss_count": 0,
                "top1_miss_fingerprints": [],
            },
        }
        self.assertEqual(
            MODULE.top1_miss_cluster_audit([base])["status"], "no-misses"
        )

        repeated = []
        for index, knobs in enumerate(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])):
            repeated.append(
                {
                    "id": f"run-{index}",
                    "knobs": knobs,
                    "observed": {
                        "processed_events": 4000,
                        "exact_top1_miss_count": 1,
                        "top1_miss_fingerprints": [
                            {
                                "query_ordinal": index,
                                "query_progress_quartile": index,
                                "source_id_blake3": f"{index + 1:064x}",
                                "modality": "text",
                                "control_top1_age_events": 1000 + index * 100,
                                "control_top1_age_ms": 10000 + index,
                            }
                        ],
                    },
                }
            )
        result = MODULE.top1_miss_cluster_audit(repeated)
        self.assertFalse(result["passed"])
        self.assertIn("modality", result["clustered_dimensions"])

    def test_required_corpus_points_are_midpoint_endpoints_and_one_axis_edges(self):
        self.assertEqual(
            MODULE.REQUIRED_CORPUS_KNOB_POINTS,
            {
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (0.5, 0.5, 0.5),
                (0.0, 0.5, 0.5),
                (1.0, 0.5, 0.5),
                (0.5, 0.0, 0.5),
                (0.5, 1.0, 0.5),
                (0.5, 0.5, 0.0),
                (0.5, 0.5, 1.0),
            },
        )

    def test_endpoint_and_midpoint_parameters(self):
        low = MODULE.expected_parameters(0.0, 0.0, 0.0)
        midpoint = MODULE.expected_parameters(0.5, 0.5, 0.5)
        high = MODULE.expected_parameters(1.0, 1.0, 1.0)

        self.assertEqual((low["route_capacity"], low["backfill_batch_size"]), (256, 64))
        self.assertEqual(
            (midpoint["route_capacity"], midpoint["backfill_batch_size"]),
            (512, 128),
        )
        self.assertEqual(
            (high["route_capacity"], high["backfill_batch_size"]), (768, 192)
        )
        self.assertEqual(
            (
                low["activation_identity_target"],
                midpoint["activation_identity_target"],
                high["activation_identity_target"],
            ),
            (640, 1280, 1920),
        )
        self.assertEqual(
            (
                low["total_query_row_budget"],
                midpoint["total_query_row_budget"],
                high["total_query_row_budget"],
            ),
            (2944, 5888, 8832),
        )
        self.assertEqual(
            (low["maximum_level"], midpoint["maximum_level"], high["maximum_level"]),
            (8, 9, 10),
        )

    def test_active_epoch_limits_have_literal_endpoint_and_axis_expectations(self):
        expected = {
            (0.0, 0.0, 0.0): (256, 16384, 33554432),
            (0.5, 0.5, 0.5): (512, 32768, 67108864),
            (1.0, 1.0, 1.0): (768, 49152, 100663296),
            (0.0, 0.5, 0.5): (384, 24576, 50331648),
            (1.0, 0.5, 0.5): (640, 40960, 83886080),
            (0.5, 0.0, 0.5): (448, 28672, 58720256),
            (0.5, 1.0, 0.5): (576, 36864, 75497472),
            (0.5, 0.5, 0.0): (448, 28672, 58720256),
            (0.5, 0.5, 1.0): (576, 36864, 75497472),
        }
        for point, values in expected.items():
            limits = MODULE.expected_active_epoch_limits(*point)
            self.assertEqual(
                (
                    limits["event_count"],
                    limits["mutation_count"],
                    limits["allocated_bytes"],
                    limits["row_batch_size"],
                ),
                (*values, MODULE.expected_parameters(*point)["backfill_batch_size"]),
            )

    def test_parameter_builders_reject_nonfinite_and_clamp_finite_values(self):
        for function in (
            MODULE.expected_parameters,
            MODULE.expected_active_epoch_limits,
        ):
            for point in (
                (math.nan, 0.5, 0.5),
                (0.5, math.inf, 0.5),
                (0.5, 0.5, -math.inf),
            ):
                with self.subTest(function=function.__name__, point=point):
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        function(*point)
            self.assertEqual(function(-1.0, -2.0, -3.0), function(0.0, 0.0, 0.0))
            self.assertEqual(function(2.0, 3.0, 4.0), function(1.0, 1.0, 1.0))

    def test_all_operational_limits_are_knob_derived(self):
        low = MODULE.expected_operational_limits(
            MODULE.expected_parameters(0.0, 0.0, 0.0)
        )
        high = MODULE.expected_operational_limits(
            MODULE.expected_parameters(1.0, 1.0, 1.0)
        )
        for name in low:
            if name in {
                "activation_search_effort_step",
                "activation_search_node_budget_step",
            }:
                self.assertEqual((low[name], high[name]), (4, 12))
                continue
            self.assertLess(low[name], high[name], name)

    def test_midpoint_backfill_and_dirty_sentinels_are_derived(self):
        midpoint = MODULE.expected_operational_limits(
            MODULE.expected_parameters(0.5, 0.5, 0.5)
        )
        self.assertEqual(midpoint["historical_backfill_drain_limit"], 128)
        self.assertEqual(midpoint["historical_backfill_overflow_probe_limit"], 129)
        self.assertEqual(midpoint["dirty_identity_drain_limit"], 512)
        self.assertEqual(midpoint["dirty_identity_overflow_probe_limit"], 513)

    def test_executable_identity_content_addresses_binary_without_local_path(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "example_binary"
            binary.write_bytes(b"exact executable bytes")
            identity = MODULE.executable_identity(binary, "example_target")

        self.assertEqual(identity["target"], "example_target")
        self.assertEqual(identity["filename"], "example_binary")
        self.assertEqual(identity["size_bytes"], 22)
        self.assertEqual(
            identity["sha256"],
            "135a9af43260004bfc617b97f806fcc6600e211fe0219d6c4311ef8cc6d59b48",
        )
        self.assertNotIn(str(binary.parent), identity.values())

    def test_link_decoder_proves_nonempty_upper_adjacency(self):
        blob = (
            (2).to_bytes(4, "little")
            + (1).to_bytes(4, "little")
            + (11).to_bytes(8, "little", signed=True)
            + (2).to_bytes(4, "little")
            + (17).to_bytes(8, "little", signed=True)
            + (19).to_bytes(8, "little", signed=True)
        )
        self.assertEqual(MODULE.decode_link_levels(blob, 1), [[11], [17, 19]])


if __name__ == "__main__":
    unittest.main()
