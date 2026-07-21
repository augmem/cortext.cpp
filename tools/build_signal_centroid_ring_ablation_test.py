import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_signal_centroid_ring_ablation.py")
SPEC = importlib.util.spec_from_file_location("ring_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SignalCentroidRingAblationTest(unittest.TestCase):
    def valid_black_box_fixture(self, root: Path):
        raw = root / "raw"
        audits = root / "audits"
        raw.mkdir()
        audits.mkdir()
        counter = MODULE.RING_MUTATION_COUNTER
        for name, knobs in MODULE.POINTS.items():
            profile = {
                "focus": knobs[0],
                "sensitivity": knobs[1],
                "stability": knobs[2],
                "processed_events": 2000,
                "wall_ms": 1,
                "active_signal_embedding_ring": {
                    "capacity": MODULE.expected_capacity(*knobs),
                    "modality_branching": False,
                    "source_id_branching": False,
                },
                "working_set_curve": [
                    {
                        "event_index": event_index,
                        "operation_ms": {counter: 1.0} if event_index == 0 else {},
                    }
                    for event_index in range(2000)
                ],
            }
            audit = {
                "passed": True,
                "query_count": 512,
                "source_digest_count": 4,
                "exact_top1": 1.0,
                "exact_identity_recall_at_k": 1.0,
                "semantic_coverage": 1.0,
                "deterministic_tie_order": True,
            }
            profile_path = raw / f"{name}.json"
            profile_path.write_text(json.dumps(profile))
            audit["private_profile_sha256"] = MODULE.file_sha256(profile_path)
            (audits / f"{name}-audit.json").write_text(json.dumps(audit))
        binary = root / "benchmark"
        binary.write_bytes(b"benchmark")
        return raw, audits, binary

    def test_capacity_is_knob_derived_at_structural_grid_points(self):
        capacities = {
            (focus, sensitivity, stability): MODULE.expected_capacity(
                focus, sensitivity, stability
            )
            for focus in (0.0, 0.5, 1.0)
            for sensitivity in (0.0, 0.5, 1.0)
            for stability in (0.0, 0.5, 1.0)
        }
        self.assertEqual(len(capacities), 27)
        self.assertEqual(capacities[(0.0, 0.0, 0.0)], 64)
        self.assertEqual(capacities[(0.5, 0.5, 0.5)], 128)
        self.assertEqual(capacities[(1.0, 1.0, 1.0)], 192)

    def test_named_execution_points_cover_each_axis(self):
        self.assertEqual(len(MODULE.POINTS), 9)
        self.assertEqual(MODULE.POINTS["focus-low"], (0.0, 0.5, 0.5))
        self.assertEqual(MODULE.POINTS["sensitivity-high"], (0.5, 1.0, 0.5))
        self.assertEqual(MODULE.POINTS["stability-low"], (0.5, 0.5, 0.0))

    def test_capacity_matches_std_lround_at_half_integer_tie(self):
        self.assertEqual(MODULE.expected_capacity(1.0 / 128.0, 0.0, 0.0), 65)

    def test_source_manifest_is_exact_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            paths = ("src/one.cpp", "tools/two.py")
            for relative, contents in zip(paths, (b"one", b"two")):
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            previous = MODULE.SOURCE_MANIFEST_PATHS
            MODULE.SOURCE_MANIFEST_PATHS = paths
            try:
                digest, entries = MODULE.source_code_manifest(repo)
            finally:
                MODULE.SOURCE_MANIFEST_PATHS = previous
            expected = hashlib.sha256()
            for relative, contents in zip(paths, (b"one", b"two")):
                expected.update(relative.encode("utf-8"))
                expected.update(b"\0")
                expected.update(hashlib.sha256(contents).digest())
            self.assertEqual(digest, expected.hexdigest())
            self.assertEqual([entry["path"] for entry in entries], list(paths))
            self.assertEqual(entries[0]["sha256"], hashlib.sha256(b"one").hexdigest())

    def test_build_rejects_missing_or_false_claim_fields(self):
        repo = MODULE_PATH.parents[1]
        cases = (
            ("missing counter", lambda profile, audit: profile[
                "working_set_curve"][0]["operation_ms"].clear()),
            ("wrong event count", lambda profile, audit: profile.update(
                processed_events=1999)),
            ("wrong query count", lambda profile, audit: audit.update(
                query_count=511)),
            ("quality below exact", lambda profile, audit: audit.update(
                exact_top1=0.999)),
            ("nondeterministic ties", lambda profile, audit: audit.update(
                deterministic_tie_order=False)),
            ("truncated rows", lambda profile, audit: profile[
                "working_set_curve"].pop()),
            ("duplicate row", lambda profile, audit: profile[
                "working_set_curve"][-1].update(event_index=1998)),
            ("missing profile digest", lambda profile, audit: audit.pop(
                "private_profile_sha256")),
            ("mismatched profile digest", lambda profile, audit: audit.update(
                private_profile_sha256="0" * 64)),
        )
        for label, corrupt in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw, audits, binary = self.valid_black_box_fixture(root)
                profile_path = raw / "midpoint.json"
                audit_path = audits / "midpoint-audit.json"
                profile = json.loads(profile_path.read_text())
                audit = json.loads(audit_path.read_text())
                corrupt(profile, audit)
                profile_path.write_text(json.dumps(profile))
                if label not in {
                    "missing profile digest", "mismatched profile digest"
                }:
                    audit["private_profile_sha256"] = MODULE.file_sha256(
                        profile_path
                    )
                audit_path.write_text(json.dumps(audit))
                with self.assertRaises(ValueError):
                    MODULE.build(raw, audits, binary, repo)


if __name__ == "__main__":
    unittest.main()
